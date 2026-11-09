import test from 'node:test';
import assert from 'node:assert/strict';
import {MemoryEditor,characterCount} from '../memory-editor.js';

const storage=()=>{const values=new Map();return {getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,v),removeItem:k=>values.delete(k)};};
const memory=(sid='a',v=3,text='原始记忆')=>({status:'ok',session_id:sid,memory_version:v,canEdit:true,canRollback:v>3,episodic:{'30sec':[{id:'first',content:text,editable:true},{id:'second',content:'另一段',editable:true}]}});
const response=(data,status=200)=>({ok:status<400,status,json:async()=>data});
const setup=options=>new MemoryEditor({storage:storage(),newKey:()=> 'memory-edit-test-key',...options});

test('save sends only changed records and a base version; undo uses the returned version',async()=>{
 const calls=[];let current=memory();
 const editor=setup({fetch:async(url,options)=>{calls.push({url,options});if(options.method){current=memory('a',current.memory_version+1,options.method==='PUT'?'新记忆':'原始记忆');return response({status:'ok',memory:current});}return response(current);}});
 editor.receive('a',current);editor.draft('a','first','新记忆');await editor.save('a');
 assert.deepEqual(JSON.parse(calls[0].options.body),{baseVersion:3,records:[{id:'first',content:'新记忆'}]});
 assert.equal(calls[0].options.headers['Idempotency-Key'],'memory-edit-test-key');
 assert.equal(editor.state('a').memory.memory_version,4);assert.deepEqual(editor.state('a').drafts,{});
 await editor.undo('a');assert.equal(calls[2].options.method,'POST');
 assert.deepEqual(JSON.parse(calls[2].options.body),{currentVersion:4});
 assert.equal(editor.state('a').memory.episodic['30sec'][0].content,'原始记忆');
});

test('an uncertain save survives reload and retries the exact receipt, then reads the latest snapshot',async()=>{
 const saved=storage(),calls=[];let fail=true;
 const fetch=async(url,options)=>{calls.push({url,options});if(options.method){if(fail)throw new Error('connection lost');return response({memory:memory('a',4,'草稿')});}return response(memory('a',5,'其他标签页的新版本'));};
 const first=setup({storage:saved,fetch});first.receive('a',memory());first.draft('a','first','草稿');await first.save('a');
 assert.equal(first.state('a').phase,'uncertain');const original=JSON.stringify(first.state('a').pending);
 const reloaded=setup({storage:saved,fetch,newKey:()=>{throw new Error('Must not create a new key');}});
 reloaded.receive('a',memory('a',4,'草稿'));assert.equal(JSON.stringify(reloaded.state('a').pending),original);
 await reloaded.save('a');assert.equal(calls.length,1,'ordinary save must not duplicate an uncertain operation');
 fail=false;await reloaded.retry('a');
 assert.equal(calls[1].options.body,calls[0].options.body);assert.deepEqual(calls[1].options.headers,calls[0].options.headers);
 assert.equal(reloaded.state('a').memory.memory_version,5);assert.equal(reloaded.state('a').pending,null);
 assert.equal(saved.getItem(reloaded.key('a')),null);
});

test('version conflicts keep drafts and require reading and reviewing the latest version',async()=>{
 let writes=0;
 const editor=setup({fetch:async(url,options)=>options.method?(writes++,response({error:{code:'memory_version_conflict',message:'版本冲突'}},409)):response(memory('a',4,'另一人的修改'))});
 editor.receive('a',memory());editor.draft('a','first','我的草稿');await editor.save('a');
 assert.equal(editor.state('a').conflict,true);assert.equal(editor.state('a').drafts.first.content,'我的草稿');
 editor.acceptLatest('a');assert.equal(editor.state('a').baseVersion,3,'cannot acknowledge before loading the new snapshot');
 await editor.refresh('a');assert.equal(editor.state('a').baseVersion,3);
 await editor.save('a');assert.equal(writes,1,'a newer version must not be silently overwritten');
 editor.acceptLatest('a');assert.equal(editor.state('a').baseVersion,4);assert.equal(editor.state('a').drafts.first.original,'另一人的修改');
 assert.equal(editor.state('a').drafts.first.content,'我的草稿');
});

test('Unicode limits and empty edits are checked before a write',async()=>{
 let calls=0;const editor=setup({fetch:async()=>{calls++;throw new Error('not expected');}});
 editor.receive('a',memory());assert.equal(characterCount('😀'),1);
 editor.draft('a','first','😀'.repeat(4000));assert.equal(editor.changes('a').error,'');
 editor.draft('a','first','😀'.repeat(4001));await editor.save('a');assert.equal(calls,0);
 editor.draft('a','first','  ');await editor.save('a');assert.equal(calls,0);
 editor.draft('a','first','原始记忆');assert.equal(editor.changes('a').records.length,0);
});

test('switching sessions or double clicking cannot redirect or duplicate a pending mutation',async()=>{
 let release,writes=0;const published=[];
 const editor=setup({onPublished:(sid,mem)=>published.push([sid,mem.session_id]),fetch:async(url,options)=>{
  if(options.method){writes++;return new Promise(resolve=>{release=()=>resolve(response({memory:memory('a',4,'A的新内容')}));});}
  return response(memory('a',4,'A的新内容'));
 }});
 editor.receive('a',memory());editor.draft('a','first','A的新内容');const saving=editor.save('a');
 editor.receive('b',memory('b'));editor.open('b');await editor.save('a');await editor.retry('a');
 assert.equal(writes,1);release();await saving;
 assert.equal(editor.state('b').memory.memory_version,3);assert.equal(editor.state('a').memory.memory_version,4);
 assert.deepEqual(published,[['a','a'],['a','a']]);
});

test('a failed rebuild keeps the draft and never publishes a success state',async()=>{
 const editor=setup({onPublished:()=>assert.fail('failed rebuild must not publish'),fetch:async()=>response({error:{code:'memory_derivation_failed',message:'摘要生成失败'}},503)});
 editor.receive('a',memory());editor.draft('a','first','需要保留');await editor.save('a');
 assert.equal(editor.state('a').memory.memory_version,3);assert.equal(editor.state('a').pending,null);
 assert.equal(editor.state('a').drafts.first.content,'需要保留');assert.match(editor.state('a').message,/摘要生成失败/);
});

test('browser timeout exceptions keep the pending receipt instead of authorizing a new write',async()=>{
 const saved=storage();const editor=setup({storage:saved,fetch:async()=>{throw new DOMException('The operation timed out','TimeoutError');}});
 editor.receive('a',memory());editor.draft('a','first','仍可能在后台保存');await editor.save('a');
 assert.equal(editor.state('a').phase,'uncertain');assert.ok(editor.state('a').pending);
 assert.equal(JSON.parse(saved.getItem(editor.key('a'))).pending.key,'memory-edit-test-key');
});

test('closing and reopening keeps a draft; inability to store a receipt prevents an untraceable write',async()=>{
 const saved=storage();const editor=setup({storage:saved});editor.receive('a',memory());editor.draft('a','first','待续草稿');editor.close('a');
 const reloaded=setup({storage:saved});reloaded.receive('a',memory());reloaded.open('a');assert.equal(reloaded.state('a').drafts.first.content,'待续草稿');
 let writes=0;const blocked=setup({storage:{getItem:()=>null,setItem:()=>{throw new Error('storage blocked');},removeItem:()=>{}},fetch:async()=>{writes++;}});
 blocked.receive('a',memory());blocked.draft('a','first','草稿');await blocked.save('a');assert.equal(writes,0);assert.equal(blocked.state('a').pending,null);
});
