import test from 'node:test';
import assert from 'node:assert/strict';

// These catch duplicate paid submissions, lost task recovery, clock offsets,
// and a result belonging to the wrong session. No external API is called.
const moduleURL = new URL('../core.js', import.meta.url);
const available = await import('node:fs').then(fs => fs.existsSync(moduleURL));
const core = available ? await import(moduleURL) : {};
const storage = () => {
  const values = new Map();
  return { getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k) };
};
const response = data => ({ ok: true, status: 200, json: async () => data });

test('admin mode is locked by default and stores only a tab-scoped unlock marker', () => {
  assert.equal(typeof core.AdminMode, 'function', 'admin mode is not implemented');
  const saved = storage();
  const changes = [];
  const admin = new core.AdminMode({storage: saved, code: '老师演示码', onChange: unlocked => changes.push(unlocked)});
  assert.equal(admin.isUnlocked(), false);
  assert.equal(admin.unlock('wrong'), false);
  assert.equal(admin.isUnlocked(), false);
  assert.equal(admin.unlock('老师演示码'), true);
  assert.equal(admin.isUnlocked(), true);
  assert.equal(saved.getItem(core.ADMIN_MODE_STORAGE_KEY), 'unlocked');
  assert.equal(String(saved.getItem(core.ADMIN_MODE_STORAGE_KEY)).includes('老师演示码'), false);
  admin.lock();
  assert.equal(admin.isUnlocked(), false);
  assert.deepEqual(changes, [true, false]);
});

test('admin mode survives a reload in the same tab but not a different storage', () => {
  const saved = storage();
  const first = new core.AdminMode({storage: saved, code: 'demo'});
  assert.equal(first.unlock('demo'), true);
  const reloaded = new core.AdminMode({storage: saved, code: 'demo'});
  assert.equal(reloaded.isUnlocked(), true);
  const otherTab = new core.AdminMode({storage: storage(), code: 'demo'});
  assert.equal(otherTab.isUnlocked(), false);
});

test('admin mode fails closed when browser storage is unavailable', () => {
  const admin = new core.AdminMode({storage: null, code: 'demo'});
  assert.equal(admin.unlock('demo'), false);
  assert.equal(admin.isUnlocked(), false);
});

test('an interrupted poll resumes its persisted task with no second POST', async () => {
  assert.equal(typeof core.QueryController, 'function', 'query recovery is not implemented');
  const saved = storage();
  const calls = [];
  let broken = true;
  const fetch = async (url, options = {}) => {
    calls.push({url, options});
    if (options.method === 'POST') return response({status: 'queued', task_id: 'task-1'});
    if (broken) throw new Error('network interruption');
    return response({status: 'done', session_id: 'a', result: {status: 'ok', answer: '真实回答', generation_status: 'text_success'}});
  };
  const first = new core.QueryController({fetch, storage: saved, sleep: async () => {}});
  await first.submit('a', '问题一');
  assert.equal(first.state('a').turns[0].status, 'paused');
  broken = false;
  const recovered = new core.QueryController({fetch, storage: saved, sleep: async () => {}});
  await recovered.resume('a');
  assert.equal(recovered.state('a').turns[0].result.answer, '真实回答');
  assert.equal(calls.filter(c => c.options.method === 'POST').length, 1);
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    question: '问题一', mode: 'async', response_mode: 'legacy', memory_mode: 'auto', cache_mode: 'auto',
    use_image_evidence: true, max_image_frames: 6, max_image_evidence: 6,
    answer_tts: false, client_source: 'history_demo', input_method: 'text'
  });
});

test('an uncertain POST stays blocked across reload and is never automatically repeated', async () => {
  assert.equal(typeof core.QueryController, 'function', 'submission safety is not implemented');
  const saved = storage(); let posts = 0;
  const fetch = async () => { posts++; throw new Error('connection lost after sending'); };
  const controller = new core.QueryController({fetch, storage: saved});
  await controller.submit('a', '可能已提交');
  const reloaded = new core.QueryController({fetch, storage: saved});
  assert.equal(reloaded.state('a').turns[0].status, 'uncertain');
  await reloaded.resume('a');
  await reloaded.submit('a', '重复问题');
  assert.equal(posts, 1);
});

test('late poll results cannot enter another session and a second turn keeps the first', async () => {
  assert.equal(typeof core.QueryController, 'function', 'per-session turns are not implemented');
  let resolveA; let task = 0;
  const fetch = async (url, options = {}) => {
    if (options.method === 'POST') return response({task_id: `task-${++task}`});
    if (url.endsWith('task-1?compact=1')) return new Promise(resolve => { resolveA = resolve; });
    return response({status: 'done', session_id: 'b', result: {answer: `回答${task}`, status: 'ok'}});
  };
  const controller = new core.QueryController({fetch, storage: storage()});
  const pending = controller.submit('a', 'A的问题');
  while (!resolveA) await new Promise(resolve => setImmediate(resolve));
  controller.stop('a');
  await controller.submit('b', 'B的问题');
  resolveA(response({status:'done', session_id:'a', result:{answer:'A的迟到回答'}}));
  await pending;
  await controller.submit('b', 'B的追问');
  assert.equal(controller.state('b').turns.length, 2);
  assert.equal(controller.state('b').turns[0].result.answer, '回答2');
  assert.equal(controller.state('b').turns[1].result.answer, '回答3');
  assert.equal(controller.state('a').turns[0].status, 'paused');
});

test('evidence clocks map once to playback and reject unknown/outside clocks', () => {
  assert.equal(typeof core.playbackSeconds, 'function', 'evidence clock mapping is not implemented');
  const media = {time_origin:'session_relative_seconds', video_start_seconds:100,time_alignment_known:true,duration_seconds:60};
  assert.equal(core.playbackSeconds({capture_time_seconds: 112}, media), 12);
  assert.equal(core.playbackSeconds({timestamp: '00:01:52'}, media), 12);
  assert.equal(core.playbackSeconds({timestamp:112, frame_directory_offset:1000}, media),12);
  assert.equal(core.playbackSeconds({timestamp:99}, media),null);
  assert.equal(core.playbackSeconds({timestamp:200}, media),null);
  assert.equal(core.playbackSeconds({path:'frame_000112.jpg'}, media),null);
  assert.equal(core.playbackSeconds({timestamp:12}, {...media,time_alignment_known:false}),null);
});

test('generation failures and local fallback remain visible despite an ok task', () => {
  assert.equal(typeof core.generationLabel, 'function', 'generation status is not implemented');
  assert.match(core.generationLabel({status:'failed',generation_status:'multimodal_success'}),/失败/);
  assert.match(core.generationLabel({status:'ok',generation_status:'local_evidence_summary'}),/降级/);
  assert.match(core.generationLabel({status:'ok',generation_status:'text_success'}),/文本/);
  assert.match(core.generationLabel({status:'ok',generation_status:'multimodal_success'}),/图文/);
});

test('backend start/end evidence ranges seek to their start', () => {
  const media = {session_id:'a',time_alignment_known:true,video_start_seconds:10,duration_seconds:60};
  assert.equal(core.playbackSeconds({start:12,end:18},media),2);
  assert.equal(core.playbackSeconds({start:9,end:18},media),null);
});

test('evidence from another owner never seeks the selected session video', () => {
  const media = {session_id:'a',time_alignment_known:true,video_start_seconds:0,duration_seconds:60};
  assert.equal(core.playbackSeconds({timestamp:12,owner_session_id:'other'},media),null);
  assert.equal(core.playbackSeconds({timestamp:12,owner_session_id:'a'},media),12);
  assert.equal(core.playbackSeconds({timestamp:12},media),12);
});

test('explicit video origins accept finite signed seconds', () => {
  const media = {time_alignment_known:true,video_start_seconds:-5,duration_seconds:60};
  assert.equal(core.playbackSeconds({capture_time_seconds:0},media),5);
  assert.equal(core.playbackSeconds({capture_time_seconds:56},media),null);
  for (const origin of [NaN,Infinity,-Infinity,'-5',null]) {
    assert.equal(core.playbackSeconds({capture_time_seconds:0},{...media,video_start_seconds:origin}),null);
  }
});

test('a provider text-only fallback is explicitly labeled as degraded', () => {
  assert.equal(core.generationLabel({status:'ok',generation_status:'text_only_success'}),'生成降级：纯文本回答');
  assert.equal(core.generationLabel({status:'ok',raw:{llm_debug:{generation_status:'text_only_success'}}}),'生成降级：纯文本回答');
});

test('history survives reload, merges a live task once, and never POSTs', async () => {
  const saved=storage(),calls=[];
  const rows=[{session_id:'a',turn_id:'turn-1',task_id:'task-1',created_at:'2026-11-04T01:00:00Z',question:'之前的问题',answer:'之前的回答',status:'done'}];
  const fetch=async(url,options={})=>{calls.push(options.method||'GET');return response({status:'ok',session_id:'a',items:rows});};
  const controller=new core.QueryController({fetch,storage:saved});
  await controller.loadHistory('a');await controller.loadHistory('a');
  assert.equal(controller.state('a').turns.length,1);
  assert.equal(controller.state('a').turns[0].result.answer,'之前的回答');
  const reloaded=new core.QueryController({fetch,storage:saved});await reloaded.loadHistory('a');
  assert.equal(reloaded.state('a').turns.length,1);
  assert.ok(calls.every(method=>method==='GET'));
});

test('history cannot inject a different session and expired details keep saved answers', async () => {
  let detail=false;
  const fetch=async()=>response(detail?{status:'not_found'}:{status:'ok',session_id:'a',items:[
    {session_id:'b',turn_id:'other',question:'foreign',answer:'foreign'},
    {session_id:'a',turn_id:'own',task_id:'expired',question:'own',answer:'saved',status:'done'}]});
  const controller=new core.QueryController({fetch,storage:storage()});await controller.loadHistory('a');
  assert.equal(controller.state('a').turns.length,1);detail=true;
  await controller.loadTurn('a',controller.state('a').turns[0]);
  assert.equal(controller.state('a').turns[0].result.answer,'saved');
});

test('partial text appears during polling and failed summaries never become successful turns', async () => {
  let polls=0;const snapshots=[];
  const fetch=async(url,options={})=>response(options.method==='POST'?{task_id:'task'}:++polls===1?
    {status:'in_progress',progress:{stage:'streaming',partial_answer:'正在回答'}}:
    {status:'done',result:{status:'ok',generation_status:'local_evidence_summary',answer:'Raw evidence'}});
  const controller=new core.QueryController({fetch,storage:storage(),sleep:async()=>{},
    onChange:(_sid,state)=>snapshots.push(state.turns[0]?.partialAnswer)});
  await controller.submit('a','问句');
  assert.ok(snapshots.includes('正在回答'));
  assert.equal(controller.state('a').turns[0].status,'failed');
  assert.equal(controller.state('a').turns[0].partialAnswer,'');
  assert.equal(core.failedAnswer({answer:'LLM answer generation failed after retries. Local evidence summary: raw'}),true);
});

test('a completed history record releases the restored pending lock without duplicating its turn', async () => {
  const saved=storage(),key='lightmem-history.pending.v1:a',calls=[];
  saved.setItem(key,JSON.stringify({question:'长时间等待的问题',taskId:'slow-task',createdAt:1}));
  const row={session_id:'a',task_id:'slow-task',turn_id:'saved-turn',question:'长时间等待的问题',answer:'后台已完成',status:'done'};
  const controller=new core.QueryController({storage:saved,fetch:async(url,options={})=>{
    calls.push(options.method||'GET');return response({status:'ok',session_id:'a',items:[row,row]});
  }});
  assert.equal(controller.pending('a').status,'paused');
  await controller.loadHistory('a');
  assert.equal(controller.pending('a'),undefined);
  assert.equal(saved.getItem(key),null);
  assert.equal(controller.state('a').turns.length,1);
  assert.equal(controller.state('a').turns[0].result.answer,'后台已完成');
  assert.equal(controller.state('a').turns[0].detailLoaded,false,'full evidence remains available for lazy loading');
  assert.deepEqual(calls,['GET']);
});

test('a timed-out wait can check the same completed task and unlock without another POST', async () => {
  const saved=storage(),calls=[];let now=0,done=false;
  const controller=new core.QueryController({storage:saved,now:()=>now,pollBudget:10,sleep:async()=>{now+=10;},
    fetch:async(url,options={})=>{calls.push({url,method:options.method||'GET'});return response(options.method==='POST'
      ?{task_id:'slow-task'}:done?{status:'done',session_id:'a',result:{status:'ok',answer:'延迟完成的回答'}}:{status:'running'});}});
  await controller.submit('a','长请求');
  assert.equal(controller.pending('a').status,'paused');
  done=true;await controller.checkPending('a');
  assert.equal(controller.pending('a'),undefined);
  assert.equal(controller.state('a').turns[0].result.answer,'延迟完成的回答');
  assert.equal(saved.getItem(controller.key('a')),null);
  assert.equal(calls.filter(call=>call.method==='POST').length,1);
  assert.ok(calls.filter(call=>call.method==='GET').every(call=>call.url==='/api/query_task/slow-task?compact=1'));
});

test('running tasks and nonterminal history never release a pending lock', async () => {
  const saved=storage(),key='lightmem-history.pending.v1:a';
  saved.setItem(key,JSON.stringify({question:'后台仍在运行',taskId:'running-task',createdAt:1}));
  const controller=new core.QueryController({storage:saved,fetch:async url=>response(url.includes('/qa_history')
    ?{status:'ok',session_id:'a',items:[{task_id:'running-task',status:'running',answer:''}]}
    :{status:'running',session_id:'a'})});
  await controller.loadHistory('a');await controller.checkPending('a');
  assert.equal(controller.pending('a').status,'paused');
  assert.ok(saved.getItem(key));
  assert.equal(controller.state('a').checkingPending,false);
});

test('checks are deduplicated and ignore responses invalidated by leaving the session', async () => {
  const saved=storage();saved.setItem('lightmem-history.pending.v1:a',JSON.stringify({question:'原问题',taskId:'old-task'}));
  let release,gets=0;
  const controller=new core.QueryController({storage:saved,fetch:()=>{gets++;return new Promise(resolve=>{release=resolve;});}});
  const waiting=controller.checkPending('a');await controller.checkPending('a');
  assert.equal(gets,1);
  controller.stop('a');
  release(response({status:'done',session_id:'a',result:{answer:'迟到的旧响应'}}));await waiting;
  assert.equal(controller.pending('a').status,'paused');
  assert.equal(controller.state('a').turns[0].result,undefined);
  assert.ok(saved.getItem(controller.key('a')));
});

test('wrong-session and failed rechecks preserve the pending task for later recovery', async () => {
  const saved=storage();saved.setItem('lightmem-history.pending.v1:a',JSON.stringify({question:'原问题',taskId:'own-task'}));
  let broken=false;
  const controller=new core.QueryController({storage:saved,fetch:async()=>{
    if(broken)throw new Error('offline');
    return response({status:'done',session_id:'b',result:{answer:'别的会话'}});
  }});
  await controller.checkPending('a');
  assert.equal(controller.pending('a').result,undefined);
  broken=true;await controller.checkPending('a');
  assert.equal(controller.pending('a').status,'paused');
  assert.ok(saved.getItem(controller.key('a')));
});

test('loading old answer details cannot remove the newer pending task from storage', async () => {
  const saved=storage(),key='lightmem-history.pending.v1:a',pending=JSON.stringify({question:'新问题',taskId:'new-task'});
  saved.setItem(key,pending);
  const controller=new core.QueryController({storage:saved,fetch:async()=>response({status:'done',session_id:'a',result:{answer:'旧回答'}})});
  const old={taskId:'old-task',status:'done',result:{answer:'旧回答摘要'},fromHistory:true};
  controller.state('a').turns.unshift(old);await controller.loadTurn('a',old);
  assert.equal(saved.getItem(key),pending);
  assert.equal(controller.pending('a').taskId,'new-task');
});

test('saved history recovers a completed answer after its task file expires', async () => {
  const saved=storage();saved.setItem('lightmem-history.pending.v1:a',JSON.stringify({question:'旧任务',taskId:'expired-task'}));
  const controller=new core.QueryController({storage:saved,fetch:async url=>response(url.includes('/qa_history')
    ?{status:'ok',session_id:'a',items:[{task_id:'expired-task',status:'done',answer:'持久化的回答'}]}:{status:'not_found'})});
  await controller.checkPending('a');
  assert.equal(controller.pending('a'),undefined);
  assert.equal(controller.state('a').turns[0].result.answer,'持久化的回答');
});

test('history completion invalidates a late poll instead of relocking the completed turn', async () => {
  const saved=storage();saved.setItem('lightmem-history.pending.v1:a',JSON.stringify({question:'原问题',taskId:'original-task'}));
  let release;
  const controller=new core.QueryController({storage:saved,fetch:async url=>url.includes('/qa_history')
    ?response({status:'ok',session_id:'a',items:[{task_id:'original-task',status:'done',answer:'已完成'}]})
    :new Promise(resolve=>{release=resolve;})});
  const polling=controller.resume('a');await controller.loadHistory('a');
  release(response({status:'running'}));await polling;
  assert.equal(controller.pending('a'),undefined);
  assert.equal(controller.state('a').turns[0].status,'done');
  assert.equal(controller.state('a').turns[0].result.answer,'已完成');
});
test('API paths support a configured backend without duplicating the API prefix', () => {
  assert.equal(core.apiPath('/session/a'),'/api/session/a');
  assert.equal(core.apiPath('/api/session/a'),'/api/session/a');
  assert.equal(core.apiPath('/api/session/a','https://backend.test/api/'),'https://backend.test/api/session/a');
});
