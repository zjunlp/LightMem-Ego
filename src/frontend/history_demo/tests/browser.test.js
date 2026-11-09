import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {launchBrowser} from './browser-env.js';
const root = new URL('../../online_web/dist/', import.meta.url);
await mkdir(new URL('artifacts/', import.meta.url), {recursive:true});
const clip = fileURLToPath(new URL('artifacts/fixture.mp4', import.meta.url));
execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=0x182539:s=320x180:r=10','-t','3','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',clip]);
const server = createServer(async (req,res) => {
  const pathname = new URL(req.url,'http://localhost').pathname;
  const name = pathname.endsWith('/') ? pathname.slice(1)+'index.html' : pathname.slice(1);
  if (!/^[\w./-]+$/.test(name)||name.split('/').includes('..')) {res.writeHead(404).end();return;}
  try {const body=await readFile(new URL(name,root));res.setHeader('Content-Type',name.endsWith('.js')?'text/javascript':name.endsWith('.css')?'text/css':'text/html');res.end(body);} catch {res.writeHead(404).end('Not implemented');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await launchBrowser();
const page = await browser.newPage();
page.setDefaultTimeout(5000);
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
let postCount=0; let pollBroken=false; let uncertain=false; let delayedA=null;
let releaseInitialMemories;
let initialMemoriesWait=new Promise(resolve=>{releaseInitialMemories=resolve;});
let mediaRevision=0,aAvailable=true,cTitle='分页会话',delayedMedia=null,brokenVideo=false;
const memoryStates=new Map(),memoryReceipts=new Map(),memoryWrites=[];
let loseMemoryReceipt=false;
let uploadedId=null,uploadReady=false,uploadPosts=0;
const uploadReceipt=()=>({upload_id:uploadedId,session_id:`video_${uploadedId}`,filename:'fixture.mp4',
  status:uploadReady?'ready':'processing',stage:uploadReady?'ready':'memory',progress:uploadReady?100:80,query_ready:uploadReady});
const fixtureMemory=sid=>{
  if(!memoryStates.has(sid))memoryStates.set(sid,{status:'ok',session_id:sid,memory_version:3,canEdit:true,canRollback:false,updated_at:'2026-11-02T00:00:00Z',episodic:{'30sec':[{id:'record-1',editable:true,content:'<img src=x onerror="window.injected=true"> 记忆内容',event_time:{start_seconds:1,end_seconds:2}}],'3min':[],'10min':[],'1h':[]},semantic:[{content:'语义记录'}],visual:[{content:'视觉记录'}],counts:{total:3}});
  return memoryStates.get(sid);
};
const savedHistory=new Map(),submittedTasks=new Map();
const longCaption='画面证据：'+('这是一段真实证据事件说明，包含可见动作与场景细节。'.repeat(150))+'完整描述结束';
const json = (route,data,status=200) => route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
await page.route('**/*',async route=>{
  const req=route.request();const url=new URL(req.url());
  assert.equal(url.origin,base,'offline tests must never contact another host');
  if(!url.pathname.startsWith('/api/')) return route.continue();
  const path=url.pathname;
  if(path==='/api/video-uploads'&&req.method()==='POST'){
    uploadPosts++;
    const body=req.postDataBuffer().toString('latin1');
    uploadedId=body.match(/name="upload_id"\r\n\r\n([^\r]+)/)[1];
    assert.ok(body.includes('filename="fixture.mp4"'));
    return json(route,uploadReceipt(),202);
  }
  if(path===`/api/video-uploads/${uploadedId}`)return json(route,uploadReceipt());
  if(path.endsWith('/qa_history')) {
    const sid=path.split('/')[3];return json(route,{status:'ok',session_id:sid,items:savedHistory.get(sid)||[]});
  }
  if(path==='/api/memories/sessions') return json(route,{status:'ok',items:url.searchParams.has('cursor')?[{session_id:'c'}]:[{session_id:'a',title:'室内回看'},{session_id:'b',title:'无视频会话'}],has_more:!url.searchParams.has('cursor'),next_cursor:url.searchParams.has('cursor')?null:'page2'});
  if(path.endsWith('/memories')) {
    if(path.includes('/a/')&&initialMemoriesWait){const wait=initialMemoriesWait;initialMemoriesWait=null;await wait;}
    return json(route,fixtureMemory(path.split('/')[3]));
  }
  if(/\/memories\/30sec(?:\/rollback)?$/.test(path)){
    const sid=path.split('/')[3],body=req.postDataJSON(),key=req.headers()['idempotency-key'];
    memoryWrites.push({sid,body,key});assert.ok(key);
    if(memoryReceipts.has(key))return json(route,memoryReceipts.get(key));
    const memory=fixtureMemory(sid),rollback=path.endsWith('/rollback');
    if((body.baseVersion??body.currentVersion)!==memory.memory_version)return json(route,{error:{code:'memory_version_conflict',message:'记忆版本已更新'}},409);
    if(rollback){memory.episodic['30sec'][0].content=memory.backup;memory.canRollback=false;}
    else{memory.backup=memory.episodic['30sec'][0].content;memory.episodic['30sec'][0].content=body.records[0].content;memory.canRollback=true;}
    memory.memory_version++;
    const result=structuredClone({status:'ok',memory});memoryReceipts.set(key,result);
    if(loseMemoryReceipt){loseMemoryReceipt=false;return route.abort('failed');}
    return json(route,result);
  }
  if(path.endsWith('/demo-media')) {
    const media={status:'ok',session_id:path.split('/')[3],title:path.includes('/c/')?cTitle:'室内回看',available:!path.includes('/b/')&&aAvailable,video_url:'/session/a/demo-video',duration_seconds:3,time_origin:'session_relative_seconds',video_start_seconds:0,time_alignment_known:true,memory_until_seconds:2,view_description:`原视频前视 ${mediaRevision}`,reason:path.includes('/b/')?'没有绑定视频':!aAvailable?'已取消视频绑定':null};
    if(path.includes('/a/')&&delayedMedia){const delay=delayedMedia;delayedMedia=null;delay.started();await delay.wait;}
    return json(route,media);
  }
  if(path.endsWith('/demo-video')) {
    if(brokenVideo)return route.fulfill({status:200,contentType:'video/mp4',body:'interrupted or invalid media'});
    const bytes=await readFile(clip);const match=req.headers().range?.match(/bytes=(\d+)-(\d*)/);
    if(match){const start=Number(match[1]),end=match[2]?Math.min(Number(match[2]),bytes.length-1):bytes.length-1;
      return route.fulfill({status:206,headers:{'Content-Type':'video/mp4','Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${bytes.length}`},body:bytes.subarray(start,end+1)});}
    return route.fulfill({status:200,headers:{'Content-Type':'video/mp4','Accept-Ranges':'bytes'},body:bytes});
  }
  if(path.endsWith('/ask')) {
    postCount++;const body=req.postDataJSON();
    assert.equal(body.memory_mode,'auto');assert.equal(body.max_image_evidence,6);assert.equal(body.answer_tts,false);
    if(uncertain)return route.abort('failed');
    submittedTasks.set(`task-${postCount}`,{sid:path.split('/')[3],question:body.question});
    return json(route,{status:'queued',task_id:`task-${postCount}`},202);
  }
  if(path.startsWith('/api/query_task/')) {
    if(path.endsWith('/legacy-failed'))return json(route,{status:'done',session_id:'c',result:{status:'ok',generation_status:'local_evidence_summary',answer:'LLM answer generation failed after retries. Local evidence summary: RAW_FAILURE_EVIDENCE'}});
    if(pollBroken)return route.abort('failed');
    if(delayedA) {const wait=delayedA;delayedA=null;await wait;}
    const taskId=path.split('/').at(-1),task=submittedTasks.get(taskId);
    if(task){const rows=savedHistory.get(task.sid)||[];if(!rows.some(row=>row.task_id===taskId))rows.push({turn_id:taskId,task_id:taskId,session_id:task.sid,question:task.question,answer:`真实回答 ${taskId.split('-').at(-1)}`,status:'done',created_at:new Date().toISOString()});savedHistory.set(task.sid,rows);}
    return json(route,{status:'done',result:{status:'ok',answer:`真实回答 ${path.split('-').at(-1)}`,generation_status:path.endsWith('task-2')?'text_only_success':'multimodal_success',used_memory_sources:['M_lt'],timestamps:[{start:0,end:2}],evidence_frames:[
      {image_url:`${base}/session/a/file?path=fixture.jpg`,relative_file_url:'/session/a/file?path=fixture.jpg',owner_session_id:'a',capture_time_seconds:1,source:'M_lt',caption:longCaption},
      {image_url:`${base}/session/other/file?path=fixture.jpg`,relative_file_url:'/session/other/file?path=fixture.jpg',owner_session_id:'other',capture_time_seconds:2,source:'M_lt',caption:'其他会话画面证据'},
      ...Array.from({length:4},()=>({relative_file_url:'/session/a/file?path=fixture.jpg',owner_session_id:'a',source:'M_lt',caption:longCaption}))
    ]}});
  }
  if(/^\/api\/session\/(a|other)\/file$/.test(path))return route.fulfill({status:200,contentType:'image/png',body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j05sAAAAASUVORK5CYII=','base64')});
  throw new Error(`Unhandled fixture ${path}`);
});
try {
  await page.goto(`${base}/history-demo/`);
  assert.equal(await page.getByRole('heading',{name:'已有视频演示',exact:true}).count(),1,'history demo page is missing');
  await page.getByRole('button',{name:/室内回看/}).click();
  await page.waitForFunction(()=>document.querySelector('video')?.readyState>=1);
  assert.equal(await page.getByText('语义记录',{exact:true}).count(),0,'video starts buffering before slow memory loading finishes');
  releaseInitialMemories();
  await page.getByText('语义记录',{exact:true}).waitFor();
  assert.equal(await page.evaluate(()=>window.injected),undefined,'memory content must stay text');
  await page.waitForFunction(()=>document.querySelector('video')?.readyState>=1);
  assert.match(await page.locator('#coverage').innerText(),/00:02/);
  await page.locator('video').evaluate(v=>{v.currentTime=1.25;});
  await page.waitForFunction(()=>Number(sessionStorage.getItem('lightmem-video.position.v1:a'))>=1.2);
  const previousVideoLoad=await page.locator('video').getAttribute('src');
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2&&document.querySelector('video').currentTime>=1.2);
  assert.notEqual(await page.locator('video').getAttribute('src'),previousVideoLoad,'reload must not reuse a locked partial video cache entry');
  assert.equal(await page.locator('video').evaluate(v=>v.paused),true,'refresh restores position without unexpectedly autoplaying');
  brokenVideo=true;
  await page.locator('video').evaluate(v=>{v.src='/api/session/a/demo-video?broken=1';v.load();});
  await page.getByRole('button',{name:'从暂停处继续播放',exact:true}).waitFor();
  assert.match(await page.locator('#video-status').innerText(),/00:01/);
  brokenVideo=false;
  await page.getByRole('button',{name:'从暂停处继续播放',exact:true}).click();
  await page.waitForFunction(()=>{const v=document.querySelector('video');return !v.paused&&!v.error&&v.currentTime>=1.2;});
  await page.locator('video').evaluate(v=>{v.pause();v.currentTime=0;});
  await page.waitForFunction(()=>Number(sessionStorage.getItem('lightmem-video.position.v1:a'))===0);
  await page.getByLabel('向所选会话提问').fill('第一个问题');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText('真实回答 1',{exact:true}).waitFor();
  await page.getByText('图文生成',{exact:true}).waitFor();
  await page.getByText('画面证据（6张）',{exact:true}).click();
  assert.match(await page.locator('#conversation').innerText(),/M_lt/);
  const evidenceImages=page.locator('#conversation [data-testid="evidence-image"]');
  await page.waitForFunction(()=>[...document.querySelectorAll('#conversation [data-testid="evidence-image"]')].every(img=>img.complete&&img.naturalWidth>0));
  assert.equal(await evidenceImages.count(),6);
  assert.equal(await evidenceImages.first().getAttribute('src'),`${base}/api/session/a/file?path=fixture.jpg`,'backend relative URL must win over its absolute URL');
  const longEvidence=page.locator('.evidence-card').first();
  const fullDescription=longEvidence.locator('details');
  assert.equal(await fullDescription.count(),1,'long evidence description must be expandable');
  assert.equal(await fullDescription.getAttribute('open'),null,'long description starts collapsed');
  assert.ok((await longEvidence.innerText()).length<300,'collapsed evidence must stay compact');
  assert.equal(await longEvidence.getByRole('button',{name:/跳转视频/}).isVisible(),true);
  assert.equal(await longEvidence.getByText('来源：M_lt',{exact:true}).isVisible(),true);
  const imageLink=longEvidence.locator('a').filter({has:page.locator('img')});
  assert.equal(await imageLink.getAttribute('href'),`${base}/api/session/a/file?path=fixture.jpg`);
  assert.equal(await imageLink.getAttribute('target'),'_blank');
  assert.match(await imageLink.getAttribute('rel'),/noopener/);
  await longEvidence.getByText('完整证据描述',{exact:true}).click();
  assert.equal(await page.locator('.evidence-grid').evaluate(grid=>getComputedStyle(grid).gridTemplateColumns.split(' ').length),3,'desktop evidence should form three clear columns');
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.locator('.evidence-grid').evaluate(grid=>getComputedStyle(grid).gridTemplateColumns.split(' ').length),2,'mobile evidence should keep two columns');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.setViewportSize({width:1280,height:720});
  assert.equal(await fullDescription.locator('p').innerText(),longCaption,'expanded description must preserve the original text');
  await longEvidence.getByText('完整证据描述',{exact:true}).click();
  assert.equal(await page.locator('.evidence-card').filter({hasText:'其他会话画面证据'}).getByRole('button').isDisabled(),true,'another owner cannot seek this video');
  const rangeButton=page.locator('#conversation .turn-actions').getByRole('button',{name:'跳转视频 · 00:00–00:02',exact:true});
  await page.locator('#conversation').getByRole('button',{name:/跳转视频.*00:01/}).click();
  await page.waitForFunction(()=>Math.round(document.querySelector('video').currentTime)===1);
  assert.equal(await page.locator('video').evaluate(v=>Math.round(v.currentTime)),1);
  await rangeButton.click();
  await page.waitForFunction(()=>document.querySelector('video').currentTime===0);
  await page.getByLabel('向所选会话提问').fill('第二个问题');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText('真实回答 2',{exact:true}).waitFor();
  await page.getByText('生成降级：纯文本回答',{exact:true}).waitFor();
  assert.equal(await page.getByText('真实回答 1',{exact:true}).count(),1,'prior turn must remain visible');
  await page.getByLabel('向所选会话提问').fill('刷新保留的草稿');
  await page.locator('video').evaluate(async video=>{video.muted=true;video.loop=true;window.refreshVideoLoads=0;video.addEventListener('loadstart',()=>window.refreshVideoLoads++);await video.play();});
  mediaRevision=1;
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await page.getByText(/原视频前视 1/).waitFor();
  assert.equal(await page.locator('video').evaluate(video=>video.paused),false,'unchanged video URL must keep playing');
  assert.equal(await page.evaluate(()=>window.refreshVideoLoads),0,'metadata refresh must not reload the same video');
  assert.equal(await page.getByLabel('向所选会话提问').inputValue(),'刷新保留的草稿');
  assert.equal(await page.getByText('真实回答 1',{exact:true}).count(),1);
  aAvailable=false;mediaRevision=2;
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await page.getByText(/已取消视频绑定/).waitFor();
  assert.equal(await page.locator('video').isHidden(),true,'removed binding must hide the previously loaded video');
  assert.equal(await page.locator('video').getAttribute('src'),null,'removed binding must clear its old source');
  aAvailable=true;mediaRevision=3;
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await page.getByText(/原视频前视 3/).waitFor();
  await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
  assert.equal(postCount,2,'refresh must preserve completed turns without submitting questions');
  let releaseMedia,markMediaStarted;
  const mediaStarted=new Promise(resolve=>{markMediaStarted=resolve;});
  delayedMedia={started:markMediaStarted,wait:new Promise(resolve=>{releaseMedia=resolve;})};
  const lateMediaResponse=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/session/a/demo-media');
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await mediaStarted;
  await page.locator('[data-session-id="b"]').click();
  await page.getByText(/没有绑定视频/).waitFor();
  releaseMedia();await lateMediaResponse;
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  assert.equal(await page.locator('#session-id').innerText(),'b');
  assert.equal(await page.locator('video').isHidden(),true,'late refreshed A media must not replace B');
  assert.match(await page.locator('#media-message').innerText(),/没有绑定视频/);
  await page.locator('[data-session-id="a"]').click();
  await page.waitForFunction(()=>document.querySelector('video').readyState>=1&&!document.querySelector('video').hidden);
  pollBroken=true;
  await page.getByLabel('向所选会话提问').fill('等待恢复');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByRole('button',{name:'继续等待结果',exact:true}).waitFor();
  await page.getByRole('button',{name:'核对任务状态',exact:true}).waitFor();
  assert.match(await page.locator('#ask-hint').innerText(),/核对/);
  assert.doesNotMatch(await page.locator('#ask-hint').innerText(),/问题正在处理/);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'pending recovery controls must fit mobile');
  await page.setViewportSize({width:1280,height:720});
  await page.reload();
  await page.getByRole('button',{name:/室内回看/}).click();
  await page.getByRole('button',{name:'继续等待结果',exact:true}).waitFor();
  pollBroken=false;
  // Returning to the tab must settle a finished task, even with its turn collapsed.
  await page.locator('.turn').last().evaluate(turn=>{turn.open=false;});
  await page.evaluate(()=>window.dispatchEvent(new Event('focus')));
  await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  assert.equal(await page.getByRole('button',{name:'核对任务状态',exact:true}).count(),0);
  await page.locator('.turn').last().locator('summary').first().click();
  await page.getByText('真实回答 3',{exact:true}).waitFor();
  assert.equal(await page.getByText('第一个问题',{exact:true}).count(),1,'completed history must survive refresh without duplicating the active task');
  assert.equal(await page.getByText('第二个问题',{exact:true}).count(),1);
  assert.equal(postCount,3,'resume must use GET, never repeat POST');
  await page.getByRole('button',{name:'加载更多会话',exact:true}).click();
  await page.getByRole('button',{name:/分页会话/}).waitFor();
  let release; delayedA=new Promise(resolve=>{release=resolve});
  await page.getByLabel('向所选会话提问').fill('切换前提问');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText('正在等待真实问答结果…',{exact:true}).waitFor();
  cTitle='更新分页会话';mediaRevision=4;
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await page.getByText(/原视频前视 4/).waitFor();
  assert.equal(await page.getByText('正在等待真实问答结果…',{exact:true}).count(),1,'refresh must keep the existing task polling');
  assert.equal(postCount,4,'refresh must not resubmit the pending question');
  assert.equal(await page.getByText('真实回答 3',{exact:true}).count(),1);
  await page.getByRole('button',{name:'加载更多会话',exact:true}).click();
  await page.getByRole('button',{name:/更新分页会话/}).waitFor();
  await page.getByRole('button',{name:/无视频会话/}).click();
  release();
  await page.getByText(/没有绑定视频/).waitFor();
  assert.equal(await page.getByText('真实回答 4',{exact:true}).count(),0,'late A answer must not render in B');
  uncertain=true;
  await page.getByLabel('向所选会话提问').fill('结果不确定的问题');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText(/提交结果不确定/).waitFor();
  assert.equal(await page.getByRole('button',{name:'发送问题',exact:true}).isDisabled(),true);
  assert.equal(postCount,5);
  savedHistory.set('c',[{turn_id:'failed-old',task_id:'legacy-failed',session_id:'c',question:'历史失败的问题',answer:'LLM answer generation failed after retries. Local evidence summary: RAW_FAILURE_EVIDENCE',status:'done',created_at:'2026-11-04T12:00:00Z'}]);
  await page.locator('[data-session-id="c"]').click();
  assert.equal(await page.locator('#view-memory-session').getAttribute('href'),'/memories/?memory_session=c','selected session should link to its memory catalog entry');
  await page.locator('[data-testid="answer-error"]').waitFor();
  assert.equal(await page.locator('[data-testid="answer"]').count(),0,'a local evidence fallback must not appear as a valid answer');
  const raw=page.getByText(/RAW_FAILURE_EVIDENCE/);assert.equal(await raw.isVisible(),false);
  await page.getByText('查看检索文字（未生成有效答案）',{exact:true}).click();assert.equal(await raw.isVisible(),true);
  assert.equal(postCount,5,'loading an old failed answer must never submit it again');
  // As an answer grows, readers at the newest turn should keep seeing it.
  // Readers scrolling back through older turns must keep their position.
  savedHistory.get('c').push(...Array.from({length:20},(_,i)=>({turn_id:`older-${i}`,session_id:'c',question:`更早的问题 ${i}`,answer:`更早的回答 ${i}`,status:'done',created_at:`2026-11-01T00:${String(i).padStart(2,'0')}:00Z`})));
  await page.getByRole('button',{name:'刷新记录',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.turn').length===21);
  uncertain=false;
  let releaseGrowth;delayedA=new Promise(resolve=>{releaseGrowth=resolve;});
  await page.getByLabel('向所选会话提问').fill('验证新回答自动跟随');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText('正在等待真实问答结果…',{exact:true}).waitFor();
  releaseGrowth();await page.getByText('真实回答 6',{exact:true}).waitFor();
  assert.ok(await page.locator('#conversation').evaluate(c=>c.scrollHeight-c.clientHeight-c.scrollTop<4),'growing latest answer must remain in view');
  let releaseReading;delayedA=new Promise(resolve=>{releaseReading=resolve;});
  await page.getByLabel('向所选会话提问').fill('验证阅读旧记录时不跳动');
  await page.getByRole('button',{name:'发送问题',exact:true}).click();
  await page.getByText('正在等待真实问答结果…',{exact:true}).waitFor();
  await page.locator('#conversation').evaluate(c=>{c.scrollTop=0;});
  releaseReading();await page.getByText('真实回答 7',{exact:true}).waitFor({state:'attached'});
  assert.equal(await page.locator('#conversation').evaluate(c=>c.scrollTop),0,'new answer must not pull a reader away from old history');
  assert.equal(postCount,7);
  // Editing happens inside the video page and must not reset the player,
  // question draft, or conversation. Network uncertainty reuses one receipt.
  page.on('dialog',dialog=>dialog.accept());
  await page.getByLabel('向所选会话提问').fill('编辑记忆时仍保留的问题草稿');
  await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
  await page.locator('video').evaluate(v=>{v.pause();v.currentTime=1.25;window.editVideoLoads=0;v.addEventListener('loadstart',()=>window.editVideoLoads++);});
  assert.equal(await page.locator('#edit-memory').isDisabled(),true,'ordinary presentation mode must not expose editing');
  await page.locator('#admin-mode-toggle').click();
  await page.locator('#admin-code').fill('lightmem-admin');
  await page.getByRole('button',{name:'进入管理员模式',exact:true}).last().click();
  await page.getByRole('button',{name:'编辑长期记忆',exact:true}).waitFor();
  await page.locator('#edit-memory').click();
  assert.equal(new URL(page.url()).pathname,'/history-demo/');
  const editField=page.locator('#memory-edit-fields textarea').first();
  await editField.fill('用户修订的事件描述');
  await page.locator('#memory-edit-close').click();await page.locator('#edit-memory').click();
  assert.equal(await editField.inputValue(),'用户修订的事件描述');
  await page.locator('#memory-edit-save').click();
  await page.getByText(/保存成功，新版本已发布/).waitFor();
  assert.equal(memoryWrites.length,1);assert.equal(memoryWrites[0].sid,'c');
  assert.equal(memoryStates.get('c').memory_version,4);
  assert.equal(await page.getByLabel('向所选会话提问').inputValue(),'编辑记忆时仍保留的问题草稿');
  assert.ok(Math.abs(await page.locator('video').evaluate(v=>v.currentTime)-1.25)<.1);
  assert.equal(await page.evaluate(()=>window.editVideoLoads),0);
  assert.equal(await page.locator('.turn').count(),23);
  await page.locator('#memory-edit-undo').click();await page.getByText(/已撤销上次修改/).waitFor();
  assert.equal(memoryStates.get('c').memory_version,5);assert.equal(memoryStates.get('c').canRollback,false);
  loseMemoryReceipt=true;await editField.fill('保存成功但回执丢失的修改');await page.locator('#memory-edit-save').click();
  await page.locator('#memory-edit-retry').waitFor();const lostReceipt=structuredClone(memoryWrites.at(-1));
  await page.reload();await page.locator('#edit-memory').click();await page.locator('#memory-edit-retry').waitFor();
  assert.equal(memoryWrites.length,3,'refresh never resubmits a memory edit automatically');
  await page.locator('#memory-edit-retry').click();await page.getByText(/保存成功，新版本已发布/).waitFor();
  assert.deepEqual(memoryWrites.at(-1),lostReceipt,'retry keeps exactly the same key and body');
  assert.equal(memoryStates.get('c').memory_version,6,'replayed receipt must not create another version');
  await editField.fill('冲突时必须保留的草稿');
  memoryStates.get('c').memory_version=7;memoryStates.get('c').episodic['30sec'][0].content='另一个标签页的修改';
  await page.locator('#memory-edit-save').click();await page.locator('#memory-edit-latest').waitFor();
  assert.equal(await editField.inputValue(),'冲突时必须保留的草稿');
  assert.equal(await page.locator('#memory-edit-save').isDisabled(),true);
  await page.locator('#memory-edit-latest').click();await page.locator('#memory-edit-review').waitFor();
  assert.equal(await editField.inputValue(),'冲突时必须保留的草稿');
  await page.locator('#memory-editor').getByText('另一个标签页的修改',{exact:true}).waitFor();
  await page.locator('#memory-edit-review').click();await page.locator('#memory-edit-save').click();
  await page.getByText(/保存成功，新版本已发布/).waitFor();assert.equal(memoryWrites.at(-1).body.baseVersion,7);
  assert.equal(postCount,7,'memory editing must not submit or replay questions');
  await page.screenshot({path:fileURLToPath(new URL('artifacts/memory-editor.png',import.meta.url)),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:fileURLToPath(new URL('artifacts/memory-editor-mobile.png',import.meta.url)),fullPage:true});
  await page.locator('#memory-edit-close').click();
  await page.setViewportSize({width:1280,height:720});
  await page.screenshot({path:fileURLToPath(new URL('artifacts/desktop.png',import.meta.url)),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:fileURLToPath(new URL('artifacts/mobile.png',import.meta.url)),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'mobile page must fit viewport');
  // Appearance changes keep the live DOM's video, user draft and backend text.
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#question').fill('不要翻译或丢掉这个问题草稿');
  const appearanceVideo=await page.locator('#video').getAttribute('src');
  const appearanceTime=await page.locator('#video').evaluate(video=>video.currentTime);
  await page.locator('#language-toggle').click();
  await page.getByRole('heading',{name:'Video & Ask',exact:true}).waitFor();
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  assert.equal(await page.locator('#send').innerText(),'Send question');
  assert.match(await page.locator('#catalog-status').innerText(),/3 sessions loaded/);
  assert.match(await page.locator('#media-detail').innerText(),/Duration 00:03/);
  assert.match(await page.locator('#coverage').innerText(),/Input received through/);
  assert.equal(await page.locator('#question').inputValue(),'不要翻译或丢掉这个问题草稿');
  assert.equal(await page.locator('#video').getAttribute('src'),appearanceVideo);
  assert.ok(Math.abs(await page.locator('#video').evaluate(video=>video.currentTime)-appearanceTime)<.1);
  assert.ok((await page.locator('#conversation').innerText()).includes('真实回答 7'));
  await page.locator('#theme-toggle').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  assert.equal(await page.evaluate(()=>localStorage.getItem('lightmem-ego.ui-theme.v1')),'dark');
  // A new upload is not a question and remains gated while memory is building.
  await page.locator('#show-upload').click();
  await page.locator('#video-upload-file').setInputFiles(clip);
  await page.locator('#video-upload-submit').click();
  await page.waitForFunction(()=>document.querySelector('#video-upload-status').textContent.includes('Building long-term memories'));
  assert.equal(await page.locator('#video-upload-open').isHidden(),true);
  assert.equal(uploadPosts,1);assert.equal(postCount,7);
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('#video-upload-status').textContent.includes('Building long-term memories'));
  assert.equal(uploadPosts,1,'refresh checks the receipt and never reuploads');
  assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  uploadReady=true;
  await page.locator('#video-upload-check').click();
  await page.locator('#video-upload-open').waitFor();
  assert.equal(await page.locator('#video-upload-memories').getAttribute('href'),`/memories/?memory_session=video_${uploadedId}`);
  await page.screenshot({path:fileURLToPath(new URL('artifacts/upload-ready-dark-en.png',import.meta.url)),fullPage:true});
  await page.locator('#video-upload-open').click();
  assert.equal(await page.locator('#session-id').innerText(),`video_${uploadedId}`);
  await page.getByLabel('Ask about this session').fill('What happened in the uploaded video?');
  await page.getByRole('button',{name:'Send question',exact:true}).click();
  await page.getByText('真实回答 8',{exact:true}).waitFor();
  assert.equal(uploadPosts,1);assert.equal(postCount,8);
  await page.locator('#language-toggle').click();await page.locator('#theme-toggle').click();
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'upload page must fit mobile in Chinese');
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:fileURLToPath(new URL('artifacts/upload-mobile-zh.png',import.meta.url)),fullPage:true});
  assert.deepEqual(errors,[]);
  console.log('PASS offline browser: catalog, media, QA, inline memory save/undo, uncertain-write recovery, version conflicts, preserved video/question, mobile');
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
