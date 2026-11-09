import {apiPath,clock,seconds,playbackSeconds,generationLabel,requestJSON,QueryController,AdminMode,isLocalSummary,failedAnswer,answerFailureMessage} from './core.js';
import {MemoryEditor,characterCount,memoryRows} from './memory-editor.js';
import {translate,readLanguage,LANGUAGE_STORAGE_KEY} from './i18n.js';
import {VideoUploadController} from './video-upload.js';

const $=id=>document.getElementById(id);
const el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);if(className)node.className=className;return node;};
let language=readLanguage();
const t=(text,values)=>translate(language,text,values);
function uiText(node,text){node.dataset.uiText=String(text??'');node.textContent=t(text);return node;}
function uiEl(tag,text,className){const node=el(tag,undefined,className);if(text!==undefined)uiText(node,text);return node;}
function verbatimText(node,text){delete node.dataset.i18n;delete node.dataset.uiText;node.textContent=String(text??'');return node;}
const button=(text,handler,className='secondary')=>{const node=uiEl('button',text,className);node.type='button';node.addEventListener('click',handler);return node;};
const data={items:[],cursor:null,hasMore:false,selected:null,epoch:0,memories:null,media:null,loadingCatalog:false};
const mediaRequests=new Map();
const playbackPositions=new Map();
let activeVideoSession=null,restorePlayback=null,playAfterRestore=false,recoveringVideo=false;
let videoLoadId=crypto.randomUUID();
const controller=new QueryController({onChange:sid=>{if(sid===data.selected)renderConversation();}});
let editorFormKey='';
const memoryEditor=new MemoryEditor({
  onChange:sid=>{if(sid===data.selected)renderMemoryEditor();},
  onPublished:(sid,memory)=>{if(sid===data.selected){data.memories=memory;renderMemories();}},
});
const adminMode=new AdminMode({onChange:()=>{renderAdminMode();renderMemoryEditor();}});
const videoUpload=new VideoUploadController({onChange:()=>renderUpload()});

function renderUpload(){
  const state=videoUpload.state,busy=['uploading','processing','uncertain'].includes(state.phase);
  $('video-upload-file').disabled=busy;
  $('video-upload-submit').disabled=busy;
  $('video-upload-result').hidden=state.phase==='idle';
  $('video-upload-filename').textContent=state.filename||'';
  $('video-upload-progress').value=Number(state.progress)||0;
  $('video-upload-check').hidden=!state.upload_id||videoUpload.canOpen();
  $('video-upload-resume').hidden=state.phase!=='failed';
  $('video-upload-new').hidden=!['ready','failed','rejected','not_found'].includes(state.phase);
  $('video-upload-open').hidden=!videoUpload.canOpen();
  $('video-upload-memories').hidden=!videoUpload.canOpen();
  if(videoUpload.canOpen())$('video-upload-memories').href=memoryCatalogURL(state.session_id);
  $('video-upload-result').classList.toggle('is-error',['failed','rejected','not_found'].includes(state.phase));
  const stages={queued:'上传完成，正在等待后台处理。',validating:'正在校验视频',transcoding:'正在优化播放格式',
    preprocessing:'正在提取视频和语音',evidence:'正在生成事件描述',memory:'正在构建长期记忆',indexing:'正在准备检索索引'};
  const text=videoUpload.canOpen()?'记忆已就绪，可以回看提问。'
    :state.phase==='uploading'?`正在上传 ${state.progress||0}%`
    :state.phase==='uncertain'?'网络中断，正在核对原上传。'
    :state.phase==='failed'?`处理失败：${state.error||''}`
    :state.phase==='rejected'||state.phase==='not_found'?state.error
    :state.checkError?'上传状态暂时不可用，已保留记录。':stages[state.stage]||'尚未开始上传';
  uiText($('video-upload-status'),text);
}
$('show-upload').addEventListener('click',()=>{$('upload-panel').hidden=!$('upload-panel').hidden;if(!$('upload-panel').hidden)$('upload-panel').scrollIntoView({block:'start',behavior:'smooth'});});
$('hide-upload').addEventListener('click',()=>{$('upload-panel').hidden=true;});
$('video-upload-form').addEventListener('submit',event=>{event.preventDefault();videoUpload.submit($('video-upload-file').files[0]);});
$('video-upload-check').addEventListener('click',()=>videoUpload.check());
$('video-upload-resume').addEventListener('click',()=>videoUpload.resume());
$('video-upload-new').addEventListener('click',()=>{videoUpload.reset();$('video-upload-file').value='';});
$('video-upload-open').addEventListener('click',async()=>{
  if(!videoUpload.canOpen())return;
  const {session_id,filename}=videoUpload.state;
  const item=data.items.find(item=>item.session_id===session_id)||{session_id,title:filename.replace(/\.[^.]+$/,'')};
  if(!data.items.includes(item))data.items.unshift(item);
  await selectSession(item);
});
function recoverUpload(){if(!document.hidden&&videoUpload.state.upload_id&&!videoUpload.canOpen())videoUpload.check();}
window.addEventListener('focus',recoverUpload);
window.addEventListener('pagehide',()=>videoUpload.stop());
if(videoUpload.state.upload_id){$('upload-panel').hidden=false;recoverUpload();}
renderUpload();

function savedPlayback(sid){
  if(playbackPositions.has(sid))return playbackPositions.get(sid);
  let value=0;try{value=seconds(sessionStorage.getItem(`lightmem-video.position.v1:${sid}`))??0;}catch{}
  playbackPositions.set(sid,value);return value;
}
function rememberPlayback(allowZero=false){
  const video=$('video'),sid=activeVideoSession;
  if(!sid||restorePlayback!==null||video.error||video.seeking)return;
  const time=seconds(video.currentTime);if(time===null||(!allowZero&&time===0))return;
  const previous=playbackPositions.get(sid);playbackPositions.set(sid,time);
  if(previous===undefined||Math.floor(previous)!==Math.floor(time)){
    try{sessionStorage.setItem(`lightmem-video.position.v1:${sid}`,String(time));}catch{}
  }
}
async function resumeVideo(){
  if(recoveringVideo||!data.selected)return;
  rememberPlayback();const sid=data.selected,epoch=data.epoch,position=savedPlayback(sid);
  recoveringVideo=true;$('resume-video').disabled=true;
  uiText($('video-status'),`正在从 ${clock(position)} 恢复播放…`);
  try{
    mediaRequests.delete(sid);const media=await getMedia(sid);
    if(epoch!==data.epoch||sid!==data.selected)return;
    if(!media.available)throw new Error('视频暂时不可用');
    data.media=media;restorePlayback=position;playAfterRestore=true;videoLoadId=crypto.randomUUID();
    renderMedia();$('video').load();
  }catch(error){
    if(epoch!==data.epoch)return;
    restorePlayback=null;playAfterRestore=false;
    uiText($('video-status'),`暂时无法恢复，已保留 ${clock(position)}。${error.message}`);
    $('resume-video').hidden=false;
  }finally{if(epoch===data.epoch){recoveringVideo=false;$('resume-video').disabled=false;}}
}
$('resume-video').addEventListener('click',resumeVideo);

function themePreference(){try{return localStorage.getItem('lightmem-ego.ui-theme.v1');}catch{return null;}}
function updateThemeButton(){const dark=document.documentElement.dataset.theme==='dark',label=t(dark?'切换至浅色主题':'切换至深色主题');$('theme-toggle').setAttribute('aria-label',label);$('theme-toggle').title=label;$('theme-toggle').setAttribute('aria-pressed',String(dark));$('theme-moon').toggleAttribute('hidden',dark);$('theme-sun').toggleAttribute('hidden',!dark);}
function applyTheme(){const saved=themePreference();const theme=['dark','light'].includes(saved)?saved:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.dataset.theme=theme;document.documentElement.style.colorScheme=theme;document.querySelector('meta[name="theme-color"]').content=theme==='dark'?'#0e1015':'#f5f5f7';updateThemeButton();}
applyTheme();
window.addEventListener('storage',event=>{if(event.key==='lightmem-ego.ui-theme.v1')applyTheme();});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',applyTheme);
$('theme-toggle').addEventListener('click',()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('lightmem-ego.ui-theme.v1',theme);}catch{}document.documentElement.dataset.theme=theme;document.documentElement.style.colorScheme=theme;document.querySelector('meta[name="theme-color"]').content=theme==='dark'?'#0e1015':'#f5f5f7';updateThemeButton();});
function applyFont(){let value='normal';try{value=localStorage.getItem('lightmem-ego.ui-font-size.v1')||'normal';}catch{}document.documentElement.dataset.fontScale=['small','normal','large'].includes(value)?value:'normal';}
applyFont();
$('font-toggle').addEventListener('click',()=>{const choices=['normal','large','small'],value=choices[(choices.indexOf(document.documentElement.dataset.fontScale)+1)%3];try{localStorage.setItem('lightmem-ego.ui-font-size.v1',value);}catch{}document.documentElement.dataset.fontScale=value;});
window.addEventListener('storage',event=>{if(event.key==='lightmem-ego.ui-font-size.v1')applyFont();});

function renderAdminMode(){
  const unlocked=adminMode.isUnlocked(),toggle=$('admin-mode-toggle');
  document.documentElement.dataset.adminMode=unlocked?'unlocked':'locked';
  uiText($('admin-mode-label'),unlocked?'退出管理员模式':'管理员模式');
  uiText($('admin-mode-state'),unlocked?'已解锁':'普通浏览');
  toggle.classList.toggle('is-unlocked',unlocked);
  toggle.setAttribute('aria-expanded',String(unlocked));
  toggle.setAttribute('aria-label',t(unlocked?'退出管理员模式':'进入管理员模式'));
  toggle.title=t(unlocked?'退出管理员模式':'进入管理员模式后才能编辑记忆');
}
function closeAdminDialog(){
  const dialog=$('admin-dialog');
  if(dialog.open)dialog.close();
  $('admin-code').value='';uiText($('admin-status'),'');$('admin-status').className='admin-status';
  $('admin-mode-toggle').setAttribute('aria-expanded',String(adminMode.isUnlocked()));
}
$('admin-mode-toggle').addEventListener('click',()=>{
  if(adminMode.isUnlocked()){adminMode.lock();return;}
  const dialog=$('admin-dialog');$('admin-code').value='';uiText($('admin-status'),'');
  if(typeof dialog.showModal==='function')dialog.showModal();else dialog.open=true;
  queueMicrotask(()=>$('admin-code').focus());
});
$('admin-form').addEventListener('submit',event=>{
  event.preventDefault();
  if(adminMode.unlock($('admin-code').value)){closeAdminDialog();return;}
  uiText($('admin-status'),'解锁码不正确，请重试。');$('admin-status').className='admin-status is-error';$('admin-code').select();
});
$('admin-cancel').addEventListener('click',closeAdminDialog);
$('admin-dialog').addEventListener('cancel',event=>{event.preventDefault();closeAdminDialog();});
renderAdminMode();
const dateLabel=value=>{const date=new Date(value);return Number.isNaN(date.getTime())?'':date.toLocaleString(language==='zh'?'zh-CN':'en-GB',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});};
function memoryCatalogURL(sid){
  const value=String(sid||'');
  if(!/^[a-zA-Z0-9_-]{1,200}$/.test(value))return '/memories/';
  const url=new URL('/memories/',location.href);url.searchParams.set('memory_session',value);
  return `${url.pathname}${url.search}${url.hash}`;
}

// Accept same-origin media and the explicitly configured API origin. Backend
// URLs are prefixed once; server-returned content stays plain text.
function mediaURL(value){
  if(typeof value!=='string'||!value.trim())return null;
  try{const path=value.startsWith('/session/')||value.startsWith('/api/')?apiPath(value):value;const url=new URL(path,location.origin);const apiOrigin=new URL(apiPath('/'),location.origin).origin;return [location.origin,apiOrigin].includes(url.origin)&&['http:','https:'].includes(url.protocol)?url.href:null;}catch{return null;}
}
async function get(path){return requestJSON(fetch.bind(window),apiPath(path));}
function getMedia(sid){
  if(!mediaRequests.has(sid)){
    const request=get(`/session/${encodeURIComponent(sid)}/demo-media`).catch(error=>{if(mediaRequests.get(sid)===request)mediaRequests.delete(sid);throw error;});
    mediaRequests.set(sid,request);
  }
  return mediaRequests.get(sid);
}

function renderCatalog(){
  const search=$('session-search').value.trim().toLowerCase();
  const visible=data.items.filter(item=>!search||`${item.title||''} ${item.demo_title||''} ${item.session_id}`.toLowerCase().includes(search));
  $('sessions').replaceChildren(...visible.map(item=>{
    const node=button('',()=>selectSession(item),'session-button');node.dataset.sessionId=item.session_id;node.setAttribute('aria-pressed',String(item.session_id===data.selected));
    node.title=item.session_id;node.append(el('span',item.title||item.demo_title||item.session_id));
    if(item.updated_at)node.append(uiEl('small',`更新于 ${dateLabel(item.updated_at)}`));return node;
  }));
  if(!visible.length)$('sessions').append(uiEl('p',search?'已载入会话中没有匹配项，可继续加载更多。':'暂无可用会话。','muted'));
  $('load-more').hidden=!data.hasMore;$('load-more').disabled=data.loadingCatalog;
}
$('session-search').addEventListener('input',renderCatalog);
async function loadCatalog(reset=false){
  if(data.loadingCatalog)return;
  data.loadingCatalog=true;uiText($('catalog-status'),'正在读取会话…');$('refresh').disabled=true;renderCatalog();
  try{
    const result=await get(`/memories/sessions${!reset&&data.cursor?`?cursor=${encodeURIComponent(data.cursor)}`:''}`);
    if(result.status!=='ok'||!Array.isArray(result.items))throw new Error(result.message||'会话目录格式无效');
    const items=reset?[]:data.items;const seen=new Set(items.map(i=>i.session_id));
    for(const item of result.items)if(item.session_id&&!seen.has(item.session_id)){items.push(item);seen.add(item.session_id);}
    data.items=items;data.cursor=result.next_cursor;data.hasMore=Boolean(result.has_more&&result.next_cursor);
    uiText($('catalog-status'),items.length?`已载入 ${items.length} 个会话`:'暂时没有已就绪的会话。');
    for(const item of items.filter(item=>!item.title&&!item.demo_title)){
      const request=getMedia(item.session_id);
      request.then(media=>{
        if(mediaRequests.get(item.session_id)!==request||!data.items.includes(item))return;
        if(typeof media.title==='string'&&media.title.trim()){item.demo_title=media.title;renderCatalog();}
      }).catch(()=>{});
    }
  }catch(error){uiText($('catalog-status'),`目录读取失败：${error.message}。可点击刷新重试。`);}
  finally{data.loadingCatalog=false;$('refresh').disabled=false;renderCatalog();}
}
$('load-more').addEventListener('click',()=>loadCatalog());
$('refresh').addEventListener('click',async()=>{
  if(data.loadingCatalog)return;
  const sid=data.selected,epoch=data.epoch;
  mediaRequests.clear();for(const item of data.items)delete item.demo_title;
  await loadCatalog(true);
  if(sid&&epoch===data.epoch)await loadSelectedMedia(sid,epoch);
});

async function selectSession(item){
  if(data.selected===item.session_id)return;
  rememberPlayback();activeVideoSession=null;restorePlayback=null;playAfterRestore=false;recoveringVideo=false;
  videoLoadId=crypto.randomUUID();
  uiText($('video-status'),'');$('resume-video').hidden=true;$('resume-video').disabled=false;
  if(data.selected)controller.stop(data.selected);
  data.selected=item.session_id;data.memories=null;data.media=null;
  try{localStorage.setItem('lightmem-ego.history-session.v1',data.selected);}catch{}
  renderMemoryEditor();
  const epoch=++data.epoch,sid=data.selected;
  const url=new URL(location.href);url.searchParams.set('session',sid);history.replaceState(null,'',url);
  renderCatalog();verbatimText($('session-title'),item.title||item.demo_title||sid);$('session-id').textContent=sid;
  const video=$('video');video.pause();video.removeAttribute('src');video.load();video.hidden=true;
  $('media-message').hidden=false;uiText($('media-message'),'正在读取视频信息…');uiText($('media-detail'),'');
  uiText($('playback'),'播放位置：—');uiText($('coverage'),'已生成记忆覆盖：读取中');
  uiText($('memory-status'),'正在读取已有记忆…');$('memory-records').replaceChildren();uiText($('memory-count'),'—');
  $('question').value='';renderConversation();
  // Let the browser buffer video while the larger memory/history payloads load.
  const mediaLoading=loadSelectedMedia(sid,epoch);
  controller.loadHistory(sid).then(()=>{if(data.selected===sid)controller.checkPending(sid);});
  try{
    const memories=await get(`/session/${encodeURIComponent(sid)}/memories`);
    if(epoch!==data.epoch)return;
    if(memories.status!=='ok')throw new Error(memories.message||'记忆尚未就绪');
    data.memories=memories;memoryEditor.receive(sid,memories);renderMemories();
  }catch(error){if(epoch!==data.epoch)return;uiText($('memory-status'),`记忆读取失败：${error.message}`);$('memory-records').append(button('重新读取',()=>{data.selected=null;selectSession(item);},'secondary retry'));}
  if(epoch!==data.epoch)return;
  await mediaLoading;
}
async function loadSelectedMedia(sid,epoch){
  const request=getMedia(sid);
  try{
    const media=await request;
    if(epoch!==data.epoch||mediaRequests.get(sid)!==request)return;
    data.media=media;renderMedia();renderMemories();renderConversation();
  }catch(error){if(epoch!==data.epoch||(mediaRequests.has(sid)&&mediaRequests.get(sid)!==request))return;$('media-message').hidden=false;uiText($('media-message'),`视频信息读取失败：${error.message}。仍可查看记忆和提问。`);uiText($('coverage'),'已生成记忆覆盖：未提供');}
}

function renderMedia(){
  const media=data.media;if(!media)return;
  const item=data.items.find(item=>item.session_id===data.selected);
  verbatimText($('session-title'),item?.title||media.title||data.selected);
  let url=media.available?mediaURL(media.video_url):null;
  if(url){const requestURL=new URL(url);requestURL.searchParams.set('playback',videoLoadId);url=requestURL.href;}
  // Reuse the URL during playback/metadata refresh. A new player load gets a
  // fresh cache entry, avoiding locks left by an interrupted large range stream.
  if(url){
    const video=$('video');video.hidden=false;
    if(video.getAttribute('src')!==url){
      rememberPlayback();
      if(restorePlayback===null)restorePlayback=savedPlayback(data.selected);
      playAfterRestore=playAfterRestore||(!video.paused&&activeVideoSession===data.selected);
      activeVideoSession=data.selected;video.src=url;
    }
    $('media-message').hidden=true;
  }
  else{const video=$('video');video.pause();video.hidden=true;if(video.hasAttribute('src')){video.removeAttribute('src');video.load();}uiText($('playback'),'播放位置：—');$('media-message').hidden=false;uiText($('media-message'),`${media.reason||'本会话没有可播放的视频'}。仍可查看记忆和提问。`);}
  const details=[media.view_description,media.duration_seconds!==null&&media.duration_seconds!==undefined?t(`视频时长 ${clock(media.duration_seconds)}`):null];
  if(!media.time_alignment_known)details.push(t('时间轴对齐未确认，证据时间不跳转视频'));
  uiText($('media-detail'),details.filter(Boolean).join(' · '));
  uiText($('coverage'),`已生成记忆覆盖：${seconds(media.memory_until_seconds)===null?t('未提供'):t(`已输入至 ${clock(media.memory_until_seconds)}`)}（与播放位置独立）`);
}
$('video').addEventListener('timeupdate',()=>{
  rememberPlayback();uiText($('playback'),`播放位置：${clock($('video').currentTime)}`);
});
function finishPlaybackRestore(){
  restorePlayback=null;rememberPlayback(true);
  if(playAfterRestore){playAfterRestore=false;$('video').play().catch(()=>{uiText($('video-status'),'已恢复播放位置，点击播放即可继续。');});}
}
$('video').addEventListener('loadedmetadata',()=>{
  const video=$('video');if(activeVideoSession!==data.selected)return;
  const position=restorePlayback??savedPlayback(data.selected);
  if(position>0&&Number.isFinite(video.duration)&&position<video.duration-.1){
    restorePlayback=position;video.currentTime=position;
  }else finishPlaybackRestore();
});
$('video').addEventListener('seeked',()=>{if(restorePlayback!==null)finishPlaybackRestore();else rememberPlayback(true);});
$('video').addEventListener('ended',()=>{
  if(!activeVideoSession)return;playbackPositions.set(activeVideoSession,0);
  try{sessionStorage.removeItem(`lightmem-video.position.v1:${activeVideoSession}`);}catch{}
});
$('video').addEventListener('waiting',()=>{uiText($('video-status'),'正在缓冲，播放位置已保留。');});
$('video').addEventListener('playing',()=>{uiText($('video-status'),'');$('resume-video').hidden=true;});
$('video').addEventListener('play',()=>{if($('video').error)resumeVideo();});
$('video').addEventListener('error',()=>{
  if(!$('video').getAttribute('src'))return;
  restorePlayback=null;playAfterRestore=false;
  uiText($('video-status'),`视频连接或解码中断，已保留 ${clock(savedPlayback(data.selected))}，可从这里继续。`);
  $('resume-video').hidden=false;
});
function seek(frame){
  const target=playbackSeconds(frame,data.media),video=$('video');
  if(target===null||video.hidden||!video.getAttribute('src'))return;
  if(Number.isFinite(video.duration)&&target>video.duration)return;
  video.currentTime=target;video.scrollIntoView({behavior:'smooth',block:'center'});video.focus();
}
function seekButton(frame){
  const target=playbackSeconds(frame,data.media);const available=target!==null&&data.media?.available;
  const capture=frame.capture_time_seconds??frame.timestamp_seconds??frame.timestamp??frame.start_seconds??frame.start;
  const end=seconds(frame.end_seconds??frame.end),hasRange=end!==null&&seconds(capture)!==null&&end>=seconds(capture);
  const endTarget=hasRange?playbackSeconds({capture_time_seconds:end,owner_session_id:frame.owner_session_id},data.media):null;
  const playbackTime=`${clock(target)}${endTarget!==null?`–${clock(endTarget)}`:''}`;
  const evidenceTime=`${clock(capture)}${hasRange?`–${clock(end)}`:''}`;
  const node=button(available?`跳转视频 · ${playbackTime}`:`证据时间 ${evidenceTime}`,()=>seek(frame));
  node.className='memory-time';node.dataset.testid='evidence-time';node.disabled=!available;
  if(!available)node.title=frame.owner_session_id&&frame.owner_session_id!==data.selected?'证据来自其他会话，不能跳转当前视频。':'时间轴未确认、超出视频范围或本会话没有视频；保留原证据时间。';return node;
}
function renderMemories(){
  const memories=data.memories;if(!memories)return;
  uiText($('memory-status'),`已有记忆 · 版本 ${memories.memory_version??'未提供'}${memories.updated_at?` · ${memories.updated_at}`:''}${memories.field_truncation?' · 部分原始字段已截断':''}`);
  const groups=[['30 秒情景','30sec'],['3 分钟情景','3min'],['10 分钟情景','10min'],['1 小时情景','1h']].map(([name,key])=>[name,memories.episodic?.[key]||[]]);
  groups.push(['语义记忆',memories.semantic||[]],['视觉记忆',memories.visual||[]]);
  uiText($('memory-count'),`${groups.reduce((n,[,rows])=>n+rows.length,0)} 条`);
  $('memory-records').replaceChildren(...groups.map(([name,rows])=>{
    const group=el('section',undefined,'memory-group');group.append(el('h3',`${t(name)} · ${rows.length}`));
    if(!rows.length)group.append(uiEl('p','暂无记录','memory-empty'));
    for(const row of rows){const record=el('article',undefined,'memory-row'),time=row.event_time||{};
      const capture=time.timestamp_seconds??time.start_seconds;
      if(seconds(capture)!==null)record.append(seekButton({capture_time_seconds:capture}));
      const text=String(row.content||'此记录没有文字内容');
      record.append(el('p',text.length>300?`${text.slice(0,300)}…`:text));
      if(text.length>300){const details=el('details',undefined,'memory-full');details.append(uiEl('summary','完整记忆内容'),el('p',text));record.append(details);}
      group.append(record);
    }return group;
  }));
}

function renderMemoryEditor(){
  const sid=data.selected;if(!sid)return;
  const state=memoryEditor.state(sid),change=memoryEditor.changes(sid),busy=state.phase==='saving'||state.phase==='loading';
  const memoryLink=$('view-memory-session');memoryLink.hidden=false;memoryLink.href=memoryCatalogURL(sid);memoryLink.title='在主站记忆目录中打开当前会话';
  const unlocked=adminMode.isUnlocked(),hasDraft=change.records.length>0;
  const available=unlocked&&Boolean(data.memories?.canEdit||state.pending||hasDraft);
  $('edit-memory').hidden=false;$('edit-memory').disabled=!available;
  uiText($('edit-memory'),!unlocked?'进入管理员模式后编辑':state.open?'编辑已展开':state.pending?'查看保存进度':hasDraft?'继续编辑草稿':'编辑长期记忆');
  $('edit-memory').title=!unlocked?'请先点击顶部“管理员模式”并解锁':available?'在本页修改所选会话的长期记忆':'此会话暂时没有可编辑的 30 秒事件';
  $('edit-memory').setAttribute('aria-expanded',String(state.open));
  $('memory-editor').hidden=!state.open||!unlocked;$('memory-records').hidden=state.open&&unlocked;
  if(!state.open||!unlocked)return;
  uiText($('memory-editor-version'),`草稿基于版本 ${state.baseVersion??'—'} · 已发布版本 ${state.memory?.memory_version??'—'}`);
  const formKey=`${sid}:${state.revision}`;
  if(editorFormKey!==formKey){
    editorFormKey=formKey;
    const rows=memoryRows(state.memory);
    $('memory-edit-fields').replaceChildren(...rows.map((row,index)=>{
      const block=el('div',undefined,'memory-edit-record'),time=row.event_time||{};
      const start=seconds(time.start_seconds??time.timestamp_seconds),end=seconds(time.end_seconds);
      const title=start===null?`事件 ${index+1}`:`事件 ${index+1} · ${clock(start)}${end!==null?`–${clock(end)}`:''}`;
      const label=uiEl('label',title),textarea=el('textarea'),counter=el('span',undefined,'memory-edit-character-count');
      textarea.id=`memory-record-${index}`;label.htmlFor=textarea.id;
      textarea.value=state.drafts[row.id]?.content??row.content??'';
      textarea.dataset.memoryRecordId=row.id;textarea.dataset.editable=String(Boolean(row.editable));
      textarea.setAttribute('aria-describedby',`memory-record-count-${index}`);counter.id=`memory-record-count-${index}`;
      textarea.addEventListener('input',()=>memoryEditor.draft(sid,row.id,textarea.value));
      block.append(label,textarea,counter);
      if(!row.editable)block.append(uiEl('p','此记录暂时不可编辑。','muted'));
      if(state.conflict&&state.drafts[row.id]){
        const compare=el('details',undefined,'memory-edit-compare');
        compare.append(uiEl('summary','核对当前已保存的内容'),el('p',row.content||''));compare.open=true;block.append(compare);
      }
      return block;
    }));
    for(const [id,draft] of Object.entries(state.drafts))if(!rows.some(row=>row.id===id)){
      const orphan=el('details',undefined,'memory-edit-compare');orphan.open=true;
      orphan.append(uiEl('summary','原记录已不存在，保留的草稿'),el('p',draft.content));$('memory-edit-fields').append(orphan);
    }
  }
  for(const field of $('memory-edit-fields').querySelectorAll('textarea')){
    field.disabled=!unlocked||Boolean(state.pending)||busy||field.dataset.editable!=='true';
    const count=characterCount(field.value);field.setAttribute('aria-invalid',String(count>change.perRecord||!field.value.trim()));
    uiText(document.getElementById(field.getAttribute('aria-describedby')),`${count} / ${change.perRecord} 字符`);
  }
  uiText($('memory-edit-count'),`已修改 ${change.records.length} 条 · ${change.total} / ${change.maxTotal} 字符`);
  uiText($('memory-edit-status'),change.error||state.message||'修改后点击保存；收起编辑或切换会话会保留草稿。');
  $('memory-edit-status').className=busy?'muted pending-status':change.error||state.conflict||state.phase==='uncertain'?'muted memory-edit-warning':'muted';
  $('memory-edit-save').disabled=!unlocked||busy||Boolean(state.pending)||state.conflict||!hasDraft||Boolean(change.error)||!state.memory?.canEdit;
  $('memory-edit-retry').hidden=!state.pending||state.phase!=='uncertain';
  $('memory-edit-latest').hidden=!state.conflict;$('memory-edit-latest').disabled=!unlocked||busy;
  $('memory-edit-review').hidden=!state.conflict||state.baseVersion===state.memory?.memory_version;$('memory-edit-review').disabled=!unlocked||busy||Boolean(change.error);
  $('memory-edit-undo').hidden=!state.memory?.canRollback;$('memory-edit-undo').disabled=!unlocked||busy||Boolean(state.pending)||hasDraft;
  $('memory-edit-discard').hidden=!hasDraft||Boolean(state.pending);$('memory-edit-discard').disabled=!unlocked||busy;
}
$('edit-memory').addEventListener('click',()=>{if(!adminMode.isUnlocked()){return;}if(data.selected){$('memory-details').open=true;memoryEditor.open(data.selected);}});
$('memory-edit-form').addEventListener('submit',event=>{event.preventDefault();if(adminMode.isUnlocked()&&data.selected)memoryEditor.save(data.selected);});
$('memory-edit-close').addEventListener('click',()=>{if(data.selected)memoryEditor.close(data.selected);});
$('memory-edit-retry').addEventListener('click',()=>{if(adminMode.isUnlocked()&&data.selected)memoryEditor.retry(data.selected);});
$('memory-edit-latest').addEventListener('click',()=>{if(adminMode.isUnlocked()&&data.selected)memoryEditor.refresh(data.selected);});
$('memory-edit-review').addEventListener('click',()=>{if(adminMode.isUnlocked()&&data.selected)memoryEditor.acceptLatest(data.selected);});
$('memory-edit-discard').addEventListener('click',()=>{if(adminMode.isUnlocked()&&data.selected&&window.confirm('放弃此会话尚未保存的修改？'))memoryEditor.discard(data.selected);});
$('memory-edit-undo').addEventListener('click',()=>{if(adminMode.isUnlocked()&&data.selected&&window.confirm('撤销此会话上一次记忆修改，并发布恢复后的内容？'))memoryEditor.undo(data.selected);});

function answerNode(text){
  const node=el('div',undefined,'answer');node.dataset.testid='answer';
  // Render a small, safe subset of Markdown without accepting HTML from a model.
  for(const paragraph of String(text).split(/\n\s*\n/)){
    const p=el('p');for(const part of paragraph.split(/(\*\*[^*\n]+\*\*)/g))p.append(part.startsWith('**')&&part.endsWith('**')?el('strong',part.slice(2,-2)):document.createTextNode(part));node.append(p);
  }return node;
}
function renderConversation(){
  const sid=data.selected;if(!sid)return;
  const state=controller.state(sid),pending=controller.pending(sid),container=$('conversation');
  const scrollTop=container.scrollTop,last=state.turns.at(-1);
  const followingLatest=container.scrollHeight-container.clientHeight-scrollTop<48;
  const focusKey=String(last?.taskId||last?.turnId||last?.createdAt||'');
  const newTurn=container.dataset.focusKey!==focusKey;container.dataset.focusKey=focusKey;
  $('question').disabled=Boolean(pending);$('send').disabled=Boolean(pending);
  uiText($('ask-hint'),pending?.status==='paused'
    ?state.checkingPending?'正在核对原任务，完成后将自动恢复提问。':'等待已暂停，页面会自动核对结果，也可手动核对。'
    :pending?.status==='uncertain'?'提交结果尚未确认，请展开该问题核对后再提问。'
    :pending?'问题正在处理，请勿重复发送。':'可继续追问，也可说明想了解的时段。');
  $('check-task').hidden=pending?.status!=='paused'||!pending.taskId;
  $('check-task').disabled=state.checkingPending;
  uiText($('conversation-count'),`${state.turns.length} 条对话`);
  uiText($('history-status'),state.historyLoading?'正在读取历史记录…':state.historyError||'已保存的对话 · 展开可查看回答和证据');
  $('refresh-history').disabled=state.historyLoading;
  $('earlier-history').hidden=!state.historyHasMore;$('earlier-history').disabled=state.historyLoading;
  const lazy=[];
  const turns=state.turns.map(turn=>{
    const article=el('details',undefined,'turn');article.dataset.testid='qa-turn';article.open=turn.open!==false;
    const summary=el('summary'),copy=el('span',undefined,'turn-summary-copy');
    copy.append(el('span',turn.question,'turn-question'));
    const stateName={done:'已回答',failed:'生成失败',polling:'处理中',submitting:'提交中',paused:'等待恢复',uncertain:'待核对'}[turn.status]||'已记录';
    copy.append(el('time',`${dateLabel(turn.createdAt)} · ${t(stateName)}`,'turn-time'));
    summary.append(copy,el('span','›','turn-chevron'));article.append(summary);
    article.addEventListener('toggle',()=>{if(!article.isConnected)return;turn.open=article.open;if(article.open&&turn.fromHistory&&!turn.detailLoaded&&!turn.detailLoading&&!turn.detailError)controller.loadTurn(sid,turn);});
    const body=el('div',undefined,'turn-body');article.append(body);
    const status=uiEl('p',turn.message,'turn-status');status.dataset.testid='task-status';status.setAttribute('role','status');
    if(['uncertain','paused'].includes(turn.status))status.classList.add('warning');
    if(['submitting','polling'].includes(turn.status))status.classList.add('pending-status');
    if(turn.message&&!turn.result)body.append(status);
    if(turn.status==='paused')body.append(button('继续等待结果',()=>controller.resume(sid)));
    if(turn.status==='uncertain')body.append(button('已核对，允许新提问',()=>controller.acknowledge(sid)));
    if(turn.partialAnswer&&!turn.result)body.append(answerNode(turn.partialAnswer));
    if(turn.result){
      const result=turn.result,failed=failedAnswer(result)||turn.status==='failed';
      const heading=el('div',undefined,'answer-heading');heading.append(el('span','LightMem-Ego'));
      const label=failed?'生成失败':turn.fromHistory&&!turn.detailLoaded&&!result.generation_status?'历史回答':generationLabel(result);
      const meta=uiEl('span',label,'turn-status generation-meta');
      if(/降级/.test(label))meta.classList.add('warning');if(failed)meta.classList.add('error');heading.append(meta);body.append(heading);
      if(failed){
        const message=uiEl('div',answerFailureMessage(result),'error-banner');message.dataset.testid='answer-error';body.append(message);
        const retry=button('重新填写这个问题',()=>{$('question').value=turn.question;$('question').focus();});retry.disabled=Boolean(pending);body.append(retry);
        const local=result.local_evidence_summary||(isLocalSummary(result)?result.answer:'');
        if(local){const details=el('details',undefined,'failure-details');details.append(uiEl('summary','查看检索文字（未生成有效答案）'),el('pre',local));body.append(details);}
      }else{
        body.append(answerNode(result.answer||result.answer_text||'此记录没有回答正文。'));
        const sources=result.used_memory_sources||result.usedMemorySources||[];
        const sourceNames={M_cur:'当前记忆',M_st:'短期记忆',M_lt:'长期记忆',M_cache:'交互缓存'};
        if(sources.length)body.append(uiEl('p',`记忆来源：${sources.map(v=>t(sourceNames[v]||String(v))).join(' · ')}`,'sources'));
        const frames=Array.isArray(result.evidence_frames)?result.evidence_frames:[];
        if(frames.length){
          const evidence=el('details',undefined,'evidence-section');evidence.open=Boolean(turn.evidenceOpen);
          evidence.append(uiEl('summary',`画面证据（${Math.min(6,frames.length)}张）`));
          const grid=el('div',undefined,'evidence-grid');for(const frame of frames.slice(0,6))grid.append(evidenceCard(frame));evidence.append(grid);body.append(evidence);
          evidence.addEventListener('toggle',()=>{if(evidence.isConnected)turn.evidenceOpen=evidence.open;});
        }
        if(Array.isArray(result.timestamps)&&result.timestamps.length){const times=el('div',undefined,'turn-actions');
          for(const time of result.timestamps.slice(0,20))times.append(seekButton(typeof time==='object'?time:{timestamp:time}));body.append(times);}
      }
      if(turn.detailLoading)body.append(uiEl('p','正在加载这条回答的证据…','muted'));
      if(turn.detailError){body.append(el('p',turn.detailError,'turn-status warning'),button('重新读取证据',()=>controller.loadTurn(sid,turn)));}
      const details=el('details',undefined,'processing-details');details.append(uiEl('summary','处理详情'));
      if(turn.taskId)details.append(uiEl('p',`任务 ID · ${turn.taskId}`,'task-id'));
      const ms=result.latency?.total_ms;if(Number.isFinite(ms))details.append(uiEl('p',`处理耗时 ${(ms/1000).toFixed(1)} 秒`,'muted'));
      body.append(details);
    }
    if(article.open&&turn.fromHistory&&!turn.detailLoaded&&!turn.detailLoading&&!turn.detailError)lazy.push(turn);
    return article;
  });
  container.replaceChildren(...(turns.length?turns:[uiEl('p','记忆已就绪，开始一段对话。\n可以问过去发生的事，也可以继续追问。','empty')]));
  container.scrollTop=newTurn&&turns.length?Math.max(0,turns.at(-1).offsetTop-55)
    :followingLatest?container.scrollHeight:scrollTop;
  for(const turn of lazy)queueMicrotask(()=>controller.loadTurn(sid,turn));
}
$('refresh-history').addEventListener('click',async()=>{const sid=data.selected;if(sid){await controller.loadHistory(sid);if(data.selected===sid)controller.checkPending(sid);}});
$('check-task').addEventListener('click',()=>{if(data.selected)controller.checkPending(data.selected);});
$('earlier-history').addEventListener('click',()=>{if(data.selected)controller.loadHistory(data.selected,true);});
function evidenceCard(frame){
  const card=el('figure',undefined,'evidence-card');
  const url=mediaURL(frame.relative_file_url||frame.image_url||frame.file_url||frame.url||frame.preview_url);
  const description=String(frame.caption||frame.description||'画面证据'),preview=description.length>160?`${description.slice(0,160)}…`:description;
  if(url){const img=el('img');img.src=url;img.alt=preview;img.loading='lazy';img.dataset.testid='evidence-image';
    const link=el('a');link.href=url;link.target='_blank';link.rel='noopener noreferrer';link.setAttribute('aria-label','打开证据原图');
    img.addEventListener('error',()=>{if(!img.hidden){img.hidden=true;card.prepend(uiEl('p','证据图片加载失败','missing-image'));}});link.append(img);card.append(link);}
  else card.append(uiEl('p','证据图片 URL 未提供或不可加载','missing-image'));
  const caption=el('figcaption');caption.append(el('span',preview,'evidence-preview'));
  if(frame.source||frame.source_memory)caption.append(uiEl('div',`来源：${frame.source||frame.source_memory}`));caption.append(seekButton(frame));
  if(description.length>160){const details=el('details',undefined,'evidence-description');details.append(uiEl('summary','完整证据描述'),el('p',description));caption.append(details);}
  card.append(caption);return card;
}
$('ask-form').addEventListener('submit',event=>{event.preventDefault();if(!data.selected||controller.pending(data.selected))return;const question=$('question').value.trim();if(!question)return;$('question').value='';controller.submit(data.selected,question);});
window.addEventListener('pagehide',()=>{rememberPlayback();if(data.selected)controller.stop(data.selected);});
// Recover the original task with GET only, including after a long wait,
// returning to the tab, or restoring a page from the browser's back cache.
function recoverPendingTask(){if(!document.hidden&&data.selected)controller.checkPending(data.selected);}
window.addEventListener('focus',recoverPendingTask);
window.addEventListener('pageshow',recoverPendingTask);
document.addEventListener('visibilitychange',recoverPendingTask);
setInterval(recoverPendingTask,15000);

function applyLanguage(){
  document.documentElement.lang=language==='zh'?'zh-CN':'en';
  try{localStorage.setItem(LANGUAGE_STORAGE_KEY,language);}catch{}
  for(const node of document.querySelectorAll('[data-i18n]'))if(node.dataset.uiText===undefined)uiText(node,node.dataset.i18n);
  for(const node of document.querySelectorAll('[data-ui-text]'))node.textContent=t(node.dataset.uiText);
  for(const [dataKey,attribute] of [['i18nPlaceholder','placeholder'],['i18nLabel','aria-label'],['i18nTitle','title']]){
    for(const node of document.querySelectorAll(`[data-${dataKey.replace(/[A-Z]/g,c=>'-'+c.toLowerCase())}]`))node.setAttribute(attribute,t(node.dataset[dataKey]));
  }
  $('language-toggle').textContent=language==='zh'?'English':'中文';
  updateThemeButton();renderAdminMode();renderCatalog();
  renderUpload();
  if(data.selected){if(data.media)renderMedia();renderMemories();renderMemoryEditor();renderConversation();}
}
$('language-toggle').addEventListener('click',()=>{
  language=language==='zh'?'en':'zh';
  try{localStorage.setItem(LANGUAGE_STORAGE_KEY,language);}catch{}
  applyLanguage();
});
window.addEventListener('storage',event=>{if(event.key===LANGUAGE_STORAGE_KEY||event.key===null){language=readLanguage();applyLanguage();}});
applyLanguage();

await loadCatalog(true);
const requested=new URLSearchParams(location.search).get('session');
if(requested){
  while(!data.items.some(item=>item.session_id===requested)&&data.hasMore){const before=data.items.length;await loadCatalog();if(before===data.items.length)break;}
  const item=data.items.find(item=>item.session_id===requested);
  if(item)await selectSession(item);else uiText($('catalog-status'),'链接中的会话不在已就绪目录，请选择其他会话或刷新。');
}
