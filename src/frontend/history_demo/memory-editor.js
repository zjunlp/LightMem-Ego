import {apiPath} from './core.js';

const version = memory => memory?.memory_version ?? memory?.memoryVersion;
export const characterCount = text => [...String(text)].length;
export const memoryRows = memory => memory?.episodic?.['30sec'] || [];

// Each session keeps its own draft and mutation receipt. A lost connection is
// not proof that a write failed: only the same idempotency key may be retried.
export class MemoryEditor {
  constructor({fetch=globalThis.fetch.bind(globalThis),storage=globalThis.sessionStorage,
    onChange=()=>{},onPublished=()=>{},newKey=()=>crypto.randomUUID(),now=Date.now}={}) {
    Object.assign(this,{fetch,storage,onChange,onPublished,newKey,now});
    this.sessions=new Map();
  }
  key(sid){return `lightmem-history.memory-edit.v1:${sid}`;}
  state(sid){
    if(!this.sessions.has(sid)){
      const state={sid,memory:null,baseVersion:null,drafts:{},pending:null,open:false,phase:'idle',message:'',conflict:false,revision:0};
      try{
        const saved=JSON.parse(this.storage.getItem(this.key(sid))||'null');
        if(saved&&saved.sid===sid&&Number.isInteger(saved.baseVersion)){
          Object.assign(state,{baseVersion:saved.baseVersion,drafts:saved.drafts||{},pending:saved.pending||null});
          if(state.pending){state.phase='uncertain';state.message='上次保存结果尚未确认，请检查保存结果。';}
        }
      }catch{state.message='本地草稿无法读取。';}
      this.sessions.set(sid,state);
    }
    return this.sessions.get(sid);
  }
  notify(sid){this.onChange(sid,this.state(sid));}
  persist(state){
    if(!Object.keys(state.drafts).length&&!state.pending){this.storage.removeItem(this.key(state.sid));return;}
    this.storage.setItem(this.key(state.sid),JSON.stringify({sid:state.sid,baseVersion:state.baseVersion,drafts:state.drafts,pending:state.pending}));
  }
  remember(state){try{this.persist(state);}catch{state.message='浏览器未能保存草稿，请保持此页打开。';}}
  receive(sid,memory){
    const state=this.state(sid);state.memory=memory;state.revision++;
    if(!Object.keys(state.drafts).length&&!state.pending){state.baseVersion=version(memory);state.conflict=false;}
    else if(state.baseVersion!==version(memory)&&!state.pending){state.conflict=true;state.message='已载入较新的记忆版本，草稿仍保留。请核对下面的已保存内容。';}
    this.notify(sid);
  }
  open(sid){const state=this.state(sid);state.open=true;state.revision++;this.notify(sid);}
  close(sid){const state=this.state(sid);state.open=false;this.remember(state);this.notify(sid);}
  draft(sid,id,content){
    const state=this.state(sid);if(state.pending)return;
    const row=memoryRows(state.memory).find(row=>row.id===id);if(!row?.editable)return;
    const original=state.drafts[id]?.original??row.content;
    if(content===original)delete state.drafts[id];else state.drafts[id]={original,content};
    this.remember(state);this.notify(sid);
  }
  changes(sid){
    const state=this.state(sid),policy=state.memory?.editPolicy||{};
    const perRecord=policy.maxCharsPerRecord||4000,maxTotal=policy.maxTotalChars||32000;
    const records=Object.entries(state.drafts).map(([id,draft])=>({id,content:draft.content}));
    let error='',total=0;
    for(const record of records){
      const row=memoryRows(state.memory).find(row=>row.id===record.id),count=characterCount(record.content);total+=count;
      if(!row?.editable)error='部分草稿对应的记录已不可编辑，请先保留需要的文字，再放弃该草稿。';
      else if(!record.content.trim())error='记忆内容不能为空。';
      else if(count>perRecord)error=`单条记忆不能超过 ${perRecord} 个字符。`;
    }
    if(total>maxTotal)error=`本次修改不能超过 ${maxTotal} 个字符。`;
    return {records,total,perRecord,maxTotal,error};
  }
  discard(sid){
    const state=this.state(sid);if(state.pending)return;
    state.drafts={};state.baseVersion=version(state.memory);state.conflict=false;state.message='已放弃未保存的修改。';state.revision++;
    this.remember(state);this.notify(sid);
  }
  async request(url,options={}){
    const response=await this.fetch(apiPath(url),{...options,signal:AbortSignal.timeout(options.method?900000:20000),headers:{'Content-Type':'application/json',...options.headers}});
    const payload=await response.json();
    if(!response.ok||payload.status==='error'){
      const error=new Error(payload.error?.message||payload.message||`HTTP ${response.status}`);
      error.code=payload.error?.code;error.status=response.status;throw error;
    }
    return payload;
  }
  async refresh(sid){
    const state=this.state(sid);if(state.phase==='saving'||state.phase==='loading')return;
    state.phase='loading';this.notify(sid);
    try{
      const memory=await this.request(`/session/${encodeURIComponent(sid)}/memories`);
      this.receive(sid,memory);this.onPublished(sid,memory);
      if(!state.conflict&&!state.pending)state.message='已读取最新记忆。';
    }catch(error){state.message=`读取失败：${error.message}；草稿已保留。`;}
    finally{state.phase=state.pending?'uncertain':'idle';this.notify(sid);}
  }
  acceptLatest(sid){
    const state=this.state(sid);if(state.pending||!state.conflict)return;
    if(version(state.memory)===state.baseVersion){state.message='请先读取最新版本，再核对草稿。';this.notify(sid);return;}
    if(this.changes(sid).error){state.message=this.changes(sid).error;this.notify(sid);return;}
    for(const [id,draft] of Object.entries(state.drafts)){
      const row=memoryRows(state.memory).find(row=>row.id===id);
      if(draft.content===row.content)delete state.drafts[id];else draft.original=row.content;
    }
    state.baseVersion=version(state.memory);state.conflict=false;state.message='已按最新版本保留草稿，可以保存。';
    this.remember(state);state.revision++;this.notify(sid);
  }
  async save(sid){
    const state=this.state(sid),change=this.changes(sid);
    if(state.pending||state.phase==='loading'||state.conflict||!state.memory?.canEdit)return;
    if(change.error||!change.records.length){state.message=change.error||'没有需要保存的修改。';this.notify(sid);return;}
    return this.start(sid,'update',{baseVersion:state.baseVersion,records:change.records});
  }
  async undo(sid){
    const state=this.state(sid);
    if(state.pending||state.phase==='loading'||Object.keys(state.drafts).length||!state.memory?.canRollback)return;
    return this.start(sid,'rollback',{currentVersion:version(state.memory)});
  }
  async start(sid,operation,body){
    const state=this.state(sid);
    state.pending={operation,body,key:this.newKey(),startedAt:this.now()};
    try{this.persist(state);}catch{
      state.pending=null;state.message='无法保存操作凭据，尚未提交修改。请检查浏览器存储后重试。';this.notify(sid);return;
    }
    return this.retry(sid);
  }
  async retry(sid){
    const state=this.state(sid),pending=state.pending;if(!pending||state.phase==='saving'||state.phase==='loading')return;
    state.phase='saving';state.message='正在保存并更新关联记忆，可能需要几分钟。';this.notify(sid);
    try{
      const rollback=pending.operation==='rollback';
      const result=await this.request(`/session/${encodeURIComponent(sid)}/memories/30sec${rollback?'/rollback':''}`,{
        method:rollback?'POST':'PUT',headers:{'Idempotency-Key':pending.key},body:JSON.stringify(pending.body),
      });
      const expected=(pending.body.baseVersion??pending.body.currentVersion)+1;
      if(result.memory?.session_id!==sid||!Number.isInteger(version(result.memory))||version(result.memory)<expected)throw new Error('保存回执不完整');
      state.pending=null;state.drafts={};state.conflict=false;state.phase='loading';
      this.receive(sid,result.memory);this.onPublished(sid,result.memory);
      // A replayed receipt can predate another tab's newer edit. Read the
      // current snapshot instead of treating the receipt as the latest version.
      try{
        const latest=await this.request(`/session/${encodeURIComponent(sid)}/memories`);
        this.receive(sid,latest);this.onPublished(sid,latest);
      }catch{ /* The successful receipt still proves this operation committed. */ }
      state.phase='idle';state.message=rollback?'已撤销上次修改，恢复内容已发布。':'保存成功，新版本已发布。后续提问会使用已发布的记忆。';
      this.remember(state);
    }catch(error){
      // Browser timeout/abort exceptions have numeric `code` values. They do
      // not establish whether the server committed the write.
      const uncertain=!Number.isInteger(error.status)||typeof error.code!=='string'||!error.code||error.code==='memory_update_in_progress';
      if(uncertain){state.phase='uncertain';state.message='连接中断或后台仍在处理，保存结果尚未确认。请点击“检查保存结果”，不要重复提交新修改。';}
      else{
        state.pending=null;state.phase='idle';state.message=`${error.message}；草稿已保留。`;
        if(error.code==='memory_version_conflict'){state.conflict=true;state.message='记忆版本已更新，尚未覆盖新内容。草稿已保留，请读取最新版本并核对。';}
      }
      this.remember(state);
    }finally{this.notify(sid);}
  }
}
