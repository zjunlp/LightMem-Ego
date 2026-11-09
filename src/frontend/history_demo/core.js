export const apiPath = (path, base = import.meta.env?.VITE_API_BASE_URL || '/api') =>
  `${base.replace(/\/+$/, '')}/${path.replace(/^\/api(?=\/|$)/, '').replace(/^\/+/, '')}`;

// The history page uses this small, tab-scoped gate to keep the edit controls
// out of an ordinary presentation. It is deliberately not treated as
// authentication: the backend's canEdit flag, version checks and idempotency
// keys remain the authority for every write.
export const ADMIN_MODE_STORAGE_KEY = 'lightmem-history.admin-mode.v1';
const defaultAdminCode = () => {
  const configured = typeof globalThis !== 'undefined' ? globalThis.__LIGHTMEM_ADMIN_CODE__ : '';
  return String(configured || 'lightmem-admin');
};

export class AdminMode {
  constructor({storage = globalThis.sessionStorage, code = null, codeProvider = defaultAdminCode, onChange = () => {}} = {}) {
    this.storage = storage;
    this.code = code;
    this.codeProvider = codeProvider;
    this.onChange = onChange;
  }
  isUnlocked() {
    try { return this.storage?.getItem(ADMIN_MODE_STORAGE_KEY) === 'unlocked'; }
    catch { return false; }
  }
  unlock(candidate) {
    const expected = this.code === null ? this.codeProvider() : this.code;
    if (!expected || String(candidate ?? '') !== String(expected)) return false;
    if (!this.storage?.setItem) return false;
    try { this.storage.setItem(ADMIN_MODE_STORAGE_KEY, 'unlocked'); }
    catch { return false; }
    if (!this.isUnlocked()) return false;
    this.onChange(true);
    return true;
  }
  lock() {
    try { this.storage?.removeItem(ADMIN_MODE_STORAGE_KEY); }
    catch { /* a locked UI is the safe fallback when storage is unavailable */ }
    this.onChange(false);
  }
}

export function seconds(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  if (typeof value === 'number') return Number.isFinite(value) && value >= 0 ? value : null;
  if (typeof value !== 'string') return null;
  if (/^\d+(?:\.\d+)?$/.test(value.trim())) return Number(value);
  const match = value.trim().match(/^(\d+):([0-5]\d)(?::([0-5]\d(?:\.\d+)?))?$/);
  if (!match) return null;
  return match[3] === undefined ? Number(match[1])*60+Number(match[2]) : Number(match[1])*3600+Number(match[2])*60+Number(match[3]);
}

export function clock(value) {
  const time = seconds(value);
  if (time === null) return '时间未知';
  const total = Math.floor(time), h = Math.floor(total/3600), m = Math.floor(total%3600/60), s = total%60;
  return `${h ? `${String(h).padStart(2,'0')}:` : ''}${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

export function playbackSeconds(frame, media) {
  if (media?.time_alignment_known !== true || typeof media.video_start_seconds !== 'number' || !Number.isFinite(media.video_start_seconds)) return null;
  if (frame.owner_session_id && frame.owner_session_id !== media.session_id) return null;
  const capture = seconds(frame.capture_time_seconds ?? frame.timestamp_seconds ?? frame.timestamp ?? frame.start_seconds ?? frame.start);
  const start = media.video_start_seconds;
  if (capture === null) return null;
  const position = capture-start;
  const duration = seconds(media.duration_seconds);
  return position < 0 || (duration !== null && position > duration) ? null : position;
}

export function generationLabel(result = {}) {
  const debug = result.raw?.llm_debug || {}, answerDebug = result.raw?.answer_debug || {};
  if (['failed','error','not_ready','cancelled','aborted'].includes(result.status) || result.generation_status === 'failed') return '生成失败';
  if (debug.local_fallback_used || answerDebug.local_fallback_used || (answerDebug.fallback_used && answerDebug.error_debug)) return '生成降级：本地证据摘要';
  const status = result.generation_status || debug.generation_status || answerDebug.generation_status;
  const labels = {local_evidence_summary:'生成降级：本地证据摘要',text_only_success:'生成降级：纯文本回答',text_success:'文本生成',multimodal_success:'图文生成',failed:'生成失败'};
  return labels[status] || (status ? `生成状态：${status}` : '生成状态未提供');
}

export function isLocalSummary(result = {}) {
  return result.generation_status === 'local_evidence_summary' || Boolean(result.local_evidence_summary)
    || Boolean(result.raw?.llm_debug?.local_fallback_used || result.raw?.answer_debug?.local_fallback_used)
    || /^LLM answer generation failed.*Local evidence summary:/s.test(result.answer || '');
}

export function failedAnswer(result = {}) {
  return isLocalSummary(result) || ['failed','error','not_ready','aborted','cancelled'].includes(result.status)
    || result.generation_status === 'failed';
}

export function answerFailureMessage(result = {}) {
  return result.message || (isLocalSummary(result)
    ? '回答服务未能完成本次生成。下方保留了检索到的文字证据，但它不是问题的答案。'
    : '本次未能生成回答，请稍后重试。');
}

export async function requestJSON(fetcher, url, options = {}) {
  const response = await fetcher(url, {...options, signal: AbortSignal.timeout(20000), headers:{'Content-Type':'application/json',...options.headers}});
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.message || data.detail || `HTTP ${response.status}`);
    error.httpStatus = response.status;
    throw error;
  }
  return data;
}

const terminalTask = status => ['done','completed','success','ok','failed','error','not_found','aborted','cancelled'].includes(status);
const failedTask = status => ['failed','error','not_found','aborted','cancelled'].includes(status);

// A session owns its turns and pending task. Leaving it invalidates only its
// polling epoch; it never creates a replacement request or moves another turn.
export class QueryController {
  constructor({fetch = globalThis.fetch.bind(globalThis), storage = globalThis.sessionStorage,
    onChange = () => {}, sleep = ms => new Promise(resolve => setTimeout(resolve,ms)), now = Date.now,
    pollInterval = 1500, pollBudget = 240000} = {}) {
    Object.assign(this,{fetch,storage,onChange,sleep,now,pollInterval,pollBudget});
    this.sessions = new Map();
  }
  key(sid) { return `lightmem-history.pending.v1:${sid}`; }
  state(sid) {
    if (!this.sessions.has(sid)) {
      const state = {sid, turns:[], epoch:0,historyLoading:false,historyError:'',historyLimit:100,historyHasMore:false,checkingPending:false};
      try {
        const saved = JSON.parse(this.storage.getItem(this.key(sid)) || 'null');
        if (saved && typeof saved.question === 'string') state.turns.push({
          question:saved.question,taskId:saved.taskId || null,createdAt:saved.createdAt,
          status:saved.taskId ? 'paused' : 'uncertain',open:true,
          message:saved.taskId ? '已恢复任务，可继续等待结果。' : '提交结果不确定，请核对后再开始新问题。'
        });
      } catch { /* An unreadable storage entry does not authorize a request. */
        state.turns.push({question:'未恢复的任务',status:'uncertain',message:'任务记录不可读取，请核对提交记录。'});
      }
      this.sessions.set(sid,state);
    }
    return this.sessions.get(sid);
  }
  pending(sid) { return this.state(sid).turns.find(turn => ['submitting','polling','paused','uncertain'].includes(turn.status)); }
  notify(sid) { this.onChange(sid,this.state(sid)); }
  persist(sid,turn) {
    this.storage.setItem(this.key(sid),JSON.stringify({question:turn.question,taskId:turn.taskId || null,createdAt:turn.createdAt}));
  }
  clear(sid) { try {this.storage.removeItem(this.key(sid));} catch {} }
  finishTurn(sid,turn,result,{failed=false,detailLoaded=true}={}) {
    // Reading an old answer must never clear a newer pending submission.
    const wasPending=this.pending(sid)===turn;
    if(wasPending)this.state(sid).epoch++;
    turn.result=result;turn.status=failed||failedAnswer(result)?'failed':'done';
    turn.partialAnswer='';turn.detailLoaded=detailLoaded;
    turn.message=turn.status==='failed'?answerFailureMessage(result):'';
    if(wasPending)this.clear(sid);
  }
  validateTask(sid,data) {
    if(data.session_id&&data.session_id!==sid)throw new Error('返回的任务属于其他会话，已停止展示');
    if(data.result?.session_id&&data.result.session_id!==sid)throw new Error('返回的回答属于其他会话，已停止展示');
  }
  async loadHistory(sid, more=false) {
    const state=this.state(sid);
    if(state.historyLoading)return;
    state.historyLoading=true;state.historyError='';this.notify(sid);
    const limit=state.historyLimit+(more?100:0);
    try {
      const data=await requestJSON(this.fetch,apiPath(`/session/${encodeURIComponent(sid)}/qa_history?limit=${limit}`));
      if(data.status!=='ok'||!Array.isArray(data.items)||(data.session_id&&data.session_id!==sid))throw new Error('历史记录返回格式无效');
      const existing=new Map(state.turns.map(t=>[t.taskId||t.turnId,t]).filter(([key])=>key));
      const wasEmpty=state.turns.length===0;
      for(const row of data.items) {
        if(row.session_id&&row.session_id!==sid)continue;
        const key=row.task_id||row.turn_id;
        if(!key||!terminalTask(row.status))continue;
        const meta=row.metadata||{};
        const result={answer:row.answer||'',status:failedTask(row.status)?'failed':'ok',
          generation_status:meta.generation_status,message:row.error||'',latency:meta.latency,
          used_memory_sources:meta.used_memory_sources||[]};
        const previous=existing.get(key);
        if(previous) {
          if(this.pending(sid)===previous) {
            previous.turnId=row.turn_id;previous.fromHistory=true;
            this.finishTurn(sid,previous,result,{detailLoaded:!row.task_id});
          }
          continue;
        }
        const turn={question:row.question||'',taskId:row.task_id,turnId:row.turn_id,
          createdAt:Date.parse(meta.question_submitted_at||row.created_at)||0,status:failedAnswer(result)?'failed':'done',
          result,fromHistory:true,detailLoaded:!row.task_id,open:false,message:''};
        state.turns.push(turn);existing.set(key,turn);
      }
      state.turns.sort((a,b)=>(Number(a.createdAt)||0)-(Number(b.createdAt)||0));
      if(wasEmpty&&state.turns.length)state.turns.at(-1).open=true;
      state.historyLimit=limit;state.historyHasMore=data.items.length>=limit;
    }catch(error){state.historyError=`历史记录读取失败：${error.message}`;}
    finally{state.historyLoading=false;this.notify(sid);}
  }
  async loadTurn(sid,turn) {
    if(!turn.taskId||turn.detailLoaded||turn.detailLoading)return;
    turn.detailLoading=true;turn.detailError='';this.notify(sid);
    try {
      const data=await requestJSON(this.fetch,apiPath(`/query_task/${encodeURIComponent(turn.taskId)}?compact=1`));
      this.validateTask(sid,data);
      if(!terminalTask(data.status))return;
      if(data.result&&typeof data.result==='object') {
        this.finishTurn(sid,turn,data.result,{failed:failedTask(data.status)});
      }
      // Expired task files do not remove the persisted question/answer.
      turn.detailLoaded=true;
    }catch(error){turn.detailError=`证据详情暂时无法加载：${error.message}`;}
    finally{turn.detailLoading=false;this.notify(sid);}
  }
  async checkPending(sid) {
    const state=this.state(sid),turn=this.pending(sid),epoch=state.epoch;
    if(!turn?.taskId||turn.status!=='paused'||state.checkingPending)return;
    state.checkingPending=true;this.notify(sid);
    try {
      const data=await requestJSON(this.fetch,apiPath(`/query_task/${encodeURIComponent(turn.taskId)}?compact=1`));
      if(state.epoch!==epoch||this.pending(sid)!==turn)return;
      this.validateTask(sid,data);
      // A task file may have expired after its answer was saved in history.
      if(data.status==='not_found') {
        await this.loadHistory(sid);
        if(state.epoch!==epoch||this.pending(sid)!==turn)return;
      }
      if(terminalTask(data.status)) {
        this.finishTurn(sid,turn,data.result&&typeof data.result==='object'?data.result:data,{failed:failedTask(data.status)});
      }else {
        turn.message=`${data.status==='queued'?'任务仍在排队':'任务仍在后台处理'}，页面会自动核对结果，也可继续等待。`;
      }
    }catch(error) {
      if(state.epoch!==epoch||this.pending(sid)!==turn)return;
      turn.message=`暂时无法核对任务状态：${error.message}。页面会自动重查，无需重新提交。`;
    }finally{state.checkingPending=false;this.notify(sid);}
  }
  stop(sid) {
    const state=this.state(sid);state.epoch++;
    const turn=this.pending(sid);
    if (turn?.taskId) {turn.status='paused';turn.message='已暂停等待；任务仍可能在后台运行。';}
    this.notify(sid);
  }
  acknowledge(sid) {
    const turn=this.pending(sid);
    if(turn?.status!=='uncertain') return;
    turn.status='unconfirmed';turn.message='已由用户核对，原提交结果仍未确认。';this.clear(sid);this.notify(sid);
  }
  async submit(sid,question) {
    if(this.pending(sid) || !question.trim()) return;
    const state=this.state(sid), epoch=++state.epoch;
    for(const item of state.turns)item.open=false;
    const turn={question:question.trim(),status:'submitting',createdAt:this.now(),open:true,message:'正在提交…'};
    state.turns.push(turn);this.notify(sid);
    try {this.persist(sid,turn);} catch {
      turn.status='failed';turn.message='浏览器不能保存任务记录，未发送问题。请允许此站点的会话存储。';this.notify(sid);return;
    }
    try {
      const data=await requestJSON(this.fetch,apiPath(`/session/${encodeURIComponent(sid)}/ask`),{method:'POST',body:JSON.stringify({
        question:turn.question,mode:'async',response_mode:'legacy',memory_mode:'auto',cache_mode:'auto',
        use_image_evidence:true,max_image_frames:6,max_image_evidence:6,answer_tts:false,client_source:'history_demo',input_method:'text'
      })});
      if (!data.task_id || (data.session_id && data.session_id!==sid)) throw new Error('提交返回的任务标识无效');
      turn.taskId=data.task_id;turn.status='paused';turn.message='任务已提交。';
      try {this.persist(sid,turn);} catch {turn.message='任务已提交，但浏览器无法保存；请记下任务 ID。';}
      this.notify(sid);
      if(state.epoch===epoch) await this.resume(sid);
    } catch(error) {
      if (error.httpStatus>=400 && error.httpStatus<500) {turn.status='failed';turn.message=`提交被拒绝：${error.message}`;this.clear(sid);}
      else {turn.status='uncertain';turn.message=`提交结果不确定，请核对原会话记录。不会自动重试。${error.message}`;}
      this.notify(sid);
    }
  }
  async resume(sid) {
    const state=this.state(sid),turn=this.pending(sid);
    if(!turn?.taskId || turn.status==='polling') return;
    const epoch=++state.epoch,started=this.now();
    turn.status='polling';turn.message='正在等待真实问答结果…';this.notify(sid);
    try {
      while(state.epoch===epoch) {
        const data=await requestJSON(this.fetch,apiPath(`/query_task/${encodeURIComponent(turn.taskId)}?compact=1`));
        if(state.epoch!==epoch) return;
        this.validateTask(sid,data);
        if(terminalTask(data.status)) {
          this.finishTurn(sid,turn,data.result&&typeof data.result==='object'?data.result:data,{failed:failedTask(data.status)});
          this.notify(sid);return;
        }
        const progress=data.progress||{};
        const stageLabels={retrieving:'正在检索记忆',generating:'正在等待回答服务',streaming:'正在生成回答',text_fallback:'正在尝试纯文字回答',finalizing:'正在整理回答与证据'};
        const elapsed=Math.max(0,Math.floor((this.now()-(Date.parse(data.created_at)||turn.createdAt||started))/1000));
        turn.message=`${data.status==='queued'?'等待处理':stageLabels[progress.stage]||'正在处理问题'} · 已等待 ${elapsed} 秒`;
        turn.partialAnswer=progress.partial_answer||'';this.notify(sid);
        if(this.now()-started>=this.pollBudget) throw new Error('本次等待已超时；后台任务可能仍在运行');
        await this.sleep(this.pollInterval);
      }
    } catch(error) {
      if(state.epoch!==epoch) return;
      turn.status='paused';turn.message=`等待中断：${error.message}。可继续等待同一任务，无需重新提问。`;this.notify(sid);
    }
  }
}
