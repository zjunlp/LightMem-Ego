import {apiPath,requestJSON} from './core.js';

const STORAGE_KEY='lightmem-history.video-upload.v1';
const terminal=state=>['ready','failed','rejected','not_found'].includes(state.phase);
const normalized=value=>String(value||'').replaceAll('-','').toLowerCase();
function uploadFile(file,id,onProgress){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest(),form=new FormData();
    form.append('upload_id',id);form.append('video',file,file.name);
    xhr.open('POST',apiPath('/video-uploads'));xhr.responseType='json';xhr.timeout=20*60*1000;
    xhr.upload.onprogress=event=>{if(event.lengthComputable)onProgress(Math.round(event.loaded/event.total*100));};
    xhr.onload=()=>{
      if(xhr.status>=200&&xhr.status<300&&xhr.response)return resolve(xhr.response);
      const error=new Error(xhr.response?.message||`HTTP ${xhr.status}`);error.httpStatus=xhr.status;reject(error);
    };
    xhr.onerror=()=>reject(new TypeError('Network interrupted'));
    xhr.ontimeout=()=>reject(new Error('Upload response timed out'));
    xhr.onabort=()=>reject(new Error('Upload interrupted'));
    xhr.send(form);
  });
}

export class VideoUploadController{
  constructor({fetch=globalThis.fetch.bind(globalThis),storage=globalThis.sessionStorage,
    onChange=()=>{},send=uploadFile,newId=()=>crypto.randomUUID(),schedule=fn=>setTimeout(fn,3000)}={}){
    Object.assign(this,{fetch,storage,onChange,send,newId,schedule});
    this.state={phase:'idle',status:'idle',progress:0};this.checking=false;this.sending=false;this.timer=null;
    try{
      const stored=JSON.parse(storage.getItem(STORAGE_KEY)||'null');
      if(stored&&/^[a-f0-9]{32}$/.test(stored.upload_id))this.state={...stored,phase:stored.phase==='uploading'?'uncertain':stored.phase};
    }catch{}
  }
  notify(){try{this.storage.setItem(STORAGE_KEY,JSON.stringify(this.state));}catch{}this.onChange(this.state);}
  canOpen(){return this.state.phase==='ready'&&this.state.query_ready===true&&this.state.session_id===`video_${this.state.upload_id}`;}
  accept(receipt){
    if(normalized(receipt.upload_id)!==this.state.upload_id||receipt.session_id!==`video_${this.state.upload_id}`)throw new Error('Wrong upload receipt');
    const phase=receipt.status==='ready'&&receipt.query_ready===true?'ready':receipt.status==='failed'?'failed':'processing';
    this.state={...this.state,...receipt,phase,checkError:null};this.notify();
    if(!terminal(this.state))this.pollLater();
  }
  pollLater(){clearTimeout(this.timer);this.timer=this.schedule(()=>this.check());}
  async submit(file){
    if(this.sending||this.state.upload_id&&!terminal(this.state))return;
    let error='';
    if(!file||file.size<=0)error='请选择视频文件。';
    else if(!/\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name))error='请选择支持的视频格式。';
    else if(file.size>95*1024*1024)error='文件超过 95 MiB 限制。';
    if(error){this.state={phase:'rejected',status:'rejected',progress:0,error};this.notify();return;}
    const upload_id=normalized(this.newId());
    this.state={upload_id,filename:file.name,size_bytes:file.size,phase:'uploading',status:'uploading',stage:'uploading',progress:0,query_ready:false};
    try{this.storage.setItem(STORAGE_KEY,JSON.stringify(this.state));}
    catch{this.state={phase:'rejected',error:'浏览器不能保存上传记录，尚未发送文件。'};this.onChange(this.state);return;}
    this.sending=true;this.notify();
    try{this.accept(await this.send(file,upload_id,progress=>{this.state.progress=progress;this.notify();}));}
    catch(error){this.state.phase=error.httpStatus>=400&&error.httpStatus<500?'rejected':'uncertain';this.state.error=error.message;this.notify();}
    finally{this.sending=false;}
  }
  async check(){
    if(!this.state.upload_id||this.checking||this.sending)return;
    clearTimeout(this.timer);this.checking=true;
    try{this.accept(await requestJSON(this.fetch,apiPath(`/video-uploads/${this.state.upload_id}`)));}
    catch(error){
      if(error.httpStatus===404)this.state={...this.state,phase:'not_found',error:'上传未完成，请重新选择视频上传。'};
      else this.state={...this.state,checkError:error.message};
      this.notify();
      if(!terminal(this.state))this.pollLater();
    }finally{this.checking=false;}
  }
  async resume(){
    if(!this.state.upload_id||this.checking||this.sending)return;
    this.checking=true;
    try{this.accept(await requestJSON(this.fetch,apiPath(`/video-uploads/${this.state.upload_id}/resume`),{method:'POST'}));}
    catch(error){this.state.checkError=error.message;this.notify();}
    finally{this.checking=false;}
  }
  reset(){
    if(this.sending||this.state.upload_id&&!terminal(this.state))return;
    clearTimeout(this.timer);this.state={phase:'idle',status:'idle',progress:0};
    try{this.storage.removeItem(STORAGE_KEY);}catch{}this.onChange(this.state);
  }
  stop(){clearTimeout(this.timer);}
}
