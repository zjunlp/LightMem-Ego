import test from 'node:test';
import assert from 'node:assert/strict';
import {VideoUploadController} from '../video-upload.js';

const storage=()=>{const values=new Map();return {getItem:k=>values.get(k)||null,setItem:(k,v)=>values.set(k,v),removeItem:k=>values.delete(k)};};
const id='123456781234123412341234567890ab';
const receipt={upload_id:id,session_id:`video_${id}`,status:'processing',stage:'memory',progress:80,query_ready:false};
const response=(data,status=200)=>({ok:status<400,status,json:async()=>data});
const file={name:'cabin.mp4',size:123,type:'video/mp4'};

test('reload recovers by GET and never repeats an upload or unlocks an incomplete memory',async()=>{
 const saved=storage();let sends=0;let ready=false;
 const options={storage:saved,newId:()=>id,schedule:()=>{},send:async()=>{sends++;return receipt;},fetch:async()=>response({...receipt,...(ready?{status:'ready',stage:'ready',query_ready:true}: {})})};
 const first=new VideoUploadController(options);await first.submit(file);
 assert.equal(first.canOpen(),false);
 const reloaded=new VideoUploadController(options);await reloaded.check();
 assert.equal(sends,1);assert.equal(reloaded.state.session_id,receipt.session_id);assert.equal(reloaded.canOpen(),false);
 ready=true;await reloaded.check();assert.equal(reloaded.canOpen(),true);
});

test('ambiguous network failure keeps a receipt and status checks do not re-upload',async()=>{
 let sends=0;const calls=[];
 const controller=new VideoUploadController({storage:storage(),newId:()=>id,schedule:()=>{},send:async()=>{sends++;throw new TypeError('lost reply');},fetch:async url=>{calls.push(url);return response(receipt);}});
 await controller.submit(file);assert.equal(controller.state.phase,'uncertain');
 await controller.check();assert.equal(sends,1);assert.ok(calls.every(url=>url===`/api/video-uploads/${id}`));
 assert.equal(controller.canOpen(),false);
});

test('empty, unsupported and oversized files never reach the upload service',async()=>{
 let sends=0;const controller=new VideoUploadController({storage:storage(),send:()=>{sends++;},schedule:()=>{}});
 for(const invalid of [{...file,size:0},{...file,name:'note.txt',type:'text/plain'},{...file,size:96*1024*1024}])await controller.submit(invalid);
 assert.equal(sends,0);
});

test('a receipt for a different upload cannot open another session',async()=>{
 const controller=new VideoUploadController({storage:storage(),newId:()=>id,schedule:()=>{},send:async()=>({...receipt,upload_id:'b'.repeat(32),status:'ready',query_ready:true})});
 await controller.submit(file);assert.equal(controller.canOpen(),false);assert.equal(controller.state.phase,'uncertain');
});

test('successful status recovery clears a previous connection error',async()=>{
 let disconnected=true;
 const controller=new VideoUploadController({storage:storage(),newId:()=>id,schedule:()=>{},send:async()=>receipt,
   fetch:async()=>{if(disconnected)throw new TypeError('offline');return response(receipt);}});
 await controller.submit(file);await controller.check();
 assert.equal(controller.state.checkError,'offline');
 disconnected=false;await controller.check();
 assert.equal(controller.state.checkError,null);
 assert.equal(controller.state.phase,'processing');
});
