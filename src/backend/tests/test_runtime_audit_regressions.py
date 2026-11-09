"""Offline API/queue/edit regressions; all sessions live under tmp_path."""
import asyncio
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from online_preprocess.io_utils import read_json, write_json_atomic
from online_preprocess import task_queue


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name, value in {'EM2MEM_ENABLE_DEMO_ROUTES':'0','EM2MEM_SINGLE_ACTIVE_SESSION':'0', 'EM2MEM_QUERY_WARMUP_CURRENT_MODEL':'0','EM2MEM_QUERY_WARMUP_VLM2VEC':'0','EM2MEM_ENABLE_SYNC_ASK':'1','EM2MEM_QUERY_CURRENT_FAST_PATH':'1','EM2MEM_MEMORY_GENERATION_BACKEND':'rule','EM2MEM_AUTO_VISUAL_EMBEDDING':'0'}.items():
        monkeypatch.setenv(name,value)


def test_warmup_uses_supported_loader_contract(tmp_path, monkeypatch):
    import online_query.query_engine as engine
    import online_query.warmup as warmup
    from online_query.query_cache import SessionEngineCache
    from online_memory_edit.service import MemoryEditService
    signature = inspect.signature(engine.load_query_engine)
    def load(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return SimpleNamespace(last_accessed_at=time.time())
    monkeypatch.setattr(engine, 'load_query_engine', load)
    path = tmp_path/'s'/'em2mem'/'memory_config.json'
    write_json_atomic(path, {'status':'memory_ready','latest_ready_memory_version':1})
    result = warmup.warm_query_session(session_id='s',sessions_root=tmp_path, cache=SessionEngineCache(),wait_for_memory=False)
    assert result['status']=='ready', result
    import online_query_worker as worker
    queued=task_queue.enqueue_query_warmup_task(tmp_path,'s',reason='memory_ready')
    assert worker._process_task(tmp_path,tmp_path,SessionEngineCache(),queued)
    task=read_json(task_queue.get_queue_dirs(tmp_path)['query_done']/queued.name)
    assert task['result']['status']=='ready'


def test_sync_preserves_not_ready_and_failed_task_history(tmp_path, monkeypatch):
    import api_server as api
    import online_query.query_engine as engine
    from online_qa_history import load_qa_history
    sessions=tmp_path/'online_sessions'
    write_json_atomic(sessions/'s'/'em2mem'/'memory_config.json',{'status':'building'})
    write_json_atomic(sessions/'s'/'status.json',{'stage':'memory_building','progress':50})
    monkeypatch.setattr(api,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(api,'ONLINE_SESSIONS_DIR',sessions)
    monkeypatch.setattr(engine,'PROJECT_ROOT',tmp_path)
    response=asyncio.run(api.ask_session('s',api.AskRequest(question='Who moved the cup yesterday?',mode='sync')))
    result=json.loads(response.body)
    assert result['status']=='not_ready',result
    failed=list(task_queue.get_queue_dirs(tmp_path)['query_failed'].glob('*.json'))
    assert len(failed)==1
    assert read_json(failed[0])['result']['status']=='not_ready'
    history=load_qa_history(sessions/'s')
    assert history[-1]['status']=='failed'


@pytest.mark.parametrize('budget',[0,None])
def test_worker_retains_explicit_zero_image_limits(tmp_path,monkeypatch,budget):
    import api_server as api
    import online_query.query_engine as engine
    import online_query_worker as worker
    from online_current.mcur_store import MCurStore
    from online_query.query_cache import SessionEngineCache
    sessions=tmp_path/'online_sessions'; session=sessions/'s'
    frame=session/'stream/frames/frame_000001.jpg'; frame.parent.mkdir(parents=True)
    Image.new('RGB',(8,8)).save(frame)
    MCurStore(session).update_from_frame_stream(frame_index=1,frame_path='stream/frames/frame_000001.jpg',relative_ts_ms=1000)
    class RecordingModel:
        def __init__(self): self.prompts=[]
        def generate(self,prompt): self.prompts.append(prompt); return 'A cup.'
    model=RecordingModel()
    observed=[]
    real_query=worker.query_session
    def query(**kwargs):
        observed.append(kwargs['max_image_frames'])
        return real_query(**kwargs)
    monkeypatch.setattr(worker,'query_session',query)
    monkeypatch.setattr(engine,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(engine,'_get_short_term_answer_model',lambda:model)
    monkeypatch.setattr(api,'PROJECT_ROOT',tmp_path); monkeypatch.setattr(api,'ONLINE_SESSIONS_DIR',sessions)
    response=asyncio.run(api.ask_session('s',api.AskRequest(question='What can you see now?',mode='async',retrieval_mode='current',max_image_evidence=budget,max_image_frames=0,use_interaction_cache=False)))
    queued=task_queue.get_queue_dirs(tmp_path)['query_queued']/f"{json.loads(response.body)['task_id']}.json"
    assert worker._process_task(tmp_path,sessions,SessionEngineCache(),queued)
    done=read_json(task_queue.get_queue_dirs(tmp_path)['query_done']/queued.name)
    assert done['result']['sent_image_count']==(0 if budget==0 else 1)
    assert len(model.prompts)==1
    assert observed==[0]


def test_refine_listing_ignores_peer_claim(tmp_path,monkeypatch):
    queued=task_queue.enqueue_mst_refine_task(tmp_path,'s',backend='mock',limit_events=1,event_id='e')
    original=Path.stat
    claimed_once=[]
    claimed_paths=[]
    def stat(path,*args,**kwargs):
        if path==queued and not claimed_once:
            claimed_once.append(True)
            claimed=task_queue.claim_mst_refine_task(tmp_path,queued)
            assert claimed is not None
            claimed_paths.append(claimed[0])
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',stat)
    import argparse
    import online_mst_refine_worker
    online_mst_refine_worker.run_worker(argparse.Namespace(
        project_root=str(tmp_path),sessions_root=str(tmp_path/'online_sessions'),
        backend='mock',model=None,max_concurrency=1,scan_sessions=False,once=True,poll_interval=0))
    assert not queued.exists()
    assert len(claimed_paths)==1 and claimed_paths[0].exists()


def test_consolidation_default_backend_reaches_append(tmp_path,monkeypatch):
    import online_mst_to_em2mem as module
    session=tmp_path/'s'
    write_json_atomic(session/'captions/mst_session_30sec_captioned.json',[])
    write_json_atomic(session/'evidence/mst_session_evidence.json',[])
    monkeypatch.setattr(module,'build_episodic_from_mst',lambda **kwargs:{})
    class Appender:
        def __init__(self,**kwargs): pass
        def append_ready_episodes(self,**kwargs):
            return SimpleNamespace(em2mem_update_mode='incremental_append',to_dict=lambda:{'status':'ok'})
    monkeypatch.setattr(module,'IncrementalMemoryAppender',Appender)
    result=module.consolidate_short_term_to_em2mem(session_id='s',sessions_root=tmp_path,backend='rule')
    assert result['updated_em2mem'] is True
    assert result['incremental_append']['status']=='ok'


def test_edit_runtime_supported_loader_contract(tmp_path,monkeypatch):
    import online_query.query_engine as engine
    from online_memory_edit.service import MemoryEditService
    signature=inspect.signature(engine.load_query_engine)
    def load(*args,**kwargs):
        signature.bind(*args,**kwargs)
        return object()
    monkeypatch.setattr(engine,'load_query_engine',load)
    assert MemoryEditService(tmp_path,'s')._validate_runtime(tmp_path/'config.json',{}) is not None


@pytest.mark.parametrize('rollback',[False,True])
def test_edit_then_append_keeps_active_snapshot(tmp_path,monkeypatch,rollback):
    from online_memory_edit.service import MemoryEditService
    from online_memory_incremental.incremental_store import IncrementalMemoryAppender
    from online_memory.evidence_to_em2mem import evidence_doc_to_caption_item,write_caption_files,write_sidecar_files,write_semantic_files,_load_triplet_map
    from online_memory.em2mem_layout import Em2MemOnlineLayout,ensure_em2mem_layout
    from online_memory_view.repository import LongTermMemoryRepository
    session=tmp_path/'s'; layout=Em2MemOnlineLayout(session,'s'); ensure_em2mem_layout(layout)
    record=evidence_doc_to_caption_item('s',{'doc_id':'old','start':0,'end':30,'start_time':0,'end_time':30,'fine_caption':'Original caption.'},0)
    write_caption_files(layout,[record],model_name='fake',generation_backend='rule')
    scales={s:read_json(layout.caption_root/f's_{s}.json',default=[]) for s in ('30sec','3min','10min','1h')}
    sides=write_sidecar_files(layout,'fake',scales,generation_backend='rule')
    write_semantic_files(layout,'fake',[record],generation_backend='rule',triplet_map=_load_triplet_map(sides['30sec']['triplets']))
    config={'status':'memory_ready','memory_version':1,'latest_ready_memory_version':1,'semantic_version':1,'graph_version':1,'visual_version':1,'caption_root':'em2mem/caption_root','sidecar_root':'em2mem/sidecar_root','semantic_root':'em2mem/semantic_root','visual_root':'em2mem/visual_root','memory_generation_backend':'rule','query_rag_args':{'subject':'s','retriever_model':'fake','episodic_caption_root':'em2mem/caption_root','episodic_sidecar_root':'em2mem/sidecar_root','semantic_root':'em2mem/semantic_root'}}
    write_json_atomic(layout.memory_config_path,config)
    service=MemoryEditService(tmp_path,'s',runtime_validator=lambda *a:None,derivation_backend_override='rule')
    index=service._editable_index([record]); rid=next(iter(index)); field=index[rid]['contentField']
    edited=service.update({'baseVersion':1,'records':[{'id':rid,'content':'Manually corrected caption.'}]},'test-edit-key')
    if rollback:
        # Make the stale writable root differ from the rollback snapshot as well.
        stale=read_json(layout.caption_30sec_path); stale[0][field]='Stale writable root.'
        write_json_atomic(layout.caption_30sec_path,stale)
        restored=service.rollback({'currentVersion':edited.payload['memoryVersion']},'test-rollback-key')
        assert restored.payload['memoryVersion']==3
    active_config=read_json(layout.memory_config_path)
    immutable=session/active_config['query_rag_args']['episodic_caption_root']/ 's_30sec.json'
    before=immutable.read_bytes()
    write_json_atomic(session/'em2mem/mst_episodic/mst_30sec_episodes.json',[{'episode_id':'new','start_time':30,'end_time':60,'status':'ready'}])
    write_json_atomic(session/'evidence/mst_session_evidence.json',[{'doc_id':'new','episode_id':'new','start_time':30,'end_time':60,'fine_caption':'Next event.'}])
    appender=IncrementalMemoryAppender(session_id='s',sessions_root=tmp_path,project_root=tmp_path,model_name='fake')
    monkeypatch.setattr(appender,'_refresh_hipporag_cache_isolated',lambda **kwargs:{'status':'ok','healthy':True})
    import fcntl
    import online_memory_incremental.incremental_store as module
    real_materialize=module.materialize_active_snapshot
    def materialize(session_dir):
        with (session_dir/'memory/memory_mutation.lock').open('a+') as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return real_materialize(session_dir)
    monkeypatch.setattr(module,'materialize_active_snapshot',materialize)
    result=appender.append_ready_episodes(skip_graph_semantic=True)
    repo=LongTermMemoryRepository(tmp_path,'s'); active=repo.resolve_active_root()
    records=repo.load_episodic(active.config)['30sec']
    assert result.status=='ok' and len(records)==2
    assert result.fast_memory_version==(4 if rollback else 3)
    assert next(x for x in records if x['doc_id']=='old')[field]==('Original caption.' if rollback else 'Manually corrected caption.')
    assert immutable.read_bytes()==before
