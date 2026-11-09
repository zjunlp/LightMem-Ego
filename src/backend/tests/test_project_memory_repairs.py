"""Offline persistence/queue regression tests for the project review repairs."""
from __future__ import annotations

import contextlib
import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from online_memory_incremental.incremental_store import IncrementalMemoryAppender
from online_memory_incremental.snapshot_manager import SnapshotManager
from online_short_term.mst_store import MSTStore
from online_short_term.mst_to_episodic import build_episodic_from_mst


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Unexpected network access')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    for name, value in {
        'EM2MEM_MEMORY_GENERATION_BACKEND':'rule',
        'EM2MEM_INCREMENTAL_GRAPH_BACKEND':'rule',
        'EM2MEM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND':'rule',
        'EM2MEM_AUTO_VISUAL_EMBEDDING':'0',
        'EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE':'0',
        'EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE_ON_NOOP':'0',
    }.items():
        monkeypatch.setenv(name, value)


SNAP_ARGS = dict(components={'episodic':1}, long_term_partial_ready=True,
                 long_term_full_ready=False, semantic_lagging=True, graph_lagging=True)


def test_snapshot_rebuild_failure_retains_published_generation(tmp_path, monkeypatch):
    session=tmp_path/'session'
    caption=session/'em2mem/caption_root/session_30sec.json'
    write(caption, [{'text':'first'}])
    manager=SnapshotManager(session)
    old=manager.build_fast_snapshot(1, **SNAP_ARGS)
    manager.update_config_snapshot_fields(session/'em2mem/memory_config.json', old, 1)
    config_before=(session/'em2mem/memory_config.json').read_bytes()
    monkeypatch.setattr(manager, '_copy_if_exists', lambda *args: (_ for _ in ()).throw(OSError('copy failed')))
    with pytest.raises(OSError):
        manager.build_fast_snapshot(1, **SNAP_ARGS)
    assert old.is_dir()
    assert (session/'em2mem/memory_config.json').read_bytes()==config_before
    assert not list(manager.root.glob('*.tmp'))


def test_same_memory_version_gets_new_immutable_generation(tmp_path):
    session=tmp_path/'session'
    caption=session/'em2mem/caption_root/session_30sec.json'
    write(caption, [{'text':'first'}])
    manager=SnapshotManager(session)
    old=manager.build_fast_snapshot(1, **SNAP_ARGS)
    write(caption, [{'text':'second'}])
    new=manager.build_fast_snapshot(1, **SNAP_ARGS)
    assert old!=new
    assert json.loads((old/'caption_root/session_30sec.json').read_text())==[{'text':'first'}]
    assert json.loads((new/'caption_root/session_30sec.json').read_text())==[{'text':'second'}]


def appender(tmp_path, session_id='session'):
    return IncrementalMemoryAppender(session_id=session_id, sessions_root=tmp_path/'sessions',
                                      project_root=tmp_path/'project', model_name='offline')


def test_append_plan_reads_latest_log_once(tmp_path):
    app=appender(tmp_path)
    episodes=[{'episode_id':f'e{i}','start_time':i*30,'end_time':(i+1)*30} for i in range(25)]
    write(app.session_dir/'em2mem/mst_episodic/mst_30sec_episodes.json', episodes)
    for ep in episodes:
        app.append_log.append_status(episode=ep,status='fully_ready')
    with patch.object(app.append_log,'read_entries',wraps=app.append_log.read_entries) as reads:
        result=app.plan_ready_episodes()
    assert result['candidate_count']==0
    assert reads.call_count==1


@pytest.mark.parametrize('existing',[False,True])
def test_append_dry_run_is_read_only(tmp_path, existing):
    app=appender(tmp_path)
    if existing:
        write(app.session_dir/'em2mem/caption_root/session_30sec.json',[{'doc_id':'old'}])
    def files():
        return {str(p.relative_to(tmp_path)):p.read_bytes() if p.is_file() else None for p in tmp_path.rglob('*')}
    before=files()
    result=app.append_ready_episodes(dry_run=True)
    assert result.status=='dry_run'
    assert files()==before


def worker_environment(tmp_path, monkeypatch):
    import online_mst_consolidation_worker as worker
    project=tmp_path/'project'
    project.mkdir(exist_ok=True)
    sessions=tmp_path/'sessions'
    session=sessions/'session'
    store=MSTStore(session)
    store.append_events([{'event_id':'e0','start_time':0,'end_time':30,'status':'refined',
                         'needs_refine':False,'event_caption_refined':'A person works at a table.'}])
    from online_short_term.refine_status import write_refine_status
    write_refine_status(store)
    monkeypatch.chdir(project)
    monkeypatch.setattr(worker,'__file__',str(project/'worker.py'))
    monkeypatch.setattr(worker,'session_is_active_or_allowed',lambda *args:True)
    monkeypatch.setattr(worker,'write_worker_runtime',lambda *args,**kwargs:None)
    monkeypatch.setattr(worker,'WorkerTaskHeartbeat',lambda *args,**kwargs:contextlib.nullcontext())
    monkeypatch.setattr(worker,'refresh_session_pipeline_state',lambda *args:None)
    monkeypatch.setattr(worker,'append_timeline_event',lambda *args,**kwargs:None)
    return worker,project,sessions,session


def run_once(worker,sessions):
    worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=True,
                      interval_seconds=0,limit_windows=None,once=True,verbose=False)


def test_consolidation_scan_disabled_preserves_explicit_queue_and_handoff(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    from online_preprocess.task_queue import (
        enqueue_mst_consolidation_task, list_queued_memory_tasks,
    )
    with patch.object(worker,'consolidate_short_term_to_em2mem',wraps=worker.consolidate_short_term_to_em2mem) as builds:
        worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=True,
                          interval_seconds=0,limit_windows=None,once=True,verbose=False,
                          scan_sessions=False)
        assert builds.call_count==0
        assert not (session/'short_term/consolidation_state.json').exists()
        assert list_queued_memory_tasks(project)==[]
        enqueue_mst_consolidation_task(project_root=project,session_id='session',
                                       backend='rule',update_em2mem=True)
        worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=True,
                          interval_seconds=0,limit_windows=None,once=True,verbose=False,
                          scan_sessions=False)
        assert builds.call_count==1
    assert len(list_queued_memory_tasks(project))==1
    assert len(list((project/'online_tasks/mst_consolidation_done').glob('*.json')))==1


def test_consolidation_scan_enabled_by_default(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    from online_preprocess.task_queue import list_queued_memory_tasks
    run_once(worker,sessions)
    assert len(list_queued_memory_tasks(project))==1


def test_worker_recovers_handoff_after_enqueue_failure_without_regeneration(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    enqueue=worker.enqueue_memory_task
    attempts=[]
    def fail_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts)==1:
            raise OSError('temporary queue error')
        return enqueue(**kwargs)
    monkeypatch.setattr(worker,'enqueue_memory_task',fail_once)
    with patch.object(worker,'consolidate_short_term_to_em2mem',wraps=worker.consolidate_short_term_to_em2mem) as builds:
        run_once(worker,sessions)
        state=json.loads((session/'short_term/consolidation_state.json').read_text())
        assert state.get('pending_memory_handoffs')
        run_once(worker,sessions)
        assert len(attempts)==2
        assert builds.call_count==1
    from online_preprocess.task_queue import list_queued_memory_tasks
    queued=list_queued_memory_tasks(project)
    assert len(queued)==1
    run_once(worker,sessions)
    assert len(list_queued_memory_tasks(project))==1


def test_force_reconsolidation_version_survives_handoff(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule',force=True)
    state=json.loads((session/'short_term/consolidation_state.json').read_text())
    pending=state.get('pending_memory_handoffs',{})
    assert pending['mst_ep_000000_000030']['episode_version']==2
    assert pending['mst_ep_000000_000030']['force'] is True
    run_once(worker,sessions)
    from online_preprocess.task_queue import list_queued_memory_tasks
    task=json.loads(list_queued_memory_tasks(project)[0].read_text())
    assert task['force'] is True


def test_append_new_source_version_replaces_old_caption_without_global_force(tmp_path):
    app=appender(tmp_path)
    def seed(version,text):
        episode={'episode_id':'e0','start_time':0,'end_time':30,'version':version}
        write(app.session_dir/'em2mem/mst_episodic/mst_30sec_episodes.json',[episode])
        write(app.session_dir/'evidence/mst_session_evidence.json',[
            {**episode,'doc_id':'e0','caption':text,'fine_caption':text}])
    seed(1,'Old observation.')
    app.append_ready_episodes(skip_graph_semantic=True)
    seed(2,'Corrected observation.')
    result=app.append_ready_episodes(skip_graph_semantic=True)
    assert result.appended_episode_ids==['e0']
    captions=json.loads(app.layout.caption_30sec_path.read_text())
    assert 'Corrected observation.' in captions[0]['text']
    assert app.append_log.latest_by_episode()['e0']['episode_version']==2


def pending_state(session):
    return json.loads((session/'short_term/consolidation_state.json').read_text())


def test_pending_only_acknowledges_applied_source_version(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    from online_memory_incremental.append_log import MemoryAppendLog
    from online_preprocess.task_queue import list_queued_memory_tasks, ensure_queue_dirs
    log=MemoryAppendLog(session/'em2mem/incremental/memory_append_log.jsonl')
    episode={'episode_id':'mst_ep_000000_000030','start_time':0,'end_time':30,'version':1}
    # The older task is already running while a correction publishes v2.
    first=worker._drain_memory_handoffs(project,session,reason='test')
    queued=list_queued_memory_tasks(project)[0]
    running=ensure_queue_dirs(project)['memory_in_progress']/queued.name
    queued.replace(running)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule',force=True)
    log.append_status(episode=episode,status='fully_ready',fast_memory_version=1)
    result=worker._drain_memory_handoffs(project,session,reason='test')
    assert not result['memory_task_paths']
    assert pending_state(session)['pending_memory_handoffs'][episode['episode_id']]['episode_version']==2
    running.unlink()
    worker._drain_memory_handoffs(project,session,reason='test')
    task_path=list_queued_memory_tasks(project)[0]
    task=json.loads(task_path.read_text())
    assert task['force'] is True
    app=IncrementalMemoryAppender(session_id='session',sessions_root=sessions,project_root=project,model_name='offline')
    app.append_ready_episodes(episode_ids=task['episode_ids'],force=task['force'],skip_graph_semantic=True)
    task_path.unlink()
    worker._drain_memory_handoffs(project,session,reason='test')
    assert pending_state(session)['pending_memory_handoffs']=={}
    assert not list_queued_memory_tasks(project)


def test_failed_queue_task_is_reenqueued_from_pending(tmp_path, monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    run_once(worker,sessions)
    from online_preprocess.task_queue import list_queued_memory_tasks, ensure_queue_dirs
    queued=list_queued_memory_tasks(project)[0]
    queued.replace(ensure_queue_dirs(project)['memory_failed']/queued.name)
    run_once(worker,sessions)
    assert len(list_queued_memory_tasks(project))==1
    assert pending_state(session)['pending_memory_handoffs']


@pytest.mark.parametrize('existing_lt,caption_present,expect_queue',[
    (False,False,True), (True,False,False), (False,True,False),
])
def test_legacy_handoff_repair_does_not_recreate_unknown_edited_lt(tmp_path,monkeypatch,existing_lt,caption_present,expect_queue):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    state=pending_state(session)
    state.pop('pending_memory_handoffs')
    write(session/'short_term/consolidation_state.json',state)
    if existing_lt:
        write(session/'em2mem/memory_config.json',{'memory_version':3,'memory_edit_propagation':{'version':3}})
    if caption_present:
        write(session/'em2mem/caption_root/session_30sec.json',[{'episode_id':'mst_ep_000000_000030'}])
    monkeypatch.setattr(worker,'consolidate_short_term_to_em2mem',lambda **kwargs:(_ for _ in ()).throw(AssertionError('Must not regenerate')))
    run_once(worker,sessions)
    from online_preprocess.task_queue import list_queued_memory_tasks
    assert bool(list_queued_memory_tasks(project)) is expect_queue


def test_explicit_failed_legacy_append_can_recover_with_existing_lt(tmp_path,monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    state=pending_state(session)
    state.pop('pending_memory_handoffs')
    write(session/'short_term/consolidation_state.json',state)
    write(session/'em2mem/memory_config.json',{'memory_version':3})
    from online_memory_incremental.append_log import MemoryAppendLog
    MemoryAppendLog(session/'em2mem/incremental/memory_append_log.jsonl').append_status(
        episode={'episode_id':'mst_ep_000000_000030','version':1},status='failed')
    run_once(worker,sessions)
    from online_preprocess.task_queue import list_queued_memory_tasks
    assert len(list_queued_memory_tasks(project))==1


def test_update_disabled_keeps_intent_without_queueing(tmp_path,monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=False,
                      interval_seconds=0,limit_windows=None,once=True,verbose=False)
    assert pending_state(session)['pending_memory_handoffs']
    from online_preprocess.task_queue import list_queued_memory_tasks
    assert list_queued_memory_tasks(project)==[]


def test_concurrent_handoff_drains_share_one_queue_task(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _:worker._drain_memory_handoffs(project,session,reason='test'),range(2)))
    from online_preprocess.task_queue import list_queued_memory_tasks
    assert len(list_queued_memory_tasks(project))==1


def test_snapshot_pointer_write_failure_retains_old_pointer_and_generations(tmp_path,monkeypatch):
    import online_memory_incremental.snapshot_manager as module
    session=tmp_path/'session'
    write(session/'em2mem/caption_root/session_30sec.json',[{'text':'first'}])
    manager=SnapshotManager(session)
    old=manager.build_fast_snapshot(1,**SNAP_ARGS)
    config=session/'em2mem/memory_config.json'
    manager.update_config_snapshot_fields(config,old,1)
    before=config.read_bytes()
    new=manager.build_fast_snapshot(1,**SNAP_ARGS)
    monkeypatch.setattr(module,'write_json_atomic',lambda *args:(_ for _ in ()).throw(OSError('pointer write failed')))
    with pytest.raises(OSError):
        manager.update_config_snapshot_fields(config,new,1)
    assert config.read_bytes()==before
    assert old.is_dir() and new.is_dir()


def test_snapshot_validation_failure_keeps_previous_generation(tmp_path):
    session=tmp_path/'session'
    caption=session/'em2mem/caption_root/session_30sec.json'
    write(caption,[{'text':'first'}])
    manager=SnapshotManager(session)
    old=manager.build_fast_snapshot(1,**SNAP_ARGS)
    caption.unlink()
    with pytest.raises(FileNotFoundError):
        manager.build_fast_snapshot(1,**SNAP_ARGS)
    assert old.is_dir()
    assert not list(manager.root.glob('*.tmp'))


def test_mismatched_evidence_version_does_not_ack_new_source(tmp_path):
    app=appender(tmp_path)
    episode={'episode_id':'e0','start_time':0,'end_time':30,'version':2}
    write(app.session_dir/'em2mem/mst_episodic/mst_30sec_episodes.json',[episode])
    write(app.session_dir/'evidence/mst_session_evidence.json',[{**episode,'version':1,'doc_id':'e0','caption':'stale'}])
    result=app.append_ready_episodes(skip_graph_semantic=True)
    assert result.failed_episode_ids==['e0']
    assert app.append_log.latest_by_episode()['e0']['status']=='failed'
    assert not app.layout.caption_30sec_path.exists()


def test_explicit_update_task_recovers_when_global_updates_disabled(tmp_path,monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    from online_preprocess.task_queue import enqueue_mst_consolidation_task, list_queued_memory_tasks
    enqueue_mst_consolidation_task(project,'session',backend='rule',update_em2mem=True)
    original=worker.enqueue_memory_task
    calls=[]
    def once_failed(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            raise OSError('first handoff queue failed')
        return original(**kwargs)
    monkeypatch.setattr(worker,'enqueue_memory_task',once_failed)
    with patch.object(worker,'consolidate_short_term_to_em2mem',wraps=worker.consolidate_short_term_to_em2mem) as builds:
        for _ in range(2):
            worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=False,
                              interval_seconds=0,limit_windows=None,once=True,verbose=False)
        assert builds.call_count==1
    assert len(list_queued_memory_tasks(project))==1
    assert len(calls)==2


def test_offline_windows_are_not_swept_by_explicit_update_of_another_window(tmp_path,monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    MSTStore(session).append_events([{'event_id':'e1','start_time':30,'end_time':60,'status':'refined',
                                    'needs_refine':False,'event_caption_refined':'A person walks out.'}])
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule',window_start=0,window_end=30)
    from online_preprocess.task_queue import enqueue_mst_consolidation_task,list_queued_memory_tasks
    enqueue_mst_consolidation_task(project,'session',backend='rule',update_em2mem=True,window_start=30,window_end=60)
    worker.run_worker(sessions_root=sessions,backend='rule',update_em2mem=False,
                      interval_seconds=0,limit_windows=None,once=True,verbose=False)
    queued=list_queued_memory_tasks(project)
    assert len(queued)==1
    assert json.loads(queued[0].read_text())['episode_ids']==['mst_ep_000030_000060']
    assert pending_state(session)['pending_memory_handoffs']['mst_ep_000000_000030']['requested'] is False


def test_reopened_window_gets_new_source_version_without_forcing_other_windows(tmp_path,monkeypatch):
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    first=build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    state=pending_state(session)
    state['window_to_episode']={}
    write(session/'short_term/consolidation_state.json',state)
    result=build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule',force=False)
    assert first['generated_episode_ids']==result['generated_episode_ids']
    episode=json.loads((session/'em2mem/mst_episodic/mst_30sec_episodes.json').read_text())[0]
    assert episode['version']==2
    assert pending_state(session)['pending_memory_handoffs'][episode['episode_id']]['episode_version']==2


def test_component_repair_cannot_ack_source_arriving_during_build(tmp_path,monkeypatch):
    import online_memory_incremental.incremental_store as module
    app=appender(tmp_path)
    episode_path=app.session_dir/'em2mem/mst_episodic/mst_30sec_episodes.json'
    evidence_path=app.session_dir/'evidence/mst_session_evidence.json'
    episode={'episode_id':'e0','start_time':0,'end_time':30,'version':1}
    write(episode_path,[episode])
    write(evidence_path,[{**episode,'doc_id':'e0','caption':'Old observation.'}])
    app.append_ready_episodes(skip_graph_semantic=True)
    def graph(**kwargs):
        write(episode_path,[{**episode,'version':2}])
        write(evidence_path,[{**episode,'version':2,'doc_id':'e0','caption':'New observation.'}])
        return {'graph_lagging':False,'latest_graph_ready_version':1,'delta_path':'em2mem/incremental/graph/stub.jsonl'}
    monkeypatch.setattr(module,'generate_graph_delta',graph)
    monkeypatch.setattr(module,'generate_semantic_delta',lambda **kwargs:{'semantic_lagging':False,'latest_semantic_ready_version':1})
    app.append_ready_episodes(skip_graph_semantic=False)
    assert app.append_log.latest_by_episode()['e0']['episode_version']==1
    assert app.plan_ready_episodes()['candidate_count']==1


def test_pending_new_version_requeues_after_existing_stale_recovery(tmp_path,monkeypatch):
    from online_memory_worker import _recover_stale_memory_tasks
    from online_preprocess.task_queue import list_queued_memory_tasks,ensure_queue_dirs
    worker,project,sessions,session=worker_environment(tmp_path,monkeypatch)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule')
    worker._drain_memory_handoffs(project,session,reason='test')
    old=list_queued_memory_tasks(project)[0]
    running=ensure_queue_dirs(project)['memory_in_progress']/old.name
    task=json.loads(old.read_text())
    task.update(status='in_progress',updated_at='2000-01-01T00:00:00+00:00')
    write(old,task)
    old.replace(running)
    build_episodic_from_mst(session_id='session',sessions_root=sessions,backend='rule',force=True)
    worker._drain_memory_handoffs(project,session,reason='test')
    assert list_queued_memory_tasks(project)==[]
    monkeypatch.setenv('EM2MEM_MEMORY_REQUEUE_STALE_IN_PROGRESS','0')
    monkeypatch.setenv('EM2MEM_MEMORY_TASK_STALE_SECONDS','300')
    recovered=_recover_stale_memory_tasks(project)
    assert recovered['recovered_count']==1
    assert recovered['tasks'][0]['target_queue']=='memory_failed'
    worker._drain_memory_handoffs(project,session,reason='test')
    queued=list_queued_memory_tasks(project)
    assert len(queued)==1
    assert json.loads(queued[0].read_text())['force'] is True
    assert pending_state(session)['pending_memory_handoffs']['mst_ep_000000_000030']['episode_version']==2


def test_metadata_update_serializes_with_memory_publication(tmp_path,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import online_short_term.mst_to_episodic as module
    from online_memory_edit.lock import session_memory_mutation_lock as real_lock
    session=tmp_path/'session'
    config=session/'em2mem/memory_config.json'
    write(config,{'memory_version':1,'caption_root':'old/root','semantic_memory_ready':False})
    entered=threading.Event()
    release_read=threading.Event()
    original_read=module.read_json
    def controlled_read(path,**kwargs):
        value=original_read(path,**kwargs)
        if threading.current_thread().name.startswith('metadata_writer') and path==config:
            entered.set()
            assert release_read.wait(3)
        return value
    @contextlib.contextmanager
    def observed_lock(path):
        entered.set()
        with real_lock(path):
            yield
    monkeypatch.setattr(module,'read_json',controlled_read)
    monkeypatch.setattr(module,'session_memory_mutation_lock',observed_lock,raising=False)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='metadata_writer') as pool:
        with real_lock(session):
            future=pool.submit(module._update_memory_config_metadata,session)
            assert entered.wait(3)
            write(config,{'memory_version':2,'caption_root':'new/root','semantic_memory_ready':True,
                          'query_rag_args':{'episodic_caption_root':'new/root'}})
            release_read.set()
        future.result(timeout=3)
    result=json.loads(config.read_text())
    assert result['memory_version']==2
    assert result['caption_root']=='new/root'
    assert result['semantic_memory_ready'] is True
    assert result['query_rag_args']['episodic_caption_root']=='new/root'
    assert result['mst_episodic_ready'] is True


def test_metadata_and_status_updates_do_not_hold_mst_lock(tmp_path,monkeypatch):
    import online_short_term.mst_to_episodic as module
    session=tmp_path/'session'
    store=MSTStore(session)
    store.append_events([{'event_id':'e0','start_time':0,'end_time':30,'status':'refined',
                         'needs_refine':False,'event_caption_refined':'A person works.'}])
    depth=0
    original=MSTStore.mutation_lock
    @contextlib.contextmanager
    def tracked(self):
        nonlocal depth
        with original(self):
            depth+=1
            try:
                yield
            finally:
                depth-=1
    calls=[]
    def outside(*args):
        assert depth==0
        calls.append(True)
    monkeypatch.setattr(MSTStore,'mutation_lock',tracked)
    monkeypatch.setattr(module,'_update_memory_config_metadata',outside)
    monkeypatch.setattr(module,'_update_status_outputs',outside)
    result=build_episodic_from_mst(session_id='session',sessions_root=tmp_path,backend='rule')
    assert result['generated_episode_count']==1
    assert len(calls)==2
