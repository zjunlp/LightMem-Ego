"""Real failure shapes without network calls or paid generations."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from online_query import query_engine as engine
from online_query.generation_errors import GenerationCallError, failure_details, mark_answer_failure


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('error_kind', ['connection', 'timeout', 'rate_limit'])
def test_transport_errors_do_not_trigger_a_second_text_request(tmp_path, monkeypatch, streaming, error_kind):
    import sys
    from retrieval_source_helpers import UTIL
    monkeypatch.setitem(sys.modules, 'em2mem.memory.utils', UTIL)
    monkeypatch.setenv('EM2MEM_QUERY_ANSWER_RETRIES', '0')
    request=httpx.Request('POST','https://example.invalid/v1/chat/completions')
    calls=[]
    class Model:
        last_debug={'api_timing':{'api_duration_ms':120000}}
        def generate(self,prompt):
            calls.append(prompt)
            if error_kind=='timeout':raise APITimeoutError(request=request)
            if error_kind=='rate_limit':
                raise RateLimitError('busy',response=httpx.Response(429,request=request),body=None)
            try:raise httpx.ConnectError('TLS peer closed connection',request=request)
            except Exception as cause:raise APIConnectionError(request=request) from cause
        def stream_generate(self,prompt,on_chunk):return self.generate(prompt)
    loaded=object.__new__(engine.LoadedQueryEngine)
    loaded.session_dir=tmp_path;loaded.em2mem_memory=SimpleNamespace(respond_llm_model=Model())
    events=[]
    qa=loaded._generate_answer_from_evidence('杯子后来在哪里？','memory_router',[],[],[],True,1,
        selected_image_paths=['frame.jpg'],stream_handler=events.append if streaming else None)
    assert len(calls)==1,'A transport failure must not cause an extra text-only call'
    assert qa.llm_debug['generation_status']=='local_evidence_summary'
    assert qa.llm_debug['text_fallback_skipped']
    if streaming:
        assert any(event['type']=='error' for event in events)
        assert not any(event['type']=='final' and event.get('answer') for event in events)
    failure=qa.llm_debug['primary_failure']
    assert failure['diagnostics']['attempts'][0]['last_debug']['api_timing']['api_duration_ms']==120000
    if error_kind=='connection':
        assert any(row['type']=='ConnectError' for row in failure['error_chain'])
    result=mark_answer_failure({'status':'ok','answer':qa.answer,
        'generation_status':'local_evidence_summary','raw':{'llm_debug':qa.llm_debug}})
    assert result['status']=='failed' and result['answer']==''
    assert result['local_evidence_summary']==qa.answer
    assert result['sent_image_count']==0


def test_error_chain_retains_cause_but_redacts_credentials(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','private-test-credential')
    class Model:
        def generate(self,prompt):raise ConnectionError('failure private-test-credential')
    with pytest.raises(GenerationCallError) as caught:
        engine._llm_generate_with_retries(Model(),'test',1)
    assert isinstance(caught.value.__cause__,ConnectionError)
    assert 'private-test-credential' not in json.dumps(failure_details(caught.value))


def test_progress_is_separate_from_queue_and_keeps_partial_text(tmp_path):
    from online_query_worker import _PollingProgressHandler
    handler=_PollingProgressHandler(tmp_path,'s_task')
    handler({'type':'progress','stage':'generating','reset_answer':True})
    handler({'type':'delta','delta':'杯子在'})
    handler({'type':'delta','delta':'桌上'})
    handler({'type':'final','text':'杯子在桌上'})
    progress=json.loads(handler.path.read_text())
    assert progress['stage']=='finalizing'
    assert progress['partial_answer']=='杯子在桌上'
    assert not (tmp_path/'online_tasks/query_in_progress/s_task.json').exists()
    handler({'type':'progress','stage':'text_fallback','reset_answer':True})
    assert json.loads(handler.path.read_text())['partial_answer']==''
    handler.close();assert not handler.path.exists()


def test_failed_worker_records_failure_metadata(tmp_path,monkeypatch):
    import online_query_worker as worker
    from online_preprocess.task_queue import enqueue_query_task,get_queue_dirs
    from online_preprocess.io_utils import read_json
    from online_query.query_cache import SessionEngineCache
    from online_qa_history import load_qa_history
    monkeypatch.setenv('EM2MEM_SINGLE_ACTIVE_SESSION','0')
    monkeypatch.setattr(worker,'query_session',lambda **kwargs:{'status':'failed','answer':'',
        'message':'回答服务等待超时','generation_status':'local_evidence_summary',
        'error_type':'upstream_timeout','sent_image_count':0,'latency':{'total_ms':120000}})
    sessions=tmp_path/'online_sessions';(sessions/'s').mkdir(parents=True)
    queued=enqueue_query_task(tmp_path,'s','杯子在哪里？')
    assert worker._process_task(tmp_path,sessions,SessionEngineCache(),queued)
    task=read_json(get_queue_dirs(tmp_path)['query_failed']/queued.name)
    assert task['status']=='failed'
    history=load_qa_history(sessions/'s')
    assert history[-1]['status']=='failed' and history[-1]['answer']==''
    assert history[-1]['metadata']['generation_status']=='local_evidence_summary'


def test_task_poll_exposes_progress_only_while_running(tmp_path,monkeypatch):
    import asyncio
    import api_server as api
    from online_preprocess.task_queue import enqueue_query_task,claim_query_task,finish_query_task
    from online_query_worker import _PollingProgressHandler
    monkeypatch.setenv('EM2MEM_SINGLE_ACTIVE_SESSION','0')
    monkeypatch.setattr(api,'PROJECT_ROOT',tmp_path)
    queued=enqueue_query_task(tmp_path,'s','What happened?')
    claimed,task=claim_query_task(tmp_path,queued)
    handler=_PollingProgressHandler(tmp_path,task['task_id'])
    handler({'type':'delta','delta':'An object moved'})
    response=asyncio.run(api.query_task_status(task['task_id']))
    assert json.loads(response.body)['progress']['partial_answer']=='An object moved'
    finish_query_task(tmp_path,claimed,task,status='done',result={'status':'ok','answer':'Done'})
    response=asyncio.run(api.query_task_status(task['task_id']))
    assert 'progress' not in json.loads(response.body),'A final result must not inherit stale draft text'
    handler.close()
