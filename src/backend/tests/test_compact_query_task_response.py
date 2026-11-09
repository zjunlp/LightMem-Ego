import copy
import json

from fastapi.testclient import TestClient
import api_server


def test_compact_response_preserves_complete_answer_and_images_without_retrieval_trace():
    answer='A complete answer. '*300
    payload={'status':'done','task_id':'task','session_id':'sid','result':{
        'status':'ok','answer':answer,'generation_status':'multimodal_success',
        'timestamps':[26,43],'used_memory_sources':['M_lt','M_st'],'latency':{'total_ms':20},
        'evidence_frames':[{'image_url':'/api/image.jpg','capture_time_seconds':26,
            'owner_session_id':'sid','caption':'Full caption','frame_origins':[{'trace':'x'*200000}]}],
        'raw':{'retrieval_trace':'x'*200000},'memory_results':{'trace':'x'*200000}}}
    original=copy.deepcopy(payload)
    compact=api_server._compact_query_task_payload(payload)
    assert compact['result']['answer']==answer
    assert compact['result']['timestamps']==[26,43]
    assert compact['result']['evidence_frames'][0]=={
        'image_url':'/api/image.jpg','capture_time_seconds':26,'owner_session_id':'sid','caption':'Full caption'}
    assert compact['result']['generation_status']=='multimodal_success'
    assert len(json.dumps(compact))<len(json.dumps(payload))*.05
    assert payload==original


def test_client_view_is_opt_in_and_both_task_routes_preserve_failure_semantics(monkeypatch,tmp_path):
    payload={'status':'failed','task_id':'task','session_id':'sid','result':{
        'status':'failed','answer':'','generation_status':'local_evidence_summary',
        'message':'Answer generation failed','raw':{'llm_debug':{'local_fallback_used':True,'trace':'private debug'}},
        'evidence_frames':[]}}
    monkeypatch.setattr(api_server,'_find_query_task',lambda _:('query_failed',tmp_path/'task.json',payload))
    monkeypatch.setattr(api_server,'_augment_evidence_frames_for_response',lambda value,*args,**kwargs:value)
    client=TestClient(api_server.app)
    for path in ['/query_task/task','/task/query/task']:
        full=client.get(path).json()
        compact=client.get(path+'?compact=1').json()
        assert full['result']['raw']['llm_debug']['trace']=='private debug'
        assert compact['status']=='failed'
        assert compact['result']['generation_status']=='local_evidence_summary'
        assert compact['result']['raw']['llm_debug']['local_fallback_used'] is True
        assert compact['result']['message']=='Answer generation failed'
