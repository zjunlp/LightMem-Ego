"""Generic recall regressions: no model or embedding service calls."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from online_llm_config import chat_response_text
from online_query.memory_router import MemoryRouter
from online_query.memory_fusion import MemoryFusion
from online_query.memory_plan import RetrievalPlanner
from online_query.evidence_packer import EvidencePacker
from online_query_constraints import parse_query_constraints
from online_short_term.mst_retriever import MSTRetriever
from online_short_term.semantic_index import ShortTermSemanticIndex


class Encoder:
    def __init__(self):
        self.documents = []
        self.queries = []
    def encode_text(self, texts, **kwargs):
        self.documents.extend(texts)
        return np.array([[1.,0.] if 'cabinet' in text else [0.,1.] for text in texts])
    def encode_query(self, question):
        self.queries.append(question)
        return np.array([[1.,0.]])


def fine(identity, text):
    return {'event_id':identity,'start_time':10,'end_time':20,'status':'refined',
            'event_caption_refined':text}


def test_incremental_index_reuses_unchanged_events_and_rejects_stale_event_vectors(tmp_path):
    encoder = Encoder()
    index = ShortTermSemanticIndex(tmp_path, encoder, namespace='test')
    events = [fine('one','The person opens the cabinet.'), fine('two','The person walks away.')]
    assert index.build(events)['encoded'] == 2
    assert index.build(events)['encoded'] == 0
    assert len(encoder.documents) == 2
    assert index.scores('他打开柜子了吗',events) == {'one':1.,'two':0.}
    changed = [fine('one','The person walks away.'), events[1]]
    assert set(index.scores('question',changed)) == {'two'}
    assert index.last_search['status'] == 'partial'
    assert index.build(changed)['encoded'] == 1
    assert index.scores('question',changed)['one'] == 0.
    # A separately loaded reader uses the same persisted vectors.
    reader = ShortTermSemanticIndex(tmp_path, encoder, namespace='test')
    assert reader.scores('question',changed) == {'one':0.,'two':0.}
    assert ShortTermSemanticIndex(tmp_path, encoder, namespace='different-model').scores('question',events) == {}


def test_missing_index_does_not_compute_document_or_query_vectors_in_question_path(tmp_path):
    encoder = Encoder()
    index = ShortTermSemanticIndex(tmp_path, encoder, namespace='empty')
    assert index.scores('question',[fine('one','A cabinet.')]) == {}
    assert encoder.documents == [] and encoder.queries == []


def test_corrupt_optional_index_falls_back_without_failing_a_question(tmp_path):
    encoder = Encoder()
    index = ShortTermSemanticIndex(tmp_path, encoder, namespace='corrupt')
    index.root.mkdir(parents=True)
    index.path.write_bytes(b'broken cached vectors')
    assert index.scores('question',[fine('one','A cabinet.')]) == {}
    assert encoder.documents == [] and encoder.queries == []


def test_prefix_index_preparation_cannot_evict_other_published_event_vectors(tmp_path):
    encoder=Encoder()
    index=ShortTermSemanticIndex(tmp_path,encoder,namespace='prefix')
    events=[fine('one','The cabinet opens.'),fine('two','The person walks away.')]
    index.build(events)
    index.build(events[:1])
    assert set(index.scores('question',events))=={'one','two'}
    assert index.last_search['status']=='ready'
    assert len(encoder.documents)==2


def test_dense_and_lexical_paths_keep_independent_candidates_and_original_signals():
    events=[fine('lexical','The person carries an umbrella.'),
            fine('semantic','The person opens the cabinet.'),
            fine('distractor','The room is quiet.')]
    class Index:
        last_search={'status':'ready'}
        def prepare(self, events): pass
        def scores(self, question, events):
            return {event['event_id']: {'semantic':.98,'distractor':.1,'lexical':.05}[event['event_id']] for event in events}
    store=SimpleNamespace(get_state=lambda:{'mst_version':1},load_events=lambda:events)
    result=MSTRetriever(store,semantic_index=Index()).search('Earlier, who carried the umbrella?',top_k=2)
    by_id={row['event_id']:row for row in result}
    assert set(by_id)=={'semantic','lexical'}
    assert by_id['semantic']['bm25_score']==0 and by_id['semantic']['semantic_score']==.98
    assert by_id['lexical']['bm25_score']>0 and by_id['lexical']['semantic_score']==.05
    fused=MemoryFusion().fuse({'short_term_results':result},{},{'final_evidence_k':2},'entity_tracking')
    sem=next(row for row in fused['final_evidence'] if row['evidence_id']=='semantic')
    assert sem['semantic_score']==.98 and sem['visual_score']==0


def test_dense_path_never_sees_candidates_outside_explicit_time_scope():
    events=[{**fine('past','The cabinet opens.'),'start_time':10,'end_time':20},
            {**fine('future','The cabinet closes.'),'start_time':80,'end_time':90}]
    class Index:
        last_search={'status':'ready'}
        def prepare(self, events): pass
        def scores(self,question,eligible):
            assert [event['event_id'] for event in eligible]==['past']
            return {'past':.8}
    store=SimpleNamespace(get_state=lambda:{'mst_version':1},load_events=lambda:events)
    result=MSTRetriever(store,semantic_index=Index()).search('第10～20秒做了什么',top_k=16)
    assert [row['event_id'] for row in result]==['past']


@pytest.mark.parametrize('question',['较早拿过什么东西？','第10～20秒拿着什么？','之前说了什么？'])
def test_history_details_remain_eligible_after_complete_lt_aggregation(question):
    state={'short_term_ready':True,'long_term_ready':True,'current_ready':True,'current_stale':False,
           'mst_time_span':[0,120],'mlt_time_span':[0,120],'mcur_time_span':[90,120]}
    assert MemoryRouter().route(question,{},state)['memory_route']['use_short_term']
    assert not MemoryRouter().route(question,{'use_short_term':False},state)['memory_route']['use_short_term']
    out_of_range={**state,'mst_time_span':[60,120]}
    explicit='第10～20秒拿着什么？'
    assert not MemoryRouter().route(explicit,{},out_of_range)['memory_route']['use_short_term']


def test_selective_other_person_caption_can_still_be_inspected():
    event={'evidence_id':'scene','source_memory':'M_st','start_time':10,'end_time':20,
        'keyframe_paths':['frame_000015.jpg'],
        'visual_observations':[{'timestamp':15,'path':'frame_000015.jpg',
            'actor':'rear passenger','action':'holds','objects':['bottle'],
            'description':'The rear passenger holds a bottle.'}]}
    output=EvidencePacker().pack('前排的人拿过手机吗？',{'use_image_evidence':True,'max_image_evidence':3},
                                {'final_evidence':[event]})
    assert output['selected_image_paths_for_mllm']==['frame_000015.jpg']


def test_other_actor_text_stays_optional_context_with_its_original_actor():
    from online_evidence_content import prompt_evidence
    observation={'timestamp':15,'path':'frame_000015.jpg','actor':'rear passenger',
                 'action':'holds','objects':['bottle'],'description':'The rear passenger holds a bottle.'}
    event={'evidence_id':'scene','source_memory':'M_st','caption':'Front and rear passengers sit together.',
           'start_time':10,'end_time':20,'visual_observations':[observation]}
    view,_=prompt_evidence('前排的人拿过手机吗？',[event],max_tokens=1000)
    assert view[0]['visual_observations']==[observation]


@pytest.mark.parametrize('kind',['recent_recall','visual_attribute','general_qa','speech_query'])
def test_source_protection_is_not_limited_to_temporal_query_labels(kind):
    long=[{'memory_id':f'history{i}','start_time':i*30,'end_time':i*30+30,
           'caption':'A cabinet is opened, then closed.','score':.1} for i in range(2)]
    short=[{**fine(f'fine{i}','People are seated.'),'score':.9,'start_time':100+i*10,'end_time':110+i*10}
           for i in range(16)]
    result=MemoryFusion().fuse({'text_results':long,'short_term_results':short},
                               {},{'final_evidence_k':12},kind)
    assert {'history0','history1'}.issubset({row['evidence_id'] for row in result['final_evidence']})
    assert len(result['final_evidence'])==12


@pytest.mark.parametrize('kind',['recent_recall','followup','general_qa','temporal_reasoning'])
def test_unspecified_text_capacity_keeps_independent_history_and_fine_details(monkeypatch,kind):
    for name in ('EM2MEM_UNIFIED_FINAL_EVIDENCE_K','EM2MEM_RECENT_FINAL_EVIDENCE_K'):
        monkeypatch.delenv(name,raising=False)
    plan=RetrievalPlanner().plan({'query_type':kind,'memory_route':{
        'use_long_term':True,'use_short_term':True}}, {'max_image_evidence':6},
        {'long_term_ready':True,'short_term_ready':True})
    layers=plan['retrieval_plan']
    assert plan['final_evidence_k']==min(32,
        layers['M_lt']['candidate_budget']+layers['M_st']['candidate_budget'])
    long=[{'memory_id':f'history{i}','start_time':i*30,'end_time':i*30+30,
           'caption':'A cabinet changes state.','score':.1} for i in range(8)]
    short=[{**fine(f'fine{i}','People are seated.'),'score':.9,'start_time':300+i*10,'end_time':310+i*10}
           for i in range(16)]
    result=MemoryFusion().fuse({'text_results':long,'short_term_results':short},{},plan,kind)
    assert len(result['final_evidence'])==24
    assert sum(row['source_memory']=='M_lt' for row in result['final_evidence'])==8
    assert plan['max_image_evidence']==6


def test_explicit_text_capacity_and_disabled_sources_remain_authoritative(monkeypatch):
    planner=RetrievalPlanner()
    decision={'query_type':'recent_recall','memory_route':{'use_short_term':True}}
    state={'short_term_ready':True,'long_term_ready':True}
    for name in ('EM2MEM_UNIFIED_FINAL_EVIDENCE_K','EM2MEM_RECENT_FINAL_EVIDENCE_K'):
        monkeypatch.delenv(name,raising=False)
    plan=planner.plan(decision,{},state)
    assert plan['final_evidence_k']==16 and not plan['retrieval_plan']['M_lt']['enabled']
    monkeypatch.setenv('EM2MEM_UNIFIED_FINAL_EVIDENCE_K','14')
    assert planner.plan(decision,{},state)['final_evidence_k']==14
    monkeypatch.setenv('EM2MEM_RECENT_FINAL_EVIDENCE_K','7')
    assert planner.plan(decision,{},state)['final_evidence_k']==7
    assert planner.plan(decision,{'final_evidence_k':4},state)['final_evidence_k']==4


def test_top_sparse_event_keeps_temporal_inspection_despite_rich_lower_events():
    top={'evidence_id':'top','source_memory':'M_lt','start_time':10,'end_time':20,
         'caption':'A cabinet changes from closed to open.','final_score':.9,
         'keyframe_paths':[f'frame_{t:06d}.jpg' for t in [9,10,15,20,21]],
         'evidence_frames':[{'path':f'frame_{t:06d}.jpg','timestamp':t} for t in [9,10,15,20,21]]}
    fine_events=[{'evidence_id':f'fine{i}','source_memory':'M_st','start_time':100+i*10,'end_time':105+i*10,
        'caption':'A person stands beside a cabinet.','final_score':.8,
        'keyframe_paths':[f'frame_{100+i*10:06d}.jpg'],
        'visual_observations':[{'timestamp':100+i*10,'path':f'frame_{100+i*10:06d}.jpg',
                                'description':'A person opens a cabinet.'}]} for i in range(4)]
    packed=EvidencePacker().pack('Did the person open the cabinet?',
        {'use_image_evidence':True,'max_image_evidence':6,'final_evidence_k':6},
        {'final_evidence':[top]+fine_events})
    paths=packed['selected_image_paths_for_mllm']
    assert {'frame_000010.jpg','frame_000020.jpg'}.issubset(paths)
    assert 'frame_000009.jpg' not in paths and 'frame_000021.jpg' not in paths
    assert any(int(path[6:12])>=100 for path in paths) and len(paths)<=6


@pytest.mark.parametrize('shape',['sdk','dict','json_envelope','raw_json','content_parts'])
def test_gateway_response_shapes_are_read_without_assuming_choices_attribute(shape):
    content=json.dumps({'visual_summary':'A wooden cabinet is open.'})
    envelope={'choices':[{'message':{'content':content}}]}
    response={'sdk':SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]),
              'dict':envelope,'json_envelope':json.dumps(envelope),'raw_json':content,
              'content_parts':{'choices':[{'message':{'content':[{'type':'text','text':content}]}}]}}[shape]
    assert json.loads(chat_response_text(response))['visual_summary']=='A wooden cabinet is open.'


def test_visual_request_includes_json_instruction_in_user_content(tmp_path):
    from test_visual_summary_generation import _builder_with_api_boundary, _events, _write_frame, WINDOW
    path=_write_frame(tmp_path)
    builder,calls=_builder_with_api_boundary({'visual_summary':'A cabinet is open.'})
    builder.build_episode(WINDOW,_events(path),session_id='test',session_dir=tmp_path)
    request=next(call for call in calls if isinstance(call['messages'][-1]['content'],list))
    assert 'json' in request['messages'][-1]['content'][0]['text'].lower()


def test_short_term_refinement_accepts_a_json_string_gateway_response(tmp_path):
    from online_short_term.micro_event_refiner import MicroEventRefiner
    from test_visual_summary_generation import _write_frame
    path=_write_frame(tmp_path)
    refiner=MicroEventRefiner(backend='openai',model='offline')
    refiner._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs:json.dumps({'refined_caption':'A cabinet is open.','confidence':.9}))))
    event={**fine('one',''),'status':'provisional','keyframes':[{'path':path,'timestamp':15}]}
    refined=refiner.refine_event(event,tmp_path)
    assert refined['status']=='refined' and refined['event_caption_refined']=='A cabinet is open.'


def test_visual_repair_publishes_new_snapshot_without_changing_text_facts(memory_session):
    from online_memory.visual_summary_repair import active_records, publish_repair, record_fingerprint, record_id
    root,sid,layout=memory_session
    before_config,before=active_records(root/sid)
    semantic={p.name:p.read_bytes() for p in layout.semantic_root.glob('*.json')}
    record=before[0]
    draft={record_id(record):{'fingerprint':record_fingerprint(record),
        'visual_summary':'A red bag rests on a light seat.','visual_summary_status':'generated',
        'visual_summary_debug':{'source':'test'},'generated_at':'2026-11-06T00:00:00Z'}}
    result=publish_repair(root/sid,draft,runtime_validator=lambda *_:None)
    config,after=active_records(root/sid)
    assert result['version']==2 and config['memory_version']==2
    assert after[0]['visual_summary']=='A red bag rests on a light seat.'
    assert after[0]['text']==record['text']
    assert semantic=={p.name:p.read_bytes() for p in layout.semantic_root.glob('*.json')}
    assert json.loads(layout.caption_30sec_path.read_text())==before
    assert publish_repair(root/sid,draft,runtime_validator=lambda *_:None)['status']=='unchanged'


def test_failed_visual_repair_validation_keeps_active_publication(memory_session):
    from online_memory.visual_summary_repair import active_records, publish_repair, record_fingerprint, record_id
    root,sid,layout=memory_session
    before_config,before=active_records(root/sid);record=before[0]
    config_bytes=layout.memory_config_path.read_bytes()
    draft={record_id(record):{'fingerprint':record_fingerprint(record),'visual_summary':'A bag is visible.',
                             'visual_summary_status':'generated'}}
    with pytest.raises(Exception,match='候选查询运行时加载失败'):
        publish_repair(root/sid,draft,runtime_validator=lambda *_:(_ for _ in ()).throw(RuntimeError('bad index')))
    assert layout.memory_config_path.read_bytes()==config_bytes
    assert active_records(root/sid)[1]==before


from test_memory_editor_connection import memory_session
