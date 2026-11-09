"""Recall under uncertain annotations; explicit time and budgets remain strict."""
from types import SimpleNamespace
import pytest

from online_query.cache_schema import has_exact_frame_binding
from online_query.memory_fusion import MemoryFusion
from online_query.evidence_packer import EvidencePacker
from online_query.evidence_sufficiency import EvidenceSufficiencyEvaluator
from online_query.memory_router import MemoryRouter
from online_query_constraints import parse_query_constraints


def cache(exact=False):
    return {'is_followup':True,'binding_valid':True,'confidence':.9,'temporal_relation':'same_time',
            'referenced_time_ranges':[{'start':12,'end':12,
                'source':'explicit_previous_image' if exact else 'answer_cited_sent_frames'}]}


def memory(identity, start, end, text, score=.7):
    return {'memory_id':identity,'segment_id':identity,'start_time':start,'end_time':end,
            'caption':text,'score':score,'keyframe_paths':[f'{identity}.jpg'],
            'visual_observations':[{'timestamp':start,'path':f'{identity}.jpg','description':text}]}


@pytest.mark.parametrize('exact',[False,True])
def test_cached_citation_is_not_a_hard_window_but_explicit_frame_is(exact):
    memories={'text_results':[memory('anchor',12,12,'A person opens a parcel.'),
                              memory('alternative',35,40,'A person opens a parcel and takes out a mug.')]}
    context=cache(exact)
    fusion=MemoryFusion().fuse(memories,{'memory_priority':{'M_lt':1}}, {'final_evidence_k':2},'followup',context)
    ids={e['evidence_id'] for e in fusion['final_evidence']}
    assert ids == ({'anchor'} if exact else {'anchor','alternative'})
    packed=EvidencePacker().pack('What was inside the parcel at that time?',
        {'use_image_evidence':True,'max_image_evidence':2,'final_evidence_k':2},fusion,context)
    sent=packed['selected_image_paths_for_mllm']
    assert 'anchor.jpg' in sent
    assert ('alternative.jpg' in sent) is not exact
    assert len(sent)<=2 and has_exact_frame_binding(context) is exact


def test_explicit_user_window_wins_over_a_conflicting_cache_reference():
    source={'text_results':[memory('anchor',12,12,'A mug.'),memory('requested',35,40,'A mug.') ]}
    constraints={'time_ranges':[[35,40]],'time_explicit':True}
    result=MemoryFusion().fuse(source,{'query_constraints':constraints}, {'final_evidence_k':2},'detail',cache(True))
    assert [e['evidence_id'] for e in result['final_evidence']]==['requested']
    packed=EvidencePacker().pack('What was there from 35 to 40 seconds?',
        {'use_image_evidence':True,'max_image_evidence':2,'query_constraints':constraints},result,cache(True))
    assert packed['selected_image_paths_for_mllm']==['requested.jpg']


@pytest.mark.parametrize('question',[
    '厨师切菜时，旁边的人在做什么？', '机器人转身的时候，桌边的人在做什么？',
    'When the nurse opens the cabinet, what is the visitor doing?',
])
def test_event_clause_does_not_erase_history_in_favour_of_current_activity(question):
    scope=parse_query_constraints(question,500)
    assert scope['requires_history'] and not scope['requires_current']
    route=MemoryRouter().route(question,{}, {'current_ready':True,'short_term_ready':True,'long_term_ready':True},
                                query_constraints=scope)
    assert route['query_type']!='current_perception' and route['memory_route']['use_long_term']


@pytest.mark.parametrize('question',[
    '前面穿蓝衣那段，他拿着什么？', 'Earlier, was the box still open?',
    '他刚开始头上戴着什么？',
])
def test_history_cannot_reserve_the_current_tail_just_because_of_a_type_label(question):
    scope=parse_query_constraints(question,500)
    result=EvidenceSufficiencyEvaluator().evaluate(question=question,query_type='recent_recall',
        query_constraints=scope,short_term_results=[{'start_time':480,'end_time':490,'status':'refined'}],
        current_results=[{'current_context':{'window_end_time':500}}])
    assert not result['requires_current_tail']


def test_recent_request_still_protects_the_unpublished_tail():
    result=EvidenceSufficiencyEvaluator().evaluate(question='最近发生了什么？',query_type='recent_recall',
        short_term_results=[{'start_time':480,'end_time':490,'status':'refined'}],
        current_results=[{'current_context':{'window_end_time':500}}])
    assert result['requires_current_tail']


@pytest.mark.parametrize('question',[
    '最开始桌上的东西，和后来打开包的时候，哪些能看清？',
    'At the beginning and later, which tools can be identified?',
])
def test_multiple_phases_are_not_collapsed_to_zero(question):
    scope=parse_query_constraints(question,500)
    assert scope['time_ranges']==[] and scope['requirements']['comparison']


def test_sparse_descriptions_use_available_inspection_budget_without_resurrecting_excluded_events(tmp_path):
    events=[]
    for i in range(4):
        name=f'event{i}'; path=f'{name}.jpg'; (tmp_path/path).write_bytes(f'image{i}'.encode())
        events.append({'evidence_id':name,'source_memory':'M_lt','start_time':i*10,'end_time':i*10+9,
            'keyframe_paths':[path], 'visual_observations':[{'timestamp':i*10,'path':path,
                'description':'A person holds a bottle.'}]})
    output=EvidencePacker(tmp_path).pack('他拿过手机吗？',
        {'use_image_evidence':True,'max_image_evidence':6,'final_evidence_k':4},
        {'final_evidence':events,'evidence_frames':[{'path':'unselected.jpg','timestamp':99}]})
    assert output['selected_image_paths_for_mllm']==[f'event{i}.jpg' for i in range(4)]
    assert output['image_selection_summary']['inspection_fallback_count']==4


def test_earlier_preference_protects_early_event_and_its_image_without_forbidding_later_evidence():
    question='她刚开始在忙什么？'
    scope=parse_query_constraints(question,500)
    memories={'text_results':[memory('later',400,430,'A person closes a box.',.9),
                              memory('earlier',10,40,'A person opens a box.',.3)]}
    result=MemoryFusion().fuse(memories,{'query_constraints':scope}, {'final_evidence_k':2},'general_qa')
    assert result['final_evidence'][0]['evidence_id']=='earlier'
    output=EvidencePacker().pack(question,{'use_image_evidence':True,'max_image_evidence':2,'query_constraints':scope}, result)
    assert output['selected_image_paths_for_mllm'][0]=='earlier.jpg'
    assert 'later.jpg' in output['selected_image_paths_for_mllm']


def test_selector_alternative_keeps_scores_and_budget_without_padding_a_full_selection():
    from copy import deepcopy
    from retrieval_source_helpers import UNIFIED
    select=UNIFIED.EM2Memory._retain_retrieval_alternative
    pool=[{'doc_id':'chosen','episodic_score':.4},{'doc_id':'other','episodic_score':.8},
          {'doc_id':'third','episodic_score':.6}]
    before=deepcopy(pool)
    assert select(['chosen'],pool,6)==(['chosen','other'],'other')
    assert select(['chosen'],pool,1)==(['chosen'],None)
    assert select(['chosen','third'],pool,6)==(['chosen','third'],None)
    assert select([],pool,6)==([],None)
    assert pool==before


@pytest.mark.parametrize('initial_description', [
    'The person is not holding an umbrella.',
    'Only a partly cropped object is visible; its identity is unclear.',
])
def test_initial_later_comparison_keeps_the_initial_negative_or_uncertain_state(initial_description):
    from test_packer_inspection_coverage import event, sent_times
    question = 'At the beginning and later, was the person holding an umbrella?'
    scope = parse_query_constraints(question, 100)
    observed = event('one', [7, 20, 30], [initial_description,
        'The person holds an umbrella.', 'The person holds an umbrella.'])
    packed = EvidencePacker().pack(question, {'query_constraints': scope,
        'use_image_evidence': True, 'max_image_evidence': 2}, {'final_evidence': [observed]})
    times = sent_times(packed)
    assert scope['time_ranges'] == [] and times[0] == 7
    assert len(times) == 2 and times[1] in {20, 30}


@pytest.mark.parametrize('valid_binding', [False, True])
def test_uncaptioned_current_inspection_requires_a_valid_reference_and_stays_in_its_span(valid_binding):
    from test_packer_inspection_coverage import event, sent_times
    history = event('known', [12], ['The person holds an umbrella.'])
    current = {'evidence_id': 'cur', 'source_memory': 'M_cur', 'start_time': 10, 'end_time': 25,
        'caption': 'Current rolling video window; open',
        'keyframe_paths': [f'current/frames/cur_kf_{t*1000:09d}.jpg' for t in [12, 15, 25]]}
    context = {'binding_valid': valid_binding, 'is_followup': valid_binding,
        'confidence': .9 if valid_binding else .1, 'temporal_relation': 'related',
        'referenced_time_ranges': [{'start': 12, 'end': 16, 'source': 'answer_cited_sent_frames'}]}
    packed = EvidencePacker().pack('How is the person holding the umbrella?',
        {'use_image_evidence': True, 'max_image_evidence': 3},
        {'final_evidence': [history, current]}, context)
    times = sent_times(packed)
    assert 12 in times and 25 not in times
    assert (15 in times) is valid_binding
    assert len(times) <= 3


def test_inspection_reserve_can_fill_slots_left_by_final_per_event_limits():
    from test_packer_inspection_coverage import event, sent_times
    history = event('known', list(range(12, 18)), ['The person holds an umbrella.'] * 6)
    current = {'evidence_id': 'cur', 'source_memory': 'M_cur', 'start_time': 18, 'end_time': 25,
        'caption': 'Current rolling video window; open',
        'keyframe_paths': [f'current/frames/cur_kf_{t*1000:09d}.jpg' for t in [18, 19, 25]]}
    context = {'binding_valid': True, 'is_followup': True, 'confidence': .9, 'temporal_relation': 'related',
        'referenced_time_ranges': [{'start': 12, 'end': 20, 'source': 'answer_cited_sent_frames'}]}
    packed = EvidencePacker().pack('How is the person holding the umbrella?',
        {'use_image_evidence': True, 'max_image_evidence': 4}, {'final_evidence': [history, current]}, context)
    times = sent_times(packed)
    assert 19 in times and 25 not in times
    assert len(times) <= 4
    assert sum(12 <= t <= 17 for t in times) <= 3
