"""Local regressions for event bindings; no models, sessions, or live services."""
from types import SimpleNamespace

import pytest

from online_query import cache_schema
from online_query.coreference_resolver import CoreferenceResolver


def turn(name, question, answer, start, end=None, **fields):
    return dict(turn_id=name, question=question, answer=answer,
                anchor_status='confirmed', entities=[],
                time_ranges=[dict(start=start, end=start if end is None else end)],
                evidence_frames=[], **fields)


def resolve(question, turns, latest=200):
    cache = SimpleNamespace(latest_context=lambda: dict(
        recent_interactions=turns, latest_interaction=turns[-1]))
    return CoreferenceResolver().resolve(question, cache, latest_seconds=latest)


def test_zero_match_is_unbound_instead_of_latest_success():
    result = resolve('刚才低头俯身时，他戴什么？', [
        turn('unrelated', '他在做什么？', '他坐直看向车窗。', 150)])
    assert result['is_followup']
    assert result.get('binding_valid') is False
    assert not result.get('referenced_time_ranges')
    assert not result.get('matched_turn_id')


@pytest.mark.parametrize('question', [
    '回到刚才低头的时候，他手里有什么？',
    '右前排这位现在的姿势，跟刚才低头俯身时有什么不同？',
])
def test_action_synonym_finds_older_event_after_interruption(question):
    result = resolve(question, [
        turn('bending', '现在前排右边在做什么？', '前排右边的人低着头看向腿部。', 75),
        turn('interruption', '开头在做什么？', '坐直看向前方。', 0, 30)])
    assert result.get('binding_valid') is True
    assert result['matched_turn_id'] == 'bending'
    assert result['referenced_time_ranges'][0]['start'] == 75


def test_discussion_topic_uses_prior_question_without_inventing_an_action():
    result = resolve('跟刚才聊衣服的时候比，墨镜的位置有变化吗？', [
        turn('clothes', '先说衣服，她穿的跟前面有什么变化？', '她穿黑色羽绒服和白色上衣。', 390),
        turn('glasses', '现在墨镜在哪里？', '墨镜戴在眼前。', 540)], latest=540)
    assert result['matched_turn_id'] == 'clothes'
    assert result.get('binding_valid') is True
    assert result['temporal_relation'] == 'related'


def test_comparison_text_after_before_marker_participates_in_event_match():
    result = resolve('回到后来前方开阔的那次，当时左侧有什么？', [
        turn('change', '后来路况有什么变化？',
             '之前前车靠近；后来前方开阔，车辆拉远。', 90, 110),
        turn('other', '现在有车吗？', '现在有一辆车。', 118)], latest=118)
    assert result.get('matched_turn_id') == 'change'
    assert result.get('binding_valid') is True


@pytest.mark.parametrize('answer', [
    '没有拿起杯子，他仍坐着。', '无法确认他是否拿起杯子。',
    'He did not lift the phone.', 'He might have lifted the phone.',
])
def test_negated_or_uncertain_action_never_establishes_requested_event(answer):
    chinese = '杯子' in answer
    result = resolve('回到刚才拿起杯子的那次，他穿什么？' if chinese else
                     'When he lifted the phone, what was he wearing?', [
        turn('uncertain', '他拿起东西了吗？', answer, 60)])
    assert result.get('binding_valid') is False
    assert not result.get('referenced_time_ranges')


def test_summary_does_not_inherit_a_single_old_point_as_its_scope():
    result = resolve('帮我把刚才到现在前面车流的变化捋一下。', [
        turn('old', '前面情况变了吗？', '看不出明显变化。', 60),
        turn('open', '后来前面有没有开阔？',
             '有。之前车较近；之后车辆拉远，路口近处开阔。', 100, 110),
        turn('latest', '前面有事故吗？', '未见明显事故，当前近处开阔。', 118)], latest=118)
    assert result['resolved_question'] == '帮我把刚才到现在前面车流的变化捋一下。'
    assert result.get('binding_valid') is False
    assert not result.get('referenced_time_ranges')


@pytest.mark.parametrize('question', [
    '刚才那段画面中他戴墨镜吗？', '概括刚才那段画面中发生了什么。',
])
def test_confirmed_same_time_reference_still_binds_the_previous_event(question):
    result = resolve(question, [
        turn('scene', '他在做什么？', '他拿着包。', 90, 105)])
    assert result['binding_valid'] is True
    assert result['temporal_relation'] == 'same_time'
    assert result['referenced_time_ranges'] == [dict(start=90, end=105)]


def test_shared_object_does_not_confirm_opposite_action():
    result = resolve('回到乘客拿起杯子的时候，他戴眼镜吗？', [
        turn('lower', '乘客在做什么？', '乘客放下杯子。', 80)])
    assert result.get('binding_valid') is False
    assert not result.get('referenced_time_ranges')


def test_legacy_binding_requires_reliable_time_not_just_topic():
    helper = cache_schema.has_valid_binding
    reliable = dict(is_followup=True, confidence=.9,
                    referenced_time_ranges=[dict(start=10, end=20)])
    assert helper(reliable)
    assert not helper(dict(reliable, binding_valid=False))
    assert not helper(dict(reliable, reference_ambiguous=True))
    assert not helper(dict(reliable, confidence=.4))
    assert not helper(dict(is_followup=True, binding_valid=True,
                           referenced_entities=[dict(name='phone')]))


def test_unbound_followup_does_not_override_summary_router_policy():
    from online_query.query_router import QueryRouter
    question = '总结整个视频中车流的变化。'
    state = dict(current_ready=True, short_term_ready=True, long_term_ready=True)
    base = QueryRouter().route(question, session_context=state)
    unbound = QueryRouter().route(question, session_context=state, cache_context=dict(
        is_followup=True, binding_valid=False, confidence=.4, reference_ambiguous=True))
    assert unbound['query_type'] == base['query_type'] == 'long_term_summary'


def _mst_events():
    return [dict(event_id=name, start_time=start, end_time=start + 5,
                 event_caption_refined='A person holds a phone.',
                 retrieval_text='A person holds a phone.')
            for name, start in [('earlier', 10), ('later', 90)]]


@pytest.mark.parametrize('flags', [
    dict(binding_valid=False, confidence=.95),
    dict(confidence=.4),
    dict(confidence=.95, reference_ambiguous=True),
    dict(confidence=.95, anchor_status='unresolved'),
])
def test_mst_invalid_binding_cannot_filter_or_add_time_score(flags):
    from online_short_term.mst_retriever import MSTRetriever
    events = _mst_events()
    store = SimpleNamespace(get_state=lambda: dict(mst_version=1), load_events=lambda: events)
    retriever = MSTRetriever(store)
    context = dict(is_followup=True, temporal_relation='same_time',
                   referenced_time_ranges=[dict(start=10, end=15)], **flags)
    baseline = retriever.search('phone', top_k=2)
    actual = retriever.search('phone', top_k=2, cache_context=context)
    assert {hit['event_id'] for hit in actual} == {'earlier', 'later'}
    assert [(hit['event_id'], hit['score'], hit['cache_score']) for hit in actual] == [
        (hit['event_id'], hit['score'], hit['cache_score']) for hit in baseline]


@pytest.mark.parametrize('binding_fields', [dict(binding_valid=True), {}])
def test_mst_valid_or_legacy_binding_boosts_time_without_discarding_alternatives(binding_fields):
    from online_short_term.mst_retriever import MSTRetriever
    events = _mst_events()
    store = SimpleNamespace(get_state=lambda: dict(mst_version=1), load_events=lambda: events)
    context = dict(is_followup=True, temporal_relation='same_time', confidence=.95,
                   referenced_time_ranges=[dict(start=10, end=15)], **binding_fields)
    hits = MSTRetriever(store).search('phone', top_k=2, cache_context=context)
    assert [hit['event_id'] for hit in hits] == ['earlier','later']
    assert hits[0]['cache_score'] == .8


def test_mst_invalid_time_binding_retains_entity_topic_hint():
    from online_short_term.mst_retriever import MSTRetriever
    context = dict(binding_valid=False, confidence=.95,
                   referenced_time_ranges=[dict(start=10, end=15)],
                   referenced_entities=[dict(name='phone')])
    scorer = MSTRetriever(None)
    assert scorer._cache_context_score(_mst_events()[0], context) == .7
    assert scorer._cache_context_score(_mst_events()[1], context) == .7
