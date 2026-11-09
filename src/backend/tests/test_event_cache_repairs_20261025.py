from online_query.coreference_resolver import CoreferenceResolver
from online_query.interaction_cache import InteractionCache
from online_query.memory_fusion import MemoryFusion


def cache(tmp_path, turns):
    result = InteractionCache('repair', tmp_path)
    result.data['recent_interactions'] = turns
    return result


def turn(name, question, start):
    return {'turn_id': name, 'question': question, 'answer': '', 'anchor_status': 'confirmed',
            'time_ranges': [{'start': start, 'end': start + 2}],
            'retrieved_memory_ids': [name], 'evidence_frames': [{'path': name + '.jpg', 'timestamp': start}]}


def test_after_hint_clipped_to_observed_video(tmp_path):
    result = CoreferenceResolver().resolve('What happened after that?', cache(tmp_path, [turn('a', 'person sitting', 10)]), latest_seconds=20)
    assert result['referenced_time_ranges'][0]['end'] == 20
    assert result['referenced_evidence_frames'] == []


def test_chinese_after_reference_uses_clipped_search_hint(tmp_path):
    result = CoreferenceResolver().resolve('那以后呢？', cache(tmp_path, [turn('a', '男子坐着', 10)]), latest_seconds=20)
    assert result.get('temporal_relation') == 'after'
    assert result['referenced_time_ranges'][0]['end'] == 20
    assert result['referenced_evidence_frames'] == []


def test_action_reference_selects_older_turn_with_its_provenance(tmp_path):
    turns = [turn('action', 'When did the passenger lift the phone?', 10), turn('interruption', 'What color is the dashboard?', 30)]
    turns[0]['answer'] = 'The passenger did lift the phone at 10 seconds.'
    result = CoreferenceResolver().resolve('At that time when the passenger lift the phone, was he smiling?', cache(tmp_path, turns))
    assert result['referenced_memory_ids'] == ['action']
    assert result['referenced_evidence_frames'][0]['path'] == 'action.jpg'
    assert result['matched_turn_id'] == 'action'


def test_ambiguous_repeated_action_does_not_force_hard_time(tmp_path):
    turns = [turn('first', 'passenger lift phone', 10), turn('second', 'passenger lift phone', 30)]
    for item in turns:
        item['answer'] = 'The passenger did lift the phone.'
    result = CoreferenceResolver().resolve('At that time when the passenger lift phone, was he smiling?', cache(tmp_path, turns))
    assert result.get('temporal_relation') != 'same_time'
    assert result.get('reference_ambiguous') is True


def fuse(context, results=None, constraints=None, k=2):
    return MemoryFusion().fuse(results or {}, {'query_constraints': constraints or {}, 'memory_priority': {'M_lt': 1, 'M_cur': 0}}, {'final_evidence_k': k}, 'visual_question', context)['final_evidence']


def test_search_only_cache_hints_never_become_final_facts():
    assert fuse({'is_followup': True, 'referenced_time_ranges': [{'start': 12, 'end': 20}], 'referenced_memory_ids': ['old'], 'referenced_entities': [{'name': 'phone'}]}) == []


def test_real_cached_frame_keeps_provenance():
    evidence = fuse({'is_followup': True, 'referenced_evidence_frames': [{'path': 'real.jpg', 'timestamp': 12, 'source': 'observed', 'segment_id': 'actual'}]})
    assert evidence[0]['metadata']['segment_id'] == 'actual'


def test_low_confidence_cache_window_cannot_exclude_real_evidence():
    evidence = fuse({'temporal_relation': 'same_time', 'confidence': .3, 'referenced_time_ranges': [{'start': 10, 'end': 12}]}, {'text_results': [{'memory_id': 'real', 'start_time': 30, 'end_time': 32, 'caption': 'observed'}]})
    assert [e['evidence_id'] for e in evidence] == ['real']


def test_mixed_current_history_reserves_current_within_budget():
    results = {'current_results': [{'current_context': {'state': {'window_start_time': 90, 'window_end_time': 100}}}], 'text_results': [{'memory_id': str(i), 'start_time': i, 'end_time': i + 1, 'caption': 'observed', 'score': 1} for i in range(4)]}
    evidence = fuse({}, results, {'requires_current': True, 'requires_history': True})
    assert len(evidence) == 2
    assert {e['source_memory'] for e in evidence} == {'M_cur', 'M_lt'}


def test_conflicting_actor_reference_does_not_bind_wrong_turn(tmp_path):
    result = CoreferenceResolver().resolve('At that time when the driver lift phone, was he smiling?', cache(tmp_path, [turn('wrong', 'passenger lift phone', 10)]))
    assert result.get('temporal_relation') != 'same_time'
    assert result.get('referenced_time_ranges') == []


def test_unrelated_specific_reference_does_not_bind_latest(tmp_path):
    result = CoreferenceResolver().resolve('At that time when the cyclist crossed the street, was he smiling?', cache(tmp_path, [turn('wrong', 'passenger lift phone', 10)]))
    assert result.get('temporal_relation') != 'same_time'


def test_chinese_action_reference_survives_interruption(tmp_path):
    turns = [turn('action', '乘客拿起手机时穿什么衣服？', 10), turn('interruption', '仪表盘是什么颜色？', 30)]
    turns[0]['answer'] = '乘客在10秒拿起手机。'
    result = CoreferenceResolver().resolve('当时乘客拿起手机的时候，他戴眼镜吗？', cache(tmp_path, turns))
    assert result['referenced_memory_ids'] == ['action']


def test_prior_question_without_an_observed_answer_is_not_an_action_anchor(tmp_path):
    result = CoreferenceResolver().resolve(
        'At that time when the passenger lift the phone, was he smiling?',
        cache(tmp_path, [turn('unanswered', 'When did the passenger lift the phone?', 10)]))
    assert result['referenced_time_ranges'] == []
    assert result.get('temporal_relation') != 'same_time'


def test_current_source_optout_and_zero_final_budget_remain_binding():
    results = {'current_results': [{'current_context': {'state': {'window_start_time': 90, 'window_end_time': 100}}}], 'text_results': [{'memory_id': 'old', 'start_time': 1, 'end_time': 2, 'caption': 'observed'}]}
    decision = {'query_constraints': {'requires_current': True, 'requires_history': True}, 'memory_route': {'use_current': False, 'use_long_term': True}}
    fusion = MemoryFusion()
    selected = fusion.fuse(results, decision, {'final_evidence_k': 2, 'retrieval_plan': {'M_cur': {'enabled': False}}}, 'general_qa')['final_evidence']
    assert {e['source_memory'] for e in selected} == {'M_lt'}
    assert fusion.fuse(results, {}, {'final_evidence_k': 0}, 'general_qa')['final_evidence'] == []


def test_mixed_budget_one_reports_missing_history():
    result = MemoryFusion().fuse({'current_results': [{'current_context': {}}], 'text_results': [{'memory_id': 'old', 'start_time': 1, 'end_time': 2}]}, {'query_constraints': {'requires_current': True, 'requires_history': True}}, {'final_evidence_k': 1}, 'general_qa')
    assert len(result['final_evidence']) == 1
    assert result['fusion_summary']['missing_required_sources'] == [['M_cache', 'M_lt', 'M_st']]


def test_matched_turn_cannot_borrow_unrelated_hot_ranges(tmp_path):
    old = turn('action', 'passenger lift phone', 10)
    old['time_ranges'] = []
    old['evidence_frames'] = []
    c = cache(tmp_path, [old, turn('other', 'dashboard color', 30)])
    c.data['hot_time_ranges'] = [{'start': 90, 'end': 92}]
    c.data['hot_entities'] = [{'name': 'unrelated hot entity'}]
    result = CoreferenceResolver().resolve('At that time when passenger lift phone, was he smiling?', c)
    assert result['referenced_time_ranges'] == []
    assert result['referenced_entities'] == []


def test_history_requirement_preserves_selected_comparison_supplements():
    results = {'text_results': [
        {'memory_id': 'before', 'start_time': 1, 'end_time': 2, 'supplemental': True, 'score': .3},
        {'memory_id': 'after', 'start_time': 5, 'end_time': 6, 'supplemental': True, 'score': .3},
        {'memory_id': 'neighbor', 'start_time': 3, 'end_time': 4, 'score': 1}]}
    selected = fuse({}, results, {'requires_history': True}, k=2)
    assert {item['evidence_id'] for item in selected} == {'before', 'after'}


def test_cache_source_route_optout_overrides_observed_frames():
    result = MemoryFusion().fuse({}, {'memory_route': {'use_interaction_cache': False}}, {'final_evidence_k': 2}, 'general_qa', {'is_followup': True, 'referenced_evidence_frames': [{'path': 'real.jpg', 'timestamp': 12}]})
    assert result['final_evidence'] == []


def test_required_tail_decision_prevents_cached_same_time_exclusion():
    result = MemoryFusion().fuse({'current_results': [{'current_context': {'state': {'window_start_time': 90, 'window_end_time': 100}}}]}, {'requires_current_tail': True}, {'final_evidence_k': 1}, 'general_qa', {'temporal_relation': 'same_time', 'confidence': .95, 'referenced_time_ranges': [{'start': 10, 'end': 12}]})
    assert result['final_evidence'][0]['source_memory'] == 'M_cur'
