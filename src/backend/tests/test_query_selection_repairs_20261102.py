"""Behavioral regressions for retaining evidence needed by a question."""
from online_query.memory_fusion import MemoryFusion
from online_query_constraints import parse_query_constraints


def fuse(events, *, constraints=None, current=False, cache=None, limit=2):
    memories = {'text_results': events}
    if current:
        memories['current_results'] = [{
            'current_context': {'state': {'window_start_time': 90, 'window_end_time': 100}},
        }]
    return MemoryFusion().fuse(
        memories,
        {'query_constraints': constraints or {},
         'memory_priority': {'M_cur': .8, 'M_st': .8, 'M_lt': .85}},
        {'final_evidence_k': limit}, 'temporal_reasoning', cache or {},
    )


def event(name, score, **extra):
    return {'memory_id': name, 'segment_id': name, 'start_time': 10,
            'end_time': 20, 'caption': 'Observed historical event', 'score': score, **extra}


def test_joint_inspection_candidate_survives_distractors_and_current_reservation():
    positive = event('joint', .01, supplement_match_score=.6,
                     requirement_support={'confirmed': False, 'simultaneous': True,
                                          'observation_matched': True, 'conditions_co_mentioned': True,
                                          'requested_conditions': ['mask', 'sunglasses'],
                                          'signals': {'mask': 'confirmed', 'sunglasses': 'confirmed'}})
    results = fuse([event('related', .95), positive], current=True,
                   constraints={'requires_current': True, 'requires_history': True,
                                'requirements': {'simultaneous': True}})
    selected = results['final_evidence']
    assert len(selected) == 2
    assert any(row['source_memory'] == 'M_cur' for row in selected)
    assert any(row['evidence_id'] == 'joint' for row in selected)


def test_single_condition_supplement_does_not_take_joint_evidence_slot():
    weak = event('one_condition', None, supplemental=True, supplement_match_score=.99,
                 score_policy='unscored_supplemental_observation',
                 requirement_support={'confirmed': False, 'simultaneous': True,
                                      'requested_conditions': ['mask', 'sunglasses'],
                                      'signals': {'mask': 'confirmed', 'sunglasses': 'unknown'}})
    results = fuse([weak, event('ordinary', .9)], current=True,
                   constraints={'requires_current': True, 'requires_history': True,
                                'requirements': {'simultaneous': True}})
    assert {row['evidence_id'] for row in results['final_evidence']} - {
        row['evidence_id'] for row in results['final_evidence'] if row['source_memory'] == 'M_cur'
    } == {'ordinary'}


def test_unscored_supplement_keeps_missing_raw_score_and_separate_match_signal():
    row = event('new', None, raw_score=None, supplemental=True,
                supplement_match_score=.9, score_policy='unscored_supplemental_observation')
    selected = fuse([row], limit=1)['final_evidence'][0]
    assert selected['retrieval_score'] == 0
    assert selected['raw_retrieval_score'] is None
    assert selected['metadata']['supplement_match_score'] == .9
    assert selected['score_policy'] == 'unscored_supplemental_observation'


def test_unbound_followup_does_not_disable_temporal_evidence_coverage():
    memories = {'text_results': [event('old', .01)], 'short_term_results': [
        {'event_id': 'recent', 'start_time': 80, 'end_time': 90,
         'event_caption_refined': 'Recent related event', 'score': .95},
    ]}
    result = MemoryFusion().fuse(
        memories, {'memory_priority': {'M_st': .9, 'M_lt': .8}},
        {'final_evidence_k': 2}, 'temporal_reasoning',
        {'is_followup': True, 'binding_valid': False, 'reference_ambiguous': True,
         'confidence': .4, 'referenced_time_ranges': []},
    )
    assert result['fusion_summary']['selection_policy'] == 'long_term_context_and_short_term_details'


def test_explicitly_invalid_binding_cannot_filter_history_to_stale_point():
    result = fuse([event('observed', .5)], cache={
        'is_followup': True, 'binding_valid': False, 'temporal_relation': 'same_time',
        'confidence': .95, 'referenced_time_ranges': [{'start': 50, 'end': 50}],
    })
    assert [row['evidence_id'] for row in result['final_evidence']] == ['observed']


def test_summary_until_present_requires_current_endpoint():
    constraints = parse_query_constraints('帮我把刚才到现在前面车流的变化捋一下。', 100)
    assert constraints['summary']
    assert constraints['requires_history']
    assert constraints['requires_current']
    assert constraints['time_range'] is None


def test_present_posture_comparison_keeps_historical_side():
    constraints = parse_query_constraints('跟前面能认出手机的时候比，他现在拿手机的姿势变了吗？', 100)
    assert constraints['requirements']['comparison']
    assert constraints['requires_current']
    assert constraints['requires_history']


def test_present_clothing_comparison_keeps_historical_side():
    constraints = parse_query_constraints('她现在穿的跟前面有什么变化？', 100)
    assert constraints['requirements']['comparison']
    assert constraints['requires_current']
    assert constraints['requires_history']


def test_spatial_front_vehicle_question_remains_current_only():
    constraints = parse_query_constraints('现在前面有什么车？', 100)
    assert constraints['requires_current']
    assert not constraints['requires_history']


def test_canonical_dedup_keeps_confirmed_support_from_lower_scored_copy():
    support = {'confirmed': True, 'simultaneous': True,
               'requested_conditions': ['mask', 'sunglasses'],
               'signals': {'mask': 'confirmed', 'sunglasses': 'confirmed'}}
    ordinary = event('ordinary_copy', .9, canonical_segment_id='same', requirement_support={})
    checked = event('checked_copy', .01, canonical_segment_id='same', requirement_support=support,
                    supplement_match_score=.8)
    selected = fuse([ordinary, checked], constraints={
        'requirements': {'simultaneous': True}}, limit=1)['final_evidence']
    assert len(selected) == 1
    assert selected[0]['metadata']['requirement_support']['confirmed'] is True
    assert selected[0]['retrieval_score'] == .9


def test_explicit_two_windows_keep_joint_support_and_other_window():
    positive = event('joint', .01, requirement_support={
        'confirmed': False, 'simultaneous': True,
        'observation_matched': True, 'conditions_co_mentioned': True,
        'requested_conditions': ['mask', 'sunglasses'],
        'signals': {'mask': 'confirmed', 'sunglasses': 'confirmed'}})
    other_window = {**event('other_window', .8), 'start_time': 85, 'end_time': 95}
    selected = fuse([event('nearby', .99), positive, other_window], constraints={
        'time_ranges': [[0, 30], [80, 100]],
        'requirements': {'simultaneous': True}}, limit=2)['final_evidence']
    assert {row['evidence_id'] for row in selected} == {'joint', 'other_window'}


def test_change_question_after_past_comparison_implies_present_side():
    constraints = parse_query_constraints('旁边那个座位跟刚才相比，有变化吗？', 100)
    assert constraints['requires_current']
    assert constraints['requires_history']
    assert constraints['time_range'] is None


def test_historical_comparison_target_does_not_become_present():
    constraints = parse_query_constraints('当时跟更早之前相比，有什么变化？', 100)
    assert not constraints['requires_current']
    assert constraints['requires_history']


def test_numbered_historical_comparison_does_not_require_present():
    constraints = parse_query_constraints('与10秒时相比，20秒时有变化吗？', 100)
    assert not constraints['requires_current']
