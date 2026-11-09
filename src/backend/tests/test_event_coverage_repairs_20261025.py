from online_query.evidence_packer import EvidencePacker
from online_query.evidence_sufficiency import EvidenceSufficiencyEvaluator
from online_query.evidence_validation import validate_evidence


def event(name, times):
    return {'evidence_id': name, 'source_memory': 'M_st', 'start_time': min(times), 'end_time': max(times),
            'keyframe_paths': [f'stream/frames/frame_{t:06d}.jpg' for t in times],
            'metadata': {'visual_observations': [{'path': f'stream/frames/frame_{t:06d}.jpg', 'timestamp': t,
                'description': d} for t, d in zip(times, ['The man picks up a bag.', 'The man opens the bag.', 'The man puts the bag down.'])]}}


def test_summary_constraint_selects_latest_endpoint_without_summary_query_type():
    out = EvidencePacker().pack('说说这段吧', {'query_type': 'recent_recall', 'query_constraints': {'summary': True},
        'use_image_evidence': True, 'max_image_evidence': 2}, {'final_evidence': [event('one', [10, 20, 30])]})
    assert {10, 30} == {f['timestamp'] for f in out['selected_evidence_frames'] if f['path'] in out['selected_image_paths_for_mllm']}


def test_recent_open_object_query_covers_distinct_timed_states():
    out = EvidencePacker().pack('刚才那个包怎么回事？', {'query_type': 'recent_recall', 'use_image_evidence': True,
        'max_image_evidence': 3}, {'final_evidence': [event('one', [10, 20, 30])]})
    assert len(out['selected_image_paths_for_mllm']) == 3


def test_sufficient_st_reports_unrefined_current_tail_without_changing_decision():
    st = {'event_id': 'closed', 'end_time': 601, 'event_caption_refined': 'A man walks across the room.',
          'score': 1, 'recency_score': 1, 'lexical_score': 1, 'bm25_score': 1}
    current = [{'current_context': {'window_start_time': 601, 'window_end_time': 610, 'open_event': {'start_time': 601}},
                'current_selection': {'evidence_frames': [{'timestamp': 606, 'path': 'current/606.jpg'}]}}]
    out = EvidenceSufficiencyEvaluator().evaluate(question='What just happened?', query_type='recent_recall',
        short_term_results=[st], current_results=current)
    assert out['sufficient'] and out['decision'] == 'use_st'
    assert out['requires_current_tail']
    assert out['current_tail_coverage']['tail_range'] == [601, 610]
    assert out['current_tail_coverage']['current_frame_times'] == [606]


def test_mixed_coverage_diagnostics_do_not_claim_visual_semantics():
    out = validate_evidence('Compare then and now', [{'source_memory': 'M_st'}], [],
        {'requires_current': True, 'requires_history': True})
    assert 'required_current_evidence_missing' in out['issues']
    assert 'required_history_evidence_missing' not in out['issues']


def test_saved_open_recent_holding_query_retains_later_object_state():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent / 'fixtures/evidence/open_bag.json'
    raw = json.loads(path.read_text()); result = raw['result']
    out = EvidencePacker().pack(raw['question'], result['route_decision'], {'final_evidence': result['selected_evidence']})
    times = {f['timestamp'] for f in out['selected_evidence_frames'] if f['path'] in out['selected_image_paths_for_mllm']}
    assert times.intersection({96, 97, 100})
    assert len(times) <= 6


def test_required_current_preference_survives_mixed_history_budget():
    history = event('history', [10, 20, 30])
    current_path = 'current/frames/cur_kf_000610000.jpg'
    current = {'evidence_id': 'tail', 'source_memory': 'M_cur', 'keyframe_paths': [current_path], 'start_time': 601, 'end_time': 610}
    for cap in (0, 1):
        out = EvidencePacker().pack('Compare then and now', {'query_type': 'recent_recall',
            'query_constraints': {'requires_current': True, 'requires_history': True},
            'use_image_evidence': True, 'max_image_evidence': cap}, {'final_evidence': [history, current],
            'current_selection': {'selected_image_paths_for_mllm': [current_path]}})
        assert out['selected_image_paths_for_mllm'] == ([current_path] if cap else [])


def test_current_tail_gap_and_stale_window_are_only_coverage_diagnostics():
    st = {'end_time': 601}
    current = [{'current_context': {'window_end_time': 610, 'is_stale': True}}]
    out = EvidenceSufficiencyEvaluator().evaluate(question='What just happened?', query_type='recent_recall',
        short_term_results=[st], current_results=current)
    assert not out['requires_current_tail']
    diagnostic = validate_evidence('What just happened?', [], [], {'current_tail_range': [601, 610]})
    assert 'current_tail_has_no_direct_image' in diagnostic['issues']


def test_explicit_historical_window_does_not_require_current_tail():
    current = [{'current_context': {'window_end_time': 610}}]
    for question in ('Only the first 30 seconds, what happened?', 'What happened earlier?'):
        out = EvidenceSufficiencyEvaluator().evaluate(question=question, query_type='recent_recall',
            short_term_results=[{'end_time': 601}], current_results=current)
        assert not out['requires_current_tail']


def test_same_time_cache_hint_only_hard_filters_when_reliable_and_no_required_current():
    history = event('history', [10, 20, 30])
    current_path = 'current/frames/cur_kf_000610000.jpg'
    current = {'evidence_id': 'tail', 'source_memory': 'M_cur', 'keyframe_paths': [current_path]}
    for cache_extra, constraint_extra, route_extra in [
        ({'confidence': .2}, {}, {}),
        ({'confidence': .9, 'reference_ambiguous': True}, {}, {}),
        ({'confidence': .9}, {'requires_current': True, 'requires_history': True}, {}),
        ({'confidence': .9}, {'requires_current_tail': True}, {}),
        ({'confidence': .9}, {}, {'requires_current_tail': True}),
        ({'confidence': .9}, {}, {'requires_current': True}),
    ]:
        out = EvidencePacker().pack('Show these moments', {'query_type': 'recent_recall',
            'query_constraints': {'summary': True, **constraint_extra}, 'use_image_evidence': True,
            'max_image_evidence': 6, **route_extra}, {'final_evidence': [history, current]},
            {'temporal_relation': 'same_time', 'referenced_time_ranges': [{'start': 10, 'end': 10}], **cache_extra})
        assert current_path in out['selected_image_paths_for_mllm']


def test_explicit_cached_frame_and_user_windows_stay_binding():
    for constraints, cache, expected in [
        ({'summary': True}, {'temporal_relation': 'same_time', 'confidence': .9,
            'referenced_time_ranges': [{'start': 10, 'end': 10, 'source':'explicit_previous_image'}]}, {10}),
        ({'summary': True, 'requires_current': True, 'time_range': [20, 20]},
            {'temporal_relation': 'same_time', 'confidence': .2,
             'referenced_time_ranges': [{'start': 10, 'end': 10}]}, {20}),
    ]:
        out = EvidencePacker().pack('Show these moments', {'query_constraints': constraints,
            'use_image_evidence': True, 'max_image_evidence': 6}, {'final_evidence': [event('history', [10, 20, 30])]}, cache)
        times = {frame['timestamp'] for frame in out['selected_evidence_frames'] if frame['path'] in out['selected_image_paths_for_mllm']}
        assert times == expected


def test_mixed_query_reserves_history_before_extra_current_images():
    history = event('history', [10])
    paths = ['current/frames/cur_kf_000100000.jpg', 'current/frames/cur_kf_000110000.jpg']
    current = {'evidence_id': 'tail', 'source_memory': 'M_cur', 'keyframe_paths': paths}
    for cap in (1, 2):
        out = EvidencePacker().pack('Compare earlier and now', {'query_constraints': {
            'requires_current': True, 'requires_history': True}, 'use_image_evidence': True,
            'max_image_evidence': cap}, {'final_evidence': [history, current],
            'current_selection': {'selected_image_paths_for_mllm': paths}})
        sent = out['selected_image_paths_for_mllm']
        assert sent[0] == paths[0]
        assert len(sent) == cap
        if cap == 2:
            assert history['keyframe_paths'][0] in sent
            assert out['evidence_pack_summary']['required_image_sources_missing'] == []
        else:
            assert out['evidence_pack_summary']['required_image_sources_missing'] == ['history']
