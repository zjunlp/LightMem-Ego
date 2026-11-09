"""Inspection candidates survive weak captions while explicit scope stays strict."""
import json
from pathlib import Path

import pytest
from online_query.evidence_packer import EvidencePacker


def event(name, times, descriptions, caption=''):
    return {'evidence_id': name, 'source_memory': 'M_st', 'start_time': min(times), 'end_time': max(times),
            'caption': caption, 'keyframe_paths': [f'stream/frames/frame_{t:06d}.jpg' for t in times],
            'metadata': {'visual_observations': [{'timestamp': t, 'path': f'stream/frames/frame_{t:06d}.jpg', 'description': d}
                                               for t, d in zip(times, descriptions)]}}


def pack(question, events, **constraints):
    return EvidencePacker().pack(question, {'max_image_evidence': 6, 'use_image_evidence': True,
        'final_evidence_k': 8, 'query_constraints': constraints}, {'final_evidence': events})


def sent_times(result):
    paths = result['selected_image_paths_for_mllm']
    return [f['timestamp'] for f in result['selected_evidence_frames'] if f['path'] in paths]


def test_weak_device_precedes_optional_other_actor_inspection():
    a = event('a', [0], ['The front-left man wearing glasses holds a phone.'])
    b = event('b', [81, 85], ['The front-left man holds a dark rectangular device in both hands.',
                            'The front-right man holds a phone.'])
    ts = sent_times(pack('找左侧前排戴眼镜的人拿手机的图', [a, b]))
    assert ts[:2] == [0, 81]  # The other-actor view remains a lower-priority inspection candidate.


def test_one_event_covers_both_requested_windows_and_excludes_gap():
    e = event('a', [20, 25, 50, 90, 95], ['The man holds a phone.'] * 5)
    ts = sent_times(pack('分别给两个时段手机图片', [e], time_ranges=[[20, 25], [90, 95]], time_explicit=True))
    assert any(20 <= t <= 25 for t in ts) and any(90 <= t <= 95 for t in ts)
    assert 50 not in ts and len(ts) <= 3


def test_simultaneous_exact_frame_beats_negative_and_shared_caption():
    e = event('a', [350, 353], ['The person wears a mask; no phone is clearly visible.',
                              'The person holds a smartphone in front of their masked face.'], 'The person wears a mask.')
    assert sent_times(pack('找同时戴口罩并拿手机的图', [e]))[0] == 353


def test_comparison_reserves_other_target_in_same_event():
    e = event('a', [25, 26, 31, 90], ['The man holds a phone.'] * 3 + ['The man wears sunglasses.'])
    ts = sent_times(pack('比较先拿手机还是先戴墨镜', [e]))
    assert 25 in ts and 90 in ts


def test_current_non_authoritative_keeps_preferred_latest():
    frames = [{'path': f'current/frames/cur_kf_{t*1000:09d}.jpg', 'timestamp': t, 'source': 'M_cur'} for t in [26, 16, 18]]
    p = EvidencePacker().pack('现在拿什么', {'query_type': 'current_perception', 'use_image_evidence': True, 'max_image_evidence': 3},
        {'current_selection': {'evidence_frames': frames, 'selected_image_paths_for_mllm': [f['path'] for f in frames]}})
    assert sent_times(p)[0] == 26


def test_only_final_events_and_existing_files_are_eligible(tmp_path):
    e = event('selected', [81, 90], ['The man holds a phone.'] * 2)
    existing = tmp_path / e['keyframe_paths'][0]
    existing.parent.mkdir(parents=True); existing.write_bytes(b'frame')
    other = event('not-selected', [1], ['The man holds a phone.'])
    out = EvidencePacker(tmp_path).pack('手机', {'use_image_evidence': True, 'max_image_evidence': 6},
        {'final_evidence': [e], 'short_term_results': [other]})
    assert sent_times(out) == [81]


@pytest.mark.parametrize('relative,required', [
    ('seven_phone', {81}),
    ('conditions', {353}),
    ('conditions_window', {353}),
    ('sequence', {25, 90}),
    ('summary', {0, 90, 116}),
])
def test_saved_response_pure_selection_replay(relative, required):
    path = Path(__file__).resolve().parent / 'fixtures/evidence' / (relative + '.json')
    raw = json.loads(path.read_text()); result = raw['result']
    route = dict(result['route_decision']); route['max_image_evidence'] = 6
    out = EvidencePacker().pack(raw['question'], route, {'final_evidence': result['selected_evidence']}, result.get('cache_context'))
    assert required <= set(sent_times(out))
    assert len(sent_times(out)) <= 6


def test_separate_phone_occurrences_in_one_event_keep_both_sides_of_lowering():
    e = event('a', [25, 26, 28, 31, 32], ['The man holds a phone at his ear.',
        'The man holds a phone at his ear.', 'The man lowers the phone away from his ear.',
        'The man holds a phone at his ear again.', 'The man holds a phone at his ear.'])
    ts = sent_times(pack('什么时候把手机贴到耳边', [e]))
    assert any(t in ts for t in [25, 26]) and any(t in ts for t in [31, 32])


def test_one_unverified_selected_event_capture_is_allowed_but_placeholder_current_is_not_padding():
    e = event('a', [81, 88], ['The front-left man holds a dark rectangular device low in both hands.',
        'A rectangular object stands on the console. The front-right man bends down, while the front-left man looks left.'])
    current = {'evidence_id': 'cur', 'source_memory': 'M_cur', 'caption': 'Current rolling video window; open',
               'keyframe_paths': ['stream/frames/frame_000097.jpg']}
    result = pack('找左侧前排拿手机的图', [e, current])
    assert sent_times(result) == [81,88]
    assert result['image_selection_summary']['inspection_fallback_count']==1


def test_saved_followup_prioritizes_each_cache_period_without_excluding_other_evidence():
    path = Path(__file__).resolve().parent / 'fixtures/evidence/followup.json'
    raw = json.loads(path.read_text()); result = raw['result']
    out = EvidencePacker().pack(raw['question'], result['route_decision'],
        {'final_evidence': result['selected_evidence']}, result['cache_context'])
    ts = sent_times(out)
    assert ts and len(ts)<=6
    assert all(23 <= t <= 26 or 30 <= t <= 31 for t in ts[:2])
    assert any(23 <= t <= 26 for t in ts) and any(30 <= t <= 31 for t in ts)


def test_explicit_point_state_verification_preserves_negative_evidence():
    e = event('a', [0, 30], ['The man has no sunglasses and holds a cup.', 'The man wears sunglasses.'])
    assert sent_times(pack('只看第0秒，他戴墨镜吗？请给图', [e], time_range=[0, 0], time_explicit=True)) == [0]


def test_explicit_multiple_points_state_verification_preserves_each_negative_or_unclear():
    e = event('a', [0, 20, 30], ['The man has no sunglasses.', 'The man wears sunglasses.',
                              'The man turns away; his eyes are unclear.'])
    assert set(sent_times(pack('分别给第0秒和第30秒，他是否戴墨镜？', [e], time_ranges=[[0, 0], [30, 30]], time_explicit=True))) == {0, 30}


def test_scoped_conjunction_verification_keeps_negative_alongside_positive():
    e = event('a', [350, 353], ['The person wears a mask but does not hold a phone.',
                              'The person wears a mask and holds a phone.'])
    assert set(sent_times(pack('350～353秒是否同时戴口罩拿手机？请给图', [e], time_range=[350, 353]))) == {350, 353}


def test_positive_lookup_keeps_positive_first_and_negative_as_inspection_context():
    e = event('a', [350, 353], ['The person does not hold a phone.', 'The person holds a phone.'])
    assert sent_times(pack('找350～353秒拿手机的图片', [e], time_range=[350, 353])) == [353, 350]


def test_state_verification_keeps_time_scope_with_selective_actor_captions():
    e = event('a', [0, 1, 30], ['The front-left man has no sunglasses.',
        'The front-right man has no sunglasses.', 'The front-left man wears sunglasses.'])
    assert sent_times(pack('只看第0～1秒，前排左侧的人戴墨镜吗？', [e], time_range=[0, 1])) == [0, 1]


def test_generic_direct_observation_can_supply_temporally_spread_inspection():
    e = event('a', [10, 20, 30], ['The man is seated.'] * 3)
    out = pack('找拿手机的画面', [e])
    assert sent_times(out) == [10, 30]
    assert out['image_selection_summary']['inspection_fallback_count'] == 2
    assert 'temporal coverage' in out['image_selection_summary']['inspection_fallback_reasons'][0]
    assert 'content unverified' in out['image_selection_summary']['inspection_fallback_reasons'][0]


def test_text_conjunction_does_not_remove_alternative_inspection_frame():
    e = event('a', [10, 20], ['The man is seated.', 'The man wears a mask and holds a phone.'])
    out = pack('找同时戴口罩拿手机的画面', [e])
    assert sent_times(out) == [20, 10]
    assert out['image_selection_summary']['inspection_fallback_count'] == 1
