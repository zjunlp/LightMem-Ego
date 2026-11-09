"""Final evidence must reach dialogue focus without turning topics into facts."""
import json
from pathlib import Path

from online_query.interaction_cache import InteractionCache

FIXTURES = Path(__file__).resolve().parent / 'fixtures/evidence'

def save(tmp_path, result):
    cache = InteractionCache('focus', tmp_path, ttl_seconds=0)
    cache.update_from_query_result('他拿着什么？', '他拿着什么？', result)
    return cache.recent_interactions()[-1]


def event(name='violet umbrella', eid='used', source='M_st'):
    return {'evidence_id': eid, 'source_memory': source, 'start_time': 10, 'end_time': 12,
            'metadata': {'visual_objects': [{'name': name, 'confidence': .99}]}}


def test_saved_gray_backpack_survives_and_beats_background(tmp_path):
    p = FIXTURES / 'cache_bag.json'
    result = json.loads(p.read_text())['result']
    turn = save(tmp_path, result)
    assert any(e['name'] == 'gray backpack' for e in turn['entities'])
    assert [e['name'] for e in turn['focus_entities']] == ['gray backpack']
    assert any(e['source_memory'] == 'M_st' for e in turn['focus_evidence_refs'])
    assert all(e['source_memory'] != 'M_lt' for e in turn['focus_evidence_refs'])


def test_selected_entities_and_ids_replace_ghost_retrieval(tmp_path):
    result = {'answer': 'He holds a violet umbrella.', 'selected_evidence': [event()],
              'retrieved_memories': [{'memory_id': 'ghost', 'caption': 'phone'}],
              'supporting_semantic_facts': [{'triple': ['phone', 'on', 'table']}]}
    turn = save(tmp_path, result)
    assert [e['name'] for e in turn['entities']] == ['violet umbrella']
    assert turn['retrieved_memory_ids'] == ['used']
    assert turn['focus_entities'][0]['name'] == 'violet umbrella'


def test_two_primary_objects_remain_ambiguous(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds a red bag and a blue bag.',
                         'raw': {'selected_evidence': [event('red bag'), event('blue bag', 'second')]}})
    assert {e['name'] for e in turn['focus_entities']} == {'red bag', 'blue bag'}


def test_current_answer_topic_is_not_a_confirmed_presence(tmp_path):
    turn = save(tmp_path, {'answer': '无法确认是否拿着背包。', 'raw': {
        'current_context': {'state': {'current_time': 75}},
        'current_selection': {'evidence_frames': [{'path': 'cur75.jpg', 'timestamp': 75}]},
        'image_paths_used': ['cur75.jpg']}})
    assert turn['entities'] == []
    assert turn['focus_entities'][0]['source'] == 'answer_topic_hint'
    assert turn['focus_entities'][0]['last_state'] is None
    assert turn['focus_evidence_refs'][0]['start_time'] == 75


def test_empty_selected_does_not_reintroduce_ghost_entities(tmp_path):
    turn = save(tmp_path, {'answer': 'No object is visible.', 'selected_evidence': [],
                          'retrieved_memories': [{'memory_id': 'ghost', 'caption': 'phone'}]})
    assert turn['entities'] == []
    assert turn['retrieved_memory_ids'] == []
    assert turn['focus_entities'] == []


def test_focus_bounds_and_nonobject_negative_control(tmp_path):
    turn = save(tmp_path, {'answer': 'The car is moving.', 'selected_evidence': [
        event(f'person {i}', str(i)) for i in range(12)]})
    assert turn['focus_entities'] == []
    assert len(turn['focus_evidence_refs']) <= 6


def test_saved_direct_current_retains_used_frames_despite_empty_selected(tmp_path):
    result = json.loads((FIXTURES / 'cache_current.json').read_text())['result']
    turn = save(tmp_path, result)
    assert turn['focus_evidence_refs']
    assert turn['focus_evidence_refs'][0]['source_memory'] == 'M_cur'
    assert turn['focus_evidence_refs'][0]['end_time'] == 75


def test_uncertain_and_negative_focus_has_no_asserted_action_state(tmp_path):
    for answer in ('无法确认他是否拿着背包。', 'He is not holding the bag.'):
        item = event('gray backpack')
        item['metadata']['main_actions'] = [{'actor': 'front-right man', 'action': 'holds bag', 'objects': ['gray backpack']}]
        turn = save(tmp_path, {'answer': answer, 'selected_evidence': [item]})
        assert turn['focus_entities'][0]['last_state'] is None
        assert turn['focus_entities'][0]['source'] == 'answer_topic_hint'


def test_reloaded_focus_fields_are_bounded(tmp_path):
    cache = InteractionCache('focus', tmp_path, ttl_seconds=0)
    cache.data['recent_interactions'] = [{'focus_entities': [{'name': str(i)} for i in range(20)],
        'focus_evidence_refs': [{'evidence_id': str(i), 'keyframe_paths': ['x'] * 20} for i in range(20)]}]
    cache.flush()
    cache.load()
    turn = cache.recent_interactions()[-1]
    assert len(turn['focus_entities']) == 4
    assert len(turn['focus_evidence_refs']) == 6
    assert len(turn['focus_evidence_refs'][0]['keyframe_paths']) == 4


def test_focus_refs_exclude_selected_unrelated_historical_objects(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds a gray backpack. Previously he held a phone.',
        'selected_evidence': [event('phone', 'old_phone', 'M_lt'),
                              event('gray backpack', 'bag_event'),
                              {'evidence_id': 'live', 'source_memory': 'M_cur',
                               'start_time': 12, 'end_time': 12, 'keyframe_paths': ['live.jpg']} ]})
    assert [ref['evidence_id'] for ref in turn['focus_evidence_refs']] == ['bag_event', 'live']


def test_focus_refs_preserve_events_for_multiple_primary_objects(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds a red bag and a blue bag.',
        'selected_evidence': [event('red bag', 'red'), event('blue bag', 'blue'),
                              event('phone', 'old_phone', 'M_lt')]})
    assert [ref['evidence_id'] for ref in turn['focus_evidence_refs']] == ['red', 'blue']


def test_specific_primary_object_excludes_other_color_from_focus_and_refs(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds a gray backpack. Previously a red bag was on the seat.',
        'selected_evidence': [event('gray backpack', 'gray'), event('red bag', 'red')]})
    assert [e['name'] for e in turn['focus_entities']] == ['gray backpack']
    assert [r['evidence_id'] for r in turn['focus_evidence_refs']] == ['gray']


def test_translated_color_and_bag_aliases_keep_only_matching_primary(tmp_path):
    for answer in ('他拿着灰色包。此前拿着红色背包。', 'He may hold a grey bag. Previously a red backpack was visible.'):
        turn = save(tmp_path, {'answer': answer, 'selected_evidence': [
            event('gray backpack', 'gray'), event('red backpack', 'red')]})
        assert [e['name'] for e in turn['focus_entities']] == ['gray backpack']
        assert [r['evidence_id'] for r in turn['focus_evidence_refs']] == ['gray']


def test_explicit_small_backpack_beats_other_named_backpack(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds the small backpack.', 'selected_evidence': [
        event('small backpack', 'small'), event('large backpack', 'large')]})
    assert [e['name'] for e in turn['focus_entities']] == ['small backpack']
    assert [r['evidence_id'] for r in turn['focus_evidence_refs']] == ['small']


def test_explicit_backpack_does_not_select_plastic_bag(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds the backpack.', 'selected_evidence': [
        event('backpack', 'backpack'), event('plastic bag', 'plastic')]})
    assert [e['name'] for e in turn['focus_entities']] == ['backpack']
    assert [r['evidence_id'] for r in turn['focus_evidence_refs']] == ['backpack']


def test_action_generic_alias_does_not_create_second_primary(tmp_path):
    item = event('gray backpack')
    item['metadata']['main_actions'] = [{'actor': 'front-right man', 'action': 'holds', 'objects': ['backpack']}]
    turn = save(tmp_path, {'answer': 'He holds a gray backpack.', 'selected_evidence': [item]})
    assert [e['name'] for e in turn['focus_entities']] == ['gray backpack']


def test_saved_current_now_anchors_latest_sent_observation(tmp_path):
    result = json.loads((FIXTURES / 'cache_current.json').read_text())['result']
    result['answer'] = '他正在拿着手机。'
    result['timestamps'] = [{'start': 45, 'end': 75}]
    cache = InteractionCache('now', tmp_path, ttl_seconds=0)
    cache.update_from_query_result('现在他在做什么？', '现在他在做什么？', result)
    assert [(r['start'], r['end']) for r in cache.recent_interactions()[-1]['time_ranges']] == [(75, 75)]


def test_current_without_sent_images_does_not_create_now_visual_anchor(tmp_path):
    result = {'answer': '无法确认当前动作。', 'retrieval_mode': 'current', 'selected_evidence': [],
              'timestamps': [{'start': 45, 'end': 75}], 'raw': {'image_paths_used': [],
              'current_context': {'state': {'current_time': 75}},
              'current_selection': {'evidence_frames': [{'path': 'past.jpg', 'timestamp': 65}]}}}
    cache = InteractionCache('now', tmp_path, ttl_seconds=0)
    cache.update_from_query_result('现在他在做什么？', '现在他在做什么？', result)
    assert cache.recent_interactions()[-1]['time_ranges'] == []


def test_generic_named_object_in_separate_event_is_not_collapsed(tmp_path):
    turn = save(tmp_path, {'answer': 'He holds a gray backpack and another backpack.',
                          'selected_evidence': [event('gray backpack', 'gray'), event('backpack', 'other')]})
    assert {e['name'] for e in turn['focus_entities']} == {'gray backpack', 'backpack'}


def test_now_uses_latest_actually_sent_frame_not_context_clock(tmp_path):
    result = {'answer': 'He holds a backpack.', 'retrieval_mode': 'current', 'selected_evidence': [],
              'timestamps': [{'start': 45, 'end': 75}], 'raw': {'image_paths_used': ['past.jpg'],
              'current_context': {'state': {'current_time': 75}},
              'current_selection': {'evidence_frames': [{'path': 'past.jpg', 'timestamp': 65},
                                                       {'path': 'latest.jpg', 'timestamp': 75}]}}}
    cache = InteractionCache('now', tmp_path, ttl_seconds=0)
    cache.update_from_query_result('现在他在做什么？', '现在他在做什么？', result)
    assert [(r['start'], r['end']) for r in cache.recent_interactions()[-1]['time_ranges']] == [(65, 65)]


def test_equivalent_alias_names_in_same_event_do_not_add_primary(tmp_path):
    item = event('gray backpack')
    item['metadata']['entities'] = ['grey bag']
    turn = save(tmp_path, {'answer': 'He holds a gray backpack.', 'selected_evidence': [item]})
    assert [e['name'] for e in turn['focus_entities']] == ['gray backpack']


def test_explicit_same_event_multiplicity_preserves_primary_ambiguity(tmp_path):
    for answer in ('He holds a gray backpack and another backpack.', '他拿着灰色背包和另一个背包。', 'He holds two backpacks, including a gray backpack.'):
        item = event('gray backpack')
        item['metadata']['main_actions'] = [{'actor': 'front-right man', 'action': 'holds', 'objects': ['backpack']}]
        turn = save(tmp_path, {'answer': answer, 'selected_evidence': [item]})
        assert {e['name'] for e in turn['focus_entities']} == {'gray backpack', 'backpack'}
