from PIL import Image
from online_evidence_content import actor_binding_fit
from online_query.evidence_packer import EvidencePacker, image_evidence_label
from online_query.memory_fusion import MemoryFusion


def test_scene_direction_is_not_a_competing_actor():
    q = 'What is the front-left person wearing?'
    scene = {'description': 'Two front occupants are visible with an open right-side doorway; laptop keyboard at center.'}
    assert actor_binding_fit(q, scene) > 0
    assert actor_binding_fit(q, {'description': 'The front-right man wears a hat.'}) == 0
    assert actor_binding_fit(q, {'actor': 'front-right man', **scene}) == 0


def test_same_pixels_share_one_slot_and_keep_provenance(tmp_path):
    for name, fmt in [('current.png', 'PNG'), ('cached.bmp', 'BMP'), ('state.png', 'PNG')]:
        image = Image.new('RGB', (80, 80), 'white')
        if name == 'state.png':
            image.putpixel((79, 79), (0, 0, 0))
        image.save(tmp_path / name, format=fmt)
    frames = [dict(path=p, source=s, evidence_id=s, timestamp=t, segment_id=s) for p,s,t in [('missing.png','missing',0), ('current.png','current',180), ('cached.bmp','history',180), ('state.png','other',181)]]
    selected, removed = EvidencePacker(tmp_path)._dedup_frames(frames, None, 3)
    assert [f['path'] for f in selected] == ['current.png', 'state.png']
    assert {r['source'] for r in selected[0]['duplicate_provenance']} == {'history'}
    assert removed == 2


def test_identical_pixels_at_distinct_capture_times_keep_both_labels(tmp_path):
    for name in ('early.png', 'later.png'):
        Image.new('RGB', (8, 8), 'white').save(tmp_path / name)
    frames = [dict(path=p, timestamp=t, segment_id=p) for p, t in [('early.png', 0), ('later.png', 180)]]
    selected, removed = EvidencePacker(tmp_path)._dedup_frames(frames, None)
    labels = [image_evidence_label(frame['path'], [], selected) for frame in selected]
    assert [label['capture_time_seconds'] for label in labels] == [0, 180]
    assert removed == 0


def test_identical_pixels_cover_each_explicit_requested_window(tmp_path):
    events = []
    for name, stamp in [('early', 0), ('later', 180)]:
        path = name + '.png'
        Image.new('RGB', (8, 8), 'white').save(tmp_path / path)
        events.append(dict(evidence_id=name, source_memory='M_st', start_time=stamp, end_time=stamp,
                           keyframe_paths=[path], visual_observations=[dict(
                               path=path, timestamp=stamp, description='The person holds a phone.')]))
    result = EvidencePacker(tmp_path).pack(
        'Compare the phone at 0 and 180 seconds.',
        dict(use_image_evidence=True, max_image_evidence=2,
             query_constraints={'time_ranges': [[0, 0], [180, 180]], 'time_explicit': True}),
        {'final_evidence': events})
    paths = result['selected_image_paths_for_mllm']
    labels = [image_evidence_label(path, events) for path in paths]
    assert {label['capture_time_seconds'] for label in labels} == {0, 180}


def test_same_capture_time_normalizes_numeric_metadata_before_content_dedup(tmp_path):
    for name in ('one.png', 'two.png'):
        Image.new('RGB', (8, 8), 'white').save(tmp_path / name)
    frames = [dict(path=p, timestamp=t, segment_id=p) for p, t in [('one.png', '180'), ('two.png', 180.0)]]
    selected, removed = EvidencePacker(tmp_path)._dedup_frames(frames, None)
    assert len(selected) == 1 and removed == 1


def test_supplemental_event_reservation_uses_canonical_pool():
    items = [dict(memory_id=f'{seg}-{suffix}', segment_id=seg, canonical_segment_id=seg, start_time=t, end_time=t+30, caption='The man holds a phone.', supplemental=True, score=score) for seg,t in [('a',150), ('b',570)] for suffix,score in [('text',.9), ('other',.8)]]
    result = MemoryFusion().fuse({'text_results': items}, {}, {'final_evidence_k': 4}, 'detail')
    final = result['final_evidence']
    assert len(final) == 2
    assert {e['start_time'] for e in final} == {150,570}


def test_exact_zero_scene_frame_keeps_selective_actor_caption_as_unverified():
    observation = {'timestamp': 0., 'path': 'frame0.jpg', 'description': 'The vehicle interior contains two front occupants, several rear occupants, and an open right-side doorway; a laptop and keyboard lie across the center.'}
    event = dict(evidence_id='opening', start_time=0, end_time=30, visual_observations=[observation])
    frame = dict(path='frame0.jpg', timestamp=0., evidence_id='opening')
    packer = EvidencePacker()
    assert packer._order_event_frames('我说的是前排左边那位，他开头手里拿着什么？', [dict(frame)], [event], 3, total_images=3, windows=[[0,0]])
    event['visual_observations'] = [{**observation, 'actor': 'front-right man'}]
    # A caption about the right actor does not establish that the left actor is
    # absent from the same image. Keep the exact requested capture eligible,
    # while retaining the conflict and giving it no positive local match.
    candidates = packer._order_event_frames('前排左边那位拿着什么？', [dict(frame)], [event], 3, total_images=3, windows=[[0,0]])
    assert len(candidates) == 1
    assert candidates[0]['_actor_conflict'] is True
    assert candidates[0]['_inspection_unknown'] is True
    assert candidates[0]['_local_relevance'] == 0


def test_identical_encoded_bytes_dedup_without_image_decoder(tmp_path):
    (tmp_path / 'one').write_bytes(b'bounded saved image payload')
    (tmp_path / 'two').write_bytes(b'bounded saved image payload')
    frames = [dict(path=p, segment_id=p, source=p) for p in ('one', 'two')]
    selected, removed = EvidencePacker(tmp_path)._dedup_frames(frames, None)
    assert len(selected) == 1 and removed == 1


def test_missing_path_never_consumes_event_quota(tmp_path):
    Image.new('RGB', (4, 4), 'red').save(tmp_path / 'good.png')
    frames = [dict(path=p, segment_id='event') for p in ('missing.png', 'good.png')]
    assert EvidencePacker(tmp_path)._dedup_frames(frames, None, 1)[0][0]['path'] == 'good.png'


def test_same_path_cross_source_keeps_duplicate_provenance(tmp_path):
    Image.new('RGB', (4, 4), 'red').save(tmp_path / 'shared.png')
    frames = [dict(path='shared.png', source=s, evidence_id=s) for s in ('M_cur', 'M_cache')]
    selected, _ = EvidencePacker(tmp_path)._dedup_frames(frames, None)
    assert selected[0]['duplicate_provenance'][0]['source'] == 'M_cache'


def test_bound_object_focus_selects_bag_without_importing_cache_time(monkeypatch):
    monkeypatch.setenv('EM2MEM_IMAGES_PER_EVIDENCE_LIMIT', '1')
    event = dict(evidence_id='event', start_time=0, end_time=30,
                 keyframe_paths=['phone.jpg', 'bag.jpg'], visual_observations=[
                     dict(timestamp=0., path='phone.jpg', description='The person holds a phone.'),
                     dict(timestamp=20., path='bag.jpg', description='The gray backpack hangs from one hand.')])
    route = dict(use_image_evidence=True, max_image_evidence=1)
    cache = {'bound_focus_entities': [{'name': 'gray backpack'}],
             'referenced_time_ranges': [{'start': 0, 'end': 0}]}
    def packed(context, decision=route):
        return EvidencePacker().pack('那个东西怎么拿？', decision, {'final_evidence': [event]}, context)['selected_image_paths_for_mllm']
    assert packed(cache) == ['bag.jpg']
    assert packed({'focus_entities': [{'name': 'gray backpack'}]}) == packed({})
    assert packed({'referenced_entities': [{'name': 'gray backpack'}]}) == packed({})
    assert packed({'bound_focus_entities': [{'name': 'gray backpack'}, {'name': 'phone'}]}) == packed({})
    assert packed(cache, {**route, 'query_constraints': {'time_ranges': [[0, 0]]}}) == ['phone.jpg']


def test_bound_focus_still_guides_cached_same_time_window(monkeypatch):
    monkeypatch.setenv('EM2MEM_IMAGES_PER_EVIDENCE_LIMIT', '1')
    event = dict(evidence_id='event', start_time=90, end_time=110,
                 keyframe_paths=['phone.jpg', 'bag.jpg'], visual_observations=[
                     dict(timestamp=96., path='phone.jpg', description='The person holds a phone.'),
                     dict(timestamp=108., path='bag.jpg', description='The gray backpack hangs from one hand.')])
    cache = {'bound_focus_entities': [{'name': 'gray backpack'}], 'temporal_relation': 'same_time',
             'confidence': 1., 'referenced_time_ranges': [{'start': 96, 'end': 108}]}
    result = EvidencePacker().pack('那个东西怎么拿？', dict(use_image_evidence=True, max_image_evidence=1), {'final_evidence': [event]}, cache)
    assert result['selected_image_paths_for_mllm'] == ['bag.jpg']
