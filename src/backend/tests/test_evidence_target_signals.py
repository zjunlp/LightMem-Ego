from online_evidence_content import (
    actor_binding_fit, focus_match, observation_match, query_focus,
    supplemental_observation_matches, target_signals, visual_anchors,
)


def test_actor_descriptor_is_not_the_search_target():
    question = '找画面左侧前排戴眼镜的人拿着手机的画面'
    assert query_focus(question) == {'phone'}
    assert focus_match('他有没有戴眼镜？', 'The man wears glasses.') == 1
    assert observation_match(question, {'description': 'The front-left man wears glasses and points at a laptop.'}) < observation_match(question, {'description': 'The front-left man holds a dark rectangular device.'})


def test_uncertain_device_is_candidate_without_rewriting():
    question = '找他拿手机的画面'
    for text in ('The man holds a dark rectangular device.', 'The man holds a device, possibly a phone.'):
        observation = {'timestamp': 5, 'description': text}
        assert target_signals(question, observation)['phone'] == 'candidate'
        assert 0 < observation_match(question, observation) < observation_match(question, {'description': 'The man holds a phone.'})
        assert observation['description'] == text


def test_negation_is_local_and_positive_same_frame_wins():
    question = '找他同时戴口罩并拿手机的画面'
    positive = {'timestamp': 353, 'description': 'The man wearing a white mask holds a black phone.'}
    negative = {'timestamp': 350, 'description': 'The man wears a white mask, no phone is visible.'}
    assert target_signals(question, negative)['phone'] == 'negative'
    assert target_signals(question, {'description': 'The man holds a phone without glasses.'})['phone'] == 'confirmed'
    assert observation_match(question, positive) > observation_match(question, negative)
    event = {'doc_id': 'event', 'visual_observations': [negative, positive]}
    assert visual_anchors(question, event)[0]['start'] == 353
    assert supplemental_observation_matches(question, [event], {'requirements': {'simultaneous': True}})[0][0] == 'event'


def test_same_event_structured_alias_resolves_actor_not_computer():
    question = '找画面左侧前排戴眼镜的人拿着手机的画面'
    device = {'timestamp': 81, 'description': 'The bald man holds a dark rectangular device.'}
    computer = {'timestamp': 82, 'description': 'The front-left man points to the laptop.'}
    event = {'visual_objects': [{'name': 'bald man', 'attributes': ['front-left position', 'glasses']}], 'visual_observations': [computer, device]}
    assert actor_binding_fit(question, device, event) == 1
    assert visual_anchors(question, event)[0]['start'] == 81
    conflict = {'actor': 'front-right man', 'description': 'The bald man holds a phone.'}
    assert actor_binding_fit(question, conflict, event) == 0


def test_different_people_remain_lower_ranked_inspection_candidates():
    question = '找他同时戴口罩并拿手机'
    mixed = {'timestamp': 4, 'description': 'One man wears a mask while another man holds a phone.'}
    assert observation_match(question, mixed) < observation_match(question, {'description': 'The man wears a mask and holds a phone.'})
    hits = supplemental_observation_matches(question, [{'doc_id': 'mixed', 'visual_observations': [mixed]}], {'requirements': {'simultaneous': True}})
    assert hits and hits[0][0] == 'mixed'


def test_two_explicit_subjects_joined_by_and_are_not_same_person():
    q = '找同时戴口罩并拿手机的画面'
    mixed = {'description': 'The front-left man wears a mask and the front-right man holds a phone.'}
    assert set(target_signals(q, mixed).values()) == {'candidate'}


def test_candidate_coverage_does_not_claim_confirmed_phone():
    from online_evidence_coverage import coverage_features
    assert coverage_features('他拿手机', {'visual_observations': [{'description': 'He holds a rectangular device, possibly a phone.'}]}) == {'candidate:phone'}


def test_comparison_recovers_distinct_timed_periods_and_union_windows():
    from online_evidence_content import comparison_observation_matches
    events = [{'doc_id': str(t), 'visual_observations': [{'timestamp': t, 'description': 'The man holds a phone.'}]} for t in (10, 50, 100)]
    c = {'requirements': {'comparison': True}, 'time_ranges': [[0, 20], [90, 110]]}
    assert [doc for doc, _ in comparison_observation_matches('比较之前和之后拿手机的画面', events, c)] == ['10', '100']


def test_masked_face_and_structured_action_objects_are_supported():
    assert target_signals('找同时戴口罩并拿手机的图', {'description': 'The person holds a smartphone in front of their masked face.'}) == {'mask': 'confirmed', 'phone': 'confirmed'}
    assert target_signals('找他拿手机', {'actor': 'man', 'action': 'holds', 'objects': ['phone']})['phone'] == 'confirmed'


def test_coverage_negation_does_not_remove_another_worn_object():
    from online_evidence_coverage import coverage_features
    features = coverage_features('概括穿戴变化', {'visual_observations': [{'description': 'The man wears a coat without a mask.'}]})
    assert 'state:coat:removed' not in features
    assert 'state:coat:worn' in features
    assert 'state:mask:removed' in features


def test_separate_actor_sentences_cannot_supply_conjunction():
    q = '找同时戴口罩并拿手机的画面'
    for text in ('One man wears a mask. Another man holds a phone.', 'The front-left man wears a mask. The front-right man holds a phone.'):
        assert set(target_signals(q, {'description': text}).values()) == {'candidate'}


def test_negation_before_another_target_does_not_cross_clause_boundary():
    assert target_signals('找拿手机的画面', {'description': 'The man has no glasses, holding a phone.'})['phone'] == 'confirmed'
    signals = target_signals('找口罩和手机', {'description': 'The man wears a mask, phone is not visible.'})
    assert signals == {'mask': 'confirmed', 'phone': 'negative'}


def test_unpositioned_people_and_cross_sentence_pronouns_are_not_same_frame_proof():
    q = '同时戴口罩拿手机'
    for text in ('The man wears a mask. The woman holds a phone.', 'The person wears a mask. He holds a phone.'):
        assert set(target_signals(q, text).values()) == {'candidate'}
        from online_evidence_content import supplemental_requirement_support
        support = supplemental_requirement_support(q, {'doc_id': 'split', 'visual_observations': [{'timestamp': 1, 'description': text}]}, {'requirements': {'simultaneous': True}})
        assert support['confirmed'] is False and support['support_level'] == 'text_candidate'
    assert set(target_signals(q, 'The person wears a mask while holding a phone.').values()) == {'confirmed'}


def test_common_phone_negation_and_unclear_visibility_are_not_positive():
    for text in ('The man does not hold a phone.', 'The man is not holding a phone.', 'The phone is not visible.', 'No phone is visible.'):
        assert target_signals('找拿手机的画面', text)['phone'] == 'negative'
    for text in ('The phone is not clearly visible.', 'The phone is not fully visible.'):
        assert target_signals('找拿手机的画面', text)['phone'] == 'candidate'


def test_contracted_and_adverbial_negation_never_confirms_a_phone():
    q = '同时戴口罩拿手机'
    for text in ("The man wears a mask but doesn't hold a phone.",
                 "The man wears a mask but didn’t hold a phone.",
                 'The man wears a mask and does not hold a phone.',
                 'The man wears a mask and is not currently holding a phone.',
                 '他戴着口罩，但是没有拿着手机。'):
        assert target_signals(q, text)['phone'] == 'negative'
        assert observation_match(q, {'description': text}) < .1


def test_packer_preserves_actor_clause_boundaries_in_simultaneous_signals():
    from online_query.evidence_packer import EvidencePacker
    q = '找他同时戴口罩并拿手机的画面'
    def event(identity, time, text):
        path = f'{identity}.jpg'
        return {'evidence_id': identity, 'start_time': time, 'end_time': time + 1,
                'visual_observations': [{'timestamp': time, 'path': path, 'description': text}],
                'keyframe_paths': [path]}
    mixed = event('mixed', 10, 'The man wears a mask while the woman holds a phone.')
    same = event('same', 20, 'The person wears a mask while holding a phone.')
    packed = EvidencePacker().pack(q, {'use_image_evidence': True, 'max_image_evidence': 2}, {'final_evidence': [mixed, same]})
    assert packed['selected_image_paths_for_mllm'] == ['same.jpg', 'mixed.jpg']
    observation = mixed['visual_observations'][0]
    assert observation['description'] == 'The man wears a mask while the woman holds a phone.'
