"""Behavior regressions for references to a previously returned segment."""
import importlib.util
import os
from pathlib import Path

import pytest

from online_query import coreference_resolver as module


class PreviousPhoneContext:
    def latest_context(self):
        return {'latest_interaction': {
            'time_ranges': [{'start': 58, 'end': 67}, {'start': 115, 'end': 120}],
            'entities': [{'canonical_name': 'front-seat man'}],
            'evidence_frames': [{'timestamp': 60, 'path': 'frame_000060.jpg'},
                                {'timestamp': 116, 'path': 'frame_000116.jpg'}],
        }}


@pytest.mark.parametrize('question', [
    '在刚才那段画面里，他身上是黑色短袖还是厚外套？',
    '刚才的那段画面里他穿什么？',
    '刚刚那段画面中他戴口罩吗？',
    '上一段画面里有什么？',
    'Describe the previous segment.',
    'Describe the previous scene.',
])
def test_previous_segment_keeps_previous_time_ranges(question):
    result = module.CoreferenceResolver().resolve(question, PreviousPhoneContext())
    assert result['is_followup']
    assert result['temporal_relation'] == 'same_time'
    assert [(r['start'], r['end']) for r in result['referenced_time_ranges']] == [(58, 67), (115, 120)]


def test_after_previous_segment_keeps_after_semantics():
    result = module.CoreferenceResolver().resolve('刚才那段之后，他穿什么？', PreviousPhoneContext())
    assert result['temporal_relation'] == 'after'
    assert [(r['start'], r['end']) for r in result['referenced_time_ranges']] == [(67, 127), (120, 180)]
    assert result['referenced_evidence_frames'] == []


@pytest.mark.parametrize('question', ['现在他穿什么？', '第0秒他穿什么？'])
def test_explicit_current_or_timestamp_overrides_previous_segment(question):
    result = module.CoreferenceResolver().resolve(question, PreviousPhoneContext())
    assert not result['is_followup']
    assert not result['referenced_time_ranges']


def test_entity_pronoun_alone_does_not_force_same_time():
    result = module.CoreferenceResolver().resolve('他穿什么？', PreviousPhoneContext())
    assert result['temporal_relation'] == 'related'


def test_earliest_displayed_image_stays_an_exact_image_reference():
    result = module.CoreferenceResolver().resolve('上一答最早那张图里穿什么？', PreviousPhoneContext())
    assert result['temporal_relation'] == 'same_time'
    assert result['referenced_time_ranges'] == [{'start': 60, 'end': 60, 'source': 'explicit_previous_image'}]
