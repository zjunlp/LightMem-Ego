"""Behavior regressions exposed by real Dongfeng prefix demonstrations."""
import importlib.util
import os
from pathlib import Path

import pytest

import online_query_constraints as module


@pytest.mark.parametrize('question', [
    '概括前排两个人的主要活动，分别给出开头、中间和后段的画面依据。',
    '从开头到结束，这个人主要在做什么？给出相隔至少30秒的两处画面。',
    '按时间概括他的穿戴变化，包含开头和最后的状态，并给关键画面。',
    'Summarize the beginning, middle, and final state of the video.',
])
def test_summary_stages_are_not_single_instants(question):
    result = module.parse_query_constraints(question, 707)
    assert result['summary']
    assert result['scope'] == 'history'
    assert result['time_range'] is None


@pytest.mark.parametrize('question,expected', [
    ('只概括开头10秒发生的事。', [0,10]),
    ('概括前两分钟的穿戴变化。', [0,120]),
    ('请概括第30～60秒的开头、中间、结尾状态。', [30,60]),
    ('视频第0秒，他戴墨镜吗？', [0,0]),
    ('视频开头，他手里有什么？', [0,0]),
    ('第40秒，他在做什么？', [40,40]),
    ('视频最后画面，他穿什么？', [707,707]),
])
def test_explicit_local_time_constraints_remain_binding(question,expected):
    assert module.parse_query_constraints(question,707)['time_range'] == expected


@pytest.mark.parametrize('question', [
    '给出相隔至少30秒的两张图。',
    '找两处间隔不少于30秒的手机画面。',
    'Find two images at least 30 seconds apart.',
    '描述持续30秒的动作。',
])
def test_durations_and_spacing_are_not_absolute_timestamps(question):
    assert module.parse_query_constraints(question,707)['time_range'] is None


def test_explicit_point_after_video_duration_is_preserved():
    result = module.parse_query_constraints('请看30秒的视频第10秒，他在做什么？',707)
    assert result['time_range'] == [10,10]
