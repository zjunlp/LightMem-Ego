"""Scope regressions: temporal words must not erase either comparison side."""
import pytest
from online_query_constraints import parse_query_constraints
from online_query.memory_router import MemoryRouter
from online_query.query_router import QueryRouter
from online_query_constraints import constraint_windows


@pytest.mark.parametrize('question,current,history', [
    ('再回到现在，桌上有什么？', True, False),
    ('Switch back to now: what is on the desk?', True, False),
    ('截至现在，画面里有没有出现过杯子？', False, True),
    ('Up until now, has anyone entered?', False, True),
    ('现在和刚才相比有什么变化？', True, True),
    ('How does the scene now differ from earlier?', True, True),
    ('Compare the current view with the beginning of the video', True, True),
    ('What happened earlier?', False, True),
    ('不要看现在，只看第10秒的杯子', False, True),
])
def test_temporal_requirements(question, current, history):
    c = parse_query_constraints(question, 120)
    assert c.get('requires_current') is current
    assert c.get('requires_history') is history
    if current and not history:
        assert c['scope'] == 'current'


@pytest.mark.parametrize('question,flag', [
    ('帮我捋一遍这段视频发生的事情', 'summary'),
    ('Walk me through the video from start to finish', 'summary'),
    ('Give me a rundown of the events', 'summary'),
    ('画面现在跟刚才有什么不一样？', 'comparison'),
    ('How has the scene changed since earlier?', 'comparison'),
    ('有没有既拿手机又喝水的时候？', 'simultaneous'),
    ('Was anyone holding a cup while opening the door?', 'simultaneous'),
])
def test_intent_paraphrases(question, flag):
    c = parse_query_constraints(question, 120)
    assert (c.get('summary') if flag == 'summary' else c['requirements'][flag]) is True


STATE = dict(current_ready=True, current_stale=False, short_term_ready=True,
             long_term_ready=True, visual_ready=True, mcur_time_span=[110,120],
             mst_time_span=[60,120], mlt_time_span=[0,120])


@pytest.mark.parametrize('question,current,history', [
    ('再回到现在，桌上有什么？', True, False),
    ('截至现在，画面里有没有出现过杯子？', False, True),
    ('How does the scene now differ from earlier?', True, True),
    ('Compare the current view with the beginning of the video', True, True),
])
def test_both_routers_keep_required_sources(question, current, history):
    routes = [MemoryRouter().route(question, {}, STATE),
              QueryRouter().route(question, session_context=STATE)]
    for result in routes:
        assert result['memory_route']['use_current'] is current
        assert result['memory_route']['use_long_term'] is history
        if current and history:
            assert result['query_type'] != 'current_perception'


def test_router_explicit_source_and_image_opt_outs_remain_binding():
    opts = dict(use_current=False, use_long_term=False, use_interaction_cache=False,
                use_image_evidence=False, max_image_evidence=0)
    q = 'How does the scene now differ from earlier?'
    for result in [MemoryRouter().route(q, opts, STATE), QueryRouter().route(q, opts, STATE)]:
        assert result['memory_route']['use_current'] is False
        assert result['memory_route']['use_long_term'] is False
        assert result['memory_route']['use_interaction_cache'] is False
    result = QueryRouter().route(q, opts, STATE)
    assert result['use_image_evidence'] is False
    assert result['max_image_evidence'] == 0


def test_then_versus_now_with_question_clause_keeps_history():
    q = '之前有没有口罩和墨镜都戴着的时候？那时跟现在有什么不同？'
    c = parse_query_constraints(q, 120)
    assert c['requires_current'] is True
    assert c['requires_history'] is True
    assert c['requirements']['comparison'] is True


def test_joint_state_when_both_conditions_hold_is_simultaneous():
    c = parse_query_constraints('之前有没有口罩和墨镜都戴着的时候？', 120)
    assert c['requirements']['simultaneous'] is True


@pytest.mark.parametrize('q', [
    'Compare what he wore at that time with what he wears now',
    '把他当时的衣服跟现在的比较一下',
])
def test_comparison_with_referenced_past_time_requires_both(q):
    c = parse_query_constraints(q, 120)
    assert c['requires_current'] is True
    assert c['requires_history'] is True
    assert c['scope'] == 'history'


@pytest.mark.parametrize('question,expected', [
    ('他开头拿的和现在拿的，能确定是同一个东西吗？', [[0.0,0.0],[120,120]]),
    ('Compare what he held at 10 seconds with what he holds now', [[10.0,10.0],[120,120]]),
    ('第10～20秒和现在相比，有什么不同？', [[10.0,20.0],[120,120]]),
])
def test_explicit_past_window_with_now_keeps_both_temporal_sides(question, expected):
    c = parse_query_constraints(question, 120)
    assert c['requires_current'] and c['requires_history']
    assert constraint_windows(c) == expected


def test_mixed_point_without_known_now_does_not_exclude_current_evidence():
    c = parse_query_constraints('Compare at 10 seconds with now')
    assert c['requires_current'] and c['requires_history']
    assert not c['time_explicit']
    assert not constraint_windows(c)
    assert c['time_parse_incomplete']


def test_appear_at_is_not_the_chinese_now_keyword():
    c = parse_query_constraints('截至第450秒，拿手机是否出现在穿外套之前和之后', 688)
    assert not c['requires_current']
    assert constraint_windows(c) == [[0,450]]


@pytest.mark.parametrize('q', [
    '帮我把刚才到现在前面车流的变化捋一下。',
    '请整理一下这段视频里事件发生的过程',
    '把这一路的变化过程整理一下',
])
def test_colloquial_event_summary_requests_overview(q):
    assert parse_query_constraints(q, 120)['summary'] is True


@pytest.mark.parametrize('q', [
    '他刚才有没有整理衣服？',
    '现在是谁在整理桌子？',
    '她整理衣服的过程持续了多久？',
])
def test_tidying_action_is_not_a_summary_request(q):
    assert parse_query_constraints(q, 120)['summary'] is False


@pytest.mark.parametrize('q', ['之前看到什么？', '刚才看到什么？', '当时能看到什么？',
                              'What could you see earlier?', 'What can you see at that time?'])
def test_historical_perception_words_do_not_request_current(q):
    c = parse_query_constraints(q, 120)
    assert c['requires_current'] is False
    assert c['requires_history'] is True
    assert c['scope'] != 'current'


@pytest.mark.parametrize('q', [
    '之前拿过手机吗，现在还拿着吗？',
    'What did he hold earlier, and what is he holding now?',
    '刚才拿着什么？现在呢？',
])
def test_independent_past_and_present_clauses_require_both(q):
    c = parse_query_constraints(q, 120)
    assert c['requires_current'] is True
    assert c['requires_history'] is True
    assert c['scope'] == 'history'


@pytest.mark.parametrize('q,current,history', [
    ('不要看之前，现在拿着什么', True, False),
    ('不要看现在，之前拿着什么', False, True),
])
def test_negated_time_clause_does_not_create_mixed_requirement(q, current, history):
    c = parse_query_constraints(q, 120)
    assert c['requires_current'] is current
    assert c['requires_history'] is history
