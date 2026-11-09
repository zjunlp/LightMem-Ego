"""Independent fine recall and preserved event context, without model calls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from online_evidence_content import prompt_evidence
from online_query.memory_fusion import MemoryFusion
from online_query.memory_plan import RetrievalPlanner
from online_query.query_engine import LoadedQueryEngine
from online_query_constraints import parse_query_constraints
from online_short_term.mst_retriever import MSTRetriever
from test_event_query_integration_20261025 import fallback_session, _fallback


@pytest.mark.parametrize('requested', [None, 4])
def test_default_expands_independent_details_but_explicit_limit_wins(monkeypatch, requested):
    for key in ('EM2MEM_UNIFIED_FINAL_EVIDENCE_K', 'EM2MEM_UNIFIED_MST_TOP_K'):
        monkeypatch.delenv(key, raising=False)
    loaded = object.__new__(LoadedQueryEngine)
    loaded.session_id = 'test'
    route, plan, _, _ = loaded._route_and_plan(
        '之前他拿着什么？', {'final_evidence_k': requested},
        {'long_term_ready': True, 'short_term_ready': True, 'mst_time_span': [0, 120],
         'mlt_time_span': [0, 120]}, {})
    expected = requested or min(32, sum(layer['candidate_budget']
        for layer in plan['retrieval_plan'].values() if layer['enabled']))
    assert route['final_evidence_k'] == expected
    assert route['short_term_policy']['top_k'] == 16
    assert plan['retrieval_plan']['M_st']['top_k'] == 16


def test_prepublication_short_term_path_uses_same_default_budget(fallback_session, monkeypatch):
    monkeypatch.delenv('EM2MEM_RECENT_FINAL_EVIDENCE_K', raising=False)
    monkeypatch.delenv('EM2MEM_UNIFIED_MST_TOP_K', raising=False)
    result = _fallback(fallback_session, final_evidence_k=None)
    route = result['route_decision']
    assert route['final_evidence_k'] >= route['short_term_policy']['top_k']
    assert route['short_term_policy']['top_k'] == 16


def test_short_term_details_keep_their_own_source_and_parent_caption():
    loaded = object.__new__(LoadedQueryEngine)
    loaded.evidence_by_doc_id = {'long': {'source_micro_event_ids': ['fine0', 'fine1']}}
    parent = {'doc_id': 'long', 'text': 'A person prepares a parcel and closes it.',
              'start_time': 0, 'end_time': 30, 'score': .8}
    short = [{'event_id': f'fine{i}', 'start_time': i*10, 'end_time': i*10+10,
              'event_caption_refined': caption, 'status': 'refined', 'score': .5,
              'visual_observations': [{'timestamp': i*10+5, 'description': caption}]}
             for i, caption in enumerate(['The person inserts a mug.', 'The person tapes the parcel.'])]
    original = deepcopy((parent, short))
    result = MemoryFusion().fuse(
        {'text_results': loaded._build_text_results([parent], []), 'short_term_results': short},
        {}, {'final_evidence_k': 3}, 'temporal_reasoning')
    packed, _ = prompt_evidence('What did the person do after preparing the parcel?', result['final_evidence'])
    long = next(row for row in packed if row['source'] == 'M_lt')
    assert long['caption'] == parent['text'] and not long.get('visual_observations')
    fine = [row for row in packed if row['source'] == 'M_st']
    assert {row['evidence_id'] for row in fine} == {'fine0', 'fine1'}
    assert all(row['caption'] and row['visual_observations'] for row in fine)
    assert (parent, short) == original


@pytest.mark.parametrize('query_type', ['temporal_reasoning', 'entity_tracking'])
def test_detail_budget_retains_multiple_st_events_even_when_lt_scores_are_higher(query_type):
    long = [{'memory_id': f'long{i}', 'start_time': i*30, 'end_time': i*30+30,
             'caption': f'Long event {i}', 'score': .95} for i in range(8)]
    short = [{'event_id': f'fine{i}', 'start_time': i*10, 'end_time': i*10+10,
              'event_caption_refined': f'Fine event {i}', 'score': .1} for i in range(16)]
    result = MemoryFusion().fuse({'text_results': long, 'short_term_results': short},
                                {}, {'final_evidence_k': 12}, query_type)
    selected = result['final_evidence']
    assert len(selected) == 12
    assert sum(row['source_memory'] == 'M_lt' for row in selected) == 8
    assert sum(row['source_memory'] == 'M_st' for row in selected) == 4


def test_summary_policy_still_prefers_history_coverage_and_recent_tail():
    plan = RetrievalPlanner().plan({'query_type': 'long_term_summary'}, {}, {})
    assert plan['final_evidence_k'] == 10
    long = [{'memory_id': f'long{i}', 'start_time': i*30, 'end_time': i*30+30,
             'caption': f'Stage {i}', 'score': .3} for i in range(8)]
    short = [{'event_id': f'fine{i}', 'start_time': 240+i*10, 'end_time': 250+i*10,
              'event_caption_refined': f'Last detail {i}', 'score': .9} for i in range(4)]
    result = MemoryFusion().fuse({'text_results': long, 'short_term_results': short},
                                {}, {'final_evidence_k': 6}, 'long_term_summary')
    selected = result['final_evidence']
    assert sum(row['source_memory'] == 'M_lt' for row in selected) == 5
    assert 'fine3' in {row['evidence_id'] for row in selected}


@pytest.mark.parametrize('budget', [150, 1400])
def test_whole_synopses_survive_image_observation_priority_and_small_budget(budget):
    caption = 'The person opens the cabinet. They take out a mug, then close the door.'
    visual_summary = 'An open wooden cabinet and a mug on the counter.'
    events = [{'evidence_id': 'long', 'source_memory': 'M_lt', 'caption': caption,
               'visual_summary': visual_summary},
              {'evidence_id': 'fine', 'source_memory': 'M_st',
               'caption': 'They hold the mug with both hands and place it on the counter.',
               'visual_observations': [{'timestamp': i, 'path': f'{i}.jpg',
                   'description': 'The mug is visible in both hands. ' * 8} for i in range(20)]}]
    snapshot = deepcopy(events)
    packed, stats = prompt_evidence('How was the mug held?', events, max_tokens=budget,
                                     image_paths=[f'{i}.jpg' for i in range(20)])
    assert packed[0]['caption'] == caption
    assert packed[0]['visual_summary'] == visual_summary
    assert packed[1]['caption'] == events[1]['caption']
    assert stats['retained_event_captions'] == 2 and stats['omitted_units'] > 0
    assert stats['budget_exceeded'] == (stats['estimated_tokens'] > budget)
    assert events == snapshot


def test_event_relative_after_does_not_favor_recent_unrelated_events():
    events = [{'event_id': f'event{i}', 'start_time': t, 'end_time': t+10,
               'event_caption_refined': 'A person opens a cabinet.', 'status': 'refined',
               'diff_score': .2} for i, t in enumerate([10, 90])]
    store = SimpleNamespace(get_state=lambda: {'mst_version': 1}, load_events=lambda: events)
    retriever = MSTRetriever(store)
    history = retriever.search('After opening the cabinet, what did the person do?', top_k=2)
    assert history[0]['score'] == history[1]['score']
    assert all(row['recency_score_component'] == 0 for row in history)
    recent = retriever.search('他最近拿过杯子吗', top_k=2)
    assert recent[0]['start_time'] == 90
    assert recent[0]['recency_score_component'] > recent[1]['recency_score_component']


def test_more_candidates_still_obey_requested_time_window():
    events = [{'event_id': f'event{i}', 'start_time': i*10, 'end_time': i*10+9,
               'event_caption_refined': 'The person holds a mug.', 'status': 'refined'} for i in range(20)]
    store = SimpleNamespace(get_state=lambda: {'mst_version': 1}, load_events=lambda: events)
    question = '第10～29秒拿着什么？'
    scope = parse_query_constraints(question, 190)
    selected = MSTRetriever(store).search(question, top_k=16, cache_context={'query_constraints': scope})
    assert {row['event_id'] for row in selected} == {'event1', 'event2'}
