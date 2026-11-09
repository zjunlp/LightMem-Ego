"""Evidence membership, image availability and answer provenance contracts."""
from __future__ import annotations

import copy

import pytest

from online_query.evidence_packer import EvidencePacker
from online_query.memory_fusion import MemoryFusion
from online_query.memory_plan import RetrievalPlanner


def touch(root, *paths):
    for path in paths:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()


def test_only_final_fused_events_can_supply_images(tmp_path):
    touch(tmp_path, 'selected.jpg', 'excluded.jpg', 'cache.jpg')
    memories = {
        'text_results': [{'memory_id': 'lt', 'segment_id': 'lt', 'score': .6,
                          'start_time': 100., 'end_time': 130., 'caption': 'selected event',
                          'keyframe_paths': ['selected.jpg']}],
        'short_term_results': [{'event_id': 'st', 'score': .99, 'start_time': 0., 'end_time': 10.,
                                'event_caption_refined': 'excluded event',
                                'keyframes': [{'path': 'excluded.jpg', 'timestamp': 5.}]}],
    }
    decision = {'memory_priority': {'M_lt': 1., 'M_st': 0.}}
    fused = MemoryFusion().fuse(memories, decision, {
        'final_evidence_k': 1, 'retrieval_plan': {'M_cache': {'enabled': False}},
    }, 'general_qa')
    assert [ev['evidence_id'] for ev in fused['final_evidence']] == ['lt']
    packed = EvidencePacker(tmp_path).pack('where?', {
        'use_image_evidence': True, 'max_image_evidence': 1, 'final_evidence_k': 1,
    }, {**memories, **fused}, {'is_followup': True, 'referenced_evidence_frames': [{'path': 'cache.jpg'}]})
    assert packed['selected_image_paths_for_mllm'] == ['selected.jpg']
    assert {f['path'] for f in packed['selected_evidence_frames']} == {'selected.jpg'}
    assert packed['evidence_pack_summary']['final_evidence_count'] == len(packed['selected_evidence']) == 1


def test_empty_final_evidence_does_not_resurrect_raw_candidates(tmp_path):
    touch(tmp_path, 'excluded.jpg')
    packed = EvidencePacker(tmp_path).pack('where?', {
        'use_image_evidence': True, 'max_image_evidence': 1,
    }, {'final_evidence': [], 'evidence_frames': [{'path': 'excluded.jpg', 'timestamp': 5.}]})
    assert packed['selected_image_paths_for_mllm'] == []


def test_missing_frames_do_not_consume_segment_or_time_quota(tmp_path):
    touch(tmp_path, 'valid.jpg')
    frames = [{'path': name, 'timestamp': t, 'segment_id': 'event'} for name, t in [
        ('missing1.jpg', 0.), ('missing2.jpg', 2.), ('missing3.jpg', 4.), ('valid.jpg', 4.1),
    ]]
    packed = EvidencePacker(tmp_path).pack('show', {
        'use_image_evidence': True, 'max_image_evidence': 3,
    }, {'evidence_frames': frames})
    assert packed['selected_image_paths_for_mllm'] == ['valid.jpg']


def test_missing_current_preference_falls_back_within_cap(tmp_path):
    touch(tmp_path, 'current/valid1.jpg', 'current/valid2.jpg', 'current/valid3.jpg', 'current/valid4.jpg')
    frames = [{'path': f'current/{name}.jpg', 'timestamp': i * 2., 'source': 'M_cur'}
              for i, name in enumerate(['missing', 'valid1', 'valid2', 'valid3', 'valid4'])]
    packed = EvidencePacker(tmp_path).pack('now', {
        'use_image_evidence': True, 'max_image_evidence': 9,
    }, {'current_selection': {
        'evidence_frames': frames, 'selected_image_paths_for_mllm': ['current/missing.jpg'],
    }})
    assert packed['selected_image_paths_for_mllm'] == ['current/valid1.jpg', 'current/valid2.jpg', 'current/valid3.jpg']


@pytest.mark.parametrize('kind', ['referenced_evidence_frames', 'referenced_time_ranges', 'referenced_memory_ids'])
def test_zero_cache_budget_is_respected(monkeypatch, kind):
    monkeypatch.setenv('EM2MEM_UNIFIED_MCACHE_TOP_K', '0')
    values = {'referenced_evidence_frames': [{'path': 'cache.jpg'}],
              'referenced_time_ranges': [{'start': 0, 'end': 10}], 'referenced_memory_ids': ['memory']}
    cache = {'is_followup': True, kind: values[kind]}
    decision = {'query_type': 'general_qa', 'memory_route': {'use_interaction_cache': True}}
    plan = RetrievalPlanner().plan(decision, {}, {}, cache)
    assert plan['retrieval_plan']['M_cache']['candidate_budget'] == 0
    fused = MemoryFusion().fuse({}, decision, plan, 'general_qa', cache)
    assert fused['fusion_summary']['input_counts']['M_cache'] == 0
    assert fused['final_evidence'] == []


@pytest.mark.parametrize('short_term_enabled', [True, False])
def test_post_answer_supplement_is_never_generation_evidence(tmp_path, monkeypatch, short_term_enabled):
    from online_query import query_engine as engine
    event = {'event_id': 'unseen', 'start_time': 100., 'end_time': 110.,
             'event_caption_refined': 'Never sent to the model', 'keyframes': []}
    class Store:
        def __init__(self, path):
            pass
        def load_events(self):
            return [event]
    monkeypatch.setattr(engine, 'MSTStore', Store)
    result = {'answer': 'Generated using old evidence', 'query_type': 'long_term_summary',
              'selected_evidence': [{'evidence_id': 'old', 'source_memory': 'M_lt', 'start_time': 0., 'end_time': 30.}],
              'memory_route': {'use_short_term': short_term_enabled},
              'fusion_summary': {'final_evidence_count': 1, 'selected_memory_sources': ['M_lt']}}
    before = copy.deepcopy(result)
    changed = engine._append_recent_stream_supplement(result, session_dir=tmp_path, question='总结整个视频')
    assert result['selected_evidence'] == before['selected_evidence']
    assert result['fusion_summary'] == before['fusion_summary']
    assert result['answer'] == before['answer']
    if short_term_enabled:
        assert changed and result['stream_supplement'][0]['evidence_id'] == 'unseen'
        assert result['stream_supplement_used_for_generation'] is False
    else:
        assert not changed and not result.get('stream_supplement')


def test_chinese_lexical_scores_work_for_all_four_sources():
    from retrieval_source_helpers import EPISODIC, UNIFIED, caption
    memory = EPISODIC.EpisodicMemoryRAG(None, None, None)
    doc = caption('cn', '00000000', '00003000', '男子拿起红色手机。', visual_object_threads=['红色手机'])
    memory._process_caption_data([doc], '30sec')
    memory._process_triplet_data({'triplet_map': {'cn': [['男子', '拿起', '红色手机']]}}, '30sec')
    memory._process_graph_data({'doc_id_to_event_id': {'cn': 'event_cn'},
        'nodes': [{'id': 'event_cn', 'type': 'Event', 'text': doc['text']}], 'edges': []}, '30sec')
    entry = memory.captions['30sec'][0]
    tokens = EPISODIC._tokenize('男子拿起了什么手机？')
    assert tokens
    for candidate in [memory._entry_retrieval_tokens(entry), memory._entry_metadata_tokens(entry),
                      memory.graph_sidecar['30sec']['cn'].graph_tokens,
                      memory._entry_raw_triplet_tokens('cn', '30sec')]:
        assert memory._overlap_score(tokens, candidate) > 0
    assert UNIFIED.EM2Memory._tokenize(None, '男子拿手机')
    assert EPISODIC._tokenize('the RED phone') == {'red', 'phone'}
