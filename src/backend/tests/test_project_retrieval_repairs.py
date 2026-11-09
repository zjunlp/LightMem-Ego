"""General retrieval fixes: unchanged source definitions, no model or network calls."""
import json
import pickle
import threading
import types

import numpy as np
import pytest

from retrieval_source_helpers import (
    EPISODIC, SEMANTIC, UNIFIED, VISUAL, EARLY, LATE, EARLY_CUTOFF,
    LATE_CUTOFF, FakeEmbedding, NoModels, make_memory,
)


@pytest.fixture
def semantic_runtime(monkeypatch):
    class Graph:
        builds = 0
        def __init__(self):
            type(self).builds += 1
            self.es = {}
        def add_vertices(self, vertices):
            self.vertices = vertices
        def add_edges(self, edges):
            self.edges = edges
    tensor_calls = []
    def tensor(data, **kwargs):
        tensor_calls.append(np.asarray(data))
        return np.asarray(data)
    monkeypatch.setattr(SEMANTIC, 'ig', types.SimpleNamespace(Graph=Graph), raising=False)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        tensor=tensor, float32=np.float32)
    monkeypatch.setattr(SEMANTIC, 'torch', torch)
    monkeypatch.setattr(VISUAL, 'torch', torch)
    return Graph, tensor_calls


def semantic(tmp_path=None, model=None, mode='flat_facts'):
    value = SEMANTIC.SemanticMemory(model or FakeEmbedding())
    value.embedding_cache_dir = tmp_path
    value.load_triples_from_data({'100': [['person', 'holds', 'cup']]})
    value.index_mode = mode
    return value


@pytest.mark.parametrize('query', ['总结整个视频', '30到60秒发生了什么'])
def test_answer_preserves_explicit_empty_constraints(tmp_path, query):
    memory = make_memory(tmp_path)
    memory.answer(query, until_time=LATE_CUTOFF, query_constraints={},
                  use_image_evidence=False, generate_answer=False)
    assert memory.episodic_memory._thread_local.query_constraints == {}


@pytest.mark.parametrize(('query', 'summary', 'expected_calls'), [
    ('总结整个视频', False, 0), ('alpha task', True, 1),
])
def test_all_summary_branches_follow_authoritative_scope(tmp_path, monkeypatch, query, summary, expected_calls):
    import online_evidence_coverage
    calls = []
    def coverage(query, items, limit):
        calls.append(True)
        return items[:limit]
    monkeypatch.setattr(online_evidence_coverage, 'select_coverage', coverage)
    memory = make_memory(tmp_path)
    memory.answer(query, until_time=LATE_CUTOFF, query_constraints={'summary': summary},
                  use_image_evidence=False, generate_answer=False)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(('field', 'before', 'after'), [
    ('dimension', 2, 3), ('revision', 'r1', 'r2'),
    ('text_backend', 'local', 'remote'), ('remote_url', 'http://one', 'http://two'),
])
def test_semantic_identity_invalidates_both_vector_caches(tmp_path, field, before, after):
    first_model = FakeEmbedding('same-name')
    setattr(first_model, field, before)
    first = semantic(tmp_path, first_model)
    entries = first.timestamp_to_triples[100]
    first._encode_fact_texts(entries)
    old_key, _ = first._semantic_cache_key(entries=entries, until_time=100, closest_timestamp=100)
    second_model = FakeEmbedding('same-name')
    setattr(second_model, field, after)
    second = semantic(tmp_path, second_model)
    new_key, _ = second._semantic_cache_key(entries=entries, until_time=100, closest_timestamp=100)
    assert old_key != new_key
    assert second._encode_fact_texts(entries).shape == (1, second_model.dimension)
    assert second_model.calls == ['person holds cup']


def test_semantic_identity_does_not_load_lazy_model(tmp_path):
    class Lazy(FakeEmbedding):
        @property
        def text_model(self):
            raise AssertionError('must not load model for identity')
    value = semantic(tmp_path, Lazy())
    value._semantic_cache_key(entries=value.timestamp_to_triples[100], until_time=100, closest_timestamp=100)


def test_cache_identity_stays_stable_after_lazy_initialization(tmp_path):
    value = semantic(tmp_path)
    before = value._embedding_model_id()
    value.embedding_model._text_model = types.SimpleNamespace(model_name='same-underlying-model')
    assert value._embedding_model_id() == before


def test_remote_identity_config_invalidates_without_exposing_credentials(tmp_path, monkeypatch):
    model = FakeEmbedding('fixed')
    model.text_backend = 'remote'
    monkeypatch.setenv('EM2MEM_TEXT_EMBED_URL', 'https://user:secret@one.example/?api_key=sensitive')
    first = semantic(tmp_path, model)._embedding_model_id()
    identity = EPISODIC.embedding_cache_identity(model)
    assert 'secret' not in json.dumps(identity) and 'sensitive' not in json.dumps(identity)
    monkeypatch.setenv('EM2MEM_TEXT_EMBED_URL', 'https://two.example')
    assert semantic(tmp_path, model)._embedding_model_id() != first


def test_query_vectors_follow_changed_model_identity(tmp_path):
    value = semantic(tmp_path)
    assert value._get_query_embedding('alpha task').shape == (1, 2)
    value.embedding_model.dimension = 3
    assert value._get_query_embedding('alpha task').shape == (1, 3)


@pytest.mark.parametrize('bad', [np.array([[1., 0., 0.]]), np.array([[np.nan, 0.]])])
def test_semantic_aggregate_rejects_bad_dimension_or_finite(tmp_path, bad):
    value = semantic(tmp_path)
    key, _ = value._semantic_cache_key(entries=value.timestamp_to_triples[100], until_time=100, closest_timestamp=100)
    path = value._semantic_cache_path(key)
    np.savez(path, embeddings=bad, manifest=np.array(json.dumps({'cache_key': key})))
    assert value._load_embedding_cache(cache_path=path, cache_key=key, expected_count=1) is None


@pytest.mark.parametrize('bad', [np.array([1., 0., 0.]), np.array([np.inf, 0.])])
def test_semantic_fact_cache_reencodes_invalid_vector(tmp_path, bad):
    value = semantic(tmp_path)
    entries = value.timestamp_to_triples[100]
    value._encode_fact_texts(entries)
    np.save(next((tmp_path/'fact_vectors').glob('*.npy')), bad)
    value.embedding_model.calls.clear()
    assert value._encode_fact_texts(entries).shape == (1, 2)
    assert value.embedding_model.calls == ['person holds cup']


@pytest.mark.parametrize('bad', [np.array([[np.nan, 0.]]), np.zeros((0, 2)), np.zeros((1, 3))])
def test_semantic_invalid_encoder_response_never_persists(tmp_path, monkeypatch, bad):
    value = semantic(tmp_path)
    monkeypatch.setattr(value.embedding_model, 'encode_text', lambda *a, **k: bad)
    with pytest.raises(ValueError, match='embedding|vector|dimension'):
        value._encode_fact_texts(value.timestamp_to_triples[100])
    assert not list(tmp_path.rglob('*.npy'))


@pytest.mark.parametrize('mode', ['snapshot', 'flat_facts'])
def test_semantic_loader_invalidates_index_at_same_timestamp(semantic_runtime, mode):
    value = semantic(mode=mode)
    value.index(100)
    value.load_triples_from_data({'100': [['person', 'holds', 'book']]})
    value.index_mode = mode
    value.index(100)
    assert [item.object for item in value.indexed_entries] == ['book']


@pytest.mark.parametrize('kind', ['semantic', 'visual'])
def test_loader_invalidates_established_worker_index(semantic_runtime, kind):
    value = semantic() if kind == 'semantic' else VISUAL.VisualMemory()
    if kind == 'visual':
        value.load_clips_from_data([EARLY])
    cutoff = 100 if kind == 'semantic' else LATE_CUTOFF
    ready, loaded = threading.Event(), threading.Event()
    result, errors = [], []
    def worker():
        try:
            value.index(cutoff)
            ready.set()
            assert loaded.wait(5)
            value.index(cutoff)
            result.extend(item.object if kind == 'semantic' else item.doc_id for item in value.indexed_entries)
        except BaseException as exc:
            errors.append(exc)
            ready.set()
    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(5)
    if kind == 'semantic':
        value.load_triples_from_data({'100': [['person', 'holds', 'book']]})
    else:
        value.load_clips_from_data([LATE])
    loaded.set()
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert result == (['book'] if kind == 'semantic' else ['late'])


def test_visual_embedding_loader_refreshes_existing_clips(tmp_path, semantic_runtime):
    value = VISUAL.VisualMemory()
    value.load_clips_from_data([{**EARLY, 'video_path': 'clip.mp4'}])
    value.index(LATE_CUTOFF)
    path = tmp_path/'embeddings.pkl'
    path.write_bytes(pickle.dumps({'clip.mp4': np.array([1., 0.])}))
    value.load_embeddings_from_file(str(path))
    value.index(LATE_CUTOFF)
    assert value.embeddings.tolist() == [[1., 0.]]


def test_flat_facts_reuses_graph_tensor_and_aggregate_across_cutoffs(tmp_path, semantic_runtime):
    graph, tensors = semantic_runtime
    value = semantic(tmp_path)
    for cutoff in range(100, 110):
        value.index(cutoff)
    assert value.indexed_time == 109
    assert graph.builds == len(tensors) == 1
    assert len(list(tmp_path.glob('semantic_embeddings_*.npz'))) == 1
    assert value.embedding_model.calls == ['person holds cup']


def test_aggregate_survives_new_instance_with_different_cutoff(tmp_path, semantic_runtime):
    first = semantic(tmp_path)
    first.index(100)
    second = semantic(tmp_path)
    second.index(150)
    assert second.embedding_cache_status['hit'] is True
    assert second.embedding_model.calls == []
    assert len(list(tmp_path.glob('semantic_embeddings_*.npz'))) == 1


@pytest.mark.parametrize('kind', ['semantic', 'visual'])
def test_explicit_reset_invalidates_other_worker_derived_state(semantic_runtime, kind):
    value = semantic() if kind == 'semantic' else VISUAL.VisualMemory()
    ready, reset = threading.Event(), threading.Event()
    result = []
    def worker():
        value.indexed_entries = ['old view']
        value.indexed_time = 100
        ready.set()
        assert reset.wait(5)
        result.append((value.indexed_entries, value.indexed_time))
    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(5)
    value.reset_index()
    reset.set()
    thread.join(5)
    assert not thread.is_alive()
    assert result == [([], 0)]


def test_flat_reuse_never_hides_new_fact_view_or_weight(tmp_path, semantic_runtime):
    graph, _ = semantic_runtime
    value = semantic(tmp_path)
    value.timestamp_to_triples[200] = [SEMANTIC.SemanticTripleEntry(
        'semantic_100_0', 'person', 'holds', 'cup', 200,
        confidence=.9, support_count=4, source_doc_ids=['a','b'])]
    value.available_timestamps.append(200)
    value.index(100)
    old_graph = value.graph
    value.index(200)
    assert value.graph is not old_graph
    assert value.indexed_entries[0].confidence == .9
    assert value.graph.es['weight'] == [pytest.approx(.9 * 1.4)]
    assert len(list(tmp_path.glob('semantic_embeddings_*.npz'))) == 1
    value.index(100)
    assert value.indexed_entries[0].support_count == 1
    assert graph.builds == 3


def test_aggregate_identity_ignores_cutoff_but_not_content(tmp_path):
    value = semantic(tmp_path)
    entries = value.timestamp_to_triples[100]
    first = value._semantic_cache_key(entries=entries, until_time=100, closest_timestamp=100)[0]
    later = value._semantic_cache_key(entries=entries, until_time=101, closest_timestamp=100)[0]
    assert first == later
    entries[0].object = 'book'
    assert value._semantic_cache_key(entries=entries, until_time=101, closest_timestamp=100)[0] != first


@pytest.mark.parametrize(('cutoff', 'constraints', 'expected'), [
    (EARLY_CUTOFF, {}, ['early']),
    (LATE_CUTOFF, {'time_range': [0., 30.]}, ['early']),
    (LATE_CUTOFF, {}, ['early', 'late']),
])
def test_family_graph_expansion_obeys_asof_and_explicit_window(tmp_path, cutoff, constraints, expected):
    value = EPISODIC.EpisodicMemoryRAG(FakeEmbedding(), NoModels(), NoModels(), granularities=['30sec'])
    value.dense_cache_dir = str(tmp_path)
    value.load_captions_from_data({'30sec': [EARLY, LATE]})
    value.graph_sidecar['30sec'] = {
        'early': EPISODIC.GraphEventSidecarRAG('e', 'early', '30sec', next_doc_id='late'),
        'late': EPISODIC.GraphEventSidecarRAG('l', 'late', '30sec', prev_doc_id='early'),
    }
    value.index(cutoff)
    value.set_query_constraints(constraints)
    result = value.retrieve_ranked_with_family('alpha task', {'question_family': 'event'})
    assert [entry.doc_id for entry, _ in result] == expected


@pytest.mark.parametrize(('cutoff', 'constraints'), [
    (EARLY_CUTOFF, {}),
    (LATE_CUTOFF, {'time_range': [0., 30.]}),
])
def test_family_fallback_filters_stage_candidates_before_topk(tmp_path, monkeypatch, cutoff, constraints):
    value = EPISODIC.EpisodicMemoryRAG(FakeEmbedding(), NoModels(), NoModels(), granularities=['30sec'])
    value.dense_cache_dir = str(tmp_path)
    value.load_captions_from_data({'30sec': [EARLY, LATE]})
    value.index(cutoff)
    value.set_query_constraints(constraints)
    policy = value._family_policy({'question_family': 'event'})
    monkeypatch.setattr(value, '_family_policy', lambda _: {**policy, 'final_top_k': 1})
    # Boundary fault injection: an optional projection/refine implementation
    # may hand back a stale or overinclusive candidate pool. The final fallback
    # must enforce scope itself and fill from the next eligible candidate.
    monkeypatch.setattr(value, '_project_ranked_candidates_to_30s', lambda **_: {'late': 9., 'early': 1.})
    monkeypatch.setattr(value, '_family_graph_refine_30s', lambda *args: [])
    result = value.retrieve_ranked_with_family('alpha task', {'question_family': 'event'})
    assert [entry.doc_id for entry, _ in result] == ['early']
