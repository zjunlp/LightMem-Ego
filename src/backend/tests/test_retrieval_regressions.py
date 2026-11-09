"""Offline retrieval regressions using source definitions and deterministic embeddings.

Heavy imports are excluded by the helper; all retrieval/indexing methods execute
unchanged production source. No network, models, or external video codecs run.
"""
import json
import threading
import types
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval_source_helpers import (
    EPISODIC, SEMANTIC, UNIFIED, VISUAL, EARLY, LATE, EARLY_CUTOFF,
    LATE_CUTOFF, FakeEmbedding, NoModels, caption, make_memory,
)


def test_query_cutoff_can_move_backwards_and_reuse_dense_embeddings(tmp_path):
    memory = make_memory(tmp_path)
    memory.index(LATE_CUTOFF)
    document_calls = list(memory.embedding_model.calls)
    result = memory.answer('target', until_time=EARLY_CUTOFF,
                           use_image_evidence=False, generate_answer=False)
    assert result.selected_doc_ids == ['early']
    assert memory.embedding_model.calls == document_calls + ['target']
    assert memory.visual_memory.indexed_time == EARLY_CUTOFF
    assert memory.semantic_memory.indexed_time == EARLY_CUTOFF


def test_same_cutoff_query_initializes_each_thread(tmp_path):
    memory = make_memory(tmp_path)
    memory.index(LATE_CUTOFF)
    results, failures = [], []
    def run():
        try:
            memory.index(LATE_CUTOFF)
            result = memory.answer('target', until_time=LATE_CUTOFF,
                                   use_image_evidence=False, generate_answer=False)
            results.append(set(result.selected_doc_ids))
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert not failures
    assert results == [{'early', 'late'}]


def episode(model, cache):
    memory = EPISODIC.EpisodicMemoryRAG(model, NoModels(), NoModels(), granularities=['30sec'])
    memory.dense_cache_dir = str(cache)
    memory.load_captions_from_data({'30sec': [EARLY, LATE]})
    memory.index(LATE_CUTOFF)
    return memory


@pytest.mark.parametrize('dimension', [2, 3])
def test_dense_cache_rejects_different_embedding_models(tmp_path, dimension):
    episode(FakeEmbedding('fake/A', 2), tmp_path)
    model = FakeEmbedding('fake/B', dimension, rotate=True)
    memory = episode(model, tmp_path)
    assert model.calls == ['alpha task', 'beta task']
    assert memory.retrieve_ranked('target', top_k_per_granularity=1)[0][0].doc_id == 'early'


def test_dense_cache_reuses_same_model(tmp_path):
    episode(FakeEmbedding('fake/A'), tmp_path)
    model = FakeEmbedding('fake/A')
    memory = episode(model, tmp_path)
    assert model.calls == []
    assert memory.retrieve_ranked('target', top_k_per_granularity=1)[0][0].doc_id == 'early'


@pytest.mark.parametrize('damage', ['legacy', 'dimension', 'version', 'config'])
def test_dense_cache_rejects_stale_or_incompatible_metadata(tmp_path, damage):
    episode(FakeEmbedding(), tmp_path)
    meta_path = tmp_path / '30sec/meta.json'
    if damage == 'dimension':
        np.save(tmp_path / '30sec/doc_embeddings.npy', np.zeros((2, 5), dtype=np.float32))
    else:
        meta = json.loads(meta_path.read_text())
        if damage == 'legacy':
            meta = {key: meta[key] for key in ('doc_ids', 'text_hashes', 'end_times')}
        elif damage == 'version':
            meta['cache_version'] = -1
        else:
            meta['embedding_identity'] = {'different': True}
        meta_path.write_text(json.dumps(meta))
    model = FakeEmbedding()
    memory = episode(model, tmp_path)
    # The aggregate cache is untrusted; valid independent text vectors can
    # reconstruct it without re-encoding unchanged documents.
    assert memory.doc_embeddings['30sec'].shape == (2, 2)
    assert np.isfinite(memory.doc_embeddings['30sec']).all()
    assert memory.retrieve_ranked('target', top_k_per_granularity=1)[0][0].doc_id == 'early'


def test_parent_context_respects_current_cutoff(tmp_path):
    future = 'FUTURE_ONLY: password changes to violet'
    parent = caption('parent', '00000000', '00030000', future)
    memory = make_memory(tmp_path, {'30sec': [EARLY, LATE], '3min': [parent]})
    result = memory.answer('target', until_time=EARLY_CUTOFF,
                           use_image_evidence=False, generate_answer=False)
    assert result.selected_doc_ids == ['early']
    assert all(future not in item.content for item in result.retrieved_items)
    candidates = memory._build_event_selector_candidates('target', {'early': 1.}, {'early': 1.}, {})
    assert not candidates[0]['parent_3min_caption']
    memory.index(100030000)
    assert memory.episodic_memory.get_parent_caption('early').text == future
    memory.index(EARLY_CUTOFF)
    assert memory.episodic_memory.get_parent_caption('early') is None


def test_visual_time_range_resolves_video_relative_to_base_dir(tmp_path, monkeypatch):
    class FakeReader:
        def __init__(self, path, ctx):
            assert Path(path).exists()
        def get_avg_fps(self): return 30.
        def __len__(self): return 900
        def get_batch(self, indices):
            return types.SimpleNamespace(asnumpy=lambda: np.zeros((len(indices), 2, 2, 3), dtype=np.uint8))
    monkeypatch.setitem(sys.modules, 'decord', types.SimpleNamespace(VideoReader=FakeReader, cpu=lambda x: x))
    video = tmp_path / 'clip.mp4'
    video.touch()
    def retrieve(path):
        memory = VISUAL.VisualMemory()
        memory.set_base_dir(tmp_path)
        memory.load_clips_from_data([caption('early', '00000000', '00003000', '', video_path=str(path))])
        return memory.retrieve('DAY1 00:00:01 - DAY1 00:00:02', as_context=False)
    assert len(retrieve('clip.mp4')) == len(retrieve(video)) == 2


def test_retrieval_timing_includes_dense_and_semantic_work(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    memory.index(LATE_CUTOFF)
    clock = types.SimpleNamespace(seconds=0.)
    monkeypatch.setattr(UNIFIED, 'time', types.SimpleNamespace(perf_counter=lambda: clock.seconds))
    encode = memory.embedding_model.encode_text
    def timed_encode(*args, **kwargs):
        clock.seconds += 2.
        return encode(*args, **kwargs)
    monkeypatch.setattr(memory.embedding_model, 'encode_text', timed_encode)
    semantic_retrieve = memory.semantic_memory.retrieve
    def timed_semantic(*args, **kwargs):
        clock.seconds += 3.
        return semantic_retrieve(*args, **kwargs)
    monkeypatch.setattr(memory.semantic_memory, 'retrieve', timed_semantic)
    result = memory.answer('target', until_time=LATE_CUTOFF,
                           use_image_evidence=False, generate_answer=False)
    assert result.timing_ms['retrieval_ms'] == 5000
    assert result.timing_ms['selector_ms'] == 0


def test_all_query_indexes_are_thread_local_and_allow_cutoff_reversal(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    memory.visual_memory.load_clips_from_data([EARLY, LATE])
    semantic = memory.semantic_memory
    semantic.index_mode = 'flat_facts'
    semantic.available_timestamps = [EARLY_CUTOFF, LATE_CUTOFF]
    semantic.timestamp_to_triples = {
        ts: [types.SimpleNamespace(id=name, subject=name, object='', confidence=1.,
                                   support_count=1, text=name, timestamp=ts)]
        for ts, name in [(EARLY_CUTOFF, 'early'), (LATE_CUTOFF, 'late')]
    }
    class FakeGraph:
        def add_vertices(self, vertices): pass
    monkeypatch.setattr(SEMANTIC, 'ig', types.SimpleNamespace(Graph=FakeGraph), raising=False)
    monkeypatch.setattr(SEMANTIC, 'torch', types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float32=np.float32,
        tensor=lambda data, **kwargs: np.asarray(data)))
    barrier = threading.Barrier(2)
    results, failures = {}, []
    def run(name, cutoff):
        try:
            memory.index(cutoff)
            barrier.wait(timeout=5)
            results[name] = (
                {e.doc_id for e, _ in memory.episodic_memory.retrieve_ranked('target')},
                {e.id for e in semantic.indexed_entries},
                {e.doc_id for e in memory.visual_memory.indexed_entries},
            )
        except BaseException as exc:
            failures.append(exc)
    workers = [threading.Thread(target=run, args=('early', EARLY_CUTOFF)),
               threading.Thread(target=run, args=('late', LATE_CUTOFF))]
    for worker in workers: worker.start()
    for worker in workers: worker.join(timeout=10)
    assert not failures
    assert results == {'early': ({'early'}, {'early'}, {'early'}),
                       'late': ({'early', 'late'}, {'early', 'late'}, {'early', 'late'})}
    for cutoff, expected in [(LATE_CUTOFF, {'early', 'late'}),
                             (EARLY_CUTOFF, {'early'}), (LATE_CUTOFF, {'early', 'late'})]:
        memory.index(cutoff)
        assert {e.id for e in semantic.indexed_entries} == expected
        assert {e.doc_id for e in memory.visual_memory.indexed_entries} == expected


def test_concurrent_cold_indexes_encode_documents_once(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    episode_memory = memory.episodic_memory
    original_build = episode_memory.build_dense_index
    original_encode = memory.embedding_model.encode_text
    second_entered = threading.Event()
    lock = threading.Lock()
    callers = []
    def build(*args, **kwargs):
        with lock:
            callers.append(threading.get_ident())
            if len(callers) >= 2:
                second_entered.set()
        return original_build(*args, **kwargs)
    def encode(*args, **kwargs):
        assert second_entered.wait(timeout=5)
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(episode_memory, 'build_dense_index', build)
    monkeypatch.setattr(memory.embedding_model, 'encode_text', encode)
    errors = []
    def run():
        try:
            memory.index(LATE_CUTOFF)
        except BaseException as exc:
            errors.append(exc)
    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers: worker.join(timeout=10)
    assert not errors
    assert memory.embedding_model.calls == ['alpha task', 'beta task']


def test_no_evidence_still_reports_retrieval_time(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    clock = types.SimpleNamespace(seconds=0.)
    monkeypatch.setattr(UNIFIED, 'time', types.SimpleNamespace(perf_counter=lambda: clock.seconds))
    def empty_retrieve(**kwargs):
        clock.seconds += 2.
        return []
    monkeypatch.setattr(memory.episodic_memory, 'retrieve_ranked', empty_retrieve)
    result = memory.answer('target', until_time=EARLY_CUTOFF,
                           use_image_evidence=False, generate_answer=False)
    assert result.retrieved_items == []
    assert result.timing_ms['retrieval_ms'] == 2000


@pytest.mark.parametrize('kind', ['semantic', 'visual'])
def test_query_index_cleanup_remains_compatible_with_thread_local_properties(kind):
    memory = SEMANTIC.SemanticMemory(FakeEmbedding()) if kind == 'semantic' else VISUAL.VisualMemory()
    memory.embeddings = object()
    memory.cleanup()
    assert memory.embeddings is None
    memory.reset_index()
    assert memory.indexed_entries == []
    assert memory.indexed_time == 0


def test_dense_cache_rejects_changed_dimension_with_same_model_name(tmp_path):
    episode(FakeEmbedding('fake/A', 2), tmp_path)
    model = FakeEmbedding('fake/A', 3)
    memory = episode(model, tmp_path)
    assert model.calls == ['alpha task', 'beta task']
    assert memory.retrieve_ranked('target', top_k_per_granularity=1)[0][0].doc_id == 'early'


def test_dense_cache_tracks_normalization_config_and_reuses_matching_config(tmp_path):
    first = FakeEmbedding('fake/A')
    first.normalize = False
    episode(first, tmp_path)
    changed = FakeEmbedding('fake/A')
    changed.normalize = True
    episode(changed, tmp_path)
    assert changed.calls == ['alpha task', 'beta task']
    unchanged = FakeEmbedding('fake/A')
    unchanged.normalize = True
    episode(unchanged, tmp_path)
    assert unchanged.calls == []
