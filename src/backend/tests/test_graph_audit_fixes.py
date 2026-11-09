"""Graph/semantic recovery and replacement using real temporary persistence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from online_memory_incremental import hipporag_cache as cache
from online_memory_incremental import incremental_store as store
from online_memory_incremental.graph_delta import generate_graph_delta


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def episode(eid, start=0):
    return {'episode_id': eid, 'session_id': 'test', 'segment_id': eid,
            'start_time': start, 'end_time': start + 30}


def evidence(eid, start=0, obj='phone'):
    return {**episode(eid, start), 'doc_id': eid, 'caption': f'person holds {obj}',
            'fine_caption': f'person holds {obj}',
            'main_actions': [{'actor': 'person', 'action': 'holds', 'objects': [obj]}]}


@pytest.fixture
def app(tmp_path, monkeypatch):
    for key, val in {
        'EM2MEM_MEMORY_GENERATION_BACKEND': 'rule', 'EM2MEM_INCREMENTAL_GRAPH_BACKEND': 'rule',
        'EM2MEM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND': 'rule', 'EM2MEM_AUTO_VISUAL_EMBEDDING': '0',
        'EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE': '0',
        'EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE_ON_NOOP': '0',
    }.items():
        monkeypatch.setenv(key, val)
    value = store.IncrementalMemoryAppender(session_id='test', sessions_root=tmp_path / 'sessions',
                                            project_root=tmp_path / 'project', model_name='offline')
    value.project_root.mkdir()
    seed(value, [episode('A')], [evidence('A')])
    return value


def seed(app, episodes, docs):
    write(app.session_dir / 'em2mem/mst_episodic/mst_30sec_episodes.json', episodes)
    write(app.session_dir / 'evidence/mst_session_evidence.json', docs)


def facts(app):
    p = app.session_dir / 'em2mem/semantic_root/semantic_memory_offline.json'
    return json.loads(p.read_text())['facts'] if p.is_file() else []


def support_docs(app):
    return {d for fact in facts(app) for d in fact['support_docs']}


def test_cache_refresh_passes_constructed_memory_to_sync(tmp_path, monkeypatch):
    monkeypatch.setenv('EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE', '1')
    memory = SimpleNamespace(cleanup=lambda: None)
    calls = []
    monkeypatch.setattr(cache, '_build_em2mem_memory_for_cache', lambda **kw: memory)
    monkeypatch.setattr(cache, '_copy_granularity_cache_dirs', lambda **kw: None)
    monkeypatch.setattr(cache, '_granularities_to_refresh', lambda *args: ['30sec'])
    monkeypatch.setattr(cache, 'inspect_hipporag_cache_health', lambda *args, **kw: {
        'healthy': True, 'levels': {'30sec': {'healthy': True}},
    })
    monkeypatch.setattr(cache, '_sync_granularity_in_tmp', lambda **kw: calls.append(kw['em2mem_memory']) or {})
    monkeypatch.setattr(cache, '_replace_granularity_cache_dirs', lambda **kw: ['30sec'])
    result = cache.refresh_hipporag_cache(session_dir=tmp_path / 's', project_root=tmp_path, force=True)
    assert calls == [memory]
    assert result['healthy'] is True


def test_cache_factory_uses_the_backend_required_by_real_sync(tmp_path, monkeypatch):
    import em2mem.embedding as embedding
    import em2mem.llm as llm
    import em2mem.memory.episodic.memory as legacy

    monkeypatch.setattr(embedding, 'EmbeddingModel', lambda: object())
    monkeypatch.setattr(llm, 'LLMModel', lambda **kw: object())
    monkeypatch.setattr(llm, 'PromptTemplateManager', lambda: object())
    calls = []
    class OfflineHippoRAG:
        def __init__(self, **kwargs):
            pass
        def update(self, *, docs):
            calls.append(docs)
        def prepare_retrieval_objects(self):
            calls.append('prepared')
    monkeypatch.setattr(legacy, 'HippoRAG', OfflineHippoRAG)
    session = tmp_path / 's'
    write(session / 'em2mem/caption_root/s_30sec.json', [{
        'doc_id': 'A', 'date': 'DAY1', 'start_time': '00000000', 'end_time': '00003000',
        'text': 'person holds phone',
    }])
    memory = cache._build_em2mem_memory_for_cache(session_dir=session, cache_tag='offline', model_name='offline')
    try:
        cache._sync_granularity_in_tmp(em2mem_memory=memory, session_dir=session,
                                      granularity='30sec', info={}, tmp_tag='offline', project_root=tmp_path)
        assert calls == [['person holds phone'], 'prepared']
    finally:
        memory.cleanup()


@pytest.mark.parametrize('new_episode', [False, True])
@pytest.mark.parametrize('component', ['graph', 'semantic'])
def test_transient_persistence_failure_is_repaired_without_fast_duplication(app, new_episode, component):
    # A real path conflict causes persistence failure after text append succeeds.
    blocked = app.session_dir / f'em2mem/incremental/{component}/deltas/{component}_delta_v000001.jsonl'
    blocked.mkdir(parents=True)
    first = app.append_ready_episodes()
    assert first.semantic_lagging
    if component == 'graph':
        assert first.graph_lagging
    blocked.rmdir()
    before = app.layout.caption_30sec_path.read_bytes()
    if new_episode:
        seed(app, [episode('A'), episode('B', 30)], [evidence('A'), evidence('B', 30, 'book')])
    retry = app.append_ready_episodes(episode_ids=['B'] if new_episode else None)
    assert not retry.graph_lagging and not retry.semantic_lagging
    assert support_docs(app) == ({'A', 'B'} if new_episode else {'A'})
    assert retry.fast_memory_version == (first.fast_memory_version + 1 if new_episode else first.fast_memory_version)
    assert retry.appended_episode_ids == (['B'] if new_episode else [])
    if not new_episode:
        assert app.layout.caption_30sec_path.read_bytes() == before
    noop = app.append_ready_episodes()
    assert not noop.graph_lagging and not noop.semantic_lagging
    assert noop.fast_memory_version == retry.fast_memory_version


def test_new_version_cannot_mask_an_older_unrepaired_episode(app):
    blocked = app.session_dir / 'em2mem/incremental/graph/deltas/graph_delta_v000001.jsonl'
    blocked.mkdir(parents=True)
    app.append_ready_episodes()
    blocked.rmdir()
    blocked = blocked.with_name('graph_delta_v000002.jsonl')
    blocked.mkdir()
    seed(app, [episode('A'), episode('B', 30)], [evidence('A'), evidence('B', 30, 'book')])
    second = app.append_ready_episodes(episode_ids=['B'])
    assert second.graph_lagging and second.semantic_lagging
    config = app.load_memory_config()
    assert config['latest_fast_ready_version'] == 2
    assert config['latest_graph_ready_version'] < 2
    assert config['latest_semantic_ready_version'] < 2
    blocked.rmdir()
    repaired = app.append_ready_episodes()
    assert repaired.fast_memory_version == 2 and repaired.appended_episode_ids == []
    assert set(repaired.repaired_episode_ids) == {'A', 'B'}
    assert support_docs(app) == {'A', 'B'}


def test_multilingual_entities_keep_distinct_graph_nodes():
    from online_memory.evidence_to_em2mem import _build_graph_payload
    triples = [['乘员', '拿着', '手机'], ['乘员', '佩戴', '墨镜'],
               ['cup-a', 'near', 'cup a'], ['x' * 81 + 'a', 'near', 'x' * 81 + 'b']]
    graph = _build_graph_payload([{'doc_id': 'A', 'text': 'test'}], {'A': triples}, '30sec')
    entities = [n for n in graph['nodes'] if n['type'] == 'Entity']
    assert len(entities) == len({value for h, _, t in triples for value in (h, t)})
    labels = {n['id']: n['label'] for n in entities}
    reconstructed = [(labels[e['source']], e['type'], labels[e['target']])
                     for e in graph['edges'] if e['type'] not in {'mentions', 'before'}]
    assert reconstructed == [tuple(t) for t in triples]
    assert not any(e['source'] == e['target'] for e in graph['edges'])


def test_noop_reports_actual_lag_when_components_are_disabled(app):
    first = app.append_ready_episodes(skip_graph_semantic=True)
    assert first.graph_lagging and first.semantic_lagging
    second = app.append_ready_episodes(skip_graph_semantic=True)
    assert second.graph_lagging and second.semantic_lagging


def test_missing_episode_evidence_stays_retryable(app):
    seed(app, [episode('A'), episode('B', 30)], [evidence('A')])
    first = app.append_ready_episodes()
    assert first.appended_episode_ids == ['A']
    assert app.append_log.latest_by_episode()['B']['status'] == 'failed'
    seed(app, [episode('A'), episode('B', 30)], [evidence('A'), evidence('B', 30)])
    second = app.append_ready_episodes()
    assert second.appended_episode_ids == ['B']
    assert len(json.loads(app.layout.caption_30sec_path.read_text())) == 2
    assert support_docs(app) == {'A', 'B'}


def test_all_missing_evidence_does_not_publish_an_empty_success(app):
    seed(app, [episode('A')], [])
    result = app.append_ready_episodes()
    assert result.appended_episode_ids == []
    assert app.append_log.latest_by_episode()['A']['status'] == 'failed'
    assert not app.layout.caption_30sec_path.exists()


def test_rebuilding_graph_replaces_only_the_changed_document(app):
    seed(app, [episode('A'), episode('B', 30)], [evidence('A'), evidence('B', 30)])
    app.append_ready_episodes()
    seed(app, [episode('A'), episode('B', 30)], [evidence('A', obj='book'), evidence('B', 30)])
    app.append_ready_episodes(episode_ids=['A'], force=True)
    p = app.layout.sidecar_root / '30s/episodic_triplets_30s_offline.json'
    triples = json.loads(p.read_text())['triplet_map']
    assert ['person', 'holds', 'phone'] not in triples['A']
    assert ['person', 'holds', 'book'] in triples['A']
    assert ['person', 'holds', 'phone'] in triples['B']


@pytest.mark.parametrize('legacy_store', [False, True])
def test_rebuilding_one_document_keeps_other_semantic_support(app, legacy_store):
    seed(app, [episode('A'), episode('B', 30)], [evidence('A'), evidence('B', 30)])
    app.append_ready_episodes()
    merged = next(f for f in facts(app) if f['triple'] == ['person', 'holds', 'phone'])
    assert set(merged['support_docs']) == {'A', 'B'}
    if legacy_store:
        path = app.layout.semantic_root / 'semantic_memory_offline.json'
        old = json.loads(path.read_text())
        for fact in old['facts']:
            fact.pop('support_records', None)
        write(path, old)
    seed(app, [episode('A'), episode('B', 30)], [evidence('A', obj='book'), evidence('B', 30)])
    app.append_ready_episodes(episode_ids=['A'], force=True)
    phone = next(f for f in facts(app) if f['triple'] == ['person', 'holds', 'phone'])
    assert phone['support_docs'] == ['B'] and phone['support_count'] == 1
    assert phone['first_seen']['start_time'] == '00003000'
    assert phone['last_seen']['end_time'] == '00010000'
    assert phone['source_doc_ids'] == phone['provenance_root_ids'] == ['B']
    payload = json.loads((app.layout.semantic_root / 'semantic_memory_offline.json').read_text())
    assert {fid for row in payload['timeline'] for fid in row['fact_ids']} <= {f['fact_id'] for f in payload['facts']}


@pytest.mark.parametrize('stdout', [
    'Using OpenAI base URL: offline\n{"status":"failed","healthy":false,"error":"broken"}',
    'log {not JSON}\n{\n "status": "failed", "healthy": false, "error": "broken"\n}\n',
    'no result at all',
])
def test_cache_subprocess_logs_cannot_turn_failure_into_success(app, monkeypatch, stdout):
    monkeypatch.setenv('EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE', '1')
    monkeypatch.setattr(store.subprocess, 'run', lambda *args, **kw: SimpleNamespace(returncode=0, stdout=stdout, stderr=''))
    result = app._refresh_hipporag_cache_isolated()
    assert result['status'] == 'failed' and result['healthy'] is False
    assert result.get('error')


def test_successful_cache_json_with_log_prefix_stays_successful(app, monkeypatch):
    monkeypatch.setenv('EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE', '1')
    monkeypatch.setattr(store.subprocess, 'run', lambda *args, **kw: SimpleNamespace(
        returncode=0, stdout='initialization log\n{"status":"healthy","healthy":true}', stderr='',
    ))
    assert app._refresh_hipporag_cache_isolated()['healthy'] is True


def test_noop_retains_independent_component_lag_flags(app):
    blocked = app.session_dir / 'em2mem/incremental/semantic/deltas/semantic_delta_v000001.jsonl'
    blocked.mkdir(parents=True)
    result = app.append_ready_episodes()
    assert not result.graph_lagging and result.semantic_lagging
    noop = app.append_ready_episodes(skip_graph_semantic=True)
    assert not noop.graph_lagging and noop.semantic_lagging
