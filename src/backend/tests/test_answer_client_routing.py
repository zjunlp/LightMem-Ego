"""Answer models must follow the selected local/remote configuration."""
import json
from types import SimpleNamespace

import pytest

from online_query import query_engine as engine


@pytest.fixture
def clients(monkeypatch):
    monkeypatch.setenv('EM2MEM_EXTERNAL_OPENAI_BASE_URL', 'https://relay.example/v1')
    monkeypatch.setenv('EM2MEM_EXTERNAL_OPENAI_API_KEY', 'test-relay-key')
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_BASE_URL', 'http://127.0.0.1:18100/v1')
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_API_KEY', 'test-local-key')
    monkeypatch.setenv('EM2MEM_QUERY_RESPOND_MODEL', 'test-model')
    monkeypatch.setenv('EM2MEM_MCUR_ANSWER_MODEL', 'test-model')
    monkeypatch.setattr(engine, '_SHORT_TERM_ANSWER_MODELS', {})
    made = []
    def make(**kwargs):
        model = SimpleNamespace(**kwargs)
        made.append(model)
        return model
    return made, make


@pytest.mark.parametrize('local', [False, True])
def test_long_term_answer_loader_uses_selected_endpoint(tmp_path, monkeypatch, clients, local):
    made, make = clients
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_ENABLED', '1' if local else '0')
    class StopBeforeMemoryLoad(Exception):
        pass
    def templates():
        raise StopBeforeMemoryLoad
    monkeypatch.setattr(engine, '_em2mem_classes', lambda *args: (object, make, templates, None, None))
    monkeypatch.setattr(engine, '_query_rag_helpers', lambda: None)
    session = tmp_path / 's'
    session.mkdir()
    config = {'status': 'memory_ready', 'latest_ready_memory_version': 1,
              'query_rag_args': {'episodic_caption_root': 'captions',
                                 'episodic_sidecar_root': 'sidecars', 'semantic_root': 'semantic'}}
    path = session / 'memory_config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(StopBeforeMemoryLoad):
        engine.load_query_engine('s', tmp_path, memory_config_override=(path, config))
    assert len(made) == 2
    assert made[0].base_url == 'https://relay.example/v1'
    assert made[0].local_mode is False
    assert made[1].base_url == ('http://127.0.0.1:18100/v1' if local else 'https://relay.example/v1')
    assert made[1].api_key == ('test-local-key' if local else 'test-relay-key')
    assert made[1].local_mode is local


@pytest.mark.parametrize('local', [False, True])
def test_current_and_short_term_answer_client_uses_selected_endpoint(monkeypatch, clients, local):
    _made, make = clients
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_ENABLED', '1' if local else '0')
    monkeypatch.setattr(engine, '_em2mem_classes', lambda *args: (None, make, None, None, None))
    model = engine._get_short_term_answer_model()
    assert model.base_url == ('http://127.0.0.1:18100/v1' if local else 'https://relay.example/v1')
    assert model.api_key == ('test-local-key' if local else 'test-relay-key')
    assert model.local_mode is local


def test_answer_client_cache_does_not_reuse_a_different_profile(monkeypatch, clients):
    _made, make = clients
    monkeypatch.setattr(engine, '_em2mem_classes', lambda *args: (None, make, None, None, None))
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_ENABLED', '1')
    first = engine._get_short_term_answer_model()
    monkeypatch.setenv('EM2MEM_LOCAL_LLM_ENABLED', '0')
    second = engine._get_short_term_answer_model()
    assert second is not first
    assert second.base_url == 'https://relay.example/v1'
    assert engine._get_short_term_answer_model() is second
