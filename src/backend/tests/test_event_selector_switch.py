"""The demo switch bypasses API selection while retaining scores and budget."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from online_query import query_engine  # Sets up the project's src import path.
from em2mem.memory.EM2Memory import EM2Memory


def memory():
    instance = EM2Memory.__new__(EM2Memory)
    instance.retriever_llm_model = SimpleNamespace(generate=Mock(return_value='{"selected_indices":[2]}'))
    instance._build_query_with_time = lambda **kwargs: kwargs["query"]
    return instance


def candidates():
    return [{"index": i, "doc_id": f"event-{i}", "episodic_score": score}
            for i, score in [(1, .2), (2, .9), (3, .5)]]


@pytest.mark.parametrize("flag", ["0", "false", "off", "no"])
def test_disabled_selector_uses_original_scores_without_an_api_request(monkeypatch, flag):
    monkeypatch.setenv("EM2MEM_LLM_EVENT_SELECTOR_ENABLED", flag)
    instance = memory()
    selected, reason = instance._select_top_events_with_llm("question", None, None, candidates(), 2)
    assert selected == ["event-2", "event-3"]
    assert "disabled" in reason
    instance.retriever_llm_model.generate.assert_not_called()


def test_default_selector_still_calls_model_when_pool_exceeds_budget(monkeypatch):
    monkeypatch.delenv("EM2MEM_LLM_EVENT_SELECTOR_ENABLED", raising=False)
    instance = memory()
    selected, _ = instance._select_top_events_with_llm("question", None, None, candidates(), 2)
    assert selected == ["event-2"]
    instance.retriever_llm_model.generate.assert_called_once()


def test_enabled_shortcut_keeps_existing_no_api_behavior(monkeypatch):
    monkeypatch.setenv("EM2MEM_LLM_EVENT_SELECTOR_ENABLED", "1")
    instance = memory()
    selected, _ = instance._select_top_events_with_llm("question", None, None, candidates(), 3)
    assert selected == ["event-1", "event-2", "event-3"]
    instance.retriever_llm_model.generate.assert_not_called()
