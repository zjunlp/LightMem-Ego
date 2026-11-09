"""Original request scope is separate from dialogue and retrieval wording."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import online_query_constraints as constraints
from online_query.coreference_resolver import CoreferenceResolver
from online_query.memory_router import MemoryRouter
from online_query.query_router import QueryRouter
from online_query.evidence_packer import EvidencePacker
from online_query import query_engine as engine
from test_event_query_integration_20261025 import fallback_session, _fallback
from test_query_reference_round2 import current_session


def runtime():
    return {"current_ready": True, "short_term_ready": True, "long_term_ready": True,
            "mcur_time_span": [271, 300], "mst_time_span": [0, 290], "mlt_time_span": [0, 900]}


@pytest.mark.parametrize("router", [QueryRouter, MemoryRouter])
def test_authoritative_empty_scope_is_not_reparsed_by_router(router):
    r = router()
    result = r.route("at 900 seconds what was on the meeting table?", {}, runtime(), query_constraints={})
    assert result["query_constraints"] == {}


@pytest.mark.parametrize("router", [QueryRouter, MemoryRouter])
def test_router_keeps_supplied_relative_window_and_does_not_mutate_it(router):
    original = constraints.parse_query_constraints("What did the cook do in the past 2 minutes?", 300)
    before = deepcopy(original)
    result = router().route("What did the cook do in the past 2 minutes?", {}, runtime(),
                            {"query_constraints": {"time_range": [800, 900]}}, query_constraints=original)
    assert result["query_constraints"]["time_range"] == [180, 300]
    assert original == before


def test_constraint_helper_copies_authoritative_values_even_when_empty():
    given = {"requirements": {"comparison": True}, "time_ranges": [[10, 20]]}
    result = constraints.resolve_query_constraints("at900 seconds", query_constraints=given, latest=900)
    result["requirements"]["comparison"] = False
    assert given["requirements"]["comparison"] is True
    assert constraints.resolve_query_constraints("at900 seconds", query_constraints={}, latest=900) == {}


def test_observed_frontier_precedes_a_different_long_term_sessions_later_clock():
    assert constraints.observed_query_time(runtime()) == 300
    assert constraints.observed_query_time({"mst_time_span": [0, 200], "mlt_time_span": [0, 100]}) == 200
    assert constraints.observed_query_time({"mcur_time_span": [0, 0], "mlt_time_span": [0, 900]}) == 0
    assert constraints.observed_query_time({"mcur_ready": False, "current_time": 0, "mst_time_span": [0, 200]}) == 200
    assert constraints.observed_query_time({"current_ready": False, "mcur_time_span": [0, 0]}) is None


def cache():
    turn = {"turn_id": "earlier", "question": "What is on the table?", "answer": "A binder is on the table.",
            "anchor_status": "confirmed", "time_ranges": [{"start": 20, "end": 20}],
            "entities": [{"name": "binder"}], "evidence_frames": [{"path": "binder.jpg", "timestamp": 20}]}
    return SimpleNamespace(latest_context=lambda: {"latest_interaction": turn, "recent_interactions": [turn]})


def test_resolver_anchors_relative_user_time_to_observed_frontier():
    result = CoreferenceResolver().resolve("最近两分钟他在做什么？", cache(), latest_seconds=300)
    assert not result.get("referenced_time_ranges")
    assert not result.get("is_followup")


def test_resolver_does_not_reparse_search_text_with_authoritative_empty_scope():
    result = CoreferenceResolver().resolve("At900 seconds, what about it?", cache(),
                                           latest_seconds=300, query_constraints={})
    assert result.get("is_followup")


def test_packer_explicit_empty_scope_wins_over_stale_cache_window():
    event = {"evidence_id": "meeting", "source_memory": "M_st", "start_time": 0, "end_time": 30,
             "keyframes": [{"path": "binder.jpg", "timestamp": 10}], "keyframe_paths": ["binder.jpg"],
             "visual_observations": [{"path": "binder.jpg", "timestamp": 10, "description": "A person reads a binder."}]}
    result = EvidencePacker().pack("What happened at900 seconds?",
        {"query_constraints": {}, "use_image_evidence": True, "max_image_evidence": 1},
        {"final_evidence": [event]}, {"query_constraints": {"time_range": [900, 900]}})
    assert result["evidence_pack_summary"]["query_constraints"] == {}
    assert result["selected_image_paths_for_mllm"] == ["binder.jpg"]


def test_loaded_routing_parses_original_scope_once(monkeypatch):
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.session_id = "meeting"
    calls = []
    parser = constraints.parse_query_constraints
    def counted(question, latest=None):
        calls.append((question, latest))
        return parser(question, latest)
    monkeypatch.setattr(constraints, "parse_query_constraints", counted)
    context = {"query_constraints": {"time_range": [800, 900]}}
    route, _, _, _ = loaded._route_and_plan("What changed in the past 2 minutes?", {}, runtime(), context)
    assert calls == [("What changed in the past 2 minutes?", 300)]
    assert route["query_constraints"]["time_range"] == [180, 300]
    assert context["query_constraints"]["time_range"] == [180, 300]


def test_current_answer_prompt_carries_the_same_request_scope(current_session, monkeypatch):
    root, _ = current_session
    prompts = []
    class Model:
        def generate(self, prompt):
            prompts.append(prompt)
            return "有人看向腿边。"
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    result = engine.query_session("s", "现在能看到什么？", sessions_root=root,
                                  cache_mode="off", max_image_evidence=1)
    text = prompts[0][-1]["content"][0]["text"]
    assert "Query constraints:" in text
    assert '"requires_current": true' in text
    assert result["route_decision"]["query_constraints"]["requires_current"]
    assert result["raw"]["answer_debug"]["actual_prompt_text"] == text


def test_fallback_keeps_authoritative_scope_and_derived_tail_request_local(fallback_session):
    given = constraints.parse_query_constraints("他最近做了什么？", 610)
    before = deepcopy(given)
    result = _fallback(fallback_session, question="他最近做了什么？", query_constraints=given)
    actual = result["route_decision"]["query_constraints"]
    assert actual["scope"] == "recent"
    assert actual["requires_current_tail"]
    assert result["raw"]["cache_context"]["query_constraints"]["requires_current_tail"]
    assert given == before


def test_relative_numeric_scope_is_preserved_through_actual_fallback(fallback_session):
    given = constraints.parse_query_constraints("What happened in the past 2 minutes?", 610)
    result = _fallback(fallback_session, question="What happened in the past 2 minutes?", query_constraints=given)
    assert result["route_decision"]["query_constraints"]["time_range"] == [490, 610]
    assert result["raw"]["cache_context"]["query_constraints"]["time_range"] == [490, 610]


def test_st_only_resolver_uses_the_same_observed_cutoff_as_scope(fallback_session, monkeypatch):
    fallback_session.context["state"].update(mcur_ready=False, current_time=0, window_end_time=0)
    seen = []
    class Resolver:
        def resolve(self, question, interaction_cache, latest_seconds=None, **kwargs):
            seen.append(latest_seconds)
            return {"cache_hit": False, "is_followup": False, "confidence": 0}
    monkeypatch.setattr(engine, "CoreferenceResolver", Resolver)
    _fallback(fallback_session, question="What happened after that?", use_interaction_cache=True, cache_mode="read_only")
    assert seen == [601]


@pytest.mark.parametrize("provided,expected", [({}, {"early", "late"}), ({"time_range": [10, 20]}, {"late"})])
def test_real_short_term_search_keeps_supplied_scope(tmp_path, provided, expected):
    from online_short_term.mst_store import MSTStore
    from online_short_term.mst_retriever import MSTRetriever
    store = MSTStore(tmp_path / "science")
    store.save_events([{"event_id": name, "start_time": stamp, "end_time": stamp + 1,
                        "event_caption_refined": "A person studies an astrolabe.", "status": "refined"}
                       for name, stamp in [("early", 5), ("late", 15)]])
    before = deepcopy(provided)
    hits = MSTRetriever(store).search("At900 seconds, what happened to the astrolabe?", top_k=3,
                                     cache_context={"query_constraints": provided})
    assert {hit["event_id"] for hit in hits} == expected
    assert provided == before
