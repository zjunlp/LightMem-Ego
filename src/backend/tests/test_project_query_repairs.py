"""Long-session, bounded-work and native-state regressions, without services."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest
from PIL import Image

from online_evidence_content import prompt_evidence, token_estimate
from online_evidence_coverage import coverage_features, select_coverage
from online_query.interaction_cache import InteractionCache
from online_query.evidence_packer import EvidencePacker


def _result():
    return {"status": "ok", "answer": "The brass astrolabe rests on the desk.",
            "selected_evidence": [{"evidence_id": "desk", "source_memory": "M_st",
                "start_time": 0, "end_time": 10,
                "metadata": {"visual_objects": [{"name": "brass astrolabe"}]}}]}


def _append(cache, number):
    return cache.update_from_query_result(f"Where is the object? Turn {number}", "", _result())["turn_id"]


def test_cache_ids_remain_unique_after_retention_reload_and_expiration(tmp_path):
    cache = InteractionCache("session", tmp_path, ttl_seconds=0)
    ids = [_append(cache, i) for i in range(24)]
    reloaded = InteractionCache("session", tmp_path, ttl_seconds=0)
    ids.append(_append(reloaded, 24))
    assert len(set(ids)) == len(ids)
    # Emulate TTL cleanup while retaining the on-disk allocation cursor.
    reloaded.data["recent_interactions"] = []
    reloaded.flush()
    ids.append(_append(reloaded, 25))
    assert len(set(ids)) == len(ids)


def test_legacy_cache_cursor_uses_highest_id_before_trimming(tmp_path):
    cache = InteractionCache("legacy", tmp_path, max_interactions=2, ttl_seconds=0)
    payload = cache.data
    payload.pop("next_turn_number", None)
    payload["recent_interactions"] = [{"turn_id": "turn_000099"}, {"turn_id": "turn_000002"}, {"turn_id": "turn_000003"}]
    cache.cache_path.write_text(json.dumps(payload))
    fresh = InteractionCache("legacy", tmp_path, max_interactions=2, ttl_seconds=0)
    assert _append(fresh, 1) == "turn_000100"


def test_two_cache_instances_allocate_unique_ids_and_bound_support(tmp_path):
    caches = [InteractionCache("s", tmp_path, max_interactions=3, ttl_seconds=0) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(lambda i: _append(caches[i % 2], i), range(12)))
    assert len(set(ids)) == len(ids)
    saved = json.loads(caches[0].cache_path.read_text())
    assert len(saved["recent_interactions"]) == 3
    assert all(len(row["supporting_turn_ids"]) <= 3 for row in saved["hot_entities"])
    assert all(len(row["source_turn_ids"]) <= 3 for row in saved["hot_memories"])


@pytest.mark.parametrize("enabled,total,per_event", [(False, 6, 3), (True, 0, 3), (True, 6, 0)])
def test_unsent_images_are_not_decoded_but_metadata_survives(tmp_path, monkeypatch, enabled, total, per_event):
    monkeypatch.setenv("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", str(per_event))
    for name in ("first.png", "last.png"):
        Image.new("RGB", (8, 8), "blue").save(tmp_path / name)
    event = {"evidence_id": "shelf", "source_memory": "M_st", "start_time": 0, "end_time": 8,
             "caption": "A box rests on the shelf.", "evidence_frames": [
                 {"path": "first.png", "timestamp": 0}, {"path": "last.png", "timestamp": 8}]}
    packer = EvidencePacker(tmp_path)
    calls = []
    monkeypatch.setattr(packer, "_content_signature", lambda path: calls.append(path))
    packed = packer.pack("Summarize the scene", {"use_image_evidence": enabled,
        "max_image_evidence": total, "query_constraints": {"summary": True}}, {"final_evidence": [event]})
    assert calls == []
    assert packed["selected_image_paths_for_mllm"] == []
    assert packed["selected_evidence"][0]["caption"] == event["caption"]
    assert packed["selected_evidence_frames"]


def _state(entity="storage crate", **extra):
    return {"entity": entity, "actor": "worker", "before": "closed", "after": "open",
            "timestamp": 42, **extra}


@pytest.mark.parametrize("entity", ["storage crate", "music stand", "折叠屏风"])
def test_unknown_native_entities_receive_bounded_transition_features(entity):
    row = _state(entity)
    original = deepcopy(row)
    features = coverage_features("Summarize all state changes", {"state_changes": [row]})
    assert any(f.startswith("transition:generic:") for f in features)
    assert row == original
    assert len(features) == 1


@pytest.mark.parametrize("extra", [{"certainty": "uncertain"}, {"polarity": "negative"},
    {"negated": True}, {"confidence_label": "unknown"}, {"description": "The storage crate possibly opens."},
    {"description": "The storage crate may now be open."},
    {"description": "It is unclear whether the storage crate is open."},
    {"after": "possibly open"}, {"after": {"state": "open", "certainty": "uncertain"}}])
def test_generic_transitions_do_not_promote_negated_or_uncertain_occurrences(extra):
    features = coverage_features("Summarize all state changes", {"state_changes": [_state(**extra)]})
    assert not any(f.startswith("transition:generic:") for f in features)


@pytest.mark.parametrize("extra", [{"before": "open"}, {"before": None}, {"after": None},
    {"entity": ""}, {"timestamp": None}, {"before": " OPEN ", "after": "open"}])
def test_generic_transition_requires_complete_distinct_timed_native_states(extra):
    assert not coverage_features("Summarize all state changes", {"state_changes": [_state(**extra)]})


def test_generic_actor_binding_and_unrelated_target_are_preserved():
    def features(actor, question="Summarize all state changes"):
        return coverage_features(question, {"state_changes": [_state(actor=actor)]})
    assert features("front-left person") != features("front-right person")
    assert not features("front-right person", "Summarize the front-left person's changes")
    assert not features("worker", "Summarize changes to the bottle")


def test_unbound_native_actors_are_not_assumed_to_share_identity():
    rows = [_state(actor=None, description=f"The {side} person opens the storage crate.")
            for side in ("front-left", "front-right")]
    features = [coverage_features("Summarize all state changes", {"state_changes": [row]}) for row in rows]
    assert features[0] and features[1] and features[0] != features[1]


def test_native_scalar_states_keep_zero_and_false():
    for before, after in ((0, 1), (False, True)):
        assert coverage_features("Summarize all state changes", {"state_changes": [_state(before=before, after=after)]})


def test_unknown_transition_competes_with_redundant_events_within_existing_budget():
    events = [{"evidence_id": str(i), "start_time": t, "caption": "A quiet room."}
              for i, t in enumerate((0, 50, 51, 100))]
    events[2]["state_changes"] = [_state(timestamp=51)]
    selected = select_coverage("Summarize all state changes", events, 3)
    assert {e["evidence_id"] for e in selected} == {"0", "2", "3"}


def test_low_text_target_reports_base_overflow_without_losing_event_identity():
    events = [{"evidence_id": f"micro_event_{i:06d}", "source_memory": "M_st",
               "start_time": i * 10, "end_time": i * 10 + 10, "caption": "A quiet room."} for i in range(16)]
    packed, stats = prompt_evidence("Summarize", events, max_tokens=256)
    assert [e["evidence_id"] for e in packed] == [e["evidence_id"] for e in events]
    assert stats["base_tokens"] == token_estimate(packed)
    assert stats["budget_exceeded"] is True
    assert stats["budget_shortfall_tokens"] == stats["estimated_tokens"] - 256
    assert stats["budget_scope"] == "evidence_text_estimate"


def test_sufficient_text_target_keeps_existing_content_and_reports_no_overflow():
    packed, stats = prompt_evidence("Summarize", [{"evidence_id": "one", "caption": "A quiet room."}])
    assert packed[0]["caption"] == "A quiet room."
    assert stats["budget_exceeded"] is False
    assert stats["budget_shortfall_tokens"] == 0


def test_loaded_and_short_term_prompt_report_the_same_soft_budget(monkeypatch):
    from online_query.query_engine import LoadedQueryEngine, _build_short_term_llm_prompt
    monkeypatch.setenv("EM2MEM_PROMPT_EVIDENCE_TOKENS", "256")
    events = [{"evidence_id": f"micro_event_{i:06d}", "source_memory": "M_st",
               "start_time": i * 10, "end_time": i * 10 + 10, "caption": "A quiet room."} for i in range(16)]
    loaded_stats, fallback_stats = {}, {}
    engine = LoadedQueryEngine.__new__(LoadedQueryEngine)
    engine._build_visual_answer_prompt("Summarize", "text_only", [], [], [],
        selected_evidence=events, query_constraints={}, evidence_budget=loaded_stats)
    _build_short_term_llm_prompt("Summarize", [], {"query_constraints": {}},
        selected_evidence=events, evidence_budget=fallback_stats)
    assert loaded_stats == fallback_stats
    assert loaded_stats["base_tokens"] > 256 and loaded_stats["budget_exceeded"]


def test_prompt_compression_obeys_authoritative_summary_flag():
    event = {"evidence_id": "one", "visual_observations": [{"description": "The shelf is visible."}],
        "parent_context": [{"doc_id": "coarse", "caption": "A quiet room. " * 300}]}
    suppressed, _ = prompt_evidence("Summarize", [event], max_tokens=600, constraints={"summary": False})
    enabled, _ = prompt_evidence("Describe the scene", [event], max_tokens=600, constraints={"summary": True})
    assert not suppressed[0].get("parent_context")
    assert enabled[0].get("parent_context")
