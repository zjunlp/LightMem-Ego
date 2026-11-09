"""Regression tests for supplemental evidence without model or network calls."""
from copy import deepcopy

import pytest

import online_evidence_content as content
from online_query import query_engine as engine


QUESTION = "之前有没有口罩和墨镜都戴着的时候？那时跟现在有什么不同？"
CONSTRAINTS = {"requirements": {"simultaneous": True, "comparison": True}}


def event(name, observations):
    return {"doc_id": name, "start_time": 0, "end_time": 20,
            "visual_observations": observations}


def observation(text, timestamp=10):
    return {"timestamp": timestamp, "description": text}


@pytest.mark.parametrize("observations", [
    [observation("A masked person sits upright.")],
    [observation("A person wears a mask."), observation("A person wears sunglasses.", 12)],
])
def test_comparison_admits_partial_phase_without_claiming_simultaneous_conditions(observations):
    source = event("partial", observations)
    assert content.comparison_observation_matches(QUESTION, [source], CONSTRAINTS)
    support = content.supplemental_requirement_support(QUESTION, source, CONSTRAINTS)
    assert support["observation_matched"] is True
    assert support["conditions_co_mentioned"] is False and support["confirmed"] is False
    # The same evidence cannot prove the positive conjunction outside comparison.
    assert not content.supplemental_requirement_support(QUESTION, source,
        {"requirements": {"simultaneous": True}})["observation_matched"]


def test_comparison_keeps_same_observation_same_actor_positive():
    hit = event("good", [observation("A person wears a white mask and dark sunglasses.")])
    assert [name for name, _ in content.comparison_observation_matches(QUESTION, [hit], CONSTRAINTS)] == ["good"]


def merge(selected, ids, scores, catalog, hits, constraints=CONSTRAINTS):
    source = {row["doc_id"]: row for row in catalog}
    return engine._merge_supplemental_events(
        question=QUESTION, selected_events=selected, selected_doc_ids=ids,
        event_scores=scores, catalog=catalog, hits=hits, constraints=constraints,
        summarize=lambda doc_id: [deepcopy(source[doc_id])],
    )


def test_existing_selected_positive_gains_support_without_duplicate_or_score_override():
    source = event("good", [observation("A person wears a mask and sunglasses.")])
    selected = [{"doc_id": "good", "score": .06, "raw_score": .064,
                 "score_policy": "absolute_projected_score_s_over_1_plus_s"}]
    ids = ["good"]
    added = merge(selected, ids, {}, [source], [("good", .95)])
    assert added == [] and ids == ["good"] and len(selected) == 1
    assert selected[0]["score"] == .06 and selected[0]["raw_score"] == .064
    assert selected[0]["supplement_match_score"] == .95
    assert selected[0]["requirement_support"]["confirmed"] is False
    assert selected[0]["requirement_support"]["conditions_co_mentioned"] is True


def test_new_supplement_uses_original_projected_score_if_selector_omitted_it():
    source = event("good", [observation("A person wears a mask and sunglasses.")])
    selected, ids = [], []
    scores = {"good": {"score": .12, "raw_score": .136,
                       "score_policy": "absolute_projected_score_s_over_1_plus_s"}}
    merge(selected, ids, scores, [source], [("good", .95)])
    assert selected[0]["score"] == .12 and selected[0]["raw_score"] == .136
    assert selected[0]["supplement_match_score"] == .95
    assert selected[0]["score_provenance"] == "episodic_projection"


def test_unscored_supplement_remains_unscored_and_retains_provenance_through_text_results():
    source = event("good", [observation("A person wears a mask and sunglasses.")])
    selected, ids = [], []
    merge(selected, ids, {}, [source], [("good", .95)])
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.evidence_by_doc_id = {}
    result = loaded._build_text_results(selected, [])[0]
    assert result["score"] is None and result["raw_score"] is None
    assert result["score_policy"] == "unscored_supplemental_observation"
    assert result["score_provenance"] == "not_retrieved"
    assert result["supplement_match_score"] == .95
    assert result["requirement_support"]["confirmed"] is False
    assert result["requirement_support"]["support_level"] == "text_candidate"


def test_comparison_observation_support_does_not_claim_whole_question_complete():
    source = event("one", [observation("A person wears a mask and sunglasses.")])
    selected, ids = [], []
    merge(selected, ids, {}, [source], [("one", .8)], {"requirements": {"comparison": True}})
    assert selected[0]["requirement_support"]["observation_matched"] is True
    assert selected[0]["requirement_support"]["confirmed"] is False


def test_supplied_score_policy_is_not_relabelled_as_projected():
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.evidence_by_doc_id = {}
    result = loaded._build_text_results([{"doc_id": "x", "score": .25, "raw_score": .25,
                                         "score_policy": "external_original"}], [])[0]
    assert result["score_policy"] == "external_original"


def test_mask_only_posture_state_is_not_simultaneous_positive():
    source = {"doc_id": "posture", "state_changes": [{"entity": "masked person",
              "attribute": "posture", "before": "leaning forward", "after": "sitting upright",
              "time_span": [10, 12]}]}
    support = content.supplemental_requirement_support(QUESTION, source, CONSTRAINTS)
    assert support["confirmed"] is False and not support["conditions_co_mentioned"]


def test_recovered_unselected_candidate_scores_are_available_without_rule_scoring(tmp_path, monkeypatch):
    from retrieval_source_helpers import make_memory, LATE_CUTOFF
    memory = make_memory(tmp_path)
    monkeypatch.setattr(memory, "_select_top_events_with_llm", lambda **kwargs: (["early"], "test selector"))
    result = memory.answer("target", until_time=LATE_CUTOFF,
                           use_image_evidence=False, generate_answer=False)
    assert result.selected_doc_ids == ["early"]
    assert "late" in result.event_scores
    original = result.event_scores["late"]
    assert original["score"] == pytest.approx(original["raw_score"] / (1 + original["raw_score"]))
    assert original["score_provenance"] == "episodic_projection"


def test_invalid_explicit_binding_cannot_suppress_recent_open_tail():
    route = {"query_constraints": {}}
    sufficiency = {"requires_current_tail": True, "current_tail_coverage": {"tail_range": [91, 100]}}
    engine._apply_current_tail_requirement(route, sufficiency, {
        "is_followup": True, "binding_valid": False, "confidence": .9,
        "temporal_relation": "same_time", "referenced_time_ranges": [{"start": 10, "end": 20}],
    })
    assert sufficiency["requires_current_tail"] is True
    assert route["query_constraints"]["requires_current_tail"] is True


def test_invalid_explicit_binding_cannot_boost_cached_segment():
    loaded = object.__new__(engine.LoadedQueryEngine)
    boost, reasons = loaded._cache_boost_for_segment("old", {"text": {"start_time": "00001000", "end_time": "00002000"}}, {
        "is_followup": True, "binding_valid": False, "confidence": .9,
        "referenced_segment_ids": ["old"], "referenced_time_ranges": [{"start": 10, "end": 20}],
    })
    assert boost == 0 and reasons == []


def test_unscored_text_survives_hybrid_fusion_without_invented_retrieval_score():
    from online_query.memory_fusion import MemoryFusion
    source = event("good", [observation("A person wears a mask and sunglasses.")])
    selected, ids = [], []
    merge(selected, ids, {}, [source], [("good", .95)])
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.evidence_by_doc_id = {}
    text = loaded._build_text_results(selected, [])
    hybrid = loaded._fuse_results(text, [], final_evidence_k=4)
    assert hybrid[0]["text_score"] == 0
    assert hybrid[0]["text"]["score"] is None
    assert hybrid[0]["text"]["supplement_match_score"] == .95
    result = MemoryFusion().fuse({"text_results": text, "fused_results": hybrid},
        {"query_constraints": CONSTRAINTS}, {"final_evidence_k": 2}, "temporal_reasoning")
    assert len(result["final_evidence"]) == 1
    evidence = result["final_evidence"][0]
    assert evidence["retrieval_score"] == 0
    assert evidence["metadata"]["supplement_match_score"] == .95
    assert evidence["metadata"]["requirement_support"]["confirmed"] is False
    assert evidence["metadata"]["requirement_support"]["conditions_co_mentioned"] is True


def test_checked_archived_observation_is_available_to_text_and_image_packing():
    source = {"doc_id": "good", "source_micro_events": [{"event_id": "linked",
              "visual_observations": [observation("A person wears a mask and sunglasses.", 11)]}]}
    selected = [{"doc_id": "good", "score": .06, "visual_observations": []}]
    original_source = deepcopy(source)
    merge(selected, ["good"], {}, [source], [("good", .95)])
    loaded = object.__new__(engine.LoadedQueryEngine)
    loaded.evidence_by_doc_id = {"good": original_source}
    text = loaded._build_text_results(selected, [])[0]
    assert text["visual_observations"] == original_source["source_micro_events"][0]["visual_observations"]
    assert source == original_source
