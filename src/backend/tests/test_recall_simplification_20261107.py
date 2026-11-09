"""Recall survives auxiliary heuristics, schema differences and phase changes."""
from copy import deepcopy

import pytest

from online_evidence_content import comparison_observation_matches, prompt_evidence
from online_query.evidence_packer import EvidencePacker
from online_query.evidence_validation import validate_evidence
from online_query.memory_fusion import MemoryFusion
from online_query_constraints import parse_query_constraints, prompt_query_constraints
from test_packer_inspection_coverage import event, sent_times


def pack(question, items, cap=4, context=None):
    return EvidencePacker().pack(question, {"query_constraints": parse_query_constraints(question),
        "use_image_evidence": True, "max_image_evidence": cap, "final_evidence_k": len(items)},
        {"final_evidence": items}, context)


def test_summary_retrieval_keeps_the_strongest_narrative_anchors(tmp_path, monkeypatch):
    from retrieval_source_helpers import make_memory, caption
    def timestamp(seconds):
        return f"{seconds//3600:02d}{seconds//60%60:02d}{seconds%60:02d}00"
    rows = [caption(str(i), timestamp(i*30), timestamp((i+1)*30), text) for i, text in enumerate([
        "A person sits down.", "The person wears glasses.", "The glasses move to the head.",
        "The person folds the fabric around their shoulders and brings the two sides together.",
        "The person closes the fastener and smooths the material around the neck.",
        "The person removes glasses.", "The person wears a hat.", "The person leaves."])]
    memory = make_memory(tmp_path, {"30sec": rows})
    memory.set_retrieval_top_k(episodic=3, semantic=3, visual=1)
    monkeypatch.setenv("EM2MEM_LLM_EVENT_SELECTOR_ENABLED", "0")
    monkeypatch.setattr(memory, "_project_episodic_candidates_to_30s",
        lambda *a, **k: {str(i): score for i, score in enumerate([.1, .3, .2, .99, .98, .4, .25, .05])})
    result = memory.answer("Summarize the clothing changes", until_time=100040000,
        query_constraints={"summary": True}, generate_answer=False, use_image_evidence=False)
    assert {"3", "4"} <= set(result.selected_doc_ids)
    assert len(result.selected_doc_ids) <= 3


def test_summary_fusion_does_not_reselect_away_relevant_narratives():
    rows = [{"memory_id": str(i), "start_time": i*30, "end_time": i*30+30,
             "caption": caption, "score": score} for i, caption, score in [
        (0, "A person wears glasses.", .1),
        (1, "Fabric moves from above the head to around the shoulders.", .99),
        (2, "The material is fastened around the torso.", .98),
        (3, "Glasses are removed.", .3), (4, "A hat is worn.", .2)]]
    out = MemoryFusion().fuse({"text_results": rows},
        {"question": "Summarize the clothing changes", "query_constraints": {"summary": True}},
        {"final_evidence_k": 3}, "long_term_summary")
    assert {"1", "2"} <= {row["evidence_id"] for row in out["final_evidence"]}


def test_summary_tail_is_newest_available_not_just_newest_closed_short_term():
    memories = {"text_results": [{"memory_id": str(i), "score": .9, "start_time": i*30,
                    "end_time": (i+1)*30, "caption": "A recorded earlier event."} for i in range(4)],
                "short_term_results": [{"event_id": "closed", "start_time": 120, "end_time": 130,
                    "event_caption_refined": "The person sits down.", "score": .9}],
                "current_results": [{"current_context": {"state": {"window_start_time": 111,
                    "window_end_time": 140, "current_time": 140}}}]}
    out = MemoryFusion().fuse(memories, {}, {"final_evidence_k": 3}, "long_term_summary")
    assert any(row["source_memory"] == "M_cur" and row["end_time"] == 140 for row in out["final_evidence"])
    assert sum(row["source_memory"] == "M_lt" for row in out["final_evidence"]) == 2


@pytest.mark.parametrize("question", [
    "之前那辆车，到现在还能看见吗？", "较早拿走的那个物品，到现在是否还在桌上？",
    "The earlier vehicle, by now can we still identify it?",
])
def test_current_endpoint_survives_a_history_clause(question):
    scope = parse_query_constraints(question, 100)
    assert scope["requires_current"] and scope["requires_history"]


@pytest.mark.parametrize("question", [
    "从前面拿着雨伞到后来放下，手上的状态有什么变化？",
    "从前面穿短袖到后来套上外套，他的衣着是怎么变化的？",
    "前后箱子的状态有什么变化？",
])
def test_phase_comparisons_do_not_require_one_fixed_wording(question):
    scope = parse_query_constraints(question)
    assert scope["requirements"]["comparison"] and not scope["time_explicit"]


@pytest.mark.parametrize("question", [
    "前面刚能认出那本书的时候，他是怎么拿着的？",
    "最初能够看到雨伞的时候，它在哪？",
    "When the device was first clearly visible, how was it held?",
])
def test_initial_discovery_is_a_soft_earlier_preference(question):
    scope = parse_query_constraints(question)
    assert scope["temporal_preference"] == "earlier"
    assert not scope["time_explicit"] and scope["time_range"] is None


def test_equivalent_prose_is_not_a_lower_inspection_class():
    earlier = event("prose", [10], ["The person lifts the blue mug with both hands."])
    later = event("structured", [50], ["The person sits at the table."])
    later["metadata"]["main_actions"] = [{"actor": "person", "action": "touches", "objects": ["mug"],
                                          "time_span": [50, 50]}]
    assert sent_times(pack("Show the person lifting the mug", [earlier, later], cap=1)) == [10]


def test_reference_end_is_visible_even_without_an_annotation():
    history = event("handling", [10, 11, 12, 13, 14], ["The person holds a blue mug."]*5)
    current = {"evidence_id": "current", "source_memory": "M_cur", "start_time": 15, "end_time": 20,
        "keyframe_paths": ["current/frames/cur_kf_000020000.jpg"], "caption": "Unrefined current window"}
    context = {"binding_valid": True, "is_followup": True, "confidence": .9, "temporal_relation": "related",
        "referenced_time_ranges": [{"start": 10, "end": 20, "source": "answer_cited_sent_frames"}]}
    assert 20 in sent_times(pack("How is the person holding the mug?", [history, current], cap=3, context=context))


def test_comparison_later_side_need_not_repeat_the_initial_conditions():
    before = event("before", [10, 11, 12, 13], ["The person wears sunglasses and holds an umbrella."]*4)
    after = event("after", [70, 80], ["The person faces the camera with empty hands."]*2)
    question = "Compare earlier wearing sunglasses and holding an umbrella with later. What changed?"
    times = sent_times(pack(question, [before, after], cap=3))
    assert any(t <= 13 for t in times) and 80 in times


def test_comparison_supplement_recalls_a_negative_end_state():
    before = event("before", [10], ["The person wears sunglasses and holds an umbrella."])
    after = event("after", [80], ["The person no longer wears sunglasses."])
    hits = comparison_observation_matches("Compare simultaneously wearing sunglasses and holding an umbrella with later",
        [before, after], {"requirements": {"comparison": True, "simultaneous": True}})
    assert {doc for doc, _ in hits} == {"before", "after"}


def test_comparison_supplement_survives_merge_source_budget_and_final_fusion():
    from online_query.query_engine import _merge_supplemental_events, LoadedQueryEngine
    before = event("before", [10], ["The person wears sunglasses and holds an umbrella."])
    after = event("after", [80], ["The person no longer wears sunglasses."])
    question = "Compare simultaneously wearing sunglasses and holding an umbrella with later"
    constraints = {"requirements": {"comparison": True, "simultaneous": True}}
    catalog = [{**row, "doc_id": row["evidence_id"]} for row in (before, after)]
    selected = [{"doc_id": "before", "score": .9, "text": "A person holds an umbrella.",
                 "start_time": 10, "end_time": 10}]
    _merge_supplemental_events(question=question, selected_events=selected, selected_doc_ids=["before"],
        event_scores={}, catalog=catalog, hits=comparison_observation_matches(question, catalog, constraints),
        constraints=constraints, summarize=lambda doc: [deepcopy(next(r for r in catalog if r["doc_id"] == doc))])
    engine = LoadedQueryEngine.__new__(LoadedQueryEngine)
    engine.evidence_by_doc_id = {row["doc_id"]: row for row in catalog}
    results = engine._build_text_results(selected, [])
    out = MemoryFusion().fuse({"text_results": results}, {"query_constraints": constraints},
        {"final_evidence_k": 2, "retrieval_plan": {"M_lt": {"candidate_budget": 1}}}, "temporal_reasoning")
    assert {r["evidence_id"] for r in out["final_evidence"]} == {"before", "after"}
    later = next(r for r in out["final_evidence"] if r["evidence_id"] == "after")
    assert later["metadata"]["requirement_support"]["confirmed"] is False
    assert later["score_policy"] == "unscored_supplemental_observation"


def test_summary_image_coverage_preserves_top_historical_event():
    opening = event("opening", [0], ["The person sits wearing sunglasses."])
    middle = event("transition", [50], ["The person pulls material around the shoulders."])
    ending = event("ending", [100], ["The person sits wearing sunglasses."])
    for row, score in zip((opening, middle, ending), (.1, .9, .3)):
        row.update(source_memory="M_lt", final_score=score)
    latest = event("live", [110], ["The person sits wearing sunglasses."])
    latest["final_score"] = 1.0
    times = sent_times(pack("Summarize the clothing changes", [opening, middle, ending, latest], cap=3))
    assert times == [0, 50, 110]


def test_parent_process_survives_a_small_soft_budget_once_and_in_order():
    process = ["The worker unfolds the material around the shoulders.",
               "They fasten the opening beneath the chin.", "Then they sit down and remove the glasses."]
    parent = {"doc_id": "shared", "start_time": 0, "end_time": 180,
              "caption": " ".join(process + [process[-1]]*40)}
    rows = [{"evidence_id": str(i), "caption": "A person is seated.", "parent_context": [parent]} for i in range(2)]
    original = deepcopy(rows)
    out, stats = prompt_evidence("Summarize the clothing changes", rows, max_tokens=120)
    parents = [p for row in out for p in row.get("parent_context", [])]
    assert len(parents) == 1 and parents[0]["caption"] == " ".join(process)
    assert rows == original
    assert stats["budget_exceeded"] == (stats["estimated_tokens"] > 120)


def test_parser_misses_are_not_injected_as_negative_semantic_facts():
    hints = prompt_query_constraints({"time_ranges": [[0, 10]], "time_explicit": True,
        "requires_current": False, "requirements": {"earliest": False, "comparison": True}})
    assert hints["requested_time_ranges"] == [[0, 10]]
    assert hints["retrieval_hints"]["comparison"] is True
    assert "earliest" not in hints["retrieval_hints"] and "requires_current" not in hints["retrieval_hints"]
    check = validate_evidence("What happened?", [{"caption": "A person is seated."}], [], {})
    assert check["mechanical_checks_passed"] is True and check["sufficient"] is None
