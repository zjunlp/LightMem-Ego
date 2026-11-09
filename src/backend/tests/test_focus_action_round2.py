"""An object binding identifies the topic, not an unobserved action."""
from types import SimpleNamespace

import pytest

from online_query.coreference_resolver import CoreferenceResolver
from online_query.interaction_cache import InteractionCache


def turn(name, answer, second):
    return {"turn_id": name, "question": "What is he holding?", "answer": answer,
            "anchor_status": "confirmed", "time_ranges": [{"start": second, "end": second}],
            "entities": [{"name": "gray backpack"}],
            "focus_entities": [{"name": "gray backpack"}],
            "focus_evidence_refs": [{"evidence_id": name, "source_memory": "M_st",
                                     "start_time": second, "end_time": second}],
            "evidence_frames": [{"path": name + ".jpg", "timestamp": second}]}


def resolve(question, turns):
    cache = SimpleNamespace(latest_context=lambda: {
        "recent_interactions": turns, "latest_interaction": turns[-1]})
    return CoreferenceResolver().resolve(question, cache)


@pytest.mark.parametrize("question", [
    "当它掉在地上的时候，他当时穿什么？",
    "When he dropped it, what was he wearing at that time?",
    "你说的灰色背包被放下的时候，他当时穿什么？",
])
def test_bound_topic_does_not_prove_an_unobserved_action(question):
    result = resolve(question, [turn("holding", "He holds a gray backpack.", 20)])
    assert not result.get("referenced_time_ranges")
    assert result.get("temporal_relation") != "same_time"


def test_bound_topic_can_find_an_earlier_observed_action():
    result = resolve("When he dropped it, what was he wearing at that time?", [
        turn("dropped", "He dropped the gray backpack.", 10),
        turn("holding", "He holds a gray backpack.", 20)])
    assert result["matched_turn_id"] == "dropped"
    assert [(r["start"], r["end"]) for r in result["referenced_time_ranges"]] == [(10, 10)]
    assert [f["path"] for f in result["referenced_evidence_frames"]] == ["dropped.jpg"]


def test_bound_topic_can_find_a_translated_earlier_action():
    result = resolve("当它掉在地上的时候，他当时穿什么？", [
        turn("dropped", "灰色背包掉在地上。", 10),
        turn("holding", "他拿着灰色背包。", 20)])
    assert result["matched_turn_id"] == "dropped"
    assert [(r["start"], r["end"]) for r in result["referenced_time_ranges"]] == [(10, 10)]


@pytest.mark.parametrize("other_object", ["red backpack", "phone"])
def test_earlier_action_on_a_different_object_does_not_match_bound_topic(other_object):
    other = turn("other", f"He dropped a {other_object}.", 10)
    other["focus_entities"] = [{"name": other_object}]
    other["entities"] = [{"name": other_object}]
    result = resolve("When he dropped it, what was he wearing at that time?", [
        other, turn("holding", "He holds a gray backpack.", 20)])
    assert not result.get("referenced_time_ranges")


@pytest.mark.parametrize("answer", [
    "He did not drop the gray backpack.",
    "He might have dropped the gray backpack.",
])
def test_bound_topic_does_not_bypass_negation_or_uncertainty(answer):
    result = resolve("When he dropped it, what was he wearing at that time?", [turn("uncertain", answer, 20)])
    assert not result.get("referenced_time_ranges")


@pytest.mark.parametrize("question,answer,followup", [
    ("Can you confirm he dropped the gray backpack?", "He holds the gray backpack.",
     "When he dropped it, what was he wearing at that time?"),
    ("他有没有把灰色背包掉在地上？", "他拿着灰色背包。",
     "当它掉在地上的时候，他当时穿什么？"),
])
def test_question_action_is_not_an_observed_answer_action(question, answer, followup):
    candidate = turn("holding", answer, 20)
    candidate["question"] = question
    result = resolve(followup, [candidate])
    assert not result.get("referenced_time_ranges")


@pytest.mark.parametrize("question", ["它怎么拿着？", "那个东西是什么颜色？", "那个东西，当时怎么拿着？"])
def test_generic_bound_topic_still_uses_previous_observation(question):
    result = resolve(question, [turn("holding", "He holds a gray backpack.", 20)])
    assert result["matched_turn_id"] == "holding"
    assert [(r["start"], r["end"]) for r in result["referenced_time_ranges"]] == [(20, 20)]


def test_explicit_same_event_multiplicity_stays_ambiguous_after_resolution(tmp_path):
    cache = InteractionCache("two", tmp_path, ttl_seconds=0)
    cache.update_from_query_result("What does he hold?", "What does he hold?", {
        "answer": "He holds a gray backpack and another backpack.",
        "selected_evidence": [{"evidence_id": "two", "source_memory": "M_st", "start_time": 10,
                               "end_time": 12, "metadata": {
                                   "visual_objects": [{"name": "gray backpack"}],
                                   "main_actions": [{"action": "holds", "objects": ["backpack"]}]}}]})
    result = CoreferenceResolver().resolve("那个东西是什么颜色？", cache)
    assert result.get("reference_ambiguous")
    assert not result.get("referenced_time_ranges")
