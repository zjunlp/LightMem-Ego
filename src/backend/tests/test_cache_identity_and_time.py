"""Topic inheritance and observation times must not bind a new historical search."""
from types import SimpleNamespace

import pytest

from online_query.cache_schema import has_entity_binding, has_valid_binding, resolved_search_question
from online_query.coreference_resolver import CoreferenceResolver
from online_query.interaction_cache import InteractionCache


def previous_turn(**overrides):
    return {
        "turn_id": "previous", "question": "前排右侧乘客在做什么？",
        "answer": "前排右侧乘客低头拿着杯子。", "anchor_status": "confirmed",
        "entities": [{"name": "front-right passenger (wearing a black jacket)", "type": "person",
                      "last_state": "looks down and holds a cup"}, {"name": "cup", "type": "object"}],
        "focus_entities": [{"name": "blue cup", "type": "object", "last_state": "held near the chest"}],
        "time_ranges": [{"start": 70, "end": 80}], "retrieved_memory_ids": ["old-event"],
        "retrieved_segment_ids": ["old-segment"], "visual_ids": ["old-visual"],
        "evidence_frames": [{"path": "old.jpg", "timestamp": 75}], **overrides,
    }


def resolve(question, turn=None):
    turn = previous_turn() if turn is None else turn
    cache = SimpleNamespace(latest_context=lambda: {"latest_interaction": turn, "recent_interactions": [turn]})
    return CoreferenceResolver().resolve(question, cache, latest_seconds=200)


@pytest.mark.parametrize("question", [
    "他什么时候拿起过杯子？", "她何时打开过车窗？", "他有没有拿过杯子？",
    "她是否曾经戴过帽子？", "他早先在做什么？",
    "When did he hold the cup?", "At what time was she holding a cup?",
    "Has he ever opened the window?",
])
def test_fresh_historical_search_inherits_identity_without_previous_times_or_states(question):
    context = resolve(question)
    assert context["is_followup"] and has_entity_binding(context)
    assert context["binding_valid"] is False and not has_valid_binding(context)
    assert context["referenced_entities"] == [{"name": "front-right passenger", "type": "person"}]
    for field in ["referenced_time_ranges", "referenced_memory_ids", "referenced_segment_ids", "referenced_visual_ids", "referenced_evidence_frames"]:
        assert context[field] == []
    rewritten = resolved_search_question(question, context)
    assert question in rewritten and "front-right passenger" in rewritten
    assert "black jacket" not in rewritten and "70" not in rewritten and "80" not in rewritten
    assert context["entity_turn_id"] == "previous"


@pytest.mark.parametrize("question", ["它什么时候被放到桌子上？", "When was it placed on the table?"])
def test_object_identity_survives_without_its_old_held_state(question):
    context = resolve(question)
    assert has_entity_binding(context) and not has_valid_binding(context)
    assert context["referenced_entities"] == [{"name": "blue cup", "type": "object"}]
    assert "held near the chest" not in context["resolved_question"]
    assert not context["referenced_evidence_frames"]


def test_two_distinct_subjects_do_not_become_a_random_identity():
    context = resolve("他什么时候拿过杯子？", previous_turn(entities=[
        {"name": "front-right passenger", "type": "person"}, {"name": "rear-left passenger", "type": "person"}]))
    assert context["reference_ambiguous"] and not has_entity_binding(context)
    assert resolved_search_question("他什么时候拿过杯子？", context) == "他什么时候拿过杯子？"
    assert not context["referenced_time_ranges"]


@pytest.mark.parametrize("question", ["这个视频什么时候开始的？", "这个门什么时候打开过？", "这个司机什么时候拿过杯子？"])
def test_new_named_subject_does_not_inherit_an_unrelated_old_object(question):
    context = resolve(question)
    assert not has_entity_binding(context)
    assert context["referenced_entities"] == []
    assert resolved_search_question(question, context) == question


def test_event_when_clause_still_matches_the_observed_previous_action():
    question = "When he held the cup, what was he wearing?"
    context = resolve(question, previous_turn(answer="The front-right passenger held the cup at 75 seconds."))
    assert has_valid_binding(context)
    assert context["referenced_time_ranges"] == [{"start": 70, "end": 80}]


@pytest.mark.parametrize("name,identity", [
    ("front-right passenger in a black jacket", "front-right passenger"),
    ("front-right passenger wearing sunglasses", "front-right passenger"),
    ("穿黑色外套的前排乘客", "前排乘客"),
    ("前排乘客戴着帽子", "前排乘客"),
])
def test_dynamic_appearance_is_not_used_as_person_identity(name, identity):
    context = resolve("他什么时候举起杯子？", previous_turn(entities=[{"name": name, "type": "person"}]))
    assert has_entity_binding(context)
    assert context["referenced_entities"] == [{"name": identity, "type": "person"}]


@pytest.mark.parametrize("question", ["那个时候他有没有戴帽子？", "刚才那段他在做什么？"])
def test_same_scene_reference_still_inherits_the_event_time(question):
    context = resolve(question)
    assert has_valid_binding(context)
    assert context["referenced_time_ranges"] == [{"start": 70, "end": 80}]


def test_entity_only_context_keeps_question_router_and_mst_scores_time_neutral():
    from online_query.query_router import QueryRouter
    from online_short_term.mst_retriever import MSTRetriever
    question = "他什么时候拿过杯子？"
    context = resolve(question)
    state = dict(current_ready=True, short_term_ready=True, long_term_ready=True)
    baseline = QueryRouter().route(question, session_context=state)
    actual = QueryRouter().route(question, session_context=state, cache_context=context)
    assert actual["query_type"] == baseline["query_type"]
    assert actual["memory_route"]["use_interaction_cache"]
    scorer = MSTRetriever(None)
    early = dict(event_id="early", start_time=10, end_time=20, retrieval_text="The front-right passenger holds a cup.")
    late = dict(early, event_id="late", start_time=70, end_time=80)
    assert scorer._cache_context_score(early, context) == scorer._cache_context_score(late, context)


def action_result():
    return {
        "question": "他什么时候戴帽子？", "answer": "47秒和49秒画面中，他已经戴好帽子。",
        "generation_status": "multimodal_success",
        "evidence_frames": [{"path": f"frame-{stamp}.jpg", "timestamp": stamp} for stamp in [47, 49]],
        "raw": {"image_paths_used": ["frame-47.jpg", "frame-49.jpg"], "selected_evidence": [{
            "evidence_id": "event", "start_time": 30, "end_time": 60,
            "main_actions": [{"actor": "person", "action": "puts a hat on the head", "start_time": 40, "end_time": 48}],
        }]},
    }


def test_action_onset_is_preserved_separately_from_direct_observation_window(tmp_path):
    cache = InteractionCache("separate-times", tmp_path, ttl_seconds=0)
    result = action_result()
    cache.update_from_query_result(result["question"], result["question"], result)
    turn = cache.recent_interactions()[-1]
    assert [(window["start"], window["end"]) for window in turn["time_ranges"]] == [(47, 49)]
    assert [(window["start"], window["end"]) for window in turn["action_time_ranges"]] == [(40, 48)]
    context = CoreferenceResolver().resolve("那个时候他坐在哪里？", cache)
    assert [(window["start"], window["end"]) for window in context["referenced_time_ranges"]] == [(47, 49)]


def test_loading_legacy_cache_clips_each_citation_window_without_merging_occurrences(tmp_path):
    cache = InteractionCache("legacy", tmp_path, ttl_seconds=0)
    legacy = previous_turn(time_ranges=[
        {"start": 40, "end": 49, "source": "answer_cited_sent_frames"},
        {"start": 90, "end": 95, "source": "answer_cited_sent_frames"},
        {"start": 0, "end": 30, "source": "explicit_time"}], evidence_frames=[
        {"path": "a.jpg", "timestamp": 47}, {"path": "b.jpg", "timestamp": 49},
        {"path": "c.jpg", "timestamp": 94}])
    cache.data["recent_interactions"] = [legacy]
    cache.flush()
    reloaded = InteractionCache("legacy", tmp_path, ttl_seconds=0)
    windows = reloaded.recent_interactions()[-1]["time_ranges"]
    assert [(window["start"], window["end"]) for window in windows] == [(47, 49), (94, 94), (0, 30)]


def test_captures_beyond_an_action_do_not_artificially_extend_it(tmp_path):
    cache = InteractionCache("points", tmp_path, ttl_seconds=0)
    result = action_result()
    result["raw"]["selected_evidence"][0]["main_actions"][0].update(start_time=10, end_time=20)
    assert cache._extract_action_time_ranges(result) == []
    assert [(window["start"], window["end"]) for window in cache._extract_time_ranges(result)] == [(47, 49)]
