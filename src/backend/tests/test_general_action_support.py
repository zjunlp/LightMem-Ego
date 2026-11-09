"""Action bindings outrank incidental scene mentions across unrelated scenes."""
import copy

import pytest

from online_evidence_content import observation_match, visual_anchors
from online_query.evidence_packer import EvidencePacker


def observation(t, action, obj, description, **extra):
    return {"timestamp": t, "path": f"frame_{t:06d}.jpg", "actor": "person",
            "action": action, "objects": [obj], "description": description, **extra}


def event(observations):
    return {"evidence_id": "scene", "source_memory": "M_st", "start_time": 0, "end_time": 100,
            "keyframe_paths": [o["path"] for o in observations],
            "metadata": {"visual_observations": observations}}


def pack(question, observations, limit=1, **constraints):
    return EvidencePacker().pack(question, {"use_image_evidence": True, "max_image_evidence": limit,
        "query_constraints": constraints}, {"final_evidence": [event(observations)]})


@pytest.mark.parametrize("question,action,obj,background_action,background_obj", [
    ("When is the person carrying the umbrella?", "carrying", "umbrella", "adjusting", "glasses"),
    ("When is the person lifting the cup?", "lifting", "cup", "opening", "drawer"),
    ("Show the person moving the binder", "moving", "binder", "typing", "keyboard"),
    ("Show the person pushing the cart", "pushing", "cart", "reading", "label"),
    ("人什么时候搬动文件夹？", "搬动", "文件夹", "整理", "眼镜"),
    ("人什么时候推购物车？", "推", "购物车", "阅读", "标签"),
])
def test_bound_action_beats_background_object(question, action, obj, background_action, background_obj):
    background = observation(10, background_action, background_obj,
        f"The person {background_action} {background_obj}; {obj} is in the scene.")
    actual = observation(20, action, obj, f"The person {action} {obj}.")
    assert observation_match(question, actual) > observation_match(question, background)
    out = pack(question, [background, actual])
    assert out["selected_image_paths_for_mllm"] == [actual["path"]]


def test_unknown_free_text_is_bounded_inspection_and_never_hard_absence():
    rows = [{"timestamp": t, "path": f"frame_{t:06d}.jpg", "description": "A person is visible."}
            for t in (10, 20, 30)]
    out = pack("Show the person carrying the astrolabe", rows, limit=6)
    assert len(out["selected_image_paths_for_mllm"]) == 2
    assert out["image_selection_summary"]["inspection_fallback_count"] == 2


def test_explicit_scope_can_inspect_background_and_preserves_input():
    rows = [observation(10, "adjusting", "glasses", "An umbrella rests on the seat."),
            observation(20, "carrying", "umbrella", "The person carries an umbrella.")]
    original = copy.deepcopy(rows)
    out = pack("At 10 seconds is the person carrying the umbrella?", rows,
               time_range=[10, 10], time_explicit=True)
    assert out["selected_image_paths_for_mllm"] == [rows[0]["path"]]
    assert rows == original


def test_bound_uncertain_or_negative_action_is_not_positive_support():
    question = "Show the person carrying the astrolabe"
    positive = observation(20, "carrying", "astrolabe", "The person is carrying the astrolabe.")
    negative = observation(10, "carrying", "astrolabe", "The person is not carrying the astrolabe.", polarity="negative")
    uncertain = observation(15, "carrying", "astrolabe", "Possibly carrying the astrolabe.", certainty="uncertain")
    assert observation_match(question, positive) > observation_match(question, uncertain) > observation_match(question, negative)
    assert pack(question, [negative, positive])["selected_image_paths_for_mllm"] == [positive["path"]]


def test_actor_conflict_does_not_borrow_background_target():
    wrong = observation(10, "carrying", "umbrella", "The front-right person carries an umbrella.", actor="front-right person")
    correct = observation(20, "carrying", "umbrella", "The front-left person carries an umbrella.", actor="front-left person")
    out = pack("Show front-left person carrying umbrella", [wrong, correct])
    assert out["selected_image_paths_for_mllm"] == [correct["path"]]


def test_generic_structured_boundary_preserves_repeated_occurrences():
    rows = [observation(10, "carrying", "umbrella", "Person carrying umbrella."),
            observation(11, "carrying", "umbrella", "Person carrying umbrella."),
            observation(12, "placing", "umbrella", "Person places umbrella on seat."),
            observation(14, "carrying", "umbrella", "Person carrying umbrella."),
            observation(15, "carrying", "umbrella", "Person carrying umbrella.")]
    out = pack("Show person carrying umbrella", rows, limit=2)
    assert out["selected_image_paths_for_mllm"] == [rows[0]["path"], rows[3]["path"]]


@pytest.mark.parametrize("question", ["When does the person carry the umbrella?", "When was the umbrella carried by the person?"])
def test_action_inflections_use_native_bindings(question):
    rows = [observation(10, "adjusting", "glasses", "Umbrella visible on seat."),
            observation(20, "carrying", "umbrella", "Person carrying umbrella.")]
    assert pack(question, rows)["selected_image_paths_for_mllm"] == [rows[1]["path"]]


def test_free_text_unknown_object_negative_keeps_polarity():
    rows = [{"path": f"frame_{t:06d}.jpg", "timestamp": t, "description": text}
            for t, text in [(10, "Person is not carrying the astrolabe."),
                            (20, "Person is carrying the astrolabe.")]]
    assert observation_match("person carrying astrolabe", rows[1]) > observation_match("person carrying astrolabe", rows[0])
    assert pack("Show person carrying astrolabe", rows)["selected_image_paths_for_mllm"] == [rows[1]["path"]]


@pytest.mark.parametrize("field", ["main_actions", "state_changes"])
def test_native_action_and_state_records_localize_without_observation_labels(field):
    e = event([])
    e["keyframe_paths"] = ["frame_000010.jpg", "frame_000020.jpg"]
    e["metadata"]["keyframes"] = [{"path": f"frame_{t:06d}.jpg", "timestamp": t} for t in (10, 20)]
    if field == "main_actions":
        rows = [observation(10, "adjusting", "glasses", "Umbrella on seat."),
                observation(20, "carrying", "umbrella", "Person carries umbrella.")]
    else:
        rows = [{"timestamp": 10, "entity": "umbrella", "before": "carrying", "after": "on seat"},
                {"timestamp": 20, "entity": "umbrella", "before": "on seat", "after": "carrying"}]
    e["metadata"][field] = rows
    out = EvidencePacker().pack("Show person carrying umbrella", {"use_image_evidence": True,
        "max_image_evidence": 1, "query_constraints": {}}, {"final_evidence": [e]})
    assert out["selected_image_paths_for_mllm"] == ["frame_000020.jpg"]


@pytest.mark.parametrize("cap", [0, 1])
def test_general_action_support_obeys_zero_and_one_caps(cap):
    rows = [observation(10, "carrying", "umbrella", "Person carries umbrella.")]
    assert len(pack("Show person carrying umbrella", rows, limit=cap)["selected_image_paths_for_mllm"]) == cap


def test_discarded_candidate_cannot_relabel_selected_frame_or_hard_window():
    selected = {"evidence_id": "selected", "source_memory": "M_st", "start_time": 10, "end_time": 15,
        "keyframe_paths": ["shared.jpg"], "metadata": {"keyframes": [
            {"path": "shared.jpg", "timestamp": 12, "caption": "selected caption", "source": "M_st"}]}}
    rejected = {"evidence_id": "rejected", "source": "M_lt", "path": "shared.jpg",
                "timestamp": 99, "caption": "rejected caption"}
    out = EvidencePacker().pack("Show the scene", {"use_image_evidence": True, "max_image_evidence": 1,
        "query_constraints": {"time_range": [12, 12]}},
        {"final_evidence": [selected], "evidence_frames": [rejected]})
    assert out["selected_image_paths_for_mllm"] == ["shared.jpg"]
    frame = out["selected_evidence_frames"][0]
    assert frame["timestamp"] == 12 and frame["caption"] == "selected caption"
    assert frame["source"] == "M_st"
    assert any(origin.get("evidence_id") == "selected" for origin in frame["frame_origins"])
    assert any(origin.get("evidence_id") == "rejected" and origin.get("selection_association") == "unselected_candidate"
               for origin in frame["frame_origins"])


def test_matching_selected_candidate_can_fill_unknown_capture_time_only():
    selected = {"evidence_id": "selected", "source_memory": "M_st", "keyframe_paths": ["shared.jpg"]}
    matching = {"evidence_id": "selected", "source": "M_st", "path": "shared.jpg", "timestamp": 12}
    frames = EvidencePacker()._rank_frames([], [matching], final_evidence=[selected])
    assert frames[0]["timestamp"] == 12


def test_time_bound_action_records_preserve_repeated_occurrences_without_path_labels():
    times = [10, 11, 12, 14, 15]
    rows = [observation(t, action, "umbrella", f"Person {action} umbrella.")
            for t, action in zip(times, ["carrying", "carrying", "placing", "carrying", "carrying"])]
    e = event(rows)
    e["metadata"] = {"keyframes": [{"path": row["path"], "timestamp": row["timestamp"]} for row in rows],
                     "main_actions": [{key: value for key, value in row.items() if key != "path"} for row in rows]}
    out = EvidencePacker().pack("Show person carrying umbrella", {"use_image_evidence": True,
        "max_image_evidence": 2, "query_constraints": {}}, {"final_evidence": [e]})
    assert out["selected_image_paths_for_mllm"] == [rows[0]["path"], rows[3]["path"]]


@pytest.mark.parametrize("description,status", [
    ("He did not carry the astrolabe.", "negative"),
    ("He may have carried the astrolabe.", "candidate"),
    ("He might have carried the astrolabe.", "candidate"),
    ("He is carrying possibly the astrolabe.", "candidate"),
    ("He is not smiling while carrying the astrolabe.", "supported"),
    ("He does not smile and carries the astrolabe.", "supported"),
])
def test_action_polarity_tracks_predicate_inflections_and_local_clauses(description, status):
    from online_evidence_content import action_support
    row = observation(10, "carrying", "astrolabe", description)
    assert action_support("Show person carrying astrolabe", row)["status"] == status


@pytest.mark.parametrize("description,status", [
    ("他没有拿着星盘。", "negative"),
    ("他可能拿着星盘。", "candidate"),
    ("他没有微笑，同时拿着星盘。", "supported"),
    ("他没有微笑但拿着星盘。", "supported"),
])
def test_chinese_action_polarity_remains_local(description, status):
    from online_evidence_content import action_support
    row = observation(10, "拿着", "星盘", description)
    assert action_support("给我拿着星盘的画面", row)["status"] == status


@pytest.mark.parametrize("action,description", [
    ("carrying and inspecting", "He carries the astrolabe but does not inspect it."),
    ("carrying and inspecting", "He does not inspect the astrolabe while carrying it."),
    ("拿着并观察", "他没有观察星盘，但拿着它。"),
])
def test_partial_compound_action_does_not_borrow_other_predicate_negation(action, description):
    from online_evidence_content import action_support
    obj = "星盘" if "拿" in action else "astrolabe"
    question = "给我拿着星盘的画面" if "拿" in action else "Show person carrying astrolabe"
    assert action_support(question, observation(10, action, obj, description))["status"] == "supported"


def test_explicit_polarity_and_certainty_override_description_inference():
    from online_evidence_content import action_support
    q = "Show person carrying astrolabe"
    row = observation(10, "carrying", "astrolabe", "He did not carry the astrolabe.", polarity="positive", certainty="confirmed")
    assert action_support(q, row)["status"] == "supported"
    row = observation(10, "carrying", "astrolabe", "He carries the astrolabe.", polarity="negative")
    assert action_support(q, row)["status"] == "negative"
    row = observation(10, "carrying", "astrolabe", "He might have carried the astrolabe.", certainty="confirmed")
    assert action_support(q, row)["status"] == "supported"


def test_state_entity_is_object_and_cannot_hide_explicit_other_actor():
    from online_evidence_content import actor_binding_fit
    row = {"timestamp": 10, "path": "frame_000010.jpg", "entity": "umbrella", "before": "on seat",
           "after": "carrying", "description": "The front-right person carrying umbrella."}
    q = "Show front-left person carrying umbrella"
    assert actor_binding_fit(q, row) == 0
    inspected = pack(q, [row])
    assert inspected["selected_image_paths_for_mllm"] == ["frame_000010.jpg"]
    assert inspected["image_selection_summary"]["inspection_fallback_count"] == 1


@pytest.mark.parametrize("obj", ["glasses", "astrolabe"])
def test_unrelated_description_can_be_inspected_without_becoming_action_support(obj):
    row = observation(10, "polishing", obj, f"The person is polishing the {obj}.")
    out = pack("Show person carrying umbrella", [row], limit=6)
    assert out["selected_image_paths_for_mllm"] == ["frame_000010.jpg"]
    assert out['image_selection_summary']['inspection_fallback_count']==1
    from online_evidence_content import action_support
    assert action_support('Show person carrying umbrella',row)['status']=='background'


def mixed_event(obj, action, background, actual):
    return {"evidence_id": "mixed", "source_memory": "M_st", "start_time": 10, "end_time": 20,
        "keyframe_paths": ["frame_000010.jpg", "frame_000020.jpg"], "metadata": {
            "main_actions": [{"actor": "person", "action": action, "objects": [obj], "time_span": [20, 20]}],
            "visual_observations": [
                {"path": "frame_000010.jpg", "timestamp": 10, "description": background},
                {"path": "frame_000020.jpg", "timestamp": 20, "description": actual}]}}


def pack_mixed(question, e, cap=1, **constraints):
    return EvidencePacker().pack(question, {"use_image_evidence": True, "max_image_evidence": cap,
        "query_constraints": constraints}, {"final_evidence": [e]})


@pytest.mark.parametrize("obj,action,question,background,actual", [
    ("cup", "lifting the cup", "Show the person lifting the cup", "A cup sits on the counter while the person adjusts glasses.", "A person is visible."),
    ("binder", "moving the binder", "How does the person handle the binder?", "A binder sits on the desk while the person adjusts glasses.", "A person is visible."),
    ("umbrella", "carrying the umbrella", "Show the person carrying the umbrella", "An umbrella sits on the seat while the person adjusts glasses.", "A person is visible."),
    ("cart", "pushing the cart", "How does the person move the cart?", "A cart stands nearby while the person reads a label.", "A person is visible."),
    ("cup", "lifting the cup", "这个杯子他是怎么拿的？", "A cup sits on the counter while the person adjusts glasses.", "A person is visible."),
])
def test_mixed_schema_native_action_retains_inspection_ahead_of_topic_mentions(obj, action, question, background, actual):
    e = mixed_event(obj, action, background, actual)
    assert pack_mixed(question, e)["selected_image_paths_for_mllm"] == ["frame_000020.jpg"]


def test_object_noun_inside_action_phrase_is_not_predicate_proof():
    from online_evidence_content import action_support
    row = observation(10, "carrying the cup", "cup", "The person carries the cup.")
    support = action_support("Where is the cup?", row)
    assert support["object_match"] is True and support["action_match"] is False


def test_cross_language_how_action_remains_an_uncertain_native_candidate():
    from online_evidence_content import action_support
    row = observation(10, "lifting the cup", "cup", "The person lifts the cup.")
    support = action_support("这个杯子他是怎么拿的？", row)
    assert support["object_match"] is True and support["action_match"] is False
    assert support["status"] == "candidate"


def test_mixed_schema_property_open_and_summary_keep_free_text_inspection():
    e = mixed_event("cup", "lifting the cup", "The cup is white on the counter.", "A person is visible.")
    assert pack_mixed("What color is the cup?", e)["selected_image_paths_for_mllm"] == ["frame_000010.jpg"]
    assert len(pack_mixed("What happened?", e, cap=2)["selected_image_paths_for_mllm"]) == 2
    assert set(pack_mixed("Summarize the scene", e, cap=2, summary=True)["selected_image_paths_for_mllm"]) == {"frame_000010.jpg", "frame_000020.jpg"}


def test_mixed_schema_action_category_preserves_explicit_window_and_caps():
    e = mixed_event("cup", "lifting the cup", "A cup rests on the counter.", "A person is visible.")
    assert pack_mixed("Show person lifting cup", e, cap=0)["selected_image_paths_for_mllm"] == []
    assert pack_mixed("At 10 seconds is the person lifting the cup?", e,
        time_range=[10, 10], time_explicit=True)["selected_image_paths_for_mllm"] == ["frame_000010.jpg"]


def test_native_actor_words_inside_action_phrase_are_not_predicate_proof():
    from online_evidence_content import action_support
    row = observation(10, "person polishing the umbrella", "umbrella", "The person polishes the umbrella.")
    assert action_support("Show person carrying umbrella", row)["action_match"] is False
    row = observation(10, "Alice polishing the umbrella", "umbrella", "Alice polishes the umbrella.", actor="Alice")
    assert action_support("Show Alice carrying umbrella", row)["action_match"] is False


@pytest.mark.parametrize("question", ["How many cups are visible?", "How much is visible?", "How old is it?",
                                      "How long is it?", "How far is it?", "How often is it visible?", "How fast is it moving?"])
def test_quantity_and_degree_how_questions_do_not_request_manner(question):
    from online_evidence_content import query_requests_action
    assert query_requests_action(question) is False


@pytest.mark.parametrize("question", ["How does the person carry the cup?", "How is the cup carried?", "这个杯子怎么拿的？"])
def test_manner_question_grammar_remains_available(question):
    from online_evidence_content import query_requests_action
    assert query_requests_action(question) is True


@pytest.mark.parametrize("obj,question,background", [
    ("bag", "How many items are inside the bag?", "The bag contains two items."),
    ("cup", "How old is the cup?", "The cup is visibly old and worn."),
    ("cup", "How long has the cup been on the counter?", "The cup has remained on the counter throughout this observation."),
])
def test_mixed_quantity_and_degree_questions_keep_property_inspection(obj, question, background):
    e = mixed_event(obj, f"lifting the {obj}", background, "A person is visible.")
    assert pack_mixed(question, e)["selected_image_paths_for_mllm"] == ["frame_000010.jpg"]
