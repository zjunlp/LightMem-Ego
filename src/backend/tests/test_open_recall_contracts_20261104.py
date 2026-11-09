"""Open recall must preserve inspectable evidence and the user's own scope."""
import pytest

from online_evidence_content import inspection_terms
from online_query.coreference_resolver import CoreferenceResolver
from online_query.evidence_packer import EvidencePacker
from online_query.memory_router import MemoryRouter
from online_query_constraints import parse_query_constraints


class Cache:
    def __init__(self, turns): self.turns = turns
    def latest_context(self):
        return {"recent_interactions": self.turns, "latest_interaction": self.turns[-1]}


def turn(identity, question, answer, time):
    return {"turn_id": identity, "question": question, "answer": answer,
            "time_ranges": [{"start": time, "end": time}],
            "entities": [], "focus_entities": [], "evidence_frames": []}


@pytest.mark.parametrize("question", ["左边那位在做什么？", "后排那个人做什么？", "What is the person on the right doing?"])
def test_open_activity_question_does_not_turn_actor_words_into_an_object_filter(question):
    assert inspection_terms(question) == set()


@pytest.mark.parametrize("question", ["前排右边那位在做什么？", "左边那位在干什么？", "What is the passenger doing?"])
def test_progressive_present_activity_consults_current_memory(question):
    scope = parse_query_constraints(question, 83)
    assert scope["requires_current"] and not scope["requires_history"]
    decision = MemoryRouter().route(question, {}, {
        "current_ready": True, "current_stale": False, "short_term_ready": True,
        "long_term_ready": True, "mcur_time_span": [54, 83], "mst_time_span": [0, 80], "mlt_time_span": [0, 60]},
        {}, query_constraints=scope)
    assert decision["memory_route"]["use_current"]


@pytest.mark.parametrize("question", ["刚开始左边那位在做什么？", "之前右侧的人在干什么？", "What was the passenger doing earlier?"])
def test_past_activity_preserves_history_instead_of_becoming_current(question):
    scope = parse_query_constraints(question, 83)
    assert scope["requires_history"] and not scope["requires_current"]


@pytest.mark.parametrize("question", [
    "她早先戴的帽子后来怎样了？", "从刚开始到后来，他的穿戴有哪些变化？",
    "他早先把手机贴耳，能确定当时在通话吗？", "最早见到杯子时，它放在哪儿？",
    "前面戴口罩那段，他穿什么？",
])
def test_self_contained_history_does_not_inherit_last_topics_time(question):
    cache = Cache([turn("unrelated", "刚才椅子在哪里？", "椅子在画面右侧。", 700)])
    result = CoreferenceResolver().resolve(question, cache, latest_seconds=710)
    assert not result["binding_valid"] and not result["referenced_time_ranges"]
    assert result["resolved_question"] in (None, question)


@pytest.mark.parametrize("topic_a,topic_b,question_a,answer_a,question_b,answer_b", [
    ("黑色外套", "白色上衣", "她现在穿的衣服有什么特点？", "她穿黑色厚羽绒服。", "她白色上衣那段戴帽子吗？", "她穿白色上衣并戴着帽子。"),
    ("蓝色背包", "红色背包", "那个蓝色背包放哪儿了？", "蓝色背包在左侧。", "红色背包的位置呢？", "红色背包在右侧。"),
])
def test_topic_reference_uses_referenced_attributes_not_new_question_object(topic_a, topic_b, question_a, answer_a, question_b, answer_b):
    cache = Cache([turn("referenced", question_a, answer_a, 80), turn("distractor", question_b, answer_b, 160)])
    result = CoreferenceResolver().resolve(f"跟前面说{topic_a}那会儿比，它的位置有什么变化？", cache, latest_seconds=200)
    assert result["binding_valid"] and result["matched_turn_id"] == "referenced"


def test_open_question_retains_images_instead_of_all_being_filtered(tmp_path):
    path = "stream/frames/observed.jpg"
    (tmp_path / path).parent.mkdir(parents=True)
    (tmp_path / path).write_bytes(b"saved capture")
    event = {"evidence_id": "observation", "source_memory": "M_st", "start_time": 10, "end_time": 20,
             "visual_observations": [{"timestamp": 15, "path": path,
                                      "description": "The front-right person is operating a device."}],
             "keyframe_paths": [path]}
    result = EvidencePacker(tmp_path).pack("前排右边那位在做什么？", {
        "use_image_evidence": True, "max_image_evidence": 3}, {"final_evidence": [event]})
    assert result["selected_image_paths_for_mllm"] == [path]


def test_chronological_review_keeps_multiple_periods_instead_of_one_cached_event():
    question = "他拿着杯子时，头的朝向怎么变化？按先后说说。"
    scope = parse_query_constraints(question, 300)
    assert scope["summary"] and scope["time_range"] is None
    result = CoreferenceResolver().resolve(question, Cache([turn("old", "他刚才在做什么？", "他低头拿着杯子。", 200)]), 300,
                                           query_constraints=scope)
    assert not result["binding_valid"] and not result["referenced_time_ranges"]


@pytest.mark.parametrize("question", ["从最初到后来，衣着有哪些变化？", "最初是红灯，到后来指示灯有哪些变化？"])
def test_multiple_named_phases_request_broad_history_without_invented_timestamps(question):
    scope = parse_query_constraints(question, 800)
    assert scope["summary"] and scope["requires_history"]
    assert not scope["time_explicit"] and scope["time_range"] is None


@pytest.mark.parametrize("question,text", [
    ("他拿手机又戴帽子的那阵，穿什么？", "The man holds a phone and wears a hat."),
    ("拿着杯子和包的时候，他坐在哪里？", "The person holds a cup and a bag."),
    ("戴眼镜拿水瓶那段是什么姿势？", "The woman wears glasses and holds a bottle."),
])
def test_multi_object_recall_can_recover_stored_observations_without_special_conjunction_word(question, text):
    from online_evidence_content import supplemental_observation_matches, supplemental_requirement_support
    event = {"doc_id": "detail", "start_time": 10, "end_time": 20,
             "visual_observations": [{"timestamp": 15, "path": "saved.jpg", "description": text}]}
    scope = parse_query_constraints(question, 80)
    assert not scope["requirements"]["simultaneous"]
    assert supplemental_observation_matches(question, [event], scope)[0][0] == "detail"
    support = supplemental_requirement_support(question, event, scope)
    assert support["observation_matched"] and not support["confirmed"]
    outside = {**scope, "time_range": [40, 50], "time_ranges": [[40, 50]]}
    assert supplemental_observation_matches(question, [event], outside) == []


def test_recovery_keeps_only_child_captures_inside_the_parent_event():
    from online_evidence_content import supplemental_observation_matches
    question = "拿杯子和包那会儿是什么姿势？"
    parent = {"doc_id": "parent", "start_time": 0, "end_time": 30,
              "source_micro_events": [{"event_id": "overlap", "start_time": 25, "end_time": 35,
                  "visual_observations": [{"timestamp": 34, "description": "The person holds a cup and a bag."}]}]}
    assert supplemental_observation_matches(question, [parent], {}) == []
