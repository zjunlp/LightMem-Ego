"""Regression contracts from real dialogue failures; every model call is local fake."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from online_current.mcur_store import MCurStore
from online_query import query_engine as engine
from online_query.coreference_resolver import CoreferenceResolver
from online_query.evidence_sufficiency import EvidenceSufficiencyEvaluator
from online_query.interaction_cache import InteractionCache
from online_query_constraints import parse_query_constraints
from test_event_query_integration_20261025 import fallback_session, _fallback


FIXTURES = Path(__file__).resolve().parent / "fixtures/evidence"


def test_saved_recent_tail_uses_original_scope_with_cache_rewritten_text():
    answer = json.loads((FIXTURES / "recent_tail.json").read_text())["result"]
    memories = answer["raw"]["memory_results"]
    evaluation = EvidenceSufficiencyEvaluator().evaluate(
        question=answer["resolved_question"], query_type=answer["query_type"],
        query_constraints=answer["route_decision"]["query_constraints"],
        short_term_results=memories["short_term_results"], current_results=memories["current_results"],
    )
    assert evaluation["requires_current_tail"]
    assert evaluation["current_tail_coverage"]["tail_range"] == [601, 610]


def test_explicit_past_scope_does_not_request_unrelated_tail():
    evaluation = EvidenceSufficiencyEvaluator().evaluate(
        question="第10秒在做什么", query_type="recent_recall",
        query_constraints=parse_query_constraints("第10秒在做什么", 40),
        short_term_results=[{"event_id": "past", "start_time": 0, "end_time": 30, "status": "refined"}],
        current_results=[{"current_context": {"window_end_time": 40}, "current_selection": {}}],
    )
    assert not evaluation["requires_current_tail"]


@pytest.fixture
def current_session(tmp_path, monkeypatch):
    session = tmp_path / "s"
    path = session / "stream/frames/frame_000075.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(path)
    MCurStore(session).update_from_frame_stream(
        frame_index=75, frame_path=str(path.relative_to(session)), relative_ts_ms=75000,
    )
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("EM2MEM_CURRENT_FAST_PATH_ENABLED", "1")
    monkeypatch.setenv("EM2MEM_INTERACTION_CACHE_ENABLED", "1")
    monkeypatch.setenv("EM2MEM_MCUR_ANSWER_RETRIES", "1")
    return tmp_path, session


@pytest.mark.parametrize("mode,enabled,writes", [
    ("auto", True, True), ("write_only", True, True), ("read_only", True, False),
    ("off", True, False), ("auto", False, False),
])
def test_direct_current_cache_write_modes(current_session, monkeypatch, mode, enabled, writes):
    root, session = current_session
    class Model:
        def generate(self, prompt): return "前排右边的人低头俯身看向腿边。"
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    monkeypatch.setattr(InteractionCache, "latest_context", lambda *_: pytest.fail("current fast path read dialogue"))
    answer = engine.query_session("s", "现在前排右边这位在干什么？", sessions_root=root,
                                  cache_mode=mode, use_interaction_cache=enabled, max_image_evidence=1)
    cache_path = session / "em2mem/cache/interaction_cache.json"
    turns = json.loads(cache_path.read_text())["recent_interactions"] if cache_path.exists() else []
    assert len(turns) == int(writes)
    assert answer["fast_path"] == "M_cur_direct"
    assert answer["cache_update"]["updated"] is writes
    if writes:
        assert turns[0]["question"] == "现在前排右边这位在干什么？"
        assert any(frame["timestamp"] == 75 for frame in turns[0]["evidence_frames"])


def test_failed_current_generation_does_not_write_normal_dialogue(current_session, monkeypatch):
    root, session = current_session
    class Model:
        def generate(self, prompt): raise RuntimeError("mock model failure")
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    answer = engine.query_session("s", "现在前排右边这位在干什么？", sessions_root=root,
                                  cache_mode="auto", use_interaction_cache=True, max_image_evidence=0)
    path = session / "em2mem/cache/interaction_cache.json"
    turns = json.loads(path.read_text())["recent_interactions"] if path.exists() else []
    assert not turns
    assert answer["cache_update"]["updated"] is False


def _cache(turns):
    return SimpleNamespace(latest_context=lambda: {"recent_interactions": turns, "latest_interaction": turns[-1]})


def _turn(name, question, answer, second):
    return {"turn_id": name, "question": question, "answer": answer, "anchor_status": "confirmed",
            "time_ranges": [{"start": second, "end": second}], "entities": [],
            "evidence_frames": [{"path": name + ".jpg", "timestamp": second}]}


def test_bending_action_beats_generic_question_words():
    turns = [_turn("opening", "说回开头，前排左边那位拿着东西时，右边那位在做什么？", "指向设备。", 0),
             _turn("bending", "现在前排右边这位在干什么？", "前排右边的人正低头俯身看向腿边。", 75),
             _turn("other", "那个设备能看清吗？", "无法看清。", 180)]
    resolved = CoreferenceResolver().resolve("回到前排右边那位低头俯身的时候，后排靠左那位当时在做什么？", _cache(turns))
    assert resolved["matched_turn_id"] == "bending"
    assert resolved["referenced_time_ranges"][0]["start"] == 75


def test_explicit_action_reference_without_a_match_stays_unanchored():
    resolved = CoreferenceResolver().resolve(
        "回到他打开车门的时候，他穿什么？",
        _cache([_turn("wrong", "说回开头，他在做什么？", "他拿着手机坐着。", 0)]),
    )
    assert not resolved.get("referenced_time_ranges")
    assert resolved.get("temporal_relation") != "same_time"


def test_generic_object_followup_inherits_primary_topic_not_background_entities():
    turn = _turn("bag", "最近手里拿什么？", "他拿着灰色背包。此前拿过手机和眼镜。", 108)
    turn["entities"] = [{"name": "smartphone"}, {"name": "sunglasses"}, {"name": "gray backpack"}]
    turn["focus_entities"] = [{"name": "gray backpack", "aliases": ["背包"], "source": "final_evidence"}]
    turn["focus_evidence_refs"] = [{"evidence_id": "bag_event", "source_memory": "M_st", "start_time": 96, "end_time": 108}]
    resolved = CoreferenceResolver().resolve("你说的那个东西，他是怎么拿着的？", _cache([turn]))
    assert "gray backpack" in resolved["resolved_question"]
    assert "smartphone" not in resolved["resolved_question"]
    assert "sunglasses" not in resolved["resolved_question"]


def test_two_primary_objects_do_not_get_arbitrarily_resolved():
    turn = _turn("two", "手里有什么", "拿着包和杯子。", 20)
    turn["focus_entities"] = [{"name": "backpack"}, {"name": "cup"}]
    resolved = CoreferenceResolver().resolve("那个东西是什么颜色？", _cache([turn]))
    assert resolved.get("reference_ambiguous")
    assert not resolved.get("referenced_time_ranges")


def test_same_object_different_action_does_not_establish_requested_event():
    turn = _turn("other_action", "乘客放下杯子时穿什么？", "乘客放下杯子。", 20)
    resolved = CoreferenceResolver().resolve("回到乘客拿起杯子的时候，他戴眼镜吗？", _cache([turn]))
    assert not resolved.get("referenced_time_ranges")
    assert resolved.get("temporal_relation") != "same_time"


def test_shared_unknown_object_is_not_a_matching_action():
    turn = _turn("closed", "What did the passenger do?", "The passenger closed the door.", 20)
    resolved = CoreferenceResolver().resolve("When he opened the door, what was he wearing?", _cache([turn]))
    assert not resolved.get("referenced_time_ranges")


def test_explicit_opposite_seated_actor_does_not_match_same_action():
    turn = _turn("left", "左前排那位在做什么？", "前排左边的人低头俯身看向腿边。", 20)
    resolved = CoreferenceResolver().resolve("回到前排右边的人低头俯身的时候，他手里有什么？", _cache([turn]))
    assert not resolved.get("referenced_time_ranges")


def test_negated_action_is_not_a_confirmed_event_reference():
    turn = _turn("no", "When did he lift the phone?", "He did not lift the phone.", 20)
    resolved = CoreferenceResolver().resolve("When he lift the phone, what was he wearing?", _cache([turn]))
    assert not resolved.get("referenced_time_ranges")


@pytest.mark.parametrize("answer", ["It is unclear whether he lifted the phone.", "He might have lifted the phone."])
def test_uncertain_action_remains_a_hint_not_a_hard_event_anchor(answer):
    turn = _turn("uncertain", "Did he lift the phone?", answer, 20)
    resolved = CoreferenceResolver().resolve("When he lifted the phone, what was he wearing?", _cache([turn]))
    assert not resolved.get("referenced_time_ranges")


def test_primary_focus_scopes_times_images_and_ids_together():
    turn = _turn("bag", "手里有什么", "他拿着灰色背包。此前拿过手机。", 108)
    turn.update(time_ranges=[{"start": 1, "end": 2}, {"start": 96, "end": 108}],
                focus_entities=[{"name": "gray backpack"}],
                focus_evidence_refs=[{"evidence_id": "bag_event", "source_memory": "M_st", "start_time": 96, "end_time": 108,
                                      "keyframe_paths": ["bag.jpg"]}],
                evidence_frames=[{"path": "phone.jpg", "timestamp": 1}, {"path": "bag.jpg", "timestamp": 100}],
                retrieved_segment_ids=["unrelated_phone"], visual_ids=["unrelated_phone_image"])
    resolved = CoreferenceResolver().resolve("你说的那个东西，当时他是怎么拿着的？", _cache([turn]))
    assert [(r["start"], r["end"]) for r in resolved["referenced_time_ranges"]] == [(96,108)]
    assert [f["path"] for f in resolved["referenced_evidence_frames"]] == ["bag.jpg"]
    assert "unrelated_phone" not in resolved["referenced_segment_ids"]
    assert not resolved["referenced_visual_ids"]


def test_multiple_actual_periods_for_one_focus_remain_separate():
    turn = _turn("bag", "拿着什么", "两段拿包。", 108)
    turn.update(time_ranges=[{"start": 10, "end": 20}, {"start": 96, "end": 108}],
                focus_entities=[{"name": "backpack"}],
                focus_evidence_refs=[{"evidence_id": "bag1", "source_memory": "M_st", "start_time": 10, "end_time": 20},
                                     {"evidence_id": "bag2", "source_memory": "M_st", "start_time": 96, "end_time": 108}],
                evidence_frames=[{"path": "first.jpg", "timestamp": 15}, {"path": "second.jpg", "timestamp": 100}])
    resolved = CoreferenceResolver().resolve("那个东西，当时他怎么拿着的？", _cache([turn]))
    assert [(r["start"], r["end"]) for r in resolved["referenced_time_ranges"]] == [(10,20),(96,108)]


def test_overlapping_source_refs_do_not_invent_multiple_occurrences():
    turn = _turn("bag", "拿着什么", "96到108秒拿包。", 108)
    turn.update(time_ranges=[{"start": 96, "end": 108}], focus_entities=[{"name": "backpack"}],
                focus_evidence_refs=[{"evidence_id": "cur", "source_memory": "M_cur", "start_time": 79, "end_time": 108},
                                     {"evidence_id": "st", "source_memory": "M_st", "start_time": 90, "end_time": 100}])
    resolved = CoreferenceResolver().resolve("那个东西，当时怎么拿着？", _cache([turn]))
    assert [(r["start"], r["end"]) for r in resolved["referenced_time_ranges"]] == [(96,108)]
    assert "occurrence_instruction" not in resolved


def test_short_term_model_prompt_is_saved_exactly(fallback_session, monkeypatch):
    calls = []
    class Model:
        def generate(self, prompt):
            calls.append(prompt)
            return "当前证据不足以确认。"
    monkeypatch.setenv("EM2MEM_MST_ANSWER_WITH_LLM", "1")
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    result = _fallback(fallback_session)
    sent_prompt = calls[0][-1]["content"][0]["text"] if isinstance(calls[0], list) else calls[0]
    assert result["raw"]["answer_debug"]["actual_prompt_text"] == sent_prompt


def test_streamed_current_turn_is_written_once(current_session, monkeypatch):
    root, session = current_session
    class Model:
        def stream_generate(self, prompt, on_chunk):
            on_chunk("他低头看向腿边。")
            return "他低头看向腿边。"
    chunks = []
    monkeypatch.setattr(engine, "_get_short_term_answer_model", lambda: Model())
    result = engine.query_session("s", "他现在在做什么？", sessions_root=root,
                                  cache_mode="auto", use_interaction_cache=True, stream_handler=chunks.append)
    turns = json.loads((session / "em2mem/cache/interaction_cache.json").read_text())["recent_interactions"]
    assert len(turns) == 1 and result["cache_update"]["updated"]
    assert any(chunk.get("type") == "delta" for chunk in chunks)
