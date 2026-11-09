"""Regressions for time coverage, real frame grounding, and publication races."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from online_query_constraints import actor_fit, parse_query_constraints
from online_query.coreference_resolver import CoreferenceResolver
from online_query.evidence_packer import EvidencePacker, image_evidence_label
from online_query.interaction_cache import InteractionCache
from online_query.memory_router import MemoryRouter
from online_query.memory_fusion import MemoryFusion
from online_query.query_engine import LoadedQueryEngine
from online_evidence_content import prompt_evidence
from online_short_term.micro_event_refiner import MicroEventRefiner, _select_keyframes


@pytest.mark.parametrize("question, expected", [
    ("第10～20秒他拿什么", [10, 20]),
    ("从10秒到20秒发生什么", [10, 20]),
    ("前两分钟有没有拿手机", [0, 120]),
    ("最近十秒发生什么", [170, 180]),
    ("between 10 and 20 seconds", [10, 20]),
    ("00:10–00:20发生什么", [10, 20]),
    ("第1分20秒到1分40秒", [80, 100]),
    ("视频开始他穿什么", [0, 0]),
    ("视频结尾穿什么", [180, 180]),
    ("第26秒他戴眼镜吗", [26, 26]),
    ("在26秒之后", [26, 180]),
    ("在26秒之前", [0, 26]),
    ("总结这个3分钟的视频", None),
    ("上一答最早那张图", None),
    ("从视频开始到最后依次发生了什么", None),
    ("总结整个视频，从视频开始到视频结尾", None),
])
def test_recognized_explicit_video_times(question, expected):
    assert parse_query_constraints(question, 180)["time_range"] == expected


RUNTIME = {"current_ready": True, "current_stale": False, "short_term_ready": True,
           "long_term_ready": True, "mcur_time_span": [16, 45],
           "mst_time_span": [0, 40], "mlt_time_span": [0, 30]}


def test_history_fills_only_the_requested_unaggregated_range():
    router = MemoryRouter()
    history = router.route("右侧男子什么时候把手机贴耳", {}, RUNTIME)
    assert history["memory_route"]["use_short_term"]
    assert history["memory_route"]["use_current"]
    earlier = router.route("第10～20秒他拿着什么", {}, RUNTIME)
    assert earlier["memory_route"]["use_short_term"]
    now = router.route("他现在拿着手机吗", {}, RUNTIME)
    assert now["query_type"] == "current_perception"
    assert now["memory_route"]["use_current"]
    assert not now["memory_route"]["use_long_term"]
    explicit = router.route("什么时候把手机贴耳", {"use_short_term": False, "use_current": False}, RUNTIME)
    assert not explicit["memory_route"]["use_short_term"]
    assert not explicit["memory_route"]["use_current"]


def test_when_question_keeps_long_term_context_and_multiple_short_term_details():
    memory = {"text_results": [{"memory_id": f"long{i}", "start_time": i*30, "end_time": (i+1)*30, "score": .1} for i in range(3)],
              "short_term_results": [{"event_id": f"recent{i}", "start_time": 30+i*10, "end_time": 40+i*10, "score": 1} for i in range(8)]}
    constraints = parse_query_constraints("什么时候把手机贴耳")
    result = MemoryFusion().fuse(memory, {"memory_priority": {"M_lt": .85, "M_st": .75}, "query_constraints": constraints},
                                 {"final_evidence_k": 4}, "temporal_localization")
    selected = result["final_evidence"]
    assert len(selected) == 4
    assert sum(event["source_memory"] == "M_lt" for event in selected) == 1
    assert sum(event["source_memory"] == "M_st" for event in selected) == 3
    assert all(event["time_relevance"] == 0 for event in result["final_evidence"])


def test_from_beginning_to_end_uses_summary_policy_not_start_frame_policy():
    route = MemoryRouter().route("从视频开始到最后，右侧的人依次做了哪些事", {}, RUNTIME)
    assert route["query_type"] == "long_term_summary"
    assert route["query_constraints"]["time_range"] is None
    assert route["short_term_tail_range"] == [30, 40]


def test_all_explicit_source_switches_off_do_not_silently_reenable_long_term():
    route = MemoryRouter().route("什么时候拿手机", {"use_current":False, "use_short_term":False, "use_long_term":False}, RUNTIME)
    assert not any(route["memory_route"][key] for key in ("use_current", "use_short_term", "use_long_term"))


def test_requested_time_filters_dense_candidates_before_topk(tmp_path):
    from retrieval_source_helpers import EPISODIC, FakeEmbedding, NoModels, EARLY, LATE, LATE_CUTOFF
    memory = EPISODIC.EpisodicMemoryRAG(FakeEmbedding(), NoModels(), NoModels(), granularities=["30sec"])
    memory.dense_cache_dir = str(tmp_path)
    memory.load_captions_from_data({"30sec": [EARLY, LATE]})
    memory.index(LATE_CUTOFF)
    memory.set_query_constraints({"time_range": [70, 80]})
    found = memory.retrieve_ranked("alpha task", 1)
    assert [entry.doc_id for entry, _ in found] == ["late"]


def test_published_version_wins_even_when_mtime_is_identical(tmp_path):
    config = tmp_path / "memory_config.json"
    config.write_text(json.dumps({"status": "memory_ready", "memory_version": 4, "latest_ready_memory_version": 4}))
    engine = object.__new__(LoadedQueryEngine)
    engine.memory_config_path = config
    engine.memory_config_mtime = config.stat().st_mtime
    engine.active_query_memory_version = 3
    assert engine.needs_reload()


def test_time_scope_survives_fusion_and_image_packing():
    constraints = parse_query_constraints("第10～20秒", 90)
    memory = {"text_results": [
        {"memory_id": "early", "start_time": 0, "end_time": 30, "score": .1, "keyframe_paths": [f"frame_{t:06d}.jpg" for t in [0, 14, 26]], "keyframes": [{"path": f"frame_{t:06d}.jpg", "timestamp": t} for t in [0, 14, 26]]},
        {"memory_id": "late", "start_time": 60, "end_time": 90, "score": 1, "keyframe_paths": ["frame_000070.jpg"]},
    ]}
    fused = MemoryFusion().fuse(memory, {"memory_priority": {"M_lt": 1}, "query_constraints": constraints}, {"final_evidence_k": 4}, "temporal_localization")
    assert [e["evidence_id"] for e in fused["final_evidence"]] == ["early"]
    packed = EvidencePacker().pack("第10～20秒", {"query_constraints": constraints, "use_image_evidence": True, "max_image_evidence": 6}, fused)
    assert packed["selected_image_paths_for_mllm"] == ["frame_000014.jpg"]


def test_single_cached_frame_keeps_capture_time_without_numeric_filename():
    event = {"evidence_id": "cache", "source_memory": "M_cache", "keyframe_paths": ["saved.jpg"],
             "metadata": {"path": "saved.jpg", "timestamp": 26, "source": "M_lt"}}
    label = image_evidence_label("saved.jpg", [event])
    assert label["capture_time_seconds"] == 26
    assert label["cache_reused"] and label["original_source"] == "M_lt"


def test_failed_generation_does_not_create_a_normal_cache_turn(tmp_path):
    cache = InteractionCache("s", tmp_path)
    result = {"status": "ok", "answer": "Local evidence summary", "raw": {"llm_debug": {"local_fallback_used": True}}}
    assert not cache.update_from_query_result("什么时候", "什么时候", result)["updated"]
    assert not cache.recent_interactions()


def _cached_result(times, answer="请看这些画面"):
    return {"status": "ok", "answer": answer, "timestamps": [{"start": 0, "end": 180}],
            "raw": {"image_paths_used": [f"frame_{t:06d}.jpg" for t in times]},
            "evidence_frames": [{"path": f"frame_{t:06d}.jpg", "timestamp": t} for t in times]}


def test_unresolved_action_does_not_reuse_a_previous_hot_time(tmp_path):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("什么时候", "什么时候", _cached_result([35], "35秒手机贴耳"))
    cache.update_from_query_result("什么时候贴耳", "什么时候贴耳", _cached_result([21], "无法确定具体时间，只看到21秒看手机"))
    resolved = CoreferenceResolver().resolve("那个时候戴墨镜了吗", cache)
    assert resolved["anchor_status"] == "unresolved"
    assert not resolved["referenced_time_ranges"]


def test_unknown_precise_onset_does_not_erase_a_verified_sampled_occurrence(tmp_path):
    cache = InteractionCache("s", tmp_path)
    result = _cached_result([26], "第26秒的画面已清楚显示手机贴在耳边。现有画面不能确定这一动作开始的精确时刻。")
    cache.update_from_query_result("什么时候把手机贴到耳边", "", result)
    assert cache.recent_interactions()[-1]["anchor_status"] == "confirmed"
    assert CoreferenceResolver().resolve("那个时候他戴墨镜了吗", cache)["referenced_time_ranges"][0]["start"] == 26


@pytest.mark.parametrize("question,time", [("上一答第一张图", 35), ("上一答最早那张图", 26), ("上一答最后一张图", 84)])
def test_explicit_image_references_use_actual_displayed_images(tmp_path, question, time):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("看看画面", "看看画面", _cached_result([35, 26, 84]))
    reference = CoreferenceResolver().resolve(question, cache)
    assert reference["referenced_time_ranges"] == [{"start": time, "end": time, "source": "explicit_previous_image"}]


def test_independent_time_request_does_not_inherit_cache_window(tmp_path):
    cache = InteractionCache("s", tmp_path)
    cache.update_from_query_result("什么时候", "什么时候", _cached_result([35], "35秒手机贴耳"))
    reference = CoreferenceResolver().resolve("第0秒他穿什么", cache)
    assert not reference["is_followup"] and not reference["referenced_time_ranges"]


def test_separate_occurrences_are_not_one_continuous_cache_window(tmp_path):
    cache = InteractionCache("s", tmp_path)
    result = _cached_result([26, 123], "26秒和123秒都有手机画面")
    cache.update_from_query_result("什么时候", "什么时候", result)
    ranges = cache.recent_interactions()[-1]["time_ranges"]
    assert [(row["start"], row["end"]) for row in ranges] == [(26, 26), (123, 123)]


def test_lowering_the_phone_separates_nearby_occurrences(tmp_path):
    cache = InteractionCache("s", tmp_path)
    result = _cached_result([25, 26, 35, 40], "25、26秒贴耳；35、40秒再次贴耳")
    result["selected_evidence"] = [{"visual_observations": [
        {"timestamp": 25, "path": "frame_000025.jpg", "description": "The front-right man holds a phone at his ear"},
        {"timestamp": 28, "path": "frame_000028.jpg", "description": "The front-right man has lowered the phone"},
        {"timestamp": 35, "path": "frame_000035.jpg", "description": "The front-right man holds a phone at his ear"}]}]
    cache.update_from_query_result("右侧男子什么时候把手机贴耳", "", result)
    ranges = cache.recent_interactions()[-1]["time_ranges"]
    assert [(row["start"], row["end"]) for row in ranges] == [(25, 26), (35, 40)]


def test_requested_person_beats_another_person_with_more_matching_keywords():
    question = "画面右侧穿黑色上衣的人什么时候把手机贴到耳边"
    assert actor_fit(question, {"description": "The front-left man holds a phone in his right hand"}) == 0
    event = {"evidence_id": "event", "source_memory": "M_lt", "start_time": 0, "end_time": 30, "final_score": .5,
             "keyframe_paths": ["frame_000000.jpg", "frame_000026.jpg"], "visual_observations": [
                 {"timestamp": 0, "path": "frame_000000.jpg", "description": "The rear man in a black shirt holds a phone"},
                 {"timestamp": 26, "path": "frame_000026.jpg", "description": "The front-right man holds a phone near his face"}]}
    packed = EvidencePacker().pack(question, {"use_image_evidence": True, "max_image_evidence": 1}, {"final_evidence": [event]})
    assert packed["selected_image_paths_for_mllm"] == ["frame_000026.jpg"]


def test_refinement_default_sees_all_eight_retained_frames_and_tracks_provenance(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.delenv("EM2MEM_MST_REFINE_MAX_IMAGES", raising=False)
    times = [20, 21, 23, 25, 26, 28, 29, 30]
    frames = [{"path": f"f{time}.jpg", "timestamp": time} for time in times]
    for frame in frames:
        Image.new("RGB", (32, 32)).save(tmp_path / frame["path"])
    model_payloads = []
    def generate(**kwargs):
        model_payloads.append(kwargs)
        payload = {"refined_caption": "A man raises a phone", "visual_observations": [
            {"timestamp": 26, "path": "f26.jpg", "description": "phone at ear"},
            {"timestamp": 99, "path": "unseen.jpg", "description": "unseen"}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    refiner = MicroEventRefiner(backend="openai")
    refiner._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=generate)))
    event = {"event_id": "e", "start_time": 20, "end_time": 30, "keyframes": frames}
    payload = refiner._refine_openai(event, tmp_path)
    refined = refiner._apply_payload(event, payload)
    assert len(refined["refine"]["observed_frames"]) == 8
    assert [row["timestamp"] for row in refined["visual_observations"]] == [26]
    assert sum(block["type"] == "image_url" for block in model_payloads[0]["messages"][1]["content"]) == 8


def test_event_synopsis_is_not_removed_to_fit_tiny_summary_budget():
    event = {"evidence_id": "e", "caption": "A person sits in the car. " * 100,
             "main_actions": [{"action": "person sitting"}] * 30,
             "visual_observations": [{"timestamp": 26, "path": "frame_000026.jpg", "description": "The man holds a phone against his ear"}]}
    view, debug = prompt_evidence("总结视频", [event], 400)
    assert view[0]["caption"] == event["caption"]
    assert debug["budget_exceeded"] and debug["omitted_units"] > 0


def test_targeted_supplement_requires_the_conditions_in_one_observation():
    from online_evidence_content import supplemental_observation_matches
    question = "画面右侧的人是否同时戴白口罩和拿手机"
    def event(name, descriptions):
        return {"doc_id": name, "visual_observations": [{"timestamp": t, "description": text} for t, text in descriptions]}
    separate = event("separate", [(10, "The front-right man wears a white mask"), (30, "The front-right man holds a phone")])
    together = event("together", [(15, "The front-right man wears a white mask while holding a phone")])
    constraints = parse_query_constraints(question)
    assert supplemental_observation_matches(question, [separate, together], constraints)[0][0] == "together"
    assert not supplemental_observation_matches(question, [separate], constraints)


def test_failed_engine_reload_preserves_the_usable_previous_engine():
    from online_query.query_cache import SessionEngineCache
    from online_retrieval_scheme import retrieval_scheme_cache_key
    import time
    class Engine:
        session_id = "s"
        long_term_retrieval_scheme = "em2memory"
        last_accessed_at = time.time()
        closed = False
        def needs_reload(self): return True
        def close(self): self.closed = True
    old = Engine()
    cache = SessionEngineCache()
    key = retrieval_scheme_cache_key("s", "em2memory")
    cache._items[key] = old
    def failed(*args): raise RuntimeError("new snapshot cannot load")
    result = cache.reload_changed(failed)
    assert result[0]["status"] == "reload_failed"
    assert cache._items[key] is old and not old.closed
