"""Shared evidence boundaries work across native stores and ordinary scenes."""
from copy import deepcopy

import pytest

from online_query.evidence_schema import evidence_frames, normalize_evidence
from online_query.coreference_resolver import CoreferenceResolver
from online_query.interaction_cache import InteractionCache
from online_query.memory_fusion import MemoryFusion


@pytest.mark.parametrize("event,source,path,time", [
    ({"current_context": {"open_event": {"open_event_id": "cooking"}},
      "current_selection": {"evidence_frames": [{"path": "pot.jpg", "timestamp": 0, "caption": "stirs soup"}]}}, "M_cur", "pot.jpg", 0),
    ({"event_id": "meeting", "keyframes": [{"path": "slides.jpg", "timestamp": "12.125", "score": 0}]}, "M_st", "slides.jpg", 12.125),
    ({"memory_id": "shopping", "keyframe_captions": [{"path": "basket.jpg", "timestamp": 1200}],
      "start_time": "00200000", "end_time": "00210000"}, "M_lt", "basket.jpg", 1200),
    ({"canonical_segment_id": "walking", "text": {"caption": "Carries a striped parcel"},
      "visual_items": [{"image_path": "parcel.jpg", "timestamp": 5, "visual_score": 0.75}]}, "M_lt", "parcel.jpg", 5),
    ({"source_memory": "M_cache", "metadata": {"path": "whisk.jpg", "timestamp": 9, "source": "M_st"}}, "M_cache", "whisk.jpg", 9),
])
def test_native_schemas_share_capture_and_provenance(event, source, path, time):
    before = deepcopy(event)
    normalized = normalize_evidence(event, source=source)
    frame = normalized["evidence_frames"][0]
    assert (frame["path"], frame["timestamp"], frame["source"]) == (path, time, source)
    assert frame["capture_time_seconds"] == time
    assert frame["frame_origins"]
    assert normalize_evidence(event)["source_memory"] == source
    if source == "M_cache":
        assert frame["original_source"] == "M_st"
    assert event == before


def test_event_bounds_are_decoded_separately_from_capture():
    event = {"memory_id": "meal", "start_time": "00003000", "end_time": "00010000",
             "timestamp": 45, "score": 0, "keyframe_paths": ["frame_000040.jpg"]}
    result = normalize_evidence(event)
    assert (result["start_time"], result["end_time"], result["timestamp"], result["score"]) == (30, 60, 45, 0)
    assert result["evidence_id"] == "meal"
    assert result["evidence_frames"][0]["timestamp"] is None


def test_native_current_window_span_does_not_supply_capture_time():
    event = {"current_context": {"state": {"window_start_time": 0, "window_end_time": 60, "current_time": 60},
                                  "open_event": {"open_event_id": "walking"}},
             "current_selection": {"evidence_frames": [{"path": "flowers.jpg", "timestamp": None}]}}
    result = normalize_evidence(event)
    assert (result["evidence_id"], result["start_time"], result["end_time"]) == ("walking", 0, 60)
    assert result["evidence_frames"][0]["timestamp"] is None


@pytest.mark.parametrize("value,path,expected", [
    (None, "frame_000040.jpg", None), (None, "kf_000028000_proc_000005.jpg", 28),
    (0, "kf_000028000_proc_000005.jpg", 0), (1200, "kf_000028000.jpg", 1200),
    (12.123456, "plain.jpg", 12.123456), ("00003000", "plain.jpg", 3000),
    (True, "plain.jpg", None), (float("nan"), "plain.jpg", None),
])
def test_capture_time_uses_only_numeric_metadata_or_known_path(value, path, expected):
    frame = evidence_frames({"path": path, "timestamp": value, "start_time": 10, "end_time": 20})[0]
    assert frame["timestamp"] == expected


def test_multiple_origins_preserve_native_metadata_scores_and_canonical_identity():
    event = {"evidence_id": "meeting", "canonical_segment_id": "seg_000000_000030", "source_memory": "M_lt",
             "keyframe_paths": ["board.jpg"], "final_score": 0.8,
             "metadata": {"text": {"keyframe_captions": [{"path": "board.jpg", "timestamp": 0, "score": 0, "role": "whiteboard"}]},
                          "visual_items": [{"image_path": "board.jpg", "timestamp": 0, "score": 0.9, "visual_score": 0.7}]}}
    before = deepcopy(event)
    frames = evidence_frames(event)
    assert len(frames) == 1
    assert frames[0]["timestamp"] == 0
    assert frames[0]["canonical_segment_id"] == "seg_000000_000030"
    assert frames[0]["role"] == "whiteboard"
    native_scores = [origin["metadata"].get("score") for origin in frames[0]["frame_origins"]]
    assert 0 in native_scores and 0.9 in native_scores
    assert event == before
    assert normalize_evidence(normalize_evidence(event))["evidence_frames"] == normalize_evidence(event)["evidence_frames"]


def test_fusion_exports_shared_frames_without_mutating_native_metadata():
    native = {"event_id": "cooking", "start_time": 10, "end_time": 20,
              "keyframes": [{"path": "spatula.jpg", "timestamp": 12, "score": 0}]}
    before = deepcopy(native)
    event = MemoryFusion()._from_short_term(native, {"M_st": 1})
    assert event["evidence_frames"][0]["timestamp"] == 12
    event["metadata"]["keyframes"][0]["timestamp"] = 99
    assert native == before


def test_cache_normalizes_selected_events_and_preserves_explicit_empty(tmp_path):
    cache = InteractionCache("general", tmp_path, ttl_seconds=0)
    selected = cache._selected_events({"selected_evidence": [{"event_id": "shopping", "keyframes": [{"path": "basket.jpg", "timestamp": 0}]}]})
    assert selected[0]["evidence_id"] == "shopping"
    assert selected[0]["evidence_frames"][0]["timestamp"] == 0
    assert cache._selected_events({"selected_evidence": [], "raw": {"selected_evidence": [{"event_id": "ghost"}]}}) == []


def test_cache_direct_current_requires_actual_sent_frames_and_keeps_capture_metadata(tmp_path):
    cache = InteractionCache("general", tmp_path, ttl_seconds=0)
    result = {"selected_evidence": [], "retrieval_mode": "current", "raw": {
        "image_paths_used": ["mug.jpg"], "current_context": {"state": {"current_time": 100}},
        "current_selection": {"evidence_frames": [{"path": "mug.jpg", "timestamp": 0}, {"path": "plate.jpg", "timestamp": 95}]}}}
    events = cache._selected_events(result)
    assert len(events) == 1
    assert events[0]["start_time"] == events[0]["end_time"] == 0
    assert [frame["path"] for frame in events[0]["evidence_frames"]] == ["mug.jpg"]
    result["raw"]["image_paths_used"] = []
    assert cache._selected_events(result) == []


def test_explicit_constraints_including_empty_override_stale_cache_window():
    results = {"short_term_results": [{"event_id": "walk", "start_time": 40, "end_time": 50}]}
    stale = {"query_constraints": {"time_range": [0, 1]}}
    selected = MemoryFusion().fuse(results, {"query_constraints": {}}, {"final_evidence_k": 2}, "general_qa", stale)["final_evidence"]
    assert [event["evidence_id"] for event in selected] == ["walk"]
    selected = MemoryFusion().fuse(results, {"query_constraints": {"time_range": [40, 50]}}, {"final_evidence_k": 2}, "general_qa", stale)["final_evidence"]
    assert [event["evidence_id"] for event in selected] == ["walk"]
    assert MemoryFusion().fuse(results, {}, {"final_evidence_k": 2}, "general_qa", stale)["final_evidence"] == []


def test_explicit_long_term_identity_precedes_generic_keyframe_container():
    native = {"memory_id": "shopping", "keyframes": [{"path": "ceramic_bowl.jpg", "timestamp": 3}]}
    assert normalize_evidence(native)["source_memory"] == "M_lt"
    assert evidence_frames(native)[0]["source"] == "M_lt"
    native["metadata"] = {"path": "ceramic_bowl.jpg", "source": "M_st"}
    assert normalize_evidence(native)["source_memory"] == "M_lt"


def test_cache_write_reload_resolve_and_reuse_retains_bounded_frame_provenance(tmp_path):
    cache = InteractionCache("general", tmp_path, ttl_seconds=0)
    event = normalize_evidence({"memory_id": "gardening", "canonical_segment_id": "seg_000000_000030",
        "start_time": 0, "end_time": 30, "source_memory": "M_lt",
        "keyframes": [{"path": "terracotta_pot.jpg", "timestamp": 12.123456,
                       "caption": "The gardener waters a terracotta pot", "score": 0,
                       "metadata": {"unbounded_event": "large" * 10000}}]})
    frame = event["evidence_frames"][0]
    cache.update_from_query_result("What did the gardener water?", "", {
        "answer": "The gardener watered a terracotta pot.", "selected_evidence": [event],
        "evidence_frames": [frame], "raw": {"image_paths_used": [frame["path"]]}})
    reloaded = InteractionCache("general", tmp_path, ttl_seconds=0)
    resolved = CoreferenceResolver().resolve("At that time what was visible?", reloaded, query_constraints={})
    ref = resolved["referenced_evidence_frames"][0]
    assert ref["original_source"] == "M_lt"
    assert ref["evidence_id"] == "gardening"
    assert ref["canonical_segment_id"] == "seg_000000_000030"
    assert ref["score"] == 0
    assert ref["frame_origins"][0]["source"] == "M_lt"
    assert "unbounded_event" not in str(ref)
    assert len(str(ref)) < 10000
    fused = MemoryFusion().fuse({}, {}, {"final_evidence_k": 1}, "general_qa", resolved)["final_evidence"][0]
    reused = evidence_frames(fused)[0]
    assert reused["timestamp"] == 12.123456
    assert reused["source"] == "M_cache" and reused["original_source"] == "M_lt"
    assert {origin["source"] for origin in reused["frame_origins"]} == {"M_lt", "M_cache"}
    cache.update_from_query_result("What was visible then?", "", {
        "answer": "A terracotta pot was visible.", "selected_evidence": [fused],
        "evidence_frames": [reused], "raw": {"image_paths_used": [reused["path"]]}})
    again = CoreferenceResolver().resolve("At that time what else?", InteractionCache("general", tmp_path, ttl_seconds=0), query_constraints={})
    assert again["referenced_evidence_frames"][0]["original_source"] == "M_lt"
    assert {origin["source"] for origin in again["referenced_evidence_frames"][0]["frame_origins"]} == {"M_lt", "M_cache"}


def test_cache_origin_cap_preserves_source_hops_before_repeated_native_records(tmp_path):
    cache = InteractionCache("general", tmp_path, ttl_seconds=0)
    frame = {"path": "sketchbook.jpg", "timestamp": 2, "source": "M_cache", "original_source": "M_st",
             "frame_origins": [{"source": "M_st", "original_source": "M_st", "metadata": {"score": n / 10}} for n in range(10)]
                              + [{"source": "M_cache", "original_source": "M_st", "metadata": {"path": "sketchbook.jpg"}}]}
    trimmed = cache._trim_evidence_frames([frame] * 20)
    assert len(trimmed) == 8
    assert len(trimmed[0]["frame_origins"]) == 4
    assert {origin["source"] for origin in trimmed[0]["frame_origins"]} == {"M_st", "M_cache"}


def test_cache_trim_preserves_selected_and_unselected_origin_associations(tmp_path):
    cache = InteractionCache("general", tmp_path, ttl_seconds=0)
    frame = {"path": "watering_can.jpg", "timestamp": 4, "source": "M_lt", "frame_origins": [
        {"source": "M_lt", "selection_association": "selected", "metadata": {"path": "watering_can.jpg", "timestamp": 4}},
        {"source": "M_lt", "selection_association": "unselected_candidate", "metadata": {"path": "watering_can.jpg", "timestamp": 9}},
    ]}
    trimmed = cache._trim_evidence_frames([frame])[0]
    assert [origin["selection_association"] for origin in trimmed["frame_origins"]] == ["selected", "unselected_candidate"]
    assert evidence_frames(trimmed)[0]["frame_origins"] == trimmed["frame_origins"]
