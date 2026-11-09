"""Regression coverage for evidence time units, frame provenance, and event caps."""
import pytest

from online_query.evidence_packer import EvidencePacker, _normalize_frame_timestamp, _timestamp_from_frame_path
from online_query.interaction_cache import _normalize_frame_timestamp as cache_timestamp
from online_query.memory_fusion import MemoryFusion
from online_query.query_engine import _keyframe_timestamp
from online_memory_view.normalizer import MemoryNormalizer


def _event(**overrides):
    return {
        "memory_id": "mst_evd_000000_000030",
        "segment_id": "seg_000000_000030",
        "start_time": "00000000", "end_time": "00003000",
        "caption": "driver uses a phone", "score": 1.0,
        **overrides,
    }


@pytest.mark.parametrize("start,end,midpoint", [
    ("00000000", "00003000", 15.0),
    ("00003000", "00010000", 45.0),
    (30.0, 60.0, 45.0),
])
def test_long_term_fusion_decodes_timecodes(start, end, midpoint):
    fused = MemoryFusion()._from_text(_event(start_time=start, end_time=end), {"M_lt": 1.0})
    assert fused["timestamp"] == midpoint
    assert fused["start_time"] == midpoint - 15.0
    assert fused["end_time"] == midpoint + 15.0


def test_fused_event_without_visual_uses_seconds():
    fused = MemoryFusion()._from_fused({"text": _event(), "visual_items": []}, {"M_lt": 1.0})
    assert fused["timestamp"] == 15.0


def test_memory_view_decodes_timecodes_and_preserves_numeric_seconds():
    normalizer = MemoryNormalizer("test")
    episodic = normalizer.episodic(_event(), "30sec")
    visual = normalizer.visual({"start_time": "00003000", "end_time": "00010000", "timestamp": 1200.0})
    assert episodic["event_time"]["end_seconds"] == 30.0
    assert visual["event_time"] == {"start_seconds": 30.0, "end_seconds": 60.0, "timestamp_seconds": 1200.0}


def test_packing_preserves_capture_times_instead_of_event_midpoint(monkeypatch):
    monkeypatch.setenv("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", "3")
    paths = [f"frame_{n:06d}.jpg" for n in (0, 10, 20)]
    event = MemoryFusion()._from_text(_event(start_time=0.0, end_time=30.0, keyframe_paths=paths), {"M_lt": 1.0})
    frames = [{"path": p, "timestamp": float(t), "segment_id": "seg_000000_000030"}
              for p, t in zip(paths, (0, 10, 20))]
    packed = EvidencePacker().pack("show the event", {
        "use_image_evidence": True, "max_image_evidence": 9,
    }, {"final_evidence": [event], "evidence_frames": frames})
    assert {f["path"]: f["timestamp"] for f in packed["selected_evidence_frames"]} == dict(zip(paths, (0.0, 10.0, 20.0)))
    assert len(packed["selected_image_paths_for_mllm"]) == 3


def test_same_event_cannot_bypass_image_cap_via_two_ids(monkeypatch):
    monkeypatch.setenv("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", "3")
    paths = [f"frame_{n:06d}.jpg" for n in (0, 2, 4, 6, 8, 10)]
    event = MemoryFusion()._from_text(_event(keyframe_paths=paths), {"M_lt": 1.0})
    frames = [{"path": p, "timestamp": float(t), "segment_id": "seg_000000_000030"}
              for p, t in zip(paths, (0, 2, 4, 6, 8, 10))]
    packed = EvidencePacker().pack("show the event", {
        "use_image_evidence": True, "max_image_evidence": 9,
    }, {"final_evidence": [event], "evidence_frames": frames})
    assert len(packed["selected_image_paths_for_mllm"]) == 3
    assert {f["canonical_segment_id"] for f in packed["selected_evidence_frames"]} == {"seg_000000_000030"}


def test_final_evidence_recovers_per_frame_metadata_without_raw_candidates():
    paths = ["frame_000040.jpg", "frame_000080.jpg"]
    event = MemoryFusion()._from_short_term({
        "event_id": "micro_1", "start_time": 0.0, "end_time": 15.0,
        "keyframes": [{"path": paths[0], "timestamp": 5.0}, {"path": paths[1], "timestamp": 10.0}],
    }, {"M_st": 1.0})
    frames = EvidencePacker()._rank_frames([], [], final_evidence=[event])
    assert [f["timestamp"] for f in frames] == [5.0, 10.0]


def test_unknown_capture_time_stays_unknown():
    event = MemoryFusion()._from_text(_event(keyframe_paths=["frame_000040.jpg"]), {"M_lt": 1.0})
    frames = EvidencePacker()._rank_frames([], [], final_evidence=[event])
    assert frames[0]["timestamp"] is None


@pytest.mark.parametrize("normalize", [_normalize_frame_timestamp, cache_timestamp])
def test_second_metadata_survives_long_videos_and_repeated_normalization(normalize):
    path = "stream/keyframes/kf_001200000_proc_000240.jpg"
    first = normalize(1200.0, path)
    assert first == 1200.0
    assert normalize(first, path) == 1200.0
    assert _keyframe_timestamp(path, {"timestamp": 1200.0}) == 1200.0


@pytest.mark.parametrize("path,expected", [
    ("stream/keyframes/kf_000028000_proc_000005.jpg", 28.0),
    ("stream/keyframes/kf_000000500_proc_000005.jpg", 0.5),
    ("keyframes/kf_000015.jpg", 15.0),
    ("current/cur_kf_001200000.jpg", 1200.0),
    ("frames/frame_000040.jpg", None),
])
def test_filename_fallback_uses_keyframe_time_token(path, expected):
    assert _keyframe_timestamp(path) == expected
    assert _timestamp_from_frame_path(path) == expected


def test_preferred_current_images_respect_zero_image_budget():
    packed = EvidencePacker().pack("show current scene", {
        "use_image_evidence": True, "max_image_evidence": 0,
    }, {"current_selection": {
        "evidence_frames": [{"path": "current/a.jpg", "timestamp": 1.0, "source": "M_cur"}],
        "selected_image_paths_for_mllm": ["current/a.jpg"],
    }})
    assert packed["selected_image_paths_for_mllm"] == []


def test_query_latency_counts_each_long_term_phase_once(tmp_path, monkeypatch):
    """Run the real query orchestration with offline retrieval and a known phase clock."""
    from types import SimpleNamespace
    from online_query import query_engine as engine
    from retrieval_source_helpers import make_memory, EARLY_CUTOFF, UTIL
    import sys

    # Reuse the unchanged utility definitions already loaded by the offline
    # harness. Importing the parent memory package initializes unrelated model
    # backends (including torch), although this test exercises no model client.
    monkeypatch.setitem(sys.modules, "em2mem.memory.utils", UTIL)

    memory = make_memory(tmp_path / "cache")
    original_answer = memory.answer

    def timed_answer(**kwargs):
        result = original_answer(**kwargs)
        result.answer = "A past event."
        result.timing_ms = {"retrieval_ms": 100, "selector_ms": 20, "pack_ms": 3}
        return result

    monkeypatch.setattr(memory, "answer", timed_answer)
    # These helpers serialize retrieval outputs; no model or network client is
    # constructed just to exercise the surrounding query and response timing.
    helpers = SimpleNamespace(
        build_until_timestamp=lambda date, time: EARLY_CUTOFF,
        summarize_selected_events=lambda memory, ids: [{
            "doc_id": "early", "date": "DAY1", "start_time": "00000000",
            "end_time": "00003000", "text": "A past event.", "keyframe_paths": [],
        }] if ids else [],
        summarize_semantic_facts=lambda memory, ids: [],
        summarize_retrieved_items=lambda result: [],
    )
    monkeypatch.setattr(engine, "_query_rag_helpers", lambda: helpers)
    monkeypatch.setattr(engine, "_em2mem_classes", lambda *args: (None, None, None, None, str))
    loaded = engine.LoadedQueryEngine(
        session_id="s", session_dir=tmp_path / "s", memory_config_path=tmp_path / "config.json",
        memory_config={"status": "memory_ready", "latest_ready_memory_version": 1},
        em2mem_memory=memory, query_args={}, visual_evidence_data=[], semantic_path="",
        latest_ready_memory_version=1,
    )
    result = loaded.query("What happened earlier?", use_current=False, use_short_term=False,
                          use_long_term=True, use_image_evidence=False, retrieval_mode="text_only",
                          use_interaction_cache=False)
    assert result["status"] == "ok"
    assert result["latency"]["long_term_pipeline_ms"] == 123
    assert result["latency"]["long_term_rag_ms"] == 100
    assert result["latency"]["long_term_selector_ms"] == 20
    assert result["latency"]["long_term_pack_ms"] == 3


@pytest.mark.parametrize("source", ["visual_index", "memory_view"])
def test_visual_records_do_not_fabricate_capture_time_from_event_start(tmp_path, source):
    import json
    from online_visual.visual_items import build_visual_items
    from online_memory_view.repository import _build_visual_items
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps([_event(
        start_time=10.0, end_time=40.0,
        keyframe_paths=["stream/keyframes/kf_000028000_proc_000005.jpg", "stream/frames/frame_000040.jpg", "stream/keyframes/kf_001200000_proc_000240.jpg"],
        keyframe_captions=[{"path": "stream/keyframes/kf_001200000_proc_000240.jpg", "timestamp": 1200.0}],
    )]))
    items = (build_visual_items(tmp_path, evidence_path=evidence) if source == "visual_index"
             else _build_visual_items(tmp_path, evidence))
    assert [item["timestamp"] for item in items] == [28.0, None, 1200.0]
