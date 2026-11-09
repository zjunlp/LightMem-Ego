"""Current image budgets must retain the selected core process, not old padding."""
from collections import Counter
from copy import deepcopy

import pytest

from online_query.evidence_packer import EvidencePacker


def current_input(tmp_path, end=110, core_seconds=10):
    # The upstream preference puts an outer-window reference before the process.
    offsets = [-23, -10, -4, -3, 0]
    frames = []
    for offset, diff in zip(offsets, [.26, .27, .31, .28, .12]):
        timestamp = end + offset
        path = f"current/frames/cur_kf_{timestamp * 1000:09d}.jpg"
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"distinct capture {timestamp}".encode())
        frames.append(dict(path=path, timestamp=timestamp, source="M_cur",
                           segment_id="M_cur", canonical_segment_id="M_cur",
                           diff_score=diff, role="latest" if offset == 0 else "recent"))
    by_time = {frame["timestamp"]: frame for frame in frames}
    preference = [by_time[end + offset]["path"] for offset in [0, -23, -10, -3, -4]]
    return {
        "current_context": {"current_time": end, "window_start_time": end - 29,
                            "window_end_time": end, "state": {"core_seconds": core_seconds}},
        "current_selection": {"evidence_frames": frames,
                              "selected_image_paths_for_mllm": preference},
    }


def pack(tmp_path, retrieval, *, budget=6, constraints=None):
    route = {"query_type": "current_perception", "use_image_evidence": True,
             "max_image_evidence": budget, "final_evidence_k": 6,
             "query_constraints": constraints or {"scope": "current", "requires_current": True}}
    return EvidencePacker(tmp_path).pack("现在最近有什么变化？", route, retrieval)


def sent_times(result):
    sent = set(result["selected_image_paths_for_mllm"])
    return [frame["timestamp"] for frame in result["selected_evidence_frames"] if frame["path"] in sent]


@pytest.mark.parametrize("end", [110, 940])
def test_core_change_and_before_survive_outer_reference_preference(tmp_path, monkeypatch, end):
    monkeypatch.setenv("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", "3")
    data = current_input(tmp_path, end=end)
    before = deepcopy(data)
    result = pack(tmp_path, data)
    times = sent_times(result)
    assert times[0] == end
    assert any(time in times for time in [end - 4, end - 3])
    assert end - 10 in times  # The selected pre-change observation is retained.
    assert len(times) == 3
    assert data == before


def test_current_core_uses_configured_window(tmp_path):
    data = current_input(tmp_path, core_seconds=5)
    times = sent_times(pack(tmp_path, data))
    assert set(times) == {106, 107, 110}


def test_localized_core_observation_precedes_larger_unrelated_change(tmp_path):
    data = current_input(tmp_path)
    data["current_context"]["open_event"] = {"visual_observations": [{
        "timestamp": 107, "description": "指示灯发生变化，从红色变为绿色。",
    }]}
    result = EvidencePacker(tmp_path).pack("现在指示灯有什么变化？", {
        "query_type": "current_perception", "use_image_evidence": True,
        "max_image_evidence": 3, "query_constraints": {"scope": "current", "requires_current": True},
    }, data)
    assert sent_times(result) == [110, 107, 100]


def test_unavailable_preferred_capture_leaves_room_for_core_backup(tmp_path):
    data = current_input(tmp_path)
    latest = data["current_selection"]["evidence_frames"][-1]
    (tmp_path / latest["path"]).unlink()
    assert sent_times(pack(tmp_path, data)) == [107, 106, 100]


def test_explicit_scope_still_excludes_unrequested_latest_and_process(tmp_path):
    data = current_input(tmp_path)
    result = pack(tmp_path, data, constraints={"scope": "point", "time_range": [87, 87],
                  "time_explicit": True, "requires_current": True})
    assert sent_times(result) == [87]


def test_mixed_current_history_preserves_budget_priority_and_duplicate_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", "3")
    monkeypatch.setenv("EM2MEM_QUERY_MAX_TOTAL_IMAGES", "6")
    data = current_input(tmp_path)
    history = []
    for segment, start in [("first", 10), ("second", 20)]:
        for time in range(start, start + 3):
            path = f"stream/frames/frame_{time:06d}.jpg"
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"history {time}".encode())
            history.append(dict(path=path, timestamp=time, source="M_lt", segment_id=segment,
                                canonical_segment_id=segment, score=1.))
    latest = data["current_selection"]["evidence_frames"][-1]
    duplicate = {**latest, "source": "M_cache", "evidence_id": "cached-copy",
                 "segment_id": "cached-copy", "canonical_segment_id": "cached-copy"}
    data["evidence_frames"] = history + [duplicate]
    result = pack(tmp_path, data, constraints={"scope": "history", "requires_current": True,
                                              "requires_history": True})
    sent = set(result["selected_image_paths_for_mllm"])
    frames = [frame for frame in result["selected_evidence_frames"] if frame["path"] in sent]
    assert len(sent) == 6
    assert {100, 106, 110} <= set(sent_times(result))
    assert all(count <= 3 for count in Counter(frame["canonical_segment_id"] for frame in frames).values())
    assert all(frame["canonical_segment_id"] in {"M_cur", "first"} for frame in frames)
    assert next(frame for frame in frames if frame["timestamp"] == 110).get("duplicate_provenance")


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_smaller_global_budgets_remain_strict(tmp_path, budget):
    result = pack(tmp_path, current_input(tmp_path), budget=budget)
    assert len(result["selected_image_paths_for_mllm"]) == budget
    if budget:
        assert sent_times(result)[0] == 110


@pytest.mark.parametrize("binding", [
    {"binding_valid": False}, {"anchor_status": "unresolved"},
    {"binding_valid": True, "confidence": .4}, {"reference_ambiguous": True},
])
def test_unbound_cache_cannot_filter_packed_history_to_its_old_point(binding):
    cache = {"is_followup": True, "confidence": .9, "temporal_relation": "same_time",
             "referenced_time_ranges": [{"start": 50, "end": 50}], **binding}
    event = {"evidence_id": "observed", "source_memory": "M_lt", "start_time": 10,
             "end_time": 10, "evidence_frames": [
                 {"path": "stream/frames/frame_000010.jpg", "timestamp": 10}]}
    result = EvidencePacker().pack("回到那段，当时发生了什么？", {
        "use_image_evidence": True, "max_image_evidence": 3,
        "query_constraints": {"scope": "history", "requires_history": True},
    }, {"final_evidence": [event]}, cache)
    assert sent_times(result) == [10]


def test_unanchored_current_metadata_keeps_existing_preference():
    paths = ["current/frames/cur_kf_000100000.jpg", "current/frames/cur_kf_000110000.jpg"]
    result = EvidencePacker().pack("现在有什么？", {
        "use_image_evidence": True, "max_image_evidence": 1,
        "query_constraints": {"requires_current": True},
    }, {"final_evidence": [{"evidence_id": "current", "source_memory": "M_cur", "keyframe_paths": paths}],
        "current_selection": {"selected_image_paths_for_mllm": paths}})
    assert result["selected_image_paths_for_mllm"] == paths[:1]
