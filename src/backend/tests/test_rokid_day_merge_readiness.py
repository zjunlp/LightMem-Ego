import json
import sys
import types
from pathlib import Path


def _install_lightweight_import_stubs() -> None:
    online_memory = types.ModuleType("online_memory")
    online_memory.build_online_worldmm_memory = lambda **kwargs: Path("worldmm") / "memory_config.json"
    sys.modules.setdefault("online_memory", online_memory)

    online_preprocess = types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.relative_to_session = lambda path, session_dir: Path(path).relative_to(session_dir).as_posix()
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json_atomic = lambda path, data: _write_json(Path(path), data)
    io_utils.ffmpeg_bin = lambda: "ffmpeg"
    io_utils.ffprobe_bin = lambda: "ffprobe"
    sys.modules.setdefault("online_preprocess", online_preprocess)
    sys.modules.setdefault("online_preprocess.io_utils", io_utils)


_install_lightweight_import_stubs()

from online_pipeline.rokid_day_merge import missing_child_outputs  # noqa: E402


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_required_outputs(session_dir: Path) -> None:
    for rel_path in (
        Path("worldmm") / "mst_episodic" / "mst_30sec_episodes.json",
        Path("evidence") / "mst_session_evidence.json",
        Path("captions") / "mst_session_30sec_captioned.json",
    ):
        _write_json(session_dir / rel_path, [])


def test_missing_child_outputs_waits_for_child_pipeline_to_finish(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0001"
    _write_required_outputs(child_dir)
    _write_json(
        child_dir / "status.json",
        {
            "status": "done",
            "stage": "memory_incremental_ready",
            "outputs": {"visual_lagging": True},
        },
    )
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})

    missing = missing_child_outputs(child_dir)

    assert "status.json:stage=memory_incremental_ready" in missing


def test_missing_child_outputs_allows_visual_embedding_ready_child(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0001"
    _write_required_outputs(child_dir)
    _write_json(
        child_dir / "status.json",
        {
            "status": "done",
            "stage": "visual_embedding_ready",
            "outputs": {
                "graph_lagging": False,
                "semantic_lagging": False,
                "visual_lagging": False,
            },
        },
    )
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})

    assert missing_child_outputs(child_dir) == []


def test_missing_child_outputs_allows_ended_stream_with_ready_memory(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0002"
    _write_required_outputs(child_dir)
    _write_json(child_dir / "status.json", {"status": "stream_ended", "stage": "stream_ended"})
    _write_json(child_dir / "stream" / "stream_state.json", {"status": "ended"})
    _write_json(child_dir / "worldmm" / "memory_config.json", {
        "status": "memory_ready",
        "memory_build_state": "ready",
        "long_term_partial_ready": True,
        "visual_embedding_ready": True,
        "visual_lagging": False,
        "readiness": {"visual_ready": True},
        "lag": {"visual_lagging": False},
    })
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})

    assert missing_child_outputs(child_dir) == []


def test_missing_child_outputs_allows_ending_stream_with_ended_at_and_ready_memory(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0002"
    _write_required_outputs(child_dir)
    _write_json(child_dir / "status.json", {"status": "done", "stage": "memory_incremental_ready"})
    _write_json(child_dir / "stream" / "stream_state.json", {"status": "ending", "ended_at": "2026-07-08T00:00:00+00:00"})
    _write_json(child_dir / "worldmm" / "memory_config.json", {
        "status": "memory_ready",
        "memory_build_state": "ready",
        "long_term_partial_ready": True,
        "visual_embedding_ready": False,
        "visual_lagging": False,
        "readiness": {"visual_ready": False},
        "lag": {"visual_lagging": False},
    })
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})

    assert missing_child_outputs(child_dir) == []


def test_missing_child_outputs_waits_when_memory_ready_but_stream_still_running(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0002"
    _write_required_outputs(child_dir)
    _write_json(child_dir / "status.json", {"status": "streaming", "stage": "stream_started"})
    _write_json(child_dir / "stream" / "stream_state.json", {"status": "running"})
    _write_json(child_dir / "worldmm" / "memory_config.json", {
        "status": "memory_ready",
        "memory_build_state": "ready",
        "long_term_partial_ready": True,
        "visual_embedding_ready": True,
        "visual_lagging": False,
        "readiness": {"visual_ready": True},
        "lag": {"visual_lagging": False},
    })
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})

    missing = missing_child_outputs(child_dir)

    assert "status.json:status=streaming" in missing
    assert "status.json:stage=stream_started" in missing


def test_missing_child_outputs_waits_for_pending_refine_and_memory(tmp_path: Path) -> None:
    child_dir = tmp_path / "parent__day0001"
    _write_required_outputs(child_dir)
    _write_json(child_dir / "status.json", {"status": "done", "stage": "visual_embedding_ready"})
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 2})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 1})
    _write_json(child_dir / "worldmm" / "incremental" / "append_state.json", {"pending_count": 3, "failed_count": 0})

    missing = missing_child_outputs(child_dir)

    assert "short_term/refine/refine_state.json:pending_event_count=2" in missing
    assert "short_term/consolidation_state.json:pending_ready_window_count=1" in missing
    assert "worldmm/incremental/append_state.json:pending_count=3" in missing
