import json
import sys
import types
from pathlib import Path


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _install_lightweight_import_stubs() -> None:
    online_memory = sys.modules.get("online_memory") or types.ModuleType("online_memory")
    online_memory.build_online_em2mem_memory = lambda **kwargs: kwargs["sessions_root"] / kwargs["session_id"] / "em2mem" / "memory_config.json"
    sys.modules["online_memory"] = online_memory

    online_preprocess = sys.modules.get("online_preprocess") or types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = sys.modules.get("online_preprocess.io_utils") or types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.relative_to_session = lambda path, session_dir: Path(path).relative_to(session_dir).as_posix()
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json = lambda path, data: _write_json(Path(path), data)
    io_utils.write_json_atomic = lambda path, data: _write_json(Path(path), data)
    sys.modules["online_preprocess"] = online_preprocess
    sys.modules["online_preprocess.io_utils"] = io_utils


_install_lightweight_import_stubs()

from online_pipeline.rokid_day import build_rokid_time_context, child_metadata_patch  # noqa: E402
from online_pipeline.rokid_day_merge import merge_rokid_day_child  # noqa: E402


def _write_ready_child_outputs(child_dir: Path) -> None:
    _write_json(child_dir / "status.json", {"status": "done", "stage": "visual_embedding_ready"})
    _write_json(child_dir / "short_term" / "refine" / "refine_state.json", {"pending_event_count": 0})
    _write_json(child_dir / "short_term" / "consolidation_state.json", {"pending_ready_window_count": 0})
    _write_json(child_dir / "em2mem" / "incremental" / "append_state.json", {"pending_count": 0, "failed_count": 0})
    _write_json(
        child_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json",
        [{"episode_id": "ep1", "start": 60.0, "end": 90.0, "text": "opened the laptop"}],
    )
    _write_json(
        child_dir / "evidence" / "mst_session_evidence.json",
        [{"evidence_doc_id": "ev1", "start": 60.0, "end": 90.0, "text": "laptop on desk"}],
    )
    _write_json(
        child_dir / "captions" / "mst_session_30sec_captioned.json",
        [{"doc_id": "cap1", "start": 60.0, "end": 90.0, "caption": "a laptop is visible"}],
    )


def test_merge_writes_display_time_fields_into_parent_memory(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent1"
    child_dir = tmp_path / "parent1__day0002"
    parent_dir.mkdir()
    child_dir.mkdir()
    run = {
        "parent_session_id": "parent1",
        "child_session_id": "parent1__day0002",
        "day_label": "DAY2",
        "day_index": 2,
        "run_id": "run-2",
        **build_rokid_time_context(
            {
                "client_start_datetime": "2026-07-08 10:03:12",
                "client_timezone_id": "Asia/Shanghai",
            }
        ),
    }
    _write_json(child_dir / "metadata.json", {"session_id": "parent1__day0002", **child_metadata_patch(run)})
    _write_json(parent_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json", [])
    _write_json(parent_dir / "evidence" / "mst_session_evidence.json", [])
    _write_json(parent_dir / "captions" / "mst_session_30sec_captioned.json", [])
    _write_ready_child_outputs(child_dir)

    result = merge_rokid_day_child(
        sessions_root=tmp_path,
        parent_session_id="parent1",
        child_session_id="parent1__day0002",
        day_label="DAY2",
        day_index=2,
        run_id="run-2",
    )

    assert result["status"] == "done"
    episode = json.loads((parent_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json").read_text(encoding="utf-8"))[0]
    evidence = json.loads((parent_dir / "evidence" / "mst_session_evidence.json").read_text(encoding="utf-8"))[0]
    caption = json.loads((parent_dir / "captions" / "mst_session_30sec_captioned.json").read_text(encoding="utf-8"))[0]

    for item in (episode, evidence, caption):
        assert item["date"] == "DAY2"
        assert item["start_time"] == "10041200"
        assert item["end_time"] == "10044200"
        assert item["local_start_time"] == 60.0
        assert item["display_date"] == "2026年7月8日"
        assert item["display_time_range"] == "10:04:12-10:04:42"
        assert item["display_datetime_start"] == "2026-07-08 10:04:12"
        assert item["timezone"] == "Asia/Shanghai"

    assert episode["text"].startswith("[2026年7月8日 10:04:12-10:04:42]")
    assert evidence["text"].startswith("[2026年7月8日 10:04:12-10:04:42]")
    assert caption["caption"].startswith("[2026年7月8日 10:04:12-10:04:42]")
