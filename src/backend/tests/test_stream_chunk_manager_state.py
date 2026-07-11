import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_stream_chunk_manager():
    online_preprocess = types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json_atomic = lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8")
    io_utils.ffmpeg_bin = lambda: "ffmpeg"
    io_utils.ffprobe_bin = lambda: "ffprobe"
    sys.modules.setdefault("online_preprocess", online_preprocess)
    sys.modules.setdefault("online_preprocess.io_utils", io_utils)

    online_short_term = types.ModuleType("online_short_term")
    online_short_term.__path__ = [str(ROOT / "online_short_term")]
    sys.modules.setdefault("online_short_term", online_short_term)

    spec = importlib.util.spec_from_file_location("stream_chunk_manager_under_test", ROOT / "online_short_term" / "stream_chunk_manager.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.StreamChunkManager


def test_save_stream_state_does_not_downgrade_terminal_status(tmp_path: Path) -> None:
    StreamChunkManager = _load_stream_chunk_manager()
    manager = StreamChunkManager(tmp_path / "session1")

    manager.save_stream_state(
        {
            "session_id": "session1",
            "stream_id": "stream1",
            "status": "ended",
            "ended_at": "2026-07-08T00:00:00+00:00",
        }
    )
    manager.save_stream_state(
        {
            "session_id": "session1",
            "stream_id": "stream1",
            "status": "ending",
            "final_chunk_index": -1,
            "stream_end_task_id": "session1_end",
        }
    )

    state = json.loads((tmp_path / "session1" / "stream" / "stream_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "ended"
    assert state["ended_at"] == "2026-07-08T00:00:00+00:00"
    assert state["final_chunk_index"] == -1
    assert state["stream_end_task_id"] == "session1_end"
