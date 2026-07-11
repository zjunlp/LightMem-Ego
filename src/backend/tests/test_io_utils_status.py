import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_io_utils():
    spec = importlib.util.spec_from_file_location("io_utils_under_test", ROOT / "online_preprocess" / "io_utils.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_status_preserves_existing_outputs_when_outputs_omitted(tmp_path: Path) -> None:
    io_utils = _load_io_utils()
    session_dir = tmp_path / "session1"
    io_utils.write_json(
        session_dir / "status.json",
        {
            "session_id": "session1",
            "status": "done",
            "stage": "visual_embedding_ready",
            "progress": 100,
            "outputs": {"visual_embedding_ready": True, "visual_lagging": False},
        },
    )

    io_utils.write_status(
        session_dir=session_dir,
        session_id="session1",
        status="stream_ended",
        stage="stream_ended",
        progress=100,
    )

    status = json.loads((session_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "stream_ended"
    assert status["stage"] == "stream_ended"
    assert status["outputs"]["visual_embedding_ready"] is True
    assert status["outputs"]["visual_lagging"] is False


def test_write_status_merges_new_outputs_with_existing_outputs(tmp_path: Path) -> None:
    io_utils = _load_io_utils()
    session_dir = tmp_path / "session1"
    io_utils.write_json(
        session_dir / "status.json",
        {
            "session_id": "session1",
            "status": "processing",
            "stage": "memory_building",
            "progress": 92,
            "outputs": {"visual_embedding_ready": False, "old_key": "kept"},
        },
    )

    io_utils.write_status(
        session_dir=session_dir,
        session_id="session1",
        status="done",
        stage="memory_incremental_ready",
        progress=100,
        outputs={"memory_ready": True},
    )

    status = json.loads((session_dir / "status.json").read_text(encoding="utf-8"))
    assert status["outputs"] == {
        "visual_embedding_ready": False,
        "old_key": "kept",
        "memory_ready": True,
    }
