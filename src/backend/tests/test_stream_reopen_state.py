import json
from pathlib import Path


from online_short_term.stream_chunk_manager import StreamChunkManager


def test_init_stream_reopens_terminal_stream_state(tmp_path: Path) -> None:
    session_dir = tmp_path / "session1"
    manager = StreamChunkManager(session_dir)
    first = manager.init_stream(metadata={"run_id": "day1"})
    assert first["status"] == "running"

    def mark_ended(state: dict) -> None:
        state["status"] = "ended"
        state["ended_at"] = "2026-07-16T00:01:00+00:00"
        state["stream_end_task_id"] = "end-task"
        state["stream_end_task_path"] = "tasks/end-task.json"
        state["final_chunk_index"] = 0
        state["final_upload_chunk_index"] = 0
        state["close_open_event"] = True

    manager.update_stream_state_locked(mark_ended)
    assert manager.load_stream_state(default={})["status"] == "ended"

    reopened = manager.init_stream(metadata={"run_id": "day2"})
    persisted = manager.load_stream_state(default={})

    assert reopened["status"] == "running"
    assert persisted["status"] == "running"
    assert persisted["ended_at"] is None
    assert persisted["stream_end_task_id"] is None
    assert persisted["stream_end_task_path"] is None
    assert persisted["final_chunk_index"] is None
    assert persisted["final_upload_chunk_index"] is None
    assert persisted["close_open_event"] is None
