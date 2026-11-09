import json
from pathlib import Path


from online_short_term.stream_chunk_manager import StreamChunkManager


def test_save_stream_state_does_not_downgrade_terminal_status(tmp_path: Path) -> None:
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
