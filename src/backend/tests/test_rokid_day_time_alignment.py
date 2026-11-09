import json
from pathlib import Path


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


from online_pipeline.rokid_day import (
    apply_demo_day_fields,
    build_rokid_time_context,
    reserve_single_session_day_run,
    resolve_query_session_context,
    rokid_display_payload_for_relative_time,
)


def test_client_device_time_context_and_relative_display_time() -> None:
    context = build_rokid_time_context(
        {
            "client_start_datetime": "2026-07-08 10:03:12",
            "client_timezone_id": "Asia/Shanghai",
            "client_timezone_offset_minutes": 480,
        }
    )
    assert context["display_date"] == "2026年7月8日"
    assert context["display_time"] == "10:03:12"
    assert context["timezone"] == "Asia/Shanghai"
    assert rokid_display_payload_for_relative_time(context, 78.4)["display_hhmmssff"] == "10043040"


def test_single_session_day_runs_increment_and_preserve_offsets(tmp_path: Path) -> None:
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    run1, _ = reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="run-1",
        input_mode="rokid_frame_audio",
        metadata={"device_type": "rokid"},
    )
    _write_json(session_dir / "stream" / "frame_state.json", {"latest_frame_index": 41, "latest_relative_ts_ms": 90_000})
    _write_json(session_dir / "stream" / "audio_state.json", {"latest_audio_index": 76, "latest_relative_ts_ms": 88_000})
    run2, _ = reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="run-2",
        input_mode="rokid_frame_audio",
        metadata={"device_type": "rokid"},
    )

    assert run1["display_day_label"] == "DAY1 周一"
    assert run1["next_frame_index"] == 0
    assert run2["display_day_label"] == "DAY2 周二"
    assert run2["next_frame_index"] == 42
    assert run2["next_audio_index"] == 77
    assert run2["start_relative_ts_ms"] == 90_001
    assert not (tmp_path / "session1__day0002").exists()


def test_run_reservation_is_idempotent_and_query_stays_in_one_session(tmp_path: Path) -> None:
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    first, _ = reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="same-run",
        input_mode="rokid_frame_audio",
        metadata={},
    )
    repeated, state = reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="same-run",
        input_mode="rokid_frame_audio",
        metadata={},
    )
    context = resolve_query_session_context("session1", tmp_path)

    assert repeated == first
    assert state["next_day_index"] == 2
    assert context["long_term_session_id"] == "session1"
    assert context["day_context"]["mode"] == "single_session"
    assert "parent_session_id" not in context
    assert "child_session_id" not in context


def test_memory_item_uses_day_for_its_relative_time(tmp_path: Path) -> None:
    session_dir = tmp_path / "session1"
    session_dir.mkdir()
    reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="run-1",
        input_mode="rokid_frame_audio",
        metadata={},
    )
    _write_json(session_dir / "stream" / "frame_state.json", {"latest_frame_index": 10, "latest_relative_ts_ms": 60_000})
    reserve_single_session_day_run(
        session_dir=session_dir,
        session_id="session1",
        run_id="run-2",
        input_mode="rokid_frame_audio",
        metadata={},
    )
    item = {"start_time": 61.0, "end_time": 70.0}
    apply_demo_day_fields(item, session_dir=session_dir, start_seconds=61.0, end_seconds=70.0)
    assert item["date"] == "DAY2"
    assert item["display_day_label"] == "DAY2 周二"
    assert item["local_start_time"] == 0.999
