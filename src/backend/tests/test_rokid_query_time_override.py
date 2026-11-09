import json
from pathlib import Path

from online_query import query_engine


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_rokid_query_time_override_uses_latest_runtime_relative_time(tmp_path: Path) -> None:
    _write_json(tmp_path / "stream" / "rokid_state.json", {"latest_frame_relative_ts_ms": 90_000})
    day_context = {
        "day_label": "DAY2",
        "display_day_label": "DAY2 周二",
        "weekday_label": "周二",
        "relative_ts_base_ms": 60_001,
    }

    override = query_engine._rokid_query_time_override(day_context, tmp_path)

    assert override["until_date"] == "DAY2"
    assert override["until_time"] == "00003000"
    assert override["relative_seconds"] == 90.0
    assert override["local_day_seconds"] == 29.999
