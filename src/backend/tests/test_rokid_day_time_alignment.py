import json
import sys
import types
from pathlib import Path


def _install_lightweight_import_stubs() -> None:
    online_preprocess = sys.modules.get("online_preprocess") or types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = sys.modules.get("online_preprocess.io_utils") or types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json = lambda path, data: _write_json(Path(path), data)
    io_utils.write_json_atomic = lambda path, data: _write_json(Path(path), data)
    sys.modules["online_preprocess"] = online_preprocess
    sys.modules["online_preprocess.io_utils"] = io_utils


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


_install_lightweight_import_stubs()

from online_pipeline.rokid_day import (  # noqa: E402
    build_rokid_time_context,
    child_metadata_patch,
    load_rokid_day_child_metadata,
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

    assert context["start_datetime"] == "2026-07-08 10:03:12"
    assert context["display_date"] == "2026年7月8日"
    assert context["display_time"] == "10:03:12"
    assert context["timezone"] == "Asia/Shanghai"
    assert context["time_source"] == "client_device"

    display = rokid_display_payload_for_relative_time(context, 78.4)

    assert display["display_datetime"] == "2026-07-08 10:04:30"
    assert display["display_hhmmssff"] == "10043040"


def test_child_metadata_and_query_context_include_time_fields(tmp_path: Path) -> None:
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
    child_dir = tmp_path / "parent1__day0002"
    _write_json(child_dir / "metadata.json", {"session_id": "parent1__day0002", **child_metadata_patch(run)})

    metadata = load_rokid_day_child_metadata(child_dir)
    query_context = resolve_query_session_context("parent1__day0002", tmp_path)

    assert metadata is not None
    assert metadata["display_date"] == "2026年7月8日"
    assert query_context["is_rokid_day_child"] is True
    assert query_context["long_term_session_id"] == "parent1"
    assert query_context["display_datetime"] == "2026-07-08 10:03:12"
    assert query_context["timezone"] == "Asia/Shanghai"


def test_missing_client_time_falls_back_to_server_receive_time() -> None:
    context = build_rokid_time_context({})

    assert context["time_source"] == "server_receive_fallback"
    assert context["start_datetime"] == "2026-07-08 00:00:00"
    assert context["timezone"] == "UTC"
