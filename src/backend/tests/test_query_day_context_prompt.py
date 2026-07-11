import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "online_query" / "day_prompt_context.py"
SPEC = importlib.util.spec_from_file_location("day_prompt_context", MODULE_PATH)
assert SPEC is not None
day_prompt_context = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(day_prompt_context)
build_day_context_block = day_prompt_context.build_day_context_block


def test_builds_rokid_day_context_for_day_child() -> None:
    day_context = {
        "is_rokid_day_child": True,
        "day_label": "DAY3",
        "day_index": 3,
        "child_session_id": "ea2b4bb5a1c3__day0003",
        "parent_session_id": "ea2b4bb5a1c3",
        "run_id": "83d7d0b9-76d3-4ee8-8af1-466ba7339e1b",
        "display_datetime": "2026-07-08 10:03:12",
        "timezone": "Asia/Shanghai",
        "time_source": "client_device",
    }

    result = build_day_context_block(day_context)

    assert "Current Rokid day/session context:" in result
    assert "- current_day_label: DAY3" in result
    assert "- current_day_index: 3" in result
    assert "- current_day_start_datetime: 2026-07-08 10:03:12" in result
    assert "- timezone: Asia/Shanghai" in result
    assert "- time_source: client_device" in result
    assert "Interpret today/yesterday/earlier/later" in result
    assert "Use this current day/session context as authoritative" in result


def test_does_not_modify_non_day_child_query() -> None:
    assert build_day_context_block({"is_rokid_day_child": False}) == ""
    assert build_day_context_block(None) == ""
