from __future__ import annotations

from typing import Any


def build_day_context_block(day_context: dict[str, Any] | None) -> str:
    if not isinstance(day_context, dict) or not day_context.get("is_rokid_day_child"):
        return ""
    day_label = str(day_context.get("day_label") or "").strip()
    day_index = day_context.get("day_index")
    child_session_id = str(day_context.get("child_session_id") or day_context.get("realtime_session_id") or "").strip()
    parent_session_id = str(day_context.get("parent_session_id") or day_context.get("long_term_session_id") or "").strip()
    run_id = str(day_context.get("run_id") or "").strip()
    display_date = str(day_context.get("display_date") or "").strip()
    display_time = str(day_context.get("display_time") or "").strip()
    display_datetime = str(day_context.get("display_datetime") or day_context.get("start_datetime") or "").strip()
    timezone = str(day_context.get("timezone") or "").strip()
    time_source = str(day_context.get("time_source") or "").strip()

    lines = ["Current Rokid day/session context:"]
    if day_label:
        lines.append(f"- current_day_label: {day_label}")
    if day_index is not None:
        lines.append(f"- current_day_index: {day_index}")
    if child_session_id:
        lines.append(f"- current_child_session_id: {child_session_id}")
    if parent_session_id:
        lines.append(f"- parent_session_id: {parent_session_id}")
    if run_id:
        lines.append(f"- current_run_id: {run_id}")
    if display_datetime:
        lines.append(f"- current_day_start_datetime: {display_datetime}")
    elif display_date or display_time:
        lines.append(f"- current_day_start_datetime: {display_date} {display_time}".strip())
    if timezone:
        lines.append(f"- timezone: {timezone}")
    if time_source:
        lines.append(f"- time_source: {time_source}")
    lines.append("Use this current day/session context as authoritative for questions about the current session or relative day references.")
    lines.append("Interpret today/yesterday/earlier/later relative to this Rokid day context and the Query Time when provided.")
    return "\n".join(lines)
