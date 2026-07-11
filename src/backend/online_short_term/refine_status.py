from __future__ import annotations

from pathlib import Path
from typing import Any

from online_preprocess.io_utils import utc_now_iso, write_json_atomic
from online_short_term.mst_store import MSTStore


def _window_id(start: float, end: float) -> str:
    return f"win_{int(start):06d}_{int(end):06d}"


def build_refined_ready_windows(store: MSTStore, window_seconds: float = 30.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = store.load_archive_events()
    state = store.get_state()
    last_processed = float(state.get("last_processed_time", 0.0) or 0.0)
    if not events:
        payload: list[dict[str, Any]] = []
        summary = {
            "session_id": store.session_id,
            "total_event_count": 0,
            "refined_event_count": 0,
            "pending_event_count": 0,
            "ready_30s_window_count": 0,
            "not_ready_30s_window_count": 0,
            "updated_at": utc_now_iso(),
        }
        return payload, summary

    max_end = max(float(event.get("end_time", 0.0)) for event in events)
    last_processed = max(last_processed, max_end)
    windows = []
    start = 0.0
    while start < max(max_end, last_processed) + 1e-6:
        end = start + window_seconds
        window_events = [
            event
            for event in events
            if float(event.get("start_time", 0.0)) < end and float(event.get("end_time", 0.0)) > start
        ]
        if window_events:
            refined = [
                event
                for event in window_events
                if event.get("status") in {"refined", "final"}
                and not event.get("needs_refine")
                and not event.get("refined_stale")
            ]
            failed = [event for event in window_events if event.get("status") == "refine_failed"]
            provisional = [
                event
                for event in window_events
                if event.get("status") not in {"refined", "final", "refine_failed"}
                or event.get("needs_refine")
                or event.get("refined_stale")
            ]
            closed = end <= last_processed + 1e-6
            ready = bool(closed and len(refined) == len(window_events) and window_events)
            windows.append(
                {
                    "window_id": _window_id(start, end),
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "event_ids": [event.get("event_id") for event in window_events],
                    "event_count": len(window_events),
                    "refined_event_count": len(refined),
                    "provisional_event_count": len(provisional),
                    "refine_failed_event_count": len(failed),
                    "is_refined_ready": ready,
                    "is_closed_window": closed,
                    "ready_for_30s_episodic": ready,
                }
            )
        start = end

    refined_count = sum(
        1
        for event in events
        if event.get("status") in {"refined", "final"}
        and not event.get("needs_refine")
        and not event.get("refined_stale")
    )
    ready_count = sum(1 for window in windows if window.get("ready_for_30s_episodic"))
    summary = {
        "session_id": store.session_id,
        "total_event_count": len(events),
        "refined_event_count": refined_count,
        "pending_event_count": len(events) - refined_count,
        "ready_30s_window_count": ready_count,
        "not_ready_30s_window_count": len(windows) - ready_count,
        "updated_at": utc_now_iso(),
    }
    return windows, summary


def write_refine_status(store: MSTStore) -> tuple[Path, Path]:
    windows, summary = build_refined_ready_windows(store)
    store.refine_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(store.ready_windows_path, windows)
    write_json_atomic(store.refine_state_path, summary)
    return store.ready_windows_path, store.refine_state_path
