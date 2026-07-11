from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def rel_to_session(session_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(session_dir))
    except ValueError:
        return str(path)


def fmt_time(value: float) -> str:
    return f"{int(round(float(value))):06d}"


def chunk_id(start: float, end: float) -> str:
    return f"chunk_{fmt_time(start)}_{fmt_time(end)}"


def event_time_id(session_id: str, start_time: float, end_time: float, boundary_index: int = 0) -> str:
    start_ms = int(round(float(start_time) * 1000))
    end_ms = int(round(float(end_time) * 1000))
    return f"mst_{session_id}_{start_ms:09d}_{end_ms:09d}_{int(boundary_index):04d}"


def effective_caption(event: dict[str, Any]) -> tuple[str, str]:
    refined = str(event.get("event_caption_refined") or "").strip()
    if refined:
        return refined, "refined"
    fast = str(event.get("event_caption_fast") or "").strip()
    if fast and "visual change is detected" not in fast.lower() and "provisional short-term" not in fast.lower():
        return fast, "fast"
    placeholder = str(event.get("event_caption_placeholder") or event.get("event_caption_fast") or "").strip()
    if placeholder:
        return placeholder, "placeholder"
    return "A provisional short-term video event is available.", "placeholder"


def build_retrieval_text(event: dict[str, Any]) -> str:
    caption, source = effective_caption(event)
    parts = []
    if source == "refined":
        parts.append(f"Refined caption: {caption}")
    elif source == "fast":
        parts.append(f"Fast caption: {caption}")
    transcript = str(event.get("transcript") or "").strip()
    if transcript:
        parts.append(f"Transcript: {transcript}")
    if source == "placeholder":
        parts.append(f"Placeholder caption: {caption}")
    objects = event.get("visual_objects") or []
    actions = event.get("main_actions") or []
    changes = event.get("state_changes") or []
    if objects:
        parts.append(f"Visual objects: {objects}")
    if actions:
        parts.append(f"Actions: {actions}")
    if changes:
        parts.append(f"State changes: {changes}")
    parts.append(f"Time: {event.get('start_time')} to {event.get('end_time')} seconds")
    parts.append(f"Boundary: {event.get('boundary_reason')} diff_score={event.get('diff_score')}")
    return "\n".join(str(part) for part in parts if part)


def mst_event_stub(
    *,
    session_id: str,
    chunk_id_value: str,
    start_time: float,
    end_time: float,
    boundary_reason: str,
    diff_score: float,
    diff_stats: dict[str, Any],
    chunk_path: str,
    boundary_index: int = 0,
) -> dict[str, Any]:
    duration = max(0.0, round(float(end_time) - float(start_time), 3))
    placeholder = (
        f"A visual change is detected between {start_time:.1f}s and {end_time:.1f}s."
        if boundary_reason == "visual_change"
        else f"The scene remains mostly stable between {start_time:.1f}s and {end_time:.1f}s."
        if boundary_reason == "low_motion"
        else f"A provisional short-term video event occurs between {start_time:.1f}s and {end_time:.1f}s."
    )
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    event = {
        "event_id": event_time_id(session_id, start_time, end_time, boundary_index),
        "session_id": session_id,
        "chunk_id": chunk_id_value,
        "start_time": round(float(start_time), 3),
        "end_time": round(float(end_time), 3),
        "duration": duration,
        "available_at": round(float(end_time), 3),
        "status": "provisional",
        "version": 1,
        "boundary_reason": boundary_reason,
        "diff_score": round(float(diff_score), 4),
        "diff_stats": diff_stats,
        "event_caption_placeholder": placeholder,
        "event_caption_fast": "",
        "event_caption_refined": None,
        "caption_source": "placeholder",
        "retrieval_text": "",
        "transcript": "",
        "transcript_segments": [],
        "transcript_version": 0,
        "transcript_source": None,
        "transcript_updated_at": None,
        "keyframes": [],
        "entities": [],
        "visual_objects": [],
        "main_actions": [],
        "state_changes": [],
        "needs_refine": False,
        "refined_stale": False,
        "stale_reason": None,
        "refine_completed_at": None,
        "refine_speed": None,
        "needs_reconsolidation": False,
        "dirty_reason": None,
        "dirty_window_id": None,
        "dirty_time_range": None,
        "source": {
            "type": "stream_frame_diff",
            "chunk_path": chunk_path,
        },
        "refine": {
            "refine_attempts": 0,
            "last_refine_at": None,
            "last_refine_queued_at": None,
            "last_refine_worker_started_at": None,
            "last_refine_completed_at": None,
            "last_refine_failed_at": None,
            "last_refine_retry_queued_at": None,
            "last_refine_error": None,
            "backend": None,
            "last_refine_task_id": None,
            "last_refine_task_reason": None,
            "refine_timeline": [],
        },
        "merged_to_long_term": False,
        "merged_episode_id": None,
        "merged_at": None,
        "confidence": 0.55,
        "created_at": now,
        "updated_at": now,
    }
    event["retrieval_text"] = build_retrieval_text(event)
    return event
