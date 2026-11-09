"""Resolve explicitly bound demo videos without leaving their session."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .session_labels import session_labels


def _finite_number(value: Any, *, nonnegative: bool = False) -> int | float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return value if math.isfinite(value) and (not nonnegative or value >= 0) else None
    except OverflowError:
        return None


def load_demo_media(sessions_root: Path, session_id: str) -> tuple[dict[str, Any], Path | None]:
    """Return public metadata and a safe MP4 path; never mutate a session."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise ValueError("invalid session_id")
    root = Path(sessions_root).resolve()
    session = (root / session_id).resolve()
    try:
        session.relative_to(root)
    except ValueError as exc:
        raise ValueError("session path escapes sessions directory") from exc
    if not session.is_dir():
        raise FileNotFoundError("session not found")

    result: dict[str, Any] = {
        "status": "ok", "session_id": session_id, "title": session_id,
        "available": False, "video_url": None, "video_version": None, "duration_seconds": None,
        "time_origin": "session_relative_seconds", "view_description": "",
        "video_start_seconds": None, "time_alignment_known": False, "memory_until_seconds": None,
        "reason": "video_not_found",
    }
    metadata_path = session / "metadata.json"
    try:
        metadata_path.resolve().relative_to(session)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError("invalid metadata")
    except (OSError, ValueError):
        result["reason"] = "invalid_metadata"
        return result, None
    binding = metadata.get("demo_media", {})
    if not isinstance(binding, dict):
        result["reason"] = "invalid_media_binding"
        return result, None
    result["title"] = session_labels(session, metadata)["title"]
    for key in ("time_origin", "view_description"):
        if isinstance(binding.get(key), str) and binding[key].strip():
            result[key] = binding[key].strip()
    result["duration_seconds"] = _finite_number(binding.get("duration_seconds"), nonnegative=True)
    start = _finite_number(binding.get("video_start_seconds"))
    if start is not None and binding.get("time_origin") == "session_relative_seconds":
        result.update(video_start_seconds=start, time_alignment_known=True)
    result["memory_until_seconds"] = _finite_number(
        binding.get("memory_until_seconds", metadata.get("memory_until_seconds")), nonnegative=True
    )
    state_path = session / "stream/frame_state.json"
    try:
        state_path.resolve().relative_to(session)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        milliseconds = _finite_number(state.get("latest_relative_ts_ms"), nonnegative=True) if isinstance(state, dict) else None
        if milliseconds is not None:
            result["memory_until_seconds"] = milliseconds / 1000
    except (OSError, ValueError):
        pass

    value = binding.get("path", "input.mp4")
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        result["reason"] = "invalid_media_path"
        return result, None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
        result["reason"] = "invalid_media_path"
        return result, None
    try:
        target = (session / path).resolve()
        target.relative_to(session)
        if target.suffix.lower() != ".mp4":
            result["reason"] = "invalid_media_path"
            return result, None
        if not target.is_file():
            return result, None
    except (OSError, ValueError, RuntimeError):
        result["reason"] = "invalid_media_path"
        return result, None
    stat = target.stat()
    version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
    result.update(available=True, video_version=version,
                  video_url=f"/session/{session_id}/demo-video?v={version}", reason=None)
    return result, target
