from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_SESSIONS_ROOT = Path("online_sessions")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def rel_to_session(session_dir: Path, path: Path | str) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(session_dir.resolve()))
    except Exception:
        return str(value)


def frame_id(timestamp: float) -> str:
    return f"cur_frame_{int(round(float(timestamp) * 1000)):09d}"


def current_frame_name(timestamp: float) -> str:
    return f"cur_kf_{int(round(float(timestamp) * 1000)):09d}.jpg"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = compact_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    if isinstance(value, list):
        parts = [compact_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()
