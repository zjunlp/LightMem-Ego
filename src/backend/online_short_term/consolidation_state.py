from __future__ import annotations

from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


def consolidation_state_path(session_dir: Path) -> Path:
    return session_dir / "short_term" / "consolidation_state.json"


def load_consolidation_state(session_dir: Path, session_id: str) -> dict[str, Any]:
    path = consolidation_state_path(session_dir)
    state = read_json(path, default={})
    if not isinstance(state, dict):
        state = {}
    return {
        "session_id": session_id,
        "consolidation_version": int(state.get("consolidation_version", 0) or 0),
        "last_consolidated_window_end": float(state.get("last_consolidated_window_end", 0.0) or 0.0),
        "generated_episode_count": int(state.get("generated_episode_count", 0) or 0),
        "generated_episode_ids": list(state.get("generated_episode_ids", []) or []),
        "window_to_episode": dict(state.get("window_to_episode", {}) or {}),
        "pending_ready_window_count": int(state.get("pending_ready_window_count", 0) or 0),
        "not_ready_window_count": int(state.get("not_ready_window_count", 0) or 0),
        "skipped_windows": list(state.get("skipped_windows", []) or []),
        "last_run_at": state.get("last_run_at"),
        "updated_at": state.get("updated_at"),
    }


def write_consolidation_state(session_dir: Path, state: dict[str, Any]) -> Path:
    path = consolidation_state_path(session_dir)
    current = read_json(path, default={})
    version = int((current if isinstance(current, dict) else {}).get("consolidation_version", 0) or 0)
    payload = dict(state)
    payload["consolidation_version"] = version + 1
    payload["updated_at"] = utc_now_iso()
    payload.setdefault("last_run_at", payload["updated_at"])
    write_json_atomic(path, payload)
    return path
