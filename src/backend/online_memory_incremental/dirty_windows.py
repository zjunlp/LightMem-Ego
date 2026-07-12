from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


KNOWN_LEVEL_SECONDS = {
    "1min": 60,
    "3min": 180,
    "10min": 600,
    "1h": 3600,
}


def _parse_levels_env() -> dict[str, int]:
    raw = os.getenv("EM2MEM_MEMORY_LEVELS") or os.getenv("EM2MEM_INCREMENTAL_LEVELS") or ""
    levels: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, seconds = part.split(":", 1)
            try:
                levels[name.strip()] = int(float(seconds))
            except Exception:
                continue
        elif part in KNOWN_LEVEL_SECONDS:
            levels[part] = KNOWN_LEVEL_SECONDS[part]
    return levels


def infer_multiscale_levels(session_dir: Path) -> dict[str, int]:
    """Infer configured Em2Mem levels from env and existing caption files.

    The current project usually has 3min/10min/1h, but this keeps 1min or
    future levels opt-in via env or existing files.
    """

    env_levels = _parse_levels_env()
    if env_levels:
        return dict(sorted(env_levels.items(), key=lambda x: x[1]))

    caption_root = session_dir / "em2mem" / "caption_root"
    found: dict[str, int] = {}
    if caption_root.exists():
        for path in caption_root.glob(f"{session_dir.name}_*.json"):
            stem = path.stem
            suffix = stem.replace(f"{session_dir.name}_", "", 1)
            if suffix in KNOWN_LEVEL_SECONDS and suffix != "30sec":
                found[suffix] = KNOWN_LEVEL_SECONDS[suffix]
    if found:
        return dict(sorted(found.items(), key=lambda x: x[1]))

    return {"3min": 180, "10min": 600, "1h": 3600}


def window_for_range(level: str, seconds: int, start_time: float, end_time: float) -> dict[str, Any]:
    start = int(float(start_time) // seconds) * seconds
    end = start + seconds
    return {
        "level": level,
        "window_id": f"{level}_{start:06d}_{end:06d}",
        "start_time": float(start),
        "end_time": float(end),
    }


class DirtyWindowManager:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.path = self.session_dir / "em2mem" / "incremental" / "dirty_windows.json"

    def load(self) -> dict[str, Any]:
        data = read_json(self.path, default={})
        if isinstance(data, dict):
            data.setdefault("session_id", self.session_dir.name)
            data.setdefault("dirty_version", 0)
            data.setdefault("windows", [])
            return data
        return {"session_id": self.session_dir.name, "dirty_version": 0, "windows": []}

    def mark_dirty(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        state = self.load()
        windows_by_id = {
            str(item.get("window_id")): dict(item)
            for item in state.get("windows", [])
            if isinstance(item, dict) and item.get("window_id")
        }
        episode_id = str(episode.get("episode_id") or "")
        levels = infer_multiscale_levels(self.session_dir)
        touched: list[dict[str, Any]] = []
        for level, seconds in levels.items():
            window = window_for_range(level, seconds, float(episode.get("start_time") or 0.0), float(episode.get("end_time") or 0.0))
            existing = windows_by_id.get(window["window_id"], {})
            source_ids = list(existing.get("source_episode_ids") or [])
            if episode_id and episode_id not in source_ids:
                source_ids.append(episode_id)
            window.update(
                {
                    "source_episode_ids": source_ids,
                    "status": "dirty",
                    "updated_at": utc_now_iso(),
                }
            )
            windows_by_id[window["window_id"]] = window
            touched.append(window)
        state["dirty_version"] = int(state.get("dirty_version") or 0) + 1
        state["windows"] = sorted(windows_by_id.values(), key=lambda x: (x.get("level", ""), x.get("start_time", 0.0)))
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.path, state)
        return touched

    def mark_clean(self, window_ids: set[str]) -> dict[str, Any]:
        state = self.load()
        for window in state.get("windows", []):
            if isinstance(window, dict) and str(window.get("window_id")) in window_ids:
                window["status"] = "clean"
                window["updated_at"] = utc_now_iso()
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.path, state)
        return state
