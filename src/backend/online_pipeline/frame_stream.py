from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


ALLOWED_FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FRAME_STREAM_INPUT_MODES = {"frame_audio_stream", "rokid_frame_audio"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def frame_stream_input_mode(input_mode: Any) -> str:
    mode = str(input_mode or "chunk").strip().lower()
    return "frame_audio_stream" if mode == "frame_stream" else mode


def is_frame_stream_mode(input_mode: Any) -> bool:
    return frame_stream_input_mode(input_mode) in FRAME_STREAM_INPUT_MODES


def detect_frame_suffix(path: Path) -> str | None:
    with Path(path).open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return ".webp"
    return None


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrameStreamStore:
    """Lightweight, lossy frame-stream state used by the M_cur fast path."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.stream_dir = self.session_dir / "stream"
        self.frames_dir = self.stream_dir / "frames"
        self.tmp_dir = self.stream_dir / "tmp"
        self.state_path = self.stream_dir / "frame_state.json"
        self.events_path = self.stream_dir / "frame_events.jsonl"
        self.lock_path = self.stream_dir / "frame_state.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(
        self,
        *,
        stream_id: str,
        input_mode: str = "frame_audio_stream",
        target_fps: float | None = None,
    ) -> dict[str, Any]:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with self.lock():
            state = self._load_unlocked()
            if not state:
                state = self._empty_state(
                    stream_id=stream_id,
                    input_mode=input_mode,
                    target_fps=target_fps,
                )
            else:
                state["session_id"] = self.session_dir.name
                state["stream_id"] = stream_id or state.get("stream_id")
                state["input_mode"] = frame_stream_input_mode(input_mode)
                state["enabled"] = True
                state.setdefault("frames", [])
                state.setdefault("target_fps", target_fps or _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0))
                state.setdefault("memory_target_fps", state.get("target_fps") or _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0))
                state.setdefault("preview_target_fps", _env_float("WORLDMM_FRAME_STREAM_PREVIEW_FPS", 8.0))
                state.setdefault("preview_received_count", state.get("received_count", 0))
                state.setdefault("preview_accepted_count", state.get("accepted_count", 0))
                state.setdefault("memory_accepted_count", 0)
                state.setdefault("memory_dropped_count", 0)
                state.setdefault("preview_recent", [])
                state.setdefault("memory_recent", [])
                state["updated_at"] = utc_now_iso()
            self._save_unlocked(state)
            self.events_path.touch(exist_ok=True)
            return dict(state)

    def load(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def register_frame(
        self,
        *,
        tmp_path: Path,
        frame_index: int,
        checksum: str,
        suffix: str,
        size_bytes: int,
        client_ts_ms: int | None,
        relative_ts_ms: int | None,
        source_ts_ms: int | None,
        timestamp_source: str | None,
        width: int | None,
        height: int | None,
        source: str,
    ) -> dict[str, Any]:
        suffix = suffix.lower()
        if suffix not in ALLOWED_FRAME_SUFFIXES:
            raise ValueError(f"unsupported frame suffix: {suffix}")
        with self.lock():
            state = self._load_unlocked()
            if not state:
                raise RuntimeError("frame stream is not initialized")
            state["received_count"] = int(state.get("received_count", 0) or 0) + 1
            state["preview_received_count"] = int(state.get("preview_received_count", state.get("received_count", 0) - 1) or 0) + 1
            latest_index = state.get("latest_frame_index")
            latest_index = int(latest_index) if latest_index is not None else None
            effective_relative_ts_ms = self._effective_relative_ts_ms(state, relative_ts_ms)

            if latest_index is not None and int(frame_index) < latest_index:
                state["dropped_count"] = int(state.get("dropped_count", 0) or 0) + 1
                state["updated_at"] = utc_now_iso()
                self._save_unlocked(state)
                self._append_event_unlocked(
                    "frame_outdated_dropped",
                    frame_index=frame_index,
                    relative_ts_ms=relative_ts_ms,
                    metadata={"latest_frame_index": latest_index},
                )
                tmp_path.unlink(missing_ok=True)
                return {"status": "outdated_dropped", "state": dict(state)}

            if latest_index is not None and int(frame_index) == latest_index:
                existing = self._frame_record(state, frame_index)
                existing_checksum = str((existing or {}).get("checksum") or "")
                if existing_checksum and existing_checksum == checksum:
                    state["duplicate_count"] = int(state.get("duplicate_count", 0) or 0) + 1
                    status = "duplicate_ignored"
                    event_type = "frame_duplicate_ignored"
                else:
                    state["conflict_count"] = int(state.get("conflict_count", 0) or 0) + 1
                    status = "ignored_conflict"
                    event_type = "frame_conflict_ignored"
                state["updated_at"] = utc_now_iso()
                self._save_unlocked(state)
                self._append_event_unlocked(
                    event_type,
                    frame_index=frame_index,
                    relative_ts_ms=relative_ts_ms,
                    metadata={"latest_frame_index": latest_index},
                )
                tmp_path.unlink(missing_ok=True)
                return {"status": status, "state": dict(state)}

            frame_id = f"frame_{int(frame_index):06d}"
            final_path = self.frames_dir / f"{frame_id}{suffix}"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_path, final_path)
            saved_path = final_path.relative_to(self.session_dir).as_posix()
            memory_accepted = self._should_accept_memory_frame(state, effective_relative_ts_ms)
            record = {
                "frame_id": frame_id,
                "frame_index": int(frame_index),
                "client_ts_ms": client_ts_ms,
                "relative_ts_ms": int(effective_relative_ts_ms),
                "source_relative_ts_ms": relative_ts_ms,
                "source_ts_ms": source_ts_ms,
                "timestamp_source": timestamp_source or ("client" if client_ts_ms is not None else "provided_relative" if relative_ts_ms is not None else "server_interval"),
                "width": width,
                "height": height,
                "source": source,
                "format": suffix.lstrip("."),
                "checksum": checksum,
                "size_bytes": int(size_bytes),
                "saved_path": saved_path,
                "received_at": utc_now_iso(),
                "memory_accepted": memory_accepted,
            }
            frames = [item for item in state.get("frames", []) or [] if isinstance(item, dict)]
            frames.append(record)
            state["frames"] = frames[-max(1, _env_int("WORLDMM_FRAME_STREAM_STATE_HISTORY", 500)):]
            state["accepted_count"] = int(state.get("accepted_count", 0) or 0) + 1
            state["preview_accepted_count"] = int(state.get("preview_accepted_count", 0) or 0) + 1
            state["latest_frame_index"] = int(frame_index)
            state["latest_relative_ts_ms"] = int(effective_relative_ts_ms)
            state["latest_source_ts_ms"] = source_ts_ms
            state["latest_timestamp_source"] = record["timestamp_source"]
            state["latest_frame_path"] = saved_path
            state["latest_frame_at"] = record["received_at"]
            state["ready"] = True
            state["preview_recent"] = self._append_recent_sample(
                state.get("preview_recent"),
                frame_index=int(frame_index),
                relative_ts_ms=int(effective_relative_ts_ms),
                received_at=record["received_at"],
            )
            if memory_accepted:
                state["memory_accepted_count"] = int(state.get("memory_accepted_count", 0) or 0) + 1
                state["latest_memory_frame_index"] = int(frame_index)
                state["latest_memory_relative_ts_ms"] = int(effective_relative_ts_ms)
                state["latest_memory_frame_path"] = saved_path
                state["latest_memory_frame_at"] = record["received_at"]
                state["memory_recent"] = self._append_recent_sample(
                    state.get("memory_recent"),
                    frame_index=int(frame_index),
                    relative_ts_ms=int(effective_relative_ts_ms),
                    received_at=record["received_at"],
                )
            else:
                state["memory_dropped_count"] = int(state.get("memory_dropped_count", 0) or 0) + 1
            state["updated_at"] = utc_now_iso()
            state["frames"] = self._prune_preview_frames_unlocked(state["frames"])
            self._save_unlocked(state)
            self._append_event_unlocked(
                "frame_received",
                frame_index=frame_index,
                relative_ts_ms=effective_relative_ts_ms,
                metadata={"saved_path": saved_path, "size_bytes": int(size_bytes), "memory_accepted": memory_accepted},
            )
            return {"status": "frame_received", "state": dict(state), "frame": record}

    def mark_mcur_updated(
        self,
        *,
        frame_index: int,
        current_frame_path: str | None,
        mcur_state: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock():
            state = self._load_unlocked()
            latest_frame_index = state.get("latest_frame_index")
            if latest_frame_index is None or int(latest_frame_index) != int(frame_index):
                return state
            state["latest_current_frame_path"] = current_frame_path
            state["mcur_ready"] = bool(mcur_state.get("mcur_ready"))
            state["mcur_version"] = int(mcur_state.get("mcur_version", 0) or 0)
            state["updated_at"] = utc_now_iso()
            self._save_unlocked(state)
            self._append_event_unlocked(
                "mcur_updated_from_frame",
                frame_index=frame_index,
                relative_ts_ms=state.get("latest_relative_ts_ms"),
                metadata={
                    "current_frame_path": current_frame_path,
                    "mcur_version": state["mcur_version"],
                },
            )
            return dict(state)

    def public_status(self, *, input_mode: Any = None) -> dict[str, Any]:
        state = self.load()
        mode = frame_stream_input_mode(input_mode or state.get("input_mode"))
        if not state or not is_frame_stream_mode(mode):
            return {
                "enabled": is_frame_stream_mode(mode),
                "input_mode": mode,
                "ready": False,
                "target_fps": _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0),
                "memory_target_fps": _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0),
                "preview_target_fps": _env_float("WORLDMM_FRAME_STREAM_PREVIEW_FPS", 8.0),
                "latest_frame_index": None,
                "latest_frame_at": None,
                "mcur_ready": False,
            }
        fields = (
            "enabled",
            "input_mode",
            "ready",
            "received_count",
            "accepted_count",
            "preview_received_count",
            "preview_accepted_count",
            "memory_accepted_count",
            "memory_dropped_count",
            "dropped_count",
            "duplicate_count",
            "conflict_count",
            "latest_frame_index",
            "latest_relative_ts_ms",
            "latest_source_ts_ms",
            "latest_timestamp_source",
            "latest_frame_at",
            "latest_frame_path",
            "latest_current_frame_path",
            "latest_memory_frame_index",
            "latest_memory_relative_ts_ms",
            "latest_memory_frame_path",
            "latest_memory_frame_at",
            "mcur_ready",
            "mcur_version",
            "target_fps",
            "memory_target_fps",
            "preview_target_fps",
            "updated_at",
        )
        status = {field: state.get(field) for field in fields}
        status["preview_fps"] = self._estimate_fps(state.get("preview_recent"))
        status["memory_fps"] = self._estimate_fps(state.get("memory_recent"))
        return status

    def _load_unlocked(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def _empty_state(self, *, stream_id: str, input_mode: str, target_fps: float | None) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "session_id": self.session_dir.name,
            "stream_id": stream_id,
            "input_mode": frame_stream_input_mode(input_mode),
            "enabled": True,
            "ready": False,
            "received_count": 0,
            "accepted_count": 0,
            "preview_received_count": 0,
            "preview_accepted_count": 0,
            "memory_accepted_count": 0,
            "memory_dropped_count": 0,
            "dropped_count": 0,
            "duplicate_count": 0,
            "conflict_count": 0,
            "latest_frame_index": None,
            "latest_relative_ts_ms": None,
            "latest_frame_path": None,
            "latest_current_frame_path": None,
            "latest_frame_at": None,
            "latest_memory_frame_index": None,
            "latest_memory_relative_ts_ms": None,
            "latest_memory_frame_path": None,
            "latest_memory_frame_at": None,
            "mcur_ready": False,
            "mcur_version": 0,
            "target_fps": float(target_fps or _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0)),
            "memory_target_fps": float(target_fps or _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0)),
            "preview_target_fps": float(_env_float("WORLDMM_FRAME_STREAM_PREVIEW_FPS", 8.0)),
            "preview_recent": [],
            "memory_recent": [],
            "frames": [],
            "created_at": now,
            "updated_at": now,
        }

    def _effective_relative_ts_ms(self, state: dict[str, Any], value: int | None) -> int:
        latest = state.get("latest_relative_ts_ms")
        target_step = max(1, int(round(1000.0 / max(0.01, float(state.get("target_fps") or 1.0)))))
        if value is None:
            return 0 if latest is None else int(latest) + target_step
        parsed = max(0, int(value))
        return parsed if latest is None else max(parsed, int(latest) + 1)

    def _frame_record(self, state: dict[str, Any], frame_index: int) -> dict[str, Any] | None:
        for item in reversed(state.get("frames", []) or []):
            if isinstance(item, dict) and item.get("frame_index") is not None and int(item.get("frame_index")) == int(frame_index):
                return item
        return None

    def _should_accept_memory_frame(self, state: dict[str, Any], relative_ts_ms: int) -> bool:
        latest_memory_ts = state.get("latest_memory_relative_ts_ms")
        if latest_memory_ts is None:
            return True
        fps = max(0.01, float(state.get("memory_target_fps") or state.get("target_fps") or _env_float("WORLDMM_FRAME_STREAM_TARGET_FPS", 1.0)))
        interval_ms = max(1, int(round(1000.0 / fps)))
        return int(relative_ts_ms) - int(latest_memory_ts) >= interval_ms

    def _append_recent_sample(
        self,
        samples: Any,
        *,
        frame_index: int,
        relative_ts_ms: int,
        received_at: str,
    ) -> list[dict[str, Any]]:
        rows = [item for item in (samples or []) if isinstance(item, dict)]
        rows.append(
            {
                "frame_index": int(frame_index),
                "relative_ts_ms": int(relative_ts_ms),
                "received_at": received_at,
            }
        )
        return rows[-50:]

    def _estimate_fps(self, samples: Any) -> float | None:
        rows = [item for item in (samples or []) if isinstance(item, dict) and item.get("relative_ts_ms") is not None]
        if len(rows) < 2:
            return None
        first = int(rows[0].get("relative_ts_ms") or 0)
        last = int(rows[-1].get("relative_ts_ms") or 0)
        span_seconds = max(0.001, (last - first) / 1000.0)
        return round((len(rows) - 1) / span_seconds, 2)

    def _prune_preview_frames_unlocked(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preview_history = max(1, _env_int("WORLDMM_FRAME_STREAM_PREVIEW_HISTORY", 16))
        memory_history = max(1, _env_int("WORLDMM_FRAME_STREAM_MEMORY_HISTORY", 500))
        recent_preview = frames[-preview_history:]
        recent_memory = [item for item in frames if item.get("memory_accepted")][-memory_history:]
        keep_paths = {
            str(item.get("saved_path") or "")
            for item in [*recent_preview, *recent_memory]
            if item.get("saved_path")
        }
        pruned: list[dict[str, Any]] = []
        for item in frames:
            saved_path = str(item.get("saved_path") or "")
            if saved_path in keep_paths:
                pruned.append(item)
                continue
            if saved_path:
                try:
                    (self.session_dir / saved_path).unlink()
                except FileNotFoundError:
                    pass
        pruned.sort(key=lambda item: int(item.get("frame_index", -1) or -1))
        return pruned

    def _append_event_unlocked(
        self,
        event_type: str,
        *,
        frame_index: int,
        relative_ts_ms: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event_id": f"frame_evt_{uuid4().hex[:12]}",
            "session_id": self.session_dir.name,
            "event_type": event_type,
            "frame_index": int(frame_index),
            "relative_ts_ms": relative_ts_ms,
            "timestamp": utc_now_iso(),
            "metadata": metadata or {},
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def public_frame_stream_status_block(session_dir: Path, *, input_mode: Any = None) -> dict[str, Any]:
    return FrameStreamStore(session_dir).public_status(input_mode=input_mode)
