from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from online_pipeline.frame_stream import frame_stream_input_mode, is_frame_stream_mode
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


ALLOWED_AUDIO_SUFFIXES = {".mp3", ".aac", ".m4a", ".wav", ".pcm", ".webm", ".opus", ".ogg"}
MIME_TO_AUDIO_SUFFIX = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/pcm": ".pcm",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_audio_mime(content_type: Any) -> str | None:
    text = str(content_type or "").strip().lower()
    if not text:
        return None
    return text.split(";", 1)[0].strip() or None


def audio_codec_from_mime(content_type: Any) -> str | None:
    text = str(content_type or "").strip().lower()
    if ";" not in text:
        return None
    for part in text.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip() == "codecs":
            return value.strip().strip('"') or None
    return None


def audio_suffix_from_format(format_hint: Any, filename: str | None = None, content_type: Any = None) -> str | None:
    candidates = []
    if format_hint:
        candidates.append(str(format_hint).strip().lower().lstrip("."))
    if filename:
        suffix = Path(str(filename)).suffix.lower().lstrip(".")
        if suffix:
            candidates.append(suffix)
    mime_base = normalize_audio_mime(content_type)
    if mime_base:
        mime_suffix = MIME_TO_AUDIO_SUFFIX.get(mime_base)
        if mime_suffix:
            candidates.append(mime_suffix.lstrip("."))
    aliases = {"mpeg": "mp3", "x-m4a": "m4a", "mp4": "m4a", "webm;codecs=opus": "webm", "ogg;codecs=opus": "ogg"}
    for item in candidates:
        normalized = aliases.get(item, item)
        suffix = f".{normalized}"
        if suffix in ALLOWED_AUDIO_SUFFIXES:
            return suffix
    return None


class AudioStreamStore:
    """Lossy realtime audio-chunk state for frame_audio_stream sessions."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.stream_dir = self.session_dir / "stream"
        self.audio_dir = self.stream_dir / "audio_chunks"
        self.tmp_dir = self.stream_dir / "tmp"
        self.state_path = self.stream_dir / "audio_state.json"
        self.events_path = self.stream_dir / "audio_events.jsonl"
        self.buffer_path = self.stream_dir / "audio_buffer_index.json"
        self.asr_state_path = self.stream_dir / "audio_asr_state.json"
        self.asr_windows_dir = self.stream_dir / "audio_asr" / "windows"
        self.lock_path = self.stream_dir / "audio_state.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(self, *, stream_id: str, input_mode: str = "frame_audio_stream") -> dict[str, Any]:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with self.lock():
            state = self._load_unlocked()
            if not state:
                state = self._empty_state(stream_id=stream_id, input_mode=input_mode)
            else:
                state["session_id"] = self.session_dir.name
                state["stream_id"] = stream_id or state.get("stream_id")
                state["input_mode"] = frame_stream_input_mode(input_mode)
                state["enabled"] = True
                state.setdefault("chunks", [])
                state["updated_at"] = utc_now_iso()
            self._save_unlocked(state)
            self.events_path.touch(exist_ok=True)
            if not self.buffer_path.exists():
                write_json_atomic(self.buffer_path, self._empty_buffer())
            if not self.asr_state_path.exists():
                write_json_atomic(self.asr_state_path, self._empty_asr_state(stream_id=stream_id))
            return dict(state)

    def load(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def register_audio_chunk(
        self,
        *,
        tmp_path: Path,
        audio_index: int,
        checksum: str,
        suffix: str,
        size_bytes: int,
        client_ts_ms: int | None,
        relative_ts_ms: int | None,
        source_ts_ms: int | None,
        timestamp_source: str | None,
        duration_ms: int | None,
        sample_rate: int | None,
        channels: int | None,
        source: str,
        mime_type: str | None = None,
        codec: str | None = None,
    ) -> dict[str, Any]:
        suffix = suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio suffix: {suffix}")
        with self.lock():
            state = self._load_unlocked()
            if not state:
                raise RuntimeError("audio stream is not initialized")
            state["received_count"] = int(state.get("received_count", 0) or 0) + 1
            latest_index = state.get("latest_audio_index")
            latest_index = int(latest_index) if latest_index is not None else None

            if latest_index is not None and int(audio_index) < latest_index:
                state["dropped_count"] = int(state.get("dropped_count", 0) or 0) + 1
                state["updated_at"] = utc_now_iso()
                self._save_unlocked(state)
                self._append_event_unlocked(
                    "audio_outdated_dropped",
                    audio_index=audio_index,
                    relative_ts_ms=relative_ts_ms,
                    duration_ms=duration_ms,
                    status="outdated_dropped",
                    metadata={"latest_audio_index": latest_index},
                )
                tmp_path.unlink(missing_ok=True)
                return {"status": "outdated_dropped", "state": dict(state)}

            if latest_index is not None and int(audio_index) == latest_index:
                existing = self._audio_record(state, audio_index)
                existing_checksum = str((existing or {}).get("checksum") or "")
                if existing_checksum and existing_checksum == checksum:
                    state["duplicate_count"] = int(state.get("duplicate_count", 0) or 0) + 1
                    status = "duplicate_ignored"
                    event_type = "audio_duplicate_ignored"
                else:
                    state["conflict_count"] = int(state.get("conflict_count", 0) or 0) + 1
                    status = "ignored_conflict"
                    event_type = "audio_ignored_conflict"
                state["updated_at"] = utc_now_iso()
                self._save_unlocked(state)
                self._append_event_unlocked(
                    event_type,
                    audio_index=audio_index,
                    relative_ts_ms=relative_ts_ms,
                    duration_ms=duration_ms,
                    status=status,
                    metadata={"latest_audio_index": latest_index},
                )
                tmp_path.unlink(missing_ok=True)
                return {"status": status, "state": dict(state)}

            audio_id = f"audio_{int(audio_index):06d}"
            final_path = self.audio_dir / f"{audio_id}{suffix}"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_path, final_path)
            saved_path = final_path.relative_to(self.session_dir).as_posix()
            rel_ms = int(relative_ts_ms if relative_ts_ms is not None else state.get("latest_relative_ts_ms") or 0)
            dur_ms = max(0, int(duration_ms or 0))
            record = {
                "audio_id": audio_id,
                "audio_index": int(audio_index),
                "client_ts_ms": client_ts_ms,
                "relative_ts_ms": rel_ms,
                "source_ts_ms": source_ts_ms,
                "timestamp_source": timestamp_source or ("client" if client_ts_ms is not None else "provided_relative" if relative_ts_ms is not None else "server_interval"),
                "duration_ms": dur_ms,
                "sample_rate": sample_rate,
                "channels": channels,
                "source": source,
                "format": suffix.lstrip("."),
                "mime_type": mime_type,
                "codec": codec,
                "checksum": checksum,
                "size_bytes": int(size_bytes),
                "decode_ok": True,
                "path": saved_path,
                "received_at": utc_now_iso(),
            }
            chunks = [item for item in state.get("chunks", []) or [] if isinstance(item, dict)]
            chunks.append(record)
            state["chunks"] = chunks[-max(1, _env_int("WORLDMM_AUDIO_STREAM_STATE_HISTORY", 500)) :]
            state["accepted_count"] = int(state.get("accepted_count", 0) or 0) + 1
            state["latest_audio_index"] = int(audio_index)
            state["latest_relative_ts_ms"] = rel_ms
            state["latest_source_ts_ms"] = source_ts_ms
            state["latest_timestamp_source"] = record["timestamp_source"]
            state["latest_duration_ms"] = dur_ms
            state["latest_audio_path"] = saved_path
            state["latest_audio_at"] = record["received_at"]
            state["total_duration_ms"] = int(state.get("total_duration_ms", 0) or 0) + dur_ms
            state["ready"] = True
            state["rolling_buffer_ready"] = True
            state["asr_ready"] = False
            state["asr_status"] = "not_started"
            state["updated_at"] = utc_now_iso()
            self._save_unlocked(state)
            buffer_state = self._update_buffer_unlocked(record)
            self._append_event_unlocked(
                "audio_chunk_received",
                audio_index=audio_index,
                relative_ts_ms=rel_ms,
                duration_ms=dur_ms,
                status="audio_chunk_received",
                metadata={
                    "saved_path": saved_path,
                    "size_bytes": int(size_bytes),
                    "format": suffix.lstrip("."),
                    "mime_type": mime_type,
                    "codec": codec,
                    "source_ts_ms": source_ts_ms,
                    "timestamp_source": record["timestamp_source"],
                },
            )
            return {"status": "audio_chunk_received", "state": dict(state), "audio": record, "buffer": buffer_state}

    def maybe_enqueue_asr_windows(self, *, project_root: Path, stream_id: str | None = None) -> dict[str, Any]:
        """Create rolling ASR tasks without doing ASR in the request path."""
        enabled = _env_bool("WORLDMM_AUDIO_ASR_ENABLED", True)
        with self.lock():
            asr_state = self._load_asr_state_unlocked()
            if not asr_state:
                asr_state = self._empty_asr_state(stream_id=stream_id or "")
            asr_state["enabled"] = bool(enabled)
            asr_state["backend"] = os.getenv("WORLDMM_AUDIO_ASR_BACKEND", asr_state.get("backend", "whisperx"))
            if not enabled:
                asr_state["asr_status"] = "not_started"
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": False, "enqueued": [], "asr_status": "not_started"}

            buffer_state = read_json(self.buffer_path, default={})
            if not isinstance(buffer_state, dict):
                buffer_state = {}
            latest_end = int(buffer_state.get("window_end_ms", 0) or 0)
            window_ms = max(1000, _env_int("WORLDMM_AUDIO_ASR_WINDOW_MS", 3000))
            hop_ms = max(500, _env_int("WORLDMM_AUDIO_ASR_HOP_MS", 3000))
            min_window_ms = max(500, _env_int("WORLDMM_AUDIO_ASR_MIN_WINDOW_MS", 2000))
            flush_min_ms = max(0, _env_int("WORLDMM_AUDIO_ASR_FLUSH_MIN_MS", 1000))
            max_window_ms = max(window_ms, _env_int("WORLDMM_AUDIO_ASR_MAX_WINDOW_MS", 4000))
            max_pending = max(1, _env_int("WORLDMM_AUDIO_ASR_MAX_PENDING_WINDOWS", 5))
            asr_state["window_ms"] = window_ms
            asr_state["hop_ms"] = hop_ms
            asr_state["min_window_ms"] = min_window_ms
            asr_state["flush_min_ms"] = flush_min_ms
            asr_state["max_window_ms"] = max_window_ms
            asr_state["max_pending_windows"] = max_pending
            last_end = int(asr_state.get("last_enqueued_window_end_ms", 0) or 0)
            asr_state["buffered_audio_ms"] = max(0, latest_end - last_end)
            if latest_end - last_end < min_window_ms:
                asr_state["asr_status"] = "not_started"
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": "not_started", "reason": "insufficient_audio_window"}

            if last_end > 0 and latest_end - last_end < hop_ms:
                asr_state["asr_status"] = self._derive_asr_status(asr_state)
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": asr_state["asr_status"], "reason": "hop_not_reached"}

            pending_count = self._pending_audio_asr_window_count(project_root)
            if pending_count >= max_pending:
                asr_state["asr_status"] = "queued"
                asr_state["pending_window_count"] = pending_count
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": "queued", "reason": "max_pending_windows"}

            selection = self._select_unconsumed_audio_window(
                buffer_state,
                last_end_ms=last_end,
                target_window_ms=window_ms,
                min_window_ms=min_window_ms,
                max_window_ms=max_window_ms,
                flush=False,
            )
            if not selection:
                asr_state["asr_status"] = self._derive_asr_status(asr_state)
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": asr_state["asr_status"], "reason": "no_complete_audio_window"}
            window_start, window_end, chunks = selection
            window_id = f"{self._asr_window_prefix(chunks)}_{window_start:09d}_{window_end:09d}"
            known_ids = {str(item.get("window_id")) for item in asr_state.get("windows", []) or [] if isinstance(item, dict)}
            if window_id in known_ids:
                asr_state["asr_status"] = self._derive_asr_status(asr_state)
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": asr_state["asr_status"], "reason": "window_already_known"}

            if not chunks:
                asr_state["asr_status"] = self._derive_asr_status(asr_state)
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": asr_state["asr_status"], "reason": "no_audio_chunks_in_window"}

            self.asr_windows_dir.mkdir(parents=True, exist_ok=True)
            output_audio_path = (self.asr_windows_dir / f"{window_id}.wav").relative_to(self.session_dir).as_posix()
            from online_preprocess.task_queue import enqueue_stream_asr_task

            task_path = enqueue_stream_asr_task(
                project_root=Path(project_root),
                session_id=self.session_dir.name,
                stream_id=stream_id or str(asr_state.get("stream_id") or ""),
                upload_chunk_id=window_id,
                upload_chunk_index=-1,
                upload_chunk_path="",
                processing_chunks=[],
                global_start_time=window_start / 1000.0,
                global_end_time=window_end / 1000.0,
                asr_backend=str(os.getenv("WORLDMM_AUDIO_ASR_BACKEND") or asr_state.get("backend") or "whisperx"),
                reason="audio_chunk_rolling_asr",
                source="audio_chunk_window",
                window_id=window_id,
                window_start_ms=window_start,
                window_end_ms=window_end,
                duration_ms=window_end - window_start,
                audio_chunk_paths=[str(item.get("path")) for item in chunks if item.get("path")],
                output_audio_path=output_audio_path,
                asr_window_path=output_audio_path,
                input_source=self._asr_input_source(chunks),
                is_flush=False,
            )
            now = utc_now_iso()
            window_record = {
                "window_id": window_id,
                "window_start_ms": window_start,
                "window_end_ms": window_end,
                "duration_ms": window_end - window_start,
                "audio_chunk_paths": [str(item.get("path")) for item in chunks if item.get("path")],
                "output_audio_path": output_audio_path,
                "asr_window_path": output_audio_path,
                "input_source": self._asr_input_source(chunks),
                "is_flush": False,
                "task_id": task_path.stem,
                "status": "queued",
                "queued_at": now,
            }
            windows = [dict(item) for item in asr_state.get("windows", []) or [] if isinstance(item, dict)]
            windows.append(window_record)
            asr_state["windows"] = windows[-max(1, _env_int("WORLDMM_AUDIO_ASR_STATE_HISTORY", 200)) :]
            asr_state["last_enqueued_window_start_ms"] = window_start
            asr_state["last_enqueued_window_end_ms"] = window_end
            asr_state["latest_window_id"] = window_id
            asr_state["latest_asr_window_duration_ms"] = window_end - window_start
            asr_state["queued_window_count"] = int(asr_state.get("queued_window_count", 0) or 0) + 1
            asr_state["asr_windows_enqueued"] = int(asr_state.get("queued_window_count", 0) or 0)
            asr_state["pending_window_count"] = pending_count + 1
            asr_state["buffered_audio_ms"] = max(0, latest_end - window_end)
            asr_state["asr_status"] = "queued"
            asr_state["updated_at"] = now
            self._save_asr_state_unlocked(asr_state)
            self._append_event_unlocked(
                "audio_asr_window_queued",
                audio_index=None,
                relative_ts_ms=window_start,
                duration_ms=window_end - window_start,
                status="queued",
                metadata={"window_id": window_id, "task_id": task_path.stem, "window_end_ms": window_end, "is_flush": False},
            )
            return {"enabled": True, "enqueued": [window_record], "asr_status": "queued", "task_path": str(task_path)}

    def flush_asr_tail(self, *, project_root: Path, stream_id: str | None = None, reason: str = "stream_stop") -> dict[str, Any]:
        """Enqueue a final short ASR window for remaining buffered audio."""
        enabled = _env_bool("WORLDMM_AUDIO_ASR_ENABLED", True)
        with self.lock():
            asr_state = self._load_asr_state_unlocked()
            if not asr_state:
                asr_state = self._empty_asr_state(stream_id=stream_id or "")
            if not enabled:
                return {"enabled": False, "enqueued": [], "asr_status": "not_started", "reason": "asr_disabled"}
            buffer_state = read_json(self.buffer_path, default={})
            if not isinstance(buffer_state, dict):
                buffer_state = {}
            latest_end = int(buffer_state.get("window_end_ms", 0) or 0)
            last_end = int(asr_state.get("last_enqueued_window_end_ms", 0) or 0)
            flush_min_ms = max(0, _env_int("WORLDMM_AUDIO_ASR_FLUSH_MIN_MS", 1000))
            max_window_ms = max(1000, _env_int("WORLDMM_AUDIO_ASR_MAX_WINDOW_MS", 4000))
            remaining_ms = max(0, latest_end - last_end)
            asr_state["flush_min_ms"] = flush_min_ms
            asr_state["buffered_audio_ms"] = remaining_ms
            if remaining_ms <= 0:
                asr_state["dropped_tail_ms"] = int(asr_state.get("dropped_tail_ms", 0) or 0)
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "reason": "no_tail_audio", "remaining_ms": remaining_ms}
            if remaining_ms < flush_min_ms:
                asr_state["dropped_tail_ms"] = int(asr_state.get("dropped_tail_ms", 0) or 0) + remaining_ms
                asr_state["tail_flush_enqueued"] = False
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                self._append_event_unlocked(
                    "audio_asr_tail_dropped",
                    audio_index=None,
                    relative_ts_ms=last_end,
                    duration_ms=remaining_ms,
                    status="dropped",
                    metadata={"reason": reason, "flush_min_ms": flush_min_ms},
                )
                return {"enabled": True, "enqueued": [], "reason": "tail_too_short", "remaining_ms": remaining_ms}
            pending_count = self._pending_audio_asr_window_count(project_root)
            max_pending = max(1, _env_int("WORLDMM_AUDIO_ASR_MAX_PENDING_WINDOWS", 5))
            asr_state["max_pending_windows"] = max_pending
            if pending_count >= max_pending:
                asr_state["asr_status"] = "queued"
                asr_state["pending_window_count"] = pending_count
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "asr_status": "queued", "reason": "max_pending_windows"}
            selection = self._select_unconsumed_audio_window(
                buffer_state,
                last_end_ms=last_end,
                target_window_ms=remaining_ms,
                min_window_ms=flush_min_ms,
                max_window_ms=max_window_ms,
                flush=True,
            )
            if not selection:
                asr_state["updated_at"] = utc_now_iso()
                self._save_asr_state_unlocked(asr_state)
                return {"enabled": True, "enqueued": [], "reason": "no_tail_chunks", "remaining_ms": remaining_ms}
            window_start, window_end, chunks = selection
            window_id = f"{self._asr_window_prefix(chunks)}_flush_{window_start:09d}_{window_end:09d}"
            known_ids = {str(item.get("window_id")) for item in asr_state.get("windows", []) or [] if isinstance(item, dict)}
            if window_id in known_ids:
                return {"enabled": True, "enqueued": [], "reason": "window_already_known", "window_id": window_id}
            self.asr_windows_dir.mkdir(parents=True, exist_ok=True)
            output_audio_path = (self.asr_windows_dir / f"{window_id}.wav").relative_to(self.session_dir).as_posix()
            from online_preprocess.task_queue import enqueue_stream_asr_task

            task_path = enqueue_stream_asr_task(
                project_root=Path(project_root),
                session_id=self.session_dir.name,
                stream_id=stream_id or str(asr_state.get("stream_id") or ""),
                upload_chunk_id=window_id,
                upload_chunk_index=-1,
                upload_chunk_path="",
                processing_chunks=[],
                global_start_time=window_start / 1000.0,
                global_end_time=window_end / 1000.0,
                asr_backend=str(os.getenv("WORLDMM_AUDIO_ASR_BACKEND") or asr_state.get("backend") or "whisperx"),
                reason="audio_chunk_rolling_asr_flush",
                source="audio_chunk_window",
                window_id=window_id,
                window_start_ms=window_start,
                window_end_ms=window_end,
                duration_ms=window_end - window_start,
                audio_chunk_paths=[str(item.get("path")) for item in chunks if item.get("path")],
                output_audio_path=output_audio_path,
                asr_window_path=output_audio_path,
                input_source=self._asr_input_source(chunks),
                is_flush=True,
            )
            now = utc_now_iso()
            window_record = {
                "window_id": window_id,
                "window_start_ms": window_start,
                "window_end_ms": window_end,
                "duration_ms": window_end - window_start,
                "audio_chunk_paths": [str(item.get("path")) for item in chunks if item.get("path")],
                "output_audio_path": output_audio_path,
                "asr_window_path": output_audio_path,
                "input_source": self._asr_input_source(chunks),
                "is_flush": True,
                "task_id": task_path.stem,
                "status": "queued",
                "queued_at": now,
            }
            windows = [dict(item) for item in asr_state.get("windows", []) or [] if isinstance(item, dict)]
            windows.append(window_record)
            asr_state["windows"] = windows[-max(1, _env_int("WORLDMM_AUDIO_ASR_STATE_HISTORY", 200)) :]
            asr_state["last_enqueued_window_start_ms"] = window_start
            asr_state["last_enqueued_window_end_ms"] = window_end
            asr_state["latest_window_id"] = window_id
            asr_state["latest_asr_window_duration_ms"] = window_end - window_start
            asr_state["queued_window_count"] = int(asr_state.get("queued_window_count", 0) or 0) + 1
            asr_state["asr_windows_enqueued"] = int(asr_state.get("queued_window_count", 0) or 0)
            asr_state["pending_window_count"] = pending_count + 1
            asr_state["tail_flush_enqueued"] = True
            asr_state["buffered_audio_ms"] = max(0, latest_end - window_end)
            asr_state["asr_status"] = "queued"
            asr_state["updated_at"] = now
            self._save_asr_state_unlocked(asr_state)
            self._append_event_unlocked(
                "audio_asr_window_queued",
                audio_index=None,
                relative_ts_ms=window_start,
                duration_ms=window_end - window_start,
                status="queued",
                metadata={"window_id": window_id, "task_id": task_path.stem, "window_end_ms": window_end, "is_flush": True},
            )
            return {"enabled": True, "enqueued": [window_record], "asr_status": "queued", "task_path": str(task_path), "is_flush": True}

    def mark_asr_window_started(self, window_id: str) -> dict[str, Any]:
        return self._update_asr_window_status(window_id, status="running", event_type="audio_asr_window_started")

    def mark_asr_window_done(self, window_id: str, *, segment_count: int, latest_transcript_at: str | None = None) -> dict[str, Any]:
        with self.lock():
            state = self._load_asr_state_unlocked()
            now = utc_now_iso()
            self._set_window_status(state, window_id, "done", now)
            try:
                window_end = int(str(window_id).split("_")[-1])
            except Exception:
                window_end = int(state.get("last_completed_window_end_ms", 0) or 0)
            state["last_completed_window_end_ms"] = max(int(state.get("last_completed_window_end_ms", 0) or 0), window_end)
            state["completed_window_count"] = int(state.get("completed_window_count", 0) or 0) + 1
            state["asr_windows_completed"] = int(state.get("completed_window_count", 0) or 0)
            state["pending_window_count"] = max(0, int(state.get("pending_window_count", 0) or 0) - 1)
            state["latest_window_id"] = window_id
            state["latest_transcript_at"] = latest_transcript_at or now
            state["transcript_segment_count"] = int(segment_count)
            state["asr_ready"] = True
            state["asr_status"] = self._derive_asr_status(state)
            state["updated_at"] = now
            self._save_asr_state_unlocked(state)
            self._append_event_unlocked(
                "audio_asr_window_done",
                audio_index=None,
                relative_ts_ms=None,
                duration_ms=None,
                status="done",
                metadata={"window_id": window_id, "segment_count": int(segment_count)},
            )
            return state

    def mark_asr_window_failed(self, window_id: str, error: str) -> dict[str, Any]:
        with self.lock():
            state = self._load_asr_state_unlocked()
            now = utc_now_iso()
            self._set_window_status(state, window_id, "failed", now, error=error)
            state["failed_window_count"] = int(state.get("failed_window_count", 0) or 0) + 1
            state["asr_windows_failed"] = int(state.get("failed_window_count", 0) or 0)
            state["pending_window_count"] = max(0, int(state.get("pending_window_count", 0) or 0) - 1)
            state["latest_window_id"] = window_id
            state["last_error"] = error
            state["asr_ready"] = int(state.get("completed_window_count", 0) or 0) > 0
            state["asr_status"] = self._derive_asr_status(state)
            state["updated_at"] = now
            self._save_asr_state_unlocked(state)
            self._append_event_unlocked(
                "audio_asr_window_failed",
                audio_index=None,
                relative_ts_ms=None,
                duration_ms=None,
                status="failed",
                metadata={"window_id": window_id},
                error=error,
            )
            return state

    def public_status(self, *, input_mode: Any = None) -> dict[str, Any]:
        state = self.load()
        mode = frame_stream_input_mode(input_mode or state.get("input_mode"))
        enabled = _env_bool("WORLDMM_AUDIO_STREAM_ENABLED", True) and is_frame_stream_mode(mode)
        if not state or not enabled:
            return {
                "enabled": bool(enabled),
                "input_mode": mode,
                "ready": False,
                "latest_audio_index": None,
                "latest_audio_at": None,
                "rolling_buffer_ready": False,
                "asr_ready": False,
                "asr_status": "not_started",
            }
        fields = (
            "enabled",
            "input_mode",
            "ready",
            "received_count",
            "accepted_count",
            "dropped_count",
            "duplicate_count",
            "conflict_count",
            "latest_audio_index",
            "latest_relative_ts_ms",
            "latest_source_ts_ms",
            "latest_timestamp_source",
            "latest_duration_ms",
            "latest_audio_path",
            "latest_audio_at",
            "total_duration_ms",
            "rolling_buffer_ready",
            "asr_ready",
            "asr_status",
            "updated_at",
        )
        status = {key: state.get(key) for key in fields}
        asr_state = self._load_asr_state()
        if asr_state:
            transcript_state = read_json(self.session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
            if not isinstance(transcript_state, dict):
                transcript_state = {}
            status.update(
                {
                    "asr_ready": bool(asr_state.get("asr_ready")),
                    "asr_status": asr_state.get("asr_status", status.get("asr_status", "not_started")),
                    "asr_backend": asr_state.get("backend"),
                    "asr_window_ms": asr_state.get("window_ms"),
                    "asr_hop_ms": asr_state.get("hop_ms"),
                    "asr_min_window_ms": asr_state.get("min_window_ms"),
                    "asr_flush_min_ms": asr_state.get("flush_min_ms"),
                    "asr_max_window_ms": asr_state.get("max_window_ms"),
                    "asr_max_pending_windows": asr_state.get("max_pending_windows"),
                    "buffered_audio_ms": asr_state.get("buffered_audio_ms", 0),
                    "last_enqueued_window_start_ms": asr_state.get("last_enqueued_window_start_ms"),
                    "last_enqueued_window_end_ms": asr_state.get("last_enqueued_window_end_ms"),
                    "latest_asr_window_duration_ms": asr_state.get("latest_asr_window_duration_ms"),
                    "asr_windows_enqueued": asr_state.get("asr_windows_enqueued", asr_state.get("queued_window_count", 0)),
                    "asr_windows_completed": asr_state.get("asr_windows_completed", asr_state.get("completed_window_count", 0)),
                    "asr_windows_failed": asr_state.get("asr_windows_failed", asr_state.get("failed_window_count", 0)),
                    "tail_flush_enqueued": bool(asr_state.get("tail_flush_enqueued", False)),
                    "dropped_tail_ms": asr_state.get("dropped_tail_ms", 0),
                    "queued_window_count": asr_state.get("queued_window_count", 0),
                    "completed_window_count": asr_state.get("completed_window_count", 0),
                    "failed_window_count": asr_state.get("failed_window_count", 0),
                    "pending_window_count": asr_state.get("pending_window_count", 0),
                    "latest_window_id": asr_state.get("latest_window_id"),
                    "latest_transcript_at": transcript_state.get("updated_at") or asr_state.get("latest_transcript_at"),
                    "transcript_segment_count": max(
                        int(asr_state.get("transcript_segment_count", 0) or 0),
                        int(transcript_state.get("segment_count", 0) or 0),
                    ),
                }
            )
        return status

    def _load_unlocked(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def _empty_state(self, *, stream_id: str, input_mode: str) -> dict[str, Any]:
        return {
            "session_id": self.session_dir.name,
            "stream_id": stream_id,
            "input_mode": frame_stream_input_mode(input_mode),
            "enabled": True,
            "ready": False,
            "received_count": 0,
            "accepted_count": 0,
            "dropped_count": 0,
            "duplicate_count": 0,
            "conflict_count": 0,
            "latest_audio_index": None,
            "latest_relative_ts_ms": None,
            "latest_duration_ms": None,
            "latest_audio_path": None,
            "latest_audio_at": None,
            "total_duration_ms": 0,
            "rolling_buffer_ready": False,
            "asr_ready": False,
            "asr_status": "not_started",
            "chunks": [],
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }

    def _empty_asr_state(self, *, stream_id: str) -> dict[str, Any]:
        return {
            "session_id": self.session_dir.name,
            "stream_id": stream_id,
            "enabled": _env_bool("WORLDMM_AUDIO_ASR_ENABLED", True),
            "backend": os.getenv("WORLDMM_AUDIO_ASR_BACKEND", "whisperx"),
            "window_ms": _env_int("WORLDMM_AUDIO_ASR_WINDOW_MS", 3000),
            "hop_ms": _env_int("WORLDMM_AUDIO_ASR_HOP_MS", 3000),
            "min_window_ms": _env_int("WORLDMM_AUDIO_ASR_MIN_WINDOW_MS", 2000),
            "flush_min_ms": _env_int("WORLDMM_AUDIO_ASR_FLUSH_MIN_MS", 1000),
            "max_window_ms": _env_int("WORLDMM_AUDIO_ASR_MAX_WINDOW_MS", 4000),
            "max_pending_windows": _env_int("WORLDMM_AUDIO_ASR_MAX_PENDING_WINDOWS", 5),
            "last_enqueued_window_end_ms": 0,
            "last_enqueued_window_start_ms": 0,
            "last_completed_window_end_ms": 0,
            "queued_window_count": 0,
            "completed_window_count": 0,
            "failed_window_count": 0,
            "asr_windows_enqueued": 0,
            "asr_windows_completed": 0,
            "asr_windows_failed": 0,
            "pending_window_count": 0,
            "buffered_audio_ms": 0,
            "latest_asr_window_duration_ms": None,
            "tail_flush_enqueued": False,
            "dropped_tail_ms": 0,
            "latest_window_id": None,
            "latest_transcript_at": None,
            "transcript_segment_count": 0,
            "asr_ready": False,
            "asr_status": "not_started",
            "windows": [],
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }

    def _load_asr_state(self) -> dict[str, Any]:
        payload = read_json(self.asr_state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def _load_asr_state_unlocked(self) -> dict[str, Any]:
        return self._load_asr_state()

    def _save_asr_state_unlocked(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.asr_state_path, state)

    def _audio_record(self, state: dict[str, Any], audio_index: int) -> dict[str, Any] | None:
        for item in state.get("chunks", []) or []:
            if isinstance(item, dict) and int(item.get("audio_index", -1)) == int(audio_index):
                return item
        return None

    def _empty_buffer(self) -> dict[str, Any]:
        return {"chunks": [], "window_start_ms": None, "window_end_ms": None, "chunk_count": 0, "updated_at": utc_now_iso()}

    def _update_buffer_unlocked(self, record: dict[str, Any]) -> dict[str, Any]:
        buffer_state = read_json(self.buffer_path, default={})
        if not isinstance(buffer_state, dict):
            buffer_state = self._empty_buffer()
        chunks = [item for item in buffer_state.get("chunks", []) or [] if isinstance(item, dict)]
        chunks.append(
            {
                "audio_index": record.get("audio_index"),
                "relative_ts_ms": record.get("relative_ts_ms"),
                "source_ts_ms": record.get("source_ts_ms"),
                "timestamp_source": record.get("timestamp_source"),
                "duration_ms": record.get("duration_ms"),
                "path": record.get("path"),
                "source": record.get("source"),
                "format": record.get("format"),
                "mime_type": record.get("mime_type"),
                "codec": record.get("codec"),
                "size_bytes": record.get("size_bytes"),
                "decode_ok": record.get("decode_ok", True),
            }
        )
        window_ms = max(1000, _env_int("WORLDMM_AUDIO_BUFFER_WINDOW_SECONDS", 60) * 1000)
        latest_end = int(record.get("relative_ts_ms", 0) or 0) + int(record.get("duration_ms", 0) or 0)
        window_start = max(0, latest_end - window_ms)
        chunks = [
            item
            for item in chunks
            if int(item.get("relative_ts_ms", 0) or 0) + int(item.get("duration_ms", 0) or 0) >= window_start
        ]
        kept_chunks = chunks[-max(1, _env_int("WORLDMM_AUDIO_BUFFER_MAX_CHUNKS", 120)) :]
        buffer_state = {
            "chunks": kept_chunks,
            "window_start_ms": window_start,
            "window_end_ms": latest_end,
            "chunk_count": len(kept_chunks),
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.buffer_path, buffer_state)
        return buffer_state

    def _audio_chunks_for_window(self, buffer_state: dict[str, Any], window_start_ms: int, window_end_ms: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for item in buffer_state.get("chunks", []) or []:
            if not isinstance(item, dict):
                continue
            start = int(item.get("relative_ts_ms", 0) or 0)
            end = start + int(item.get("duration_ms", 0) or 0)
            if max(start, int(window_start_ms)) < min(end, int(window_end_ms)) and item.get("path"):
                selected.append(dict(item))
        return selected

    def _select_unconsumed_audio_window(
        self,
        buffer_state: dict[str, Any],
        *,
        last_end_ms: int,
        target_window_ms: int,
        min_window_ms: int,
        max_window_ms: int,
        flush: bool,
    ) -> tuple[int, int, list[dict[str, Any]]] | None:
        chunks = [dict(item) for item in buffer_state.get("chunks", []) or [] if isinstance(item, dict) and item.get("path")]
        chunks.sort(key=lambda item: (int(item.get("relative_ts_ms", 0) or 0), int(item.get("audio_index", -1) or -1)))
        candidates: list[dict[str, Any]] = []
        for item in chunks:
            start = int(item.get("relative_ts_ms", 0) or 0)
            end = start + max(0, int(item.get("duration_ms", 0) or 0))
            if end <= int(last_end_ms):
                continue
            item["window_start_ms"] = start
            item["window_end_ms"] = end
            candidates.append(item)
        if not candidates:
            return None
        selected: list[dict[str, Any]] = []
        window_start = int(candidates[0].get("window_start_ms", 0) or 0)
        window_end = window_start
        for item in candidates:
            item_start = int(item.get("window_start_ms", 0) or 0)
            item_end = int(item.get("window_end_ms", item_start) or item_start)
            prospective_end = max(window_end, item_end)
            prospective_duration = prospective_end - window_start
            if selected and prospective_duration > max_window_ms and not flush:
                break
            selected.append(item)
            window_end = prospective_end
            if not flush and window_end - window_start >= target_window_ms:
                break
        duration = max(0, window_end - window_start)
        if duration < int(min_window_ms):
            return None
        return window_start, window_end, selected

    def _asr_input_source(self, chunks: list[dict[str, Any]]) -> str:
        sources = {str(item.get("source") or "").strip() for item in chunks if item.get("source")}
        if any("webrtc" in item for item in sources):
            return "web_webrtc_whip"
        if any("rtmp" in item or "live" in item for item in sources):
            return "live_media"
        return "audio_chunk"

    def _asr_window_prefix(self, chunks: list[dict[str, Any]]) -> str:
        input_source = self._asr_input_source(chunks)
        return "live_audio_asr" if input_source in {"web_webrtc_whip", "live_media"} else "audio_asr"

    def _pending_audio_asr_window_count(self, project_root: Path) -> int:
        from online_preprocess.io_utils import read_json
        from online_preprocess.task_queue import ensure_queue_dirs

        dirs = ensure_queue_dirs(Path(project_root))
        count = 0
        for key in ("stream_asr_queued", "stream_asr_in_progress"):
            for path in dirs[key].glob(f"{self.session_dir.name}_*.json"):
                payload = read_json(path, default={})
                if isinstance(payload, dict) and payload.get("source") == "audio_chunk_window":
                    count += 1
        return count

    def _derive_asr_status(self, state: dict[str, Any]) -> str:
        pending = int(state.get("pending_window_count", 0) or 0)
        completed = int(state.get("completed_window_count", 0) or 0)
        failed = int(state.get("failed_window_count", 0) or 0)
        if pending > 0:
            return "queued"
        if completed > 0:
            return "ready"
        if failed > 0:
            return "failed"
        return "not_started"

    def _set_window_status(self, state: dict[str, Any], window_id: str, status: str, now: str, error: str | None = None) -> None:
        windows = []
        found = False
        for item in state.get("windows", []) or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if str(item.get("window_id") or "") == str(window_id):
                item["status"] = status
                item[f"{status}_at"] = now
                if error:
                    item["error"] = error
                elif status in {"queued", "running", "done"}:
                    item.pop("error", None)
                    item.pop("failed_at", None)
                found = True
            windows.append(item)
        if not found:
            payload = {"window_id": window_id, "status": status, f"{status}_at": now}
            if error:
                payload["error"] = error
            windows.append(payload)
        state["windows"] = windows[-max(1, _env_int("WORLDMM_AUDIO_ASR_STATE_HISTORY", 200)) :]

    def _update_asr_window_status(self, window_id: str, *, status: str, event_type: str) -> dict[str, Any]:
        with self.lock():
            state = self._load_asr_state_unlocked()
            now = utc_now_iso()
            self._set_window_status(state, window_id, status, now)
            state["latest_window_id"] = window_id
            state["asr_status"] = status
            state["updated_at"] = now
            self._save_asr_state_unlocked(state)
            self._append_event_unlocked(
                event_type,
                audio_index=None,
                relative_ts_ms=None,
                duration_ms=None,
                status=status,
                metadata={"window_id": window_id},
            )
            return state

    def _append_event_unlocked(
        self,
        event_type: str,
        *,
        audio_index: int | None,
        relative_ts_ms: int | None,
        duration_ms: int | None,
        status: str,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_type": event_type,
            "session_id": self.session_dir.name,
            "audio_index": audio_index,
            "relative_ts_ms": relative_ts_ms,
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
            "created_at": utc_now_iso(),
        }
        if metadata:
            payload.update(metadata)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def public_audio_stream_status_block(session_dir: Path, *, input_mode: Any = None) -> dict[str, Any]:
    return AudioStreamStore(Path(session_dir)).public_status(input_mode=input_mode)
