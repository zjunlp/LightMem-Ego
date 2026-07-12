from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import ffmpeg_bin, ffprobe_bin, read_json, utc_now_iso, write_json_atomic
from online_short_term.schemas import chunk_id, rel_to_session


STREAM_TERMINAL_STATUSES = {"ended", "stopped", "aborted", "cancelled", "canceled", "failed"}


def _run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        message = (
            f"Command failed with code {proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        raise RuntimeError(message)
    return proc


def probe_duration(video_path: Path) -> float:
    proc = _run_checked(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse ffprobe duration for {video_path}: {proc.stdout}") from exc


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _elapsed_ms(start: Any, end: Any) -> int | None:
    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0, int(round((end_dt - start_dt).total_seconds() * 1000)))


def prepare_output_session(
    *,
    sessions_root: Path,
    source_session_id: str,
    output_session_id: str | None,
    force: bool = False,
) -> tuple[str, Path]:
    if not output_session_id or output_session_id == source_session_id:
        return source_session_id, sessions_root / source_session_id

    source_dir = sessions_root / source_session_id
    target_dir = sessions_root / output_session_id
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    input_src = source_dir / "input.mp4"
    input_dst = target_dir / "input.mp4"
    if input_src.exists() and (force or not input_dst.exists()):
        shutil.copy2(input_src, input_dst)
    transcript_src = source_dir / "preprocess" / "transcript.json"
    transcript_dst = target_dir / "preprocess" / "transcript.json"
    if transcript_src.exists() and (force or not transcript_dst.exists()):
        transcript_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(transcript_src, transcript_dst)
    return output_session_id, target_dir


class StreamChunkManager:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.stream_dir = session_dir / "stream"
        self.chunks_dir = self.stream_dir / "chunks"
        self.upload_chunks_dir = self.stream_dir / "upload_chunks"
        self.processed_chunks_path = self.stream_dir / "processed_chunks.jsonl"
        self.stream_state_path = self.stream_dir / "stream_state.json"
        self.stream_state_lock_path = self.stream_dir / "stream_state.lock"
        self.event_state_path = self.stream_dir / "event_state.json"
        self.tmp_dir = self.stream_dir / "tmp"
        self.state_dir = self.stream_dir / "state"

    @contextmanager
    def stream_state_lock(self, *, timeout_seconds: float = 30.0):
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with self.stream_state_lock_path.open("a+") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - start >= timeout_seconds:
                        print(
                            f"[stream_state] lock timeout session_id={self.session_dir.name} "
                            f"lock={self.stream_state_lock_path}",
                            flush=True,
                        )
                        raise TimeoutError(f"timed out waiting for stream_state lock: {self.stream_state_lock_path}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_stream_state_unlocked(self, default: Any = None) -> dict[str, Any]:
        state = read_json(self.stream_state_path, default=default if default is not None else {})
        return state if isinstance(state, dict) else {}

    def split_input_video(
        self,
        *,
        chunk_seconds: float = 10.0,
        max_chunks: int | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        input_video = self.session_dir / "input.mp4"
        if not input_video.exists():
            raise FileNotFoundError(f"input.mp4 not found: {input_video}")
        if force and self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir)
        if force and self.processed_chunks_path.exists():
            self.processed_chunks_path.unlink()
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        duration = probe_duration(input_video)
        chunks = []
        start = 0.0
        idx = 0
        while start < duration - 1e-3:
            if max_chunks is not None and idx >= max_chunks:
                break
            end = min(duration, start + float(chunk_seconds))
            cid = chunk_id(start, end)
            out_path = self.chunks_dir / f"{cid}.mp4"
            if force or not out_path.exists():
                if out_path.exists():
                    out_path.unlink()
                _run_checked(
                    [
                        ffmpeg_bin(),
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-i",
                        str(input_video),
                        "-t",
                        f"{end - start:.3f}",
                        "-c",
                        "copy",
                        str(out_path),
                    ]
                )
            chunks.append(
                {
                    "chunk_id": cid,
                    "chunk_index": idx,
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "duration": round(end - start, 3),
                    "path": rel_to_session(self.session_dir, out_path),
                    "created_at": utc_now_iso(),
                }
            )
            start = end
            idx += 1
        self._write_stream_state(duration=duration, chunks=chunks)
        return chunks

    def init_stream(
        self,
        *,
        stream_id: str | None = None,
        chunk_duration: float = 5.0,
        metadata: dict[str, Any] | None = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        if reset and self.stream_dir.exists():
            shutil.rmtree(self.stream_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.upload_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        existing = self.load_stream_state(default={})
        if existing and not reset:
            existing.setdefault("session_id", self.session_dir.name)
            existing.setdefault("stream_id", stream_id or existing.get("stream_id") or uuid4().hex[:12])
            previous_status = existing.get("status")
            terminal_keys = (
                "ended_at",
                "stream_end_task_id",
                "stream_end_task_path",
                "final_chunk_index",
                "final_upload_chunk_index",
                "close_open_event",
            )
            should_reopen = previous_status in {None, "", "created", "ending", "ended"} or any(
                existing.get(key) is not None for key in terminal_keys
            )
            existing["status"] = "running" if should_reopen else previous_status
            if should_reopen:
                for key in terminal_keys:
                    existing[key] = None
            existing["chunk_duration"] = float(existing.get("chunk_duration") or chunk_duration)
            existing["processing_chunk_seconds"] = float(existing.get("processing_chunk_seconds") or chunk_duration)
            existing.setdefault("upload_chunks", existing.get("received_chunks", []))
            existing.setdefault("processing_chunks", [])
            existing.setdefault("processed_chunks", [])
            existing.setdefault("failed_chunks", [])
            existing.setdefault("missing_chunks", [])
            existing.setdefault("duplicate_chunks", [])
            existing.setdefault("conflict_chunks", [])
            existing.setdefault("retry_required_chunks", [])
            existing.setdefault("waiting_chunks", [])
            existing.setdefault("latency", {})
            existing.setdefault("next_expected_upload_chunk_index", 0)
            existing.setdefault("next_expected_proc_index", existing.get("next_expected_chunk_index", 0))
            existing.setdefault("last_processed_proc_index", existing.get("last_processed_chunk_index", -1))
            existing.setdefault("stream_timeline_end", 0.0)
            existing["updated_at"] = now
            self.save_stream_state(existing)
            if not self.event_state_path.exists():
                self.save_event_state(self.empty_event_state(existing["stream_id"]))
            return existing
        state = {
            "session_id": self.session_dir.name,
            "stream_id": stream_id or uuid4().hex[:12],
            "status": "running",
            "chunk_duration": float(chunk_duration),
            "processing_chunk_seconds": float(chunk_duration),
            "metadata": metadata or {},
            "received_chunks": [],
            "upload_chunks": [],
            "processing_chunks": [],
            "processed_chunks": [],
            "failed_chunks": [],
            "missing_chunks": [],
            "duplicate_chunks": [],
            "conflict_chunks": [],
            "retry_required_chunks": [],
            "waiting_chunks": [],
            "latency": {},
            "next_expected_upload_chunk_index": 0,
            "next_expected_proc_index": 0,
            "next_expected_chunk_index": 0,
            "last_processed_proc_index": -1,
            "last_processed_chunk_index": -1,
            "stream_timeline_end": 0.0,
            "created_at": now,
            "updated_at": now,
        }
        state["chunks"] = state["received_chunks"]
        self.save_stream_state(state)
        self.save_event_state(self.empty_event_state(state["stream_id"]))
        return state

    @property
    def keyframes_dir(self) -> Path:
        return self.stream_dir / "keyframes"

    def empty_event_state(self, stream_id: str | None = None) -> dict[str, Any]:
        return {
            "session_id": self.session_dir.name,
            "stream_id": stream_id,
            "event_seq": 0,
            "last_candidate_frame": None,
            "last_boundary_time": None,
            "open_event": None,
            "updated_at": utc_now_iso(),
        }

    def load_stream_state(self, default: Any = None) -> dict[str, Any]:
        state = read_json(self.stream_state_path, default=default if default is not None else {})
        return state if isinstance(state, dict) else {}

    def save_stream_state(self, state: dict[str, Any]) -> None:
        with self.stream_state_lock():
            current = self._load_stream_state_unlocked(default={})
            payload = self._merge_stream_state(current, state) if current else dict(state)
            self._save_stream_state_unlocked(payload)

    def update_stream_state_locked(self, mutator) -> tuple[dict[str, Any], Any]:
        """Read latest stream_state, apply a mutation, and write it under one lock."""
        with self.stream_state_lock():
            state = self._load_stream_state_unlocked(default={})
            result = mutator(state)
            self._save_stream_state_unlocked(state)
            return self._load_stream_state_unlocked(default={}), result

    def _save_stream_state_unlocked(self, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload["received_chunks"] = sorted(
            [dict(item) for item in payload.get("received_chunks", []) if isinstance(item, dict)],
            key=lambda item: int(item.get("chunk_index", 0) or 0),
        )
        payload["upload_chunks"] = sorted(
            [dict(item) for item in payload.get("upload_chunks", payload.get("received_chunks", [])) if isinstance(item, dict)],
            key=lambda item: int(item.get("upload_chunk_index", item.get("chunk_index", 0)) or 0),
        )
        payload["processing_chunks"] = sorted(
            [dict(item) for item in payload.get("processing_chunks", []) if isinstance(item, dict)],
            key=lambda item: int(item.get("proc_index", item.get("chunk_index", 0)) or 0),
        )
        payload["chunks"] = payload["processing_chunks"] or payload["received_chunks"]
        payload["missing_chunks"] = self._compute_missing_chunks(payload)
        payload["conflict_chunks"] = [dict(item) for item in payload.get("conflict_chunks", []) if isinstance(item, dict)][-100:]
        payload["duplicate_chunks"] = [dict(item) for item in payload.get("duplicate_chunks", []) if isinstance(item, dict)][-100:]
        payload["waiting_chunks"] = self._compute_waiting_chunks(payload)
        payload["retry_required_chunks"] = self._compute_retry_required_chunks(payload)
        payload["failed_chunks"] = sorted(set(int(x) for x in payload.get("failed_chunks", []) or [] if str(x).lstrip("-").isdigit()))
        payload["latency"] = self._compute_latency(payload)
        payload["updated_at"] = utc_now_iso()
        try:
            write_json_atomic(self.stream_state_path, payload)
        except Exception as exc:
            print(f"[stream_state] write failed session_id={self.session_dir.name}: {exc}", flush=True)
            raise

    def _merge_stream_state(self, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        if not current:
            return dict(incoming)
        merged = dict(current)
        for key, value in incoming.items():
            if key in {"received_chunks", "upload_chunks", "processing_chunks"}:
                continue
            if key == "status":
                current_status = str(merged.get("status") or "").strip().lower()
                incoming_status = str(value or "").strip().lower()
                if current_status in STREAM_TERMINAL_STATUSES and incoming_status not in STREAM_TERMINAL_STATUSES:
                    continue
            merged[key] = value
        for key, id_keys in (
            ("upload_chunks", ("upload_chunk_index", "chunk_index")),
            ("received_chunks", ("upload_chunk_index", "chunk_index")),
            ("processing_chunks", ("proc_index", "chunk_index")),
        ):
            merged[key] = self._merge_record_list(
                current.get(key, []) or [],
                incoming.get(key, []) or [],
                id_keys=id_keys,
            )
        return merged

    def _merge_record_list(self, current: list[Any], incoming: list[Any], *, id_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        records: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        for source_name, rows in (("current", current), ("incoming", incoming)):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    key = next(int(row.get(id_key)) for id_key in id_keys if row.get(id_key) is not None)
                except Exception:
                    continue
                if key not in records:
                    records[key] = dict(row)
                    order.append(key)
                    continue
                previous = records[key]
                combined = dict(previous)
                for field, value in row.items():
                    if value is None:
                        continue
                    if field == "processing_chunks" and not value and previous.get("processing_chunks"):
                        continue
                    if field == "status" and source_name == "incoming":
                        previous_status = str(previous.get("status") or "")
                        incoming_status = str(value or "")
                        if previous_status == "processed" and incoming_status in {"received", "queued", "processing", "waiting_for_previous"}:
                            print(
                                f"[stream_state] merge preserved processed status session_id={self.session_dir.name} "
                                f"record={key} incoming_status={incoming_status}",
                                flush=True,
                            )
                            continue
                    combined[field] = value
                records[key] = combined
        return [records[key] for key in sorted(order)]

    def load_event_state(self) -> dict[str, Any]:
        state = read_json(self.event_state_path, default=None)
        if isinstance(state, dict) and state:
            return state
        stream_state = self.load_stream_state(default={})
        return self.empty_event_state(stream_state.get("stream_id"))

    def save_event_state(self, state: dict[str, Any]) -> None:
        payload = dict(state)
        payload.setdefault("session_id", self.session_dir.name)
        payload["updated_at"] = utc_now_iso()
        write_json_atomic(self.event_state_path, payload)

    def make_chunk_id(self, chunk_index: int, start_time: float, end_time: float) -> str:
        return (
            f"chunk_{int(chunk_index):06d}_"
            f"{int(round(float(start_time))):06d}_"
            f"{int(round(float(end_time))):06d}"
        )

    def make_processing_chunk_id(self, proc_index: int, start_time: float, end_time: float) -> str:
        return (
            f"proc_{int(proc_index):06d}_"
            f"{int(round(float(start_time) * 1000)):09d}_"
            f"{int(round(float(end_time) * 1000)):09d}"
        )

    def make_upload_chunk_id(self, upload_chunk_index: int) -> str:
        return f"upload_{int(upload_chunk_index):06d}"

    def _use_direct_upload_processing(self, state: dict[str, Any]) -> bool:
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        if not metadata:
            session_metadata = read_json(self.session_dir / "metadata.json", default={})
            if isinstance(session_metadata, dict) and isinstance(session_metadata.get("metadata"), dict):
                metadata = session_metadata.get("metadata") or {}
        mode = str(metadata.get("mode") or "").strip().lower()
        source = str(metadata.get("source") or "").strip().lower()
        return mode == "camera_live" or source == "wechat_miniprogram"

    def _processing_segment_ranges(
        self,
        *,
        actual_duration: float,
        processing_chunk_seconds: float,
        direct_upload_processing: bool,
    ) -> list[tuple[float, float]]:
        duration = max(0.0, float(actual_duration))
        if duration <= 1e-3:
            return []
        if direct_upload_processing:
            return [(0.0, duration)]
        proc_seconds = max(0.1, float(processing_chunk_seconds))
        min_tail_seconds = max(0.0, float(os.getenv("EM2MEM_STREAM_MIN_PROCESSING_TAIL_SECONDS", "1.0")))
        ranges: list[tuple[float, float]] = []
        local_start = 0.0
        while local_start < duration - 1e-3:
            local_end = min(duration, local_start + proc_seconds)
            remaining = duration - local_end
            if 1e-3 < remaining < min_tail_seconds:
                local_end = duration
            ranges.append((local_start, local_end))
            local_start = local_end
        return ranges

    def split_upload_chunk_to_processing_chunks(
        self,
        *,
        upload_chunk_path: Path,
        upload_chunk_index: int,
        checksum: str,
        size_bytes: int,
        actual_duration: float,
        processing_chunk_seconds: float,
        client_timestamp: str | None = None,
        is_last: bool = False,
    ) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        if not state:
            state = self.init_stream(chunk_duration=processing_chunk_seconds)
        upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) if isinstance(item, dict)]
        existing = next((item for item in upload_chunks if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(upload_chunk_index)), None)
        duplicate_record = {
            "upload_chunk_index": int(upload_chunk_index),
            "checksum": checksum,
            "received_at": utc_now_iso(),
            "path": rel_to_session(self.session_dir, upload_chunk_path),
        }
        if existing is not None:
            if str(existing.get("checksum") or "") == checksum:
                duplicate_record["status"] = "duplicate_ignored"
                duplicates = list(state.get("duplicate_chunks", []) or [])
                duplicates.append(duplicate_record)
                state["duplicate_chunks"] = duplicates[-100:]
                self.save_stream_state(state)
                return {"status": "duplicate_ignored", "upload_chunk": existing, "processing_chunks": existing.get("processing_chunks", []), "duplicate": True}
            duplicate_record["status"] = "duplicate_conflict"
            duplicate_record["existing_checksum"] = existing.get("checksum")
            duplicates = list(state.get("duplicate_chunks", []) or [])
            duplicates.append(duplicate_record)
            state["duplicate_chunks"] = duplicates[-100:]
            conflicts = list(state.get("conflict_chunks", []) or [])
            conflicts.append(duplicate_record)
            state["conflict_chunks"] = conflicts[-100:]
            self.save_stream_state(state)
            raise ValueError(f"upload_chunk_index {upload_chunk_index} already exists with different checksum")

        next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
        upload_status = "received" if int(upload_chunk_index) == next_upload else "waiting_for_previous"
        upload_record = {
            "upload_chunk_id": self.make_upload_chunk_id(upload_chunk_index),
            "chunk_id": self.make_upload_chunk_id(upload_chunk_index),
            "upload_chunk_index": int(upload_chunk_index),
            "chunk_index": int(upload_chunk_index),
            "path": rel_to_session(self.session_dir, upload_chunk_path),
            "checksum": checksum,
            "size_bytes": int(size_bytes),
            "upload_duration": round(float(actual_duration), 3),
            "actual_duration": round(float(actual_duration), 3),
            "stream_start_time": None,
            "stream_end_time": None,
            "status": upload_status,
            "client_timestamp": client_timestamp,
            "is_last": bool(is_last),
            "processing_chunks": [],
            "received_at": utc_now_iso(),
            "processed_at": None,
            "error": None,
        }
        upload_chunks.append(upload_record)
        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        if is_last:
            state["final_upload_chunk_index"] = int(upload_chunk_index)
            state["status"] = "ending"
        state = self._materialize_ready_upload_chunks(state, max(0.1, float(processing_chunk_seconds)))
        materialized = next(
            (
                dict(item)
                for item in state.get("upload_chunks", [])
                if isinstance(item, dict) and int(item.get("upload_chunk_index", -1)) == int(upload_chunk_index)
            ),
            upload_record,
        )
        if is_last:
            final_proc = self._derive_final_proc_index(state, int(upload_chunk_index))
            if final_proc is not None:
                state["final_chunk_index"] = final_proc
        self.save_stream_state(state)
        return {
            "status": upload_status,
            "upload_chunk": materialized,
            "processing_chunks": materialized.get("processing_chunks", []),
            "duplicate": False,
            "state": self.load_stream_state(default={}),
        }

    def register_upload_chunk(
        self,
        *,
        upload_chunk_path: Path,
        upload_chunk_index: int,
        checksum: str,
        size_bytes: int,
        client_timestamp: str | None = None,
        is_last: bool = False,
    ) -> dict[str, Any]:
        """Record an uploaded chunk without doing media work in the API process."""
        state = self.load_stream_state(default={})
        if not state:
            state = self.init_stream()
        upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) if isinstance(item, dict)]
        existing = next((item for item in upload_chunks if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(upload_chunk_index)), None)
        duplicate_record = {
            "upload_chunk_index": int(upload_chunk_index),
            "checksum": checksum,
            "received_at": utc_now_iso(),
            "path": rel_to_session(self.session_dir, upload_chunk_path),
        }
        if existing is not None:
            if str(existing.get("checksum") or "") == checksum:
                duplicate_record["status"] = "duplicate_ignored"
                duplicates = list(state.get("duplicate_chunks", []) or [])
                duplicates.append(duplicate_record)
                state["duplicate_chunks"] = duplicates[-100:]
                self.save_stream_state(state)
                return {"status": "duplicate_ignored", "upload_chunk": existing, "duplicate": True, "state": state}
            duplicate_record["status"] = "duplicate_conflict"
            duplicate_record["existing_checksum"] = existing.get("checksum")
            duplicates = list(state.get("duplicate_chunks", []) or [])
            duplicates.append(duplicate_record)
            state["duplicate_chunks"] = duplicates[-100:]
            conflicts = list(state.get("conflict_chunks", []) or [])
            conflicts.append(duplicate_record)
            state["conflict_chunks"] = conflicts[-100:]
            self.save_stream_state(state)
            raise ValueError(f"upload_chunk_index {upload_chunk_index} already exists with different checksum")

        next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
        upload_status = "received" if int(upload_chunk_index) == next_upload else "waiting_for_previous"
        upload_record = {
            "upload_chunk_id": self.make_upload_chunk_id(upload_chunk_index),
            "chunk_id": self.make_upload_chunk_id(upload_chunk_index),
            "upload_chunk_index": int(upload_chunk_index),
            "chunk_index": int(upload_chunk_index),
            "path": rel_to_session(self.session_dir, upload_chunk_path),
            "checksum": checksum,
            "size_bytes": int(size_bytes),
            "upload_duration": None,
            "actual_duration": None,
            "stream_start_time": None,
            "stream_end_time": None,
            "status": upload_status,
            "client_timestamp": client_timestamp,
            "is_last": bool(is_last),
            "processing_chunks": [],
            "received_at": utc_now_iso(),
            "processed_at": None,
            "error": None,
        }
        upload_chunks.append(upload_record)
        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        if is_last:
            state["final_upload_chunk_index"] = int(upload_chunk_index)
            state["status"] = "ending"
        self.save_stream_state(state)
        return {
            "status": upload_status,
            "upload_chunk": upload_record,
            "duplicate": False,
            "state": self.load_stream_state(default={}),
        }

    def register_upload_chunk_transaction(
        self,
        *,
        tmp_upload_path: Path,
        upload_chunk_index: int,
        checksum: str,
        size_bytes: int,
        actual_duration: float | None = None,
        client_timestamp: str | None = None,
        is_last: bool = False,
    ) -> dict[str, Any]:
        upload_chunk_id = self.make_upload_chunk_id(upload_chunk_index)
        final_path = self.upload_chunks_dir / f"{upload_chunk_id}.mp4"
        self.upload_chunks_dir.mkdir(parents=True, exist_ok=True)
        with self.stream_state_lock():
            state = self._load_stream_state_unlocked(default={})
            if not state:
                state = {
                    "session_id": self.session_dir.name,
                    "stream_id": uuid4().hex[:12],
                    "status": "running",
                    "chunk_duration": 5.0,
                    "processing_chunk_seconds": 5.0,
                    "received_chunks": [],
                    "upload_chunks": [],
                    "processing_chunks": [],
                    "processed_chunks": [],
                    "failed_chunks": [],
                    "missing_chunks": [],
                    "duplicate_chunks": [],
                    "conflict_chunks": [],
                    "retry_required_chunks": [],
                    "waiting_chunks": [],
                    "latency": {},
                    "next_expected_upload_chunk_index": 0,
                    "next_expected_proc_index": 0,
                    "next_expected_chunk_index": 0,
                    "last_processed_proc_index": -1,
                    "last_processed_chunk_index": -1,
                    "stream_timeline_end": 0.0,
                    "created_at": utc_now_iso(),
                }
            upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) if isinstance(item, dict)]
            existing = next(
                (
                    item
                    for item in upload_chunks
                    if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(upload_chunk_index)
                ),
                None,
            )
            duplicate_record = {
                "upload_chunk_index": int(upload_chunk_index),
                "checksum": checksum,
                "received_at": utc_now_iso(),
                "path": rel_to_session(self.session_dir, final_path),
            }
            if existing is not None:
                tmp_upload_path.unlink(missing_ok=True)
                if str(existing.get("checksum") or "") == checksum:
                    duplicate_record["status"] = "duplicate_ignored"
                    duplicates = list(state.get("duplicate_chunks", []) or [])
                    duplicates.append(duplicate_record)
                    state["duplicate_chunks"] = duplicates[-100:]
                    self._save_stream_state_unlocked(state)
                    return {"status": "duplicate_ignored", "upload_chunk": existing, "duplicate": True, "state": state}
                duplicate_record["status"] = "duplicate_conflict"
                duplicate_record["existing_checksum"] = existing.get("checksum")
                duplicates = list(state.get("duplicate_chunks", []) or [])
                duplicates.append(duplicate_record)
                state["duplicate_chunks"] = duplicates[-100:]
                conflicts = list(state.get("conflict_chunks", []) or [])
                conflicts.append(duplicate_record)
                state["conflict_chunks"] = conflicts[-100:]
                self._save_stream_state_unlocked(state)
                raise ValueError(f"upload_chunk_index {upload_chunk_index} already exists with different checksum")

            recovered_from_orphan = False
            if final_path.exists():
                existing_checksum = sha256_path(final_path)
                tmp_upload_path.unlink(missing_ok=True)
                if existing_checksum != checksum:
                    duplicate_record["status"] = "duplicate_conflict"
                    duplicate_record["existing_checksum"] = existing_checksum
                    conflicts = list(state.get("conflict_chunks", []) or [])
                    conflicts.append(duplicate_record)
                    state["conflict_chunks"] = conflicts[-100:]
                    self._save_stream_state_unlocked(state)
                    raise ValueError(f"upload_chunk_index {upload_chunk_index} already exists with different checksum")
                recovered_from_orphan = True
                print(
                    f"[stream_state] recovered orphan upload during register session_id={self.session_dir.name} "
                    f"chunk_index={upload_chunk_index}",
                    flush=True,
                )
            else:
                try:
                    os.replace(tmp_upload_path, final_path)
                except Exception as exc:
                    print(
                        f"[stream_state] upload replace failed session_id={self.session_dir.name} "
                        f"chunk_index={upload_chunk_index} tmp={tmp_upload_path} final={final_path} "
                        f"tmp_exists={tmp_upload_path.exists()} target_parent_exists={final_path.parent.exists()} "
                        f"error={exc}",
                        flush=True,
                    )
                    raise

            next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
            upload_status = "received" if int(upload_chunk_index) == next_upload else "waiting_for_previous"
            upload_record = {
                "upload_chunk_id": upload_chunk_id,
                "chunk_id": upload_chunk_id,
                "upload_chunk_index": int(upload_chunk_index),
                "chunk_index": int(upload_chunk_index),
                "path": rel_to_session(self.session_dir, final_path),
                "checksum": checksum,
                "size_bytes": int(size_bytes),
                "upload_duration": round(float(actual_duration), 3) if actual_duration is not None else None,
                "actual_duration": round(float(actual_duration), 3) if actual_duration is not None else None,
                "stream_start_time": None,
                "stream_end_time": None,
                "status": upload_status,
                "client_timestamp": client_timestamp,
                "is_last": bool(is_last),
                "processing_chunks": [],
                "received_at": utc_now_iso(),
                "processed_at": None,
                "error": None,
                "recovered_from_orphan_file": recovered_from_orphan,
            }
            upload_chunks.append(upload_record)
            state["upload_chunks"] = upload_chunks
            state["received_chunks"] = upload_chunks
            if is_last:
                state["final_upload_chunk_index"] = int(upload_chunk_index)
                state["status"] = "ending"
            self._save_stream_state_unlocked(state)
            return {
                "status": upload_status,
                "upload_chunk": upload_record,
                "duplicate": False,
                "state": self._load_stream_state_unlocked(default={}),
            }

    def enqueue_ready_upload_chunk(self, project_root: Path) -> Path | None:
        from online_preprocess.task_queue import enqueue_stream_upload_task

        state = self.load_stream_state(default={})
        next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
        upload = next(
            (
                dict(item)
                for item in state.get("upload_chunks", state.get("received_chunks", [])) or []
                if isinstance(item, dict)
                and int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == next_upload
            ),
            None,
        )
        if not upload or upload.get("processing_chunks"):
            return None
        if str(upload.get("status") or "") not in {"received", "waiting_for_previous"}:
            return None
        task_path = enqueue_stream_upload_task(
            project_root=project_root,
            session_id=self.session_dir.name,
            stream_id=str(state.get("stream_id") or ""),
            upload_chunk_id=str(upload.get("upload_chunk_id") or upload.get("chunk_id") or ""),
            upload_chunk_index=next_upload,
            upload_chunk_path=str(upload.get("path") or ""),
            checksum=str(upload.get("checksum") or ""),
        )
        upload_chunks = []
        for item in state.get("upload_chunks", state.get("received_chunks", [])) or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == next_upload:
                item["status"] = "queued"
                item["task_id"] = task_path.stem
                item["task_path"] = str(task_path)
                item["stream_chunk_queued_at"] = utc_now_iso()
            upload_chunks.append(item)
        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        self.save_stream_state(state)
        return task_path

    def reconcile_stream_upload_chunks(self) -> dict[str, Any]:
        recovered: list[dict[str, Any]] = []
        self.upload_chunks_dir.mkdir(parents=True, exist_ok=True)
        with self.stream_state_lock():
            state = self._load_stream_state_unlocked(default={})
            if not state:
                return {"status": "no_stream_state", "recovered_count": 0, "recovered_chunks": []}
            upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) if isinstance(item, dict)]
            registered = {
                int(item.get("upload_chunk_index", item.get("chunk_index", -1)))
                for item in upload_chunks
                if str(item.get("upload_chunk_index", item.get("chunk_index", ""))).lstrip("-").isdigit()
            }
            for path in sorted(self.upload_chunks_dir.glob("upload_*.mp4")):
                stem = path.stem
                try:
                    upload_index = int(stem.split("_", 1)[1])
                except Exception:
                    continue
                if upload_index in registered:
                    continue
                try:
                    checksum = sha256_path(path)
                except Exception as exc:
                    print(
                        f"[stream_state] orphan checksum failed session_id={self.session_dir.name} "
                        f"chunk_index={upload_index}: {exc}",
                        flush=True,
                    )
                    continue
                try:
                    duration = probe_duration(path)
                except Exception as exc:
                    print(
                        f"[stream_state] orphan duration probe failed session_id={self.session_dir.name} "
                        f"chunk_index={upload_index}: {exc}",
                        flush=True,
                    )
                    duration = None
                next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
                record = {
                    "upload_chunk_id": self.make_upload_chunk_id(upload_index),
                    "chunk_id": self.make_upload_chunk_id(upload_index),
                    "upload_chunk_index": upload_index,
                    "chunk_index": upload_index,
                    "path": rel_to_session(self.session_dir, path),
                    "checksum": checksum,
                    "size_bytes": int(path.stat().st_size),
                    "upload_duration": round(float(duration), 3) if duration is not None else None,
                    "actual_duration": round(float(duration), 3) if duration is not None else None,
                    "stream_start_time": None,
                    "stream_end_time": None,
                    "status": "received" if upload_index == next_upload else "waiting_for_previous",
                    "processing_chunks": [],
                    "received_at": utc_now_iso(),
                    "processed_at": None,
                    "error": None,
                    "recovered_from_orphan_file": True,
                }
                print(
                    f"[stream_state] recovered orphan upload file session_id={self.session_dir.name} "
                    f"chunk_index={upload_index} path={path}",
                    flush=True,
                )
                upload_chunks.append(record)
                registered.add(upload_index)
                recovered.append(record)
            if recovered:
                state["upload_chunks"] = upload_chunks
                state["received_chunks"] = upload_chunks
                self._save_stream_state_unlocked(state)
        return {"status": "ok", "recovered_count": len(recovered), "recovered_chunks": recovered}

    def materialize_ready_upload_chunks(self, *, processing_chunk_seconds: float) -> dict[str, Any]:
        with self.stream_state_lock():
            state = self._load_stream_state_unlocked(default={})
            if not state:
                now = utc_now_iso()
                state = {
                    "session_id": self.session_dir.name,
                    "stream_id": uuid4().hex[:12],
                    "status": "running",
                    "chunk_duration": float(processing_chunk_seconds),
                    "processing_chunk_seconds": float(processing_chunk_seconds),
                    "received_chunks": [],
                    "upload_chunks": [],
                    "processing_chunks": [],
                    "processed_chunks": [],
                    "failed_chunks": [],
                    "missing_chunks": [],
                    "duplicate_chunks": [],
                    "conflict_chunks": [],
                    "retry_required_chunks": [],
                    "waiting_chunks": [],
                    "latency": {},
                    "next_expected_upload_chunk_index": 0,
                    "next_expected_proc_index": 0,
                    "next_expected_chunk_index": 0,
                    "last_processed_proc_index": -1,
                    "last_processed_chunk_index": -1,
                    "stream_timeline_end": 0.0,
                    "created_at": now,
                    "updated_at": now,
                }
            state = self._materialize_ready_upload_chunks(state, max(0.1, float(processing_chunk_seconds)))
            self._save_stream_state_unlocked(state)
            return self._load_stream_state_unlocked(default={})

    def _materialize_ready_upload_chunks(self, state: dict[str, Any], processing_chunk_seconds: float) -> dict[str, Any]:
        """Split only contiguous upload chunks into processing chunks.

        Upload chunks may arrive out of order.  The stream timeline can only be
        assigned once all previous upload chunks are known, otherwise upload #2
        could incorrectly start at t=0.  This method advances the timeline in
        upload_chunk_index order and leaves later chunks waiting.
        """
        upload_chunks = [dict(item) for item in state.get("upload_chunks", state.get("received_chunks", [])) if isinstance(item, dict)]
        processing_chunks = [dict(item) for item in state.get("processing_chunks", []) if isinstance(item, dict)]
        next_upload = int(state.get("next_expected_upload_chunk_index", 0) or 0)
        proc_index = int(state.get("next_processing_chunk_index", len(processing_chunks)) or 0)
        timeline_start = float(state.get("stream_timeline_end", 0.0) or 0.0)
        proc_seconds = max(0.1, float(processing_chunk_seconds))
        direct_upload_processing = self._use_direct_upload_processing(state)
        state["processing_chunk_strategy"] = "upload_direct" if direct_upload_processing else "fixed_seconds"

        while True:
            record = next(
                (
                    item
                    for item in upload_chunks
                    if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == next_upload
                ),
                None,
            )
            if record is None:
                break
            if record.get("processing_chunks"):
                next_upload += 1
                timeline_start = float(record.get("stream_end_time") or timeline_start)
                continue
            local_start = 0.0
            actual_duration = max(0.0, float(record.get("actual_duration") or record.get("upload_duration") or 0.0))
            upload_path = self.session_dir / str(record.get("path") or "")
            if actual_duration <= 1e-3:
                actual_duration = probe_duration(upload_path)
                record["actual_duration"] = round(actual_duration, 3)
                record["upload_duration"] = round(actual_duration, 3)
            new_processing: list[dict[str, Any]] = []
            segment_ranges = self._processing_segment_ranges(
                actual_duration=actual_duration,
                processing_chunk_seconds=proc_seconds,
                direct_upload_processing=direct_upload_processing,
            )
            for local_start, local_end in segment_ranges:
                start_time = timeline_start + local_start
                end_time = timeline_start + local_end
                proc_id = self.make_processing_chunk_id(proc_index, start_time, end_time)
                proc_path = self.chunks_dir / f"{proc_id}.mp4"
                if not proc_path.exists():
                    if direct_upload_processing:
                        try:
                            os.link(upload_path, proc_path)
                        except Exception:
                            shutil.copy2(upload_path, proc_path)
                    else:
                        _run_checked(
                            [
                                ffmpeg_bin(),
                                "-y",
                                "-ss",
                                f"{local_start:.3f}",
                                "-i",
                                str(upload_path),
                                "-t",
                                f"{local_end - local_start:.3f}",
                                "-c",
                                "copy",
                                str(proc_path),
                            ]
                        )
                proc_record = {
                    "proc_index": proc_index,
                    "chunk_index": proc_index,
                    "processing_chunk_id": proc_id,
                    "chunk_id": proc_id,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "duration": round(local_end - local_start, 3),
                    "local_start_time": round(local_start, 3),
                    "local_end_time": round(local_end, 3),
                    "source_upload_chunk_index": int(record.get("upload_chunk_index", next_upload)),
                    "source_upload_chunk_id": str(record.get("upload_chunk_id") or self.make_upload_chunk_id(next_upload)),
                    "upload_chunk_index": int(record.get("upload_chunk_index", next_upload)),
                    "processing_strategy": "upload_direct" if direct_upload_processing else "fixed_seconds",
                    "path": rel_to_session(self.session_dir, proc_path),
                    "status": "received"
                    if proc_index == int(state.get("next_expected_proc_index", 0) or 0)
                    else "waiting_for_previous",
                    "task_id": None,
                    "task_path": None,
                    "created_at": utc_now_iso(),
                    "processed_at": None,
                    "error": None,
                }
                new_processing.append(proc_record)
                processing_chunks.append(proc_record)
                proc_index += 1
            if direct_upload_processing:
                print(
                    f"[stream_materialize] camera_live no-split session_id={self.session_dir.name} "
                    f"upload_chunk_index={record.get('upload_chunk_index', next_upload)} "
                    f"actual_duration={round(float(actual_duration), 3)} "
                    f"processing_chunk_ids={[item.get('chunk_id') for item in new_processing]} "
                    f"processing_chunk_count={len(new_processing)}",
                    flush=True,
                )
            else:
                short_segments = [
                    item
                    for item in new_processing
                    if float(item.get("duration") or 0.0) < float(os.getenv("EM2MEM_STREAM_MIN_PROCESSING_TAIL_SECONDS", "1.0"))
                ]
                if short_segments:
                    print(
                        f"[stream_materialize] short processing chunk kept session_id={self.session_dir.name} "
                        f"upload_chunk_index={record.get('upload_chunk_index', next_upload)} "
                        f"actual_duration={round(float(actual_duration), 3)} "
                        f"processing_chunk_ids={[item.get('chunk_id') for item in short_segments]} "
                        f"processing_chunk_count={len(new_processing)}",
                        flush=True,
                    )
            record["status"] = "received"
            record["stream_start_time"] = round(timeline_start, 3)
            record["stream_end_time"] = round(timeline_start + actual_duration, 3)
            record["processing_chunks"] = new_processing
            timeline_start += actual_duration
            next_upload += 1

        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        state["processing_chunks"] = processing_chunks
        state["next_expected_upload_chunk_index"] = next_upload
        state["next_processing_chunk_index"] = proc_index
        state["stream_timeline_end"] = round(timeline_start, 3)
        final_upload = state.get("final_upload_chunk_index")
        if final_upload is not None:
            final_proc = self._derive_final_proc_index(state, int(final_upload))
            if final_proc is not None:
                state["final_chunk_index"] = final_proc
        return state

    def _derive_final_proc_index(self, state: dict[str, Any], final_upload_chunk_index: int) -> int | None:
        candidates = [
            int(item.get("proc_index", item.get("chunk_index", -1)))
            for item in state.get("processing_chunks", []) or []
            if isinstance(item, dict)
            and int(item.get("source_upload_chunk_index", item.get("upload_chunk_index", -1))) <= int(final_upload_chunk_index)
        ]
        return max(candidates) if candidates else None

    def register_received_chunk(
        self,
        *,
        chunk_index: int,
        start_time: float,
        end_time: float,
        path: str,
        checksum: str,
        size_bytes: int,
        client_timestamp: str | None = None,
        is_last: bool = False,
    ) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        if not state:
            state = self.init_stream()
        received = [dict(item) for item in state.get("received_chunks", []) if isinstance(item, dict)]
        existing = next((item for item in received if int(item.get("chunk_index", -1)) == int(chunk_index)), None)
        duplicate_record = {
            "chunk_index": int(chunk_index),
            "checksum": checksum,
            "received_at": utc_now_iso(),
            "path": path,
        }
        if existing is not None:
            if str(existing.get("checksum") or "") == checksum:
                duplicate_record["status"] = "duplicate_ignored"
                duplicates = list(state.get("duplicate_chunks", []) or [])
                duplicates.append(duplicate_record)
                state["duplicate_chunks"] = duplicates[-100:]
                self.save_stream_state(state)
                return {"status": "duplicate_ignored", "chunk": existing, "state": state, "duplicate": True}
            duplicate_record["status"] = "duplicate_conflict"
            duplicate_record["existing_checksum"] = existing.get("checksum")
            duplicates = list(state.get("duplicate_chunks", []) or [])
            duplicates.append(duplicate_record)
            state["duplicate_chunks"] = duplicates[-100:]
            conflicts = list(state.get("conflict_chunks", []) or [])
            conflicts.append(duplicate_record)
            state["conflict_chunks"] = conflicts[-100:]
            self.save_stream_state(state)
            raise ValueError(f"chunk_index {chunk_index} already exists with different checksum")

        next_expected = int(state.get("next_expected_chunk_index", 0) or 0)
        status = "received" if int(chunk_index) == next_expected else "waiting_for_previous"
        chunk = {
            "chunk_id": self.make_chunk_id(chunk_index, start_time, end_time),
            "chunk_index": int(chunk_index),
            "start_time": round(float(start_time), 3),
            "end_time": round(float(end_time), 3),
            "duration": round(max(0.0, float(end_time) - float(start_time)), 3),
            "path": path,
            "checksum": checksum,
            "size_bytes": int(size_bytes),
            "status": status,
            "task_id": None,
            "task_path": None,
            "client_timestamp": client_timestamp,
            "is_last": bool(is_last),
            "received_at": utc_now_iso(),
            "processed_at": None,
            "error": None,
        }
        received.append(chunk)
        state["received_chunks"] = received
        if is_last:
            state["final_chunk_index"] = int(chunk_index)
            state["status"] = "ending"
        self.save_stream_state(state)
        return {"status": status, "chunk": chunk, "state": self.load_stream_state(default={}), "duplicate": False}

    def get_chunk(self, chunk_index: int) -> dict[str, Any] | None:
        state = self.load_stream_state(default={})
        for item in state.get("processing_chunks", []) or []:
            if isinstance(item, dict) and int(item.get("proc_index", item.get("chunk_index", -1))) == int(chunk_index):
                return dict(item)
        if state.get("upload_chunks"):
            return None
        for item in state.get("received_chunks", []) or []:
            if isinstance(item, dict) and int(item.get("chunk_index", -1)) == int(chunk_index):
                return dict(item)
        return None

    def update_chunk_status(
        self,
        chunk_index: int,
        *,
        status: str,
        task_id: str | None = None,
        task_path: str | None = None,
        error: str | None = None,
        processed_at: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        processing_mode = bool(state.get("processing_chunks"))
        key = "processing_chunks" if processing_mode else "received_chunks"
        chunks = []
        for item in state.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item_index = int(item.get("proc_index", item.get("chunk_index", -1))) if processing_mode else int(item.get("chunk_index", -1))
            if item_index == int(chunk_index):
                item["status"] = status
                now = utc_now_iso()
                if task_id is not None:
                    item["task_id"] = task_id
                if task_path is not None:
                    item["task_path"] = task_path
                if error is not None:
                    item["error"] = error
                if status == "queued":
                    item["stream_chunk_queued_at"] = item.get("stream_chunk_queued_at") or now
                if status == "processing":
                    item["stream_chunk_processing_started_at"] = item.get("stream_chunk_processing_started_at") or now
                if processed_at is not None:
                    item["processed_at"] = processed_at
                    item["stream_chunk_processed_at"] = processed_at
                    item["mcur_updated_at"] = processed_at
                if extra:
                    for key, value in extra.items():
                        if value is not None:
                            item[key] = value
            chunks.append(item)
        state[key] = chunks
        if processing_mode:
            status_by_proc = {int(item.get("proc_index", item.get("chunk_index", -1))): dict(item) for item in chunks}
            upload_chunks = []
            for upload in state.get("upload_chunks", state.get("received_chunks", [])) or []:
                if not isinstance(upload, dict):
                    continue
                upload = dict(upload)
                nested = []
                for proc in upload.get("processing_chunks", []) or []:
                    if not isinstance(proc, dict):
                        continue
                    proc_index = int(proc.get("proc_index", proc.get("chunk_index", -1)))
                    nested.append(status_by_proc.get(proc_index, dict(proc)))
                upload["processing_chunks"] = nested
                if nested and all(str(item.get("status") or "") == "processed" for item in nested):
                    upload["status"] = "processed"
                    upload["processed_at"] = max(str(item.get("processed_at") or "") for item in nested)
                elif nested and any(str(item.get("status") or "") in {"queued", "processing", "processed"} for item in nested):
                    upload["status"] = "processing"
                elif not nested and str(upload.get("status") or "") == "queued":
                    upload["status"] = "received"
                upload_chunks.append(upload)
            state["upload_chunks"] = upload_chunks
            state["received_chunks"] = upload_chunks
        elif state.get("upload_chunks"):
            state["upload_chunks"] = chunks
            state["received_chunks"] = chunks
        if status == "processed":
            processed = list(state.get("processed_chunks", []) or [])
            if int(chunk_index) not in processed:
                processed.append(int(chunk_index))
            state["processed_chunks"] = sorted(processed)
            try:
                previous_last = int(state.get("last_processed_proc_index" if processing_mode else "last_processed_chunk_index", -1))
            except Exception:
                previous_last = -1
            last_value = max(previous_last, int(chunk_index))
            if processing_mode:
                state["last_processed_proc_index"] = last_value
                state["last_processed_chunk_index"] = last_value
                state["next_expected_proc_index"] = last_value + 1
                state["next_expected_chunk_index"] = last_value + 1
            else:
                state["last_processed_chunk_index"] = last_value
                state["next_expected_chunk_index"] = last_value + 1
        if status == "failed":
            failed = list(state.get("failed_chunks", []) or [])
            if int(chunk_index) not in failed:
                failed.append(int(chunk_index))
            state["failed_chunks"] = sorted(failed)
        self.save_stream_state(state)
        return self.load_stream_state(default={})

    def enqueue_ready_chunk(self, project_root: Path) -> Path | None:
        from online_preprocess.task_queue import enqueue_stream_chunk_task

        state = self.load_stream_state(default={})
        processing_mode = bool(state.get("processing_chunks"))
        if not processing_mode and state.get("upload_chunks"):
            return None
        next_index = int(state.get("next_expected_proc_index" if processing_mode else "next_expected_chunk_index", 0) or 0)
        chunk = self.get_chunk(next_index)
        if not chunk:
            return None
        if str(chunk.get("status") or "") not in {"received", "waiting_for_previous"}:
            return None
        task_path = enqueue_stream_chunk_task(
            project_root=project_root,
            session_id=self.session_dir.name,
            stream_id=str(state.get("stream_id") or ""),
            chunk_id=str(chunk.get("chunk_id") or ""),
            chunk_index=next_index,
            proc_index=next_index if processing_mode else None,
            upload_chunk_index=chunk.get("upload_chunk_index") or chunk.get("source_upload_chunk_index"),
            source_upload_chunk_id=chunk.get("source_upload_chunk_id"),
            chunk_path=str(chunk.get("path") or ""),
            start_time=float(chunk.get("start_time") or 0.0),
            end_time=float(chunk.get("end_time") or 0.0),
            duration=float(chunk.get("duration") or 0.0),
            checksum=str(chunk.get("checksum") or ""),
        )
        self.update_chunk_status(
            next_index,
            status="queued",
            task_id=task_path.stem,
            task_path=str(task_path),
        )
        return task_path

    def mark_stream_ending(self, *, final_chunk_index: int | None, close_open_event: bool) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        state["status"] = "ending"
        processing_mode = bool(state.get("processing_chunks"))
        if final_chunk_index is not None:
            if processing_mode:
                state["final_upload_chunk_index"] = int(final_chunk_index)
                final_proc = self._derive_final_proc_index(state, int(final_chunk_index))
                state["final_chunk_index"] = final_proc if final_proc is not None else int(state.get("last_processed_proc_index", -1) or -1)
            else:
                state["final_chunk_index"] = int(final_chunk_index)
        elif state.get("final_chunk_index") is None:
            if processing_mode:
                indexes = [int(item.get("proc_index", item.get("chunk_index", -1))) for item in state.get("processing_chunks", []) or [] if isinstance(item, dict)]
                state["final_chunk_index"] = max(indexes) if indexes else int(state.get("last_processed_proc_index", -1) or -1)
            else:
                indexes = [int(item.get("chunk_index", -1)) for item in state.get("received_chunks", []) or [] if isinstance(item, dict)]
                state["final_chunk_index"] = max(indexes) if indexes else int(state.get("last_processed_chunk_index", -1) or -1)
        state["close_open_event"] = bool(close_open_event)
        self.save_stream_state(state)
        return self.load_stream_state(default={})

    def enqueue_stream_end_if_ready(self, project_root: Path) -> Path | None:
        from online_preprocess.task_queue import enqueue_stream_end_task

        state = self.load_stream_state(default={})
        if state.get("status") != "ending":
            return None
        final_index = state.get("final_chunk_index")
        try:
            final_int = int(final_index)
        except Exception:
            try:
                final_int = int(state.get("last_processed_proc_index", state.get("last_processed_chunk_index", -1)))
            except Exception:
                final_int = -1
        try:
            last_processed = int(state.get("last_processed_proc_index", state.get("last_processed_chunk_index", -1)))
        except Exception:
            last_processed = -1
        if last_processed < final_int:
            return None
        task_path = enqueue_stream_end_task(
            project_root=project_root,
            session_id=self.session_dir.name,
            stream_id=str(state.get("stream_id") or ""),
            final_chunk_index=final_int,
            close_open_event=bool(state.get("close_open_event", True)),
        )
        state["stream_end_task_id"] = task_path.stem
        state["stream_end_task_path"] = str(task_path)
        self.save_stream_state(state)
        return task_path

    def mark_stream_ended(self) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        state["status"] = "ended"
        state["ended_at"] = utc_now_iso()
        self.save_stream_state(state)
        return self.load_stream_state(default={})

    def summary(self) -> dict[str, Any]:
        state = self.load_stream_state(default={})
        event_state = self.load_event_state()
        upload_chunks = [item for item in state.get("upload_chunks", state.get("received_chunks", [])) or [] if isinstance(item, dict)]
        processing_chunks = [item for item in state.get("processing_chunks", []) or [] if isinstance(item, dict)]
        processed_processing = [
            item for item in processing_chunks if isinstance(item, dict) and str(item.get("status") or "") == "processed"
        ]
        open_event = event_state.get("open_event") if isinstance(event_state, dict) else None
        last_frame = event_state.get("last_candidate_frame") if isinstance(event_state, dict) else None
        diff_count = self._count_diff_records()
        cross_diff_count = self._count_diff_records(cross_chunk_only=True)
        transcript_state = read_json(self.stream_dir / "transcript" / "partial_transcript_state.json", default={})
        if not isinstance(transcript_state, dict):
            transcript_state = {}
        return {
            "session_id": self.session_dir.name,
            "stream_id": state.get("stream_id"),
            "stream_status": state.get("status", "not_started"),
            "chunk_duration": state.get("chunk_duration"),
            "processing_chunk_seconds": state.get("processing_chunk_seconds"),
            "processing_chunk_strategy": state.get("processing_chunk_strategy"),
            "upload_received_count": len(upload_chunks),
            "upload_processed_count": len(
                [item for item in upload_chunks if isinstance(item, dict) and str(item.get("status") or "") == "processed"]
            ),
            "processing_chunk_count": len(processing_chunks),
            "processing_done_count": len(processed_processing),
            "received_chunk_count": len(upload_chunks),
            "received_upload_chunk_count": len(upload_chunks),
            "generated_processing_chunk_count": len(processing_chunks),
            "processed_chunk_count": len(processed_processing) if processing_chunks else len(state.get("processed_chunks", []) or []),
            "processed_processing_chunk_count": len(processed_processing),
            "next_expected_chunk_index": state.get("next_expected_chunk_index", 0),
            "next_expected_upload_chunk_index": state.get("next_expected_upload_chunk_index", 0),
            "next_expected_proc_index": state.get("next_expected_proc_index", 0),
            "last_processed_chunk_index": state.get("last_processed_chunk_index", -1),
            "last_processed_proc_index": state.get("last_processed_proc_index", -1),
            "missing_chunks": state.get("missing_chunks", []),
            "failed_chunks": state.get("failed_chunks", []),
            "duplicate_chunks": state.get("duplicate_chunks", []),
            "conflict_chunks": state.get("conflict_chunks", []),
            "retry_required_chunks": state.get("retry_required_chunks", []),
            "waiting_chunks": state.get("waiting_chunks", []),
            "has_open_event": bool(open_event),
            "open_event_start": open_event.get("start_time") if isinstance(open_event, dict) else None,
            "open_event_end": open_event.get("last_update_time") if isinstance(open_event, dict) else None,
            "last_candidate_frame_time": last_frame.get("timestamp") if isinstance(last_frame, dict) else None,
            "diff_record_count": diff_count,
            "cross_chunk_diff_count": cross_diff_count,
            "stream_asr": {
                "partial_transcript_version": transcript_state.get("partial_transcript_version", 0),
                "partial_transcript_segment_count": transcript_state.get("segment_count", 0),
                "time_span": transcript_state.get("time_span", [0.0, 0.0]),
                "processed_asr_chunks": transcript_state.get("processed_asr_chunks", []),
                "failed_asr_chunks": transcript_state.get("failed_asr_chunks", []),
                "last_asr_chunk_index": transcript_state.get("last_asr_chunk_index"),
            },
            "upload_chunks": upload_chunks,
            "processing_chunks": processing_chunks,
            "chunks": processing_chunks or upload_chunks,
            "latency": state.get("latency", {}),
            "updated_at": state.get("updated_at"),
        }

    def _count_diff_records(self, *, cross_chunk_only: bool = False) -> int:
        path = self.stream_dir / "diff_records.jsonl"
        if not path.exists():
            return 0
        count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if not cross_chunk_only:
                        count += 1
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict) and payload.get("is_cross_chunk"):
                        count += 1
        except Exception:
            return 0
        return count

    def append_processed_chunk(self, record: dict[str, Any]) -> None:
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload["processed_at"] = utc_now_iso()
        with self.processed_chunks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_stream_state(self, *, duration: float, chunks: list[dict[str, Any]]) -> None:
        now = utc_now_iso()
        state = {
            "session_id": self.session_dir.name,
            "stream_id": uuid4().hex[:12],
            "status": "simulated",
            "duration": round(duration, 3),
            "chunk_duration": chunks[0]["duration"] if chunks else None,
            "chunk_count": len(chunks),
            "received_chunks": [{**chunk, "status": "received"} for chunk in chunks],
            "processed_chunks": [],
            "failed_chunks": [],
            "missing_chunks": [],
            "duplicate_chunks": [],
            "next_expected_chunk_index": 0,
            "last_processed_chunk_index": -1,
            "created_at": now,
            "updated_at": now,
        }
        self.save_stream_state(state)
        if not self.event_state_path.exists():
            self.save_event_state(self.empty_event_state(state["stream_id"]))

    def _compute_missing_chunks(self, state: dict[str, Any]) -> list[int]:
        chunks = [item for item in state.get("upload_chunks", state.get("received_chunks", [])) or [] if isinstance(item, dict)]
        if not chunks:
            return []
        received = {int(item.get("upload_chunk_index", item.get("chunk_index", -1))) for item in chunks}
        max_received = max(received)
        return [idx for idx in range(0, max_received + 1) if idx not in received]

    def _compute_waiting_chunks(self, state: dict[str, Any]) -> list[int]:
        waiting: set[int] = set()
        for item in state.get("upload_chunks", state.get("received_chunks", [])) or []:
            if isinstance(item, dict) and str(item.get("status") or "") == "waiting_for_previous":
                waiting.add(int(item.get("upload_chunk_index", item.get("chunk_index", -1))))
        for item in state.get("processing_chunks", []) or []:
            if isinstance(item, dict) and str(item.get("status") or "") == "waiting_for_previous":
                waiting.add(int(item.get("proc_index", item.get("chunk_index", -1))))
        return sorted(idx for idx in waiting if idx >= 0)

    def _compute_retry_required_chunks(self, state: dict[str, Any]) -> list[int]:
        retry: set[int] = set()
        for item in state.get("upload_chunks", state.get("received_chunks", [])) or []:
            if isinstance(item, dict) and str(item.get("status") or "") == "failed":
                retry.add(int(item.get("upload_chunk_index", item.get("chunk_index", -1))))
            if isinstance(item, dict) and str(item.get("asr_status") or "") == "failed":
                retry.add(int(item.get("upload_chunk_index", item.get("chunk_index", -1))))
        for item in state.get("processing_chunks", []) or []:
            if isinstance(item, dict) and str(item.get("status") or "") == "failed":
                retry.add(int(item.get("source_upload_chunk_index", item.get("upload_chunk_index", item.get("chunk_index", -1)))))
        return sorted(idx for idx in retry if idx >= 0)

    def _compute_latency(self, state: dict[str, Any]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for upload in state.get("upload_chunks", state.get("received_chunks", [])) or []:
            if not isinstance(upload, dict):
                continue
            nested = [dict(item) for item in upload.get("processing_chunks", []) or [] if isinstance(item, dict)]
            row = dict(upload)
            if nested:
                for key in ("stream_chunk_queued_at", "stream_chunk_processing_started_at", "stream_chunk_processed_at", "mcur_updated_at", "mst_event_closed_at"):
                    values = [item.get(key) for item in nested if item.get(key)]
                    if values:
                        row[key] = max(values)
            rows.append(row)
        if not rows:
            return state.get("latency", {}) if isinstance(state.get("latency"), dict) else {}

        def last_ms(target_key: str) -> int | None:
            candidates = [
                (item.get("received_at"), item.get(target_key))
                for item in rows
                if item.get("received_at") and item.get(target_key)
            ]
            if not candidates:
                return None
            return _elapsed_ms(*candidates[-1])

        def avg_ms(target_key: str) -> int | None:
            values = [
                _elapsed_ms(item.get("received_at"), item.get(target_key))
                for item in rows
                if item.get("received_at") and item.get(target_key)
            ]
            values = [value for value in values if value is not None]
            if not values:
                return None
            return int(round(sum(values) / len(values)))

        return {
            "last_chunk_upload_to_mcur_ms": last_ms("mcur_updated_at"),
            "last_chunk_upload_to_mst_ms": last_ms("mst_event_closed_at"),
            "last_chunk_upload_to_asr_ms": last_ms("asr_done_at"),
            "last_chunk_upload_to_refine_ms": last_ms("refine_queued_at"),
            "last_chunk_upload_to_memory_ms": last_ms("memory_appended_at"),
            "avg_upload_to_mcur_ms": avg_ms("mcur_updated_at"),
            "avg_upload_to_asr_ms": avg_ms("asr_done_at"),
            "avg_upload_to_memory_ms": avg_ms("memory_appended_at"),
        }


def discover_chunks(session_dir: Path) -> list[dict[str, Any]]:
    chunks_dir = session_dir / "stream" / "chunks"
    records = []
    for idx, path in enumerate(sorted(chunks_dir.glob("chunk_*.mp4"))):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            start = float(int(parts[1]))
            end = float(int(parts[2]))
        else:
            start, end = 0.0, probe_duration(path)
        records.append(
            {
                "chunk_id": stem,
                "chunk_index": idx,
                "start_time": start,
                "end_time": end,
                "duration": max(0.0, end - start),
                "path": rel_to_session(session_dir, path),
            }
        )
    return records
