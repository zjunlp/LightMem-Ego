from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from online_pipeline.frame_stream import frame_stream_input_mode
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_preprocess.task_queue import enqueue_mst_refine_task
from online_short_term.frame_diff_detector import CandidateFrame, FrameDiffEventDetector, cv2
from online_short_term.mst_store import MSTStore
from online_short_term.schemas import build_retrieval_text, env_float, env_int, mst_event_stub


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class FrameStreamMicroEventBuilder:
    """Adapter from lossy frame stream images into the existing M_st store."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.stream_dir = self.session_dir / "stream"
        self.state_path = self.stream_dir / "frame_event_state.json"
        self.lock_path = self.stream_dir / "frame_event_state.lock"
        self.frame_events_path = self.stream_dir / "frame_events.jsonl"
        self.diff_records_path = self.stream_dir / "frame_diff_records.jsonl"
        self.detector = FrameDiffEventDetector(
            self.session_dir,
            diff_threshold=env_float("WORLDMM_FRAME_STREAM_DIFF_THRESHOLD", env_float("WORLDMM_MST_DIFF_THRESHOLD", 0.40)),
            min_event_duration=env_float("WORLDMM_FRAME_STREAM_MIN_EVENT_SECONDS", 1.0),
            max_event_duration=env_float("WORLDMM_FRAME_STREAM_MAX_EVENT_SECONDS", 10.0),
            min_boundary_gap=env_float("WORLDMM_FRAME_STREAM_MIN_BOUNDARY_GAP_SECONDS", 1.0),
        )
        self.store = MSTStore(self.session_dir)
        self.max_keyframes = env_int("WORLDMM_FRAME_STREAM_MAX_KEYFRAMES_PER_EVENT", 8)

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.stream_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def process_frame(
        self,
        *,
        frame_record: dict[str, Any],
        current_frame_path: str | None = None,
        project_root: Path | None = None,
        enqueue_refine: bool = True,
    ) -> dict[str, Any]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for frame stream M_st detection")
        frame = self._candidate_from_record(frame_record, current_frame_path=current_frame_path)
        if frame is None:
            return self._empty_result(status="skipped_missing_frame")
        with self.lock():
            state = self._load_state_unlocked()
            frame, diff_record = self._compute_diff(frame, state.get("last_accepted_frame"))
            closed_events, state, opened_events = self._update_state_with_frame(state, frame, diff_record)
            appended = self.store.append_events(closed_events)
            refine_task_paths: list[str] = []
            if appended and enqueue_refine and project_root is not None and _env_bool("WORLDMM_FRAME_STREAM_ENQUEUE_REFINE", True):
                task_path = enqueue_mst_refine_task(
                    project_root=Path(project_root),
                    session_id=self.session_id,
                    backend=os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai"),
                    limit_events=max(1, int(os.getenv("WORLDMM_MST_REFINE_LIMIT_EVENTS", "10") or 10)),
                    event_id=None,
                    force_refine=False,
                    reason="frame_stream_batch",
                )
                refine_task_paths.append(str(task_path))
                state["refine_queued_count"] = int(state.get("refine_queued_count", 0) or 0) + len(refine_task_paths)
                state["latest_refine_task_paths"] = refine_task_paths[-10:]
            state["updated_at"] = utc_now_iso()
            self._save_state_unlocked(state)
            if diff_record:
                self._append_diff_record_unlocked(diff_record)
            self._append_frame_event_unlocked(
                "frame_mst_updated",
                frame_index=int(frame_record.get("frame_index", -1) or -1),
                metadata={
                    "diff_score": frame.diff_score,
                    "has_open_event": bool(state.get("open_event")),
                    "closed_event_count": len(appended),
                    "closed_event_ids": [event.get("event_id") for event in appended],
                },
            )
            for event in appended:
                self._append_frame_event_unlocked(
                    "frame_mst_closed",
                    frame_index=int(frame_record.get("frame_index", -1) or -1),
                    metadata={
                        "event_id": event.get("event_id"),
                        "start_time": event.get("start_time"),
                        "end_time": event.get("end_time"),
                        "duration": event.get("duration"),
                        "keyframe_count": len(event.get("keyframes", []) or []),
                        "diff_score": event.get("diff_score"),
                    },
                )
            return {
                "enabled": True,
                "status": "ok",
                "event_detector_ready": True,
                "has_open_event": bool(state.get("open_event")),
                "open_event_start_time": (state.get("open_event") or {}).get("start_time") if isinstance(state.get("open_event"), dict) else None,
                "open_event_duration": self._open_event_duration(state.get("open_event")),
                "closed_event_count": len(appended),
                "closed_event_ids": [event.get("event_id") for event in appended],
                "opened_event_count": len(opened_events),
                "opened_events": opened_events,
                "latest_event_id": state.get("latest_event_id"),
                "latest_event_time_range": state.get("latest_event_time_range"),
                "diff_score": frame.diff_score,
                "refine_task_paths": refine_task_paths,
                "refine_queued_count": state.get("refine_queued_count", 0),
                "mst_state": self.store.get_state(),
            }

    def close_open_event(
        self,
        *,
        project_root: Path | None = None,
        enqueue_refine: bool = True,
        reason: str = "stream_end",
    ) -> dict[str, Any]:
        with self.lock():
            state = self._load_state_unlocked()
            open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
            if not open_event:
                return self._empty_result(status="no_open_event")
            duration = self._open_event_duration(open_event) or 0.0
            if duration < env_float("WORLDMM_FRAME_STREAM_MIN_EVENT_SECONDS", 1.0):
                state["open_event"] = None
                state["ignored_short_event_count"] = int(state.get("ignored_short_event_count", 0) or 0) + 1
                state["updated_at"] = utc_now_iso()
                self._save_state_unlocked(state)
                self._append_frame_event_unlocked(
                    "frame_mst_ignored_short_event",
                    frame_index=None,
                    metadata={"reason": reason, "duration": duration},
                )
                return self._empty_result(status="ignored_short_event")
            event = self._mst_event_from_open_event(open_event, reason=reason, event_seq=int(state.get("event_count", 0) or 0))
            appended = self.store.append_events([event])
            state["open_event"] = None
            state["event_count"] = int(state.get("event_count", 0) or 0) + len(appended)
            state["closed_event_count"] = int(state.get("closed_event_count", 0) or 0) + len(appended)
            if appended:
                state["latest_event_id"] = appended[-1].get("event_id")
                state["latest_event_time_range"] = [appended[-1].get("start_time"), appended[-1].get("end_time")]
            refine_task_paths: list[str] = []
            if appended and enqueue_refine and project_root is not None and _env_bool("WORLDMM_FRAME_STREAM_ENQUEUE_REFINE", True):
                task_path = enqueue_mst_refine_task(
                    project_root=Path(project_root),
                    session_id=self.session_id,
                    backend=os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai"),
                    limit_events=max(1, int(os.getenv("WORLDMM_MST_REFINE_LIMIT_EVENTS", "10") or 10)),
                    event_id=None,
                    force_refine=False,
                    reason="frame_stream_batch",
                )
                refine_task_paths.append(str(task_path))
                state["refine_queued_count"] = int(state.get("refine_queued_count", 0) or 0) + len(refine_task_paths)
            state["updated_at"] = utc_now_iso()
            self._save_state_unlocked(state)
            for event in appended:
                self._append_frame_event_unlocked(
                    "frame_mst_closed",
                    frame_index=None,
                    metadata={
                        "event_id": event.get("event_id"),
                        "start_time": event.get("start_time"),
                        "end_time": event.get("end_time"),
                        "duration": event.get("duration"),
                        "boundary_reason": reason,
                    },
                )
            return {
                "enabled": True,
                "status": "ok",
                "event_detector_ready": cv2 is not None,
                "closed_event_count": len(appended),
                "closed_event_ids": [event.get("event_id") for event in appended],
                "has_open_event": False,
                "refine_task_paths": refine_task_paths,
                "mst_state": self.store.get_state(),
            }

    def summary(self) -> dict[str, Any]:
        state = load_frame_mst_state(self.session_dir)
        return public_frame_mst_status_block(self.session_dir, state=state)

    def _candidate_from_record(self, frame_record: dict[str, Any], current_frame_path: str | None) -> CandidateFrame | None:
        rel_path = str(frame_record.get("saved_path") or current_frame_path or "")
        path = self.session_dir / rel_path
        if not rel_path or not path.exists():
            return None
        timestamp = float(frame_record.get("relative_ts_ms", 0) or 0) / 1000.0
        frame_id = str(frame_record.get("frame_id") or f"frame_{int(frame_record.get('frame_index', 0) or 0):06d}")
        return CandidateFrame(timestamp=round(timestamp, 3), path=rel_path, chunk_id=frame_id)

    def _compute_diff(self, frame: CandidateFrame, prev_meta: Any) -> tuple[CandidateFrame, dict[str, Any] | None]:
        image = cv2.imread(str(self.session_dir / frame.path))
        if image is None:
            return frame, None
        current_hash = self.detector._phash(image)
        frame.phash = self.detector._hash_to_string(current_hash)
        frame.image_checksum = self.detector._image_checksum(image)
        if not isinstance(prev_meta, dict) or not prev_meta.get("path"):
            return frame, None
        prev_image = cv2.imread(str(self.session_dir / str(prev_meta.get("path"))))
        if prev_image is None:
            return frame, None
        prev_hash = self.detector._phash(prev_image)
        hist_diff = self.detector._histogram_diff(prev_image, image)
        phash_diff = self.detector._hash_diff(prev_hash, current_hash)
        frame.histogram_diff = round(hist_diff, 4)
        frame.phash_diff = round(phash_diff, 4)
        frame.diff_score = round(0.6 * hist_diff + 0.4 * phash_diff, 4)
        return frame, {
            "prev_timestamp": round(float(prev_meta.get("timestamp", 0.0) or 0.0), 3),
            "curr_timestamp": round(float(frame.timestamp), 3),
            "prev_frame_id": prev_meta.get("frame_id") or prev_meta.get("chunk_id"),
            "curr_frame_id": frame.chunk_id,
            "diff_score": frame.diff_score,
            "histogram_diff": frame.histogram_diff,
            "phash_diff": frame.phash_diff,
            "boundary_triggered": False,
            "created_at": utc_now_iso(),
        }

    def _update_state_with_frame(
        self,
        state: dict[str, Any],
        frame: CandidateFrame,
        diff_record: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        now = utc_now_iso()
        event_seq = int(state.get("event_count", 0) or 0)
        open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
        closed_events: list[dict[str, Any]] = []
        opened_events: list[dict[str, Any]] = []
        if open_event is None:
            open_event = self._new_open_event(frame)
            opened_events.append(
                {
                    "open_event_id": open_event.get("open_event_id"),
                    "start_time": open_event.get("start_time"),
                    "frame_id": frame.chunk_id,
                }
            )
            self._append_frame_event_unlocked(
                "frame_mst_opened",
                frame_index=self._frame_index(frame),
                metadata={"start_time": open_event.get("start_time"), "frame_id": frame.chunk_id},
            )
        self._extend_open_event(open_event, frame)
        duration = self._open_event_duration(open_event) or 0.0
        visual_boundary = frame.diff_score >= self.detector.diff_threshold and duration >= self.detector.min_event_duration
        max_duration = duration >= self.detector.max_event_duration
        if visual_boundary or max_duration:
            reason = "visual_change" if visual_boundary else "max_duration"
            if diff_record is not None:
                diff_record["boundary_triggered"] = True
                diff_record["boundary_reason"] = reason
            closed_events.append(self._mst_event_from_open_event(open_event, reason=reason, event_seq=event_seq))
            event_seq += 1
            open_event = self._new_open_event(frame)
            opened_events.append(
                {
                    "open_event_id": open_event.get("open_event_id"),
                    "start_time": open_event.get("start_time"),
                    "frame_id": frame.chunk_id,
                    "reason": f"after_{reason}",
                }
            )
            self._append_frame_event_unlocked(
                "frame_mst_opened",
                frame_index=self._frame_index(frame),
                metadata={"start_time": open_event.get("start_time"), "frame_id": frame.chunk_id, "reason": f"after_{reason}"},
            )
        state["session_id"] = self.session_id
        state["input_mode"] = frame_stream_input_mode(state.get("input_mode") or "frame_audio_stream")
        state["event_count"] = event_seq
        state["closed_event_count"] = int(state.get("closed_event_count", 0) or 0) + len(closed_events)
        state["open_event"] = open_event
        state["last_accepted_frame"] = self._frame_meta(frame, role="latest")
        state["last_boundary_frame"] = (open_event.get("start_frame") if isinstance(open_event, dict) else None)
        if closed_events:
            state["latest_event_id"] = closed_events[-1].get("event_id")
            state["latest_event_time_range"] = [closed_events[-1].get("start_time"), closed_events[-1].get("end_time")]
        state["latest_diff_score"] = frame.diff_score
        state["updated_at"] = now
        return closed_events, state, opened_events

    def _new_open_event(self, frame: CandidateFrame) -> dict[str, Any]:
        meta = self._frame_meta(frame, role="start")
        return {
            "open_event_id": f"open_frame_{int(round(float(frame.timestamp) * 1000)):09d}",
            "start_time": round(float(frame.timestamp), 3),
            "last_update_time": round(float(frame.timestamp), 3),
            "start_frame": meta,
            "latest_frame": meta,
            "keyframes": [meta],
            "source_frame_indices": [self._frame_index(frame)],
            "diff_scores": [],
            "diff_stats": {"max_diff": 0.0, "mean_diff": 0.0, "last_diff": 0.0},
            "status": "open",
        }

    def _extend_open_event(self, open_event: dict[str, Any], frame: CandidateFrame) -> None:
        meta = self._frame_meta(frame, role="latest")
        open_event["last_update_time"] = round(float(frame.timestamp), 3)
        open_event["latest_frame"] = meta
        frame_indices = [int(item) for item in open_event.get("source_frame_indices", []) or [] if item is not None]
        idx = self._frame_index(frame)
        if idx not in frame_indices:
            frame_indices.append(idx)
        open_event["source_frame_indices"] = frame_indices
        keyframes = [item for item in open_event.get("keyframes", []) if isinstance(item, dict)]
        if not any(str(item.get("path") or "") == str(meta.get("path") or "") for item in keyframes):
            keyframes.append(meta)
        open_event["keyframes"] = self._select_keyframes(keyframes)
        scores = [float(item) for item in open_event.get("diff_scores", []) if item is not None]
        scores.append(float(frame.diff_score or 0.0))
        open_event["diff_scores"] = scores[-100:]
        open_event["diff_stats"] = {
            "max_diff": round(max(scores or [0.0]), 4),
            "mean_diff": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "last_diff": round(float(frame.diff_score or 0.0), 4),
        }

    def _mst_event_from_open_event(self, open_event: dict[str, Any], *, reason: str, event_seq: int) -> dict[str, Any]:
        start = float(open_event.get("start_time", 0.0) or 0.0)
        end = float(open_event.get("last_update_time", start) or start)
        start_ms = int(round(start * 1000))
        end_ms = int(round(end * 1000))
        event = mst_event_stub(
            session_id=self.session_id,
            chunk_id_value="frame_stream",
            start_time=start,
            end_time=end,
            boundary_reason=reason,
            diff_score=float((open_event.get("diff_stats") or {}).get("max_diff") or 0.0),
            diff_stats=dict(open_event.get("diff_stats") or {}),
            chunk_path="stream/frames",
            boundary_index=event_seq,
        )
        event["event_id"] = f"mst_frame_{self.session_id}_{start_ms:09d}_{end_ms:09d}_{event_seq:04d}"
        event["source"] = {
            "type": "frame_audio_stream",
            "input_source": "frame_stream",
            "chunk_path": "stream/frames",
        }
        event["input_source"] = "frame_stream"
        event["source_frame_indices"] = list(open_event.get("source_frame_indices") or [])
        event["keyframes"] = self._select_keyframes([item for item in open_event.get("keyframes", []) if isinstance(item, dict)])
        event["evidence_frames"] = list(event["keyframes"])
        event["transcript"] = ""
        event["transcript_segments"] = []
        event["needs_refine"] = True
        event["caption_source"] = "placeholder"
        event["event_caption_placeholder"] = (
            f"A provisional frame-stream event occurs between {start:.1f}s and {end:.1f}s "
            f"from {len(event.get('source_frame_indices') or [])} uploaded frames."
        )
        event["retrieval_text"] = build_retrieval_text(event)
        return event

    def _frame_meta(self, frame: CandidateFrame, role: str) -> dict[str, Any]:
        return {
            "timestamp": round(float(frame.timestamp), 3),
            "path": frame.path,
            "role": role,
            "diff_score": round(float(frame.diff_score or 0.0), 4),
            "histogram_diff": round(float(frame.histogram_diff or 0.0), 4),
            "phash_diff": round(float(frame.phash_diff or 0.0), 4),
            "phash": frame.phash,
            "image_checksum": frame.image_checksum,
            "frame_id": frame.chunk_id,
            "frame_index": self._frame_index(frame),
        }

    def _select_keyframes(self, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_items = max(2, int(self.max_keyframes))
        if len(keyframes) <= max_items:
            return keyframes
        start = keyframes[0]
        end = keyframes[-1]
        middle = sorted(keyframes[1:-1], key=lambda item: float(item.get("diff_score", 0.0) or 0.0), reverse=True)
        selected = [start] + middle[: max_items - 2] + [end]
        return sorted(selected, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))

    def _frame_index(self, frame: CandidateFrame) -> int:
        try:
            return int(str(frame.chunk_id).split("_")[-1])
        except Exception:
            return -1

    def _load_state_unlocked(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def _save_state_unlocked(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def _append_frame_event_unlocked(self, event_type: str, frame_index: int | None, metadata: dict[str, Any]) -> None:
        self.frame_events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_id": f"frame_mst_{uuid4().hex[:12]}",
            "session_id": self.session_id,
            "event_type": event_type,
            "frame_index": frame_index,
            "timestamp": utc_now_iso(),
            "metadata": metadata,
        }
        with self.frame_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _append_diff_record_unlocked(self, record: dict[str, Any]) -> None:
        self.diff_records_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**record, "session_id": self.session_id}
        with self.diff_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _open_event_duration(self, open_event: Any) -> float | None:
        if not isinstance(open_event, dict):
            return None
        try:
            return round(float(open_event.get("last_update_time", 0.0) or 0.0) - float(open_event.get("start_time", 0.0) or 0.0), 3)
        except Exception:
            return None

    def _empty_result(self, status: str) -> dict[str, Any]:
        return {
            "enabled": True,
            "status": status,
            "event_detector_ready": cv2 is not None,
            "has_open_event": False,
            "closed_event_count": 0,
            "closed_event_ids": [],
            "refine_task_paths": [],
            "mst_state": self.store.get_state(),
        }


def load_frame_mst_state(session_dir: Path) -> dict[str, Any]:
    payload = read_json(Path(session_dir) / "stream" / "frame_event_state.json", default={})
    return payload if isinstance(payload, dict) else {}


def public_frame_mst_status_block(session_dir: Path, *, state: dict[str, Any] | None = None, input_mode: Any = None) -> dict[str, Any]:
    from online_pipeline.frame_stream import frame_stream_input_mode, is_frame_stream_mode

    state = state if isinstance(state, dict) else load_frame_mst_state(Path(session_dir))
    mode = input_mode if input_mode is not None else state.get("input_mode")
    normalized_mode = frame_stream_input_mode(mode)
    enabled = _env_bool("WORLDMM_FRAME_STREAM_ENABLE_MST", True) and is_frame_stream_mode(mode)
    mst_state = MSTStore(Path(session_dir)).get_state()
    open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
    duration = None
    if open_event:
        try:
            duration = round(float(open_event.get("last_update_time", 0.0) or 0.0) - float(open_event.get("start_time", 0.0) or 0.0), 3)
        except Exception:
            duration = None
    return {
        "enabled": bool(enabled),
        "input_mode": normalized_mode,
        "event_detector_ready": cv2 is not None,
        "has_open_event": bool(open_event),
        "open_event_start_time": open_event.get("start_time") if open_event else None,
        "open_event_duration": duration,
        "closed_event_count": int(state.get("closed_event_count", 0) or 0),
        "latest_event_id": state.get("latest_event_id"),
        "latest_event_time_range": state.get("latest_event_time_range"),
        "mst_ready": bool(mst_state.get("short_term_ready")),
        "mst_version": int(mst_state.get("mst_version", 0) or 0),
        "mst_event_count": int(mst_state.get("event_count", 0) or 0),
        "refine_queued_count": int(state.get("refine_queued_count", 0) or 0),
        "ignored_short_event_count": int(state.get("ignored_short_event_count", 0) or 0),
        "latest_diff_score": state.get("latest_diff_score"),
        "updated_at": state.get("updated_at"),
    }
