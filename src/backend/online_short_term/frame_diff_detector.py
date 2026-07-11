from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from online_short_term.schemas import env_float, rel_to_session

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class CandidateFrame:
    timestamp: float
    path: str
    chunk_id: str = ""
    diff_score: float = 0.0
    histogram_diff: float = 0.0
    phash_diff: float = 0.0
    phash: str = ""
    image_checksum: str = ""
    duplicate_of_previous: bool = False


class FrameDiffEventDetector:
    def __init__(
        self,
        session_dir: Path,
        candidate_fps: float | None = None,
        diff_threshold: float | None = None,
        min_event_duration: float | None = None,
        max_event_duration: float | None = None,
        min_boundary_gap: float | None = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for frame-diff M_st detection")
        self.session_dir = session_dir
        self.keyframes_dir = session_dir / "stream" / "keyframes"
        self.candidate_fps = candidate_fps or env_float("WORLDMM_MST_CANDIDATE_FPS", 1.0)
        self.diff_threshold = diff_threshold or env_float("WORLDMM_MST_DIFF_THRESHOLD", 0.40)
        self.min_event_duration = min_event_duration or env_float("WORLDMM_MST_MIN_EVENT_DURATION", 2.0)
        self.max_event_duration = max_event_duration or env_float("WORLDMM_MST_MAX_EVENT_DURATION", 8.0)
        self.min_boundary_gap = min_boundary_gap or env_float("WORLDMM_MST_MIN_BOUNDARY_GAP", 1.5)

    def detect(
        self,
        *,
        chunk_path: Path,
        chunk_id: str,
        chunk_global_start_time: float,
        chunk_global_end_time: float,
    ) -> dict[str, Any]:
        frames = self._extract_candidate_frames(
            chunk_path=chunk_path,
            chunk_id=chunk_id,
            chunk_global_start_time=chunk_global_start_time,
            chunk_global_end_time=chunk_global_end_time,
        )
        if not frames:
            return {"frames": [], "events": []}
        self._compute_frame_diffs(frames)
        events = self._cut_events(frames, chunk_global_end_time)
        return {"frames": [frame.__dict__ for frame in frames], "events": events}

    def detect_stream_chunk(
        self,
        *,
        chunk_path: Path,
        chunk_id: str,
        chunk_global_start_time: float,
        chunk_global_end_time: float,
        previous_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if str(chunk_id).startswith("upload_"):
            raise RuntimeError(f"detect_stream_chunk requires processing chunk id, got upload chunk id: {chunk_id}")
        if chunk_global_end_time < chunk_global_start_time:
            raise RuntimeError(
                f"processing chunk timeline must be monotonic, got start={chunk_global_start_time}, end={chunk_global_end_time}, chunk_id={chunk_id}"
            )
        frames = self._extract_candidate_frames(
            chunk_path=chunk_path,
            chunk_id=chunk_id,
            chunk_global_start_time=chunk_global_start_time,
            chunk_global_end_time=chunk_global_end_time,
        )
        state = dict(previous_state or {})
        if not frames:
            return {
                "candidate_frames": [],
                "closed_events": [],
                "new_event_state": state,
                "diff_records": [],
            }
        diff_records = self._compute_stream_frame_diffs(frames, state)
        closed_events, new_state = self._update_stream_event_state(
            frames=frames,
            previous_state=state,
            diff_records=diff_records,
            chunk_id=chunk_id,
        )
        return {
            "candidate_frames": [frame.__dict__ for frame in frames],
            "closed_events": closed_events,
            "new_event_state": new_state,
            "diff_records": diff_records,
        }

    def _extract_candidate_frames(
        self,
        *,
        chunk_path: Path,
        chunk_id: str,
        chunk_global_start_time: float,
        chunk_global_end_time: float,
    ) -> list[CandidateFrame]:
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(chunk_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open chunk video: {chunk_path}")
        step = 1.0 / max(self.candidate_fps, 0.1)
        duration = max(0.0, chunk_global_end_time - chunk_global_start_time)
        offsets = list(np.arange(0.0, duration + 1e-3, step))
        if not offsets or offsets[-1] < duration - 0.25:
            offsets.append(duration)
        frames: list[CandidateFrame] = []
        for offset in offsets:
            global_ts = chunk_global_start_time + float(offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(offset)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            safe_chunk_id = "".join(ch if ch.isalnum() else "_" for ch in str(chunk_id))[-80:]
            out_path = self.keyframes_dir / f"kf_{int(round(global_ts * 1000)):09d}_{safe_chunk_id}.jpg"
            cv2.imwrite(str(out_path), frame)
            frames.append(CandidateFrame(timestamp=round(global_ts, 3), path=rel_to_session(self.session_dir, out_path), chunk_id=chunk_id))
        cap.release()
        return frames

    def _compute_frame_diffs(self, frames: list[CandidateFrame]) -> None:
        prev_img = None
        prev_hash = None
        for frame in frames:
            img = cv2.imread(str(self.session_dir / frame.path))
            if img is None:
                continue
            if prev_timestamp is not None and float(frame.timestamp) < float(prev_timestamp) - 1e-6:
                frame.duplicate_of_previous = True
                continue
            current_hash = self._phash(img)
            frame.phash = self._hash_to_string(current_hash)
            frame.image_checksum = self._image_checksum(img)
            if prev_img is not None and prev_hash is not None:
                hist_diff = self._histogram_diff(prev_img, img)
                phash_diff = self._hash_diff(prev_hash, current_hash)
                frame.histogram_diff = round(hist_diff, 4)
                frame.phash_diff = round(phash_diff, 4)
                frame.diff_score = round(0.6 * hist_diff + 0.4 * phash_diff, 4)
            prev_img = img
            prev_hash = current_hash

    def _compute_stream_frame_diffs(self, frames: list[CandidateFrame], previous_state: dict[str, Any]) -> list[dict[str, Any]]:
        diff_records: list[dict[str, Any]] = []
        prev_img = None
        prev_hash = None
        prev_meta = previous_state.get("last_candidate_frame") if isinstance(previous_state, dict) else None
        prev_timestamp = None
        prev_chunk_id = None
        prev_checksum = None
        prev_phash_string = None
        skipped_cross_boundary_duplicate = False
        if isinstance(prev_meta, dict) and prev_meta.get("path"):
            prev_path = self.session_dir / str(prev_meta.get("path"))
            prev_img = cv2.imread(str(prev_path))
            if prev_img is not None:
                prev_hash = self._phash(prev_img)
                prev_timestamp = float(prev_meta.get("timestamp", 0.0) or 0.0)
                prev_chunk_id = str(prev_meta.get("chunk_id") or "")
                prev_checksum = str(prev_meta.get("image_checksum") or self._image_checksum(prev_img))
                prev_phash_string = str(prev_meta.get("phash") or self._hash_to_string(prev_hash))

        for frame in frames:
            img = cv2.imread(str(self.session_dir / frame.path))
            if img is None:
                continue
            current_hash = self._phash(img)
            frame.phash = self._hash_to_string(current_hash)
            frame.image_checksum = self._image_checksum(img)
            if prev_img is not None and prev_hash is not None and prev_timestamp is not None:
                is_cross_chunk = bool(prev_chunk_id and prev_chunk_id != frame.chunk_id)
                duplicate_boundary_frame = (
                    is_cross_chunk
                    and not skipped_cross_boundary_duplicate
                    and (
                        abs(float(frame.timestamp) - float(prev_timestamp)) < 1e-6
                        or (prev_checksum is not None and prev_checksum == frame.image_checksum)
                        or (prev_phash_string is not None and prev_phash_string == frame.phash)
                    )
                )
                if duplicate_boundary_frame:
                    frame.duplicate_of_previous = True
                    frame.diff_score = 0.0
                    frame.histogram_diff = 0.0
                    frame.phash_diff = 0.0
                    # Keep the previous unique frame as the comparison anchor so
                    # the next non-duplicate frame becomes the real cross-chunk
                    # diff record.
                    skipped_cross_boundary_duplicate = True
                    continue
                hist_diff = self._histogram_diff(prev_img, img)
                phash_diff = self._hash_diff(prev_hash, current_hash)
                frame.histogram_diff = round(hist_diff, 4)
                frame.phash_diff = round(phash_diff, 4)
                frame.diff_score = round(0.6 * hist_diff + 0.4 * phash_diff, 4)
                diff_records.append(
                    {
                        "prev_timestamp": round(float(prev_timestamp), 3),
                        "curr_timestamp": round(float(frame.timestamp), 3),
                        "prev_chunk_id": prev_chunk_id,
                        "curr_chunk_id": frame.chunk_id,
                        "prev_processing_chunk_id": prev_chunk_id,
                        "curr_processing_chunk_id": frame.chunk_id,
                        "diff_score": frame.diff_score,
                        "histogram_diff": frame.histogram_diff,
                        "phash_diff": frame.phash_diff,
                        "is_cross_chunk": is_cross_chunk,
                        "boundary_triggered": False,
                    }
                )
            prev_img = img
            prev_hash = current_hash
            prev_timestamp = frame.timestamp
            prev_chunk_id = frame.chunk_id
            prev_checksum = frame.image_checksum
            prev_phash_string = frame.phash
            skipped_cross_boundary_duplicate = False
        return diff_records

    def _update_stream_event_state(
        self,
        *,
        frames: list[CandidateFrame],
        previous_state: dict[str, Any],
        diff_records: list[dict[str, Any]],
        chunk_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state = dict(previous_state or {})
        event_seq = int(state.get("event_seq", 0) or 0)
        last_boundary_time = state.get("last_boundary_time")
        try:
            last_boundary = float(last_boundary_time)
        except Exception:
            last_boundary = float(frames[0].timestamp)
        open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
        closed_events: list[dict[str, Any]] = []

        last_effective_frame: CandidateFrame | None = None
        for frame in frames:
            if frame.duplicate_of_previous:
                continue
            if open_event is None:
                open_event = self._new_open_event(frame)
                last_boundary = float(frame.timestamp)
            self._extend_open_event(open_event, frame)
            last_effective_frame = frame
            duration = float(frame.timestamp) - float(open_event.get("start_time", frame.timestamp))
            is_visual_boundary = (
                frame.diff_score >= self.diff_threshold
                and duration >= self.min_event_duration
                and float(frame.timestamp) - last_boundary >= self.min_boundary_gap
            )
            is_max_duration = duration >= self.max_event_duration
            if is_visual_boundary or is_max_duration:
                reason = "visual_change" if is_visual_boundary else "max_duration"
                for record in diff_records:
                    if abs(float(record.get("curr_timestamp", -1.0)) - float(frame.timestamp)) < 1e-6:
                        record["boundary_triggered"] = True
                        record["boundary_reason"] = reason
                closed_events.append(self._event_from_open_event(open_event, reason, event_seq))
                event_seq += 1
                open_event = self._new_open_event(frame)
                last_boundary = float(frame.timestamp)

        if last_effective_frame is None:
            state["event_seq"] = event_seq
            state["last_boundary_time"] = round(float(last_boundary), 3)
            state["open_event"] = open_event
            return closed_events, state

        last_frame = last_effective_frame
        new_state = {
            "session_id": self.session_dir.name,
            "stream_id": state.get("stream_id"),
            "event_seq": event_seq,
            "last_candidate_frame": self._frame_meta(last_frame, role="latest"),
            "last_boundary_time": round(float(last_boundary), 3),
            "open_event": open_event,
        }
        return closed_events, new_state

    def close_stream_open_event(self, previous_state: dict[str, Any] | None, reason: str = "stream_end") -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state = dict(previous_state or {})
        open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
        if not open_event:
            state["open_event"] = None
            return [], state
        event_seq = int(state.get("event_seq", 0) or 0)
        closed = self._event_from_open_event(open_event, reason, event_seq)
        state["event_seq"] = event_seq + 1
        state["last_boundary_time"] = closed.get("end_time")
        state["open_event"] = None
        return [closed], state

    def _new_open_event(self, frame: CandidateFrame) -> dict[str, Any]:
        frame_meta = self._frame_meta(frame, role="start")
        return {
            "open_event_id": f"open_{int(round(float(frame.timestamp) * 1000)):09d}",
            "start_time": round(float(frame.timestamp), 3),
            "last_update_time": round(float(frame.timestamp), 3),
            "start_frame": frame_meta,
            "latest_frame": frame_meta,
            "keyframes": [frame_meta],
            "source_chunks": [frame.chunk_id],
            "diff_scores": [],
            "diff_stats": {"max_diff": 0.0, "mean_diff": 0.0, "last_diff": 0.0},
            "status": "open",
        }

    def _extend_open_event(self, open_event: dict[str, Any], frame: CandidateFrame) -> None:
        frame_meta = self._frame_meta(frame, role="latest")
        open_event["last_update_time"] = round(float(frame.timestamp), 3)
        open_event["latest_frame"] = frame_meta
        chunks = list(open_event.get("source_chunks") or [])
        if frame.chunk_id and frame.chunk_id not in chunks:
            chunks.append(frame.chunk_id)
        open_event["source_chunks"] = chunks
        keyframes = [item for item in open_event.get("keyframes", []) if isinstance(item, dict)]
        if not any(abs(float(item.get("timestamp", -1.0)) - float(frame.timestamp)) < 1e-6 for item in keyframes):
            keyframes.append(frame_meta)
        open_event["keyframes"] = keyframes[-12:]
        scores = [float(item) for item in open_event.get("diff_scores", []) if item is not None]
        scores.append(float(frame.diff_score or 0.0))
        open_event["diff_scores"] = scores[-100:]
        open_event["diff_stats"] = {
            "max_diff": round(max(scores or [0.0]), 4),
            "mean_diff": round(float(np.mean(scores)) if scores else 0.0, 4),
            "last_diff": round(float(frame.diff_score or 0.0), 4),
        }

    def _event_from_open_event(self, open_event: dict[str, Any], reason: str, event_seq: int) -> dict[str, Any]:
        keyframes = [item for item in open_event.get("keyframes", []) if isinstance(item, dict)]
        start_frame = dict(open_event.get("start_frame") or (keyframes[0] if keyframes else {}))
        end_frame = dict(open_event.get("latest_frame") or (keyframes[-1] if keyframes else start_frame))
        start_frame["role"] = "start"
        end_frame["role"] = "end"
        compact_keyframes = [start_frame]
        if end_frame.get("path") != start_frame.get("path") or end_frame.get("timestamp") != start_frame.get("timestamp"):
            compact_keyframes.append(end_frame)
        source_chunks = list(dict.fromkeys(str(item) for item in open_event.get("source_chunks", []) if item))
        diff_stats = dict(open_event.get("diff_stats") or {})
        return {
            "start_time": round(float(open_event.get("start_time", start_frame.get("timestamp", 0.0)) or 0.0), 3),
            "end_time": round(float(open_event.get("last_update_time", end_frame.get("timestamp", 0.0)) or 0.0), 3),
            "boundary_reason": reason,
            "diff_score": float(diff_stats.get("max_diff") or 0.0),
            "diff_stats": diff_stats,
            "keyframes": compact_keyframes,
            "source_chunks": source_chunks,
            "cross_chunk": len(source_chunks) > 1,
            "boundary_index": int(event_seq),
        }

    def _frame_meta(self, frame: CandidateFrame, role: str = "candidate") -> dict[str, Any]:
        return {
            "timestamp": round(float(frame.timestamp), 3),
            "path": frame.path,
            "role": role,
            "diff_score": round(float(frame.diff_score or 0.0), 4),
            "histogram_diff": round(float(frame.histogram_diff or 0.0), 4),
            "phash_diff": round(float(frame.phash_diff or 0.0), 4),
            "phash": frame.phash,
            "image_checksum": frame.image_checksum,
            "duplicate_of_previous": bool(frame.duplicate_of_previous),
            "chunk_id": frame.chunk_id,
        }

    def _cut_events(self, frames: list[CandidateFrame], chunk_end: float) -> list[dict[str, Any]]:
        events = []
        current_start = frames[0]
        last_boundary_time = current_start.timestamp
        max_diffs: list[float] = []
        for frame in frames[1:]:
            duration = frame.timestamp - current_start.timestamp
            max_diffs.append(frame.diff_score)
            is_visual_boundary = (
                frame.diff_score >= self.diff_threshold
                and duration >= self.min_event_duration
                and frame.timestamp - last_boundary_time >= self.min_boundary_gap
            )
            is_max_duration = duration >= self.max_event_duration
            if is_visual_boundary or is_max_duration:
                reason = "visual_change" if is_visual_boundary else "max_duration"
                events.append(self._event_record(current_start, frame, reason, max_diffs))
                current_start = frame
                last_boundary_time = frame.timestamp
                max_diffs = []
        if current_start.timestamp < chunk_end - 0.1:
            end_frame = frames[-1]
            reason = "low_motion" if not events and max(max_diffs or [0.0]) < self.diff_threshold else "chunk_end"
            events.append(self._event_record(current_start, end_frame, reason, max_diffs))
        if not events and frames:
            events.append(self._event_record(frames[0], frames[-1], "low_motion", [f.diff_score for f in frames]))
        return events

    def _event_record(
        self,
        start_frame: CandidateFrame,
        end_frame: CandidateFrame,
        reason: str,
        diffs: list[float],
    ) -> dict[str, Any]:
        max_diff = max(diffs or [end_frame.diff_score, 0.0])
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        return {
            "start_time": start_frame.timestamp,
            "end_time": max(end_frame.timestamp, start_frame.timestamp),
            "boundary_reason": reason,
            "diff_score": max_diff,
            "diff_stats": {"max_diff": round(max_diff, 4), "mean_diff": round(mean_diff, 4)},
            "keyframes": [
                {"timestamp": start_frame.timestamp, "path": start_frame.path, "role": "start"},
                {"timestamp": end_frame.timestamp, "path": end_frame.path, "role": "end"},
            ],
        }

    def _histogram_diff(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        return float(np.clip(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA), 0.0, 1.0))

    def _phash(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        low = dct[:8, :8]
        median = np.median(low[1:, 1:])
        return low > median

    def _hash_diff(self, hash_a: np.ndarray, hash_b: np.ndarray) -> float:
        return float(np.count_nonzero(hash_a != hash_b) / hash_a.size)

    def _hash_to_string(self, value: np.ndarray) -> str:
        return "".join("1" if bool(item) else "0" for item in value.astype(bool).flatten())

    def _image_checksum(self, img: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
