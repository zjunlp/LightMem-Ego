from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from online_current.schemas import (
    as_float,
    current_frame_name,
    env_float,
    env_int,
    frame_id,
    rel_to_session,
)
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


class MCurStore:
    """Rolling current-memory buffer for the latest stream window.

    M_cur is deliberately lightweight: it stores sampled frame metadata,
    recent transcript segments, and an open event summary. It does not run VLM
    or LLM during updates.
    """

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.current_dir = self.session_dir / "current"
        self.frames_dir = self.current_dir / "frames"
        self.state_path = self.current_dir / "current_state.json"
        self.frames_path = self.current_dir / "current_frames.jsonl"
        self.transcript_path = self.current_dir / "transcript_partial.jsonl"
        self.open_event_path = self.current_dir / "open_event.json"
        self.window_seconds = env_float("EM2MEM_MCUR_WINDOW_SECONDS", 30.0)
        self.core_seconds = env_float("EM2MEM_MCUR_CORE_SECONDS", 10.0)
        self.max_frames = env_int("EM2MEM_MCUR_MAX_FRAMES", 30)
        self.stale_seconds = env_float("EM2MEM_MCUR_STALE_SECONDS", 60.0)

    def load(self) -> dict[str, Any]:
        state = read_json(self.state_path, default={})
        return state if isinstance(state, dict) else {}

    def load_frames(self) -> list[dict[str, Any]]:
        if not self.frames_path.exists():
            return []
        frames: list[dict[str, Any]] = []
        with self.frames_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    item["timestamp"] = round(as_float(item.get("timestamp")), 3)
                    frames.append(item)
        frames.sort(key=lambda item: as_float(item.get("timestamp")))
        return frames

    def load_transcript_segments(self) -> list[dict[str, Any]]:
        if not self.transcript_path.exists():
            return []
        segments: list[dict[str, Any]] = []
        with self.transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    segments.append(item)
        return segments

    def get_state(self) -> dict[str, Any]:
        state = self.load()
        if state:
            state["is_stale"] = self.is_stale(state)
        return state

    def is_ready(self) -> bool:
        state = self.get_state()
        return bool(state.get("mcur_ready") and (self.frames_path.exists() or state.get("current_text_ready") or state.get("audio_current_ready")))

    def is_stale(self, state: dict[str, Any] | None = None) -> bool:
        state = state or self.load()
        if not state or not state.get("mcur_ready"):
            return True
        last_update = str(state.get("last_update_at") or state.get("updated_at") or "")
        try:
            if last_update.endswith("Z"):
                last_update = last_update[:-1] + "+00:00"
            updated = datetime.fromisoformat(last_update)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds()
            if age > self.stale_seconds:
                return True
        except Exception:
            return True
        return False

    def clear(self) -> dict[str, Any]:
        if self.current_dir.exists():
            shutil.rmtree(self.current_dir)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        state = self._empty_state()
        write_json_atomic(self.state_path, state)
        self.frames_path.write_text("", encoding="utf-8")
        self.transcript_path.write_text("", encoding="utf-8")
        write_json_atomic(self.open_event_path, {})
        return state

    def update_from_chunk(
        self,
        chunk_info: dict[str, Any],
        frames: list[dict[str, Any]],
        transcript_segments: list[dict[str, Any]],
        diff_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        current_time = as_float(chunk_info.get("end_time"), 0.0)
        if frames:
            current_time = max(current_time, max(as_float(frame.get("timestamp")) for frame in frames))
        window_start = max(0.0, current_time - self.window_seconds)

        existing = self.load_frames()
        merged_by_second_path: dict[tuple[int, str], dict[str, Any]] = {}
        for item in existing:
            ts = as_float(item.get("timestamp"))
            if ts < window_start - 1e-6 or ts > current_time + 1e-6:
                continue
            path = str(item.get("path") or "")
            merged_by_second_path[(int(round(ts * 1000)), path)] = item

        for frame in frames:
            ts = round(as_float(frame.get("timestamp")), 3)
            if ts < window_start - 1e-6 or ts > current_time + 1e-6:
                continue
            source_rel = str(frame.get("path") or frame.get("source_path") or "")
            source_abs = self.session_dir / source_rel
            cur_abs = self.frames_dir / current_frame_name(ts)
            if source_abs.suffix.lower() in {".jpeg", ".png", ".webp"}:
                cur_abs = cur_abs.with_suffix(source_abs.suffix.lower())
            if source_abs.exists() and not cur_abs.exists():
                try:
                    os.link(source_abs, cur_abs)
                except Exception:
                    shutil.copy2(source_abs, cur_abs)
            cur_rel = rel_to_session(self.session_dir, cur_abs)
            diff_score = as_float(frame.get("diff_score"), 0.0)
            role = "change" if diff_score >= 0.4 else "recent"
            item = {
                "frame_id": frame_id(ts),
                "session_id": self.session_id,
                "timestamp": ts,
                "path": cur_rel,
                "source_path": source_rel,
                "diff_score": round(diff_score, 4),
                "role": role,
                "created_at": str(frame.get("created_at") or utc_now_iso()),
            }
            merged_by_second_path[(int(round(ts * 1000)), cur_rel)] = item

        merged = sorted(merged_by_second_path.values(), key=lambda item: as_float(item.get("timestamp")))
        if self.max_frames > 0 and len(merged) > self.max_frames:
            merged = merged[-self.max_frames :]
            window_start = as_float(merged[0].get("timestamp"), window_start)
        if merged:
            latest_ts = max(as_float(item.get("timestamp")) for item in merged)
            for item in merged:
                if abs(as_float(item.get("timestamp")) - latest_ts) < 1e-6:
                    item["role"] = "latest"

        kept_paths = {str(item.get("path") or "") for item in merged}
        self._write_jsonl(self.frames_path, merged)
        self._cleanup_current_frames(kept_paths)

        transcript_segments = [
            dict(seg)
            for seg in transcript_segments
            if as_float(seg.get("end"), as_float(seg.get("start"))) >= window_start
            and as_float(seg.get("start")) <= current_time
        ]
        self._write_jsonl(self.transcript_path, transcript_segments)
        transcript = " ".join(str(seg.get("text") or "").strip() for seg in transcript_segments if seg.get("text")).strip()

        diff_values = [as_float(item.get("diff_score")) for item in merged]
        diff_stats = dict(diff_stats or {})
        if diff_values:
            diff_stats.setdefault("max_diff", max(diff_values))
            diff_stats.setdefault("mean_diff", sum(diff_values) / len(diff_values))
            diff_stats.setdefault("last_diff", diff_values[-1])
        else:
            diff_stats.setdefault("max_diff", 0.0)
            diff_stats.setdefault("mean_diff", 0.0)
            diff_stats.setdefault("last_diff", 0.0)

        open_start = max(window_start, current_time - self.core_seconds)
        open_event = {
            "open_event_id": f"mcur_open_{int(round(open_start * 1000)):09d}_{int(round(current_time * 1000)):09d}",
            "session_id": self.session_id,
            "start_time": round(open_start, 3),
            "end_time": round(current_time, 3),
            "duration": round(max(0.0, current_time - open_start), 3),
            "boundary_status": "open",
            "diff_stats": {
                "max_diff": round(as_float(diff_stats.get("max_diff")), 4),
                "mean_diff": round(as_float(diff_stats.get("mean_diff")), 4),
                "last_diff": round(as_float(diff_stats.get("last_diff")), 4),
            },
            "keyframes": [
                {
                    "timestamp": item.get("timestamp"),
                    "path": item.get("path"),
                    "role": item.get("role"),
                    "diff_score": item.get("diff_score"),
                }
                for item in merged
                if as_float(item.get("timestamp")) >= open_start - 1e-6
            ],
            "transcript": transcript,
            "transcript_segments": transcript_segments,
            "status": "open",
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.open_event_path, open_event)

        previous_state = self.load()
        version = int(previous_state.get("mcur_version", 0) or 0) + 1
        state = {
            "session_id": self.session_id,
            "mcur_ready": bool(merged),
            "mcur_version": version,
            "window_seconds": self.window_seconds,
            "core_seconds": self.core_seconds,
            "window_start_time": round(window_start, 3),
            "window_end_time": round(current_time, 3),
            "current_time": round(current_time, 3),
            "frame_count": len(merged),
            "transcript_segment_count": len(transcript_segments),
            "open_event_id": open_event["open_event_id"],
            "open_event_start": open_event["start_time"],
            "open_event_end": open_event["end_time"],
            "last_update_at": utc_now_iso(),
            "is_stale": False,
            "stale_seconds": self.stale_seconds,
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.state_path, state)
        return state

    def update_from_frame_stream(
        self,
        *,
        frame_index: int,
        frame_path: str,
        relative_ts_ms: int,
        client_ts_ms: int | None = None,
        source: str = "camera_take_photo",
    ) -> dict[str, Any]:
        """Feed one lossy realtime frame into the existing M_cur rolling store."""
        timestamp = round(max(0.0, float(relative_ts_ms) / 1000.0), 3)
        state = self.update_from_chunk(
            chunk_info={
                "chunk_id": f"frame_{int(frame_index):06d}",
                "start_time": timestamp,
                "end_time": timestamp,
                "source": "frame_stream",
            },
            frames=[
                {
                    "timestamp": timestamp,
                    "path": frame_path,
                    "source_path": frame_path,
                    "diff_score": 0.0,
                    "frame_index": int(frame_index),
                    "client_ts_ms": client_ts_ms,
                    "source": source,
                    "created_at": utc_now_iso(),
                }
            ],
            transcript_segments=self.load_transcript_segments(),
        )
        current_frame_path = None
        for item in reversed(self.load_frames()):
            if str(item.get("source_path") or "") == str(frame_path):
                current_frame_path = str(item.get("path") or "")
                break
        return {**state, "current_frame_path": current_frame_path}

    def update_transcript_segments(
        self,
        segments: list[dict[str, Any]],
        source: str = "stream_asr",
    ) -> dict[str, Any]:
        segments = [dict(seg) for seg in segments if isinstance(seg, dict) and str(seg.get("text") or "").strip()]
        if not segments:
            return {"backfilled_current": False, "segment_count": 0, "mcur_version": int(self.load().get("mcur_version", 0) or 0)}
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if not self.frames_path.exists():
            self.frames_path.write_text("", encoding="utf-8")
        state = self.load()
        if not state:
            return self._write_audio_only_transcript_state(segments, source=source, state={})
        try:
            window_start = as_float(state.get("window_start_time"), 0.0)
            window_end = as_float(state.get("window_end_time"), state.get("current_time") or 0.0)
        except Exception:
            return self._write_audio_only_transcript_state(segments, source=source, state=state)
        matched = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = as_float(segment.get("start"), 0.0)
            end = as_float(segment.get("end"), start)
            if max(start, window_start) <= min(end, window_end):
                matched.append(dict(segment))
        if not matched:
            if not self.load_frames() or state.get("audio_current_ready") or state.get("current_text_ready"):
                return self._write_audio_only_transcript_state(segments, source=source, state=state)
            return {"backfilled_current": False, "segment_count": 0, "mcur_version": int(state.get("mcur_version", 0) or 0)}

        existing = self.load_transcript_segments()
        merged = self._merge_transcript_segments(existing, matched)
        self._write_jsonl(self.transcript_path, merged)
        transcript = " ".join(str(seg.get("text") or "").strip() for seg in merged if seg.get("text")).strip()

        open_event = read_json(self.open_event_path, default={})
        if not isinstance(open_event, dict):
            open_event = {}
        self._backfill_open_event_transcript(open_event, matched, source=source)
        write_json_atomic(self.open_event_path, open_event)

        version = int(state.get("mcur_version", 0) or 0) + 1
        state["mcur_version"] = version
        state["transcript_segment_count"] = len(merged)
        state["transcript_source"] = source
        state["transcript_updated_at"] = utc_now_iso()
        state["current_text_ready"] = bool(merged)
        state["audio_current_ready"] = bool(merged) and str(source) in {"audio_asr_backfill", "audio_chunk_asr", "stream_asr", "transcript_backfill"}
        state["mcur_ready"] = bool(state.get("mcur_ready") or merged)
        state["last_update_at"] = utc_now_iso()
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.state_path, state)
        return {
            "backfilled_current": True,
            "segment_count": len(matched),
            "total_transcript_segment_count": len(merged),
            "mcur_version": version,
            "transcript": transcript,
        }

    def get_current_context(self) -> dict[str, Any]:
        state = self.get_state()
        frames = self.load_frames()
        transcript_segments = self.load_transcript_segments()
        open_event = read_json(self.open_event_path, default={})
        if not isinstance(open_event, dict):
            open_event = {}
        transcript = " ".join(str(seg.get("text") or "").strip() for seg in transcript_segments if seg.get("text")).strip()
        return {
            "session_id": self.session_id,
            "state": state,
            "frames": frames,
            "transcript_segments": transcript_segments,
            "transcript": transcript,
            "open_event": open_event,
            "mcur_ready": bool(state.get("mcur_ready") and (frames or transcript_segments or state.get("current_text_ready") or state.get("audio_current_ready"))),
            "current_text_ready": bool(state.get("current_text_ready") or transcript_segments),
            "audio_current_ready": bool(state.get("audio_current_ready")),
            "is_stale": self.is_stale(state),
            "window_start_time": state.get("window_start_time"),
            "window_end_time": state.get("window_end_time"),
            "current_time": state.get("current_time"),
        }

    def summary(self, limit: int = 5) -> dict[str, Any]:
        state = self.get_state()
        frames = self.load_frames()
        return {
            "session_id": self.session_id,
            "mcur_ready": bool(state.get("mcur_ready") and (frames or state.get("current_text_ready") or state.get("audio_current_ready") or state.get("transcript_segment_count", 0))),
            "current_text_ready": bool(state.get("current_text_ready") or state.get("transcript_segment_count", 0)),
            "audio_current_ready": bool(state.get("audio_current_ready")),
            "mcur_version": state.get("mcur_version", 0),
            "window_start_time": state.get("window_start_time"),
            "window_end_time": state.get("window_end_time"),
            "current_time": state.get("current_time"),
            "frame_count": len(frames),
            "transcript_segment_count": state.get("transcript_segment_count", 0),
            "is_stale": self.is_stale(state),
            "latest_frames": frames[-max(1, limit) :],
            "state": state,
        }

    def _empty_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mcur_ready": False,
            "mcur_version": 0,
            "window_seconds": self.window_seconds,
            "core_seconds": self.core_seconds,
            "window_start_time": None,
            "window_end_time": None,
            "current_time": None,
            "frame_count": 0,
            "transcript_segment_count": 0,
            "current_text_ready": False,
            "audio_current_ready": False,
            "open_event_id": None,
            "open_event_start": None,
            "open_event_end": None,
            "last_update_at": None,
            "is_stale": True,
            "stale_seconds": self.stale_seconds,
            "updated_at": utc_now_iso(),
        }

    def _write_audio_only_transcript_state(
        self,
        segments: list[dict[str, Any]],
        *,
        source: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        existing = self.load_transcript_segments()
        merged = self._merge_transcript_segments(existing, segments)
        self._write_jsonl(self.transcript_path, merged)
        starts = [as_float(seg.get("start"), 0.0) for seg in merged]
        ends = [as_float(seg.get("end"), as_float(seg.get("start"), 0.0)) for seg in merged]
        window_start = min(starts) if starts else 0.0
        window_end = max(ends) if ends else window_start
        transcript = " ".join(str(seg.get("text") or "").strip() for seg in merged if seg.get("text")).strip()
        now = utc_now_iso()
        version = int(state.get("mcur_version", 0) or 0) + 1
        open_event = {
            "open_event_id": f"mcur_audio_open_{int(round(window_start * 1000)):09d}_{int(round(window_end * 1000)):09d}",
            "session_id": self.session_id,
            "start_time": round(window_start, 3),
            "end_time": round(window_end, 3),
            "duration": round(max(0.0, window_end - window_start), 3),
            "boundary_status": "open",
            "input_source": "audio_chunk_asr",
            "source": source,
            "keyframes": [],
            "transcript": transcript,
            "transcript_segments": merged,
            "transcript_source": source,
            "transcript_updated_at": now,
            "status": "open",
            "updated_at": now,
        }
        write_json_atomic(self.open_event_path, open_event)
        new_state = {
            **state,
            "session_id": self.session_id,
            "mcur_ready": True,
            "current_text_ready": True,
            "audio_current_ready": True,
            "mcur_version": version,
            "window_seconds": state.get("window_seconds", self.window_seconds),
            "core_seconds": state.get("core_seconds", self.core_seconds),
            "window_start_time": round(window_start, 3),
            "window_end_time": round(window_end, 3),
            "current_time": round(window_end, 3),
            "frame_count": len(self.load_frames()),
            "transcript_segment_count": len(merged),
            "transcript_source": source,
            "transcript_updated_at": now,
            "open_event_id": open_event["open_event_id"],
            "open_event_start": open_event["start_time"],
            "open_event_end": open_event["end_time"],
            "last_update_at": now,
            "is_stale": False,
            "stale_seconds": self.stale_seconds,
            "updated_at": now,
        }
        write_json_atomic(self.state_path, new_state)
        return {
            "backfilled_current": True,
            "segment_count": len(segments),
            "total_transcript_segment_count": len(merged),
            "mcur_version": version,
            "transcript": transcript,
            "audio_only": True,
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _merge_transcript_segments(self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(item: dict[str, Any]) -> str:
            explicit = str(item.get("segment_id") or "").strip()
            if explicit:
                return explicit
            start = int(round(as_float(item.get("start"), 0.0) * 1000))
            end = int(round(as_float(item.get("end"), as_float(item.get("start"), 0.0)) * 1000))
            return f"{start:09d}_{end:09d}_{str(item.get('text') or '').strip()}"

        by_key = {key(item): dict(item) for item in existing if isinstance(item, dict)}
        for segment in incoming:
            by_key[key(segment)] = {**by_key.get(key(segment), {}), **dict(segment)}
        return sorted(by_key.values(), key=lambda item: (as_float(item.get("start")), as_float(item.get("end")), str(item.get("segment_id") or "")))

    def _backfill_open_event_transcript(self, open_event: dict[str, Any], segments: list[dict[str, Any]], source: str) -> None:
        def overlaps_event(event: dict[str, Any], segment: dict[str, Any]) -> bool:
            if not event:
                return False
            start = as_float(event.get("start_time"), event.get("open_event_start") or 0.0)
            end = as_float(event.get("end_time"), event.get("last_update_time") or event.get("open_event_end") or start)
            s = as_float(segment.get("start"), 0.0)
            e = as_float(segment.get("end"), s)
            return max(start, s) <= min(end, e)

        matched = [seg for seg in segments if overlaps_event(open_event, seg)]
        if matched:
            merged = self._merge_transcript_segments(list(open_event.get("transcript_segments", []) or []), matched)
            open_event["transcript_segments"] = merged
            open_event["transcript"] = " ".join(str(seg.get("text") or "").strip() for seg in merged if seg.get("text")).strip()
            open_event["transcript_source"] = source
            open_event["transcript_updated_at"] = utc_now_iso()
        stream_open = open_event.get("stream_open_event")
        if isinstance(stream_open, dict):
            matched_stream = [seg for seg in segments if overlaps_event(stream_open, seg)]
            if matched_stream:
                merged_stream = self._merge_transcript_segments(list(stream_open.get("transcript_segments", []) or []), matched_stream)
                stream_open["transcript_segments"] = merged_stream
                stream_open["transcript"] = " ".join(str(seg.get("text") or "").strip() for seg in merged_stream if seg.get("text")).strip()
                stream_open["transcript_source"] = source
                stream_open["transcript_updated_at"] = utc_now_iso()
                open_event["stream_open_event"] = stream_open

    def _cleanup_current_frames(self, kept_rel_paths: set[str]) -> None:
        if not self.frames_dir.exists():
            return
        for path in self.frames_dir.glob("cur_kf_*"):
            rel = rel_to_session(self.session_dir, path)
            if rel not in kept_rel_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
