from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


def _segment_key(segment: dict[str, Any]) -> str:
    explicit = str(segment.get("segment_id") or "").strip()
    if explicit:
        return explicit
    start = int(round(float(segment.get("start", 0.0) or 0.0) * 1000))
    end = int(round(float(segment.get("end", segment.get("start", 0.0)) or 0.0) * 1000))
    text = str(segment.get("text") or "").strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{start:09d}_{end:09d}_{digest}"


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _time_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    try:
        ls = float(left.get("start", 0.0) or 0.0)
        le = float(left.get("end", ls) or ls)
        rs = float(right.get("start", 0.0) or 0.0)
        re_ = float(right.get("end", rs) or rs)
    except Exception:
        return 0.0
    inter = max(0.0, min(le, re_) - max(ls, rs))
    union = max(le, re_) - min(ls, rs)
    return inter / union if union > 0 else 0.0


class PartialTranscriptStore:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.transcript_dir = self.session_dir / "stream" / "transcript"
        self.path = self.transcript_dir / "partial_transcript.jsonl"
        self.state_path = self.transcript_dir / "partial_transcript_state.json"
        self.errors_path = self.transcript_dir / "asr_errors.jsonl"

    def load_segments(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        segments: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
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
        segments.sort(key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0), str(item.get("segment_id") or "")))
        return segments

    def append_segments(
        self,
        segments: list[dict[str, Any]],
        *,
        upload_chunk_index: int | None = None,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        existing = self.load_segments()
        by_key = {_segment_key(item): dict(item) for item in existing}
        now = utc_now_iso()
        added = 0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            item = dict(segment)
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            item.setdefault("session_id", self.session_id)
            item.setdefault("stream_id", stream_id)
            item.setdefault("version", 1)
            item.setdefault("created_at", now)
            item["updated_at"] = now
            key = _segment_key(item)
            if key not in by_key:
                duplicate_key = self._find_overlap_duplicate_key(by_key, item)
                if duplicate_key is not None:
                    key = duplicate_key
            if key not in by_key:
                added += 1
            by_key[key] = {**by_key.get(key, {}), **item}
        merged = sorted(by_key.values(), key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0), str(item.get("segment_id") or "")))
        with self.path.open("w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        state = self._write_state(merged, stream_id=stream_id, upload_chunk_index=upload_chunk_index)
        return {
            "segment_count": len(merged),
            "added_segment_count": added,
            "partial_transcript_path": str(self.path.relative_to(self.session_dir)),
            "state": state,
        }

    def mark_chunk_processed(self, upload_chunk_index: int, *, stream_id: str | None, segment_count: int, no_audio: bool = False) -> dict[str, Any]:
        state = self._state()
        processed = list(state.get("processed_asr_chunks", []) or [])
        if int(upload_chunk_index) not in processed:
            processed.append(int(upload_chunk_index))
        state["processed_asr_chunks"] = sorted(processed)
        state["last_asr_chunk_index"] = int(upload_chunk_index)
        state["last_asr_segment_count"] = int(segment_count)
        state["last_asr_no_audio"] = bool(no_audio)
        state["stream_id"] = stream_id or state.get("stream_id")
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.state_path, state)
        return state

    def mark_audio_window_processed(self, window_id: str, *, stream_id: str | None, segment_count: int) -> dict[str, Any]:
        state = self._state()
        processed = list(state.get("processed_asr_windows", []) or [])
        if str(window_id) not in processed:
            processed.append(str(window_id))
        state["processed_asr_windows"] = processed
        failed = [str(item) for item in state.get("failed_asr_windows", []) or [] if str(item) != str(window_id)]
        state["failed_asr_windows"] = failed
        if not failed:
            state.pop("last_error", None)
        state["last_asr_window_id"] = str(window_id)
        state["last_asr_segment_count"] = int(segment_count)
        state["stream_id"] = stream_id or state.get("stream_id")
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.state_path, state)
        return state

    def mark_chunk_failed(self, upload_chunk_index: int, error: str, *, stream_id: str | None = None) -> dict[str, Any]:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        state = self._state()
        failed = list(state.get("failed_asr_chunks", []) or [])
        if int(upload_chunk_index) not in failed:
            failed.append(int(upload_chunk_index))
        state["failed_asr_chunks"] = sorted(failed)
        state["last_asr_chunk_index"] = int(upload_chunk_index)
        state["last_error"] = error
        state["stream_id"] = stream_id or state.get("stream_id")
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.state_path, state)
        with self.errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"upload_chunk_index": int(upload_chunk_index), "error": error, "created_at": utc_now_iso()}, ensure_ascii=False) + "\n")
        return state

    def mark_audio_window_failed(self, window_id: str, error: str, *, stream_id: str | None = None) -> dict[str, Any]:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        state = self._state()
        failed = list(state.get("failed_asr_windows", []) or [])
        if str(window_id) not in failed:
            failed.append(str(window_id))
        state["failed_asr_windows"] = failed
        state["last_asr_window_id"] = str(window_id)
        state["last_error"] = error
        state["stream_id"] = stream_id or state.get("stream_id")
        state["updated_at"] = utc_now_iso()
        write_json_atomic(self.state_path, state)
        with self.errors_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"window_id": str(window_id), "error": error, "created_at": utc_now_iso()}, ensure_ascii=False) + "\n")
        return state

    def _state(self) -> dict[str, Any]:
        state = read_json(self.state_path, default={})
        if not isinstance(state, dict):
            state = {}
        state.setdefault("session_id", self.session_id)
        state.setdefault("partial_transcript_version", 0)
        state.setdefault("segment_count", len(self.load_segments()))
        state.setdefault("processed_asr_chunks", [])
        state.setdefault("failed_asr_chunks", [])
        state.setdefault("processed_asr_windows", [])
        state.setdefault("failed_asr_windows", [])
        return state

    def _write_state(
        self,
        segments: list[dict[str, Any]],
        *,
        stream_id: str | None,
        upload_chunk_index: int | None,
    ) -> dict[str, Any]:
        old = self._state()
        starts = [float(item.get("start", 0.0) or 0.0) for item in segments]
        ends = [float(item.get("end", item.get("start", 0.0)) or 0.0) for item in segments]
        state = {
            **old,
            "session_id": self.session_id,
            "stream_id": stream_id or old.get("stream_id"),
            "partial_transcript_version": int(old.get("partial_transcript_version", 0) or 0) + 1,
            "segment_count": len(segments),
            "time_span": [round(min(starts), 3), round(max(ends), 3)] if segments else [0.0, 0.0],
            "last_asr_chunk_index": upload_chunk_index if upload_chunk_index is not None else old.get("last_asr_chunk_index"),
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.state_path, state)
        return state

    def _find_overlap_duplicate_key(self, by_key: dict[str, dict[str, Any]], item: dict[str, Any]) -> str | None:
        text = _normalize_text(item.get("text"))
        if not text:
            return None
        for key, existing in by_key.items():
            if _normalize_text(existing.get("text")) != text:
                continue
            if _time_iou(existing, item) > 0.5:
                return key
        return None
