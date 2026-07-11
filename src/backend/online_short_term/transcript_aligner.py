from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json


class TranscriptAligner:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.transcript_path = session_dir / "preprocess" / "transcript.json"
        self.stream_transcript_path = session_dir / "stream" / "transcript" / "partial_transcript.jsonl"
        self.segments = self._load_segments()

    def align(self, start_time: float, end_time: float) -> dict[str, Any]:
        self.reload()
        matched = []
        for segment in self.segments:
            s = float(segment.get("start", 0.0))
            e = float(segment.get("end", s))
            if max(s, start_time) <= min(e, end_time):
                matched.append(dict(segment))
        text = " ".join(str(item.get("text", "")).strip() for item in matched if item.get("text")).strip()
        return {"transcript": text, "transcript_segments": matched}

    def reload(self) -> list[dict[str, Any]]:
        self.segments = self._load_segments()
        return self.segments

    def align_from_stream(self, session_dir: Path, start_time: float, end_time: float) -> list[dict[str, Any]]:
        previous = self.session_dir
        self.session_dir = Path(session_dir)
        self.transcript_path = self.session_dir / "preprocess" / "transcript.json"
        self.stream_transcript_path = self.session_dir / "stream" / "transcript" / "partial_transcript.jsonl"
        try:
            return self.align(start_time, end_time)["transcript_segments"]
        finally:
            self.session_dir = previous
            self.transcript_path = previous / "preprocess" / "transcript.json"
            self.stream_transcript_path = previous / "stream" / "transcript" / "partial_transcript.jsonl"
            self.reload()

    def _load_segments(self) -> list[dict[str, Any]]:
        segments = self._load_preprocess_segments()
        segments.extend(self._load_stream_segments())
        dedup: dict[str, dict[str, Any]] = {}
        for item in segments:
            key = str(item.get("segment_id") or f"{item.get('start')}:{item.get('end')}:{item.get('text')}")
            dedup[key] = item
        return sorted(dedup.values(), key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)), str(item.get("segment_id") or "")))

    def _load_preprocess_segments(self) -> list[dict[str, Any]]:
        if not self.transcript_path.exists():
            return []
        payload = read_json(self.transcript_path, default={})
        raw_segments = []
        if isinstance(payload, dict):
            if isinstance(payload.get("segments"), list):
                raw_segments = payload["segments"]
            elif isinstance(payload.get("transcript_segments"), list):
                raw_segments = payload["transcript_segments"]
            elif isinstance(payload.get("result"), dict) and isinstance(payload["result"].get("segments"), list):
                raw_segments = payload["result"]["segments"]
        elif isinstance(payload, list):
            raw_segments = payload
        segments = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start", item.get("start_time", 0.0)))
                end = float(item.get("end", item.get("end_time", start)))
            except Exception:
                continue
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": str(item.get("text", "")).strip(),
                    "speaker": item.get("speaker"),
                    "source": item.get("source") or "preprocess",
                    "segment_id": item.get("segment_id"),
                }
            )
        return segments

    def _load_stream_segments(self) -> list[dict[str, Any]]:
        if not self.stream_transcript_path.exists():
            return []
        segments: list[dict[str, Any]] = []
        with self.stream_transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                try:
                    start = float(item.get("start", item.get("start_time", 0.0)))
                    end = float(item.get("end", item.get("end_time", start)))
                except Exception:
                    continue
                segments.append(
                    {
                        **item,
                        "start": start,
                        "end": end,
                        "text": str(item.get("text", "")).strip(),
                        "speaker": item.get("speaker"),
                        "source": item.get("source") or "stream_asr",
                    }
                )
        return segments
