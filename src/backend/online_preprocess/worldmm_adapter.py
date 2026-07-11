from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import write_json


def seconds_to_hhmmssff(seconds: float, fps_for_code: int = 100) -> str:
    total_frames = max(0, int(round(seconds * fps_for_code)))
    total_seconds, frames = divmod(total_frames, fps_for_code)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}{frames:02d}"


def _transcript_entries_for_sync(segment: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for item in segment.get("transcript_segments", []) or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        entries.append(
            {
                "start": round(float(item.get("start") or 0.0), 3),
                "end": round(float(item.get("end") or 0.0), 3),
                "text": text,
                "type": "transcript",
                "speaker": item.get("speaker"),
            }
        )
    return entries


def build_session_sync(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sync_items = []
    for segment in segments:
        sync_items.append(
            {
                "video_file": segment["clip_path"],
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "data": _transcript_entries_for_sync(segment),
            }
        )
    return sync_items


def build_session_30sec(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for segment in segments:
        docs.append(
            {
                "doc_id": segment["segment_id"],
                "start_time": seconds_to_hhmmssff(float(segment["start"])),
                "end_time": seconds_to_hhmmssff(float(segment["end"])),
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "date": "DAY1",
                "text": segment.get("transcript", ""),
                "transcript_text": segment.get("transcript", ""),
                "transcript_segments": segment.get("transcript_segments", []),
                "video_path": segment.get("source_video_path", "input.mp4"),
                "clip_path": segment.get("clip_path", ""),
                "keyframe_paths": [item["path"] for item in segment.get("keyframes", [])],
                "keyframes": segment.get("keyframes", []),
            }
        )
    return docs


def write_worldmm_session_files(
    segments: list[dict[str, Any]],
    preprocess_dir: Path,
) -> tuple[Path, Path]:
    session_sync_path = preprocess_dir / "session_sync.json"
    session_30sec_path = preprocess_dir / "session_30sec.json"
    write_json(session_sync_path, build_session_sync(segments))
    write_json(session_30sec_path, build_session_30sec(segments))
    return session_sync_path, session_30sec_path
