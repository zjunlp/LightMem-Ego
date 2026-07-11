from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io_utils import OnlinePreprocessError, ensure_dir, ffmpeg_bin, relative_to_session, run_command


SEGMENT_SECONDS = 30.0


def _seconds_tag(value: float) -> str:
    return f"{int(round(value)):06d}"


def _build_segment_record(
    session_dir: Path,
    clip_path: Path,
    start: float,
    end: float,
) -> dict[str, Any]:
    start_tag = _seconds_tag(start)
    end_tag = _seconds_tag(end)
    return {
        "segment_id": f"seg_{start_tag}_{end_tag}",
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "clip_path": relative_to_session(clip_path, session_dir),
        "source_video_path": "input.mp4",
        "transcript": "",
        "transcript_segments": [],
        "keyframes": [],
    }


def _render_clip(
    input_video: Path,
    output_clip: Path,
    start: float,
    duration: float,
) -> None:
    ensure_dir(output_clip.parent)
    command = [
        ffmpeg_bin(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_clip),
    ]
    try:
        run_command(command, description=f"segment video clip {output_clip.name}")
    except OnlinePreprocessError as exc:
        if "Unknown encoder 'libx264'" not in str(exc):
            raise
        fallback = list(command)
        fallback[fallback.index("libx264")] = "libopenh264"
        run_command(fallback, description=f"segment video clip {output_clip.name} with libopenh264")


def segment_video_into_clips(
    input_video: Path,
    clips_dir: Path,
    session_dir: Path,
    duration: float,
    force: bool = False,
    segment_seconds: float = SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    ensure_dir(clips_dir)

    total_segments = max(1, int(math.ceil(duration / segment_seconds)))
    segments: list[dict[str, Any]] = []
    for idx in range(total_segments):
        start = idx * segment_seconds
        end = min((idx + 1) * segment_seconds, duration)
        start_tag = _seconds_tag(start)
        end_tag = _seconds_tag(end)
        clip_path = clips_dir / f"clip_{start_tag}_{end_tag}.mp4"
        if force or not clip_path.exists():
            _render_clip(input_video, clip_path, start=start, duration=end - start)
        segments.append(_build_segment_record(session_dir=session_dir, clip_path=clip_path, start=start, end=end))
    return segments


def align_transcript_to_segments(
    segments: list[dict[str, Any]],
    transcript_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for segment in segments:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])

        matched = [
            {
                "start": float(entry.get("start") or 0.0),
                "end": float(entry.get("end") or 0.0),
                "text": str(entry.get("text", "")).strip(),
                "speaker": entry.get("speaker"),
            }
            for entry in transcript_entries
            if float(entry.get("end") or 0.0) > segment_start and float(entry.get("start") or 0.0) < segment_end
        ]
        matched = [entry for entry in matched if entry["text"]]

        segment["transcript_segments"] = matched
        segment["transcript"] = " ".join(entry["text"] for entry in matched).strip()

    return segments
