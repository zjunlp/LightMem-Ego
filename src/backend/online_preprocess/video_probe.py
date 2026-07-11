from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from .io_utils import OnlinePreprocessError, ffprobe_bin, read_json, relative_to_session, run_command, write_json


def _parse_fps(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def probe_video(
    input_video: Path,
    output_json: Path,
    session_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    if output_json.exists() and not force:
        cached = read_json(output_json, default={})
        if cached:
            return cached

    if not input_video.exists():
        raise OnlinePreprocessError(f"Input video does not exist: {input_video}")

    command = [
        ffprobe_bin(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_video),
    ]
    result = run_command(command, description="ffprobe video")
    payload = json.loads(result.stdout)

    streams = payload.get("streams", [])
    format_info = payload.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise OnlinePreprocessError(f"No readable video stream found in {input_video}")

    duration = float(format_info.get("duration") or video_stream.get("duration") or 0.0)
    if duration <= 0:
        raise OnlinePreprocessError(f"Video duration is invalid: {duration}")

    meta = {
        "video_path": relative_to_session(input_video, session_dir),
        "duration": duration,
        "fps": _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "has_audio": audio_stream is not None,
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "probe_ok": True,
    }
    write_json(output_json, meta)
    return meta
