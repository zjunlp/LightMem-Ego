"""Time conversions at memory/evidence boundaries, without model imports."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any


def seconds_value(value: Any) -> float | None:
    """Explicit numeric timestamps in metadata are already seconds."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def event_time_seconds(value: Any) -> float | None:
    """Decode HHMMSSff strings; numeric values and other numeric strings are seconds."""
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 8 and text.isdigit():
            hours, minutes, seconds, centiseconds = (int(text[i:i + 2]) for i in range(0, 8, 2))
            if minutes >= 60 or seconds >= 60:
                return None
            return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0
        parts = text.split(":")
        if len(parts) == 3:
            try:
                return seconds_value(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            except ValueError:
                return None
    return seconds_value(value)


def keyframe_path_seconds(path: str) -> float | None:
    """Read only known kf time tokens, never a processing ID or a frame index.

    Legacy preprocess uses six-digit seconds; streaming/current use nine-digit
    milliseconds. Suffixes identify chunks or filename collisions, not time.
    """
    match = re.search(r"(?:^|_)kf_(\d+)(?:_|$)", Path(path).stem)
    if not match:
        return None
    token = match.group(1)
    return round(int(token) / (1000.0 if len(token) >= 7 else 1.0), 3)


def frame_capture_seconds(path: str, metadata: dict[str, Any]) -> float | None:
    timestamp = seconds_value(metadata.get("timestamp"))
    return timestamp if timestamp is not None else keyframe_path_seconds(path)
