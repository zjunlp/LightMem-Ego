from __future__ import annotations

import math
from typing import Any


def event_windows(event: dict[str, Any], window_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Return windows intersecting the event's half-open time range."""
    start = float(event.get("start_time", 0.0) or 0.0)
    end = max(start, float(event.get("end_time", start) or start))
    first = math.floor(start / window_seconds)
    stop = max(first + 1, math.ceil(end / window_seconds))
    return [
        {
            "window_id": f"win_{int(index * window_seconds):06d}_{int((index + 1) * window_seconds):06d}",
            "start_time": round(index * window_seconds, 3),
            "end_time": round((index + 1) * window_seconds, 3),
        }
        for index in range(first, stop)
    ]
