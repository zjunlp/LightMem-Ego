"""Persist capture-grounded image descriptions independently of event prose."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from online_time import event_time_seconds, keyframe_path_seconds, seconds_value


def merge_frame_captions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep distinct observations and their provenance; replace empty labels."""
    described = {(row.get("path"), seconds_value(row.get("timestamp"))) for row in rows
                 if isinstance(row, dict) and (row.get("caption") or row.get("description"))}
    output, seen = [], set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = deepcopy(raw)
        text = row.get("caption") or row.get("description") or ""
        text = " ".join(text.split()) if isinstance(text, str) else ""
        capture = seconds_value(row.get("timestamp"))
        if not text and (row.get("path"), capture) in described:
            continue
        row["caption"] = text
        identity = (row.get("path"), capture, text, row.get("source_micro_event_id"))
        if identity not in seen:
            seen.add(identity)
            output.append(row)
    return output


def short_term_frame_captions(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Reuse observed image labels, never assign an event synopsis to a frame."""
    rows, captures = [], {}
    for frame in event.get("keyframes", []) or []:
        if not isinstance(frame, dict):
            continue
        path = frame.get("path")
        capture = seconds_value(frame.get("timestamp"))
        if path and capture is not None:
            captures[path] = capture
        caption = frame.get("caption") or frame.get("keyframe_caption") or ""
        if path or caption:
            rows.append({"timestamp": frame.get("timestamp"), "path": path, "caption": caption})
    for frame in event.get("keyframe_captions", []) or []:
        if isinstance(frame, dict):
            rows.append(frame)
        elif isinstance(frame, str) and frame.strip():
            rows.append({"caption": frame.strip()})
    paths = set(event.get("keyframe_paths", []) or []) | set(captures)
    paths.update(row["path"] for row in rows if row.get("path"))
    for observation in event.get("visual_observations", []) or []:
        if not isinstance(observation, dict):
            continue
        path, capture = observation.get("path"), seconds_value(observation.get("timestamp"))
        description = observation.get("description")
        if path not in paths or capture is None or not isinstance(description, str) or not description.strip():
            continue
        known_capture = captures.get(path, keyframe_path_seconds(path))
        if known_capture is not None and abs(known_capture - capture) > 0.01:
            continue
        row = {**deepcopy(observation), "timestamp": capture, "caption": description.strip(),
               "description_source": "short_term_visual_observation"}
        if event.get("event_id"):
            row["source_micro_event_id"] = event["event_id"]
        if event.get("version") is not None:
            row["source_micro_event_version"] = event["version"]
        rows.append(row)
    return merge_frame_captions(rows)


def inherit_frame_captions(record: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Backfill an existing event from exact linked captures, without model calls.

    A publication repair invokes this before publishing a new snapshot. Query
    readers only use the resulting LT metadata and do not fetch ST records.
    """
    paths = set(record.get("keyframe_paths", []) or [])
    previous = record.get("keyframe_captions", []) or []
    paths.update(row["path"] for row in previous if isinstance(row, dict) and row.get("path"))
    start = event_time_seconds(record.get("start_time", record.get("start")))
    end = event_time_seconds(record.get("end_time", record.get("end")))
    source_ids = set(record.get("source_micro_event_ids", []) or [])
    rows = list(previous)
    for source in sources:
        if source_ids and source.get("event_id") not in source_ids:
            continue
        for row in short_term_frame_captions(source):
            capture = seconds_value(row.get("timestamp"))
            if not row.get("caption") or row.get("path") not in paths or capture is None:
                continue
            if start is not None and capture < start or end is not None and capture > end:
                continue
            rows.append(row)
    return {**record, "keyframe_captions": merge_frame_captions(rows)}
