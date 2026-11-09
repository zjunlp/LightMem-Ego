"""Read selected evidence through one bounded, lossless schema boundary.

This adapter understands the existing stores; it does not rewrite their native
metadata. Event ranges and frame capture timestamps deliberately use different
time parsers. An event midpoint never supplies a missing frame timestamp.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from online_time import event_time_seconds, keyframe_path_seconds, seconds_value

_FRAME_LISTS = ("evidence_frames", "selected_frames", "keyframes", "keyframe_captions", "visual_items", "visual_observations")
_CONTAINERS = ("metadata", "text", "current_selection", "current_context", "open_event")
_IDENTITIES = ("evidence_id", "event_id", "memory_id", "evidence_doc_id", "doc_id", "open_event_id", "visual_id")


def _present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((row[key] for key in keys if row.get(key) is not None and row[key] != ""), None)


def _event_rows(event: dict[str, Any]):
    """Only traverse known event containers, with a depth/cycle bound."""
    def walk(row, depth, ancestors):
        if not isinstance(row, dict) or depth > 6 or id(row) in ancestors:
            return
        yield row
        for key in (*_CONTAINERS, "state"):
            yield from walk(row.get(key), depth + 1, ancestors | {id(row)})
    yield from walk(event, 0, set())


def _identity(event: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in _event_rows(event):
        for key in ("evidence_id", "segment_id", "canonical_segment_id"):
            value = _present(row, _IDENTITIES) if key == "evidence_id" else row.get(key)
            if key not in result and value is not None and value != "":
                result[key] = value
    if "canonical_segment_id" not in result and result.get("segment_id"):
        result["canonical_segment_id"] = result["segment_id"]
    if "evidence_id" not in result and result.get("canonical_segment_id"):
        result["evidence_id"] = result["canonical_segment_id"]
    return result


def _source(event: dict[str, Any], source: str | None) -> str | None:
    if source is not None:
        return source
    explicit_source = event.get("source_memory") or event.get("source")
    if explicit_source:
        return explicit_source
    if event.get("current_context") or event.get("current_selection"):
        return "M_cur"
    # Native identities are stronger than frame-container names shared by
    # several stores (long-term events may also carry ``keyframes``).
    if any(event.get(key) for key in ("memory_id", "evidence_doc_id", "visual_id")):
        return "M_lt"
    if event.get("event_id"):
        return "M_st"
    for row in _event_rows(event):
        value = row.get("source_memory") or row.get("source")
        if value:
            return value
    if any(key in event for key in ("text", "visual_items")):
        return "M_lt"
    if event.get("keyframes"):
        return "M_st"
    return None


def evidence_frames(event: Any, *, source: str | None = None) -> list[dict[str, Any]]:
    """Return copied, path-deduplicated frame records from native evidence.

    Native fields and scores stay on each frame. ``frame_origins`` retains every
    native record (including conflicting scores) and its schema location. The
    normalized ``timestamp``/``capture_time_seconds`` is seconds or None; only a
    known keyframe filename token may fill missing capture metadata.
    """
    if not isinstance(event, dict):
        return []
    owner = _identity(event)
    owner_source = _source(event, source)
    frames: dict[str, dict[str, Any]] = {}

    def add(raw, location, inherited):
        if not isinstance(raw, dict):
            return
        path = raw.get("path") or raw.get("image_path")
        if not isinstance(path, str) or not path:
            return
        row = deepcopy(raw)
        capture = seconds_value(raw.get("timestamp"))
        if capture is None:
            capture = seconds_value(raw.get("capture_time_seconds"))
        capture_origin = "metadata" if capture is not None else "path"
        if capture is None:
            capture = keyframe_path_seconds(path)
        if capture is None:
            capture_origin = "unknown"
        frame_source = owner_source or raw.get("source_memory") or raw.get("source")
        original_source = raw.get("original_source") or raw.get("source") or raw.get("source_memory") or frame_source
        row.update(path=path, timestamp=capture, capture_time_seconds=capture,
                   capture_time_origin=raw.get("capture_time_origin") or capture_origin,
                   source=frame_source, original_source=original_source)
        for key, value in {**owner, **inherited}.items():
            if row.get(key) is None:
                row[key] = value
        if row.get("canonical_segment_id") is None and row.get("segment_id"):
            row["canonical_segment_id"] = row["segment_id"]
        origins = deepcopy(raw.get("frame_origins")) if isinstance(raw.get("frame_origins"), list) else []
        if not origins:
            native = {key: deepcopy(value) for key, value in raw.items() if key != "frame_origins"}
            origins = [{"schema_path": location, "source": frame_source,
                        "original_source": original_source, **{key: row[key] for key in owner if key in row},
                        "metadata": native}]
        elif frame_source and not any(isinstance(origin, dict) and origin.get("source") == frame_source for origin in origins):
            # Reuse changes the selected source while retaining the original
            # capture's source. Record the hop once, stable on normalization.
            origins.append({"schema_path": location, "source": frame_source,
                            "original_source": original_source, **owner,
                            "metadata": {"path": path, "timestamp": capture}})
        row["frame_origins"] = origins
        existing = frames.get(path)
        if existing is None:
            frames[path] = row
            return
        # Prefer exact metadata over a filename-derived time, retaining all
        # native records so disagreements remain inspectable.
        if (existing.get("timestamp") is None or existing.get("capture_time_origin") == "path") and capture_origin == "metadata":
            for key in ("timestamp", "capture_time_seconds", "capture_time_origin"):
                existing[key] = row[key]
        for key, value in row.items():
            if existing.get(key) is None or (key in {"caption", "keyframe_caption", "description"}
                                              and not existing.get(key) and value):
                existing[key] = value
        for origin in origins:
            if origin not in existing["frame_origins"]:
                existing["frame_origins"].append(origin)

    def walk(row, location, depth, ancestors, inherited):
        if not isinstance(row, dict) or depth > 6 or id(row) in ancestors:
            return
        identity = {**inherited, **{key: row[key] for key in ("evidence_id", "segment_id", "canonical_segment_id") if row.get(key) is not None}}
        add(row, location, identity)
        for key in _FRAME_LISTS:
            values = row.get(key)
            if isinstance(values, (list, tuple)):
                for index, frame in enumerate(values):
                    add(frame, f"{location}.{key}[{index}]", identity)
        for key in _CONTAINERS:
            walk(row.get(key), f"{location}.{key}", depth + 1, ancestors | {id(row)}, identity)
        # Path-only fallback comes last, so richer native records win.
        paths = row.get("keyframe_paths")
        if isinstance(paths, (list, tuple)):
            for index, path in enumerate(paths):
                if isinstance(path, str) and path not in frames:
                    add({"path": path}, f"{location}.keyframe_paths[{index}]", identity)

    walk(event, "$", 0, set(), owner)
    return list(frames.values())


def normalize_evidence(event: Any, *, source: str | None = None) -> dict[str, Any]:
    """Copy an existing event and expose shared identities, ranges and frames."""
    if not isinstance(event, dict):
        return {}
    result = deepcopy(event)
    for key, value in _identity(event).items():
        if result.get(key) is None:
            result[key] = value
    source_memory = _source(event, source)
    if source_memory is not None:
        result["source_memory"] = source_memory
    rows = list(_event_rows(event))
    for key in ("start_time", "end_time"):
        aliases = (key, "window_start_time" if key == "start_time" else "window_end_time")
        value = next((_present(row, aliases) for row in rows if _present(row, aliases) is not None), None)
        if value is not None or key in result:
            result[key] = event_time_seconds(value)
    frames = evidence_frames(event, source=source_memory)
    result["evidence_frames"] = frames
    paths = list(result.get("keyframe_paths") or [])
    result["keyframe_paths"] = list(dict.fromkeys(paths + [frame["path"] for frame in frames]))
    return result
