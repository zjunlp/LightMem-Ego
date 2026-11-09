from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any

from online_time import event_time_seconds
from online_memory.content import effective_episodic_content, effective_episodic_content_field

from .schemas import (
    EPISODIC_DATA_FIELDS,
    FORBIDDEN_NESTED_KEYS,
    SEMANTIC_DATA_FIELDS,
    VISUAL_DATA_FIELDS,
)


class MemoryNormalizer:
    def __init__(self, session_id: str, max_array_items: int = 100, max_depth: int = 6) -> None:
        self.session_id = session_id
        self.max_array_items = max_array_items
        self.max_depth = max_depth
        self.truncated_field_count = 0

    def episodic(self, record: dict[str, Any], granularity: str) -> dict[str, Any]:
        source_id = _source_id(record, "doc_id", "evidence_doc_id", "segment_id")
        content = effective_episodic_content(record)
        content_field = episodic_content_field(record)
        return {
            "id": self._stable_id("ep", granularity, source_id, content),
            "source_id": source_id or None,
            "memory_source": "M_lt",
            "memory_type": "episodic",
            "granularity": granularity,
            "content": content,
            "editable": bool(granularity == "30sec" and source_id and content_field),
            "editDisabledReason": None if granularity == "30sec" and source_id and content_field else (
                "missing_stable_id" if granularity == "30sec" and not source_id else
                "missing_editable_text_field" if granularity == "30sec" else
                "unsupported_source_record"
            ),
            "confidence": _number_or_none(record.get("confidence")),
            "status": _text_or_none(record.get("status")),
            "event_time": {
                "date": _text_or_none(record.get("date")),
                "start_seconds": _time_number(record.get("start_time", record.get("start"))),
                "end_seconds": _time_number(record.get("end_time", record.get("end"))),
            },
            "data": self._data(record, EPISODIC_DATA_FIELDS),
        }

    def semantic(self, record: dict[str, Any]) -> dict[str, Any]:
        source_id = _source_id(record, "fact_id")
        content = _first_text(record, "semantic_summary") or _triple_text(record)
        return {
            "id": self._stable_id("sem", "", source_id, content),
            "source_id": source_id or None,
            "memory_source": "M_lt",
            "memory_type": "semantic",
            "granularity": None,
            "content": content,
            "confidence": _number_or_none(record.get("confidence")),
            "status": _text_or_none(record.get("status")),
            "event_time": {
                "first_seen": _scalar_or_none(record.get("first_seen")),
                "last_seen": _scalar_or_none(record.get("last_seen")),
            },
            "data": self._data(record, SEMANTIC_DATA_FIELDS),
        }

    def visual(self, record: dict[str, Any]) -> dict[str, Any]:
        source_id = _source_id(record, "visual_id", "evidence_doc_id", "segment_id")
        content = _first_text(record, "keyframe_caption", "segment_caption", "scene")
        return {
            "id": self._stable_id("vis", "", source_id, content),
            "source_id": source_id or None,
            "memory_source": "M_lt",
            "memory_type": "visual",
            "granularity": None,
            "content": content,
            "confidence": _number_or_none(record.get("confidence")),
            "status": _text_or_none(record.get("status")),
            "event_time": {
                "start_seconds": _time_number(record.get("start_time")),
                "end_seconds": _time_number(record.get("end_time")),
                "timestamp_seconds": _time_number(record.get("timestamp")),
            },
            "data": self._data(record, VISUAL_DATA_FIELDS),
        }

    def _data(self, record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
        return {
            key: self._safe_value(record.get(key), depth=0)
            for key in fields
            if key in record
        }

    def _safe_value(self, value: Any, depth: int) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if depth >= self.max_depth:
            self.truncated_field_count += 1
            return {"truncated": True, "reason": "max_depth"}
        if isinstance(value, list):
            original_count = len(value)
            selected = value[: self.max_array_items]
            items = [self._safe_value(item, depth + 1) for item in selected]
            if original_count > self.max_array_items:
                self.truncated_field_count += 1
                return {
                    "items": items,
                    "truncated": True,
                    "original_count": original_count,
                    "returned_count": len(items),
                }
            return items
        if isinstance(value, dict):
            return {
                str(key): self._safe_value(item, depth + 1)
                for key, item in value.items()
                if not _forbidden_key(str(key))
            }
        return str(value)

    def _stable_id(self, prefix: str, granularity: str, source_id: str, content: str) -> str:
        raw = "\x1f".join((self.session_id, prefix, granularity, source_id or content))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"mlt_{prefix}_{digest}"


def episodic_sort_key(record: dict[str, Any]) -> tuple[int, float, str]:
    event_time = record.get("event_time") if isinstance(record.get("event_time"), dict) else {}
    value = event_time.get("start_seconds")
    day_offset = _day_offset(event_time.get("date"))
    if value is not None and day_offset is not None:
        value = day_offset + float(value)
    return (1 if value is None else 0, float(value or 0), str(record.get("id") or ""))


def visual_sort_key(record: dict[str, Any]) -> tuple[int, float, str]:
    event_time = record.get("event_time") if isinstance(record.get("event_time"), dict) else {}
    value = event_time.get("timestamp_seconds")
    if value is None:
        value = event_time.get("start_seconds")
    return (1 if value is None else 0, float(value or 0), str(record.get("id") or ""))


def semantic_sort_key(record: dict[str, Any]) -> tuple[int, int, float | str, str]:
    event_time = record.get("event_time") if isinstance(record.get("event_time"), dict) else {}
    value = event_time.get("first_seen")
    parsed = _sortable_time(value)
    kind = 0 if isinstance(parsed, (int, float)) else 1
    return (
        1 if parsed is None else 0,
        kind,
        parsed if parsed is not None else "",
        str(record.get("id") or ""),
    )


def _source_id(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def episodic_content_field(record: dict[str, Any]) -> str | None:
    return effective_episodic_content_field(record)


def episodic_source_id(record: dict[str, Any]) -> str:
    return _source_id(record, "doc_id", "evidence_doc_id", "segment_id")


def stable_episodic_id(session_id: str, granularity: str, record: dict[str, Any]) -> str:
    source_id = episodic_source_id(record)
    content = effective_episodic_content(record)
    raw = "\x1f".join((session_id, "ep", granularity, source_id or content))
    return f"mlt_ep_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def source_record_fingerprint(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _triple_text(record: dict[str, Any]) -> str:
    triple = record.get("triple")
    if isinstance(triple, list):
        text = " ".join(str(item).strip() for item in triple if str(item).strip())
        if text:
            return text
    return " ".join(
        value
        for value in (
            _text_or_none(record.get("head")),
            _text_or_none(record.get("relation")),
            _text_or_none(record.get("tail")),
        )
        if value
    )


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _scalar_or_none(value: Any) -> str | int | float | bool | None:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    try:
        parsed = float(str(value))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def _time_number(value: Any) -> float | None:
    return event_time_seconds(value)


def _sortable_time(value: Any) -> float | str | None:
    numeric = _time_number(value)
    if numeric is not None:
        return numeric
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return text


def _day_offset(value: Any) -> float | None:
    if value is None:
        return 0.0
    text = str(value).strip().upper()
    if text.startswith("DAY"):
        try:
            return max(0, int(text[3:]) - 1) * 86400.0
        except ValueError:
            return None
    return 0.0


def _forbidden_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in FORBIDDEN_NESTED_KEYS:
        return True
    return normalized.endswith("_path") or normalized.endswith("_paths") or "embedding" in normalized
