from __future__ import annotations

import json
import re
from typing import Any


MEMORY_KEYS = ("M_cur", "M_st", "M_lt", "M_cache")


def parse_auto_bool(value: Any, default: str = "auto") -> bool | str:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return "auto"
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    parsed = parse_auto_bool(value)
    return parsed if isinstance(parsed, bool) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp01(value: Any) -> float:
    return max(0.0, min(1.0, safe_float(value)))


def contains_any(text: str, keywords: list[str]) -> bool:
    lower = (text or "").lower()
    for keyword in keywords:
        needle = str(keyword or "").strip().lower()
        if not needle:
            continue
        if needle.isascii():
            prefix = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
            suffix = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
            if re.search(prefix + re.escape(needle) + suffix, lower):
                return True
        elif needle in lower:
            return True
    return False


def flatten_text(value: Any) -> list[str]:
    """Convert nested evidence fields to prompt-safe strings."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            child = " ".join(flatten_text(item)).strip()
            if child:
                parts.append(f"{key}: {child}")
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(flatten_text(item))
        return parts
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    text = text.strip()
    return [text] if text else []


def compact_text(value: Any, max_chars: int = 500) -> str:
    text = "; ".join(flatten_text(value)).strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def time_overlap(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float:
    a0 = safe_float(a_start)
    a1 = safe_float(a_end, a0)
    b0 = safe_float(b_start)
    b1 = safe_float(b_end, b0)
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(max(a1 - a0, 0.0), max(b1 - b0, 0.0)))
    return max(0.0, min(1.0, inter / denom))


def window_contains(window: list[Any] | tuple[Any, ...] | None, start: Any, end: Any) -> bool:
    if not window or len(window) < 2:
        return False
    return time_overlap(window[0], window[1], start, end) > 0.0
