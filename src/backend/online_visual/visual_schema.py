from __future__ import annotations

from typing import Any


VisualItem = dict[str, Any]
VisualResult = dict[str, Any]


VALID_RETRIEVAL_MODES = {"text_only", "visual_only", "hybrid", "current"}


def normalize_retrieval_mode(value: str | None) -> str:
    mode = (value or "hybrid").strip().lower()
    if mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode: {value}. Expected one of {sorted(VALID_RETRIEVAL_MODES)}")
    return mode
