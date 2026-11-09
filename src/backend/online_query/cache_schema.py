from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from online_time import seconds_value


def has_valid_binding(context: dict[str, Any] | None) -> bool:
    """Whether a cached *event time* may guide retrieval, not just a topic.

    Older callers have no binding_valid field. Accept their grounded ranges,
    but never turn a low-confidence/ambiguous reference or an entity alone
    into a temporal binding.
    """
    context = context or {}
    if (context.get("binding_valid") is False or context.get("is_followup") is False
            or context.get("reference_ambiguous")):
        return False
    if context.get("anchor_status") == "unresolved":
        return False
    try:
        if float(context.get("confidence", 1.0)) < 0.7:
            return False
    except (TypeError, ValueError):
        return False
    for window in context.get("referenced_time_ranges", []) or []:
        if not isinstance(window, dict):
            continue
        start, end = seconds_value(window.get("start")), seconds_value(window.get("end"))
        if start is not None and end is not None and 0 <= start <= end:
            return True
    return False


def has_exact_frame_binding(context: dict[str, Any] | None) -> bool:
    """Only an explicit reference to a displayed frame fixes a capture time.

    Answer citations locate observations, not the full extent of an event.
    Even a confident dialogue binding is otherwise a retrieval preference.
    """
    context = context or {}
    ranges = context.get("referenced_time_ranges") or []
    return bool(has_valid_binding(context) and ranges and all(
        isinstance(window, dict) and window.get("source") == "explicit_previous_image"
        for window in ranges))


def has_entity_binding(context: dict[str, Any] | None) -> bool:
    """An identity hint may be valid without referring to an old event time."""
    context = context or {}
    if not context.get("entity_binding_valid") or not context.get("is_followup") or context.get("reference_ambiguous"):
        return False
    try:
        return float(context.get("confidence", 0)) >= 0.7 and bool(context.get("referenced_entities"))
    except (TypeError, ValueError):
        return False


def resolved_search_question(question: str, context: dict[str, Any] | None) -> str:
    context = context or {}
    if has_valid_binding(context) or has_entity_binding(context):
        return str(context.get("resolved_question") or question)
    return question


@dataclass
class CacheContext:
    cache_hit: bool = False
    is_followup: bool = False
    binding_valid: bool = False
    entity_binding_valid: bool = False
    resolved_question: str | None = None
    referenced_entities: list[dict[str, Any]] = field(default_factory=list)
    referenced_time_ranges: list[dict[str, Any]] = field(default_factory=list)
    referenced_memory_ids: list[str] = field(default_factory=list)
    referenced_segment_ids: list[str] = field(default_factory=list)
    referenced_visual_ids: list[str] = field(default_factory=list)
    referenced_evidence_frames: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hit": self.cache_hit,
            "is_followup": self.is_followup,
            "binding_valid": self.binding_valid,
            "entity_binding_valid": self.entity_binding_valid,
            "resolved_question": self.resolved_question,
            "referenced_entities": self.referenced_entities,
            "referenced_time_ranges": self.referenced_time_ranges,
            "referenced_memory_ids": self.referenced_memory_ids,
            "referenced_segment_ids": self.referenced_segment_ids,
            "referenced_visual_ids": self.referenced_visual_ids,
            "referenced_evidence_frames": self.referenced_evidence_frames,
            "confidence": self.confidence,
            "reason": self.reason,
            "warnings": self.warnings,
        }
