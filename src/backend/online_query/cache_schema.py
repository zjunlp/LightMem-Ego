from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheContext:
    cache_hit: bool = False
    is_followup: bool = False
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
