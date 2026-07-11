from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryLatency:
    total_ms: int = 0
    cache_lookup_ms: int = 0
    cache_hit: bool = False
    engine_load_ms: int = 0
    router_ms: int | None = None
    memory_router_ms: int | None = None
    retrieval_planner_ms: int | None = None
    retrieval_ms: int | None = None
    text_retrieval_ms: int | None = None
    visual_retrieval_ms: int | None = None
    short_term_retrieval_ms: int | None = None
    fusion_ms: int | None = None
    memory_fusion_ms: int | None = None
    evidence_pack_ms: int | None = None
    generation_ms: int | None = None
    query_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "cache_lookup_ms": self.cache_lookup_ms,
            "cache_hit": self.cache_hit,
            "engine_load_ms": self.engine_load_ms,
            "router_ms": self.router_ms,
            "memory_router_ms": self.memory_router_ms,
            "retrieval_planner_ms": self.retrieval_planner_ms,
            "retrieval_ms": self.retrieval_ms,
            "text_retrieval_ms": self.text_retrieval_ms,
            "visual_retrieval_ms": self.visual_retrieval_ms,
            "short_term_retrieval_ms": self.short_term_retrieval_ms,
            "fusion_ms": self.fusion_ms,
            "memory_fusion_ms": self.memory_fusion_ms,
            "evidence_pack_ms": self.evidence_pack_ms,
            "generation_ms": self.generation_ms,
            "query_ms": self.query_ms,
        }


@dataclass
class LoadedQueryEngineInfo:
    session_id: str
    memory_config_path: Path
    loaded_at: float
    last_accessed_at: float
    query_count: int = 0
    recent_queries: list[dict[str, Any]] = field(default_factory=list)
