from __future__ import annotations

import os
import re
from typing import Any

from online_query.cache_schema import CacheContext
from online_query.interaction_cache import InteractionCache


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
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


class CoreferenceResolver:
    """Lightweight rule-based follow-up/coreference resolver."""

    PRONOUN_KWS = [
        "他", "她", "它", "这个", "那个", "这些", "那些", "这个东西", "那个人", "那个物体", "刚才那个",
        "he", "she", "it", "they", "this", "that", "those", "the object", "the person", "the thing",
        "that object", "this object", "that person", "this person", "the previous one",
    ]
    FOLLOWUP_KWS = [
        "刚才", "刚刚", "上一段", "那个时候", "后来呢", "之后呢", "然后呢", "接下来呢", "后来", "之后",
        "then", "later", "after that", "what about it", "what about that", "where did it go",
        "just now", "recently", "a moment ago", "a few seconds ago", "moments ago",
        "previous moment", "previously", "last segment", "last scene", "what just happened",
    ]
    LATER_KWS = ["后来", "之后", "然后", "接下来", "later", "after that", "then", "next", "then what"]

    def __init__(self, followup_window_seconds: int | None = None) -> None:
        self.followup_window_seconds = followup_window_seconds or int(os.getenv("WORLDMM_CACHE_FOLLOWUP_WINDOW_SECONDS", "60"))

    def resolve(self, question: str, interaction_cache: InteractionCache) -> dict[str, Any]:
        q = (question or "").strip()
        context = interaction_cache.latest_context()
        latest = context.get("latest_interaction") or {}
        hot_entities = list(context.get("hot_entities", []) or [])
        hot_ranges = list(context.get("hot_time_ranges", []) or [])
        hot_memories = list(context.get("hot_memories", []) or [])

        if not latest and not hot_entities and not hot_ranges:
            return CacheContext(reason="interaction cache is empty").to_dict()

        has_pronoun = _contains_any(q, self.PRONOUN_KWS)
        has_followup = _contains_any(q, self.FOLLOWUP_KWS)
        if has_pronoun and not has_followup:
            video_summary_reference = _contains_any(q, ["这个视频", "这段视频", "this video", "the video"])
            specific_reference = _contains_any(q, ["这个东西", "那个东西", "那个人", "那个物体", "刚才那个", "the object", "the person", "the thing"])
            if video_summary_reference and not specific_reference:
                has_pronoun = False
        is_followup = bool(has_pronoun or has_followup)
        if not is_followup:
            return CacheContext(
                cache_hit=True,
                reason="cache exists but question has no follow-up signal",
                referenced_entities=hot_entities[:5],
                referenced_time_ranges=hot_ranges[:3],
                referenced_memory_ids=[str(item.get("memory_id")) for item in hot_memories[:5] if item.get("memory_id")],
                referenced_segment_ids=[str(item.get("segment_id")) for item in hot_memories[:5] if item.get("segment_id")],
                referenced_evidence_frames=list(latest.get("evidence_frames", []) or [])[:2],
            ).to_dict()

        referenced_entities = list(latest.get("entities", []) or [])[:5] or hot_entities[:5]
        referenced_ranges = list(latest.get("time_ranges", []) or [])[:3] or hot_ranges[:3]
        referenced_memory_ids = list(latest.get("retrieved_memory_ids", []) or [])
        referenced_segment_ids = list(latest.get("retrieved_segment_ids", []) or [])
        referenced_visual_ids = list(latest.get("visual_ids", []) or [])
        referenced_evidence_frames = list(latest.get("evidence_frames", []) or [])[:2]

        entity_names = [
            str(item.get("canonical_name") or item.get("name") or item.get("entity_key"))
            for item in referenced_entities
            if item.get("canonical_name") or item.get("name") or item.get("entity_key")
        ]
        range_text = self._format_ranges(referenced_ranges)
        resolved_question = q
        reason = "question contains follow-up/coreference signals and cache has recent context"
        confidence = 0.62

        if _contains_any(q, self.LATER_KWS) and referenced_ranges:
            expanded = self._expand_after_ranges(referenced_ranges)
            referenced_ranges = expanded
            resolved_question = (
                f"After the previously retrieved event around {range_text}"
                f"{self._entity_clause(entity_names)}, answer: {q}"
            )
            confidence = 0.74
        elif entity_names or referenced_ranges:
            resolved_question = (
                f"Using the previous context"
                f"{self._entity_clause(entity_names)}"
                f"{self._time_clause(range_text)}, answer: {q}"
            )
            confidence = 0.7

        return CacheContext(
            cache_hit=True,
            is_followup=True,
            resolved_question=resolved_question,
            referenced_entities=referenced_entities,
            referenced_time_ranges=referenced_ranges,
            referenced_memory_ids=referenced_memory_ids,
            referenced_segment_ids=referenced_segment_ids,
            referenced_visual_ids=referenced_visual_ids,
            referenced_evidence_frames=referenced_evidence_frames,
            confidence=confidence,
            reason=reason,
        ).to_dict()

    def _format_ranges(self, ranges: list[dict[str, Any]]) -> str:
        parts = []
        for item in ranges[:2]:
            start = item.get("start")
            end = item.get("end")
            if start is not None and end is not None:
                parts.append(f"{float(start):.1f}-{float(end):.1f}s")
        return ", ".join(parts)

    def _expand_after_ranges(self, ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = []
        for item in ranges[:3]:
            end = float(item.get("end") or item.get("start") or 0.0)
            expanded.append({
                "start": end,
                "end": end + float(self.followup_window_seconds),
                "score": item.get("score", item.get("confidence", 0.7)),
                "source": "interaction_cache_followup_after",
            })
        return expanded

    def _entity_clause(self, names: list[str]) -> str:
        names = [name for name in names if name][:4]
        return f" involving {', '.join(names)}" if names else ""

    def _time_clause(self, range_text: str) -> str:
        return f" around {range_text}" if range_text else ""
