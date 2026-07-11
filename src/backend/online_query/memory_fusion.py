from __future__ import annotations

from typing import Any

from .router_schema import clamp01, compact_text, safe_float, time_overlap


class MemoryFusion:
    """Normalize, score, deduplicate, and crop evidence across memory sources."""

    def fuse(
        self,
        memory_results: dict[str, Any],
        memory_decision: dict[str, Any],
        retrieval_plan: dict[str, Any],
        query_type: str,
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_context = cache_context or {}
        priorities = memory_decision.get("memory_priority") or {}
        memory_route = memory_decision.get("memory_route") or {}
        plan_map = retrieval_plan.get("retrieval_plan") or {}
        current_only = bool(memory_route.get("use_current")) and not memory_route.get("use_short_term") and not memory_route.get("use_long_term")
        candidates: list[dict[str, Any]] = []
        input_counts = {"M_cur": 0, "M_st": 0, "M_lt": 0, "M_cache": 0}
        raw_input_counts = {
            "M_cur": len(memory_results.get("current_results", []) or []),
            "M_st": len(memory_results.get("short_term_results", []) or []),
            "M_lt": (
                len(memory_results.get("text_results", []) or [])
                + len(memory_results.get("fused_results", []) or [])
                + (0 if memory_results.get("fused_results") else len(memory_results.get("visual_results", []) or []))
            ),
            "M_cache": 0,
        }

        for item in self._limit_items(memory_results.get("current_results", []) or [], self._source_budget(plan_map, "M_cur", 1)):
            ev = self._from_current(item, priorities, query_type=query_type, current_only=current_only)
            candidates.append(ev)
            input_counts["M_cur"] += 1
        for item in self._limit_items(memory_results.get("short_term_results", []) or [], self._source_budget(plan_map, "M_st", 8)):
            ev = self._from_short_term(item, priorities)
            candidates.append(ev)
            input_counts["M_st"] += 1
        mlt_budget = self._source_budget(plan_map, "M_lt", 8)
        fused_items = list(memory_results.get("fused_results", []) or [])
        text_budget = mlt_budget
        if fused_items:
            text_budget = max(1, min(len(memory_results.get("text_results", []) or []), mlt_budget // 2))
        mlt_remaining = mlt_budget
        for item in self._limit_items(memory_results.get("text_results", []) or [], text_budget):
            ev = self._from_text(item, priorities)
            candidates.append(ev)
            input_counts["M_lt"] += 1
            mlt_remaining -= 1
        for item in self._limit_items(fused_items, mlt_remaining):
            ev = self._from_fused(item, priorities)
            candidates.append(ev)
            input_counts["M_lt"] += 1
            mlt_remaining -= 1
        if not fused_items:
            for item in self._limit_items(memory_results.get("visual_results", []) or [], mlt_remaining):
                ev = self._from_visual(item, priorities)
                candidates.append(ev)
                input_counts["M_lt"] += 1
                mlt_remaining -= 1
        cache_plan = plan_map.get("M_cache") or {}
        cache_budget = self._source_budget(plan_map, "M_cache", 3)
        if cache_plan.get("enabled", True):
            cache_items = self._cache_candidates(cache_context, limit=cache_budget)
            raw_input_counts["M_cache"] = len(cache_items)
            for item in cache_items:
                ev = self._with_scores(item, priorities.get("M_cache", 0.3), item.get("retrieval_score", 0.35), 0.0, item.get("cache_score", 0.6), item.get("visual_score", 0.0))
                candidates.append(ev)
                input_counts["M_cache"] += 1

        for ev in candidates:
            ev.setdefault("candidate_api", "em2memory_style")
            ev["time_relevance"] = max(safe_float(ev.get("time_relevance")), self._cache_time_relevance(ev, cache_context))
            ev["cache_score"] = max(safe_float(ev.get("cache_score")), self._cache_entity_relevance(ev, cache_context))
            ev["final_score"] = self._score(ev)

        selected, removed = self._dedup(candidates)
        final_k = int(retrieval_plan.get("final_evidence_k") or 4)
        selected = selected[: max(1, final_k)]
        selected_sources = list(dict.fromkeys(ev.get("source_memory") for ev in selected if ev.get("source_memory")))
        return {
            "final_evidence": selected,
            "fusion_summary": {
                "input_counts": input_counts,
                "raw_input_counts": raw_input_counts,
                "candidate_budgets": retrieval_plan.get("candidate_budgets", {}),
                "selected_memory_sources": selected_sources,
                "dedup_removed": removed,
                "final_evidence_count": len(selected),
                "query_type": query_type,
            },
        }

    def _source_budget(self, plan_map: dict[str, Any], source: str, default: int) -> int:
        plan = plan_map.get(source) or {}
        for key in ("candidate_budget", "top_k", "text_top_k"):
            value = plan.get(key)
            try:
                if value is not None:
                    return max(0, int(value))
            except Exception:
                continue
        return max(0, int(default))

    def _limit_items(self, items: list[Any], limit: int) -> list[Any]:
        items = list(items or [])
        if limit <= 0:
            return []
        return items[:limit]

    def _from_current(
        self,
        item: dict[str, Any],
        priorities: dict[str, float],
        *,
        query_type: str,
        current_only: bool,
    ) -> dict[str, Any]:
        context = item.get("current_context") or {}
        selection = item.get("current_selection") or {}
        state = context.get("state") or {}
        open_event = context.get("open_event") or {}
        frames = selection.get("evidence_frames", []) or []
        caption = compact_text([
            "Current rolling video window",
            open_event.get("retrieval_text") or open_event.get("event_caption_fast") or open_event.get("status"),
            context.get("transcript"),
        ])
        if current_only and query_type == "current_perception":
            retrieval_score = 0.85
            time_relevance = 0.8
            visual_score = 0.4
            score_policy = "primary_current"
        else:
            retrieval_score = 0.45
            time_relevance = 0.35
            visual_score = 0.2
            score_policy = "supplemental_current"
        return self._with_scores(
            {
                "evidence_id": open_event.get("open_event_id") or f"mcur_{state.get('window_start_time')}_{state.get('window_end_time')}",
                "source_memory": "M_cur",
                "source_type": "current_frame",
                "start_time": state.get("window_start_time"),
                "end_time": state.get("window_end_time"),
                "timestamp": state.get("current_time") or state.get("window_end_time"),
                "caption": caption,
                "transcript": context.get("transcript", ""),
                "keyframe_paths": [frame.get("path") for frame in frames if frame.get("path")],
                "status": "current",
                "metadata": {"current_selection": selection, "open_event": open_event, "score_policy": score_policy},
            },
            priorities.get("M_cur", 0.0),
            retrieval_score,
            time_relevance,
            0.0,
            visual_score,
        )

    def _from_short_term(self, item: dict[str, Any], priorities: dict[str, float]) -> dict[str, Any]:
        frames = item.get("keyframes", []) or []
        caption = (
            item.get("event_caption_refined")
            or item.get("event_caption_fast")
            or item.get("event_caption_placeholder")
            or item.get("retrieval_text")
            or item.get("transcript")
            or ""
        )
        return self._with_scores(
            {
                "evidence_id": item.get("event_id"),
                "source_memory": "M_st",
                "source_type": "micro_event",
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "timestamp": (safe_float(item.get("start_time")) + safe_float(item.get("end_time"), item.get("start_time"))) / 2.0,
                "caption": compact_text(caption),
                "transcript": item.get("transcript", ""),
                "keyframe_paths": [frame.get("path") for frame in frames if isinstance(frame, dict) and frame.get("path")],
                "retrieval_score": item.get("score"),
                "status": item.get("status") or "provisional",
                "metadata": item,
            },
            priorities.get("M_st", 0.0),
            item.get("score", 0.5),
            0.7,
            0.0,
            item.get("diff_score", 0.0),
        )

    def _from_text(self, item: dict[str, Any], priorities: dict[str, float]) -> dict[str, Any]:
        return self._with_scores(
            {
                "evidence_id": item.get("memory_id") or item.get("evidence_doc_id") or item.get("segment_id"),
                "source_memory": "M_lt",
                "source_type": "episodic_memory",
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "timestamp": (safe_float(item.get("start_time")) + safe_float(item.get("end_time"), item.get("start_time"))) / 2.0,
                "caption": compact_text(item.get("caption")),
                "transcript": item.get("transcript", ""),
                "keyframe_paths": item.get("keyframe_paths", []) or [],
                "retrieval_score": item.get("score"),
                "status": "final",
                "metadata": item,
            },
            priorities.get("M_lt", 0.0),
            item.get("score", 0.5),
            0.2,
            0.0,
            item.get("semantic_score", 0.0),
        )

    def _from_fused(self, item: dict[str, Any], priorities: dict[str, float]) -> dict[str, Any]:
        text = item.get("text") if isinstance(item.get("text"), dict) else {}
        visual_items = item.get("visual_items", []) or []
        first_visual = visual_items[0] if visual_items else {}
        start = text.get("start_time") if text else first_visual.get("start_time")
        end = text.get("end_time") if text else first_visual.get("end_time", start)
        caption = compact_text([
            text.get("caption") if text else "",
            [visual.get("keyframe_caption") or visual.get("segment_caption") for visual in visual_items[:2]],
        ])
        return self._with_scores(
            {
                "evidence_id": item.get("canonical_segment_id") or item.get("segment_id"),
                "source_memory": "M_lt",
                "source_type": "episodic_memory",
                "start_time": start,
                "end_time": end,
                "timestamp": first_visual.get("timestamp") if first_visual else (safe_float(start) + safe_float(end, start)) / 2.0,
                "caption": caption,
                "transcript": text.get("transcript", "") if text else "",
                "keyframe_paths": [visual.get("image_path") for visual in visual_items if visual.get("image_path")] or text.get("keyframe_paths", []),
                "retrieval_score": item.get("fused_score"),
                "visual_score": item.get("visual_score"),
                "semantic_score": item.get("text_score"),
                "status": "final",
                "metadata": item,
            },
            priorities.get("M_lt", 0.0),
            item.get("fused_score", 0.5),
            0.2,
            item.get("cache_boost", 0.0),
            max(safe_float(item.get("visual_score")), safe_float(item.get("text_score"))),
        )

    def _from_visual(self, item: dict[str, Any], priorities: dict[str, float]) -> dict[str, Any]:
        return self._with_scores(
            {
                "evidence_id": item.get("visual_id") or item.get("image_path"),
                "source_memory": "M_lt",
                "source_type": "visual_frame",
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "timestamp": item.get("timestamp"),
                "caption": compact_text(item.get("keyframe_caption") or item.get("segment_caption")),
                "keyframe_paths": [item.get("image_path")] if item.get("image_path") else [],
                "retrieval_score": item.get("score"),
                "visual_score": item.get("visual_score") or item.get("score"),
                "status": "final",
                "metadata": item,
            },
            priorities.get("M_lt", 0.0),
            item.get("score", 0.5),
            0.2,
            0.0,
            item.get("visual_score", item.get("score", 0.0)),
        )

    def _cache_candidates(self, cache_context: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
        if not (cache_context.get("cache_hit") or cache_context.get("is_followup")):
            return []
        out: list[dict[str, Any]] = []
        followup = bool(cache_context.get("is_followup"))
        base_score = 0.62 if followup else 0.36
        for frame in (cache_context.get("referenced_evidence_frames", []) or [])[:2]:
            if not isinstance(frame, dict):
                continue
            out.append({
                "evidence_id": frame.get("path") or frame.get("image_path"),
                "source_memory": "M_cache",
                "source_type": "cache_frame",
                "timestamp": frame.get("timestamp"),
                "caption": frame.get("caption", ""),
                "keyframe_paths": [frame.get("path") or frame.get("image_path")] if (frame.get("path") or frame.get("image_path")) else [],
                "retrieval_score": base_score,
                "cache_score": base_score,
                "metadata": frame,
            })
            if len(out) >= limit:
                return out
        for idx, window in enumerate(cache_context.get("referenced_time_ranges", []) or []):
            if not isinstance(window, dict):
                continue
            start = window.get("start")
            end = window.get("end", start)
            out.append({
                "evidence_id": f"cache_time_{idx}_{start}_{end}",
                "source_memory": "M_cache",
                "source_type": "cache_time_range",
                "start_time": start,
                "end_time": end,
                "timestamp": start,
                "caption": f"Interaction cache referenced time range {start}-{end}s.",
                "retrieval_score": max(base_score - 0.05, 0.0),
                "cache_score": base_score,
                "metadata": window,
            })
            if len(out) >= limit:
                return out
        memory_ids = list(dict.fromkeys(
            [str(value) for value in cache_context.get("referenced_memory_ids", []) or [] if value]
            + [str(value) for value in cache_context.get("referenced_segment_ids", []) or [] if value]
        ))
        for memory_id in memory_ids:
            out.append({
                "evidence_id": memory_id,
                "source_memory": "M_cache",
                "source_type": "cache_memory_ref",
                "caption": f"Interaction cache referenced previous memory {memory_id}.",
                "retrieval_score": max(base_score - 0.08, 0.0),
                "cache_score": base_score,
                "metadata": {"memory_id": memory_id},
            })
            if len(out) >= limit:
                return out
        entities = []
        for entity in cache_context.get("referenced_entities", []) or []:
            if not isinstance(entity, dict):
                continue
            name = entity.get("canonical_name") or entity.get("name") or entity.get("entity_key")
            if name:
                entities.append(str(name))
        if entities and len(out) < limit:
            out.append({
                "evidence_id": "cache_entities_" + "_".join(entities[:3]),
                "source_memory": "M_cache",
                "source_type": "cache_entity_ref",
                "caption": "Interaction cache referenced entities: " + ", ".join(entities[:6]),
                "retrieval_score": max(base_score - 0.1, 0.0),
                "cache_score": base_score,
                "metadata": {"entities": entities[:6]},
            })
        return out

    def _with_scores(
        self,
        item: dict[str, Any],
        memory_priority: Any,
        retrieval_score: Any,
        time_relevance: Any,
        cache_score: Any,
        visual_or_semantic_score: Any,
    ) -> dict[str, Any]:
        item["memory_priority"] = clamp01(memory_priority)
        item["retrieval_score"] = clamp01(item.get("retrieval_score", retrieval_score))
        item["time_relevance"] = clamp01(time_relevance)
        item["cache_score"] = clamp01(cache_score)
        item["visual_score"] = clamp01(item.get("visual_score", visual_or_semantic_score))
        item["semantic_score"] = clamp01(item.get("semantic_score", visual_or_semantic_score))
        item["final_score"] = self._score(item)
        return item

    def _score(self, item: dict[str, Any]) -> float:
        visual_or_semantic = max(clamp01(item.get("visual_score")), clamp01(item.get("semantic_score")))
        return round(
            0.30 * clamp01(item.get("memory_priority"))
            + 0.30 * clamp01(item.get("retrieval_score"))
            + 0.15 * clamp01(item.get("time_relevance"))
            + 0.15 * clamp01(item.get("cache_score"))
            + 0.10 * visual_or_semantic,
            6,
        )

    def _cache_time_relevance(self, item: dict[str, Any], cache_context: dict[str, Any]) -> float:
        best = 0.0
        for window in cache_context.get("referenced_time_ranges", []) or []:
            if not isinstance(window, dict):
                continue
            best = max(best, time_overlap(item.get("start_time"), item.get("end_time"), window.get("start"), window.get("end")))
        return best

    def _cache_entity_relevance(self, item: dict[str, Any], cache_context: dict[str, Any]) -> float:
        entities = []
        for ent in cache_context.get("referenced_entities", []) or []:
            if isinstance(ent, dict):
                entities.extend([ent.get("canonical_name"), ent.get("name"), ent.get("entity_key")])
            else:
                entities.append(ent)
        terms = [str(term).lower() for term in entities if term]
        if not terms:
            return 0.0
        haystack = f"{item.get('caption','')} {item.get('transcript','')} {item.get('metadata','')}".lower()
        return 0.7 if any(term in haystack for term in terms) else 0.0

    def _dedup(self, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        candidates = sorted(candidates, key=lambda item: -safe_float(item.get("final_score")))
        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        removed = 0
        for item in candidates:
            evidence_id = str(item.get("evidence_id") or "")
            paths = {str(path) for path in item.get("keyframe_paths", []) or [] if path}
            if evidence_id and evidence_id in seen_ids:
                removed += 1
                continue
            if paths and paths.issubset(seen_paths):
                removed += 1
                continue
            if self._is_redundant_time(item, selected):
                removed += 1
                continue
            selected.append(item)
            if evidence_id:
                seen_ids.add(evidence_id)
            seen_paths.update(paths)
        return selected, removed

    def _is_redundant_time(self, item: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
        if item.get("source_memory") != "M_lt":
            return False
        for existing in selected:
            if existing.get("source_memory") != item.get("source_memory"):
                continue
            if time_overlap(item.get("start_time"), item.get("end_time"), existing.get("start_time"), existing.get("end_time")) >= 0.9:
                return True
        return False
