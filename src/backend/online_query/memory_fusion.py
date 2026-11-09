from __future__ import annotations

import json

from typing import Any

from online_time import event_time_seconds
from .cache_schema import has_valid_binding, has_exact_frame_binding
from .evidence_schema import normalize_evidence
from .router_schema import clamp01, compact_text, safe_float, time_overlap
from online_query_constraints import constraint_window, constraint_windows, reserve_time_coverage, overlaps, time_fit


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
        constraints = memory_decision.get("query_constraints")
        if constraints is None:
            constraints = cache_context.get("query_constraints")
        if constraints is None:
            constraints = {}
        windows = constraint_windows(constraints)
        def limited(items, budget):
            return reserve_time_coverage(items or [], windows, budget) if len(windows) > 1 else self._limit_items(items or [], budget)
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

        for item in limited(memory_results.get("current_results", []) or [], self._source_budget(plan_map, "M_cur", 1)):
            ev = self._from_current(item, priorities, query_type=query_type, current_only=current_only)
            candidates.append(ev)
            input_counts["M_cur"] += 1
        for item in limited(memory_results.get("short_term_results", []) or [], self._source_budget(plan_map, "M_st", 16)):
            ev = self._from_short_term(item, priorities)
            candidates.append(ev)
            input_counts["M_st"] += 1
        mlt_budget = self._source_budget(plan_map, "M_lt", 8)
        fused_items = list(memory_results.get("fused_results", []) or [])
        # Text and hybrid views often refer to the same LT event. Count each
        # unique event once before its source budget can discard other history.
        long_term = [self._from_text(item, priorities) for item in memory_results.get("text_results", []) or []]
        long_term.extend(self._from_fused(item, priorities) for item in fused_items)
        if not fused_items:
            long_term.extend(self._from_visual(item, priorities) for item in memory_results.get("visual_results", []) or [])
        long_term, duplicate_long_term_views = self._dedup(long_term)
        supplemental = self._supported_candidates(long_term, constraints) if mlt_budget > 0 else []
        long_term = limited(long_term, mlt_budget)
        # The retrieval budget has already been spent before a bounded
        # observation supplement is appended. Do not silently crop that pass
        # away a second time; the final shared evidence limit still applies.
        for item in supplemental[:2 if (constraints.get("requirements") or {}).get("comparison") else 1]:
            if item not in long_term:
                long_term.append(item)
        candidates.extend(long_term)
        input_counts["M_lt"] = len(long_term)
        cache_plan = plan_map.get("M_cache") or {}
        cache_budget = self._source_budget(plan_map, "M_cache", 3)
        if cache_plan.get("enabled", True):
            cache_items = self._cache_candidates(cache_context, limit=cache_budget)
            raw_input_counts["M_cache"] = len(cache_items)
            for item in cache_items:
                ev = self._with_scores(item, priorities.get("M_cache", 0.3), item.get("retrieval_score", 0.35), 0.0, item.get("cache_score", 0.6), item.get("visual_score", 0.0))
                candidates.append(ev)
                input_counts["M_cache"] += 1

        requested_window = constraint_window(constraints)
        reliable_binding = has_valid_binding(cache_context)
        reliable_same_time = (reliable_binding and cache_context.get("temporal_relation") == "same_time"
            and not constraints.get("requires_current")
            and not constraints.get("requires_current_tail")
            and not memory_decision.get("requires_current")
            and not memory_decision.get("requires_current_tail"))
        before_time_filter = len(candidates)
        exact_cache_time = reliable_same_time and has_exact_frame_binding(cache_context)
        if requested_window:
            candidates = [item for item in candidates if overlaps(item.get("start_time", item.get("timestamp")), item.get("end_time", item.get("timestamp")), requested_window)]
        if not requested_window and exact_cache_time and cache_context.get("referenced_time_ranges"):
            ranges = cache_context.get("referenced_time_ranges") or []
            def in_scope(item):
                start = item.get("start_time")
                if start is None:
                    start = item.get("timestamp")
                end = item.get("end_time", start)
                if start is None:
                    return item.get("source_type") in {"cache_entity_ref", "cache_memory_ref"}
                return any(safe_float(start) <= safe_float(window.get("end"), safe_float(window.get("start"))) and safe_float(end, safe_float(start)) >= safe_float(window.get("start")) for window in ranges)
            candidates = [item for item in candidates if in_scope(item)]
        after_time_filter = len(candidates)
        for ev in candidates:
            ev.setdefault("candidate_api", "em2memory_style")
            ev["time_relevance"] = (time_fit(ev.get("start_time", ev.get("timestamp")), ev.get("end_time", ev.get("timestamp")), requested_window)
                                    if requested_window else max(safe_float(ev.get("time_relevance")), self._cache_time_relevance(ev, cache_context) if reliable_binding else 0.0))
            if not requested_window and reliable_binding:
                ev["time_relevance"] = self._cache_time_relevance(ev, cache_context)
            elif not requested_window and constraints.get("scope") == "history":
                # An unconstrained historical question supplies no recency
                # preference. A source-wide constant must not favor short-term
                # events merely because they are stored in that layer.
                ev["time_relevance"] = 0.0
            ev["cache_score"] = max(safe_float(ev.get("cache_score")), self._cache_entity_relevance(ev, cache_context))
            ev["final_score"] = self._score(ev)

        source_routes = {"M_cur": "use_current", "M_st": "use_short_term", "M_lt": "use_long_term", "M_cache": "use_interaction_cache"}
        candidates = [item for item in candidates
            if (plan_map.get(item.get("source_memory")) or {}).get("enabled", True)
            and memory_route.get(source_routes.get(item.get("source_memory")), True)]
        selected, removed = self._dedup(candidates)
        candidates = selected  # All later reservations must use canonical evidence.
        final_k = max(0, int(retrieval_plan.get("final_evidence_k", 4)))
        selection_policy = "fusion_score"
        if len(constraint_windows(constraints)) > 1:
            selected = reserve_time_coverage(selected, constraint_windows(constraints), max(1, final_k))
            selection_policy = "requested_time_window_coverage"
        elif reliable_same_time and cache_context.get("referenced_time_ranges"):
            selected = self._preserve_followup_context(selected, cache_context["referenced_time_ranges"], max(1, final_k),
                                                       allow_alternative=not exact_cache_time)
            selection_policy = "previous_occurrence_coverage" if exact_cache_time else "soft_previous_occurrence_with_alternative"
        elif query_type != "current_perception" and not reliable_binding:
            selected = self._preserve_temporal_context(selected, max(1, final_k), query_type, memory_decision.get("question") or constraints.get("question") or "")
            selection_policy = "long_term_anchors_and_short_term_tail" if query_type == "long_term_summary" else "long_term_context_and_short_term_details"
        else:
            selected = selected[: max(1, final_k)]
        temporal_preference = constraints.get("temporal_preference")
        if (temporal_preference in {"earlier", "earlier_and_later"} and not requested_window
                and (not reliable_binding or (temporal_preference == "earlier_and_later" and not exact_cache_time))):
            history = [item for item in candidates if item.get("source_memory") in {"M_lt", "M_st"}
                       and item.get("start_time") is not None and safe_float(item.get("retrieval_score")) > 0]
            if history:
                earliest = min(history, key=lambda item: (safe_float(item["start_time"]), -safe_float(item.get("final_score"))))
                selected = ([earliest] + [item for item in selected if item is not earliest])[:max(1, final_k)]
                selection_policy = "earlier_retrieved_anchor_and_ranked_alternatives"
        supplements = self._supported_candidates(candidates, constraints)
        reserved_supplements = []
        if supplements:
            # A verified condition is evidence even when its retrieval score
            # is small. Its match strength never replaces that score.
            requirements = constraints.get("requirements") or {}
            supplement_limit = 2 if requirements.get("comparison") else 1
            # Preserve the historical two-period behavior of legacy callers.
            if not requirements and all(not (item.get("metadata") or {}).get("requirement_support") for item in supplements):
                supplement_limit = 2
            reserved_supplements = supplements[:min(supplement_limit, max(1, final_k))]
            selected = (reserved_supplements + [item for item in selected if item not in reserved_supplements])[:max(1, final_k)]
        if len(constraint_windows(constraints)) > 1:
            ranked = sorted(candidates, key=lambda item: -safe_float(item.get("final_score")))
            preferred = reserved_supplements + [item for item in ranked if item not in reserved_supplements]
            selected = reserve_time_coverage(preferred, constraint_windows(constraints), max(1, final_k))
        required_sources = []
        requires_current = any(container.get("requires_current") or container.get("requires_current_tail") for container in (constraints, memory_decision))
        if requires_current and memory_route.get("use_current", True) and (plan_map.get("M_cur") or {}).get("enabled", True):
            required_sources.append({"M_cur"})
        if constraints.get("requires_history"):
            required_sources.append({"M_st", "M_lt", "M_cache"})
        reserved = []
        for sources in required_sources:
            match = next((item for item in selected + sorted(candidates, key=lambda row: -safe_float(row.get("final_score"))) if item.get("source_memory") in sources), None)
            if match is not None and match not in reserved:
                reserved.append(match)
        selected = (reserved + [item for item in selected if item not in reserved])[:final_k]
        missing_required_sources = [sorted(sources) for sources in required_sources if not any(item.get("source_memory") in sources for item in selected)]
        if query_type == "long_term_summary" and selected:
            from online_evidence_content import content_fields
            # Shared summaries remain useful when their child loses an event
            # slot. Preserve each once as explicitly broader background; they
            # do not authorize images from a discarded event.
            parents = {}
            for item in candidates:
                for parent in content_fields(item).get("parent_context", []) or []:
                    if isinstance(parent, dict):
                        key = parent.get("doc_id") or json.dumps(parent, sort_keys=True, default=str)
                        parents.setdefault(key, {**parent, "scope": "coarse overview background; use its own time range, not this child's time"})
            if parents:
                first = dict(selected[0])
                first["metadata"] = {**(first.get("metadata") or {}), "parent_context": list(parents.values())}
                first["parent_context"] = list(parents.values())
                selected[0] = first
        selected_sources = list(dict.fromkeys(ev.get("source_memory") for ev in selected if ev.get("source_memory")))
        return {
            "final_evidence": selected,
            "fusion_summary": {
                "input_counts": input_counts,
                "raw_input_counts": raw_input_counts,
                "candidate_budgets": retrieval_plan.get("candidate_budgets", {}),
                "selected_memory_sources": selected_sources,
                "dedup_removed": removed + duplicate_long_term_views,
                "long_term_duplicate_views_removed": duplicate_long_term_views,
                "final_evidence_count": len(selected),
                "query_type": query_type,
                "selection_policy": selection_policy,
                "missing_required_sources": missing_required_sources,
                "supported_evidence_ids": [item.get("evidence_id") for item in selected if item in supplements],
                "time_filter": {"before": before_time_filter, "after": after_time_filter,
                    "policy": "explicit_request" if requested_window else "explicit_cached_frame" if exact_cache_time
                    else "soft_cache_preference" if reliable_same_time else "none"},
            },
        }

    def _supported_candidates(self, candidates, constraints):
        """Reserve a bounded inspection candidate without asserting its truth."""
        requirements = constraints.get("requirements") or {}
        supported = []
        for item in candidates:
            metadata = item.get("metadata") or {}
            support = metadata.get("requirement_support") or {}
            if requirements.get("comparison") and support.get("comparison_phase"):
                accepted = support.get("observation_matched") is True
            elif requirements.get("simultaneous"):
                conditions = support.get("requested_conditions") or []
                signals = support.get("signals") or {}
                accepted = (support.get("observation_matched") is True
                            and support.get("conditions_co_mentioned") is True and bool(conditions)
                            and all(signals.get(name) in {"confirmed", "candidate"} for name in conditions))
            elif support:
                accepted = support.get("observation_matched") is True or support.get("confirmed") is True
            else:
                # Compatibility for already scored comparison supplements;
                # a new unscored candidate needs an actual support record.
                accepted = bool(metadata.get("supplemental")) and item.get("score_policy") != "unscored_supplemental_observation"
            if accepted:
                supported.append(item)
        if requirements.get("earliest"):
            supported.sort(key=lambda item: safe_float(((item.get("metadata") or {}).get("requirement_support") or {}).get("start"), safe_float(item.get("start_time"))))
        if requirements.get("comparison") and len(supported) > 1:
            timeline = sorted(supported, key=lambda item: safe_float(((item.get("metadata") or {}).get("requirement_support") or {}).get("start"), safe_float(item.get("start_time"))))
            supported = [timeline[0], timeline[-1]] + timeline[1:-1]
        return supported

    def _preserve_followup_context(self, candidates, ranges, limit, *, allow_alternative=False):
        reserved = []
        for window in ranges:
            matching = [item for item in candidates if overlaps(item.get("start_time", item.get("timestamp")), item.get("end_time", item.get("timestamp")), [window["start"], window.get("end", window["start"])])]
            # A bare cache time reference is a search constraint, not evidence.
            matching = [item for item in matching if item.get("source_type") not in {"cache_time_range", "cache_entity_ref", "cache_memory_ref"}]
            if matching and not any(item in matching for item in reserved):
                reserved.append(max(matching, key=lambda item: item.get("final_score", 0)))
        if allow_alternative and limit > 1:
            # Keep one independently retrieved alternative: citations may only
            # show a small part of an action, or the resolver may bind wrongly.
            alternative = next((item for item in candidates if item.get("source_memory") != "M_cache"
                and not any(overlaps(item.get("start_time"), item.get("end_time"), [w["start"], w.get("end",w["start"])]) for w in ranges)), None)
            if alternative is not None:
                reserved = reserved[:limit-1] + [alternative]
        reserved = reserved[:limit]
        return (reserved + [item for item in candidates if item not in reserved])[:limit]

    def _preserve_temporal_context(self, candidates: list[dict[str, Any]], limit: int, query_type: str, question: str = "") -> list[dict[str, Any]]:
        """Protect selected historical anchors from a heap of recent details."""
        long_term = [item for item in candidates if item.get("source_memory") == "M_lt"]
        short_term = [item for item in candidates if item.get("source_memory") == "M_st"]
        reserved = []
        live_context = [item for item in candidates if item.get("source_memory") in {"M_cur", "M_st"}]
        has_live_context = bool(live_context)
        long_budget = max(1, limit - (1 if has_live_context and limit > 1 else 0))
        if query_type == "long_term_summary":
            # Retrieval has already selected these anchors. A second keyword
            # coverage selector must not replace its strongest events.
            reserved.extend(long_term[:long_budget])
            if live_context and len(reserved) < limit:
                # Reserve the actual newest available endpoint. A closed ST
                # event can stop before the current window; more historical
                # candidates must not make that open tail disappear.
                reserved.append(max(live_context, key=lambda item: (
                    safe_float(item.get("end_time")), item.get("source_memory") == "M_st")))
        else:
            # Guarantee a small independent view of each enabled source, then
            # share remaining positions by fusion rank rather than fixed ratios.
            reserved.extend(long_term[:2 if limit >= 6 else 1])
            fine_floor = 2 if limit >= 6 else 1
            reserved.extend(short_term[:min(fine_floor, max(0, limit-len(reserved)))])
            current = next((item for item in live_context if item.get("source_memory") == "M_cur"), None)
            if current is not None and len(reserved) < limit:
                reserved.append(current)
        seen = {id(item) for item in reserved}
        for item in candidates:
            if len(reserved) >= limit:
                break
            if id(item) not in seen:
                reserved.append(item)
                seen.add(id(item))
        if query_type == "long_term_summary":
            reserved.sort(key=lambda item: (safe_float(item.get("start_time")), -safe_float(item.get("final_score"))))
        else:
            reserved.sort(key=lambda item: -safe_float(item.get("final_score")))
        return reserved[:limit]

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
            visual_score = 0.0
            score_policy = "primary_current"
        else:
            retrieval_score = 0.45
            time_relevance = 0.35
            visual_score = 0.0
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
                "score_policy": score_policy,
                "metadata": {"current_selection": selection, "open_event": open_event, "score_policy": score_policy, "raw_score": None},
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
                "timestamp": (safe_float(item.get("start_time")) + safe_float(item.get("end_time"), safe_float(item.get("start_time")))) / 2.0,
                "caption": compact_text(caption, max_chars=0),
                "transcript": item.get("transcript", ""),
                "keyframe_paths": [frame.get("path") for frame in frames if isinstance(frame, dict) and frame.get("path")],
                "retrieval_score": item.get("score"),
                "status": item.get("status") or "provisional",
                "metadata": item,
                "diff_score": item.get("diff_score", 0.0),
                "visual_score": 0.0,
                "semantic_score": safe_float(item.get("semantic_score")),
                "score_policy": item.get("score_policy") or ("summary_timeline_coverage_priority" if item.get("summary_timeline_fallback") else "short_term_bm25_lexical_recency_cache_continuity"),
            },
            priorities.get("M_st", 0.0),
            item.get("score", 0.5),
            0.7,
            0.0,
            safe_float(item.get("semantic_score")),
        )

    def _from_text(self, item: dict[str, Any], priorities: dict[str, float]) -> dict[str, Any]:
        start = event_time_seconds(item.get("start_time"))
        end = event_time_seconds(item.get("end_time"))
        end = start if end is None else end
        return self._with_scores(
            {
                "evidence_id": item.get("memory_id") or item.get("evidence_doc_id") or item.get("segment_id"),
                "source_memory": "M_lt",
                "source_type": "episodic_memory",
                "segment_id": item.get("segment_id"),
                "canonical_segment_id": item.get("canonical_segment_id") or item.get("segment_id"),
                "start_time": start,
                "end_time": end,
                "timestamp": (start + end) / 2.0 if start is not None and end is not None else None,
                "caption": compact_text(item.get("caption"), max_chars=0),
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
        start = event_time_seconds(text.get("start_time") if text else first_visual.get("start_time"))
        end = event_time_seconds(text.get("end_time") if text else first_visual.get("end_time"))
        end = start if end is None else end
        caption = compact_text([
            text.get("caption") if text else "",
            [visual.get("keyframe_caption") or visual.get("segment_caption") for visual in visual_items[:2]],
        ], max_chars=0)
        return self._with_scores(
            {
                "evidence_id": item.get("canonical_segment_id") or item.get("segment_id"),
                "segment_id": item.get("segment_id"),
                "canonical_segment_id": item.get("canonical_segment_id") or item.get("segment_id"),
                "source_memory": "M_lt",
                "source_type": "episodic_memory",
                "start_time": start,
                "end_time": end,
                "timestamp": first_visual.get("timestamp") if first_visual else ((start + end) / 2.0 if start is not None and end is not None else None),
                "caption": caption,
                "transcript": text.get("transcript", "") if text else "",
                "keyframe_paths": [visual.get("image_path") for visual in visual_items if visual.get("image_path")] or text.get("keyframe_paths", []),
                "retrieval_score": item.get("fused_score"),
                "visual_score": item.get("visual_score"),
                "semantic_score": item.get("text_score"),
                "status": "final",
                "metadata": {**item, **{key: text[key] for key in (
                    "supplemental", "supplement_match_score", "requirement_support", "score_provenance",
                ) if key in text}},
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
        if limit <= 0 or not (cache_context.get("cache_hit") or cache_context.get("is_followup")):
            return []
        out: list[dict[str, Any]] = []
        followup = bool(cache_context.get("is_followup"))
        base_score = 0.62 if followup else 0.36
        frames = list(cache_context.get("referenced_evidence_frames", []) or [])
        if followup and cache_context.get("referenced_time_ranges"):
            # First offer one real frame from every occurrence, then fill.
            anchors = []
            for window in cache_context["referenced_time_ranges"]:
                match = next((frame for frame in frames if frame.get("timestamp") is not None and overlaps(frame["timestamp"], frame["timestamp"], [window["start"], window.get("end", window["start"])])), None)
                if match is not None and match not in anchors:
                    anchors.append(match)
            frames = anchors + [frame for frame in frames if frame not in anchors]
        for frame in frames:
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
        # Ranges, IDs and entities guide retrieval; only observed frames are evidence.
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
        item = normalize_evidence(item)
        item["memory_priority"] = clamp01(memory_priority)
        raw_score = item.get("retrieval_score", retrieval_score)
        item["raw_retrieval_score"] = (item.get("metadata") or {}).get("raw_score", raw_score)
        item.setdefault("score_policy", (item.get("metadata") or {}).get("score_policy") or "source_specific_retrieval")
        item["retrieval_score"] = clamp01(raw_score)
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
            best = max(best, time_fit(item.get("start_time"), item.get("end_time"), [safe_float(window.get("start")), safe_float(window.get("end"), safe_float(window.get("start")))]))
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
        seen_ids: dict[tuple[str, str], dict[str, Any]] = {}
        seen_content: set[tuple[Any, ...]] = set()
        removed = 0
        for item in candidates:
            evidence_id = str(item.get("evidence_id") or "")
            identity_id = evidence_id
            if item.get("source_memory") == "M_lt" and item.get("source_type") == "episodic_memory":
                identity_id = str(item.get("canonical_segment_id") or item.get("segment_id") or evidence_id)
            identity = (str(item.get("source_memory") or ""), identity_id)
            if evidence_id and identity in seen_ids:
                existing = seen_ids[identity]
                existing["keyframe_paths"] = list(dict.fromkeys((existing.get("keyframe_paths") or []) + (item.get("keyframe_paths") or [])))
                for key in ("caption", "transcript"):
                    old, new = str(existing.get(key) or ""), str(item.get(key) or "")
                    if new and new not in old:
                        existing[key] = new if old in new else old + "\n" + new
                # Merge metadata rather than dropping richer hybrid/text fields.
                incoming_metadata = item.get("metadata") or {}
                existing_metadata = existing.get("metadata") or {}
                merged_metadata = {**incoming_metadata, **existing_metadata}
                supports = [support for support in (
                    existing_metadata.get("requirement_support"), incoming_metadata.get("requirement_support")
                ) if isinstance(support, dict) and support]
                if supports:
                    # Keep an observation atomically; never manufacture a
                    # conjunction by merging signals from separate records.
                    merged_metadata["requirement_support"] = max(supports, key=lambda support: (
                        support.get("confirmed") is True,
                        support.get("observation_matched") is True,
                        safe_float(support.get("match_score")),
                    ))
                existing["metadata"] = merged_metadata
                removed += 1
                continue
            fingerprint = (item.get("source_memory"), item.get("start_time"), item.get("end_time"), item.get("caption"), item.get("transcript"), json.dumps(item.get("metadata") or {}, sort_keys=True, default=str))
            if fingerprint in seen_content:
                removed += 1
                continue
            selected.append(item)
            if evidence_id:
                seen_ids[identity] = item
            seen_content.add(fingerprint)
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
