from __future__ import annotations

import os
from typing import Any

from .router_schema import parse_auto_bool


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(default))


class RetrievalPlanner:
    """Stage 7 level-2 planner: choose retrieval/evidence policy per memory."""

    def plan(
        self,
        memory_decision: dict[str, Any],
        request_options: dict[str, Any] | None,
        runtime_state: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_options = request_options or {}
        cache_context = cache_context or {}
        query_type = str(memory_decision.get("query_type") or "general_qa")
        memory_route = dict(memory_decision.get("memory_route") or {})
        top_k = _coerce_positive_int(request_options.get("top_k"), _env_int("EM2MEM_QUERY_ROUTER_TEXT_EVIDENCE_K", 5))
        budgets = self._candidate_budgets(query_type=query_type, base_top_k=top_k)
        mst_top_k = budgets["M_st"]
        mlt_top_k = budgets["M_lt"]
        if request_options.get("text_top_k") is not None:
            mlt_top_k = _coerce_positive_int(request_options.get("text_top_k"), mlt_top_k)
        text_top_k = mlt_top_k
        visual_top_k = int(request_options.get("visual_top_k") or _env_int("EM2MEM_QUERY_ROUTER_VISUAL_TOP_K", 8))
        requested_final_k = _coerce_positive_int(
            request_options.get("final_evidence_k"),
            _env_int("EM2MEM_FINAL_TEXT_EVIDENCE_K", _env_int("EM2MEM_QUERY_ROUTER_FINAL_EVIDENCE_K", 4)),
        )
        final_k = self._final_evidence_budget(query_type=query_type, requested=requested_final_k)
        frames_k = _env_int("EM2MEM_FINAL_EVIDENCE_FRAMES_K", _env_int("EM2MEM_QUERY_ROUTER_EVIDENCE_FRAMES_K", 5))
        max_images_default = _env_int("EM2MEM_FINAL_MAX_IMAGE_EVIDENCE", _env_int("EM2MEM_QUERY_ROUTER_MAX_IMAGE_EVIDENCE", 3))

        mlt_mode = self._default_mlt_mode(query_type)
        retrieval_mode_source = "auto"
        requested_mode = str(request_options.get("retrieval_mode") or "auto").strip().lower()
        if requested_mode in {"current", "text_only", "visual_only", "hybrid"}:
            retrieval_mode_source = "user_override"
            if requested_mode == "current":
                memory_route["use_current"] = True
            else:
                mlt_mode = requested_mode

        requested_image = parse_auto_bool(request_options.get("use_image_evidence", "auto"))
        use_image_source = "auto"
        if isinstance(requested_image, bool):
            use_image = requested_image
            use_image_source = "user_override"
        else:
            use_image = True
        try:
            max_images = int(request_options.get("max_image_evidence") if request_options.get("max_image_evidence") is not None else max_images_default)
        except Exception:
            max_images = max_images_default
        max_images = max(0, min(max_images, max_images_default))
        if not use_image:
            max_images = 0

        plan = {
            "retrieval_plan": {
                "M_cur": {
                    "enabled": bool(memory_route.get("use_current") and runtime_state.get("current_ready") and not runtime_state.get("current_stale")),
                    "mode": "current",
                    "candidate_budget": budgets["M_cur"],
                    "top_k": budgets["M_cur"],
                    "max_images": max_images,
                    "max_evidence_frames": frames_k,
                },
                "M_st": {
                    "enabled": bool(memory_route.get("use_short_term") and runtime_state.get("short_term_ready")),
                    "mode": "short_term",
                    "candidate_budget": mst_top_k,
                    "top_k": mst_top_k,
                    "include_keyframes": True,
                    "cache_boost": bool(memory_route.get("use_interaction_cache")),
                },
                "M_lt": {
                    "enabled": bool(memory_route.get("use_long_term") and runtime_state.get("long_term_ready")),
                    "mode": mlt_mode,
                    "candidate_budget": mlt_top_k,
                    "text_top_k": text_top_k,
                    "visual_top_k": visual_top_k,
                    "semantic_top_k": max(text_top_k, 5),
                    "retrieval_mode_source": retrieval_mode_source,
                },
                "M_cache": {
                    "enabled": bool(memory_route.get("use_interaction_cache")),
                    "mode": "context_boost",
                    "candidate_budget": budgets["M_cache"],
                    "top_k": budgets["M_cache"],
                    "use_entities": True,
                    "use_time_ranges": True,
                    "cache_hit": bool(cache_context.get("cache_hit")),
                    "is_followup": bool(cache_context.get("is_followup")),
                },
            },
            "candidate_budgets": budgets,
            "use_image_evidence": bool(use_image),
            "use_image_evidence_source": use_image_source,
            "max_image_evidence": max_images,
            "final_evidence_k": max(1, final_k),
            "evidence_frames_k": max(1, frames_k),
            "text_top_k": max(1, text_top_k),
            "visual_top_k": max(1, visual_top_k),
            "retrieval_mode": "current" if memory_route.get("use_current") and not memory_route.get("use_long_term") and not memory_route.get("use_short_term") else mlt_mode,
            "retrieval_mode_source": retrieval_mode_source,
        }
        if plan["retrieval_plan"]["M_lt"]["enabled"] and mlt_mode in {"hybrid", "visual_only"} and not runtime_state.get("visual_embedding_ready"):
            if mlt_mode == "visual_only":
                plan["retrieval_plan"]["M_lt"]["mode"] = "text_only"
            plan.setdefault("warnings", []).append("visual index is not ready; long-term visual retrieval will fallback to text-only where needed")
        return plan

    def _default_mlt_mode(self, query_type: str) -> str:
        if query_type == "long_term_summary":
            return "text_only"
        if query_type in {"visual_attribute", "entity_tracking"}:
            return "hybrid"
        if query_type in {"temporal_reasoning", "temporal_localization", "temporal_count"}:
            return "hybrid"
        if query_type == "speech_query":
            return "text_only"
        return "hybrid"

    def _candidate_budgets(self, *, query_type: str, base_top_k: int) -> dict[str, int]:
        mcur = _env_int("EM2MEM_UNIFIED_MCUR_TOP_K", 1)
        mcache = _env_int("EM2MEM_UNIFIED_MCACHE_TOP_K", 3)
        mst = max(base_top_k, _env_int("EM2MEM_UNIFIED_MST_TOP_K", 8))
        mlt = max(base_top_k, _env_int("EM2MEM_UNIFIED_MLT_TOP_K", 8))
        if query_type == "current_perception":
            return {"M_cur": max(1, mcur), "M_st": min(mst, 2), "M_lt": min(mlt, 2), "M_cache": min(mcache, 2)}
        if query_type == "temporal_count":
            temporal_top_k = _env_int("EM2MEM_TEMPORAL_COUNT_TOP_K", 12)
            return {"M_cur": max(1, mcur), "M_st": max(mst, temporal_top_k), "M_lt": max(mlt, temporal_top_k), "M_cache": max(mcache, 3)}
        if query_type in {"recent_recall", "followup"}:
            return {"M_cur": max(1, mcur), "M_st": max(mst, 8), "M_lt": max(mlt, 8), "M_cache": max(mcache, 4)}
        if query_type in {"long_term_summary", "temporal_reasoning", "temporal_localization", "entity_tracking"}:
            return {"M_cur": max(1, mcur), "M_st": max(mst, 6), "M_lt": max(mlt, 10), "M_cache": max(mcache, 3)}
        return {"M_cur": max(1, mcur), "M_st": mst, "M_lt": mlt, "M_cache": mcache}

    def _final_evidence_budget(self, *, query_type: str, requested: int) -> int:
        if query_type == "current_perception":
            return min(max(1, requested), _env_int("EM2MEM_CURRENT_FINAL_EVIDENCE_K", 4))
        default_cap = _env_int("EM2MEM_UNIFIED_FINAL_EVIDENCE_K", 10)
        if query_type == "temporal_count":
            default_cap = _env_int("EM2MEM_TEMPORAL_COUNT_FINAL_EVIDENCE_K", 12)
        elif query_type in {"recent_recall", "followup"}:
            default_cap = _env_int("EM2MEM_RECENT_FINAL_EVIDENCE_K", 10)
        elif query_type == "long_term_summary":
            default_cap = _env_int("EM2MEM_SUMMARY_FINAL_EVIDENCE_K", 10)
        return max(requested, default_cap)
