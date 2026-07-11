from __future__ import annotations

import os
import re
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _contains_any(text: str, keywords: list[str]) -> bool:
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


def parse_auto_bool(value: Any, default: str = "auto") -> bool | str:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"auto", ""}:
        return "auto"
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


class QueryRouter:
    """Lightweight rule-based router for Stage 4A.

    This router only chooses retrieval/evidence policy. It intentionally does
    not implement M_cur/M_st/M_cache yet; those fields stay as placeholders.
    """

    SUMMARY_KWS = [
        "主要发生了什么", "总结", "概括", "整体", "这个视频讲了什么", "发生了什么", "主要内容", "总结一下",
        "到目前为止", "目前为止", "到现在", "从开始", "从头", "从头到尾", "全过程", "整个视频", "全程",
        "summarize", "summary", "recap", "overall", "overview", "brief summary", "briefly summarize",
        "overall summary", "full summary", "main content", "main points", "tell me the main points",
        "give me an overview", "what happened", "what has happened so far", "what happened so far",
        "what is this video about", "what is going on in the video", "describe the video",
        "describe what happened", "from beginning to end", "entire video", "whole video",
        "so far", "up to now", "from the beginning",
    ]
    VISUAL_KWS = [
        "颜色", "形状", "外观", "长什么样", "穿什么", "拿的是什么", "手里是什么", "左边", "右边",
        "位置", "有没有红色", "有没有黑色", "画面里", "屏幕", "显示", "看起来",
        "这是什么", "这个是什么", "这是啥", "这是什么东西", "看到什么", "看到了什么", "能看到什么", "当前场景",
        "color", "shape", "appearance", "look like", "wearing", "holding", "left", "right",
        "position", "visible", "visual", "image", "frame", "scene", "view", "in view", "in the frame",
        "screen", "display", "showing", "shown", "displayed", "on screen", "on the screen",
        "what do you see", "what can you see", "what is visible", "what can be seen",
        "what is shown", "what is on screen", "what's on screen", "what is on the screen",
        "what's on the screen", "what is this thing", "what is this object", "what is that thing",
        "what is that object", "what am i looking at", "describe the scene", "text on screen",
        "read the screen", "read this",
    ]
    ENTITY_KWS = [
        "谁", "哪个人", "这个人", "那个东西", "物体", "手机", "杯子", "包", "车", "在哪里",
        "去哪了", "放在哪里", "拿走", "移动", "拿起",
        "who", "which person", "person", "this person", "that person", "object", "thing", "item",
        "phone", "cup", "bag", "car", "where", "where is", "where did", "where did it go",
        "where was it placed", "moved", "moved to", "placed", "put", "put down", "picked up",
        "picked", "grabbed", "held", "holding", "carried", "took", "dropped", "left", "entered",
        "appeared", "disappeared",
    ]
    TEMPORAL_LOCALIZATION_KWS = [
        "什么时候", "几秒", "哪一段", "在哪个时间", "之前", "之后", "先后",
        "\u4eca\u5929", "\u6628\u5929", "\u524d\u5929", "\u4e0a\u5348", "\u4e2d\u5348", "\u4e0b\u5348", "\u665a\u4e0a", "\u65e9\u4e0a", "\u51cc\u6668", "\u591c\u91cc",
        "when", "when did", "what time", "at what time", "at what moment", "which moment",
        "which part", "which segment", "during which part", "timestamp", "time range", "seconds",
        "where in the video", "before", "after", "earlier", "later", "today", "yesterday",
        "the day before yesterday", "this morning", "this afternoon", "tonight", "this evening",
    ]
    TEMPORAL_REASONING_KWS = [
        "为什么", "怎么", "先后关系", "后来", "之前发生了什么", "原因",
        "why", "why did", "how", "how did", "reason", "what caused", "because", "cause",
        "sequence", "order", "what happened before", "what happened after", "what happened next",
        "then what", "next",
    ]
    CURRENT_KWS = [
        "现在", "当前", "此刻", "正在", "目前", "眼下", "实时", "此时", "当下",
        "现在画面", "画面里现在", "当前画面", "正在发生", "正在做", "现在在做",
        "这是什么", "这个是什么", "这是啥", "这是什么东西", "看到什么", "看到了什么", "能看到什么",
        "now", "currently", "at this moment", "right now", "current frame", "in the current frame",
        "at the moment", "at present", "current scene", "current view", "current screen", "current image",
        "in the current scene", "in the current view", "in front of me", "live", "realtime", "real-time",
        "what is happening now", "what's happening now", "what is going on now", "what's going on now",
        "what do you see", "what can you see", "what is in the current scene",
        "what is in the current view", "what is in front of me", "what am i seeing",
        "what am i looking at", "what is on screen", "what's on screen", "what is on the screen",
        "what's on the screen", "what is visible now", "what can you see now", "describe current scene",
        "describe the current scene", "read the current screen", "what is this?", "what's this?",
        "what is that?", "what's that?",
    ]
    RECENT_KWS = [
        "刚才", "刚刚", "上一段", "最近",
        "just now", "recently", "a moment ago", "a few seconds ago", "moments ago",
        "a minute ago", "a little while ago", "earlier just now", "shortly before",
        "right before", "previous moment", "previously", "last segment", "last scene",
        "recent scene", "what just happened",
    ]
    COUNT_KWS = [
        "一共", "总共", "总计", "总共有", "一共有", "多少", "几个", "几次", "几幅", "几张", "数量",
        "count", "how many", "total", "in total", "altogether",
    ]
    SPAN_KWS = [
        "从刚才到现在", "从刚刚到现在", "刚才到现在", "刚刚到现在", "到现在", "到目前为止",
        "since just now", "from just now", "so far", "up to now",
    ]
    LONG_TERM_SCOPE_KWS = [
        "到目前为止", "目前为止", "到现在", "从开始", "从头", "从头到尾", "全过程", "整个视频", "全程",
        "so far", "up to now", "from the beginning", "from beginning to end", "entire video", "whole video",
    ]

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or os.getenv("WORLDMM_QUERY_ROUTER_BACKEND", "rule")
        self.default_retrieval_mode = os.getenv("WORLDMM_QUERY_ROUTER_DEFAULT_RETRIEVAL_MODE", "auto")
        self.default_use_image = os.getenv("WORLDMM_QUERY_ROUTER_DEFAULT_USE_IMAGE", "auto")
        self.max_image_evidence = _env_int("WORLDMM_QUERY_ROUTER_MAX_IMAGE_EVIDENCE", 3)
        self.text_evidence_k = _env_int("WORLDMM_QUERY_ROUTER_TEXT_EVIDENCE_K", 5)
        self.final_evidence_k = _env_int("WORLDMM_QUERY_ROUTER_FINAL_EVIDENCE_K", 4)
        self.evidence_frames_k = _env_int("WORLDMM_QUERY_ROUTER_EVIDENCE_FRAMES_K", 5)
        self.visual_top_k = _env_int("WORLDMM_QUERY_ROUTER_VISUAL_TOP_K", 8)

    def route(
        self,
        question: str,
        request_options: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_options = request_options or {}
        session_context = session_context or {}
        cache_context = cache_context or {}
        q = (question or "").strip().lower()
        warnings: list[str] = []

        has_visual = _contains_any(q, self.VISUAL_KWS)
        has_entity = _contains_any(q, self.ENTITY_KWS)
        has_summary = _contains_any(q, self.SUMMARY_KWS)
        has_temporal_loc = _contains_any(q, self.TEMPORAL_LOCALIZATION_KWS)
        has_temporal_reason = _contains_any(q, self.TEMPORAL_REASONING_KWS)
        has_current = _contains_any(q, self.CURRENT_KWS)
        has_recent = _contains_any(q, self.RECENT_KWS)
        has_count = _contains_any(q, self.COUNT_KWS)
        has_span = _contains_any(q, self.SPAN_KWS) or (has_recent and has_current)
        is_span_count = bool(has_count and (has_span or has_recent))
        has_long_term_scope = _contains_any(q, self.LONG_TERM_SCOPE_KWS)
        is_long_term_summary = bool(has_summary and ((has_long_term_scope and not has_recent) or not (has_current or has_recent)))

        query_type = "general_qa"
        retrieval_mode = "hybrid"
        use_image_evidence: bool = True
        confidence = 0.55
        reason = "Default hybrid retrieval for general question."

        if is_span_count:
            query_type = "temporal_count"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = 0.78
            reason = "Question asks for a count across a recent-to-current time span; using short-term and long-term event memory."
        elif is_long_term_summary:
            query_type = "long_term_summary"
            retrieval_mode = "text_only"
            use_image_evidence = True
            confidence = 0.8
            reason = "Summary/span question is best served by text episodic/semantic memories."
        elif has_recent:
            query_type = "recent_recall"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = 0.75
            reason = "Question asks about recent recall; using short-term micro-events when available."
        elif has_current:
            query_type = "current_perception"
            retrieval_mode = "current"
            use_image_evidence = True
            confidence = 0.75
            reason = "Question asks about current perception; using current rolling memory."
        elif has_visual:
            query_type = "visual_attribute"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = 0.85
            reason = "Visual attribute/location/screen question benefits from visual retrieval and image evidence."
        elif has_temporal_loc:
            query_type = "temporal_localization"
            retrieval_mode = "hybrid" if has_entity else "text_only"
            use_image_evidence = True
            confidence = 0.75
            reason = "Temporal localization should preserve timestamps; images are usually not needed."
        elif has_temporal_reason:
            query_type = "temporal_reasoning"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = 0.7
            reason = "Temporal reasoning benefits from text and visual event alignment."
        elif has_entity:
            query_type = "entity_tracking"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = 0.75
            reason = "Entity/object tracking benefits from hybrid retrieval."
        elif has_summary:
            query_type = "long_term_summary"
            retrieval_mode = "text_only"
            use_image_evidence = True
            confidence = 0.8
            reason = "Summary question is best served by text episodic/semantic memories."

        memory_route = {
            "use_current": False,
            "use_short_term": False,
            "use_long_term": True,
            "use_interaction_cache": False,
        }
        short_term_reason = ""

        if cache_context.get("is_followup"):
            memory_route["use_interaction_cache"] = True
            memory_route["use_short_term"] = True
            if has_temporal_reason or has_recent or _contains_any(q, ["后来呢", "之后呢", "然后呢", "then", "later", "after that"]):
                query_type = "temporal_reasoning"
            elif cache_context.get("referenced_entities"):
                query_type = "entity_tracking"
            else:
                query_type = "recent_recall"
            retrieval_mode = "hybrid"
            use_image_evidence = True
            confidence = max(confidence, float(cache_context.get("confidence") or 0.0))
            reason = "Follow-up question detected; using interaction cache context with hybrid retrieval."
            short_term_reason = "follow-up query benefits from short-term micro-events"
            warnings.append("M_cache follow-up context is used as soft retrieval guidance")

        if query_type == "current_perception":
            memory_route["use_current"] = True
            memory_route["use_short_term"] = False
            memory_route["use_long_term"] = False
            memory_route["use_interaction_cache"] = bool(request_options.get("use_interaction_cache", True))
        if query_type == "temporal_count":
            memory_route["use_current"] = has_current
            memory_route["use_short_term"] = True
            memory_route["use_long_term"] = True
            memory_route["use_interaction_cache"] = bool(request_options.get("use_interaction_cache", True))
            short_term_reason = "count-over-time query should aggregate short-term micro-events and long-term event memory"
        if query_type in {"recent_recall"}:
            memory_route["use_short_term"] = True
            short_term_reason = "recent/current query should consult short-term micro-events"
        if query_type == "temporal_reasoning" and _contains_any(q, ["后来", "之后", "after that", "later", "then"]):
            memory_route["use_short_term"] = True
            short_term_reason = "later/after query should consult short-term micro-events"
        if query_type == "entity_tracking" and cache_context.get("is_followup"):
            memory_route["use_short_term"] = True
            short_term_reason = "follow-up entity tracking should consult short-term micro-events"

        short_term_ready = bool(session_context.get("short_term_ready", False))
        current_ready = bool(session_context.get("current_ready", False))
        current_stale = bool(session_context.get("current_stale", True))
        long_term_ready = bool(session_context.get("long_term_ready", True))
        if query_type == "current_perception" and (not current_ready or current_stale):
            memory_route["use_current"] = True
            memory_route["use_short_term"] = True
            memory_route["use_long_term"] = False
            short_term_reason = "current memory is stale or unavailable; falling back to short-term memory"
            warnings.append("current memory is stale or unavailable; falling back to short-term memory")
        if not long_term_ready and short_term_ready:
            memory_route["use_long_term"] = False
            if query_type in {"recent_recall", "current_perception", "entity_tracking", "temporal_reasoning", "temporal_count"}:
                memory_route["use_short_term"] = True
                short_term_reason = short_term_reason or "long-term memory is not ready; using short-term memory"

        requested_mode = str(request_options.get("retrieval_mode") or self.default_retrieval_mode or "auto").strip().lower()
        retrieval_mode_source = "auto"
        if requested_mode in {"text_only", "visual_only", "hybrid", "current"}:
            retrieval_mode = requested_mode
            retrieval_mode_source = "user_override"
            if requested_mode != "current" and query_type == "current_perception":
                memory_route["use_current"] = False
                memory_route["use_short_term"] = requested_mode in {"hybrid", "visual_only"}
                memory_route["use_long_term"] = True
        elif requested_mode not in {"", "auto"}:
            warnings.append(f"unsupported retrieval_mode={requested_mode}; using auto decision")

        requested_image = parse_auto_bool(request_options.get("use_image_evidence", self.default_use_image))
        use_image_evidence_source = "auto"
        if isinstance(requested_image, bool):
            use_image_evidence = requested_image
            use_image_evidence_source = "user_override"

        max_image_evidence = request_options.get("max_image_evidence")
        try:
            max_image_evidence = int(max_image_evidence) if max_image_evidence is not None else self.max_image_evidence
        except Exception:
            max_image_evidence = self.max_image_evidence
        max_image_evidence = max(0, min(max_image_evidence, self.max_image_evidence))

        text_top_k = int(request_options.get("text_top_k") or request_options.get("top_k") or self.text_evidence_k)
        final_evidence_k = int(request_options.get("final_evidence_k") or self.final_evidence_k)
        visual_top_k = int(request_options.get("visual_top_k") or self.visual_top_k)

        if retrieval_mode in {"hybrid", "visual_only"} and not session_context.get("visual_ready", False):
            warnings.append("visual index is not ready; query engine will fallback to text-only where needed")

        return {
            "query_type": query_type,
            "memory_route": memory_route,
            "retrieval_mode": retrieval_mode,
            "retrieval_mode_source": retrieval_mode_source,
            "use_image_evidence": bool(use_image_evidence),
            "use_image_evidence_source": use_image_evidence_source,
            "max_image_evidence": max_image_evidence,
            "text_top_k": max(1, text_top_k),
            "visual_top_k": max(1, visual_top_k),
            "final_evidence_k": max(1, final_evidence_k),
            "evidence_frames_k": max(1, self.evidence_frames_k),
            "confidence": confidence,
            "reason": reason,
            "warnings": warnings,
            "router_backend": self.backend,
            "short_term_policy": {
                "enabled": bool(memory_route["use_short_term"]),
                "ready": short_term_ready,
                "reason": short_term_reason,
                "top_k": max(1, text_top_k),
            },
            "current_policy": {
                "enabled": bool(memory_route["use_current"]),
                "ready": current_ready,
                "stale": current_stale,
                "reason": reason if query_type == "current_perception" else "",
                "top_k": 1,
            },
            "cache_context": {
                "cache_hit": bool(cache_context.get("cache_hit")),
                "is_followup": bool(cache_context.get("is_followup")),
                "referenced_entities": cache_context.get("referenced_entities", []),
                "referenced_time_ranges": cache_context.get("referenced_time_ranges", []),
                "referenced_memory_ids": cache_context.get("referenced_memory_ids", []),
                "referenced_segment_ids": cache_context.get("referenced_segment_ids", []),
                "confidence": cache_context.get("confidence", 0.0),
                "reason": cache_context.get("reason", ""),
            },
        }
