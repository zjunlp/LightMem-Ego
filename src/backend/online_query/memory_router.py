from __future__ import annotations

from typing import Any

from .router_schema import contains_any, parse_optional_bool, safe_float, time_overlap


class MemoryRouter:
    """Stage 7 level-1 router: choose memory sources before retrieval mode."""

    CURRENT_KWS = [
        "现在", "当前", "此刻", "正在", "目前", "当下", "眼下", "实时", "此时", "眼前",
        "现在画面", "当前画面", "画面里现在", "此刻画面", "正在发生", "正在做", "现在在做",
        "这是什么", "这个是什么", "这是啥", "这是什么东西", "看到什么", "看到了什么", "能看到什么", "当前场景",
        "now", "currently", "right now", "at this moment", "at the moment", "at present",
        "current frame", "in the current frame", "current scene", "current view", "current screen",
        "current image", "in the current scene", "in the current view", "in front of me",
        "live", "realtime", "real-time", "what is happening now", "what's happening now",
        "what is going on now", "what's going on now", "what do you see", "what can you see",
        "what is in the current scene", "what is in the current view", "what is in front of me",
        "what am i seeing", "what am i looking at", "what is on screen", "what's on screen",
        "what is on the screen", "what's on the screen", "what is visible now",
        "what can you see now", "describe current scene", "describe the current scene",
        "read the current screen", "what is this?", "what's this?", "what is that?", "what's that?",
    ]
    RECENT_KWS = [
        "刚才", "刚刚", "最近", "上一段", "前面一点", "刚发生", "不久前", "前一会", "前面",
        "just now", "recently", "a moment ago", "a few seconds ago", "moments ago", "a minute ago",
        "a little while ago", "earlier just now", "shortly before", "right before", "earlier",
        "previous moment", "previously", "a little earlier", "last segment", "last scene",
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
    SUMMARY_ACTION_KWS = [
        "发生了什么", "发生过什么", "出现了什么", "看到了什么", "看到过什么", "有哪些", "总结", "概括",
        "what happened", "what has happened", "what did we see", "what appeared", "summarize", "summary",
    ]
    FOLLOWUP_KWS = [
        "他", "她", "它", "这个", "那个", "这些", "那些", "这个东西", "那个东西", "那个人",
        "那个物体", "刚才那个", "那后来呢", "之后呢", "然后呢", "接下来呢", "后来呢",
        "he", "she", "it", "they", "this", "that", "those", "the object", "the person",
        "the thing", "that object", "that person", "the previous one", "then", "later",
        "after that", "what about it", "what about that", "where did it go",
    ]
    SUMMARY_KWS = [
        "总结", "概括", "整体", "整个视频", "全过程", "从头到尾", "主要发生了什么",
        "这个视频讲了什么", "主要内容", "总结一下", "总体", "到目前为止", "目前为止",
        "summary", "summarize", "recap", "overall", "overview", "brief summary", "briefly summarize",
        "overall summary", "full summary", "entire video", "whole video", "from beginning to end",
        "main content", "main points", "tell me the main points", "give me an overview",
        "what happened in the video", "what has happened so far", "what happened so far",
        "what is this video about", "what is going on in the video", "describe the video",
        "describe what happened",
    ]
    VISUAL_KWS = [
        "颜色", "外观", "长什么样", "穿什么", "穿着", "手里", "拿着", "拿的", "左边", "右边",
        "画面里", "可见", "看起来", "形状", "位置", "红色", "黑色", "蓝色", "绿色",
        "color", "appearance", "look like", "wearing", "holding", "left", "right", "visible",
        "visual", "image", "frame", "scene", "view", "in view", "in the frame", "shape",
        "position", "screen", "display", "showing", "shown", "displayed", "on screen",
        "on the screen", "what do you see", "what can you see", "what is visible",
        "what can be seen", "what is shown", "what is on screen", "what's on screen",
        "what is on the screen", "what's on the screen", "what is this thing",
        "what is this object", "what is that thing", "what is that object",
        "what am i looking at", "describe the scene", "text on screen", "read the screen",
        "read this", "red", "black", "blue", "green",
    ]
    TEMPORAL_KWS = [
        "什么时候", "哪一段", "几秒", "在哪个时间", "之前", "之后", "后来", "先后", "原因", "为什么",
        "怎么", "接下来", "然后",
        "\u4eca\u5929", "\u6628\u5929", "\u524d\u5929", "\u4e0a\u5348", "\u4e2d\u5348", "\u4e0b\u5348", "\u665a\u4e0a", "\u65e9\u4e0a", "\u51cc\u6668", "\u591c\u91cc",
        "when", "when did", "what time", "at what time", "at what moment", "which moment",
        "which part", "which segment", "during which part", "timestamp", "time range", "seconds",
        "where in the video", "before", "after", "earlier", "later", "why", "why did",
        "how", "how did", "reason", "what caused", "because", "cause", "sequence", "order",
        "what happened before", "what happened after", "what happened next", "then", "then what", "next",
        "today", "yesterday", "the day before yesterday", "this morning", "this afternoon", "tonight", "this evening",
    ]
    SPEECH_KWS = [
        "他说了什么", "她说了什么", "说了什么", "说话", "语音", "对话", "谈论", "提到", "讲话",
        "say", "said", "what did he say", "what did she say", "what did they say",
        "what was said", "speaking", "speech", "conversation", "talked about", "mentioned",
        "transcript", "audio", "voice",
    ]
    ENTITY_KWS = [
        "谁", "哪个人", "这个人", "那个人", "物体", "东西", "手机", "杯子", "包", "车", "在哪里",
        "去哪了", "放在哪里", "拿走", "移动", "拿起",
        "who", "which person", "person", "this person", "that person", "object", "thing", "item",
        "phone", "cup", "bag", "car", "where", "where is", "where did", "where did it go",
        "where was it placed", "moved", "moved to", "placed", "put", "put down", "picked up",
        "picked", "grabbed", "held", "holding", "carried", "took", "dropped", "left", "entered",
        "appeared", "disappeared",
    ]

    def route(
        self,
        question: str,
        request_options: dict[str, Any] | None,
        runtime_state: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_options = request_options or {}
        cache_context = cache_context or {}
        q = (question or "").strip().lower()
        warnings: list[str] = []

        has_current = contains_any(q, self.CURRENT_KWS)
        has_recent = contains_any(q, self.RECENT_KWS)
        has_followup = bool(cache_context.get("is_followup")) or contains_any(q, self.FOLLOWUP_KWS)
        has_long_term_scope = contains_any(q, self.LONG_TERM_SCOPE_KWS)
        has_summary = contains_any(q, self.SUMMARY_KWS) or (has_long_term_scope and contains_any(q, self.SUMMARY_ACTION_KWS))
        has_visual = contains_any(q, self.VISUAL_KWS)
        has_temporal = contains_any(q, self.TEMPORAL_KWS)
        has_speech = contains_any(q, self.SPEECH_KWS)
        has_entity = contains_any(q, self.ENTITY_KWS)
        has_count = contains_any(q, self.COUNT_KWS)
        has_span = contains_any(q, self.SPAN_KWS) or (has_recent and has_current)
        is_span_count = bool(has_count and (has_span or has_recent))

        query_type = "general_qa"
        if is_span_count:
            query_type = "temporal_count"
        elif has_summary and not has_recent:
            query_type = "long_term_summary"
        elif has_recent:
            query_type = "recent_recall"
        elif has_speech:
            query_type = "speech_query"
        elif has_temporal:
            query_type = "temporal_reasoning" if contains_any(q, ["为什么", "原因", "why", "how", "后来", "之后", "after", "later"]) else "temporal_localization"
        elif has_followup:
            query_type = "followup"
        elif has_entity:
            query_type = "entity_tracking"
        elif has_current:
            query_type = "current_perception"
        elif has_visual:
            query_type = "visual_attribute"

        route = {
            "use_current": False,
            "use_short_term": False,
            "use_long_term": True,
            "use_interaction_cache": bool(request_options.get("use_interaction_cache", True)),
        }
        fallback_order = ["M_lt"]
        priority = {"M_cur": 0.0, "M_st": 0.0, "M_lt": 1.0, "M_cache": 0.3 if route["use_interaction_cache"] else 0.0}
        reason = "General query defaults to long-term memory."

        if query_type == "current_perception":
            route.update({"use_current": True, "use_short_term": False, "use_long_term": False, "use_interaction_cache": True})
            fallback_order = ["M_cur", "M_st", "M_lt"]
            priority.update({"M_cur": 1.0, "M_st": 0.45, "M_lt": 0.1, "M_cache": 0.3})
            reason = "Current-perception query should use the current rolling memory first."
        elif query_type == "recent_recall":
            route.update({"use_current": True, "use_short_term": True, "use_long_term": True, "use_interaction_cache": True})
            fallback_order = ["M_st", "M_lt", "M_cur"]
            priority.update({"M_cur": 0.35, "M_st": 0.9, "M_lt": 0.8, "M_cache": 0.35})
            reason = "Recent-recall query should use short-term events and long-term event memory; current memory is supplemental."
        elif query_type == "temporal_count":
            route.update({"use_current": has_current, "use_short_term": True, "use_long_term": True, "use_interaction_cache": True})
            fallback_order = ["M_st", "M_lt", "M_cur"]
            priority.update({"M_cur": 0.45 if has_current else 0.1, "M_st": 0.95, "M_lt": 0.85, "M_cache": 0.35})
            reason = "Count-over-time query should aggregate short-term micro-events and long-term event memory instead of using current memory only."
        elif query_type == "followup":
            route.update({"use_current": False, "use_short_term": True, "use_long_term": True, "use_interaction_cache": True})
            fallback_order = ["M_cache", "M_st", "M_lt", "M_cur"]
            priority.update({"M_cur": 0.35, "M_st": 0.75, "M_lt": 0.65, "M_cache": 0.8})
            reason = "Follow-up/coreference query uses interaction cache plus short/long-term context."
            self._apply_followup_time_routing(route, priority, runtime_state, cache_context)
        elif query_type == "long_term_summary":
            route.update({"use_current": False, "use_short_term": False, "use_long_term": True})
            fallback_order = ["M_lt"]
            priority.update({"M_cur": 0.0, "M_st": 0.15, "M_lt": 1.0, "M_cache": 0.1})
            reason = "Summary query should use long-term episodic/semantic memories."
        elif query_type == "visual_attribute":
            route.update({"use_current": has_current, "use_short_term": has_recent, "use_long_term": not (has_current or has_recent)})
            route["use_interaction_cache"] = True
            fallback_order = ["M_cur", "M_st", "M_lt"] if has_current or has_recent else ["M_lt", "M_st"]
            priority.update({"M_cur": 0.9 if has_current else 0.2, "M_st": 0.8 if has_recent else 0.25, "M_lt": 0.85 if not (has_current or has_recent) else 0.3, "M_cache": 0.35})
            reason = "Visual attribute query routes by temporal scope."
        elif query_type in {"temporal_localization", "temporal_reasoning"}:
            route.update({"use_current": has_current, "use_short_term": has_recent or has_followup, "use_long_term": not has_recent or has_summary or not has_current, "use_interaction_cache": True})
            fallback_order = ["M_st", "M_lt", "M_cur"] if has_recent or has_followup else ["M_lt", "M_st"]
            priority.update({"M_cur": 0.65 if has_current else 0.1, "M_st": 0.7 if (has_recent or has_followup) else 0.35, "M_lt": 0.85, "M_cache": 0.45})
            reason = "Temporal query needs timestamps and may span short and long-term memory."
        elif query_type == "speech_query":
            route.update({"use_current": has_current, "use_short_term": has_recent, "use_long_term": not (has_current or has_recent), "use_interaction_cache": True})
            fallback_order = ["M_cur", "M_st", "M_lt"] if has_current or has_recent else ["M_lt", "M_st"]
            priority.update({"M_cur": 0.85 if has_current else 0.15, "M_st": 0.8 if has_recent else 0.3, "M_lt": 0.8 if not (has_current or has_recent) else 0.35, "M_cache": 0.35})
            reason = "Speech query routes to the transcript-bearing memory for its temporal scope."
        elif query_type == "entity_tracking":
            route.update({"use_current": has_current, "use_short_term": has_recent or has_followup, "use_long_term": True, "use_interaction_cache": True})
            fallback_order = ["M_st", "M_lt", "M_cur"] if has_recent or has_followup else ["M_lt", "M_st"]
            priority.update({"M_cur": 0.55 if has_current else 0.1, "M_st": 0.7 if (has_recent or has_followup) else 0.4, "M_lt": 0.8, "M_cache": 0.45})
            reason = "Entity tracking may need cache references and long-term state history."

        self._apply_readiness(route, priority, fallback_order, runtime_state, query_type, warnings)
        self._apply_user_overrides(route, request_options, warnings)
        requested_mode = str(request_options.get("retrieval_mode") or "auto").strip().lower()
        if query_type == "long_term_summary" and requested_mode != "current":
            if route.get("use_current"):
                warnings.append("ignored use_current for long-term summary unless retrieval_mode=current")
            route["use_current"] = False
            route["use_short_term"] = False
            route["use_long_term"] = True

        return {
            "query_type": query_type,
            "memory_route": route,
            "fallback_order": fallback_order,
            "memory_priority": priority,
            "reason": reason,
            "warnings": warnings,
            "runtime_state": runtime_state,
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

    def _apply_followup_time_routing(
        self,
        route: dict[str, bool],
        priority: dict[str, float],
        runtime_state: dict[str, Any],
        cache_context: dict[str, Any],
    ) -> None:
        ranges = cache_context.get("referenced_time_ranges", []) or []
        if not ranges:
            return
        mcur = runtime_state.get("mcur_time_span")
        mst = runtime_state.get("mst_time_span")
        matched = False
        for item in ranges:
            start = item.get("start") if isinstance(item, dict) else None
            end = item.get("end") if isinstance(item, dict) else start
            if mcur and time_overlap(mcur[0], mcur[1], start, end) > 0:
                route["use_current"] = True
                priority["M_cur"] = max(priority["M_cur"], 0.75)
                matched = True
            if mst and time_overlap(mst[0], mst[1], start, end) > 0:
                route["use_short_term"] = True
                priority["M_st"] = max(priority["M_st"], 0.8)
                matched = True
        if not matched:
            route["use_long_term"] = True
            priority["M_lt"] = max(priority["M_lt"], 0.75)

    def _apply_readiness(
        self,
        route: dict[str, bool],
        priority: dict[str, float],
        fallback_order: list[str],
        runtime_state: dict[str, Any],
        query_type: str,
        warnings: list[str],
    ) -> None:
        if route.get("use_current") and (not runtime_state.get("current_ready") or runtime_state.get("current_stale")):
            route["use_current"] = False
            if runtime_state.get("short_term_ready"):
                route["use_short_term"] = True
                priority["M_st"] = max(priority.get("M_st", 0.0), 0.75)
            elif runtime_state.get("long_term_ready"):
                route["use_long_term"] = True
            warnings.append("current memory unavailable or stale; falling back to short-term memory")
        if route.get("use_short_term") and not runtime_state.get("short_term_ready"):
            route["use_short_term"] = False
            if runtime_state.get("long_term_ready"):
                route["use_long_term"] = True
            warnings.append("short-term memory unavailable; falling back to long-term memory")
        if route.get("use_long_term") and not runtime_state.get("long_term_ready"):
            route["use_long_term"] = False
            if runtime_state.get("short_term_ready") and query_type not in {"long_term_summary"}:
                route["use_short_term"] = True
                warnings.append("long-term memory unavailable; using short-term memory")
            elif runtime_state.get("current_ready") and not runtime_state.get("current_stale") and query_type in {"current_perception", "recent_recall"}:
                route["use_current"] = True
                warnings.append("long-term memory unavailable; using current memory")
        if not any(route.get(key) for key in ("use_current", "use_short_term", "use_long_term")):
            if runtime_state.get("long_term_ready"):
                route["use_long_term"] = True
                fallback_order[:] = ["M_lt"]
            elif runtime_state.get("short_term_ready"):
                route["use_short_term"] = True
                fallback_order[:] = ["M_st"]
            elif runtime_state.get("current_ready") and not runtime_state.get("current_stale"):
                route["use_current"] = True
                fallback_order[:] = ["M_cur"]

    def _apply_user_overrides(
        self,
        route: dict[str, bool],
        request_options: dict[str, Any],
        warnings: list[str],
    ) -> None:
        override_map = {
            "use_current": "use_current",
            "use_short_term": "use_short_term",
            "use_long_term": "use_long_term",
            "use_interaction_cache": "use_interaction_cache",
        }
        for opt_key, route_key in override_map.items():
            parsed = parse_optional_bool(request_options.get(opt_key))
            if parsed is not None:
                route[route_key] = parsed
                if route_key != "use_interaction_cache":
                    warnings.append(f"{route_key} set by user_override")
        if not any(route.get(key) for key in ("use_current", "use_short_term", "use_long_term")):
            route["use_long_term"] = True
            warnings.append("all memory sources disabled by overrides; falling back to long-term memory")
