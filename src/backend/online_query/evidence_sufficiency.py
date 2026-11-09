from __future__ import annotations

import json
import math
import os
import re
from typing import Any


_RECENT_PHRASES = (
    "刚才", "刚刚", "最近", "上一段", "前面一点", "刚发生", "不久前", "前一会",
    "just now", "recently", "a moment ago", "a few seconds ago", "moments ago",
    "a minute ago", "a little while ago", "shortly before", "right before",
    "previous moment", "last segment", "last scene", "recent scene",
)
_AGGREGATE_PHRASES = (
    "一共", "总共", "几次", "多少次", "从开始", "从头", "全过程", "整个视频",
    "到目前为止", "总结", "概括", "通常", "经常", "习惯", "偏好", "第一次",
    "count", "how many", "in total", "from the beginning", "entire video", "whole video",
    "summarize", "summary", "usually", "often", "habit", "prefer", "first time",
)
_SPEECH_PHRASES = (
    "说了什么", "说什么", "谁说", "讲话", "对话", "谈论", "提到", "语音",
    "what did he say", "what did she say", "what did they say", "what did i say",
    "what was said", "who said", "say", "said", "speech", "conversation",
    "talked about", "mentioned",
)
_LOCATION_PHRASES = (
    "在哪里", "在哪", "哪里", "哪儿", "放哪", "位置", "去哪", "where", "location",
)
_PLACEMENT_PHRASES = (
    "放", "搁", "留下", "put", "place", "placed", "leave", "left", "drop", "dropped",
)
_ACTOR_PHRASES = ("谁", "哪个人", "who", "which person")
_VISUAL_ATTRIBUTE_PHRASES = (
    "颜色", "什么颜色", "外观", "长什么样", "形状", "穿什么", "拿的是什么", "手里是什么",
    "color", "appearance", "look like", "shape", "wearing", "what was i holding",
    "what did i hold", "what was in my hand",
)
_GENERIC_RECENT_PHRASES = (
    "发生了什么", "刚才做了什么", "刚刚做了什么", "上一段是什么", "what happened",
    "what just happened", "what did i just do", "what was happening",
)
_ACTION_WORDS = {
    "放", "拿", "关", "开", "走", "进入", "离开", "吃", "喝", "说", "给", "穿", "脱",
    "put", "place", "placed", "leave", "left", "pick", "picked", "take", "took", "carry",
    "carried", "close", "closed", "open", "opened", "enter", "entered", "exit", "walk",
    "eat", "ate", "drink", "drank", "say", "said", "give", "gave", "wear", "wore",
}
_LOCATION_WORDS = (
    "桌", "台", "厨房", "卧室", "房间", "办公室", "地面", "地板", "架", "柜", "包里",
    "门口", "旁边", "上面", "下面", "里面", "外面", "左边", "右边",
)
_ENGLISH_LOCATION_RE = re.compile(
    r"\b(?:on|in|into|onto|at|inside|outside|beside|near|under|above|behind|by|next to|in front of)\b",
    re.IGNORECASE,
)
_QUERY_STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "we", "you", "did", "do", "does", "was", "were",
    "is", "are", "what", "which", "where", "who", "when", "how", "it", "this", "that", "just",
    "now", "recently", "before", "please", "tell", "happened", "happen",
    "我", "你", "他", "她", "它", "的", "了", "吗", "呢", "把", "被", "刚", "才", "刚才", "刚刚",
    "最近", "什么", "哪里", "哪儿", "在哪", "谁", "怎么", "如何", "请", "告诉", "发生",
}
_INTENT_TERMS = {
    "put", "place", "placed", "leave", "left", "drop", "dropped", "say", "said", "speech",
    "color", "appearance", "shape", "where", "location", "who",
    "放", "搁", "留下", "说", "讲话", "颜色", "外观", "形状", "位置",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lower = str(text or "").lower()
    return any(phrase.lower() in lower for phrase in phrases)


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _event_text(event: dict[str, Any]) -> str:
    fields = (
        "retrieval_text",
        "event_caption_refined",
        "event_caption_fast",
        "transcript",
        "entities",
        "visual_objects",
        "main_actions",
        "state_changes",
    )
    return " ".join(_flatten(event.get(field)) for field in fields if event.get(field)).strip()


def _tokens(text: str) -> set[str]:
    lower = str(text or "").lower()
    words = {
        word for word in re.findall(r"[a-z0-9_]+", lower)
        if len(word) > 1 and word not in _QUERY_STOPWORDS
    }
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lower)
    for run in chinese_runs:
        chars = [char for char in run if char not in _QUERY_STOPWORDS]
        words.update(char for char in chars if char not in _QUERY_STOPWORDS)
        words.update(
            "".join(chars[index : index + 2])
            for index in range(max(0, len(chars) - 1))
        )
    return {word for word in words if word and word not in _QUERY_STOPWORDS}


def _query_terms(question: str) -> set[str]:
    text = str(question or "").lower()
    for phrase in _RECENT_PHRASES:
        text = text.replace(phrase.lower(), " ")
    return _tokens(text)


def _query_coverage(question: str, evidence_text: str) -> float:
    query_terms = _query_terms(question)
    if not query_terms:
        return 1.0
    evidence_terms = _tokens(evidence_text)
    return len(query_terms & evidence_terms) / max(len(query_terms), 1)


def _target_coverage(question: str, evidence_text: str) -> float:
    target_terms = _query_terms(question) - _ACTION_WORDS - _INTENT_TERMS
    if not target_terms:
        return 1.0
    evidence_terms = _tokens(evidence_text)
    return len(target_terms & evidence_terms) / max(len(target_terms), 1)


def _has_location(text: str, event: dict[str, Any]) -> bool:
    lower = text.lower()
    if any(word in lower for word in _LOCATION_WORDS) or _ENGLISH_LOCATION_RE.search(lower):
        return True
    for change in event.get("state_changes", []) or []:
        if not isinstance(change, dict):
            continue
        if str(change.get("attribute") or "").lower() in {"location", "position", "place"} and change.get("after"):
            return True
    return False


def _has_placement(text: str, event: dict[str, Any]) -> bool:
    if _contains_any(text, _PLACEMENT_PHRASES):
        return True
    actions = _flatten(event.get("main_actions")).lower()
    changes = _flatten(event.get("state_changes")).lower()
    return _contains_any(actions + " " + changes, _PLACEMENT_PHRASES)


def _has_actor(event: dict[str, Any], text: str) -> bool:
    for action in event.get("main_actions", []) or []:
        if not isinstance(action, dict):
            continue
        actor = str(action.get("actor") or "").strip().lower()
        if actor and actor not in {"unknown", "unknown person", "unknown speaker", "person", "someone"}:
            return True
    entities = [str(item).strip().lower() for item in (event.get("entities", []) or [])]
    if any(item and item not in {"unknown person", "unknown speaker", "person", "someone"} for item in entities):
        return True
    return bool(re.search(r"\b(?:man|woman|boy|girl|doctor|teacher|alex|speaker)\b", text, re.IGNORECASE))


def _has_visual_grounding(event: dict[str, Any]) -> bool:
    return bool(
        event.get("keyframes")
        or event.get("evidence_frames")
        or event.get("visual_objects")
    )


def _substantive_text(event: dict[str, Any]) -> bool:
    refined = str(event.get("event_caption_refined") or "").strip()
    fast = str(event.get("event_caption_fast") or "").strip()
    transcript = str(event.get("transcript") or "").strip()
    text = refined or fast or transcript
    if len(text) < 4:
        return False
    lower = text.lower()
    placeholder_only = (
        "provisional" in lower and "event occurs" in lower
    ) or lower.startswith("audio transcript is available between")
    return not placeholder_only


def _evidence_quality(event: dict[str, Any]) -> float:
    refined = bool(str(event.get("event_caption_refined") or "").strip())
    fast = bool(str(event.get("event_caption_fast") or "").strip())
    transcript = bool(str(event.get("transcript") or "").strip())
    frames = _has_visual_grounding(event)
    if refined:
        base = 1.0
    elif fast and (transcript or frames):
        base = 0.80
    elif fast:
        base = 0.65
    elif transcript:
        base = 0.55
    elif frames:
        base = 0.45
    else:
        base = 0.15
    if event.get("needs_refine") or event.get("refined_stale"):
        base -= 0.15
    event_confidence = event.get("confidence")
    if event_confidence is not None:
        base = 0.8 * base + 0.2 * _clamp01(event_confidence)
    return _clamp01(base)


def _intent(question: str) -> str:
    lower = str(question or "").lower()
    if _contains_any(lower, _SPEECH_PHRASES):
        return "speech"
    if _contains_any(lower, _LOCATION_PHRASES):
        return "location"
    if _contains_any(lower, _ACTOR_PHRASES):
        return "actor"
    if _contains_any(lower, _VISUAL_ATTRIBUTE_PHRASES):
        return "visual_attribute"
    if _contains_any(lower, _GENERIC_RECENT_PHRASES):
        return "recent_event"
    return "event_detail"


def _predicate_signals(question: str, event: dict[str, Any], evidence_text: str) -> tuple[float, list[str]]:
    intent = _intent(question)
    missing: list[str] = []
    checks: list[bool] = []
    if intent == "speech":
        transcript = str(event.get("transcript") or "").strip()
        checks.append(len(transcript) >= 4)
        if not checks[-1]:
            missing.append("speech_content")
        if _contains_any(question, _ACTOR_PHRASES):
            checks.append(_has_actor(event, evidence_text))
            if not checks[-1]:
                missing.append("speaker")
    elif intent == "location":
        checks.append(_has_location(evidence_text, event))
        if not checks[-1]:
            missing.append("location")
        if _contains_any(question, _PLACEMENT_PHRASES):
            checks.append(_has_placement(evidence_text, event))
            if not checks[-1]:
                missing.append("placement_action")
    elif intent == "actor":
        checks.append(_has_actor(event, evidence_text))
        if not checks[-1]:
            missing.append("actor")
        action_terms = _query_terms(question) & _ACTION_WORDS
        if action_terms:
            checks.append(bool(action_terms & _tokens(evidence_text)))
            if not checks[-1]:
                missing.append("queried_action")
    elif intent == "visual_attribute":
        checks.append(_has_visual_grounding(event))
        if not checks[-1]:
            missing.append("visual_grounding")
        checks.append(_substantive_text(event))
        if not checks[-1]:
            missing.append("attribute_content")
    else:
        checks.append(_substantive_text(event))
        if not checks[-1]:
            missing.append("event_description")
        action_terms = _query_terms(question) & _ACTION_WORDS
        if action_terms:
            checks.append(bool(action_terms & _tokens(evidence_text)))
            if not checks[-1]:
                missing.append("queried_action")
    return sum(1.0 for check in checks if check) / max(len(checks), 1), missing


def _agreement(first: dict[str, Any], second: dict[str, Any] | None) -> float:
    if second is None:
        return 0.6
    first_terms = _tokens(_event_text(first))
    second_terms = _tokens(_event_text(second))
    if not first_terms or not second_terms:
        return 0.0
    return min(1.0, len(first_terms & second_terms) / max(1.0, min(len(first_terms), len(second_terms))))


class EvidenceSufficiencyEvaluator:
    """Conservative, LLM-free gate for recent-recall ST-first retrieval."""

    def evaluate(
        self,
        *,
        question: str,
        query_type: str,
        short_term_results: list[dict[str, Any]] | None,
        current_results: list[dict[str, Any]] | None = None,
        explicit_long_term_override: bool = False,
        query_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results = [item for item in (short_term_results or []) if isinstance(item, dict)]
        closed_end = max((_safe_float(item.get("end_time")) for item in results
                          if not item.get("synthetic") and item.get("status") not in {"open", "provisional"}), default=None)
        current_end = None
        frame_times = []
        for item in current_results or []:
            context = item.get("current_context") or {}
            if context.get("is_stale"):
                continue
            end = context.get("window_end_time")
            if end is not None:
                current_end = max(current_end if current_end is not None else float(end), float(end))
            for frame in (item.get("current_selection") or {}).get("evidence_frames", []) or []:
                timestamp = frame.get("timestamp", frame.get("capture_time_seconds"))
                if timestamp is not None:
                    frame_times.append(float(timestamp))
        from online_query_constraints import resolve_query_constraints, constraint_windows
        # The caller's original-question constraints are authoritative. Cache
        # hints may appear in rewritten search text but are not user windows.
        constraints = resolve_query_constraints(question, query_constraints=query_constraints, latest=current_end)
        windows = constraint_windows(constraints)
        tail_in_scope = not windows or any(span[1] > (closed_end if closed_end is not None else float("-inf")) for span in windows)
        historical_only = constraints.get("scope") == "history" and not constraints.get("requires_current")
        requires_tail = bool(str(query_type or "") == "recent_recall" and not historical_only and tail_in_scope and closed_end is not None
                             and current_end is not None and current_end > closed_end)
        base = {
            "requires_current_tail": requires_tail,
            "current_tail_coverage": {
                "closed_st_end": closed_end, "current_end": current_end,
                "tail_range": [closed_end, current_end] if requires_tail else None,
                "current_frame_times": sorted(set(t for t in frame_times if closed_end is None or t > closed_end)),
                "provenance": "unrefined_current_window",
            },
            "enabled": True,
            "evaluated": False,
            "sufficient": False,
            "confidence": 0.0,
            "decision": "escalate_lt",
            "reason": "not_evaluated",
            "intent": _intent(question),
            "missing_slots": [],
            "supporting_event_ids": [],
            "signals": {},
            "current_evidence_count": len(current_results or []),
            "short_term_evidence_count": len(results),
        }
        if str(query_type or "") != "recent_recall":
            return {**base, "enabled": False, "reason": "query_type_not_recent_recall"}
        if explicit_long_term_override:
            return {**base, "enabled": False, "reason": "explicit_long_term_override"}
        if _contains_any(question, _AGGREGATE_PHRASES):
            return {**base, "evaluated": True, "reason": "aggregate_or_long_scope_query"}
        if not results:
            return {**base, "evaluated": True, "reason": "no_short_term_evidence", "missing_slots": ["short_term_evidence"]}

        ranked = sorted(results, key=lambda item: -_safe_float(item.get("score")))
        top_two = ranked[:2]
        best: dict[str, Any] | None = None
        for rank, event in enumerate(ranked[:3]):
            text = _event_text(event)
            generic_recent = _intent(question) == "recent_event"
            retrieval = (
                0.45 * _clamp01(event.get("bm25_score"))
                + 0.35 * _clamp01(event.get("lexical_score"))
                + 0.20 * _clamp01(event.get("score"))
            )
            if generic_recent:
                retrieval = max(
                    retrieval,
                    0.50 * _clamp01(event.get("score")) + 0.50 * _clamp01(event.get("recency_score")),
                )
            coverage = 1.0 if generic_recent else max(
                _query_coverage(question, text),
                _clamp01(event.get("lexical_score")),
            )
            target_coverage = 1.0 if generic_recent else _target_coverage(question, text)
            predicate, missing = _predicate_signals(question, event, text)
            quality = _evidence_quality(event)
            temporal = _clamp01(event.get("recency_score"))
            second = top_two[1] if top_two and event is top_two[0] and len(top_two) > 1 else None
            agreement = _agreement(event, second)
            margin = max(0.0, _safe_float(event.get("score")) - _safe_float(second.get("score"))) if second else 0.12
            stability = max(agreement, min(1.0, margin / 0.12))
            confidence = (
                0.20 * retrieval
                + 0.20 * coverage
                + 0.25 * predicate
                + 0.15 * quality
                + 0.15 * temporal
                + 0.05 * stability
            )
            candidate = {
                "event": event,
                "rank": rank,
                "confidence": confidence,
                "missing": missing,
                "signals": {
                    "retrieval": round(retrieval, 4),
                    "query_coverage": round(coverage, 4),
                    "target_coverage": round(target_coverage, 4),
                    "predicate_coverage": round(predicate, 4),
                    "evidence_quality": round(quality, 4),
                    "temporal_fit": round(temporal, 4),
                    "stability": round(stability, 4),
                },
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate

        assert best is not None
        threshold = _env_float("EM2MEM_RECENT_ST_SUFFICIENCY_THRESHOLD", 0.72)
        min_retrieval = _env_float("EM2MEM_RECENT_ST_MIN_RETRIEVAL", 0.42)
        min_coverage = _env_float("EM2MEM_RECENT_ST_MIN_QUERY_COVERAGE", 0.50)
        min_target_coverage = _env_float("EM2MEM_RECENT_ST_MIN_TARGET_COVERAGE", 0.50)
        min_temporal = _env_float("EM2MEM_RECENT_ST_MIN_TEMPORAL_FIT", 0.55)
        generic_recent = _intent(question) == "recent_event"
        sufficient = bool(
            best["confidence"] >= threshold
            and not best["missing"]
            and best["signals"]["predicate_coverage"] >= 1.0
            and best["signals"]["retrieval"] >= (0.30 if generic_recent else min_retrieval)
            and best["signals"]["query_coverage"] >= (0.0 if generic_recent else min_coverage)
            and best["signals"]["target_coverage"] >= (0.0 if generic_recent else min_target_coverage)
            and best["signals"]["temporal_fit"] >= min_temporal
        )
        event_id = str(best["event"].get("event_id") or "")
        return {
            **base,
            "evaluated": True,
            "sufficient": sufficient,
            "confidence": round(best["confidence"], 4),
            "decision": "use_st" if sufficient else "escalate_lt",
            "reason": "short_term_evidence_sufficient" if sufficient else "short_term_evidence_below_threshold_or_incomplete",
            "missing_slots": list(best["missing"]),
            "supporting_event_ids": [event_id] if event_id else [],
            "signals": best["signals"],
            "thresholds": {
                "confidence": threshold,
                "retrieval": 0.30 if generic_recent else min_retrieval,
                "query_coverage": 0.0 if generic_recent else min_coverage,
                "target_coverage": 0.0 if generic_recent else min_target_coverage,
                "temporal_fit": min_temporal,
            },
        }


def apply_recent_recall_decision(
    route_decision: dict[str, Any],
    retrieval_plan: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Apply an ST-fast-path decision consistently across router/planner views."""

    route_decision["recent_recall_sufficiency"] = decision
    if not decision.get("sufficient"):
        if decision.get("evaluated"):
            route_decision["recent_recall_escalated_to_long_term"] = True
            route_decision["recent_recall_escalation_reason"] = decision.get("reason")
        return False

    route_decision.setdefault("memory_route", {})["use_long_term"] = False
    nested_decision = route_decision.get("memory_router_decision")
    if isinstance(nested_decision, dict):
        nested_decision.setdefault("memory_route", {})["use_long_term"] = False
    plan_map = retrieval_plan.get("retrieval_plan") or {}
    if isinstance(plan_map.get("M_lt"), dict):
        plan_map["M_lt"]["enabled"] = False
        plan_map["M_lt"]["disabled_reason"] = "short_term_evidence_sufficient"
    route_decision["recent_recall_fast_path"] = True
    route_decision["recent_recall_fast_path_reason"] = decision.get("reason")
    warnings = route_decision.setdefault("warnings", [])
    message = "recent recall answered from sufficient short-term evidence; long-term retrieval skipped"
    if message not in warnings:
        warnings.append(message)
    return True
