from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
HIPPO_SRC_ROOT = PROJECT_ROOT / "src" / "HippoRAG" / "src"
for _path in (SRC_ROOT, HIPPO_SRC_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from online_preprocess.io_utils import read_json, utc_now_iso, write_json, write_json_atomic
from online_current.mcur_query import build_current_prompt, build_local_current_answer
from online_current.mcur_selector import MCurFrameSelector
from online_current.mcur_store import MCurStore
from online_query.coreference_resolver import CoreferenceResolver
from online_query.evidence_packer import EvidencePacker
from online_query.interaction_cache import InteractionCache
from online_query.memory_fusion import MemoryFusion
from online_query.memory_plan import RetrievalPlanner
from online_query.memory_router import MemoryRouter
from online_query.query_router import QueryRouter
from online_query.day_prompt_context import build_day_context_block
from online_short_term.mst_retriever import MSTRetriever
from online_short_term.mst_store import MSTStore
from online_query.stream_query_context import load_stream_query_context
from online_pipeline.stream_timeline import append_timeline_event
from online_pipeline.rokid_day import (
    query_memory_ready,
    resolve_query_long_term_candidates,
    resolve_query_session_context,
    rokid_display_payload_for_relative_time,
)
from online_visual.visual_index import VisualSearchIndex, load_visual_index
from online_visual.visual_items import read_visual_items
from online_visual.visual_schema import normalize_retrieval_mode
from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime, l2_normalize
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


def _query_rag_helpers() -> Any:
    from eval import query_rag

    return query_rag


def _worldmm_classes(long_term_retrieval_scheme: str | None = None) -> tuple[Any, Any, Any, Any, Any]:
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import transform_timestamp

    scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    if scheme == "em2memory":
        from worldmm.memory.WorldMemory import WorldMemory
    elif scheme == "worldmm_legacy":
        from worldmm.memory.memory import WorldMemory
    else:
        raise ValueError(f"unsupported long-term retrieval scheme: {scheme}")

    return EmbeddingModel, LLMModel, PromptTemplateManager, WorldMemory, transform_timestamp


def _configure_semantic_embedding_cache(world_memory: Any, semantic_path: str | None) -> None:
    if not semantic_path:
        return
    semantic_memory = getattr(world_memory, "semantic_memory", None)
    if semantic_memory is None or not hasattr(semantic_memory, "embedding_cache_dir"):
        return
    configured_cache_dir = os.getenv("WORLDMM_SEMANTIC_EMBED_CACHE_DIR", "").strip()
    cache_dir = Path(configured_cache_dir) if configured_cache_dir else Path(semantic_path).parent / ".semantic_embedding_cache"
    try:
        semantic_memory.embedding_cache_dir = cache_dir
        status = getattr(semantic_memory, "embedding_cache_status", None)
        if isinstance(status, dict):
            status.setdefault("enabled", getattr(semantic_memory, "embedding_cache_enabled", True))
            status["dir"] = str(cache_dir)
    except Exception:
        return


def _image_path_runtime_scoped(path: Any) -> bool:
    text = str(path or "").replace("\\", "/").lstrip("/")
    if not text:
        return False
    if text.startswith("stream/day_assets/"):
        return False
    return text.startswith(
        (
            "current/",
            "stream/",
            "short_term/",
            "mcur/",
            "frame_stream/",
            "audio_stream/",
        )
    )


def _hhmmssff_to_seconds(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().zfill(8)
    try:
        hours = int(text[0:2])
        minutes = int(text[2:4])
        seconds = int(text[4:6])
        frames = int(text[6:8])
    except ValueError:
        return 0.0
    return hours * 3600 + minutes * 60 + seconds + frames / 100.0


def _ms(start: float, end: float | None = None) -> int:
    end = time.perf_counter() if end is None else end
    return int(round((end - start) * 1000))


def _qa_generation_ms(qa_result: Any) -> int | None:
    debug = getattr(qa_result, "llm_debug", None)
    if not isinstance(debug, dict):
        return None
    candidates = [
        debug.get("answer_generation_ms"),
        debug.get("text_only_fallback_generation_ms"),
        debug.get("primary_answer_generation_ms"),
    ]
    fallback_debug = debug.get("text_only_fallback_debug")
    if isinstance(fallback_debug, dict):
        candidates.insert(0, fallback_debug.get("answer_generation_ms"))
    for value in candidates:
        try:
            if value is not None:
                return max(0, int(round(float(value))))
        except Exception:
            continue
    return None


def _qa_timing_ms(qa_result: Any) -> dict[str, Any]:
    timing = getattr(qa_result, "timing_ms", None)
    return dict(timing) if isinstance(timing, dict) else {}


def _eval_utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _eval_duration_ms(start: float) -> int:
    return int(round(max(0.0, time.perf_counter() - start) * 1000))


def _new_eval_trace() -> dict[str, Any]:
    return {
        "trace_schema_version": 1,
        "query_started_at": _eval_utc_now_iso(),
        "retrieval_started_at": None,
        "retrieval_finished_at": None,
        "prompt_ready_at": None,
        "generation_started_at": None,
        "generation_finished_at": None,
        "query_finished_at": None,
        "stage_durations_ms": {},
        "llm_api": None,
    }


def _eval_mark(trace: dict[str, Any] | None, key: str) -> None:
    if isinstance(trace, dict) and not trace.get(key):
        trace[key] = _eval_utc_now_iso()


def _first_api_timing(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    timing = value.get("api_timing")
    if isinstance(timing, dict):
        return timing
    if value.get("llm_call_started_at") or value.get("llm_call_finished_at"):
        return {
            "api_request_started_at": value.get("llm_call_started_at"),
            "api_response_finished_at": value.get("llm_call_finished_at"),
            "api_duration_ms": value.get("llm_call_duration_ms"),
            "request_path": "model.generate",
        }
    for key in ("primary_generation", "text_only_fallback", "last_debug"):
        nested = value.get(key)
        found = _first_api_timing(nested)
        if found:
            return found
    return None


def _populate_eval_trace_from_result(trace: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    llm_debug = raw.get("llm_debug") if isinstance(raw, dict) else None
    timing = _first_api_timing(llm_debug)
    if timing:
        trace["llm_api"] = timing
        trace.setdefault("llm_api_history", [])
        if timing not in trace["llm_api_history"]:
            trace["llm_api_history"].append(timing)
        trace["prompt_ready_at"] = trace.get("prompt_ready_at") or timing.get("api_request_started_at")
        trace["generation_started_at"] = trace.get("generation_started_at") or timing.get("api_request_started_at")
        trace["generation_finished_at"] = trace.get("generation_finished_at") or timing.get("api_response_finished_at")
    trace["retrieval_started_at"] = trace.get("retrieval_started_at") or trace.get("query_started_at")
    trace["retrieval_finished_at"] = trace.get("retrieval_finished_at") or trace.get("generation_started_at")
    trace["query_finished_at"] = trace.get("query_finished_at") or _eval_utc_now_iso()
    return trace


def _eval_score_and_source(item: dict[str, Any]) -> tuple[Any, str]:
    for key in ("final_score", "retrieval_score", "score", "visual_score", "semantic_score"):
        value = item.get(key)
        if value is not None:
            return value, key
    source = str(item.get("source_memory") or item.get("source_type") or "").lower()
    if "cur" in source or "current" in source:
        return None, "not_available_for_current_observation"
    return None, "not_available"


def _eval_evidence_type(item: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key) or "").lower()
        for key in ("source_memory", "source_type", "source", "evidence_id")
    )
    if "semantic" in text or "m_sem" in text or "msem" in text:
        return "long_term_semantic"
    if "episodic" in text or "m_lt" in text or "mlt" in text or "long" in text:
        return "long_term_episodic"
    if "m_st" in text or "mst" in text or "short" in text or item.get("event_id"):
        return "short_term_event"
    if "m_cur" in text or "mcur" in text or "current" in text:
        return "current_observation"
    if item.get("transcript"):
        return "transcript_snippet"
    if item.get("image_path") or item.get("keyframe_paths") or item.get("path"):
        return "visual_frame"
    return "unknown"


def _annotate_eval_evidence(items: Any) -> Any:
    if not isinstance(items, list):
        return items
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        item.setdefault("eval_rank", index)
        score, source = _eval_score_and_source(item)
        item.setdefault("eval_score", score)
        item.setdefault("eval_score_source", source)
        item.setdefault("eval_evidence_type", _eval_evidence_type(item))
    return items


def _finalize_eval_trace(result: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    trace = _populate_eval_trace_from_result(trace, result)
    _annotate_eval_evidence(result.get("selected_evidence"))
    start = trace.get("query_started_at")
    finish = trace.get("query_finished_at")
    durations = trace.setdefault("stage_durations_ms", {})
    latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
    if latency:
        durations.setdefault("retrieval_ms", sum(
            int(latency.get(key) or 0)
            for key in (
                "text_retrieval_ms",
                "visual_retrieval_ms",
                "short_term_retrieval_ms",
                "fusion_ms",
                "memory_fusion_ms",
                "evidence_pack_ms",
            )
        ))
        durations.setdefault("answer_generation_ms", latency.get("generation_ms") or latency.get("worldmm_answer_ms"))
        durations.setdefault("end_to_end_qa_ms", latency.get("total_ms"))
    result["eval_trace"] = trace
    return result


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _answer_language_instruction() -> str:
    language = str(os.getenv("WORLDMM_ANSWER_LANGUAGE", "zh") or "zh").strip().lower()
    if language in {"zh", "cn", "chinese", "中文", "zh-cn", "zh_hans", "zh-hans", "simplified_chinese"}:
        return "请始终用简体中文回答，不要使用繁体中文。即使证据文本是英文，也要翻译和概括成简体中文；专有名词、文件名、模型名可以保留原文。"
    if language in {"en", "english"}:
        return "Answer in English."
    if language in {"auto", "same", "same_as_question"}:
        return "Answer in the same language as the question."
    return f"Answer in {language}."


_SHORT_TERM_ANSWER_MODELS: dict[tuple[str, int], Any] = {}


def _contains_keyword(text: str, keyword: str) -> bool:
    lower = (text or "").lower()
    needle = str(keyword or "").strip().lower()
    if not needle:
        return False
    if needle.isascii():
        prefix = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
        suffix = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
        return re.search(prefix + re.escape(needle) + suffix, lower) is not None
    return needle in lower


def _contains_keyword_any(text: str, keywords: Any) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in keywords)


def _current_fast_path_enabled() -> bool:
    return _env_bool("WORLDMM_CURRENT_FAST_PATH_ENABLED", True)


def _is_current_fast_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if _is_non_current_history_scope_question(q):
        return False
    return _contains_keyword_any(q, _EXPLICIT_CURRENT_FAST_KEYWORDS)


def _wants_current_fast_path(
    question: str,
    *,
    retrieval_mode: str = "auto",
    memory_mode: str = "auto",
    use_current: bool | None = None,
    use_short_term: bool | None = None,
    use_long_term: bool | None = None,
) -> bool:
    if _is_span_count_question(question):
        return False
    if not _current_fast_path_enabled() or use_current is False:
        return False
    retrieval = str(retrieval_mode or "auto").strip().lower()
    memory = str(memory_mode or "auto").strip().lower()
    if retrieval == "current" or memory == "current":
        return True
    if use_current is True and use_short_term is False and use_long_term is False:
        return True
    return _is_current_fast_question(question)


def _model_last_debug(model: Any) -> dict[str, Any]:
    debug = getattr(model, "last_debug", None)
    if isinstance(debug, dict) and debug:
        return dict(debug)
    inner = getattr(model, "model", None)
    debug = getattr(inner, "last_debug", None)
    return dict(debug) if isinstance(debug, dict) else {}


def _llm_generate_with_retries(model: Any, prompt: Any, attempts: int) -> tuple[str, dict[str, Any]]:
    attempts = max(1, int(attempts or 1))
    errors: list[str] = []
    last_debug: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        call_started_at = _eval_utc_now_iso()
        call_start = time.perf_counter()
        try:
            answer = str(model.generate(prompt)).strip()
            last_debug = _model_last_debug(model)
            if answer:
                return answer, {
                    "attempts": attempt,
                    "attempt_errors": errors,
                    "last_debug": last_debug,
                    "llm_call_started_at": call_started_at,
                    "llm_call_finished_at": _eval_utc_now_iso(),
                    "llm_call_duration_ms": _eval_duration_ms(call_start),
                }
            raise RuntimeError("empty answer from LLM")
        except Exception as exc:
            last_debug = _model_last_debug(model)
            errors.append(f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(min(2.0, 0.25 * attempt))
    raise RuntimeError("; ".join(errors) or "LLM generation failed")


def _llm_stream_with_retries(
    model: Any,
    prompt: Any,
    attempts: int,
    on_chunk: Any = None,
) -> tuple[str, dict[str, Any]]:
    attempts = max(1, int(attempts or 1))
    errors: list[str] = []
    last_debug: dict[str, Any] = {}
    for attempt in range(1, attempts + 1):
        call_started_at = _eval_utc_now_iso()
        call_start = time.perf_counter()
        try:
            stream_fn = getattr(model, "stream_generate", None)
            if callable(stream_fn):
                answer = str(stream_fn(prompt, on_chunk=on_chunk)).strip()
            else:
                answer = str(model.generate(prompt)).strip()
            last_debug = _model_last_debug(model)
            if answer:
                return answer, {
                    "attempts": attempt,
                    "attempt_errors": errors,
                    "last_debug": last_debug,
                    "llm_call_started_at": call_started_at,
                    "llm_call_finished_at": _eval_utc_now_iso(),
                    "llm_call_duration_ms": _eval_duration_ms(call_start),
                }
            raise RuntimeError("empty answer from LLM")
        except Exception as exc:
            last_debug = _model_last_debug(model)
            errors.append(f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(min(2.0, 0.25 * attempt))
    raise RuntimeError("; ".join(errors) or "LLM generation failed")


def _emit_stream_event(stream_handler: Any, event: dict[str, Any]) -> None:
    if not callable(stream_handler):
        return
    try:
        stream_handler(event)
    except Exception:
        pass


def _session_path(session_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(session_dir / path)


def _session_config_path(session_dir: Path, config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else session_dir / path


def _load_memory_config(session_dir: Path) -> tuple[Path, dict[str, Any]]:
    memory_config_path = session_dir / "worldmm" / "memory_config.json"
    if not memory_config_path.exists():
        raise FileNotFoundError("memory is not ready: worldmm/memory_config.json not found")
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict) or config.get("status") != "memory_ready":
        raise RuntimeError("memory is not ready")
    return memory_config_path, config


def _memory_version_from_config(config: dict[str, Any]) -> int | None:
    for key in ("latest_ready_memory_version", "memory_version", "version"):
        value = config.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            return parsed
    if config.get("status") == "memory_ready":
        return 1
    return None


def _status_not_ready(session_dir: Path) -> dict[str, Any]:
    status = read_json(session_dir / "status.json", default={})
    if not isinstance(status, dict):
        status = {}
    return {
        "status": "not_ready",
        "message": "memory is not ready",
        "stage": status.get("stage"),
        "progress": status.get("progress"),
    }


def _is_current_memory_ready(store: MCurStore, state: dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False
    text_ready = bool(
        state.get("current_text_ready")
        or state.get("audio_current_ready")
        or int(state.get("transcript_segment_count", 0) or 0) > 0
        or store.transcript_path.exists()
    )
    return bool(state.get("mcur_ready") and (store.frames_path.exists() or text_ready))


def _stream_answer_completeness(stream_context: dict[str, Any]) -> dict[str, Any]:
    if not stream_context.get("is_stream_session"):
        return {"level": "complete", "reason": "none", "visible_to_user": False}
    if stream_context.get("recommended_answer_policy") == "limited":
        return {"level": "limited", "reason": "current_only", "visible_to_user": False}
    reasons = []
    if stream_context.get("asr_lagging"):
        reasons.append("asr_lagging")
    if stream_context.get("semantic_lagging"):
        reasons.append("semantic_lagging")
    if stream_context.get("memory_lagging"):
        reasons.append("memory_lagging")
    if reasons:
        return {"level": "partial", "reason": "|".join(reasons), "visible_to_user": False}
    return {"level": "complete", "reason": "none", "visible_to_user": False}


def _stream_warnings(stream_context: dict[str, Any]) -> list[str]:
    if not stream_context.get("is_stream_session"):
        return []
    warnings = []
    if stream_context.get("asr_lagging"):
        warnings.append("ASR is still processing recent chunks")
    if stream_context.get("semantic_lagging"):
        warnings.append("Semantic memory is lagging behind fast episodic memory")
    if stream_context.get("graph_lagging"):
        warnings.append("Graph memory is lagging behind fast episodic memory")
    if stream_context.get("memory_lagging"):
        warnings.append("Long-term memory is still catching up with recent stream events")
    if stream_context.get("latest_processed_chunk_index", -1) < stream_context.get("latest_uploaded_chunk_index", -1):
        warnings.append("Stream processing is behind latest uploaded chunk")
    return warnings


_PROVISIONAL_CAPTION_SOURCES = {"", "placeholder", "provisional", "fast", "rule", "mock", "none", "null"}
_VISUAL_EVENT_KEYWORDS = (
    "现在", "当前", "此刻", "正在", "画面", "看见", "看到", "出现", "东西", "物体", "场景",
    "刚才", "刚刚", "最近", "上一段", "发生了什么", "做什么", "有什么",
    "这是什么", "这个是什么", "这是啥", "这是什么东西", "看到什么", "看到了什么", "能看到什么", "当前场景",
    "what is in", "what's in", "what happened", "what just happened", "recently", "just now",
    "a moment ago", "a few seconds ago", "now", "currently", "right now", "at this moment",
    "current frame", "current scene", "current view", "current screen", "in the current frame",
    "in the current scene", "in front of me", "visible", "object", "thing", "scene", "frame",
    "screen", "display", "on screen", "on the screen", "what do you see", "what can you see",
    "what is visible", "what can be seen", "what is shown", "what is on screen",
    "what's on screen", "what is on the screen", "what's on the screen", "what am i seeing",
    "what am i looking at", "describe the scene", "describe current scene",
    "describe the current scene", "read the screen", "read the current screen", "text on screen",
    "what is this?", "what's this?", "what is that?", "what's that?",
)
_SUMMARY_KEYWORDS = (
    "总结", "概括", "到目前为止", "整个", "全程", "主要发生",
    "summary", "summarize", "recap", "overall", "overview", "brief summary",
    "briefly summarize", "main content", "main points", "what has happened so far",
    "what happened so far", "what is this video about", "what is going on in the video",
    "describe the video", "describe what happened", "entire video", "whole video",
)
_MEMORY_STATUS_KEYWORDS = (
    "检索库", "记忆库", "多少记忆", "构建了多少", "ready", "准备好", "长期记忆", "短期记忆",
    "m_cur", "m_st", "m_lt", "memory status", "memory ready", "index ready", "component version",
)
_COUNT_QUERY_KEYWORDS = (
    "一共", "总共", "总计", "总共有", "一共有", "多少", "几个", "几次", "几幅", "几张", "数量",
    "count", "how many", "total", "in total", "altogether",
)
_SPAN_QUERY_KEYWORDS = (
    "从刚才到现在", "从刚刚到现在", "刚才到现在", "刚刚到现在", "到现在", "到目前为止",
    "since just now", "from just now", "so far", "up to now",
)
_CURRENT_SURFACE_KEYWORDS = (
    "当前画面", "现在画面", "此刻画面", "画面里现在", "当前场景", "当前帧", "眼前",
    "current frame", "in the current frame", "current scene", "current view", "current screen",
    "current image", "in the current scene", "in the current view", "in front of me",
)
_EXPLICIT_CURRENT_FAST_KEYWORDS = _CURRENT_SURFACE_KEYWORDS + (
    "此刻", "正在", "正在发生", "正在做", "现在在做", "现在发生", "眼下", "实时",
    "这是什么", "这个是什么", "这是啥", "这是什么东西", "看到什么", "看到了什么", "能看到什么",
    "right now", "at this moment", "what is happening now", "what's happening now",
    "what is going on now", "what's going on now", "what do you see", "what can you see",
    "what am i seeing", "what am i looking at", "what is on screen", "what's on screen",
    "what is on the screen", "what's on the screen", "what is visible now",
    "what can you see now", "describe current scene", "describe the current scene",
    "read the current screen", "what is this?", "what's this?", "what is that?", "what's that?",
)
_HISTORY_SCOPE_FAST_PATH_BLOCK_KEYWORDS = (
    "刚才", "刚刚", "最近", "上一段", "前面一点", "刚发生", "不久前", "前一会", "前面",
    "之前", "以前", "过去", "从开始", "从头", "从头到尾", "全过程", "整个视频", "全程",
    "到现在", "到目前", "到目前为止", "目前为止", "出现过", "有哪些", "哪些", "列出",
    "总结", "概括", "整体", "主要发生", "发生了什么",
    "just now", "recently", "earlier", "previously", "last segment", "last scene",
    "before", "after", "so far", "up to now", "from the beginning", "entire video",
    "whole video", "summary", "summarize", "recap", "overall", "overview", "what happened",
    "what happened so far", "what has happened so far",
)


def _is_visual_event_question(question: str, query_type: str | None = None) -> bool:
    text = str(question or "").lower()
    if str(query_type or "") in {"current_perception", "recent_recall"}:
        return True
    return _contains_keyword_any(text, _VISUAL_EVENT_KEYWORDS)


def _is_summary_question_text(question: str, query_type: str | None = None) -> bool:
    text = str(question or "").lower()
    return str(query_type or "") in {"long_term_summary"} or _contains_keyword_any(text, _SUMMARY_KEYWORDS)


def _is_memory_status_question(question: str) -> bool:
    text = str(question or "").lower()
    return _contains_keyword_any(text, _MEMORY_STATUS_KEYWORDS)


def _is_span_count_question(question: str) -> bool:
    text = str(question or "").lower()
    if not _contains_keyword_any(text, _COUNT_QUERY_KEYWORDS):
        return False
    has_span = _contains_keyword_any(text, _SPAN_QUERY_KEYWORDS)
    has_recent = _contains_keyword_any(text, QueryRouter.RECENT_KWS)
    has_current = _contains_keyword_any(text, QueryRouter.CURRENT_KWS)
    return bool(has_span or (has_recent and has_current))


def _is_non_current_history_scope_question(question: str) -> bool:
    text = str(question or "").lower()
    if _is_span_count_question(text):
        return True
    if _contains_keyword_any(text, _HISTORY_SCOPE_FAST_PATH_BLOCK_KEYWORDS):
        return True
    has_aggregate = _contains_keyword_any(text, _COUNT_QUERY_KEYWORDS) or _contains_keyword_any(text, ["所有", "全部", "有哪些", "which", "list"])
    has_current_surface = _contains_keyword_any(text, _CURRENT_SURFACE_KEYWORDS)
    return bool(has_aggregate and not has_current_surface)


def _event_is_provisional(event: dict[str, Any]) -> bool:
    caption_source = str(event.get("caption_source") or "").strip().lower()
    status = str(event.get("status") or "").strip().lower()
    refined = str(event.get("event_caption_refined") or "").strip()
    return (
        caption_source in _PROVISIONAL_CAPTION_SOURCES
        or status in {"", "open", "provisional", "refine_failed"}
        or not refined
    )


def _diagnose_partial_memory(
    *,
    question: str,
    route_decision: dict[str, Any],
    current_results: list[dict[str, Any]] | None,
    short_term_results: list[dict[str, Any]] | None,
    memory_config: dict[str, Any] | None,
    visual_ready: bool,
    long_term_ready: bool,
) -> dict[str, Any]:
    current_results = current_results or []
    short_term_results = short_term_results or []
    config = memory_config if isinstance(memory_config, dict) else {}
    readiness = config.get("readiness") if isinstance(config.get("readiness"), dict) else {}
    lag = config.get("lag") if isinstance(config.get("lag"), dict) else {}

    provisional_events = [event for event in short_term_results if isinstance(event, dict) and _event_is_provisional(event)]
    has_short_term_frames = any((event.get("keyframes") or event.get("keyframe_paths")) for event in short_term_results if isinstance(event, dict))
    has_current_frames = any(
        ((item.get("current_selection") or {}).get("evidence_frames") or [])
        for item in current_results
        if isinstance(item, dict)
    )
    long_term_event_ready = bool(
        long_term_ready
        or config.get("long_term_partial_ready")
        or config.get("latest_ready_memory_version")
        or config.get("latest_fast_ready_version")
        or config.get("episodic_index_ready")
        or readiness.get("episodic_ready")
    )
    semantic_ready = bool(
        config.get("semantic_memory_ready")
        or config.get("latest_semantic_ready_version")
        or readiness.get("semantic_ready")
    )
    long_term_full_ready = bool(
        config.get("long_term_full_ready")
        or readiness.get("long_term_full_ready")
        or readiness.get("long_term_ready")
    )
    semantic_lagging = bool(lag.get("semantic_lagging") or config.get("semantic_lagging"))
    graph_lagging = bool(lag.get("graph_lagging") or config.get("graph_lagging"))

    reasons: list[str] = []
    if provisional_events:
        reasons.append("short_term_provisional")
        if any(not str(event.get("event_caption_refined") or "").strip() for event in provisional_events):
            reasons.append("refined_missing")
    if not long_term_event_ready:
        reasons.append("long_term_missing")
    elif not long_term_full_ready:
        reasons.append("long_term_partial")
    if not semantic_ready or semantic_lagging or graph_lagging:
        reasons.append("semantic_partial")
    if not visual_ready and (has_current_frames or has_short_term_frames):
        reasons.append("visual_index_missing")
    if current_results and not short_term_results and not long_term_ready:
        reasons.append("current_only")

    query_type = str(route_decision.get("query_type") or "")
    visual_event_question = _is_visual_event_question(question, query_type)
    summary_question = _is_summary_question_text(question, query_type)
    has_local_frames = bool(has_current_frames or has_short_term_frames)
    should_use_images = bool(has_local_frames and (visual_event_question or provisional_events or reasons))
    slow_memory_incomplete = bool("long_term_missing" in set(reasons))
    prefer_fast_local = bool(
        (current_results or short_term_results)
        and should_use_images
        and (visual_event_question or provisional_events or slow_memory_incomplete)
        and not long_term_event_ready
    )
    if not reasons and (current_results or short_term_results):
        completeness = "full"
    elif (current_results or short_term_results) and not long_term_ready:
        completeness = "provisional_only"
    else:
        completeness = "partial"
    return {
        "memory_completeness": completeness,
        "used_image_fallback": should_use_images,
        "image_fallback_reason": reasons[0] if reasons else ("current_only" if current_results else None),
        "image_fallback_reasons": list(dict.fromkeys(reasons)),
        "attached_image_count": 0,
        "provisional_event_count": len(provisional_events),
        "prefer_fast_local": prefer_fast_local,
        "visual_event_question": visual_event_question,
        "summary_question": summary_question,
        "slow_memory_incomplete": slow_memory_incomplete,
        "has_local_frames": has_local_frames,
        "long_term_event_ready": long_term_event_ready,
    }


def _apply_image_fallback_to_route(route_decision: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    if not diagnostics.get("used_image_fallback"):
        return
    route_decision["use_image_evidence"] = True
    route_decision["use_image_evidence_source"] = "auto_partial_memory_fallback"
    route_decision["max_image_evidence"] = max(
        int(route_decision.get("max_image_evidence") or 0),
        _env_int("WORLDMM_PARTIAL_MEMORY_IMAGE_FALLBACK_MAX_IMAGES", 4),
    )
    route_decision["evidence_frames_k"] = max(
        int(route_decision.get("evidence_frames_k") or 0),
        _env_int("WORLDMM_PARTIAL_MEMORY_IMAGE_FALLBACK_FRAME_K", 6),
    )
    warnings = route_decision.setdefault("warnings", [])
    message = "partial/provisional memory detected; attaching local visual frames as primary evidence"
    if message not in warnings:
        warnings.append(message)


def _attach_memory_completeness(result: dict[str, Any], diagnostics: dict[str, Any], attached_image_count: int | None = None) -> None:
    if not diagnostics:
        return
    count = int(attached_image_count if attached_image_count is not None else diagnostics.get("attached_image_count") or 0)
    result["memory_completeness"] = diagnostics.get("memory_completeness") or "full"
    result["used_image_fallback"] = bool(diagnostics.get("used_image_fallback") and count > 0)
    result["image_fallback_reason"] = diagnostics.get("image_fallback_reason")
    result["image_fallback_reasons"] = diagnostics.get("image_fallback_reasons", [])
    result["attached_image_count"] = count
    result["provisional_event_count"] = int(diagnostics.get("provisional_event_count") or 0)
    result.setdefault("raw", {})["partial_memory_diagnostics"] = {**diagnostics, "attached_image_count": count}


def _used_memory_sources(result: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    fusion = result.get("fusion_summary") if isinstance(result.get("fusion_summary"), dict) else {}
    for item in fusion.get("selected_memory_sources", []) or []:
        if item and item not in sources:
            sources.append(str(item))
    for key, source in (("current_results", "M_cur"), ("short_term_results", "M_st"), ("text_results", "M_lt"), ("retrieved_memories", "M_lt")):
        values = result.get(key)
        if values and source not in sources:
            sources.append(source)
    return sources


def _frame_open_event_synthetic_evidence(session_dir: Path) -> dict[str, Any] | None:
    state = read_json(session_dir / "stream" / "frame_event_state.json", default={})
    if not isinstance(state, dict):
        return None
    open_event = state.get("open_event") if isinstance(state.get("open_event"), dict) else None
    if not open_event:
        return None
    start = _safe_float(open_event.get("start_time"), 0.0)
    end = _safe_float(open_event.get("last_update_time"), start)
    keyframes = [dict(item) for item in open_event.get("keyframes", []) or [] if isinstance(item, dict)]
    if not keyframes:
        latest = open_event.get("latest_frame") if isinstance(open_event.get("latest_frame"), dict) else {}
        if latest:
            keyframes = [dict(latest)]
    frame_indices = list(open_event.get("source_frame_indices") or [])
    retrieval_text = (
        f"Open frame-stream event from {start:.1f}s to {end:.1f}s with "
        f"{len(frame_indices) or len(keyframes)} uploaded frames; event has not been closed/refined yet."
    )
    return {
        "event_id": open_event.get("open_event_id") or f"synthetic_frame_open_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}",
        "input_source": "frame_stream",
        "source": {"type": "frame_audio_stream", "input_source": "frame_stream"},
        "status": "open_provisional",
        "caption_source": "placeholder",
        "event_caption_placeholder": retrieval_text,
        "retrieval_text": retrieval_text,
        "start_time": start,
        "end_time": end,
        "duration": round(max(0.0, end - start), 3),
        "keyframes": keyframes,
        "evidence_frames": list(keyframes),
        "source_frame_indices": frame_indices,
        "diff_stats": dict(open_event.get("diff_stats") or {}),
        "transcript": open_event.get("transcript", ""),
        "transcript_segments": list(open_event.get("transcript_segments") or []),
        "needs_refine": True,
        "synthetic": True,
    }


def _has_current_evidence(session_dir: Path) -> bool:
    try:
        store = MCurStore(session_dir)
        state = store.get_state()
        return _is_current_memory_ready(store, state)
    except Exception:
        return False


def _has_any_query_evidence(session_dir: Path) -> bool:
    if _has_current_evidence(session_dir):
        return True
    if _frame_open_event_synthetic_evidence(session_dir):
        return True
    try:
        store = MSTStore(session_dir)
        if store.is_ready():
            return True
        if store.events_path.exists() and store.events_path.stat().st_size > 0:
            return True
    except Exception:
        pass
    memory_config = read_json(session_dir / "worldmm" / "memory_config.json", default={})
    if isinstance(memory_config, dict):
        readiness = memory_config.get("readiness") if isinstance(memory_config.get("readiness"), dict) else {}
        return bool(
            memory_config.get("latest_fast_ready_version")
            or memory_config.get("latest_ready_memory_version")
            or readiness.get("long_term_partial_ready")
            or readiness.get("long_term_full_ready")
        )
    return False


def _mark_partial_fallback(
    result: dict[str, Any],
    *,
    reason: str,
    sources: list[str],
    warning: str,
    visible: bool = True,
) -> dict[str, Any]:
    result["status"] = "ok" if result.get("status") in {None, "not_ready"} else result.get("status")
    result["memory_completeness"] = "partial"
    result["fallback_reason"] = reason
    result["used_memory_sources"] = list(dict.fromkeys(sources))
    result["answer_completeness"] = {"level": "partial", "reason": reason, "visible_to_user": visible}
    warnings = result.setdefault("warnings", [])
    if warning and warning not in warnings:
        warnings.append(warning)
    stream_warnings = result.setdefault("stream_warnings", [])
    if warning and warning not in stream_warnings:
        stream_warnings.append(warning)
    result.setdefault("raw", {})["fallback_reason"] = reason
    return result


def _answer_memory_status_question(session_id: str, session_dir: Path, question: str, *, total_start: float = 0.0) -> dict[str, Any]:
    current_state = read_json(session_dir / "current" / "current_state.json", default={})
    frame_state = read_json(session_dir / "stream" / "frame_state.json", default={})
    mst_state = read_json(session_dir / "short_term" / "mst_state.json", default={})
    memory_config = read_json(session_dir / "worldmm" / "memory_config.json", default={})
    component_versions = read_json(session_dir / "worldmm" / "incremental" / "component_versions.json", default={})
    current_state = current_state if isinstance(current_state, dict) else {}
    frame_state = frame_state if isinstance(frame_state, dict) else {}
    mst_state = mst_state if isinstance(mst_state, dict) else {}
    memory_config = memory_config if isinstance(memory_config, dict) else {}
    component_versions = component_versions if isinstance(component_versions, dict) else {}
    summary = {
        "mcur_ready": bool(current_state.get("mcur_ready")),
        "mcur_version": current_state.get("mcur_version", 0),
        "mcur_frame_count": current_state.get("frame_count", 0),
        "frame_stream_accepted_count": frame_state.get("accepted_count", 0),
        "short_term_ready": bool(mst_state.get("short_term_ready")),
        "mst_version": mst_state.get("mst_version", 0),
        "mst_event_count": mst_state.get("event_count", 0),
        "latest_ready_memory_version": memory_config.get("latest_ready_memory_version"),
        "latest_fast_ready_version": memory_config.get("latest_fast_ready_version"),
        "latest_semantic_ready_version": memory_config.get("latest_semantic_ready_version"),
        "latest_graph_ready_version": memory_config.get("latest_graph_ready_version"),
        "long_term_full_ready": bool(memory_config.get("long_term_full_ready") or ((memory_config.get("readiness") or {}).get("long_term_full_ready") if isinstance(memory_config.get("readiness"), dict) else False)),
        "component_versions": component_versions,
    }
    answer = (
        f"Visible memory status: M_cur ready={summary['mcur_ready']} version={summary['mcur_version']} frames={summary['mcur_frame_count']}; "
        f"frame_stream accepted={summary['frame_stream_accepted_count']}; "
        f"M_st ready={summary['short_term_ready']} version={summary['mst_version']} events={summary['mst_event_count']}; "
        f"M_lt latest_ready={summary['latest_ready_memory_version']} fast={summary['latest_fast_ready_version']} "
        f"semantic={summary['latest_semantic_ready_version']} graph={summary['latest_graph_ready_version']} "
        f"full_ready={summary['long_term_full_ready']}."
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "question": question,
        "query_type": "memory_status",
        "answer": answer,
        "memory_status": summary,
        "memory_completeness": "partial" if not summary["long_term_full_ready"] else "full",
        "answer_completeness": {
            "level": "partial" if not summary["long_term_full_ready"] else "complete",
            "reason": "memory_lagging" if not summary["long_term_full_ready"] else "none",
            "visible_to_user": not summary["long_term_full_ready"],
        },
        "warnings": ["long-term index may still be building; counts reflect currently visible components"] if not summary["long_term_full_ready"] else [],
        "used_memory_sources": ["runtime_state"],
        "timestamps": [],
        "evidence_frames": [],
        "retrieved_memories": [],
        "supporting_semantic_facts": [],
        "latency": {"total_ms": _ms(total_start)} if total_start else {},
        "raw": {"memory_status": summary},
    }


def _is_summary_or_semantic_question(question: str, result: dict[str, Any]) -> bool:
    query_type = str(result.get("query_type") or "")
    if query_type in {"long_term_summary", "temporal_reasoning", "entity_tracking"}:
        return True
    text = str(question or "").lower()
    keywords = ("总结", "概括", "到目前为止", "主要发生", "整个视频", "关系", "状态变化", "summary", "summarize", "overall", "relationship")
    return any(keyword in text for keyword in keywords)


def _append_recent_stream_supplement(
    result: dict[str, Any],
    *,
    session_dir: Path,
    question: str,
    limit: int = 2,
) -> bool:
    if not _is_summary_or_semantic_question(question, result):
        return False
    try:
        events = MSTStore(session_dir).load_events()
    except Exception:
        return False
    if not events:
        return False
    existing_ranges: list[tuple[float, float]] = []
    for key in ("selected_evidence", "retrieved_memories", "text_results"):
        for item in result.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            try:
                existing_ranges.append((float(item.get("start_time", 0.0) or 0.0), float(item.get("end_time", item.get("start_time", 0.0)) or 0.0)))
            except Exception:
                continue
    supplements = []
    for event in sorted(events, key=lambda item: float(item.get("end_time", item.get("start_time", 0.0)) or 0.0), reverse=True):
        try:
            start = float(event.get("start_time", 0.0) or 0.0)
            end = float(event.get("end_time", start) or start)
        except Exception:
            continue
        if any(max(start, a) <= min(end, b) and min(end, b) - max(start, a) >= 0.5 * max(0.1, end - start) for a, b in existing_ranges):
            continue
        caption = (
            event.get("event_caption_refined")
            or event.get("event_caption_fast")
            or event.get("retrieval_text")
            or event.get("transcript")
            or ""
        )
        if not caption:
            continue
        frames = event.get("keyframes", []) or []
        supplements.append(
            {
                "evidence_id": event.get("event_id"),
                "source_memory": "M_st",
                "source": "recent_stream_supplement",
                "source_type": "micro_event",
                "start_time": start,
                "end_time": end,
                "timestamp": (start + end) / 2.0,
                "caption": caption,
                "transcript": event.get("transcript", ""),
                "keyframe_paths": [frame.get("path") for frame in frames if isinstance(frame, dict) and frame.get("path")],
                "status": event.get("status"),
                "metadata": event,
            }
        )
        if len(supplements) >= limit:
            break
    if not supplements:
        return False
    result.setdefault("selected_evidence", [])
    result["selected_evidence"].extend(supplements)
    result.setdefault("memory_results", {})
    if isinstance(result["memory_results"], dict):
        result["memory_results"].setdefault("short_term_results", [])
        result["memory_results"]["short_term_results"].extend([item["metadata"] for item in supplements])
    result.setdefault("fusion_summary", {})
    if isinstance(result["fusion_summary"], dict):
        sources = list(result["fusion_summary"].get("selected_memory_sources", []) or [])
        if "M_st" not in sources:
            sources.append("M_st")
        result["fusion_summary"]["selected_memory_sources"] = sources
        result["fusion_summary"]["recent_stream_supplement_count"] = len(supplements)
    return True


def _attach_stream_query_awareness(result: dict[str, Any], *, session_id: str, session_dir: Path, sessions_root: Path, project_root: Path, question: str) -> dict[str, Any]:
    stream_context = load_stream_query_context(session_id, sessions_root=sessions_root, project_root=project_root, question=question)
    if not stream_context.get("is_stream_session"):
        return result
    used_supplement = _append_recent_stream_supplement(result, session_dir=session_dir, question=question)
    result["stream_context"] = stream_context
    existing_completeness = result.get("answer_completeness") if isinstance(result.get("answer_completeness"), dict) else {}
    if existing_completeness.get("visible_to_user") or existing_completeness.get("level") in {"partial", "limited"}:
        result["answer_completeness"] = existing_completeness
    else:
        result["answer_completeness"] = _stream_answer_completeness(stream_context)
    existing_sources = list(result.get("used_memory_sources") or [])
    result["used_memory_sources"] = list(dict.fromkeys(existing_sources + _used_memory_sources(result)))
    result["used_stream_supplement"] = bool(used_supplement)
    result["stream_warnings"] = list(dict.fromkeys((result.get("stream_warnings") or []) + _stream_warnings(stream_context)))
    result.setdefault("raw", {})
    if isinstance(result["raw"], dict):
        result["raw"]["stream_context"] = stream_context
        result["raw"]["answer_completeness"] = result["answer_completeness"]
    append_timeline_event(
        session_dir,
        "query_answered",
        metadata={
            "question": question,
            "query_type": result.get("query_type"),
            "used_memory_sources": result.get("used_memory_sources", []),
            "stream_policy": stream_context.get("recommended_answer_policy"),
            "asr_lagging": stream_context.get("asr_lagging"),
            "semantic_lagging": stream_context.get("semantic_lagging"),
            "latency_ms": (result.get("latency") or {}).get("total_ms") if isinstance(result.get("latency"), dict) else None,
            "answer_completeness": result["answer_completeness"].get("level"),
        },
    )
    return result


def _event_timestamps(event: dict[str, Any]) -> dict[str, float]:
    return {
        "start": _hhmmssff_to_seconds(event.get("start_time", "0")),
        "end": _hhmmssff_to_seconds(event.get("end_time", "0")),
    }


def _keyframe_timestamp(path: str, evidence_frame: dict[str, Any] | None = None) -> float | None:
    if evidence_frame and evidence_frame.get("timestamp") is not None:
        try:
            return _normalize_keyframe_timestamp(evidence_frame["timestamp"], path)
        except Exception:
            pass
    stem = Path(path).stem
    parts = stem.split("_")
    for part in reversed(parts):
        try:
            value = int(part)
            return _normalize_keyframe_timestamp(value, path, token=part)
        except Exception:
            continue
    return None


def _normalize_keyframe_timestamp(value: Any, path: str | None = None, token: str | None = None) -> float | None:
    """Normalize outward keyframe timestamps to seconds.

    Stream-generated MST keyframes use millisecond filenames such as
    kf_000008000.jpg, while legacy preprocess keyframes use second filenames
    such as kf_000015.jpg. Keep long-video second timestamps intact unless the
    path/token clearly indicates millisecond encoding.
    """
    try:
        timestamp = float(value)
    except Exception:
        return None
    path_text = str(path or "")
    token_text = str(token or "")
    if not token_text and path_text:
        match = re.search(r"kf_(\d{7,})(?:\D|$)", Path(path_text).stem)
        token_text = match.group(1) if match else ""
    is_stream_keyframe = "stream/keyframes" in path_text.replace("\\", "/")
    if timestamp >= 1000.0 and (is_stream_keyframe or len(token_text) >= 7):
        timestamp = timestamp / 1000.0
    return round(timestamp, 3)


def _normalize_frame_timestamps_in_context(value: Any) -> Any:
    if isinstance(value, dict):
        item = dict(value)
        if "timestamp" in item:
            path = str(item.get("path") or item.get("image_path") or "")
            item["timestamp"] = _normalize_keyframe_timestamp(item.get("timestamp"), path)
        for key in ("referenced_evidence_frames", "evidence_frames", "keyframes"):
            if key in item:
                item[key] = _normalize_frame_timestamps_in_context(item[key])
        return item
    if isinstance(value, list):
        return [_normalize_frame_timestamps_in_context(item) for item in value]
    return value


def _canonical_segment_id(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"(seg_\d{6}_\d{6})(?:_\d{4})?", text)
    if match:
        return match.group(1)
    parts = text.split("_")
    if len(parts) == 5 and parts[0] == "seg" and all(p.isdigit() for p in parts[1:]):
        return "_".join(parts[:4])
    return text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _latest_rokid_relative_seconds(session_dir: Path) -> float:
    candidates: list[float] = []
    for rel_path, keys in (
        (Path("stream") / "frame_state.json", ("latest_memory_relative_ts_ms", "latest_relative_ts_ms")),
        (Path("stream") / "audio_state.json", ("latest_relative_ts_ms",)),
        (Path("stream") / "rokid_state.json", ("latest_frame_relative_ts_ms", "latest_audio_relative_ts_ms")),
    ):
        payload = read_json(session_dir / rel_path, default={})
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if payload.get(key) is not None:
                candidates.append(max(0.0, _safe_float(payload.get(key), 0.0) / 1000.0))
    current_state = read_json(session_dir / "current" / "current_state.json", default={})
    if isinstance(current_state, dict):
        for key in ("current_time", "window_end_time"):
            if current_state.get(key) is not None:
                candidates.append(max(0.0, _safe_float(current_state.get(key), 0.0)))
    return max(candidates) if candidates else 0.0


def _rokid_query_time_override(day_context: dict[str, Any] | None, runtime_session_dir: Path) -> dict[str, Any] | None:
    if not isinstance(day_context, dict) or not day_context.get("is_rokid_day_child"):
        return None
    day_label = str(day_context.get("day_label") or "").strip()
    if not day_label:
        return None
    relative_seconds = _latest_rokid_relative_seconds(runtime_session_dir)
    display = rokid_display_payload_for_relative_time(day_context, relative_seconds)
    return {
        "until_date": day_label,
        "until_time": display["display_hhmmssff"],
        "relative_seconds": round(relative_seconds, 3),
        **display,
    }


def _mark_active_query_component_versions(
    session_dir: Path,
    config: dict[str, Any],
    *,
    active_query_memory_version: int | None,
) -> None:
    """Record which component versions are actually loaded by query runtime.

    This is runtime metadata only. It must not build or mutate retrieval
    artifacts, but it keeps monitor/inspect output aligned with the query
    worker's loaded snapshot.
    """
    if not active_query_memory_version:
        return
    component_path = session_dir / "worldmm" / "incremental" / "component_versions.json"
    data = read_json(component_path, default={})
    if not isinstance(data, dict):
        data = {"session_id": session_dir.name}
    data["session_id"] = session_dir.name
    fast = data.setdefault("fast", {})
    if isinstance(fast, dict):
        fast.setdefault("latest_ready_version", _safe_int(config.get("latest_fast_ready_version") or active_query_memory_version, active_query_memory_version))
        fast["active_query_version"] = _safe_int(active_query_memory_version, active_query_memory_version)
    component_keys = {
        "visual": config.get("latest_visual_ready_version") or config.get("visual_version"),
        "graph": config.get("latest_graph_ready_version") or config.get("graph_version"),
        "semantic": config.get("latest_semantic_ready_version") or config.get("semantic_version"),
    }
    for key, version in component_keys.items():
        section = data.setdefault(key, {})
        if isinstance(section, dict) and version is not None:
            section["active_query_version"] = _safe_int(version, 0)
    full = data.setdefault("full", {})
    if isinstance(full, dict):
        full["active_query_version"] = _safe_int(active_query_memory_version, active_query_memory_version)
    data["active_query_memory_version"] = _safe_int(active_query_memory_version, active_query_memory_version)
    data["active_query_updated_at"] = utc_now_iso()
    data["updated_at"] = utc_now_iso()
    write_json_atomic(component_path, data)


class LoadedQueryEngine:
    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        memory_config_path: Path,
        memory_config: dict[str, Any],
        world_memory: Any,
        query_args: dict[str, Any],
        visual_evidence_data: list[dict[str, Any]],
        semantic_path: str,
        visual_ready: bool = False,
        visual_items: dict[str, dict[str, Any]] | None = None,
        visual_id_mapping: dict[str, Any] | None = None,
        visual_index: VisualSearchIndex | None = None,
        visual_embedding_model: str | None = None,
        visual_version: int = 0,
        strict_load_only: bool = True,
        latest_ready_memory_version: int | None = None,
        building_memory_version: int | None = None,
        active_query_memory_version: int | None = None,
        preload_status: str = "loaded",
        long_term_retrieval_scheme: str | None = None,
    ) -> None:
        now = time.time()
        self.session_id = session_id
        self.session_dir = session_dir
        self.memory_config_path = memory_config_path
        self.memory_config_mtime = memory_config_path.stat().st_mtime if memory_config_path.exists() else 0.0
        self.memory_config = memory_config
        self.world_memory = world_memory
        self.query_args = query_args
        self.visual_evidence_data = visual_evidence_data
        self.semantic_path = semantic_path
        self.visual_ready = visual_ready
        self.visual_items = visual_items or {}
        self.visual_id_mapping = visual_id_mapping or {}
        self.visual_index = visual_index
        self.visual_embedding_model = visual_embedding_model or ""
        self.visual_loaded_at = now if visual_ready else 0.0
        self.visual_version = visual_version
        self.strict_load_only = strict_load_only
        self.latest_ready_memory_version = latest_ready_memory_version
        self.building_memory_version = building_memory_version
        self.active_query_memory_version = active_query_memory_version or latest_ready_memory_version
        self.preload_status = preload_status
        self.long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
        self.loaded_at = now
        self.last_accessed_at = now
        self.query_count = 0
        self.recent_queries: deque[dict[str, Any]] = deque(maxlen=20)
        self.interaction_cache = InteractionCache(session_id=session_id, session_dir=session_dir)
        self.mst_store = MSTStore(session_dir)
        self.mst_retriever = MSTRetriever(self.mst_store)
        mst_state = self.mst_store.get_state()
        self.short_term_ready = bool(mst_state.get("short_term_ready") and self.mst_store.events_path.exists())
        self.mst_version = int(mst_state.get("mst_version", 0) or 0)
        self.mst_loaded_at = now if self.short_term_ready else 0.0
        self.mcur_store = MCurStore(session_dir)
        mcur_state = self.mcur_store.get_state()
        self.current_ready = _is_current_memory_ready(self.mcur_store, mcur_state)
        self.current_stale = self.mcur_store.is_stale(mcur_state)
        self.mcur_version = int(mcur_state.get("mcur_version", 0) or 0)
        self.mcur_loaded_at = now if self.current_ready else 0.0
        self.evidence_by_doc_id: dict[str, dict[str, Any]] = {}
        for item in visual_evidence_data:
            if not isinstance(item, dict):
                continue
            for key in ("doc_id", "evidence_doc_id", "segment_id"):
                value = item.get(key)
                if value:
                    self.evidence_by_doc_id[str(value)] = item
                    canonical = _canonical_segment_id(value)
                    if canonical:
                        self.evidence_by_doc_id.setdefault(canonical, item)

    @contextmanager
    def runtime_session_context(
        self,
        *,
        realtime_session_id: str | None = None,
        short_term_session_id: str | None = None,
        interaction_cache_session_id: str | None = None,
        requested_session_id: str | None = None,
        day_context: dict[str, Any] | None = None,
    ):
        original = {
            "runtime_session_id": getattr(self, "runtime_session_id", self.session_id),
            "runtime_session_dir": getattr(self, "runtime_session_dir", self.session_dir),
            "interaction_cache": self.interaction_cache,
            "mst_store": self.mst_store,
            "mst_retriever": self.mst_retriever,
            "short_term_ready": self.short_term_ready,
            "mst_version": self.mst_version,
            "mst_loaded_at": self.mst_loaded_at,
            "mcur_store": self.mcur_store,
            "current_ready": self.current_ready,
            "current_stale": self.current_stale,
            "mcur_version": self.mcur_version,
            "mcur_loaded_at": self.mcur_loaded_at,
            "requested_session_id": getattr(self, "requested_session_id", self.session_id),
            "day_context": getattr(self, "day_context", None),
            "query_time_override": getattr(self, "query_time_override", None),
        }
        realtime_id = str(realtime_session_id or self.session_id)
        short_term_id = str(short_term_session_id or realtime_id)
        cache_id = str(interaction_cache_session_id or realtime_id)
        runtime_dir = self.session_dir.parent / realtime_id
        short_term_dir = self.session_dir.parent / short_term_id
        cache_dir = self.session_dir.parent / cache_id
        try:
            self.runtime_session_id = realtime_id
            self.runtime_session_dir = runtime_dir
            self.requested_session_id = str(requested_session_id or realtime_id)
            self.day_context = dict(day_context or {})
            self.query_time_override = _rokid_query_time_override(self.day_context, runtime_dir)
            self.interaction_cache = InteractionCache(session_id=cache_id, session_dir=cache_dir)
            self.mst_store = MSTStore(short_term_dir)
            self.mst_retriever = MSTRetriever(self.mst_store)
            mst_state = self.mst_store.get_state()
            self.short_term_ready = bool(mst_state.get("short_term_ready") and self.mst_store.events_path.exists())
            self.mst_version = int(mst_state.get("mst_version", 0) or 0)
            self.mst_loaded_at = time.time() if self.short_term_ready else 0.0
            self.mcur_store = MCurStore(runtime_dir)
            mcur_state = self.mcur_store.get_state()
            self.current_ready = _is_current_memory_ready(self.mcur_store, mcur_state)
            self.current_stale = self.mcur_store.is_stale(mcur_state)
            self.mcur_version = int(mcur_state.get("mcur_version", 0) or 0)
            self.mcur_loaded_at = time.time() if self.current_ready else 0.0
            yield self
        finally:
            self.runtime_session_id = original["runtime_session_id"]
            self.runtime_session_dir = original["runtime_session_dir"]
            self.interaction_cache = original["interaction_cache"]
            self.mst_store = original["mst_store"]
            self.mst_retriever = original["mst_retriever"]
            self.short_term_ready = original["short_term_ready"]
            self.mst_version = original["mst_version"]
            self.mst_loaded_at = original["mst_loaded_at"]
            self.mcur_store = original["mcur_store"]
            self.current_ready = original["current_ready"]
            self.current_stale = original["current_stale"]
            self.mcur_version = original["mcur_version"]
            self.mcur_loaded_at = original["mcur_loaded_at"]
            self.requested_session_id = original["requested_session_id"]
            self.day_context = original["day_context"]
            self.query_time_override = original["query_time_override"]

    def touch(self) -> None:
        self.last_accessed_at = time.time()

    def close(self) -> None:
        try:
            self.world_memory.cleanup()
        except Exception:
            pass

    def _memory_component_versions(self) -> dict[str, Any]:
        config = self.memory_config if isinstance(self.memory_config, dict) else {}
        lag = config.get("lag") if isinstance(config.get("lag"), dict) else {}
        return {
            "fast": config.get("latest_fast_ready_version") or config.get("latest_ready_memory_version") or self.latest_ready_memory_version,
            "visual": config.get("latest_visual_ready_version") or config.get("visual_version") or self.visual_version,
            "graph": config.get("latest_graph_ready_version") or config.get("graph_version"),
            "semantic": config.get("latest_semantic_ready_version") or config.get("semantic_version"),
            "semantic_lagging": bool(lag.get("semantic_lagging")),
            "graph_lagging": bool(lag.get("graph_lagging")),
            "visual_lagging": bool(lag.get("visual_lagging")),
            "readiness": config.get("readiness") if isinstance(config.get("readiness"), dict) else {},
            "worldmm_update_mode": config.get("worldmm_update_mode"),
            "latest_snapshot_version": config.get("latest_snapshot_version"),
            "latest_snapshot_path": config.get("latest_snapshot_path"),
        }

    def needs_reload(self) -> bool:
        try:
            mtime_changed = self.memory_config_path.stat().st_mtime > self.memory_config_mtime
        except FileNotFoundError:
            return True
        if not mtime_changed:
            return False
        config = read_json(self.memory_config_path, default={})
        if not isinstance(config, dict):
            return True
        latest_ready = _memory_version_from_config(config)
        if latest_ready and self.active_query_memory_version:
            if int(latest_ready) > int(self.active_query_memory_version):
                return True
        if latest_ready and not self.active_query_memory_version:
            return True
        old_components = self._memory_component_versions()
        component_pairs = (
            ("latest_fast_ready_version", "fast"),
            ("latest_visual_ready_version", "visual"),
            ("latest_graph_ready_version", "graph"),
            ("latest_semantic_ready_version", "semantic"),
        )
        for config_key, component_key in component_pairs:
            new_value = _safe_int(config.get(config_key), 0)
            old_value = _safe_int(old_components.get(component_key), 0)
            if new_value and new_value > old_value:
                return True
        return False

    def _refresh_short_term(self) -> dict[str, Any]:
        state = self.mst_store.get_state()
        ready = bool(state.get("short_term_ready") and self.mst_store.events_path.exists())
        version = int(state.get("mst_version", 0) or 0)
        if ready != self.short_term_ready or version != self.mst_version:
            self.short_term_ready = ready
            self.mst_version = version
            self.mst_retriever.refresh_if_needed()
            self.mst_loaded_at = time.time() if ready else 0.0
        return state

    def _refresh_current(self) -> dict[str, Any]:
        state = self.mcur_store.get_state()
        ready = _is_current_memory_ready(self.mcur_store, state)
        stale = self.mcur_store.is_stale(state)
        version = int(state.get("mcur_version", 0) or 0)
        if ready != self.current_ready or stale != self.current_stale or version != self.mcur_version:
            self.current_ready = ready
            self.current_stale = stale
            self.mcur_version = version
            self.mcur_loaded_at = time.time() if ready else 0.0
        return state

    def _retrieve_short_term(
        self,
        question: str,
        route_decision: dict[str, Any],
        cache_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._refresh_short_term()
        if not self.short_term_ready:
            return []
        memory_route = route_decision.get("memory_route", {}) or {}
        if not memory_route.get("use_short_term"):
            return []
        top_k = int((route_decision.get("short_term_policy") or {}).get("top_k") or route_decision.get("text_top_k") or 5)
        return self.mst_retriever.search(question, top_k=top_k, cache_context=cache_context)

    def _runtime_state(self, mst_state: dict[str, Any], mcur_state: dict[str, Any], interaction_enabled: bool) -> dict[str, Any]:
        mst_span = mst_state.get("active_time_span") or mst_state.get("archive_time_span")
        if not mst_span and mst_state.get("last_processed_time") is not None:
            mst_span = [0.0, mst_state.get("last_processed_time")]
        mlt_span = self.memory_config.get("time_span") or self.memory_config.get("video_time_span")
        return {
            "current_ready": self.current_ready,
            "current_stale": self.current_stale,
            "short_term_ready": self.short_term_ready,
            "long_term_ready": True,
            "visual_embedding_ready": self.visual_ready,
            "visual_ready": self.visual_ready,
            "cache_ready": bool(interaction_enabled),
            "mcur_time_span": [mcur_state.get("window_start_time"), mcur_state.get("window_end_time")] if mcur_state else None,
            "mst_time_span": mst_span,
            "mlt_time_span": mlt_span,
            "session_id": getattr(self, "requested_session_id", self.session_id),
        }

    def _route_and_plan(
        self,
        question: str,
        request_options: dict[str, Any],
        runtime_state: dict[str, Any],
        cache_context: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], int, int]:
        memory_router_start = time.perf_counter()
        memory_decision = MemoryRouter().route(
            question=question,
            request_options=request_options,
            runtime_state=runtime_state,
            cache_context=cache_context,
        )
        memory_router_ms = _ms(memory_router_start)
        planner_start = time.perf_counter()
        retrieval_plan = RetrievalPlanner().plan(
            memory_decision=memory_decision,
            request_options=request_options,
            runtime_state=runtime_state,
            cache_context=cache_context,
        )
        planner_ms = _ms(planner_start)
        route_decision = QueryRouter().route(
            question,
            request_options={**request_options, "retrieval_mode": retrieval_plan.get("retrieval_mode", request_options.get("retrieval_mode"))},
            session_context={
                "visual_ready": runtime_state.get("visual_embedding_ready"),
                "short_term_ready": runtime_state.get("short_term_ready"),
                "current_ready": runtime_state.get("current_ready"),
                "current_stale": runtime_state.get("current_stale"),
                "long_term_ready": runtime_state.get("long_term_ready"),
                "session_id": getattr(self, "requested_session_id", self.session_id),
            },
            cache_context=cache_context,
        )
        route_decision["query_type"] = memory_decision.get("query_type")
        route_decision["memory_route"] = memory_decision.get("memory_route", {})
        route_decision["fallback_order"] = memory_decision.get("fallback_order", [])
        route_decision["memory_priority"] = memory_decision.get("memory_priority", {})
        route_decision["memory_router_reason"] = memory_decision.get("reason", "")
        route_decision["retrieval_plan"] = retrieval_plan.get("retrieval_plan", {})
        plan_map = retrieval_plan.get("retrieval_plan", {}) or {}
        route_decision["short_term_policy"] = {
            "enabled": bool((plan_map.get("M_st") or {}).get("enabled")),
            "ready": bool(runtime_state.get("short_term_ready")),
            "reason": memory_decision.get("reason", ""),
            "top_k": int((plan_map.get("M_st") or {}).get("top_k") or request_options.get("top_k") or 5),
        }
        route_decision["current_policy"] = {
            "enabled": bool((plan_map.get("M_cur") or {}).get("enabled")),
            "ready": bool(runtime_state.get("current_ready")),
            "stale": bool(runtime_state.get("current_stale")),
            "reason": memory_decision.get("reason", ""),
            "top_k": 1,
        }
        route_decision["retrieval_mode"] = retrieval_plan.get("retrieval_mode") or route_decision.get("retrieval_mode")
        route_decision["retrieval_mode_source"] = retrieval_plan.get("retrieval_mode_source") or route_decision.get("retrieval_mode_source")
        route_decision["use_image_evidence"] = bool(retrieval_plan.get("use_image_evidence"))
        route_decision["use_image_evidence_source"] = retrieval_plan.get("use_image_evidence_source") or route_decision.get("use_image_evidence_source")
        route_decision["max_image_evidence"] = int(retrieval_plan.get("max_image_evidence") or 0)
        route_decision["candidate_budgets"] = retrieval_plan.get("candidate_budgets", {})
        route_decision["text_top_k"] = int(retrieval_plan.get("text_top_k") or route_decision.get("text_top_k") or request_options.get("text_top_k") or request_options.get("top_k") or 5)
        route_decision["visual_top_k"] = int(retrieval_plan.get("visual_top_k") or route_decision.get("visual_top_k") or request_options.get("visual_top_k") or 8)
        route_decision["final_evidence_k"] = int(retrieval_plan.get("final_evidence_k") or route_decision.get("final_evidence_k") or 4)
        route_decision["evidence_frames_k"] = int(retrieval_plan.get("evidence_frames_k") or route_decision.get("evidence_frames_k") or 5)
        route_decision["memory_router_decision"] = memory_decision
        route_decision.setdefault("warnings", [])
        for warning in memory_decision.get("warnings", []) + retrieval_plan.get("warnings", []):
            if warning not in route_decision["warnings"]:
                route_decision["warnings"].append(warning)
        return route_decision, retrieval_plan, memory_router_ms, planner_ms

    def _retrieve_current_results(
        self,
        question: str,
        route_decision: dict[str, Any],
        retrieval_plan: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        plan = (retrieval_plan.get("retrieval_plan") or {}).get("M_cur") or {}
        if not plan.get("enabled"):
            return [], {}, {}
        current_context = self.mcur_store.get_current_context()
        if not current_context.get("mcur_ready") or current_context.get("is_stale"):
            return [], current_context, {}
        selection = MCurFrameSelector().select_frames_for_query(
            current_context,
            question,
            max_images=plan.get("max_images") or route_decision.get("max_image_evidence") or 0,
            max_frames=plan.get("max_evidence_frames") or route_decision.get("evidence_frames_k") or 5,
        )
        return [{"current_context": current_context, "current_selection": selection}], current_context, selection

    def _fuse_memory_results(
        self,
        memory_results: dict[str, Any],
        route_decision: dict[str, Any],
        retrieval_plan: dict[str, Any],
        cache_context: dict[str, Any],
    ) -> dict[str, Any]:
        memory_decision = route_decision.get("memory_router_decision") or {
            "memory_route": route_decision.get("memory_route", {}),
            "memory_priority": route_decision.get("memory_priority", {}),
        }
        return MemoryFusion().fuse(
            memory_results=memory_results,
            memory_decision=memory_decision,
            retrieval_plan=retrieval_plan,
            query_type=str(route_decision.get("query_type") or "general_qa"),
            cache_context=cache_context,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        latency: dict[str, Any] | None = None,
        use_image_evidence: Any = "auto",
        max_image_frames: int = 4,
        retrieval_mode: str = "auto",
        max_image_evidence: int | None = None,
        text_top_k: int | None = None,
        visual_top_k: int = 8,
        final_evidence_k: int = 4,
        use_interaction_cache: bool = True,
        cache_mode: str = "auto",
        memory_mode: str = "auto",
        use_current: bool | None = None,
        use_short_term: bool | None = None,
        use_long_term: bool | None = None,
        debug_router: bool = False,
        stream_handler: Any = None,
        eval_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _eval_mark(eval_trace, "retrieval_started_at")
        cache_mode = str(cache_mode or "auto").strip().lower()
        if cache_mode not in {"auto", "off", "read_only", "write_only"}:
            cache_mode = "auto"
        interaction_enabled = (
            bool(use_interaction_cache)
            and _env_bool("WORLDMM_INTERACTION_CACHE_ENABLED", True)
            and cache_mode != "off"
        )
        cache_read_enabled = interaction_enabled and cache_mode in {"auto", "read_only"}
        cache_write_enabled = interaction_enabled and cache_mode in {"auto", "write_only"}
        cache_context: dict[str, Any] = {
            "cache_hit": False,
            "is_followup": False,
            "confidence": 0.0,
            "reason": "interaction cache disabled" if not interaction_enabled else "no cache context",
        }
        resolved_question = question
        if cache_read_enabled:
            try:
                cache_context = CoreferenceResolver().resolve(question, self.interaction_cache)
                resolved_question = str(cache_context.get("resolved_question") or question)
                if cache_context.get("confidence", 0.0) < 0.45:
                    resolved_question = question
            except Exception as exc:
                cache_context = {
                    "cache_hit": False,
                    "is_followup": False,
                    "confidence": 0.0,
                    "reason": f"cache resolver failed: {exc}",
                    "warnings": [str(exc)],
                }
        cache_context = _normalize_frame_timestamps_in_context(cache_context)

        mst_state = self._refresh_short_term()
        mcur_state = self._refresh_current()
        runtime_state = self._runtime_state(mst_state, mcur_state, interaction_enabled)
        request_options = {
            "memory_mode": memory_mode,
            "retrieval_mode": retrieval_mode,
            "use_image_evidence": use_image_evidence,
            "max_image_evidence": max_image_evidence,
            "top_k": top_k,
            "text_top_k": text_top_k,
            "visual_top_k": visual_top_k,
            "final_evidence_k": final_evidence_k,
            "use_current": use_current,
            "use_short_term": use_short_term,
            "use_long_term": use_long_term,
            "use_interaction_cache": interaction_enabled,
            "debug_router": debug_router,
        }
        route_decision, retrieval_plan, memory_router_ms, retrieval_planner_ms = self._route_and_plan(
            question=question,
            request_options=request_options,
            runtime_state=runtime_state,
            cache_context=cache_context,
        )
        router_ms = memory_router_ms + retrieval_planner_ms
        retrieval_mode = normalize_retrieval_mode(route_decision["retrieval_mode"])
        use_image_evidence = bool(route_decision["use_image_evidence"])
        max_image_evidence = int(route_decision.get("max_image_evidence") or 0)
        text_top_k = int(route_decision.get("text_top_k") or top_k)
        visual_top_k = int(route_decision.get("visual_top_k") or visual_top_k)
        final_evidence_k = int(route_decision.get("final_evidence_k") or final_evidence_k)
        memory_route = route_decision.get("memory_route") or {}
        retrieval_plan_map = retrieval_plan.get("retrieval_plan") or {}
        current_results, current_context, current_selection = self._retrieve_current_results(
            resolved_question,
            route_decision,
            retrieval_plan,
        )
        current_partial_diag = _diagnose_partial_memory(
            question=question,
            route_decision=route_decision,
            current_results=current_results,
            short_term_results=[],
            memory_config=self.memory_config,
            visual_ready=self.visual_ready,
            long_term_ready=bool(self.latest_ready_memory_version),
        )
        _apply_image_fallback_to_route(route_decision, current_partial_diag)
        use_image_evidence = bool(route_decision["use_image_evidence"])
        max_image_evidence = int(route_decision.get("max_image_evidence") or 0)
        current_only = bool(memory_route.get("use_current")) and not memory_route.get("use_short_term") and not memory_route.get("use_long_term")
        if current_only and (retrieval_mode == "current" or (route_decision.get("memory_route") or {}).get("use_current")):
            if self.current_ready and not self.current_stale:
                result = _answer_current_memory(
                    session_id=getattr(self, "requested_session_id", self.session_id),
                    session_dir=getattr(self, "runtime_session_dir", self.session_dir),
                    question=question,
                    resolved_question=resolved_question,
                    route_decision=route_decision,
                    cache_context=cache_context,
                    latency=dict(latency or {}),
                    router_ms=router_ms,
                    long_term_ready=True,
                    short_term_ready=self.short_term_ready,
                    cache_used=interaction_enabled,
                    cache_hit=bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                    cache_mode=cache_mode,
                    day_context=getattr(self, "day_context", None),
                    stream_handler=stream_handler,
                )
                result["visual_embedding_ready"] = self.visual_ready
                result["pipeline_mode"] = self.memory_config.get("pipeline_mode", os.getenv("WORLDMM_PIPELINE_MODE", "mst"))
                result["active_30s_source"] = self.memory_config.get("active_30s_source") or self.memory_config.get("worldmm_30s_input_source")
                result["episodic_source"] = self.memory_config.get("episodic_source")
                result["legacy_evidence_used"] = bool(self.memory_config.get("legacy_evidence_used") or self.memory_config.get("legacy_evidence_fallback_used"))
                result["retrieval_plan"] = retrieval_plan.get("retrieval_plan", {})
                result["memory_results"] = {
                    "current_results": current_results,
                    "short_term_results": [],
                    "long_term_results": [],
                    "cache_context": cache_context,
                }
                result["fusion_summary"] = {
                    "input_counts": {"M_cur": len(current_results), "M_st": 0, "M_lt": 0, "M_cache": 0},
                    "selected_memory_sources": ["M_cur"],
                    "dedup_removed": 0,
                    "final_evidence_count": len(current_results),
                }
                result["selected_evidence"] = []
                result.setdefault("latency", {})["memory_router_ms"] = memory_router_ms
                result["latency"]["retrieval_planner_ms"] = retrieval_planner_ms
                result["latency"]["memory_fusion_ms"] = 0
                self.recent_queries.append(
                    {
                        "question": question,
                        "resolved_question": resolved_question,
                        "answer": result.get("answer"),
                        "retrieved_memory_ids": [],
                        "timestamps": result.get("timestamps", []),
                        "evidence_frames": result.get("evidence_frames", []),
                        "latency": result.get("latency", {}),
                        "image_evidence_enabled": bool(result.get("use_image_evidence")),
                        "image_paths_used": result.get("raw", {}).get("image_paths_used", []),
                        "cache_used": bool(interaction_enabled),
                        "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                        "created_at": utc_now_iso(),
                    }
                )
                _attach_memory_completeness(
                    result,
                    current_partial_diag,
                    attached_image_count=int(result.get("sent_image_count") or 0),
                )
                return self._finalize_interaction_cache(
                    result=result,
                    question=question,
                    resolved_question=resolved_question,
                    cache_mode=cache_mode,
                    interaction_enabled=interaction_enabled,
                    cache_write_enabled=cache_write_enabled,
                    cache_context=cache_context,
                )
            warnings = route_decision.setdefault("warnings", [])
            message = "current memory is stale or unavailable; falling back to short-term memory"
            if message not in warnings:
                warnings.append(message)
            memory_route = route_decision.setdefault("memory_route", {})
            memory_route["use_current"] = False
            memory_route["use_short_term"] = True
            if retrieval_mode == "current":
                retrieval_mode = "hybrid" if self.short_term_ready else "text_only"
                route_decision["retrieval_mode"] = retrieval_mode
        short_term_start = time.perf_counter()
        short_term_results = self._retrieve_short_term(resolved_question, route_decision, cache_context)
        short_term_retrieval_ms = _ms(short_term_start)
        partial_diag = _diagnose_partial_memory(
            question=question,
            route_decision=route_decision,
            current_results=current_results,
            short_term_results=short_term_results,
            memory_config=self.memory_config,
            visual_ready=self.visual_ready,
            long_term_ready=bool(self.latest_ready_memory_version),
        )
        _apply_image_fallback_to_route(route_decision, partial_diag)
        if partial_diag.get("prefer_fast_local") and not partial_diag.get("long_term_event_ready"):
            memory_route = route_decision.setdefault("memory_route", {})
            memory_route["use_long_term"] = False
            route_decision["partial_memory_fast_path"] = True
            route_decision["partial_memory_fast_path_reason"] = partial_diag.get("image_fallback_reason")
        use_image_evidence = bool(route_decision["use_image_evidence"])
        max_image_evidence = int(route_decision.get("max_image_evidence") or 0)
        pack_result: dict[str, Any] = {}
        packer = EvidencePacker(
            session_dir=self.session_dir,
            runtime_session_dir=getattr(self, "runtime_session_dir", self.session_dir),
        )
        memory_route = route_decision.get("memory_route") or {}
        if not memory_route.get("use_long_term"):
            fusion_start = time.perf_counter()
            memory_results = {
                "current_results": current_results,
                "short_term_results": short_term_results,
                "long_term_results": [],
                "text_results": [],
                "visual_results": [],
                "fused_results": [],
                "cache_context": cache_context,
            }
            fusion_result = self._fuse_memory_results(memory_results, route_decision, retrieval_plan, cache_context)
            memory_fusion_ms = _ms(fusion_start)
            pack_start = time.perf_counter()
            pack_result = packer.pack(
                query=question,
                route_decision=route_decision,
                retrieval_result={
                    **memory_results,
                    "current_context": current_context,
                    "current_selection": current_selection,
                    "evidence_frames": current_selection.get("evidence_frames", []) if current_selection else [],
                    "final_evidence": fusion_result.get("final_evidence", []),
                    "fusion_summary": fusion_result.get("fusion_summary", {}),
                },
                cache_context=cache_context,
            )
            pack_ms = _ms(pack_start)
            generation_start = time.perf_counter()
            final_raw_qa = self._generate_answer_from_evidence(
                question=resolved_question,
                retrieval_mode="memory_router",
                text_results=[],
                visual_results=[],
                fused_results=[],
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                selected_image_paths=pack_result.get("selected_image_paths_for_mllm"),
                evidence_pack_summary=pack_result.get("evidence_pack_summary"),
                short_term_results=short_term_results,
                selected_evidence=pack_result.get("selected_evidence", []),
                stream_handler=stream_handler,
            )
            generation_ms = _ms(generation_start)
            image_count = sum(final_raw_qa.visual_event_image_counts.values())
            selected_evidence = pack_result.get("selected_evidence", [])
            timestamps = [
                {"start": ev.get("start_time"), "end": ev.get("end_time")}
                for ev in selected_evidence
                if ev.get("start_time") is not None or ev.get("end_time") is not None
            ]
            latency = dict(latency or {})
            latency.update({
                "router_ms": router_ms,
                "memory_router_ms": memory_router_ms,
                "retrieval_planner_ms": retrieval_planner_ms,
                "short_term_retrieval_ms": short_term_retrieval_ms,
                "text_retrieval_ms": 0,
                "visual_retrieval_ms": 0,
                "fusion_ms": 0,
                "memory_fusion_ms": memory_fusion_ms,
                "evidence_pack_ms": pack_ms,
                "generation_ms": generation_ms,
                "query_ms": short_term_retrieval_ms + memory_fusion_ms + pack_ms + generation_ms,
                "image_evidence_enabled": bool(use_image_evidence),
                "image_blocks_count": image_count,
            })
            result = {
                "status": "ok",
                "session_id": self.session_id,
                "question": question,
                "resolved_question": resolved_question,
                "cache_used": bool(interaction_enabled),
                "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                "cache_mode": cache_mode,
                "cache_context": cache_context,
                "query_type": route_decision.get("query_type"),
                "memory_route": memory_route,
                "retrieval_plan": retrieval_plan.get("retrieval_plan", {}),
                "route_decision": route_decision,
                "retrieval_mode": retrieval_mode,
                "retrieval_mode_source": route_decision.get("retrieval_mode_source"),
                "long_term_ready": True,
                "short_term_ready": self.short_term_ready,
                "short_term_only": bool(short_term_results and not current_results),
                "short_term_results": short_term_results,
                "current_ready": self.current_ready,
                "current_used": bool(current_results),
                "current_stale": self.current_stale,
                "visual_embedding_ready": self.visual_ready,
                "visual_fallback": False,
                "use_image_evidence": bool(use_image_evidence),
                "use_image_evidence_source": route_decision.get("use_image_evidence_source"),
                "max_image_evidence": max_image_evidence,
                "sent_image_count": image_count,
                "answer": final_raw_qa.answer,
                "timestamps": timestamps,
                "evidence_frames": pack_result.get("selected_evidence_frames", []),
                "retrieved_memories": [],
                "supporting_semantic_facts": [],
                "text_results": [],
                "visual_results": [],
                "fused_results": [],
                "memory_results": memory_results,
                "fusion_summary": fusion_result.get("fusion_summary", {}),
                "selected_evidence": selected_evidence,
                "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
                "warnings": route_decision.get("warnings", []),
                "latency": latency,
                "raw": {
                    "route_decision": route_decision,
                    "retrieval_plan": retrieval_plan.get("retrieval_plan", {}),
                    "runtime_state": runtime_state,
                    "memory_results": memory_results,
                    "fusion_summary": fusion_result.get("fusion_summary", {}),
                    "cache_context": cache_context,
                    "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
                    "image_selection_summary": pack_result.get("image_selection_summary", {}),
                    "model_response_text": final_raw_qa.model_response_text,
                    "error_debug": final_raw_qa.error_debug,
                    "fallback_used": final_raw_qa.fallback_used,
                    "llm_debug": final_raw_qa.llm_debug,
                },
            }
            _attach_memory_completeness(
                result,
                partial_diag,
                attached_image_count=sum(final_raw_qa.visual_event_image_counts.values()),
            )
            self.recent_queries.append(
                {
                    "question": question,
                    "resolved_question": resolved_question,
                    "answer": final_raw_qa.answer,
                    "retrieved_memory_ids": [],
                    "timestamps": timestamps,
                    "evidence_frames": result["evidence_frames"],
                    "latency": latency,
                    "image_evidence_enabled": bool(use_image_evidence),
                    "image_paths_used": final_raw_qa.image_paths_used,
                    "cache_used": bool(interaction_enabled),
                    "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                    "created_at": utc_now_iso(),
                }
            )
            return self._finalize_interaction_cache(
                result=result,
                question=question,
                resolved_question=resolved_question,
                cache_mode=cache_mode,
                interaction_enabled=interaction_enabled,
                cache_write_enabled=cache_write_enabled,
                cache_context=cache_context,
            )
        if retrieval_mode == "visual_only":
            result = self._query_visual_only(
                question=question,
                search_question=resolved_question,
                top_k=top_k,
                latency=latency,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                visual_top_k=visual_top_k,
                final_evidence_k=final_evidence_k,
                route_decision=route_decision,
                router_ms=router_ms,
                packer=packer,
                cache_context=cache_context,
                short_term_results=short_term_results,
                stream_handler=stream_handler,
            )
            return self._finalize_interaction_cache(
                result=result,
                question=question,
                resolved_question=resolved_question,
                cache_mode=cache_mode,
                interaction_enabled=interaction_enabled,
                cache_write_enabled=cache_write_enabled,
                cache_context=cache_context,
            )

        query_rag = _query_rag_helpers()
        _, _, _, _, transform_timestamp = _worldmm_classes(self.long_term_retrieval_scheme)
        query_start = time.perf_counter()
        self.touch()
        self.query_count += 1
        self.world_memory.set_retrieval_top_k(
            episodic=text_top_k,
            semantic=max(text_top_k, 8),
            visual=min(max(text_top_k, 1), 5),
        )
        query_time_override = getattr(self, "query_time_override", None)
        effective_until_date = str(
            (query_time_override or {}).get("until_date")
            or self.query_args.get("until_date")
            or "DAY1"
        )
        effective_until_time = str(
            (query_time_override or {}).get("until_time")
            or self.query_args.get("until_time")
            or "23595999"
        )
        until_time = query_rag.build_until_timestamp(effective_until_date, effective_until_time)
        day_context_block = build_day_context_block(getattr(self, "day_context", None))
        qa_result = self.world_memory.answer(
            query=resolved_question,
            choices=None,
            until_time=until_time,
            answer_mode="open_ended",
            # Only retrieve long-term event anchors here; EvidencePacker selects final images later.
            use_image_evidence=False,
            max_image_frames=0,
            stream_handler=stream_handler,
            prompt_context=day_context_block or None,
            generate_answer=False,
        )
        worldmm_answer_ms = _ms(query_start)
        long_term_timing = _qa_timing_ms(qa_result)
        long_term_rag_ms = int(long_term_timing.get("retrieval_ms") or 0)
        long_term_selector_ms = int(long_term_timing.get("selector_ms") or 0)
        long_term_pack_ms = int(long_term_timing.get("pack_ms") or 0)
        long_term_retrieval_ms = long_term_rag_ms + long_term_selector_ms + long_term_pack_ms
        text_query_ms = long_term_retrieval_ms
        episodic_retrieval_debug = getattr(getattr(self.world_memory, "episodic_memory", None), "last_retrieval_debug", {})

        selected_events = query_rag.summarize_selected_events(self.world_memory, qa_result.selected_doc_ids)
        supporting_semantic_facts = query_rag.summarize_semantic_facts(self.world_memory, qa_result.semantic_fact_ids)
        retrieved_items_summary = query_rag.summarize_retrieved_items(qa_result)
        timestamps = [_event_timestamps(event) for event in selected_events if isinstance(event, dict)]
        evidence_frames = self._build_evidence_frames(selected_events)
        retrieved_memories = self._build_retrieved_memories(selected_events)
        text_results = self._build_text_results(selected_events, retrieved_memories)
        visual_results: list[dict[str, Any]] = []
        fused_results: list[dict[str, Any]] = []
        memory_results: dict[str, Any] = {}
        fusion_result: dict[str, Any] = {"final_evidence": [], "fusion_summary": {}}
        memory_fusion_ms = 0
        visual_fallback = False
        visual_retrieval_ms = 0
        fusion_ms = 0
        final_answer = ""
        final_raw_qa = qa_result
        generation_ms: int | None = None

        if retrieval_mode == "hybrid":
            visual_start = time.perf_counter()
            if self.visual_ready:
                visual_results = self._retrieve_visual(resolved_question, visual_top_k)
            else:
                visual_fallback = True
            visual_retrieval_ms = _ms(visual_start)
            fusion_start = time.perf_counter()
            fused_results = self._fuse_results(
                text_results,
                visual_results,
                final_evidence_k=final_evidence_k,
                cache_context=cache_context,
            )
            evidence_frames = self._merge_evidence_frames(evidence_frames, fused_results)
            fusion_ms = _ms(fusion_start)
            memory_fusion_start = time.perf_counter()
            memory_results = {
                "current_results": current_results,
                "short_term_results": short_term_results,
                "long_term_results": fused_results or text_results,
                "text_results": text_results,
                "visual_results": visual_results,
                "fused_results": fused_results,
                "cache_context": cache_context,
            }
            fusion_result = self._fuse_memory_results(memory_results, route_decision, retrieval_plan, cache_context)
            memory_fusion_ms = _ms(memory_fusion_start)
            pack_start = time.perf_counter()
            pack_result = packer.pack(
                query=question,
                route_decision=route_decision,
                retrieval_result={
                    "current_context": current_context,
                    "current_selection": current_selection,
                    "text_results": text_results,
                    "visual_results": visual_results,
                    "fused_results": fused_results,
                    "evidence_frames": evidence_frames,
                    "short_term_results": short_term_results,
                    "final_evidence": fusion_result.get("final_evidence", []),
                    "fusion_summary": fusion_result.get("fusion_summary", {}),
                },
                cache_context=cache_context,
            )
            text_results = pack_result["packed_text_results"]
            fused_results = pack_result["packed_fused_results"]
            evidence_frames = pack_result["selected_evidence_frames"]
            pack_ms = _ms(pack_start)
            if visual_results or short_term_results or current_results:
                generation_start = time.perf_counter()
                final_raw_qa = self._generate_answer_from_evidence(
                    question=resolved_question,
                    retrieval_mode=retrieval_mode,
                    text_results=text_results,
                    visual_results=visual_results,
                    fused_results=fused_results,
                    use_image_evidence=use_image_evidence,
                    max_image_evidence=max_image_evidence,
                    selected_image_paths=pack_result.get("selected_image_paths_for_mllm"),
                    evidence_pack_summary=pack_result.get("evidence_pack_summary"),
                    short_term_results=short_term_results,
                    selected_evidence=pack_result.get("selected_evidence", []),
                    stream_handler=stream_handler,
                )
                generation_ms = _ms(generation_start)
                final_answer = final_raw_qa.answer
        else:
            memory_fusion_start = time.perf_counter()
            memory_results = {
                "current_results": current_results,
                "short_term_results": short_term_results,
                "long_term_results": text_results,
                "text_results": text_results,
                "visual_results": visual_results,
                "fused_results": fused_results,
                "cache_context": cache_context,
            }
            fusion_result = self._fuse_memory_results(memory_results, route_decision, retrieval_plan, cache_context)
            memory_fusion_ms = _ms(memory_fusion_start)
            pack_start = time.perf_counter()
            pack_result = packer.pack(
                query=question,
                route_decision=route_decision,
                retrieval_result={
                    "current_context": current_context,
                    "current_selection": current_selection,
                    "text_results": text_results,
                    "visual_results": visual_results,
                    "fused_results": fused_results,
                    "evidence_frames": evidence_frames,
                    "short_term_results": short_term_results,
                    "final_evidence": fusion_result.get("final_evidence", []),
                    "fusion_summary": fusion_result.get("fusion_summary", {}),
                },
                cache_context=cache_context,
            )
            text_results = pack_result["packed_text_results"]
            evidence_frames = pack_result["selected_evidence_frames"]
            pack_ms = _ms(pack_start)
            if short_term_results or current_results:
                generation_start = time.perf_counter()
                final_raw_qa = self._generate_answer_from_evidence(
                    question=resolved_question,
                    retrieval_mode=f"{retrieval_mode}+short_term",
                    text_results=text_results,
                    visual_results=[],
                    fused_results=[],
                    use_image_evidence=use_image_evidence,
                    max_image_evidence=max_image_evidence,
                    selected_image_paths=pack_result.get("selected_image_paths_for_mllm"),
                    evidence_pack_summary=pack_result.get("evidence_pack_summary"),
                    short_term_results=short_term_results,
                    selected_evidence=pack_result.get("selected_evidence", []),
                    stream_handler=stream_handler,
                )
                generation_ms = _ms(generation_start)
                final_answer = final_raw_qa.answer

        if (
            (not str(final_answer or "").strip() or str(final_answer).strip() == "Unable to generate answer")
            and (text_results or fused_results or visual_results or short_term_results)
        ):
            primary_error_debug = final_raw_qa.error_debug
            primary_traceback = ""
            if isinstance(final_raw_qa.llm_debug, dict):
                primary_traceback = str(
                    final_raw_qa.llm_debug.get("traceback")
                    or final_raw_qa.llm_debug.get("text_only_fallback_traceback")
                    or ""
                )
            fallback_generation_start = time.perf_counter()
            fallback_raw = self._generate_answer_from_evidence(
                question=resolved_question,
                retrieval_mode=f"{retrieval_mode}_text_fallback",
                text_results=text_results,
                visual_results=visual_results,
                fused_results=fused_results,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                selected_image_paths=pack_result.get("selected_image_paths_for_mllm") if use_image_evidence else [],
                evidence_pack_summary=pack_result.get("evidence_pack_summary"),
                short_term_results=short_term_results,
                selected_evidence=pack_result.get("selected_evidence", []),
                stream_handler=stream_handler,
            )
            generation_ms = _ms(fallback_generation_start)
            fallback_raw.fallback_used = True
            if primary_error_debug:
                if str(fallback_raw.error_debug or "").strip():
                    fallback_raw.error_debug = "simplified_text_fallback_error_debug:\n" + str(fallback_raw.error_debug)
                elif str(fallback_raw.answer or "").strip() == "Unable to generate answer":
                    fallback_raw.error_debug = "worldmm_primary_error_debug:\n" + str(primary_error_debug)
            if isinstance(fallback_raw.llm_debug, dict):
                fallback_raw.llm_debug["worldmm_primary_error_debug"] = primary_error_debug
                if primary_traceback:
                    fallback_raw.llm_debug["worldmm_primary_traceback"] = primary_traceback
            final_raw_qa = fallback_raw
            final_answer = final_raw_qa.answer

        latency = dict(latency or {})
        latency["query_ms"] = text_query_ms + visual_retrieval_ms + fusion_ms
        latency["long_term_pipeline_ms"] = long_term_retrieval_ms + long_term_selector_ms + long_term_pack_ms
        latency["router_ms"] = router_ms
        latency["memory_router_ms"] = memory_router_ms
        latency["retrieval_planner_ms"] = retrieval_planner_ms
        latency["text_retrieval_ms"] = text_query_ms
        latency["worldmm_answer_ms"] = worldmm_answer_ms
        latency["long_term_rag_ms"] = long_term_rag_ms
        latency["long_term_retrieval_ms"] = long_term_retrieval_ms
        latency["long_term_selector_ms"] = long_term_selector_ms
        latency["long_term_pack_ms"] = long_term_pack_ms
        latency["visual_retrieval_ms"] = visual_retrieval_ms
        latency["short_term_retrieval_ms"] = short_term_retrieval_ms
        latency["fusion_ms"] = fusion_ms
        latency["memory_fusion_ms"] = memory_fusion_ms
        latency["evidence_pack_ms"] = pack_ms
        latency["generation_ms"] = generation_ms
        latency["final_generation_ms"] = generation_ms
        latency["answer_generation_ms"] = generation_ms
        latency["image_evidence_enabled"] = bool(use_image_evidence)
        latency["image_blocks_count"] = sum(final_raw_qa.visual_event_image_counts.values())

        raw_traceback = ""
        if isinstance(final_raw_qa.llm_debug, dict):
            raw_traceback = str(
                final_raw_qa.llm_debug.get("traceback")
                or final_raw_qa.llm_debug.get("worldmm_primary_traceback")
                or ""
            )

        raw = {
            "answer_mode_used": final_raw_qa.answer_mode,
            "qa_template_name": final_raw_qa.qa_template_name,
            "num_rounds": final_raw_qa.num_rounds,
            "round_history": qa_result.round_history,
            "selected_event_doc_ids": qa_result.selected_doc_ids,
            "selector_reason": qa_result.selector_reason,
            "supporting_semantic_fact_ids": qa_result.semantic_fact_ids,
            "visual_event_image_counts": final_raw_qa.visual_event_image_counts,
            "image_evidence_enabled": final_raw_qa.image_evidence_enabled,
            "image_mode_used": bool(use_image_evidence and sum(final_raw_qa.visual_event_image_counts.values()) > 0),
            "image_blocks_count": sum(final_raw_qa.visual_event_image_counts.values()),
            "image_paths_used": final_raw_qa.image_paths_used,
            "model_response_text": final_raw_qa.model_response_text,
            "error_debug": final_raw_qa.error_debug,
            "traceback": raw_traceback,
            "primary_error_debug": (
                final_raw_qa.llm_debug.get("worldmm_primary_error_debug", "")
                if isinstance(final_raw_qa.llm_debug, dict)
                else ""
            ),
            "fallback_used": final_raw_qa.fallback_used,
            "llm_debug": final_raw_qa.llm_debug,
            "router_input_summary": {
                "retrieval_mode": retrieval_mode,
                "use_image_evidence": use_image_evidence,
                "top_k": top_k,
                "visual_ready": self.visual_ready,
                "short_term_ready": self.short_term_ready,
                "current_ready": self.current_ready,
                "original_question": question,
                "resolved_question": resolved_question,
            },
            "route_decision": route_decision,
            "retrieval_plan": retrieval_plan.get("retrieval_plan", {}),
            "runtime_state": runtime_state,
            "memory_results": memory_results,
            "fusion_summary": fusion_result.get("fusion_summary", {}),
            "selected_evidence": pack_result.get("selected_evidence", []),
            "cache_context": cache_context,
            "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
            "image_selection_summary": pack_result.get("image_selection_summary", {}),
            "episodic_retrieval_debug": episodic_retrieval_debug,
            "pipeline": {
                "pipeline_mode": self.memory_config.get("pipeline_mode", os.getenv("WORLDMM_PIPELINE_MODE", "mst")),
                "active_30s_source": self.memory_config.get("active_30s_source") or self.memory_config.get("worldmm_30s_input_source"),
                "episodic_source": self.memory_config.get("episodic_source"),
                "legacy_evidence_used": bool(self.memory_config.get("legacy_evidence_used") or self.memory_config.get("legacy_evidence_fallback_used")),
            },
            "mst_state": mst_state,
            "mcur_state": mcur_state,
            "selected_events": selected_events,
            "supporting_semantic_facts": supporting_semantic_facts,
            "retrieved_items_summary": retrieved_items_summary,
            "until_timestamp": until_time,
            "until_timestamp_str": transform_timestamp(str(until_time)),
            "query_time_override": query_time_override,
            "effective_until_date": effective_until_date,
            "effective_until_time": effective_until_time,
        }
        if day_context_block:
            raw["day_context_prompt"] = {
                "injected": True,
                "content": day_context_block,
            }
        result = {
            "status": "ok",
            "session_id": getattr(self, "requested_session_id", self.session_id),
            "question": question,
            "resolved_question": resolved_question,
            "cache_used": bool(interaction_enabled),
            "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
            "cache_mode": cache_mode,
            "cache_context": cache_context,
            "query_type": route_decision.get("query_type"),
            "memory_route": route_decision.get("memory_route", {}),
            "retrieval_plan": retrieval_plan.get("retrieval_plan", {}),
            "route_decision": route_decision,
            "retrieval_mode": retrieval_mode,
            "retrieval_mode_source": route_decision.get("retrieval_mode_source"),
            "pipeline_mode": self.memory_config.get("pipeline_mode", os.getenv("WORLDMM_PIPELINE_MODE", "mst")),
            "active_30s_source": self.memory_config.get("active_30s_source") or self.memory_config.get("worldmm_30s_input_source"),
            "episodic_source": self.memory_config.get("episodic_source"),
            "legacy_evidence_used": bool(self.memory_config.get("legacy_evidence_used") or self.memory_config.get("legacy_evidence_fallback_used")),
            "long_term_ready": True,
            "short_term_ready": self.short_term_ready,
            "short_term_only": False,
            "short_term_results": short_term_results,
            "current_ready": self.current_ready,
            "current_used": bool(current_results),
            "current_stale": self.current_stale,
            "visual_embedding_ready": self.visual_ready,
            "visual_fallback": visual_fallback,
            "use_image_evidence": bool(use_image_evidence),
            "use_image_evidence_source": route_decision.get("use_image_evidence_source"),
            "max_image_evidence": max_image_evidence,
            "sent_image_count": sum(final_raw_qa.visual_event_image_counts.values()),
            "answer": final_answer,
            "timestamps": timestamps,
            "evidence_frames": evidence_frames,
            "retrieved_memories": retrieved_memories,
            "supporting_semantic_facts": supporting_semantic_facts,
            "text_results": text_results,
            "visual_results": visual_results,
            "fused_results": fused_results,
            "memory_results": memory_results,
            "fusion_summary": fusion_result.get("fusion_summary", {}),
            "selected_evidence": pack_result.get("selected_evidence", []),
            "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
            "episodic_retrieval_debug": episodic_retrieval_debug,
            "warnings": route_decision.get("warnings", []),
            "latency": latency,
            "raw": raw,
            "timing": {
                "long_term": long_term_timing,
                "long_term_retrieval_ms": long_term_retrieval_ms,
                "final_generation_ms": generation_ms,
            },
        }
        _attach_memory_completeness(
            result,
            partial_diag,
            attached_image_count=sum(final_raw_qa.visual_event_image_counts.values()),
        )
        self.recent_queries.append(
            {
                "question": question,
                "resolved_question": resolved_question,
                "answer": final_answer,
                "retrieved_memory_ids": [m["memory_id"] for m in retrieved_memories],
                "timestamps": timestamps,
                "evidence_frames": evidence_frames,
                "latency": latency,
                "image_evidence_enabled": bool(use_image_evidence),
                "image_paths_used": final_raw_qa.image_paths_used,
                "cache_used": bool(interaction_enabled),
                "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                "created_at": utc_now_iso(),
            }
        )
        return self._finalize_interaction_cache(
            result=result,
            question=question,
            resolved_question=resolved_question,
            cache_mode=cache_mode,
            interaction_enabled=interaction_enabled,
            cache_write_enabled=cache_write_enabled,
            cache_context=cache_context,
        )

    def _finalize_interaction_cache(
        self,
        result: dict[str, Any],
        question: str,
        resolved_question: str,
        cache_mode: str,
        interaction_enabled: bool,
        cache_write_enabled: bool,
        cache_context: dict[str, Any],
    ) -> dict[str, Any]:
        cache_update = {
            "updated": False,
            "reason": "interaction cache disabled" if not interaction_enabled else "cache_mode does not allow writes",
        }
        if cache_write_enabled and result.get("status") == "ok":
            try:
                cache_update = self.interaction_cache.update_from_query_result(
                    question=question,
                    resolved_question=resolved_question,
                    result=result,
                )
            except Exception as exc:
                cache_update = {"updated": False, "error": str(exc)}
        result["cache_used"] = bool(interaction_enabled)
        result["cache_hit"] = bool(cache_context.get("cache_hit") and cache_context.get("is_followup"))
        result["cache_mode"] = cache_mode
        result["resolved_question"] = resolved_question
        cache_context = _normalize_frame_timestamps_in_context(cache_context)
        result["cache_context"] = cache_context
        result["cache_update"] = cache_update
        result.setdefault("memory_component_versions", self._memory_component_versions())
        result.setdefault("raw", {})["cache_context"] = cache_context
        result["raw"]["cache_update"] = cache_update
        result["raw"].setdefault("memory_component_versions", self._memory_component_versions())
        return result

    def _query_visual_only(
        self,
        question: str,
        search_question: str,
        top_k: int,
        latency: dict[str, Any] | None,
        use_image_evidence: bool,
        max_image_evidence: int,
        visual_top_k: int,
        final_evidence_k: int,
        route_decision: dict[str, Any],
        router_ms: int,
        packer: EvidencePacker,
        cache_context: dict[str, Any] | None = None,
        short_term_results: list[dict[str, Any]] | None = None,
        stream_handler: Any = None,
    ) -> dict[str, Any]:
        short_term_results = short_term_results or []
        if not self.visual_ready:
            result = self.query(
                question=question,
                top_k=top_k,
                latency=latency,
                use_image_evidence=use_image_evidence,
                max_image_frames=max_image_evidence,
                retrieval_mode="text_only",
                use_interaction_cache=False,
                cache_mode="off",
            )
            result["retrieval_mode"] = "visual_only"
            result["visual_fallback"] = True
            result["visual_embedding_ready"] = False
            result["visual_results"] = []
            result["fused_results"] = []
            return result

        self.touch()
        self.query_count += 1
        visual_start = time.perf_counter()
        visual_results = self._retrieve_visual(search_question, max(visual_top_k, top_k))
        visual_retrieval_ms = _ms(visual_start)
        fusion_start = time.perf_counter()
        fused_results = self._fuse_results([], visual_results, final_evidence_k=final_evidence_k, cache_context=cache_context)
        evidence_frames = self._merge_evidence_frames([], fused_results)
        fusion_ms = _ms(fusion_start)
        pack_start = time.perf_counter()
        pack_result = packer.pack(
            query=question,
            route_decision=route_decision,
            retrieval_result={
                "text_results": [],
                "visual_results": visual_results,
                "fused_results": fused_results,
                "evidence_frames": evidence_frames,
                "short_term_results": short_term_results,
            },
            cache_context=cache_context,
        )
        fused_results = pack_result["packed_fused_results"]
        evidence_frames = pack_result["selected_evidence_frames"]
        pack_ms = _ms(pack_start)
        qa_start = time.perf_counter()
        final_raw_qa = self._generate_answer_from_evidence(
            question=search_question,
            retrieval_mode="visual_only",
            text_results=[],
            visual_results=visual_results,
            fused_results=fused_results,
            use_image_evidence=use_image_evidence,
            max_image_evidence=max_image_evidence,
            selected_image_paths=pack_result.get("selected_image_paths_for_mllm"),
            evidence_pack_summary=pack_result.get("evidence_pack_summary"),
            short_term_results=short_term_results,
            stream_handler=stream_handler,
        )
        generation_ms = _ms(qa_start)
        timestamps = self._timestamps_from_visual_results(fused_results)
        retrieved_memories = self._retrieved_memories_from_fused(fused_results)
        image_count = sum(final_raw_qa.visual_event_image_counts.values())
        latency = dict(latency or {})
        latency.update({
            "query_ms": visual_retrieval_ms + fusion_ms + generation_ms,
            "router_ms": router_ms,
            "text_retrieval_ms": 0,
            "visual_retrieval_ms": visual_retrieval_ms,
            "fusion_ms": fusion_ms,
            "evidence_pack_ms": pack_ms,
            "generation_ms": generation_ms,
            "image_evidence_enabled": bool(use_image_evidence),
            "image_blocks_count": image_count,
        })
        result = {
            "status": "ok",
            "session_id": self.session_id,
            "question": question,
            "resolved_question": search_question,
            "cache_used": bool((cache_context or {}).get("cache_hit")),
            "cache_hit": bool((cache_context or {}).get("cache_hit") and (cache_context or {}).get("is_followup")),
            "cache_context": cache_context or {},
            "query_type": route_decision.get("query_type"),
            "memory_route": route_decision.get("memory_route", {}),
            "retrieval_plan": route_decision.get("retrieval_plan", {}),
            "route_decision": route_decision,
            "retrieval_mode": "visual_only",
            "retrieval_mode_source": route_decision.get("retrieval_mode_source"),
            "pipeline_mode": self.memory_config.get("pipeline_mode", os.getenv("WORLDMM_PIPELINE_MODE", "mst")),
            "active_30s_source": self.memory_config.get("active_30s_source") or self.memory_config.get("worldmm_30s_input_source"),
            "episodic_source": self.memory_config.get("episodic_source"),
            "legacy_evidence_used": bool(self.memory_config.get("legacy_evidence_used") or self.memory_config.get("legacy_evidence_fallback_used")),
            "visual_embedding_ready": self.visual_ready,
            "long_term_ready": True,
            "short_term_ready": self.short_term_ready,
            "short_term_only": False,
            "short_term_results": short_term_results,
            "visual_fallback": False,
            "use_image_evidence": bool(use_image_evidence),
            "use_image_evidence_source": route_decision.get("use_image_evidence_source"),
            "max_image_evidence": max_image_evidence,
            "sent_image_count": image_count,
            "answer": final_raw_qa.answer,
            "timestamps": timestamps,
            "evidence_frames": evidence_frames,
            "retrieved_memories": retrieved_memories,
            "supporting_semantic_facts": [],
            "text_results": [],
            "visual_results": visual_results,
            "fused_results": fused_results,
            "memory_results": {
                "current_results": [],
                "short_term_results": short_term_results,
                "long_term_results": fused_results or visual_results,
                "cache_context": cache_context or {},
            },
            "fusion_summary": {
                "selected_memory_sources": ["M_lt"] if (fused_results or visual_results) else [],
                "final_evidence_count": len(fused_results),
                "dedup_removed": 0,
            },
            "selected_evidence": pack_result.get("selected_evidence", []),
            "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
            "latency": latency,
            "raw": {
                "answer_mode_used": final_raw_qa.answer_mode,
                "qa_template_name": final_raw_qa.qa_template_name,
                "num_rounds": final_raw_qa.num_rounds,
                "round_history": final_raw_qa.round_history,
                "selected_event_doc_ids": final_raw_qa.selected_doc_ids,
                "selector_reason": final_raw_qa.selector_reason,
                "supporting_semantic_fact_ids": final_raw_qa.semantic_fact_ids,
                "visual_event_image_counts": final_raw_qa.visual_event_image_counts,
                "image_evidence_enabled": final_raw_qa.image_evidence_enabled,
                "image_mode_used": bool(use_image_evidence and image_count > 0),
                "image_blocks_count": image_count,
                "image_paths_used": final_raw_qa.image_paths_used,
                "model_response_text": final_raw_qa.model_response_text,
                "error_debug": final_raw_qa.error_debug,
                "fallback_used": final_raw_qa.fallback_used,
                "llm_debug": final_raw_qa.llm_debug,
                "router_input_summary": {
                    "retrieval_mode": "visual_only",
                    "use_image_evidence": use_image_evidence,
                    "visual_ready": self.visual_ready,
                    "short_term_ready": self.short_term_ready,
                    "original_question": question,
                    "resolved_question": search_question,
                },
                "route_decision": route_decision,
                "retrieval_plan": route_decision.get("retrieval_plan", {}),
                "cache_context": cache_context or {},
                "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
                "image_selection_summary": pack_result.get("image_selection_summary", {}),
            },
        }
        _emit_stream_event(
            stream_handler,
            {
                "type": "final",
                "stage": "complete",
                "text": final_raw_qa.answer,
                "answer": final_raw_qa.answer,
                "raw": result.get("raw", {}),
                "latency": latency,
            },
        )
        self.recent_queries.append({
            "question": question,
            "resolved_question": search_question,
            "answer": final_raw_qa.answer,
            "retrieved_memory_ids": [m["memory_id"] for m in retrieved_memories],
            "timestamps": timestamps,
            "evidence_frames": evidence_frames,
            "latency": latency,
            "image_evidence_enabled": bool(use_image_evidence),
            "image_paths_used": final_raw_qa.image_paths_used,
            "cache_hit": bool((cache_context or {}).get("cache_hit") and (cache_context or {}).get("is_followup")),
            "created_at": utc_now_iso(),
        })
        return result

    def _generate_answer_from_evidence(
        self,
        question: str,
        retrieval_mode: str,
        text_results: list[dict[str, Any]],
        visual_results: list[dict[str, Any]],
        fused_results: list[dict[str, Any]],
        use_image_evidence: bool,
        max_image_evidence: int,
        selected_image_paths: list[str] | None = None,
        evidence_pack_summary: dict[str, Any] | None = None,
        short_term_results: list[dict[str, Any]] | None = None,
        selected_evidence: list[dict[str, Any]] | None = None,
        stream_handler: Any = None,
    ) -> Any:
        from worldmm.memory.utils import QAResult, RetrievedItem

        prompt_text = self._build_visual_answer_prompt(
            question=question,
            retrieval_mode=retrieval_mode,
            text_results=text_results,
            visual_results=visual_results,
            fused_results=fused_results,
            short_term_results=short_term_results or [],
            selected_evidence=selected_evidence or [],
        )
        image_paths_used: list[str] = []
        image_warnings: list[str] = []
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if use_image_evidence:
            if selected_image_paths is not None:
                image_paths_used = list(selected_image_paths)[:max_image_evidence]
            else:
                image_paths_used, image_warnings = self._select_generation_image_paths(
                    visual_results=visual_results,
                    fused_results=fused_results,
                    max_image_evidence=max_image_evidence,
                )
            runtime_dir = getattr(self, "runtime_session_dir", self.session_dir)
            for rel_path in image_paths_used:
                base_dir = runtime_dir if _image_path_runtime_scoped(rel_path) else self.session_dir
                user_content.append({"type": "image", "image": base_dir / rel_path})

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided long-video evidence. "
                    "If evidence is insufficient, say what is uncertain. "
                    f"{_answer_language_instruction()}"
                ),
            },
            {"role": "user", "content": user_content},
        ]
        fallback_used = False
        error_debug = ""
        llm_debug: dict[str, Any] = {
            "retrieval_mode": retrieval_mode,
            "visual_results_count": len(visual_results),
            "fused_results_count": len(fused_results),
            "short_term_results_count": len(short_term_results or []),
            "selected_evidence_count": len(selected_evidence or []),
            "image_warnings": image_warnings,
            "evidence_pack_summary": evidence_pack_summary or {},
        }
        answer_attempts = _env_int("WORLDMM_QUERY_ANSWER_RETRIES", 3)
        try:
            if callable(stream_handler):
                answer, primary_debug = _llm_stream_with_retries(
                    self.world_memory.respond_llm_model,
                    messages,
                    answer_attempts,
                    on_chunk=lambda text: _emit_stream_event(
                        stream_handler,
                        {"type": "delta", "stage": "answer", "delta": text},
                    ),
                )
            else:
                answer, primary_debug = _llm_generate_with_retries(
                    self.world_memory.respond_llm_model,
                    messages,
                    answer_attempts,
                )
            llm_debug["primary_generation"] = primary_debug
        except Exception as exc:
            primary_traceback = traceback.format_exc()
            error_debug = f"{type(exc).__name__}: {exc}\n{primary_traceback}"
            llm_debug["primary_error"] = error_debug
            llm_debug["traceback"] = primary_traceback
            if use_image_evidence:
                fallback_used = True
                try:
                    if callable(stream_handler):
                        answer, fallback_debug = _llm_stream_with_retries(
                            self.world_memory.respond_llm_model,
                            prompt_text,
                            answer_attempts,
                            on_chunk=lambda text: _emit_stream_event(
                                stream_handler,
                                {"type": "delta", "stage": "answer", "delta": text},
                            ),
                        )
                    else:
                        answer, fallback_debug = _llm_generate_with_retries(
                            self.world_memory.respond_llm_model,
                            prompt_text,
                            answer_attempts,
                        )
                    llm_debug["fallback_text_only_debug"] = fallback_debug
                except Exception as fallback_exc:
                    fallback_traceback = traceback.format_exc()
                    error_debug += f"\nfallback={type(fallback_exc).__name__}: {fallback_exc}\n{fallback_traceback}"
                    answer = _build_local_evidence_answer(
                        question=question,
                        text_results=text_results,
                        visual_results=visual_results,
                        fused_results=fused_results,
                        short_term_results=short_term_results or [],
                    )
                    llm_debug["fallback_traceback"] = fallback_traceback
                    llm_debug["local_fallback_used"] = True
            else:
                fallback_used = True
                answer = _build_local_evidence_answer(
                    question=question,
                    text_results=text_results,
                    visual_results=visual_results,
                    fused_results=fused_results,
                    short_term_results=short_term_results or [],
                )
                llm_debug["local_fallback_used"] = True

        selected_doc_ids = []
        for item in text_results:
            eid = item.get("memory_id") or item.get("evidence_doc_id") or item.get("segment_id")
            if eid:
                selected_doc_ids.append(str(eid))
        for result in fused_results:
            text = result.get("text") or {}
            if isinstance(text, dict) and text.get("memory_id"):
                selected_doc_ids.append(str(text["memory_id"]))
            for item in result.get("visual_items", []) or []:
                eid = item.get("evidence_doc_id") or item.get("segment_id")
                if eid:
                    selected_doc_ids.append(str(eid))
        selected_doc_ids = list(dict.fromkeys(selected_doc_ids))
        _emit_stream_event(
            stream_handler,
            {
                "type": "final",
                "stage": "answer",
                "text": answer,
                "answer": answer,
                "retrieval_mode": retrieval_mode,
            },
        )
        return QAResult(
            question=question,
            answer=answer,
            retrieved_items=[
                RetrievedItem(
                    memory_type="visual" if retrieval_mode == "visual_only" else "hybrid",
                    content=prompt_text,
                    query=question,
                    round_num=1,
                )
            ],
            round_history=[
                {
                    "round_num": 1,
                    "decision": "search",
                    "memory_type": retrieval_mode,
                    "search_query": question,
                    "retrieved_content": f"visual={len(visual_results)} fused={len(fused_results)}",
                }
            ],
            num_rounds=1,
            answer_mode="open_ended",
            qa_template_name=f"online_{retrieval_mode}",
            selected_doc_ids=selected_doc_ids,
            selector_reason="Online visual/hybrid retrieval evidence package.",
            semantic_fact_ids=[],
            visual_event_image_counts={"visual_retrieval": len(image_paths_used)} if use_image_evidence else {},
            image_evidence_enabled=bool(use_image_evidence),
            image_paths_used=image_paths_used,
            model_response_text=answer,
            error_debug=error_debug,
            fallback_used=fallback_used,
            llm_debug=llm_debug,
        )

    def _build_visual_answer_prompt(
        self,
        question: str,
        retrieval_mode: str,
        text_results: list[dict[str, Any]],
        visual_results: list[dict[str, Any]],
        fused_results: list[dict[str, Any]],
        short_term_results: list[dict[str, Any]] | None = None,
        selected_evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        short_term_results = short_term_results or []
        selected_evidence = selected_evidence or []
        text_evidence = []
        for item in text_results[:5]:
            text_evidence.append({
                "segment_id": item.get("segment_id"),
                "score": item.get("score"),
                "caption": item.get("caption"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "evidence_doc_id": item.get("evidence_doc_id"),
            })
        visual_evidence = []
        seen_visual = set()
        source_items: list[dict[str, Any]] = []
        for fused in fused_results:
            source_items.extend(fused.get("visual_items", []) or [])
        source_items.extend(visual_results)
        for item in source_items:
            visual_id = str(item.get("visual_id") or item.get("image_path") or "")
            if not visual_id or visual_id in seen_visual:
                continue
            seen_visual.add(visual_id)
            visual_evidence.append({
                "visual_id": item.get("visual_id"),
                "segment_id": item.get("segment_id"),
                "timestamp": item.get("timestamp"),
                "image_path": item.get("image_path"),
                "keyframe_caption": item.get("keyframe_caption"),
                "visual_score": item.get("visual_score") or item.get("score"),
                "segment_caption": item.get("segment_caption"),
                "scene": item.get("scene"),
                "objects": item.get("visual_objects"),
                "actions": item.get("main_actions"),
                "state_changes": item.get("state_changes"),
            })
            if len(visual_evidence) >= 8:
                break
        fused_evidence = []
        for item in fused_results:
            fused_evidence.append({
                "segment_id": item.get("segment_id"),
                "text_score": item.get("text_score"),
                "visual_score": item.get("visual_score"),
                "fused_score": item.get("fused_score"),
            })
        short_term_evidence = []
        for item in short_term_results[:5]:
            caption = (
                item.get("event_caption_refined")
                or item.get("event_caption_fast")
                or item.get("event_caption_placeholder")
                or ""
            )
            short_term_evidence.append({
                "event_id": item.get("event_id"),
                "score": item.get("score"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "status": item.get("status"),
                "caption_source": item.get("caption_source"),
                "boundary_reason": item.get("boundary_reason"),
                "caption": caption,
                "event_caption_refined": item.get("event_caption_refined"),
                "event_caption_fast": item.get("event_caption_fast"),
                "transcript": item.get("transcript"),
                "keyframes": [
                    {
                        "timestamp": frame.get("timestamp"),
                        "path": frame.get("path"),
                        "role": frame.get("role"),
                    }
                    for frame in (item.get("keyframes") or [])[:2]
                    if isinstance(frame, dict)
                ],
                "note": (
                    "This short-term event has a refined caption."
                    if item.get("status") in {"refined", "final"}
                    else "This short-term event is provisional and may not yet have refined caption."
                ),
            })
        final_evidence = []
        final_prompt_limit = _env_int("WORLDMM_PROMPT_FINAL_EVIDENCE_K", 10)
        for item in selected_evidence[: max(1, final_prompt_limit)]:
            final_evidence.append({
                "evidence_id": item.get("evidence_id"),
                "source_memory": item.get("source_memory"),
                "source_type": item.get("source_type"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "timestamp": item.get("timestamp"),
                "caption": item.get("caption"),
                "transcript": item.get("transcript"),
                "keyframe_paths": item.get("keyframe_paths"),
                "final_score": item.get("final_score"),
                "status": item.get("status"),
            })
        return (
            "You are answering a question about a first-person video session.\n"
            "Use only the evidence below. Do not invent hidden intentions. "
            "Some retrieved memory entries may be provisional and not fully refined; "
            "when images are attached, use the visible frames as primary evidence and treat placeholder captions only as time-range hints. "
            f"{_answer_language_instruction()}\n\n"
            f"Question: {question}\n"
            f"Retrieval mode: {retrieval_mode}\n\n"
            "Text evidence:\n"
            f"{json.dumps(text_evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Visual retrieval evidence:\n"
            f"{json.dumps(visual_evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Fused evidence scores:\n"
            f"{json.dumps(fused_evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Final selected evidence after memory fusion:\n"
            f"{json.dumps(final_evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Short-term provisional micro-events:\n"
            f"{json.dumps(short_term_evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
            "Give a concise grounded answer. Mention timestamps only when useful."
        )

    def _select_generation_image_paths(
        self,
        visual_results: list[dict[str, Any]],
        fused_results: list[dict[str, Any]],
        max_image_evidence: int,
    ) -> tuple[list[str], list[str]]:
        max_image_evidence = max(0, int(max_image_evidence or 0))
        if max_image_evidence <= 0:
            return [], []
        candidates: list[dict[str, Any]] = []
        for fused in sorted(fused_results, key=lambda x: -_safe_float(x.get("fused_score"))):
            for item in sorted(fused.get("visual_items", []) or [], key=lambda x: -_safe_float(x.get("visual_score") or x.get("score"))):
                candidate = dict(item)
                candidate["fused_score"] = fused.get("fused_score")
                candidates.append(candidate)
        candidates.extend(sorted(visual_results, key=lambda x: -_safe_float(x.get("visual_score") or x.get("score"))))

        selected: list[str] = []
        warnings: list[str] = []
        seen_paths: set[str] = set()
        seen_segments: dict[str, int] = {}
        for item in candidates:
            rel_path = str(item.get("image_path") or "")
            if not rel_path or rel_path in seen_paths:
                continue
            segment_id = _canonical_segment_id(item.get("segment_id") or item.get("evidence_doc_id"))
            if seen_segments.get(segment_id, 0) >= 2:
                continue
            base_dir = getattr(self, "runtime_session_dir", self.session_dir) if _image_path_runtime_scoped(rel_path) else self.session_dir
            abs_path = base_dir / rel_path
            if not abs_path.exists():
                warnings.append(f"missing image: {rel_path}")
                continue
            selected.append(rel_path)
            seen_paths.add(rel_path)
            seen_segments[segment_id] = seen_segments.get(segment_id, 0) + 1
            if len(selected) >= max_image_evidence:
                break
        return selected, warnings

    def _timestamps_from_visual_results(self, fused_results: list[dict[str, Any]]) -> list[dict[str, float]]:
        timestamps: list[dict[str, float]] = []
        seen: set[tuple[float, float]] = set()
        for fused in fused_results:
            items = fused.get("visual_items", []) or []
            if not items:
                continue
            item = items[0]
            start = _safe_float(item.get("start_time"), _safe_float(item.get("timestamp")))
            end = _safe_float(item.get("end_time"), start)
            pair = (start, end)
            if pair not in seen:
                seen.add(pair)
                timestamps.append({"start": start, "end": end})
        return timestamps

    def _retrieved_memories_from_fused(self, fused_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        for fused in fused_results:
            items = fused.get("visual_items", []) or []
            item = items[0] if items else {}
            memory_id = item.get("evidence_doc_id") or item.get("segment_id") or item.get("visual_id")
            memories.append({
                "memory_id": memory_id,
                "segment_id": item.get("segment_id"),
                "score": fused.get("fused_score"),
                "caption": item.get("segment_caption") or item.get("keyframe_caption") or "",
                "evidence_doc_id": item.get("evidence_doc_id"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
            })
        return memories

    def _build_text_results(self, selected_events: list[dict[str, Any]], retrieved_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for idx, event in enumerate(selected_events):
            doc_id = str(event.get("doc_id") or "")
            canonical_doc_id = _canonical_segment_id(doc_id)
            evidence = self.evidence_by_doc_id.get(doc_id) or self.evidence_by_doc_id.get(canonical_doc_id) or {}
            segment_id = evidence.get("segment_id") if isinstance(evidence, dict) and evidence.get("segment_id") else canonical_doc_id or doc_id
            canonical_segment_id = _canonical_segment_id(segment_id)
            results.append({
                "memory_id": doc_id,
                "segment_id": segment_id,
                "canonical_segment_id": canonical_segment_id,
                "evidence_doc_id": evidence.get("evidence_doc_id", doc_id) if isinstance(evidence, dict) else doc_id,
                "caption": event.get("text", ""),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "score": 1.0 if idx == 0 else max(0.0, 1.0 - idx * 0.1),
                "keyframe_paths": list(event.get("keyframe_paths", []) or []),
            })
        return results

    def _retrieve_visual(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if not self.visual_ready or self.visual_index is None:
            return []
        runtime = get_global_vlm2vec_runtime(
            backend=self.memory_config.get("visual_embedding_backend"),
            model_path=self.memory_config.get("visual_embedding_model"),
        )
        query_vec = runtime.encode_texts([question])
        query_vec = l2_normalize(query_vec)
        scores, indices = self.visual_index.search(query_vec, top_k)
        row_to_visual_id = self.visual_id_mapping.get("row_to_visual_id", {})
        results = []
        if len(indices) == 0:
            return results
        for score, row in zip(scores[0].tolist(), indices[0].tolist()):
            if row < 0:
                continue
            visual_id = row_to_visual_id.get(str(row)) or row_to_visual_id.get(row)
            item = self.visual_items.get(str(visual_id))
            if not item:
                continue
            result = dict(item)
            result["score"] = float(score)
            result["visual_score"] = float(score)
            result["image_path"] = str(item.get("image_path") or "")
            result["canonical_segment_id"] = _canonical_segment_id(item.get("segment_id") or item.get("evidence_doc_id"))
            results.append(result)
        return self._normalize_visual_scores(results)

    def _normalize_visual_scores(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return results
        scores = [float(r.get("score") or 0.0) for r in results]
        mn, mx = min(scores), max(scores)
        for r in results:
            score = float(r.get("score") or 0.0)
            r["score"] = 1.0 if mx == mn else (score - mn) / (mx - mn)
            r["visual_score"] = r["score"]
        return results

    def _fuse_results(
        self,
        text_results: list[dict[str, Any]],
        visual_results: list[dict[str, Any]],
        final_evidence_k: int,
        cache_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        cache_context = cache_context or {}
        by_segment: dict[str, dict[str, Any]] = {}
        for item in text_results:
            seg = _canonical_segment_id(item.get("canonical_segment_id") or item.get("segment_id") or item.get("evidence_doc_id") or item.get("memory_id"))
            by_segment.setdefault(seg, {"segment_id": seg, "canonical_segment_id": seg, "text_score": 0.0, "visual_score": 0.0, "text": item, "visual_items": []})
            by_segment[seg]["text_score"] = max(float(by_segment[seg]["text_score"]), float(item.get("score") or 0.0))
            if by_segment[seg].get("text") is None:
                by_segment[seg]["text"] = item
        for item in visual_results:
            seg = _canonical_segment_id(item.get("canonical_segment_id") or item.get("segment_id") or item.get("evidence_doc_id") or item.get("visual_id"))
            by_segment.setdefault(seg, {"segment_id": seg, "canonical_segment_id": seg, "text_score": 0.0, "visual_score": 0.0, "text": None, "visual_items": []})
            by_segment[seg]["visual_score"] = max(float(by_segment[seg]["visual_score"]), float(item.get("visual_score") or item.get("score") or 0.0))
            by_segment[seg]["visual_items"].append(item)
        fused = []
        for seg, item in by_segment.items():
            text_score = float(item["text_score"])
            visual_score = float(item["visual_score"])
            fused_score = 0.6 * text_score + 0.4 * visual_score
            cache_boost, cache_reasons = self._cache_boost_for_segment(seg, item, cache_context)
            fused_score = min(1.0, fused_score + cache_boost)
            best_visual = sorted(item["visual_items"], key=lambda x: -float(x.get("visual_score") or 0.0))[:2]
            fused.append({
                "segment_id": seg,
                "canonical_segment_id": seg,
                "text_score": text_score,
                "visual_score": visual_score,
                "fused_score": fused_score,
                "cache_boost": cache_boost,
                "cache_match_reasons": cache_reasons,
                "text": item.get("text"),
                "visual_items": best_visual,
                "has_text": item.get("text") is not None,
                "visual_item_count": len(item.get("visual_items", []) or []),
            })
        fused.sort(key=lambda x: -float(x.get("fused_score") or 0.0))
        return fused[:max(final_evidence_k, 1)]

    def _cache_boost_for_segment(
        self,
        segment_id: str,
        fused_item: dict[str, Any],
        cache_context: dict[str, Any],
    ) -> tuple[float, list[str]]:
        if not cache_context.get("is_followup"):
            return 0.0, []
        reasons: list[str] = []
        boost = 0.0
        canonical = _canonical_segment_id(segment_id)
        referenced_segments = {
            _canonical_segment_id(value)
            for value in cache_context.get("referenced_segment_ids", []) or []
            if value
        }
        referenced_memories = {
            _canonical_segment_id(value)
            for value in cache_context.get("referenced_memory_ids", []) or []
            if value
        }
        if canonical in referenced_segments or canonical in referenced_memories:
            boost += 0.16
            reasons.append("referenced_segment")

        text = fused_item.get("text") or {}
        start = end = None
        if isinstance(text, dict):
            start = _hhmmssff_to_seconds(text.get("start_time", "0"))
            end = _hhmmssff_to_seconds(text.get("end_time", "0"))
        elif fused_item.get("visual_items"):
            visual = fused_item["visual_items"][0]
            start = _safe_float(visual.get("start_time"), _safe_float(visual.get("timestamp")))
            end = _safe_float(visual.get("end_time"), start)
        if start is not None and end is not None:
            for window in cache_context.get("referenced_time_ranges", []) or []:
                w_start = _safe_float(window.get("start"))
                w_end = _safe_float(window.get("end"), w_start)
                if max(start, w_start) <= min(end, w_end):
                    boost += 0.1
                    reasons.append("referenced_time_overlap")
                    break

        entity_terms = []
        for entity in cache_context.get("referenced_entities", []) or []:
            for key in ("canonical_name", "name", "entity_key"):
                value = entity.get(key) if isinstance(entity, dict) else None
                if value:
                    entity_terms.append(str(value).lower())
        if entity_terms:
            haystack = json.dumps(fused_item, ensure_ascii=False, default=str).lower()
            if any(term and term in haystack for term in entity_terms):
                boost += 0.08
                reasons.append("referenced_entity")

        return min(boost, 0.25), reasons

    def _merge_evidence_frames(self, base_frames: list[dict[str, Any]], fused_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frames = list(base_frames)
        seen = {str(f.get("path")) for f in frames if f.get("path")}
        for fused in fused_results:
            for item in fused.get("visual_items", []) or []:
                path = str(item.get("image_path") or "")
                if not path or path in seen:
                    continue
                seen.add(path)
                frames.append({
                    "path": path,
                    "timestamp": (
                        _normalize_keyframe_timestamp(item.get("timestamp"), path)
                        if item.get("timestamp") is not None
                        else _keyframe_timestamp(path)
                    ),
                    "caption": item.get("keyframe_caption", ""),
                    "visual_score": item.get("visual_score"),
                    "fused_score": fused.get("fused_score"),
                    "source": "hybrid_retrieval",
                })
        return frames

    def _build_evidence_frames(self, selected_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        seen = set()
        for event in selected_events:
            doc_id = str(event.get("doc_id") or "")
            evidence = self.evidence_by_doc_id.get(doc_id) or self.evidence_by_doc_id.get(_canonical_segment_id(doc_id)) or {}
            frame_by_path = {}
            if isinstance(evidence, dict):
                for item in evidence.get("keyframe_captions", []) or []:
                    if isinstance(item, dict) and item.get("path"):
                        frame_by_path[str(item["path"])] = item
            for path in event.get("keyframe_paths", []) or []:
                path = str(path)
                if not path or path in seen:
                    continue
                seen.add(path)
                evidence_frame = frame_by_path.get(path)
                frames.append(
                    {
                        "path": path,
                        "timestamp": _keyframe_timestamp(path, evidence_frame),
                        "caption": str((evidence_frame or {}).get("caption", event.get("keyframe_caption", "")) or ""),
                    }
                )
        return frames

    def _build_retrieved_memories(self, selected_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        memories = []
        for idx, event in enumerate(selected_events):
            if not isinstance(event, dict):
                continue
            doc_id = event.get("doc_id") or f"selected_{idx}"
            canonical_doc_id = _canonical_segment_id(doc_id)
            evidence = self.evidence_by_doc_id.get(str(doc_id)) or self.evidence_by_doc_id.get(canonical_doc_id) or {}
            segment_id = evidence.get("segment_id") if isinstance(evidence, dict) and evidence.get("segment_id") else canonical_doc_id or doc_id
            memories.append(
                {
                    "memory_id": doc_id,
                    "segment_id": segment_id,
                    "canonical_segment_id": _canonical_segment_id(segment_id),
                    "score": None,
                    "caption": event.get("text", ""),
                    "evidence_doc_id": evidence.get("evidence_doc_id", doc_id) if isinstance(evidence, dict) else doc_id,
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                }
            )
        return memories

    def runtime_info(self) -> dict[str, Any]:
        episodic_memory = getattr(self.world_memory, "episodic_memory", None)
        semantic_memory = getattr(self.world_memory, "semantic_memory", None)
        episodic_counts: dict[str, int] = {}
        indexed_counts: dict[str, int] = {}
        hipporag_counts: dict[str, Any] = {}
        if episodic_memory is not None:
            for granularity in getattr(episodic_memory, "granularities", []) or []:
                episodic_counts[granularity] = len(getattr(episodic_memory, "captions", {}).get(granularity, []) or [])
                indexed_counts[granularity] = len(getattr(episodic_memory, "indexed_entries", {}).get(granularity, []) or [])
            for granularity, hipporag in (getattr(episodic_memory, "hipporag", {}) or {}).items():
                try:
                    hipporag_counts[granularity] = {
                        "ready_to_retrieve": bool(getattr(hipporag, "ready_to_retrieve", False)),
                        "passages": len(hipporag.chunk_embedding_store.get_all_ids()),
                        "entities": len(hipporag.entity_embedding_store.get_all_ids()),
                        "facts": len(hipporag.fact_embedding_store.get_all_ids()),
                        "graph_nodes": hipporag.graph.vcount(),
                    }
                except Exception as exc:
                    hipporag_counts[granularity] = {"error": f"{type(exc).__name__}: {exc}"}
        semantic_count = 0
        if semantic_memory is not None:
            semantic_count = len(
                getattr(semantic_memory, "indexed_entries", [])
                or getattr(semantic_memory, "triple_id_to_entry", {})
                or getattr(semantic_memory, "memory_entries", [])
                or getattr(semantic_memory, "facts", [])
                or []
            )
        return {
            "session_id": self.session_id,
            "memory_config_path": str(self.memory_config_path),
            "memory_config_mtime": self.memory_config_mtime,
            "pipeline_mode": self.memory_config.get("pipeline_mode"),
            "active_30s_source": self.memory_config.get("active_30s_source") or self.memory_config.get("worldmm_30s_input_source"),
            "episodic_source": self.memory_config.get("episodic_source"),
            "legacy_evidence_used": bool(self.memory_config.get("legacy_evidence_used") or self.memory_config.get("legacy_evidence_fallback_used")),
            "worldmm_update_mode": self.memory_config.get("worldmm_update_mode"),
            "strict_load_only": self.strict_load_only,
            "preload_status": self.preload_status,
            "latest_ready_memory_version": self.latest_ready_memory_version,
            "building_memory_version": self.building_memory_version,
            "active_query_memory_version": self.active_query_memory_version,
            "memory_build_state": self.memory_config.get("memory_build_state"),
            "memory_component_versions": self._memory_component_versions(),
            "loaded_document_counts": {
                "episodic_captions": episodic_counts,
                "episodic_indexed": indexed_counts,
                "semantic_facts": semantic_count,
                "visual_items": len(self.visual_items),
                "visual_evidence": len(self.visual_evidence_data),
            },
            "hipporag_cache_counts": hipporag_counts,
            "using_stale_while_building": bool(
                self.building_memory_version
                and self.active_query_memory_version
                and int(self.building_memory_version) > int(self.active_query_memory_version)
            ),
            "loaded_at": self.loaded_at,
            "last_accessed_at": self.last_accessed_at,
            "query_count": self.query_count,
            "visual_ready": self.visual_ready,
            "visual_item_count": len(self.visual_items),
            "visual_embedding_model": self.visual_embedding_model,
            "visual_version": self.visual_version,
            "short_term_ready": self.short_term_ready,
            "mst_version": self.mst_version,
            "mst_loaded_at": self.mst_loaded_at,
            "mst_state": self.mst_store.get_state(),
            "current_ready": self.current_ready,
            "current_stale": self.current_stale,
            "mcur_version": self.mcur_version,
            "mcur_loaded_at": self.mcur_loaded_at,
            "mcur_state": self.mcur_store.get_state(),
            "recent_queries": list(self.recent_queries),
            "interaction_cache": self.interaction_cache.summary(),
        }


def load_query_engine(
    session_id: str,
    sessions_root: Path = Path("online_sessions"),
    long_term_retrieval_scheme: str | None = None,
) -> LoadedQueryEngine:
    query_rag = _query_rag_helpers()
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    EmbeddingModel, LLMModel, PromptTemplateManager, WorldMemory, _ = _worldmm_classes(long_term_retrieval_scheme)
    session_dir = sessions_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"session not found: {session_dir}")
    memory_config_path, config = _load_memory_config(session_dir)
    config["long_term_retrieval_scheme"] = long_term_retrieval_scheme
    strict_load_only = _env_bool("WORLDMM_QUERY_STRICT_LOAD_ONLY", True)
    if strict_load_only:
        os.environ.setdefault("WORLDMM_QUERY_USE_CACHED_HIPPORAG", "1")
        os.environ.setdefault("WORLDMM_QUERY_SKIP_REINDEX", "1")
    latest_ready_memory_version = _memory_version_from_config(config)
    building_memory_version = None
    try:
        building_memory_version = int(config.get("building_memory_version")) if config.get("building_memory_version") is not None else None
    except Exception:
        building_memory_version = None
    if strict_load_only and not latest_ready_memory_version:
        raise RuntimeError("no ready long-term memory snapshot for strict load-only query")
    query_args = dict(config.get("query_rag_args", {}) or {})
    subject = str(query_args.get("subject") or session_id)
    default_query_model = os.getenv("WORLDMM_QUERY_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"
    retriever_model_name = str(
        os.getenv("WORLDMM_QUERY_RETRIEVER_MODEL")
        or default_query_model
        or query_args.get("retriever_model")
        or "gpt-5.4"
    )
    respond_model_name = str(
        os.getenv("WORLDMM_QUERY_RESPOND_MODEL")
        or os.getenv("WORLDMM_RESPOND_MODEL")
        or default_query_model
        or query_args.get("respond_model")
        or retriever_model_name
    )

    episodic_caption_root = _session_path(session_dir, query_args.get("episodic_caption_root"))
    episodic_sidecar_root = _session_path(session_dir, query_args.get("episodic_sidecar_root"))
    semantic_root = _session_path(session_dir, query_args.get("semantic_root"))
    visual_root = _session_path(session_dir, query_args.get("visual_root"))
    visual_evidence_file = _session_path(session_dir, query_args.get("visual_evidence_file"))
    if not episodic_caption_root or not episodic_sidecar_root or not semantic_root:
        raise RuntimeError("memory_config query_rag_args are incomplete")

    embedding_model = EmbeddingModel()
    answer_retries = _env_int("WORLDMM_QUERY_ANSWER_RETRIES", 3)
    retriever_llm_model = LLMModel(model_name=retriever_model_name, max_retries=answer_retries)
    respond_llm_model = LLMModel(model_name=respond_model_name, fps=1, max_retries=answer_retries)
    prompt_template_manager = PromptTemplateManager()
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=1,
        max_errors=3,
        episodic_cache_tag=f"online_{session_id}",
    )

    episodic_caption_files = query_rag.filter_existing_files(
        query_rag.build_episodic_caption_file_map(episodic_caption_root, subject)
    )
    if "30sec" not in episodic_caption_files:
        raise FileNotFoundError("30sec caption file is required.")
    skip_episodic_sidecar = _env_bool("WORLDMM_SKIP_EPISODIC_SIDECAR", False)
    if skip_episodic_sidecar:
        episodic_triplet_files = {}
        episodic_graph_files = {}
        config["episodic_sidecar_load_skipped"] = True
    else:
        episodic_triplet_files, episodic_graph_files = query_rag.build_episodic_sidecar_file_maps(
            episodic_sidecar_root,
            retriever_model_name,
        )
        episodic_triplet_files = query_rag.filter_existing_files(episodic_triplet_files)
        episodic_graph_files = query_rag.filter_existing_files(episodic_graph_files)
    semantic_path = None
    semantic_results = []
    try:
        semantic_path = query_rag.resolve_semantic_path(semantic_root, retriever_model_name)
    except FileNotFoundError as exc:
        config["semantic_memory_ready"] = False
        config["semantic_load_warning"] = str(exc)
    if semantic_path and Path(semantic_path).exists():
        semantic_results = query_rag.load_json(semantic_path)
    elif semantic_path:
        semantic_results = []
        config["semantic_memory_ready"] = False
        config["semantic_load_warning"] = f"semantic memory file missing: {semantic_path}"
    inferred_visual_evidence = query_rag.infer_visual_evidence_file(
        episodic_caption_root=episodic_caption_root,
        subject=subject,
        user_specified=visual_evidence_file,
    )
    visual_evidence_data = query_rag.load_json(inferred_visual_evidence or episodic_caption_files["30sec"])
    visual_embeddings_path = None
    if visual_root:
        candidate_visual_path = Path(visual_root) / "visual_embeddings.pkl"
        if candidate_visual_path.exists():
            visual_embeddings_path = str(candidate_visual_path)

    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    if episodic_triplet_files or episodic_graph_files:
        world_memory.load_episodic_sidecar(
            triplet_files=episodic_triplet_files,
            graph_files=episodic_graph_files,
        )
    _configure_semantic_embedding_cache(world_memory, semantic_path)
    world_memory.load_semantic_triples(data=semantic_results)
    world_memory.load_visual_clips(
        embeddings_path=visual_embeddings_path,
        clips_data=visual_evidence_data,
    )
    if hasattr(world_memory.visual_memory, "set_base_dir"):
        world_memory.visual_memory.set_base_dir(session_dir)
    until_time = query_rag.build_until_timestamp(
        str(query_args.get("until_date") or "DAY1"),
        str(query_args.get("until_time") or "23595999"),
    )
    world_memory.index(until_time)

    visual_ready = False
    visual_items: dict[str, dict[str, Any]] = {}
    visual_id_mapping: dict[str, Any] = {}
    visual_index: VisualSearchIndex | None = None
    visual_embedding_model = None
    visual_version = int(config.get("visual_version") or 0)
    if config.get("visual_embedding_ready"):
        try:
            visual_items_path = _session_config_path(session_dir, config, "visual_items_path")
            visual_mapping_path = _session_config_path(session_dir, config, "visual_id_mapping_path")
            visual_faiss_path = _session_config_path(session_dir, config, "visual_faiss_path")
            if not visual_items_path or not visual_mapping_path or not visual_faiss_path:
                raise FileNotFoundError("visual index paths are incomplete in memory_config.json")
            if not visual_items_path.exists() or not visual_mapping_path.exists() or not visual_faiss_path.exists():
                raise FileNotFoundError("visual index files are missing")
            visual_item_list = read_visual_items(visual_items_path)
            visual_items = {str(item.get("visual_id")): item for item in visual_item_list if item.get("visual_id")}
            visual_id_mapping = read_json(visual_mapping_path, default={})
            if not isinstance(visual_id_mapping, dict):
                visual_id_mapping = {}
            visual_index = load_visual_index(visual_faiss_path)
            visual_ready = True
            visual_embedding_model = str(config.get("visual_embedding_model") or "")
        except Exception as exc:
            config["visual_embedding_ready"] = False
            config["visual_embedding_error"] = f"failed to load visual index: {exc}"

    _mark_active_query_component_versions(
        session_dir,
        config,
        active_query_memory_version=latest_ready_memory_version,
    )

    return LoadedQueryEngine(
        session_id=session_id,
        session_dir=session_dir,
        memory_config_path=memory_config_path,
        memory_config=config,
        world_memory=world_memory,
        query_args=query_args,
        visual_evidence_data=visual_evidence_data,
        semantic_path=semantic_path,
        visual_ready=visual_ready,
        visual_items=visual_items,
        visual_id_mapping=visual_id_mapping,
        visual_index=visual_index,
        visual_embedding_model=visual_embedding_model,
        visual_version=visual_version,
        strict_load_only=strict_load_only,
        latest_ready_memory_version=latest_ready_memory_version,
        building_memory_version=building_memory_version,
        active_query_memory_version=latest_ready_memory_version,
        preload_status="loaded",
        long_term_retrieval_scheme=long_term_retrieval_scheme,
    )


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _build_short_term_answer(question: str, short_term_results: list[dict[str, Any]]) -> str:
    del question
    if not short_term_results:
        return "No recent short-term events are available yet."
    lines = []
    refined_count = 0
    for item in short_term_results[:3]:
        start = _safe_float(item.get("start_time"))
        end = _safe_float(item.get("end_time"), start)
        caption = (
            item.get("event_caption_refined")
            or item.get("event_caption_fast")
            or item.get("event_caption_placeholder")
            or "A provisional short-term event is available."
        )
        if item.get("event_caption_refined"):
            refined_count += 1
        transcript = item.get("transcript") or ""
        suffix = ""
        if transcript and transcript not in caption:
            suffix = f" transcript: {transcript}"
        lines.append(f"{start:.1f}-{end:.1f}s: {caption}{suffix}")
    all_refined = refined_count == min(3, len(short_term_results))
    note = "These short-term micro-events include refined captions." if all_refined else "Some short-term micro-events are still provisional."
    return "From short-term memory: " + "; ".join(lines) + f". {note}"


def _build_local_evidence_answer(
    *,
    question: str,
    text_results: list[dict[str, Any]],
    visual_results: list[dict[str, Any]],
    fused_results: list[dict[str, Any]],
    short_term_results: list[dict[str, Any]],
) -> str:
    if short_term_results and not (text_results or visual_results or fused_results):
        return _build_short_term_answer(question, short_term_results)
    snippets: list[str] = []
    for item in text_results[:3]:
        caption = str(item.get("caption") or "").strip()
        if caption:
            snippets.append(f"{item.get('start_time', '')}-{item.get('end_time', '')}: {caption}")
    source_visuals: list[dict[str, Any]] = []
    for fused in fused_results[:3]:
        source_visuals.extend(fused.get("visual_items", []) or [])
    source_visuals.extend(visual_results[:3])
    for item in source_visuals[:3]:
        caption = str(item.get("keyframe_caption") or item.get("segment_caption") or "").strip()
        if caption:
            snippets.append(f"{item.get('timestamp', '')}s: {caption}")
    for item in short_term_results[:3]:
        caption = (
            item.get("event_caption_refined")
            or item.get("event_caption_fast")
            or item.get("event_caption_placeholder")
            or item.get("transcript")
            or ""
        )
        if caption:
            snippets.append(f"{item.get('start_time')}-{item.get('end_time')}s: {caption}")
    if not snippets:
        return "Evidence was retrieved, but LLM answer generation failed after retries."
    return "LLM answer generation failed after retries. Local evidence summary: " + "; ".join(snippets[:5])


def _build_short_term_llm_prompt(question: str, short_term_results: list[dict[str, Any]], pack_summary: dict[str, Any]) -> str:
    evidence = []
    prompt_limit = _env_int("WORLDMM_MST_SUMMARY_PROMPT_EVENTS", 16) if _is_summary_question_text(question) else 8
    for item in short_term_results[: max(1, prompt_limit)]:
        evidence.append({
            "event_id": item.get("event_id"),
            "score": item.get("score"),
            "start_time": item.get("start_time"),
            "end_time": item.get("end_time"),
            "status": item.get("status"),
            "caption_source": item.get("caption_source"),
            "caption": (
                item.get("event_caption_refined")
                or item.get("event_caption_fast")
                or item.get("event_caption_placeholder")
                or ""
            ),
            "transcript": item.get("transcript"),
            "boundary_reason": item.get("boundary_reason"),
            "diff_score": item.get("diff_score"),
            "keyframes": [
                {
                    "timestamp": frame.get("timestamp"),
                    "path": frame.get("path"),
                    "role": frame.get("role"),
                }
                for frame in (item.get("keyframes") or [])[:2]
                if isinstance(frame, dict)
            ],
        })
    return (
        "You are answering a question about recent first-person video short-term memory.\n"
        "Use only the retrieved M_st micro-events below. Some events may be provisional; say uncertainty when needed.\n"
        "If keyframes/images are attached, use the visible content as primary evidence and do not treat placeholder captions as final facts.\n"
        f"{_answer_language_instruction()} Be concise.\n\n"
        f"Question: {question}\n\n"
        f"Evidence pack summary:\n{json.dumps(pack_summary or {}, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Short-term micro-events:\n{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}"
    )


def _get_short_term_answer_model() -> Any:
    model_name = (
        os.getenv("WORLDMM_MCUR_ANSWER_MODEL")
        or os.getenv("WORLDMM_MST_ANSWER_MODEL")
        or os.getenv("WORLDMM_QUERY_RESPOND_MODEL")
        or os.getenv("WORLDMM_RESPOND_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.4"
    )
    retries = _env_int("WORLDMM_MCUR_ANSWER_MODEL_RETRIES", _env_int("WORLDMM_QUERY_ANSWER_RETRIES", 3))
    key = (model_name, retries)
    if key not in _SHORT_TERM_ANSWER_MODELS:
        _, LLMModel, _, _, _ = _worldmm_classes()
        _SHORT_TERM_ANSWER_MODELS[key] = LLMModel(model_name=model_name, fps=1, max_retries=retries)
    return _SHORT_TERM_ANSWER_MODELS[key]


def _answer_current_memory(
    *,
    session_id: str,
    session_dir: Path,
    question: str,
    resolved_question: str,
    route_decision: dict[str, Any],
    cache_context: dict[str, Any],
    latency: dict[str, Any],
    router_ms: int,
    long_term_ready: bool,
    short_term_ready: bool,
    cache_used: bool,
    cache_hit: bool,
    cache_mode: str,
    total_start: float | None = None,
    day_context: dict[str, Any] | None = None,
    stream_handler: Any = None,
) -> dict[str, Any]:
    store = MCurStore(session_dir)
    current_context = store.get_current_context()
    if not current_context.get("mcur_ready") or current_context.get("is_stale"):
        return {
            "status": "not_ready",
            "session_id": session_id,
            "question": question,
            "resolved_question": resolved_question,
            "message": "current memory is stale or unavailable",
            "current_ready": bool(current_context.get("mcur_ready")),
            "current_used": False,
            "current_stale": bool(current_context.get("is_stale", True)),
            "long_term_ready": long_term_ready,
            "short_term_ready": short_term_ready,
        }

    current_image_cap = max(0, _env_int("WORLDMM_MCUR_MAX_QUERY_IMAGES", 3))
    try:
        requested_max_images = int(route_decision.get("max_image_evidence")) if route_decision.get("max_image_evidence") is not None else current_image_cap
    except Exception:
        requested_max_images = current_image_cap
    max_images = min(max(0, requested_max_images), current_image_cap)
    max_frames = int(route_decision.get("evidence_frames_k") or _env_int("WORLDMM_MCUR_MAX_EVIDENCE_FRAMES", 5))
    use_image = bool(route_decision.get("use_image_evidence", True))
    if not use_image and _is_visual_event_question(question, route_decision.get("query_type")):
        use_image = True
        fallback_max_images = max(0, _env_int("WORLDMM_MCUR_VISUAL_FALLBACK_MAX_IMAGES", 3))
        max_images = min(max(max_images, fallback_max_images), current_image_cap)
        route_decision["use_image_evidence"] = True
        route_decision["use_image_evidence_source"] = "auto_current_visual_question"
    route_decision["max_image_evidence"] = max_images
    selection = MCurFrameSelector().select_frames_for_query(
        current_context,
        resolved_question,
        max_images=max_images,
        max_frames=max_frames,
    )
    pack_start = time.perf_counter()
    packer = EvidencePacker(session_dir=session_dir)
    pack_result = packer.pack(
        query=question,
        route_decision=route_decision,
        retrieval_result={
            "current_context": current_context,
            "current_selection": selection,
            "text_results": [],
            "visual_results": [],
            "fused_results": [],
            "evidence_frames": selection.get("evidence_frames", []),
            "short_term_results": [],
        },
        cache_context=cache_context,
    )
    pack_ms = _ms(pack_start)
    selected_image_paths = list(pack_result.get("selected_image_paths_for_mllm") or [])[:max_images]
    if not use_image:
        selected_image_paths = []

    day_context_block = build_day_context_block(day_context)
    prompt_text = build_current_prompt(resolved_question, current_context, selection)
    if day_context_block:
        prompt_text = f"{day_context_block}\n\n{prompt_text}"
    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for rel_path in selected_image_paths:
        user_content.append({"type": "image", "image": session_dir / rel_path})
    messages = [
        {
            "role": "system",
            "content": "Answer current-perception questions using only the current rolling video memory and attached frames.",
        },
        {"role": "user", "content": user_content},
    ]
    generation_start = time.perf_counter()
    fallback_used = False
    answer_debug: dict[str, Any] = {
        "image_paths_used": selected_image_paths,
        "selection_reason": selection.get("selection_reason"),
    }
    error_debug = ""
    try:
        if callable(stream_handler):
            answer, llm_debug = _llm_stream_with_retries(
                _get_short_term_answer_model(),
                messages if selected_image_paths else prompt_text,
                _env_int("WORLDMM_MCUR_ANSWER_RETRIES", 1),
                on_chunk=lambda text: _emit_stream_event(
                    stream_handler,
                    {"type": "delta", "stage": "answer", "delta": text},
                ),
            )
        else:
            answer, llm_debug = _llm_generate_with_retries(
                _get_short_term_answer_model(),
                messages if selected_image_paths else prompt_text,
                _env_int("WORLDMM_MCUR_ANSWER_RETRIES", 1),
            )
        answer_debug["llm_debug"] = llm_debug
    except Exception as exc:
        fallback_used = True
        error_debug = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        answer_debug["error_debug"] = error_debug
        if selected_image_paths:
            try:
                if callable(stream_handler):
                    answer, fallback_debug = _llm_stream_with_retries(
                        _get_short_term_answer_model(),
                        prompt_text,
                        _env_int("WORLDMM_MCUR_ANSWER_RETRIES", 1),
                        on_chunk=lambda text: _emit_stream_event(
                            stream_handler,
                            {"type": "delta", "stage": "answer", "delta": text},
                        ),
                    )
                else:
                    answer, fallback_debug = _llm_generate_with_retries(
                        _get_short_term_answer_model(),
                        prompt_text,
                        _env_int("WORLDMM_MCUR_ANSWER_RETRIES", 1),
                    )
                answer_debug["text_only_fallback_debug"] = fallback_debug
            except Exception as fallback_exc:
                answer_debug["text_only_fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
                answer = build_local_current_answer(resolved_question, current_context, selection)
        else:
            answer = build_local_current_answer(resolved_question, current_context, selection)
    generation_ms = _ms(generation_start)

    state = current_context.get("state") or {}
    open_event = current_context.get("open_event") or {}
    current_context_public = {
        "window_start_time": state.get("window_start_time"),
        "window_end_time": state.get("window_end_time"),
        "current_time": state.get("current_time"),
        "open_event": open_event,
        "selected_frame_count": len(selection.get("evidence_frames", []) or []),
        "transcript": current_context.get("transcript", ""),
    }
    if day_context:
        current_context_public["day_context"] = day_context
    evidence_frames = pack_result.get("selected_evidence_frames", selection.get("evidence_frames", []))
    current_memory_results = {
        "current_results": [{"current_context": current_context_public, "current_selection": selection}],
        "short_term_results": [],
        "long_term_results": [],
        "cache_context": cache_context,
    }
    current_fusion_summary = {
        "input_counts": {"M_cur": 1, "M_st": 0, "M_lt": 0, "M_cache": 0},
        "selected_memory_sources": ["M_cur"],
        "dedup_removed": 0,
        "final_evidence_count": 1,
    }
    latency = dict(latency or {})
    latency.update(
        {
            "router_ms": router_ms,
            "text_retrieval_ms": 0,
            "visual_retrieval_ms": 0,
            "short_term_retrieval_ms": 0,
            "fusion_ms": 0,
            "evidence_pack_ms": pack_ms,
            "generation_ms": generation_ms,
            "query_ms": generation_ms + pack_ms,
            "image_evidence_enabled": bool(use_image),
            "image_blocks_count": len(selected_image_paths),
        }
    )
    if total_start is not None:
        latency["total_ms"] = _ms(total_start)
    result = {
        "status": "ok",
        "session_id": session_id,
        "question": question,
        "resolved_question": resolved_question,
        "cache_used": cache_used,
        "cache_hit": cache_hit,
        "cache_mode": cache_mode,
        "cache_context": cache_context,
        "query_type": route_decision.get("query_type"),
        "memory_route": route_decision.get("memory_route", {}),
        "retrieval_plan": route_decision.get("retrieval_plan", {}),
        "route_decision": route_decision,
        "retrieval_mode": "current",
        "retrieval_mode_source": route_decision.get("retrieval_mode_source"),
        "long_term_ready": long_term_ready,
        "short_term_ready": short_term_ready,
        "short_term_only": False,
        "short_term_results": [],
        "current_ready": True,
        "current_used": True,
        "current_stale": False,
        "current_context": current_context_public,
        "visual_embedding_ready": False,
        "visual_fallback": False,
        "use_image_evidence": bool(use_image),
        "use_image_evidence_source": route_decision.get("use_image_evidence_source"),
        "max_image_evidence": max_images,
        "sent_image_count": len(selected_image_paths),
        "memory_completeness": "provisional_only" if not long_term_ready else "partial",
        "used_image_fallback": bool(selected_image_paths),
        "image_fallback_reason": "current_only",
        "image_fallback_reasons": ["current_only"],
        "attached_image_count": len(selected_image_paths),
        "provisional_event_count": 0,
        "answer": answer,
        "timestamps": [
            {
                "start": state.get("window_start_time"),
                "end": state.get("window_end_time"),
            }
        ],
        "evidence_frames": evidence_frames,
        "retrieved_memories": [],
        "supporting_semantic_facts": [],
        "text_results": [],
        "visual_results": [],
        "fused_results": [],
        "memory_results": current_memory_results,
        "fusion_summary": current_fusion_summary,
        "selected_evidence": [],
        "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
        "warnings": route_decision.get("warnings", []),
        "latency": latency,
        "raw": {
            "route_decision": route_decision,
            "retrieval_plan": route_decision.get("retrieval_plan", {}),
            "cache_context": cache_context,
            "memory_results": current_memory_results,
            "fusion_summary": current_fusion_summary,
            "current_context": current_context_public,
            "current_selection": selection,
            "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
            "image_selection_summary": pack_result.get("image_selection_summary", {}),
            "image_mode_used": bool(use_image and selected_image_paths),
            "image_paths_used": selected_image_paths,
            "model_response_text": answer,
            "error_debug": error_debug,
            "fallback_used": fallback_used,
            "answer_debug": answer_debug,
        },
    }
    if day_context_block:
        result["day_context"] = day_context
        result["raw"]["day_context_prompt"] = {"injected": True, "content": day_context_block}
    _emit_stream_event(
        stream_handler,
        {
            "type": "final",
            "stage": "complete",
            "text": answer,
            "answer": answer,
            "raw": result.get("raw", {}),
            "latency": latency,
        },
    )
    return result


def _query_short_term_only(
    *,
    session_id: str,
    question: str,
    sessions_root: Path,
    top_k: int,
    latency: dict[str, Any],
    retrieval_mode: str,
    use_image_evidence: Any,
    max_image_evidence: int | None,
    text_top_k: int | None,
    visual_top_k: int | None,
    final_evidence_k: int | None,
    use_interaction_cache: bool,
    cache_mode: str,
    memory_mode: str = "auto",
    use_current: bool | None = None,
    use_short_term: bool | None = None,
    use_long_term: bool | None = None,
    debug_router: bool = False,
    long_term_retrieval_scheme: str | None = None,
    total_start: float = 0.0,
    day_context: dict[str, Any] | None = None,
    stream_handler: Any = None,
) -> dict[str, Any]:
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    session_dir = sessions_root / session_id
    store = MSTStore(session_dir)
    retriever = MSTRetriever(store)
    current_store = MCurStore(session_dir)
    current_state = current_store.get_state()
    current_ready = _is_current_memory_ready(current_store, current_state)
    current_stale = current_store.is_stale(current_state)
    synthetic_open_event = _frame_open_event_synthetic_evidence(session_dir)
    interaction_cache = InteractionCache(session_id=session_id, session_dir=session_dir)
    cache_mode = str(cache_mode or "auto").strip().lower()
    if cache_mode not in {"auto", "off", "read_only", "write_only"}:
        cache_mode = "auto"
    interaction_enabled = bool(use_interaction_cache) and _env_bool("WORLDMM_INTERACTION_CACHE_ENABLED", True) and cache_mode != "off"
    cache_read_enabled = interaction_enabled and cache_mode in {"auto", "read_only"}
    cache_write_enabled = interaction_enabled and cache_mode in {"auto", "write_only"}
    cache_context: dict[str, Any] = {"cache_hit": False, "is_followup": False, "reason": "no cache context", "confidence": 0.0}
    resolved_question = question
    if cache_read_enabled:
        cache_context = CoreferenceResolver().resolve(question, interaction_cache)
        resolved_question = str(cache_context.get("resolved_question") or question)
        if cache_context.get("confidence", 0.0) < 0.45:
            resolved_question = question
    cache_context = _normalize_frame_timestamps_in_context(cache_context)
    route_start = time.perf_counter()
    route_decision = QueryRouter().route(
        question,
        request_options={
            "retrieval_mode": retrieval_mode,
            "use_image_evidence": use_image_evidence,
            "max_image_evidence": max_image_evidence,
            "top_k": top_k,
            "text_top_k": text_top_k,
            "visual_top_k": visual_top_k or 8,
            "final_evidence_k": final_evidence_k or 4,
            "memory_mode": memory_mode,
            "use_current": use_current,
            "use_short_term": use_short_term,
            "use_long_term": use_long_term,
            "long_term_retrieval_scheme": long_term_retrieval_scheme,
        },
        session_context={
            "visual_ready": False,
            "short_term_ready": bool(store.is_ready() or synthetic_open_event),
            "current_ready": current_ready,
            "current_stale": current_stale,
            "long_term_ready": False,
            "session_id": session_id,
        },
        cache_context=cache_context,
    )
    router_ms = _ms(route_start)
    query_type = route_decision.get("query_type")
    allowed = route_decision.get("memory_route", {}).get("use_short_term") or query_type in {
        "recent_recall",
        "current_perception",
        "entity_tracking",
        "temporal_reasoning",
        "temporal_count",
    }
    any_evidence_available = bool(store.is_ready() or synthetic_open_event or (current_ready and not current_stale))
    if not allowed and any_evidence_available:
        route_decision.setdefault("warnings", []).append("preferred memory is not ready; using available current/short-term fallback")
        memory_route = route_decision.setdefault("memory_route", {})
        if synthetic_open_event or store.is_ready():
            memory_route["use_short_term"] = True
        elif current_ready and not current_stale:
            memory_route["use_current"] = True
        allowed = True
    if not allowed:
        status = _status_not_ready(session_dir)
        status.update(
            {
                "session_id": session_id,
                "long_term_ready": False,
                "short_term_ready": store.is_ready(),
                "short_term_only": False,
                "message": "long-term memory is not ready for this query type",
                "route_decision": route_decision,
                "memory_route": route_decision.get("memory_route", {}),
            }
        )
        return status

    search_start = time.perf_counter()
    short_term_results = retriever.search(resolved_question, top_k=top_k, cache_context=cache_context) if store.is_ready() else []
    if not short_term_results and synthetic_open_event:
        short_term_results = [synthetic_open_event]
    search_ms = _ms(search_start)
    if not short_term_results and current_ready and not current_stale:
        current_route = dict(route_decision)
        current_route["query_type"] = route_decision.get("query_type") or "recent_recall"
        current_route["retrieval_mode"] = "current"
        current_route.setdefault("memory_route", {})
        current_route["memory_route"]["use_current"] = True
        current_route["memory_route"]["use_short_term"] = False
        current_route["memory_route"]["use_long_term"] = False
        result = _answer_current_memory(
            session_id=session_id,
            session_dir=session_dir,
            question=question,
            resolved_question=resolved_question,
            route_decision=current_route,
            cache_context=cache_context,
            latency={**latency, "short_term_retrieval_ms": search_ms},
            router_ms=router_ms,
            long_term_ready=False,
            short_term_ready=False,
            cache_used=interaction_enabled,
            cache_hit=bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
            cache_mode=cache_mode,
            total_start=total_start,
            day_context=day_context,
        )
        return _mark_partial_fallback(
            result,
            reason="current_only",
            sources=["M_cur_fallback"],
            warning="answer is based on current frame only because short-term/long-term memory is still building",
        )
    partial_diag = _diagnose_partial_memory(
        question=question,
        route_decision=route_decision,
        current_results=[],
        short_term_results=short_term_results,
        memory_config={},
        visual_ready=False,
        long_term_ready=False,
    )
    if synthetic_open_event and short_term_results and short_term_results[0].get("synthetic"):
        partial_diag["memory_completeness"] = "provisional_only"
        partial_diag["image_fallback_reasons"] = list(dict.fromkeys(["short_term_provisional", "refined_missing"] + list(partial_diag.get("image_fallback_reasons") or [])))
        partial_diag["image_fallback_reason"] = partial_diag.get("image_fallback_reason") or "short_term_provisional"
    _apply_image_fallback_to_route(route_decision, partial_diag)
    packer = EvidencePacker(session_dir=session_dir)
    pack_start = time.perf_counter()
    pack_result = packer.pack(
        query=question,
        route_decision=route_decision,
        retrieval_result={
            "text_results": [],
            "visual_results": [],
            "fused_results": [],
            "evidence_frames": [],
            "short_term_results": short_term_results,
        },
        cache_context=cache_context,
    )
    pack_ms = _ms(pack_start)
    evidence_frames = pack_result.get("selected_evidence_frames", [])
    selected_image_paths = list(pack_result.get("selected_image_paths_for_mllm") or [])
    timestamps = [
        {"start": _safe_float(item.get("start_time")), "end": _safe_float(item.get("end_time"))}
        for item in short_term_results
    ]
    local_answer = _build_short_term_answer(question, short_term_results)
    answer = local_answer
    generation_start = time.perf_counter()
    answer_debug: dict[str, Any] = {
        "answer_model": (
            os.getenv("WORLDMM_MST_ANSWER_MODEL")
            or os.getenv("WORLDMM_QUERY_RESPOND_MODEL")
            or os.getenv("WORLDMM_RESPOND_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-5.4"
        ),
        "llm_enabled": _env_bool("WORLDMM_MST_ANSWER_WITH_LLM", True),
        "fallback_used": False,
        "local_fallback_answer": local_answer,
    }
    if answer_debug["llm_enabled"]:
        try:
            prompt = _build_short_term_llm_prompt(
                resolved_question,
                short_term_results,
                pack_result.get("evidence_pack_summary", {}),
            )
            if selected_image_paths:
                user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for rel_path in selected_image_paths[: int(route_decision.get("max_image_evidence") or 4)]:
                    if (session_dir / rel_path).exists():
                        user_content.append({"type": "image", "image": session_dir / rel_path})
                prompt_or_messages: Any = [
                    {
                        "role": "system",
                        "content": (
                            "Answer recent-video questions using only the retrieved micro-events and attached frames. "
                            "Treat provisional captions as uncertain. "
                            f"{_answer_language_instruction()}"
                        ),
                    },
                    {"role": "user", "content": user_content},
                ]
            else:
                prompt_or_messages = prompt
            if callable(stream_handler):
                answer, llm_debug = _llm_stream_with_retries(
                    _get_short_term_answer_model(),
                    prompt_or_messages,
                    _env_int("WORLDMM_MST_ANSWER_RETRIES", _env_int("WORLDMM_QUERY_ANSWER_RETRIES", 3)),
                    on_chunk=lambda text: _emit_stream_event(
                        stream_handler,
                        {"type": "delta", "stage": "answer", "delta": text},
                    ),
                )
            else:
                answer, llm_debug = _llm_generate_with_retries(
                    _get_short_term_answer_model(),
                    prompt_or_messages,
                    _env_int("WORLDMM_MST_ANSWER_RETRIES", _env_int("WORLDMM_QUERY_ANSWER_RETRIES", 3)),
                )
            answer_debug["llm_debug"] = llm_debug
        except Exception as exc:
            answer = local_answer
            answer_debug["fallback_used"] = True
            answer_debug["error_debug"] = f"{type(exc).__name__}: {exc}"
    generation_ms = _ms(generation_start)
    result = {
        "status": "ok",
        "session_id": session_id,
        "question": question,
        "resolved_question": resolved_question,
        "query_type": query_type,
        "route_decision": route_decision,
        "memory_route": route_decision.get("memory_route", {}),
        "retrieval_mode": "short_term_only",
        "retrieval_mode_source": route_decision.get("retrieval_mode_source"),
        "long_term_ready": False,
        "short_term_ready": bool(store.is_ready() or synthetic_open_event),
        "short_term_only": True,
        "short_term_results": short_term_results,
        "visual_embedding_ready": False,
        "visual_fallback": True,
        "use_image_evidence": bool(route_decision.get("use_image_evidence")),
        "use_image_evidence_source": route_decision.get("use_image_evidence_source"),
        "max_image_evidence": int(route_decision.get("max_image_evidence") or 0),
        "sent_image_count": len(selected_image_paths),
        "memory_completeness": partial_diag.get("memory_completeness"),
        "used_image_fallback": bool(selected_image_paths),
        "image_fallback_reason": partial_diag.get("image_fallback_reason"),
        "image_fallback_reasons": partial_diag.get("image_fallback_reasons", []),
        "attached_image_count": len(selected_image_paths),
        "provisional_event_count": partial_diag.get("provisional_event_count", 0),
        "answer": answer,
        "timestamps": timestamps,
        "evidence_frames": evidence_frames,
        "retrieved_memories": [],
        "supporting_semantic_facts": [],
        "text_results": [],
        "visual_results": [],
        "fused_results": [],
        "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
        "cache_used": interaction_enabled,
        "cache_hit": bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
        "cache_mode": cache_mode,
        "cache_context": cache_context,
        "warnings": [
            (
                "short-term event is still open; using provisional frame-stream evidence"
                if synthetic_open_event and short_term_results and short_term_results[0].get("synthetic")
                else "long-term memory is not ready; answering from short-term memory only"
            )
        ],
        "latency": {
            **latency,
            "router_ms": router_ms,
            "short_term_retrieval_ms": search_ms,
            "evidence_pack_ms": pack_ms,
            "generation_ms": generation_ms,
            "total_ms": _ms(total_start),
        },
        "raw": {
            "route_decision": route_decision,
            "cache_context": cache_context,
            "mst_state": store.get_state(),
            "evidence_pack_summary": pack_result.get("evidence_pack_summary", {}),
            "answer_debug": answer_debug,
            "partial_memory_diagnostics": {**partial_diag, "attached_image_count": len(selected_image_paths)},
        },
    }
    _emit_stream_event(
        stream_handler,
        {
            "type": "final",
            "stage": "complete",
            "text": answer,
            "answer": answer,
            "raw": result.get("raw", {}),
            "latency": result.get("latency", {}),
        },
    )
    if synthetic_open_event and short_term_results and short_term_results[0].get("synthetic"):
        _mark_partial_fallback(
            result,
            reason="short_term_provisional",
            sources=["M_st_open_event"],
            warning="short-term event is still open; using provisional frame-stream evidence",
        )
    elif short_term_results:
        _mark_partial_fallback(
            result,
            reason="long_term_missing",
            sources=["M_st_partial"],
            warning="long-term memory is still building; using short-term fallback",
        )
    cache_update = {"updated": False, "reason": "cache_mode does not allow writes"}
    if cache_write_enabled:
        cache_update = interaction_cache.update_from_query_result(question, resolved_question, result)
    result["cache_update"] = cache_update
    result["raw"]["cache_update"] = cache_update
    return result


def query_session(
    session_id: str,
    question: str,
    sessions_root: Path = Path("online_sessions"),
    top_k: int = 5,
    no_cache: bool = False,
    cache: Any = None,
    output_json: Path | None = None,
    use_image_evidence: Any = "auto",
    max_image_frames: int = 4,
    retrieval_mode: str = "auto",
    max_image_evidence: int | None = 3,
    text_top_k: int | None = None,
    visual_top_k: int | None = None,
    final_evidence_k: int | None = None,
    use_interaction_cache: bool = True,
    cache_mode: str = "auto",
    memory_mode: str = "auto",
    use_current: bool | None = None,
    use_short_term: bool | None = None,
    use_long_term: bool | None = None,
    debug_router: bool = False,
    long_term_retrieval_scheme: str | None = None,
    stream_handler: Any = None,
) -> dict[str, Any]:
    from .query_cache import GLOBAL_SESSION_ENGINE_CACHE

    total_start = time.perf_counter()
    eval_trace = _new_eval_trace()
    query_context = resolve_query_session_context(session_id, sessions_root)
    requested_session_id = session_id
    realtime_session_id = str(query_context.get("realtime_session_id") or session_id)
    short_term_session_id = str(query_context.get("short_term_session_id") or realtime_session_id)
    interaction_cache_session_id = str(query_context.get("interaction_cache_session_id") or realtime_session_id)
    long_term_selection = resolve_query_long_term_candidates(
        session_id,
        sessions_root,
        question=question,
        query_context=query_context,
    )
    long_term_session_id = str(long_term_selection.get("selected_session_id") or query_context.get("long_term_session_id") or session_id)
    query_context["active_long_term_session_id"] = long_term_session_id
    query_context["long_term_selection"] = long_term_selection
    session_dir = sessions_root / realtime_session_id
    short_term_session_dir = sessions_root / short_term_session_id
    long_term_session_dir = sessions_root / long_term_session_id
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)

    def _finalize_stream_result(result: dict[str, Any]) -> dict[str, Any]:
        result = _finalize_eval_trace(result, eval_trace)
        result["session_id"] = requested_session_id
        result["query_session_context"] = query_context
        result["day_context"] = {
            "day_label": query_context.get("day_label"),
            "day_index": query_context.get("day_index"),
            "run_id": query_context.get("run_id"),
        } if query_context.get("is_rokid_day_child") else None
        result["long_term_retrieval_scheme"] = long_term_retrieval_scheme
        raw = result.setdefault("raw", {})
        raw["long_term_retrieval_scheme"] = long_term_retrieval_scheme
        raw["query_session_context"] = query_context
        raw["long_term_selection"] = long_term_selection
        try:
            return _attach_stream_query_awareness(
                result,
                session_id=requested_session_id,
                session_dir=session_dir,
                sessions_root=sessions_root,
                project_root=PROJECT_ROOT,
                question=question,
            )
        except Exception as exc:
            result.setdefault("stream_context_error", str(exc))
            return result

    if _wants_current_fast_path(
        question,
        retrieval_mode=retrieval_mode,
        memory_mode=memory_mode,
        use_current=use_current,
        use_short_term=use_short_term,
        use_long_term=use_long_term,
    ):
        current_store = MCurStore(session_dir)
        current_state = current_store.get_state()
        current_ready = _is_current_memory_ready(current_store, current_state)
        current_stale = current_store.is_stale(current_state)
        if current_ready and not current_stale:
            short_term_ready = MSTStore(session_dir).is_ready()
            fast_route_decision = QueryRouter().route(
                question,
                request_options={
                    "retrieval_mode": "current",
                    "use_image_evidence": use_image_evidence,
                    "max_image_evidence": min(
                        int(max_image_evidence) if max_image_evidence is not None else _env_int("WORLDMM_MCUR_MAX_QUERY_IMAGES", 3),
                        _env_int("WORLDMM_MCUR_MAX_QUERY_IMAGES", 3),
                    ),
                    "top_k": top_k,
                    "text_top_k": text_top_k,
                    "visual_top_k": visual_top_k,
                    "final_evidence_k": final_evidence_k,
                    "use_current": True,
                    "use_short_term": False,
                    "use_long_term": False,
                    "use_interaction_cache": False,
                    "long_term_retrieval_scheme": long_term_retrieval_scheme,
                },
                session_context={
                    "visual_ready": False,
                    "short_term_ready": short_term_ready,
                    "current_ready": current_ready,
                    "current_stale": current_stale,
                    "long_term_ready": False,
                    "session_id": requested_session_id,
                },
                cache_context={},
            )
            fast_route_decision["retrieval_mode_source"] = "current_fast_path"
            fast_route_decision["fast_path"] = "M_cur_direct"
            result = _answer_current_memory(
                session_id=requested_session_id,
                session_dir=session_dir,
                question=question,
                resolved_question=question,
                route_decision=fast_route_decision,
                cache_context={"cache_hit": False, "is_followup": False, "confidence": 0.0, "reason": "current fast path skips interaction cache"},
                latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0, "fast_path": "M_cur_direct"},
                router_ms=0,
                long_term_ready=query_memory_ready(long_term_session_dir),
                short_term_ready=short_term_ready,
                cache_used=False,
                cache_hit=False,
                cache_mode="off",
                total_start=total_start,
                day_context=query_context if query_context.get("is_rokid_day_child") else None,
                stream_handler=stream_handler,
            )
            result["fast_path"] = "M_cur_direct"
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result

    if not query_memory_ready(long_term_session_dir):
        current_store = MCurStore(session_dir)
        store = MSTStore(short_term_session_dir)
        current_state = current_store.get_state()
        current_ready = _is_current_memory_ready(current_store, current_state)
        current_stale = current_store.is_stale(current_state)
        if _is_memory_status_question(question) and session_dir.exists():
            result = _answer_memory_status_question(requested_session_id, session_dir, question, total_start=total_start)
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result
        route_decision = QueryRouter().route(
            question,
            request_options={
                "retrieval_mode": retrieval_mode,
                "use_image_evidence": use_image_evidence,
                "max_image_evidence": max_image_evidence,
                "top_k": top_k,
                "text_top_k": text_top_k,
                "visual_top_k": visual_top_k,
                "final_evidence_k": final_evidence_k,
                "use_current": use_current,
                "use_short_term": use_short_term,
                "use_long_term": use_long_term,
                "use_interaction_cache": use_interaction_cache,
                "long_term_retrieval_scheme": long_term_retrieval_scheme,
            },
            session_context={
                "visual_ready": False,
                "short_term_ready": store.is_ready(),
                "current_ready": current_ready,
                "current_stale": current_stale,
                "long_term_ready": False,
                "session_id": requested_session_id,
            },
            cache_context={},
        )
        if use_current is not False and route_decision.get("retrieval_mode") == "current" and current_ready and not current_stale:
            result = _answer_current_memory(
                session_id=requested_session_id,
                session_dir=session_dir,
                question=question,
                resolved_question=question,
                route_decision=route_decision,
                cache_context={},
                latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0},
                router_ms=0,
                long_term_ready=False,
                short_term_ready=store.is_ready(),
                cache_used=False,
                cache_hit=False,
                cache_mode="off",
                total_start=total_start,
                day_context=query_context if query_context.get("is_rokid_day_child") else None,
                stream_handler=stream_handler,
            )
            result["short_term_only"] = False
            result = _mark_partial_fallback(
                result,
                reason="current_only",
                sources=["M_cur_fallback"],
                warning="answer is based on current frame only because long-term memory is still building",
            )
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result
        if store.is_ready():
            result = _query_short_term_only(
                session_id=short_term_session_id,
                question=question,
                sessions_root=sessions_root,
                top_k=top_k,
                latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0},
                retrieval_mode=retrieval_mode,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                text_top_k=text_top_k,
                visual_top_k=visual_top_k,
                final_evidence_k=final_evidence_k,
                use_interaction_cache=use_interaction_cache,
                cache_mode=cache_mode,
                use_current=use_current,
                use_short_term=use_short_term,
                use_long_term=use_long_term,
                debug_router=debug_router,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                total_start=total_start,
                day_context=query_context if query_context.get("is_rokid_day_child") else None,
                stream_handler=stream_handler,
            )
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result
        if _frame_open_event_synthetic_evidence(session_dir) or (current_ready and not current_stale):
            result = _query_short_term_only(
                session_id=short_term_session_id,
                question=question,
                sessions_root=sessions_root,
                top_k=top_k,
                latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0},
                retrieval_mode=retrieval_mode,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                text_top_k=text_top_k,
                visual_top_k=visual_top_k,
                final_evidence_k=final_evidence_k,
                use_interaction_cache=use_interaction_cache,
                cache_mode=cache_mode,
                use_current=use_current,
                use_short_term=use_short_term,
                use_long_term=use_long_term,
                debug_router=debug_router,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                total_start=total_start,
                day_context=query_context if query_context.get("is_rokid_day_child") else None,
                stream_handler=stream_handler,
            )
            result = _mark_partial_fallback(
                result,
                reason="fallback_memory_available",
                sources=result.get("used_memory_sources") or ["M_st_open_event" if _frame_open_event_synthetic_evidence(session_dir) else "M_cur_fallback"],
                warning="long-term memory is still building; using available current/short-term fallback",
            )
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result
        output = _status_not_ready(session_dir)
        output["session_id"] = requested_session_id
        output["short_term_ready"] = False
        output["current_ready"] = current_ready
        output["current_stale"] = current_stale
        output["long_term_ready"] = False
        output = _finalize_stream_result(output)
        if output_json:
            write_json(output_json, output)
        return output

    current_store = MCurStore(session_dir)
    current_state = current_store.get_state()
    current_ready = _is_current_memory_ready(current_store, current_state)
    current_stale = current_store.is_stale(current_state)
    short_term_ready = MSTStore(short_term_session_dir).is_ready()
    if _is_memory_status_question(question) and session_dir.exists():
        result = _answer_memory_status_question(requested_session_id, session_dir, question, total_start=total_start)
        result = _finalize_stream_result(result)
        if output_json:
            write_json(output_json, result)
        return result
    if use_current is not False and current_ready and not current_stale and str(memory_mode or "auto").strip().lower() != "auto":
        direct_cache_mode = str(cache_mode or "auto").strip().lower()
        if direct_cache_mode not in {"auto", "off", "read_only", "write_only"}:
            direct_cache_mode = "auto"
        interaction_enabled = bool(use_interaction_cache) and _env_bool("WORLDMM_INTERACTION_CACHE_ENABLED", True) and direct_cache_mode != "off"
        cache_read_enabled = interaction_enabled and direct_cache_mode in {"auto", "read_only"}
        cache_write_enabled = interaction_enabled and direct_cache_mode in {"auto", "write_only"}
        interaction_cache = InteractionCache(session_id=interaction_cache_session_id, session_dir=sessions_root / interaction_cache_session_id)
        cache_context: dict[str, Any] = {"cache_hit": False, "is_followup": False, "confidence": 0.0, "reason": "no cache context"}
        resolved_question = question
        if cache_read_enabled:
            try:
                cache_context = CoreferenceResolver().resolve(question, interaction_cache)
                resolved_question = str(cache_context.get("resolved_question") or question)
                if cache_context.get("confidence", 0.0) < 0.45:
                    resolved_question = question
            except Exception as exc:
                cache_context = {"cache_hit": False, "is_followup": False, "confidence": 0.0, "reason": f"cache resolver failed: {exc}"}
        cache_context = _normalize_frame_timestamps_in_context(cache_context)
        direct_router_start = time.perf_counter()
        direct_route_decision = QueryRouter().route(
            question,
            request_options={
                "retrieval_mode": retrieval_mode,
                "use_image_evidence": use_image_evidence,
                "max_image_evidence": max_image_evidence,
                "top_k": top_k,
                "text_top_k": text_top_k,
                "visual_top_k": visual_top_k,
                "final_evidence_k": final_evidence_k,
                "use_current": use_current,
                "use_short_term": use_short_term,
                "use_long_term": use_long_term,
                "use_interaction_cache": interaction_enabled,
                "long_term_retrieval_scheme": long_term_retrieval_scheme,
            },
            session_context={
                "visual_ready": False,
                "short_term_ready": short_term_ready,
                "current_ready": current_ready,
                "current_stale": current_stale,
                "long_term_ready": True,
                "session_id": requested_session_id,
            },
            cache_context=cache_context,
        )
        direct_router_ms = _ms(direct_router_start)
        direct_memory_route = direct_route_decision.get("memory_route") or {}
        direct_current_only = (
            bool(direct_memory_route.get("use_current"))
            and not direct_memory_route.get("use_short_term")
            and not direct_memory_route.get("use_long_term")
        )
        if direct_route_decision.get("retrieval_mode") == "current" or direct_current_only:
            result = _answer_current_memory(
                session_id=requested_session_id,
                session_dir=session_dir,
                question=question,
                resolved_question=resolved_question,
                route_decision=direct_route_decision,
                cache_context=cache_context,
                latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0},
                router_ms=direct_router_ms,
                long_term_ready=True,
                short_term_ready=short_term_ready,
                cache_used=interaction_enabled,
                cache_hit=bool(cache_context.get("cache_hit") and cache_context.get("is_followup")),
                cache_mode=direct_cache_mode,
                total_start=total_start,
                day_context=query_context if query_context.get("is_rokid_day_child") else None,
                stream_handler=stream_handler,
            )
            cache_update = {"updated": False, "reason": "cache_mode does not allow writes"}
            if cache_write_enabled and result.get("status") == "ok":
                try:
                    cache_update = interaction_cache.update_from_query_result(question, resolved_question, result)
                except Exception as exc:
                    cache_update = {"updated": False, "error": str(exc)}
            result["cache_update"] = cache_update
            result.setdefault("raw", {})["cache_update"] = cache_update
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result

    cache = cache or GLOBAL_SESSION_ENGINE_CACHE
    try:
        if no_cache:
            load_start = time.perf_counter()
            engine = load_query_engine(
                long_term_session_id,
                sessions_root=sessions_root,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
            )
            engine_load_ms = _ms(load_start)
            cache_hit = False
            cache_lookup_ms = 0
        else:
            lookup_start = time.perf_counter()
            engine, cache_hit, engine_load_ms = cache.get_or_load(
                session_id=long_term_session_id,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                loader=lambda sid: load_query_engine(
                    sid,
                    sessions_root=sessions_root,
                    long_term_retrieval_scheme=long_term_retrieval_scheme,
                ),
            )
            cache_lookup_ms = _ms(lookup_start) - engine_load_ms
    except Exception as exc:
        current_store = MCurStore(session_dir)
        current_state = current_store.get_state()
        current_ready = _is_current_memory_ready(current_store, current_state)
        current_stale = current_store.is_stale(current_state)
        store = MSTStore(short_term_session_dir)
        if use_current is not False and current_ready and not current_stale:
            route_decision = QueryRouter().route(
                question,
                request_options={
                    "retrieval_mode": retrieval_mode,
                    "use_image_evidence": use_image_evidence,
                    "max_image_evidence": max_image_evidence,
                    "top_k": top_k,
                    "text_top_k": text_top_k,
                    "visual_top_k": visual_top_k,
                    "final_evidence_k": final_evidence_k,
                    "memory_mode": memory_mode,
                    "use_current": use_current,
                    "use_short_term": use_short_term,
                    "use_long_term": use_long_term,
                    "use_interaction_cache": use_interaction_cache,
                    "long_term_retrieval_scheme": long_term_retrieval_scheme,
                },
                session_context={
                    "visual_ready": False,
                    "short_term_ready": store.is_ready(),
                    "current_ready": current_ready,
                    "current_stale": current_stale,
                    "long_term_ready": False,
                    "session_id": requested_session_id,
                },
                cache_context={},
            )
            route_memory = route_decision.get("memory_route") or {}
            route_current_only = (
                bool(route_memory.get("use_current"))
                and not route_memory.get("use_short_term")
                and not route_memory.get("use_long_term")
            )
            if route_decision.get("retrieval_mode") == "current" or route_current_only:
                result = _answer_current_memory(
                    session_id=requested_session_id,
                    session_dir=session_dir,
                    question=question,
                    resolved_question=question,
                    route_decision=route_decision,
                    cache_context={},
                    latency={"cache_lookup_ms": 0, "cache_hit": False, "engine_load_ms": 0, "load_error": str(exc)},
                    router_ms=0,
                    long_term_ready=False,
                    short_term_ready=store.is_ready(),
                    cache_used=False,
                    cache_hit=False,
                    cache_mode="off",
                    total_start=total_start,
                    day_context=query_context if query_context.get("is_rokid_day_child") else None,
                    stream_handler=stream_handler,
                )
                result.setdefault("warnings", []).append(f"long-term snapshot load failed; using current fallback: {exc}")
                result["long_term_available"] = False
                result["fallback_reason"] = "no loadable ready long-term snapshot"
                result.setdefault("raw", {})["long_term_load_error"] = str(exc)
                result = _mark_partial_fallback(
                    result,
                    reason="current_only",
                    sources=["M_cur_fallback"],
                    warning="answer is based on current frame only because long-term memory is unavailable",
                )
                result = _finalize_stream_result(result)
                if output_json:
                    write_json(output_json, result)
                return result
        if store.is_ready() or _frame_open_event_synthetic_evidence(session_dir) or (current_ready and not current_stale):
            result = _query_short_term_only(
                session_id=short_term_session_id,
                question=question,
                sessions_root=sessions_root,
                top_k=top_k,
                latency={
                    "cache_lookup_ms": 0,
                    "cache_hit": False,
                    "engine_load_ms": 0,
                    "load_error": str(exc),
                },
                retrieval_mode=retrieval_mode,
                use_image_evidence=use_image_evidence,
                max_image_evidence=max_image_evidence,
                text_top_k=text_top_k,
                visual_top_k=visual_top_k,
                final_evidence_k=final_evidence_k,
                use_interaction_cache=use_interaction_cache,
                cache_mode=cache_mode,
                memory_mode=memory_mode,
                use_current=use_current,
                use_short_term=use_short_term,
                use_long_term=use_long_term,
                debug_router=debug_router,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                total_start=total_start,
                stream_handler=stream_handler,
            )
            result.setdefault("warnings", []).append(f"long-term snapshot load failed; using current/short-term fallback: {exc}")
            result["long_term_available"] = False
            result["fallback_reason"] = "no loadable ready long-term snapshot"
            result.setdefault("raw", {})["long_term_load_error"] = str(exc)
            result = _finalize_stream_result(result)
            if output_json:
                write_json(output_json, result)
            return result
        output = _status_not_ready(session_dir)
        output.update(
            {
                "session_id": requested_session_id,
                "long_term_ready": False,
                "long_term_available": False,
                "fallback_reason": "no loadable ready long-term snapshot",
                "load_error": str(exc),
            }
        )
        output = _finalize_stream_result(output)
        if output_json:
            write_json(output_json, output)
        return output

    with engine.runtime_session_context(
        realtime_session_id=realtime_session_id,
        short_term_session_id=short_term_session_id,
        interaction_cache_session_id=interaction_cache_session_id,
        requested_session_id=requested_session_id,
        day_context=query_context if query_context.get("is_rokid_day_child") else None,
    ):
        result = engine.query(
            question=question,
            top_k=top_k,
            use_image_evidence=use_image_evidence,
            max_image_frames=max_image_frames,
            retrieval_mode=retrieval_mode,
            max_image_evidence=max_image_evidence,
            text_top_k=text_top_k,
            visual_top_k=visual_top_k or 8,
            final_evidence_k=final_evidence_k or 4,
            use_interaction_cache=use_interaction_cache,
            cache_mode=cache_mode,
            memory_mode=memory_mode,
            use_current=use_current,
            use_short_term=use_short_term,
            use_long_term=use_long_term,
            debug_router=debug_router,
            latency={
                "cache_lookup_ms": max(0, cache_lookup_ms),
                "cache_hit": cache_hit,
                "engine_load_ms": engine_load_ms,
            },
            stream_handler=stream_handler,
            eval_trace=eval_trace,
        )
    result["latency"]["total_ms"] = _ms(total_start)
    result["long_term_retrieval_scheme"] = engine.long_term_retrieval_scheme
    result.setdefault("raw", {})["long_term_retrieval_scheme"] = engine.long_term_retrieval_scheme
    result["active_query_memory_version"] = engine.active_query_memory_version
    result["latest_ready_memory_version"] = engine.latest_ready_memory_version
    result["building_memory_version"] = engine.building_memory_version
    result.setdefault("memory_component_versions", engine._memory_component_versions())
    result["using_stale_while_building"] = bool(
        engine.building_memory_version
        and engine.active_query_memory_version
        and int(engine.building_memory_version) > int(engine.active_query_memory_version)
    )
    result = _finalize_stream_result(result)
    if no_cache:
        engine.close()
    if output_json:
        write_json(output_json, result)
    return result
