from __future__ import annotations

from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import queue_counts
from online_pipeline.rokid_day import resolve_query_long_term_candidates, resolve_query_session_context
from online_preprocess.io_utils import read_json


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _question_policy(question: str, *, stream_status: str, current_ready: bool, short_term_ready: bool, long_term_partial_ready: bool, asr_lagging: bool, semantic_lagging: bool, graph_lagging: bool) -> str:
    q = str(question or "").lower()
    current_words = (
        "现在", "当前", "此刻", "正在", "目前", "当前画面", "当前场景",
        "now", "currently", "right now", "at this moment", "at the moment", "current frame",
        "in the current frame", "current scene", "current view", "current screen",
        "what is happening now", "what's happening now", "what is going on now",
        "what's going on now", "what is in the current scene", "what am i seeing",
        "what am i looking at", "what is on screen", "what's on screen", "what is on the screen",
        "what's on the screen", "describe current scene", "describe the current scene",
    )
    recent_words = (
        "刚才", "刚刚", "最近", "上一段", "刚发生",
        "just now", "recently", "earlier", "a moment ago", "a few seconds ago", "moments ago",
        "previous moment", "previously", "last segment", "last scene", "what just happened",
    )
    summary_words = (
        "总结", "概括", "到目前为止", "目前为止", "到现在", "从开始", "从头", "主要发生", "整个视频", "全过程",
        "summary", "summarize", "recap", "overall", "overview", "brief summary",
        "briefly summarize", "main content", "main points", "what has happened so far",
        "what happened so far", "what is this video about", "describe the video", "so far", "up to now",
    )
    semantic_words = (
        "关系", "状态变化", "长期事实", "人物和物体",
        "relation", "relationship", "state change", "long-term fact", "semantic memory",
        "people and objects", "person and object",
    )
    count_words = (
        "一共", "总共", "总计", "总共有", "一共有", "多少", "几个", "几次", "几幅", "几张", "数量",
        "count", "how many", "total", "in total", "altogether",
    )
    span_words = (
        "从刚才到现在", "从刚刚到现在", "刚才到现在", "刚刚到现在", "到现在", "到目前为止",
        "从开始", "从头", "目前为止", "since just now", "from just now", "so far", "up to now",
        "from the beginning",
    )
    has_recent = any(word in q for word in recent_words)
    has_summary = any(word in q for word in summary_words)
    has_span = any(word in q for word in span_words)
    has_count = any(word in q for word in count_words)
    if has_count and (has_span or has_recent):
        return "partial_summary" if long_term_partial_ready else ("recent_fast" if (current_ready or short_term_ready) else "limited")
    if has_recent:
        return "recent_fast" if (current_ready or short_term_ready or long_term_partial_ready) else "limited"
    if any(word in q for word in semantic_words):
        return "partial_summary" if (semantic_lagging or graph_lagging) and long_term_partial_ready else ("full_summary" if long_term_partial_ready else "limited")
    if has_summary:
        if stream_status in {"running", "ending"} or semantic_lagging or asr_lagging:
            return "partial_summary" if long_term_partial_ready or short_term_ready or current_ready else "limited"
        return "full_summary" if long_term_partial_ready else ("partial_summary" if short_term_ready or current_ready else "limited")
    if any(word in q for word in current_words):
        return "current_fast" if current_ready else "limited"
    return "recent_fast" if (current_ready or short_term_ready) else ("partial_summary" if long_term_partial_ready else "limited")


def load_stream_query_context(
    session_id: str,
    *,
    sessions_root: Path = Path("online_sessions"),
    project_root: Path | None = None,
    question: str = "",
) -> dict[str, Any]:
    sessions_root = Path(sessions_root)
    session_dir = sessions_root / session_id
    try:
        query_context = resolve_query_session_context(session_id, sessions_root)
    except Exception:
        query_context = {
            "is_rokid_day_child": False,
            "long_term_session_id": session_id,
            "parent_session_id": session_id,
        }
    long_term_selection = resolve_query_long_term_candidates(
        session_id,
        sessions_root,
        question=question,
        query_context=query_context,
    )
    long_term_session_id = str(long_term_selection.get("selected_session_id") or query_context.get("long_term_session_id") or session_id)
    long_term_session_dir = sessions_root / long_term_session_id
    stream_state_path = session_dir / "stream" / "stream_state.json"
    stream_state = read_json(stream_state_path, default={})
    if not isinstance(stream_state, dict) or not stream_state_path.exists():
        return {
            "is_stream_session": False,
            "recommended_answer_policy": "full_summary",
            "long_term_session_id": long_term_session_id,
            "parent_session_id": query_context.get("parent_session_id"),
            "is_rokid_day_child": bool(query_context.get("is_rokid_day_child")),
            "long_term_selection": long_term_selection,
        }
    pipeline_state = read_json(session_dir / "pipeline_state.json", default={})
    long_term_pipeline_state = read_json(long_term_session_dir / "pipeline_state.json", default={})
    memory_config = read_json(long_term_session_dir / "em2mem" / "memory_config.json", default={})
    transcript_state = read_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
    frame_state = read_json(session_dir / "stream" / "frame_state.json", default={})
    frame_event_state = read_json(session_dir / "stream" / "frame_event_state.json", default={})
    current_state = read_json(session_dir / "current" / "current_state.json", default={})
    query_runtime = read_json((project_root or Path.cwd()) / "online_tasks" / "query_runtime.json", default={}) if project_root else {}
    if not isinstance(pipeline_state, dict):
        pipeline_state = {}
    if not isinstance(long_term_pipeline_state, dict):
        long_term_pipeline_state = {}
    if not isinstance(memory_config, dict):
        memory_config = {}
    if not isinstance(transcript_state, dict):
        transcript_state = {}
    if not isinstance(frame_state, dict):
        frame_state = {}
    if not isinstance(frame_event_state, dict):
        frame_event_state = {}
    if not isinstance(current_state, dict):
        current_state = {}
    current = pipeline_state.get("current") if isinstance(pipeline_state.get("current"), dict) else {}
    short_term = pipeline_state.get("short_term") if isinstance(pipeline_state.get("short_term"), dict) else {}
    long_term = long_term_pipeline_state.get("long_term") if isinstance(long_term_pipeline_state.get("long_term"), dict) else {}
    upload_chunks = [item for item in stream_state.get("upload_chunks", stream_state.get("received_chunks", [])) or [] if isinstance(item, dict)]
    processing_chunks = [item for item in stream_state.get("processing_chunks", []) or [] if isinstance(item, dict)]
    latest_uploaded = max([_as_int(item.get("upload_chunk_index", item.get("chunk_index", -1)), -1) for item in upload_chunks], default=-1)
    latest_processed = _as_int(stream_state.get("last_processed_proc_index", stream_state.get("last_processed_chunk_index", -1)), -1)
    latest_asr = _as_int(transcript_state.get("last_asr_chunk_index"), -1)
    latest_asr_window = transcript_state.get("last_asr_window_id")
    transcript_segment_count = _as_int(transcript_state.get("segment_count"), 0)
    counts = queue_counts(project_root or Path.cwd())
    asr_pending = _as_int(counts.get("stream_asr_queued"), 0) + _as_int(counts.get("stream_asr_in_progress"), 0)
    memory_pending = _as_int(counts.get("memory_queued"), 0) + _as_int(counts.get("memory_in_progress"), 0)
    frame_stream_ready = bool(frame_state.get("mcur_ready") or frame_state.get("ready"))
    frame_open_event = frame_event_state.get("open_event") if isinstance(frame_event_state.get("open_event"), dict) else {}
    frame_open_event_ready = bool(frame_open_event)
    current_text_ready = bool(
        current_state.get("current_text_ready")
        or current_state.get("audio_current_ready")
        or _as_int(current_state.get("transcript_segment_count"), 0) > 0
        or transcript_segment_count > 0
    )
    current_ready = bool(current.get("ready") or frame_stream_ready or current_text_ready)
    short_term_ready = bool(short_term.get("ready") or frame_open_event_ready)
    long_term_partial_ready = bool(long_term.get("long_term_partial_ready") or memory_config.get("latest_fast_ready_version") or memory_config.get("latest_ready_memory_version"))
    long_term_full_ready = bool(long_term.get("long_term_full_ready"))
    latest_fast = long_term.get("latest_fast_ready_version") or memory_config.get("latest_fast_ready_version")
    latest_semantic = long_term.get("latest_semantic_ready_version") or memory_config.get("latest_semantic_ready_version")
    latest_graph = long_term.get("latest_graph_ready_version") or memory_config.get("latest_graph_ready_version")
    semantic_lagging = bool(long_term.get("semantic_lagging") or (latest_fast and latest_semantic and _as_int(latest_semantic) < _as_int(latest_fast)))
    graph_lagging = bool(long_term.get("graph_lagging") or (latest_fast and latest_graph and _as_int(latest_graph) < _as_int(latest_fast)))
    asr_lagging = bool(asr_pending or (latest_uploaded >= 0 and latest_asr < latest_uploaded))
    memory_lagging = bool(memory_pending or (latest_fast and latest_semantic and _as_int(latest_semantic) < _as_int(latest_fast)))
    policy = _question_policy(
        question,
        stream_status=str(stream_state.get("status") or "not_started"),
        current_ready=current_ready,
        short_term_ready=short_term_ready,
        long_term_partial_ready=long_term_partial_ready,
        asr_lagging=asr_lagging,
        semantic_lagging=semantic_lagging,
        graph_lagging=graph_lagging,
    )
    return {
        "is_stream_session": True,
        "session_id": session_id,
        "long_term_session_id": long_term_session_id,
        "parent_session_id": query_context.get("parent_session_id"),
        "is_rokid_day_child": bool(query_context.get("is_rokid_day_child")),
        "long_term_selection": long_term_selection,
        "stream_status": stream_state.get("status", "not_started"),
        "latest_uploaded_chunk_index": latest_uploaded,
        "latest_processed_chunk_index": latest_processed,
        "latest_asr_chunk_index": latest_asr,
        "current_ready": current_ready,
        "current_text_ready": current_text_ready,
        "audio_current_ready": bool(current_state.get("audio_current_ready") or transcript_segment_count > 0),
        "short_term_ready": short_term_ready,
        "frame_stream_ready": frame_stream_ready,
        "frame_open_event_ready": frame_open_event_ready,
        "frame_open_event_start_time": frame_open_event.get("start_time") if frame_open_event else None,
        "frame_open_event_duration": (
            round(float(frame_open_event.get("last_update_time", 0.0) or 0.0) - float(frame_open_event.get("start_time", 0.0) or 0.0), 3)
            if frame_open_event
            else None
        ),
        "long_term_partial_ready": long_term_partial_ready,
        "long_term_full_ready": long_term_full_ready,
        "asr_ready": bool(latest_asr >= 0 or latest_asr_window or transcript_segment_count > 0),
        "latest_asr_window_id": latest_asr_window,
        "transcript_segment_count": transcript_segment_count,
        "asr_lagging": asr_lagging,
        "asr_pending": asr_pending,
        "semantic_lagging": semantic_lagging,
        "graph_lagging": graph_lagging,
        "memory_lagging": memory_lagging,
        "latest_fast_ready_version": _as_int(latest_fast, 0) if latest_fast is not None else None,
        "latest_semantic_ready_version": _as_int(latest_semantic, 0) if latest_semantic is not None else None,
        "latest_graph_ready_version": _as_int(latest_graph, 0) if latest_graph is not None else None,
        "active_query_memory_version": _active_query_version(query_runtime, long_term_session_id),
        "can_answer_current": current_ready,
        "can_answer_recent": bool(current_ready or short_term_ready or frame_open_event_ready or transcript_segment_count > 0),
        "can_answer_summary": bool(long_term_partial_ready or short_term_ready or current_ready),
        "recommended_answer_policy": policy,
        "processing_chunk_count": len(processing_chunks),
    }


def _active_query_version(query_runtime: Any, session_id: str) -> int | None:
    if not isinstance(query_runtime, dict):
        return None
    loaded = query_runtime.get("loaded_sessions") or ((query_runtime.get("cache") or {}).get("loaded_sessions") if isinstance(query_runtime.get("cache"), dict) else [])
    if not isinstance(loaded, list):
        return None
    for item in loaded:
        if isinstance(item, dict) and str(item.get("session_id") or "") == str(session_id):
            value = item.get("active_query_memory_version")
            return _as_int(value, 0) if value is not None else None
    return None
