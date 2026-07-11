from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from .io_utils import ensure_dir, read_json, utc_now_iso, write_json_atomic
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


def _active_session_helpers():
    from online_pipeline.active_session import abort_task_file, task_belongs_to_inactive_session

    return abort_task_file, task_belongs_to_inactive_session


CURRENT_QUERY_KEYWORDS = [
    "现在",
    "当前",
    "此刻",
    "正在",
    "目前",
    "眼下",
    "实时",
    "此时",
    "当下",
    "现在画面",
    "画面里现在",
    "当前画面",
    "正在发生",
    "正在做",
    "现在在做",
    "这是什么",
    "这个是什么",
    "这是啥",
    "这是什么东西",
    "看到什么",
    "看到了什么",
    "能看到什么",
    "当前场景",
    "now",
    "currently",
    "at this moment",
    "right now",
    "at the moment",
    "at present",
    "current frame",
    "in the current frame",
    "current scene",
    "current view",
    "current screen",
    "current image",
    "in the current scene",
    "in the current view",
    "in front of me",
    "live",
    "realtime",
    "real-time",
    "what is happening now",
    "what's happening now",
    "what is going on now",
    "what's going on now",
    "what do you see",
    "what can you see",
    "what is in the current scene",
    "what is in the current view",
    "what is in front of me",
    "what am i seeing",
    "what am i looking at",
    "what is on screen",
    "what's on screen",
    "what is on the screen",
    "what's on the screen",
    "what is visible now",
    "what can you see now",
    "describe current scene",
    "describe the current scene",
    "read the current screen",
    "what is this?",
    "what's this?",
    "what is that?",
    "what's that?",
]
RECENT_QUERY_KEYWORDS = [
    "刚才",
    "刚刚",
    "上一段",
    "最近",
    "just now",
    "recently",
    "a moment ago",
    "a few seconds ago",
    "moments ago",
    "a minute ago",
    "a little while ago",
    "earlier just now",
    "shortly before",
    "right before",
    "previous moment",
    "previously",
    "last segment",
    "last scene",
    "recent scene",
    "what just happened",
]
COUNT_QUERY_KEYWORDS = [
    "一共", "总共", "总计", "总共有", "一共有", "多少", "几个", "几次", "几幅", "几张", "数量",
    "count", "how many", "total", "in total", "altogether",
]
SPAN_QUERY_KEYWORDS = [
    "从刚才到现在", "从刚刚到现在", "刚才到现在", "刚刚到现在", "到现在", "到目前为止",
    "since just now", "from just now", "so far", "up to now",
]

def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = (text or "").lower()
    for keyword in keywords:
        needle = str(keyword or "").strip().lower()
        if not needle:
            continue
        if needle.isascii():
            prefix = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
            suffix = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
            if re.search(prefix + re.escape(needle) + suffix, lowered):
                return True
        elif needle in lowered:
            return True
    return False


def _query_priority(
    *,
    question: str,
    retrieval_mode: str = "auto",
    memory_mode: str = "auto",
    use_current: bool | None = None,
    use_short_term: bool | None = None,
    use_long_term: bool | None = None,
) -> tuple[int, str]:
    retrieval = str(retrieval_mode or "auto").strip().lower()
    memory = str(memory_mode or "auto").strip().lower()
    has_count = _contains_any(question, COUNT_QUERY_KEYWORDS)
    has_recent = _contains_any(question, RECENT_QUERY_KEYWORDS)
    has_current = _contains_any(question, CURRENT_QUERY_KEYWORDS)
    has_span = _contains_any(question, SPAN_QUERY_KEYWORDS) or (has_recent and has_current)
    if has_count and (has_span or has_recent):
        return 1, "temporal_count_question"
    if retrieval == "current" or memory == "current":
        return 0, "explicit_current_mode"
    if use_current is True and use_short_term is False and use_long_term is False:
        return 0, "explicit_current_only"
    if has_current:
        return 0, "current_question_keyword"
    if use_current is True:
        return 1, "use_current_requested"
    if has_recent:
        return 1, "recent_question_keyword"
    return 5, "normal"


TASKS_ROOT_NAME = "online_tasks"
PREPROCESS_QUEUE_NAME = "preprocess"
IN_PROGRESS_QUEUE_NAME = "in_progress"
DONE_QUEUE_NAME = "done"
FAILED_QUEUE_NAME = "failed"
EVIDENCE_QUEUE_NAME = "evidence"
EVIDENCE_IN_PROGRESS_QUEUE_NAME = "evidence_in_progress"
EVIDENCE_DONE_QUEUE_NAME = "evidence_done"
EVIDENCE_FAILED_QUEUE_NAME = "evidence_failed"
MEMORY_QUEUE_NAME = "memory"
MEMORY_IN_PROGRESS_QUEUE_NAME = "memory_in_progress"
MEMORY_DONE_QUEUE_NAME = "memory_done"
MEMORY_FAILED_QUEUE_NAME = "memory_failed"
QUERY_QUEUE_NAME = "query"
QUERY_IN_PROGRESS_QUEUE_NAME = "query_in_progress"
QUERY_DONE_QUEUE_NAME = "query_done"
QUERY_FAILED_QUEUE_NAME = "query_failed"
VISUAL_QUEUE_NAME = "visual"
VISUAL_IN_PROGRESS_QUEUE_NAME = "visual_in_progress"
VISUAL_DONE_QUEUE_NAME = "visual_done"
VISUAL_FAILED_QUEUE_NAME = "visual_failed"
MST_REFINE_QUEUE_NAME = "mst_refine"
MST_REFINE_IN_PROGRESS_QUEUE_NAME = "mst_refine_in_progress"
MST_REFINE_DONE_QUEUE_NAME = "mst_refine_done"
MST_REFINE_FAILED_QUEUE_NAME = "mst_refine_failed"
MST_CONSOLIDATION_QUEUE_NAME = "mst_consolidation"
MST_CONSOLIDATION_IN_PROGRESS_QUEUE_NAME = "mst_consolidation_in_progress"
MST_CONSOLIDATION_DONE_QUEUE_NAME = "mst_consolidation_done"
MST_CONSOLIDATION_FAILED_QUEUE_NAME = "mst_consolidation_failed"
STREAM_CHUNK_QUEUE_NAME = "stream_chunk"
STREAM_CHUNK_IN_PROGRESS_QUEUE_NAME = "stream_chunk_in_progress"
STREAM_CHUNK_DONE_QUEUE_NAME = "stream_chunk_done"
STREAM_CHUNK_FAILED_QUEUE_NAME = "stream_chunk_failed"
STREAM_ASR_QUEUE_NAME = "stream_asr"
STREAM_ASR_IN_PROGRESS_QUEUE_NAME = "stream_asr_in_progress"
STREAM_ASR_DONE_QUEUE_NAME = "stream_asr_done"
STREAM_ASR_FAILED_QUEUE_NAME = "stream_asr_failed"
LIVE_INGEST_QUEUE_NAME = "live_ingest"
LIVE_INGEST_IN_PROGRESS_QUEUE_NAME = "live_ingest_in_progress"
LIVE_INGEST_DONE_QUEUE_NAME = "live_ingest_done"
LIVE_INGEST_FAILED_QUEUE_NAME = "live_ingest_failed"
ROKID_DAY_MERGE_QUEUE_NAME = "rokid_day_merge"
ROKID_DAY_MERGE_IN_PROGRESS_QUEUE_NAME = "rokid_day_merge_in_progress"
ROKID_DAY_MERGE_DONE_QUEUE_NAME = "rokid_day_merge_done"
ROKID_DAY_MERGE_FAILED_QUEUE_NAME = "rokid_day_merge_failed"


def get_queue_dirs(project_root: Path) -> dict[str, Path]:
    root = project_root / TASKS_ROOT_NAME
    return {
        "queued": root / PREPROCESS_QUEUE_NAME,
        "in_progress": root / IN_PROGRESS_QUEUE_NAME,
        "done": root / DONE_QUEUE_NAME,
        "failed": root / FAILED_QUEUE_NAME,
        "evidence_queued": root / EVIDENCE_QUEUE_NAME,
        "evidence_in_progress": root / EVIDENCE_IN_PROGRESS_QUEUE_NAME,
        "evidence_done": root / EVIDENCE_DONE_QUEUE_NAME,
        "evidence_failed": root / EVIDENCE_FAILED_QUEUE_NAME,
        "memory_queued": root / MEMORY_QUEUE_NAME,
        "memory_in_progress": root / MEMORY_IN_PROGRESS_QUEUE_NAME,
        "memory_done": root / MEMORY_DONE_QUEUE_NAME,
        "memory_failed": root / MEMORY_FAILED_QUEUE_NAME,
        "query_queued": root / QUERY_QUEUE_NAME,
        "query_in_progress": root / QUERY_IN_PROGRESS_QUEUE_NAME,
        "query_done": root / QUERY_DONE_QUEUE_NAME,
        "query_failed": root / QUERY_FAILED_QUEUE_NAME,
        "visual_queued": root / VISUAL_QUEUE_NAME,
        "visual_in_progress": root / VISUAL_IN_PROGRESS_QUEUE_NAME,
        "visual_done": root / VISUAL_DONE_QUEUE_NAME,
        "visual_failed": root / VISUAL_FAILED_QUEUE_NAME,
        "mst_refine_queued": root / MST_REFINE_QUEUE_NAME,
        "mst_refine_in_progress": root / MST_REFINE_IN_PROGRESS_QUEUE_NAME,
        "mst_refine_done": root / MST_REFINE_DONE_QUEUE_NAME,
        "mst_refine_failed": root / MST_REFINE_FAILED_QUEUE_NAME,
        "mst_consolidation_queued": root / MST_CONSOLIDATION_QUEUE_NAME,
        "mst_consolidation_in_progress": root / MST_CONSOLIDATION_IN_PROGRESS_QUEUE_NAME,
        "mst_consolidation_done": root / MST_CONSOLIDATION_DONE_QUEUE_NAME,
        "mst_consolidation_failed": root / MST_CONSOLIDATION_FAILED_QUEUE_NAME,
        "stream_chunk_queued": root / STREAM_CHUNK_QUEUE_NAME,
        "stream_chunk_in_progress": root / STREAM_CHUNK_IN_PROGRESS_QUEUE_NAME,
        "stream_chunk_done": root / STREAM_CHUNK_DONE_QUEUE_NAME,
        "stream_chunk_failed": root / STREAM_CHUNK_FAILED_QUEUE_NAME,
        "stream_asr_queued": root / STREAM_ASR_QUEUE_NAME,
        "stream_asr_in_progress": root / STREAM_ASR_IN_PROGRESS_QUEUE_NAME,
        "stream_asr_done": root / STREAM_ASR_DONE_QUEUE_NAME,
        "stream_asr_failed": root / STREAM_ASR_FAILED_QUEUE_NAME,
        "live_ingest_queued": root / LIVE_INGEST_QUEUE_NAME,
        "live_ingest_in_progress": root / LIVE_INGEST_IN_PROGRESS_QUEUE_NAME,
        "live_ingest_done": root / LIVE_INGEST_DONE_QUEUE_NAME,
        "live_ingest_failed": root / LIVE_INGEST_FAILED_QUEUE_NAME,
        "rokid_day_merge_queued": root / ROKID_DAY_MERGE_QUEUE_NAME,
        "rokid_day_merge_in_progress": root / ROKID_DAY_MERGE_IN_PROGRESS_QUEUE_NAME,
        "rokid_day_merge_done": root / ROKID_DAY_MERGE_DONE_QUEUE_NAME,
        "rokid_day_merge_failed": root / ROKID_DAY_MERGE_FAILED_QUEUE_NAME,
    }


def ensure_queue_dirs(project_root: Path) -> dict[str, Path]:
    dirs = get_queue_dirs(project_root)
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def enqueue_preprocess_task(
    project_root: Path,
    session_id: str,
    force: bool = False,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs["queued"] / f"{task_id}.json"
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": "preprocess",
            "session_id": session_id,
            "force": force,
            "status": "queued",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    return task_path


def enqueue_evidence_task(
    project_root: Path,
    session_id: str,
    force: bool = False,
    backend: str | None = None,
    limit_segments: int | None = None,
    pipeline_mode: str | None = None,
    role: str | None = None,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs["evidence_queued"] / f"{task_id}.json"
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": "evidence",
            "session_id": session_id,
            "force": force,
            "backend": backend,
            "limit_segments": limit_segments,
            "pipeline_mode": pipeline_mode,
            "role": role,
            "status": "queued",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    return task_path


def enqueue_memory_task(
    project_root: Path,
    session_id: str,
    force: bool = False,
    skip_visual_embedding: bool = True,
    skip_semantic: bool = False,
    limit_segments: int | None = None,
    source: str = "auto",
    update_mode: str | None = None,
    append_ready_episodes: bool | None = None,
    episode_ids: list[str] | None = None,
    reason: str | None = None,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task_type = "memory_append" if str(update_mode or "").strip().lower() in {"incremental", "incremental_append", "append"} else "memory"
    episode_ids_key = None
    if isinstance(episode_ids, list) and episode_ids:
        normalized_episode_ids = sorted({str(item) for item in episode_ids if str(item or "").strip()})
        episode_ids_key = ",".join(normalized_episode_ids)
        episode_ids = normalized_episode_ids
    dedupe_key = None
    if task_type == "memory_append":
        dedupe_key = episode_ids_key or ("append_ready_all" if append_ready_episodes else None)
        if dedupe_key:
            existing = _find_existing_task(
                project_root,
                keys=("memory_queued", "memory_in_progress"),
                session_id=session_id,
                task_type=task_type,
                match_fields={
                    "source": source,
                    "update_mode": update_mode,
                    "dedupe_key": dedupe_key,
                },
            )
            if existing is not None:
                return existing
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs["memory_queued"] / f"{task_id}.json"
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": task_type,
            "session_id": session_id,
            "force": force,
            "skip_visual_embedding": skip_visual_embedding,
            "skip_semantic": skip_semantic,
            "limit_segments": limit_segments,
            "source": source,
            "update_mode": update_mode,
            "append_ready_episodes": append_ready_episodes,
            "episode_ids": episode_ids,
            "episode_ids_key": episode_ids_key,
            "dedupe_key": dedupe_key,
            "reason": reason,
            "status": "queued",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    return task_path


def enqueue_query_task(
    project_root: Path,
    session_id: str,
    question: str,
    top_k: int = 5,
    use_image_evidence: bool | str = "auto",
    max_image_frames: int = 4,
    retrieval_mode: str = "auto",
    max_image_evidence: int | None = 3,
    text_top_k: int | None = None,
    visual_top_k: int | None = None,
    final_evidence_k: int | None = None,
    memory_mode: str = "auto",
    use_interaction_cache: bool = True,
    cache_mode: str = "auto",
    use_current: bool | None = None,
    use_short_term: bool | None = None,
    use_long_term: bool | None = None,
    debug_router: bool = False,
    long_term_retrieval_scheme: str | None = None,
    retrieval_scheme: str | None = None,
    client_source: str = "unknown",
    input_method: str = "unknown",
    allow_inactive_session: bool = False,
    task_source: str = "api",
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs["query_queued"] / f"{task_id}.json"
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        long_term_retrieval_scheme or retrieval_scheme
    )
    retrieval_scheme = long_term_retrieval_scheme
    priority, priority_reason = _query_priority(
        question=question,
        retrieval_mode=retrieval_mode,
        memory_mode=memory_mode,
        use_current=use_current,
        use_short_term=use_short_term,
        use_long_term=use_long_term,
    )
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": "query",
            "task_source": task_source,
            "session_id": session_id,
            "question": question,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "use_image_evidence": use_image_evidence,
            "max_image_frames": max_image_frames,
            "max_image_evidence": max_image_evidence if max_image_evidence is not None else 3,
            "text_top_k": text_top_k,
            "visual_top_k": visual_top_k,
            "final_evidence_k": final_evidence_k,
            "memory_mode": memory_mode,
            "use_interaction_cache": use_interaction_cache,
            "cache_mode": cache_mode,
            "use_current": use_current,
            "use_short_term": use_short_term,
            "use_long_term": use_long_term,
            "debug_router": debug_router,
            "long_term_retrieval_scheme": long_term_retrieval_scheme,
            "retrieval_scheme": retrieval_scheme,
            "client_source": client_source,
            "input_method": input_method,
            "allow_inactive_session": bool(allow_inactive_session),
            "priority": priority,
            "query_priority_reason": priority_reason,
            "status": "queued",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    return task_path


def enqueue_query_warmup_task(
    project_root: Path,
    session_id: str,
    *,
    reason: str = "memory_ready",
    wait_for_memory: bool = False,
    force: bool = False,
    long_term_retrieval_scheme: str | None = None,
    retrieval_scheme: str | None = None,
) -> Path:
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        long_term_retrieval_scheme or retrieval_scheme
    )
    retrieval_scheme = long_term_retrieval_scheme
    if not force:
        existing = _find_existing_task(
            project_root,
            keys=("query_queued", "query_in_progress", "query_done"),
            session_id=session_id,
            task_type="query_warmup",
            match_fields={
                "reason": reason,
                "long_term_retrieval_scheme": long_term_retrieval_scheme,
            },
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "query_queued",
        "query_warmup",
        session_id,
        reason=reason,
        wait_for_memory=bool(wait_for_memory),
        long_term_retrieval_scheme=long_term_retrieval_scheme,
        retrieval_scheme=retrieval_scheme,
        priority=9,
        query_priority_reason="warmup",
    )


def _enqueue_task(project_root: Path, queue_key: str, task_type: str, session_id: str, **fields: object) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs[queue_key] / f"{task_id}.json"
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "session_id": session_id,
        "status": "queued",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    payload.update(fields)
    write_json_atomic(task_path, payload)
    return task_path


def _normalize_event_ids(event_id: str | None = None, event_ids: list[str] | None = None) -> list[str]:
    merged: list[str] = []
    if event_id:
        merged.append(str(event_id))
    for value in event_ids or []:
        text = str(value or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def _merge_mst_refine_batch_task(
    project_root: Path,
    *,
    session_id: str,
    backend: str | None,
    limit_events: int,
    force_refine: bool,
    reason: str | None,
    event_ids: list[str],
) -> Path | None:
    if not reason or not event_ids:
        return None
    dirs = ensure_queue_dirs(project_root)
    for path in sorted(dirs["mst_refine_queued"].glob(f"{session_id}_*.json")):
        payload = read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("session_id") or "") != str(session_id):
            continue
        if str(payload.get("task_type") or "") != "mst_refine":
            continue
        if str(payload.get("reason") or "") != str(reason):
            continue
        if str(payload.get("event_id") or "").strip():
            continue
        existing_force_refine = bool(payload.get("force_refine", False))
        if existing_force_refine != bool(force_refine):
            continue
        existing_event_ids = _normalize_event_ids(event_ids=payload.get("event_ids") if isinstance(payload.get("event_ids"), list) else None)
        merged_event_ids = _normalize_event_ids(event_ids=existing_event_ids + event_ids)
        if merged_event_ids == existing_event_ids:
            return path
        payload["event_ids"] = merged_event_ids
        payload["limit_events"] = max(int(payload.get("limit_events") or 0), int(limit_events or 0), len(merged_event_ids))
        payload["updated_at"] = utc_now_iso()
        if backend and not payload.get("backend"):
            payload["backend"] = backend
        write_json_atomic(path, payload)
        return path
    return None


def enqueue_visual_task(
    project_root: Path,
    session_id: str,
    force: bool = False,
    backend: str | None = None,
    limit_items: int | None = None,
    task_type: str = "visual",
    episode_ids: list[str] | None = None,
    keyframe_paths: list[str] | None = None,
    target_visual_version: int | None = None,
) -> Path:
    return _enqueue_task(
        project_root,
        "visual_queued",
        task_type,
        session_id,
        force=force,
        backend=backend,
        limit_items=limit_items,
        episode_ids=episode_ids,
        keyframe_paths=keyframe_paths,
        target_visual_version=target_visual_version,
    )


def enqueue_mst_refine_task(
    project_root: Path,
    session_id: str,
    backend: str | None = None,
    limit_events: int = 10,
    event_id: str | None = None,
    event_ids: list[str] | None = None,
    force_refine: bool = False,
    reason: str | None = None,
) -> Path:
    normalized_event_ids = _normalize_event_ids(event_id=event_id, event_ids=event_ids)
    if normalized_event_ids and not event_id:
        merged = _merge_mst_refine_batch_task(
            project_root,
            session_id=session_id,
            backend=backend,
            limit_events=max(int(limit_events or 0), len(normalized_event_ids)),
            force_refine=force_refine,
            reason=reason,
            event_ids=normalized_event_ids,
        )
        if merged is not None:
            return merged
    if event_id and reason:
        existing = _find_existing_task(
            project_root,
            keys=("mst_refine_queued", "mst_refine_in_progress"),
            session_id=session_id,
            task_type="mst_refine",
            match_fields={"event_id": event_id, "reason": reason},
        )
        if existing is not None:
            return existing
    elif reason and not force_refine:
        existing = _find_existing_task(
            project_root,
            keys=("mst_refine_queued",),
            session_id=session_id,
            task_type="mst_refine",
            match_fields={"event_id": None, "reason": reason},
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "mst_refine_queued",
        "mst_refine",
        session_id,
        backend=backend,
        limit_events=max(int(limit_events or 0), len(normalized_event_ids)),
        event_id=event_id,
        event_ids=normalized_event_ids or None,
        force_refine=force_refine,
        reason=reason,
    )


def enqueue_mst_consolidation_task(
    project_root: Path,
    session_id: str,
    backend: str | None = None,
    update_worldmm: bool = True,
    force: bool = False,
    limit_windows: int | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
    reason: str | None = None,
) -> Path:
    if reason and window_start is not None and window_end is not None:
        existing = _find_existing_task(
            project_root,
            keys=("mst_consolidation_queued", "mst_consolidation_in_progress"),
            session_id=session_id,
            task_type="mst_consolidation",
            match_fields={"reason": reason, "window_start": float(window_start), "window_end": float(window_end)},
        )
        if existing is not None:
            return existing
    elif reason and not force:
        existing = _find_existing_task(
            project_root,
            keys=("mst_consolidation_queued",),
            session_id=session_id,
            task_type="mst_consolidation",
            match_fields={"reason": reason, "window_start": None, "window_end": None},
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "mst_consolidation_queued",
        "mst_consolidation",
        session_id,
        backend=backend,
        update_worldmm=update_worldmm,
        force=force,
        limit_windows=limit_windows,
        window_start=window_start,
        window_end=window_end,
        reason=reason,
    )


def _field_equal(left: object, right: object) -> bool:
    if isinstance(right, float):
        try:
            return abs(float(left) - right) < 1e-6
        except Exception:
            return False
    return left == right


def _find_existing_task(
    project_root: Path,
    *,
    keys: tuple[str, ...],
    session_id: str,
    task_type: str | None = None,
    match_fields: dict[str, object] | None = None,
) -> Path | None:
    dirs = ensure_queue_dirs(project_root)
    for key in keys:
        for path in sorted(dirs[key].glob(f"{session_id}_*.json")):
            payload = read_json(path, default={})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("session_id") or "") != str(session_id):
                continue
            if task_type is not None and str(payload.get("task_type") or "") != str(task_type):
                continue
            if match_fields:
                if any(not _field_equal(payload.get(field), value) for field, value in match_fields.items()):
                    continue
            return path
    return None


def _find_existing_stream_task(
    project_root: Path,
    session_id: str,
    chunk_index: int | None = None,
    proc_index: int | None = None,
    task_type: str | None = None,
) -> Path | None:
    dirs = ensure_queue_dirs(project_root)
    keys = (
        "stream_chunk_queued",
        "stream_chunk_in_progress",
        "stream_chunk_done",
        "stream_chunk_failed",
    )
    for key in keys:
        for path in sorted(dirs[key].glob(f"{session_id}_*.json")):
            payload = read_json(path, default={})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("session_id") or "") != str(session_id):
                continue
            if task_type is not None and str(payload.get("task_type") or "") != str(task_type):
                continue
            if chunk_index is not None:
                try:
                    if int(payload.get("chunk_index")) != int(chunk_index):
                        continue
                except Exception:
                    continue
            if proc_index is not None:
                try:
                    if int(payload.get("proc_index", payload.get("chunk_index"))) != int(proc_index):
                        continue
                except Exception:
                    continue
            return path
    return None


def enqueue_stream_chunk_task(
    project_root: Path,
    session_id: str,
    stream_id: str,
    chunk_id: str,
    chunk_index: int,
    chunk_path: str,
    start_time: float,
    end_time: float,
    duration: float,
    checksum: str | None = None,
    proc_index: int | None = None,
    upload_chunk_index: int | None = None,
    source_upload_chunk_id: str | None = None,
    force: bool = False,
) -> Path:
    proc_index = int(proc_index if proc_index is not None else chunk_index)
    if not force:
        existing = _find_existing_stream_task(
            project_root,
            session_id=session_id,
            proc_index=proc_index,
            task_type="stream_chunk",
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "stream_chunk_queued",
        "stream_chunk",
        session_id,
        stream_id=stream_id,
        chunk_id=chunk_id,
        chunk_index=proc_index,
        proc_index=proc_index,
        upload_chunk_index=upload_chunk_index,
        source_upload_chunk_id=source_upload_chunk_id,
        chunk_path=chunk_path,
        start_time=round(float(start_time), 3),
        end_time=round(float(end_time), 3),
        duration=round(float(duration), 3),
        checksum=checksum,
    )


def enqueue_stream_upload_task(
    project_root: Path,
    session_id: str,
    stream_id: str,
    upload_chunk_id: str,
    upload_chunk_index: int,
    upload_chunk_path: str,
    checksum: str | None = None,
    force: bool = False,
) -> Path:
    if not force:
        existing = _find_existing_stream_task(
            project_root,
            session_id=session_id,
            chunk_index=upload_chunk_index,
            task_type="stream_upload_chunk",
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "stream_chunk_queued",
        "stream_upload_chunk",
        session_id,
        stream_id=stream_id,
        upload_chunk_id=upload_chunk_id,
        upload_chunk_index=int(upload_chunk_index),
        chunk_id=upload_chunk_id,
        chunk_index=int(upload_chunk_index),
        upload_chunk_path=upload_chunk_path,
        chunk_path=upload_chunk_path,
        checksum=checksum,
    )


def _find_existing_stream_asr_task(
    project_root: Path,
    session_id: str,
    upload_chunk_index: int | None = None,
    *,
    window_id: str | None = None,
    include_failed: bool = False,
) -> Path | None:
    keys = ["stream_asr_queued", "stream_asr_in_progress", "stream_asr_done"]
    if include_failed:
        keys.append("stream_asr_failed")
    if window_id:
        return _find_existing_task(
            project_root,
            keys=tuple(keys),
            session_id=session_id,
            task_type="stream_asr",
            match_fields={"window_id": str(window_id)},
        )
    if upload_chunk_index is None:
        return None
    return _find_existing_task(
        project_root,
        keys=tuple(keys),
        session_id=session_id,
        task_type="stream_asr",
        match_fields={"upload_chunk_index": int(upload_chunk_index)},
    )


def enqueue_stream_asr_task(
    project_root: Path,
    session_id: str,
    stream_id: str,
    upload_chunk_id: str,
    upload_chunk_index: int,
    upload_chunk_path: str,
    processing_chunks: list[dict] | None = None,
    global_start_time: float | None = None,
    global_end_time: float | None = None,
    asr_backend: str = "whisperx",
    reason: str = "stream_upload_chunk",
    force: bool = False,
    retry_failed: bool = False,
    source: str | None = None,
    window_id: str | None = None,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    duration_ms: int | None = None,
    audio_chunk_paths: list[str] | None = None,
    output_audio_path: str | None = None,
    asr_window_path: str | None = None,
    input_source: str | None = None,
    is_flush: bool = False,
) -> Path:
    if not force:
        existing = _find_existing_stream_asr_task(
            project_root,
            session_id=session_id,
            upload_chunk_index=upload_chunk_index,
            window_id=window_id,
            include_failed=retry_failed,
        )
        if existing is not None:
            return existing
    chunks = [dict(item) for item in processing_chunks or [] if isinstance(item, dict)]
    start = global_start_time
    end = global_end_time
    if start is None and chunks:
        start = min(float(item.get("start_time", 0.0) or 0.0) for item in chunks)
    if end is None and chunks:
        end = max(float(item.get("end_time", 0.0) or 0.0) for item in chunks)
    previous_failed = _find_existing_stream_asr_task(
        project_root,
        session_id=session_id,
        upload_chunk_index=upload_chunk_index,
        window_id=window_id,
        include_failed=True,
    )
    retry_count = 0
    if previous_failed is not None and previous_failed.parent.name == STREAM_ASR_FAILED_QUEUE_NAME:
        payload = read_json(previous_failed, default={})
        if isinstance(payload, dict):
            retry_count = int(payload.get("retry_count", 0) or 0) + 1
    return _enqueue_task(
        project_root,
        "stream_asr_queued",
        "stream_asr",
        session_id,
        stream_id=stream_id,
        upload_chunk_id=upload_chunk_id,
        upload_chunk_index=int(upload_chunk_index),
        upload_chunk_path=upload_chunk_path,
        processing_chunks=chunks,
        global_start_time=round(float(start or 0.0), 3),
        global_end_time=round(float(end or start or 0.0), 3),
        asr_backend=asr_backend,
        reason=reason,
        source=source,
        window_id=window_id,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        duration_ms=duration_ms,
        audio_chunk_paths=[str(item) for item in audio_chunk_paths or []],
        output_audio_path=output_audio_path,
        asr_window_path=asr_window_path or output_audio_path,
        input_source=input_source,
        is_flush=bool(is_flush),
        retry_count=retry_count,
    )


def enqueue_stream_end_task(
    project_root: Path,
    session_id: str,
    stream_id: str,
    final_chunk_index: int | None = None,
    close_open_event: bool = True,
    force: bool = False,
) -> Path:
    if not force:
        existing = _find_existing_stream_task(
            project_root,
            session_id=session_id,
            chunk_index=None,
            task_type="stream_end",
        )
        if existing is not None:
            return existing
    return _enqueue_task(
        project_root,
        "stream_chunk_queued",
        "stream_end",
        session_id,
        stream_id=stream_id,
        final_chunk_index=final_chunk_index,
        close_open_event=bool(close_open_event),
    )


def list_queued_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_evidence_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["evidence_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_memory_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["memory_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_query_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    def _sort_key(path: Path) -> tuple[int, float]:
        priority = 5
        payload = read_json(path, default={})
        if isinstance(payload, dict):
            try:
                priority = int(payload.get("priority"))
            except Exception:
                priority, _ = _query_priority(
                    question=str(payload.get("question") or ""),
                    retrieval_mode=str(payload.get("retrieval_mode") or "auto"),
                    memory_mode=str(payload.get("memory_mode") or "auto"),
                    use_current=_coerce_optional_bool(payload.get("use_current")),
                    use_short_term=_coerce_optional_bool(payload.get("use_short_term")),
                    use_long_term=_coerce_optional_bool(payload.get("use_long_term")),
                )
        return priority, path.stat().st_mtime

    paths = sorted(dirs["query_queued"].glob("*.json"), key=_sort_key)
    try:
        from online_pipeline.active_session import read_active_session_id, single_active_session_enabled

        if single_active_session_enabled():
            active_session_id = read_active_session_id(project_root)
            if active_session_id:
                active_paths = []
                other_paths = []
                for path in paths:
                    payload = read_json(path, default={})
                    if isinstance(payload, dict) and str(payload.get("session_id") or "") == active_session_id:
                        active_paths.append(path)
                    else:
                        other_paths.append(path)
                return active_paths + other_paths
    except Exception:
        pass
    return paths


def list_queued_visual_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["visual_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_mst_refine_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["mst_refine_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_mst_consolidation_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["mst_consolidation_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_stream_chunk_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["stream_chunk_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_stream_asr_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["stream_asr_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def enqueue_live_ingest_task(
    project_root: Path,
    session_id: str,
    stream_id: str,
    source_url: str,
    *,
    source: str = "srs_rtmp",
    input_mode: str = "live_pusher_rtmp",
    reason: str = "live_ingest_start",
    force: bool = False,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    for key in ("live_ingest_queued", "live_ingest_in_progress"):
        for existing in dirs[key].glob(f"{session_id}_*.json"):
            payload = read_json(existing, default={})
            if isinstance(payload, dict) and str(payload.get("session_id") or "") == str(session_id):
                if not force:
                    return existing
    task_id = f"{session_id}_{uuid4().hex[:8]}"
    task_path = dirs["live_ingest_queued"] / f"{task_id}.json"
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": "live_ingest",
            "session_id": session_id,
            "stream_id": stream_id,
            "source_url": source_url,
            "source": source,
            "input_mode": input_mode,
            "reason": reason,
            "status": "queued",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    return task_path


def enqueue_rokid_day_merge_task(
    project_root: Path,
    *,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    day_index: int,
    run_id: str,
    reason: str = "stream_end",
    retry_count: int = 0,
    force: bool = False,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    dedupe_key = f"{parent_session_id}:{child_session_id}:{run_id}"
    if not force:
        existing = _find_existing_task(
            project_root,
            keys=("rokid_day_merge_queued", "rokid_day_merge_in_progress", "rokid_day_merge_done"),
            session_id=child_session_id,
            task_type="rokid_day_merge",
            match_fields={"dedupe_key": dedupe_key},
        )
        if existing is not None:
            return existing
    task_id = f"{child_session_id}_{uuid4().hex[:8]}"
    task_path = dirs["rokid_day_merge_queued"] / f"{task_id}.json"
    now = utc_now_iso()
    write_json_atomic(
        task_path,
        {
            "task_id": task_id,
            "task_type": "rokid_day_merge",
            "session_id": child_session_id,
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "day_label": day_label,
            "day_index": int(day_index),
            "run_id": run_id,
            "reason": reason,
            "dedupe_key": dedupe_key,
            "retry_count": int(retry_count or 0),
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        },
    )
    return task_path


def list_queued_rokid_day_merge_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["rokid_day_merge_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def list_queued_live_ingest_tasks(project_root: Path) -> list[Path]:
    dirs = ensure_queue_dirs(project_root)
    return sorted(dirs["live_ingest_queued"].glob("*.json"), key=lambda p: p.stat().st_mtime)


def _claim_task_to(project_root: Path, task_path: Path, in_progress_key: str) -> tuple[Path, dict] | None:
    dirs = ensure_queue_dirs(project_root)
    task = read_json(task_path, default=None)
    if not isinstance(task, dict):
        return None
    if str(task.get("task_type") or "") != "rokid_day_merge":
        abort_task_file, task_belongs_to_inactive_session = _active_session_helpers()
        if task_belongs_to_inactive_session(project_root, task):
            abort_task_file(project_root, task_path, task=task, reason="inactive_session_claim")
            return None
    claimed_at = utc_now_iso()
    task["status"] = "in_progress"
    task["claimed_at"] = claimed_at
    task["updated_at"] = claimed_at
    claimed_path = dirs[in_progress_key] / task_path.name
    try:
        task_path.replace(claimed_path)
    except FileNotFoundError:
        return None
    write_json_atomic(claimed_path, task)
    return claimed_path, task


def claim_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "in_progress")


def claim_evidence_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "evidence_in_progress")


def claim_memory_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "memory_in_progress")


def claim_query_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "query_in_progress")


def claim_visual_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "visual_in_progress")


def claim_mst_refine_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "mst_refine_in_progress")


def claim_mst_consolidation_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "mst_consolidation_in_progress")


def claim_stream_chunk_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "stream_chunk_in_progress")


def claim_stream_asr_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "stream_asr_in_progress")


def claim_live_ingest_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "live_ingest_in_progress")


def claim_rokid_day_merge_task(project_root: Path, task_path: Path) -> tuple[Path, dict] | None:
    return _claim_task_to(project_root, task_path, "rokid_day_merge_in_progress")


def requeue_rokid_day_merge_task(
    project_root: Path,
    claimed_path: Path,
    task: dict,
    *,
    retry_count: int,
    not_before: str | None = None,
    reason: str = "waiting_for_child_outputs",
    result: dict | None = None,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    task["status"] = "queued"
    task["retry_count"] = int(retry_count)
    task["requeue_reason"] = reason
    task["not_before"] = not_before
    task["updated_at"] = utc_now_iso()
    if result is not None:
        task["last_waiting_result"] = result
    target_path = dirs["rokid_day_merge_queued"] / claimed_path.name
    write_json_atomic(claimed_path, task)
    claimed_path.replace(target_path)
    return target_path


def _finish_task_to(
    project_root: Path,
    claimed_path: Path,
    task: dict,
    status: str,
    done_key: str,
    failed_key: str,
    result: dict | None = None,
    error: str | None = None,
) -> Path:
    dirs = ensure_queue_dirs(project_root)
    if status not in {"done", "failed"}:
        raise ValueError(f"Unsupported task final status: {status}")
    if str(task.get("task_type") or "") != "rokid_day_merge":
        abort_task_file, task_belongs_to_inactive_session = _active_session_helpers()
        if task_belongs_to_inactive_session(project_root, task):
            abort_task_file(project_root, claimed_path, task=task, reason="inactive_session_finish")
            return claimed_path
    task["status"] = status
    task["error"] = error
    if result is not None:
        task["result"] = result
    task["updated_at"] = utc_now_iso()
    target_dir = dirs[done_key] if status == "done" else dirs[failed_key]
    target_path = target_dir / claimed_path.name
    write_json_atomic(claimed_path, task)
    claimed_path.replace(target_path)
    return target_path


def finish_task(project_root: Path, claimed_path: Path, task: dict, status: str, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "done", "failed", error=error)


def finish_evidence_task(project_root: Path, claimed_path: Path, task: dict, status: str, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "evidence_done", "evidence_failed", error=error)


def finish_memory_task(
    project_root: Path,
    claimed_path: Path,
    task: dict,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "memory_done", "memory_failed", result=result, error=error)


def finish_query_task(
    project_root: Path,
    claimed_path: Path,
    task: dict,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "query_done", "query_failed", result=result, error=error)


def finish_visual_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "visual_done", "visual_failed", result=result, error=error)


def finish_mst_refine_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "mst_refine_done", "mst_refine_failed", result=result, error=error)


def finish_mst_consolidation_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "mst_consolidation_done", "mst_consolidation_failed", result=result, error=error)


def finish_stream_chunk_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "stream_chunk_done", "stream_chunk_failed", result=result, error=error)


def finish_stream_asr_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "stream_asr_done", "stream_asr_failed", result=result, error=error)


def finish_live_ingest_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "live_ingest_done", "live_ingest_failed", result=result, error=error)


def finish_rokid_day_merge_task(project_root: Path, claimed_path: Path, task: dict, status: str, result: dict | None = None, error: str | None = None) -> Path:
    return _finish_task_to(project_root, claimed_path, task, status, "rokid_day_merge_done", "rokid_day_merge_failed", result=result, error=error)
