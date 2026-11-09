from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import utc_now_iso, write_json_atomic
from online_preprocess.task_queue import (
    claim_query_task,
    finish_query_task,
    get_queue_dirs,
    list_queued_query_tasks,
)
from online_preprocess.io_utils import read_json
from online_pipeline.runtime_state import write_worker_runtime
from online_query.query_cache import SessionEngineCache
from online_query.query_engine import load_query_engine, query_session
from online_query.stream_transport import QueryStreamEventWriter, query_stream_events_path
from online_qa_history import append_qa_history
from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


PROJECT_ROOT = Path(__file__).resolve().parent
_WORKER_CONTEXT = threading.local()


def _runtime_name() -> str:
    return getattr(_WORKER_CONTEXT, "name", "query")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_optional_bool(value: Any) -> bool | None:
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


def _queue_counts(project_root: Path) -> dict[str, int]:
    dirs = get_queue_dirs(project_root)
    keys = ["query_queued", "query_in_progress", "query_done", "query_failed"]
    return {
        key: len(list(dirs[key].glob("*.json"))) if dirs[key].exists() else 0
        for key in keys
    }


def _heartbeat_interval_seconds() -> float:
    configured = os.getenv("EM2MEM_QUERY_WORKER_HEARTBEAT_SECONDS")
    if configured not in {None, ""}:
        return max(1.0, _env_float("EM2MEM_QUERY_WORKER_HEARTBEAT_SECONDS", 15.0))
    stale_seconds = max(3.0, _env_float("EM2MEM_WORKER_STALE_SECONDS", 60.0))
    return max(1.0, min(15.0, stale_seconds / 3.0))


def _touch_runtime(
    project_root: Path,
    state: str,
    *,
    extra: dict[str, Any] | None = None,
    last_task_id: str | None = None,
    last_error: str | None = None,
) -> None:
    queue_counts = _queue_counts(project_root)
    runtime_name = _runtime_name()
    worker_runtime_path = project_root / "runtime" / "workers" / f"{runtime_name}.json"
    existing_worker = read_json(worker_runtime_path, default={})
    if not isinstance(existing_worker, dict):
        existing_worker = {}
    preserved_extra: dict[str, Any] = {}
    for key in (
        "loaded_sessions",
        "pipeline_mode",
            "strict_load_only",
            "preload_recent_memory_ready",
            "text_embedding_runtime",
            "vlm2vec_runtime",
            "query_preload_results",
            "reload_results",
    ):
        if key in existing_worker:
            preserved_extra[key] = existing_worker[key]
    if extra:
        preserved_extra.update(extra)

    write_worker_runtime(
        project_root,
        runtime_name,
        status="ready" if state in {"idle", "running"} else state,
        model_name=os.getenv("EM2MEM_QUERY_RESPOND_MODEL") or os.getenv("EM2MEM_RESPOND_MODEL") or os.getenv("OPENAI_MODEL"),
        backend="openai-compatible",
        client_loaded=True,
        warmup_done=True,
        queue_pending=queue_counts.get("query_queued", 0),
        last_task_id=last_task_id or existing_worker.get("last_task_id"),
        last_error=last_error if last_error is not None else existing_worker.get("last_error"),
        extra=preserved_extra,
    )

    query_runtime_path = project_root / "online_tasks" / f"{runtime_name}_runtime.json"
    query_runtime = read_json(query_runtime_path, default={})
    if isinstance(query_runtime, dict):
        query_runtime["status"] = state
        query_runtime["updated_at"] = utc_now_iso()
        query_runtime["pid"] = os.getpid()
        query_runtime["queue_counts"] = queue_counts
        if last_task_id:
            query_runtime["last_task_id"] = last_task_id
        if last_error is not None:
            query_runtime["last_error"] = last_error
        if extra:
            query_runtime.update(extra)
        write_json_atomic(query_runtime_path, query_runtime)


class _TaskHeartbeat:
    def __init__(self, project_root: Path, task: dict[str, Any], claimed_path: Path, *, state: str = "running") -> None:
        self.runtime_name = _runtime_name()
        self.project_root = project_root
        self.task = task
        self.claimed_path = claimed_path
        self.state = state
        self.task_id = str(task.get("task_id") or claimed_path.stem)
        self.task_type = str(task.get("task_type") or "query")
        self.session_id = str(task.get("session_id") or "")
        self.started_at = str(task.get("claimed_at") or task.get("updated_at") or utc_now_iso())
        self.started_monotonic = time.monotonic()
        self.interval_seconds = _heartbeat_interval_seconds()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"query-heartbeat-{self.task_id}", daemon=True)

    def __enter__(self) -> "_TaskHeartbeat":
        self._write()
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _runtime_extra(self) -> dict[str, Any]:
        current_task: dict[str, Any] = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "session_id": self.session_id,
            "task_path": str(self.claimed_path),
            "started_at": self.started_at,
            "running_seconds": round(max(0.0, time.monotonic() - self.started_monotonic), 3),
        }
        question = str(self.task.get("question") or "").strip()
        if question:
            current_task["question_preview"] = question[:160]
            current_task["query_priority_reason"] = self.task.get("query_priority_reason")
        return {
            "current_task": current_task,
            "current_task_id": self.task_id,
            "current_session_id": self.session_id,
            "current_task_type": self.task_type,
        }

    def _write(self) -> None:
        _touch_runtime(
            self.project_root,
            self.state,
            extra=self._runtime_extra(),
            last_task_id=self.task_id,
        )

    def _run(self) -> None:
        _WORKER_CONTEXT.name = self.runtime_name
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write()
            except Exception as exc:
                print(f"[query_worker] heartbeat update failed task={self.task_id}: {exc}", flush=True)


def _write_runtime(project_root: Path, cache: SessionEngineCache, state: str, extra: dict[str, Any] | None = None) -> None:
    queue_counts = _queue_counts(project_root)
    payload = {
        "status": state,
        "updated_at": utc_now_iso(),
        "pid": os.getpid(),
        "project_root": str(project_root),
        "pipeline_mode": os.getenv("EM2MEM_PIPELINE_MODE", "mst"),
        "strict_load_only": _env_bool("EM2MEM_QUERY_STRICT_LOAD_ONLY", True),
        "skip_reindex": _env_bool("EM2MEM_QUERY_SKIP_REINDEX", True),
        "use_cached_hipporag": _env_bool("EM2MEM_QUERY_USE_CACHED_HIPPORAG", True),
        "long_term_retrieval_scheme": normalize_long_term_retrieval_scheme(None),
        "preload_recent_memory_ready": int(os.getenv("EM2MEM_PRELOAD_RECENT_MEMORY_READY", "0") or 0),
        "queue_counts": queue_counts,
        "router": {
            "memory_router_enabled": True,
            "default_memory_mode": os.getenv("EM2MEM_DEFAULT_MEMORY_MODE", "auto"),
        },
        "cache": cache.runtime_info(),
    }
    payload["loaded_sessions"] = payload["cache"].get("loaded_sessions", [])
    if extra:
        payload.update(extra)
    runtime_name = _runtime_name()
    write_json_atomic(project_root / "online_tasks" / f"{runtime_name}_runtime.json", payload)
    write_worker_runtime(
        project_root,
        runtime_name,
        status="ready" if state in {"idle", "running"} else state,
        model_name=os.getenv("EM2MEM_QUERY_RESPOND_MODEL") or os.getenv("EM2MEM_RESPOND_MODEL") or os.getenv("OPENAI_MODEL"),
        backend="openai-compatible",
        client_loaded=True,
        warmup_done=True,
        queue_pending=queue_counts.get("query_queued", 0),
        extra={
            "loaded_sessions": payload["loaded_sessions"],
            "pipeline_mode": payload["pipeline_mode"],
            "strict_load_only": payload["strict_load_only"],
            "preload_recent_memory_ready": payload["preload_recent_memory_ready"],
            "text_embedding_runtime": payload.get("text_embedding_runtime"),
            "vlm2vec_runtime": payload.get("vlm2vec_runtime"),
        },
    )


def _preload_vlm2vec() -> dict[str, Any]:
    runtime = get_global_vlm2vec_runtime()
    info = runtime.info()
    if runtime.backend == "vlm2vec":
        _ = runtime.model
    elif runtime.backend == "remote":
        info["remote_health"] = runtime.ping_remote()
    return info


def _preload_text_embedding() -> dict[str, Any] | None:
    backend = os.getenv("EM2MEM_TEXT_EMBED_BACKEND", "local").strip().lower()
    if backend != "remote":
        return {"backend": backend or "local"}
    from em2mem.embedding.remote_text_embedding import RemoteTextEmbeddingModel

    client = RemoteTextEmbeddingModel()
    health = client.ping_remote()
    return {
        "backend": "remote",
        "remote_url": client.remote_url,
        "remote_health": health,
    }


def _parse_session_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _classify_upstream_error(error: BaseException | str) -> tuple[str | None, str]:
    message = str(error)
    lowered = message.lower()
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered or "rate_limited" in lowered:
        return "rate_limited", "Model service is busy or rate-limited. Please retry later."
    if "timeout" in lowered or "timed out" in lowered or "readtimeout" in lowered or "connecttimeout" in lowered:
        return "upstream_timeout", "Model service timed out. Please retry later."
    return None, message


class _TimingOnlyStreamHandler:
    """No-op stream handler that only enables first-token timing.

    The query engine wires the LLM ``on_chunk`` callback (and therefore records
    ``latency.stream_model_first_token_ms``) only when it receives a *callable*
    ``stream_handler``.  Non-SSE (async / polling) clients used to pass ``None``,
    so TTFT was never measured for them.  Passing this callable in async mode
    keeps client-facing behaviour identical (no events are written) while
    enabling the server-side TTFT measurement.
    """

    def __call__(self, event: dict[str, Any]) -> None:
        return None


class _PollingProgressHandler:
    """Publish bounded progress separately from the queue file being claimed."""

    def __init__(self, project_root: Path, task_id: str) -> None:
        self.path = project_root / "online_tasks" / "query_progress" / f"{task_id}.json"
        self.text = ""
        self.stage = "retrieving"
        self.last_write = 0.0
        self({"type": "progress", "stage": self.stage})

    def __call__(self, event: dict[str, Any]) -> None:
        previous = self.stage
        if event.get("type") == "progress":
            self.stage = event.get("stage", self.stage)
            if event.get("reset_answer"):
                self.text = ""
        elif event.get("type") == "delta":
            self.stage = "streaming"
            self.text = (self.text + str(event.get("delta") or event.get("text") or ""))[:16000]
        elif event.get("type") == "final":
            self.stage = "finalizing"
        else:
            return
        now = time.monotonic()
        if self.stage == previous and now - self.last_write < .5:
            return
        self.last_write = now
        write_json_atomic(self.path, {"stage": self.stage, "partial_answer": self.text,
                                      "updated_at": utc_now_iso()})

    def close(self) -> None:
        self.path.unlink(missing_ok=True)


def _attach_answer_audio_for_task(
    *,
    result: dict[str, Any],
    sessions_root: Path,
    session_id: str,
    client_source: str,
    input_method: str,
    task_id: str,
    answer_tts: bool = False,
) -> dict[str, Any]:
    if not answer_tts:
        return result
    try:
        from online_preprocess.tts_xfyun import attach_answer_audio_to_result

        return attach_answer_audio_to_result(
            result=result,
            sessions_root=sessions_root,
            session_id=session_id,
            client_source=client_source,
            input_method=input_method,
            task_id=task_id,
        )
    except Exception as exc:
        result.setdefault(
            "answer_tts",
            {
                "status": "failed",
                "backend": "xfyun_tts",
                "error": str(exc),
                "created_at": utc_now_iso(),
            },
        )
        return result


def _discover_memory_ready_sessions(sessions_root: Path, limit: int) -> list[str]:
    if limit <= 0 or not sessions_root.exists():
        return []
    candidates: list[tuple[float, str]] = []
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        memory_config = session_dir / "em2mem" / "memory_config.json"
        if not memory_config.exists():
            continue
        config = read_json(memory_config, default={})
        if not isinstance(config, dict) or config.get("status") != "memory_ready":
            continue
        latest_ready = config.get("latest_ready_memory_version") or config.get("memory_version")
        if latest_ready is None and config.get("status") == "memory_ready":
            latest_ready = 1
        if not latest_ready:
            continue
        candidates.append((memory_config.stat().st_mtime, session_dir.name))
    candidates.sort(reverse=True)
    return [session_id for _, session_id in candidates[:limit]]


def _preload_query_sessions(
    sessions_root: Path,
    cache: SessionEngineCache,
    session_ids: list[str],
) -> list[dict[str, Any]]:
    results = []
    seen = set()
    for session_id in session_ids:
        if session_id in seen:
            continue
        seen.add(session_id)
        start = time.perf_counter()
        try:
            long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(None)
            _engine, cache_hit, engine_load_ms = cache.get_or_load(
                session_id=session_id,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                loader=lambda sid: load_query_engine(
                    sid,
                    sessions_root=sessions_root,
                    long_term_retrieval_scheme=long_term_retrieval_scheme,
                ),
            )
            total_ms = int(round((time.perf_counter() - start) * 1000))
            result = {
                "session_id": session_id,
                "status": "ok",
                "long_term_retrieval_scheme": long_term_retrieval_scheme,
                "cache_hit": cache_hit,
                "engine_load_ms": engine_load_ms,
                "total_ms": total_ms,
            }
            print(
                f"[query_worker] preloaded query session={session_id} "
                f"engine_load_ms={engine_load_ms} total_ms={total_ms}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "session_id": session_id,
                "status": "failed",
                "preload_status": "skipped_incomplete_snapshot",
                "error": str(exc),
            }
            print(f"[query_worker] preload session={session_id} failed: {exc}", flush=True)
        results.append(result)
    return results


def _process_task(
    project_root: Path,
    sessions_root: Path,
    cache: SessionEngineCache,
    task_path: Path,
) -> bool:
    claimed = claim_query_task(project_root, task_path)
    if claimed is None:
        return False
    claimed_path, task = claimed
    session_id = str(task.get("session_id") or "")
    task_type = str(task.get("task_type") or "query")
    task_id = str(task.get("task_id") or claimed_path.stem)

    if task_type == "query_warmup":
        long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
            task.get("long_term_retrieval_scheme") or task.get("retrieval_scheme")
        )
        print(
            f"[query_worker] warmup task={task_id} session={session_id} "
            f"reason={task.get('reason')} long_term_retrieval_scheme={long_term_retrieval_scheme}",
            flush=True,
        )
        try:
            from online_query.warmup import warm_query_session

            with _TaskHeartbeat(project_root, task, claimed_path):
                result = warm_query_session(
                    session_id=session_id,
                    sessions_root=sessions_root,
                    cache=cache,
                    wait_for_memory=_coerce_bool(task.get("wait_for_memory", False), False),
                    reason=str(task.get("reason") or "query_warmup"),
                    long_term_retrieval_scheme=long_term_retrieval_scheme,
                )
            final_status = "done" if result.get("status") in {"ready", "partial"} else "failed"
            finish_query_task(
                project_root=project_root,
                claimed_path=claimed_path,
                task=task,
                status=final_status,
                result=result,
                error=None if final_status == "done" else str(result.get("error") or result.get("status")),
            )
            print(
                f"[query_worker] warmup task={task_id} finished status={result.get('status')} "
                f"total_ms={result.get('total_ms')}",
                flush=True,
            )
        except Exception as exc:
            finish_query_task(
                project_root=project_root,
                claimed_path=claimed_path,
                task=task,
                status="failed",
                error=str(exc),
            )
            print(f"[query_worker] warmup task={task_id} failed: {exc}", flush=True)
        return True

    question = str(task.get("question") or "").strip()
    top_k = int(task.get("top_k") or 5)
    retrieval_mode = str(task.get("retrieval_mode") or "auto")
    use_image_evidence = task.get("use_image_evidence", "auto")
    max_image_frames = int(task["max_image_frames"] if task.get("max_image_frames") is not None else 3)
    max_image_evidence = int(task["max_image_evidence"] if task.get("max_image_evidence") is not None else 9)
    text_top_k = task.get("text_top_k")
    visual_top_k = task.get("visual_top_k")
    final_evidence_k = task.get("final_evidence_k")
    memory_mode = str(task.get("memory_mode") or "auto")
    use_interaction_cache = _coerce_bool(task.get("use_interaction_cache", True), True)
    use_current = _coerce_optional_bool(task.get("use_current"))
    use_short_term = _coerce_optional_bool(task.get("use_short_term"))
    use_long_term = _coerce_optional_bool(task.get("use_long_term"))
    debug_router = _coerce_bool(task.get("debug_router", False), False)
    cache_mode = str(task.get("cache_mode") or "auto")
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(
        task.get("long_term_retrieval_scheme") or task.get("retrieval_scheme")
    )
    priority = task.get("priority")
    priority_reason = task.get("query_priority_reason")
    client_source = str(task.get("client_source") or "unknown")
    input_method = str(task.get("input_method") or "unknown")
    answer_tts = _coerce_bool(task.get("answer_tts", False), False)
    response_mode = str(task.get("response_mode") or "async").strip().lower()
    is_streaming = response_mode in {"stream", "streaming", "progressive"}
    stream_writer = None
    # Always hand the engine a callable so it records stream_model_first_token_ms.
    # Polling clients receive progress snapshots; SSE clients use the writer.
    engine_stream_handler: Any = _PollingProgressHandler(project_root, task_id)
    polling_progress = engine_stream_handler
    if is_streaming:
        configured_stream_path = str(task.get("stream_events_path") or "").strip()
        if configured_stream_path:
            candidate = (project_root / configured_stream_path).resolve()
            stream_root = (project_root / "online_tasks" / "query_stream_events").resolve()
            try:
                candidate.relative_to(stream_root)
                stream_path = candidate
            except ValueError:
                stream_path = query_stream_events_path(project_root, task_id)
        else:
            stream_path = query_stream_events_path(project_root, task_id)
        stream_writer = QueryStreamEventWriter(stream_path)
        engine_stream_handler = stream_writer

    if not session_id or not question:
        polling_progress.close()
        finish_query_task(
            project_root=project_root,
            claimed_path=claimed_path,
            task=task,
            status="failed",
            error="query task requires session_id and question",
        )
        return True

    print(
        f"[query_worker] task={task_id} session={session_id} top_k={top_k} "
        f"retrieval_mode={retrieval_mode} long_term_retrieval_scheme={long_term_retrieval_scheme} "
        f"priority={priority} reason={priority_reason}",
        flush=True,
    )
    try:
        with _TaskHeartbeat(project_root, task, claimed_path):
            result = query_session(
                session_id=session_id,
                question=question,
                sessions_root=sessions_root,
                top_k=top_k,
                cache=cache,
                retrieval_mode=retrieval_mode,
                use_image_evidence=use_image_evidence,
                max_image_frames=max_image_frames,
                max_image_evidence=max_image_evidence,
                text_top_k=int(text_top_k) if text_top_k else None,
                visual_top_k=int(visual_top_k) if visual_top_k else None,
                final_evidence_k=int(final_evidence_k) if final_evidence_k else None,
                memory_mode=memory_mode,
                use_interaction_cache=use_interaction_cache,
                use_current=use_current,
                use_short_term=use_short_term,
                use_long_term=use_long_term,
                debug_router=debug_router,
                cache_mode=cache_mode,
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                stream_handler=engine_stream_handler,
            )
        try:
            from online_query.stream_query_context import load_stream_query_context

            stream_context = load_stream_query_context(session_id, sessions_root=sessions_root, project_root=project_root, question=question)
            if stream_context and not result.get("stream_context"):
                result["stream_context"] = stream_context
        except Exception:
            pass
        final_status = "done" if result.get("status") == "ok" else "failed"
        error_message = None if final_status == "done" else str(result.get("message") or result.get("status"))
        error_type = None
        if error_message:
            error_type, readable = _classify_upstream_error(error_message)
            if error_type:
                error_message = readable
                result["status"] = "failed"
                result["error_type"] = error_type
                result["message"] = readable
        if error_type:
            task["error_type"] = error_type
        if final_status == "done":
            result["response_mode"] = "stream" if is_streaming else "async"
            result = _attach_answer_audio_for_task(
                result=result,
                sessions_root=sessions_root,
                session_id=session_id,
                client_source=client_source,
                input_method=input_method,
                task_id=task_id,
                answer_tts=answer_tts,
            )
        try:
            append_qa_history(
                sessions_root / session_id,
                session_id=session_id,
                question=question,
                answer=str(result.get("answer") or result.get("answer_text") or ""),
                client_source=client_source,
                input_method=input_method,
                status=final_status,
                error=error_message or "",
                task_id=task_id,
                response_mode="stream" if is_streaming else "async",
                metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme,
                          "generation_status": result.get("generation_status"),
                          "error_type": result.get("error_type"),
                          "latency": {"total_ms": (result.get("latency") or {}).get("total_ms")},
                          "used_memory_sources": result.get("used_memory_sources", []),
                          "evidence_count": result.get("sent_image_count", 0),
                          "question_submitted_at": task.get("created_at"),
                          "original_client_source": client_source},
            )
        except Exception as history_exc:
            print(f"[query_worker] qa_history append failed task={task_id} session={session_id}: {history_exc}", flush=True)
        finish_query_task(
            project_root=project_root,
            claimed_path=claimed_path,
            task=task,
            status=final_status,
            result=result,
            error=error_message,
        )
        print(
            f"[query_worker] task={task_id} finished status={result.get('status')} "
            f"cache_hit={result.get('latency', {}).get('cache_hit')} "
            f"fast_path={result.get('fast_path') or result.get('latency', {}).get('fast_path')} "
            f"generation_ms={result.get('latency', {}).get('generation_ms')} "
            f"image_blocks={result.get('latency', {}).get('image_blocks_count')}",
            flush=True,
        )
    except Exception as exc:
        error_type, readable = _classify_upstream_error(exc)
        if stream_writer is not None:
            stream_writer(
                {
                    "type": "error",
                    "status": "error",
                    "message": readable,
                    "session_id": session_id,
                }
            )
        if error_type:
            task["error_type"] = error_type
        try:
            append_qa_history(
                sessions_root / session_id,
                session_id=session_id,
                question=question,
                client_source=client_source,
                input_method=input_method,
                status="failed",
                error=readable,
                task_id=task_id,
                response_mode="stream" if is_streaming else "async",
                metadata={"long_term_retrieval_scheme": long_term_retrieval_scheme},
            )
        except Exception as history_exc:
            print(f"[query_worker] qa_history append failed task={task_id} session={session_id}: {history_exc}", flush=True)
        finish_query_task(
            project_root=project_root,
            claimed_path=claimed_path,
            task=task,
            status="failed",
            error=readable,
        )
        print(f"[query_worker] task={task_id} failed error_type={error_type}: {exc}", flush=True)
    finally:
        polling_progress.close()
    return True


def _is_current_lane_task(task_path: Path) -> bool:
    """Only independent current-only requests may bypass the history queue."""
    from online_query_constraints import parse_query_constraints
    try:
        task = read_json(task_path, default={})
    except FileNotFoundError:
        # The other lane may have claimed it since the queue was listed.
        return False
    if not isinstance(task, dict) or task.get("task_type", "query") != "query":
        return False
    if any(_coerce_optional_bool(task.get(key)) is True for key in ("use_short_term", "use_long_term")):
        return False
    if _coerce_optional_bool(task.get("use_current")) is False:
        return False
    if str(task.get("retrieval_mode") or "").lower() == "current" or str(task.get("memory_mode") or "").lower() == "current":
        return True
    if _coerce_optional_bool(task.get("use_current")) is True and all(_coerce_optional_bool(task.get(key)) is False for key in ("use_short_term", "use_long_term")):
        return True
    constraints = parse_query_constraints(str(task.get("question") or ""))
    if constraints.get("scope") != "current" or constraints.get("summary") or constraints.get("time_explicit"):
        return False
    from online_query.memory_router import MemoryRouter
    decision = MemoryRouter().route(str(task.get("question") or ""), {}, {"current_ready": True, "current_stale": False, "short_term_ready": True, "long_term_ready": True})
    route = decision["memory_route"]
    return decision["query_type"] == "current_perception" and not route.get("use_short_term") and not route.get("use_long_term")


def _run_current_lane(project_root: Path, sessions_root: Path, stop_event: threading.Event, poll_interval: float) -> None:
    _WORKER_CONTEXT.name = "query_current"
    cache = SessionEngineCache(max_sessions=1)
    try:
        while not stop_event.is_set():
            for task_path in list_queued_query_tasks(project_root):
                if stop_event.is_set():
                    break
                if _is_current_lane_task(task_path):
                    _process_task(project_root, sessions_root, cache, task_path)
            stop_event.wait(max(0.05, min(0.5, poll_interval)))
    except Exception as exc:
        print(f"[query_worker] current lane stopped unexpectedly: {exc}", flush=True)
    finally:
        cache.clear()
        _write_runtime(project_root, cache, "stopped")


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = (project_root / args.sessions_root).resolve() if not Path(args.sessions_root).is_absolute() else Path(args.sessions_root)
    cache = SessionEngineCache(max_sessions=args.cache_max_sessions, ttl_seconds=args.ttl_seconds)
    stop = False
    current_stop = threading.Event()
    current_thread = None

    def _handle_stop(signum: int, frame: object) -> None:
        nonlocal stop
        del signum, frame
        stop = True
        current_stop.set()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    if not args.once and getattr(args, "current_lane", True) and _env_bool("EM2MEM_QUERY_CURRENT_FAST_PATH", True):
        # Start before index/model preloads so even the first history engine
        # initialization cannot hold up independent current-only requests.
        current_thread = threading.Thread(target=_run_current_lane, args=(project_root, sessions_root, current_stop, args.poll_interval), name="query-current-lane", daemon=True)
        current_thread.start()

    print("[query_worker] starting", flush=True)
    print(f"[query_worker] project_root={project_root}", flush=True)
    print(f"[query_worker] sessions_root={sessions_root}", flush=True)
    print(f"[query_worker] cache_max_sessions={args.cache_max_sessions}", flush=True)
    print(f"[query_worker] ttl_seconds={args.ttl_seconds}", flush=True)
    text_embedding_info = _preload_text_embedding()
    if text_embedding_info:
        print(f"[query_worker] text embedding runtime: {text_embedding_info}", flush=True)
    preload_info = None
    if args.preload_vlm2vec:
        print("[query_worker] preloading VLM2Vec runtime", flush=True)
        preload_info = _preload_vlm2vec()
        print(f"[query_worker] VLM2Vec runtime ready: {preload_info}", flush=True)
    preload_session_ids = _parse_session_ids(args.preload_sessions)
    if args.preload_recent_memory_ready > 0:
        preload_session_ids.extend(_discover_memory_ready_sessions(sessions_root, args.preload_recent_memory_ready))
    query_preload_results = []
    if preload_session_ids:
        print(f"[query_worker] preloading query sessions: {preload_session_ids}", flush=True)
        query_preload_results = _preload_query_sessions(sessions_root, cache, preload_session_ids)
    _write_runtime(
        project_root,
        cache,
        "running",
        {
            "text_embedding_runtime": text_embedding_info,
            "vlm2vec_runtime": preload_info,
            "query_preload_results": query_preload_results,
        },
    )

    while not stop:
        processed = False
        for task_path in list_queued_query_tasks(project_root):
            if current_thread is not None and current_thread.is_alive() and _is_current_lane_task(task_path):
                continue
            processed = _process_task(project_root, sessions_root, cache, task_path) or processed
            _write_runtime(project_root, cache, "running", {"last_task_path": str(task_path)})
            if args.once:
                stop = True
                break
        if args.once:
            break
        if not processed:
            # Claim queued work first. A full background engine rebuild used to
            # delay even current-only questions before they could be claimed.
            reload_results = cache.reload_changed(
                lambda sid, scheme: load_query_engine(sid, sessions_root=sessions_root, long_term_retrieval_scheme=scheme)
            )
            if reload_results:
                _write_runtime(project_root, cache, "running", {"reload_results": reload_results[-20:]})
            _write_runtime(project_root, cache, "idle")
            time.sleep(args.poll_interval)

    current_stop.set()
    if current_thread is not None:
        # Finish a claimed current request just as the history lane finishes
        # its current task; do not abandon it after a fixed five-second grace.
        current_thread.join()
    cache.clear()
    _write_runtime(project_root, cache, "stopped")
    print("[query_worker] stopped", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent online query worker with session-level engine cache.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("EM2MEM_QUERY_WORKER_POLL_SECONDS", "1.0")))
    parser.add_argument("--cache-max-sessions", type=int, default=int(os.getenv("EM2MEM_QUERY_CACHE_MAX_SESSIONS", "9")))
    parser.add_argument("--ttl-seconds", type=int, default=int(os.getenv("EM2MEM_QUERY_CACHE_TTL_SECONDS", "3600")))
    parser.add_argument("--preload-vlm2vec", action=argparse.BooleanOptionalAction, default=_env_bool("EM2MEM_PRELOAD_VLM2VEC", False))
    parser.add_argument("--preload-sessions", default=os.getenv("EM2MEM_PRELOAD_QUERY_SESSIONS", ""))
    parser.add_argument("--preload-recent-memory-ready", type=int, default=int(os.getenv("EM2MEM_PRELOAD_RECENT_MEMORY_READY", "0")))
    parser.add_argument("--once", action="store_true", help="Process queued tasks once and exit.")
    parser.add_argument("--current-lane", action=argparse.BooleanOptionalAction, default=_env_bool("EM2MEM_QUERY_CURRENT_LANE", True), help="Serve independent current-only requests alongside history/index tasks.")
    return parser.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
