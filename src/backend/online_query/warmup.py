from __future__ import annotations

import os
import time
import traceback
import threading
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import utc_now_iso, write_json_atomic
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


_WARMUP_STATE_CACHE: dict[str, dict[str, Any]] = {}
_WARMUP_STATE_CACHE_LOCK = threading.RLock()
_WARMUP_THREADS: dict[str, threading.Thread] = {}
_WARMUP_THREADS_LOCK = threading.RLock()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _write_warmup_state(session_dir: Path, payload: dict[str, Any]) -> None:
    try:
        target = session_dir / "em2mem" / "query_warmup_state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, payload)
    except Exception:
        pass


def _update_warmup_state_cache(session_id: str, payload: dict[str, Any]) -> None:
    with _WARMUP_STATE_CACHE_LOCK:
        _WARMUP_STATE_CACHE[session_id] = dict(payload)


def get_warmup_state(session_id: str) -> dict[str, Any] | None:
    with _WARMUP_STATE_CACHE_LOCK:
        return _WARMUP_STATE_CACHE.get(session_id)


def is_warmup_running(session_id: str) -> bool:
    with _WARMUP_THREADS_LOCK:
        thread = _WARMUP_THREADS.get(session_id)
        return thread is not None and thread.is_alive()


def warm_query_session(
    *,
    session_id: str,
    sessions_root: Path,
    cache: Any = None,
    wait_for_memory: bool = True,
    timeout_seconds: float | None = None,
    poll_interval: float | None = None,
    reason: str = "stream_start",
    long_term_retrieval_scheme: str | None = None,
    async_mode: bool = False,
    skip_long_term_preload: bool = False,
) -> dict[str, Any]:
    from online_query.query_cache import GLOBAL_SESSION_ENGINE_CACHE
    from online_query.query_engine import _get_short_term_answer_model, load_query_engine
    from online_pipeline.rokid_day import query_memory_ready, resolve_query_long_term_candidates, resolve_query_session_context
    from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime

    sessions_root = Path(sessions_root)
    requested_session_id = session_id
    session_dir = sessions_root / requested_session_id

    try:
        query_context = resolve_query_session_context(requested_session_id, sessions_root)
    except Exception:
        query_context = {
            "session_id": requested_session_id,
            "long_term_session_id": requested_session_id,
        }
    long_term_selection = resolve_query_long_term_candidates(
        requested_session_id,
        sessions_root,
        query_context=query_context,
    )
    long_term_session_id = str(long_term_selection.get("selected_session_id") or query_context.get("long_term_session_id") or requested_session_id)
    long_term_session_dir = sessions_root / long_term_session_id
    cache = cache or GLOBAL_SESSION_ENGINE_CACHE
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    timeout_seconds = _env_float("EM2MEM_QUERY_WARMUP_WAIT_MEMORY_SECONDS", 120.0) if timeout_seconds is None else float(timeout_seconds)
    poll_interval = _env_float("EM2MEM_QUERY_WARMUP_POLL_SECONDS", 3.0) if poll_interval is None else float(poll_interval)

    payload: dict[str, Any] = {
        "status": "running",
        "session_id": requested_session_id,
        "requested_session_id": requested_session_id,
        "long_term_session_id": long_term_session_id,
        "day_context": query_context.get("day_context"),
        "reason": reason,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "wait_for_memory": bool(wait_for_memory),
        "long_term_retrieval_scheme": long_term_retrieval_scheme,
        "long_term_selection": long_term_selection,
        "steps": [],
        "async_mode": async_mode,
    }
    _write_warmup_state(session_dir, payload)
    _update_warmup_state_cache(requested_session_id, payload)

    def _do_warmup() -> None:
        start = time.perf_counter()
        nonlocal payload
        try:
            if not session_dir.exists():
                raise FileNotFoundError(f"session not found: {session_dir}")

            if _env_bool("EM2MEM_QUERY_WARMUP_CURRENT_MODEL", True):
                _get_short_term_answer_model()
                payload["steps"].append({"name": "current_answer_model", "status": "ok", "at": utc_now_iso()})
                payload["updated_at"] = utc_now_iso()
                _update_warmup_state_cache(requested_session_id, payload)

            if _env_bool("EM2MEM_QUERY_WARMUP_VLM2VEC", False):
                runtime = get_global_vlm2vec_runtime()
                info = runtime.info()
                if getattr(runtime, "backend", None) == "remote":
                    info["remote_health"] = runtime.ping_remote()
                payload["steps"].append({"name": "vlm2vec_runtime", "status": "ok", "at": utc_now_iso(), "backend": info.get("backend"), "remote_url": info.get("remote_url")})
                payload["updated_at"] = utc_now_iso()
                _update_warmup_state_cache(requested_session_id, payload)

            if skip_long_term_preload:
                payload["steps"].append({"name": "long_term_query_engine", "status": "skipped", "reason": "skip_long_term_preload", "at": utc_now_iso()})
                payload["status"] = "partial"
            else:
                if wait_for_memory:
                    deadline = time.time() + max(0.0, timeout_seconds)
                    poll_count = 0
                    while time.time() < deadline:
                        if query_memory_ready(long_term_session_dir):
                            break
                        poll_count += 1
                        if poll_count % 10 == 0:
                            payload["updated_at"] = utc_now_iso()
                            payload["poll_count"] = poll_count
                            _update_warmup_state_cache(requested_session_id, payload)
                        time.sleep(max(0.5, poll_interval))

                memory_ready = query_memory_ready(long_term_session_dir)
                if memory_ready:
                    load_start = time.perf_counter()
                    engine, cache_hit, engine_load_ms = cache.get_or_load(
                        session_id=long_term_session_id,
                        long_term_retrieval_scheme=long_term_retrieval_scheme,
                        loader=lambda sid: load_query_engine(
                            sid,
                            sessions_root=sessions_root,
                            long_term_retrieval_scheme=long_term_retrieval_scheme,
                        ),
                    )
                    del engine
                    payload["steps"].append({
                        "name": "long_term_query_engine",
                        "status": "ok",
                        "at": utc_now_iso(),
                        "cache_hit": bool(cache_hit),
                        "engine_load_ms": int(engine_load_ms),
                        "total_load_ms": int(round((time.perf_counter() - load_start) * 1000)),
                        "long_term_retrieval_scheme": long_term_retrieval_scheme,
                        "loaded_session_id": long_term_session_id,
                    })
                    payload["status"] = "ready"
                else:
                    if wait_for_memory:
                        payload["steps"].append({"name": "long_term_query_engine", "status": "skipped", "reason": "memory_not_ready_after_wait", "at": utc_now_iso()})
                        payload["status"] = "partial"
                    else:
                        payload["steps"].append({"name": "long_term_query_engine", "status": "skipped", "reason": "wait_for_memory_false", "at": utc_now_iso()})
                        payload["status"] = "partial"

            payload["finished_at"] = utc_now_iso()
            payload["total_ms"] = int(round((time.perf_counter() - start) * 1000))
            payload["updated_at"] = utc_now_iso()
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = str(exc)
            payload["traceback"] = traceback.format_exc()
            payload["finished_at"] = utc_now_iso()
            payload["total_ms"] = int(round((time.perf_counter() - start) * 1000))
            payload["updated_at"] = utc_now_iso()
        finally:
            _write_warmup_state(session_dir, payload)
            _update_warmup_state_cache(requested_session_id, payload)
            with _WARMUP_THREADS_LOCK:
                _WARMUP_THREADS.pop(requested_session_id, None)

    if async_mode:
        with _WARMUP_THREADS_LOCK:
            if requested_session_id in _WARMUP_THREADS and _WARMUP_THREADS[requested_session_id].is_alive():
                pass
            else:
                thread = threading.Thread(target=_do_warmup, daemon=True)
                _WARMUP_THREADS[requested_session_id] = thread
                thread.start()
        payload["async_mode"] = True
        payload["status"] = "running_async"
        return payload

    _do_warmup()
    return payload
