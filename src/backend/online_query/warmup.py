from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import utc_now_iso, write_json_atomic
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


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
        target = session_dir / "worldmm" / "query_warmup_state.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, payload)
    except Exception:
        pass


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
) -> dict[str, Any]:
    from online_query.query_cache import GLOBAL_SESSION_ENGINE_CACHE
    from online_query.query_engine import _get_short_term_answer_model, load_query_engine
    from online_pipeline.rokid_day import query_memory_ready, resolve_query_long_term_candidates, resolve_query_session_context
    from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime

    start = time.perf_counter()
    sessions_root = Path(sessions_root)
    requested_session_id = session_id
    session_dir = sessions_root / requested_session_id
    try:
        query_context = resolve_query_session_context(requested_session_id, sessions_root)
    except Exception:
        query_context = {
            "session_id": requested_session_id,
            "is_rokid_day_child": False,
            "long_term_session_id": requested_session_id,
            "parent_session_id": requested_session_id,
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
    timeout_seconds = _env_float("WORLDMM_QUERY_WARMUP_WAIT_MEMORY_SECONDS", 900.0) if timeout_seconds is None else float(timeout_seconds)
    poll_interval = _env_float("WORLDMM_QUERY_WARMUP_POLL_SECONDS", 2.0) if poll_interval is None else float(poll_interval)
    deadline = time.time() + max(0.0, timeout_seconds)
    payload: dict[str, Any] = {
        "status": "running",
        "session_id": requested_session_id,
        "requested_session_id": requested_session_id,
        "long_term_session_id": long_term_session_id,
        "parent_session_id": query_context.get("parent_session_id"),
        "is_rokid_day_child": bool(query_context.get("is_rokid_day_child")),
        "reason": reason,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "wait_for_memory": bool(wait_for_memory),
        "long_term_retrieval_scheme": long_term_retrieval_scheme,
        "long_term_selection": long_term_selection,
        "steps": [],
    }
    _write_warmup_state(session_dir, payload)

    def _step(name: str, status: str = "ok", **extra: Any) -> None:
        payload["steps"].append({"name": name, "status": status, "at": utc_now_iso(), **extra})
        payload["updated_at"] = utc_now_iso()
        _write_warmup_state(session_dir, payload)

    try:
        if not session_dir.exists():
            raise FileNotFoundError(f"session not found: {session_dir}")

        # Build lazy singletons used by current-frame answers before the first ask.
        if _env_bool("WORLDMM_QUERY_WARMUP_CURRENT_MODEL", True):
            _get_short_term_answer_model()
            _step("current_answer_model")

        # Ping remote VLM2Vec once so backend health/connectivity is resolved early.
        if _env_bool("WORLDMM_QUERY_WARMUP_VLM2VEC", True):
            runtime = get_global_vlm2vec_runtime()
            info = runtime.info()
            if getattr(runtime, "backend", None) == "remote":
                info["remote_health"] = runtime.ping_remote()
            _step("vlm2vec_runtime", backend=info.get("backend"), remote_url=info.get("remote_url"))

        if wait_for_memory:
            while time.time() < deadline:
                long_term_selection = resolve_query_long_term_candidates(
                    requested_session_id,
                    sessions_root,
                    query_context=query_context,
                )
                long_term_session_id = str(long_term_selection.get("selected_session_id") or query_context.get("long_term_session_id") or requested_session_id)
                long_term_session_dir = sessions_root / long_term_session_id
                payload["long_term_session_id"] = long_term_session_id
                payload["long_term_selection"] = long_term_selection
                payload["updated_at"] = utc_now_iso()
                _write_warmup_state(session_dir, payload)
                if query_memory_ready(long_term_session_dir):
                    break
                time.sleep(max(0.2, poll_interval))

        if query_memory_ready(long_term_session_dir):
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
            _step(
                "long_term_query_engine",
                cache_hit=bool(cache_hit),
                engine_load_ms=int(engine_load_ms),
                total_load_ms=int(round((time.perf_counter() - load_start) * 1000)),
                long_term_retrieval_scheme=long_term_retrieval_scheme,
                loaded_session_id=long_term_session_id,
            )
            payload["status"] = "ready"
        else:
            _step("long_term_query_engine", status="skipped", reason="memory_not_ready")
            payload["status"] = "partial"

        payload["finished_at"] = utc_now_iso()
        payload["total_ms"] = int(round((time.perf_counter() - start) * 1000))
        payload["updated_at"] = utc_now_iso()
        _write_warmup_state(session_dir, payload)
        return payload
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        payload["finished_at"] = utc_now_iso()
        payload["total_ms"] = int(round((time.perf_counter() - start) * 1000))
        payload["updated_at"] = utc_now_iso()
        _write_warmup_state(session_dir, payload)
        return payload
