from __future__ import annotations

import os
import threading
import time
import gc
from collections import OrderedDict
from typing import Callable

from .query_engine import LoadedQueryEngine
from online_retrieval_scheme import normalize_long_term_retrieval_scheme, retrieval_scheme_cache_key


class SessionEngineCache:
    def __init__(self, max_sessions: int = 9, ttl_seconds: int = 3600) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, LoadedQueryEngine] = OrderedDict()
        self._lock = threading.RLock()

    def _close_engine(self, engine: LoadedQueryEngine) -> None:
        try:
            engine.close()
        except Exception:
            pass
        try:
            del engine
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _evict_expired_locked(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, engine in self._items.items()
            if now - engine.last_accessed_at > self.ttl_seconds
        ]
        for session_id in expired:
            engine = self._items.pop(session_id)
            self._close_engine(engine)
            print(f"[query_cache] evicted expired session={session_id}", flush=True)

    def _evict_lru_locked(self) -> None:
        while len(self._items) > self.max_sessions:
            session_id, engine = self._items.popitem(last=False)
            self._close_engine(engine)
            print(f"[query_cache] evicted lru session={session_id}", flush=True)

    def get_or_load(
        self,
        session_id: str,
        loader: Callable[[str], LoadedQueryEngine],
        long_term_retrieval_scheme: str | None = None,
    ) -> tuple[LoadedQueryEngine, bool, int]:
        scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
        cache_key = retrieval_scheme_cache_key(session_id, scheme)
        with self._lock:
            self._evict_expired_locked()
            engine = self._items.get(cache_key)
            if engine is not None:
                if engine.needs_reload():
                    old_engine = self._items.pop(cache_key)
                    self._close_engine(old_engine)
                    print(
                        f"[query_cache] reloading session={session_id} scheme={scheme} because memory_config changed",
                        flush=True,
                    )
                    engine = None
                else:
                    engine.touch()
                    self._items.move_to_end(cache_key)
                    return engine, True, 0
        start = time.perf_counter()
        engine = loader(session_id)
        engine_load_ms = int(round((time.perf_counter() - start) * 1000))

        with self._lock:
            old_engine = self._items.pop(cache_key, None)
            if old_engine is not None and old_engine is not engine:
                self._close_engine(old_engine)
            self._items[cache_key] = engine
            self._items.move_to_end(cache_key)
            self._evict_expired_locked()
            self._evict_lru_locked()
        return engine, False, engine_load_ms

    def reload_changed(
        self,
        loader: Callable[[str, str], LoadedQueryEngine],
    ) -> list[dict]:
        results = []
        with self._lock:
            cache_keys = list(self._items.keys())
        for cache_key in cache_keys:
            with self._lock:
                engine = self._items.get(cache_key)
                if engine is None or not engine.needs_reload():
                    continue
                old_engine = self._items.pop(cache_key, None)
            if old_engine is not None:
                self._close_engine(old_engine)
            start = time.perf_counter()
            try:
                session_id = getattr(old_engine, "session_id", cache_key)
                scheme = normalize_long_term_retrieval_scheme(
                    getattr(old_engine, "long_term_retrieval_scheme", None)
                )
                new_engine = loader(session_id, scheme)
                engine_load_ms = int(round((time.perf_counter() - start) * 1000))
                with self._lock:
                    self._items[cache_key] = new_engine
                    self._items.move_to_end(cache_key)
                results.append(
                    {
                        "session_id": session_id,
                        "long_term_retrieval_scheme": scheme,
                        "status": "reloaded",
                        "active_query_memory_version": new_engine.active_query_memory_version,
                        "latest_ready_memory_version": new_engine.latest_ready_memory_version,
                        "engine_load_ms": engine_load_ms,
                    }
                )
            except Exception as exc:
                if old_engine is not None:
                    with self._lock:
                        self._items[cache_key] = old_engine
                        self._items.move_to_end(cache_key)
                results.append(
                    {
                        "session_id": getattr(old_engine, "session_id", cache_key),
                        "long_term_retrieval_scheme": getattr(old_engine, "long_term_retrieval_scheme", None),
                        "status": "reload_failed",
                        "error": str(exc),
                    }
                )
        return results

    def invalidate(self, session_id: str, long_term_retrieval_scheme: str | None = None) -> bool:
        engines: list[LoadedQueryEngine] = []
        with self._lock:
            if long_term_retrieval_scheme is not None:
                cache_key = retrieval_scheme_cache_key(session_id, long_term_retrieval_scheme)
                engine = self._items.pop(cache_key, None)
                if engine is not None:
                    engines.append(engine)
            else:
                cache_keys = [
                    cache_key
                    for cache_key, engine in self._items.items()
                    if getattr(engine, "session_id", cache_key) == session_id
                ]
                for cache_key in cache_keys:
                    engine = self._items.pop(cache_key, None)
                    if engine is not None:
                        engines.append(engine)
        for engine in engines:
            self._close_engine(engine)
        return bool(engines)

    def clear(self) -> None:
        with self._lock:
            engines = list(self._items.values())
            self._items.clear()
        for engine in engines:
            self._close_engine(engine)

    def runtime_info(self) -> dict:
        with self._lock:
            self._evict_expired_locked()
            loaded_sessions = []
            for cache_key, engine in self._items.items():
                loaded_sessions.append(
                    {
                        "cache_key": cache_key,
                        "session_id": engine.session_id,
                        "long_term_retrieval_scheme": getattr(engine, "long_term_retrieval_scheme", None),
                        "loaded_at": engine.loaded_at,
                        "last_accessed_at": engine.last_accessed_at,
                        "query_count": engine.query_count,
                        "pipeline_mode": engine.memory_config.get("pipeline_mode"),
                        "active_30s_source": engine.memory_config.get("active_30s_source") or engine.memory_config.get("em2mem_30s_input_source"),
                        "episodic_source": engine.memory_config.get("episodic_source"),
                        "legacy_evidence_used": bool(engine.memory_config.get("legacy_evidence_used") or engine.memory_config.get("legacy_evidence_fallback_used")),
                        "em2mem_update_mode": engine.memory_config.get("em2mem_update_mode"),
                        "memory_config_mtime": engine.memory_config_mtime,
                        "strict_load_only": engine.strict_load_only,
                        "preload_status": engine.preload_status,
                        "latest_ready_memory_version": engine.latest_ready_memory_version,
                        "building_memory_version": engine.building_memory_version,
                        "active_query_memory_version": engine.active_query_memory_version,
                        "memory_component_versions": engine._memory_component_versions(),
                        "memory_build_state": engine.memory_config.get("memory_build_state"),
                        "using_stale_while_building": bool(
                            engine.building_memory_version
                            and engine.active_query_memory_version
                            and int(engine.building_memory_version) > int(engine.active_query_memory_version)
                        ),
                        "current_ready": engine.current_ready,
                        "current_stale": engine.current_stale,
                        "current_version": engine.mcur_version,
                        "short_term_ready": engine.short_term_ready,
                        "mst_version": engine.mst_version,
                        "long_term_ready": True,
                        "memory_version": engine.memory_config.get("memory_version") or engine.memory_config.get("updated_at") or engine.memory_config_mtime,
                        "cache_ready": True,
                        "recent_queries": list(engine.recent_queries),
                        "interaction_cache": engine.interaction_cache.summary(),
                    }
                )
            return {
                "cache_max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "loaded_sessions": loaded_sessions,
            }


GLOBAL_SESSION_ENGINE_CACHE = SessionEngineCache(
    max_sessions=int(os.getenv("EM2MEM_QUERY_CACHE_MAX_SESSIONS", "9")),
    ttl_seconds=int(os.getenv("EM2MEM_QUERY_CACHE_TTL_SECONDS", "3600")),
)
