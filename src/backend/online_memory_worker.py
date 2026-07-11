from __future__ import annotations

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from online_memory import build_online_worldmm_memory
from online_memory_incremental import IncrementalMemoryAppender
from online_pipeline.runtime_state import WorkerTaskHeartbeat, get_pipeline_mode, refresh_session_pipeline_state, write_worker_runtime
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json, utc_now_iso, write_json, write_json_atomic, write_status
from online_preprocess.task_queue import claim_memory_task, enqueue_query_warmup_task, enqueue_visual_task, ensure_queue_dirs, finish_memory_task, list_queued_memory_tasks
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log_stage(session_id: str, stage: str, **fields: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    print(f"[memory_worker][stage] session={session_id} stage={stage} {suffix}".rstrip(), flush=True)


def _mark_component_lagging(session_dir: Path, *, component: str, error: str, task_id: str | None = None) -> None:
    memory_config_path = session_dir / "worldmm" / "memory_config.json"
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        return
    config["memory_build_state"] = "ready_with_warnings" if (
        config.get("latest_fast_ready_version") or config.get("latest_ready_memory_version")
    ) else "failed"
    config[f"{component}_lagging"] = True
    config[f"{component}_error"] = error
    config[f"{component}_failed_at"] = utc_now_iso()
    if task_id:
        config["last_failed_task_id"] = task_id
    lag = config.get("lag") if isinstance(config.get("lag"), dict) else {}
    lag[f"{component}_lagging"] = True
    config["lag"] = lag
    readiness = config.get("readiness") if isinstance(config.get("readiness"), dict) else {}
    readiness[f"{component}_ready"] = False
    config["readiness"] = readiness
    config["updated_at"] = utc_now_iso()
    write_json_atomic(memory_config_path, config)


def _build_retrieval_artifacts(
    session_id: str,
    sessions_root: Path,
    long_term_retrieval_scheme: str | None = None,
) -> None:
    if not _env_bool("WORLDMM_MEMORY_BUILD_RETRIEVAL_ARTIFACTS", True):
        return
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    old_strict = os.environ.get("WORLDMM_QUERY_STRICT_LOAD_ONLY")
    old_cached = os.environ.get("WORLDMM_QUERY_USE_CACHED_HIPPORAG")
    os.environ["WORLDMM_QUERY_STRICT_LOAD_ONLY"] = "0"
    os.environ["WORLDMM_QUERY_USE_CACHED_HIPPORAG"] = "0"
    try:
        from online_query.query_engine import load_query_engine

        engine = load_query_engine(
            session_id=session_id,
            sessions_root=sessions_root,
            long_term_retrieval_scheme=long_term_retrieval_scheme,
        )
        engine.close()
        memory_config_path = sessions_root / session_id / "worldmm" / "memory_config.json"
        config = read_json(memory_config_path, default={})
        if isinstance(config, dict):
            config["hipporag_cache_ready"] = True
            config["episodic_index_ready"] = True
            config["retrieval_artifacts_ready"] = True
            config["retrieval_artifacts_built_at"] = utc_now_iso()
            config["default_long_term_retrieval_scheme"] = long_term_retrieval_scheme
            schemes = list(config.get("retrieval_artifact_schemes") or [])
            if long_term_retrieval_scheme not in schemes:
                schemes.append(long_term_retrieval_scheme)
            config["retrieval_artifact_schemes"] = schemes
            write_json(memory_config_path, config)
    finally:
        if old_strict is None:
            os.environ.pop("WORLDMM_QUERY_STRICT_LOAD_ONLY", None)
        else:
            os.environ["WORLDMM_QUERY_STRICT_LOAD_ONLY"] = old_strict
        if old_cached is None:
            os.environ.pop("WORLDMM_QUERY_USE_CACHED_HIPPORAG", None)
        else:
            os.environ["WORLDMM_QUERY_USE_CACHED_HIPPORAG"] = old_cached


def _build_retrieval_artifacts_isolated(
    session_id: str,
    sessions_root: Path,
    project_root: Path,
    long_term_retrieval_scheme: str | None = None,
) -> dict:
    if not _env_bool("WORLDMM_MEMORY_BUILD_RETRIEVAL_ARTIFACTS", True):
        return {"status": "disabled"}
    long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(long_term_retrieval_scheme)
    script = project_root / "memory_worker_subprocess.py"
    if not script.exists():
        if not _env_bool("WORLDMM_MEMORY_ALLOW_INPROCESS_RETRIEVAL_BUILD", False):
            return {"status": "skipped", "reason": "memory_worker_subprocess.py missing"}
        _build_retrieval_artifacts(
            session_id=session_id,
            sessions_root=sessions_root,
            long_term_retrieval_scheme=long_term_retrieval_scheme,
        )
        return {"status": "done", "mode": "in_process_explicit", "long_term_retrieval_scheme": long_term_retrieval_scheme}
    cmd = [
        sys.executable,
        str(script),
        "build_retrieval_artifacts",
        "--session-id",
        session_id,
        "--sessions-root",
        str(sessions_root),
        "--long-term-retrieval-scheme",
        long_term_retrieval_scheme,
    ]
    proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True, check=False)
    result = {
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "long_term_retrieval_scheme": long_term_retrieval_scheme,
    }
    if proc.returncode != 0:
        _mark_component_lagging(
            sessions_root / session_id,
            component="graph",
            error=f"retrieval subprocess failed rc={proc.returncode}: {proc.stderr[-1000:]}",
        )
    return result


def _memory_update_mode(task: dict, args: argparse.Namespace) -> str:
    mode = str(task.get("update_mode") or os.getenv("WORLDMM_MEMORY_UPDATE_MODE") or "incremental_append").strip().lower()
    if mode in {"incremental", "append"}:
        return "incremental_append"
    if mode in {"full", "full_rebuild", "full_rebuild_fallback"}:
        return "full_rebuild_fallback"
    return "incremental_append"


def _full_rebuild_allowed() -> bool:
    return _env_bool("WORLDMM_ALLOW_FULL_REBUILD_FALLBACK", False)


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _recover_stale_memory_tasks(project_root: Path) -> dict:
    """Move stale memory_in_progress tasks out of the active queue after crashes.

    By default stale tasks are marked failed instead of requeued. This prevents a
    broken semantic/NER task from being retried forever and consuming API quota.
    Set WORLDMM_MEMORY_REQUEUE_STALE_IN_PROGRESS=1 to requeue them.
    """

    stale_seconds = float(os.getenv("WORLDMM_MEMORY_TASK_STALE_SECONDS", "300"))
    requeue = _env_bool("WORLDMM_MEMORY_REQUEUE_STALE_IN_PROGRESS", False)
    retry_limit = int(os.getenv("WORLDMM_MEMORY_STALE_RETRY_LIMIT", "1"))
    dirs = ensure_queue_dirs(project_root)
    recovered: list[dict] = []
    now = datetime.now(timezone.utc)
    for path in sorted(dirs["memory_in_progress"].glob("*.json")):
        task = read_json(path, default={})
        if not isinstance(task, dict):
            continue
        updated_at = _parse_iso_datetime(task.get("updated_at") or task.get("claimed_at") or task.get("created_at"))
        age_seconds = (now - updated_at).total_seconds() if updated_at else stale_seconds + 1
        if age_seconds < stale_seconds:
            continue
        task["recovered_from_stale_in_progress"] = True
        task["stale_recovered_at"] = utc_now_iso()
        task["stale_age_seconds"] = round(age_seconds, 3)
        retry_count = int(task.get("stale_retry_count", 0) or 0)
        if requeue and retry_count < retry_limit:
            task["stale_retry_count"] = retry_count + 1
            task["status"] = "queued"
            task["updated_at"] = utc_now_iso()
            target = dirs["memory_queued"] / path.name
        else:
            task["status"] = "failed"
            task["error"] = (
                "stale memory_in_progress task recovered on worker startup"
                if not requeue
                else f"stale memory task exceeded retry limit {retry_limit}"
            )
            task["updated_at"] = utc_now_iso()
            target = dirs["memory_failed"] / path.name
        write_json_atomic(path, task)
        path.replace(target)
        recovered.append(
            {
                "task_id": task.get("task_id") or path.stem,
                "session_id": task.get("session_id"),
                "target_queue": target.parent.name,
                "age_seconds": round(age_seconds, 3),
            }
        )
    return {
        "recovered_count": len(recovered),
        "requeued": requeue,
        "retry_limit": retry_limit,
        "stale_seconds": stale_seconds,
        "tasks": recovered[-20:],
    }


def _memory_task_dedupe_key(task: dict) -> str | None:
    task_type = str(task.get("task_type") or "")
    if task_type != "memory_append":
        return None
    dedupe_key = str(task.get("dedupe_key") or "").strip()
    if not dedupe_key:
        episode_ids = task.get("episode_ids")
        if isinstance(episode_ids, list) and episode_ids:
            dedupe_key = ",".join(sorted({str(item) for item in episode_ids if str(item or "").strip()}))
        elif task.get("append_ready_episodes"):
            dedupe_key = "append_ready_all"
    if not dedupe_key:
        return None
    return "|".join(
        [
            str(task.get("session_id") or ""),
            task_type,
            str(task.get("source") or ""),
            str(task.get("update_mode") or ""),
            dedupe_key,
        ]
    )


def _dedupe_queued_memory_tasks(project_root: Path) -> dict:
    dirs = ensure_queue_dirs(project_root)
    seen: dict[str, Path] = {}
    skipped: list[dict] = []
    for path in sorted(dirs["memory_queued"].glob("*.json")):
        task = read_json(path, default={})
        if not isinstance(task, dict):
            continue
        key = _memory_task_dedupe_key(task)
        if not key:
            continue
        existing = seen.get(key)
        if existing is None:
            seen[key] = path
            continue
        task["status"] = "skipped_duplicate"
        task["duplicate_of"] = existing.name
        task["dedupe_skipped_at"] = utc_now_iso()
        task["updated_at"] = utc_now_iso()
        target = dirs["memory_failed"] / path.name
        write_json_atomic(path, task)
        path.replace(target)
        skipped.append(
            {
                "task_id": task.get("task_id") or path.stem,
                "session_id": task.get("session_id"),
                "duplicate_of": existing.name,
            }
        )
    return {"skipped_duplicate_count": len(skipped), "tasks": skipped[-20:]}


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    stale_recovery = _recover_stale_memory_tasks(project_root)
    queued_dedupe = _dedupe_queued_memory_tasks(project_root)
    print(
        "Memory worker started:",
        f"skip_visual_embedding={args.skip_visual_embedding}",
        f"skip_semantic={args.skip_semantic}",
        f"stale_recovered={stale_recovery['recovered_count']}",
        f"queued_duplicates_skipped={queued_dedupe['skipped_duplicate_count']}",
        flush=True,
    )

    def _queue_pending() -> int:
        return len(list_queued_memory_tasks(project_root))

    def _runtime_extra(session_id: str) -> dict:
        return {
            "session_id": session_id,
            "skip_visual_embedding": args.skip_visual_embedding,
            "skip_semantic": args.skip_semantic,
            "auto_visual_embedding": args.auto_visual_embedding,
            "visual_backend": args.visual_backend,
            "pipeline_mode": get_pipeline_mode(),
            "memory_source": args.source,
            "default_update_mode": os.getenv("WORLDMM_MEMORY_UPDATE_MODE", "incremental_append"),
        }
    write_worker_runtime(
        project_root,
        "memory",
        status="ready",
        backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
        model_name=args.model,
        client_loaded=True,
        warmup_done=True,
        queue_pending=len(list_queued_memory_tasks(project_root)),
        extra={
            "skip_visual_embedding": args.skip_visual_embedding,
            "skip_semantic": args.skip_semantic,
            "auto_visual_embedding": args.auto_visual_embedding,
            "visual_backend": args.visual_backend,
            "pipeline_mode": get_pipeline_mode(),
            "memory_source": args.source,
            "default_update_mode": os.getenv("WORLDMM_MEMORY_UPDATE_MODE", "incremental_append"),
            "stale_task_recovery": stale_recovery,
            "queued_task_dedupe": queued_dedupe,
        },
    )

    while True:
        queued_tasks = list_queued_memory_tasks(project_root)
        if not queued_tasks:
            write_worker_runtime(
                project_root,
                "memory",
                status="ready",
                backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
                model_name=args.model,
                client_loaded=True,
                warmup_done=True,
                queue_pending=0,
                extra={
                    "skip_visual_embedding": args.skip_visual_embedding,
                    "skip_semantic": args.skip_semantic,
                    "auto_visual_embedding": args.auto_visual_embedding,
                    "visual_backend": args.visual_backend,
                    "pipeline_mode": get_pipeline_mode(),
                    "memory_source": args.source,
                    "default_update_mode": os.getenv("WORLDMM_MEMORY_UPDATE_MODE", "incremental_append"),
                },
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued_tasks:
            claimed = claim_memory_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task["session_id"])
            session_dir = sessions_root / session_id
            try:
                write_worker_runtime(
                    project_root,
                    "memory",
                    status="busy",
                    backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
                    model_name=args.model,
                    client_loaded=True,
                    warmup_done=True,
                    queue_pending=len(list_queued_memory_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    extra={"session_id": session_id},
                )
                write_status(
                    session_dir=session_dir,
                    session_id=session_id,
                    status="processing",
                    stage="memory_building",
                    progress=92,
                    error=None,
                )
                source = str(task.get("source") or args.source)
                update_mode = _memory_update_mode(task, args)
                incremental_result = None
                heartbeat_extra = {**_runtime_extra(session_id), "update_mode": update_mode, "source": source}
                with WorkerTaskHeartbeat(
                    project_root,
                    "memory",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
                    model_name=args.model,
                    client_loaded=True,
                    warmup_done=True,
                    queue_pending=_queue_pending,
                    extra=heartbeat_extra,
                    interval_env="WORLDMM_MEMORY_HEARTBEAT_SECONDS",
                ):
                    if update_mode == "incremental_append" and source in {"auto", "mst_episodic", "mst", "mst_micro_events"}:
                        _log_stage(session_id, "fast_append", task_id=str(task.get("task_id") or claimed_path.stem), update_mode=update_mode)
                        appender = IncrementalMemoryAppender(
                            session_id=session_id,
                            sessions_root=sessions_root,
                            project_root=project_root,
                            model_name=args.model,
                            verbose=args.verbose,
                        )
                        incremental_result = appender.append_ready_episodes(
                            episode_ids=task.get("episode_ids") if isinstance(task.get("episode_ids"), list) else None,
                            force=bool(task.get("force", args.force)),
                            dry_run=False,
                            skip_graph_semantic=bool(task.get("skip_semantic", args.skip_semantic)),
                        )
                    else:
                        if update_mode == "full_rebuild_fallback" and not _full_rebuild_allowed():
                            raise RuntimeError(
                                "full_rebuild_fallback is disabled. Set WORLDMM_ALLOW_FULL_REBUILD_FALLBACK=1 to run a full rebuild."
                            )
                        _log_stage(session_id, "fast_append", task_id=str(task.get("task_id") or claimed_path.stem), update_mode=update_mode)
                        build_online_worldmm_memory(
                            session_id=session_id,
                            sessions_root=sessions_root,
                            force=bool(task.get("force", args.force)),
                            skip_visual_embedding=True,
                            skip_semantic=bool(task.get("skip_semantic", args.skip_semantic)),
                            limit_segments=task.get("limit_segments") or args.limit_segments,
                            dry_run=False,
                            model_name=args.model,
                            source=source,
                            verbose=args.verbose,
                        )
                if incremental_result is not None:
                    _log_stage(
                        session_id,
                        "component_versions_update",
                        fast_version=incremental_result.fast_memory_version,
                        semantic_lagging=incremental_result.semantic_lagging,
                        graph_lagging=incremental_result.graph_lagging,
                        visual_lagging=incremental_result.visual_lagging,
                    )
                    write_status(
                        session_dir=session_dir,
                        session_id=session_id,
                        status="done",
                        stage="memory_incremental_ready",
                        progress=100,
                        error=None,
                        outputs=incremental_result.to_dict(),
                    )
                if incremental_result is None and args.auto_visual_embedding and not bool(task.get("skip_online_visual_embedding", False)):
                    _log_stage(session_id, "visual_enqueue", backend=args.visual_backend)
                    try:
                        visual_task_path = enqueue_visual_task(
                            project_root=project_root,
                            session_id=session_id,
                            backend=args.visual_backend,
                            force=bool(task.get("force_visual_embedding", True)),
                        )
                        memory_config_path = session_dir / "worldmm" / "memory_config.json"
                        config = read_json(memory_config_path, default={})
                        if isinstance(config, dict):
                            config["visual_embedding_ready"] = False
                            config["visual_lagging"] = True
                            config["visual_task_path"] = str(visual_task_path)
                            config["memory_build_state"] = "ready_with_warnings"
                            config["updated_at"] = utc_now_iso()
                            write_json_atomic(memory_config_path, config)
                        _log_stage(
                            session_id,
                            "visual_build",
                            delegated_to="online_visual_worker",
                            visual_task_path=str(visual_task_path),
                        )
                    except Exception as exc:
                        _mark_component_lagging(
                            session_dir,
                            component="visual",
                            error=str(exc),
                            task_id=str(task.get("task_id") or claimed_path.stem),
                        )
                        print(f"[memory_worker] optional visual embedding failed session={session_id}: {exc}", flush=True)
                if incremental_result is None:
                    graph_task = {
                        **task,
                        "task_id": str(task.get("task_id") or claimed_path.stem),
                        "task_type": "memory_retrieval_artifacts",
                        "session_id": session_id,
                    }
                    with WorkerTaskHeartbeat(
                        project_root,
                        "memory",
                        task=graph_task,
                        claimed_path=claimed_path,
                        status="busy",
                        backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
                        model_name=args.model,
                        client_loaded=True,
                        warmup_done=True,
                        queue_pending=_queue_pending,
                        extra=heartbeat_extra,
                        interval_env="WORLDMM_MEMORY_HEARTBEAT_SECONDS",
                    ):
                        _log_stage(session_id, "graph_build", mode="retrieval_artifacts_subprocess")
                        long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(None)
                        retrieval_result = _build_retrieval_artifacts_isolated(
                            session_id=session_id,
                            sessions_root=sessions_root,
                            project_root=project_root,
                            long_term_retrieval_scheme=long_term_retrieval_scheme,
                        )
                        _log_stage(
                            session_id,
                            "graph_build",
                            status=retrieval_result.get("status"),
                            returncode=retrieval_result.get("returncode"),
                            long_term_retrieval_scheme=retrieval_result.get("long_term_retrieval_scheme"),
                        )
                finish_memory_task(
                    project_root,
                    claimed_path,
                    task,
                    status="done",
                    result=incremental_result.to_dict() if incremental_result is not None else None,
                )
                if _env_bool("WORLDMM_AUTO_QUERY_WARMUP_ON_MEMORY_READY", True):
                    try:
                        warmup_task_path = enqueue_query_warmup_task(
                            project_root,
                            session_id,
                            reason="memory_ready",
                            wait_for_memory=False,
                            long_term_retrieval_scheme=normalize_long_term_retrieval_scheme(None),
                            retrieval_scheme=normalize_long_term_retrieval_scheme(None),
                        )
                        _log_stage(
                            session_id,
                            "query_warmup_queued",
                            warmup_task_path=str(warmup_task_path),
                        )
                    except Exception as exc:
                        print(f"[memory_worker] query warmup enqueue failed session={session_id}: {exc}", flush=True)
                append_timeline_event(
                    session_dir,
                    "memory_append_done" if incremental_result is not None else "memory_build_done",
                    metadata={
                        "task_id": str(task.get("task_id") or claimed_path.stem),
                        "update_mode": update_mode,
                        "result": incremental_result.to_dict() if incremental_result is not None else None,
                    },
                )
                refresh_session_pipeline_state(session_dir)
            except Exception as exc:
                memory_config_path = session_dir / "worldmm" / "memory_config.json"
                if memory_config_path.exists():
                    config = read_json(memory_config_path, default={})
                    if isinstance(config, dict):
                        if config.get("memory_build_state") == "waiting":
                            config["last_waiting_error"] = str(exc)
                            config["last_waiting_at"] = utc_now_iso()
                        elif config.get("latest_fast_ready_version") or config.get("latest_ready_memory_version"):
                            config["memory_build_state"] = "ready_with_warnings"
                            config["last_build_error"] = str(exc)
                            config["last_build_failed_at"] = utc_now_iso()
                            config["last_failed_task_id"] = str(task.get("task_id") or claimed_path.stem)
                        else:
                            config["memory_build_state"] = "failed"
                            config["last_build_error"] = str(exc)
                            config["last_build_failed_at"] = utc_now_iso()
                        write_json(memory_config_path, config)
                waiting = False
                if memory_config_path.exists():
                    config = read_json(memory_config_path, default={})
                    waiting = isinstance(config, dict) and config.get("memory_build_state") == "waiting"
                write_status(
                    session_dir=session_dir,
                    session_id=session_id,
                    status="processing" if waiting else "failed",
                    stage="memory_waiting_for_mst_consolidation" if waiting else "memory_worker",
                    progress=90 if waiting else 100,
                    error=None if waiting else str(exc),
                )
                finish_memory_task(project_root, claimed_path, task, status="failed", error=str(exc))
                write_worker_runtime(
                    project_root,
                    "memory",
                    status="error",
                    backend=os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND", "llm"),
                    model_name=args.model,
                    client_loaded=True,
                    warmup_done=True,
                    queue_pending=len(list_queued_memory_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    last_error=str(exc),
                    extra={"session_id": session_id},
                )

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent worker for online WorldMM memory tasks.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", default=_env_bool("WORLDMM_FORCE_MEMORY", False))
    parser.add_argument("--skip-visual-embedding", action="store_true", default=_env_bool("WORLDMM_SKIP_VISUAL_EMBEDDING", True))
    parser.add_argument("--skip-semantic", action="store_true", default=_env_bool("WORLDMM_SKIP_SEMANTIC_MEMORY", False))
    parser.add_argument("--auto-visual-embedding", action=argparse.BooleanOptionalAction, default=_env_bool("WORLDMM_AUTO_VISUAL_EMBEDDING", True))
    parser.add_argument("--visual-backend", default=os.getenv("WORLDMM_VISUAL_BACKEND", "vlm2vec"))
    parser.add_argument("--visual-batch-size", type=int, default=int(os.getenv("WORLDMM_VISUAL_BATCH_SIZE", "8")))
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--model", default=os.getenv("WORLDMM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL"))
    parser.add_argument("--source", default=os.getenv("WORLDMM_MEMORY_SOURCE", "auto"), choices=["auto", "online_evidence", "legacy_evidence", "mst_episodic"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
