from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from online_mst_to_em2mem import consolidate_short_term_to_em2mem
from online_pipeline.active_session import session_is_active_or_allowed
from online_pipeline.runtime_state import WorkerTaskHeartbeat, refresh_session_pipeline_state, write_worker_runtime
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_preprocess.task_queue import (
    claim_mst_consolidation_task,
    enqueue_memory_task,
    finish_mst_consolidation_task,
    list_queued_mst_consolidation_tasks,
)
from online_streaming.transcript_backfill import mark_transcript_dirty_window_consolidated
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT


def _ready_window_count(session_dir: Path) -> int:
    windows_path = session_dir / "short_term" / "refine" / "refined_ready_windows.json"
    windows = read_json(windows_path, default=[])
    if not isinstance(windows, list):
        return 0
    state = read_json(session_dir / "short_term" / "consolidation_state.json", default={})
    consolidated = set((state or {}).get("window_to_episode", {}).keys()) if isinstance(state, dict) else set()
    count = 0
    for window in windows:
        if not isinstance(window, dict):
            continue
        window_id = str(window.get("window_id") or "")
        if window.get("ready_for_30s_episodic") and window.get("is_closed_window") and window_id not in consolidated:
            count += 1
    return count


def run_worker(
    *,
    sessions_root: Path,
    backend: str,
    update_em2mem: bool,
    interval_seconds: float,
    limit_windows: int | None,
    once: bool,
    verbose: bool,
) -> None:
    runtime_path = Path("online_tasks") / "mst_consolidation_runtime.json"
    project_root = Path(__file__).resolve().parent
    last_task_id = None
    last_error = None
    model_name = os.getenv("EM2MEM_MST_EPISODIC_MODEL") or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL")

    def _queue_pending() -> int:
        return len(list_queued_mst_consolidation_tasks(project_root))

    write_worker_runtime(
        project_root,
        "consolidation",
        status="ready",
        backend=backend,
        model_name=model_name,
        client_loaded=backend == "openai",
        warmup_done=True,
        queue_pending=_queue_pending(),
        extra={"update_em2mem": update_em2mem},
    )
    while True:
        results = []
        queued_tasks = list_queued_mst_consolidation_tasks(project_root)
        for task_path in queued_tasks:
            claimed = claim_mst_consolidation_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task.get("session_id") or "")
            task_id = str(task.get("task_id") or claimed_path.stem)
            last_task_id = task_id
            try:
                write_worker_runtime(
                    project_root,
                    "consolidation",
                    status="busy",
                    backend=str(task.get("backend") or backend),
                    model_name=os.getenv("EM2MEM_MST_EPISODIC_MODEL") or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL"),
                    client_loaded=(str(task.get("backend") or backend) == "openai"),
                    warmup_done=True,
                    queue_pending=len(list_queued_mst_consolidation_tasks(project_root)),
                    last_task_id=task_id,
                    extra={"session_id": session_id, "update_em2mem": bool(task.get("update_em2mem", update_em2mem))},
                )
                requested_update_em2mem = bool(task.get("update_em2mem", update_em2mem))
                task_backend = str(task.get("backend") or backend)
                with WorkerTaskHeartbeat(
                    project_root,
                    "consolidation",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    backend=task_backend,
                    model_name=model_name,
                    client_loaded=task_backend == "openai",
                    warmup_done=True,
                    queue_pending=_queue_pending,
                    extra={"session_id": session_id, "update_em2mem": requested_update_em2mem},
                    interval_env="EM2MEM_MST_CONSOLIDATION_HEARTBEAT_SECONDS",
                ):
                    result = consolidate_short_term_to_em2mem(
                        session_id=session_id,
                        sessions_root=sessions_root,
                        backend=task_backend,
                        update_em2mem=False,
                        force=bool(task.get("force", False)),
                        limit_windows=task.get("limit_windows") or limit_windows,
                        window_start=task.get("window_start"),
                        window_end=task.get("window_end"),
                        verbose=verbose,
                    )
                generated_episode_ids = result.get("generated_episode_ids") if isinstance(result.get("generated_episode_ids"), list) else []
                if requested_update_em2mem and generated_episode_ids:
                    memory_task = enqueue_memory_task(
                        project_root=project_root,
                        session_id=session_id,
                        force=bool(task.get("force", False)),
                        skip_visual_embedding=os.getenv("EM2MEM_SKIP_VISUAL_EMBEDDING", "1").lower() in {"1", "true", "yes", "on"},
                        skip_semantic=os.getenv("EM2MEM_SKIP_SEMANTIC_MEMORY", "0").lower() in {"1", "true", "yes", "on"},
                        source="mst_episodic",
                        update_mode="incremental_append",
                        append_ready_episodes=True,
                        episode_ids=generated_episode_ids,
                        reason=str(task.get("reason") or "mst_consolidation"),
                    )
                    result["memory_task_path"] = str(memory_task)
                    result["updated_em2mem"] = False
                    result["memory_update_queued"] = True
                    append_timeline_event(
                        sessions_root / session_id,
                        "memory_append_queued",
                        metadata={"task_id": memory_task.stem, "episode_ids": generated_episode_ids},
                    )
                elif requested_update_em2mem:
                    result["updated_em2mem"] = False
                    result["memory_update_queued"] = False
                    result["memory_skip_reason"] = "no newly generated episodes"
                if str(task.get("reason") or "") == "transcript_backfill" and task.get("window_start") is not None and task.get("window_end") is not None:
                    mark_transcript_dirty_window_consolidated(
                        sessions_root / session_id,
                        float(task.get("window_start")),
                        float(task.get("window_end")),
                        task_id=task_id,
                    )
                refresh_session_pipeline_state(sessions_root / session_id)
                finish_mst_consolidation_task(project_root, claimed_path, task, status="done", result=result)
                append_timeline_event(
                    sessions_root / session_id,
                    "consolidation_done",
                    metadata={
                        "task_id": task_id,
                        "generated_episode_ids": generated_episode_ids,
                        "memory_update_queued": result.get("memory_update_queued"),
                    },
                )
                results.append(result)
                last_error = None
            except Exception as exc:
                last_error = str(exc)
                finish_mst_consolidation_task(project_root, claimed_path, task, status="failed", error=str(exc))
                results.append({"session_id": session_id, "status": "error", "message": str(exc)})

        for session_dir in sorted(sessions_root.iterdir() if sessions_root.exists() else []):
            if not session_dir.is_dir():
                continue
            if not session_is_active_or_allowed(project_root, session_dir.name):
                continue
            ready_count = _ready_window_count(session_dir)
            if ready_count <= 0:
                continue
            try:
                scan_task = {
                    "task_id": f"{session_dir.name}_consolidation_scan",
                    "task_type": "mst_consolidation_scan",
                    "session_id": session_dir.name,
                    "reason": "mst_consolidation_scan",
                    "limit_windows": limit_windows,
                }
                with WorkerTaskHeartbeat(
                    project_root,
                    "consolidation",
                    task=scan_task,
                    status="busy",
                    backend=backend,
                    model_name=model_name,
                    client_loaded=backend == "openai",
                    warmup_done=True,
                    queue_pending=_queue_pending,
                    extra={"session_id": session_dir.name, "update_em2mem": update_em2mem, "ready_window_count": ready_count},
                    interval_env="EM2MEM_MST_CONSOLIDATION_HEARTBEAT_SECONDS",
                ):
                    result = consolidate_short_term_to_em2mem(
                        session_id=session_dir.name,
                        sessions_root=sessions_root,
                        backend=backend,
                        update_em2mem=False,
                        force=False,
                        limit_windows=limit_windows,
                        verbose=verbose,
                    )
                generated_episode_ids = result.get("generated_episode_ids") if isinstance(result.get("generated_episode_ids"), list) else []
                if update_em2mem and generated_episode_ids:
                    memory_task = enqueue_memory_task(
                        project_root=project_root,
                        session_id=session_dir.name,
                        force=False,
                        skip_visual_embedding=os.getenv("EM2MEM_SKIP_VISUAL_EMBEDDING", "1").lower() in {"1", "true", "yes", "on"},
                        skip_semantic=os.getenv("EM2MEM_SKIP_SEMANTIC_MEMORY", "0").lower() in {"1", "true", "yes", "on"},
                        source="mst_episodic",
                        update_mode="incremental_append",
                        append_ready_episodes=True,
                        episode_ids=generated_episode_ids,
                        reason="mst_consolidation_scan",
                    )
                    result["memory_task_path"] = str(memory_task)
                    result["updated_em2mem"] = False
                    result["memory_update_queued"] = True
                    append_timeline_event(
                        session_dir,
                        "memory_append_queued",
                        metadata={"task_id": memory_task.stem, "episode_ids": generated_episode_ids},
                    )
                elif update_em2mem:
                    result["updated_em2mem"] = False
                    result["memory_update_queued"] = False
                    result["memory_skip_reason"] = "no newly generated episodes"
                results.append(result)
                refresh_session_pipeline_state(session_dir)
                append_timeline_event(
                    session_dir,
                    "consolidation_done",
                    metadata={
                        "generated_episode_ids": generated_episode_ids,
                        "memory_update_queued": result.get("memory_update_queued"),
                    },
                )
                if verbose:
                    print(json.dumps(result, ensure_ascii=False))
            except Exception as exc:
                results.append({"session_id": session_dir.name, "status": "error", "message": str(exc)})
                last_error = str(exc)
        write_json_atomic(
            runtime_path,
            {
                "status": "ok",
                "updated_at": utc_now_iso(),
                "backend": backend,
                "update_em2mem": update_em2mem,
                "last_results": results[-20:],
            },
        )
        write_worker_runtime(
            project_root,
            "consolidation",
            status="ready",
            backend=backend,
            model_name=os.getenv("EM2MEM_MST_EPISODIC_MODEL") or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL"),
            client_loaded=backend == "openai",
            warmup_done=True,
            queue_pending=len(list_queued_mst_consolidation_tasks(project_root)),
            last_task_id=last_task_id,
            last_error=last_error,
            extra={"update_em2mem": update_em2mem, "last_results": results[-5:]},
        )
        if once:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll refined-ready M_st windows and consolidate them into 30s episodic memory.")
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default=os.getenv("EM2MEM_MST_EPISODIC_BACKEND", "openai"), choices=["openai", "rule", "mock"])
    parser.add_argument("--update-em2mem", action="store_true", default=os.getenv("EM2MEM_MST_CONSOLIDATE_UPDATE_EM2MEM", "1").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("EM2MEM_MST_CONSOLIDATION_INTERVAL_SECONDS", "30")))
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_worker(
        sessions_root=Path(args.sessions_root),
        backend=args.backend,
        update_em2mem=bool(args.update_em2mem),
        interval_seconds=args.interval_seconds,
        limit_windows=args.limit_windows,
        once=args.once,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
