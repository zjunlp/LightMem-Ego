from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from online_preprocess.evidence_builder import build_session_evidence
from online_preprocess.io_utils import write_status
from online_preprocess.task_queue import claim_evidence_task, enqueue_memory_task, finish_evidence_task, list_queued_evidence_tasks
from online_pipeline.runtime_state import WorkerTaskHeartbeat, refresh_session_pipeline_state, write_worker_runtime


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    print(
        "Evidence worker started:",
        f"backend={args.backend}",
        f"model={args.model}",
        f"max_keyframes={args.max_keyframes}",
        flush=True,
    )
    write_worker_runtime(
        project_root,
        "evidence",
        status="ready",
        backend=args.backend,
        model_name=args.model,
        client_loaded=args.backend == "openai",
        warmup_done=True,
        queue_pending=len(list_queued_evidence_tasks(project_root)),
    )

    while True:
        queued_tasks = list_queued_evidence_tasks(project_root)
        if not queued_tasks:
            write_worker_runtime(
                project_root,
                "evidence",
                status="ready",
                backend=args.backend,
                model_name=args.model,
                client_loaded=args.backend == "openai",
                warmup_done=True,
                queue_pending=0,
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued_tasks:
            claimed = claim_evidence_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task["session_id"])
            session_dir = sessions_root / session_id
            backend = task.get("backend") or args.backend
            limit_segments = task.get("limit_segments")

            try:
                write_worker_runtime(
                    project_root,
                    "evidence",
                    status="busy",
                    backend=backend,
                    model_name=args.model,
                    client_loaded=backend == "openai",
                    warmup_done=True,
                    queue_pending=len(list_queued_evidence_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    extra={"session_id": session_id},
                )
                write_status(
                    session_dir=session_dir,
                    session_id=session_id,
                    status="processing",
                    stage="evidence_building",
                    progress=70,
                    error=None,
                )
                with WorkerTaskHeartbeat(
                    project_root,
                    "evidence",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    backend=backend,
                    model_name=args.model,
                    client_loaded=backend == "openai",
                    warmup_done=True,
                    queue_pending=lambda: len(list_queued_evidence_tasks(project_root)),
                    extra={"session_id": session_id},
                    interval_env="WORLDMM_EVIDENCE_HEARTBEAT_SECONDS",
                ):
                    build_session_evidence(
                        session_id=session_id,
                        sessions_root=sessions_root,
                        backend=backend,
                        model=args.model,
                        max_keyframes=args.max_keyframes,
                        force=bool(task.get("force", args.force)),
                        limit_segments=int(limit_segments) if limit_segments is not None else args.limit_segments,
                        dry_run=False,
                    )
                if _env_bool("WORLDMM_AUTO_MEMORY", False):
                    enqueue_memory_task(
                        project_root=project_root,
                        session_id=session_id,
                        force=bool(task.get("force", args.force)),
                        skip_visual_embedding=_env_bool("WORLDMM_SKIP_VISUAL_EMBEDDING", True),
                        skip_semantic=_env_bool("WORLDMM_SKIP_SEMANTIC_MEMORY", False),
                        limit_segments=int(limit_segments) if limit_segments is not None else args.limit_segments,
                    )
                    write_status(
                        session_dir=session_dir,
                        session_id=session_id,
                        status="processing",
                        stage="memory_queued",
                        progress=90,
                        error=None,
                    )
                finish_evidence_task(project_root, claimed_path, task, status="done")
                refresh_session_pipeline_state(session_dir)
            except Exception as exc:
                write_status(
                    session_dir=session_dir,
                    session_id=session_id,
                    status="failed",
                    stage="evidence_worker",
                    progress=100,
                    error=str(exc),
                )
                finish_evidence_task(project_root, claimed_path, task, status="failed", error=str(exc))
                write_worker_runtime(
                    project_root,
                    "evidence",
                    status="error",
                    backend=backend,
                    model_name=args.model,
                    client_loaded=backend == "openai",
                    warmup_done=True,
                    queue_pending=len(list_queued_evidence_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    last_error=str(exc),
                    extra={"session_id": session_id},
                )

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent worker for online evidence tasks.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default=os.getenv("WORLDMM_EVIDENCE_CAPTION_BACKEND", "mock"), choices=["mock", "openai", "local"])
    parser.add_argument("--model", default=os.getenv("WORLDMM_VLM_MODEL"))
    parser.add_argument("--max-keyframes", type=int, default=int(os.getenv("WORLDMM_VLM_MAX_KEYFRAMES", "8")))
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", default=_env_bool("WORLDMM_FORCE_EVIDENCE", False))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
