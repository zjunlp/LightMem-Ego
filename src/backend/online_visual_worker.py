from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import WorkerTaskHeartbeat, refresh_session_pipeline_state, write_worker_runtime
from online_preprocess.task_queue import (
    claim_visual_task,
    finish_visual_task,
    list_queued_visual_tasks,
)
from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime
from online_visual_embedding_builder import append_visual_embeddings, build_visual_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def _visual_model_name(backend: str | None) -> str:
    return "VLM2Vec-remote" if backend == "remote" else ("VLM2Vec-V2.0" if backend == "vlm2vec" else "mock")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _queue_pending(project_root: Path) -> int:
    return len(list_queued_visual_tasks(project_root))


def _warmup_visual_runtime(args: argparse.Namespace) -> dict[str, Any]:
    runtime = get_global_vlm2vec_runtime(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        normalize=True,
    )
    info = runtime.info()
    if args.warmup and runtime.backend == "vlm2vec":
        _ = runtime.model
    elif args.warmup and runtime.backend == "remote":
        info["remote_health"] = runtime.ping_remote()
    return info


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    last_task_id: str | None = None
    last_error: str | None = None

    try:
        runtime_info = _warmup_visual_runtime(args)
        write_worker_runtime(
            project_root,
            "visual",
            status="ready",
            backend=runtime_info.get("backend"),
            model_name=_visual_model_name(runtime_info.get("backend")),
            model_path=runtime_info.get("model_path"),
            device=runtime_info.get("device"),
            model_loaded=True,
            warmup_done=bool(args.warmup),
            queue_pending=_queue_pending(project_root),
            extra={"runtime": runtime_info},
        )
        print(
            "[visual_worker] ready "
            f"backend={runtime_info.get('backend')} model_path={runtime_info.get('model_path')}",
            flush=True,
        )
    except Exception as exc:
        write_worker_runtime(
            project_root,
            "visual",
            status="failed",
            backend=args.backend,
            model_name=_visual_model_name(args.backend),
            model_path=args.model_path,
            device=args.device,
            model_loaded=False,
            warmup_done=False,
            last_error=str(exc),
        )
        raise

    while True:
        queued = list_queued_visual_tasks(project_root)
        if not queued:
            write_worker_runtime(
                project_root,
                "visual",
                status="ready",
                backend=args.backend,
                model_name=_visual_model_name(args.backend),
                model_path=args.model_path,
                device=args.device,
                model_loaded=True,
                warmup_done=bool(args.warmup),
                queue_pending=0,
                last_task_id=last_task_id,
                last_error=last_error,
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued:
            claimed = claim_visual_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task.get("session_id") or "")
            task_id = str(task.get("task_id") or claimed_path.stem)
            last_task_id = task_id
            try:
                write_worker_runtime(
                    project_root,
                    "visual",
                    status="busy",
                    backend=args.backend,
                    model_name=_visual_model_name(args.backend),
                    model_path=args.model_path,
                    device=args.device,
                    model_loaded=True,
                    warmup_done=bool(args.warmup),
                    queue_pending=_queue_pending(project_root),
                    last_task_id=task_id,
                    last_error=None,
                )
                task_backend = str(task.get("backend") or args.backend)
                with WorkerTaskHeartbeat(
                    project_root,
                    "visual",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    backend=task_backend,
                    model_name=_visual_model_name(task_backend),
                    model_path=args.model_path,
                    device=args.device,
                    model_loaded=True,
                    warmup_done=bool(args.warmup),
                    queue_pending=lambda: _queue_pending(project_root),
                    last_error=None,
                    extra={"session_id": session_id},
                    interval_env="WORLDMM_VISUAL_HEARTBEAT_SECONDS",
                ):
                    if str(task.get("task_type") or "") == "visual_append":
                        visual_root = append_visual_embeddings(
                            session_id=session_id,
                            sessions_root=sessions_root,
                            backend=task_backend,
                            keyframe_paths=task.get("keyframe_paths") if isinstance(task.get("keyframe_paths"), list) else None,
                            episode_ids=task.get("episode_ids") if isinstance(task.get("episode_ids"), list) else None,
                            target_visual_version=task.get("target_visual_version"),
                            batch_size=args.batch_size,
                            normalize=True,
                            verbose=args.verbose,
                        )
                    else:
                        visual_root = build_visual_embeddings(
                            session_id=session_id,
                            sessions_root=sessions_root,
                            backend=task_backend,
                            force=bool(task.get("force", args.force)),
                            limit_items=task.get("limit_items") or args.limit_items,
                            batch_size=args.batch_size,
                            normalize=True,
                            dry_run=False,
                            verbose=args.verbose,
                        )
                refresh_session_pipeline_state(sessions_root / session_id)
                result = {"session_id": session_id, "visual_root": str(visual_root)}
                finish_visual_task(project_root, claimed_path, task, status="done", result=result)
                last_error = None
                print(f"[visual_worker] task={task_id} session={session_id} done", flush=True)
            except Exception as exc:
                last_error = str(exc)
                finish_visual_task(project_root, claimed_path, task, status="failed", error=str(exc))
                write_worker_runtime(
                    project_root,
                    "visual",
                    status="error",
                    backend=args.backend,
                    model_name=_visual_model_name(args.backend),
                    model_path=args.model_path,
                    device=args.device,
                    model_loaded=True,
                    warmup_done=bool(args.warmup),
                    queue_pending=_queue_pending(project_root),
                    last_task_id=task_id,
                    last_error=str(exc),
                )
                print(f"[visual_worker] task={task_id} session={session_id} failed: {exc}", flush=True)

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent VLM2Vec visual embedding worker.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default=os.getenv("WORLDMM_VISUAL_BACKEND", "vlm2vec"), choices=["vlm2vec", "mock", "remote"])
    parser.add_argument("--model-path", default=os.getenv("WORLDMM_VLM2VEC_MODEL_PATH"))
    parser.add_argument("--device", default=os.getenv("WORLDMM_VLM2VEC_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("WORLDMM_VLM2VEC_DTYPE", "float16"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("WORLDMM_VISUAL_BATCH_SIZE", "8")))
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("WORLDMM_VISUAL_WORKER_POLL_SECONDS", "2")))
    parser.add_argument("--force", action="store_true", default=_env_bool("WORLDMM_FORCE_VISUAL_EMBEDDING", False))
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=_env_bool("WORLDMM_VISUAL_WARMUP", True))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
