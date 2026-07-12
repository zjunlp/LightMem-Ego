from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from online_preprocess.asr_whisperx import WhisperXRuntime
from online_preprocess.io_utils import write_status
from online_preprocess.task_queue import (
    claim_stream_asr_task,
    claim_task,
    enqueue_evidence_task,
    finish_stream_asr_task,
    finish_task,
    list_queued_stream_asr_tasks,
    list_queued_tasks,
)
from online_processor import process_session
from online_pipeline.runtime_state import WorkerTaskHeartbeat, get_pipeline_mode, refresh_session_pipeline_state, write_worker_runtime
from online_pipeline.stream_timeline import append_timeline_event
from online_streaming.stream_asr_processor import mark_stream_asr_failed, process_stream_asr_task


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"
DEFAULT_WHISPERX_MODEL_DIR = PROJECT_ROOT / "models" / "whisperx"
DEFAULT_WHISPERX_ALIGN_MODEL_DIR = DEFAULT_WHISPERX_MODEL_DIR / "alignment"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _auto_legacy_evidence_enabled() -> bool:
    mode = get_pipeline_mode()
    if mode == "legacy":
        return _env_bool("EM2MEM_AUTO_EVIDENCE", True)
    if mode == "hybrid":
        return _env_bool("EM2MEM_ENABLE_LEGACY_EVIDENCE_WORKER", True) and _env_bool("EM2MEM_AUTO_EVIDENCE", True)
    return _env_bool("EM2MEM_ENABLE_LEGACY_EVIDENCE_WORKER", False) and _env_bool("EM2MEM_AUTO_EVIDENCE", False)


def _split_languages(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()

    stream_asr_processed_count = 0
    last_stream_asr_task_id: str | None = None
    last_stream_asr_session_id: str | None = None
    last_stream_asr_error: str | None = None

    runtime = WhisperXRuntime(
        model_name=args.whisperx_model,
        device=args.device,
        compute_type=args.compute_type,
        model_dir=args.model_dir,
        align_model_dir=args.align_model_dir,
        preload_align_languages=_split_languages(args.preload_align_languages),
    )
    runtime.load()

    def _queue_pending() -> int:
        return len(list_queued_tasks(project_root)) + len(list_queued_stream_asr_tasks(project_root))

    def _runtime_extra(session_id: str | None = None) -> dict:
        extra = {
            "whisperx_model": args.whisperx_model,
            "compute_type": runtime.compute_type,
            "preload_align_languages": args.preload_align_languages,
            "pipeline_mode": get_pipeline_mode(),
            "auto_legacy_evidence": _auto_legacy_evidence_enabled(),
            "stream_asr_enabled": _env_bool("EM2MEM_STREAM_ASR_ENABLED", True),
            "stream_asr_queue_pending": len(list_queued_stream_asr_tasks(project_root)),
            "stream_asr_processed_count": stream_asr_processed_count,
            "last_stream_asr_task_id": last_stream_asr_task_id,
            "last_stream_asr_session_id": last_stream_asr_session_id,
            "last_stream_asr_error": last_stream_asr_error,
        }
        if session_id:
            extra["session_id"] = session_id
        return extra

    write_worker_runtime(
        project_root,
        "preprocess",
        status="ready",
        model_name="whisperx",
        model_path=str(args.model_dir),
        device=runtime.device,
        model_loaded=True,
        warmup_done=True,
        queue_pending=len(list_queued_tasks(project_root)) + len(list_queued_stream_asr_tasks(project_root)),
        extra={
            "whisperx_model": args.whisperx_model,
            "compute_type": runtime.compute_type,
            "preload_align_languages": args.preload_align_languages,
            "pipeline_mode": get_pipeline_mode(),
            "auto_legacy_evidence": _auto_legacy_evidence_enabled(),
            "stream_asr_enabled": _env_bool("EM2MEM_STREAM_ASR_ENABLED", True),
            "stream_asr_queue_pending": len(list_queued_stream_asr_tasks(project_root)),
            "stream_asr_processed_count": stream_asr_processed_count,
            "last_stream_asr_task_id": last_stream_asr_task_id,
            "last_stream_asr_session_id": last_stream_asr_session_id,
            "last_stream_asr_error": last_stream_asr_error,
        },
    )
    print(
        "WhisperX runtime loaded:",
        f"model={args.whisperx_model}",
        f"device={runtime.device}",
        f"compute_type={runtime.compute_type}",
        f"align={args.preload_align_languages}",
        flush=True,
    )

    while True:
        stream_asr_tasks = list_queued_stream_asr_tasks(project_root) if _env_bool("EM2MEM_STREAM_ASR_ENABLED", True) else []
        if stream_asr_tasks:
            for task_path in stream_asr_tasks:
                claimed = claim_stream_asr_task(project_root, task_path)
                if claimed is None:
                    continue
                claimed_path, task = claimed
                session_id = str(task.get("session_id") or "")
                task_id = str(task.get("task_id") or claimed_path.stem)
                last_stream_asr_task_id = task_id
                last_stream_asr_session_id = session_id
                try:
                    append_timeline_event(
                        sessions_root / session_id,
                        "asr_started",
                        chunk_index=int(task.get("upload_chunk_index", -1)),
                        chunk_id=str(task.get("upload_chunk_id") or ""),
                        metadata={"task_id": task_id, "backend": task.get("asr_backend")},
                    )
                    write_worker_runtime(
                        project_root,
                        "preprocess",
                        status="busy_stream_asr",
                        model_name="whisperx",
                        model_path=str(args.model_dir),
                        device=runtime.device,
                        model_loaded=True,
                        warmup_done=True,
                        queue_pending=len(list_queued_tasks(project_root)) + len(list_queued_stream_asr_tasks(project_root)),
                        last_task_id=task_id,
                        extra={
                            "session_id": session_id,
                            "pipeline_mode": get_pipeline_mode(),
                            "auto_legacy_evidence": _auto_legacy_evidence_enabled(),
                            "stream_asr_enabled": True,
                            "stream_asr_queue_pending": len(list_queued_stream_asr_tasks(project_root)),
                            "stream_asr_processed_count": stream_asr_processed_count,
                            "last_stream_asr_task_id": last_stream_asr_task_id,
                            "last_stream_asr_session_id": last_stream_asr_session_id,
                            "last_stream_asr_error": None,
                        },
                    )
                    with WorkerTaskHeartbeat(
                        project_root,
                        "preprocess",
                        task=task,
                        claimed_path=claimed_path,
                        status="busy_stream_asr",
                        model_name="whisperx",
                        model_path=str(args.model_dir),
                        device=runtime.device,
                        model_loaded=True,
                        warmup_done=True,
                        queue_pending=_queue_pending,
                        extra_fn=lambda session_id=session_id: _runtime_extra(session_id),
                        interval_env="EM2MEM_PREPROCESS_HEARTBEAT_SECONDS",
                    ):
                        result = process_stream_asr_task(
                            project_root=project_root,
                            sessions_root=sessions_root,
                            task=task,
                            asr_runtime=runtime,
                            whisperx_model=args.whisperx_model,
                            device=args.device,
                            compute_type=args.compute_type,
                            language=args.language,
                            model_dir=args.model_dir,
                            align_model_dir=args.align_model_dir,
                            force=bool(task.get("force", False)),
                        )
                    finish_stream_asr_task(project_root, claimed_path, task, status="done", result=result)
                    append_timeline_event(
                        sessions_root / session_id,
                        "asr_done",
                        chunk_index=int(task.get("upload_chunk_index", -1)),
                        chunk_id=str(task.get("upload_chunk_id") or ""),
                        metadata={"task_id": task_id, "segment_count": result.get("segment_count"), "no_audio": result.get("no_audio")},
                    )
                    append_timeline_event(
                        sessions_root / session_id,
                        "transcript_backfilled",
                        chunk_index=int(task.get("upload_chunk_index", -1)),
                        chunk_id=str(task.get("upload_chunk_id") or ""),
                        metadata=result.get("backfill") if isinstance(result.get("backfill"), dict) else {},
                    )
                    stream_asr_processed_count += 1
                    last_stream_asr_error = None
                    refresh_session_pipeline_state(sessions_root / session_id)
                except Exception as exc:
                    last_stream_asr_error = str(exc)
                    mark_stream_asr_failed(sessions_root, task, str(exc))
                    append_timeline_event(
                        sessions_root / session_id,
                        "error",
                        chunk_index=int(task.get("upload_chunk_index", -1)),
                        chunk_id=str(task.get("upload_chunk_id") or ""),
                        metadata={"stage": "stream_asr", "task_id": task_id, "error": str(exc)},
                    )
                    finish_stream_asr_task(project_root, claimed_path, task, status="failed", error=str(exc))
                    write_worker_runtime(
                        project_root,
                        "preprocess",
                        status="error",
                        model_name="whisperx",
                        model_path=str(args.model_dir),
                        device=runtime.device,
                        model_loaded=True,
                        warmup_done=True,
                        queue_pending=len(list_queued_tasks(project_root)) + len(list_queued_stream_asr_tasks(project_root)),
                        last_task_id=task_id,
                        last_error=str(exc),
                        extra={
                            "session_id": session_id,
                            "stream_asr_enabled": True,
                            "last_stream_asr_task_id": last_stream_asr_task_id,
                            "last_stream_asr_session_id": last_stream_asr_session_id,
                            "last_stream_asr_error": last_stream_asr_error,
                        },
                    )
                if args.once:
                    return
            continue

        queued_tasks = list_queued_tasks(project_root)
        if not queued_tasks:
            write_worker_runtime(
                project_root,
                "preprocess",
                status="ready",
                model_name="whisperx",
                model_path=str(args.model_dir),
                device=runtime.device,
                model_loaded=True,
                warmup_done=True,
                queue_pending=len(list_queued_stream_asr_tasks(project_root)),
                extra={
                    "whisperx_model": args.whisperx_model,
                    "compute_type": runtime.compute_type,
                    "pipeline_mode": get_pipeline_mode(),
                    "auto_legacy_evidence": _auto_legacy_evidence_enabled(),
                    "stream_asr_enabled": _env_bool("EM2MEM_STREAM_ASR_ENABLED", True),
                    "stream_asr_queue_pending": len(list_queued_stream_asr_tasks(project_root)),
                    "stream_asr_processed_count": stream_asr_processed_count,
                    "last_stream_asr_task_id": last_stream_asr_task_id,
                    "last_stream_asr_session_id": last_stream_asr_session_id,
                    "last_stream_asr_error": last_stream_asr_error,
                },
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued_tasks:
            claimed = claim_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task["session_id"])
            session_dir = sessions_root / session_id

            try:
                write_worker_runtime(
                    project_root,
                    "preprocess",
                    status="busy",
                    model_name="whisperx",
                    model_path=str(args.model_dir),
                    device=runtime.device,
                    model_loaded=True,
                    warmup_done=True,
                    queue_pending=len(list_queued_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    extra={
                        "session_id": session_id,
                        "pipeline_mode": get_pipeline_mode(),
                        "auto_legacy_evidence": _auto_legacy_evidence_enabled(),
                        "stream_asr_enabled": _env_bool("EM2MEM_STREAM_ASR_ENABLED", True),
                        "stream_asr_queue_pending": len(list_queued_stream_asr_tasks(project_root)),
                        "stream_asr_processed_count": stream_asr_processed_count,
                        "last_stream_asr_task_id": last_stream_asr_task_id,
                        "last_stream_asr_session_id": last_stream_asr_session_id,
                        "last_stream_asr_error": last_stream_asr_error,
                    },
                )
                with WorkerTaskHeartbeat(
                    project_root,
                    "preprocess",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    model_name="whisperx",
                    model_path=str(args.model_dir),
                    device=runtime.device,
                    model_loaded=True,
                    warmup_done=True,
                    queue_pending=_queue_pending,
                    extra_fn=lambda session_id=session_id: _runtime_extra(session_id),
                    interval_env="EM2MEM_PREPROCESS_HEARTBEAT_SECONDS",
                ):
                    process_session(
                        session_id=session_id,
                        sessions_root=sessions_root,
                        whisperx_model=args.whisperx_model,
                        device=args.device,
                        compute_type=args.compute_type,
                        language=args.language,
                        model_dir=args.model_dir,
                        align_model_dir=args.align_model_dir,
                        skip_asr=args.skip_asr,
                        force=bool(task.get("force", args.force)),
                        asr_runtime=runtime,
                    )
                if _auto_legacy_evidence_enabled():
                    enqueue_evidence_task(
                        project_root=project_root,
                        session_id=session_id,
                        force=bool(task.get("force", args.force)),
                        backend=os.getenv("EM2MEM_EVIDENCE_CAPTION_BACKEND"),
                        pipeline_mode=get_pipeline_mode(),
                        role="legacy_optional" if get_pipeline_mode() != "legacy" else "main",
                    )
                    write_status(
                        session_dir=session_dir,
                        session_id=session_id,
                        status="processing",
                        stage="evidence_queued",
                        progress=60,
                        error=None,
                    )
                finish_task(project_root, claimed_path, task, status="done")
                refresh_session_pipeline_state(session_dir)
            except Exception as exc:
                write_status(
                    session_dir=session_dir,
                    session_id=session_id,
                    status="failed",
                    stage="worker_preprocess",
                    progress=100,
                    error=str(exc),
                )
                finish_task(project_root, claimed_path, task, status="failed", error=str(exc))
                write_worker_runtime(
                    project_root,
                    "preprocess",
                    status="error",
                    model_name="whisperx",
                    model_path=str(args.model_dir),
                    device=runtime.device,
                    model_loaded=True,
                    warmup_done=True,
                    queue_pending=len(list_queued_tasks(project_root)),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    last_error=str(exc),
                    extra={"session_id": session_id},
                )

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent compute-node worker for online preprocessing tasks.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--whisperx-model", default=os.getenv("EM2MEM_WHISPERX_MODEL", "medium"))
    parser.add_argument("--device", default=os.getenv("EM2MEM_WHISPERX_DEVICE", "cuda"))
    parser.add_argument("--compute-type", default=os.getenv("EM2MEM_WHISPERX_COMPUTE_TYPE", "float16"))
    parser.add_argument("--language", default=os.getenv("EM2MEM_WHISPERX_LANGUAGE") or None)
    parser.add_argument("--model-dir", default=os.getenv("EM2MEM_WHISPERX_MODEL_DIR", str(DEFAULT_WHISPERX_MODEL_DIR)))
    parser.add_argument(
        "--align-model-dir",
        default=os.getenv("EM2MEM_WHISPERX_ALIGN_MODEL_DIR", str(DEFAULT_WHISPERX_ALIGN_MODEL_DIR)),
    )
    parser.add_argument("--preload-align-languages", default=os.getenv("EM2MEM_WHISPERX_ALIGN_LANGS", "zh,en"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--skip-asr", action="store_true", default=_env_bool("EM2MEM_SKIP_ASR", False))
    parser.add_argument("--force", action="store_true", default=_env_bool("EM2MEM_FORCE_PREPROCESS", False))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_worker(args)


if __name__ == "__main__":
    main()
