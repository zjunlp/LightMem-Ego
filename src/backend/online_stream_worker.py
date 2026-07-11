from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import WorkerTaskHeartbeat, get_pipeline_mode, refresh_session_pipeline_state, write_worker_runtime
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import utc_now_iso, write_status
from online_preprocess.task_queue import (
    claim_stream_chunk_task,
    enqueue_stream_asr_task,
    finish_stream_chunk_task,
    list_queued_stream_chunk_tasks,
)
from online_short_term.micro_event_builder import MicroEventBuilder
from online_short_term.stream_chunk_manager import StreamChunkManager


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def _queue_pending(project_root: Path) -> int:
    return len(list_queued_stream_chunk_tasks(project_root))


def _runtime_extra(result: dict[str, Any] | None = None) -> dict[str, Any]:
    result = result or {}
    return {
        "last_session_id": result.get("session_id"),
        "last_chunk_id": result.get("chunk_id"),
        "last_chunk_index": result.get("chunk_index"),
        "last_proc_index": result.get("proc_index", result.get("chunk_index")),
        "last_upload_chunk_index": result.get("upload_chunk_index"),
        "candidate_frame_count": int(result.get("candidate_frame_count") or 0),
        "diff_record_count": int(result.get("diff_record_count") or 0),
        "closed_event_count": int(result.get("closed_event_count") or 0),
        "has_open_event": bool(result.get("has_open_event", False)),
        "reused_frame_extraction": bool(result.get("reused_frame_extraction", True)),
    }


def _process_chunk_task(
    *,
    project_root: Path,
    sessions_root: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    session_id = str(task.get("session_id") or "")
    session_dir = sessions_root / session_id
    manager = StreamChunkManager(session_dir)
    chunk_index = int(task.get("chunk_index"))
    task_chunk_id = str(task.get("chunk_id") or "")
    task_chunk_path = str(task.get("chunk_path") or "")
    if task_chunk_id.startswith("upload_") or "/upload_chunks/" in task_chunk_path or task_chunk_path.startswith("stream/upload_chunks/"):
        upload_task = dict(task)
        upload_task["task_type"] = "stream_upload_chunk"
        upload_task["upload_chunk_id"] = task_chunk_id or task.get("upload_chunk_id")
        upload_task["upload_chunk_index"] = task.get("upload_chunk_index", chunk_index)
        upload_task["upload_chunk_path"] = task.get("upload_chunk_path") or task_chunk_path
        return _process_upload_task(project_root=project_root, sessions_root=sessions_root, task=upload_task)
    chunk = manager.get_chunk(chunk_index)
    if not chunk:
        chunk = {
            "chunk_id": task.get("chunk_id"),
            "chunk_index": chunk_index,
            "path": task.get("chunk_path"),
            "start_time": task.get("start_time"),
            "end_time": task.get("end_time"),
            "duration": task.get("duration"),
        }
    chunk_id = str(chunk.get("chunk_id") or "")
    if chunk_id.startswith("upload_") or str(chunk.get("path") or "").startswith("stream/upload_chunks/"):
        upload_task = dict(task)
        upload_task["task_type"] = "stream_upload_chunk"
        upload_task["upload_chunk_id"] = chunk_id or task.get("upload_chunk_id")
        upload_task["upload_chunk_index"] = chunk.get("upload_chunk_index", chunk.get("chunk_index", chunk_index))
        upload_task["upload_chunk_path"] = chunk.get("path") or task_chunk_path
        return _process_upload_task(project_root=project_root, sessions_root=sessions_root, task=upload_task)
    if not chunk_id.startswith("proc_") and "processing_chunk_id" not in chunk:
        raise RuntimeError(f"stream_chunk task must reference a processing chunk, got chunk_id={chunk_id!r}")
    manager.update_chunk_status(chunk_index, status="processing", task_id=str(task.get("task_id") or ""))
    append_timeline_event(
        session_dir,
        "stream_chunk_started",
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        metadata={"task_id": task.get("task_id"), "proc_index": chunk.get("proc_index", chunk_index)},
    )
    builder = MicroEventBuilder(session_dir)
    result = builder.process_stream_chunk(chunk, project_root=project_root, enqueue_refine=True)
    processed_at = utc_now_iso()
    closed_count = int(result.get("closed_event_count") or 0)
    manager.update_chunk_status(
        chunk_index,
        status="processed",
        processed_at=processed_at,
        extra={"mst_event_closed_at": processed_at if closed_count > 0 else None},
    )
    append_timeline_event(
        session_dir,
        "mcur_updated",
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        metadata={
            "candidate_frame_count": result.get("candidate_frame_count"),
            "diff_record_count": result.get("diff_record_count"),
        },
    )
    if closed_count > 0:
        append_timeline_event(
            session_dir,
            "mst_event_closed",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            metadata={"closed_event_count": closed_count, "closed_event_ids": result.get("closed_event_ids", [])},
        )
    manager.append_processed_chunk(
        {
            **chunk,
            "event_count": int(result.get("closed_event_count") or 0),
            "candidate_frame_count": int(result.get("candidate_frame_count") or 0),
            "diff_record_count": int(result.get("diff_record_count") or 0),
        }
    )
    manager.enqueue_ready_chunk(project_root)
    manager.enqueue_stream_end_if_ready(project_root)
    refresh_session_pipeline_state(session_dir)
    return result


def _process_upload_task(
    *,
    project_root: Path,
    sessions_root: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    session_id = str(task.get("session_id") or "")
    session_dir = sessions_root / session_id
    manager = StreamChunkManager(session_dir)
    upload_index = int(task.get("upload_chunk_index", task.get("chunk_index", 0)))
    state = manager.load_stream_state(default={})
    processing_seconds = float(
        state.get("processing_chunk_seconds")
        or state.get("chunk_duration")
        or 5.0
    )
    manager.update_chunk_status(upload_index, status="processing", task_id=str(task.get("task_id") or ""))
    state = manager.materialize_ready_upload_chunks(processing_chunk_seconds=processing_seconds)
    task_path = manager.enqueue_ready_chunk(project_root)
    manager.enqueue_ready_upload_chunk(project_root)
    manager.enqueue_stream_end_if_ready(project_root)
    upload = next(
        (
            dict(item)
            for item in state.get("upload_chunks", state.get("received_chunks", [])) or []
            if isinstance(item, dict)
            and int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == upload_index
        ),
        {},
    )
    processing_chunks = [dict(item) for item in upload.get("processing_chunks", []) if isinstance(item, dict)]
    asr_task_path = None
    stream_asr_enabled = os.getenv("WORLDMM_STREAM_ASR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
    if processing_chunks and stream_asr_enabled:
        asr_task_path = enqueue_stream_asr_task(
            project_root=project_root,
            session_id=session_id,
            stream_id=str(state.get("stream_id") or upload.get("stream_id") or ""),
            upload_chunk_id=str(upload.get("upload_chunk_id") or task.get("upload_chunk_id") or task.get("chunk_id") or ""),
            upload_chunk_index=upload_index,
            upload_chunk_path=str(upload.get("path") or task.get("upload_chunk_path") or task.get("chunk_path") or ""),
            processing_chunks=processing_chunks,
            global_start_time=float(upload.get("stream_start_time", processing_chunks[0].get("start_time", 0.0)) or 0.0),
            global_end_time=float(upload.get("stream_end_time", processing_chunks[-1].get("end_time", 0.0)) or 0.0),
            asr_backend=os.getenv("WORLDMM_STREAM_ASR_BACKEND", "whisperx"),
            reason="stream_upload_chunk",
        )
        append_timeline_event(
            session_dir,
            "asr_queued",
            chunk_index=upload_index,
            chunk_id=str(upload.get("upload_chunk_id") or task.get("upload_chunk_id") or task.get("chunk_id") or ""),
            metadata={"task_id": asr_task_path.stem, "processing_chunk_count": len(processing_chunks)},
        )
        state = manager.load_stream_state(default={})
        upload_chunks = []
        for item in state.get("upload_chunks", state.get("received_chunks", [])) or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == upload_index:
                item["asr_status"] = "queued"
                item["asr_task_id"] = asr_task_path.stem
                item["asr_task_path"] = str(asr_task_path)
                item["asr_queued_at"] = utc_now_iso()
            upload_chunks.append(item)
        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        manager.save_stream_state(state)
    refresh_session_pipeline_state(session_dir)
    return {
        "session_id": session_id,
        "chunk_id": str(upload.get("upload_chunk_id") or task.get("upload_chunk_id") or task.get("chunk_id") or ""),
        "chunk_index": upload_index,
        "upload_chunk_index": upload_index,
        "actual_duration": upload.get("actual_duration"),
        "processing_chunk_count": len(processing_chunks),
        "generated_processing_chunk_ids": [item.get("chunk_id") for item in processing_chunks],
        "next_processing_task_path": str(task_path) if task_path else None,
        "stream_asr_task_path": str(asr_task_path) if asr_task_path else None,
        "candidate_frame_count": 0,
        "diff_record_count": 0,
        "closed_event_count": 0,
        "has_open_event": False,
        "reused_frame_extraction": True,
    }


def _process_end_task(
    *,
    project_root: Path,
    sessions_root: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    session_id = str(task.get("session_id") or "")
    session_dir = sessions_root / session_id
    manager = StreamChunkManager(session_dir)
    if bool(task.get("close_open_event", True)):
        builder = MicroEventBuilder(session_dir)
        result = builder.close_stream_open_event(
            project_root=project_root,
            enqueue_refine=True,
            reason="stream_end",
        )
    else:
        event_state = manager.load_event_state()
        result = {
            "session_id": session_id,
            "closed_event_count": 0,
            "closed_event_ids": [],
            "has_open_event": bool(event_state.get("open_event")) if isinstance(event_state, dict) else False,
            "boundary_reason": None,
        }
    state = manager.mark_stream_ended()
    append_timeline_event(
        session_dir,
        "stream_ended",
        chunk_index=task.get("final_chunk_index"),
        chunk_id="stream_end",
        metadata={"closed_event_count": result.get("closed_event_count", 0), "closed_event_ids": result.get("closed_event_ids", [])},
    )
    write_status(
        session_dir=session_dir,
        session_id=session_id,
        status="stream_ended",
        stage="stream_ended",
        progress=100,
        error=None,
    )
    refresh_session_pipeline_state(session_dir)
    return {
        **result,
        "chunk_id": "stream_end",
        "chunk_index": task.get("final_chunk_index"),
        "stream_status": state.get("status"),
        "reused_frame_extraction": True,
    }


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    last_task_id: str | None = None
    last_error: str | None = None
    last_result: dict[str, Any] = {}

    write_worker_runtime(
        project_root,
        "stream",
        status="ready",
        model_name=None,
        client_loaded=False,
        model_loaded=False,
        warmup_done=True,
        queue_pending=_queue_pending(project_root),
        extra={"pipeline_mode": get_pipeline_mode(), **_runtime_extra(last_result)},
    )
    print("[stream_worker] ready", flush=True)

    while True:
        queued = list_queued_stream_chunk_tasks(project_root)
        if not queued:
            write_worker_runtime(
                project_root,
                "stream",
                status="ready",
                model_name=None,
                client_loaded=False,
                model_loaded=False,
                warmup_done=True,
                queue_pending=0,
                last_task_id=last_task_id,
                last_error=last_error,
                extra={"pipeline_mode": get_pipeline_mode(), **_runtime_extra(last_result)},
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued:
            claimed = claim_stream_chunk_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            task_id = str(task.get("task_id") or claimed_path.stem)
            session_id = str(task.get("session_id") or "")
            last_task_id = task_id
            try:
                write_worker_runtime(
                    project_root,
                    "stream",
                    status="busy",
                    warmup_done=True,
                    queue_pending=_queue_pending(project_root),
                    last_task_id=task_id,
                    last_error=None,
                    extra={"pipeline_mode": get_pipeline_mode(), "last_session_id": session_id},
                )
                task_type = str(task.get("task_type") or "")
                with WorkerTaskHeartbeat(
                    project_root,
                    "stream",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    warmup_done=True,
                    queue_pending=lambda: _queue_pending(project_root),
                    last_error=None,
                    extra_fn=lambda session_id=session_id: {
                        "pipeline_mode": get_pipeline_mode(),
                        "last_session_id": session_id,
                        **_runtime_extra(last_result),
                    },
                    interval_env="WORLDMM_STREAM_HEARTBEAT_SECONDS",
                ):
                    if task_type == "stream_end":
                        result = _process_end_task(project_root=project_root, sessions_root=sessions_root, task=task)
                    elif task_type == "stream_upload_chunk":
                        result = _process_upload_task(project_root=project_root, sessions_root=sessions_root, task=task)
                    else:
                        result = _process_chunk_task(project_root=project_root, sessions_root=sessions_root, task=task)
                last_result = {**result, "session_id": session_id}
                finish_stream_chunk_task(project_root, claimed_path, task, status="done", result=result)
                last_error = None
                print(f"[stream_worker] task={task_id} session={session_id} done", flush=True)
            except Exception as exc:
                last_error = str(exc)
                try:
                    if str(task.get("task_type") or "") == "stream_chunk":
                        StreamChunkManager(sessions_root / session_id).update_chunk_status(
                            int(task.get("chunk_index")),
                            status="failed",
                            error=str(exc),
                        )
                except Exception:
                    pass
                finish_stream_chunk_task(project_root, claimed_path, task, status="failed", error=str(exc))
                print(f"[stream_worker] task={task_id} session={session_id} failed: {exc}", flush=True)

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent worker for real chunk-based stream ingestion.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
