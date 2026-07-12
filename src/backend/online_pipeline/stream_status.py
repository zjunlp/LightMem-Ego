from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from online_pipeline.backpressure import compute_backpressure
from online_pipeline.audio_stream import public_audio_stream_status_block
from online_pipeline.frame_stream import public_frame_stream_status_block
from online_pipeline.live_ingest import public_live_ingest_status_block
from online_pipeline.live_rtmp import public_live_status_block, public_webrtc_status_block
from online_pipeline.runtime_state import queue_counts
from online_pipeline.rokid_ingest import public_rokid_status_block
from online_preprocess.io_utils import read_json
from online_preprocess.task_queue import ensure_queue_dirs
from online_short_term.frame_stream_event_builder import public_frame_mst_status_block
from online_short_term.stream_chunk_manager import StreamChunkManager


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _count_session_tasks(project_root: Path, session_id: str) -> dict[str, int]:
    dirs = ensure_queue_dirs(Path(project_root))
    counts: dict[str, int] = {}
    for key, path in dirs.items():
        count = 0
        for task_path in path.glob(f"{session_id}_*.json"):
            payload = read_json(task_path, default={})
            if isinstance(payload, dict) and str(payload.get("session_id") or "") == session_id:
                count += 1
        counts[key] = count
    return counts


def _count_session_audio_asr_tasks(project_root: Path, session_id: str) -> dict[str, int]:
    dirs = ensure_queue_dirs(Path(project_root))
    mapping = {
        "queued": "stream_asr_queued",
        "in_progress": "stream_asr_in_progress",
        "done": "stream_asr_done",
        "failed": "stream_asr_failed",
    }
    counts = {key: 0 for key in mapping}
    for public_key, queue_key in mapping.items():
        path = dirs.get(queue_key)
        if path is None:
            continue
        for task_path in path.glob(f"{session_id}_*.json"):
            payload = read_json(task_path, default={})
            if isinstance(payload, dict) and payload.get("source") == "audio_chunk_window":
                counts[public_key] += 1
    return counts


def _reconcile_audio_stream_asr_status(audio_stream: dict[str, Any], task_counts: dict[str, int]) -> dict[str, Any]:
    if not audio_stream.get("enabled"):
        return audio_stream
    updated = dict(audio_stream)
    queued = _as_int(task_counts.get("queued"), 0)
    in_progress = _as_int(task_counts.get("in_progress"), 0)
    done = _as_int(task_counts.get("done"), 0)
    failed = _as_int(task_counts.get("failed"), 0)
    if not any((queued, in_progress, done, failed)):
        return updated
    pending = queued + in_progress
    updated["queued_window_count"] = max(_as_int(updated.get("queued_window_count"), 0), queued + in_progress + done + failed)
    updated["completed_window_count"] = max(_as_int(updated.get("completed_window_count"), 0), done)
    updated["failed_window_count"] = max(_as_int(updated.get("failed_window_count"), 0), failed)
    updated["pending_window_count"] = pending
    if pending > 0:
        updated["asr_status"] = "running" if in_progress else "queued"
    elif _as_int(updated.get("completed_window_count"), 0) > 0:
        updated["asr_status"] = "ready"
        updated["asr_ready"] = True
    elif _as_int(updated.get("failed_window_count"), 0) > 0:
        updated["asr_status"] = "failed"
        updated["asr_ready"] = False
    return updated


def build_stream_context(session_dir: Path) -> dict[str, Any] | None:
    session_dir = Path(session_dir)
    state_path = session_dir / "stream" / "stream_state.json"
    if not state_path.exists():
        return None
    stream_state = read_json(state_path, default={})
    pipeline_state = read_json(session_dir / "pipeline_state.json", default={})
    transcript_state = read_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
    current_state = read_json(session_dir / "current" / "current_state.json", default={})
    if not isinstance(stream_state, dict):
        stream_state = {}
    if not isinstance(pipeline_state, dict):
        pipeline_state = {}
    if not isinstance(transcript_state, dict):
        transcript_state = {}
    if not isinstance(current_state, dict):
        current_state = {}
    current = pipeline_state.get("current") if isinstance(pipeline_state.get("current"), dict) else {}
    short_term = pipeline_state.get("short_term") if isinstance(pipeline_state.get("short_term"), dict) else {}
    long_term = pipeline_state.get("long_term") if isinstance(pipeline_state.get("long_term"), dict) else {}
    frame_stream = public_frame_stream_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    audio_stream = public_audio_stream_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    stream = pipeline_state.get("stream") if isinstance(pipeline_state.get("stream"), dict) else {}
    upload_chunks = stream_state.get("upload_chunks", stream_state.get("received_chunks", [])) or []
    current_text_ready = bool(
        current_state.get("current_text_ready")
        or current_state.get("audio_current_ready")
        or int(current_state.get("transcript_segment_count", 0) or 0) > 0
        or int(audio_stream.get("transcript_segment_count", 0) or 0) > 0
    )
    return {
        "stream_status": stream_state.get("status", "not_started"),
        "current_ready": bool(current.get("ready") or frame_stream.get("mcur_ready") or current_text_ready),
        "current_text_ready": current_text_ready,
        "audio_current_ready": bool(current_state.get("audio_current_ready") or int(audio_stream.get("transcript_segment_count", 0) or 0) > 0),
        "short_term_ready": bool(short_term.get("ready")),
        "long_term_partial_ready": bool(long_term.get("long_term_partial_ready")),
        "long_term_full_ready": bool(long_term.get("long_term_full_ready")),
        "semantic_lagging": bool(long_term.get("semantic_lagging")),
        "graph_lagging": bool(long_term.get("graph_lagging")),
        "asr_lagging": len(transcript_state.get("failed_asr_chunks", []) or []) > 0
        or _as_int(stream_state.get("next_expected_upload_chunk_index"), 0) > len(transcript_state.get("processed_asr_chunks", []) or []),
        "latest_processed_chunk_index": stream.get("last_processed_proc_index", stream_state.get("last_processed_proc_index", stream_state.get("last_processed_chunk_index", -1))),
        "latest_uploaded_chunk_index": max(
            [
                _as_int(item.get("upload_chunk_index", item.get("chunk_index", -1)), -1)
                for item in upload_chunks
                if isinstance(item, dict)
            ],
            default=-1,
        ),
    }


def build_stream_status(project_root: Path, session_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    session_dir = Path(session_dir)
    session_id = session_dir.name
    try:
        manager_summary = StreamChunkManager(session_dir).summary()
    except Exception as exc:
        manager_summary = {
            "session_id": session_id,
            "stream_status": "failed",
            "status_error": str(exc),
            "upload_chunks": [],
            "processing_chunks": [],
            "latency": {},
        }
    pipeline_state = read_json(session_dir / "pipeline_state.json", default={})
    memory_config = read_json(session_dir / "em2mem" / "memory_config.json", default={})
    transcript_state = read_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
    stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
    if not isinstance(pipeline_state, dict):
        pipeline_state = {}
    if not isinstance(memory_config, dict):
        memory_config = {}
    if not isinstance(transcript_state, dict):
        transcript_state = {}
    if not isinstance(stream_state, dict):
        stream_state = {}
    frame_stream = public_frame_stream_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    audio_stream = public_audio_stream_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    frame_mst = public_frame_mst_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    rokid = public_rokid_status_block(session_dir, input_mode=stream_state.get("input_mode"))
    live_ingest = public_live_ingest_status_block(session_dir)
    if isinstance(transcript_state.get("time_span"), list) and len(transcript_state.get("time_span") or []) >= 2:
        try:
            live_ingest = {
                **live_ingest,
                "latest_transcript_end_ms": int(round(float(transcript_state["time_span"][1]) * 1000)),
            }
        except Exception:
            pass
    if live_ingest.get("status") not in {None, "not_started"}:
        frame_stream = public_frame_stream_status_block(session_dir, input_mode="frame_audio_stream")
        audio_stream = public_audio_stream_status_block(session_dir, input_mode="frame_audio_stream")
        frame_mst = public_frame_mst_status_block(session_dir, input_mode="frame_audio_stream")
    current_state = read_json(session_dir / "current" / "current_state.json", default={})
    if not isinstance(current_state, dict):
        current_state = {}
    counts = _count_session_tasks(project_root, session_id)
    audio_asr_task_counts = _count_session_audio_asr_tasks(project_root, session_id)
    audio_stream = _reconcile_audio_stream_asr_status(audio_stream, audio_asr_task_counts)
    global_counts = queue_counts(project_root)
    backpressure = compute_backpressure(project_root=project_root, stream_latency=manager_summary.get("latency") if isinstance(manager_summary, dict) else None)

    upload_chunks = manager_summary.get("upload_chunks", []) if isinstance(manager_summary.get("upload_chunks"), list) else []
    processing_chunks = manager_summary.get("processing_chunks", []) if isinstance(manager_summary.get("processing_chunks"), list) else []
    conflict_chunks = manager_summary.get("conflict_chunks", []) if isinstance(manager_summary.get("conflict_chunks"), list) else []
    failed_upload_chunks = [
        _as_int(item.get("upload_chunk_index", item.get("chunk_index")), -1)
        for item in upload_chunks
        if isinstance(item, dict) and str(item.get("status") or "") == "failed"
    ]
    failed_proc_chunks = [
        _as_int(item.get("proc_index", item.get("chunk_index")), -1)
        for item in processing_chunks
        if isinstance(item, dict) and str(item.get("status") or "") == "failed"
    ]
    current = pipeline_state.get("current") if isinstance(pipeline_state.get("current"), dict) else {}
    short_term = pipeline_state.get("short_term") if isinstance(pipeline_state.get("short_term"), dict) else {}
    long_term = pipeline_state.get("long_term") if isinstance(pipeline_state.get("long_term"), dict) else {}
    open_event_start = manager_summary.get("open_event_start")
    open_event_end = manager_summary.get("open_event_end")
    try:
        open_event_duration = round(float(open_event_end) - float(open_event_start), 3) if open_event_start is not None and open_event_end is not None else None
    except Exception:
        open_event_duration = None
    current_text_ready = bool(
        current_state.get("current_text_ready")
        or current_state.get("audio_current_ready")
        or int(current_state.get("transcript_segment_count", 0) or 0) > 0
        or int(audio_stream.get("transcript_segment_count", 0) or 0) > 0
    )
    short_term_audio_ready = False
    try:
        short_term_audio_ready = any(
            '"input_source": "audio_chunk_asr"' in line or '"input_source":"audio_chunk_asr"' in line
            for line in (session_dir / "short_term" / "micro_events.jsonl").read_text(encoding="utf-8").splitlines()[-20:]
        )
    except Exception:
        short_term_audio_ready = False
    can_ask = bool(current.get("ready") or frame_stream.get("mcur_ready") or current_text_ready or short_term.get("ready") or short_term_audio_ready or long_term.get("long_term_partial_ready"))
    can_upload_next = str(manager_summary.get("stream_status") or "") in {"running", "not_started", "simulated"} and backpressure.get("recommended_action") != "pause_upload"

    asr_pending = counts.get("stream_asr_queued", 0) + counts.get("stream_asr_in_progress", 0)
    asr_done = counts.get("stream_asr_done", 0)
    asr_failed = counts.get("stream_asr_failed", 0)
    status_updated_at = max(
        [
            str(value)
            for value in (manager_summary.get("updated_at"), frame_stream.get("updated_at"), audio_stream.get("updated_at"))
            if value
        ],
        default=None,
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "stream_id": manager_summary.get("stream_id"),
        "stream_status": manager_summary.get("stream_status", "not_started"),
        "upload": {
            "upload_received_count": manager_summary.get("upload_received_count", len(upload_chunks)),
            "upload_processed_count": manager_summary.get("upload_processed_count", 0),
            "processing_chunk_count": manager_summary.get("processing_chunk_count", len(processing_chunks)),
            "processing_done_count": manager_summary.get("processing_done_count", manager_summary.get("processed_processing_chunk_count", 0)),
            "processing_chunk_strategy": manager_summary.get("processing_chunk_strategy"),
            "received_chunk_count": len(upload_chunks),
            "processed_chunk_count": manager_summary.get("processed_chunk_count", 0),
            "failed_chunk_count": len([idx for idx in failed_upload_chunks + failed_proc_chunks if idx >= 0]),
            "next_expected_chunk_index": manager_summary.get("next_expected_upload_chunk_index", manager_summary.get("next_expected_chunk_index", 0)),
            "next_expected_upload_chunk_index": manager_summary.get("next_expected_upload_chunk_index", 0),
            "next_expected_proc_index": manager_summary.get("next_expected_proc_index", 0),
            "last_received_chunk_index": max(
                [_as_int(item.get("upload_chunk_index", item.get("chunk_index")), -1) for item in upload_chunks if isinstance(item, dict)],
                default=-1,
            ),
            "last_processed_chunk_index": manager_summary.get("last_processed_proc_index", manager_summary.get("last_processed_chunk_index", -1)),
            "last_processed_upload_chunk_index": max(
                [
                    _as_int(item.get("upload_chunk_index", item.get("chunk_index")), -1)
                    for item in upload_chunks
                    if isinstance(item, dict) and str(item.get("status") or "") == "processed"
                ],
                default=-1,
            ),
            "last_processed_proc_index": manager_summary.get("last_processed_proc_index", -1),
            "missing_chunks": manager_summary.get("missing_chunks", []) or [],
            "duplicate_chunks": manager_summary.get("duplicate_chunks", []),
            "conflict_chunks": conflict_chunks or [],
            "retry_required_chunks": manager_summary.get("retry_required_chunks", []) or [],
            "waiting_chunks": manager_summary.get("waiting_chunks", []) or [],
        },
        "memory": {
            "current_ready": bool(current.get("ready") or frame_stream.get("mcur_ready") or current_text_ready),
            "current_text_ready": current_text_ready,
            "audio_current_ready": bool(current_state.get("audio_current_ready") or int(audio_stream.get("transcript_segment_count", 0) or 0) > 0),
            "mcur_version": current.get("version") or frame_stream.get("mcur_version", 0),
            "short_term_ready": bool(short_term.get("ready") or short_term_audio_ready),
            "short_term_audio_ready": short_term_audio_ready,
            "mst_version": short_term.get("mst_version", 0),
            "long_term_partial_ready": bool(long_term.get("long_term_partial_ready")),
            "latest_fast_ready_version": long_term.get("latest_fast_ready_version") or memory_config.get("latest_fast_ready_version"),
            "latest_semantic_ready_version": long_term.get("latest_semantic_ready_version") or memory_config.get("latest_semantic_ready_version"),
            "semantic_lagging": bool(long_term.get("semantic_lagging")),
            "graph_lagging": bool(long_term.get("graph_lagging")),
        },
        "live": public_live_status_block(session_dir, stream_state=stream_state),
        "webrtc": public_webrtc_status_block(session_dir, stream_state=stream_state),
        "live_ingest": live_ingest,
        "rokid": rokid,
        "frame_stream": frame_stream,
        "audio_stream": audio_stream,
        "frame_mst": frame_mst,
        "asr": {
            "enabled": _as_bool(os.getenv("EM2MEM_STREAM_ASR_ENABLED"), True),
            "backend": os.getenv("EM2MEM_STREAM_ASR_BACKEND", "whisperx"),
            "pending": asr_pending,
            "done": asr_done,
            "failed": asr_failed,
            "partial_transcript_version": transcript_state.get("partial_transcript_version", 0),
            "segment_count": transcript_state.get("segment_count", 0),
        },
        "tasks": {
            "stream_chunk_pending": counts.get("stream_chunk_queued", 0) + counts.get("stream_chunk_in_progress", 0),
            "stream_asr_pending": asr_pending,
            "mst_refine_pending": counts.get("mst_refine_queued", 0) + counts.get("mst_refine_in_progress", 0),
            "mst_consolidation_pending": counts.get("mst_consolidation_queued", 0) + counts.get("mst_consolidation_in_progress", 0),
            "memory_pending": counts.get("memory_queued", 0) + counts.get("memory_in_progress", 0),
            "visual_pending": counts.get("visual_queued", 0) + counts.get("visual_in_progress", 0),
            "query_pending": counts.get("query_queued", 0) + counts.get("query_in_progress", 0),
            "global": global_counts,
        },
        "open_event": {
            "has_open_event": bool(manager_summary.get("has_open_event")),
            "start_time": open_event_start,
            "last_update_time": open_event_end,
            "duration": open_event_duration,
        },
        "backpressure": backpressure,
        "latency": manager_summary.get("latency", {}),
        "can_ask": can_ask,
        "can_upload_next_chunk": can_upload_next,
        "updated_at": status_updated_at,
        "raw_stream": manager_summary,
    }
