from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from online_pipeline.runtime_state import collect_worker_runtime, queue_counts
from online_pipeline.stream_status import build_stream_status
from online_preprocess.io_utils import ensure_dir, read_json, utc_now_iso


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _elapsed_ms(start: Any, end: Any) -> int | None:
    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0, int(round((end_dt - start_dt).total_seconds() * 1000)))


def _percentile(values: list[int], q: float) -> int | None:
    values = sorted(int(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return int(round(values[low] * (1.0 - frac) + values[high] * frac))


def timeline_path(session_dir: Path) -> Path:
    return Path(session_dir) / "stream" / "timeline.jsonl"


def append_timeline_event(
    session_dir: Path,
    event_type: str,
    *,
    chunk_index: int | None = None,
    chunk_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    stage_started_at: str | None = None,
) -> None:
    try:
        session_dir = Path(session_dir)
        state = read_json(session_dir / "stream" / "stream_state.json", default={})
        if not isinstance(state, dict):
            state = {}
        now = utc_now_iso()
        payload = {
            "event_id": f"tl_{uuid4().hex[:12]}",
            "session_id": session_dir.name,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "event_type": str(event_type),
            "timestamp": now,
            "relative_time_ms": _elapsed_ms(state.get("created_at"), now),
            "stage_latency_ms": _elapsed_ms(stage_started_at, now) if stage_started_at else None,
            "metadata": metadata or {},
        }
        path = timeline_path(session_dir)
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def read_timeline_events(
    session_dir: Path,
    *,
    limit: int = 100,
    event_type: str | None = None,
    chunk_index: int | None = None,
) -> list[dict[str, Any]]:
    path = timeline_path(session_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                if event_type and str(item.get("event_type") or "") != str(event_type):
                    continue
                if chunk_index is not None:
                    try:
                        if int(item.get("chunk_index")) != int(chunk_index):
                            continue
                    except Exception:
                        continue
                rows.append(item)
    except Exception:
        return []
    limit = max(1, int(limit or 100))
    return rows[-limit:]


def _latency_samples_from_stream_state(session_dir: Path) -> dict[str, list[int]]:
    state = read_json(Path(session_dir) / "stream" / "stream_state.json", default={})
    if not isinstance(state, dict):
        return {}
    samples = {
        "upload_to_mcur": [],
        "upload_to_mst": [],
        "upload_to_asr": [],
        "upload_to_transcript_backfill": [],
        "upload_to_refine": [],
        "upload_to_memory": [],
    }
    for upload in state.get("upload_chunks", state.get("received_chunks", [])) or []:
        if not isinstance(upload, dict):
            continue
        received_at = upload.get("received_at")
        pairs = {
            "upload_to_asr": upload.get("asr_done_at"),
            "upload_to_transcript_backfill": upload.get("transcript_backfilled_at"),
            "upload_to_refine": upload.get("refine_queued_at"),
            "upload_to_memory": upload.get("memory_appended_at"),
        }
        nested = [item for item in upload.get("processing_chunks", []) or [] if isinstance(item, dict)]
        if nested:
            for key, target_key in (("upload_to_mcur", "mcur_updated_at"), ("upload_to_mst", "mst_event_closed_at")):
                values = [item.get(target_key) for item in nested if item.get(target_key)]
                if values:
                    pairs[key] = max(values)
        else:
            pairs["upload_to_mcur"] = upload.get("mcur_updated_at")
            pairs["upload_to_mst"] = upload.get("mst_event_closed_at")
        for key, end_at in pairs.items():
            value = _elapsed_ms(received_at, end_at)
            if value is not None:
                samples[key].append(value)
    return samples


def build_stream_metrics(project_root: Path, session_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    session_dir = Path(session_dir)
    status = build_stream_status(project_root, session_dir)
    timeline = read_timeline_events(session_dir, limit=5000)
    samples = _latency_samples_from_stream_state(session_dir)
    summary_status = status.get("upload") if isinstance(status.get("upload"), dict) else {}
    transcript_state = read_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", default={})
    append_state = read_json(session_dir / "em2mem" / "incremental" / "append_state.json", default={})
    if not isinstance(transcript_state, dict):
        transcript_state = {}
    if not isinstance(append_state, dict):
        append_state = {}
    queues_raw = queue_counts(project_root)
    worker_raw = collect_worker_runtime(project_root)
    queues = {
        "stream_chunk": {"pending": queues_raw.get("stream_chunk_queued", 0), "running": queues_raw.get("stream_chunk_in_progress", 0), "done": queues_raw.get("stream_chunk_done", 0), "failed": queues_raw.get("stream_chunk_failed", 0)},
        "stream_asr": {"pending": queues_raw.get("stream_asr_queued", 0), "running": queues_raw.get("stream_asr_in_progress", 0), "done": queues_raw.get("stream_asr_done", 0), "failed": queues_raw.get("stream_asr_failed", 0)},
        "mst_refine": {"pending": queues_raw.get("mst_refine_queued", 0), "running": queues_raw.get("mst_refine_in_progress", 0), "done": queues_raw.get("mst_refine_done", 0), "failed": queues_raw.get("mst_refine_failed", 0)},
        "mst_consolidation": {"pending": queues_raw.get("mst_consolidation_queued", 0), "running": queues_raw.get("mst_consolidation_in_progress", 0), "done": queues_raw.get("mst_consolidation_done", 0), "failed": queues_raw.get("mst_consolidation_failed", 0)},
        "memory": {"pending": queues_raw.get("memory_queued", 0), "running": queues_raw.get("memory_in_progress", 0), "done": queues_raw.get("memory_done", 0), "failed": queues_raw.get("memory_failed", 0)},
        "visual": {"pending": queues_raw.get("visual_queued", 0), "running": queues_raw.get("visual_in_progress", 0), "done": queues_raw.get("visual_done", 0), "failed": queues_raw.get("visual_failed", 0)},
        "query": {"pending": queues_raw.get("query_queued", 0), "running": queues_raw.get("query_in_progress", 0), "done": queues_raw.get("query_done", 0), "failed": queues_raw.get("query_failed", 0)},
    }
    latency_state = status.get("latency") if isinstance(status.get("latency"), dict) else {}
    latency = {
        "last_upload_to_mcur_ms": latency_state.get("last_chunk_upload_to_mcur_ms"),
        "last_upload_to_mst_ms": latency_state.get("last_chunk_upload_to_mst_ms"),
        "last_upload_to_asr_ms": latency_state.get("last_chunk_upload_to_asr_ms"),
        "last_upload_to_transcript_backfill_ms": latency_state.get("last_chunk_upload_to_asr_ms"),
        "last_upload_to_refine_ms": latency_state.get("last_chunk_upload_to_refine_ms"),
        "last_upload_to_memory_ms": latency_state.get("last_chunk_upload_to_memory_ms"),
        "avg_upload_to_mcur_ms": latency_state.get("avg_upload_to_mcur_ms"),
        "avg_upload_to_mst_ms": _percentile(samples.get("upload_to_mst", []), 0.5),
        "avg_upload_to_asr_ms": latency_state.get("avg_upload_to_asr_ms"),
        "avg_upload_to_memory_ms": latency_state.get("avg_upload_to_memory_ms"),
        "p50_upload_to_mcur_ms": _percentile(samples.get("upload_to_mcur", []), 0.5),
        "p90_upload_to_mcur_ms": _percentile(samples.get("upload_to_mcur", []), 0.9),
        "p50_upload_to_asr_ms": _percentile(samples.get("upload_to_asr", []), 0.5),
        "p90_upload_to_asr_ms": _percentile(samples.get("upload_to_asr", []), 0.9),
        "p50_upload_to_memory_ms": _percentile(samples.get("upload_to_memory", []), 0.5),
        "p90_upload_to_memory_ms": _percentile(samples.get("upload_to_memory", []), 0.9),
    }
    memory = status.get("memory") if isinstance(status.get("memory"), dict) else {}
    asr = status.get("asr") if isinstance(status.get("asr"), dict) else {}
    return {
        "status": "ok",
        "session_id": session_dir.name,
        "stream_id": status.get("stream_id"),
        "summary": {
            "upload_received_count": summary_status.get("upload_received_count", summary_status.get("received_chunk_count", 0)),
            "upload_processed_count": summary_status.get("upload_processed_count", 0),
            "processing_chunk_count": summary_status.get("processing_chunk_count", 0),
            "processing_done_count": summary_status.get("processing_done_count", summary_status.get("processed_chunk_count", 0)),
            "processing_chunk_strategy": summary_status.get("processing_chunk_strategy"),
            "received_chunk_count": summary_status.get("received_chunk_count", 0),
            "processed_chunk_count": summary_status.get("processed_chunk_count", 0),
            "failed_chunk_count": summary_status.get("failed_chunk_count", 0),
            "micro_event_count": _count_micro_events(session_dir),
            "asr_segment_count": transcript_state.get("segment_count", 0),
            "memory_append_count": append_state.get("appended_count", 0),
        },
        "latency": latency,
        "workers": {
            "stream": worker_raw.get("stream", {}),
            "preprocess": worker_raw.get("preprocess", {}),
            "mst_refine": worker_raw.get("refine", {}),
            "mst_consolidation": worker_raw.get("consolidation", {}),
            "memory": worker_raw.get("memory", {}),
            "visual": worker_raw.get("visual", {}),
            "query": worker_raw.get("query", {}),
        },
        "queues": queues,
        "lagging": {
            "asr_lagging": bool(asr.get("pending", 0)),
            "semantic_lagging": bool(memory.get("semantic_lagging")),
            "graph_lagging": bool(memory.get("graph_lagging")),
            "memory_lagging": bool(memory.get("latest_fast_ready_version") and memory.get("latest_semantic_ready_version") and memory.get("latest_fast_ready_version") != memory.get("latest_semantic_ready_version")),
        },
        "recent_timeline_events": timeline[-10:],
        "updated_at": utc_now_iso(),
    }


def _count_micro_events(session_dir: Path) -> int:
    path = Path(session_dir) / "short_term" / "archive" / "micro_events_all.jsonl"
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
