from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import queue_counts


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _max_latency_seconds(latency: dict[str, Any] | None) -> float:
    if not isinstance(latency, dict):
        return 0.0
    values: list[float] = []
    for key in (
        "last_chunk_upload_to_mcur_ms",
        "last_chunk_upload_to_mst_ms",
        "last_chunk_upload_to_asr_ms",
        "last_chunk_upload_to_refine_ms",
        "last_chunk_upload_to_memory_ms",
        "avg_upload_to_mcur_ms",
        "avg_upload_to_asr_ms",
        "avg_upload_to_memory_ms",
    ):
        try:
            value = latency.get(key)
            if value is not None:
                values.append(float(value) / 1000.0)
        except Exception:
            continue
    return max(values, default=0.0)


def compute_backpressure(
    *,
    project_root: Path,
    stream_latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enabled = _env_bool("EM2MEM_BACKPRESSURE_ENABLED", False)
    counts = queue_counts(Path(project_root))
    stream_pending = int(counts.get("stream_chunk_queued", 0) or 0) + int(counts.get("stream_chunk_in_progress", 0) or 0)
    asr_pending = int(counts.get("stream_asr_queued", 0) or 0) + int(counts.get("stream_asr_in_progress", 0) or 0)
    memory_pending = int(counts.get("memory_queued", 0) or 0) + int(counts.get("memory_in_progress", 0) or 0)
    refine_pending = int(counts.get("mst_refine_queued", 0) or 0) + int(counts.get("mst_refine_in_progress", 0) or 0)
    consolidation_pending = int(counts.get("mst_consolidation_queued", 0) or 0) + int(counts.get("mst_consolidation_in_progress", 0) or 0)

    stream_high = _env_int("EM2MEM_BACKPRESSURE_STREAM_PENDING_HIGH", 10)
    asr_high = _env_int("EM2MEM_BACKPRESSURE_ASR_PENDING_HIGH", 5)
    memory_high = _env_int("EM2MEM_BACKPRESSURE_MEMORY_PENDING_HIGH", 3)
    latency_high = _env_float("EM2MEM_BACKPRESSURE_LATENCY_HIGH_SECONDS", 30.0)
    max_latency = _max_latency_seconds(stream_latency)

    reasons: list[str] = []
    severity = 0
    if stream_pending >= stream_high:
        severity = max(severity, 3)
        reasons.append(f"stream queue pending={stream_pending} >= {stream_high}")
    elif stream_pending >= max(1, int(stream_high * 0.6)):
        severity = max(severity, 2)
        reasons.append(f"stream queue pending={stream_pending}")
    elif stream_pending > 0:
        severity = max(severity, 1)

    if asr_pending >= asr_high:
        severity = max(severity, 2)
        reasons.append(f"asr queue pending={asr_pending} >= {asr_high}")
    elif asr_pending > 0:
        severity = max(severity, 1)

    if memory_pending >= memory_high:
        severity = max(severity, 3)
        reasons.append(f"memory queue pending={memory_pending} >= {memory_high}")
    elif memory_pending > 0 or refine_pending > 0 or consolidation_pending > 0:
        severity = max(severity, 1)

    if max_latency >= latency_high:
        severity = max(severity, 2)
        reasons.append(f"latency={max_latency:.1f}s >= {latency_high:.1f}s")

    if not enabled:
        return {
            "enabled": False,
            "level": "none",
            "reason": None,
            "recommended_action": "continue",
            "recommended_chunk_duration": 5.0,
            "retry_after_seconds": 0,
        }

    if severity >= 3:
        return {
            "enabled": True,
            "level": "high",
            "reason": "; ".join(reasons) or "queue pressure is high",
            "recommended_action": "pause_upload",
            "recommended_chunk_duration": 10.0,
            "retry_after_seconds": 10,
        }
    if severity == 2:
        return {
            "enabled": True,
            "level": "medium",
            "reason": "; ".join(reasons) or "queue pressure is elevated",
            "recommended_action": "slow_down",
            "recommended_chunk_duration": 8.0,
            "retry_after_seconds": 3,
        }
    if severity == 1:
        return {
            "enabled": True,
            "level": "low",
            "reason": "; ".join(reasons) or "minor queue backlog",
            "recommended_action": "continue",
            "recommended_chunk_duration": 5.0,
            "retry_after_seconds": 0,
        }
    return {
        "enabled": True,
        "level": "none",
        "reason": None,
        "recommended_action": "continue",
        "recommended_chunk_duration": 5.0,
        "retry_after_seconds": 0,
    }
