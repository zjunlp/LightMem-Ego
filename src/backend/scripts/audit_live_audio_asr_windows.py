#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online_preprocess.io_utils import read_json  # noqa: E402
from online_preprocess.task_queue import ensure_queue_dirs  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _duration_ms(window: dict[str, Any]) -> int:
    if window.get("duration_ms") is not None:
        try:
            return int(window.get("duration_ms"))
        except Exception:
            pass
    try:
        return int(window.get("window_end_ms")) - int(window.get("window_start_ms"))
    except Exception:
        return 0


def _session_stream_asr_tasks(project_root: Path, session_id: str) -> dict[str, list[dict[str, Any]]]:
    dirs = ensure_queue_dirs(project_root)
    mapping = {
        "queued": "stream_asr_queued",
        "in_progress": "stream_asr_in_progress",
        "done": "stream_asr_done",
        "failed": "stream_asr_failed",
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for public_key, queue_key in mapping.items():
        rows: list[dict[str, Any]] = []
        for path in sorted(dirs[queue_key].glob(f"{session_id}_*.json")):
            payload = read_json(path, default={})
            if isinstance(payload, dict) and payload.get("source") == "audio_chunk_window":
                rows.append({**payload, "_task_path": str(path)})
        result[public_key] = rows
    return result


def build_report(project_root: Path, session_id: str) -> dict[str, Any]:
    session_dir = project_root / "online_sessions" / session_id
    warnings: list[str] = []
    if not session_dir.exists():
        return {"session_id": session_id, "exists": False, "warnings": ["session directory not found"]}
    audio_state = read_json(session_dir / "stream" / "audio_state.json", default={})
    audio_buffer = read_json(session_dir / "stream" / "audio_buffer_index.json", default={})
    asr_state = read_json(session_dir / "stream" / "audio_asr_state.json", default={})
    if not isinstance(audio_state, dict):
        audio_state = {}
    if not isinstance(audio_buffer, dict):
        audio_buffer = {}
    if not isinstance(asr_state, dict):
        asr_state = {}
    chunks = [item for item in audio_state.get("chunks", []) or [] if isinstance(item, dict)]
    buffer_chunks = [item for item in audio_buffer.get("chunks", []) or [] if isinstance(item, dict)]
    windows = [item for item in asr_state.get("windows", []) or [] if isinstance(item, dict)]
    tasks = _session_stream_asr_tasks(project_root, session_id)
    state_window_ids = [str(item.get("window_id")) for item in windows if item.get("window_id")]
    active_task_window_ids = [
        str(item.get("window_id"))
        for key in ("queued", "in_progress")
        for item in tasks.get(key, [])
        if item.get("window_id")
    ]
    failed_task_window_ids = [str(item.get("window_id")) for item in tasks.get("failed", []) if item.get("window_id")]
    done_task_window_ids = [str(item.get("window_id")) for item in tasks.get("done", []) if item.get("window_id")]
    state_duplicates = sorted([window_id for window_id, count in Counter(state_window_ids).items() if count > 1])
    active_task_duplicates = sorted([window_id for window_id, count in Counter(active_task_window_ids).items() if count > 1])
    historical_failed_retried = sorted(set(failed_task_window_ids) & set(done_task_window_ids))
    duplicates = sorted(set(state_duplicates + active_task_duplicates))
    durations = [_duration_ms(item) for item in windows if _duration_ms(item) > 0]
    ordinary_durations = [_duration_ms(item) for item in windows if _duration_ms(item) > 0 and not item.get("is_flush")]
    too_short = [
        {
            "window_id": item.get("window_id"),
            "duration_ms": _duration_ms(item),
            "is_flush": bool(item.get("is_flush")),
        }
        for item in windows
        if _duration_ms(item) > 0 and _duration_ms(item) < 2000
    ]
    short_ordinary = [
        {
            "window_id": item.get("window_id"),
            "duration_ms": _duration_ms(item),
        }
        for item in windows
        if _duration_ms(item) > 0 and _duration_ms(item) < int(asr_state.get("min_window_ms", 2000) or 2000) and not item.get("is_flush")
    ]
    if duplicates:
        warnings.append("duplicate ASR window_id detected")
    if short_ordinary:
        warnings.append("ordinary ASR window shorter than min_window_ms detected")
    if too_short:
        warnings.append("ASR window shorter than 2s detected")
    transcript_segments = _read_jsonl(session_dir / "stream" / "transcript" / "partial_transcript.jsonl")
    return {
        "session_id": session_id,
        "exists": True,
        "audio_chunks": len(chunks),
        "buffer_chunks": len(buffer_chunks),
        "audio_ready": bool(audio_state.get("ready")),
        "asr_status": asr_state.get("asr_status", audio_state.get("asr_status")),
        "config": {
            "asr_window_ms": asr_state.get("window_ms"),
            "asr_hop_ms": asr_state.get("hop_ms"),
            "asr_min_window_ms": asr_state.get("min_window_ms"),
            "asr_flush_min_ms": asr_state.get("flush_min_ms"),
            "asr_max_window_ms": asr_state.get("max_window_ms"),
            "max_pending_windows": asr_state.get("max_pending_windows"),
        },
        "asr_windows": len(windows),
        "window_duration_ms": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "avg": round(sum(durations) / len(durations), 1) if durations else None,
            "ordinary_min": min(ordinary_durations) if ordinary_durations else None,
            "ordinary_max": max(ordinary_durations) if ordinary_durations else None,
        },
        "too_short_windows": len(too_short),
        "too_short_window_details": too_short,
        "short_ordinary_windows": len(short_ordinary),
        "duplicate_windows": len(duplicates),
        "duplicate_window_ids": duplicates,
        "duplicate_state_window_ids": state_duplicates,
        "duplicate_active_task_window_ids": active_task_duplicates,
        "historical_failed_retried_windows": historical_failed_retried,
        "task_counts": {key: len(value) for key, value in tasks.items()},
        "transcript_segments": len(transcript_segments),
        "tail_flush_enqueued": bool(asr_state.get("tail_flush_enqueued", False)),
        "dropped_tail_ms": int(asr_state.get("dropped_tail_ms", 0) or 0),
        "latest_window_id": asr_state.get("latest_window_id"),
        "latest_asr_window_duration_ms": asr_state.get("latest_asr_window_duration_ms"),
        "buffered_audio_ms": asr_state.get("buffered_audio_ms", 0),
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit for live audio rolling ASR windows.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(Path(args.project_root).resolve(), args.session_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"session_id: {report.get('session_id')}")
        print(f"audio_chunks: {report.get('audio_chunks')}")
        print(f"asr_windows: {report.get('asr_windows')}")
        print(f"window_duration_ms: {report.get('window_duration_ms')}")
        print(f"transcript_segments: {report.get('transcript_segments')}")
        print(f"historical_failed_retried_windows: {report.get('historical_failed_retried_windows')}")
        print(f"warnings: {report.get('warnings')}")


if __name__ == "__main__":
    main()
