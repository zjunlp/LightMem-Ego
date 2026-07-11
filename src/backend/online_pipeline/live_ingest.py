from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from online_pipeline.live_rtmp import load_live_source, save_live_source
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


def live_ingest_state_path(session_dir: Path) -> Path:
    return Path(session_dir) / "stream" / "live_ingest_state.json"


def load_live_ingest_state(session_dir: Path) -> dict[str, Any]:
    payload = read_json(live_ingest_state_path(session_dir), default={})
    return payload if isinstance(payload, dict) else {}


def write_live_ingest_state(session_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    path = live_ingest_state_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = utc_now_iso()
    write_json_atomic(path, payload)
    return payload


def update_live_ingest_state(session_dir: Path, **updates: Any) -> dict[str, Any]:
    state = load_live_ingest_state(session_dir)
    # None is meaningful for runtime fields such as cleared pids/errors.
    state.update(updates)
    return write_live_ingest_state(session_dir, state)


def choose_live_source_url(live_source: dict[str, Any]) -> str:
    return str(live_source.get("pull_url_internal") or live_source.get("pull_url_public") or live_source.get("push_url_public") or "").strip()


def choose_effective_live_pull_url(live_source: dict[str, Any], fallback_url: str = "") -> tuple[str, str, bool]:
    """Return the worker-only pull URL and its source.

    `WORLDMM_LIVE_PULL_BASE_URL` lets isolated compute nodes pull via a local
    tunnel, e.g. rtmp://127.0.0.1:11935/live/<stream_name>, without mutating
    live_source.json or frontend-facing public URLs.
    """
    stream_name = str(live_source.get("stream_name") or "").strip()
    base_override = str(os.getenv("WORLDMM_LIVE_PULL_BASE_URL") or "").strip()
    if base_override and stream_name:
        return f"{base_override.rstrip('/')}/{stream_name}", "env_override", True
    internal = str(live_source.get("pull_url_internal") or "").strip()
    if internal:
        return internal, "pull_url_internal", False
    public = str(live_source.get("pull_url_public") or "").strip()
    if public:
        return public, "pull_url_public", False
    fallback = str(fallback_url or live_source.get("push_url_public") or "").strip()
    if fallback:
        return fallback, "task_source_url", False
    return "", "none", bool(base_override)


def initialize_live_ingest_state(
    session_dir: Path,
    *,
    stream_id: str,
    source_url: str,
    frame_fps: float,
    audio_segment_ms: int,
    source: str = "srs_rtmp",
    input_mode: str = "live_pusher_rtmp",
    status: str = "queued",
) -> dict[str, Any]:
    existing = load_live_ingest_state(session_dir)
    now = utc_now_iso()
    rtmp_video_only = str(input_mode or "").strip() == "rokid_live_rtmp"
    state = {
        "session_id": Path(session_dir).name,
        "stream_id": stream_id,
        "input_mode": input_mode,
        "source_url": source_url,
        "effective_pull_url": existing.get("effective_pull_url") or source_url,
        "pull_url_source": existing.get("pull_url_source") or "source_url",
        "pull_base_override_enabled": bool(existing.get("pull_base_override_enabled", False)),
        "status": status,
        "started_at": existing.get("started_at"),
        "updated_at": now,
        "stopped_at": existing.get("stopped_at"),
        "frame_fps": float(frame_fps),
        "audio_segment_ms": int(audio_segment_ms),
        "frame_index": int(existing.get("frame_index")) if existing.get("frame_index") is not None else -1,
        "audio_index": int(existing.get("audio_index")) if existing.get("audio_index") is not None else -1,
        "frames_ingested": int(existing.get("frames_ingested", 0) or 0),
        "audio_chunks_ingested": int(existing.get("audio_chunks_ingested", 0) or 0),
        "last_frame_relative_ts_ms": existing.get("last_frame_relative_ts_ms"),
        "last_audio_relative_ts_ms": existing.get("last_audio_relative_ts_ms"),
        "latest_frame_relative_ts_ms": existing.get("latest_frame_relative_ts_ms", existing.get("last_frame_relative_ts_ms")),
        "latest_audio_relative_ts_ms": existing.get("latest_audio_relative_ts_ms", existing.get("last_audio_relative_ts_ms")),
        "latest_audio_end_relative_ts_ms": existing.get("latest_audio_end_relative_ts_ms"),
        "latest_transcript_end_ms": existing.get("latest_transcript_end_ms"),
        "timeline_version": int(existing.get("timeline_version", 1) or 1),
        "timeline_origin_wall_ms": existing.get("timeline_origin_wall_ms"),
        "timeline_origin_monotonic_ns": existing.get("timeline_origin_monotonic_ns"),
        "timestamp_mode": existing.get("timestamp_mode"),
        "timestamp_source": existing.get("timestamp_source"),
        "av_skew_ms": existing.get("av_skew_ms"),
        "frame_monotonic_violations": int(existing.get("frame_monotonic_violations", 0) or 0),
        "audio_monotonic_violations": int(existing.get("audio_monotonic_violations", 0) or 0),
        "sync_status": existing.get("sync_status", "unknown"),
        "rtmp_video_only": rtmp_video_only,
        "rtmp_audio_expected": not rtmp_video_only,
        "rtmp_audio_disabled": rtmp_video_only,
        "rtmp_transport_mode": "video_rtmp_audio_http" if rtmp_video_only else "av_rtmp",
        "rtmp_rw_timeout_us": existing.get("rtmp_rw_timeout_us"),
        "rtmp_startup_wait_seconds": existing.get("rtmp_startup_wait_seconds"),
        "rtmp_recovery_wait_seconds": existing.get("rtmp_recovery_wait_seconds"),
        "video_reconnect_count": int(existing.get("video_reconnect_count", 0) or 0),
        "video_stall_count": int(existing.get("video_stall_count", 0) or 0),
        "last_frame_at": existing.get("last_frame_at"),
        "last_audio_at": existing.get("last_audio_at"),
        "last_error": existing.get("last_error"),
        "ffmpeg_video_pid": existing.get("ffmpeg_video_pid"),
        "ffmpeg_audio_pid": existing.get("ffmpeg_audio_pid"),
        "source": source,
        "stop_requested": bool(existing.get("stop_requested", False)),
        "audio_unavailable": False if rtmp_video_only else bool(existing.get("audio_unavailable", False)),
        "pull_attempt_count": int(existing.get("pull_attempt_count", 0) or 0),
        "ffprobe_ok": existing.get("ffprobe_ok"),
        "last_probe_error": existing.get("last_probe_error"),
        "last_pull_error": existing.get("last_pull_error"),
        "waiting_reason": existing.get("waiting_reason"),
        "last_output_file_at": existing.get("last_output_file_at"),
        "video_output_dir": existing.get("video_output_dir"),
        "audio_output_dir": existing.get("audio_output_dir"),
        "last_ffmpeg_stderr_tail": existing.get("last_ffmpeg_stderr_tail"),
        "ffmpeg_bin": existing.get("ffmpeg_bin"),
        "ffprobe_bin": existing.get("ffprobe_bin"),
        "ffprobe_cmd": existing.get("ffprobe_cmd"),
        "ffmpeg_video_cmd": existing.get("ffmpeg_video_cmd"),
        "ffmpeg_audio_cmd": existing.get("ffmpeg_audio_cmd"),
        "ffmpeg_video_exit_code": existing.get("ffmpeg_video_exit_code"),
        "ffmpeg_audio_exit_code": existing.get("ffmpeg_audio_exit_code"),
        "video_output_glob": existing.get("video_output_glob"),
        "audio_output_glob": existing.get("audio_output_glob"),
        "detected_video_file_count": int(existing.get("detected_video_file_count", 0) or 0),
        "detected_audio_file_count": int(existing.get("detected_audio_file_count", 0) or 0),
    }
    return write_live_ingest_state(session_dir, state)


def update_live_source_ingest(session_dir: Path, **updates: Any) -> dict[str, Any]:
    live_source = load_live_source(session_dir)
    if not live_source:
        return {}
    live_source.update({key: value for key, value in updates.items() if value is not None})
    live_source["updated_at"] = utc_now_iso()
    save_live_source(session_dir, live_source)
    return live_source


def request_live_ingest_stop(session_dir: Path, reason: str = "api_stop") -> dict[str, Any]:
    state = load_live_ingest_state(session_dir)
    state["stop_requested"] = True
    state["stop_reason"] = reason
    if str(state.get("status") or "") in {
        "running",
        "starting",
        "waiting_stream",
        "waiting_rtmp_output",
        "waiting_keyframe",
        "queued",
    }:
        state["status"] = "stopping"
    state = write_live_ingest_state(session_dir, state)
    update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_error=state.get("last_error"))
    return state


def public_live_ingest_status_block(session_dir: Path) -> dict[str, Any]:
    state = load_live_ingest_state(session_dir)
    if not state:
        return {
            "status": "not_started",
            "frames_ingested": 0,
            "audio_chunks_ingested": 0,
            "last_error": None,
        }
    keys = (
        "status",
        "source_url",
        "effective_pull_url",
        "pull_url_source",
        "pull_base_override_enabled",
        "source",
        "frame_fps",
        "audio_segment_ms",
        "frame_index",
        "audio_index",
        "frames_ingested",
        "audio_chunks_ingested",
        "last_frame_relative_ts_ms",
        "last_audio_relative_ts_ms",
        "latest_frame_relative_ts_ms",
        "latest_audio_relative_ts_ms",
        "latest_audio_end_relative_ts_ms",
        "latest_transcript_end_ms",
        "timeline_version",
        "timeline_origin_wall_ms",
        "timeline_origin_monotonic_ns",
        "timestamp_mode",
        "timestamp_source",
        "av_skew_ms",
        "frame_monotonic_violations",
        "audio_monotonic_violations",
        "sync_status",
        "rtmp_video_only",
        "rtmp_audio_expected",
        "rtmp_audio_disabled",
        "rtmp_transport_mode",
        "rtmp_rw_timeout_us",
        "rtmp_startup_wait_seconds",
        "rtmp_recovery_wait_seconds",
        "video_reconnect_count",
        "video_stall_count",
        "last_frame_at",
        "last_audio_at",
        "last_error",
        "pull_attempt_count",
        "ffprobe_ok",
        "last_probe_error",
        "last_pull_error",
        "waiting_reason",
        "last_output_file_at",
        "video_output_dir",
        "audio_output_dir",
        "last_ffmpeg_stderr_tail",
        "ffmpeg_bin",
        "ffprobe_bin",
        "ffprobe_cmd",
        "ffmpeg_video_cmd",
        "ffmpeg_audio_cmd",
        "ffmpeg_video_exit_code",
        "ffmpeg_audio_exit_code",
        "video_output_glob",
        "audio_output_glob",
        "detected_video_file_count",
        "detected_audio_file_count",
        "audio_asr_tail_flush",
        "frame_mst_close",
        "frame_mst_closed_on_stop",
        "live_finalize_error",
        "audio_unavailable",
        "stop_requested",
        "ffmpeg_video_pid",
        "ffmpeg_audio_pid",
        "started_at",
        "stopped_at",
        "updated_at",
    )
    return {key: state.get(key) for key in keys}
