from __future__ import annotations

import os
import wave
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


ROKID_INPUT_MODE = "rokid_frame_audio"
ROKID_LIVE_RTMP_INPUT_MODE = "rokid_live_rtmp"
ROKID_INPUT_MODES = {ROKID_INPUT_MODE, ROKID_LIVE_RTMP_INPUT_MODE}
ROKID_DEFAULT_VIDEO_SOURCE = "rokid_sdk_video"
ROKID_DEFAULT_AUDIO_SOURCE = "rokid_sdk_audio"
ROKID_TIMESTAMP_MODE = "connector_relative_ts_ms"

ROKID_ACCEPTED_VIDEO_FORMATS = ["jpg", "jpeg", "png", "webp"]
ROKID_OPTIONAL_RAW_VIDEO_FORMATS = ["nv21"]
ROKID_ACCEPTED_AUDIO_FORMATS = ["wav", "pcm", "aac", "m4a", "mp3"]


class RokidNormalizationError(ValueError):
    pass


def is_rokid_input_mode(input_mode: Any) -> bool:
    return str(input_mode or "").strip().lower() in ROKID_INPUT_MODES


def default_rokid_metadata(metadata: dict[str, Any] | None = None, *, input_mode: str = ROKID_INPUT_MODE) -> dict[str, Any]:
    input_mode = str(input_mode or ROKID_INPUT_MODE).strip().lower()
    payload = dict(metadata or {})
    payload.setdefault("source", "rokid_glass")
    payload.setdefault("device_type", "rokid")
    payload.setdefault("transport", "rtmp_srs" if input_mode == ROKID_LIVE_RTMP_INPUT_MODE else "phone_sdk")
    payload.setdefault("sdk", "android_rootencoder_rtmp" if input_mode == ROKID_LIVE_RTMP_INPUT_MODE else "rokid")
    payload.setdefault("timestamp_mode", "srs_live_ingest" if input_mode == ROKID_LIVE_RTMP_INPUT_MODE else ROKID_TIMESTAMP_MODE)
    payload["input_mode"] = input_mode
    return payload


def rokid_state_path(session_dir: Path) -> Path:
    return Path(session_dir) / "stream" / "rokid_state.json"


def load_rokid_state(session_dir: Path) -> dict[str, Any]:
    payload = read_json(rokid_state_path(session_dir), default={})
    return payload if isinstance(payload, dict) else {}


def _metadata_from_session(session_dir: Path, stream_state: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = {}
    if isinstance(stream_state, dict) and isinstance(stream_state.get("metadata"), dict):
        metadata.update(stream_state.get("metadata") or {})
    session_metadata = read_json(Path(session_dir) / "metadata.json", default={})
    if isinstance(session_metadata, dict) and isinstance(session_metadata.get("metadata"), dict):
        metadata = {**session_metadata.get("metadata", {}), **metadata}
    return metadata


def initialize_rokid_state(session_dir: Path, *, stream_id: str, metadata: dict[str, Any] | None = None, input_mode: str = ROKID_INPUT_MODE) -> dict[str, Any]:
    metadata = default_rokid_metadata(metadata, input_mode=input_mode)
    input_mode = str(metadata.get("input_mode") or input_mode or ROKID_INPUT_MODE).strip().lower()
    existing = load_rokid_state(session_dir)
    now = utc_now_iso()
    state = {
        "enabled": True,
        "session_id": Path(session_dir).name,
        "stream_id": stream_id,
        "input_mode": input_mode,
        "device_type": metadata.get("device_type", "rokid"),
        "transport": metadata.get("transport", "phone_sdk"),
        "sdk": metadata.get("sdk", "rokid"),
        "sdk_version": metadata.get("sdk_version"),
        "device_id": metadata.get("device_id"),
        "glass_sn": metadata.get("glass_sn"),
        "phone_model": metadata.get("phone_model"),
        "timestamp_mode": metadata.get("timestamp_mode") or ROKID_TIMESTAMP_MODE,
        "video_source": existing.get("video_source") or ROKID_DEFAULT_VIDEO_SOURCE,
        "audio_source": existing.get("audio_source") or ROKID_DEFAULT_AUDIO_SOURCE,
        "frames_received": int(existing.get("frames_received", 0) or 0),
        "audio_chunks_received": int(existing.get("audio_chunks_received", 0) or 0),
        "latest_frame_relative_ts_ms": existing.get("latest_frame_relative_ts_ms"),
        "latest_audio_relative_ts_ms": existing.get("latest_audio_relative_ts_ms"),
        "timestamp_warnings": int(existing.get("timestamp_warnings", 0) or 0),
        "warnings": list(existing.get("warnings", []) or [])[-50:],
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
    }
    write_json_atomic(rokid_state_path(session_dir), state)
    return state


def public_rokid_start_block(metadata: dict[str, Any] | None = None, *, input_mode: str = ROKID_INPUT_MODE) -> dict[str, Any]:
    metadata = default_rokid_metadata(metadata, input_mode=input_mode)
    input_mode = str(metadata.get("input_mode") or input_mode or ROKID_INPUT_MODE).strip().lower()
    return {
        "enabled": True,
        "input_mode": input_mode,
        "adapter": "rtmp_srs_live" if input_mode == ROKID_LIVE_RTMP_INPUT_MODE else "phone_sdk_connector",
        "device_type": metadata.get("device_type", "rokid"),
        "transport": metadata.get("transport", "phone_sdk"),
        "expected_video": {
            "preferred_format": "jpeg",
            "accepted_formats": ROKID_ACCEPTED_VIDEO_FORMATS,
            "optional_raw_formats": ROKID_OPTIONAL_RAW_VIDEO_FORMATS,
            "recommended_fps": 1,
            "recommended_width": 640,
            "recommended_quality": 75,
        },
        "expected_audio": {
            "preferred_format": "wav",
            "accepted_formats": ROKID_ACCEPTED_AUDIO_FORMATS,
            "recommended_chunk_ms": 1000,
            "asr_window_ms": _env_int("WORLDMM_AUDIO_ASR_WINDOW_MS", 5000),
        },
        "timestamp": {
            "required": "relative_ts_ms",
            "recommended_clock": "android_elapsed_realtime",
        },
    }


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
    return parsed.astimezone(timezone.utc)


def _server_elapsed_ms(stream_state: dict[str, Any]) -> int:
    start = _parse_time(stream_state.get("started_at") or stream_state.get("created_at"))
    if start is None:
        return 0
    return max(0, int(round((datetime.now(timezone.utc) - start).total_seconds() * 1000)))


def prepare_rokid_relative_ts(
    session_dir: Path,
    stream_state: dict[str, Any],
    *,
    media_type: str,
    relative_ts_ms: int | None,
    timestamp_source: str | None = None,
) -> tuple[int | None, str | None, list[str]]:
    if relative_ts_ms is not None:
        return max(0, int(relative_ts_ms)), timestamp_source or ROKID_TIMESTAMP_MODE, []
    fallback = _server_elapsed_ms(stream_state)
    warning = (
        f"{media_type} relative_ts_ms missing; using server receive elapsed time. "
        "Connector should send SystemClock.elapsedRealtime() - streamStartElapsedMs."
    )
    return fallback, timestamp_source or "server_receive_fallback", [warning]


def record_rokid_media_ingest(
    session_dir: Path,
    *,
    media_type: str,
    source: str,
    relative_ts_ms: int | None,
    accepted: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    state = load_rokid_state(session_dir)
    if not state:
        stream_state = read_json(Path(session_dir) / "stream" / "stream_state.json", default={})
        metadata = _metadata_from_session(session_dir, stream_state if isinstance(stream_state, dict) else {})
        state = initialize_rokid_state(
            session_dir,
            stream_id=str((stream_state or {}).get("stream_id") or ""),
            metadata=metadata,
        )
    state["enabled"] = True
    state["input_mode"] = ROKID_INPUT_MODE
    state["timestamp_mode"] = ROKID_TIMESTAMP_MODE
    if media_type == "frame":
        state["video_source"] = source or state.get("video_source") or ROKID_DEFAULT_VIDEO_SOURCE
        if accepted:
            state["frames_received"] = int(state.get("frames_received", 0) or 0) + 1
            state["latest_frame_relative_ts_ms"] = relative_ts_ms
    elif media_type == "audio":
        state["audio_source"] = source or state.get("audio_source") or ROKID_DEFAULT_AUDIO_SOURCE
        if accepted:
            state["audio_chunks_received"] = int(state.get("audio_chunks_received", 0) or 0) + 1
            state["latest_audio_relative_ts_ms"] = relative_ts_ms
    if warnings:
        rows = list(state.get("warnings", []) or [])
        for message in warnings:
            rows.append({"media_type": media_type, "message": str(message), "created_at": utc_now_iso()})
        state["warnings"] = rows[-50:]
        state["timestamp_warnings"] = int(state.get("timestamp_warnings", 0) or 0) + len(warnings)
    state["updated_at"] = utc_now_iso()
    write_json_atomic(rokid_state_path(session_dir), state)
    return state


def normalize_rokid_frame_upload(
    frame_bytes: bytes,
    *,
    format_hint: str | None = None,
    filename: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[bytes, str | None, dict[str, Any]]:
    del width, height
    fmt = _normalize_format(format_hint) or _normalize_format(Path(str(filename or "")).suffix)
    if fmt == "nv21":
        raise RokidNormalizationError("raw nv21 upload is not enabled; please convert to jpeg on Rokid Connector")
    return frame_bytes, fmt, {}


def normalize_rokid_audio_upload(
    audio_bytes: bytes,
    *,
    format_hint: str | None = None,
    filename: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    sample_width: int | None = None,
    encoding: str | None = None,
) -> tuple[bytes, str | None, dict[str, Any]]:
    fmt = _normalize_format(format_hint) or _normalize_format(Path(str(filename or "")).suffix)
    if fmt not in {"pcm", "pcm_s16le", "s16le"}:
        return audio_bytes, fmt, {}
    encoding_value = str(encoding or "pcm_s16le").strip().lower()
    if encoding_value not in {"pcm_s16le", "s16le"}:
        raise RokidNormalizationError("raw pcm upload only supports encoding=pcm_s16le; please upload wav from Rokid Connector")
    if int(sample_width or 0) != 2:
        raise RokidNormalizationError("raw pcm upload requires sample_width=2; please upload wav from Rokid Connector")
    if int(sample_rate or 0) <= 0 or int(channels or 0) <= 0:
        raise RokidNormalizationError("raw pcm upload requires sample_rate and channels; please upload wav from Rokid Connector")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(int(channels or 1))
        wav.setsampwidth(int(sample_width or 2))
        wav.setframerate(int(sample_rate or 16000))
        wav.writeframes(audio_bytes)
    return output.getvalue(), "wav", {"normalized_from": fmt, "encoding": encoding_value}


def public_rokid_status_block(session_dir: Path, *, input_mode: Any = None) -> dict[str, Any]:
    session_dir = Path(session_dir)
    stream_state = read_json(session_dir / "stream" / "stream_state.json", default={})
    if not isinstance(stream_state, dict):
        stream_state = {}
    mode = str(input_mode if input_mode is not None else stream_state.get("input_mode") or "").strip().lower()
    if not is_rokid_input_mode(mode):
        return {"enabled": False, "input_mode": mode or None}
    metadata = _metadata_from_session(session_dir, stream_state)
    rokid_state = load_rokid_state(session_dir)
    frame_state = read_json(session_dir / "stream" / "frame_state.json", default={})
    audio_state = read_json(session_dir / "stream" / "audio_state.json", default={})
    live_ingest_state = read_json(session_dir / "stream" / "live_ingest_state.json", default={})
    asr_state = read_json(session_dir / "stream" / "audio_asr_state.json", default={})
    if not isinstance(frame_state, dict):
        frame_state = {}
    if not isinstance(audio_state, dict):
        audio_state = {}
    if not isinstance(live_ingest_state, dict):
        live_ingest_state = {}
    if not isinstance(asr_state, dict):
        asr_state = {}
    warnings = list(rokid_state.get("warnings", []) or [])[-10:]
    return {
        "enabled": True,
        "input_mode": mode,
        "device_type": rokid_state.get("device_type") or metadata.get("device_type", "rokid"),
        "transport": rokid_state.get("transport") or metadata.get("transport", "rtmp_srs" if mode == ROKID_LIVE_RTMP_INPUT_MODE else "phone_sdk"),
        "video_source": rokid_state.get("video_source") or ROKID_DEFAULT_VIDEO_SOURCE,
        "audio_source": rokid_state.get("audio_source") or ROKID_DEFAULT_AUDIO_SOURCE,
        "frames_received": max(int(frame_state.get("received_count", rokid_state.get("frames_received", 0)) or 0), int(live_ingest_state.get("frames_ingested", 0) or 0)),
        "audio_chunks_received": max(int(audio_state.get("received_count", rokid_state.get("audio_chunks_received", 0)) or 0), int(live_ingest_state.get("audio_chunks_ingested", 0) or 0)),
        "latest_frame_relative_ts_ms": frame_state.get("latest_relative_ts_ms", rokid_state.get("latest_frame_relative_ts_ms")) or live_ingest_state.get("latest_frame_relative_ts_ms"),
        "latest_audio_relative_ts_ms": audio_state.get("latest_relative_ts_ms", rokid_state.get("latest_audio_relative_ts_ms")) or live_ingest_state.get("latest_audio_relative_ts_ms"),
        "audio_asr_window_ms": asr_state.get("window_ms") or _env_int("WORLDMM_AUDIO_ASR_WINDOW_MS", 5000),
        "timestamp_mode": rokid_state.get("timestamp_mode") or ROKID_TIMESTAMP_MODE,
        "timestamp_warnings": int(rokid_state.get("timestamp_warnings", 0) or 0),
        "warnings": warnings,
    }


def _normalize_format(value: Any) -> str | None:
    text = str(value or "").strip().lower().lstrip(".")
    return text or None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
