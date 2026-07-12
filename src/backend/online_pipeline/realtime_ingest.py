from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from online_pipeline.audio_stream import (
    AudioStreamStore,
    audio_codec_from_mime,
    audio_suffix_from_format,
    normalize_audio_mime,
    public_audio_stream_status_block,
    sha256_path as audio_sha256_path,
)
from online_pipeline.frame_stream import (
    ALLOWED_FRAME_SUFFIXES,
    FrameStreamStore,
    detect_frame_suffix,
    frame_stream_input_mode,
    is_frame_stream_mode,
    sha256_path,
)
from online_pipeline.rokid_ingest import is_rokid_input_mode, prepare_rokid_relative_ts, record_rokid_media_ingest
from online_pipeline.runtime_state import refresh_session_pipeline_state
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json
from online_short_term.stream_chunk_manager import StreamChunkManager


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _append_timeline_event_safe(session_dir: Path, event_type: str, **kwargs: Any) -> None:
    try:
        append_timeline_event(session_dir, event_type, **kwargs)
    except Exception as exc:
        print(f"[timeline] append failed session_id={session_dir.name} event_type={event_type}: {exc}", flush=True)


def _session_dir(project_root: Path, session_id: str) -> Path:
    return Path(project_root) / "online_sessions" / session_id


def _valid_session_id(session_id: str) -> bool:
    return bool(session_id) and all(ch.isalnum() or ch in {"-", "_"} for ch in session_id)


def _with_status(payload: dict[str, Any], status_code: int) -> dict[str, Any]:
    payload["_http_status_code"] = int(status_code)
    return payload


def _load_stream_state(session_dir: Path) -> dict[str, Any]:
    try:
        return StreamChunkManager(session_dir).load_stream_state(default={})
    except Exception:
        payload = read_json(session_dir / "stream" / "stream_state.json", default={})
        return payload if isinstance(payload, dict) else {}


def _is_live_adapter_mode(mode: str) -> bool:
    return mode in {"live_pusher_rtmp", "web_webrtc_whip", "rokid_live_rtmp"}


def ingest_frame(
    project_root: Path,
    session_id: str,
    frame_bytes: bytes,
    *,
    frame_index: int,
    client_ts_ms: int | None = None,
    relative_ts_ms: int | None = None,
    source_ts_ms: int | None = None,
    timestamp_source: str | None = None,
    width: int | None = None,
    height: int | None = None,
    format: str | None = None,
    source: str = "http_frame_upload",
    input_mode: str = "frame_audio_stream",
    filename_hint: str | None = None,
    update_mcur: bool = True,
    update_mst: bool = True,
    allow_live_input: bool = False,
) -> dict[str, Any]:
    """Ingest one realtime image frame without depending on FastAPI UploadFile.

    Future WebRTC/RTMP receivers should call this after extracting image bytes
    and timeline metadata. This function keeps the existing frame_audio_stream
    M_cur/M_st behavior and does not call LLM/VLM/query.
    """

    project_root = Path(project_root)
    if not _valid_session_id(session_id):
        return _with_status({"status": "error", "message": "invalid session_id"}, 400)
    try:
        frame_index = int(frame_index)
    except Exception:
        return _with_status({"status": "error", "message": "frame_index must be an integer"}, 400)
    if frame_index < 0:
        return _with_status({"status": "error", "message": "frame_index must be >= 0"}, 400)
    session_dir = _session_dir(project_root, session_id)
    if not session_dir.exists():
        return _with_status({"status": "error", "message": f"session not found: {session_id}"}, 404)
    stream_state = _load_stream_state(session_dir)
    if not stream_state:
        return _with_status({"status": "error", "message": "stream not started"}, 409)
    mode = frame_stream_input_mode(stream_state.get("input_mode") or input_mode)
    if not is_frame_stream_mode(mode) and not (allow_live_input and _is_live_adapter_mode(mode)):
        return _with_status(
            {
                "status": "error",
                "message": "frame upload requires input_mode=frame_audio_stream or rokid_frame_audio",
                "input_mode": mode,
            },
            409,
        )
    if str(stream_state.get("status") or "") in {"ending", "ended"}:
        return _with_status(
            {
                "status": "stream_not_accepting_frames",
                "message": "stream is ending or ended",
                "session_id": session_id,
                "stream_status": stream_state.get("status"),
            },
            409,
        )
    max_bytes = int(os.getenv("EM2MEM_FRAME_STREAM_MAX_BYTES", "524288") or 524288)
    size_bytes = len(frame_bytes or b"")
    if size_bytes <= 0:
        return _with_status({"status": "error", "message": "uploaded frame is empty"}, 400)
    if size_bytes > max_bytes:
        return _with_status({"status": "error", "message": f"frame exceeds EM2MEM_FRAME_STREAM_MAX_BYTES={max_bytes}"}, 400)
    rokid_warnings: list[str] = []
    if is_rokid_input_mode(mode):
        relative_ts_ms, timestamp_source, rokid_warnings = prepare_rokid_relative_ts(
            session_dir,
            stream_state,
            media_type="frame",
            relative_ts_ms=relative_ts_ms,
            timestamp_source=timestamp_source,
        )

    store_mode = "frame_audio_stream" if _is_live_adapter_mode(mode) else mode
    store = FrameStreamStore(session_dir)
    store.initialize(stream_id=str(stream_state.get("stream_id") or ""), input_mode=store_mode)
    tmp_path = store.tmp_dir / f"frame_{frame_index:06d}_{uuid4().hex[:8]}.tmp"
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(frame_bytes)
        detected_suffix = detect_frame_suffix(tmp_path)
        if detected_suffix not in ALLOWED_FRAME_SUFFIXES:
            return _with_status({"status": "error", "message": "unsupported image content; expected jpg, jpeg, png, or webp"}, 400)
        if format and f".{str(format).strip().lower().lstrip('.')}" not in ALLOWED_FRAME_SUFFIXES:
            return _with_status({"status": "error", "message": "unsupported format field; expected jpg, jpeg, png, or webp"}, 400)
        registration = store.register_frame(
            tmp_path=tmp_path,
            frame_index=frame_index,
            checksum=sha256_path(tmp_path),
            suffix=detected_suffix,
            size_bytes=size_bytes,
            client_ts_ms=client_ts_ms,
            relative_ts_ms=relative_ts_ms,
            source_ts_ms=source_ts_ms,
            timestamp_source=timestamp_source,
            width=width,
            height=height,
            source=str(source or "http_frame_upload"),
        )
        frame_state = registration.get("state") if isinstance(registration.get("state"), dict) else {}
        response_status = str(registration.get("status") or "frame_received")
        if is_rokid_input_mode(mode) and response_status != "frame_received" and rokid_warnings:
            record_rokid_media_ingest(
                session_dir,
                media_type="frame",
                source=str(source or "rokid_sdk_video"),
                relative_ts_ms=frame_state.get("latest_relative_ts_ms", relative_ts_ms),
                accepted=False,
                warnings=rokid_warnings,
            )
        if response_status != "frame_received":
            timeline_event_type = {
                "duplicate_ignored": "frame_duplicate_ignored",
                "outdated_dropped": "frame_outdated_dropped",
                "ignored_conflict": "frame_conflict_ignored",
            }.get(response_status, response_status)
            _append_timeline_event_safe(
                session_dir,
                timeline_event_type,
                chunk_index=frame_index,
                chunk_id=f"frame_{frame_index:06d}",
                metadata={"input_mode": mode},
            )
            return _with_status(
                {
                    "status": response_status,
                    "session_id": session_id,
                    "stream_id": stream_state.get("stream_id"),
                    "frame_index": frame_index,
                    "latest_frame_index": frame_state.get("latest_frame_index"),
                    "mcur_ready": bool(frame_state.get("mcur_ready")),
                    "mcur_version": frame_state.get("mcur_version", 0),
                    "can_ask": bool(frame_state.get("mcur_ready")),
                    "warnings": rokid_warnings,
                },
                409 if response_status == "ignored_conflict" else 200,
            )

        frame_record = registration.get("frame") if isinstance(registration.get("frame"), dict) else {}
        memory_accepted = bool(frame_record.get("memory_accepted"))
        mcur_error = None
        frame_mst_error = None
        frame_mst_result = None
        if update_mcur and memory_accepted:
            try:
                from online_current.mcur_store import MCurStore

                mcur_result = MCurStore(session_dir).update_from_frame_stream(
                    frame_index=frame_index,
                    frame_path=str(frame_record.get("saved_path") or ""),
                    relative_ts_ms=int(frame_record.get("relative_ts_ms", 0) or 0),
                    client_ts_ms=client_ts_ms,
                    source=str(source or "http_frame_upload"),
                )
                frame_state = store.mark_mcur_updated(
                    frame_index=frame_index,
                    current_frame_path=str(mcur_result.get("current_frame_path") or "") or None,
                    mcur_state=mcur_result,
                )
                refresh_session_pipeline_state(session_dir)
            except Exception as exc:
                mcur_error = str(exc)
                frame_state = store.load()
                print(f"[frame_stream] M_cur update failed session_id={session_id} frame_index={frame_index}: {exc}", flush=True)
                _append_timeline_event_safe(
                    session_dir,
                    "error",
                    chunk_index=frame_index,
                    chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                    metadata={"stage": "mcur_updated_from_frame", "error": mcur_error},
                )
        if update_mst and memory_accepted:
            try:
                if _env_bool("EM2MEM_FRAME_STREAM_ENABLE_MST", True):
                    from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder

                    frame_mst_result = FrameStreamMicroEventBuilder(session_dir).process_frame(
                        frame_record=frame_record,
                        current_frame_path=str(frame_state.get("latest_current_frame_path") or "") or None,
                        project_root=project_root,
                        enqueue_refine=_env_bool("EM2MEM_FRAME_STREAM_ENQUEUE_REFINE", True),
                    )
                    refresh_session_pipeline_state(session_dir)
                    _append_timeline_event_safe(
                        session_dir,
                        "frame_mst_updated",
                        chunk_index=frame_index,
                        chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                        metadata={
                            "diff_score": frame_mst_result.get("diff_score"),
                            "has_open_event": frame_mst_result.get("has_open_event"),
                            "closed_event_count": frame_mst_result.get("closed_event_count", 0),
                        },
                    )
                    if int(frame_mst_result.get("opened_event_count", 0) or 0) > 0:
                        _append_timeline_event_safe(
                            session_dir,
                            "frame_mst_opened",
                            chunk_index=frame_index,
                            chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                            metadata={"opened_events": frame_mst_result.get("opened_events", [])},
                        )
                    if int(frame_mst_result.get("closed_event_count", 0) or 0) > 0:
                        _append_timeline_event_safe(
                            session_dir,
                            "frame_mst_closed",
                            chunk_index=frame_index,
                            chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                            metadata={
                                "closed_event_ids": frame_mst_result.get("closed_event_ids", []),
                                "refine_task_paths": frame_mst_result.get("refine_task_paths", []),
                            },
                        )
                        if frame_mst_result.get("refine_task_paths"):
                            _append_timeline_event_safe(
                                session_dir,
                                "frame_mst_refine_queued",
                                chunk_index=frame_index,
                                chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                                metadata={"refine_task_paths": frame_mst_result.get("refine_task_paths", [])},
                            )
            except Exception as exc:
                frame_mst_error = str(exc)
                print(f"[frame_stream] M_st update failed session_id={session_id} frame_index={frame_index}: {exc}", flush=True)
                _append_timeline_event_safe(
                    session_dir,
                    "frame_mst_error",
                    chunk_index=frame_index,
                    chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                    metadata={"error": frame_mst_error},
                )
        if memory_accepted:
            _append_timeline_event_safe(
                session_dir,
                "frame_received",
                chunk_index=frame_index,
                chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                metadata={
                    "relative_ts_ms": frame_record.get("relative_ts_ms"),
                    "saved_path": frame_record.get("saved_path"),
                    "mcur_version": frame_state.get("mcur_version"),
                    "source": source,
                    "filename_hint": filename_hint,
                    "memory_accepted": True,
                },
            )
        if mcur_error is None and update_mcur and memory_accepted:
            _append_timeline_event_safe(
                session_dir,
                "mcur_updated_from_frame",
                chunk_index=frame_index,
                chunk_id=str(frame_record.get("frame_id") or f"frame_{frame_index:06d}"),
                metadata={
                    "current_frame_path": frame_state.get("latest_current_frame_path"),
                    "mcur_version": frame_state.get("mcur_version"),
                },
            )
        if is_rokid_input_mode(mode):
            record_rokid_media_ingest(
                session_dir,
                media_type="frame",
                source=str(source or "rokid_sdk_video"),
                relative_ts_ms=frame_record.get("relative_ts_ms", relative_ts_ms),
                accepted=True,
                warnings=rokid_warnings,
            )
        return _with_status(
            {
                "status": "frame_received",
                "session_id": session_id,
                "stream_id": stream_state.get("stream_id"),
                "frame_index": frame_index,
                "frame_id": frame_record.get("frame_id"),
                "client_ts_ms": client_ts_ms,
                "relative_ts_ms": frame_record.get("relative_ts_ms"),
                "source_ts_ms": frame_record.get("source_ts_ms"),
                "timestamp_source": frame_record.get("timestamp_source"),
                "size_bytes": size_bytes,
                "saved_path": frame_record.get("saved_path"),
                "current_frame_path": frame_state.get("latest_current_frame_path"),
                "memory_accepted": memory_accepted,
                "memory_accepted_count": frame_state.get("memory_accepted_count", 0),
                "preview_received_count": frame_state.get("preview_received_count", frame_state.get("received_count", 0)),
                "mcur_ready": bool(frame_state.get("mcur_ready")),
                "mcur_version": frame_state.get("mcur_version", 0),
                "latest_frame_index": frame_state.get("latest_frame_index"),
                "can_ask": bool(frame_state.get("mcur_ready")),
                "mcur_error": mcur_error,
                "frame_mst": frame_mst_result,
                "frame_mst_error": frame_mst_error,
                "warnings": rokid_warnings,
            },
            200,
        )
    except Exception as exc:
        return _with_status({"status": "error", "message": str(exc), "session_id": session_id, "frame_index": frame_index}, 500)
    finally:
        tmp_path.unlink(missing_ok=True)


def ingest_audio_chunk(
    project_root: Path,
    session_id: str,
    audio_bytes: bytes,
    *,
    audio_index: int,
    client_ts_ms: int | None = None,
    relative_ts_ms: int | None = None,
    source_ts_ms: int | None = None,
    timestamp_source: str | None = None,
    duration_ms: int | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    format: str | None = None,
    content_type: str | None = None,
    source: str = "http_audio_upload",
    input_mode: str = "frame_audio_stream",
    filename_hint: str | None = None,
    enqueue_asr: bool = True,
    allow_live_input: bool = False,
) -> dict[str, Any]:
    """Ingest one realtime audio chunk without depending on FastAPI UploadFile."""

    project_root = Path(project_root)
    if not _valid_session_id(session_id):
        return _with_status({"status": "error", "message": "invalid session_id"}, 400)
    try:
        audio_index = int(audio_index)
    except Exception:
        return _with_status({"status": "error", "message": "audio_index must be an integer"}, 400)
    if audio_index < 0:
        return _with_status({"status": "error", "message": "audio_index must be >= 0"}, 400)
    if not _env_bool("EM2MEM_AUDIO_STREAM_ENABLED", True):
        return _with_status(
            {
                "status": "audio_stream_disabled",
                "message": "audio stream is disabled by EM2MEM_AUDIO_STREAM_ENABLED=0",
                "session_id": session_id,
            },
            409,
        )
    session_dir = _session_dir(project_root, session_id)
    if not session_dir.exists():
        return _with_status({"status": "error", "message": f"session not found: {session_id}"}, 404)
    stream_state = _load_stream_state(session_dir)
    if not stream_state:
        return _with_status({"status": "error", "message": "stream not started"}, 409)
    mode = frame_stream_input_mode(stream_state.get("input_mode") or input_mode)
    if not is_frame_stream_mode(mode) and not (allow_live_input and _is_live_adapter_mode(mode)):
        return _with_status(
            {
                "status": "error",
                "message": "audio_chunk upload requires input_mode=frame_audio_stream or rokid_frame_audio",
                "input_mode": mode,
            },
            409,
        )
    if str(stream_state.get("status") or "") in {"ending", "ended"}:
        return _with_status(
            {
                "status": "stream_not_accepting_audio_chunks",
                "message": "stream is ending or ended",
                "session_id": session_id,
                "stream_status": stream_state.get("status"),
            },
            409,
        )
    suffix = audio_suffix_from_format(format, filename_hint, content_type)
    if suffix is None:
        return _with_status({"status": "error", "message": "unsupported audio format; expected mp3, aac, m4a, wav, pcm, webm, opus, or ogg"}, 400)
    mime_type = str(content_type or "").strip() or None
    mime_base = normalize_audio_mime(content_type)
    codec = audio_codec_from_mime(content_type)
    max_bytes = int(os.getenv("EM2MEM_AUDIO_CHUNK_MAX_BYTES", "1048576") or 1048576)
    size_bytes = len(audio_bytes or b"")
    if size_bytes <= 0:
        return _with_status({"status": "error", "message": "uploaded audio chunk is empty", "session_id": session_id, "audio_index": audio_index}, 400)
    if size_bytes > max_bytes:
        return _with_status(
            {
                "status": "error",
                "message": f"audio chunk exceeds EM2MEM_AUDIO_CHUNK_MAX_BYTES={max_bytes}",
                "session_id": session_id,
                "audio_index": audio_index,
            },
            400,
        )
    rokid_warnings: list[str] = []
    if is_rokid_input_mode(mode):
        relative_ts_ms, timestamp_source, rokid_warnings = prepare_rokid_relative_ts(
            session_dir,
            stream_state,
            media_type="audio",
            relative_ts_ms=relative_ts_ms,
            timestamp_source=timestamp_source,
        )

    store_mode = "frame_audio_stream" if _is_live_adapter_mode(mode) else mode
    store = AudioStreamStore(session_dir)
    store.initialize(stream_id=str(stream_state.get("stream_id") or ""), input_mode=store_mode)
    tmp_path = store.tmp_dir / f"audio_{audio_index:06d}_{uuid4().hex[:8]}.tmp"
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(audio_bytes)
        registration = store.register_audio_chunk(
            tmp_path=tmp_path,
            audio_index=audio_index,
            checksum=audio_sha256_path(tmp_path),
            suffix=suffix,
            size_bytes=size_bytes,
            client_ts_ms=client_ts_ms,
            relative_ts_ms=relative_ts_ms,
            source_ts_ms=source_ts_ms,
            timestamp_source=timestamp_source,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            source=str(source or "http_audio_upload"),
            mime_type=mime_type or mime_base,
            codec=codec,
        )
        audio_state = registration.get("state") if isinstance(registration.get("state"), dict) else {}
        audio_record = registration.get("audio") if isinstance(registration.get("audio"), dict) else {}
        response_status = str(registration.get("status") or "audio_chunk_received")
        asr_enqueue = {"enabled": False, "enqueued": []}
        if response_status == "audio_chunk_received" and enqueue_asr:
            try:
                asr_enqueue = store.maybe_enqueue_asr_windows(
                    project_root=project_root,
                    stream_id=str(stream_state.get("stream_id") or ""),
                )
            except Exception as exc:
                asr_enqueue = {"enabled": _env_bool("EM2MEM_AUDIO_ASR_ENABLED", True), "enqueued": [], "error": str(exc)}
                print(f"[audio_stream] rolling ASR enqueue failed session_id={session_id} audio_index={audio_index}: {exc}", flush=True)
        timeline_event_type = {
            "audio_chunk_received": "audio_chunk_received",
            "duplicate_ignored": "audio_duplicate_ignored",
            "outdated_dropped": "audio_outdated_dropped",
            "ignored_conflict": "audio_ignored_conflict",
        }.get(response_status, response_status)
        _append_timeline_event_safe(
            session_dir,
            timeline_event_type,
            chunk_index=audio_index,
            chunk_id=audio_record.get("audio_id") or f"audio_{audio_index:06d}",
            metadata={
                "input_mode": mode,
                "relative_ts_ms": relative_ts_ms,
                "source_ts_ms": source_ts_ms,
                "timestamp_source": timestamp_source,
                "duration_ms": duration_ms,
                "size_bytes": size_bytes,
                "saved_path": audio_record.get("path"),
                "source": source,
                "filename_hint": filename_hint,
            },
        )
        if is_rokid_input_mode(mode):
            record_rokid_media_ingest(
                session_dir,
                media_type="audio",
                source=str(source or "rokid_sdk_audio"),
                relative_ts_ms=audio_record.get("relative_ts_ms", relative_ts_ms),
                accepted=response_status == "audio_chunk_received",
                warnings=rokid_warnings,
            )
        return _with_status(
            {
                "status": response_status,
                "session_id": session_id,
                "stream_id": stream_state.get("stream_id"),
                "audio_index": audio_index,
                "audio_id": audio_record.get("audio_id") or f"audio_{audio_index:06d}",
                "client_ts_ms": client_ts_ms,
                "relative_ts_ms": audio_record.get("relative_ts_ms", relative_ts_ms),
                "source_ts_ms": audio_record.get("source_ts_ms", source_ts_ms),
                "timestamp_source": audio_record.get("timestamp_source", timestamp_source),
                "duration_ms": audio_record.get("duration_ms", duration_ms),
                "sample_rate": sample_rate,
                "channels": channels,
                "format": audio_record.get("format") or suffix.lstrip("."),
                "mime_type": audio_record.get("mime_type") or mime_type,
                "codec": audio_record.get("codec") or codec,
                "size_bytes": size_bytes,
                "saved_path": audio_record.get("path"),
                "audio_ready": bool(audio_state.get("ready")),
                "latest_audio_index": audio_state.get("latest_audio_index"),
                "received_count": audio_state.get("received_count", 0),
                "accepted_count": audio_state.get("accepted_count", 0),
                "rolling_buffer_ready": bool(audio_state.get("rolling_buffer_ready")),
                "asr_ready": False,
                "asr_status": (asr_enqueue or {}).get("asr_status") or audio_state.get("asr_status", "not_started"),
                "asr_enqueue": asr_enqueue,
                "audio_stream": public_audio_stream_status_block(session_dir, input_mode=mode),
                "warnings": rokid_warnings,
            },
            409 if response_status == "ignored_conflict" else 200,
        )
    except Exception as exc:
        _append_timeline_event_safe(
            session_dir,
            "audio_chunk_error",
            chunk_index=audio_index,
            chunk_id=f"audio_{audio_index:06d}",
            metadata={"error": str(exc)},
        )
        return _with_status({"status": "error", "message": str(exc), "session_id": session_id, "audio_index": audio_index}, 500)
    finally:
        tmp_path.unlink(missing_ok=True)
