from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from online_preprocess.asr_whisperx import WhisperXRuntime, transcribe_audio_with_whisperx
from online_preprocess.extract_audio import extract_audio_wav
from online_preprocess.io_utils import ffmpeg_bin, read_json, run_command, utc_now_iso
from online_pipeline.audio_stream import AudioStreamStore
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.video_probe import probe_video
from online_streaming.partial_transcript_store import PartialTranscriptStore
from online_streaming.transcript_backfill import TranscriptBackfiller
from online_short_term.stream_chunk_manager import StreamChunkManager


_MP4_LIKE_SUFFIXES = {".m4a", ".mp4", ".m4v", ".mov"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _auto_language(value: str | None) -> str | None:
    text = str(value or "").strip()
    return None if not text or text.lower() in {"auto", "none", "null"} else text


def _append_timeline_best_effort(session_dir: Path, event_type: str, **kwargs: Any) -> None:
    try:
        append_timeline_event(session_dir, event_type, **kwargs)
    except Exception as exc:
        print(f"[stream_asr] timeline append failed session_id={session_dir.name} event_type={event_type}: {exc}", flush=True)


def _segment_id(upload_chunk_id: str, start: float, end: float) -> str:
    return f"stream_asr_{upload_chunk_id}_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def _audio_window_segment_id(window_id: str, index: int, start: float, end: float) -> str:
    return f"{window_id}_{index:04d}_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}"


def _processing_chunk_ids_for_segment(processing_chunks: list[dict[str, Any]], start: float, end: float) -> list[str]:
    ids: list[str] = []
    for chunk in processing_chunks:
        try:
            c_start = float(chunk.get("start_time", 0.0) or 0.0)
            c_end = float(chunk.get("end_time", c_start) or c_start)
        except Exception:
            continue
        if max(c_start, start) <= min(c_end, end):
            chunk_id = str(chunk.get("chunk_id") or chunk.get("processing_chunk_id") or "")
            if chunk_id:
                ids.append(chunk_id)
    return ids


def _mock_segments(task: dict[str, Any]) -> list[dict[str, Any]]:
    start = float(task.get("global_start_time", 0.0) or 0.0)
    end = float(task.get("global_end_time", start) or start)
    if end <= start:
        return []
    text = f"Mock stream transcript for upload chunk {task.get('upload_chunk_index')} from {start:.1f}s to {end:.1f}s."
    return [{"start": start, "end": min(end, start + 2.0), "text": text, "speaker": None, "confidence": None, "words": []}]


def _mock_audio_window_segments(task: dict[str, Any]) -> list[dict[str, Any]]:
    start = float(task.get("global_start_time", 0.0) or 0.0)
    end = float(task.get("global_end_time", start) or start)
    if end <= start:
        return []
    text = f"Mock audio transcript from {start:.1f}s to {end:.1f}s."
    return [{"start": 0.0, "end": min(2.0, end - start), "text": text, "speaker": None, "confidence": None, "words": []}]


def _globalize_segments(task: dict[str, Any], local_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    upload_chunk_id = str(task.get("upload_chunk_id") or f"upload_{int(task.get('upload_chunk_index', 0)):06d}")
    upload_chunk_index = int(task.get("upload_chunk_index", 0) or 0)
    stream_id = task.get("stream_id")
    global_start = float(task.get("global_start_time", 0.0) or 0.0)
    global_end = float(task.get("global_end_time", global_start) or global_start)
    processing_chunks = [dict(item) for item in task.get("processing_chunks", []) or [] if isinstance(item, dict)]
    normalized: list[dict[str, Any]] = []
    for segment in local_segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        local_start = max(0.0, float(segment.get("start", 0.0) or 0.0))
        local_end = max(local_start, float(segment.get("end", local_start) or local_start))
        start = max(global_start, global_start + local_start)
        end = min(global_end, global_start + local_end) if global_end > global_start else global_start + local_end
        if end <= start:
            continue
        words = []
        for word in segment.get("words", []) or []:
            if not isinstance(word, dict):
                continue
            word_item = dict(word)
            if word_item.get("start") is not None:
                word_item["start"] = round(global_start + float(word_item["start"]), 3)
            if word_item.get("end") is not None:
                word_item["end"] = round(global_start + float(word_item["end"]), 3)
            words.append(word_item)
        normalized.append(
            {
                "segment_id": _segment_id(upload_chunk_id, start, end),
                "session_id": task.get("session_id"),
                "stream_id": stream_id,
                "upload_chunk_id": upload_chunk_id,
                "upload_chunk_index": upload_chunk_index,
                "source_processing_chunk_ids": _processing_chunk_ids_for_segment(processing_chunks, start, end),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "speaker": segment.get("speaker"),
                "confidence": segment.get("confidence"),
                "words": words,
                "asr_backend": str(task.get("asr_backend") or "whisperx"),
                "version": 1,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        )
    return normalized


def _globalize_audio_window_segments(task: dict[str, Any], local_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    window_id = str(task.get("window_id") or "audio_asr_window")
    stream_id = task.get("stream_id")
    window_start = float(task.get("window_start_ms", 0) or 0) / 1000.0
    window_end = float(task.get("window_end_ms", 0) or 0) / 1000.0
    normalized: list[dict[str, Any]] = []
    for idx, segment in enumerate(local_segments):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        local_start = max(0.0, float(segment.get("start", 0.0) or 0.0))
        local_end = max(local_start, float(segment.get("end", local_start) or local_start))
        start = window_start + local_start
        end = min(window_end, window_start + local_end) if window_end > window_start else window_start + local_end
        if end <= start:
            continue
        words = []
        for word in segment.get("words", []) or []:
            if not isinstance(word, dict):
                continue
            word_item = dict(word)
            if word_item.get("start") is not None:
                word_item["start"] = round(window_start + float(word_item["start"]), 3)
            if word_item.get("end") is not None:
                word_item["end"] = round(window_start + float(word_item["end"]), 3)
            words.append(word_item)
        normalized.append(
            {
                "segment_id": _audio_window_segment_id(window_id, idx, start, end),
                "session_id": task.get("session_id"),
                "stream_id": stream_id,
                "source": "audio_chunk_asr",
                "window_id": window_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "start_ms": int(round(start * 1000)),
                "end_ms": int(round(end * 1000)),
                "text": text,
                "speaker": segment.get("speaker"),
                "confidence": segment.get("confidence"),
                "words": words,
                "asr_backend": str(task.get("asr_backend") or "whisperx"),
                "version": 1,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        )
    return normalized


def _quote_concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _is_whisperx_empty_window_error(exc: Exception) -> bool:
    return isinstance(exc, IndexError) and "list index out of range" in str(exc).lower()


def _transcribe_audio_with_empty_window_guard(**kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
    try:
        return transcribe_audio_with_whisperx(**kwargs), False
    except Exception as exc:
        if _is_whisperx_empty_window_error(exc):
            return [], True
        raise


def _transcode_audio_chunk_to_wav(input_path: Path, output_path: Path, *, force: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ],
        description=f"transcode audio chunk for rolling ASR: {input_path.name}",
    )
    return output_path


def _is_mp4_like_path(path: Path) -> bool:
    return path.suffix.lower() in _MP4_LIKE_SUFFIXES


def _iter_mp4_top_level_boxes(data: bytes) -> list[tuple[bytes, int, int]]:
    boxes: list[tuple[bytes, int, int]] = []
    offset = 0
    length = len(data)
    while offset + 8 <= length:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > length:
                break
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = length - offset
        if size < header_size or offset + size > length:
            break
        boxes.append((box_type, offset, offset + size))
        offset += size
    return boxes


def _mp4_init_prefix(data: bytes) -> bytes | None:
    saw_ftyp = False
    saw_moov = False
    prefix_end = 0
    for box_type, _start, end in _iter_mp4_top_level_boxes(data):
        if box_type == b"ftyp":
            saw_ftyp = True
            prefix_end = max(prefix_end, end)
        elif box_type == b"moov":
            saw_moov = True
            prefix_end = max(prefix_end, end)
        elif saw_ftyp and saw_moov and box_type in {b"moof", b"mdat"}:
            break
        elif not saw_moov:
            prefix_end = max(prefix_end, end)
        if saw_ftyp and saw_moov:
            return data[:prefix_end]
    return None


def _read_mp4_init_prefix(path: Path) -> bytes | None:
    try:
        return _mp4_init_prefix(path.read_bytes())
    except OSError:
        return None


def _find_mp4_init_prefix(paths: list[Path]) -> tuple[bytes | None, Path | None]:
    for path in paths:
        prefix = _read_mp4_init_prefix(path)
        if prefix:
            return prefix, path
    for directory in sorted({path.parent for path in paths}):
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not _is_mp4_like_path(path):
                continue
            prefix = _read_mp4_init_prefix(path)
            if prefix:
                return prefix, path
    return None, None


def _build_fragmented_mp4_audio_window_wav(session_dir: Path, task: dict[str, Any], abs_paths: list[Path], output_path: Path, *, force: bool = False) -> Path:
    if not abs_paths or not all(_is_mp4_like_path(path) for path in abs_paths):
        raise ValueError("fragmented mp4 audio fallback requires only mp4-like source chunks")
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    first_prefix = _read_mp4_init_prefix(abs_paths[0])
    init_prefix = first_prefix
    init_source: Path | None = abs_paths[0] if first_prefix else None
    prepend_init = False
    if not init_prefix:
        init_prefix, init_source = _find_mp4_init_prefix(abs_paths)
        prepend_init = bool(init_prefix)
    if not init_prefix:
        raise ValueError("fragmented mp4 audio fallback could not find ftyp/moov init metadata")
    source_suffix = abs_paths[0].suffix.lower() if abs_paths[0].suffix.lower() in _MP4_LIKE_SUFFIXES else ".mp4"
    window_source = parts_dir / f"{output_path.stem}_window_source{source_suffix}"
    if force or not window_source.exists():
        with window_source.open("wb") as handle:
            if prepend_init:
                handle.write(init_prefix)
            for path in abs_paths:
                data = path.read_bytes()
                if prepend_init:
                    prefix = _mp4_init_prefix(data)
                    if prefix:
                        data = data[len(prefix) :]
                if data:
                    handle.write(data)
    run_command(
        [
            ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(window_source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ],
        description=f"build fragmented mp4 rolling audio ASR window: {str(task.get('window_id') or output_path.stem)} from {init_source.name if init_source else 'unknown'} init",
    )
    return output_path


def _build_audio_window_wav(session_dir: Path, task: dict[str, Any], *, force: bool = False) -> Path:
    output_rel = str(task.get("output_audio_path") or "")
    if not output_rel:
        window_id = str(task.get("window_id") or "audio_asr_window")
        output_rel = f"stream/audio_asr/windows/{window_id}.wav"
    output_path = session_dir / output_rel
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rel_paths = [str(item) for item in task.get("audio_chunk_paths", []) or [] if str(item or "").strip()]
    if not rel_paths:
        raise FileNotFoundError("audio ASR window has no audio_chunk_paths")
    abs_paths = []
    for rel in rel_paths:
        path = session_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"audio chunk not found: {path}")
        abs_paths.append(path.resolve())
    # Browser MediaRecorder commonly emits small webm/opus chunks. Normalize
    # every source chunk first so concat remains stable across mixed formats.
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    part_paths: list[Path] = []
    try:
        for idx, path in enumerate(abs_paths):
            part_path = parts_dir / f"part_{idx:04d}.wav"
            part_paths.append(_transcode_audio_chunk_to_wav(path, part_path, force=force))
        concat_path = output_path.with_suffix(".concat.txt")
        concat_path.write_text("\n".join(f"file '{_quote_concat_path(path)}'" for path in part_paths) + "\n", encoding="utf-8")
        run_command(
            [
                ffmpeg_bin(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            description="build rolling audio ASR window",
        )
    except Exception:
        if not all(_is_mp4_like_path(path) for path in abs_paths):
            raise
        return _build_fragmented_mp4_audio_window_wav(session_dir, task, abs_paths, output_path, force=force)
    return output_path


def _process_audio_window_asr_task(
    *,
    project_root: Path,
    session_dir: Path,
    task: dict[str, Any],
    asr_runtime: WhisperXRuntime,
    whisperx_model: str,
    device: str,
    compute_type: str,
    language: str | None,
    model_dir: str | None,
    align_model_dir: str | None,
    force: bool,
) -> dict[str, Any]:
    window_id = str(task.get("window_id") or "")
    if not window_id:
        raise ValueError("audio_chunk_window stream_asr task missing window_id")
    store = AudioStreamStore(session_dir)
    store.mark_asr_window_started(window_id)
    stream_id = str(task.get("stream_id") or "")
    backend = (os.getenv("WORLDMM_AUDIO_ASR_BACKEND") or str(task.get("asr_backend") or "whisperx")).strip().lower()
    task = {**task, "asr_backend": backend}
    transcript_dir = session_dir / "stream" / "transcript" / window_id
    transcript_dir.mkdir(parents=True, exist_ok=True)
    no_audio = False
    if backend == "mock":
        local_segments = _mock_audio_window_segments(task)
    else:
        audio_path = _build_audio_window_wav(session_dir, task, force=force)
        local_segments, whisperx_no_audio = _transcribe_audio_with_empty_window_guard(
            audio_path=audio_path,
            output_srt=transcript_dir / "transcript.srt",
            output_json=transcript_dir / "transcript.json",
            model_name=whisperx_model,
            device=device,
            compute_type=compute_type,
            language=_auto_language(os.getenv("WORLDMM_AUDIO_ASR_LANGUAGE") or language),
            model_dir=model_dir,
            align_model_dir=align_model_dir,
            batch_size=int(os.getenv("WORLDMM_WHISPERX_BATCH_SIZE", "16") or 16),
            force=force,
            runtime=asr_runtime,
        )
        no_audio = whisperx_no_audio
    segments = _globalize_audio_window_segments(task, local_segments)
    if not segments and backend != "mock":
        no_audio = True
    partial_store = PartialTranscriptStore(session_dir)
    append_result = partial_store.append_segments(segments, upload_chunk_index=None, stream_id=stream_id)
    state = partial_store.mark_audio_window_processed(window_id, stream_id=stream_id, segment_count=len(segments))
    backfill_result = TranscriptBackfiller(
        session_dir,
        project_root=Path(project_root),
        enqueue_refine=_env_bool("WORLDMM_STREAM_ASR_ENQUEUE_REFINE", True),
        enqueue_consolidation=_env_bool("WORLDMM_STREAM_ASR_ENQUEUE_CONSOLIDATION", True),
    ).backfill_segments(segments, reason="audio_asr_backfill")
    asr_state = store.mark_asr_window_done(
        window_id,
        segment_count=int(state.get("segment_count", len(segments)) or len(segments)),
        latest_transcript_at=state.get("updated_at"),
    )
    _append_timeline_best_effort(
        session_dir,
        "audio_transcript_appended",
        chunk_id=window_id,
        metadata={"segment_count": len(segments), "partial_transcript_path": append_result.get("partial_transcript_path")},
    )
    _append_timeline_best_effort(
        session_dir,
        "audio_transcript_backfilled",
        chunk_id=window_id,
        metadata=backfill_result,
    )
    return {
        "session_id": session_dir.name,
        "stream_id": stream_id,
        "source": "audio_chunk_window",
        "window_id": window_id,
        "backend": backend,
        "no_audio": no_audio,
        "segment_count": len(segments),
        "partial_transcript_state": state,
        "partial_transcript_path": append_result.get("partial_transcript_path"),
        "audio_asr_state": asr_state,
        "backfill": backfill_result,
    }


def process_stream_asr_task(
    *,
    project_root: Path,
    sessions_root: Path,
    task: dict[str, Any],
    asr_runtime: WhisperXRuntime,
    whisperx_model: str,
    device: str,
    compute_type: str,
    language: str | None = None,
    model_dir: str | None = None,
    align_model_dir: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    session_id = str(task.get("session_id") or "")
    session_dir = Path(sessions_root) / session_id
    if str(task.get("source") or "") == "audio_chunk_window":
        return _process_audio_window_asr_task(
            project_root=Path(project_root),
            session_dir=session_dir,
            task=task,
            asr_runtime=asr_runtime,
            whisperx_model=whisperx_model,
            device=device,
            compute_type=compute_type,
            language=language,
            model_dir=model_dir,
            align_model_dir=align_model_dir,
            force=force,
        )
    upload_rel = str(task.get("upload_chunk_path") or "")
    upload_path = session_dir / upload_rel
    if not upload_path.exists():
        raise FileNotFoundError(f"stream upload chunk not found: {upload_path}")

    backend = (os.getenv("WORLDMM_STREAM_ASR_BACKEND") or str(task.get("asr_backend") or "whisperx")).strip().lower()
    stream_id = str(task.get("stream_id") or "")
    upload_chunk_id = str(task.get("upload_chunk_id") or f"upload_{int(task.get('upload_chunk_index', 0)):06d}")
    upload_chunk_index = int(task.get("upload_chunk_index", 0) or 0)
    transcript_dir = session_dir / "stream" / "transcript" / upload_chunk_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    no_audio = False
    if backend == "mock":
        local_or_global = _mock_segments(task)
        segments = [
            {
                **item,
                "segment_id": _segment_id(upload_chunk_id, float(item["start"]), float(item["end"])),
                "session_id": session_id,
                "stream_id": stream_id,
                "upload_chunk_id": upload_chunk_id,
                "upload_chunk_index": upload_chunk_index,
                "source_processing_chunk_ids": _processing_chunk_ids_for_segment(task.get("processing_chunks", []) or [], float(item["start"]), float(item["end"])),
                "asr_backend": "mock",
                "version": 1,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
            for item in local_or_global
        ]
    else:
        probe = probe_video(upload_path, transcript_dir / "probe.json", session_dir, force=force)
        if not bool(probe.get("has_audio")):
            no_audio = True
            segments = []
        else:
            audio_path = extract_audio_wav(
                input_video=upload_path,
                output_wav=transcript_dir / "audio.wav",
                has_audio=True,
                force=force,
            )
            local_segments, whisperx_no_audio = _transcribe_audio_with_empty_window_guard(
                audio_path=audio_path,
                output_srt=transcript_dir / "transcript.srt",
                output_json=transcript_dir / "transcript.json",
                model_name=whisperx_model,
                device=device,
                compute_type=compute_type,
                language=language,
                model_dir=model_dir,
                align_model_dir=align_model_dir,
                batch_size=int(os.getenv("WORLDMM_WHISPERX_BATCH_SIZE", "16") or 16),
                force=force,
                runtime=asr_runtime,
            )
            no_audio = whisperx_no_audio
            segments = _globalize_segments({**task, "asr_backend": "whisperx"}, local_segments)

    store = PartialTranscriptStore(session_dir)
    append_result = store.append_segments(
        segments,
        upload_chunk_index=upload_chunk_index,
        stream_id=stream_id,
    )
    state = store.mark_chunk_processed(upload_chunk_index, stream_id=stream_id, segment_count=len(segments), no_audio=no_audio)
    backfill_result = TranscriptBackfiller(
        session_dir,
        project_root=Path(project_root),
        enqueue_refine=_env_bool("WORLDMM_STREAM_ASR_ENQUEUE_REFINE", True),
        enqueue_consolidation=_env_bool("WORLDMM_STREAM_ASR_ENQUEUE_CONSOLIDATION", True),
    ).backfill_segments(segments, reason="transcript_backfill")
    _update_upload_asr_status(session_dir, upload_chunk_index, status="done", segment_count=len(segments))

    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "upload_chunk_id": upload_chunk_id,
        "upload_chunk_index": upload_chunk_index,
        "backend": backend,
        "no_audio": no_audio,
        "segment_count": len(segments),
        "partial_transcript_state": state,
        "partial_transcript_path": append_result.get("partial_transcript_path"),
        "backfill": backfill_result,
    }


def mark_stream_asr_failed(sessions_root: Path, task: dict[str, Any], error: str) -> None:
    session_dir = Path(sessions_root) / str(task.get("session_id") or "")
    if not session_dir.exists():
        return
    try:
        if str(task.get("source") or "") == "audio_chunk_window":
            window_id = str(task.get("window_id") or "")
            if window_id:
                PartialTranscriptStore(session_dir).mark_audio_window_failed(
                    window_id,
                    error,
                    stream_id=str(task.get("stream_id") or ""),
                )
                AudioStreamStore(session_dir).mark_asr_window_failed(window_id, error)
            return
        PartialTranscriptStore(session_dir).mark_chunk_failed(
            int(task.get("upload_chunk_index", 0) or 0),
            error,
            stream_id=str(task.get("stream_id") or ""),
        )
        _update_upload_asr_status(session_dir, int(task.get("upload_chunk_index", 0) or 0), status="failed", error=error)
    except Exception:
        return


def _update_upload_asr_status(session_dir: Path, upload_chunk_index: int, *, status: str, segment_count: int | None = None, error: str | None = None) -> None:
    manager = StreamChunkManager(session_dir)
    state = manager.load_stream_state(default={})
    upload_chunks = []
    for item in state.get("upload_chunks", state.get("received_chunks", [])) or []:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        if int(item.get("upload_chunk_index", item.get("chunk_index", -1))) == int(upload_chunk_index):
            item["asr_status"] = status
            if status == "done":
                done_at = utc_now_iso()
                item["asr_done_at"] = done_at
                item["asr_processed_at"] = done_at
                item["transcript_backfilled_at"] = done_at
            if segment_count is not None:
                item["asr_segment_count"] = int(segment_count)
            if error:
                item["asr_error"] = error
        upload_chunks.append(item)
    if upload_chunks:
        state["upload_chunks"] = upload_chunks
        state["received_chunks"] = upload_chunks
        manager.save_stream_state(state)
