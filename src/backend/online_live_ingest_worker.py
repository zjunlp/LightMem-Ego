from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from online_pipeline.live_ingest import (
    choose_effective_live_pull_url,
    load_live_ingest_state,
    update_live_ingest_state,
    update_live_source_ingest,
)
from online_pipeline.realtime_ingest import ingest_audio_chunk, ingest_frame
from online_pipeline.runtime_state import WorkerTaskHeartbeat, write_worker_runtime
from online_preprocess.io_utils import ffmpeg_bin, ffprobe_bin, read_json, utc_now_iso
from online_preprocess.task_queue import (
    claim_live_ingest_task,
    finish_live_ingest_task,
    list_queued_live_ingest_tasks,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ONLINE_SESSIONS_DIR = PROJECT_ROOT / "online_sessions"

FRAME_RE = re.compile(r"frame_(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)
AUDIO_RE = re.compile(r"audio_(\d+)\.wav$", re.IGNORECASE)
ROKID_LIVE_RTMP_MODE = "rokid_live_rtmp"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _is_rokid_live_rtmp(input_mode: Any) -> bool:
    return str(input_mode or "").strip() == ROKID_LIVE_RTMP_MODE


def _rtmp_video_only(input_mode: Any) -> bool:
    return _is_rokid_live_rtmp(input_mode)


def _live_ingest_rw_timeout_us(input_mode: Any) -> int:
    if _is_rokid_live_rtmp(input_mode):
        return _env_int("EM2MEM_ROKID_LIVE_INGEST_RW_TIMEOUT_US", 30_000_000)
    return _env_int("EM2MEM_LIVE_INGEST_RW_TIMEOUT_US", 30_000_000)


def _ffmpeg_input_options(source_url: str, input_mode: Any) -> list[str]:
    options: list[str] = []
    if str(source_url or "").strip().lower().startswith("rtmp://"):
        options.extend(["-rtmp_live", "live"])
    options.extend(["-rw_timeout", str(_live_ingest_rw_timeout_us(input_mode))])
    return options


def _pid_alive(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except Exception:
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _stable_file(path: Path, min_age_seconds: float = 0.2) -> bool:
    try:
        first = path.stat()
        if first.st_size <= 0:
            return False
        if time.time() - first.st_mtime < min_age_seconds:
            return False
        time.sleep(0.02)
        second = path.stat()
        return first.st_size == second.st_size and second.st_size > 0
    except Exception:
        return False


def _iter_indexed_files(directory: Path, pattern: str, regex: re.Pattern[str]) -> list[tuple[int, Path]]:
    rows: list[tuple[int, Path]] = []
    if not directory.exists():
        return rows
    for path in directory.glob(pattern):
        match = regex.match(path.name)
        if not match:
            continue
        try:
            rows.append((int(match.group(1)), path))
        except Exception:
            continue
    return sorted(rows, key=lambda item: item[0])


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab")


def _media_binary(tool_name: str) -> str:
    resolver = ffmpeg_bin if tool_name == "ffmpeg" else ffprobe_bin
    try:
        return resolver()
    except Exception as exc:
        env_name = f"EM2MEM_{tool_name.upper()}_BIN"
        raise RuntimeError(
            f"{tool_name} executable was not found; set {env_name} to a valid absolute path "
            f"or add {tool_name} to PATH: {exc}"
        ) from exc


def _tail_text(path: Path, max_bytes: int = 6000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _stderr_tail(video_log_path: Path, audio_log_path: Path) -> dict[str, str]:
    return {
        "video": _tail_text(video_log_path),
        "audio": _tail_text(audio_log_path),
    }


def _command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _detected_output_counts(frame_dir: Path, audio_dir: Path) -> tuple[int, int]:
    return (
        len(_iter_indexed_files(frame_dir, "frame_*.jpg", FRAME_RE)),
        len(_iter_indexed_files(audio_dir, "audio_*.wav", AUDIO_RE)),
    )


def _live_ingest_runtime_extra(session_id: str) -> dict[str, Any]:
    session_dir = ONLINE_SESSIONS_DIR / session_id
    state = load_live_ingest_state(session_dir) if session_id else {}
    return {
        "active_sessions": [session_id] if session_id else [],
        "processed_frames": state.get("frames_ingested") if isinstance(state, dict) else None,
        "processed_audio_chunks": state.get("audio_chunks_ingested") if isinstance(state, dict) else None,
        "live_status": state.get("status") if isinstance(state, dict) else None,
        "waiting_reason": state.get("waiting_reason") if isinstance(state, dict) else None,
        "ffmpeg_video_pid": state.get("ffmpeg_video_pid") if isinstance(state, dict) else None,
        "ffmpeg_audio_pid": state.get("ffmpeg_audio_pid") if isinstance(state, dict) else None,
        "last_output_file_at": state.get("last_output_file_at") if isinstance(state, dict) else None,
        "last_pull_error": state.get("last_pull_error") if isinstance(state, dict) else None,
    }


def _now_wall_ms() -> int:
    return int(round(time.time() * 1000))


def _timeline_origin_ns(state: dict[str, Any]) -> int:
    value = state.get("timeline_origin_monotonic_ns")
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return time.monotonic_ns()


def _timeline_rel_ms(state: dict[str, Any], *, kind: str, duration_ms: int = 0, fallback_ms: int = 0) -> tuple[int, dict[str, Any]]:
    """Map a detected live output file onto one shared session timeline.

    The current ffmpeg file-output path does not expose reliable source PTS.
    We therefore use the worker's monotonic origin for both video and audio,
    then clamp small regressions so downstream M_cur/M_st/ASR overlap logic sees
    a single monotonic session-relative timeline.
    """
    origin_ns = _timeline_origin_ns(state)
    raw_ms = int(round((time.monotonic_ns() - origin_ns) / 1_000_000.0)) - max(0, int(duration_ms or 0))
    if raw_ms < 0:
        raw_ms = max(0, int(fallback_ms))
    latest_key = f"latest_{kind}_relative_ts_ms"
    violation_key = f"{kind}_monotonic_violations"
    latest = state.get(latest_key, state.get(f"last_{kind}_relative_ts_ms"))
    clamped = False
    try:
        latest_int = int(latest)
    except Exception:
        latest_int = None
    if latest_int is not None and raw_ms <= latest_int:
        raw_ms = latest_int + 1
        clamped = True
    updates = {
        "timeline_version": 1,
        "timestamp_mode": "shared_monotonic",
        "timestamp_source": "shared_monotonic",
        latest_key: int(raw_ms),
        f"last_{kind}_relative_ts_ms": int(raw_ms),
    }
    if clamped:
        updates[violation_key] = int(state.get(violation_key, 0) or 0) + 1
    return int(raw_ms), updates


def _sync_updates(state: dict[str, Any], *, audio_segment_ms: int) -> dict[str, Any]:
    try:
        frame_ms = int(state.get("latest_frame_relative_ts_ms", state.get("last_frame_relative_ts_ms")))
    except Exception:
        frame_ms = None
    try:
        audio_start_ms = int(state.get("latest_audio_relative_ts_ms", state.get("last_audio_relative_ts_ms")))
    except Exception:
        audio_start_ms = None
    try:
        audio_end_ms = int(state.get("latest_audio_end_relative_ts_ms"))
    except Exception:
        audio_end_ms = None
    if audio_end_ms is None and audio_start_ms is not None:
        audio_end_ms = audio_start_ms + max(0, int(audio_segment_ms or 0))
    updates: dict[str, Any] = {}
    if frame_ms is not None and audio_end_ms is not None:
        skew = int(frame_ms - audio_end_ms)
        updates["av_skew_ms"] = skew
        violations = int(state.get("frame_monotonic_violations", 0) or 0) + int(state.get("audio_monotonic_violations", 0) or 0)
        updates["sync_status"] = "warning" if violations > 0 or abs(skew) > max(5000, 3 * max(1, int(audio_segment_ms or 0))) else "healthy"
    elif frame_ms is not None or audio_start_ms is not None:
        updates["sync_status"] = "unknown"
    else:
        updates["sync_status"] = "unknown"
    return updates


def _next_output_number(directory: Path, pattern: str, regex: re.Pattern[str], state_index: Any) -> int:
    existing = _iter_indexed_files(directory, pattern, regex)
    highest_file_index = existing[-1][0] if existing else -1
    parsed_state_index = int(state_index) if state_index is not None else -1
    return max(highest_file_index, parsed_state_index) + 1


def _probe_command(source_url: str, timeout_seconds: float) -> list[str]:
    return [
        _media_binary("ffprobe"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-rw_timeout",
        str(int(max(1.0, timeout_seconds) * 1_000_000)),
        "-show_entries",
        "stream=index,codec_type,codec_name",
        "-of",
        "json",
        source_url,
    ]


def _probe_source(source_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
    cmd = _probe_command(source_url, timeout_seconds)
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=max(1.0, timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        return False, f"ffprobe timed out after {timeout_seconds:.1f}s"
    except Exception as exc:
        return False, f"ffprobe failed to start: {exc}"
    if result.returncode == 0:
        return True, None
    detail = (result.stderr or result.stdout or f"ffprobe exited with code {result.returncode}").strip()
    return False, detail[-4000:]


def _ffmpeg_video_command(source_url: str, output_dir: Path, start_number: int, frame_fps: float, input_mode: str = "") -> list[str]:
    return [
        _media_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "info",
        *_ffmpeg_input_options(source_url, input_mode),
        "-i",
        source_url,
        "-map",
        "0:v:0?",
        "-an",
        "-vf",
        f"fps={max(0.1, frame_fps)}",
        "-q:v",
        "5",
        "-start_number",
        str(max(0, start_number)),
        str(output_dir / "frame_%06d.jpg"),
    ]


def _start_ffmpeg_video(source_url: str, output_dir: Path, start_number: int, frame_fps: float, input_mode: str = "") -> tuple[subprocess.Popen, Any, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = _open_log(output_dir.parent / "ffmpeg_video_stderr.log")
    cmd = _ffmpeg_video_command(source_url, output_dir, start_number, frame_fps, input_mode)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log_handle), log_handle, cmd


def _ffmpeg_audio_command(source_url: str, output_dir: Path, start_number: int, audio_segment_ms: int, input_mode: str = "") -> list[str]:
    segment_seconds = max(0.25, float(audio_segment_ms) / 1000.0)
    return [
        _media_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "info",
        *_ffmpeg_input_options(source_url, input_mode),
        "-i",
        source_url,
        "-map",
        "0:a:0?",
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "segment",
        "-segment_time",
        f"{segment_seconds:.3f}",
        "-segment_start_number",
        str(max(0, start_number)),
        "-reset_timestamps",
        "1",
        str(output_dir / "audio_%06d.wav"),
    ]


def _start_ffmpeg_audio(source_url: str, output_dir: Path, start_number: int, audio_segment_ms: int, input_mode: str = "") -> tuple[subprocess.Popen, Any, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = _open_log(output_dir.parent / "ffmpeg_audio_stderr.log")
    cmd = _ffmpeg_audio_command(source_url, output_dir, start_number, audio_segment_ms, input_mode)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log_handle), log_handle, cmd


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _close_handles(*handles: Any) -> None:
    for handle in handles:
        try:
            handle.close()
        except Exception:
            pass


def _stream_status(session_dir: Path) -> str:
    payload = read_json(session_dir / "stream" / "stream_state.json", default={})
    if isinstance(payload, dict):
        return str(payload.get("status") or "")
    return ""


def _stop_requested(session_dir: Path) -> bool:
    state = load_live_ingest_state(session_dir)
    return bool(state.get("stop_requested")) or _stream_status(session_dir) in {"ending", "ended"}


def _wait_for_whip_pull(
    session_dir: Path,
    source_url: str,
    *,
    probe_timeout_seconds: float,
    retry_seconds: float,
    last_output_monotonic: float,
    total_no_output_seconds: float,
) -> bool:
    while not _stop_requested(session_dir):
        state = load_live_ingest_state(session_dir)
        attempt_count = int(state.get("pull_attempt_count", 0) or 0) + 1
        probe_ok, probe_error = _probe_source(source_url, probe_timeout_seconds)
        state = update_live_ingest_state(
            session_dir,
            status="waiting_rtmp_output" if probe_ok else "waiting_stream",
            waiting_reason="waiting_for_ffmpeg_output" if probe_ok else "waiting_for_rtmp_pull",
            pull_attempt_count=attempt_count,
            ffprobe_ok=probe_ok,
            ffprobe_cmd=_command_text(_probe_command(source_url, probe_timeout_seconds)),
            last_probe_error=probe_error,
            last_error=None if probe_ok else state.get("last_error"),
        )
        update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_error=state.get("last_error"))
        if probe_ok:
            return True
        if time.monotonic() - last_output_monotonic >= total_no_output_seconds:
            raise RuntimeError(
                f"RTMP pull URL was not ready after {total_no_output_seconds:.0f}s; "
                f"last ffprobe error: {probe_error or 'unknown error'}"
            )
        time.sleep(max(0.25, retry_seconds))
    return False


def _process_new_frames(session_dir: Path, frame_dir: Path, frame_interval_ms: int, audio_segment_ms: int, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    count = 0
    last_index = int(state.get("frame_index")) if state.get("frame_index") is not None else -1
    for idx, path in _iter_indexed_files(frame_dir, "frame_*.jpg", FRAME_RE):
        if idx <= last_index or not _stable_file(path):
            continue
        rel_ms, timeline_updates = _timeline_rel_ms(
            state,
            kind="frame",
            duration_ms=0,
            fallback_ms=int(round(idx * frame_interval_ms)),
        )
        input_mode = str(state.get("input_mode") or "live_pusher_rtmp")
        source = "web_webrtc_whip_video" if input_mode == "web_webrtc_whip" else "srs_rtmp_video"
        result = ingest_frame(
            PROJECT_ROOT,
            session_dir.name,
            path.read_bytes(),
            frame_index=idx,
            relative_ts_ms=rel_ms,
            source_ts_ms=None,
            timestamp_source="shared_monotonic",
            format="jpg",
            source=source,
            input_mode=input_mode,
            filename_hint=path.name,
            allow_live_input=True,
        )
        status = str(result.get("status") or "")
        now = utc_now_iso()
        updates = {
            "status": "running",
            "frame_index": idx,
            "last_frame_relative_ts_ms": rel_ms,
            "latest_frame_relative_ts_ms": rel_ms,
            "last_frame_at": now,
            "last_frame_status": status,
            **timeline_updates,
        }
        if status == "frame_received":
            updates["frames_ingested"] = int(state.get("frames_ingested", 0) or 0) + 1
            count += 1
        elif status == "error":
            updates["last_error"] = result.get("message")
        updates.update(_sync_updates({**state, **updates}, audio_segment_ms=audio_segment_ms))
        state.update(updates)
        state = update_live_ingest_state(session_dir, **updates)
        update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_frame_at=state.get("last_frame_at"), last_error=state.get("last_error"))
        last_index = idx
    return count, state


def _process_new_audio(session_dir: Path, audio_dir: Path, audio_segment_ms: int, frame_interval_ms: int, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    count = 0
    last_index = int(state.get("audio_index")) if state.get("audio_index") is not None else -1
    for idx, path in _iter_indexed_files(audio_dir, "audio_*.wav", AUDIO_RE):
        if idx <= last_index or not _stable_file(path):
            continue
        rel_ms, timeline_updates = _timeline_rel_ms(
            state,
            kind="audio",
            duration_ms=audio_segment_ms,
            fallback_ms=int(idx * audio_segment_ms),
        )
        input_mode = str(state.get("input_mode") or "live_pusher_rtmp")
        source = "web_webrtc_whip_audio" if input_mode == "web_webrtc_whip" else "srs_rtmp_audio"
        result = ingest_audio_chunk(
            PROJECT_ROOT,
            session_dir.name,
            path.read_bytes(),
            audio_index=idx,
            relative_ts_ms=rel_ms,
            source_ts_ms=None,
            timestamp_source="shared_monotonic",
            duration_ms=audio_segment_ms,
            sample_rate=16000,
            channels=1,
            format="wav",
            content_type="audio/wav",
            source=source,
            input_mode=input_mode,
            filename_hint=path.name,
            enqueue_asr=True,
            allow_live_input=True,
        )
        status = str(result.get("status") or "")
        now = utc_now_iso()
        updates = {
            "status": "running",
            "audio_index": idx,
            "last_audio_relative_ts_ms": rel_ms,
            "latest_audio_relative_ts_ms": rel_ms,
            "latest_audio_end_relative_ts_ms": rel_ms + int(audio_segment_ms),
            "last_audio_at": now,
            "last_audio_status": status,
            "audio_unavailable": False,
            **timeline_updates,
        }
        if status == "audio_chunk_received":
            updates["audio_chunks_ingested"] = int(state.get("audio_chunks_ingested", 0) or 0) + 1
            count += 1
        elif status == "error":
            updates["last_error"] = result.get("message")
        updates.update(_sync_updates({**state, **updates}, audio_segment_ms=audio_segment_ms))
        state.update(updates)
        state = update_live_ingest_state(session_dir, **updates)
        update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_audio_at=state.get("last_audio_at"), last_error=state.get("last_error"))
        last_index = idx
    return count, state


def _mark_stale_live_ingest_sessions() -> None:
    for state_path in ONLINE_SESSIONS_DIR.glob("*/stream/live_ingest_state.json"):
        session_dir = state_path.parents[1]
        state = load_live_ingest_state(session_dir)
        if str(state.get("status") or "") not in {
            "starting",
            "waiting_stream",
            "waiting_rtmp_output",
            "waiting_keyframe",
            "running",
            "stopping",
        }:
            continue
        video_alive = _pid_alive(state.get("ffmpeg_video_pid"))
        audio_alive = _pid_alive(state.get("ffmpeg_audio_pid"))
        video_only = bool(state.get("rtmp_video_only")) or _rtmp_video_only(state.get("input_mode"))
        if not video_alive and (video_only or not audio_alive):
            update_live_ingest_state(
                session_dir,
                status="failed",
                last_error="stale live ingest state: ffmpeg pids are not alive",
                stopped_at=utc_now_iso(),
            )
            update_live_source_ingest(session_dir, ingest_status="failed", last_error="stale live ingest state: ffmpeg pids are not alive")


def _finalize_live_inputs(
    *,
    session_dir: Path,
    frame_dir: Path,
    audio_dir: Path,
    frame_interval_ms: int,
    audio_segment_ms: int,
) -> dict[str, Any]:
    state = load_live_ingest_state(session_dir)
    try:
        _, state = _process_new_frames(session_dir, frame_dir, frame_interval_ms, audio_segment_ms, state)
        if not _rtmp_video_only(state.get("input_mode")):
            _, state = _process_new_audio(session_dir, audio_dir, audio_segment_ms, frame_interval_ms, state)
    except Exception as exc:
        state = update_live_ingest_state(session_dir, live_finalize_error=str(exc))
        print(f"[live_ingest_worker] final input drain failed session_id={session_dir.name}: {exc}", flush=True)
    try:
        from online_pipeline.audio_stream import AudioStreamStore

        flush_result = AudioStreamStore(session_dir).flush_asr_tail(
            project_root=PROJECT_ROOT,
            stream_id=str(state.get("stream_id") or ""),
            reason="live_ingest_stop",
        )
        state = update_live_ingest_state(session_dir, audio_asr_tail_flush=flush_result)
    except Exception as exc:
        state = update_live_ingest_state(session_dir, audio_asr_tail_flush={"status": "error", "error": str(exc)})
        print(f"[live_ingest_worker] audio ASR tail flush failed session_id={session_dir.name}: {exc}", flush=True)
    close_result: dict[str, Any] = {}
    try:
        from online_short_term.frame_stream_event_builder import FrameStreamMicroEventBuilder

        close_result = FrameStreamMicroEventBuilder(session_dir).close_open_event(
            project_root=PROJECT_ROOT,
            enqueue_refine=True,
            reason="live_ingest_stop",
        )
        state = update_live_ingest_state(
            session_dir,
            frame_mst_close=close_result,
            frame_mst_closed_on_stop=bool(close_result.get("closed_event_count")),
        )
    except Exception as exc:
        close_result = {"status": "error", "error": str(exc)}
        state = update_live_ingest_state(session_dir, frame_mst_close=close_result)
        print(f"[live_ingest_worker] frame M_st close failed session_id={session_dir.name}: {exc}", flush=True)
    return state


def process_live_ingest_task(task: dict[str, Any]) -> dict[str, Any]:
    session_id = str(task.get("session_id") or "")
    session_dir = ONLINE_SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"session not found: {session_id}")
    task_source_url = str(task.get("source_url") or "").strip()
    if not task_source_url:
        raise ValueError("live_ingest task missing source_url")
    stream_id = str(task.get("stream_id") or "")
    live_source = read_json(session_dir / "stream" / "live_source.json", default={})
    if not isinstance(live_source, dict):
        live_source = {}
    input_mode = str(task.get("input_mode") or live_source.get("input_mode") or "live_pusher_rtmp")
    source_url, pull_url_source, pull_base_override_enabled = choose_effective_live_pull_url(live_source, fallback_url=task_source_url)
    if not source_url:
        raise ValueError("live_ingest task has no effective pull URL")
    print(f"[live_ingest_worker] effective pull url source={pull_url_source} url={source_url}", flush=True)
    rtmp_video_only = _rtmp_video_only(input_mode)
    frame_fps = _env_float("EM2MEM_LIVE_INGEST_FRAME_FPS", 1.0)
    audio_segment_ms = _env_int("EM2MEM_LIVE_INGEST_AUDIO_SEGMENT_MS", 1500)
    probe_timeout_seconds = _env_float("EM2MEM_LIVE_INGEST_PROBE_TIMEOUT_SECONDS", 10.0)
    probe_retry_seconds = _env_float("EM2MEM_LIVE_INGEST_PROBE_RETRY_SECONDS", 2.0)
    attempt_no_output_seconds = _env_float("EM2MEM_LIVE_INGEST_ATTEMPT_NO_OUTPUT_SECONDS", 25.0)
    total_no_output_seconds = _env_float("EM2MEM_LIVE_INGEST_TOTAL_WAIT_SECONDS", 90.0)
    rokid_recovery_no_output_seconds = 0.0
    if rtmp_video_only:
        total_no_output_seconds = _env_float(
            "EM2MEM_ROKID_LIVE_INGEST_STARTUP_WAIT_SECONDS",
            max(120.0, total_no_output_seconds),
        )
        rokid_recovery_no_output_seconds = _env_float("EM2MEM_ROKID_LIVE_INGEST_RECOVERY_WAIT_SECONDS", 0.0)
    frame_interval_ms = int(round(1000.0 / max(0.1, frame_fps)))
    base_dir = session_dir / "stream" / "live_ingest"
    frame_dir = base_dir / "video_frames"
    audio_dir = base_dir / "audio_segments"
    video_log_path = base_dir / "ffmpeg_video_stderr.log"
    audio_log_path = base_dir / "ffmpeg_audio_stderr.log"
    base_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg = _media_binary("ffmpeg")
    resolved_ffprobe = _media_binary("ffprobe")
    detected_video_count, detected_audio_count = _detected_output_counts(frame_dir, audio_dir)
    existing_state = load_live_ingest_state(session_dir)
    timeline_origin_monotonic_ns = existing_state.get("timeline_origin_monotonic_ns") or time.monotonic_ns()
    timeline_origin_wall_ms = existing_state.get("timeline_origin_wall_ms") or _now_wall_ms()
    state = update_live_ingest_state(
        session_dir,
        session_id=session_id,
        stream_id=stream_id,
        input_mode=input_mode,
        source_url=task_source_url,
        effective_pull_url=source_url,
        pull_url_source=pull_url_source,
        pull_base_override_enabled=pull_base_override_enabled,
        status="starting",
        started_at=utc_now_iso(),
        stopped_at=None,
        frame_fps=frame_fps,
        audio_segment_ms=audio_segment_ms,
        source=str(task.get("source") or "srs_rtmp"),
        stop_requested=False,
        last_error=None,
        last_probe_error=None,
        last_pull_error=None,
        waiting_reason="initializing_pull",
        timeline_version=1,
        timeline_origin_wall_ms=int(timeline_origin_wall_ms),
        timeline_origin_monotonic_ns=int(timeline_origin_monotonic_ns),
        timestamp_mode="shared_monotonic",
        timestamp_source="shared_monotonic",
        sync_status="unknown",
        rtmp_video_only=rtmp_video_only,
        rtmp_audio_expected=not rtmp_video_only,
        rtmp_audio_disabled=rtmp_video_only,
        rtmp_transport_mode="video_rtmp_audio_http" if rtmp_video_only else "av_rtmp",
        rtmp_rw_timeout_us=_live_ingest_rw_timeout_us(input_mode),
        rtmp_startup_wait_seconds=total_no_output_seconds,
        rtmp_recovery_wait_seconds=rokid_recovery_no_output_seconds if rtmp_video_only else total_no_output_seconds,
        video_reconnect_count=int(existing_state.get("video_reconnect_count", 0) or 0),
        video_stall_count=int(existing_state.get("video_stall_count", 0) or 0),
        frame_monotonic_violations=int(existing_state.get("frame_monotonic_violations", 0) or 0),
        audio_monotonic_violations=int(existing_state.get("audio_monotonic_violations", 0) or 0),
        ffprobe_ok=None,
        ffmpeg_bin=resolved_ffmpeg,
        ffprobe_bin=resolved_ffprobe,
        ffprobe_cmd=_command_text(_probe_command(source_url, probe_timeout_seconds)),
        ffmpeg_video_cmd=None,
        ffmpeg_audio_cmd=None,
        video_output_dir=str(frame_dir),
        audio_output_dir=str(audio_dir),
        video_output_glob=str(frame_dir / "frame_*.jpg"),
        audio_output_glob=str(audio_dir / "audio_*.wav"),
        detected_video_file_count=detected_video_count,
        detected_audio_file_count=detected_audio_count,
        last_ffmpeg_stderr_tail=_stderr_tail(video_log_path, audio_log_path),
    )
    update_live_source_ingest(session_dir, ingest_status="starting", last_error=None)

    video_proc = audio_proc = None
    video_log = audio_log = None
    last_output_monotonic = time.monotonic()
    last_diag_update = 0.0
    try:
        while not _stop_requested(session_dir):
            if input_mode == "web_webrtc_whip":
                ready = _wait_for_whip_pull(
                    session_dir,
                    source_url,
                    probe_timeout_seconds=probe_timeout_seconds,
                    retry_seconds=probe_retry_seconds,
                    last_output_monotonic=last_output_monotonic,
                    total_no_output_seconds=total_no_output_seconds,
                )
                if not ready:
                    break

            state = load_live_ingest_state(session_dir)
            next_frame_number = _next_output_number(frame_dir, "frame_*.jpg", FRAME_RE, state.get("frame_index"))
            next_audio_number = _next_output_number(audio_dir, "audio_*.wav", AUDIO_RE, state.get("audio_index"))
            pull_attempt_count = int(state.get("pull_attempt_count", 0) or 0) + 1
            attempt_last_output = time.monotonic()
            attempt_error = ""
            video_proc, video_log, video_cmd = _start_ffmpeg_video(source_url, frame_dir, next_frame_number, frame_fps, input_mode)
            if rtmp_video_only:
                audio_proc = None
                audio_log = None
                audio_cmd = []
            else:
                audio_proc, audio_log, audio_cmd = _start_ffmpeg_audio(source_url, audio_dir, next_audio_number, audio_segment_ms, input_mode)
            detected_video_count, detected_audio_count = _detected_output_counts(frame_dir, audio_dir)
            state = update_live_ingest_state(
                session_dir,
                status="waiting_rtmp_output",
                waiting_reason="waiting_for_ffmpeg_output",
                pull_attempt_count=pull_attempt_count,
                ffmpeg_video_pid=video_proc.pid,
                ffmpeg_audio_pid=audio_proc.pid if audio_proc is not None else None,
                ffmpeg_video_exit_code=None,
                ffmpeg_audio_exit_code=None,
                ffmpeg_video_cmd=_command_text(video_cmd),
                ffmpeg_audio_cmd=_command_text(audio_cmd) if audio_cmd else None,
                rtmp_video_only=rtmp_video_only,
                rtmp_audio_expected=not rtmp_video_only,
                rtmp_audio_disabled=rtmp_video_only,
                video_reconnect_count=max(0, pull_attempt_count - 1),
                detected_video_file_count=detected_video_count,
                detected_audio_file_count=detected_audio_count,
                last_pull_error=None,
                last_ffmpeg_stderr_tail=_stderr_tail(video_log_path, audio_log_path),
            )
            update_live_source_ingest(session_dir, ingest_status="waiting_rtmp_output")

            while not _stop_requested(session_dir):
                state = load_live_ingest_state(session_dir)
                frame_count, state = _process_new_frames(session_dir, frame_dir, frame_interval_ms, audio_segment_ms, state)
                if rtmp_video_only:
                    audio_count = 0
                else:
                    audio_count, state = _process_new_audio(session_dir, audio_dir, audio_segment_ms, frame_interval_ms, state)
                now_monotonic = time.monotonic()
                current_video_count, current_audio_count = _detected_output_counts(frame_dir, audio_dir)
                output_file_detected = current_video_count > detected_video_count or current_audio_count > detected_audio_count
                if output_file_detected:
                    detected_video_count = current_video_count
                    detected_audio_count = current_audio_count
                    last_output_monotonic = now_monotonic
                    attempt_last_output = now_monotonic
                    state = update_live_ingest_state(
                        session_dir,
                        last_output_file_at=utc_now_iso(),
                        detected_video_file_count=current_video_count,
                        detected_audio_file_count=current_audio_count,
                    )
                if frame_count or audio_count:
                    last_output_monotonic = now_monotonic
                    attempt_last_output = now_monotonic
                    state = update_live_ingest_state(
                        session_dir,
                        status="running",
                        waiting_reason=None,
                        last_output_file_at=utc_now_iso(),
                        detected_video_file_count=current_video_count,
                        detected_audio_file_count=current_audio_count,
                        last_error=None,
                        last_pull_error=None,
                        last_ffmpeg_stderr_tail=_stderr_tail(video_log_path, audio_log_path),
                    )
                    update_live_source_ingest(
                        session_dir,
                        ingest_status="running",
                        last_frame_at=state.get("last_frame_at"),
                        last_audio_at=state.get("last_audio_at"),
                    )
                if now_monotonic - last_diag_update >= 2.0:
                    running_without_files = (
                        (video_proc.poll() is None or (audio_proc is not None and audio_proc.poll() is None))
                        and now_monotonic - attempt_last_output >= 2.0
                    )
                    state = update_live_ingest_state(
                        session_dir,
                        detected_video_file_count=current_video_count,
                        detected_audio_file_count=current_audio_count,
                        last_pull_error="ffmpeg running but no output files detected" if running_without_files else state.get("last_pull_error"),
                        last_ffmpeg_stderr_tail=_stderr_tail(video_log_path, audio_log_path),
                    )
                    last_diag_update = now_monotonic

                video_dead = video_proc.poll() is not None
                audio_dead = True if audio_proc is None else audio_proc.poll() is not None
                if not rtmp_video_only and audio_dead and int(state.get("audio_chunks_ingested", 0) or 0) == 0:
                    state = update_live_ingest_state(session_dir, audio_unavailable=True)
                if video_dead and (rtmp_video_only or audio_dead):
                    if rtmp_video_only:
                        attempt_error = f"ffmpeg video exited before sustained output files were detected (video_rc={video_proc.returncode})"
                    else:
                        attempt_error = (
                            f"ffmpeg video/audio exited before sustained output files were detected "
                            f"(video_rc={video_proc.returncode}, audio_rc={audio_proc.returncode if audio_proc is not None else None})"
                        )
                    break
                if now_monotonic - attempt_last_output >= attempt_no_output_seconds:
                    if rtmp_video_only:
                        attempt_error = "ffmpeg video running but no video output files detected" if not video_dead else (
                            f"ffmpeg video exited and produced no output files for {attempt_no_output_seconds:.0f}s"
                        )
                    elif not video_dead or not audio_dead:
                        attempt_error = "ffmpeg running but no output files detected"
                    else:
                        attempt_error = f"ffmpeg exited and produced no output files for {attempt_no_output_seconds:.0f}s"
                    break
                time.sleep(_env_float("EM2MEM_LIVE_INGEST_POLL_SECONDS", 0.25))

            _stop_process(video_proc)
            _stop_process(audio_proc)
            video_exit_code = video_proc.returncode if video_proc is not None else None
            audio_exit_code = audio_proc.returncode if audio_proc is not None else None
            _close_handles(video_log, audio_log)
            video_proc = audio_proc = None
            video_log = audio_log = None
            if _stop_requested(session_dir):
                break

            stderr_tail = _stderr_tail(video_log_path, audio_log_path)
            running_without_output = attempt_error in {
                "ffmpeg running but no output files detected",
                "ffmpeg video running but no video output files detected",
            }
            state = update_live_ingest_state(
                session_dir,
                status="waiting_keyframe" if input_mode == "web_webrtc_whip" and running_without_output else "waiting_rtmp_output",
                waiting_reason="waiting_for_keyframe_or_rtmp_output" if input_mode == "web_webrtc_whip" and running_without_output else "restarting_ffmpeg_pull",
                last_pull_error=attempt_error or "ffmpeg pull attempt ended without output",
                last_ffmpeg_stderr_tail=stderr_tail,
                ffmpeg_video_exit_code=video_exit_code,
                ffmpeg_audio_exit_code=audio_exit_code,
                ffmpeg_video_pid=None,
                ffmpeg_audio_pid=None,
                video_stall_count=int(state.get("video_stall_count", 0) or 0) + (1 if rtmp_video_only else 0),
            )
            update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_error=state.get("last_pull_error"))
            max_no_output_seconds = total_no_output_seconds
            if rtmp_video_only and (int(state.get("frames_ingested", 0) or 0) > 0 or detected_video_count > 0):
                max_no_output_seconds = rokid_recovery_no_output_seconds
            if max_no_output_seconds > 0 and time.monotonic() - last_output_monotonic >= max_no_output_seconds:
                raise RuntimeError(
                    f"live ingest produced no output for {max_no_output_seconds:.0f}s after retries; "
                    f"last pull error: {state.get('last_pull_error')}"
                )
            time.sleep(max(0.25, probe_retry_seconds))
    finally:
        _stop_process(video_proc)
        _stop_process(audio_proc)
        _close_handles(video_log, audio_log)

    _finalize_live_inputs(
        session_dir=session_dir,
        frame_dir=frame_dir,
        audio_dir=audio_dir,
        frame_interval_ms=frame_interval_ms,
        audio_segment_ms=audio_segment_ms,
    )
    state = update_live_ingest_state(
        session_dir,
        status="stopped",
        ffmpeg_video_pid=None,
        ffmpeg_audio_pid=None,
        stopped_at=utc_now_iso(),
        stop_requested=False,
        waiting_reason=None,
        last_ffmpeg_stderr_tail=_stderr_tail(video_log_path, audio_log_path),
    )
    update_live_source_ingest(session_dir, ingest_status=state.get("status"), last_frame_at=state.get("last_frame_at"), last_audio_at=state.get("last_audio_at"), last_error=state.get("last_error"))
    return {
        "status": state.get("status"),
        "frames_ingested": state.get("frames_ingested", 0),
        "audio_chunks_ingested": state.get("audio_chunks_ingested", 0),
        "last_error": state.get("last_error"),
    }


def main() -> None:
    poll_seconds = _env_float("EM2MEM_LIVE_INGEST_WORKER_POLL_SECONDS", 1.0)
    _mark_stale_live_ingest_sessions()
    print("[live_ingest_worker] started", flush=True)
    while True:
        queued = list_queued_live_ingest_tasks(PROJECT_ROOT)
        write_worker_runtime(
            PROJECT_ROOT,
            "live_ingest",
            status="ready" if not queued else "busy",
            queue_pending=len(queued),
            extra={"active_sessions": [], "processed_frames": None, "processed_audio_chunks": None},
        )
        if not queued:
            time.sleep(poll_seconds)
            continue
        for task_path in queued:
            claimed = claim_live_ingest_task(PROJECT_ROOT, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task.get("session_id") or "")
            try:
                write_worker_runtime(
                    PROJECT_ROOT,
                    "live_ingest",
                    status="busy",
                    queue_pending=max(0, len(queued) - 1),
                    last_task_id=str(task.get("task_id") or claimed_path.stem),
                    extra={"active_sessions": [session_id]},
                )
                with WorkerTaskHeartbeat(
                    PROJECT_ROOT,
                    "live_ingest",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    queue_pending=lambda: len(list_queued_live_ingest_tasks(PROJECT_ROOT)),
                    extra_fn=lambda session_id=session_id: _live_ingest_runtime_extra(session_id),
                    interval_env="EM2MEM_LIVE_INGEST_HEARTBEAT_SECONDS",
                ):
                    result = process_live_ingest_task(task)
                finish_live_ingest_task(PROJECT_ROOT, claimed_path, task, "done", result=result)
            except Exception as exc:
                session_dir = ONLINE_SESSIONS_DIR / session_id
                if session_dir.exists():
                    log_dir = session_dir / "stream" / "live_ingest"
                    update_live_ingest_state(
                        session_dir,
                        status="failed",
                        last_error=str(exc),
                        last_pull_error=str(exc),
                        waiting_reason=None,
                        stopped_at=utc_now_iso(),
                        ffmpeg_video_pid=None,
                        ffmpeg_audio_pid=None,
                        last_ffmpeg_stderr_tail=_stderr_tail(
                            log_dir / "ffmpeg_video_stderr.log",
                            log_dir / "ffmpeg_audio_stderr.log",
                        ),
                    )
                    update_live_source_ingest(session_dir, ingest_status="failed", last_error=str(exc))
                print(f"[live_ingest_worker] task failed session_id={session_id}: {exc}", flush=True)
                finish_live_ingest_task(PROJECT_ROOT, claimed_path, task, "failed", error=str(exc))


if __name__ == "__main__":
    main()
