from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api_server as base_api  # noqa: E402
from online_current.mcur_store import MCurStore  # noqa: E402
from online_preprocess.io_utils import ffmpeg_bin, ffprobe_bin, read_json, relative_to_session, utc_now_iso, write_json, write_json_atomic, write_status  # noqa: E402
from online_preprocess.task_queue import enqueue_evidence_task, enqueue_preprocess_task  # noqa: E402


app = base_api.app
ONLINE_SESSIONS_DIR = base_api.ONLINE_SESSIONS_DIR
DEMO_STATE_NAME = "demo_state.json"
DEMO_MANIFEST_NAME = "frames_manifest.json"
DEMO_TEST_STATE_NAME = "demo_test_state.json"
DEMO_TEST_DEFAULT_DAY1_START = "2026-06-30 18:56:02"
DEMO_TEST_DEFAULT_DAY2_START = "2026-07-01 21:26:32"
if getattr(app.state, "worldmm_demo_routes_registered", False):
    raise RuntimeError("WorldMM demo routes are already registered on this app")
app.state.worldmm_demo_routes_registered = True


class DemoStartRequest(BaseModel):
    current_time: float = 0.0
    playback_speed: float = 1.0


class DemoTickRequest(BaseModel):
    current_time: float
    paused: bool = False
    playback_speed: float = 1.0


class DemoPrepareRequest(BaseModel):
    sample_fps: Optional[float] = None
    force: bool = False
    enqueue_preprocess: bool = False
    force_preprocess: bool = False


class DemoSyncInputRequest(BaseModel):
    enqueue_preprocess: bool = False
    force_preprocess: bool = False


class DemoTestStartRequest(BaseModel):
    clip_id: str = "day1"
    current_time: float = 0.0
    playback_speed: float = 1.0


class DemoTestTickRequest(BaseModel):
    clip_id: str = "day1"
    current_time: float
    paused: bool = False
    playback_speed: float = 1.0


class DemoTestPrepareRequest(BaseModel):
    clip_id: Optional[str] = None
    sample_fps: Optional[float] = None
    force: bool = False


class DemoTestOfflineRequest(BaseModel):
    force_preprocess: bool = False
    enqueue_evidence: bool = False
    force_evidence: bool = False


class DemoTestBuildMemoryRequest(BaseModel):
    force: bool = True
    allow_manifest_fallback: bool = False
    skip_semantic: bool = False
    generation_backend: Optional[str] = None


def _session_dir(session_id: str) -> Path:
    return ONLINE_SESSIONS_DIR / session_id


def _demo_dir(session_dir: Path) -> Path:
    return session_dir / "demo"


def _state_path(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / DEMO_STATE_NAME


def _manifest_path(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / DEMO_MANIFEST_NAME


def _demo_test_dir(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / "demo_test"


def _demo_test_state_path(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / DEMO_TEST_STATE_NAME


def _demo_test_clip_dir(session_dir: Path, clip_id: str) -> Path:
    return _demo_test_dir(session_dir) / _safe_clip_id(clip_id)


def _demo_test_manifest_path(session_dir: Path, clip_id: str) -> Path:
    return _demo_test_clip_dir(session_dir, clip_id) / "frames_manifest.json"


def _read_state(session_dir: Path) -> dict[str, Any]:
    state = read_json(_state_path(session_dir), default={})
    return state if isinstance(state, dict) else {}


def _write_state(session_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = utc_now_iso()
    write_json_atomic(_state_path(session_dir), state)
    return state


def _read_manifest(session_dir: Path) -> list[dict[str, Any]]:
    data = read_json(_manifest_path(session_dir), default=[])
    return data if isinstance(data, list) else []


def _read_demo_test_state(session_dir: Path) -> dict[str, Any]:
    state = read_json(_demo_test_state_path(session_dir), default={})
    return state if isinstance(state, dict) else {}


def _write_demo_test_state(session_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = utc_now_iso()
    write_json_atomic(_demo_test_state_path(session_dir), state)
    return state


def _read_demo_test_manifest(session_dir: Path, clip_id: str) -> list[dict[str, Any]]:
    data = read_json(_demo_test_manifest_path(session_dir, clip_id), default=[])
    return data if isinstance(data, list) else []


def _run(cmd: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )
    return result


def _probe_duration(video_path: Path) -> float:
    result = _run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        description="ffprobe video duration",
    )
    try:
        payload = json.loads(result.stdout)
        return max(0.0, float((payload.get("format") or {}).get("duration") or 0.0))
    except Exception:
        return 0.0


def _safe_session_id(value: Optional[str] = None) -> str:
    text = (value or "").strip()
    if text and all(ch.isalnum() or ch in {"_", "-"} for ch in text):
        return text[:64]
    return f"demo_{uuid4().hex[:12]}"


def _safe_clip_id(value: str | None) -> str:
    text = (value or "day1").strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
    return text[:32] or "day1"


def _parse_demo_datetime(value: str | None, default: str) -> datetime:
    text = str(value or default).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"invalid demo datetime: {text}; expected YYYY-MM-DD HH:MM:SS")


def _format_cn_date(value: datetime) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def _format_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S")


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _seconds_to_hhmmssff(seconds: float, fps_for_code: int = 100) -> str:
    total_frames = max(0, int(round(float(seconds) * fps_for_code)))
    total_seconds, frames = divmod(total_frames, fps_for_code)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}{frames:02d}"


def _display_datetime_for_clip(clip: dict[str, Any], local_time: float) -> datetime:
    start = _parse_demo_datetime(str(clip.get("start_datetime") or ""), DEMO_TEST_DEFAULT_DAY1_START)
    return start + timedelta(seconds=max(0.0, float(local_time or 0.0)))


def _display_payload_for_clip(clip: dict[str, Any], local_time: float) -> dict[str, Any]:
    value = _display_datetime_for_clip(clip, local_time)
    return {
        "display_date": _format_cn_date(value),
        "display_time": _format_time(value),
        "display_datetime": _format_datetime(value),
        "display_iso": value.isoformat(timespec="seconds"),
        "display_hhmmssff": _seconds_to_hhmmssff(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000),
    }


def _demo_test_child_session_id(parent_session_id: str, clip_id: str) -> str:
    return f"{parent_session_id}__{_safe_clip_id(clip_id)}"


def _public_urls(session_id: str) -> dict[str, str]:
    return {
        "video_url": f"/demo/{session_id}/video",
        "status_url": f"/demo/{session_id}/status",
        "start_url": f"/demo/{session_id}/start",
        "tick_url": f"/demo/{session_id}/tick",
        "ask_url": f"/ask/{session_id}",
        "ask_stream_url": f"/ask/{session_id}/stream",
    }


def _demo_test_public_urls(session_id: str, clips: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    clip_urls = {}
    for clip in clips or []:
        clip_id = _safe_clip_id(str(clip.get("clip_id") or ""))
        if clip_id:
            clip_urls[clip_id] = f"/demo-test/{session_id}/video/{clip_id}"
    return {
        "status_url": f"/demo-test/{session_id}/status",
        "prepare_url": f"/demo-test/{session_id}/prepare",
        "start_url": f"/demo-test/{session_id}/start",
        "tick_url": f"/demo-test/{session_id}/tick",
        "pause_url": f"/demo-test/{session_id}/pause",
        "manifest_url": f"/demo-test/{session_id}/manifest",
        "enqueue_offline_url": f"/demo-test/{session_id}/enqueue_offline",
        "build_memory_url": f"/demo-test/{session_id}/build_memory",
        "ask_url": f"/ask/{session_id}",
        "ask_stream_url": f"/ask/{session_id}/stream",
        "clip_video_urls": clip_urls,
    }


def _create_base_session(session_id: str, *, original_filename: Optional[str] = None) -> Path:
    session_dir = _session_dir(session_id)
    for subdir in ("demo", "stream", "current", "short_term"):
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "session_id": session_id,
        "demo_mode": True,
        "source": "demo_upload",
        "original_filename": original_filename,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(session_dir / "metadata.json", metadata)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_id,
            "status": "uploaded",
            "stage": "demo_uploaded",
            "progress": 10,
            "demo_mode": True,
            "updated_at": utc_now_iso(),
        },
    )
    return session_dir


def _create_demo_test_parent_session(session_id: str) -> Path:
    session_dir = _session_dir(session_id)
    for subdir in ("demo", "stream", "current", "short_term", "evidence", "captions", "preprocess", "worldmm"):
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "session_id": session_id,
        "demo_mode": True,
        "demo_test_mode": True,
        "source": "demo_test_upload",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(session_dir / "metadata.json", metadata)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_id,
            "status": "uploaded",
            "stage": "demo_test_uploaded",
            "progress": 10,
            "demo_mode": True,
            "demo_test_mode": True,
            "updated_at": utc_now_iso(),
        },
    )
    return session_dir


def _create_demo_test_child_session(parent_session_dir: Path, clip: dict[str, Any]) -> Path:
    child_session_id = str(clip["child_session_id"])
    child_dir = _session_dir(child_session_id)
    for subdir in ("demo", "stream", "current", "short_term"):
        (child_dir / subdir).mkdir(parents=True, exist_ok=True)
    source_path = parent_session_dir / str(clip["source_video"])
    input_path = child_dir / "input.mp4"
    if input_path.exists():
        input_path.unlink()
    try:
        os.link(source_path, input_path)
    except Exception:
        shutil.copy2(source_path, input_path)
    write_json_atomic(
        child_dir / "metadata.json",
        {
            "session_id": child_session_id,
            "parent_session_id": parent_session_dir.name,
            "demo_mode": True,
            "demo_test_mode": True,
            "demo_test_child": True,
            "demo_clip_id": clip["clip_id"],
            "source": "demo_test_child",
            "start_datetime": clip["start_datetime"],
            "display_date": clip["display_date"],
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        child_dir / "status.json",
        {
            "session_id": child_session_id,
            "status": "uploaded",
            "stage": "demo_test_child_uploaded",
            "progress": 10,
            "parent_session_id": parent_session_dir.name,
            "demo_clip_id": clip["clip_id"],
            "updated_at": utc_now_iso(),
        },
    )
    return child_dir


def _sync_input_video(session_dir: Path, source_path: Path) -> Path:
    """Make demo uploads visible to existing offline preprocess code."""
    input_path = session_dir / "input.mp4"
    if input_path.exists():
        try:
            if input_path.samefile(source_path):
                return input_path
        except Exception:
            pass
        input_path.unlink()
    try:
        os.link(source_path, input_path)
    except Exception:
        shutil.copy2(source_path, input_path)
    return input_path


def _enqueue_preprocess(session_dir: Path, *, force: bool = False) -> dict[str, Any]:
    input_path = session_dir / "input.mp4"
    if not input_path.exists():
        state = _read_state(session_dir)
        source_path = Path(str(state.get("source_video_abs") or ""))
        if not source_path.exists():
            source_rel = str(state.get("source_video") or "")
            source_path = session_dir / source_rel
        if source_path.exists():
            _sync_input_video(session_dir, source_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input.mp4 not found for session {session_dir.name}")

    task_path = enqueue_preprocess_task(
        project_root=PROJECT_ROOT,
        session_id=session_dir.name,
        force=force,
    )
    write_status(
        session_dir=session_dir,
        session_id=session_dir.name,
        status="processing",
        stage="preprocess_queued",
        progress=5,
        error=None,
    )
    return {
        "preprocess_queued": True,
        "preprocess_task_path": str(task_path),
        "preprocess_task_id": task_path.stem,
        "preprocess_task_url": f"/session/{session_dir.name}/status",
    }


def _extract_frames(session_dir: Path, *, sample_fps: float, force: bool = False) -> list[dict[str, Any]]:
    demo_dir = _demo_dir(session_dir)
    video_path = Path(_read_state(session_dir).get("source_video_abs") or "")
    if not video_path.exists():
        source_rel = str(_read_state(session_dir).get("source_video") or "")
        video_path = session_dir / source_rel
    if not video_path.exists():
        raise FileNotFoundError(f"demo source video not found for session {session_dir.name}")

    frames_dir = demo_dir / "frames"
    if force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _manifest_path(session_dir)
    if manifest_path.exists() and not force:
        manifest = _read_manifest(session_dir)
        if manifest:
            return manifest

    sample_fps = max(0.1, min(float(sample_fps or 1.0), 10.0))
    output_pattern = frames_dir / "demo_frame_%06d.jpg"
    _run(
        [
            ffmpeg_bin(),
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={sample_fps}",
            "-q:v",
            "2",
            str(output_pattern),
        ],
        description="extract demo video frames",
    )

    frames = sorted(frames_dir.glob("demo_frame_*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg extracted no demo frames")
    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(frames):
        timestamp = round(index / sample_fps, 3)
        manifest.append(
            {
                "frame_index": index,
                "timestamp": timestamp,
                "path": path.relative_to(session_dir).as_posix(),
                "role": "demo_frame",
            }
        )
    write_json_atomic(manifest_path, manifest)
    return manifest


def _extract_demo_test_clip_frames(
    session_dir: Path,
    clip: dict[str, Any],
    *,
    sample_fps: float,
    force: bool = False,
) -> list[dict[str, Any]]:
    clip_id = _safe_clip_id(str(clip.get("clip_id") or ""))
    clip_dir = _demo_test_clip_dir(session_dir, clip_id)
    source_path = session_dir / str(clip.get("source_video") or "")
    if not source_path.exists():
        raise FileNotFoundError(f"demo-test source video not found: {source_path}")

    frames_dir = clip_dir / "frames"
    if force and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _demo_test_manifest_path(session_dir, clip_id)
    if manifest_path.exists() and not force:
        manifest = _read_demo_test_manifest(session_dir, clip_id)
        if manifest:
            return manifest

    sample_fps = max(0.1, min(float(sample_fps or 1.0), 10.0))
    output_pattern = frames_dir / f"{clip_id}_frame_%06d.jpg"
    _run(
        [
            ffmpeg_bin(),
            "-y",
            "-i",
            str(source_path),
            "-vf",
            f"fps={sample_fps}",
            "-q:v",
            "2",
            str(output_pattern),
        ],
        description=f"extract demo-test frames for {clip_id}",
    )

    frames = sorted(frames_dir.glob(f"{clip_id}_frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no demo-test frames for {clip_id}")

    timeline_offset = float(clip.get("timeline_offset") or 0.0)
    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(frames):
        local_ts = round(index / sample_fps, 3)
        timeline_ts = round(timeline_offset + local_ts, 3)
        display = _display_payload_for_clip(clip, local_ts)
        manifest.append(
            {
                "clip_id": clip_id,
                "frame_index": index,
                "local_timestamp": local_ts,
                "timestamp": timeline_ts,
                "timeline_timestamp": timeline_ts,
                "path": path.relative_to(session_dir).as_posix(),
                "role": "demo_test_frame",
                **display,
            }
        )
    write_json_atomic(manifest_path, manifest)
    return manifest


def _prepare_demo_test_session(
    session_dir: Path,
    *,
    clip_id: str | None = None,
    sample_fps: Optional[float] = None,
    force: bool = False,
) -> dict[str, Any]:
    state = _read_demo_test_state(session_dir)
    clips = list(state.get("clips") or [])
    if clip_id:
        wanted = _safe_clip_id(clip_id)
        clips = [clip for clip in clips if _safe_clip_id(str(clip.get("clip_id") or "")) == wanted]
        if not clips:
            raise FileNotFoundError(f"demo-test clip not found: {clip_id}")

    updated_clips = []
    clip_by_id = {str(clip.get("clip_id")): dict(clip) for clip in state.get("clips", []) or []}
    for clip in clips:
        fps = float(sample_fps or clip.get("sample_fps") or state.get("sample_fps") or os.getenv("WORLDMM_DEMO_SAMPLE_FPS", "1.0") or 1.0)
        manifest = _extract_demo_test_clip_frames(session_dir, clip, sample_fps=fps, force=force)
        updated = {
            **clip,
            "prepared": True,
            "sample_fps": fps,
            "frame_count": len(manifest),
            "manifest_path": _demo_test_manifest_path(session_dir, str(clip["clip_id"])).relative_to(session_dir).as_posix(),
        }
        clip_by_id[str(clip["clip_id"])] = updated
        updated_clips.append(updated)

    final_clips = [clip_by_id[str(clip.get("clip_id"))] for clip in state.get("clips", []) or [] if str(clip.get("clip_id")) in clip_by_id]
    prepared_state = {
        **state,
        **_demo_test_public_urls(session_dir.name, final_clips),
        "status": "prepared",
        "prepared": all(bool(clip.get("prepared")) for clip in final_clips) if final_clips else False,
        "clips": final_clips,
        "frame_count": sum(int(clip.get("frame_count") or 0) for clip in final_clips),
    }
    _write_demo_test_state(session_dir, prepared_state)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_dir.name,
            "status": "ready",
            "stage": "demo_test_prepared",
            "progress": 100,
            "demo_mode": True,
            "demo_test_mode": True,
            "clips": [
                {
                    "clip_id": clip.get("clip_id"),
                    "display_date": clip.get("display_date"),
                    "start_time": clip.get("start_time"),
                    "duration": clip.get("duration"),
                    "frame_count": clip.get("frame_count"),
                }
                for clip in final_clips
            ],
            "updated_at": utc_now_iso(),
        },
    )
    return prepared_state


def _copy_or_link_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _parent_asset_rel_for_child_path(parent_dir: Path, child_dir: Path, clip_id: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    src = Path(text)
    if not src.is_absolute():
        src = child_dir / text
    if not src.exists():
        return text
    parent_asset = _demo_test_clip_dir(parent_dir, clip_id) / "evidence_assets" / src.name
    _copy_or_link_asset(src, parent_asset)
    return relative_to_session(parent_asset, parent_dir)


def _prefix_time_label(text: str, *, clip: dict[str, Any], start_local: float, end_local: float) -> str:
    start = _display_payload_for_clip(clip, start_local)
    end = _display_payload_for_clip(clip, end_local)
    label = f"[{start['display_date']} {start['display_time']}-{end['display_time']}]"
    clean = str(text or "").strip()
    return clean if clean.startswith(label) else (f"{label} {clean}" if clean else label)


def _fallback_demo_test_evidence_from_manifest(parent_dir: Path, clip: dict[str, Any], *, window_seconds: float = 30.0) -> list[dict[str, Any]]:
    clip_id = _safe_clip_id(str(clip.get("clip_id") or ""))
    manifest = _read_demo_test_manifest(parent_dir, clip_id)
    if not manifest:
        manifest = _extract_demo_test_clip_frames(parent_dir, clip, sample_fps=float(clip.get("sample_fps") or 1.0))
    buckets: dict[int, list[dict[str, Any]]] = {}
    for frame in manifest:
        local_ts = float(frame.get("local_timestamp", 0.0) or 0.0)
        buckets.setdefault(int(local_ts // window_seconds), []).append(frame)
    docs: list[dict[str, Any]] = []
    for idx, key in enumerate(sorted(buckets)):
        group = buckets[key]
        start_local = float(group[0].get("local_timestamp", 0.0) or 0.0)
        end_local = min(float(clip.get("duration") or start_local), max(start_local, float(group[-1].get("local_timestamp", start_local) or start_local) + 1.0))
        display_start = _display_payload_for_clip(clip, start_local)
        display_end = _display_payload_for_clip(clip, end_local)
        caption = (
            f"Demo-test video {clip_id} contains visual evidence from "
            f"{display_start['display_date']} {display_start['display_time']} to {display_end['display_time']}."
        )
        docs.append(
            {
                "doc_id": f"{parent_dir.name}_{clip_id}_fallback_{idx:04d}",
                "segment_id": f"{clip_id}_fallback_{idx:04d}",
                "start_time": round(float(clip.get("timeline_offset") or 0.0) + start_local, 3),
                "end_time": round(float(clip.get("timeline_offset") or 0.0) + end_local, 3),
                "duration": round(max(0.0, end_local - start_local), 3),
                "source_video_path": str(clip.get("source_video") or ""),
                "clip_path": str(clip.get("source_video") or ""),
                "keyframe_paths": [str(item.get("path")) for item in group[:8] if item.get("path")],
                "keyframe_captions": [
                    {
                        "timestamp": item.get("timeline_timestamp", item.get("timestamp")),
                        "path": item.get("path"),
                        "caption": caption,
                        "display_datetime": item.get("display_datetime"),
                    }
                    for item in group[:8]
                ],
                "fine_caption": caption,
                "scene": caption,
                "visual_objects": [],
                "main_actions": [],
                "state_changes": [],
                "conversation_focus": [],
                "transcript": "",
                "transcript_segments": [],
                "demo_test_mode": True,
                "demo_clip_id": clip_id,
                "date": clip.get("day_label"),
                "display_date": display_start["display_date"],
                "display_start_time": display_start["display_time"],
                "display_end_time": display_end["display_time"],
                "display_time_range": f"{display_start['display_time']}-{display_end['display_time']}",
                "display_datetime_start": display_start["display_datetime"],
                "display_datetime_end": display_end["display_datetime"],
                "display_hhmmssff_start": display_start["display_hhmmssff"],
                "display_hhmmssff_end": display_end["display_hhmmssff"],
                "local_start_time": round(start_local, 3),
                "local_end_time": round(end_local, 3),
            }
        )
    return docs


def _merge_demo_test_evidence(parent_dir: Path, *, allow_manifest_fallback: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = _read_demo_test_state(parent_dir)
    merged: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    for raw_clip in state.get("clips", []) or []:
        clip = dict(raw_clip)
        clip_id = _safe_clip_id(str(clip.get("clip_id") or ""))
        child_id = str(clip.get("child_session_id") or _demo_test_child_session_id(parent_dir.name, clip_id))
        child_dir = _session_dir(child_id)
        child_evidence_path = child_dir / "evidence" / "session_evidence.json"
        if not child_evidence_path.exists():
            if allow_manifest_fallback:
                merged.extend(_fallback_demo_test_evidence_from_manifest(parent_dir, clip))
                continue
            waiting.append(
                {
                    "clip_id": clip_id,
                    "child_session_id": child_id,
                    "missing": str(child_evidence_path),
                    "hint": "run preprocess/evidence for this child session, then call build_memory again",
                }
            )
            continue
        docs = read_json(child_evidence_path, default=[])
        if not isinstance(docs, list):
            waiting.append({"clip_id": clip_id, "child_session_id": child_id, "missing": "child evidence is not a JSON list"})
            continue
        for idx, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue
            local_start = float(doc.get("start_time") or doc.get("start") or 0.0)
            local_end = float(doc.get("end_time") or doc.get("end") or local_start)
            timeline_start = round(float(clip.get("timeline_offset") or 0.0) + local_start, 3)
            timeline_end = round(float(clip.get("timeline_offset") or 0.0) + local_end, 3)
            display_start = _display_payload_for_clip(clip, local_start)
            display_end = _display_payload_for_clip(clip, local_end)
            parent_doc = dict(doc)
            source_segment = str(doc.get("segment_id") or doc.get("doc_id") or f"seg_{idx:04d}")
            parent_doc.update(
                {
                    "doc_id": f"{parent_dir.name}_{clip_id}_{source_segment}",
                    "segment_id": f"{clip_id}_{source_segment}",
                    "source_child_session_id": child_id,
                    "source_child_doc_id": doc.get("doc_id"),
                    "start_time": timeline_start,
                    "end_time": timeline_end,
                    "start": timeline_start,
                    "end": timeline_end,
                    "duration": round(max(0.0, timeline_end - timeline_start), 3),
                    "source_video_path": str(clip.get("source_video") or ""),
                    "clip_path": str(clip.get("source_video") or ""),
                    "demo_test_mode": True,
                    "demo_clip_id": clip_id,
                    "date": clip.get("day_label"),
                    "display_date": display_start["display_date"],
                    "display_start_time": display_start["display_time"],
                    "display_end_time": display_end["display_time"],
                    "display_time_range": f"{display_start['display_time']}-{display_end['display_time']}",
                    "display_datetime_start": display_start["display_datetime"],
                    "display_datetime_end": display_end["display_datetime"],
                    "display_hhmmssff_start": display_start["display_hhmmssff"],
                    "display_hhmmssff_end": display_end["display_hhmmssff"],
                    "local_start_time": round(local_start, 3),
                    "local_end_time": round(local_end, 3),
                }
            )
            parent_doc["fine_caption"] = _prefix_time_label(str(parent_doc.get("fine_caption") or parent_doc.get("caption") or ""), clip=clip, start_local=local_start, end_local=local_end)
            parent_doc["scene"] = _prefix_time_label(str(parent_doc.get("scene") or ""), clip=clip, start_local=local_start, end_local=local_end)
            parent_doc["keyframe_paths"] = [
                _parent_asset_rel_for_child_path(parent_dir, child_dir, clip_id, path)
                for path in parent_doc.get("keyframe_paths", []) or []
                if path
            ]
            keyframe_captions = []
            for item in parent_doc.get("keyframe_captions", []) or []:
                if isinstance(item, dict):
                    copied = dict(item)
                    copied["path"] = _parent_asset_rel_for_child_path(parent_dir, child_dir, clip_id, copied.get("path"))
                    local_frame_ts = float(copied.get("timestamp") or local_start)
                    copied["timestamp"] = round(float(clip.get("timeline_offset") or 0.0) + local_frame_ts, 3)
                    copied.update(_display_payload_for_clip(clip, local_frame_ts))
                    keyframe_captions.append(copied)
            parent_doc["keyframe_captions"] = keyframe_captions
            merged.append(parent_doc)
    merged.sort(key=lambda item: (str(item.get("date") or ""), float(item.get("start_time") or 0.0), str(item.get("segment_id") or "")))
    return merged, waiting


def _caption_item_from_demo_test_evidence(session_id: str, doc: dict[str, Any], idx: int) -> dict[str, Any]:
    clip_id = _safe_clip_id(str(doc.get("demo_clip_id") or "clip"))
    date_label = str(doc.get("date") or ("DAY2" if clip_id == "day2" else "DAY1"))
    start_code = str(doc.get("display_hhmmssff_start") or "00000000")
    end_code = str(doc.get("display_hhmmssff_end") or start_code)
    segment_id = str(doc.get("segment_id") or f"{clip_id}_seg_{idx:04d}")
    doc_id = str(doc.get("doc_id") or f"{session_id}_{segment_id}")
    fine_caption = str(doc.get("fine_caption") or doc.get("caption") or doc.get("scene") or "").strip()
    transcript = str(doc.get("transcript") or "").strip()
    time_text = f"Display time: {doc.get('display_date')} {doc.get('display_time_range')}"
    text = " ".join(part for part in [fine_caption, transcript, time_text] if part)
    return {
        "doc_id": doc_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "date": date_label,
        "start_time": start_code,
        "end_time": end_code,
        "start": float(doc.get("start_time") or doc.get("start") or 0.0),
        "end": float(doc.get("end_time") or doc.get("end") or doc.get("start_time") or 0.0),
        "duration": round(float(doc.get("duration") or 0.0), 3),
        "video_path": str(doc.get("clip_path") or doc.get("source_video_path") or ""),
        "clip_path": str(doc.get("clip_path") or ""),
        "source_video_path": str(doc.get("source_video_path") or ""),
        "transcript": transcript,
        "transcript_text": transcript,
        "transcript_segments": doc.get("transcript_segments", []) or [],
        "text": text or "No grounded caption is available for this segment.",
        "caption": fine_caption,
        "fine_caption": fine_caption,
        "visual_summary": fine_caption,
        "scene": doc.get("scene"),
        "scene_summary": {"dominant_scene": doc.get("scene"), "scene": doc.get("scene"), "source": "demo_test"},
        "keyframe_caption": " ".join(str(item.get("caption") or "") for item in doc.get("keyframe_captions", []) or [] if isinstance(item, dict)),
        "keyframe_captions": doc.get("keyframe_captions", []) or [],
        "keyframe_paths": doc.get("keyframe_paths", []) or [],
        "visual_objects": doc.get("visual_objects", []) or [],
        "visual_object_threads": doc.get("visual_objects", []) or [],
        "main_actions": doc.get("main_actions", []) or [],
        "action_threads": doc.get("main_actions", []) or [],
        "state_changes": doc.get("state_changes", []) or [],
        "conversation_focus": doc.get("conversation_focus"),
        "topic_threads": doc.get("conversation_focus", []) if isinstance(doc.get("conversation_focus"), list) else [],
        "speakers": doc.get("speakers", []) or [],
        "speaker_stats": doc.get("speakers", []) or [],
        "critical_speech_lines": [
            str(item.get("text") or "")
            for item in doc.get("transcript_segments", []) or []
            if isinstance(item, dict) and item.get("text")
        ],
        "evidence_doc_id": doc_id,
        "demo_test_mode": True,
        "demo_clip_id": clip_id,
        "display_date": doc.get("display_date"),
        "display_time_range": doc.get("display_time_range"),
        "display_datetime_start": doc.get("display_datetime_start"),
        "display_datetime_end": doc.get("display_datetime_end"),
        "local_start_time": doc.get("local_start_time"),
        "local_end_time": doc.get("local_end_time"),
    }


def _aggregate_demo_caption_group(group: list[dict[str, Any]], scale: str, idx: int) -> dict[str, Any]:
    first = group[0]
    last = group[-1]
    text = " ".join(str(item.get("fine_caption") or item.get("text") or "") for item in group if item.get("fine_caption") or item.get("text"))
    keyframes = []
    for item in group:
        keyframes.extend(item.get("keyframe_paths", []) or [])
    return {
        "doc_id": f"{first['session_id']}_{scale}_{idx:04d}_{first['date']}_{first['start_time']}_{last['end_time']}",
        "session_id": first["session_id"],
        "date": first["date"],
        "start_time": first["start_time"],
        "end_time": last["end_time"],
        "start": first["start"],
        "end": last["end"],
        "duration": round(float(last["end"]) - float(first["start"]), 3),
        "video_path": first.get("video_path"),
        "text": text,
        "caption": text,
        "fine_caption": text,
        "visual_summary": text,
        "scene_summary": first.get("scene_summary", {}),
        "keyframe_paths": keyframes,
        "action_threads": [x for item in group for x in item.get("action_threads", []) or []],
        "object_threads": [x for item in group for x in item.get("visual_objects", []) or []],
        "visual_object_threads": [x for item in group for x in item.get("visual_objects", []) or []],
        "topic_threads": [x for item in group for x in item.get("topic_threads", []) or []],
        "critical_speech_lines": [x for item in group for x in item.get("critical_speech_lines", []) or []],
        "source_doc_ids": [item["doc_id"] for item in group],
        "child_ids": [item["doc_id"] for item in group],
        "level": scale,
        "demo_test_mode": True,
        "display_date": first.get("display_date"),
    }


def _build_demo_multiscale_caption_items(caption_30s: list[dict[str, Any]], window_seconds: int, scale: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in caption_30s:
        key = (str(item.get("date") or "DAY1"), int(float(item.get("start") or 0.0) // window_seconds))
        buckets.setdefault(key, []).append(item)
    output = []
    for idx, key in enumerate(sorted(buckets)):
        group = sorted(buckets[key], key=lambda x: float(x.get("start") or 0.0))
        if group:
            output.append(_aggregate_demo_caption_group(group, scale, idx))
    return output


def _build_demo_test_parent_memory(
    session_dir: Path,
    *,
    force: bool = True,
    allow_manifest_fallback: bool = False,
    skip_semantic: bool = False,
    generation_backend: str | None = None,
) -> dict[str, Any]:
    from online_memory.evidence_to_worldmm import write_semantic_files, write_sidecar_files
    from online_memory.worldmm_layout import WorldMMOnlineLayout, ensure_worldmm_layout

    state = _read_demo_test_state(session_dir)
    if not state:
        raise FileNotFoundError("demo-test state not found")
    evidence_docs, waiting = _merge_demo_test_evidence(session_dir, allow_manifest_fallback=allow_manifest_fallback)
    if waiting:
        return {
            "status": "waiting",
            "session_id": session_dir.name,
            "waiting_child_outputs": waiting,
            "message": "child evidence is not ready; run demo-test enqueue_offline and evidence worker, then call build_memory again",
        }
    if not evidence_docs:
        raise RuntimeError("no demo-test evidence available to build memory")

    (session_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (session_dir / "captions").mkdir(parents=True, exist_ok=True)
    (session_dir / "preprocess").mkdir(parents=True, exist_ok=True)
    write_json(session_dir / "evidence" / "session_evidence.json", evidence_docs)
    write_json(session_dir / "preprocess" / "session_30sec.json", evidence_docs)
    write_json(session_dir / "preprocess" / "segments_30s.json", evidence_docs)
    if not (session_dir / "input.mp4").exists():
        first_source = session_dir / str((state.get("clips") or [{}])[0].get("source_video") or "")
        if first_source.exists():
            _sync_input_video(session_dir, first_source)

    caption_30s = [_caption_item_from_demo_test_evidence(session_dir.name, doc, idx) for idx, doc in enumerate(evidence_docs)]
    caption_30s.sort(key=lambda item: (str(item.get("date") or ""), str(item.get("start_time") or ""), str(item.get("end_time") or "")))
    write_json(session_dir / "captions" / "session_30sec_captioned.json", caption_30s)

    layout = WorldMMOnlineLayout(session_dir=session_dir, session_id=session_dir.name)
    ensure_worldmm_layout(layout)
    caption_3min = _build_demo_multiscale_caption_items(caption_30s, 180, "3min")
    caption_10min = _build_demo_multiscale_caption_items(caption_30s, 600, "10min")
    caption_1h = _build_demo_multiscale_caption_items(caption_30s, 3600, "1h")
    write_json(layout.caption_30sec_path, caption_30s)
    write_json(layout.caption_3min_path, caption_3min)
    write_json(layout.caption_10min_path, caption_10min)
    write_json(layout.caption_1h_path, caption_1h)
    write_json(layout.visual_evidence_path, caption_30s)

    model = os.getenv("WORLDMM_QUERY_RETRIEVER_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("WORLDMM_MEMORY_MODEL") or "gpt-5"
    backend = generation_backend or os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND") or "rule"
    sidecar_paths = write_sidecar_files(
        layout=layout,
        model_name=model,
        caption_by_scale={"30sec": caption_30s, "3min": caption_3min, "10min": caption_10min, "1h": caption_1h},
        generation_backend=backend,
    )
    semantic_candidates_path = None
    semantic_memory_path = layout.semantic_root / f"semantic_memory_{model}.json"
    semantic_fact_count = 0
    if skip_semantic:
        layout.semantic_root.mkdir(parents=True, exist_ok=True)
        write_json(semantic_memory_path, {"facts": [], "timeline": [], "source": "demo_test_empty", "semantic_memory_ready": True})
        semantic_candidates_path = layout.semantic_root / "semantic_candidates.jsonl"
        semantic_candidates_path.write_text("", encoding="utf-8")
    else:
        triplet_map = {}
        if sidecar_paths.get("30sec"):
            triplets = read_json(sidecar_paths["30sec"]["triplets"], default={})
            if isinstance(triplets, dict):
                triplet_map = triplets.get("triplet_map") or {}
        semantic_candidates_path, semantic_memory_path, semantic_fact_count = write_semantic_files(
            layout=layout,
            model_name=model,
            caption_30s=caption_30s,
            generation_backend=backend,
            triplet_map=triplet_map,
        )

    previous = read_json(layout.memory_config_path, default={})
    if not isinstance(previous, dict):
        previous = {}
    previous_version = int(previous.get("latest_ready_memory_version") or previous.get("memory_version") or 0)
    memory_version = previous_version + 1 if force or previous_version <= 0 else previous_version
    last_item = max(caption_30s, key=lambda item: (str(item.get("date") or ""), str(item.get("end_time") or "")))
    config = {
        "session_id": session_dir.name,
        "status": "memory_ready",
        "memory_version": memory_version,
        "latest_ready_memory_version": memory_version,
        "building_memory_version": None,
        "memory_build_state": "ready",
        "episodic_index_ready": True,
        "hipporag_cache_ready": False,
        "long_term_partial_ready": True,
        "long_term_full_ready": bool(not skip_semantic),
        "demo_mode": True,
        "demo_test_mode": True,
        "generation_backend": backend,
        "semantic_memory_ready": True,
        "semantic_memory_path": relative_to_session(semantic_memory_path, session_dir),
        "semantic_candidates_path": relative_to_session(semantic_candidates_path, session_dir) if semantic_candidates_path else None,
        "caption_root": relative_to_session(layout.caption_root, session_dir),
        "sidecar_root": relative_to_session(layout.sidecar_root, session_dir),
        "semantic_root": relative_to_session(layout.semantic_root, session_dir),
        "visual_root": relative_to_session(layout.visual_root, session_dir),
        "visual_embedding_ready": False,
        "visual_lagging": False,
        "visual_evidence_file": relative_to_session(layout.visual_evidence_path, session_dir),
        "evidence_path": "evidence/session_evidence.json",
        "captioned_30sec_path": "captions/session_30sec_captioned.json",
        "pipeline_mode": "demo_test",
        "requested_30s_source": "demo_test_direct",
        "active_30s_source": "demo_test_direct",
        "episodic_source": "demo_test_evidence",
        "worldmm_30s_input_source": "demo_test_direct",
        "worldmm_update_mode": "demo_test_rebuild",
        "last_ready_at": utc_now_iso(),
        "created_at": previous.get("created_at") or utc_now_iso(),
        "updated_at": utc_now_iso(),
        "time_span": [min(float(item.get("start") or 0.0) for item in caption_30s), max(float(item.get("end") or 0.0) for item in caption_30s)],
        "demo_test_clips": state.get("clips", []),
        "counts": {
            "caption_30sec": len(caption_30s),
            "caption_multiscale_total": len(caption_30s) + len(caption_3min) + len(caption_10min) + len(caption_1h),
            "semantic_facts": semantic_fact_count,
            "evidence_docs": len(evidence_docs),
        },
        "query_rag_args": {
            "subject": session_dir.name,
            "retriever_model": model,
            "respond_model": os.getenv("WORLDMM_RESPOND_MODEL", model),
            "until_date": str(last_item.get("date") or "DAY2"),
            "until_time": str(last_item.get("end_time") or "23595999"),
            "episodic_caption_root": relative_to_session(layout.caption_root, session_dir),
            "episodic_sidecar_root": relative_to_session(layout.sidecar_root, session_dir),
            "semantic_root": relative_to_session(layout.semantic_root, session_dir),
            "visual_root": relative_to_session(layout.embeddings_root, session_dir),
            "visual_evidence_file": relative_to_session(layout.visual_evidence_path, session_dir),
        },
        "worldmm_files": {
            "caption_30sec": relative_to_session(layout.caption_30sec_path, session_dir),
            "caption_3min": relative_to_session(layout.caption_3min_path, session_dir),
            "caption_10min": relative_to_session(layout.caption_10min_path, session_dir),
            "caption_1h": relative_to_session(layout.caption_1h_path, session_dir),
            "visual_evidence": relative_to_session(layout.visual_evidence_path, session_dir),
        },
    }
    write_json(layout.memory_config_path, config)
    _write_demo_test_state(
        session_dir,
        {
            **state,
            "memory_ready": True,
            "memory_version": memory_version,
            "memory_config": relative_to_session(layout.memory_config_path, session_dir),
            "long_term_evidence_count": len(evidence_docs),
            "long_term_caption_count": len(caption_30s),
        },
    )
    write_status(
        session_dir,
        session_dir.name,
        status="done",
        stage="demo_test_memory_ready",
        progress=100,
        error=None,
        outputs={"memory_config": relative_to_session(layout.memory_config_path, session_dir)},
    )
    return {"status": "ready", "session_id": session_dir.name, "memory_config": str(layout.memory_config_path), "counts": config["counts"]}


def _select_window_frames(manifest: list[dict[str, Any]], current_time: float, *, window_seconds: float = 30.0) -> list[dict[str, Any]]:
    if not manifest:
        return []
    current_time = max(0.0, float(current_time))
    past = [item for item in manifest if float(item.get("timestamp", 0.0) or 0.0) <= current_time + 1e-6]
    if not past:
        past = [manifest[0]]
    window_start = max(0.0, current_time - window_seconds)
    selected = [item for item in past if float(item.get("timestamp", 0.0) or 0.0) >= window_start - 1e-6]
    if not selected:
        selected = past[-1:]
    return selected[-30:]


def _select_demo_test_window_frames(
    manifest: list[dict[str, Any]],
    timeline_time: float,
    *,
    window_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    if not manifest:
        return []
    timeline_time = max(0.0, float(timeline_time))
    past = [item for item in manifest if float(item.get("timeline_timestamp", item.get("timestamp", 0.0)) or 0.0) <= timeline_time + 1e-6]
    if not past:
        past = [manifest[0]]
    window_start = max(0.0, timeline_time - window_seconds)
    selected = [
        item
        for item in past
        if float(item.get("timeline_timestamp", item.get("timestamp", 0.0)) or 0.0) >= window_start - 1e-6
    ]
    if not selected:
        selected = past[-1:]
    return selected[-30:]


def _jsonl_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                items.append(value)
    return items


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _decorate_demo_test_current_memory(
    session_dir: Path,
    *,
    clip: dict[str, Any],
    selected_frames: list[dict[str, Any]],
    current_local_time: float,
    timeline_time: float,
) -> None:
    display = _display_payload_for_clip(clip, current_local_time)
    by_ts = {
        int(round(float(item.get("timeline_timestamp", item.get("timestamp", 0.0)) or 0.0) * 1000)): item
        for item in selected_frames
    }
    current_dir = session_dir / "current"
    frames_path = current_dir / "current_frames.jsonl"
    frames = _jsonl_items(frames_path)
    for frame in frames:
        key = int(round(float(frame.get("timestamp", 0.0) or 0.0) * 1000))
        source = by_ts.get(key)
        if not source:
            continue
        frame.update(
            {
                "demo_test_mode": True,
                "demo_clip_id": clip.get("clip_id"),
                "local_timestamp": source.get("local_timestamp"),
                "timeline_timestamp": source.get("timeline_timestamp", source.get("timestamp")),
                "display_date": source.get("display_date"),
                "display_time": source.get("display_time"),
                "display_datetime": source.get("display_datetime"),
                "display_iso": source.get("display_iso"),
            }
        )
    _write_jsonl(frames_path, frames)

    open_event_path = current_dir / "open_event.json"
    open_event = read_json(open_event_path, default={})
    if isinstance(open_event, dict) and open_event:
        start_local = max(0.0, float(current_local_time) - float(open_event.get("duration", 0.0) or 0.0))
        open_event.update(
            {
                "demo_test_mode": True,
                "demo_clip_id": clip.get("clip_id"),
                "display_date": display["display_date"],
                "display_time": display["display_time"],
                "display_datetime": display["display_datetime"],
                "display_start": _display_payload_for_clip(clip, start_local),
                "display_end": display,
            }
        )
        for frame in open_event.get("keyframes", []) or []:
            if not isinstance(frame, dict):
                continue
            key = int(round(float(frame.get("timestamp", 0.0) or 0.0) * 1000))
            source = by_ts.get(key)
            if source:
                frame.update(
                    {
                        "demo_clip_id": clip.get("clip_id"),
                        "local_timestamp": source.get("local_timestamp"),
                        "display_date": source.get("display_date"),
                        "display_time": source.get("display_time"),
                        "display_datetime": source.get("display_datetime"),
                    }
                )
        write_json_atomic(open_event_path, open_event)

    state_path = current_dir / "current_state.json"
    state = read_json(state_path, default={})
    if isinstance(state, dict) and state:
        state.update(
            {
                "demo_test_mode": True,
                "demo_clip_id": clip.get("clip_id"),
                "local_current_time": round(float(current_local_time), 3),
                "timeline_current_time": round(float(timeline_time), 3),
                "display_date": display["display_date"],
                "display_time": display["display_time"],
                "display_datetime": display["display_datetime"],
                "display_iso": display["display_iso"],
                "updated_at": utc_now_iso(),
            }
        )
        write_json_atomic(state_path, state)


def _update_demo_test_query_horizon(session_dir: Path, clip: dict[str, Any], display: dict[str, Any]) -> None:
    memory_config_path = session_dir / "worldmm" / "memory_config.json"
    if not memory_config_path.exists():
        return
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        return
    args = dict(config.get("query_rag_args") or {})
    args["until_date"] = str(clip.get("day_label") or args.get("until_date") or "DAY1")
    args["until_time"] = str(display.get("display_hhmmssff") or args.get("until_time") or "23595999")
    config["query_rag_args"] = args
    config["demo_test_active_clip_id"] = clip.get("clip_id")
    config["demo_test_query_horizon"] = {
        "date": args["until_date"],
        "time": args["until_time"],
        "display_date": display.get("display_date"),
        "display_time": display.get("display_time"),
        "display_datetime": display.get("display_datetime"),
    }
    config["updated_at"] = utc_now_iso()
    write_json_atomic(memory_config_path, config)


def _apply_demo_test_tick(
    session_dir: Path,
    *,
    clip_id: str,
    current_time: float,
    paused: bool = False,
    playback_speed: float = 1.0,
) -> dict[str, Any]:
    state = _read_demo_test_state(session_dir)
    clips = list(state.get("clips") or [])
    wanted = _safe_clip_id(clip_id)
    clip = next((dict(item) for item in clips if _safe_clip_id(str(item.get("clip_id") or "")) == wanted), None)
    if clip is None:
        raise FileNotFoundError(f"demo-test clip not found: {clip_id}")

    manifest = _read_demo_test_manifest(session_dir, wanted)
    if not manifest:
        sample_fps = float(clip.get("sample_fps") or state.get("sample_fps") or os.getenv("WORLDMM_DEMO_SAMPLE_FPS", "1.0") or 1.0)
        manifest = _extract_demo_test_clip_frames(session_dir, clip, sample_fps=sample_fps)

    duration = float(clip.get("duration") or 0.0)
    if duration > 0:
        current_time = min(max(0.0, float(current_time)), duration)
    else:
        current_time = max(0.0, float(current_time))
    timeline_time = round(float(clip.get("timeline_offset") or 0.0) + current_time, 3)
    selected = _select_demo_test_window_frames(manifest, timeline_time)
    latest = selected[-1] if selected else None
    display = _display_payload_for_clip(clip, current_time)

    frames_for_mcur = []
    for item in selected:
        ts = float(item.get("timeline_timestamp", item.get("timestamp", 0.0)) or 0.0)
        frames_for_mcur.append(
            {
                "timestamp": ts,
                "path": item.get("path"),
                "source_path": item.get("path"),
                "frame_index": item.get("frame_index"),
                "diff_score": 1.0 if item is latest else 0.0,
                "source": "demo_test_video_playback",
                "created_at": utc_now_iso(),
            }
        )

    mcur_state = MCurStore(session_dir).update_from_chunk(
        chunk_info={
            "chunk_id": f"demo_test_{wanted}_{int(round(timeline_time * 1000)):09d}",
            "start_time": selected[0].get("timeline_timestamp", selected[0].get("timestamp")) if selected else timeline_time,
            "end_time": timeline_time,
            "source": "demo_test_video_playback",
            "demo_clip_id": wanted,
        },
        frames=frames_for_mcur,
        transcript_segments=[],
        diff_stats={
            "max_diff": 1.0 if latest else 0.0,
            "mean_diff": 0.0,
            "last_diff": 1.0 if latest else 0.0,
        },
    )
    _decorate_demo_test_current_memory(
        session_dir,
        clip=clip,
        selected_frames=selected,
        current_local_time=current_time,
        timeline_time=timeline_time,
    )
    _update_demo_test_query_horizon(session_dir, clip, display)

    stream_dir = session_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    frame_state = {
        "session_id": session_dir.name,
        "stream_id": state.get("stream_id") or f"demo_test_stream_{session_dir.name}",
        "input_mode": "demo_test_video",
        "demo_mode": True,
        "demo_test_mode": True,
        "enabled": True,
        "ready": True,
        "mcur_ready": True,
        "status": "paused" if paused else "running",
        "active_clip_id": wanted,
        "accepted_count": len(
            [
                item
                for item in manifest
                if float(item.get("timeline_timestamp", item.get("timestamp", 0.0)) or 0.0) <= timeline_time + 1e-6
            ]
        ),
        "latest_frame_index": latest.get("frame_index") if latest else None,
        "latest_relative_ts_ms": int(round(timeline_time * 1000)),
        "latest_local_ts_ms": int(round(current_time * 1000)),
        "latest_frame_path": latest.get("path") if latest else None,
        "frames": selected[-50:],
        **display,
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(stream_dir / "frame_state.json", frame_state)

    open_event = read_json(session_dir / "current" / "open_event.json", default={})
    if not isinstance(open_event, dict):
        open_event = {}
    write_json_atomic(
        stream_dir / "frame_event_state.json",
        {
            "session_id": session_dir.name,
            "demo_mode": True,
            "demo_test_mode": True,
            "active_clip_id": wanted,
            "open_event": open_event,
            **display,
            "updated_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        stream_dir / "stream_state.json",
        {
            "session_id": session_dir.name,
            "stream_id": frame_state["stream_id"],
            "input_mode": "demo_test_video",
            "demo_mode": True,
            "demo_test_mode": True,
            "status": "paused" if paused else "running",
            "active_clip_id": wanted,
            "duration": duration,
            "current_time": round(timeline_time, 3),
            "local_current_time": round(current_time, 3),
            "playback_speed": playback_speed,
            "last_processed_proc_index": frame_state["latest_frame_index"],
            "last_processed_chunk_index": frame_state["latest_frame_index"],
            **display,
            "updated_at": utc_now_iso(),
        },
    )

    new_state = {
        **state,
        **_demo_test_public_urls(session_dir.name, clips),
        "status": "paused" if paused else "running",
        "prepared": True,
        "active_clip_id": wanted,
        "current_time": round(timeline_time, 3),
        "local_current_time": round(current_time, 3),
        "playback_speed": playback_speed,
        "latest_frame": latest,
        "mcur_state": mcur_state,
        **display,
    }
    _write_demo_test_state(session_dir, new_state)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_dir.name,
            "status": "streaming" if not paused else "paused",
            "stage": "demo_test_playback",
            "progress": 100,
            "demo_mode": True,
            "demo_test_mode": True,
            "active_clip_id": wanted,
            "current_ready": True,
            "current_time": round(timeline_time, 3),
            "local_current_time": round(current_time, 3),
            **display,
            "outputs": {
                "demo_test_video_url": f"/demo-test/{session_dir.name}/video/{wanted}",
                "current_frame_path": latest.get("path") if latest else None,
            },
            "updated_at": utc_now_iso(),
        },
    )
    return new_state


def _apply_demo_tick(session_dir: Path, *, current_time: float, paused: bool = False, playback_speed: float = 1.0) -> dict[str, Any]:
    manifest = _read_manifest(session_dir)
    if not manifest:
        state = _read_state(session_dir)
        sample_fps = float(state.get("sample_fps") or os.getenv("WORLDMM_DEMO_SAMPLE_FPS", "1.0") or 1.0)
        manifest = _extract_frames(session_dir, sample_fps=sample_fps)

    state = _read_state(session_dir)
    duration = float(state.get("duration") or 0.0)
    if duration > 0:
        current_time = min(max(0.0, float(current_time)), duration)
    else:
        current_time = max(0.0, float(current_time))
    selected = _select_window_frames(manifest, current_time)
    latest = selected[-1] if selected else None

    frames_for_mcur = []
    for item in selected:
        ts = float(item.get("timestamp", 0.0) or 0.0)
        frames_for_mcur.append(
            {
                "timestamp": ts,
                "path": item.get("path"),
                "source_path": item.get("path"),
                "frame_index": item.get("frame_index"),
                "diff_score": 1.0 if item is latest else 0.0,
                "source": "demo_video_playback",
                "created_at": utc_now_iso(),
            }
        )

    mcur_state = MCurStore(session_dir).update_from_chunk(
        chunk_info={
            "chunk_id": f"demo_{int(round(current_time * 1000)):09d}",
            "start_time": selected[0].get("timestamp") if selected else current_time,
            "end_time": current_time,
            "source": "demo_video_playback",
        },
        frames=frames_for_mcur,
        transcript_segments=[],
        diff_stats={
            "max_diff": 1.0 if latest else 0.0,
            "mean_diff": 0.0,
            "last_diff": 1.0 if latest else 0.0,
        },
    )

    stream_dir = session_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    frame_state = {
        "session_id": session_dir.name,
        "stream_id": state.get("stream_id") or f"demo_stream_{session_dir.name}",
        "input_mode": "demo_video",
        "demo_mode": True,
        "enabled": True,
        "ready": True,
        "mcur_ready": True,
        "status": "paused" if paused else "running",
        "accepted_count": len([item for item in manifest if float(item.get("timestamp", 0.0) or 0.0) <= current_time + 1e-6]),
        "latest_frame_index": latest.get("frame_index") if latest else None,
        "latest_relative_ts_ms": int(round(current_time * 1000)),
        "latest_frame_path": latest.get("path") if latest else None,
        "frames": selected[-50:],
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(stream_dir / "frame_state.json", frame_state)

    open_event = read_json(session_dir / "current" / "open_event.json", default={})
    if not isinstance(open_event, dict):
        open_event = {}
    write_json_atomic(
        stream_dir / "frame_event_state.json",
        {
            "session_id": session_dir.name,
            "demo_mode": True,
            "open_event": open_event,
            "updated_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        stream_dir / "stream_state.json",
        {
            "session_id": session_dir.name,
            "stream_id": frame_state["stream_id"],
            "input_mode": "demo_video",
            "demo_mode": True,
            "status": "paused" if paused else "running",
            "duration": duration,
            "current_time": round(current_time, 3),
            "playback_speed": playback_speed,
            "last_processed_proc_index": frame_state["latest_frame_index"],
            "last_processed_chunk_index": frame_state["latest_frame_index"],
            "updated_at": utc_now_iso(),
        },
    )

    new_state = {
        **state,
        **_public_urls(session_dir.name),
        "status": "paused" if paused else "running",
        "prepared": True,
        "current_time": round(current_time, 3),
        "playback_speed": playback_speed,
        "latest_frame": latest,
        "mcur_state": mcur_state,
    }
    _write_state(session_dir, new_state)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_dir.name,
            "status": "streaming" if not paused else "paused",
            "stage": "demo_playback",
            "progress": 100,
            "demo_mode": True,
            "current_ready": True,
            "current_time": round(current_time, 3),
            "outputs": {
                "demo_video_url": f"/demo/{session_dir.name}/video",
                "current_frame_path": latest.get("path") if latest else None,
            },
            "updated_at": utc_now_iso(),
        },
    )
    return new_state


def _prepare_session(session_dir: Path, *, sample_fps: Optional[float] = None, force: bool = False) -> dict[str, Any]:
    state = _read_state(session_dir)
    fps = float(sample_fps or state.get("sample_fps") or os.getenv("WORLDMM_DEMO_SAMPLE_FPS", "1.0") or 1.0)
    duration = float(state.get("duration") or 0.0)
    manifest = _extract_frames(session_dir, sample_fps=fps, force=force)
    prepared_state = {
        **state,
        **_public_urls(session_dir.name),
        "status": "prepared",
        "prepared": True,
        "sample_fps": fps,
        "duration": duration,
        "frame_count": len(manifest),
        "manifest_path": _manifest_path(session_dir).relative_to(session_dir).as_posix(),
    }
    _write_state(session_dir, prepared_state)
    write_json_atomic(
        session_dir / "status.json",
        {
            "session_id": session_dir.name,
            "status": "ready",
            "stage": "demo_prepared",
            "progress": 100,
            "demo_mode": True,
            "outputs": {"demo_video_url": f"/demo/{session_dir.name}/video", "demo_frame_count": len(manifest)},
            "updated_at": utc_now_iso(),
        },
    )
    return prepared_state


@app.post("/demo/upload")
async def demo_upload(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    sample_fps: float = Form(default=1.0),
    auto_prepare: bool = Form(default=True),
    enqueue_preprocess: bool = Form(default=False),
    force_preprocess: bool = Form(default=False),
) -> JSONResponse:
    sid = _safe_session_id(session_id)
    session_dir = _create_base_session(sid, original_filename=video.filename)
    demo_dir = _demo_dir(session_dir)
    source_suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
    if source_suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}:
        source_suffix = ".mp4"
    source_path = demo_dir / f"source{source_suffix}"
    with source_path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)
    input_path = _sync_input_video(session_dir, source_path)
    duration = _probe_duration(source_path)
    state = {
        **_public_urls(sid),
        "session_id": sid,
        "demo_mode": True,
        "status": "uploaded",
        "prepared": False,
        "source_video": source_path.relative_to(session_dir).as_posix(),
        "source_video_abs": str(source_path),
        "input_video": input_path.relative_to(session_dir).as_posix(),
        "original_filename": video.filename,
        "duration": round(duration, 3),
        "sample_fps": float(sample_fps or 1.0),
        "created_at": utc_now_iso(),
    }
    _write_state(session_dir, state)
    if auto_prepare:
        state = _prepare_session(session_dir, sample_fps=sample_fps, force=True)
    if enqueue_preprocess:
        try:
            state = {**state, **_enqueue_preprocess(session_dir, force=force_preprocess)}
        except Exception as exc:
            state = {**state, "preprocess_queued": False, "preprocess_error": str(exc)}
            _write_state(session_dir, state)
            return JSONResponse(status_code=500, content={"status": "error", **state})
    else:
        state = {**state, "preprocess_queued": False}
        _write_state(session_dir, state)
    return JSONResponse(status_code=200, content=state)


@app.post("/demo/{session_id}/prepare")
async def demo_prepare(session_id: str, request: Optional[DemoPrepareRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoPrepareRequest()
    try:
        state = _prepare_session(session_dir, sample_fps=request.sample_fps, force=request.force)
        if request.enqueue_preprocess:
            state = {**state, **_enqueue_preprocess(session_dir, force=request.force_preprocess)}
            _write_state(session_dir, state)
        else:
            state = {**state, "preprocess_queued": False}
            _write_state(session_dir, state)
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        state = _read_state(session_dir)
        state["status"] = "prepare_failed"
        state["error"] = str(exc)
        _write_state(session_dir, state)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo/{session_id}/sync_input")
async def demo_sync_input(session_id: str, request: Optional[DemoSyncInputRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoSyncInputRequest()
    try:
        state = _read_state(session_dir)
        source_path = Path(str(state.get("source_video_abs") or ""))
        if not source_path.exists():
            source_rel = str(state.get("source_video") or "")
            source_path = session_dir / source_rel
        if not source_path.exists():
            return JSONResponse(status_code=404, content={"status": "error", "message": "demo source video not found"})
        input_path = _sync_input_video(session_dir, source_path)
        state = {
            **state,
            **_public_urls(session_id),
            "input_video": input_path.relative_to(session_dir).as_posix(),
            "input_synced": True,
        }
        if request.enqueue_preprocess:
            state = {**state, **_enqueue_preprocess(session_dir, force=request.force_preprocess)}
        else:
            state["preprocess_queued"] = False
        _write_state(session_dir, state)
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.get("/demo/{session_id}/status")
async def demo_status(session_id: str) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = _read_state(session_dir)
    state.setdefault("session_id", session_id)
    state.setdefault("demo_mode", True)
    state.setdefault("frame_count", len(_read_manifest(session_dir)))
    state.update(_public_urls(session_id))
    return JSONResponse(status_code=200, content=state)


@app.get("/demo/{session_id}/video", response_model=None)
async def demo_video(session_id: str):
    session_dir = _session_dir(session_id)
    state = _read_state(session_dir)
    source_rel = str(state.get("source_video") or "")
    source_path = session_dir / source_rel
    if not source_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo video not found"})
    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    return FileResponse(source_path, media_type=media_type, filename=source_path.name)


@app.post("/demo/{session_id}/start")
async def demo_start(session_id: str, request: Optional[DemoStartRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoStartRequest()
    state = _apply_demo_tick(
        session_dir,
        current_time=request.current_time,
        paused=False,
        playback_speed=request.playback_speed,
    )
    return JSONResponse(status_code=200, content=state)


@app.post("/demo/{session_id}/tick")
async def demo_tick(session_id: str, request: DemoTickRequest) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = _apply_demo_tick(
        session_dir,
        current_time=request.current_time,
        paused=request.paused,
        playback_speed=request.playback_speed,
    )
    return JSONResponse(status_code=200, content=state)


@app.post("/demo/{session_id}/pause")
async def demo_pause(session_id: str, request: Optional[DemoTickRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    current_time = float((request.current_time if request else _read_state(session_dir).get("current_time")) or 0.0)
    state = _apply_demo_tick(session_dir, current_time=current_time, paused=True, playback_speed=(request.playback_speed if request else 1.0))
    return JSONResponse(status_code=200, content=state)


@app.post("/demo/{session_id}/stop")
async def demo_stop(session_id: str) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = _read_state(session_dir)
    state["status"] = "stopped"
    state["current_time"] = 0.0
    _write_state(session_dir, state)
    return JSONResponse(status_code=200, content=state)


@app.get("/demo/{session_id}/manifest")
async def demo_manifest(session_id: str) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    return JSONResponse(status_code=200, content={"session_id": session_id, "frames": _read_manifest(session_dir)})


def _enqueue_demo_test_offline(session_dir: Path, request: DemoTestOfflineRequest) -> dict[str, Any]:
    state = _read_demo_test_state(session_dir)
    tasks = []
    for clip in state.get("clips", []) or []:
        clip_id = _safe_clip_id(str(clip.get("clip_id") or ""))
        child_id = str(clip.get("child_session_id") or _demo_test_child_session_id(session_dir.name, clip_id))
        child_dir = _session_dir(child_id)
        if not (child_dir / "input.mp4").exists():
            _create_demo_test_child_session(session_dir, clip)
        preprocess_task = enqueue_preprocess_task(PROJECT_ROOT, child_id, force=request.force_preprocess)
        item = {
            "clip_id": clip_id,
            "child_session_id": child_id,
            "preprocess_task_path": str(preprocess_task),
            "preprocess_task_id": preprocess_task.stem,
            "evidence_task_path": None,
            "evidence_task_id": None,
            "evidence_skipped_reason": None,
        }
        if request.enqueue_evidence:
            if (child_dir / "preprocess" / "session_30sec.json").exists():
                evidence_task = enqueue_evidence_task(
                    PROJECT_ROOT,
                    child_id,
                    force=request.force_evidence,
                    backend=os.getenv("WORLDMM_EVIDENCE_CAPTION_BACKEND"),
                    pipeline_mode="legacy",
                    role="demo_test_child",
                )
                item["evidence_task_path"] = str(evidence_task)
                item["evidence_task_id"] = evidence_task.stem
            else:
                item["evidence_skipped_reason"] = "preprocess output is not ready yet; call enqueue_offline again with enqueue_evidence=true after preprocess finishes"
        tasks.append(item)
    state = {**state, "offline_tasks": tasks, "offline_queued_at": utc_now_iso()}
    _write_demo_test_state(session_dir, state)
    return {"offline_queued": True, "tasks": tasks}


@app.post("/demo-test/upload")
async def demo_test_upload(
    day1_video: UploadFile = File(...),
    day2_video: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    sample_fps: float = Form(default=1.0),
    auto_prepare: bool = Form(default=True),
    enqueue_offline: bool = Form(default=False),
    enqueue_evidence: bool = Form(default=False),
    force_preprocess: bool = Form(default=False),
    force_evidence: bool = Form(default=False),
    day1_start: str = Form(default=DEMO_TEST_DEFAULT_DAY1_START),
    day2_start: str = Form(default=DEMO_TEST_DEFAULT_DAY2_START),
    timeline_gap_seconds: float = Form(default=60.0),
) -> JSONResponse:
    try:
        sid = _safe_session_id(session_id or f"demo_test_{uuid4().hex[:12]}")
        session_dir = _create_demo_test_parent_session(sid)
        demo_test_dir = _demo_test_dir(session_dir)
        demo_test_dir.mkdir(parents=True, exist_ok=True)

        uploads = [
            ("day1", "DAY1", day1_video, _parse_demo_datetime(day1_start, DEMO_TEST_DEFAULT_DAY1_START), 0.0),
            ("day2", "DAY2", day2_video, _parse_demo_datetime(day2_start, DEMO_TEST_DEFAULT_DAY2_START), None),
        ]
        clips: list[dict[str, Any]] = []
        previous_duration = 0.0
        for clip_id, day_label, upload, start_dt, explicit_offset in uploads:
            clip_id = _safe_clip_id(clip_id)
            clip_dir = _demo_test_clip_dir(session_dir, clip_id)
            clip_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(upload.filename or "input.mp4").suffix.lower() or ".mp4"
            if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}:
                suffix = ".mp4"
            source_path = clip_dir / f"source{suffix}"
            with source_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            duration = _probe_duration(source_path)
            timeline_offset = float(explicit_offset if explicit_offset is not None else previous_duration + max(0.0, float(timeline_gap_seconds or 0.0)))
            display_start = {
                "display_date": _format_cn_date(start_dt),
                "display_time": _format_time(start_dt),
                "display_datetime": _format_datetime(start_dt),
                "display_iso": start_dt.isoformat(timespec="seconds"),
            }
            clip = {
                "clip_id": clip_id,
                "day_label": day_label,
                "source_video": source_path.relative_to(session_dir).as_posix(),
                "source_video_abs": str(source_path),
                "original_filename": upload.filename,
                "duration": round(duration, 3),
                "sample_fps": float(sample_fps or 1.0),
                "timeline_offset": round(timeline_offset, 3),
                "timeline_start": round(timeline_offset, 3),
                "timeline_end": round(timeline_offset + duration, 3),
                "start_datetime": _format_datetime(start_dt),
                "start_time": _format_time(start_dt),
                "display_date": display_start["display_date"],
                "display_time": display_start["display_time"],
                "display_datetime": display_start["display_datetime"],
                "child_session_id": _demo_test_child_session_id(sid, clip_id),
                "prepared": False,
                "video_url": f"/demo-test/{sid}/video/{clip_id}",
            }
            clips.append(clip)
            previous_duration = timeline_offset + duration

        for clip in clips:
            _create_demo_test_child_session(session_dir, clip)
        if clips:
            _sync_input_video(session_dir, session_dir / str(clips[0]["source_video"]))

        state = {
            **_demo_test_public_urls(sid, clips),
            "session_id": sid,
            "demo_mode": True,
            "demo_test_mode": True,
            "status": "uploaded",
            "prepared": False,
            "sample_fps": float(sample_fps or 1.0),
            "clips": clips,
            "created_at": utc_now_iso(),
        }
        _write_demo_test_state(session_dir, state)
        if auto_prepare:
            state = _prepare_demo_test_session(session_dir, sample_fps=sample_fps, force=True)
        offline_result = {"offline_queued": False}
        if enqueue_offline:
            offline_result = _enqueue_demo_test_offline(
                session_dir,
                DemoTestOfflineRequest(
                    force_preprocess=force_preprocess,
                    enqueue_evidence=enqueue_evidence,
                    force_evidence=force_evidence,
                ),
            )
        state = {**_read_demo_test_state(session_dir), **offline_result}
        _write_demo_test_state(session_dir, state)
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/demo-test/{session_id}/status")
async def demo_test_status(session_id: str) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = _read_demo_test_state(session_dir)
    clips = list(state.get("clips") or [])
    child_status = []
    for clip in clips:
        child_id = str(clip.get("child_session_id") or "")
        child_dir = _session_dir(child_id)
        child_status.append(
            {
                "clip_id": clip.get("clip_id"),
                "child_session_id": child_id,
                "preprocess_ready": (child_dir / "preprocess" / "session_30sec.json").exists(),
                "evidence_ready": (child_dir / "evidence" / "session_evidence.json").exists(),
                "status": read_json(child_dir / "status.json", default={}) if child_dir.exists() else {},
            }
        )
    state.setdefault("session_id", session_id)
    state.setdefault("demo_test_mode", True)
    state.update(_demo_test_public_urls(session_id, clips))
    state["child_status"] = child_status
    state["memory_ready"] = (session_dir / "worldmm" / "memory_config.json").exists()
    return JSONResponse(status_code=200, content=state)


@app.get("/demo-test/{session_id}/video/{clip_id}", response_model=None)
async def demo_test_video(session_id: str, clip_id: str):
    session_dir = _session_dir(session_id)
    state = _read_demo_test_state(session_dir)
    wanted = _safe_clip_id(clip_id)
    clip = next((item for item in state.get("clips", []) or [] if _safe_clip_id(str(item.get("clip_id") or "")) == wanted), None)
    if not clip:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"demo-test clip not found: {clip_id}"})
    source_path = session_dir / str(clip.get("source_video") or "")
    if not source_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo-test video not found"})
    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    return FileResponse(source_path, media_type=media_type, filename=source_path.name)


@app.post("/demo-test/{session_id}/prepare")
async def demo_test_prepare(session_id: str, request: Optional[DemoTestPrepareRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoTestPrepareRequest()
    try:
        state = _prepare_demo_test_session(session_dir, clip_id=request.clip_id, sample_fps=request.sample_fps, force=request.force)
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo-test/{session_id}/start")
async def demo_test_start(session_id: str, request: Optional[DemoTestStartRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoTestStartRequest()
    try:
        state = _apply_demo_test_tick(
            session_dir,
            clip_id=request.clip_id,
            current_time=request.current_time,
            paused=False,
            playback_speed=request.playback_speed,
        )
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo-test/{session_id}/tick")
async def demo_test_tick(session_id: str, request: DemoTestTickRequest) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    try:
        state = _apply_demo_test_tick(
            session_dir,
            clip_id=request.clip_id,
            current_time=request.current_time,
            paused=request.paused,
            playback_speed=request.playback_speed,
        )
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo-test/{session_id}/pause")
async def demo_test_pause(session_id: str, request: Optional[DemoTestTickRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoTestTickRequest(current_time=float(_read_demo_test_state(session_dir).get("local_current_time") or 0.0))
    try:
        state = _apply_demo_test_tick(
            session_dir,
            clip_id=request.clip_id,
            current_time=request.current_time,
            paused=True,
            playback_speed=request.playback_speed,
        )
        return JSONResponse(status_code=200, content=state)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo-test/{session_id}/enqueue_offline")
async def demo_test_enqueue_offline(session_id: str, request: Optional[DemoTestOfflineRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoTestOfflineRequest()
    try:
        result = _enqueue_demo_test_offline(session_dir, request)
        return JSONResponse(status_code=202, content={"status": "queued", "session_id": session_id, **result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.post("/demo-test/{session_id}/build_memory")
async def demo_test_build_memory(session_id: str, request: Optional[DemoTestBuildMemoryRequest] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    request = request or DemoTestBuildMemoryRequest()
    try:
        result = _build_demo_test_parent_memory(
            session_dir,
            force=request.force,
            allow_manifest_fallback=request.allow_manifest_fallback,
            skip_semantic=request.skip_semantic,
            generation_backend=request.generation_backend,
        )
        status_code = 202 if result.get("status") == "waiting" else 200
        return JSONResponse(status_code=status_code, content=result)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "session_id": session_id})


@app.get("/demo-test/{session_id}/manifest")
async def demo_test_manifest(session_id: str, clip_id: Optional[str] = None) -> JSONResponse:
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"session not found: {session_id}"})
    state = _read_demo_test_state(session_dir)
    clips = list(state.get("clips") or [])
    if clip_id:
        wanted = _safe_clip_id(clip_id)
        return JSONResponse(status_code=200, content={"session_id": session_id, "clip_id": wanted, "frames": _read_demo_test_manifest(session_dir, wanted)})
    return JSONResponse(
        status_code=200,
        content={
            "session_id": session_id,
            "clips": [
                {
                    "clip_id": clip.get("clip_id"),
                    "frames": _read_demo_test_manifest(session_dir, str(clip.get("clip_id") or "")),
                }
                for clip in clips
            ],
        },
    )
