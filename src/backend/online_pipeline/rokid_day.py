from __future__ import annotations

import copy
import shutil
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic

from .file_lock import FileLock


DAY_CHILD_SEPARATOR = "__day"
DAY_STATE_RELATIVE_PATH = Path("stream") / "day_state.json"
DAY_MERGE_STATE_RELATIVE_PATH = Path("stream") / "day_merge_state.json"


def _format_cn_date(value: datetime) -> str:
    return f"{value.year}年{value.month}月{value.day}日"


def _format_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S")


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _seconds_to_hhmmssff(seconds: float, fps_for_code: int = 100) -> str:
    total_frames = max(0, int(round(float(seconds or 0.0) * fps_for_code)))
    total_seconds, frames = divmod(total_frames, fps_for_code)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}{frames:02d}"


def _parse_iso_datetime(value: Any, tz: timezone | ZoneInfo) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            return parsed.astimezone(tz)
        except Exception:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=tz)
        except Exception:
            continue
    return None


def _timezone_from_metadata(metadata: dict[str, Any]) -> tuple[timezone | ZoneInfo, str]:
    tz_id = str(metadata.get("client_timezone_id") or metadata.get("timezone") or "").strip()
    if tz_id:
        try:
            return ZoneInfo(tz_id), tz_id
        except ZoneInfoNotFoundError:
            pass
    try:
        offset = int(metadata.get("client_timezone_offset_minutes"))
        return timezone(timedelta(minutes=offset)), f"UTC{offset / 60:+g}"
    except Exception:
        return timezone.utc, "UTC"


def _safe_epoch_ms(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def build_rokid_time_context(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(metadata or {})
    tz, tz_label = _timezone_from_metadata(payload)
    epoch_ms = _safe_epoch_ms(payload.get("client_session_start_ts_ms") or payload.get("client_start_ts_ms"))
    time_source = "client_device"
    if epoch_ms is not None:
        start = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).astimezone(tz)
    else:
        start = _parse_iso_datetime(payload.get("client_start_datetime") or payload.get("start_datetime"), tz)
        if start is None:
            start = _parse_iso_datetime(utc_now_iso(), timezone.utc) or datetime.now(timezone.utc)
            tz = timezone.utc
            tz_label = "UTC"
            time_source = "server_receive_fallback"
        else:
            time_source = "client_device"
    return {
        "start_datetime": _format_datetime(start),
        "display_date": _format_cn_date(start),
        "display_time": _format_time(start),
        "display_datetime": _format_datetime(start),
        "display_iso": start.isoformat(timespec="seconds"),
        "display_hhmmssff": _seconds_to_hhmmssff(start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1_000_000),
        "timezone": tz_label,
        "time_source": time_source,
        "client_session_start_ts_ms": epoch_ms,
        "client_timezone_offset_minutes": payload.get("client_timezone_offset_minutes"),
    }


def _parse_context_start(value: dict[str, Any]) -> datetime | None:
    tz, _ = _timezone_from_metadata(value)
    return _parse_iso_datetime(value.get("display_iso") or value.get("start_datetime") or value.get("display_datetime"), tz)


def rokid_display_payload_for_relative_time(context: dict[str, Any] | None, relative_seconds: float) -> dict[str, Any]:
    payload = dict(context or {})
    start = _parse_context_start(payload)
    if start is None:
        fallback = build_rokid_time_context(payload)
        start = _parse_context_start(fallback) or datetime.now(timezone.utc)
        payload = {**fallback, **payload}
    value = start + timedelta(seconds=max(0.0, float(relative_seconds or 0.0)))
    return {
        "display_date": _format_cn_date(value),
        "display_time": _format_time(value),
        "display_datetime": _format_datetime(value),
        "display_iso": value.isoformat(timespec="seconds"),
        "display_hhmmssff": _seconds_to_hhmmssff(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000),
        "timezone": payload.get("timezone") or "UTC",
        "time_source": payload.get("time_source") or "unknown",
    }


def valid_session_id(session_id: str) -> bool:
    return bool(session_id) and all(ch.isalnum() or ch in {"-", "_"} for ch in session_id)


def normalize_run_id(run_id: Any) -> str:
    text = str(run_id or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", text)[:128]


def normalize_day_label(day_index: Any) -> str:
    try:
        index = max(1, int(day_index))
    except Exception:
        index = 1
    return f"DAY{index}"


def child_session_id(parent_session_id: str, day_index: int) -> str:
    return f"{parent_session_id}{DAY_CHILD_SEPARATOR}{day_index:04d}"


def day_state_path(parent_dir: Path) -> Path:
    return parent_dir / DAY_STATE_RELATIVE_PATH


def day_merge_state_path(parent_dir: Path) -> Path:
    return parent_dir / DAY_MERGE_STATE_RELATIVE_PATH


def _empty_day_state(parent_session_id: str) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "parent_session_id": parent_session_id,
        "next_day_index": 1,
        "runs": {},
        "created_at": now,
        "updated_at": now,
    }


def load_day_state(parent_dir: Path, parent_session_id: str) -> dict[str, Any]:
    state = read_json(day_state_path(parent_dir), default={})
    if not isinstance(state, dict) or not state:
        return _empty_day_state(parent_session_id)
    state.setdefault("parent_session_id", parent_session_id)
    state.setdefault("next_day_index", 1)
    if not isinstance(state.get("runs"), dict):
        state["runs"] = {}
    return state


def save_day_state(parent_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    write_json_atomic(day_state_path(parent_dir), state)


def reserve_rokid_day_child(
    *,
    sessions_root: Path,
    parent_session_id: str,
    run_id: str,
    input_mode: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reserve or return the child session assigned to a Rokid Start run."""

    if not valid_session_id(parent_session_id):
        raise ValueError("invalid parent_session_id")
    normalized_run_id = normalize_run_id(run_id)
    if not normalized_run_id:
        raise ValueError("run_id is required")

    parent_dir = sessions_root / parent_session_id
    parent_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = parent_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(stream_dir / "day_state.lock"), timeout=30)
    with lock:
        state = load_day_state(parent_dir, parent_session_id)
        runs = state.setdefault("runs", {})
        existing = runs.get(normalized_run_id)
        if isinstance(existing, dict) and existing.get("child_session_id"):
            return copy.deepcopy(existing), state

        used_day_indices: set[int] = set()
        for run in runs.values():
            if not isinstance(run, dict):
                continue
            status = str(run.get("status") or "").strip().lower()
            if status not in {"reserved", "started"}:
                continue
            try:
                used_day_indices.add(max(1, int(run.get("day_index") or 1)))
            except Exception:
                continue
        next_day_index = int(state.get("next_day_index") or 1)
        while next_day_index in used_day_indices:
            next_day_index += 1
        child_id = child_session_id(parent_session_id, next_day_index)
        now = utc_now_iso()
        time_context = build_rokid_time_context(metadata)
        run = {
            "run_id": normalized_run_id,
            "day_index": next_day_index,
            "day_label": normalize_day_label(next_day_index),
            "parent_session_id": parent_session_id,
            "child_session_id": child_id,
            "status": "reserved",
            "input_mode": input_mode,
            **time_context,
            "created_at": now,
            "updated_at": now,
        }
        runs[normalized_run_id] = run
        save_day_state(parent_dir, state)
        return copy.deepcopy(run), state


def mark_rokid_day_started(
    *,
    sessions_root: Path,
    parent_session_id: str,
    run_id: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    parent_dir = sessions_root / parent_session_id
    stream_dir = parent_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    normalized_run_id = normalize_run_id(run_id)
    lock = FileLock(str(stream_dir / "day_state.lock"), timeout=30)
    with lock:
        state = load_day_state(parent_dir, parent_session_id)
        run = state.setdefault("runs", {}).get(normalized_run_id)
        if not isinstance(run, dict):
            raise ValueError(f"run_id not reserved: {normalized_run_id}")
        day_index = int(run.get("day_index") or 1)
        run["status"] = "started"
        run["started_at"] = run.get("started_at") or utc_now_iso()
        run["updated_at"] = utc_now_iso()
        run["start_response"] = copy.deepcopy(response)
        state["next_day_index"] = max(int(state.get("next_day_index") or 1), day_index + 1)
        save_day_state(parent_dir, state)
        return copy.deepcopy(run)


def mark_rokid_day_failed(
    *,
    sessions_root: Path,
    parent_session_id: str,
    run_id: str,
    error: str,
) -> None:
    parent_dir = sessions_root / parent_session_id
    stream_dir = parent_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    normalized_run_id = normalize_run_id(run_id)
    lock = FileLock(str(stream_dir / "day_state.lock"), timeout=30)
    with lock:
        state = load_day_state(parent_dir, parent_session_id)
        run = state.setdefault("runs", {}).get(normalized_run_id)
        if isinstance(run, dict) and str(run.get("status") or "") == "reserved":
            run["status"] = "failed"
            run["error"] = str(error)
            run["updated_at"] = utc_now_iso()
            state["runs"].pop(normalized_run_id, None)
            save_day_state(parent_dir, state)


def day_context_for_run(run: dict[str, Any]) -> dict[str, Any]:
    context = {
        "day_label": str(run.get("day_label") or normalize_day_label(run.get("day_index"))),
        "day_index": int(run.get("day_index") or 1),
        "run_id": str(run.get("run_id") or ""),
    }
    for key in (
        "start_datetime",
        "display_date",
        "display_time",
        "display_datetime",
        "display_iso",
        "display_hhmmssff",
        "timezone",
        "time_source",
        "client_session_start_ts_ms",
        "client_timezone_offset_minutes",
    ):
        if run.get(key) is not None:
            context[key] = run.get(key)
    return context


def enrich_start_response_for_day(response: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    content = copy.deepcopy(response)
    parent_session_id = str(run.get("parent_session_id") or "")
    child_id = str(run.get("child_session_id") or content.get("session_id") or "")
    content["session_id"] = child_id
    content["parent_session_id"] = parent_session_id
    content["child_session_id"] = child_id
    content["day_context"] = day_context_for_run(run)
    return content


def child_metadata_patch(run: dict[str, Any]) -> dict[str, Any]:
    day_context = day_context_for_run(run)
    parent_session_id = str(run.get("parent_session_id") or "")
    child_id = str(run.get("child_session_id") or "")
    return {
        "parent_session_id": parent_session_id,
        "child_session_id": child_id,
        "is_rokid_day_child": True,
        "day_label": day_context["day_label"],
        "day_index": day_context["day_index"],
        "run_id": day_context["run_id"],
        **{key: value for key, value in day_context.items() if key not in {"day_label", "day_index", "run_id"}},
    }


def update_child_metadata(session_dir: Path, run: dict[str, Any]) -> None:
    path = session_dir / "metadata.json"
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        payload = {}
    patch = child_metadata_patch(run)
    payload.update(patch)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(patch)
    payload["metadata"] = metadata
    write_json_atomic(path, payload)


def cleanup_failed_child_reservation(sessions_root: Path, child_session_id: str) -> None:
    """Remove an empty failed DAY child so the DAY number can be reclaimed."""

    child_dir = sessions_root / child_session_id
    if not child_dir.exists() or not child_dir.is_dir():
        return
    metadata = load_rokid_day_child_metadata(child_dir)
    if metadata is None:
        return
    try:
        shutil.rmtree(child_dir)
    except Exception:
        return


def load_rokid_day_child_metadata(session_dir: Path) -> dict[str, Any] | None:
    payload = read_json(session_dir / "metadata.json", default={})
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    merged = {**metadata, **payload}
    if not merged.get("is_rokid_day_child"):
        return None
    parent_session_id = str(merged.get("parent_session_id") or "").strip()
    child_id = str(merged.get("child_session_id") or payload.get("session_id") or session_dir.name).strip()
    if not parent_session_id:
        return None
    try:
        day_index = int(merged.get("day_index") or 1)
    except Exception:
        day_index = 1
    return {
        "parent_session_id": parent_session_id,
        "child_session_id": child_id,
        "day_label": str(merged.get("day_label") or normalize_day_label(day_index)),
        "day_index": day_index,
        "run_id": str(merged.get("run_id") or ""),
        **{
            key: merged.get(key)
            for key in (
                "start_datetime",
                "display_date",
                "display_time",
                "display_datetime",
                "display_iso",
                "display_hhmmssff",
                "timezone",
                "time_source",
                "client_session_start_ts_ms",
                "client_timezone_offset_minutes",
            )
            if merged.get(key) is not None
        },
    }


def query_memory_ready(session_dir: Path) -> bool:
    config = read_json(session_dir / "worldmm" / "memory_config.json", default={})
    if not isinstance(config, dict):
        return False
    return bool(
        config.get("latest_ready_memory_version")
        or config.get("latest_fast_ready_version")
        or config.get("memory_version")
        or config.get("long_term_partial_ready")
        or str(config.get("status") or "") == "memory_ready"
    )


def resolve_query_long_term_candidates(
    session_id: str,
    sessions_root: Path,
    *,
    question: str = "",
    query_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(query_context or resolve_query_session_context(session_id, sessions_root))
    sessions_root = Path(sessions_root)
    preferred_id = str(context.get("long_term_session_id") or session_id)
    parent_id = str(context.get("parent_session_id") or preferred_id)
    child_id = str(context.get("child_session_id") or session_id)
    is_day_child = bool(context.get("is_rokid_day_child"))

    def candidate(session: str, role: str, reason: str) -> dict[str, Any]:
        session_dir = sessions_root / session
        return {
            "session_id": session,
            "role": role,
            "ready": query_memory_ready(session_dir),
            "memory_config_exists": (session_dir / "worldmm" / "memory_config.json").exists(),
            "reason": reason,
        }

    ordered: list[dict[str, Any]] = []
    if is_day_child:
        ordered.append(candidate(parent_id, "parent", "preferred cross-day parent memory"))
        ordered.append(candidate(child_id, "current_child", "current day child fallback"))
    else:
        ordered.append(candidate(preferred_id, "session", "single-session long-term memory"))

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ordered:
        sid = str(item.get("session_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        unique.append(item)

    selected = next((item for item in unique if item.get("ready")), unique[0] if unique else candidate(preferred_id, "session", "fallback"))
    return {
        "preferred_session_id": preferred_id,
        "selected_session_id": selected.get("session_id"),
        "selected_role": selected.get("role"),
        "selected_reason": selected.get("reason"),
        "target_day_index": None,
        "target_child_session_id": None,
        "candidates": unique,
    }


def resolve_query_session_context(session_id: str, sessions_root: Path) -> dict[str, Any]:
    session_dir = sessions_root / session_id
    child_meta = load_rokid_day_child_metadata(session_dir)
    if child_meta:
        parent_session_id = child_meta["parent_session_id"]
        return {
            "session_id": session_id,
            "is_rokid_day_child": True,
            "realtime_session_id": session_id,
            "short_term_session_id": session_id,
            "long_term_session_id": parent_session_id,
            "interaction_cache_session_id": session_id,
            "parent_session_id": parent_session_id,
            "child_session_id": child_meta["child_session_id"],
            "day_label": child_meta["day_label"],
            "day_index": child_meta["day_index"],
            "run_id": child_meta["run_id"],
            **{
                key: child_meta.get(key)
                for key in (
                    "start_datetime",
                    "display_date",
                    "display_time",
                    "display_datetime",
                    "display_iso",
                    "display_hhmmssff",
                    "timezone",
                    "time_source",
                    "client_session_start_ts_ms",
                    "client_timezone_offset_minutes",
                )
                if child_meta.get(key) is not None
            },
        }
    return {
        "session_id": session_id,
        "is_rokid_day_child": False,
        "realtime_session_id": session_id,
        "short_term_session_id": session_id,
        "long_term_session_id": session_id,
        "interaction_cache_session_id": session_id,
        "parent_session_id": session_id,
        "child_session_id": session_id,
        "day_label": None,
        "day_index": None,
        "run_id": "",
    }
