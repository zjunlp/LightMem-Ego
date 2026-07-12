from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any

from online_memory import build_online_em2mem_memory
from online_preprocess.io_utils import read_json, relative_to_session, utc_now_iso, write_json_atomic

from .file_lock import FileLock
from .rokid_day import (
    day_merge_state_path,
    load_rokid_day_child_metadata,
    normalize_day_label,
    rokid_display_payload_for_relative_time,
)


REQUIRED_CHILD_OUTPUTS = (
    Path("em2mem") / "mst_episodic" / "mst_30sec_episodes.json",
    Path("evidence") / "mst_session_evidence.json",
    Path("captions") / "mst_session_30sec_captioned.json",
)

CHILD_READY_STATUS = {"done"}
CHILD_READY_STAGES = {"visual_embedding_ready"}
ENDED_STREAM_STATUS = {"stream_ended"}
ENDED_STREAM_STAGES = {"stream_ended"}
STREAM_TERMINAL_STATUS = {"ended", "stopped", "aborted", "cancelled", "canceled"}


def _safe_doc_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _stable_parent_doc_id(parent_session_id: str, day_label: str, child_doc_id: Any, fallback: str) -> str:
    return f"{parent_session_id}__{day_label}__{_safe_doc_token(child_doc_id, fallback)}"


def _merge_unique_by_id(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], id_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for item in existing + incoming:
        if not isinstance(item, dict):
            continue
        doc_id = ""
        for key in id_keys:
            if item.get(key):
                doc_id = str(item[key])
                break
        if not doc_id:
            doc_id = f"_row_{len(output):06d}"
        if doc_id in index_by_id:
            output[index_by_id[doc_id]] = item
        else:
            index_by_id[doc_id] = len(output)
            output.append(item)
    return output


def _copy_asset(child_dir: Path, parent_dir: Path, day_label: str, rel_path: Any) -> str:
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    if not rel:
        return rel
    source = child_dir / rel
    if not source.exists() or not source.is_file():
        return rel
    target_rel = Path("stream") / "day_assets" / day_label / rel
    target = parent_dir / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(source, target)
        except Exception:
            shutil.copy2(source, target)
    return target_rel.as_posix()


def _rewrite_keyframe_paths(item: dict[str, Any], child_dir: Path, parent_dir: Path, day_label: str) -> None:
    if isinstance(item.get("keyframe_paths"), list):
        item["keyframe_paths"] = [_copy_asset(child_dir, parent_dir, day_label, path) for path in item["keyframe_paths"]]
    if isinstance(item.get("keyframe_captions"), list):
        rewritten = []
        for caption in item["keyframe_captions"]:
            if isinstance(caption, dict):
                cap = dict(caption)
                if cap.get("path"):
                    cap["path"] = _copy_asset(child_dir, parent_dir, day_label, cap.get("path"))
                if cap.get("image_path"):
                    cap["image_path"] = _copy_asset(child_dir, parent_dir, day_label, cap.get("image_path"))
                rewritten.append(cap)
            else:
                rewritten.append(caption)
        item["keyframe_captions"] = rewritten
    for key in ("path", "image_path"):
        if item.get(key):
            item[key] = _copy_asset(child_dir, parent_dir, day_label, item.get(key))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _relative_seconds(value: Any) -> float:
    seconds = _safe_float(value, 0.0)
    return seconds / 1000.0 if seconds >= 100000.0 else seconds


def _time_range_from_item(item: dict[str, Any]) -> tuple[float, float]:
    start = _relative_seconds(
        item.get("local_start_time")
        if item.get("local_start_time") is not None
        else item.get("start")
        if item.get("start") is not None
        else item.get("start_time")
    )
    end = _relative_seconds(
        item.get("local_end_time")
        if item.get("local_end_time") is not None
        else item.get("end")
        if item.get("end") is not None
        else item.get("end_time")
        if item.get("end_time") is not None
        else start
    )
    if end < start:
        end = start
    return start, end


def _prefix_time_label(text: Any, *, display_start: dict[str, Any], display_end: dict[str, Any]) -> str:
    clean = str(text or "").strip()
    label = f"[{display_start['display_date']} {display_start['display_time']}-{display_end['display_time']}]"
    if not clean:
        return label
    return clean if clean.startswith(label) else f"{label} {clean}"


def _apply_display_time(
    output: dict[str, Any],
    *,
    day_label: str,
    time_context: dict[str, Any],
) -> None:
    local_start, local_end = _time_range_from_item(output)
    display_start = rokid_display_payload_for_relative_time(time_context, local_start)
    display_end = rokid_display_payload_for_relative_time(time_context, local_end)
    output["date"] = day_label
    output["day_label"] = day_label
    output["start_time"] = display_start["display_hhmmssff"]
    output["end_time"] = display_end["display_hhmmssff"]
    output["local_start_time"] = round(local_start, 3)
    output["local_end_time"] = round(local_end, 3)
    output["display_date"] = display_start["display_date"]
    output["display_start_time"] = display_start["display_time"]
    output["display_end_time"] = display_end["display_time"]
    output["display_time_range"] = f"{display_start['display_time']}-{display_end['display_time']}"
    output["display_datetime_start"] = display_start["display_datetime"]
    output["display_datetime_end"] = display_end["display_datetime"]
    output["display_iso_start"] = display_start["display_iso"]
    output["display_iso_end"] = display_end["display_iso"]
    output["timezone"] = display_start.get("timezone")
    output["time_source"] = display_start.get("time_source")
    for key in ("text", "caption", "fine_caption", "visual_summary", "scene"):
        if output.get(key):
            output[key] = _prefix_time_label(output.get(key), display_start=display_start, display_end=display_end)
    if isinstance(output.get("keyframe_captions"), list):
        rewritten = []
        for caption in output["keyframe_captions"]:
            if isinstance(caption, dict):
                cap = dict(caption)
                frame_seconds = _relative_seconds(cap.get("local_timestamp") if cap.get("local_timestamp") is not None else cap.get("timestamp"))
                cap.update(rokid_display_payload_for_relative_time(time_context, frame_seconds))
                rewritten.append(cap)
            else:
                rewritten.append(caption)
        output["keyframe_captions"] = rewritten


def _rewrite_common(
    item: dict[str, Any],
    *,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    child_dir: Path,
    parent_dir: Path,
    id_value: str,
    time_context: dict[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(item)
    output["session_id"] = parent_session_id
    output["parent_session_id"] = parent_session_id
    output["child_session_id"] = child_session_id
    output["source_child_session_id"] = child_session_id
    output["date"] = day_label
    output["day_label"] = day_label
    output["source_doc_id"] = str(id_value)
    output.setdefault("source_doc_ids", [str(id_value)])
    _apply_display_time(output, day_label=day_label, time_context=time_context)
    _rewrite_keyframe_paths(output, child_dir, parent_dir, day_label)
    return output


def _rewrite_caption_docs(
    docs: list[dict[str, Any]],
    *,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    child_dir: Path,
    parent_dir: Path,
    time_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for idx, item in enumerate(docs):
        if not isinstance(item, dict):
            continue
        child_doc_id = str(item.get("doc_id") or item.get("episode_id") or f"caption_{idx:06d}")
        doc_id = _stable_parent_doc_id(parent_session_id, day_label, child_doc_id, f"caption_{idx:06d}")
        output = _rewrite_common(
            item,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            id_value=child_doc_id,
            time_context=time_context,
        )
        output["doc_id"] = doc_id
        output["evidence_doc_id"] = doc_id
        output["segment_id"] = output.get("segment_id") or doc_id
        output["episode_id"] = output.get("episode_id") or doc_id
        rewritten.append(output)
    return rewritten


def _rewrite_evidence_docs(
    docs: list[dict[str, Any]],
    *,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    child_dir: Path,
    parent_dir: Path,
    time_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for idx, item in enumerate(docs):
        if not isinstance(item, dict):
            continue
        child_doc_id = str(item.get("evidence_doc_id") or item.get("doc_id") or item.get("episode_id") or f"evidence_{idx:06d}")
        doc_id = _stable_parent_doc_id(parent_session_id, day_label, child_doc_id, f"evidence_{idx:06d}")
        output = _rewrite_common(
            item,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            id_value=child_doc_id,
            time_context=time_context,
        )
        output["doc_id"] = doc_id
        output["evidence_doc_id"] = doc_id
        output["episode_id"] = output.get("episode_id") or doc_id
        output["segment_id"] = output.get("segment_id") or doc_id
        rewritten.append(output)
    return rewritten


def _rewrite_episode_docs(
    docs: list[dict[str, Any]],
    *,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    child_dir: Path,
    parent_dir: Path,
    time_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for idx, item in enumerate(docs):
        if not isinstance(item, dict):
            continue
        child_doc_id = str(item.get("episode_id") or item.get("doc_id") or f"episode_{idx:06d}")
        doc_id = _stable_parent_doc_id(parent_session_id, day_label, child_doc_id, f"episode_{idx:06d}")
        output = _rewrite_common(
            item,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            id_value=child_doc_id,
            time_context=time_context,
        )
        output["episode_id"] = doc_id
        output["doc_id"] = doc_id
        output["segment_id"] = output.get("segment_id") or doc_id
        rewritten.append(output)
    return rewritten


def _load_list(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    return data if isinstance(data, list) else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _child_memory_ready_missing(child_dir: Path) -> list[str]:
    missing: list[str] = []
    memory_config = read_json(child_dir / "em2mem" / "memory_config.json", default={})
    if not isinstance(memory_config, dict):
        return ["em2mem/memory_config.json"]
    status = str(memory_config.get("status") or "").strip()
    memory_build_state = str(memory_config.get("memory_build_state") or "").strip()
    if status != "memory_ready" and memory_build_state != "ready":
        missing.append(f"em2mem/memory_config.json:status={status or 'missing'}")
        missing.append(f"em2mem/memory_config.json:memory_build_state={memory_build_state or 'missing'}")
    if not _bool_value(memory_config.get("long_term_partial_ready"), False):
        missing.append("em2mem/memory_config.json:long_term_partial_ready=false")
    readiness = memory_config.get("readiness") if isinstance(memory_config.get("readiness"), dict) else {}
    visual_ready = _bool_value(readiness.get("visual_ready"), _bool_value(memory_config.get("visual_embedding_ready"), False))
    visual_lagging = _bool_value(memory_config.get("visual_lagging"), False)
    lag = memory_config.get("lag") if isinstance(memory_config.get("lag"), dict) else {}
    if lag:
        visual_lagging = _bool_value(lag.get("visual_lagging"), visual_lagging)
    # if not visual_ready:
    #     missing.append("em2mem/memory_config.json:visual_ready=false")
    # if visual_lagging:
    #     missing.append("em2mem/memory_config.json:visual_lagging=true")
    return missing


def _child_memory_ready(child_dir: Path) -> bool:
    return not _child_memory_ready_missing(child_dir)


def _stream_is_terminal(child_dir: Path, status_value: str, stage_value: str) -> bool:
    if status_value in CHILD_READY_STATUS and stage_value in CHILD_READY_STAGES:
        return True
    if status_value in ENDED_STREAM_STATUS and stage_value in ENDED_STREAM_STAGES:
        return True
    stream_state = read_json(child_dir / "stream" / "stream_state.json", default={})
    if not isinstance(stream_state, dict):
        return False
    stream_status = str(stream_state.get("status") or stream_state.get("stream_status") or "").strip().lower()
    if stream_status in STREAM_TERMINAL_STATUS:
        return True
    return stream_status == "ending" and bool(stream_state.get("ended_at"))


def missing_child_outputs(child_dir: Path) -> list[str]:
    missing = [path.as_posix() for path in REQUIRED_CHILD_OUTPUTS if not (child_dir / path).exists()]

    status = read_json(child_dir / "status.json", default={})
    if not isinstance(status, dict):
        status = {}
    status_value = str(status.get("status") or "").strip()
    stage_value = str(status.get("stage") or "").strip()
    status_ready = status_value in CHILD_READY_STATUS and stage_value in CHILD_READY_STAGES
    memory_ready_missing = _child_memory_ready_missing(child_dir)
    memory_ready = not memory_ready_missing
    if status_ready:
        pass
    elif memory_ready and _stream_is_terminal(child_dir, status_value, stage_value):
        pass
    elif memory_ready:
        missing.append(f"status.json:status={status_value or 'missing'}")
        missing.append(f"status.json:stage={stage_value or 'missing'}")
    else:
        missing.extend(memory_ready_missing)
        if status_value not in CHILD_READY_STATUS:
            missing.append(f"status.json:status={status_value or 'missing'}")
        if stage_value not in CHILD_READY_STAGES:
            missing.append(f"status.json:stage={stage_value or 'missing'}")

    refine_state_path = child_dir / "short_term" / "refine" / "refine_state.json"
    refine_state = read_json(refine_state_path, default=None)
    if not isinstance(refine_state, dict):
        missing.append("short_term/refine/refine_state.json")
    else:
        pending_event_count = _int_value(refine_state.get("pending_event_count"))
        if pending_event_count:
            missing.append(f"short_term/refine/refine_state.json:pending_event_count={pending_event_count}")

    consolidation_state_path = child_dir / "short_term" / "consolidation_state.json"
    consolidation_state = read_json(consolidation_state_path, default=None)
    if not isinstance(consolidation_state, dict):
        missing.append("short_term/consolidation_state.json")
    else:
        pending_ready_window_count = _int_value(consolidation_state.get("pending_ready_window_count"))
        if pending_ready_window_count:
            missing.append(f"short_term/consolidation_state.json:pending_ready_window_count={pending_ready_window_count}")

    append_state = read_json(child_dir / "em2mem" / "incremental" / "append_state.json", default={})
    if not isinstance(append_state, dict):
        missing.append("em2mem/incremental/append_state.json")
    else:
        pending_count = _int_value(append_state.get("pending_count"))
        failed_count = _int_value(append_state.get("failed_count"))
        if pending_count:
            missing.append(f"em2mem/incremental/append_state.json:pending_count={pending_count}")
        if failed_count:
            missing.append(f"em2mem/incremental/append_state.json:failed_count={failed_count}")

    return missing


def _write_merge_state(parent_dir: Path, child_session_id: str, payload: dict[str, Any]) -> None:
    state_path = day_merge_state_path(parent_dir)
    state = read_json(state_path, default={})
    if not isinstance(state, dict):
        state = {}
    merges = state.get("merges") if isinstance(state.get("merges"), dict) else {}
    merges[child_session_id] = payload
    state["merges"] = merges
    state["updated_at"] = utc_now_iso()
    write_json_atomic(state_path, state)


def record_rokid_day_merge_waiting(
    *,
    sessions_root: Path,
    parent_session_id: str,
    child_session_id: str,
    day_label: str,
    day_index: int,
    run_id: str,
    missing: list[str],
    retry_count: int,
) -> dict[str, Any]:
    parent_dir = sessions_root / parent_session_id
    payload = {
        "status": "waiting",
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "day_label": day_label,
        "day_index": int(day_index),
        "run_id": run_id,
        "missing_outputs": missing,
        "retry_count": int(retry_count),
        "updated_at": utc_now_iso(),
    }
    _write_merge_state(parent_dir, child_session_id, payload)
    return payload


def merge_rokid_day_child(
    *,
    sessions_root: Path,
    parent_session_id: str,
    child_session_id: str,
    day_label: str | None,
    day_index: int | None,
    run_id: str = "",
    force_rebuild: bool = True,
    skip_visual_embedding: bool = True,
    skip_semantic: bool = False,
) -> dict[str, Any]:
    parent_dir = sessions_root / parent_session_id
    child_dir = sessions_root / child_session_id
    if not parent_dir.exists():
        raise FileNotFoundError(f"parent session not found: {parent_session_id}")
    if not child_dir.exists():
        raise FileNotFoundError(f"child session not found: {child_session_id}")

    resolved_day_index = int(day_index or 1)
    resolved_day_label = str(day_label or normalize_day_label(resolved_day_index))
    child_meta = load_rokid_day_child_metadata(child_dir) or {}
    time_context = {
        **child_meta,
        "day_label": resolved_day_label,
        "day_index": resolved_day_index,
    }
    missing = missing_child_outputs(child_dir)
    if missing:
        return {
            "status": "waiting",
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "day_label": resolved_day_label,
            "day_index": resolved_day_index,
            "run_id": run_id,
            "missing_outputs": missing,
        }

    lock_path = parent_dir / "stream" / "day_merge_state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=60):
        child_episodes = _load_list(child_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json")
        child_evidence = _load_list(child_dir / "evidence" / "mst_session_evidence.json")
        child_captions = _load_list(child_dir / "captions" / "mst_session_30sec_captioned.json")

        incoming_episodes = _rewrite_episode_docs(
            child_episodes,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=resolved_day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            time_context=time_context,
        )
        incoming_evidence = _rewrite_evidence_docs(
            child_evidence,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=resolved_day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            time_context=time_context,
        )
        incoming_captions = _rewrite_caption_docs(
            child_captions,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            day_label=resolved_day_label,
            child_dir=child_dir,
            parent_dir=parent_dir,
            time_context=time_context,
        )

        parent_episode_path = parent_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json"
        parent_evidence_path = parent_dir / "evidence" / "mst_session_evidence.json"
        parent_caption_path = parent_dir / "captions" / "mst_session_30sec_captioned.json"
        episodes = _merge_unique_by_id(_load_list(parent_episode_path), incoming_episodes, ("episode_id", "doc_id"))
        evidence = _merge_unique_by_id(_load_list(parent_evidence_path), incoming_evidence, ("evidence_doc_id", "doc_id"))
        captions = _merge_unique_by_id(_load_list(parent_caption_path), incoming_captions, ("doc_id", "evidence_doc_id"))
        episodes.sort(key=lambda item: (str(item.get("date") or ""), _safe_float(item.get("local_start_time", item.get("start", 0.0))), str(item.get("episode_id") or "")))
        evidence.sort(key=lambda item: (str(item.get("date") or ""), _safe_float(item.get("local_start_time", item.get("start", 0.0))), str(item.get("evidence_doc_id") or item.get("doc_id") or "")))
        captions.sort(key=lambda item: (str(item.get("date") or ""), _safe_float(item.get("local_start_time", item.get("start", 0.0))), str(item.get("doc_id") or "")))
        write_json_atomic(parent_episode_path, episodes)
        write_json_atomic(parent_evidence_path, evidence)
        write_json_atomic(parent_caption_path, captions)

        build_config_path = build_online_em2mem_memory(
            session_id=parent_session_id,
            sessions_root=sessions_root,
            force=force_rebuild,
            skip_visual_embedding=skip_visual_embedding,
            skip_semantic=skip_semantic,
            source="mst_episodic",
        )
        try:
            from online_query.query_cache import GLOBAL_SESSION_ENGINE_CACHE

            GLOBAL_SESSION_ENGINE_CACHE.invalidate(parent_session_id)
        except Exception:
            pass
        payload = {
            "status": "done",
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "day_label": resolved_day_label,
            "day_index": resolved_day_index,
            "run_id": run_id,
            "time_context": time_context,
            "merged_at": utc_now_iso(),
            "parent_counts": {
                "episodes": len(episodes),
                "evidence": len(evidence),
                "captions": len(captions),
            },
            "incoming_counts": {
                "episodes": len(incoming_episodes),
                "evidence": len(incoming_evidence),
                "captions": len(incoming_captions),
            },
            "memory_config_path": relative_to_session(build_config_path, parent_dir),
        }
        _write_merge_state(parent_dir, child_session_id, payload)
        return payload
