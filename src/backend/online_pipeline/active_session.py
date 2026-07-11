from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


ACTIVE_SESSION_PATH = Path("runtime") / "active_session.json"
ABORTED_ROOT_NAME = "online_tasks_aborted"
TASKS_ROOT_NAME = "online_tasks"


ACTIVE_TASK_DIRS = {
    "query",
    "query_in_progress",
    "mst_refine",
    "mst_refine_in_progress",
    "mst_consolidation",
    "mst_consolidation_in_progress",
    "memory",
    "memory_in_progress",
    "visual",
    "visual_in_progress",
    "preprocess",
    "in_progress",
    "evidence",
    "evidence_in_progress",
    "stream_chunk",
    "stream_chunk_in_progress",
    "stream_asr",
    "stream_asr_in_progress",
    "live_ingest",
    "live_ingest_in_progress",
}


def single_active_session_enabled() -> bool:
    value = os.getenv("WORLDMM_SINGLE_ACTIVE_SESSION")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def active_session_file(project_root: Path) -> Path:
    return project_root / ACTIVE_SESSION_PATH


def read_active_session_id(project_root: Path) -> str | None:
    payload = read_json(active_session_file(project_root), default={})
    if not isinstance(payload, dict):
        return None
    session_id = str(payload.get("active_session_id") or "").strip()
    return session_id or None


def _allow_inactive_session_task(task: dict[str, Any]) -> bool:
    value = task.get("allow_inactive_session")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def write_active_session(project_root: Path, session_id: str, reason: str = "stream_start") -> dict[str, Any]:
    payload = {
        "active_session_id": session_id,
        "updated_at": utc_now_iso(),
        "reason": reason,
    }
    write_json_atomic(active_session_file(project_root), payload)
    return payload



def clear_active_session(project_root: Path, session_id: str | None = None, reason: str = "stream_end") -> dict[str, Any]:
    previous_session_id = read_active_session_id(project_root)
    if session_id and previous_session_id and previous_session_id != session_id:
        return {
            "active_session_id": previous_session_id,
            "cleared": False,
            "cleared_session_id": None,
            "updated_at": utc_now_iso(),
            "reason": "active_session_mismatch",
        }
    payload = {
        "active_session_id": None,
        "cleared": True,
        "cleared_session_id": previous_session_id,
        "updated_at": utc_now_iso(),
        "reason": reason,
    }
    write_json_atomic(active_session_file(project_root), payload)
    return payload

def _aborted_dir(project_root: Path, batch_id: str, original_dir: str) -> Path:
    return project_root / ABORTED_ROOT_NAME / batch_id / original_dir


def _is_query_task(task: dict[str, Any], task_path: Path) -> bool:
    task_type = str(task.get("task_type") or "")
    parent_name = task_path.parent.name
    return task_type == "query" or parent_name in {"query", "query_in_progress"}


def _write_cancelled_query_status(
    project_root: Path,
    task_path: Path,
    task: dict[str, Any],
    reason: str,
) -> Path:
    target_dir = project_root / TASKS_ROOT_NAME / "query_failed"
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(task)
    payload["status"] = "cancelled"
    payload["error"] = "cancelled because a new active stream session started"
    payload["error_type"] = "cancelled"
    payload["cancelled_reason"] = reason
    payload["updated_at"] = utc_now_iso()
    target_path = target_dir / task_path.name
    write_json_atomic(target_path, payload)
    return target_path


def abort_task_file(
    project_root: Path,
    task_path: Path,
    *,
    task: dict[str, Any] | None = None,
    reason: str = "old_session",
    batch_id: str | None = None,
) -> dict[str, Any]:
    payload = dict(task) if isinstance(task, dict) else read_json(task_path, default={})
    if not isinstance(payload, dict):
        payload = {}
    sid = str(payload.get("session_id") or "")
    batch = batch_id or utc_now_iso().replace(":", "").replace(".", "_")
    original_dir = task_path.parent.name
    aborted_path = _aborted_dir(project_root, batch, original_dir) / task_path.name
    aborted_path.parent.mkdir(parents=True, exist_ok=True)

    payload["status"] = "aborted"
    payload["abort_reason"] = reason
    payload["aborted_at"] = utc_now_iso()
    cancelled_path = None
    if _is_query_task(payload, task_path):
        cancelled_path = _write_cancelled_query_status(project_root, task_path, payload, reason)

    moved = False
    if task_path.exists():
        write_json_atomic(task_path, payload)
        try:
            shutil.move(str(task_path), str(aborted_path))
            moved = True
        except FileNotFoundError:
            moved = False
    print(
        f"[task:skip_old_session] session_id={sid} task_path={task_path} "
        f"aborted_path={aborted_path if moved else '<missing>'} reason={reason}",
        flush=True,
    )
    return {
        "session_id": sid,
        "task_path": str(task_path),
        "aborted_path": str(aborted_path) if moved else None,
        "cancelled_query_path": str(cancelled_path) if cancelled_path else None,
        "moved": moved,
    }


def task_belongs_to_inactive_session(project_root: Path, task: dict[str, Any]) -> bool:
    if _allow_inactive_session_task(task):
        return False
    if not single_active_session_enabled():
        return False
    active_session_id = read_active_session_id(project_root)
    if not active_session_id:
        return False
    session_id = str(task.get("session_id") or "").strip()
    return bool(session_id and session_id != active_session_id)


def session_is_active_or_allowed(project_root: Path, session_id: str) -> bool:
    if not single_active_session_enabled():
        return True
    active_session_id = read_active_session_id(project_root)
    if not active_session_id:
        return True
    return str(session_id or "").strip() == active_session_id


def clear_old_session_tasks(project_root: Path, keep_session_id: str, reason: str = "new_active_session") -> dict[str, Any]:
    batch_id = utc_now_iso().replace(":", "").replace(".", "_")
    tasks_root = project_root / TASKS_ROOT_NAME
    counts = {
        "scanned": 0,
        "aborted": 0,
        "cancelled_query": 0,
        "by_dir": {},
        "batch_id": batch_id,
        "aborted_root": str(project_root / ABORTED_ROOT_NAME / batch_id),
    }
    if not tasks_root.exists():
        return counts
    for queue_dir in sorted(tasks_root.iterdir()):
        if not queue_dir.is_dir() or queue_dir.name not in ACTIVE_TASK_DIRS:
            continue
        for task_path in sorted(queue_dir.glob("*.json")):
            counts["scanned"] += 1
            payload = read_json(task_path, default={})
            if not isinstance(payload, dict):
                continue
            session_id = str(payload.get("session_id") or "").strip()
            if not session_id or session_id == keep_session_id:
                continue
            if _allow_inactive_session_task(payload):
                continue
            result = abort_task_file(project_root, task_path, task=payload, reason=reason, batch_id=batch_id)
            counts["aborted"] += 1
            counts["by_dir"][queue_dir.name] = counts["by_dir"].get(queue_dir.name, 0) + 1
            if result.get("cancelled_query_path"):
                counts["cancelled_query"] += 1
    return counts
