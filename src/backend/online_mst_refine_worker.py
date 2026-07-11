from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import refresh_session_pipeline_state, write_worker_runtime
from online_pipeline.active_session import session_is_active_or_allowed
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.task_queue import (
    claim_mst_refine_task,
    enqueue_mst_consolidation_task,
    enqueue_mst_refine_task,
    finish_mst_refine_task,
    list_queued_mst_refine_tasks,
)
from online_short_term.mst_store import MSTStore
from online_short_term.refine_status import write_refine_status
from online_streaming.transcript_backfill import load_transcript_dirty_windows, mark_transcript_dirty_window_queued
from refine_mst_micro_events import is_auto_refine_eligible, refine_session


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"
FRAME_STREAM_BATCH_REASON = "frame_stream_batch"
MST_REFINE_READY_REASON = "mst_refine_ready_batch"
MST_REFINE_SCAN_REASON = "mst_refine_scan"


def _worker_runtime_name() -> str:
    name = str(os.getenv("WORLDMM_WORKER_INSTANCE_NAME") or "refine").strip()
    return name or "refine"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _heartbeat_interval_seconds() -> float:
    configured = os.getenv("WORLDMM_MST_REFINE_HEARTBEAT_SECONDS")
    if configured not in {None, ""}:
        return max(1.0, _env_float("WORLDMM_MST_REFINE_HEARTBEAT_SECONDS", 15.0))
    stale_seconds = max(3.0, _env_float("WORLDMM_WORKER_STALE_SECONDS", 60.0))
    return max(1.0, min(15.0, stale_seconds / 3.0))


def _queue_pending(project_root: Path) -> int:
    return len(list_queued_mst_refine_tasks(project_root))


def _write_refine_runtime(
    project_root: Path,
    *,
    status: str,
    backend: str,
    model_name: str | None,
    max_concurrency: int,
    last_task_id: str | None = None,
    last_error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload_extra = {"max_concurrency": max_concurrency}
    if extra:
        payload_extra.update(extra)
    write_worker_runtime(
        project_root,
        _worker_runtime_name(),
        status=status,
        backend=backend,
        model_name=model_name,
        client_loaded=True,
        warmup_done=True,
        queue_pending=_queue_pending(project_root),
        last_task_id=last_task_id,
        last_error=last_error,
        extra=payload_extra,
    )


class _TaskHeartbeat:
    def __init__(
        self,
        *,
        project_root: Path,
        task: dict[str, Any],
        claimed_path: Path,
        backend: str,
        model_name: str | None,
        max_concurrency: int,
    ) -> None:
        self.project_root = project_root
        self.task = task
        self.claimed_path = claimed_path
        self.backend = backend
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.task_id = str(task.get("task_id") or claimed_path.stem)
        self.session_id = str(task.get("session_id") or "")
        self.task_reason = str(task.get("reason") or "")
        self.started_at = str(task.get("claimed_at") or task.get("updated_at") or "")
        self.started_monotonic = time.monotonic()
        self.interval_seconds = _heartbeat_interval_seconds()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"refine-heartbeat-{self.task_id}", daemon=True)

    def __enter__(self) -> "_TaskHeartbeat":
        self._write()
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _runtime_extra(self) -> dict[str, Any]:
        return {
            "current_task": {
                "task_id": self.task_id,
                "task_type": str(self.task.get("task_type") or "mst_refine"),
                "session_id": self.session_id,
                "task_path": str(self.claimed_path),
                "reason": self.task_reason,
                "started_at": self.started_at,
                "running_seconds": round(max(0.0, time.monotonic() - self.started_monotonic), 3),
                "limit_events": self.task.get("limit_events"),
                "event_id": self.task.get("event_id"),
                "force_refine": bool(self.task.get("force_refine", False)),
            },
            "current_task_id": self.task_id,
            "current_session_id": self.session_id,
            "current_task_reason": self.task_reason,
        }

    def _write(self) -> None:
        _write_refine_runtime(
            self.project_root,
            status="busy",
            backend=self.backend,
            model_name=self.model_name,
            max_concurrency=self.max_concurrency,
            last_task_id=self.task_id,
            last_error=None,
            extra=self._runtime_extra(),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write()
            except Exception as exc:
                print(f"[mst_refine_worker] heartbeat update failed task={self.task_id}: {exc}", flush=True)


def _session_has_pending_events(session_dir: Path) -> bool:
    store = MSTStore(session_dir)
    return any(is_auto_refine_eligible(event) for event in store.load_archive_events())


def _enqueue_refine_followup_if_needed(
    *,
    project_root: Path,
    session_dir: Path,
    backend: str,
    limit_events: int,
    event_id: str | None,
    force_refine: bool,
    task_reason: str | None,
) -> Path | None:
    if event_id is not None or force_refine or task_reason != FRAME_STREAM_BATCH_REASON:
        return None
    store = MSTStore(session_dir)
    if not any(is_auto_refine_eligible(event) for event in store.load_archive_events()):
        return None
    return enqueue_mst_refine_task(
        project_root=project_root,
        session_id=session_dir.name,
        backend=backend,
        limit_events=limit_events,
        event_id=None,
        force_refine=False,
        reason=FRAME_STREAM_BATCH_REASON,
    )


def _discover_sessions(sessions_root: Path, limit: int) -> list[str]:
    if not sessions_root.exists() or limit <= 0:
        return []
    candidates = []
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        archive = session_dir / "short_term" / "archive" / "micro_events_all.jsonl"
        if archive.exists() and _session_has_pending_events(session_dir):
            candidates.append((archive.stat().st_mtime, session_dir.name))
    candidates.sort(reverse=True)
    return [sid for _, sid in candidates[:limit]]


def _ready_window_count(session_dir: Path) -> int:
    store = MSTStore(session_dir)
    windows_path, _state_path = write_refine_status(store)
    import json

    try:
        windows = json.loads(windows_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(windows, list):
        return 0
    return sum(1 for item in windows if isinstance(item, dict) and item.get("ready_for_30s_episodic") and item.get("is_closed_window"))


def _run_refine_task(
    *,
    project_root: Path,
    sessions_root: Path,
    session_id: str,
    backend: str,
    limit_events: int,
    event_id: str | None,
    event_ids: list[str] | None,
    force_refine: bool,
    task_reason: str | None,
    task_id: str | None,
    task_queued_at: str | None,
    task_worker_started_at: str | None,
    verbose: bool,
) -> dict[str, Any]:
    result = refine_session(
        session_id=session_id,
        sessions_root=sessions_root,
        backend=backend,
        limit_events=limit_events,
        event_id=event_id,
        event_ids=event_ids,
        force_refine=force_refine,
        only_active=False,
        only_archive=False,
        task_id=task_id,
        task_queued_at=task_queued_at,
        task_worker_started_at=task_worker_started_at,
        task_reason=task_reason,
        verbose=verbose,
    )
    session_dir = sessions_root / session_id
    ready_count = _ready_window_count(session_dir)
    result["ready_30s_window_count"] = ready_count
    if ready_count > 0 and _env_bool("WORLDMM_AUTO_MST_CONSOLIDATION", True):
        task_path = enqueue_mst_consolidation_task(
            project_root=project_root,
            session_id=session_id,
            backend=os.getenv("WORLDMM_MST_EPISODIC_BACKEND", "openai"),
            update_worldmm=_env_bool("WORLDMM_MST_CONSOLIDATE_UPDATE_WORLDMM", True),
            force=False,
            limit_windows=None,
            reason=MST_REFINE_READY_REASON,
        )
        result["consolidation_task_path"] = str(task_path)
        append_timeline_event(
            session_dir,
            "consolidation_queued",
            metadata={"task_id": task_path.stem, "ready_30s_window_count": ready_count, "reason": MST_REFINE_READY_REASON},
        )
    dirty_tasks = []
    dirty_state = load_transcript_dirty_windows(session_dir)
    for window in dirty_state.get("windows", []) or []:
        if not isinstance(window, dict) or str(window.get("status") or "") not in {"dirty", "queued"}:
            continue
        try:
            start = float(window.get("start_time"))
            end = float(window.get("end_time"))
        except Exception:
            continue
        if _env_bool("WORLDMM_AUTO_MST_CONSOLIDATION", True):
            task_path = enqueue_mst_consolidation_task(
                project_root=project_root,
                session_id=session_id,
                backend=os.getenv("WORLDMM_MST_EPISODIC_BACKEND", "openai"),
                update_worldmm=_env_bool("WORLDMM_MST_CONSOLIDATE_UPDATE_WORLDMM", True),
                force=True,
                limit_windows=1,
                window_start=start,
                window_end=end,
                reason="transcript_backfill",
            )
            mark_transcript_dirty_window_queued(session_dir, start, end, task_path.stem)
            append_timeline_event(
                session_dir,
                "consolidation_queued",
                metadata={"task_id": task_path.stem, "reason": "transcript_backfill", "window_start": start, "window_end": end},
            )
            dirty_tasks.append(str(task_path))
    if dirty_tasks:
        result["dirty_consolidation_task_paths"] = dirty_tasks
    followup_path = _enqueue_refine_followup_if_needed(
        project_root=project_root,
        session_dir=session_dir,
        backend=backend,
        limit_events=limit_events,
        event_id=event_id,
        force_refine=force_refine,
        task_reason=task_reason,
    )
    if followup_path is not None:
        result["followup_refine_task_path"] = str(followup_path)
        append_timeline_event(
            session_dir,
            "refine_followup_queued",
            metadata={"task_id": followup_path.stem, "reason": FRAME_STREAM_BATCH_REASON},
        )
    refresh_session_pipeline_state(session_dir)
    return result


def run_worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    sessions_root = Path(args.sessions_root).resolve()
    last_task_id: str | None = None
    last_error: str | None = None

    _write_refine_runtime(
        project_root,
        status="ready",
        backend=args.backend,
        model_name=args.model,
        max_concurrency=args.max_concurrency,
    )
    print(
        f"[mst_refine_worker] ready instance={_worker_runtime_name()} backend={args.backend} model={args.model} "
        f"max_concurrency={args.max_concurrency}",
        flush=True,
    )

    while True:
        queued = list_queued_mst_refine_tasks(project_root)
        if not queued and args.scan_sessions:
            for session_id in _discover_sessions(sessions_root, args.scan_limit_sessions):
                if not session_is_active_or_allowed(project_root, session_id):
                    print(f"[mst_refine_worker] skip inactive scan session={session_id}", flush=True)
                    continue
                try:
                    task_path = enqueue_mst_refine_task(
                        project_root=project_root,
                        session_id=session_id,
                        backend=args.backend,
                        limit_events=args.limit_events,
                        event_id=None,
                        force_refine=False,
                        reason=MST_REFINE_SCAN_REASON,
                    )
                    last_task_id = task_path.stem
                    last_error = None
                    print(
                        f"[mst_refine_worker] scan session={session_id} queued task={task_path.stem}",
                        flush=True,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    print(f"[mst_refine_worker] scan session={session_id} failed: {exc}", flush=True)

        queued = list_queued_mst_refine_tasks(project_root)
        if not queued:
            _write_refine_runtime(
                project_root,
                status="ready",
                backend=args.backend,
                model_name=args.model,
                max_concurrency=args.max_concurrency,
                last_task_id=last_task_id,
                last_error=last_error,
            )
            if args.once:
                return
            time.sleep(args.poll_interval)
            continue

        for task_path in queued:
            claimed = claim_mst_refine_task(project_root, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            session_id = str(task.get("session_id") or "")
            task_id = str(task.get("task_id") or claimed_path.stem)
            last_task_id = task_id
            try:
                _write_refine_runtime(
                    project_root,
                    status="busy",
                    backend=args.backend,
                    model_name=args.model,
                    max_concurrency=args.max_concurrency,
                    last_task_id=task_id,
                    last_error=None,
                )
                task_backend = str(task.get("backend") or args.backend)
                with _TaskHeartbeat(
                    project_root=project_root,
                    task=task,
                    claimed_path=claimed_path,
                    backend=task_backend,
                    model_name=args.model,
                    max_concurrency=args.max_concurrency,
                ):
                    result = _run_refine_task(
                        project_root=project_root,
                        sessions_root=sessions_root,
                        session_id=session_id,
                        backend=task_backend,
                        limit_events=int(task.get("limit_events") or args.limit_events),
                        event_id=task.get("event_id"),
                        event_ids=task.get("event_ids") if isinstance(task.get("event_ids"), list) else None,
                        force_refine=bool(task.get("force_refine", args.force_refine)),
                        task_reason=task.get("reason"),
                        task_id=task_id,
                        task_queued_at=task.get("created_at"),
                        task_worker_started_at=task.get("claimed_at") or task.get("updated_at"),
                        verbose=args.verbose,
                    )
                finish_mst_refine_task(project_root, claimed_path, task, status="done", result=result)
                append_timeline_event(
                    sessions_root / session_id,
                    "refine_done",
                    metadata={
                        "task_id": task_id,
                        "event_id": task.get("event_id"),
                        "refined_event_count": result.get("refined_event_count"),
                        "ready_30s_window_count": result.get("ready_30s_window_count"),
                    },
                )
                last_error = None
                print(f"[mst_refine_worker] task={task_id} session={session_id} done", flush=True)
            except Exception as exc:
                last_error = str(exc)
                finish_mst_refine_task(project_root, claimed_path, task, status="failed", error=str(exc))
                print(f"[mst_refine_worker] task={task_id} session={session_id} failed: {exc}", flush=True)

        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent worker for asynchronous M_st micro-event refinement.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default=os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai"), choices=["openai", "mock"])
    parser.add_argument("--model", default=os.getenv("WORLDMM_MST_REFINE_MODEL") or os.getenv("WORLDMM_VLM_MODEL") or os.getenv("OPENAI_MODEL"))
    parser.add_argument("--limit-events", type=int, default=int(os.getenv("WORLDMM_MST_REFINE_LIMIT_EVENTS", "20")))
    parser.add_argument("--force-refine", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("WORLDMM_REFINE_MAX_CONCURRENCY", "4")))
    parser.add_argument("--scan-sessions", action=argparse.BooleanOptionalAction, default=_env_bool("WORLDMM_MST_REFINE_SCAN_SESSIONS", True))
    parser.add_argument("--scan-limit-sessions", type=int, default=int(os.getenv("WORLDMM_MST_REFINE_SCAN_LIMIT_SESSIONS", "5")))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("WORLDMM_MST_REFINE_POLL_SECONDS", "10")))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
