from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from online_pipeline.rokid_day_merge import (
    merge_rokid_day_child,
    missing_child_outputs,
    record_rokid_day_merge_waiting,
)
from online_pipeline.runtime_state import WorkerTaskHeartbeat, write_worker_runtime
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json, utc_now_iso
from online_preprocess.task_queue import (
    claim_rokid_day_merge_task,
    finish_rokid_day_merge_task,
    list_queued_rokid_day_merge_tasks,
    requeue_rokid_day_merge_task,
)
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_iso(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _not_before_reached(task_path: Path) -> bool:
    task = read_json(task_path, default={})
    if not isinstance(task, dict):
        return True
    not_before = _parse_iso(task.get("not_before"))
    if not_before is None:
        return True
    return datetime.now(timezone.utc) >= not_before


def _next_not_before(delay_seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, delay_seconds))).isoformat()


def run_worker(
    *,
    sessions_root: Path,
    interval_seconds: float,
    retry_delay_seconds: float,
    once: bool,
    verbose: bool,
) -> None:
    last_task_id = None
    last_error = None

    def _queue_pending() -> int:
        return len(list_queued_rokid_day_merge_tasks(PROJECT_ROOT))

    write_worker_runtime(
        PROJECT_ROOT,
        "rokid_day_merge",
        status="ready",
        backend="filesystem",
        warmup_done=True,
        queue_pending=len(list_queued_rokid_day_merge_tasks(PROJECT_ROOT)),
    )
    while True:
        results = []
        for task_path in list_queued_rokid_day_merge_tasks(PROJECT_ROOT):
            if not _not_before_reached(task_path):
                continue
            claimed = claim_rokid_day_merge_task(PROJECT_ROOT, task_path)
            if claimed is None:
                continue
            claimed_path, task = claimed
            task_id = str(task.get("task_id") or claimed_path.stem)
            parent_session_id = str(task.get("parent_session_id") or "")
            child_session_id = str(task.get("child_session_id") or task.get("session_id") or "")
            day_label = str(task.get("day_label") or "")
            day_index = int(task.get("day_index") or 1)
            run_id = str(task.get("run_id") or "")
            last_task_id = task_id
            write_worker_runtime(
                PROJECT_ROOT,
                "rokid_day_merge",
                status="busy",
                backend="filesystem",
                warmup_done=True,
                queue_pending=len(list_queued_rokid_day_merge_tasks(PROJECT_ROOT)),
                last_task_id=task_id,
                extra={
                    "parent_session_id": parent_session_id,
                    "child_session_id": child_session_id,
                    "day_label": day_label,
                    "day_index": day_index,
                },
            )
            try:
                child_dir = sessions_root / child_session_id
                missing = missing_child_outputs(child_dir)
                if missing:
                    retry_count = int(task.get("retry_count", 0) or 0) + 1
                    result = record_rokid_day_merge_waiting(
                        sessions_root=sessions_root,
                        parent_session_id=parent_session_id,
                        child_session_id=child_session_id,
                        day_label=day_label,
                        day_index=day_index,
                        run_id=run_id,
                        missing=missing,
                        retry_count=retry_count,
                    )
                    requeue_rokid_day_merge_task(
                        PROJECT_ROOT,
                        claimed_path,
                        task,
                        retry_count=retry_count,
                        not_before=_next_not_before(retry_delay_seconds),
                        result=result,
                    )
                    results.append(result)
                    append_timeline_event(
                        sessions_root / child_session_id,
                        "rokid_day_merge_waiting",
                        metadata={"task_id": task_id, "missing_outputs": missing, "retry_count": retry_count},
                    )
                    if verbose:
                        print(f"[rokid_day_merge] waiting task={task_id} child={child_session_id} missing={missing}", flush=True)
                    continue

                with WorkerTaskHeartbeat(
                    PROJECT_ROOT,
                    "rokid_day_merge",
                    task=task,
                    claimed_path=claimed_path,
                    status="busy",
                    backend="filesystem",
                    warmup_done=True,
                    queue_pending=_queue_pending,
                    extra={
                        "parent_session_id": parent_session_id,
                        "child_session_id": child_session_id,
                        "day_label": day_label,
                        "day_index": day_index,
                    },
                    interval_env="EM2MEM_ROKID_DAY_MERGE_HEARTBEAT_SECONDS",
                ):
                    result = merge_rokid_day_child(
                        sessions_root=sessions_root,
                        parent_session_id=parent_session_id,
                        child_session_id=child_session_id,
                        day_label=day_label,
                        day_index=day_index,
                        run_id=run_id,
                        force_rebuild=True,
                        skip_visual_embedding=_env_bool("EM2MEM_ROKID_DAY_MERGE_SKIP_VISUAL_EMBEDDING", True),
                        skip_semantic=_env_bool("EM2MEM_ROKID_DAY_MERGE_SKIP_SEMANTIC", False),
                    )
                finish_rokid_day_merge_task(PROJECT_ROOT, claimed_path, task, status="done", result=result)
                append_timeline_event(
                    sessions_root / child_session_id,
                    "rokid_day_merge_done",
                    metadata={"task_id": task_id, "parent_session_id": parent_session_id, "day_label": day_label},
                )
                append_timeline_event(
                    sessions_root / parent_session_id,
                    "rokid_day_merge_done",
                    metadata={"task_id": task_id, "child_session_id": child_session_id, "day_label": day_label},
                )
                results.append(result)
                last_error = None
                if verbose:
                    print(f"[rokid_day_merge] done task={task_id} child={child_session_id} parent={parent_session_id}", flush=True)
            except Exception as exc:
                last_error = str(exc)
                finish_rokid_day_merge_task(PROJECT_ROOT, claimed_path, task, status="failed", error=str(exc))
                results.append({"status": "failed", "task_id": task_id, "error": str(exc)})
                if verbose:
                    print(f"[rokid_day_merge] failed task={task_id}: {exc}", flush=True)

        write_worker_runtime(
            PROJECT_ROOT,
            "rokid_day_merge",
            status="ready",
            backend="filesystem",
            warmup_done=True,
            queue_pending=len(list_queued_rokid_day_merge_tasks(PROJECT_ROOT)),
            last_task_id=last_task_id,
            last_error=last_error,
            extra={"last_results": results[-10:], "updated_at": utc_now_iso()},
        )
        if once:
            return
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Rokid DAY child sessions into their parent long-term memory.")
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("EM2MEM_ROKID_DAY_MERGE_INTERVAL_SECONDS", "15")))
    parser.add_argument("--retry-delay-seconds", type=float, default=float(os.getenv("EM2MEM_ROKID_DAY_MERGE_RETRY_DELAY_SECONDS", "60")))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_worker(
        sessions_root=Path(args.sessions_root),
        interval_seconds=args.interval_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        once=bool(args.once),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
