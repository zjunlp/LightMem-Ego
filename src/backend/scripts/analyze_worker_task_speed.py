from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAY_CHILD_SEPARATOR = "__day"


WORKER_DIRS = {
    "query": {
        "done": "query_done",
        "failed": "query_failed",
        "in_progress": "query_in_progress",
    },
    "refine": {
        "done": "mst_refine_done",
        "failed": "mst_refine_failed",
        "in_progress": "mst_refine_in_progress",
    },
    "rokid_day_merge": {
        "done": "rokid_day_merge_done",
        "failed": "rokid_day_merge_failed",
        "in_progress": "rokid_day_merge_in_progress",
    },
}

WORKER_ALIASES = {
    "mst_refine": "refine",
    "mst_refine_worker": "refine",
    "refine_worker": "refine",
    "query_worker": "query",
    "rokid_day_merge_worker": "rokid_day_merge",
}


@dataclass
class TaskMetric:
    worker: str
    status_bucket: str
    status: str
    task_id: str
    session_id: str
    created_at: datetime | None
    claimed_at: datetime | None
    finished_at: datetime | None
    queue_wait_s: float | None
    processing_s: float | None
    total_s: float | None
    result_latency_s: float | None
    item_count: float | None
    item_label: str
    reason: str
    precision: str
    path: str
    error: str | None

    def speed_s(self) -> float | None:
        return self.processing_s if self.processing_s is not None else self.total_s

    def items_per_minute(self) -> float | None:
        speed = self.speed_s()
        if speed is None or speed <= 0 or self.item_count is None:
            return None
        return self.item_count * 60.0 / speed


def parse_iso(value: Any) -> datetime | None:
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
    return parsed.astimezone(timezone.utc)


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 60:
        return f"{value:.2f}s"
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}m{seconds:04.1f}s"


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _result_latency_s(worker: str, task: dict[str, Any]) -> float | None:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    if worker == "query":
        latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
        value_ms = _first_number(
            latency.get("total_ms"),
            latency.get("query_ms"),
            result.get("total_ms"),
        )
        return value_ms / 1000.0 if value_ms is not None else None
    if worker == "rokid_day_merge":
        merged_at = parse_iso(result.get("merged_at"))
        claimed_at = parse_iso(task.get("claimed_at"))
        return seconds_between(claimed_at, merged_at)
    return None


def _item_count(worker: str, task: dict[str, Any]) -> tuple[float | None, str]:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    if worker == "query":
        return 1.0, "query"
    if worker == "refine":
        refined = _first_number(result.get("refined_event_count"))
        selected = _first_number(result.get("selected_event_count"))
        return refined if refined is not None else selected, "event"
    if worker == "rokid_day_merge":
        incoming = result.get("incoming_counts") if isinstance(result.get("incoming_counts"), dict) else {}
        count = _first_number(incoming.get("episodes"), incoming.get("evidence"), incoming.get("captions"))
        return count, "episode"
    return None, "item"


def task_matches_session(task: dict[str, Any], session_filter: set[str]) -> bool:
    if not session_filter:
        return True
    fields = (
        task.get("session_id"),
        task.get("parent_session_id"),
        task.get("child_session_id"),
    )
    return any(str(value or "") in session_filter for value in fields)


def task_matches_task_id(task: dict[str, Any], task_ids: set[str]) -> bool:
    if not task_ids:
        return True
    task_id = str(task.get("task_id") or "")
    return task_id in task_ids or any(needle in task_id for needle in task_ids)


def metric_from_task(
    *,
    worker: str,
    status_bucket: str,
    path: Path,
    task: dict[str, Any],
    now: datetime,
) -> TaskMetric:
    created_at = parse_iso(task.get("created_at"))
    claimed_at = parse_iso(task.get("claimed_at"))
    updated_at = parse_iso(task.get("updated_at"))
    completed_at = parse_iso(task.get("completed_at") or task.get("finished_at"))
    result_completed_at = None
    if isinstance(task.get("result"), dict):
        result_completed_at = parse_iso(
            task["result"].get("completed_at")
            or task["result"].get("finished_at")
            or task["result"].get("merged_at")
        )
    if status_bucket == "in_progress":
        finished_at = now
    else:
        finished_at = completed_at or result_completed_at or updated_at

    queue_wait_s = seconds_between(created_at, claimed_at)
    processing_s = seconds_between(claimed_at, finished_at)
    total_s = seconds_between(created_at, finished_at)
    result_latency_s = _result_latency_s(worker, task)
    item_count, item_label = _item_count(worker, task)

    if processing_s is not None:
        precision = "processing_exact"
    elif total_s is not None:
        precision = "total_only_no_claimed_at"
    else:
        precision = "missing_timestamps"

    return TaskMetric(
        worker=worker,
        status_bucket=status_bucket,
        status=str(task.get("status") or status_bucket),
        task_id=str(task.get("task_id") or path.stem),
        session_id=str(
            task.get("session_id")
            or task.get("child_session_id")
            or task.get("parent_session_id")
            or ""
        ),
        created_at=created_at,
        claimed_at=claimed_at,
        finished_at=finished_at,
        queue_wait_s=queue_wait_s,
        processing_s=processing_s,
        total_s=total_s,
        result_latency_s=result_latency_s,
        item_count=item_count,
        item_label=item_label,
        reason=str(task.get("reason") or task.get("query_priority_reason") or ""),
        precision=precision,
        path=str(path),
        error=str(task.get("error") or "")[:240] or None,
    )


def expand_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def _append_unique(items: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


def _child_session_ids_from_day_state(parent_dir: Path, parent_session_id: str) -> list[str]:
    state = read_json(parent_dir / "stream" / "day_state.json")
    if not isinstance(state, dict):
        return []
    parent_from_state = str(state.get("parent_session_id") or parent_session_id).strip()
    if parent_from_state != parent_session_id:
        return []
    children: list[str] = []
    runs = state.get("runs") if isinstance(state.get("runs"), dict) else {}
    for run in runs.values():
        if not isinstance(run, dict):
            continue
        if str(run.get("parent_session_id") or parent_session_id).strip() != parent_session_id:
            continue
        _append_unique(children, run.get("child_session_id"))
    return children


def _child_session_id_from_metadata(session_dir: Path, parent_session_id: str) -> str | None:
    payload = read_json(session_dir / "metadata.json")
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    merged = {**metadata, **payload}
    if str(merged.get("parent_session_id") or "").strip() != parent_session_id:
        return None
    child_id = str(merged.get("child_session_id") or payload.get("session_id") or session_dir.name).strip()
    return child_id or session_dir.name


def expand_session_filter(
    session_ids: list[str],
    *,
    sessions_root: Path,
    expand_child_sessions: bool,
) -> tuple[set[str], dict[str, list[str]], list[str]]:
    filter_ids: list[str] = []
    expansion_map: dict[str, list[str]] = {}
    warnings: list[str] = []
    for session_id in session_ids:
        _append_unique(filter_ids, session_id)
    if not session_ids or not expand_child_sessions:
        return set(filter_ids), expansion_map, warnings
    if not sessions_root.exists():
        warnings.append(f"sessions root not found, child session expansion skipped: {sessions_root}")
        return set(filter_ids), expansion_map, warnings

    for session_id in session_ids:
        children: list[str] = []
        parent_dir = sessions_root / session_id
        if parent_dir.exists() and parent_dir.is_dir():
            for child_id in _child_session_ids_from_day_state(parent_dir, session_id):
                _append_unique(children, child_id)

        prefix = f"{session_id}{DAY_CHILD_SEPARATOR}"
        try:
            session_dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
        except Exception as exc:
            warnings.append(f"unable to scan sessions root for child sessions: {sessions_root}: {exc}")
            session_dirs = []

        for session_dir in session_dirs:
            if session_dir.name.startswith(prefix):
                _append_unique(children, session_dir.name)
            child_id = _child_session_id_from_metadata(session_dir, session_id)
            if child_id:
                _append_unique(children, child_id)

        if children:
            expansion_map[session_id] = children
            for child_id in children:
                _append_unique(filter_ids, child_id)

    return set(filter_ids), expansion_map, warnings


def normalize_workers(values: list[str] | None) -> list[str]:
    raw = expand_csv(values)
    if not raw or "all" in raw:
        return list(WORKER_DIRS)
    workers: list[str] = []
    for value in raw:
        normalized = WORKER_ALIASES.get(value, value)
        if normalized not in WORKER_DIRS:
            raise SystemExit(f"Unknown worker: {value}. Choices: all, {', '.join(WORKER_DIRS)}")
        if normalized not in workers:
            workers.append(normalized)
    return workers


def load_metrics(args: argparse.Namespace) -> tuple[list[TaskMetric], list[str], dict[str, list[str]]]:
    tasks_root = Path(args.tasks_root).resolve()
    workers = normalize_workers(args.worker)
    requested_session_ids = expand_csv(args.session_id)
    sessions_root = Path(args.sessions_root).resolve()
    session_filter, expansion_map, expansion_warnings = expand_session_filter(
        requested_session_ids,
        sessions_root=sessions_root,
        expand_child_sessions=bool(args.expand_child_sessions),
    )
    task_ids = set(expand_csv(args.task_id))
    since = parse_iso(args.since)
    until = parse_iso(args.until)
    statuses = ["done"]
    if args.include_failed:
        statuses.append("failed")
    if args.include_in_progress:
        statuses.append("in_progress")
    if args.status:
        statuses = expand_csv(args.status)

    now = datetime.now(timezone.utc)
    metrics: list[TaskMetric] = []
    warnings: list[str] = list(expansion_warnings)

    for worker in workers:
        worker_metrics: list[TaskMetric] = []
        for status in statuses:
            dirname = WORKER_DIRS[worker].get(status)
            if dirname is None:
                continue
            directory = tasks_root / dirname
            if not directory.exists():
                warnings.append(f"missing directory: {directory}")
                continue
            for path in directory.glob("*.json"):
                task = read_json(path)
                if task is None:
                    warnings.append(f"skip unreadable json: {path}")
                    continue
                if not task_matches_session(task, session_filter):
                    continue
                if not task_matches_task_id(task, task_ids):
                    continue
                metric = metric_from_task(
                    worker=worker,
                    status_bucket=status,
                    path=path,
                    task=task,
                    now=now,
                )
                time_for_filter = metric.finished_at or metric.created_at
                if since and (time_for_filter is None or time_for_filter < since):
                    continue
                if until and (time_for_filter is None or time_for_filter > until):
                    continue
                worker_metrics.append(metric)

        worker_metrics.sort(key=lambda item: item.finished_at or item.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        if args.max_records_per_worker and args.max_records_per_worker > 0:
            worker_metrics = worker_metrics[: args.max_records_per_worker]
        metrics.extend(worker_metrics)

    return metrics, warnings, expansion_map


def summarize(metrics: list[TaskMetric]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for worker in WORKER_DIRS:
        group = [item for item in metrics if item.worker == worker]
        if not group:
            continue
        speed_values = [value for value in (item.speed_s() for item in group) if value is not None]
        total_values = [item.total_s for item in group if item.total_s is not None]
        queue_values = [item.queue_wait_s for item in group if item.queue_wait_s is not None]
        processing_values = [item.processing_s for item in group if item.processing_s is not None]
        inner_values = [item.result_latency_s for item in group if item.result_latency_s is not None]
        throughput_values = [item.items_per_minute() for item in group if item.items_per_minute() is not None]
        rows.append(
            {
                "worker": worker,
                "records": len(group),
                "done": sum(1 for item in group if item.status_bucket == "done"),
                "failed": sum(1 for item in group if item.status_bucket == "failed"),
                "in_progress": sum(1 for item in group if item.status_bucket == "in_progress"),
                "with_claimed_at": sum(1 for item in group if item.claimed_at is not None),
                "avg_speed_s": mean(speed_values),
                "p50_speed_s": percentile(speed_values, 0.50),
                "p95_speed_s": percentile(speed_values, 0.95),
                "avg_total_s": mean(total_values),
                "avg_queue_wait_s": mean(queue_values),
                "avg_processing_s": mean(processing_values),
                "avg_inner_latency_s": mean(inner_values),
                "avg_items_per_minute": mean(throughput_values),
            }
        )
    return rows


def metric_to_dict(item: TaskMetric) -> dict[str, Any]:
    return {
        "worker": item.worker,
        "status_bucket": item.status_bucket,
        "status": item.status,
        "task_id": item.task_id,
        "session_id": item.session_id,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "claimed_at": item.claimed_at.isoformat() if item.claimed_at else None,
        "finished_at": item.finished_at.isoformat() if item.finished_at else None,
        "queue_wait_s": item.queue_wait_s,
        "processing_s": item.processing_s,
        "total_s": item.total_s,
        "result_latency_s": item.result_latency_s,
        "item_count": item.item_count,
        "item_label": item.item_label,
        "items_per_minute": item.items_per_minute(),
        "reason": item.reason,
        "precision": item.precision,
        "path": item.path,
        "error": item.error,
    }


def print_table(rows: list[list[str]], headers: list[str]) -> None:
    all_rows = [headers, *rows]
    widths = [max(len(row[idx]) for row in all_rows) for idx in range(len(headers))]
    print("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def print_text_report(
    metrics: list[TaskMetric],
    warnings: list[str],
    slow_limit: int,
    expansion_map: dict[str, list[str]] | None = None,
) -> None:
    print("Worker task speed report")
    print("========================")
    if expansion_map:
        print()
        print("Expanded child sessions:")
        for parent_id, child_ids in expansion_map.items():
            print(f"- {parent_id}: {', '.join(child_ids)}")
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings[:10]:
            print(f"- {warning}")
        if len(warnings) > 10:
            print(f"- ... {len(warnings) - 10} more")

    summary = summarize(metrics)
    if not summary:
        print()
        print("No matching task records found.")
        return

    print()
    print("Summary")
    print_table(
        [
            [
                row["worker"],
                str(row["records"]),
                str(row["done"]),
                str(row["failed"]),
                str(row["in_progress"]),
                f'{row["with_claimed_at"]}/{row["records"]}',
                fmt_seconds(row["avg_speed_s"]),
                fmt_seconds(row["p50_speed_s"]),
                fmt_seconds(row["p95_speed_s"]),
                fmt_seconds(row["avg_queue_wait_s"]),
                fmt_seconds(row["avg_processing_s"]),
                fmt_seconds(row["avg_inner_latency_s"]),
                fmt_number(row["avg_items_per_minute"]),
            ]
            for row in summary
        ],
        [
            "worker",
            "records",
            "done",
            "failed",
            "running",
            "claimed",
            "avg_speed",
            "p50",
            "p95",
            "avg_wait",
            "avg_process",
            "inner_latency",
            "items/min",
        ],
    )

    print()
    print("Slowest records")
    slow = sorted(
        [item for item in metrics if item.speed_s() is not None],
        key=lambda item: item.speed_s() or 0.0,
        reverse=True,
    )[:slow_limit]
    print_table(
        [
            [
                item.worker,
                item.status,
                item.task_id,
                item.session_id,
                fmt_seconds(item.speed_s()),
                fmt_seconds(item.total_s),
                fmt_seconds(item.queue_wait_s),
                fmt_seconds(item.processing_s),
                item.precision,
                item.reason[:32] or "-",
            ]
            for item in slow
        ],
        ["worker", "status", "task_id", "session", "speed", "total", "wait", "process", "precision", "reason"],
    )

    missing_claimed = sum(1 for item in metrics if item.claimed_at is None)
    if missing_claimed:
        print()
        print(
            "Note: some records have no claimed_at timestamp, so their speed uses "
            "created_at -> updated_at total time instead of pure worker processing time."
        )


def print_csv_report(metrics: list[TaskMetric]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=list(metric_to_dict(metrics[0]).keys()) if metrics else ["worker"],
    )
    writer.writeheader()
    for item in metrics:
        writer.writerow(metric_to_dict(item))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze existing online_tasks records to estimate worker processing speed. "
            "This script is read-only and never enqueues or runs worker tasks."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Repository root. Default: this script's parent repo.")
    parser.add_argument("--tasks-root", default=None, help="Path to online_tasks. Default: <project-root>/online_tasks.")
    parser.add_argument("--sessions-root", default=None, help="Path to online_sessions. Default: <project-root>/online_sessions.")
    parser.add_argument("--worker", action="append", help="Worker to inspect: all, query, refine, rokid_day_merge. Can be comma-separated.")
    parser.add_argument(
        "--session-id",
        action="append",
        help=(
            "Filter by session_id, parent_session_id, or child_session_id. "
            "Parent Rokid sessions are expanded to matching child sessions by default. "
            "Can be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--expand-child-sessions",
        dest="expand_child_sessions",
        action="store_true",
        default=True,
        help="When --session-id is a parent Rokid session, include child sessions found under online_sessions. Default: on.",
    )
    parser.add_argument(
        "--no-expand-child-sessions",
        dest="expand_child_sessions",
        action="store_false",
        help="Only match the exact session ids passed through --session-id.",
    )
    parser.add_argument("--task-id", action="append", help="Filter by exact or partial task_id. Can be repeated or comma-separated.")
    parser.add_argument("--status", action="append", help="Override status buckets: done, failed, in_progress. Can be comma-separated.")
    parser.add_argument("--include-failed", action="store_true", help="Also include failed task records.")
    parser.add_argument("--include-in-progress", action="store_true", help="Also include currently in-progress task records.")
    parser.add_argument("--since", help="Only include records finished after this ISO timestamp.")
    parser.add_argument("--until", help="Only include records finished before this ISO timestamp.")
    parser.add_argument("--max-records-per-worker", type=int, default=10000, help="Newest records to keep per worker after filtering. Use 0 for no cap.")
    parser.add_argument("--slow-limit", type=int, default=8, help="Number of slowest records to show in text output.")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    args.tasks_root = str(Path(args.tasks_root).resolve()) if args.tasks_root else str(project_root / "online_tasks")
    args.sessions_root = str(Path(args.sessions_root).resolve()) if args.sessions_root else str(project_root / "online_sessions")

    metrics, warnings, expansion_map = load_metrics(args)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "summary": summarize(metrics),
                    "records": [metric_to_dict(item) for item in metrics],
                    "warnings": warnings,
                    "expanded_child_sessions": expansion_map,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.format == "csv":
        print_csv_report(metrics)
    else:
        print_text_report(metrics, warnings, slow_limit=max(0, int(args.slow_limit)), expansion_map=expansion_map)


if __name__ == "__main__":
    main()
