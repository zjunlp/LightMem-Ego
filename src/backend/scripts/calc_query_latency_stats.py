#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_iso(value: Any) -> datetime:
    if not value:
        return datetime.max
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def percentile(values: list[float], q: float) -> float | None:
    cleaned = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    pos = (len(cleaned) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return cleaned[lower]
    weight = pos - lower
    return cleaned[lower] * (1 - weight) + cleaned[upper] * weight


def stats(values: list[int | float | None]) -> dict[str, Any]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "count": len(cleaned),
        "mean_ms": round(statistics.mean(cleaned), 3) if cleaned else None,
        "p50_ms": round(percentile(cleaned, 0.5), 3) if cleaned else None,
        "p90_ms": round(percentile(cleaned, 0.9), 3) if cleaned else None,
    }


def stage_retrieval_ms(latency: dict[str, Any], trace: dict[str, Any] | None) -> int | None:
    if isinstance(trace, dict):
        value = (trace.get("stage_durations_ms") or {}).get("retrieval_ms")
        if value is not None:
            return int(float(value))
    keys = (
        "text_retrieval_ms",
        "visual_retrieval_ms",
        "short_term_retrieval_ms",
        "fusion_ms",
        "memory_fusion_ms",
        "evidence_pack_ms",
    )
    values = [latency.get(key) for key in keys if latency.get(key) is not None]
    return int(sum(float(v) for v in values)) if values else None


def long_term_retrieval_ms(latency: dict[str, Any]) -> int | None:
    values = [
        latency.get(key)
        for key in ("long_term_rag_ms", "long_term_selector_ms", "long_term_pack_ms")
        if latency.get(key) is not None
    ]
    if values:
        return int(sum(float(v) for v in values))
    for key in ("long_term_retrieval_ms", "text_retrieval_ms"):
        if latency.get(key) is not None:
            return int(float(latency[key]))
    return None


def answer_generation_ms(latency: dict[str, Any], trace: dict[str, Any] | None) -> int | None:
    if isinstance(trace, dict):
        value = (trace.get("stage_durations_ms") or {}).get("answer_generation_ms")
        if value is not None:
            return int(float(value))
    for key in ("answer_generation_ms", "final_generation_ms", "generation_ms", "worldmm_answer_ms"):
        if latency.get(key) is not None:
            return int(float(latency[key]))
    return None


def end_to_end_qa_ms(latency: dict[str, Any], trace: dict[str, Any] | None) -> int | None:
    if isinstance(trace, dict):
        value = (trace.get("stage_durations_ms") or {}).get("end_to_end_qa_ms")
        if value is not None:
            return int(float(value))
    if latency.get("total_ms") is not None:
        return int(float(latency["total_ms"]))
    return None


def used_memory_sources(task: dict[str, Any], result: dict[str, Any]) -> list[str]:
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    used = result.get("used_memory_sources") or raw.get("used_memory_sources") or []
    return [str(item) for item in used] if isinstance(used, list) else []


def load_rows(
    tasks_root: Path,
    session_ids: list[str],
    retrieval_metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    done_dir = tasks_root / "query_done"
    for session_id in session_ids:
        for path in sorted(done_dir.glob(f"{session_id}*.json")):
            try:
                task = json.loads(path.read_text())
            except Exception as exc:
                skipped.append({"session_id": session_id, "file": path.name, "reason": f"read_error:{exc}"})
                continue
            result = task.get("result") if isinstance(task.get("result"), dict) else {}
            latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
            question = str(task.get("question") or result.get("question") or "").strip()
            if not (
                task.get("status") == "done"
                and result.get("status") == "ok"
                and latency.get("total_ms") is not None
                and question
            ):
                skipped.append(
                    {
                        "session_id": session_id,
                        "file": path.name,
                        "task_status": task.get("status"),
                        "result_status": result.get("status"),
                        "question": question,
                    }
                )
                continue
            trace = result.get("eval_trace") if isinstance(result.get("eval_trace"), dict) else None
            retrieval_value = (
                long_term_retrieval_ms(latency)
                if retrieval_metric == "long-term"
                else stage_retrieval_ms(latency, trace)
            )
            used_sources = used_memory_sources(task, result)
            rows.append(
                {
                    "session_id": session_id,
                    "task_id": task.get("task_id") or path.stem,
                    "created_at": task.get("created_at") or "",
                    "question": question,
                    "retrieval_ms": retrieval_value,
                    "answer_generation_ms": answer_generation_ms(latency, trace),
                    "end_to_end_qa_ms": end_to_end_qa_ms(latency, trace),
                    "latency_cache_hit": latency.get("cache_hit"),
                    "has_m_cache": "M_cache" in used_sources,
                    "used_memory_sources": ",".join(used_sources),
                    "long_term_selector_ms": latency.get("long_term_selector_ms"),
                    "source_path": str(path),
                }
            )
    return rows, skipped


def filter_cache_rows(rows: list[dict[str, Any]], cache_filter: str) -> list[dict[str, Any]]:
    if cache_filter == "all":
        return rows
    if cache_filter == "latency-cache-hit-false":
        return [row for row in rows if row.get("latency_cache_hit") is False]
    if cache_filter == "no-m-cache":
        return [row for row in rows if not row.get("has_m_cache")]
    if cache_filter == "both":
        return [
            row
            for row in rows
            if row.get("latency_cache_hit") is False and not row.get("has_m_cache")
        ]
    raise ValueError(f"Unsupported cache filter: {cache_filter}")


def filter_selector_rows(rows: list[dict[str, Any]], min_selector_ms: float | None) -> list[dict[str, Any]]:
    if min_selector_ms is None:
        return rows
    return [
        row
        for row in rows
        if row.get("long_term_selector_ms") is not None
        and float(row["long_term_selector_ms"]) >= min_selector_ms
    ]


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: (r["session_id"], r["question"], parse_iso(r["created_at"]), r["task_id"])):
        key = (row["session_id"], row["question"])
        kept_by_key.setdefault(key, row)
    kept = list(kept_by_key.values())
    dropped = [row for row in rows if kept_by_key[(row["session_id"], row["question"])]["task_id"] != row["task_id"]]
    return kept, dropped


def print_table(rows: list[dict[str, Any]], session_ids: list[str]) -> None:
    print("per_session_count")
    print("session_id\tcount")
    for session_id in session_ids:
        print(f"{session_id}\t{sum(1 for row in rows if row['session_id'] == session_id)}")
    print()
    print("overall_stats_ms")
    print("metric\tcount\tmean_ms\tp50_ms\tp90_ms")
    for metric in ("retrieval_ms", "answer_generation_ms", "end_to_end_qa_ms"):
        item = stats([row[metric] for row in rows])
        print(f"{metric}\t{item['count']}\t{item['mean_ms']}\t{item['p50_ms']}\t{item['p90_ms']}")
    print()
    print("kept_rows")
    print("session_id\ttask_id\tcreated_at\tcache_hit\thas_m_cache\tused_memory_sources\tselector_ms\tretrieval_ms\tanswer_generation_ms\tend_to_end_qa_ms\tquestion")
    for row in sorted(rows, key=lambda r: (r["session_id"], parse_iso(r["created_at"]), r["task_id"])):
        print(
            f"{row['session_id']}\t{row['task_id']}\t{row['created_at']}\t"
            f"{row['latency_cache_hit']}\t{row['has_m_cache']}\t{row['used_memory_sources']}\t"
            f"{row['long_term_selector_ms']}\t"
            f"{row['retrieval_ms']}\t{row['answer_generation_ms']}\t{row['end_to_end_qa_ms']}\t{row['question']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute query latency stats from query_done task JSON files.")
    parser.add_argument("session_ids", nargs="+")
    parser.add_argument("--tasks-root", default=str(PROJECT_ROOT / "online_tasks"))
    parser.add_argument("--no-dedupe", action="store_true", help="Do not dedupe repeated questions within the same session.")
    parser.add_argument("--show-dropped", action="store_true")
    parser.add_argument(
        "--cache-filter",
        choices=("all", "latency-cache-hit-false", "no-m-cache", "both"),
        default="all",
        help="Filter rows by cache behavior before optional dedupe.",
    )
    parser.add_argument(
        "--retrieval-metric",
        choices=("stage", "long-term"),
        default="stage",
        help="Use eval_trace retrieval_ms/stage sum, or long-term RAG+selector+packet timing.",
    )
    parser.add_argument(
        "--min-selector-ms",
        type=float,
        default=None,
        help="Keep only rows whose long_term_selector_ms is at least this value.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = PROJECT_ROOT / tasks_root

    rows, skipped = load_rows(tasks_root, args.session_ids, args.retrieval_metric)
    rows = filter_cache_rows(rows, args.cache_filter)
    rows = filter_selector_rows(rows, args.min_selector_ms)
    for row in rows:
        row["_before_dedupe"] = True
    kept, dropped = (rows, []) if args.no_dedupe else dedupe_rows(rows)

    print(f"valid_before_dedupe\t{len(rows)}")
    print(f"valid_after_dedupe\t{len(kept)}")
    print(f"duplicates_dropped\t{len(dropped)}")
    print(f"skipped_invalid_or_missing_latency\t{len(skipped)}")
    print()
    print_table(kept, args.session_ids)
    if args.show_dropped:
        print()
        print("dropped_duplicates")
        print("session_id\ttask_id\tcreated_at\tquestion")
        for row in sorted(dropped, key=lambda r: (r["session_id"], r["question"], parse_iso(r["created_at"]), r["task_id"])):
            print(f"{row['session_id']}\t{row['task_id']}\t{row['created_at']}\t{row['question']}")


if __name__ == "__main__":
    main()
