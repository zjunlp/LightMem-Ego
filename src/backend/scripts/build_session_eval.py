#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_DIRS = ("query_done", "query_failed", "query_in_progress")
MEMORY_DIRS = (
    "memory_done",
    "memory_failed",
    "memory_in_progress",
    "mst_refine_done",
    "mst_refine_failed",
    "mst_refine_in_progress",
    "mst_consolidation_done",
    "mst_consolidation_failed",
    "mst_consolidation_in_progress",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
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
                    rows.append(value)
    except Exception:
        return rows
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None, default=str)
        handle.write("\n")
    tmp.replace(path)


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


def ms_between(start: Any, end: Any) -> int | None:
    left = parse_iso(start)
    right = parse_iso(end)
    if left is None or right is None:
        return None
    return int(round(max(0.0, (right - left).total_seconds()) * 1000))


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


def stats(values: list[Any]) -> dict[str, Any]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "count": len(cleaned),
        "mean_ms": round(statistics.mean(cleaned), 3) if cleaned else None,
        "p50_ms": round(percentile(cleaned, 0.5), 3) if cleaned else None,
        "p90_ms": round(percentile(cleaned, 0.9), 3) if cleaned else None,
    }


def merged_metadata(session_dir: Path) -> dict[str, Any]:
    payload = read_json(session_dir / "metadata.json", default={})
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {**metadata, **payload}


def is_child_metadata(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("is_rokid_day_child") and metadata.get("parent_session_id"))


def discover_related_sessions(session_id: str, sessions_root: Path, include_related: bool) -> list[str]:
    if not include_related:
        return [session_id]
    session_dir = sessions_root / session_id
    metadata = merged_metadata(session_dir)
    related = {session_id}
    parent = str(metadata.get("parent_session_id") or "").strip() if is_child_metadata(metadata) else session_id
    if parent:
        related.add(parent)
    for candidate in sessions_root.iterdir() if sessions_root.exists() else []:
        if not candidate.is_dir():
            continue
        name = candidate.name
        meta = merged_metadata(candidate)
        if name.startswith(f"{parent}__day") or str(meta.get("parent_session_id") or "") == parent:
            related.add(name)
    return sorted(related)


def classify_client(task_or_record: dict[str, Any], session_meta: dict[str, Any] | None = None) -> str:
    session_meta = session_meta or {}
    text = " ".join(
        str(task_or_record.get(key) or session_meta.get(key) or "").lower()
        for key in ("client_source", "input_method", "source", "device_type", "transport")
    )
    if "rokid" in text or "glass" in text or "glasses" in text:
        return "glasses"
    if "frontend" in text or "web" in text or "phone" in text or "mobile" in text:
        return "phone"
    if "script" in text or "cli" in text:
        return "script"
    return "unknown"


def score_and_source(item: dict[str, Any]) -> tuple[Any, str]:
    for key in ("eval_score", "final_score", "retrieval_score", "score", "visual_score", "semantic_score"):
        if item.get(key) is not None:
            return item.get(key), key
    source = str(item.get("source_memory") or item.get("source_type") or "").lower()
    if "cur" in source or "current" in source:
        return None, "not_available_for_current_observation"
    return None, "not_available"


def evidence_type(item: dict[str, Any]) -> str:
    if item.get("eval_evidence_type"):
        return str(item["eval_evidence_type"])
    text = " ".join(str(item.get(key) or "").lower() for key in ("source_memory", "source_type", "source", "evidence_id"))
    if "semantic" in text or "m_sem" in text or "msem" in text:
        return "long_term_semantic"
    if "episodic" in text or "m_lt" in text or "mlt" in text or "long" in text:
        return "long_term_episodic"
    if "m_st" in text or "mst" in text or "short" in text or item.get("event_id"):
        return "short_term_event"
    if "m_cur" in text or "mcur" in text or "current" in text:
        return "current_observation"
    if item.get("transcript"):
        return "transcript_snippet"
    if item.get("image_path") or item.get("keyframe_paths") or item.get("path"):
        return "visual_frame"
    return "unknown"


def normalize_evidence(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        score, source = score_and_source(copied)
        copied.setdefault("rank", copied.get("eval_rank") or rank)
        copied.setdefault("eval_rank", rank)
        copied.setdefault("eval_score", score)
        copied.setdefault("eval_score_source", source)
        copied.setdefault("eval_evidence_type", evidence_type(copied))
        normalized.append(copied)
    return normalized


def retrieval_ms(latency: dict[str, Any], trace: dict[str, Any] | None = None) -> int | None:
    if isinstance(trace, dict):
        value = (trace.get("stage_durations_ms") or {}).get("retrieval_ms")
        if value is not None:
            return int(value)
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


def answer_generation_ms(latency: dict[str, Any], trace: dict[str, Any] | None = None) -> int | None:
    if isinstance(trace, dict):
        value = (trace.get("stage_durations_ms") or {}).get("answer_generation_ms")
        if value is not None:
            return int(value)
    for key in ("generation_ms", "worldmm_answer_ms"):
        if latency.get(key) is not None:
            return int(float(latency[key]))
    return None


def task_time_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_queue_wait_ms": ms_between(task.get("created_at"), task.get("claimed_at")),
        "task_processing_ms": ms_between(task.get("claimed_at"), task.get("updated_at")),
        "task_total_ms": ms_between(task.get("created_at"), task.get("updated_at")),
    }


def load_tasks(tasks_root: Path, dirs: tuple[str, ...], session_ids: set[str]) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for dirname in dirs:
        folder = tasks_root / dirname
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            payload = read_json(path, default={})
            if not isinstance(payload, dict):
                continue
            if str(payload.get("session_id") or "") in session_ids:
                items.append((path, payload))
    return items


def build_question_from_task(path: Path, task: dict[str, Any], session_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    latency = result.get("latency") if isinstance(result.get("latency"), dict) else {}
    trace = result.get("eval_trace") if isinstance(result.get("eval_trace"), dict) else None
    times = task_time_summary(task)
    end_to_end = latency.get("total_ms") if latency.get("total_ms") is not None else times.get("task_total_ms")
    client_type = classify_client({**task, **result}, session_meta.get(str(task.get("session_id") or ""), {}))
    missing: list[str] = []
    if not result:
        missing.append("missing_query_result")
    if not trace:
        missing.append("missing_eval_trace")
    if "selected_evidence" not in result:
        missing.append("missing_selected_evidence")
    return {
        "question_id": str(task.get("task_id") or path.stem),
        "task_id": str(task.get("task_id") or path.stem),
        "session_id": str(task.get("session_id") or ""),
        "client_source": task.get("client_source") or result.get("client_source") or "unknown",
        "client_type": client_type,
        "input_method": task.get("input_method") or result.get("input_method") or "unknown",
        "response_mode": task.get("response_mode") or result.get("response_mode"),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
        "claimed_at": task.get("claimed_at"),
        "finished_at": task.get("updated_at"),
        "question": task.get("question") or result.get("question"),
        "generated_answer": result.get("answer") or result.get("answer_text") or "",
        "retrieval_settings": {
            "top_k": task.get("top_k"),
            "retrieval_mode": task.get("retrieval_mode"),
            "memory_mode": task.get("memory_mode"),
            "use_current": task.get("use_current"),
            "use_short_term": task.get("use_short_term"),
            "use_long_term": task.get("use_long_term"),
            "long_term_retrieval_scheme": task.get("long_term_retrieval_scheme") or result.get("long_term_retrieval_scheme"),
        },
        "evidence": {
            "selected": normalize_evidence(result.get("selected_evidence")),
            "candidates_by_source": result.get("memory_results") or {},
            "evidence_frames": result.get("evidence_frames") or [],
        },
        "latency": {
            **times,
            "retrieval_ms": retrieval_ms(latency, trace),
            "answer_generation_ms": answer_generation_ms(latency, trace),
            "end_to_end_qa_ms": end_to_end,
            "raw": latency,
        },
        "eval_trace": trace,
        "raw_debug": {
            "query_type": result.get("query_type"),
            "memory_route": result.get("memory_route"),
            "retrieval_plan": result.get("retrieval_plan"),
            "fusion_summary": result.get("fusion_summary"),
            "llm_api": (trace or {}).get("llm_api") if isinstance(trace, dict) else None,
        },
        "missing_data": missing,
        "source_path": str(path),
    }


def build_question_from_history(record: dict[str, Any], session_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sid = str(record.get("session_id") or "")
    return {
        "question_id": str(record.get("turn_id") or ""),
        "task_id": record.get("task_id"),
        "turn_id": record.get("turn_id"),
        "session_id": sid,
        "client_source": record.get("client_source") or "unknown",
        "client_type": classify_client(record, session_meta.get(sid, {})),
        "input_method": record.get("input_method") or "unknown",
        "response_mode": record.get("response_mode"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "claimed_at": None,
        "finished_at": record.get("created_at"),
        "question": record.get("question"),
        "generated_answer": record.get("answer") or "",
        "retrieval_settings": record.get("metadata") or {},
        "evidence": {"selected": [], "candidates_by_source": {}, "evidence_frames": []},
        "latency": {
            "task_queue_wait_ms": None,
            "task_processing_ms": None,
            "task_total_ms": None,
            "retrieval_ms": None,
            "answer_generation_ms": None,
            "end_to_end_qa_ms": None,
            "raw": {},
        },
        "eval_trace": None,
        "raw_debug": {},
        "missing_data": ["missing_query_task", "missing_selected_evidence", "missing_eval_trace", "missing_latency"],
        "source_path": "qa_history.jsonl",
    }


def summarize_memory_task(path: Path, task: dict[str, Any], session_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    times = task_time_summary(task)
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    sid = str(task.get("session_id") or "")
    return {
        "task_id": task.get("task_id") or path.stem,
        "task_type": task.get("task_type"),
        "session_id": sid,
        "client_type": classify_client(task, session_meta.get(sid, {})),
        "status": task.get("status"),
        "created_at": task.get("created_at"),
        "claimed_at": task.get("claimed_at"),
        "finished_at": task.get("updated_at"),
        "queue_wait_ms": times["task_queue_wait_ms"],
        "processing_ms": times["task_processing_ms"],
        "total_ms": times["task_total_ms"],
        "source": task.get("source"),
        "reason": task.get("reason"),
        "result_summary": {
            "status": result.get("status"),
            "updated_count": result.get("updated_count") or result.get("episode_count") or result.get("event_count"),
            "memory_version": result.get("memory_version"),
        },
        "source_path": str(path),
    }


def latency_tables(questions: list[dict[str, Any]], memory_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"overall": {"questions": questions, "memory_tasks": memory_tasks}}
    for client in sorted({q.get("client_type") for q in questions} | {m.get("client_type") for m in memory_tasks}):
        if not client:
            continue
        groups[str(client)] = {
            "questions": [q for q in questions if q.get("client_type") == client],
            "memory_tasks": [m for m in memory_tasks if m.get("client_type") == client],
        }
    tables: dict[str, Any] = {}
    for name, group in groups.items():
        qs = group["questions"]
        mts = group["memory_tasks"]
        tables[name] = {
            "memory_update": stats([m.get("total_ms") for m in mts]),
            "retrieval": stats([(q.get("latency") or {}).get("retrieval_ms") for q in qs]),
            "answer_generation": stats([(q.get("latency") or {}).get("answer_generation_ms") for q in qs]),
            "end_to_end_qa": stats([(q.get("latency") or {}).get("end_to_end_qa_ms") for q in qs]),
        }
    return tables


def build_eval(args: argparse.Namespace) -> dict[str, Any]:
    sessions_root = Path(args.sessions_root)
    if not sessions_root.is_absolute():
        sessions_root = PROJECT_ROOT / sessions_root
    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = PROJECT_ROOT / tasks_root
    included = discover_related_sessions(args.session_id, sessions_root, not args.no_related_sessions)
    included_set = set(included)
    session_meta = {sid: merged_metadata(sessions_root / sid) for sid in included}

    query_tasks = load_tasks(tasks_root, QUERY_DIRS, included_set)
    questions = [
        build_question_from_task(path, task, session_meta)
        for path, task in query_tasks
        if str(task.get("task_type") or "query") == "query"
    ]
    task_ids = {q.get("task_id") for q in questions if q.get("task_id")}
    qa_paths: list[str] = []
    for sid in included:
        history_path = sessions_root / sid / "qa_history.jsonl"
        qa_paths.append(str(history_path))
        for record in read_jsonl(history_path):
            task_id = record.get("task_id")
            if task_id and task_id in task_ids:
                continue
            questions.append(build_question_from_history(record, session_meta))

    memory_tasks = [
        summarize_memory_task(path, task, session_meta)
        for path, task in load_tasks(tasks_root, MEMORY_DIRS, included_set)
    ]
    questions.sort(key=lambda item: str(item.get("created_at") or ""))
    memory_tasks.sort(key=lambda item: str(item.get("created_at") or ""))

    return {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "session_id": args.session_id,
        "included_session_ids": included,
        "source_paths": {
            "qa_history": qa_paths,
            "query_tasks": [str(path) for path, _ in query_tasks],
            "memory_tasks": [item["source_path"] for item in memory_tasks],
        },
        "summary": {
            "question_count": len(questions),
            "completed_question_count": sum(1 for q in questions if str(q.get("status") or "").lower() in {"done", "ok"}),
            "failed_question_count": sum(1 for q in questions if str(q.get("status") or "").lower() in {"failed", "error"}),
            "memory_task_count": len(memory_tasks),
            "client_counts": {client: sum(1 for q in questions if q.get("client_type") == client) for client in sorted({q.get("client_type") for q in questions})},
        },
        "questions": questions,
        "memory_tasks": memory_tasks,
        "latency_tables": latency_tables(questions, memory_tasks),
        "metrics_placeholders": {
            "retrieval_accuracy": {
                "R@1": None,
                "R@3": None,
                "R@5": None,
                "MRR": None,
                "not_computed_reason": "requires_gold_labels",
            },
            "qa_accuracy": {
                "gpt4o_acc": None,
                "human_acc": None,
                "not_computed_reason": "requires_gold_answers",
            },
        },
        "notes": [
            "gold evidence, gold answers, and scenario labels are intentionally not recorded",
            "selected evidence rank is derived from final evidence order when eval_rank is absent",
            "historical records without query task files cannot recover evidence or detailed latency",
            "historical records without eval_trace use latency components as a fallback",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build session-level eval.json for LightMem-Ego paper evaluation.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--tasks-root", default="online_tasks")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-related-sessions", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_eval(args)
    sessions_root = Path(args.sessions_root)
    if not sessions_root.is_absolute():
        sessions_root = PROJECT_ROOT / sessions_root
    output = Path(args.output) if args.output else sessions_root / args.session_id / "eval.json"
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    write_json_atomic(output, payload, pretty=bool(args.pretty))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
