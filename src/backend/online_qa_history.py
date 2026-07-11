from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


QA_HISTORY_FILENAME = "qa_history.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def qa_history_path(session_dir: Path) -> Path:
    return Path(session_dir) / QA_HISTORY_FILENAME


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_source(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"frontend", "glasses", "unknown"} else "unknown"


def _clean_input_method(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"manual", "preset", "voice", "scripted_demo", "unknown"} else "unknown"


def _clean_status(value: Any, *, error: str = "") -> str:
    text = _clean_text(value).lower()
    if text in {"queued", "running", "done", "failed", "error", "ok"}:
        return "done" if text == "ok" else text
    return "failed" if error else "done"


def append_qa_history(
    session_dir: Path,
    *,
    session_id: str,
    question: str,
    answer: str = "",
    client_source: str = "unknown",
    input_method: str = "unknown",
    status: str = "done",
    error: str = "",
    task_id: str | None = None,
    response_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_error = _clean_text(error)
    record: dict[str, Any] = {
        "turn_id": f"turn_{utc_now_iso().replace(':', '').replace('.', '_')}_{uuid4().hex[:8]}",
        "created_at": utc_now_iso(),
        "session_id": _clean_text(session_id),
        "client_source": _clean_source(client_source),
        "input_method": _clean_input_method(input_method),
        "question": _clean_text(question),
        "answer": _clean_text(answer),
        "status": _clean_status(status, error=clean_error),
        "error": clean_error,
    }
    if task_id:
        record["task_id"] = _clean_text(task_id)
    if response_mode:
        record["response_mode"] = _clean_text(response_mode)
    if metadata:
        record["metadata"] = metadata

    path = qa_history_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return record


def load_qa_history(session_dir: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = qa_history_path(session_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows
