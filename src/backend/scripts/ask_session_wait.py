#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_server() -> str:
    _load_dotenv(ROOT_DIR / ".env")
    port = os.getenv("WORLDMM_API_PORT", "8000").strip() or "8000"
    return f"http://127.0.0.1:{port}"


def _read_json_response(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except Exception:
            detail = body
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    try:
        parsed = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"non-JSON response from {url}: {body[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"unexpected JSON response from {url}: {parsed!r}")
    return parsed


def _final_status(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "").strip().lower()
    queue_state = str(payload.get("queue_state") or "").strip().lower()
    if queue_state == "query_done":
        result = payload.get("result")
        if isinstance(result, dict):
            result_status = str(result.get("status") or "").strip().lower()
            return result_status or "ok"
        return "done"
    if queue_state == "query_failed":
        return "failed"
    return status


def _extract_answer(payload: dict[str, Any]) -> str:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if not isinstance(result, dict):
        return ""
    return str(result.get("answer") or result.get("answer_text") or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against a specific WorldMM session and wait for the answer.")
    parser.add_argument("positional_session_id", nargs="?", help="Session id to query.")
    parser.add_argument("positional_question", nargs="*", help="Question text. If omitted, use --question.")
    parser.add_argument("--session-id", dest="session_id", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--server", default=os.getenv("SERVER") or _default_server())
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("TIMEOUT_SECONDS", "600")))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("POLL_INTERVAL", "1.0")))
    parser.add_argument("--request-timeout", type=float, default=float(os.getenv("REQUEST_TIMEOUT", "30")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    parser.add_argument("--memory-mode", default=os.getenv("MEMORY_MODE", "auto"))
    parser.add_argument("--retrieval-mode", default=os.getenv("RETRIEVAL_MODE", "auto"))
    parser.add_argument("--use-current", choices=["true", "false", "auto"], default=os.getenv("USE_CURRENT", "auto"))
    parser.add_argument("--use-short-term", choices=["true", "false", "auto"], default=os.getenv("USE_SHORT_TERM", "auto"))
    parser.add_argument("--use-long-term", choices=["true", "false", "auto"], default=os.getenv("USE_LONG_TERM", "auto"))
    parser.add_argument("--client-source", default=os.getenv("CLIENT_SOURCE", "script"))
    parser.add_argument("--input-method", default=os.getenv("INPUT_METHOD", "cli"))
    parser.add_argument("--json", action="store_true", help="Print the final task payload as JSON instead of a compact answer summary.")
    return parser.parse_args()


def _optional_bool(value: str) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def main() -> int:
    args = parse_args()
    session_id = (args.session_id or args.positional_session_id or "").strip()
    question = (args.question or " ".join(args.positional_question)).strip()
    if not session_id:
        print("session_id is required", file=sys.stderr)
        return 2
    if not question:
        print("question is required", file=sys.stderr)
        return 2

    server = str(args.server).rstrip("/")
    request_payload: dict[str, Any] = {
        "question": question,
        "mode": "async",
        "memory_mode": args.memory_mode,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "client_source": args.client_source,
        "input_method": args.input_method,
    }
    for key, value in (
        ("use_current", _optional_bool(args.use_current)),
        ("use_short_term", _optional_bool(args.use_short_term)),
        ("use_long_term", _optional_bool(args.use_long_term)),
    ):
        if value is not None:
            request_payload[key] = value

    ask_url = f"{server}/session/{session_id}/ask"
    queued = _read_json_response(ask_url, method="POST", payload=request_payload, timeout=args.request_timeout)
    if str(queued.get("status") or "").lower() != "queued" or not queued.get("task_id"):
        print(json.dumps(queued, ensure_ascii=False, indent=2))
        return 1

    task_id = str(queued["task_id"])
    result_url = f"{server}{queued.get('result_url') or f'/query_task/{task_id}'}"
    print(f"queued task_id={task_id}", file=sys.stderr)

    deadline = time.monotonic() + max(0.0, args.timeout_seconds)
    last_payload: dict[str, Any] = queued
    while time.monotonic() <= deadline:
        last_payload = _read_json_response(result_url, timeout=args.request_timeout)
        status = _final_status(last_payload)
        if status in {"ok", "done"}:
            if args.json:
                print(json.dumps(last_payload, ensure_ascii=False, indent=2))
            else:
                answer = _extract_answer(last_payload)
                print(answer or json.dumps(last_payload, ensure_ascii=False, indent=2))
            return 0
        if status in {"failed", "error", "cancelled", "aborted"}:
            print(json.dumps(last_payload, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1
        time.sleep(max(0.1, args.poll_interval))

    print(f"timeout waiting for task_id={task_id}", file=sys.stderr)
    print(json.dumps(last_payload, ensure_ascii=False, indent=2), file=sys.stderr)
    return 124


if __name__ == "__main__":
    raise SystemExit(main())
