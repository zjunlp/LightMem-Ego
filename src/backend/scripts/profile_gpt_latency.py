from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
HIPPO_SRC_ROOT = PROJECT_ROOT / "src" / "HippoRAG" / "src"
for path in (PROJECT_ROOT, SRC_ROOT, HIPPO_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from online_current.mcur_query import build_current_prompt
from online_current.mcur_selector import MCurFrameSelector
from online_current.mcur_store import MCurStore
from online_query.evidence_packer import EvidencePacker
from online_query.query_engine import _build_short_term_llm_prompt
from online_query.query_router import QueryRouter
from online_short_term.mst_retriever import MSTRetriever
from online_short_term.mst_store import MSTStore
from worldmm.llm import LLMModel


def _load_env_file(path: Path) -> None:
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


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _payload_size(chat_messages: list[dict[str, Any]], model_name: str) -> tuple[int, int]:
    body = json.dumps({"model": model_name, "messages": chat_messages}, ensure_ascii=False)
    image_url_bytes = 0
    for message in chat_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url_bytes += len(str(image_url.get("url", "")))
            elif isinstance(image_url, str):
                image_url_bytes += len(image_url)
    return len(body.encode("utf-8")), image_url_bytes


def _parse_usage(debug: dict[str, Any]) -> dict[str, int | None]:
    preview = str(debug.get("raw_response_preview") or "")

    def find_int(pattern: str) -> int | None:
        match = re.search(pattern, preview)
        return int(match.group(1)) if match else None

    return {
        "prompt_tokens": find_int(r"prompt_tokens=(\d+)"),
        "cached_tokens": find_int(r"cached_tokens=(\d+)"),
        "completion_tokens": find_int(r"completion_tokens=(\d+)"),
        "total_tokens": find_int(r"total_tokens=(\d+)"),
    }


def _build_current_case(session_dir: Path, question: str) -> tuple[str, str, list[str]]:
    current_context = MCurStore(session_dir).get_current_context()
    selection = MCurFrameSelector().select_frames_for_query(
        current_context,
        question,
        max_images=4,
        max_frames=5,
    )
    route_decision = {
        "query_type": "current_perception",
        "retrieval_mode": "current",
        "use_image_evidence": True,
        "max_image_evidence": 4,
        "evidence_frames_k": 5,
        "memory_route": {"use_current": True, "use_short_term": False, "use_long_term": False},
    }
    pack_result = EvidencePacker(session_dir=session_dir).pack(
        query=question,
        route_decision=route_decision,
        retrieval_result={
            "current_context": current_context,
            "current_selection": selection,
            "text_results": [],
            "visual_results": [],
            "fused_results": [],
            "evidence_frames": selection.get("evidence_frames", []),
            "short_term_results": [],
        },
        cache_context={},
    )
    full_prompt = build_current_prompt(question, current_context, selection)
    state = current_context.get("state") or {}
    frame_times = [
        str(frame.get("timestamp"))
        for frame in selection.get("evidence_frames", [])[:5]
        if frame.get("timestamp") is not None
    ]
    compact_prompt = (
        f"问题：{question}\n"
        f"当前窗口：{state.get('window_start_time')}-{state.get('window_end_time')} 秒。\n"
        f"候选帧时间：{', '.join(frame_times)}。\n"
        "请只根据附图和上述极简上下文，用中文一句话回答；不确定就说明不确定。"
    )
    image_paths = list(pack_result.get("selected_image_paths_for_mllm") or [])
    return full_prompt, compact_prompt, image_paths


def _build_short_term_case(session_dir: Path, question: str) -> tuple[str, str, list[str]]:
    store = MSTStore(session_dir)
    short_term_results = MSTRetriever(store).search(question, top_k=5, cache_context={})
    route_decision = QueryRouter().route(
        question,
        request_options={
            "retrieval_mode": "auto",
            "use_image_evidence": True,
            "max_image_evidence": 4,
            "top_k": 5,
            "visual_top_k": 8,
            "final_evidence_k": 4,
        },
        session_context={
            "visual_ready": False,
            "short_term_ready": store.is_ready(),
            "current_ready": True,
            "current_stale": False,
            "long_term_ready": False,
        },
        cache_context={},
    )
    route_decision["use_image_evidence"] = True
    route_decision["max_image_evidence"] = 4
    pack_result = EvidencePacker(session_dir=session_dir).pack(
        query=question,
        route_decision=route_decision,
        retrieval_result={
            "text_results": [],
            "visual_results": [],
            "fused_results": [],
            "evidence_frames": [],
            "short_term_results": short_term_results,
        },
        cache_context={},
    )
    full_prompt = _build_short_term_llm_prompt(
        question,
        short_term_results,
        pack_result.get("evidence_pack_summary", {}),
    )
    lines = [f"问题：{question}", "短期事件摘要："]
    for item in short_term_results[:4]:
        caption = (
            item.get("event_caption_refined")
            or item.get("event_caption_fast")
            or item.get("event_caption_placeholder")
            or item.get("retrieval_text")
            or ""
        )
        lines.append(f"- {item.get('start_time')}-{item.get('end_time')} 秒：{caption}")
    lines.append("请结合附图和这些摘要，用中文简要回答；如果证据只是 provisional，请说明不确定。")
    compact_prompt = "\n".join(lines)
    image_paths = list(pack_result.get("selected_image_paths_for_mllm") or [])
    return full_prompt, compact_prompt, image_paths


def _make_messages(prompt: str, image_paths: list[str], session_dir: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for rel_path in image_paths:
        content.append({"type": "image", "image": session_dir / rel_path})
    return [
        {"role": "system", "content": "Answer in Chinese. Be concise and use only the supplied context and images."},
        {"role": "user", "content": content},
    ]


def _run_one(
    *,
    model: Any,
    model_name: str,
    session_id: str,
    session_dir: Path,
    case_name: str,
    prompt: str,
    image_paths: list[str],
    repeat_index: int,
    dry_run: bool,
) -> dict[str, Any]:
    run_marker = uuid.uuid4().hex[:8]
    marked_prompt = f"[latency_probe={run_marker}]\n{prompt}"
    raw_messages = _make_messages(marked_prompt, image_paths, session_dir)

    preprocess_start = time.perf_counter()
    processed_prompt = model._preprocess_prompt(raw_messages)
    preprocess_ms = round((time.perf_counter() - preprocess_start) * 1000)
    chat_messages = model._convert_prompt_to_chat_messages(processed_prompt)
    payload_bytes, image_url_bytes = _payload_size(chat_messages, model_name)

    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "case": case_name,
        "repeat_index": repeat_index,
        "image_count": len(image_paths),
        "prompt_chars": len(marked_prompt),
        "preprocess_ms": preprocess_ms,
        "payload_bytes": payload_bytes,
        "image_url_bytes": image_url_bytes,
        "model": model_name,
        "dry_run": dry_run,
    }
    if dry_run:
        return record

    api_start = time.perf_counter()
    try:
        answer = model._generate_with_chat_fallback(processed_prompt)
        api_ms = round((time.perf_counter() - api_start) * 1000)
        debug = dict(getattr(model, "last_debug", {}) or {})
        record.update(
            {
                "ok": True,
                "api_ms": api_ms,
                "total_ms": preprocess_ms + api_ms,
                "answer_chars": len(str(answer or "")),
                "answer_preview": str(answer or "")[:160],
                "request_path": debug.get("request_path"),
                **_parse_usage(debug),
            }
        )
    except Exception as exc:
        api_ms = round((time.perf_counter() - api_start) * 1000)
        debug = dict(getattr(model, "last_debug", {}) or {})
        record.update(
            {
                "ok": False,
                "api_ms": api_ms,
                "total_ms": preprocess_ms + api_ms,
                "error": f"{type(exc).__name__}: {exc}",
                "request_path": debug.get("request_path"),
            }
        )
    return record


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPT latency for online M_cur/M_st query payloads.")
    parser.add_argument("--session-id", default="11e4b80de954")
    parser.add_argument("--sessions-root", default=str(PROJECT_ROOT / "online_sessions"))
    parser.add_argument("--current-question", default="现在画面里有什么？")
    parser.add_argument("--short-question", default="刚才发生了什么？")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Only build payloads; do not call GPT.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summarize", default=None, help="Summarize an existing JSONL profile file and exit.")
    args = parser.parse_args()

    if args.summarize:
        rows = []
        for line in Path(args.summarize).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_case.setdefault(str(row.get("case")), []).append(row)
        for case, items in sorted(by_case.items()):
            ok_items = [x for x in items if x.get("ok") is True or x.get("dry_run")]
            api_values = [int(x["api_ms"]) for x in ok_items if x.get("api_ms") is not None]
            total_values = [int(x["total_ms"]) for x in ok_items if x.get("total_ms") is not None]
            payload_values = [int(x["payload_bytes"]) for x in ok_items if x.get("payload_bytes") is not None]
            summary = {
                "case": case,
                "n": len(items),
                "ok": len(ok_items),
                "image_count": items[0].get("image_count") if items else None,
                "prompt_chars": items[0].get("prompt_chars") if items else None,
                "payload_bytes": int(statistics.median(payload_values)) if payload_values else None,
                "api_ms_min": min(api_values) if api_values else None,
                "api_ms_p50": int(statistics.median(api_values)) if api_values else None,
                "api_ms_max": max(api_values) if api_values else None,
                "total_ms_p50": int(statistics.median(total_values)) if total_values else None,
            }
            print(json.dumps(summary, ensure_ascii=False))
        return

    _load_env_file(PROJECT_ROOT / ".env")
    model_name = (
        os.getenv("WORLDMM_QUERY_RESPOND_MODEL")
        or os.getenv("WORLDMM_RESPOND_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.4"
    )
    session_dir = Path(args.sessions_root) / args.session_id
    output_path = Path(args.output or PROJECT_ROOT / "runtime" / f"gpt_latency_profile_{args.session_id}_{_utc_stamp()}.jsonl")

    current_full, current_compact, current_images = _build_current_case(session_dir, args.current_question)
    short_full, short_compact, short_images = _build_short_term_case(session_dir, args.short_question)
    cases: list[tuple[str, str, list[str]]] = [
        ("probe_text_only", "用中文只回答两个字：收到", []),
        ("current_compact_0img", current_compact, []),
        ("current_compact_1img", current_compact, current_images[:1]),
        ("current_full_0img", current_full, []),
        ("current_full_1img", current_full, current_images[:1]),
        ("current_full_2img", current_full, current_images[:2]),
        ("short_compact_1img", short_compact, short_images[:1]),
        ("short_full_0img", short_full, []),
        ("short_full_2img", short_full, short_images[:2]),
        ("short_full_4img", short_full, short_images[:4]),
    ]

    model = None
    if not args.dry_run:
        model = LLMModel(model_name=model_name, fps=1, max_retries=1).model
    else:
        model = LLMModel(model_name=model_name, fps=1, max_retries=1).model

    print(json.dumps({"output": str(output_path), "cases": len(cases), "repeats": args.repeats, "dry_run": args.dry_run}, ensure_ascii=False))
    for repeat_index in range(max(1, args.repeats)):
        for case_name, prompt, image_paths in cases:
            record = _run_one(
                model=model,
                model_name=model_name,
                session_id=args.session_id,
                session_dir=session_dir,
                case_name=case_name,
                prompt=prompt,
                image_paths=image_paths,
                repeat_index=repeat_index,
                dry_run=args.dry_run,
            )
            _write_jsonl(output_path, record)
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
