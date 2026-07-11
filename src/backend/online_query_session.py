from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from online_preprocess.io_utils import read_json, write_json
from online_preprocess.task_queue import enqueue_query_task, get_queue_dirs
from online_query import query_session
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


PROJECT_ROOT = Path(__file__).resolve().parent


def _wait_query_task(task_id: str, timeout_seconds: float, poll_interval: float) -> dict:
    dirs = get_queue_dirs(PROJECT_ROOT)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        done_path = dirs["query_done"] / f"{task_id}.json"
        failed_path = dirs["query_failed"] / f"{task_id}.json"
        if done_path.exists():
            payload = read_json(done_path, default={})
            return payload.get("result", payload) if isinstance(payload, dict) else {"status": "error", "message": "invalid done payload"}
        if failed_path.exists():
            payload = read_json(failed_path, default={})
            result = payload.get("result") if isinstance(payload, dict) else None
            if isinstance(result, dict):
                return result
            return {
                "status": "error",
                "message": payload.get("error", "query task failed") if isinstance(payload, dict) else "query task failed",
                "task_id": task_id,
            }
        time.sleep(poll_interval)
    return {"status": "error", "message": f"query task timeout after {timeout_seconds}s", "task_id": task_id}


def _query_via_worker(args: argparse.Namespace, question: str, image_option: bool | str) -> dict:
    task_path = enqueue_query_task(
        project_root=PROJECT_ROOT,
        session_id=args.session_id,
        question=question,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        use_image_evidence=image_option,
        max_image_frames=args.max_image_frames,
        max_image_evidence=args.max_image_evidence,
        text_top_k=args.text_top_k,
        visual_top_k=args.visual_top_k,
        final_evidence_k=args.final_evidence_k,
        memory_mode=args.memory_mode,
        use_interaction_cache=args.use_interaction_cache,
        cache_mode=args.cache_mode,
        use_current=args.use_current,
        use_short_term=args.use_short_term,
        use_long_term=args.use_long_term,
        debug_router=args.debug_router,
        long_term_retrieval_scheme=args.long_term_retrieval_scheme,
        retrieval_scheme=args.long_term_retrieval_scheme,
    )
    return _wait_query_task(task_path.stem, timeout_seconds=args.worker_timeout, poll_interval=args.worker_poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against one online WorldMM session memory.",
        epilog=(
            "Image evidence flags are switches: use --use-image-evidence to enable, "
            "--no-image-evidence to disable, and omit both to let the router decide. "
            "Do not pass true/false after these flags."
        ),
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--question", required=True, action="append", help="Question to ask. Repeat this flag to test cache reuse in one CLI process.")
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", default="auto", choices=["auto", "current", "text_only", "visual_only", "hybrid"])
    parser.add_argument(
        "--long-term-retrieval-scheme",
        "--retrieval-scheme",
        dest="long_term_retrieval_scheme",
        default=None,
        help="Long-term retrieval scheme: em2memory (default) or worldmm_legacy.",
    )
    parser.add_argument("--retriever-model", default=None)
    parser.add_argument("--respond-model", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--no-cache", action="store_true", help="Force cold-load the query engine and do not keep it cached.")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--use-image-evidence",
        action="store_true",
        help="Enable image evidence. This is a flag; do not pass true/false after it.",
    )
    image_group.add_argument(
        "--no-image-evidence",
        action="store_true",
        help="Disable image evidence. This is a flag; do not pass true/false after it. By default the router decides.",
    )
    parser.add_argument("--max-image-frames", type=int, default=4, help="Maximum selected keyframes to send when image evidence is enabled.")
    parser.add_argument("--max-image-evidence", type=int, default=3, help="Maximum visual-retrieval keyframes to send when image evidence is enabled.")
    parser.add_argument("--text-top-k", type=int, default=None)
    parser.add_argument("--visual-top-k", type=int, default=None)
    parser.add_argument("--final-evidence-k", type=int, default=None)
    parser.add_argument("--memory-mode", default="auto", choices=["auto", "legacy"])
    parser.add_argument("--use-interaction-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-current", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-short-term", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-long-term", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache-mode", default="auto", choices=["auto", "off", "read_only", "write_only"])
    parser.add_argument("--debug-router", action="store_true")
    parser.add_argument("--via-worker", action="store_true", help="Submit query tasks to the persistent online_query_worker instead of running in this CLI process.")
    parser.add_argument("--worker-timeout", type=float, default=600.0)
    parser.add_argument("--worker-poll-interval", type=float, default=0.5)
    parser.add_argument("--print-latency", action="store_true", help="Keep latency fields in the printed JSON output.")
    parser.add_argument("--json", action="store_true", help="Print stable JSON output. This is the default for compatibility.")
    args = parser.parse_args()
    del args.retriever_model, args.respond_model
    args.long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(args.long_term_retrieval_scheme)

    outputs = []
    for question in args.question:
        image_option: bool | str = "auto"
        if args.use_image_evidence:
            image_option = True
        if args.no_image_evidence:
            image_option = False
        if args.via_worker:
            result = _query_via_worker(args, question, image_option)
        else:
            result = query_session(
                session_id=args.session_id,
                question=question,
                sessions_root=Path(args.sessions_root),
                top_k=args.top_k,
                no_cache=args.no_cache,
                retrieval_mode=args.retrieval_mode,
                use_image_evidence=image_option,
                max_image_frames=args.max_image_frames,
                max_image_evidence=args.max_image_evidence,
                text_top_k=args.text_top_k,
                visual_top_k=args.visual_top_k,
                final_evidence_k=args.final_evidence_k,
                memory_mode=args.memory_mode,
                use_interaction_cache=args.use_interaction_cache,
                cache_mode=args.cache_mode,
                use_current=args.use_current,
                use_short_term=args.use_short_term,
                use_long_term=args.use_long_term,
                debug_router=args.debug_router,
                long_term_retrieval_scheme=args.long_term_retrieval_scheme,
            )
        if not args.print_latency:
            result = dict(result)
            result.pop("latency", None)
        outputs.append(result)
    output = outputs[0] if len(outputs) == 1 else outputs
    if args.output_json:
        write_json(Path(args.output_json), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
