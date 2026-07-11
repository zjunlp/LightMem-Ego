from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_memory_incremental import IncrementalMemoryAppender, refresh_hipporag_cache
from online_pipeline.runtime_state import refresh_session_pipeline_state


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally append MST 30s episodes into WorldMM long-term memory.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--episode-id", action="append", dest="episode_ids", default=None)
    parser.add_argument("--append-ready-episodes", action="store_true", help="Append all MST 30s episodes not present in memory_append_log.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-graph-semantic", action="store_true")
    parser.add_argument("--refresh-hipporag-cache-only", action="store_true", help="Only append/reconcile HippoRAG parquet/vector/graph cache from active captions.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.refresh_hipporag_cache_only:
        try:
            result = refresh_hipporag_cache(
                session_dir=Path(args.sessions_root) / args.session_id,
                project_root=PROJECT_ROOT,
                model_name=args.model,
                force=args.force,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "session_id": args.session_id,
                "healthy": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if result.get("status") != "failed":
            refresh_session_pipeline_state(Path(args.sessions_root) / args.session_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.append_ready_episodes and not args.episode_ids:
        parser.error("Pass --append-ready-episodes or at least one --episode-id.")

    appender = IncrementalMemoryAppender(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        project_root=PROJECT_ROOT,
        model_name=args.model,
        verbose=args.verbose,
    )
    result = appender.append_ready_episodes(
        episode_ids=args.episode_ids,
        force=args.force,
        dry_run=args.dry_run,
        skip_graph_semantic=args.skip_graph_semantic,
    )
    if not args.dry_run:
        refresh_session_pipeline_state(Path(args.sessions_root) / args.session_id)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
