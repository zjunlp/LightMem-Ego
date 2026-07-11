from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from online_memory import build_online_worldmm_memory
from online_memory_incremental import IncrementalMemoryAppender


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt one online session into WorldMM-compatible memory inputs.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-visual-embedding", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--use-fake-egolife-layout", action="store_true", help="Accepted for compatibility; current adapter writes direct WorldMM roots.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--source", default="auto", choices=["auto", "online_evidence", "legacy_evidence", "mst_episodic"])
    parser.add_argument("--generation-backend", default=None, choices=["llm", "rule"], help="30s+ memory generation backend. Defaults to WORLDMM_MEMORY_GENERATION_BACKEND or llm.")
    parser.add_argument("--update-mode", default=None, choices=["incremental_append", "full_rebuild_fallback"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.generation_backend:
        import os
        os.environ["WORLDMM_MEMORY_GENERATION_BACKEND"] = args.generation_backend

    update_mode = args.update_mode or os.getenv("WORLDMM_MEMORY_UPDATE_MODE", "incremental_append")
    if args.source in {"auto", "mst_episodic"} and update_mode == "incremental_append":
        result = IncrementalMemoryAppender(
            session_id=args.session_id,
            sessions_root=Path(args.sessions_root),
            project_root=Path(__file__).resolve().parent,
            model_name=args.model,
            verbose=args.verbose,
        ).append_ready_episodes(force=args.force, dry_run=args.dry_run, skip_graph_semantic=args.skip_semantic)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        if update_mode == "full_rebuild_fallback" and os.getenv("WORLDMM_ALLOW_FULL_REBUILD_FALLBACK", "0").lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("full_rebuild_fallback is disabled. Set WORLDMM_ALLOW_FULL_REBUILD_FALLBACK=1.")
        config_path = build_online_worldmm_memory(
            session_id=args.session_id,
            sessions_root=Path(args.sessions_root),
            force=args.force,
            skip_visual_embedding=args.skip_visual_embedding,
            skip_semantic=args.skip_semantic,
            dry_run=args.dry_run,
            limit_segments=args.limit_segments,
            model_name=args.model,
            source=args.source,
            verbose=args.verbose,
        )
        print(f"Memory config: {config_path}")


if __name__ == "__main__":
    main()
