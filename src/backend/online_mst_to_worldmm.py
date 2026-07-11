from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from online_memory import build_online_worldmm_memory
from online_memory_incremental import IncrementalMemoryAppender
from online_preprocess.io_utils import relative_to_session
from online_short_term.mst_to_episodic import build_episodic_from_mst, update_mst_worldmm_metadata
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT


def consolidate_short_term_to_worldmm(
    *,
    session_id: str,
    sessions_root: Path = DEFAULT_SESSIONS_ROOT,
    backend: str = "openai",
    update_worldmm: bool = True,
    force: bool = False,
    dry_run: bool = False,
    limit_windows: int | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
    skip_visual_embedding: bool = False,
    skip_semantic: bool = False,
    model_name: str | None = None,
    verbose: bool = False,
    generation_backend: str | None = None,
) -> dict[str, Any]:
    session_dir = Path(sessions_root) / session_id
    result = build_episodic_from_mst(
        session_id=session_id,
        sessions_root=Path(sessions_root),
        backend=backend,
        force=force,
        dry_run=dry_run,
        window_start=window_start,
        window_end=window_end,
        limit_windows=limit_windows,
        verbose=verbose,
    )
    if dry_run or not update_worldmm:
        result["updated_worldmm"] = False
        result["worldmm_update_mode"] = None
        return result
    if generation_backend:
        import os
        os.environ["WORLDMM_MEMORY_GENERATION_BACKEND"] = generation_backend

    captioned_path = session_dir / "captions" / "mst_session_30sec_captioned.json"
    evidence_path = session_dir / "evidence" / "mst_session_evidence.json"
    if not captioned_path.exists() or not evidence_path.exists():
        raise FileNotFoundError("MST episodic compatible files are missing; run build_episodic_from_mst first.")

    if os.getenv("WORLDMM_ALLOW_FULL_REBUILD_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"} and os.getenv("WORLDMM_MEMORY_UPDATE_MODE", "").strip().lower() in {"full_rebuild", "full_rebuild_fallback"}:
        config_path = build_online_worldmm_memory(
            session_id=session_id,
            sessions_root=Path(sessions_root),
            force=True,
            skip_visual_embedding=skip_visual_embedding,
            skip_semantic=skip_semantic,
            dry_run=False,
            limit_segments=None,
            model_name=model_name,
            source="mst_episodic",
            verbose=verbose,
        )
        update_mst_worldmm_metadata(session_dir, update_mode="full_rebuild_fallback")
        result["worldmm_update_mode"] = "full_rebuild_fallback"
        result["memory_config_path"] = relative_to_session(config_path, session_dir)
    else:
        append_result = IncrementalMemoryAppender(
            session_id=session_id,
            sessions_root=Path(sessions_root),
            project_root=Path(__file__).resolve().parent,
            model_name=model_name,
            verbose=verbose,
        ).append_ready_episodes(
            force=force,
            dry_run=False,
            skip_graph_semantic=skip_semantic,
        )
        result["worldmm_update_mode"] = append_result.worldmm_update_mode
        result["memory_config_path"] = "worldmm/memory_config.json"
        result["incremental_append"] = append_result.to_dict()
    result["updated_worldmm"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MST 30s episodic memory and optionally feed it into the WorldMM adapter.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default="openai", choices=["openai", "rule", "mock"])
    parser.add_argument("--update-worldmm", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--window-end", type=float, default=None)
    parser.add_argument("--skip-visual-embedding", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--generation-backend", default=None, choices=["llm", "rule"], help="WorldMM 30s+ memory generation backend. Defaults to WORLDMM_MEMORY_GENERATION_BACKEND or llm.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = consolidate_short_term_to_worldmm(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        backend=args.backend,
        update_worldmm=args.update_worldmm,
        force=args.force,
        dry_run=args.dry_run,
        limit_windows=args.limit_windows,
        window_start=args.window_start,
        window_end=args.window_end,
        skip_visual_embedding=args.skip_visual_embedding,
        skip_semantic=args.skip_semantic,
        model_name=args.model,
        verbose=args.verbose,
        generation_backend=args.generation_backend,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
