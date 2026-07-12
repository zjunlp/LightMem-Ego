from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_memory_incremental import IncrementalMemoryAppender, inspect_hipporag_cache_health, reconcile_component_versions
from online_preprocess.io_utils import read_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Stage 8B component memory versions and append state.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--reconcile-component-versions", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    session_dir = Path(args.sessions_root) / args.session_id
    appender = IncrementalMemoryAppender(session_id=args.session_id, sessions_root=Path(args.sessions_root), project_root=PROJECT_ROOT)
    memory_config = read_json(session_dir / "em2mem" / "memory_config.json", default={})
    append_state = read_json(session_dir / "em2mem" / "incremental" / "append_state.json", default={})
    dirty_windows = read_json(session_dir / "em2mem" / "incremental" / "dirty_windows.json", default={})
    graph_state = read_json(session_dir / "em2mem" / "incremental" / "graph" / "graph_state.json", default={})
    semantic_state = read_json(session_dir / "em2mem" / "incremental" / "semantic" / "semantic_state.json", default={})
    reconcile_result = None
    if args.reconcile_component_versions:
        reconcile_result = reconcile_component_versions(session_dir)
    output = {
        "status": "ok",
        "session_id": args.session_id,
        "memory_config_versions": {
            "em2mem_update_mode": memory_config.get("em2mem_update_mode") if isinstance(memory_config, dict) else None,
            "latest_ready_memory_version": memory_config.get("latest_ready_memory_version") if isinstance(memory_config, dict) else None,
            "latest_fast_ready_version": memory_config.get("latest_fast_ready_version") if isinstance(memory_config, dict) else None,
            "latest_visual_ready_version": memory_config.get("latest_visual_ready_version") if isinstance(memory_config, dict) else None,
            "latest_graph_ready_version": memory_config.get("latest_graph_ready_version") if isinstance(memory_config, dict) else None,
            "latest_semantic_ready_version": memory_config.get("latest_semantic_ready_version") if isinstance(memory_config, dict) else None,
            "lag": memory_config.get("lag") if isinstance(memory_config, dict) else None,
            "readiness": memory_config.get("readiness") if isinstance(memory_config, dict) else None,
        },
        "component_versions": reconcile_result if reconcile_result is not None else appender.load_versions(),
        "reconciled_component_versions": bool(args.reconcile_component_versions),
        "append_state": append_state,
        "dirty_windows": dirty_windows,
        "graph_state": graph_state,
        "semantic_state": semantic_state,
        "hipporag_cache_health": inspect_hipporag_cache_health(session_dir, PROJECT_ROOT),
    }
    if args.verbose:
        output["memory_config"] = memory_config
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
