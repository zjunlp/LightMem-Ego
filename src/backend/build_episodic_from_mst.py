from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_short_term.mst_to_episodic import build_episodic_from_mst
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build refined 30s episodic memories from M_st archive micro-events.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default="openai", choices=["openai", "rule", "mock"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--window-end", type=float, default=None)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--update-em2mem", action="store_true", help="After building compatible MST episodic files, run the Em2Mem adapter.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.update_em2mem:
        from online_mst_to_em2mem import consolidate_short_term_to_em2mem

        result = consolidate_short_term_to_em2mem(
            session_id=args.session_id,
            sessions_root=Path(args.sessions_root),
            backend=args.backend,
            update_em2mem=True,
            force=args.force,
            dry_run=args.dry_run,
            window_start=args.window_start,
            window_end=args.window_end,
            limit_windows=args.limit_windows,
            verbose=args.verbose,
        )
    else:
        result = build_episodic_from_mst(
            session_id=args.session_id,
            sessions_root=Path(args.sessions_root),
            backend=args.backend,
            force=args.force,
            dry_run=args.dry_run,
            window_start=args.window_start,
            window_end=args.window_end,
            limit_windows=args.limit_windows,
            verbose=args.verbose,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
