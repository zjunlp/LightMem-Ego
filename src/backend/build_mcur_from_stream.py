from __future__ import annotations

import argparse
from pathlib import Path

from online_current.mcur_updater import MCurUpdater
from online_current.schemas import DEFAULT_SESSIONS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/update M_cur from existing pseudo-stream chunks.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--force", action="store_true", help="Clear existing current memory before rebuilding.")
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = Path(args.sessions_root) / args.session_id
    updater = MCurUpdater(session_dir)
    state = updater.update_from_existing_stream(force=args.force, limit_chunks=args.limit_chunks)
    if args.verbose:
        print("M_cur state:", state)
        print("current_dir:", session_dir / "current")


if __name__ == "__main__":
    main()
