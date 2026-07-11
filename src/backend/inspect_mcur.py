from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_current.mcur_store import MCurStore
from online_current.schemas import DEFAULT_SESSIONS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect current memory state.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = MCurStore(Path(args.sessions_root) / args.session_id)
    summary = store.summary(limit=10 if args.verbose else 5)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if args.verbose:
        context = store.get_current_context()
        print("\nopen_event:")
        print(json.dumps(context.get("open_event", {}), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
