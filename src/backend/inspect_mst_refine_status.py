from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_preprocess.io_utils import read_json
from online_short_term.mst_store import MSTStore
from online_short_term.refine_status import write_refine_status
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect M_st refine readiness windows.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    store = MSTStore(Path(args.sessions_root) / args.session_id)
    windows_path, refine_state_path = write_refine_status(store)
    refine_state = read_json(refine_state_path, default={})
    windows = read_json(windows_path, default=[])
    payload = {
        "session_id": args.session_id,
        "mst_state": store.get_state(),
        "archive_state": store.get_archive_state(),
        "refine_state": refine_state,
        "refined_ready_windows_path": str(windows_path),
        "ready_windows": [w for w in windows if w.get("ready_for_30s_episodic")],
    }
    if args.verbose:
        payload["windows"] = windows
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
