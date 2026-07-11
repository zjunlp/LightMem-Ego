from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_memory_worker import _build_retrieval_artifacts
from online_retrieval_scheme import normalize_long_term_retrieval_scheme


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated native-heavy memory worker subprocess tasks.")
    sub = parser.add_subparsers(dest="command", required=True)
    retrieval = sub.add_parser("build_retrieval_artifacts")
    retrieval.add_argument("--session-id", required=True)
    retrieval.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    retrieval.add_argument(
        "--long-term-retrieval-scheme",
        "--retrieval-scheme",
        dest="long_term_retrieval_scheme",
        default=None,
    )
    args = parser.parse_args()

    if args.command == "build_retrieval_artifacts":
        long_term_retrieval_scheme = normalize_long_term_retrieval_scheme(args.long_term_retrieval_scheme)
        _build_retrieval_artifacts(
            session_id=args.session_id,
            sessions_root=Path(args.sessions_root),
            long_term_retrieval_scheme=long_term_retrieval_scheme,
        )
        print(
            json.dumps(
                {
                    "status": "done",
                    "session_id": args.session_id,
                    "long_term_retrieval_scheme": long_term_retrieval_scheme,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
