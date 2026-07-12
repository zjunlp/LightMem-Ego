from __future__ import annotations

import argparse
import os
from pathlib import Path

from online_preprocess.evidence_builder import build_session_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build online EvidenceDoc files for one uploaded session.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--backend", default=os.getenv("EM2MEM_EVIDENCE_CAPTION_BACKEND", "mock"), choices=["mock", "openai", "local"])
    parser.add_argument("--model", default=os.getenv("EM2MEM_VLM_MODEL"))
    parser.add_argument("--max-keyframes", type=int, default=int(os.getenv("EM2MEM_VLM_MAX_KEYFRAMES", "8")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evidence_path, captioned_path = build_session_evidence(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        backend=args.backend,
        model=args.model,
        max_keyframes=args.max_keyframes,
        force=args.force,
        limit_segments=args.limit_segments,
        dry_run=args.dry_run,
    )
    print(f"Evidence output: {evidence_path}")
    print(f"Captioned output: {captioned_path}")


if __name__ == "__main__":
    main()
