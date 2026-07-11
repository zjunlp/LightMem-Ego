"""
Backward-compatible wrapper around the local multiscale generator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_generate_multiscale_memory():
    try:
        from .multiscale import generate_multiscale_memory
        return generate_multiscale_memory
    except ImportError:
        current_dir = Path(__file__).resolve().parent
        src_root = Path(__file__).resolve().parents[3]
        for path in (str(current_dir), str(src_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from multiscale import generate_multiscale_memory
        return generate_multiscale_memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiscale episodic memory.")
    parser.add_argument("--db_name", default="A1_JAKE")
    parser.add_argument(
        "--json_path",
        default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_evidence.json",
    )
    parser.add_argument("--diary_dir", default=".cache/events_diary")
    parser.add_argument("--save_path", default="data/EgoLife/EgoLifeCap")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--perspective",
        default="egocentric",
        choices=["egocentric", "general"],
    )
    args = parser.parse_args()

    generate_multiscale_memory = _load_generate_multiscale_memory()
    generate_multiscale_memory(
        db_name=args.db_name,
        json_path=args.json_path,
        diary_dir=args.diary_dir,
        save_path=args.save_path,
        model_name=args.model,
        perspective=args.perspective,
    )
