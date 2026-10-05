#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, List

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OPTION_KEYS = ["choice_a", "choice_b", "choice_c", "choice_d"]
DEFAULT_TIME = {"date": "DAY1", "time": "23595999"}


def load_rows(input_parquet: Path) -> List[dict[str, Any]]:
    table = pq.read_table(input_parquet)
    return table.to_pylist()


def normalize_option_text(text: str) -> str:
    return re.sub(r"^[A-D][\.\)]\s*", "", (text or "")).strip()


def reformat_row(row: dict[str, Any]) -> dict[str, Any]:
    reformatted = {
        "ID": row["question_id"],
        "query_time": dict(DEFAULT_TIME),
        "type": row["task_type"],
        "video_id": row["videoID"],
        "duration": row["duration"],
        "domain": row["domain"],
        "sub_category": row["sub_category"],
        "url": row["url"],
        "question": row["question"],
    }

    options = row.get("options") or []
    for index, key in enumerate(OPTION_KEYS):
        option_text = options[index] if index < len(options) else ""
        reformatted[key] = normalize_option_text(option_text)

    reformatted["answer"] = row["answer"]
    reformatted["target_time"] = dict(DEFAULT_TIME)
    return reformatted


def save_rows(rows: List[dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Video-MME parquet file into JSON.")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/Video-MME/videomme/test-00000-of-00001.parquet"),
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/Video-MME/videomme/test.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="all",
        choices=["all", "short", "medium", "long"],
        help="Optionally filter rows by duration before saving.",
    )
    args = parser.parse_args()

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet file not found: {args.input_parquet}")

    rows = [reformat_row(row) for row in load_rows(args.input_parquet)]
    if args.duration != "all":
        rows = [row for row in rows if row["duration"] == args.duration]
    save_rows(rows, args.output_json)
    logger.info("Converted %d rows from %s to %s", len(rows), args.input_parquet, args.output_json)


if __name__ == "__main__":
    main()
