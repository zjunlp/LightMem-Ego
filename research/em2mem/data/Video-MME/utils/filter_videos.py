#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import argparse
from pathlib import Path

import pyarrow.parquet as pq

EXTS = {".mp4", ".mkv", ".mov", ".webm"}
CHOICES = ("all", "short", "medium", "long")
PARQUET = Path("data/Video-MME/videomme/test-00000-of-00001.parquet")
ROOT = Path("data/Video-MME")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, default=PARQUET)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--duration", default="all", choices=CHOICES, help="which videos to keep based on duration")
    return parser.parse_args()


def keep_ids(parquet, duration):
    rows = pq.read_table(parquet, columns=["videoID", "duration"]).to_pylist()
    all_ = duration == "all"
    return {r["videoID"] for r in rows if all_ or r["duration"] == duration}


def video_roots(root):
    return sorted(p for p in root.glob("data*") if p.is_dir())


def video_files(roots):
    return sorted(
        p for root in roots for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )


def prune_empty(root):
    dirs = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(
        [root, *dirs], key=lambda path: len(path.parts), reverse=True
    ):
        if path.exists() and not any(path.iterdir()):
            path.rmdir()


def main():
    args = parse_args()
    if not args.input_parquet.exists():
        raise FileNotFoundError(args.input_parquet)
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)

    keep = keep_ids(args.input_parquet, args.duration)
    if not keep:
        raise ValueError(f"no videos for {args.duration}")

    roots = video_roots(args.dataset_root)
    removed = 0
    for path in video_files(roots):
        if path.stem in keep:
            continue
        path.unlink()
        removed += 1

    for root in roots:
        prune_empty(root)

    kept = len(video_files([root for root in roots if root.exists()]))
    print(f"duration={args.duration} kept={kept} removed={removed}")


if __name__ == "__main__":
    main()
