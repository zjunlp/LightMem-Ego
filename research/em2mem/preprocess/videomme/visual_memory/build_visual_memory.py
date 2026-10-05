#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import argparse
import logging
import os
import subprocess
import sys
from typing import List, Optional

_PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, _PREPROCESS_DIR)

from em2mem.embedding import EmbeddingModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def discover_videos(caption_dir: str):
    """Return sorted list of video IDs that have a 10sec.json in their subdirectory."""
    if not os.path.isdir(caption_dir):
        raise FileNotFoundError(f"Caption directory not found: {caption_dir}")
    return sorted(
        d for d in os.listdir(caption_dir)
        if os.path.isdir(os.path.join(caption_dir, d))
        and os.path.exists(os.path.join(caption_dir, d, "10sec.json"))
    )


def select_video_ids_for_split(video_ids: List[str], split_id: int, num_splits: int) -> List[str]:
    """Return the contiguous chunk of video IDs assigned to a split."""
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")
    if split_id < 0 or split_id >= num_splits:
        raise ValueError(f"split_id must be in [0, {num_splits - 1}], got {split_id}")

    videos_per_split = (len(video_ids) + num_splits - 1) // num_splits
    start_idx = split_id * videos_per_split
    end_idx = min(start_idx + videos_per_split, len(video_ids))
    return video_ids[start_idx:end_idx]


def parse_gpu_list(gpu_arg: str) -> List[str]:
    """Parse a comma-separated GPU token list while preserving duplicates."""
    gpu_tokens = [token.strip() for token in gpu_arg.split(",") if token.strip()]
    if not gpu_tokens:
        raise ValueError("At least one GPU token must be provided for visual extraction")
    return gpu_tokens


def run_visual_worker(video_ids, caption_dir, output_dir, num_frames):
    from extract_visual_features import process_caption_dir

    if not video_ids:
        logger.info("No videos assigned to this visual worker")
        return

    embedding_model = EmbeddingModel(vis_model_name="VLM2Vec/VLM2Vec-V2.0")
    embedding_model.load_model(model_type="vision")

    process_caption_dir(
        caption_dir,
        os.path.join(output_dir),
        embedding_model,
        num_frames=num_frames,
        video_ids=video_ids,
    )


def spawn_visual_workers(video_ids, caption_dir, output_dir, model_name, gpu_tokens, num_frames):
    """Launch visual workers as child build_memory.py processes."""
    script_path = os.path.abspath(__file__)
    processes = []
    num_splits = len(gpu_tokens)

    for split_id, gpu_id in enumerate(gpu_tokens):
        assigned_video_ids = select_video_ids_for_split(video_ids, split_id, num_splits)
        logger.info("Starting visual worker %d/%d on GPU %s for %d videos", split_id + 1, num_splits, gpu_id, len(assigned_video_ids))

        cmd = [sys.executable, script_path, "--caption-dir", caption_dir, "--output-dir", output_dir, "--model", model_name, "--num-frames", str(num_frames), "--split-id", str(split_id), "--num-splits", str(num_splits)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        processes.append((split_id, gpu_id, subprocess.Popen(cmd, env=env)))

    failed = False
    for split_id, gpu_id, process in processes:
        return_code = process.wait()
        if return_code == 0:
            logger.info("Visual worker %d/%d on GPU %s completed successfully", split_id + 1, num_splits, gpu_id)
        else:
            logger.error("Visual worker %d/%d on GPU %s failed with exit code %d", split_id + 1, num_splits, gpu_id, return_code)
            failed = True

    if failed:
        raise RuntimeError("One or more visual workers failed")


def run_visual(video_ids, caption_dir, output_dir, model_name, gpu_arg, num_frames,
               split_id: Optional[int] = None, num_splits: int = 1):
    if split_id is not None:
        assigned_video_ids = select_video_ids_for_split(video_ids, split_id, num_splits)
        logger.info("Visual worker %d/%d processing %d videos", split_id + 1, num_splits, len(assigned_video_ids))
        run_visual_worker(assigned_video_ids, caption_dir, output_dir, num_frames)
        return

    gpu_tokens = parse_gpu_list(gpu_arg)
    logger.info("Launching visual extraction across %d worker(s): %s", len(gpu_tokens), ",".join(gpu_tokens))
    spawn_visual_workers(video_ids, caption_dir, output_dir, model_name, gpu_tokens, num_frames)


def main():
    parser = argparse.ArgumentParser(description="Batch visual memory building.")
    parser.add_argument("--caption-dir", type=str, required=True, help="Directory containing caption subdirectories with 10sec.json files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Root output directory.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name.")
    parser.add_argument("--gpu", type=str, default="0,1,2,3", help="Comma-separated GPU token list for visual extraction.")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract from each visual segment.")
    parser.add_argument("--split-id", type=int, default=None, help="Worker split ID for internal visual processing.")
    parser.add_argument("--num-splits", type=int, default=1, help="Total worker count for internal visual processing.")
    args = parser.parse_args()

    if args.num_frames < 1:
        parser.error("--num-frames must be at least 1.")
    if args.num_splits < 1:
        parser.error("--num-splits must be at least 1.")
    if args.split_id is not None and not 0 <= args.split_id < args.num_splits:
        parser.error("--split-id must be within [0, num_splits).")
    if args.split_id is None:
        try:
            parse_gpu_list(args.gpu)
        except ValueError as exc:
            parser.error(str(exc))

    video_ids = discover_videos(args.caption_dir)
    if not video_ids:
        logger.error(f"No video subdirectories with 10sec.json found in {args.caption_dir}")
        return
    logger.info(f"Found {len(video_ids)} videos to process")

    run_visual(video_ids, args.caption_dir, args.output_dir, args.model, args.gpu, args.num_frames, split_id=args.split_id, num_splits=args.num_splits)

    logger.info("Visual memory building complete.")


if __name__ == "__main__":
    main()
