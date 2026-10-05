#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import json
import pickle
import numpy as np
import os
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

from em2mem.embedding.embedding_wrapper import EmbeddingModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _time_str_to_seconds(time_str: str) -> float:
    """Convert HHMMSSFF time string to seconds."""
    time_str = time_str.zfill(8)
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    return float(hours * 3600 + minutes * 60 + seconds)


def load_video_entries(json_path: str) -> List[dict]:
    """Load caption entries from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_embeddings(embeddings_dict: Dict[str, np.ndarray], output_path: str):
    """Save embeddings dictionary to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")


def process_caption_dir(
    caption_dir: str,
    output_dir: str,
    embedding_model: EmbeddingModel,
    num_frames: int = 16,
    video_ids: Optional[List[str]] = None,
):
    """Process caption subdirectories, producing per-video embeddings keyed by time."""
    if video_ids is None:
        video_dirs = sorted(
            d for d in os.listdir(caption_dir)
            if os.path.isdir(os.path.join(caption_dir, d))
        )
    else:
        video_dirs = list(video_ids)
    if not video_dirs:
        print(f"No video subdirectories found in {caption_dir}")
        return

    print(f"Processing visual embeddings for {len(video_dirs)} videos...")
    for video_id in tqdm(video_dirs, desc="Visual embeddings"):
        base_json = os.path.join(caption_dir, video_id, "10sec.json")
        if not os.path.exists(base_json):
            print(f"  Skipping {video_id}: no 10sec.json found")
            continue

        out_pkl = os.path.join(output_dir, video_id, "visual_embeddings.pkl")
        if os.path.exists(out_pkl):
            print(f"  Skipping {video_id}: visual_embeddings.pkl already exists")
            continue

        entries = load_video_entries(base_json)
        if not any('video_path' in entry for entry in entries):
            continue

        embeddings_dict: Dict[str, np.ndarray] = {}

        for entry in entries:
            vp = entry.get('video_path', '')
            start_time = str(entry.get('start_time', ''))
            end_time = str(entry.get('end_time', ''))
            if not vp or not os.path.exists(vp):
                continue

            start_sec = _time_str_to_seconds(start_time)
            end_sec = _time_str_to_seconds(end_time)
            key = start_time

            video_spec = {"video": vp, "video_start": start_sec, "video_end": end_sec}
            try:
                embedding = embedding_model.encode_video(
                    [video_spec], nframes=num_frames, batch_size=1,
                )
                embeddings_dict[key] = embedding[0]
            except Exception as e:
                print(f"  Error encoding {video_id} segment {start_time}-{end_time}: {e}")

        if embeddings_dict:
            os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
            save_embeddings(embeddings_dict, out_pkl)

    print("Visual embedding extraction complete.")
