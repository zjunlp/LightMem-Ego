#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import json
import pickle
import numpy as np
import os
import argparse
from typing import Dict, List
from tqdm import tqdm

from em2mem.embedding.embedding_wrapper import EmbeddingModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_video_paths(json_path: str) -> List[str]:
    """Load video paths from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_paths = []
    for item in data:
        if 'video_path' in item:
            video_paths.append(item['video_path'])
    
    print(f"Loaded {len(video_paths)} video paths from {json_path}")
    return video_paths


def process_videos_sequentially(video_paths: List[str], embedding_model: EmbeddingModel, 
                                num_frames: int = 16) -> Dict[str, np.ndarray]:
    """Process videos sequentially (one at a time) and extract embeddings."""
    embeddings_dict = {}
    
    # Process videos one by one
    for path in tqdm(video_paths, desc="Processing videos sequentially"):
        # Check if video file exists
        if not os.path.exists(path):
            print(f"Warning: Video file not found: {path}")
            continue
        
        try:
            # Extract embedding for single video
            embedding = embedding_model.encode_video(
                [path], 
                num_frames=num_frames,
                batch_size=1
            )
            
            # Store embedding in dictionary (keep as numpy array for pickle)
            embeddings_dict[path] = embedding[0]
            
        except Exception as e:
            print(f"Error processing video {path}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(embeddings_dict)} out of {len(video_paths)} videos")
    return embeddings_dict


def save_embeddings(embeddings_dict: Dict[str, np.ndarray], output_path: str):
    """Save embeddings dictionary to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")


def merge_split_embeddings(person: str, num_splits: int):
    """Merge split embedding files into a single file."""
    print(f"\n=== Merging {num_splits} split files for {person} ===")
    
    merged_embeddings = {}
    
    for split_id in range(num_splits):
        split_file = f"output/metadata/visual_memory/{person}/visual_embeddings_split_{split_id}.pkl"
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}")
            continue
        
        try:
            with open(split_file, 'rb') as f:
                split_embeddings = pickle.load(f)
            
            print(f"Loaded {len(split_embeddings)} embeddings from split {split_id}")
            merged_embeddings.update(split_embeddings)
            
        except Exception as e:
            print(f"Error loading split {split_id}: {str(e)}")
            continue

    if not merged_embeddings:
        print("Error: No embeddings to merge")
        return False
    
    # Save merged file
    output_file = f"output/metadata/visual_memory/{person}/visual_embeddings.pkl"
    try:
        save_embeddings(merged_embeddings, output_file)
        print(f"Successfully merged {len(merged_embeddings)} total embeddings")
        
        # Optionally remove split files after successful merge
        print("\nRemoving split files...")
        for split_id in range(num_splits):
            split_file = f"output/metadata/visual_memory/{person}/visual_embeddings_split_{split_id}.pkl"
            if os.path.exists(split_file):
                os.remove(split_file)
                print(f"Removed {split_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving merged embeddings: {str(e)}")
        return False


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process EgoLife visual embeddings')
    parser.add_argument('--split-id', type=int, default=None, 
                       help='Split ID for parallel processing (0-indexed). If not set, process all videos.')
    parser.add_argument('--num-splits', type=int, default=1,
                       help='Total number of splits for parallel processing')
    parser.add_argument('--num-frames', type=int, default=16,
                       help='Number of frames to extract from each video')
    parser.add_argument('--person', type=str, default='A1_JAKE',
                       help='Person to process (e.g., A1_JAKE)')
    parser.add_argument('--merge', action='store_true',
                       help='Merge split files instead of processing videos')
    
    args = parser.parse_args()
    
    name = args.person
    
    # Handle merge mode
    if args.merge:
        if args.num_splits <= 1:
            print("Error: --num_splits must be > 1 for merging")
            return
        merge_split_embeddings(name, args.num_splits)
        return
    
    # File paths
    input_json = f"data/EgoLife/EgoLifeCap/{name}/{name}.json"
    
    # Output path includes split_id if doing split processing
    if args.split_id is not None:
        output_pickle = f"output/metadata/visual_memory/{name}/visual_embeddings_split_{args.split_id}.pkl"
        print(f"\n=== Processing {name} (split {args.split_id + 1}/{args.num_splits}) ===")
    else:
        output_pickle = f"output/metadata/visual_memory/{name}/visual_embeddings.pkl"
        print(f"\n=== Processing {name} (all videos) ===")
    
    # Check if input file exists
    if not os.path.exists(input_json):
        print(f"Error: Input file not found: {input_json}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_pickle)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    embedding_model = EmbeddingModel(vis_model_name="VLM2Vec/VLM2Vec-V2.0")
    embedding_model.load_model(model_type="vision")
    
    # Load video paths
    try:
        video_paths = load_video_paths(input_json)
    except Exception as e:
        print(f"Error loading video paths: {str(e)}")
        return
    
    if not video_paths:
        print("No video paths found in input file")
        return
    
    # Split video paths for parallel processing if split_id is specified
    if args.split_id is not None:
        total_videos = len(video_paths)
        videos_per_split = (total_videos + args.num_splits - 1)
        start_idx = args.split_id * videos_per_split
        end_idx = min(start_idx + videos_per_split, total_videos)
        video_paths = video_paths[start_idx:end_idx]
        print(f"Split {args.split_id}: Processing videos {start_idx} to {end_idx-1} ({len(video_paths)} videos)")
    else:
        print(f"Processing all {len(video_paths)} videos")
    
    # Process videos and extract embeddings sequentially
    print(f"Processing videos sequentially (one at a time)...")
    embeddings_dict = process_videos_sequentially(
        video_paths, 
        embedding_model, 
        num_frames=args.num_frames
    )
    
    if not embeddings_dict:
        print("No embeddings were extracted")
        return

    # Save embeddings
    try:
        save_embeddings(embeddings_dict, output_pickle)
        if args.split_id is not None:
            print(f"Processing completed successfully for {name} split {args.split_id}!")
        else:
            print(f"Processing completed successfully for {name}!")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return


if __name__ == "__main__":
    main()
