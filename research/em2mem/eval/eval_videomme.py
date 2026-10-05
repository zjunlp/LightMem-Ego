#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors
import os
import sys
import json
import gc
import re
import torch
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from em2mem.embedding import EmbeddingModel
from em2mem.llm import LLMModel, PromptTemplateManager
from em2mem.memory import EM2Memory_videomme as EM2Memory, QAResult


def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])
    if pred_norm == gold_candidate:
        return True
    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True
    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}"),
    ]
    if pred_norm in full_patterns:
        return True
    return False


VIDEOMME_GRANULARITIES = ["10sec", "30sec", "3min", "10min"]
QUERY_TIME = int("1" + "23595999")  # DAY1 end-of-video: index everything


def build_choices(row: Dict[str, Any]) -> Dict[str, str]:
    """Build choices dict from a row."""
    choices = {}
    for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
        if key in row and row[key]:
            choices[label] = row[key]
    return choices


def load_processing_status(status_file: str) -> Dict[str, Dict[str, int]]:
    """Load the processing status from a file (if it exists)."""
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}


def save_processing_status(status_file: str, status: Dict[str, Dict[str, int]]):
    """Save the processing status to a file."""
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=4)


def get_gpu_id() -> str:
    """Return the GPU ID from CUDA_VISIBLE_DEVICES if set, else empty string."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu:
        return f"_gpu{gpu}"
    return ""


def load_oom_counter(gpu_tag: str, retriever_model: str, respond_model: str) -> Dict[str, int]:
    oom_file = f"output/{retriever_model}_{respond_model}/oom_counter{gpu_tag}.json"
    if os.path.exists(oom_file):
        with open(oom_file, 'r') as f:
            return json.load(f)
    return {}


def save_oom_counter(gpu_tag: str, counter: Dict[str, int], retriever_model: str, respond_model: str):
    oom_file = f"output/{retriever_model}_{respond_model}/oom_counter{gpu_tag}.json"
    with open(oom_file, 'w') as f:
        json.dump(counter, f, indent=2)


def increment_oom_counter(gpu_tag: str, video_id: str, retriever_model: str, respond_model: str) -> int:
    counter = load_oom_counter(gpu_tag, retriever_model, respond_model)
    new_count = counter.get(video_id, 0) + 1
    counter[video_id] = new_count
    save_oom_counter(gpu_tag, counter, retriever_model, respond_model)
    return new_count


def load_skip_set(gpu_tag: str, retriever_model: str, respond_model: str) -> set:
    skip_file = f"output/{retriever_model}_{respond_model}/skip_videos{gpu_tag}.json"
    if os.path.exists(skip_file):
        with open(skip_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_skip_set(gpu_tag: str, skip_set: set, retriever_model: str, respond_model: str):
    skip_file = f"output/{retriever_model}_{respond_model}/skip_videos{gpu_tag}.json"
    with open(skip_file, 'w') as f:
        json.dump(list(skip_set), f)


def mark_video_skip(gpu_tag: str, video_id: str, retriever_model: str, respond_model: str):
    skip_set = load_skip_set(gpu_tag, retriever_model, respond_model)
    skip_set.add(video_id)
    save_skip_set(gpu_tag, skip_set, retriever_model, respond_model)


def main():
    parser = argparse.ArgumentParser(description="Video-MMEEvaluation with event-centric EM2Memory")
    parser.add_argument("--eval-json", type=str, default="data/Video-MME/videomme/test.json", help="Path to Video-MME test JSON")
    parser.add_argument("--metadata-dir", type=str, default="output/videomme/metadata", help="Root metadata directory")
    parser.add_argument("--retriever-model", type=str, default="gpt-5-mini", help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="gpt-5", help="LLM model for reasoning and answering")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="output/videomme", help="Output directory for results")
    parser.add_argument("--duration", type=str, default=None, choices=["short", "medium", "long"], help="Optionally filter by video duration")
    parser.add_argument("--video-ids", type=str, default=None, help="Comma-separated list of video IDs to process (e.g. 'abc123,def456'). If not set, process all.")
    args = parser.parse_args()

    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm = LLMModel(model_name=args.retriever_model)
    respond_llm = LLMModel(model_name=args.respond_model, fps=1)
    prompt_template_manager = PromptTemplateManager()

    em2mem = EM2Memory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm,
        respond_llm_model=respond_llm,
        prompt_template_manager=prompt_template_manager,
        episodic_granularities=VIDEOMME_GRANULARITIES,
        qa_template_name="qa",
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )
    em2mem.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    logger.info("Loading evaluation data...")
    duration_tag = f"_{args.duration}" if args.duration else ""
    gpu_tag = get_gpu_id()
    output_filename = f"videomme_eval{duration_tag}{gpu_tag}.json"
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-','_')}_{args.respond_model.replace('-','_')}",
        output_filename,
    )
    processing_status = load_processing_status(output_path)
    skip_set = load_skip_set(gpu_tag, args.retriever_model, args.respond_model)

    eval_data = load_json(args.eval_json)
    if args.duration:
        eval_data = [r for r in eval_data if r.get("duration") == args.duration]

    queries_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_data:
        queries_by_video[row["video_id"]].append(row)

    # Filter by video-ids if provided
    if args.video_ids:
        selected_ids = set(args.video_ids.split(','))
        queries_by_video = {vid: qs for vid, qs in queries_by_video.items() if vid in selected_ids}
        logger.info(f"Filtered to {len(queries_by_video)} videos matching provided IDs")

    results: List[Dict[str, Any]] = []
    total_correct = 0
    total_processed = 0
    model_name = args.retriever_model

    video_progress = tqdm(sorted(queries_by_video.items()), desc="Videos", unit="video")
    for video_id, video_queries in video_progress:
        video_progress.set_postfix(vid=video_id, q=len(video_queries))

        # Skip videos marked as skip (OOM twice)
        if video_id in skip_set:
            logger.warning(f"Skipping video {video_id} because it is in skip set (OOM twice).")
            continue

        # Skip videos already processed successfully
        if video_id in processing_status:
            logger.info(f"Skipping video {video_id} as it is already processed.")
            results.extend(processing_status[video_id]['results'])
            total_correct += processing_status[video_id]['correct']
            total_processed += processing_status[video_id]['processed']
            continue

        processed_questions = 0
        correct_answers = 0

        try:
            em2mem.reset()

            caption_files = {}
            record_dir = os.path.join(args.metadata_dir, "multimodal_memory_cell", str(video_id))
            for g in VIDEOMME_GRANULARITIES:
                path = os.path.join(record_dir, "temporal_context_views", f"temporal_context_views_{g}.json")
                if g == "10sec":
                    path = os.path.join(record_dir, f"multimodal_event_record.json")
                if os.path.exists(path):
                    caption_files[g] = path

            if not caption_files:
                logger.warning(f"No caption files found for video {video_id} in {record_dir}, skipping.")
                continue

            em2mem.load_episodic_captions(caption_files=caption_files)

            semantic_file = os.path.join(
                args.metadata_dir, "semantic_graph", str(video_id),
                f"semantic_graph_{model_name}.json",
            )
            if os.path.exists(semantic_file):
                em2mem.load_semantic_triples(file_path=semantic_file)

            visual_pkl = os.path.join(
                args.metadata_dir, "visual_memory", str(video_id),
                "visual_embeddings.pkl",
            )
            base_caption_file = caption_files.get("10sec")
            if os.path.exists(visual_pkl) and base_caption_file:
                clips_data = load_json(os.path.join("data/Video-MME/caption", str(video_id), "10sec.json"))
                em2mem.load_visual_clips(embeddings_path=visual_pkl, clips_data=clips_data)

            em2mem.index(QUERY_TIME)

            for row in video_queries:
                choices = build_choices(row)
                question = row["question"]
                answer = row["answer"]

                qa_result: Optional[QAResult] = None
                try:
                    qa_result = em2mem.answer(
                        query=question,
                        choices=choices,
                        until_time=QUERY_TIME,
                    )
                    response = qa_result.answer
                except Exception as e:
                    logger.error(f"Error answering {row['ID']}: {e}")
                    response = "Error"

                correct = evaluate_prediction(response, answer, choices)
                correct_answers += int(correct)
                total_correct += int(correct)
                processed_questions += 1
                total_processed += 1

                results.append({
                    "ID": row["ID"],
                    "video_id": video_id,
                    "type": row.get("type", ""),
                    "duration": row.get("duration", ""),
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "response": response,
                    "round_history": qa_result.round_history if qa_result else [],
                    "num_rounds": qa_result.num_rounds if qa_result else 0,
                    "evaluate": correct,
                })

                logger.info(
                    f"{row['ID']} Pred: {response}, Gold: {answer}, "
                    f"Acc: {total_correct}/{total_processed} = {total_correct/total_processed:.4f}"
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                count = increment_oom_counter(gpu_tag, video_id, retriever_model=args.retriever_model, respond_model=args.respond_model)
                if count == 2:
                    mark_video_skip(gpu_tag, video_id, retriever_model=args.retriever_model, respond_model=args.respond_model)
                    logger.warning(f"Video {video_id} OOM twice, marked as skip.")
                logger.warning(f"OOM on video {video_id} (attempt {count}), exiting for retry.")
                sys.exit(1)
            else:
                raise e

        # Save processing status after each video
        processing_status[video_id] = {
            "correct": correct_answers,
            "processed": processed_questions,
            "results": results[-len(video_queries):],
        }
        save_processing_status(output_path, processing_status)

    # Final accuracy across all videos
    final_accuracy = total_correct / total_processed if total_processed > 0 else 0
    logger.info(f"Final Accuracy across all videos: {final_accuracy:.4f}")
    em2mem.cleanup()


if __name__ == "__main__":
    main()