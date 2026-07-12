#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from em2mem.embedding import EmbeddingModel
from em2mem.llm import LLMModel, PromptTemplateManager
from em2mem.memory import EM2Memory, transform_timestamp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")


def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_existing(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _glob_first(candidates: List[str]) -> Optional[str]:
    for pattern in candidates:
        matched = sorted(glob.glob(pattern))
        if matched:
            return matched[0]
    return None


def build_episodic_caption_file_map(base_dir: str, subject: str) -> Dict[str, str]:
    file_map: Dict[str, str] = {}
    scale_to_patterns = {
        "30sec": [
            os.path.join(base_dir, f"{subject}_EVIDENCE_30sec.json"),
            os.path.join(base_dir, f"{subject}_30sec.json"),
            os.path.join(base_dir, f"{subject}_EVIDENCE_SMOKE_30sec.json"),
        ],
        "3min": [
            os.path.join(base_dir, f"{subject}_EVIDENCE_3min.json"),
            os.path.join(base_dir, f"{subject}_3min.json"),
            os.path.join(base_dir, f"{subject}_EVIDENCE_SMOKE_3min.json"),
        ],
        "10min": [
            os.path.join(base_dir, f"{subject}_EVIDENCE_10min.json"),
            os.path.join(base_dir, f"{subject}_10min.json"),
            os.path.join(base_dir, f"{subject}_EVIDENCE_SMOKE_10min.json"),
        ],
        "1h": [
            os.path.join(base_dir, f"{subject}_EVIDENCE_1h.json"),
            os.path.join(base_dir, f"{subject}_1h.json"),
            os.path.join(base_dir, f"{subject}_EVIDENCE_SMOKE_1h.json"),
        ],
    }
    scale_to_globs = {
        "30sec": [os.path.join(base_dir, "*30sec.json"), os.path.join(base_dir, "*30s.json")],
        "3min": [os.path.join(base_dir, "*3min.json")],
        "10min": [os.path.join(base_dir, "*10min.json")],
        "1h": [os.path.join(base_dir, "*1h.json")],
    }
    for scale in ["30sec", "3min", "10min", "1h"]:
        path = _first_existing(scale_to_patterns[scale]) or _glob_first(scale_to_globs[scale])
        if path:
            file_map[scale] = path
    return file_map


def build_episodic_sidecar_file_maps(base_dir: str, model_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    triplet_files = {}
    graph_files = {}
    folder_candidates = {
        "30sec": ["30s", "30sec"],
        "3min": ["3min"],
        "10min": ["10min"],
        "1h": ["1h"],
    }
    for scale, folders in folder_candidates.items():
        triplet_candidates = []
        graph_candidates = []
        for folder in folders:
            filename_scale = "30s" if scale == "30sec" else scale
            triplet_candidates.append(os.path.join(base_dir, folder, f"episodic_triplets_{filename_scale}_{model_name}.json"))
            graph_candidates.append(os.path.join(base_dir, folder, f"episodic_graph_{filename_scale}_{model_name}.json"))
        triplet_path = _first_existing(triplet_candidates)
        graph_path = _first_existing(graph_candidates)
        if triplet_path:
            triplet_files[scale] = triplet_path
        if graph_path:
            graph_files[scale] = graph_path
    return triplet_files, graph_files


def filter_existing_files(file_map: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in file_map.items() if os.path.exists(v)}


def infer_visual_evidence_file(episodic_caption_root: str, subject: str, user_specified: Optional[str]) -> Optional[str]:
    if user_specified:
        return user_specified if os.path.exists(user_specified) else None

    candidates = [
        os.path.join(episodic_caption_root, f"{subject}_evidence.json"),
        os.path.join(episodic_caption_root, f"{subject}_EVIDENCE_30sec.json"),
        os.path.join(episodic_caption_root, f"{subject}_30sec.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    globbed = sorted(glob.glob(os.path.join(episodic_caption_root, "*evidence*.json")))
    if globbed:
        return globbed[0]
    return None


def resolve_semantic_path(semantic_root: str, retriever_model: str) -> str:
    semantic_path = _first_existing([
        os.path.join(semantic_root, f"semantic_memory_{retriever_model}.json"),
        os.path.join(semantic_root, f"semantic_memory_{OPENAI_MODEL}.json"),
        os.path.join(semantic_root, f"semantic_consolidation_results_{retriever_model}.json"),
        os.path.join(semantic_root, f"semantic_consolidation_results_{OPENAI_MODEL}.json"),
    ])
    if semantic_path is None:
        candidates = sorted(glob.glob(os.path.join(semantic_root, "semantic_memory_*.json")))
        if candidates:
            semantic_path = candidates[0]
    if semantic_path is None:
        candidates = sorted(glob.glob(os.path.join(semantic_root, "semantic_consolidation_results_*.json")))
        if candidates:
            semantic_path = candidates[0]
    if semantic_path is None or not os.path.exists(semantic_path):
        raise FileNotFoundError(f"Semantic memory file not found under: {semantic_root}")
    return semantic_path


def parse_choices(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    choices: Dict[str, str] = {}
    if args.choice_a:
        choices["A"] = args.choice_a
    if args.choice_b:
        choices["B"] = args.choice_b
    if args.choice_c:
        choices["C"] = args.choice_c
    if args.choice_d:
        choices["D"] = args.choice_d
    return choices or None


def build_until_timestamp(day: str, time_code: str) -> int:
    day_digit = day.replace("DAY", "").replace("Day", "").strip()
    return int(day_digit + str(time_code).zfill(8))


def summarize_selected_events(em2mem_memory: EM2Memory, doc_ids: List[str]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        entry = em2mem_memory.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
        visual_entry = em2mem_memory.visual_memory.get_clip_by_doc_id(doc_id)
        if entry is None:
            continue
        summaries.append({
            "doc_id": doc_id,
            "date": getattr(entry, "date", ""),
            "start_time": getattr(entry, "start_time", ""),
            "end_time": getattr(entry, "end_time", ""),
            "text": getattr(entry, "text", ""),
            "video_path": getattr(visual_entry, "video_path", "") if visual_entry is not None else "",
            "keyframe_paths": list(getattr(visual_entry, "keyframe_paths", []) or []) if visual_entry is not None else [],
            "num_keyframes": len(list(getattr(visual_entry, "keyframe_paths", []) or [])) if visual_entry is not None else 0,
            "scene_summary": dict(getattr(visual_entry, "scene_summary", {}) or {}) if visual_entry is not None else {},
            "keyframe_caption": str(getattr(visual_entry, "keyframe_caption", "") or "") if visual_entry is not None else "",
        })
    return summaries


def summarize_semantic_facts(em2mem_memory: EM2Memory, fact_ids: List[str]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for fact_id in fact_ids:
        entry = em2mem_memory.semantic_memory.triple_id_to_entry.get(fact_id)
        if entry is None:
            continue
        summaries.append({
            "fact_id": entry.id,
            "triple": list(entry.triple),
            "semantic_summary": getattr(entry, "semantic_summary", ""),
            "confidence": float(getattr(entry, "confidence", 0.0)),
            "support_count": int(getattr(entry, "support_count", 0)),
            "support_event_ids": list(em2mem_memory.semantic_memory.get_support_event_ids(entry, limit=10)),
            "provenance_root_ids": list(getattr(entry, "provenance_root_ids", []) or []),
        })
    return summaries


def summarize_retrieved_items(qa_result) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for item in qa_result.retrieved_items:
        if item.memory_type == "visual":
            content_summary = {"num_images": len(item.content) if isinstance(item.content, list) else 0}
        else:
            text = item.content if isinstance(item.content, str) else ""
            content_summary = {"preview": text[:800]}
        summaries.append({
            "memory_type": item.memory_type,
            "query": item.query,
            "round_num": item.round_num,
            **content_summary,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single RAG query with EM2Memory.")
    parser.add_argument("--subject", type=str, default="A1_JAKE")
    parser.add_argument("--retriever-model", type=str, default="gpt-5")
    parser.add_argument("--respond-model", type=str, default="gpt-5")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--until-date", type=str, default="DAY1")
    parser.add_argument("--until-time", type=str, default="11193000")
    parser.add_argument("--episodic-top-k", type=int, default=5)
    parser.add_argument("--semantic-top-k", type=int, default=8)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--episodic-caption-root", type=str, required=True)
    parser.add_argument("--episodic-sidecar-root", type=str, required=True)
    parser.add_argument("--semantic-root", type=str, required=True)
    parser.add_argument("--visual-root", type=str, default=None)
    parser.add_argument("--visual-evidence-file", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--choice-a", type=str, default="")
    parser.add_argument("--choice-b", type=str, default="")
    parser.add_argument("--choice-c", type=str, default="")
    parser.add_argument("--choice-d", type=str, default="")
    parser.add_argument("--answer-mode", type=str, default="open_ended", choices=["auto", "open_ended", "multiple_choice"])
    args = parser.parse_args()

    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(model_name=args.retriever_model)
    respond_llm_model = LLMModel(model_name=args.respond_model, fps=1)
    prompt_template_manager = PromptTemplateManager()

    episodic_cache_tag = os.path.basename(os.path.normpath(args.episodic_caption_root))
    em2mem_memory = EM2Memory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=1,
        max_errors=3,
        episodic_cache_tag=episodic_cache_tag,
    )
    em2mem_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    try:
        episodic_caption_files = filter_existing_files(
            build_episodic_caption_file_map(args.episodic_caption_root, args.subject)
        )
        if "30sec" not in episodic_caption_files:
            raise FileNotFoundError("30sec caption file is required.")

        episodic_triplet_files, episodic_graph_files = build_episodic_sidecar_file_maps(
            args.episodic_sidecar_root,
            args.retriever_model,
        )
        episodic_triplet_files = filter_existing_files(episodic_triplet_files)
        episodic_graph_files = filter_existing_files(episodic_graph_files)

        semantic_results = load_json(resolve_semantic_path(args.semantic_root, args.retriever_model))

        visual_evidence_file = infer_visual_evidence_file(
            episodic_caption_root=args.episodic_caption_root,
            subject=args.subject,
            user_specified=args.visual_evidence_file,
        )
        visual_evidence_data = load_json(visual_evidence_file or episodic_caption_files["30sec"])

        visual_embeddings_path = None
        if args.visual_root:
            candidate_visual_path = os.path.join(args.visual_root, "visual_embeddings.pkl")
            if os.path.exists(candidate_visual_path):
                visual_embeddings_path = candidate_visual_path

        em2mem_memory.load_episodic_captions(caption_files=episodic_caption_files)
        if episodic_triplet_files or episodic_graph_files:
            em2mem_memory.load_episodic_sidecar(
                triplet_files=episodic_triplet_files,
                graph_files=episodic_graph_files,
            )
        em2mem_memory.load_semantic_triples(data=semantic_results)
        em2mem_memory.load_visual_clips(
            embeddings_path=visual_embeddings_path,
            clips_data=visual_evidence_data,
        )

        until_time = build_until_timestamp(args.until_date, args.until_time)
        choices = parse_choices(args)
        qa_result = em2mem_memory.answer(
            query=args.query,
            choices=choices,
            until_time=until_time,
            answer_mode=args.answer_mode,
        )

        selected_events = summarize_selected_events(em2mem_memory, qa_result.selected_doc_ids)
        supporting_semantic_facts = summarize_semantic_facts(em2mem_memory, qa_result.semantic_fact_ids)
        retrieved_items_summary = summarize_retrieved_items(qa_result)

        result = {
            "query": args.query,
            "choices": choices or {},
            "question_type": "multiple_choice" if choices else "open_ended",
            "answer_mode_requested": args.answer_mode,
            "answer_mode_used": qa_result.answer_mode,
            "qa_template_name": qa_result.qa_template_name,
            "answer": qa_result.answer,
            "answer_text": qa_result.answer,
            "num_rounds": qa_result.num_rounds,
            "round_history": qa_result.round_history,
            "selected_event_doc_ids": qa_result.selected_doc_ids,
            "selector_reason": qa_result.selector_reason,
            "supporting_semantic_fact_ids": qa_result.semantic_fact_ids,
            "visual_event_image_counts": qa_result.visual_event_image_counts,
            "evidence_summary": {
                "num_selected_events": len(qa_result.selected_doc_ids),
                "num_supporting_semantic_facts": len(qa_result.semantic_fact_ids),
                "num_visual_images": sum(qa_result.visual_event_image_counts.values()),
                "num_retrieved_items": len(qa_result.retrieved_items),
            },
            "selected_events": selected_events,
            "supporting_semantic_facts": supporting_semantic_facts,
            "retrieved_items_summary": retrieved_items_summary,
            "until_timestamp": until_time,
            "until_timestamp_str": transform_timestamp(str(until_time)),
            "episodic_caption_root": args.episodic_caption_root,
            "episodic_sidecar_root": args.episodic_sidecar_root,
            "semantic_root": args.semantic_root,
            "visual_evidence_file": visual_evidence_file,
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))

        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    finally:
        try:
            em2mem_memory.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
