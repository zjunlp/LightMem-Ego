#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import hashlib
import argparse
import json
import logging
import os
from typing import Any, Dict, List
from pathlib import Path
from tqdm import tqdm
import sys

from em2mem.embedding import EmbeddingModel
from em2mem.llm import LLMModel
from em2mem.memory.semantic_graph.semantic_consolidation import SemanticConsolidation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def scale_rank(scale: str) -> int:
    return {
        "10sec": 0,
        "30sec": 1,
        "3min": 2,
        "10min": 3,
        "grouped_10sec": 10,
    }.get(scale, 99)


def sort_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            str(x.get("date", "")),
            str(x.get("start_time", "")),
            str(x.get("end_time", "")),
            scale_rank(str(x.get("scale", ""))),
            str(x.get("chunk_id", x.get("doc_id", ""))),
        ),
    )


def triple_fingerprint(triple: List[str]) -> str:
    raw = "||".join([str(x).strip().lower() for x in triple])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def fact_id(triple: List[str]) -> str:
    return f"sf_{triple_fingerprint(triple)}"


def _upsert_fact(all_facts_dict: Dict[str, Dict[str, Any]], fid: str, triple: List[str],
            evidence_refs: List[str], root_metadata_map: Dict[str, Dict[str, Any]]) -> None:
    if fid in all_facts_dict:
        existing = all_facts_dict[fid]

        existing_ev_set = set(existing["evidence_refs"])
        for ref in evidence_refs:
            if ref not in existing_ev_set:
                existing["evidence_refs"].append(ref)
                existing_ev_set.add(ref)

        new_docs = _ordered_unique([ref.split("#", 1)[0] for ref in evidence_refs])
        existing_support_set = set(existing["support_docs"])
        for doc in new_docs:
            if doc not in existing_support_set:
                existing["support_docs"].append(doc)
                existing_support_set.add(doc)

        existing["support_count"] = len(existing["evidence_refs"])
        existing["support_doc_count"] = len(existing["support_docs"])
        existing["support_root_count"] = existing["support_doc_count"]
        if evidence_refs:
            last_ref = evidence_refs[-1]
            last_doc = last_ref.split("#", 1)[0]
            last_meta = root_metadata_map.get(last_doc, {})
            existing["last_seen"] = {
                "doc_id": last_meta.get("doc_id", last_doc),
                "date": last_meta.get("date", ""),
                "start_time": last_meta.get("start_time", ""),
                "end_time": last_meta.get("end_time", ""),
            }
            
    else:
        support_docs = _ordered_unique([ref.split("#", 1)[0] for ref in evidence_refs])
        support_days = []
        support_scales = []
        for doc_id in support_docs:
            meta = root_metadata_map.get(doc_id, {})
            day = str(meta.get("date", ""))
            scale = str(meta.get("scale", ""))
            if day and day not in support_days:
                support_days.append(day)
            if scale and scale not in support_scales:
                support_scales.append(scale)
        provenance_root_ids = support_docs.copy()
        first_doc_meta = root_metadata_map.get(support_docs[0], {}) if support_docs else {}
        last_doc_meta = root_metadata_map.get(support_docs[-1], {}) if support_docs else {}
        all_facts_dict[fid] = {
            "fact_id": fid,
            "triple": triple,
            "support_count": len(evidence_refs),
            "support_doc_count": len(support_docs),
            "support_root_count": len(provenance_root_ids),
            "support_docs": support_docs,
            "support_days": support_days,
            "support_scales": support_scales,
            "evidence_refs": evidence_refs,
            "provenance_root_ids": provenance_root_ids,
            "first_seen": {
                "doc_id": first_doc_meta.get("doc_id", support_docs[0] if support_docs else ""),
                "date": first_doc_meta.get("date", ""),
                "start_time": first_doc_meta.get("start_time", ""),
                "end_time": first_doc_meta.get("end_time", ""),
            },
            "last_seen": {
                "doc_id": last_doc_meta.get("doc_id", support_docs[-1] if support_docs else ""),
                "date": last_doc_meta.get("date", ""),
                "start_time": last_doc_meta.get("start_time", ""),
                "end_time": last_doc_meta.get("end_time", ""),
            },
        }


def consolidate(items: List[Dict[str, Any]], consolidator: SemanticConsolidation) -> Dict[str, Any]:
    items = sort_items(items)

    root_metadata_map: Dict[str, Dict[str, Any]] = {}
    for item in items:
        for unit_meta in item.get("source_units", []) or []:
            doc_id = str(unit_meta.get("doc_id", "")).strip()
            if doc_id:
                root_metadata_map[doc_id] = {
                    "doc_id": doc_id,
                    "scale": unit_meta.get("scale", "10sec"),
                    "date": unit_meta.get("date", ""),
                    "start_time": unit_meta.get("start_time", ""),
                    "end_time": unit_meta.get("end_time", ""),
                    "provenance_root_ids": unit_meta.get("provenance_root_ids", []) or [doc_id],
                }

    accumulated_triples: List[List[str]] = []
    accumulated_evidence: List[List[str]] = []
    all_facts_dict: Dict[str, Dict[str, Any]] = {}
    timeline: Dict[str, Any] = {}

    for i, item in tqdm(enumerate(items), total=len(items), desc="Building semantic memory", leave=True):
        chunk_id = item.get("chunk_id", item.get("doc_id", ""))
        current_triples = item.get("semantic_triples", []) or []
        current_evidence = item.get("episodic_evidence_refs", []) or []

        existing_results = (accumulated_triples.copy(), accumulated_evidence.copy())
        new_results = (current_triples, current_evidence)
        consolidated_triples, consolidated_evidence, triples_to_remove = (
            consolidator.batch_semantic_consolidation(existing_results, new_results)
        )

        remove_set = {(tuple(t), tuple(e)) for t, e in triples_to_remove}

        kept_triples = []
        kept_evidence = []
        removed_fact_ids = []
        for old_t, old_e in zip(accumulated_triples, accumulated_evidence):
            if (tuple(old_t), tuple(old_e)) in remove_set:
                removed_fact_ids.append(fact_id(old_t))
                continue
            kept_triples.append(old_t)
            kept_evidence.append(old_e)

        accumulated_triples = kept_triples
        accumulated_evidence = kept_evidence

        added_fact_ids = []
        for new_t, new_e in zip(consolidated_triples, consolidated_evidence):
            deduped_evidence = _ordered_unique(new_e)
            accumulated_triples.append(new_t)
            accumulated_evidence.append(deduped_evidence)
            added_fact_ids.append(fact_id(new_t))

        for triple, evidence in zip(accumulated_triples, accumulated_evidence):
            fid = fact_id(triple)
            evidence_refs = _ordered_unique(evidence)
            _upsert_fact(all_facts_dict, fid, triple, evidence_refs, root_metadata_map)

        active_fact_ids = sorted({fact_id(t) for t in accumulated_triples})
        active_root_ids = []
        for evidence_refs in accumulated_evidence:
            for ref in evidence_refs:
                active_root_ids.append(ref.split("#", 1)[0])

        timeline[chunk_id] = {
            "order": i,
            "chunk_id": chunk_id,
            "date": item.get("date", ""),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "scale": item.get("scale", ""),
            "group_period": item.get("group_period", None),
            "source_doc_ids": item.get("source_doc_ids", []),
            "added_fact_ids": sorted(set(added_fact_ids)),
            "removed_fact_ids": sorted(set(removed_fact_ids)),
            "active_fact_ids": active_fact_ids,
            "active_provenance_root_ids": _ordered_unique(active_root_ids),
        }

    facts_list = list(all_facts_dict.values())
    facts_list.sort(key=lambda x: (-x["support_count"], x["fact_id"]))
    return {"facts": facts_list, "timeline": timeline}


def run_semantic_consolidation(
    semantic_file: str,
    output_dir: str,
    model_name: str = "gpt-5-mini",
    llm_model: LLMModel | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> None:

    output_file = os.path.join(output_dir, f"semantic_graph_{model_name}.json")
    if os.path.exists(output_file):
        logger.info("Output file %s already exists, skipping consolidation.", output_file)
        return

    semantic_results = load_json(semantic_file)

    if "items" not in semantic_results:
        raise ValueError("Expected semantic extraction file with an `items` field.")

    items = semantic_results["items"]
    logger.info("Loaded %d extraction items", len(items))

    try:
        if embedding_model is None:
            embedding_model = EmbeddingModel()
            embedding_model.load_model(model_type="text")

        if llm_model is None:
            llm_model = LLMModel(model_name=model_name)

        output = consolidate(items, SemanticConsolidation(llm_model, embedding_model))
    except Exception as e:
        if "CUDA out of memory." in str(e):
            logger.error("CUDA out of memory error during consolidation. Consider reducing batch size or using a smaller model.")
            sys.exit(1)
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Saved semantic graph to %s", output_file)
    logger.info("Final semantic fact count: %d", len(output.get("facts", [])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate semantic extraction results into semantic graph.")
    parser.add_argument("--semantic-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--video-names", type=str, default=None, help="Comma-separated list of video names to process. If not provided, all subdirectories are used.")
    args = parser.parse_args()

    # Determine video names to process
    if args.video_names:
        video_names = [name.strip() for name in args.video_names.split(",")]
    else:
        video_names = [item.name for item in os.scandir(args.semantic_dir) if item.is_dir()]

    for video_name in tqdm(video_names, total=len(video_names), desc="Processing videos"):
        semantic_file = os.path.join(args.semantic_dir, video_name, f"semantic_candidates_{args.model}.json")
        output_dir = os.path.join(args.output_dir, video_name)
        run_semantic_consolidation(
            semantic_file=semantic_file,
            output_dir=output_dir,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
