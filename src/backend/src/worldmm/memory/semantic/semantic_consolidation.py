# import json
# import os
# from typing import Dict, Any, List, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import logging
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# from .utils import ConsolidationRawOutput
# from ...llm import LLMModel, PromptTemplateManager
# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)

# class SemanticConsolidation:
#     def __init__(self, llm_model: LLMModel, embedding_model: EmbeddingModel):
#         self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
#         self.llm_model = llm_model
#         self.embedding_model = embedding_model
#         self.similarity_threshold = 0.6  # Threshold for finding relevant existing triples

#     def find_relevant_triples(self, new_triples: List[List[str]], 
#                                 existing_triples: List[Tuple[List[str], List[str]]], 
#                                 top_k: int = 20) -> List[List[Tuple[List[str], List[str]]]]:
#         """
#         Find existing triples that are relevant to a new triple using semantic similarity.
        
#         Args:
#             new_triples: List of new semantic triples [subject, predicate, object]
#             existing_triples: List of tuples (triple, evidence_list) where evidence is in "{timestamp}_{idx}" format
#             top_k: Maximum number of relevant triples to return
            
#         Returns:
#             List of List of tuples (existing_triple, evidence_list) for relevant matches
#         """
#         if not existing_triples:
#             return [[] for _ in new_triples]
        
#         # Convert triples to text for embedding
#         new_triple_texts = [" ".join(triple) for triple in new_triples]
#         existing_triple_texts = [" ".join(triple) for triple, _ in existing_triples]
        
#         # Get embeddings using text modality
#         new_embeddings = self.embedding_model.encode(new_triple_texts, modality="text")
#         existing_embeddings = self.embedding_model.encode(existing_triple_texts, modality="text")
        
#         # Compute cosine similarity matrix (new_triples x existing_triples)
#         similarities = cosine_similarity(new_embeddings, existing_embeddings)
        
#         # Get top-k indices for each new triple (sorted by similarity descending)
#         top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        
#         # Return relevant triples for each new triple
#         return [
#             [existing_triples[idx] for idx in indices if similarities[i, idx] >= self.similarity_threshold]
#             for i, indices in enumerate(top_k_indices)
#         ]

#     def consolidate_triple(self, new_triple: List[str], 
#                             relevant_existing_triples: List[List[str]]) -> Tuple[List[str], List[int]]:
#         """
#         Consolidate a new semantic triple with relevant existing ones.
        
#         Args:
#             new_triple (List[str]): A single new semantic triple [subject, predicate, object]
#             relevant_existing_triples (List[List[str]]): List of relevant existing triples
            
#         Returns:
#             Tuple of (updated_new_triple, triple_indices_to_remove)
#         """
#         if not relevant_existing_triples:
#             return new_triple, []

#         formatted_existing_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(relevant_existing_triples))
#         messages = self.prompt_template_manager.render(
#             name='semantic_consolidation',
#             new_triple=new_triple,
#             existing_triples=formatted_existing_triples
#         )

#         try:
#             # LLM INFERENCE
#             # Ensure messages is a list for chat-based templates
#             if isinstance(messages, str):
#                 raise ValueError("Expected chat template to return List[Dict], got string")
#             response = self.llm_model.generate(messages, text_format=ConsolidationRawOutput)

#         except Exception as e:
#             logger.warning(e)
#             return new_triple, []
        
#         return response.updated_triple, response.triples_to_remove

#     def batch_semantic_consolidation(self, existing_semantic_results: Tuple[List[List[str]], List[List[str]]], 
#                                       new_semantic_results: Tuple[List[List[str]], List[List[str]]]) -> Tuple[List[List[str]], List[List[str]], List[Tuple[List[str], List[str]]]]:
#         """
#         Conduct semantic consolidation for a single timestamp against existing results.
        
#         Args:
#             existing_semantic_results: Tuple of (existing_semantic_triples, existing_episodic_evidence) 
#                                      where existing_episodic_evidence is already in "{timestamp}_{idx}" format
#             new_semantic_results: Tuple of (new_semantic_triples, new_episodic_evidence)
#                                 where new_episodic_evidence is also in "{timestamp}_{idx}" format
            
#         Returns:
#             Tuple of (consolidated_semantic_triples, consolidated_episodic_evidence, triples_to_remove)
#             where triples_to_remove is a list of (triple, evidence) tuples that should be removed from accumulated state
#         """
#         existing_semantic_triples, existing_episodic_evidence = existing_semantic_results
#         new_semantic_triples, new_episodic_evidence = new_semantic_results
        
#         if not new_semantic_triples:
#             # No new triples to consolidate, return empty results
#             return [], [], []
        
#         # Create accumulated triples structure from existing results
#         # Each tuple is (triple, evidence_list)
#         accumulated_triples: List[Tuple[List[str], List[str]]] = []
#         for triple, evidence in zip(existing_semantic_triples, existing_episodic_evidence):
#             accumulated_triples.append((triple, evidence))
        
#         # Process all new triples concurrently
#         consolidated_results = self._process_timestamp_triples_concurrent(
#             new_semantic_triples, new_episodic_evidence, accumulated_triples
#         )
        
#         # Extract results
#         consolidated_triples = [result["updated_triple"] for result in consolidated_results]
#         consolidated_evidence = [result["merged_evidence"] for result in consolidated_results]
        
#         # Collect all triples to be removed across all consolidations
#         all_triples_to_remove: List[Tuple[List[str], List[str]]] = []
#         for result in consolidated_results:
#             all_triples_to_remove.extend(result["triples_to_remove"])
        
#         return consolidated_triples, consolidated_evidence, all_triples_to_remove

#     def _process_timestamp_triples_concurrent(self, current_triples: List[List[str]], 
#                                             current_evidence: List[List[str]],
#                                             accumulated_triples: List[Tuple[List[str], List[str]]]) -> List[Dict[str, Any]]:
#         """
#         Process all triples at a timestamp concurrently.
        
#         Args:
#             current_triples: New semantic triples for this timestamp
#             current_evidence: Evidence in "{timestamp}_{idx}" format
#             accumulated_triples: Existing triples from previous timestamps
            
#         Returns:
#             List of consolidation results with updated triples and merged evidence
#         """
#         # Find relevant triples for ALL new triples at once (optimization)
#         all_relevant_existing_data = self.find_relevant_triples(current_triples, accumulated_triples)
        
#         def process_single_triple(triple_idx: int) -> Dict[str, Any]:
#             new_triple = current_triples[triple_idx]
#             new_evidence = current_evidence[triple_idx]
            
#             # Get precomputed relevant triples for this specific triple
#             relevant_existing = all_relevant_existing_data[triple_idx]
            
#             if not relevant_existing:
#                 # No relevant existing triples, return as-is
#                 return {
#                     "updated_triple": new_triple,
#                     "triples_to_remove": [],
#                     "merged_evidence": new_evidence,
#                     "triple_idx": triple_idx
#                 }
            
#             relevant_triples_only = [triple for triple, _ in relevant_existing]
            
#             # Consolidate with LLM
#             updated_triple, indices_to_remove = self.consolidate_triple(new_triple, relevant_triples_only)
            
#             # Merge evidence from removed triples
#             merged_evidence = new_evidence.copy()
#             triples_to_remove_data = []
            
#             for remove_idx in indices_to_remove:
#                 if remove_idx < len(relevant_existing):
#                     removed_triple, removed_evidence = relevant_existing[remove_idx]
#                     merged_evidence.extend(removed_evidence)
#                     triples_to_remove_data.append((removed_triple, removed_evidence))
            
#             return {
#                 "updated_triple": updated_triple,
#                 "triples_to_remove": triples_to_remove_data,
#                 "merged_evidence": merged_evidence,
#                 "triple_idx": triple_idx
#             }
        
#         # Process all triples concurrently
#         results = []
#         with ThreadPoolExecutor() as executor:
#             futures = {executor.submit(process_single_triple, i): i for i in range(len(current_triples))}
#             pbar = tqdm(as_completed(futures), total=len(futures), 
#                        desc=f"Consolidating triples", leave=False)
            
#             for future in pbar:
#                 result = future.result()
#                 results.append(result)
        
#         # Sort results by original triple index to maintain order
#         results.sort(key=lambda x: x["triple_idx"])
#         return results

#     def save_results(self, results: Dict[str, Any], output_dir: str = "."):
#         """
#         Save consolidation results to a JSON file.
        
#         Args:
#             results: The results dictionary to save
#             output_dir: Output directory path.
#         """

#         # Convert results to JSON-serializable format
#         json_results = {}
#         for key, value in results.items():
#             if hasattr(value, '__dict__'):
#                 json_results[key] = value.__dict__
#             else:
#                 json_results[key] = value
        
#         os.makedirs(output_dir, exist_ok=True)
#         with open(os.path.join(output_dir, f"semantic_consolidation_results_{self.llm_model.model_name}.json"), 'w', encoding='utf-8') as f:
#             json.dump(json_results, f, indent=2, ensure_ascii=False)



#!/usr/bin/env python3
# """
# Deterministic semantic fact consolidation.

# This module consolidates rule-based semantic candidates into stable semantic facts.
# It is designed to replace the old LLM-based semantic consolidation pipeline.

# Input:
# - semantic candidates produced by semantic_extraction.py

# Output:
# - accumulated semantic facts with:
#   - support_count
#   - support_days
#   - support_scales
#   - confidence
#   - habit_strength
#   - evidence_event_ids
# """

# import json
# import os
# import re
# import hashlib
# import logging
# from copy import deepcopy
# from typing import Dict, Any, List, Tuple, Optional

# logger = logging.getLogger(__name__)


# LOW_VALUE_TOPICS_EXACT = {
#     "awkwardness",
#     "left",
#     "right",
#     "up",
#     "down",
#     "here",
#     "there",
#     "place_visibility",
#     "open_item_first",
#     "install_item_as_backup",
#     "continue_left",
#     "move_left",
#     "move_right",
#     "everyone",
#     "group",
#     "plastic_bag",
#     "lining",
#     "this_thing",
#     "question",
#     "questions",
#     "understanding",
#     "hole",
#     "notch",
#     "l1",
#     "alice",
#     "lucy",
#     "jake",
#     "tasha",
#     "katrina",
#     "shure",
# }

# LOW_VALUE_TOPIC_PATTERNS = [
#     r"\bawkward",
#     r"\bvisibility\b",
#     r"\bleft\b",
#     r"\bright\b",
#     r"\bthis thing\b",
#     r"\bplastic bag\b",
#     r"\blining\b",
# ]

# LOW_VALUE_OBJECT_EXACT = {
#     "people",
#     "chair",
#     "stool",
#     "table",
#     "desk",
#     "cheek",
#     "face",
#     "head",
#     "hair",
#     "hand",
#     "hands",
#     "arm",
#     "arms",
#     "leg",
#     "legs",
#     "left leg",
#     "right leg",
#     "eye",
#     "eyes",
#     "mouth",
#     "pole",
#     "plastic bag",
# }

# ALLOWED_OBJECTS_PRIORITY = {
#     "phone",
#     "hard drive",
#     "tripod",
#     "cable",
#     "container",
#     "padding",
#     "laptop",
#     "tablet",
#     "charger",
#     "paper",
#     "whiteboard",
#     "bag",
#     "cart",
#     "refrigerator",
# }

# RELATION_MIN_SUPPORT = {
#     "frequently_uses": 2,
#     "frequently_handles": 3,
#     "frequently_interacts_with": 2,
#     "related_to": 2,
# }

# RELATION_PRIOR = {
#     "frequently_uses": 0.62,
#     "frequently_handles": 0.56,
#     "frequently_interacts_with": 0.60,
#     "related_to": 0.58,
# }


# def canonicalize_text(x: str) -> str:
#     x = str(x).strip().lower()
#     x = re.sub(r"\s+", " ", x)
#     return x


# def normalize_day_str(date_val: Any) -> str:
#     if date_val is None:
#         return ""
#     s = str(date_val)
#     m = re.search(r"DAY(\d+)", s)
#     if m:
#         return f"DAY{m.group(1)}"
#     m = re.search(r"(\d+)", s)
#     if m:
#         return f"DAY{m.group(1)}"
#     return s


# def normalize_time_str(t: Any) -> str:
#     return str(t).zfill(8)


# def day_to_int(day_str: str) -> int:
#     m = re.search(r"(\d+)", str(day_str))
#     return int(m.group(1)) if m else 0


# def fact_id_from_key(key: Tuple[str, str, str, str, str]) -> str:
#     raw = "||".join(key)
#     return "sf_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# def is_low_value_topic(topic: str) -> bool:
#     t = canonicalize_text(topic)
#     if not t:
#         return True
#     if t in LOW_VALUE_TOPICS_EXACT:
#         return True
#     for pattern in LOW_VALUE_TOPIC_PATTERNS:
#         if re.search(pattern, t):
#             return True
#     return False


# def is_low_value_object(obj: str) -> bool:
#     o = canonicalize_text(obj)
#     if not o:
#         return True
#     if o in LOW_VALUE_OBJECT_EXACT:
#         return True
#     return False


# def relation_threshold(relation: str) -> int:
#     return RELATION_MIN_SUPPORT.get(relation, 2)


# def compute_confidence(fact: Dict[str, Any]) -> float:
#     relation = fact["relation"]
#     support_count = int(fact.get("support_count", 1))
#     support_days_count = len(fact.get("support_days", []))
#     support_scales = set(fact.get("support_scales", []))
#     prior = RELATION_PRIOR.get(relation, 0.55)

#     confidence = prior
#     confidence += 0.05 * min(support_count, 5)
#     confidence += 0.05 * max(0, support_days_count - 1)

#     if "30s" in support_scales and "3min" in support_scales:
#         confidence += 0.03

#     if len(fact.get("evidence_event_ids", [])) >= 3:
#         confidence += 0.03

#     if relation == "frequently_handles":
#         # handles is broader and noisier than uses/interacts
#         confidence -= 0.03

#     confidence = max(0.0, min(0.95, confidence))
#     return round(confidence, 3)


# def compute_habit_strength(fact: Dict[str, Any]) -> str:
#     support_count = int(fact.get("support_count", 1))
#     support_days_count = len(fact.get("support_days", []))

#     if support_count >= 5 or support_days_count >= 3:
#         return "high"
#     if support_count >= 3:
#         return "medium"
#     return "low"


# def should_keep_candidate(candidate: Dict[str, Any]) -> bool:
#     relation = str(candidate.get("relation", "")).strip()
#     head = str(candidate.get("head", "")).strip()
#     tail = str(candidate.get("tail", "")).strip()
#     head_type = str(candidate.get("head_type", "")).strip()
#     tail_type = str(candidate.get("tail_type", "")).strip()

#     if not relation or not head or not tail:
#         return False

#     if head == tail:
#         return False

#     if relation == "related_to":
#         if tail_type == "Topic" and is_low_value_topic(tail):
#             return False
#         if tail_type == "Object" and is_low_value_object(tail):
#             return False

#     if relation in {"frequently_uses", "frequently_handles"}:
#         if tail_type != "Object":
#             return False
#         if is_low_value_object(tail):
#             return False

#     if relation == "frequently_interacts_with":
#         if head_type != "Person" or tail_type != "Person":
#             return False

#     return True


# class SemanticConsolidation:
#     """
#     Deterministic semantic fact accumulator.

#     It consumes semantic candidates and accumulates them into stable semantic facts.
#     """

#     def __init__(self, llm_model=None, embedding_model=None):
#         # kept for backward compatibility with previous constructor signature
#         self.llm_model = llm_model
#         self.embedding_model = embedding_model

#     def candidate_key(self, candidate: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
#         return (
#             str(candidate["head"]).strip(),
#             str(candidate["head_type"]).strip(),
#             str(candidate["relation"]).strip(),
#             str(candidate["tail"]).strip(),
#             str(candidate["tail_type"]).strip(),
#         )

#     def init_fact_from_candidate(self, candidate: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
#         key = self.candidate_key(candidate)
#         day = normalize_day_str(unit.get("date", ""))
#         scale = str(unit.get("scale", "")).strip()
#         doc_id = str(unit.get("doc_id", "")).strip()

#         source_triplets = candidate.get("source_triplets", []) or []

#         fact = {
#             "fact_id": fact_id_from_key(key),
#             "head": key[0],
#             "head_type": key[1],
#             "relation": key[2],
#             "tail": key[3],
#             "tail_type": key[4],
#             "semantic_summary": str(candidate.get("semantic_summary", "")).strip(),

#             "support_count": 1,
#             "support_days": [day] if day else [],
#             "support_scales": [scale] if scale else [],
#             "evidence_event_ids": [doc_id] if doc_id else [],

#             "first_seen": doc_id,
#             "last_seen": doc_id,

#             "source_triplets": source_triplets[:8],
#             "confidence": 0.0,
#             "habit_strength": "low",
#         }
#         fact["confidence"] = compute_confidence(fact)
#         fact["habit_strength"] = compute_habit_strength(fact)
#         return fact

#     def update_fact_with_candidate(self, fact: Dict[str, Any], candidate: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
#         day = normalize_day_str(unit.get("date", ""))
#         scale = str(unit.get("scale", "")).strip()
#         doc_id = str(unit.get("doc_id", "")).strip()

#         fact["support_count"] += 1

#         if day and day not in fact["support_days"]:
#             fact["support_days"].append(day)
#             fact["support_days"].sort(key=day_to_int)

#         if scale and scale not in fact["support_scales"]:
#             fact["support_scales"].append(scale)

#         if doc_id and doc_id not in fact["evidence_event_ids"]:
#             fact["evidence_event_ids"].append(doc_id)

#         if not fact.get("first_seen") and doc_id:
#             fact["first_seen"] = doc_id
#         if doc_id:
#             fact["last_seen"] = doc_id

#         # keep a few source triplets for debugging / provenance
#         for tri in candidate.get("source_triplets", []) or []:
#             if tri not in fact["source_triplets"]:
#                 fact["source_triplets"].append(tri)
#             if len(fact["source_triplets"]) >= 12:
#                 break

#         fact["confidence"] = compute_confidence(fact)
#         fact["habit_strength"] = compute_habit_strength(fact)
#         return fact

#     def update_fact_store(
#         self,
#         fact_store: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
#         unit: Dict[str, Any],
#     ) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
#         """
#         Update accumulated fact store with one semantic-candidate unit.
#         """
#         candidates = unit.get("semantic_candidates", []) or []

#         for candidate in candidates:
#             if not should_keep_candidate(candidate):
#                 continue

#             key = self.candidate_key(candidate)
#             if key not in fact_store:
#                 fact_store[key] = self.init_fact_from_candidate(candidate, unit)
#             else:
#                 fact_store[key] = self.update_fact_with_candidate(fact_store[key], candidate, unit)

#         return fact_store

#     def fact_is_mature(self, fact: Dict[str, Any]) -> bool:
#         relation = fact["relation"]
#         support_count = int(fact.get("support_count", 1))
#         threshold = relation_threshold(relation)

#         if support_count < threshold:
#             return False

#         if fact["tail_type"] == "Topic" and is_low_value_topic(fact["tail"]):
#             return False
#         if fact["tail_type"] == "Object" and is_low_value_object(fact["tail"]):
#             return False

#         # frequent handles should prefer more semantically meaningful objects
#         if relation == "frequently_handles":
#             tail = canonicalize_text(fact["tail"])
#             if tail not in ALLOWED_OBJECTS_PRIORITY and support_count < 4:
#                 return False

#         return True

#     def materialize_snapshot(
#         self,
#         fact_store: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
#     ) -> List[Dict[str, Any]]:
#         """
#         Materialize current semantic facts from accumulated store.

#         Only mature facts are surfaced into the snapshot.
#         """
#         facts = []
#         for _, fact in fact_store.items():
#             if not self.fact_is_mature(fact):
#                 continue

#             out = deepcopy(fact)
#             out["support_scales"] = sorted(out.get("support_scales", []))
#             out["evidence_event_ids"] = sorted(out.get("evidence_event_ids", []))
#             out["support_days"] = sorted(out.get("support_days", []), key=day_to_int)
#             out["confidence"] = compute_confidence(out)
#             out["habit_strength"] = compute_habit_strength(out)
#             facts.append(out)

#         facts.sort(
#             key=lambda x: (
#                 -float(x.get("confidence", 0.0)),
#                 -int(x.get("support_count", 0)),
#                 x.get("head", ""),
#                 x.get("relation", ""),
#                 x.get("tail", ""),
#             )
#         )
#         return facts

#     def save_results(self, results: Dict[str, Any], output_dir: str = ".", model_name: str = "rulebased"):
#         os.makedirs(output_dir, exist_ok=True)
#         out_path = os.path.join(output_dir, f"semantic_consolidation_results_{model_name}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Semantic consolidation results saved to: {out_path}")


# """
# Deterministic semantic fact consolidation.

# This revision keeps the old interface, but improves full-data robustness by:
# 1. removing obvious first2 / person-name hardcoding
# 2. using provenance-root-aware support counting to reduce cross-scale double-counting
# 3. keeping raw mention count alongside unique support count
# 4. remaining backward compatible with downstream SemanticMemory consumers
# """

# import json
# import os
# import re
# import hashlib
# import logging
# from copy import deepcopy
# from typing import Dict, Any, List, Tuple, Iterable

# logger = logging.getLogger(__name__)

# GENERIC_LOW_VALUE_TOPICS = {
#     "awkwardness",
#     "left",
#     "right",
#     "up",
#     "down",
#     "here",
#     "there",
#     "visibility",
#     "this_thing",
#     "question",
#     "questions",
#     "understanding",
#     "everyone",
#     "group",
#     "yes",
#     "no",
#     "okay",
# }

# LOW_VALUE_TOPIC_PATTERNS = [
#     r"\bawkward",
#     r"\bvisibility\b",
#     r"\bthis thing\b",
#     r"\bmove left\b",
#     r"\bmove right\b",
#     r"\bcontinue left\b",
#     r"\bcontinue right\b",
#     r"\bturn it on\b",
#     r"\bopen it up\b",
# ]

# LOW_VALUE_OBJECT_EXACT = {
#     "people",
#     "person",
#     "chair",
#     "stool",
#     "table",
#     "desk",
#     "face",
#     "head",
#     "hair",
#     "hand",
#     "hands",
#     "arm",
#     "arms",
#     "leg",
#     "legs",
#     "eye",
#     "eyes",
#     "mouth",
#     "wall",
#     "floor",
#     "door",
#     "window",
#     "room",
# }

# PLACE_SCENE_TERMS = {
#     "living room",
#     "kitchen",
#     "bedroom",
#     "bathroom",
#     "hallway",
#     "outdoors",
#     "outdoor",
#     "outside",
#     "street",
#     "parking lot",
#     "parking",
#     "office",
#     "workspace",
#     "office/workspace",
#     "dining area",
#     "dining room",
#     "restaurant",
#     "cafe",
#     "restaurant/cafe",
#     "store",
#     "shop",
#     "store/shop",
#     "car interior",
#     "car/interior",
#     "room",
#     "house",
#     "home",
# }

# STRONG_TOPIC_FAMILIES = {
#     "coordination_or_marking",
#     "timing_or_schedule",
#     "planning_or_assignment",
#     "identity_or_count",
#     "source_or_location",
# }

# WEAK_TOPIC_FAMILIES = {
#     "setup_or_dependency",
#     "availability_or_commitment",
#     "preference_or_habit",
#     "group_activity_or_coordination",
# }

# PRIORITY_OBJECTS = {
#     "phone",
#     "hard drive",
#     "tripod",
#     "cable",
#     "container",
#     "padding",
#     "laptop",
#     "tablet",
#     "charger",
#     "paper",
#     "whiteboard",
#     "bag",
#     "cart",
#     "refrigerator",
#     "cloth",
#     "flower",
#     "coin",
#     "seed paper",
# }

# RELATION_MIN_SUPPORT = {
#     "frequently_uses": 2,
#     "frequently_handles": 3,
#     "frequently_interacts_with": 2,
#     "related_to": 3,
# }

# RELATION_PRIOR = {
#     "frequently_uses": 0.62,
#     "frequently_handles": 0.54,
#     "frequently_interacts_with": 0.60,
#     "related_to": 0.50,
# }


# def canonicalize_text(x: str) -> str:
#     x = str(x).strip().lower()
#     x = re.sub(r"\s+", " ", x)
#     return x


# def normalize_day_str(date_val: Any) -> str:
#     if date_val is None:
#         return ""
#     s = str(date_val)
#     m = re.search(r"DAY(\d+)", s)
#     if m:
#         return f"DAY{m.group(1)}"
#     m = re.search(r"(\d+)", s)
#     if m:
#         return f"DAY{m.group(1)}"
#     return s


# def normalize_time_str(t: Any) -> str:
#     return str(t).zfill(8)


# def day_to_int(day_str: str) -> int:
#     m = re.search(r"(\d+)", str(day_str))
#     return int(m.group(1)) if m else 0


# def fact_id_from_key(key: Tuple[str, str, str, str, str]) -> str:
#     raw = "||".join(key)
#     return "sf_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# def looks_like_person_name(text: str) -> bool:
#     raw = str(text).strip()
#     if not raw:
#         return False
#     return bool(re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", raw))

# def is_place_or_scene_like(text: str) -> bool:
#     s = canonicalize_text(text)
#     if not s:
#         return False
#     if s in PLACE_SCENE_TERMS:
#         return True
#     if re.search(r"(living room|kitchen|bedroom|bathroom|hallway|outdoors?|outside|street|parking|office|workspace|dining area|dining room|restaurant|cafe|store|shop|room|house|home)", s):
#         return True
#     return False


# def is_low_value_topic(topic: str) -> bool:
#     t = canonicalize_text(topic)
#     if not t:
#         return True
#     if t in GENERIC_LOW_VALUE_TOPICS:
#         return True
#     if looks_like_person_name(topic):
#         return True
#     for pattern in LOW_VALUE_TOPIC_PATTERNS:
#         if re.search(pattern, t):
#             return True
#     return False


# def is_low_value_object(obj: str) -> bool:
#     o = canonicalize_text(obj)
#     if not o:
#         return True
#     if o in LOW_VALUE_OBJECT_EXACT:
#         return True
#     if looks_like_person_name(obj):
#         return True
#     if is_place_or_scene_like(obj) or is_place_or_scene_like(o):
#         return True
#     return False


# def relation_threshold(relation: str) -> int:
#     return RELATION_MIN_SUPPORT.get(relation, 2)


# def _sorted_unique(values: Iterable[str]) -> List[str]:
#     out = sorted({str(v).strip() for v in values if str(v).strip()})
#     return out


# def _candidate_roots(candidate: Dict[str, Any], unit: Dict[str, Any]) -> List[str]:
#     roots = candidate.get("provenance_root_ids") or unit.get("provenance_root_ids") or unit.get("source_doc_ids") or []
#     roots = _sorted_unique(roots)
#     if roots:
#         return roots
#     doc_id = str(unit.get("doc_id", "")).strip()
#     return [doc_id] if doc_id else []


# def compute_confidence(fact: Dict[str, Any]) -> float:
#     relation = fact["relation"]
#     support_count = int(fact.get("support_count", 1))
#     raw_support_count = int(fact.get("raw_support_count", support_count))
#     support_days_count = len(fact.get("support_days", []))
#     support_scales = set(fact.get("support_scales", []))
#     prior = RELATION_PRIOR.get(relation, 0.55)

#     confidence = prior
#     # slower growth: distinguish low/medium support, avoid rapid top-end saturation
#     confidence += 0.020 * min(support_count, 3)
#     confidence += 0.004 * min(max(support_count - 3, 0), 10)
#     confidence += 0.004 * min(max(raw_support_count - support_count, 0), 10)
#     confidence += 0.030 * max(0, support_days_count - 1)

#     if {"30s", "3min"}.issubset(support_scales):
#         confidence += 0.018
#     elif len(support_scales) >= 2:
#         confidence += 0.012

#     if len(fact.get("evidence_event_ids", [])) >= 5:
#         confidence += 0.010

#     if relation == "frequently_handles":
#         confidence -= 0.03
#     elif relation == "related_to":
#         confidence -= 0.06
#         if str(fact.get("tail", "")) in WEAK_TOPIC_FAMILIES:
#             confidence -= 0.04

#     confidence = max(0.0, min(0.88, confidence))
#     return round(confidence, 3)


# def compute_habit_strength(fact: Dict[str, Any]) -> str:
#     support_count = int(fact.get("support_count", 1))
#     support_days_count = len(fact.get("support_days", []))
#     if support_count >= 5 or support_days_count >= 3:
#         return "high"
#     if support_count >= 3:
#         return "medium"
#     return "low"


# def should_keep_candidate(candidate: Dict[str, Any]) -> bool:
#     relation = str(candidate.get("relation", "")).strip()
#     head = str(candidate.get("head", "")).strip()
#     tail = str(candidate.get("tail", "")).strip()
#     head_type = str(candidate.get("head_type", "")).strip()
#     tail_type = str(candidate.get("tail_type", "")).strip()

#     if not relation or not head or not tail:
#         return False
#     if head == tail:
#         return False

#     if relation == "related_to":
#         if tail_type != "Topic":
#             return False
#         if is_low_value_topic(tail):
#             return False
#         source_triplets = candidate.get("source_triplets", []) or []
#         metadata_support = int(candidate.get("metadata_support_count", 0))
#         if tail in WEAK_TOPIC_FAMILIES and not source_triplets and metadata_support < 2:
#             return False

#     if relation in {"frequently_uses", "frequently_handles"}:
#         if tail_type != "Object":
#             return False
#         if is_low_value_object(tail):
#             return False

#     if relation == "frequently_interacts_with":
#         if head_type != "Person" or tail_type != "Person":
#             return False

#     return True



# class SemanticConsolidation:
#     """
#     Deterministic semantic fact accumulator.

#     It consumes semantic candidates and accumulates them into stable semantic facts.
#     """

#     def __init__(self, llm_model=None, embedding_model=None):
#         self.llm_model = llm_model
#         self.embedding_model = embedding_model

#     def candidate_key(self, candidate: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
#         return (
#             str(candidate["head"]).strip(),
#             str(candidate["head_type"]).strip(),
#             str(candidate["relation"]).strip(),
#             str(candidate["tail"]).strip(),
#             str(candidate["tail_type"]).strip(),
#         )

#     def init_fact_from_candidate(self, candidate: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
#         key = self.candidate_key(candidate)
#         day = normalize_day_str(unit.get("date", ""))
#         scale = str(unit.get("scale", "")).strip()
#         doc_id = str(unit.get("doc_id", "")).strip()
#         source_triplets = candidate.get("source_triplets", []) or []
#         provenance_root_ids = _candidate_roots(candidate, unit)

#         fact = {
#             "fact_id": fact_id_from_key(key),
#             "head": key[0],
#             "head_type": key[1],
#             "relation": key[2],
#             "tail": key[3],
#             "tail_type": key[4],
#             "semantic_summary": str(candidate.get("semantic_summary", "")).strip(),

#             # unique bottom-event support after cross-scale de-dup
#             "support_count": len(provenance_root_ids),
#             # raw mentions across units/scales kept for diagnostics
#             "raw_support_count": 1,
#             "support_days": [day] if day else [],
#             "support_scales": [scale] if scale else [],
#             "evidence_event_ids": [doc_id] if doc_id else [],
#             "provenance_root_ids": provenance_root_ids,

#             "first_seen": doc_id,
#             "last_seen": doc_id,
#             "source_triplets": source_triplets[:8],
#             "confidence": 0.0,
#             "habit_strength": "low",
#         }
#         fact["confidence"] = compute_confidence(fact)
#         fact["habit_strength"] = compute_habit_strength(fact)
#         return fact

#     def update_fact_with_candidate(self, fact: Dict[str, Any], candidate: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
#         day = normalize_day_str(unit.get("date", ""))
#         scale = str(unit.get("scale", "")).strip()
#         doc_id = str(unit.get("doc_id", "")).strip()
#         provenance_root_ids = _candidate_roots(candidate, unit)

#         fact["raw_support_count"] = int(fact.get("raw_support_count", fact.get("support_count", 1))) + 1

#         if day and day not in fact["support_days"]:
#             fact["support_days"].append(day)
#             fact["support_days"].sort(key=day_to_int)
#         if scale and scale not in fact["support_scales"]:
#             fact["support_scales"].append(scale)
#         if doc_id and doc_id not in fact["evidence_event_ids"]:
#             fact["evidence_event_ids"].append(doc_id)
#         for rid in provenance_root_ids:
#             if rid not in fact["provenance_root_ids"]:
#                 fact["provenance_root_ids"].append(rid)
#         fact["support_count"] = len(fact["provenance_root_ids"])

#         if not fact.get("first_seen") and doc_id:
#             fact["first_seen"] = doc_id
#         if doc_id:
#             fact["last_seen"] = doc_id

#         for tri in candidate.get("source_triplets", []) or []:
#             if tri not in fact["source_triplets"]:
#                 fact["source_triplets"].append(tri)
#             if len(fact["source_triplets"]) >= 12:
#                 break

#         fact["confidence"] = compute_confidence(fact)
#         fact["habit_strength"] = compute_habit_strength(fact)
#         return fact

#     def update_fact_store(
#         self,
#         fact_store: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
#         unit: Dict[str, Any],
#     ) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
#         candidates = unit.get("semantic_candidates", []) or []
#         for candidate in candidates:
#             if not should_keep_candidate(candidate):
#                 continue
#             key = self.candidate_key(candidate)
#             if key not in fact_store:
#                 fact_store[key] = self.init_fact_from_candidate(candidate, unit)
#             else:
#                 fact_store[key] = self.update_fact_with_candidate(fact_store[key], candidate, unit)
#         return fact_store

#     def fact_is_mature(self, fact: Dict[str, Any]) -> bool:
#         relation = fact["relation"]
#         support_count = int(fact.get("support_count", 1))
#         raw_support_count = int(fact.get("raw_support_count", support_count))
#         threshold = relation_threshold(relation)
#         if support_count < threshold:
#             return False
#         if fact["tail_type"] == "Topic" and is_low_value_topic(fact["tail"]):
#             return False
#         if fact["tail_type"] == "Object" and is_low_value_object(fact["tail"]):
#             return False
#         if relation == "frequently_handles":
#             tail = canonicalize_text(fact["tail"])
#             if tail not in PRIORITY_OBJECTS and support_count < 4:
#                 return False
#         if relation == "related_to":
#             tail = str(fact.get("tail", ""))
#             if tail in WEAK_TOPIC_FAMILIES:
#                 if support_count < 4 and len(fact.get("support_days", [])) < 2:
#                     return False
#                 if raw_support_count < 4:
#                     return False
#             elif tail not in STRONG_TOPIC_FAMILIES:
#                 return False
#         return True


#     def materialize_snapshot(
#         self,
#         fact_store: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
#     ) -> List[Dict[str, Any]]:
#         facts = []
#         for _, fact in fact_store.items():
#             if not self.fact_is_mature(fact):
#                 continue
#             out = deepcopy(fact)
#             out["support_scales"] = sorted(out.get("support_scales", []))
#             out["evidence_event_ids"] = sorted(out.get("evidence_event_ids", []))
#             out["provenance_root_ids"] = sorted(out.get("provenance_root_ids", []))
#             out["support_days"] = sorted(out.get("support_days", []), key=day_to_int)
#             out["support_count"] = len(out.get("provenance_root_ids", [])) or int(out.get("support_count", 1))
#             out["confidence"] = compute_confidence(out)
#             out["habit_strength"] = compute_habit_strength(out)
#             facts.append(out)
#         facts.sort(
#             key=lambda x: (
#                 -float(x.get("confidence", 0.0)),
#                 -int(x.get("support_count", 0)),
#                 x.get("head", ""),
#                 x.get("relation", ""),
#                 x.get("tail", ""),
#             )
#         )
#         return facts

#     def save_results(self, results: Dict[str, Any], output_dir: str = ".", model_name: str = "rulebased"):
#         os.makedirs(output_dir, exist_ok=True)
#         out_path = os.path.join(output_dir, f"semantic_consolidation_results_{model_name}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Semantic consolidation results saved to: {out_path}")



import json
import os
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils import ConsolidationRawOutput
from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)

class SemanticConsolidation:
    def __init__(self, llm_model: LLMModel, embedding_model: EmbeddingModel):
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.6  # Threshold for finding relevant existing triples

    def find_relevant_triples(self, new_triples: List[List[str]], 
                                existing_triples: List[Tuple[List[str], List[str]]], 
                                top_k: int = 20) -> List[List[Tuple[List[str], List[str]]]]:
        """
        Find existing triples that are relevant to a new triple using semantic similarity.
        
        Args:
            new_triples: List of new semantic triples [subject, predicate, object]
            existing_triples: List of tuples (triple, evidence_list) where evidence is in "{timestamp}_{idx}" format
            top_k: Maximum number of relevant triples to return
            
        Returns:
            List of List of tuples (existing_triple, evidence_list) for relevant matches
        """
        if not existing_triples:
            return [[] for _ in new_triples]
        
        # Convert triples to text for embedding
        new_triple_texts = [" ".join(triple) for triple in new_triples]
        existing_triple_texts = [" ".join(triple) for triple, _ in existing_triples]
        
        # Get embeddings using text modality
        new_embeddings = self.embedding_model.encode(new_triple_texts, modality="text")
        existing_embeddings = self.embedding_model.encode(existing_triple_texts, modality="text")
        
        # Compute cosine similarity matrix (new_triples x existing_triples)
        similarities = cosine_similarity(new_embeddings, existing_embeddings)
        
        # Get top-k indices for each new triple (sorted by similarity descending)
        top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        
        # Return relevant triples for each new triple
        return [
            [existing_triples[idx] for idx in indices if similarities[i, idx] >= self.similarity_threshold]
            for i, indices in enumerate(top_k_indices)
        ]

    def consolidate_triple(self, new_triple: List[str], 
                            relevant_existing_triples: List[List[str]]) -> Tuple[List[str], List[int]]:
        """
        Consolidate a new semantic triple with relevant existing ones.
        
        Args:
            new_triple (List[str]): A single new semantic triple [subject, predicate, object]
            relevant_existing_triples (List[List[str]]): List of relevant existing triples
            
        Returns:
            Tuple of (updated_new_triple, triple_indices_to_remove)
        """
        if not relevant_existing_triples:
            return new_triple, []

        formatted_existing_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(relevant_existing_triples))
        messages = self.prompt_template_manager.render(
            name='semantic_consolidation',
            new_triple=new_triple,
            existing_triples=formatted_existing_triples
        )

        try:
            # LLM INFERENCE
            # Ensure messages is a list for chat-based templates
            if isinstance(messages, str):
                raise ValueError("Expected chat template to return List[Dict], got string")
            response = self.llm_model.generate(messages, text_format=ConsolidationRawOutput)

        except Exception as e:
            logger.warning(e)
            return new_triple, []
        
        return response.updated_triple, response.triples_to_remove

    def batch_semantic_consolidation(self, existing_semantic_results: Tuple[List[List[str]], List[List[str]]], 
                                      new_semantic_results: Tuple[List[List[str]], List[List[str]]]) -> Tuple[List[List[str]], List[List[str]], List[Tuple[List[str], List[str]]]]:
        """
        Conduct semantic consolidation for a single timestamp against existing results.
        
        Args:
            existing_semantic_results: Tuple of (existing_semantic_triples, existing_episodic_evidence) 
                                     where existing_episodic_evidence is already in "{timestamp}_{idx}" format
            new_semantic_results: Tuple of (new_semantic_triples, new_episodic_evidence)
                                where new_episodic_evidence is also in "{timestamp}_{idx}" format
            
        Returns:
            Tuple of (consolidated_semantic_triples, consolidated_episodic_evidence, triples_to_remove)
            where triples_to_remove is a list of (triple, evidence) tuples that should be removed from accumulated state
        """
        existing_semantic_triples, existing_episodic_evidence = existing_semantic_results
        new_semantic_triples, new_episodic_evidence = new_semantic_results
        
        if not new_semantic_triples:
            # No new triples to consolidate, return empty results
            return [], [], []
        
        # Create accumulated triples structure from existing results
        # Each tuple is (triple, evidence_list)
        accumulated_triples: List[Tuple[List[str], List[str]]] = []
        for triple, evidence in zip(existing_semantic_triples, existing_episodic_evidence):
            accumulated_triples.append((triple, evidence))
        
        # Process all new triples concurrently
        consolidated_results = self._process_timestamp_triples_concurrent(
            new_semantic_triples, new_episodic_evidence, accumulated_triples
        )
        
        # Extract results
        consolidated_triples = [result["updated_triple"] for result in consolidated_results]
        consolidated_evidence = [result["merged_evidence"] for result in consolidated_results]
        
        # Collect all triples to be removed across all consolidations
        all_triples_to_remove: List[Tuple[List[str], List[str]]] = []
        for result in consolidated_results:
            all_triples_to_remove.extend(result["triples_to_remove"])
        
        return consolidated_triples, consolidated_evidence, all_triples_to_remove

    def _process_timestamp_triples_concurrent(self, current_triples: List[List[str]], 
                                            current_evidence: List[List[str]],
                                            accumulated_triples: List[Tuple[List[str], List[str]]]) -> List[Dict[str, Any]]:
        """
        Process all triples at a timestamp concurrently.
        
        Args:
            current_triples: New semantic triples for this timestamp
            current_evidence: Evidence in "{timestamp}_{idx}" format
            accumulated_triples: Existing triples from previous timestamps
            
        Returns:
            List of consolidation results with updated triples and merged evidence
        """
        # Find relevant triples for ALL new triples at once (optimization)
        all_relevant_existing_data = self.find_relevant_triples(current_triples, accumulated_triples)
        
        def process_single_triple(triple_idx: int) -> Dict[str, Any]:
            new_triple = current_triples[triple_idx]
            new_evidence = current_evidence[triple_idx]
            
            # Get precomputed relevant triples for this specific triple
            relevant_existing = all_relevant_existing_data[triple_idx]
            
            if not relevant_existing:
                # No relevant existing triples, return as-is
                return {
                    "updated_triple": new_triple,
                    "triples_to_remove": [],
                    "merged_evidence": new_evidence,
                    "triple_idx": triple_idx
                }
            
            relevant_triples_only = [triple for triple, _ in relevant_existing]
            
            # Consolidate with LLM
            updated_triple, indices_to_remove = self.consolidate_triple(new_triple, relevant_triples_only)
            
            # Merge evidence from removed triples
            merged_evidence = new_evidence.copy()
            triples_to_remove_data = []
            
            for remove_idx in indices_to_remove:
                if remove_idx < len(relevant_existing):
                    removed_triple, removed_evidence = relevant_existing[remove_idx]
                    merged_evidence.extend(removed_evidence)
                    triples_to_remove_data.append((removed_triple, removed_evidence))
            
            return {
                "updated_triple": updated_triple,
                "triples_to_remove": triples_to_remove_data,
                "merged_evidence": merged_evidence,
                "triple_idx": triple_idx
            }
        
        # Process all triples concurrently
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_triple, i): i for i in range(len(current_triples))}
            pbar = tqdm(as_completed(futures), total=len(futures), 
                       desc=f"Consolidating triples", leave=False)
            
            for future in pbar:
                result = future.result()
                results.append(result)
        
        # Sort results by original triple index to maintain order
        results.sort(key=lambda x: x["triple_idx"])
        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "."):
        """
        Save consolidation results to a JSON file.
        
        Args:
            results: The results dictionary to save
            output_dir: Output directory path.
        """

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"semantic_consolidation_results_{self.llm_model.model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)