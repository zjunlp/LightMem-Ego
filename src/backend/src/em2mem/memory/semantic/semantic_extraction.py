# import json
# import os
# from typing import Dict, Any, List, Tuple
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import logging

# from .utils import SemanticRawOutput, SemanticOutput
# from ...llm import LLMModel, PromptTemplateManager

# logger = logging.getLogger(__name__)

# class SemanticExtraction:
#     def __init__(self, llm_model: LLMModel):
#         self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
#         self.llm_model = llm_model

#     def semantic_extraction(self, chunk_key: str, episodic_triples: List[List[str]]) -> SemanticOutput:
#         # PREPROCESSING
#         formatted_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(episodic_triples))
#         messages = self.prompt_template_manager.render(name='semantic_extraction', episodic_triples=formatted_triples)

#         try:
#             # LLM INFERENCE (entire try-block is retried by the decorator)
#             response = self.llm_model.generate(messages, text_format=SemanticRawOutput)

#         except Exception as e:
#             logger.warning(e)
#             return SemanticOutput(
#                 chunk_id=chunk_key,
#                 semantic_triples=[],
#                 episodic_evidence=[]
#             )

#         return SemanticOutput(
#             chunk_id=chunk_key,
#             semantic_triples=response.semantic_triples,
#             episodic_evidence=response.episodic_evidence
#         )
    
#     def save_results(self, results: Dict[str, Any], output_dir: str = "."):
#         """
#         Save extraction results to a JSON file.
        
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
#         with open(os.path.join(output_dir, f"semantic_extraction_results_{self.llm_model.model_name}.json"), 'w', encoding='utf-8') as f:
#             json.dump(json_results, f, indent=2, ensure_ascii=False)

#     def batch_semantic_extraction(self, episodic_triples_batch: Dict[str, List[List[str]]], output_dir: str = ".") -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]:
#         """
#         Conduct batch semantic extraction synchronously using multi-threading.

#         Args:
#             episodic_triples_batch: A dictionary mapping chunk IDs to lists of episodic triples.
#             output_dir (str): Directory to save output file.

#         Returns:
#             Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]
#                 - A dict with keys as the chunk ids (mdhash) and values as the semantic triples
#                 - A dict with keys as the chunk ids (mdhash) and values as the episodic evidence indices
#         """
#         results = []
#         with ThreadPoolExecutor() as executor:
#             futures = {
#                 executor.submit(self.semantic_extraction, chunk_key, episodic_triples): episodic_triples
#                 for chunk_key, episodic_triples in episodic_triples_batch.items()
#             }
#             pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting semantic triples")
#             for future in pbar:
#                 result = future.result()
#                 results.append(result)

#         semantic_triples_map = {res.chunk_id: res.semantic_triples for res in results}
#         episodic_evidence_map = {res.chunk_id: res.episodic_evidence for res in results}

#         chunk_keys = list(episodic_triples_batch.keys())

#         ordered_semantic_triples = {key: semantic_triples_map.get(key, []) for key in chunk_keys}
#         ordered_episodic_evidence = {key: episodic_evidence_map.get(key, []) for key in chunk_keys}

#         combined_results = {
#             "semantic_triples": ordered_semantic_triples,
#             "episodic_evidence": ordered_episodic_evidence
#         }
#         self.save_results(combined_results, output_dir)

#         return ordered_semantic_triples, ordered_episodic_evidence


#!/usr/bin/env python3
# """
# Rule-based semantic candidate extraction from episodic triplets.

# This version is designed to produce fewer, more stable semantic candidates
# that can accumulate support across events and scales.

# Key design choices:
# 1. topic family merge:
#    - hard_drive_identity
#    - hard_drive_count
#    - meeting_plan
#    - timestamp_marking
#    - stopwatch_need
#    - power_connection
#    - hard_drive_connection
# 2. person-person interactions are treated as undirected pairs
# 3. object whitelist + object-class-aware relation folding
# 4. more conservative candidate generation, especially for 3min+
# 5. remove overly broad installation/setup merging
# """

# import json
# import os
# import re
# import logging
# from typing import Dict, Any, List, Optional, Tuple

# logger = logging.getLogger(__name__)


# # =========================================================
# # Basic normalization helpers
# # =========================================================

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
#     "mark_it",
#     "yes",
#     "no",
#     "okay",
#     "dessert_setup",
#     "second_life_setup",
#     "prior_installation_experience",
# }

# LOW_VALUE_TOPIC_PATTERNS = [
#     r"\bawkward",
#     r"\bvisibility\b",
#     r"\bleft\b",
#     r"\bright\b",
#     r"\bthis thing\b",
#     r"\bplastic bag\b",
#     r"\blining\b",
#     r"\bunderstanding\b",
#     r"\bhole\b",
#     r"\bnotch\b",
#     r"\bdessert\b",
#     r"\bsecond life\b",
#     r"\bprior installation experience\b",
# ]

# LOW_VALUE_OBJECTS = {
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
#     "box",
#     "bag",
# }

# PERSON_LIKE_STOPWORDS = {
#     "group",
#     "everyone",
#     "people",
# }

# OBJECT_CANONICAL_MAP: List[Tuple[str, str]] = [
#     (r"\bbox liner\b|\bbox lining\b|\bliner\b|\blining\b", "padding"),
#     (r"\bcell ?phone|smartphone|mobile phone\b", "phone"),
#     (r"\bphone\b", "phone"),
#     (r"\bhard drives?\b|\bexternal drives?\b|\bdrive\b", "hard drive"),
#     (r"\btripods?\b", "tripod"),
#     (r"\bcables?\b|\bdata cable\b|\busb cable\b|\bcharging cable\b", "cable"),
#     (r"\bpaddings?\b|\bfoam\b", "padding"),
#     (r"\bpapers?\b|\bdocuments?\b|\bnotes?\b", "paper"),
#     (r"\bcontainers?\b|\bboxes?\b|\bbin\b", "container"),
#     (r"\bwhiteboard\b|\bboard\b", "whiteboard"),
#     (r"\blaptop\b|\bcomputer\b", "laptop"),
#     (r"\btablet\b|\bipad\b", "tablet"),
#     (r"\bcharger\b", "charger"),
#     (r"\bbag\b|\bbackpack\b", "bag"),
#     (r"\bplastic bag\b", "bag"),
#     (r"\bcart\b|\bshopping cart\b", "cart"),
#     (r"\bfridge\b|\brefrigerator\b", "refrigerator"),
#     (r"\btable\b|\bdesk\b", "table"),
#     (r"\bchair\b|\bchairs\b", "chair"),
#     (r"\bstool\b|\bstools\b", "stool"),
# ]

# OBJECT_WHITELIST = {
#     "phone",
#     "tripod",
#     "hard drive",
#     "cable",
#     "container",
#     "padding",
#     "laptop",
#     "tablet",
#     "charger",
#     "paper",
#     "whiteboard",
# }

# DEVICE_OBJECTS = {
#     "phone",
#     "laptop",
#     "tablet",
#     "charger",
# }

# EQUIPMENT_OBJECTS = {
#     "tripod",
#     "hard drive",
#     "cable",
#     "container",
#     "padding",
#     "paper",
#     "whiteboard",
# }

# STABLE_TOPIC_FAMILIES = {
#     "timestamp_marking",
#     "stopwatch_need",
#     "hard_drive_identity",
#     "hard_drive_count",
#     "meeting_plan",
#     "power_connection",
#     "hard_drive_connection",
# }

# TOPIC_FAMILY_PATTERNS: List[Tuple[List[str], str]] = [
#     # timestamp marking
#     ([r"timestamp", r"mark"], "timestamp_marking"),
#     ([r"mark a timestamp"], "timestamp_marking"),
#     ([r"timestamp marking"], "timestamp_marking"),

#     # stopwatch need
#     ([r"stopwatch", r"need"], "stopwatch_need"),

#     # hard drive identity / naming
#     ([r"hard drive", r"name"], "hard_drive_identity"),
#     ([r"device was hard drive"], "hard_drive_identity"),
#     ([r"which hard drive"], "hard_drive_identity"),
#     ([r"correct hard drive"], "hard_drive_identity"),
#     ([r"hard drive identity"], "hard_drive_identity"),

#     # hard drive count
#     ([r"there were four hard drives"], "hard_drive_count"),
#     ([r"how many hard drives"], "hard_drive_count"),
#     ([r"number of drives"], "hard_drive_count"),
#     ([r"hard drive count"], "hard_drive_count"),

#     # meeting / plan
#     ([r"continue meetings"], "meeting_plan"),
#     ([r"plans for last day"], "meeting_plan"),
#     ([r"meeting discussion"], "meeting_plan"),
#     ([r"meeting plan"], "meeting_plan"),
#     ([r"last day plan"], "meeting_plan"),
# ]


# def canonicalize_text(x: str) -> str:
#     x = str(x).strip().lower()
#     x = re.sub(r"\s+", " ", x)
#     return x


# def title_case_name(x: str) -> str:
#     x = canonicalize_text(x)
#     if not x:
#         return ""
#     return " ".join(p.capitalize() for p in x.split())


# def singularize_basic(x: str) -> str:
#     if x.endswith("ies") and len(x) > 4:
#         return x[:-3] + "y"
#     if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
#         return x[:-1]
#     return x


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


# def canonicalize_object(obj: str) -> str:
#     s = canonicalize_text(obj)
#     if not s:
#         return ""
#     for pattern, target in OBJECT_CANONICAL_MAP:
#         if re.search(pattern, s):
#             return target
#     s = singularize_basic(s)
#     s = re.sub(r"\bsmall\b|\blarge\b|\bblack\b|\bwhite\b|\bred\b|\bblue\b|\bgreen\b", "", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s


# def is_good_object(obj: str) -> bool:
#     o = canonicalize_object(obj)
#     if not o:
#         return False
#     if o in LOW_VALUE_OBJECTS:
#         return False
#     if o in OBJECT_WHITELIST:
#         return True
#     return False


# def canonicalize_topic_family(topic: str) -> str:
#     """
#     Map raw topic-like strings into a small set of canonical topic families.
#     Returns "" if the topic should not be promoted.
#     """
#     s = canonicalize_text(topic)
#     if not s:
#         return ""

#     s = s.replace('"', "").replace("'", "")
#     s = re.sub(r"\b(a|an|the)\b", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()

#     if is_low_value_topic(s):
#         return ""

#     # -------------------------------------------------
#     # Explicit reject zone for overly broad / noisy setup topics
#     # -------------------------------------------------
#     if re.search(r"\bdessert\b", s):
#         return ""
#     if re.search(r"\bsecond life\b", s):
#         return ""
#     if re.search(r"\bprior installation experience\b", s):
#         return ""

#     # -------------------------------------------------
#     # Narrow family mapping for connection / setup-like topics
#     # -------------------------------------------------
#     # power-related connection
#     if re.search(r"\bpower bank\b|\bbattery pack\b|\bcharger\b", s) and re.search(r"\bplugged\b|\bplug in\b|\bconnected\b|\bconnect\b", s):
#         return "power_connection"

#     # hard-drive-specific connection
#     if re.search(r"\bhard drive\b|\bdrive\b", s) and re.search(r"\bconnect\b|\bconnected\b|\bplug\b|\bplugged\b|\bcable\b|\bport\b", s):
#         return "hard_drive_connection"

#     # -------------------------------------------------
#     # Stable topic family patterns
#     # -------------------------------------------------
#     for patterns, family in TOPIC_FAMILY_PATTERNS:
#         matched = True
#         for p in patterns:
#             if not re.search(p, s):
#                 matched = False
#                 break
#         if matched:
#             return family

#     # strongly suppress one-off open-text topics
#     return ""


# def looks_like_person(name: str, wearer_name: str) -> bool:
#     raw = str(name).strip()
#     if not raw:
#         return False

#     if canonicalize_text(raw) in PERSON_LIKE_STOPWORDS:
#         return False

#     if raw == wearer_name:
#         return True

#     if re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", raw):
#         return True

#     return False


# def normalize_person(name: str, wearer_name: str) -> str:
#     raw = str(name).strip()
#     if not raw:
#         return ""
#     if raw.lower() in {"i", "me", "my", "myself"}:
#         return wearer_name
#     return raw if looks_like_person(raw, wearer_name) else ""


# def relation_summary(head: str, relation: str, tail: str) -> str:
#     if relation == "frequently_interacts_with":
#         return f"{head} frequently interacts with {tail} across multiple events."
#     if relation == "frequently_uses":
#         return f"{head} frequently uses {tail} across multiple events."
#     if relation == "frequently_handles":
#         return f"{head} frequently handles {tail} across multiple events."
#     if relation == "related_to":
#         return f"{head} is repeatedly related to {tail} across multiple events."
#     return f"{head} {relation} {tail} across multiple events."


# # =========================================================
# # Semantic candidate extraction
# # =========================================================

# class SemanticExtraction:
#     """
#     Rule-based semantic candidate extraction.

#     Key upgrades:
#     - topic family merge
#     - undirected person-person interaction
#     - object whitelist
#     - object-class-aware relation folding
#     - conservative 3min+ candidate generation
#     - remove overly broad installation/setup merging
#     """

#     def __init__(self, llm_model=None, wearer_name: str = "Jake", model_name: Optional[str] = None):
#         self.llm_model = llm_model
#         self.wearer_name = title_case_name(wearer_name)
#         self.model_name = model_name or getattr(llm_model, "model_name", "rulebased")

#     def _make_candidate(
#         self,
#         head: str,
#         head_type: str,
#         relation: str,
#         tail: str,
#         tail_type: str,
#     ) -> Dict[str, Any]:
#         tail_family = tail if tail_type == "Topic" else tail
#         semantic_key = f"{head}|{head_type}|{relation}|{tail_family}|{tail_type}"
#         return {
#             "head": head,
#             "head_type": head_type,
#             "relation": relation,
#             "tail": tail,
#             "tail_type": tail_type,
#             "tail_family": tail_family,
#             "semantic_key": semantic_key,
#             "semantic_summary": relation_summary(head, relation, tail_family),
#         }

#     def _promote_person_person(
#         self,
#         h: str,
#         r: str,
#         t: str,
#     ) -> Optional[Dict[str, Any]]:
#         if r != "hand_to":
#             return None

#         p1 = normalize_person(h, self.wearer_name)
#         p2 = normalize_person(t, self.wearer_name)
#         if not p1 or not p2 or p1 == p2:
#             return None

#         # undirected pair canonicalization
#         head, tail = sorted([p1, p2])
#         return self._make_candidate(
#             head=head,
#             head_type="Person",
#             relation="frequently_interacts_with",
#             tail=tail,
#             tail_type="Person",
#         )

#     def _promote_person_object(
#         self,
#         h: str,
#         r: str,
#         t: str,
#     ) -> Optional[Dict[str, Any]]:
#         person = normalize_person(h, self.wearer_name)
#         if not person:
#             return None

#         obj = canonicalize_object(t)
#         if not is_good_object(obj):
#             return None

#         # object-class-aware folding
#         if obj in DEVICE_OBJECTS:
#             relation = "frequently_uses"
#         elif obj in EQUIPMENT_OBJECTS:
#             relation = "frequently_handles"
#         else:
#             if r == "use":
#                 relation = "frequently_uses"
#             else:
#                 relation = "frequently_handles"

#         return self._make_candidate(
#             head=person,
#             head_type="Person",
#             relation=relation,
#             tail=obj,
#             tail_type="Object",
#         )

#     def _promote_person_topic(
#         self,
#         h: str,
#         r: str,
#         t: str,
#     ) -> Optional[Dict[str, Any]]:
#         if r not in {"ask_about", "confirm", "say_about", "discuss"}:
#             return None

#         person = normalize_person(h, self.wearer_name)
#         if not person:
#             return None

#         topic_family = canonicalize_topic_family(t)
#         if not topic_family:
#             return None

#         return self._make_candidate(
#             head=person,
#             head_type="Person",
#             relation="related_to",
#             tail=topic_family,
#             tail_type="Topic",
#         )

#     def _promote_triplet(
#         self,
#         triplet: List[str],
#         scale: str,
#     ) -> Optional[Dict[str, Any]]:
#         """
#         Promote one episodic triplet into one semantic candidate.
#         Returns None if the triplet should not contribute.
#         """
#         if not isinstance(triplet, list) or len(triplet) != 3:
#             return None

#         h, r, t = [str(x).strip() for x in triplet]
#         if not h or not r or not t:
#             return None

#         # 1) person-person interaction
#         cand = self._promote_person_person(h, r, t)
#         if cand is not None:
#             return cand

#         # 2) person-object
#         if r in {"use", "hold", "inspect", "organize", "place_on", "take_from"}:
#             cand = self._promote_person_object(h, r, t)
#             if cand is not None:
#                 return cand

#         # 3) person-topic
#         cand = self._promote_person_topic(h, r, t)
#         if cand is not None:
#             return cand

#         return None

#     def _allow_candidate_for_scale(self, candidate: Dict[str, Any], scale: str) -> bool:
#         """
#         Be more conservative for 3min/10min/1h units:
#         only allow candidates already mapped into stable families / stable objects.
#         """
#         scale = str(scale).strip().lower()

#         if scale in {"30s", "30sec"}:
#             return True

#         relation = candidate["relation"]
#         tail_type = candidate["tail_type"]
#         tail = candidate["tail"]

#         if relation == "frequently_interacts_with":
#             return True

#         if tail_type == "Object" and tail in OBJECT_WHITELIST:
#             return True

#         if tail_type == "Topic" and tail in STABLE_TOPIC_FAMILIES:
#             return True

#         return False

#     def semantic_extraction(self, unit: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Extract semantic candidates for a single episodic unit.
#         """
#         doc_id = str(unit.get("doc_id", "")).strip()
#         date = str(unit.get("date", "")).strip()
#         start_time = str(unit.get("start_time", "")).strip()
#         end_time = str(unit.get("end_time", "")).strip()
#         scale = str(unit.get("scale", "")).strip() or str(unit.get("level", "")).strip() or "unknown"

#         episodic_triples = unit.get("episodic_triplets", []) or []

#         # dedup within unit by semantic_key
#         bucket: Dict[str, Dict[str, Any]] = {}

#         for idx, tri in enumerate(episodic_triples):
#             candidate = self._promote_triplet(tri, scale)
#             if candidate is None:
#                 continue

#             if not self._allow_candidate_for_scale(candidate, scale):
#                 continue

#             key = candidate["semantic_key"]

#             if key not in bucket:
#                 bucket[key] = {
#                     **candidate,
#                     "source_doc_ids": [doc_id] if doc_id else [],
#                     "source_scales": [scale] if scale else [],
#                     "support_triplet_indices": [idx],
#                     "source_triplets": [tri],
#                 }
#             else:
#                 bucket[key]["support_triplet_indices"].append(idx)
#                 bucket[key]["source_triplets"].append(tri)
#                 if doc_id and doc_id not in bucket[key]["source_doc_ids"]:
#                     bucket[key]["source_doc_ids"].append(doc_id)
#                 if scale and scale not in bucket[key]["source_scales"]:
#                     bucket[key]["source_scales"].append(scale)

#         semantic_candidates = list(bucket.values())
#         semantic_candidates.sort(
#             key=lambda x: (
#                 x["head_type"],
#                 x["head"],
#                 x["relation"],
#                 x["tail_family"],
#                 x["tail_type"],
#             )
#         )

#         return {
#             "doc_id": doc_id,
#             "date": date,
#             "start_time": start_time,
#             "end_time": end_time,
#             "scale": scale,
#             "semantic_candidates": semantic_candidates,
#         }

#     def save_results(self, results: Dict[str, Any], output_dir: str = "."):
#         os.makedirs(output_dir, exist_ok=True)
#         out_path = os.path.join(output_dir, f"semantic_candidate_results_{self.model_name}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Semantic candidate results saved to: {out_path}")

#     def batch_semantic_extraction(
#         self,
#         episodic_units: List[Dict[str, Any]],
#         output_dir: str = ".",
#     ) -> Dict[str, Any]:
#         """
#         Process episodic units in order and produce semantic candidates.

#         Returns:
#             {
#               "units": [...]
#             }
#         """
#         results = []
#         for unit in episodic_units:
#             results.append(self.semantic_extraction(unit))

#         combined_results = {
#             "units": results
#         }
#         self.save_results(combined_results, output_dir)
#         return combined_results

# """
# Rule-based semantic candidate extraction from episodic triplets.

# Goals of this revision:
# 1. be case-free and less tied to the first-two-sync subset
# 2. preserve compatibility with the old output schema
# 3. use richer unit metadata when available
# 4. carry cross-scale provenance roots for later de-duplication
# 5. support optional concurrent batch extraction with progress logging
# """

# import json
# import os
# import re
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Dict, Any, List, Optional, Tuple, Iterable

# try:
#     from tqdm.auto import tqdm
# except Exception:  # pragma: no cover
#     def tqdm(x: Iterable, *args, **kwargs):
#         return x

# logger = logging.getLogger(__name__)


# # =========================================================
# # Basic normalization helpers
# # =========================================================

# GENERIC_LOW_VALUE_TOPICS = {
#     "awkwardness",
#     "left",
#     "right",
#     "up",
#     "down",
#     "here",
#     "there",
#     "visibility",
#     "question",
#     "questions",
#     "understanding",
#     "everyone",
#     "group",
#     "this_thing",
#     "move_left",
#     "move_right",
#     "continue_left",
#     "continue_right",
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

# LOW_VALUE_OBJECTS = {
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

# PERSON_LIKE_STOPWORDS = {
#     "group",
#     "everyone",
#     "people",
#     "team",
#     "someone",
#     "somebody",
#     "person",
# }

# STOPWORDS = {
#     "a", "an", "the", "this", "that", "these", "those", "it", "its", "their", "his", "her",
#     "my", "our", "your", "some", "any", "more", "other", "another", "same",
#     "to", "of", "in", "on", "at", "for", "with", "from", "into", "onto", "over", "under",
#     "and", "or", "but", "if", "then", "so", "as", "than",
# }

# OBJECT_CANONICAL_MAP: List[Tuple[str, str]] = [
#     (r"\bcell ?phone|smartphone|mobile phone\b", "phone"),
#     (r"\bphone\b", "phone"),
#     (r"\bhard drives?\b|\bexternal drives?\b|\bdrive\b", "hard drive"),
#     (r"\btripods?\b", "tripod"),
#     (r"\bcables?\b|\bdata cable\b|\busb cable\b|\bcharging cable\b|\bpower cable\b", "cable"),
#     (r"\bbox liner\b|\bbox lining\b|\bliner\b|\blining\b|\bfoam\b|\bpadding\b", "padding"),
#     (r"\bpapers?\b|\bdocuments?\b|\bnotes?\b", "paper"),
#     (r"\bcontainers?\b|\bboxes?\b|\bbins?\b", "container"),
#     (r"\bwhiteboard\b|\bboard\b", "whiteboard"),
#     (r"\blaptop\b|\bcomputer\b", "laptop"),
#     (r"\btablet\b|\bipad\b", "tablet"),
#     (r"\bcharger\b|\bpower bank\b|\bbattery pack\b", "charger"),
#     (r"\bbags?\b|\bbackpacks?\b", "bag"),
#     (r"\bcarts?\b|\bshopping cart\b", "cart"),
#     (r"\bfridge\b|\brefrigerator\b", "refrigerator"),
#     (r"\blens cloth\b", "cloth"),
#     (r"\bflowers?\b", "flower"),
#     (r"\bcoins?\b|\bcurrency\b", "coin"),
#     (r"\bseed paper\b", "seed paper"),
# ]

# PRIORITY_OBJECTS = {
#     "phone",
#     "tripod",
#     "hard drive",
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

# DEVICE_OBJECTS = {"phone", "laptop", "tablet", "charger"}
# EQUIPMENT_OBJECTS = {"tripod", "hard drive", "cable", "container", "padding", "whiteboard", "cart", "refrigerator"}
# MATERIAL_OBJECTS = {"paper", "bag", "cloth", "flower", "coin", "seed paper"}

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

# STABLE_TOPIC_FAMILIES = {
#     "coordination_or_marking",
#     "timing_or_schedule",
#     "planning_or_assignment",
#     "identity_or_count",
#     "source_or_location",
#     "setup_or_dependency",
#     "availability_or_commitment",
#     "preference_or_habit",
#     "group_activity_or_coordination",
# }


# def canonicalize_text(x: str) -> str:
#     x = str(x).strip().lower()
#     x = re.sub(r"\s+", " ", x)
#     return x


# def title_case_name(x: str) -> str:
#     x = canonicalize_text(x)
#     if not x:
#         return ""
#     return " ".join(p.capitalize() for p in x.split())


# def singularize_basic(x: str) -> str:
#     if x.endswith("glasses"):
#         return "glasses"
#     if x.endswith("ies") and len(x) > 4:
#         return x[:-3] + "y"
#     if x.endswith("ses") and len(x) > 4 and not x.endswith("sses"):
#         return x[:-2]
#     if x.endswith("s") and len(x) > 3 and not x.endswith("ss"):
#         return x[:-1]
#     return x


# def tokenize(text: str) -> List[str]:
#     toks = re.findall(r"[a-zA-Z0-9_/-]+", canonicalize_text(text))
#     return [t for t in toks if t and t not in STOPWORDS]


# def looks_like_time_or_count_phrase(text: str) -> bool:
#     s = canonicalize_text(text)
#     if not s:
#         return True
#     if re.fullmatch(r"\d+[\d\s:/-]*", s):
#         return True
#     if re.search(r"\b\d+\s*(am|pm|hours?|minutes?|days?)\b", s):
#         return True
#     return False


# def looks_like_person_name(text: str) -> bool:
#     raw = str(text).strip()
#     if not raw:
#         return False
#     if canonicalize_text(raw) in PERSON_LIKE_STOPWORDS:
#         return False
#     if re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)?", raw):
#         return True
#     return False

# def is_place_or_scene_like(text: str) -> bool:
#     s = canonicalize_text(text)
#     if not s:
#         return False
#     if s in PLACE_SCENE_TERMS:
#         return True
#     if re.search(r"(living room|kitchen|bedroom|bathroom|hallway|outdoors?|outside|street|parking|office|workspace|dining area|dining room|restaurant|cafe|store|shop|room|house|home)", s):
#         return True
#     if re.search(r"scene|area|interior", s) and len(tokenize(s)) <= 3:
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


# def canonicalize_object(obj: str) -> str:
#     s = canonicalize_text(obj)
#     if not s:
#         return ""
#     for pattern, target in OBJECT_CANONICAL_MAP:
#         if re.search(pattern, s):
#             return target
#     s = singularize_basic(s)
#     s = re.sub(r"\bsmall\b|\blarge\b|\bblack\b|\bwhite\b|\bred\b|\bblue\b|\bgreen\b|\bwooden\b|\bcheckered\b", " ", s)
#     s = re.sub(r"\bheld\b|\bbeing held\b", " ", s)
#     s = re.sub(r"\s+", " ", s).strip(" _-")
#     return s


# def is_good_object(obj: str) -> bool:
#     o = canonicalize_object(obj)
#     if not o:
#         return False
#     if o in LOW_VALUE_OBJECTS:
#         return False
#     if looks_like_person_name(obj):
#         return False
#     if looks_like_time_or_count_phrase(obj):
#         return False
#     if is_place_or_scene_like(obj) or is_place_or_scene_like(o):
#         return False
#     if o in PRIORITY_OBJECTS:
#         return True
#     toks = tokenize(o)
#     if len(toks) >= 2:
#         if is_place_or_scene_like(" ".join(toks)):
#             return False
#         return True
#     return len(toks) == 1 and len(toks[0]) >= 4 and not is_place_or_scene_like(toks[0])


# def canonicalize_topic_family(topic: str) -> str:
#     """
#     Map raw topic-like strings into a small set of canonical topic families.
#     Returns "" if the topic should not be promoted.
#     """
#     s = canonicalize_text(topic)
#     if not s:
#         return ""

#     s = s.replace('"', "").replace("'", "")
#     s = re.sub(r"(a|an|the)", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()

#     if is_low_value_topic(s):
#         return ""

#     if re.search(r"timestamp|mark|stopwatch|marking", s):
#         return "coordination_or_marking"

#     if re.search(r"date|start time|schedule|weather|rain|three hours|last day", s):
#         return "timing_or_schedule"

#     if re.search(r"plan|plans|planning|meeting plan|assign|assigned|who will|need to bring|we should|we need to|i need to|lead|help run|auction|invite", s):
#         return "planning_or_assignment"

#     if re.search(r"what'?s this called|what is this called|identify|called|how many|only \d+|count|which one|correct", s):
#         return "identity_or_count"

#     if re.search(r"before|came from|source|where .* before|take from|bring from|upstairs|downstairs|fridge|location", s):
#         return "source_or_location"

#     if re.search(r"install|setup|open .* first|backup|connect|connected|plug|cable|power bank|charger|port", s):
#         return "setup_or_dependency"

#     if re.search(r"if you need|i will|i'll|we can|can help|can bring|available|no problem", s):
#         return "availability_or_commitment"

#     if re.search(r"usually|every time|often|likes?|dislikes?|prefer|habit|repeated", s):
#         return "preference_or_habit"

#     return ""


# def looks_like_person(name: str, wearer_name: str) -> bool:
#     raw = str(name).strip()
#     if not raw:
#         return False
#     if canonicalize_text(raw) in PERSON_LIKE_STOPWORDS:
#         return False
#     if raw.lower() in {"i", "me", "my", "myself"}:
#         return True
#     if raw == wearer_name:
#         return True
#     return looks_like_person_name(raw)


# def normalize_person(name: str, wearer_name: str) -> str:
#     raw = str(name).strip()
#     if not raw:
#         return ""
#     if raw.lower() in {"i", "me", "my", "myself"}:
#         return wearer_name
#     return raw if looks_like_person(raw, wearer_name) else ""


# def relation_summary(head: str, relation: str, tail: str) -> str:
#     if relation == "frequently_interacts_with":
#         return f"{head} frequently interacts with {tail} across multiple events."
#     if relation == "frequently_uses":
#         return f"{head} frequently uses {tail} across multiple events."
#     if relation == "frequently_handles":
#         return f"{head} frequently handles {tail} across multiple events."
#     if relation == "related_to":
#         return f"{head} is repeatedly related to {tail} across multiple events."
#     return f"{head} {relation} {tail} across multiple events."


# def extract_speaker_and_text(line: str, wearer_name: str) -> Tuple[str, str]:
#     raw = str(line).strip()
#     if not raw:
#         return "", ""
#     if ":" in raw:
#         speaker, content = raw.split(":", 1)
#         speaker = speaker.strip()
#         content = content.strip()
#     else:
#         speaker, content = wearer_name, raw
#     speaker = normalize_person(speaker, wearer_name) or wearer_name
#     return speaker, content


# def normalize_provenance_roots(unit: Dict[str, Any]) -> List[str]:
#     roots = unit.get("source_doc_ids") or unit.get("provenance_root_ids") or []
#     roots = [str(x).strip() for x in roots if str(x).strip()]
#     if roots:
#         return sorted(set(roots))
#     doc_id = str(unit.get("doc_id", "")).strip()
#     return [doc_id] if doc_id else []


# # =========================================================
# # Semantic candidate extraction
# # =========================================================

# class SemanticExtraction:
#     """
#     Rule-based semantic candidate extraction.

#     This version remains deterministic, but is broader and more case-free than
#     the earlier first2-oriented version.
#     """

#     def __init__(self, llm_model=None, wearer_name: str = "Jake", model_name: Optional[str] = None):
#         self.llm_model = llm_model
#         self.wearer_name = title_case_name(wearer_name)
#         self.model_name = model_name or getattr(llm_model, "model_name", "rulebased")

#     def _make_candidate(
#         self,
#         head: str,
#         head_type: str,
#         relation: str,
#         tail: str,
#         tail_type: str,
#     ) -> Dict[str, Any]:
#         tail_family = tail if tail_type == "Topic" else tail
#         semantic_key = f"{head}|{head_type}|{relation}|{tail_family}|{tail_type}"
#         return {
#             "head": head,
#             "head_type": head_type,
#             "relation": relation,
#             "tail": tail,
#             "tail_type": tail_type,
#             "tail_family": tail_family,
#             "semantic_key": semantic_key,
#             "semantic_summary": relation_summary(head, relation, tail_family),
#         }

#     def _promote_person_person(self, h: str, r: str, t: str) -> Optional[Dict[str, Any]]:
#         if r not in {"hand_to", "pass_to", "give_to", "help", "assign_to", "talk_to"}:
#             return None
#         p1 = normalize_person(h, self.wearer_name)
#         p2 = normalize_person(t, self.wearer_name)
#         if not p1 or not p2 or p1 == p2:
#             return None
#         head, tail = sorted([p1, p2])
#         return self._make_candidate(
#             head=head,
#             head_type="Person",
#             relation="frequently_interacts_with",
#             tail=tail,
#             tail_type="Person",
#         )

#     def _promote_person_object(self, h: str, r: str, t: str) -> Optional[Dict[str, Any]]:
#         person = normalize_person(h, self.wearer_name)
#         if not person:
#             return None
#         obj = canonicalize_object(t)
#         if not is_good_object(obj):
#             return None

#         if obj in DEVICE_OBJECTS:
#             relation = "frequently_uses"
#         elif obj in EQUIPMENT_OBJECTS:
#             relation = "frequently_handles"
#         elif obj in MATERIAL_OBJECTS:
#             relation = "frequently_handles"
#         else:
#             relation = "frequently_uses" if r == "use" else "frequently_handles"

#         return self._make_candidate(
#             head=person,
#             head_type="Person",
#             relation=relation,
#             tail=obj,
#             tail_type="Object",
#         )

#     def _promote_person_topic(self, h: str, r: str, t: str) -> Optional[Dict[str, Any]]:
#         if r not in {"ask_about", "confirm", "say_about", "discuss", "offer", "propose", "assign", "instruct", "identify", "introduce"}:
#             return None
#         person = normalize_person(h, self.wearer_name)
#         if not person:
#             return None
#         topic_family = canonicalize_topic_family(t)
#         if not topic_family:
#             return None
#         return self._make_candidate(
#             head=person,
#             head_type="Person",
#             relation="related_to",
#             tail=topic_family,
#             tail_type="Topic",
#         )

#     def _promote_triplet(self, triplet: List[str], scale: str) -> Optional[Dict[str, Any]]:
#         if not isinstance(triplet, list) or len(triplet) != 3:
#             return None
#         h, r, t = [str(x).strip() for x in triplet]
#         if not h or not r or not t:
#             return None

#         cand = self._promote_person_person(h, r, t)
#         if cand is not None:
#             return cand

#         if r in {
#             "use", "hold", "inspect", "organize", "place_on", "take_from", "pick_up", "carry",
#             "stack", "write_on", "move", "turn_off", "put_away", "identify", "introduce"
#         }:
#             cand = self._promote_person_object(h, r, t)
#             if cand is not None:
#                 return cand

#         cand = self._promote_person_topic(h, r, t)
#         if cand is not None:
#             return cand

#         return None

#     def _metadata_topic_candidates(self, unit: Dict[str, Any]) -> List[Dict[str, Any]]:
#         candidates: List[Dict[str, Any]] = []
#         seen = set()

#         topic_texts: List[Tuple[str, str, str]] = []
#         for line in unit.get("critical_speech_lines", []) or []:
#             speaker, content = extract_speaker_and_text(line, self.wearer_name)
#             if content:
#                 topic_texts.append((speaker, content, "speech"))

#         for th in unit.get("topic_threads", []) or []:
#             if not isinstance(th, dict):
#                 continue
#             raw_topic = str(th.get("canonical_label") or th.get("topic") or "").strip()
#             if raw_topic:
#                 topic_texts.append((self.wearer_name, raw_topic, "topic_thread"))

#         for speaker, content, source_kind in topic_texts:
#             family = canonicalize_topic_family(content)
#             if not family:
#                 continue
#             key = (speaker, family, source_kind)
#             if key in seen:
#                 continue
#             seen.add(key)
#             cand = self._make_candidate(
#                 head=speaker,
#                 head_type="Person",
#                 relation="related_to",
#                 tail=family,
#                 tail_type="Topic",
#             )
#             cand["metadata_source_kind"] = source_kind
#             candidates.append(cand)

#         return candidates

#     def _allow_candidate_for_scale(self, candidate: Dict[str, Any], scale: str) -> bool:
#         """
#         For coarser scales, be more conservative, especially for generic related_to topics.
#         """
#         scale = str(scale).strip().lower()
#         if scale in {"30s", "30sec"}:
#             return True

#         relation = candidate["relation"]
#         tail_type = candidate["tail_type"]
#         tail = candidate["tail"]

#         if relation == "frequently_interacts_with":
#             return True
#         if relation in {"frequently_uses", "frequently_handles"}:
#             return tail_type == "Object" and (tail in PRIORITY_OBJECTS or len(tokenize(tail)) >= 2)
#         if relation == "related_to":
#             return tail_type == "Topic" and tail in STRONG_TOPIC_FAMILIES
#         return False

#     def semantic_extraction(self, unit: Dict[str, Any]) -> Dict[str, Any]:
#         doc_id = str(unit.get("doc_id", "")).strip()
#         date = str(unit.get("date", "")).strip()
#         start_time = str(unit.get("start_time", "")).strip()
#         end_time = str(unit.get("end_time", "")).strip()
#         scale = str(unit.get("scale", "")).strip() or str(unit.get("level", "")).strip() or "unknown"

#         episodic_triples = unit.get("episodic_triplets", []) or []
#         provenance_roots = normalize_provenance_roots(unit)

#         bucket: Dict[str, Dict[str, Any]] = {}

#         def absorb(candidate: Dict[str, Any], source_triplet: Optional[List[str]], triplet_idx: Optional[int]) -> None:
#             if not self._allow_candidate_for_scale(candidate, scale):
#                 return
#             key = candidate["semantic_key"]
#             if key not in bucket:
#                 bucket[key] = {
#                     **candidate,
#                     "source_doc_ids": [doc_id] if doc_id else [],
#                     "source_scales": [scale] if scale else [],
#                     "support_triplet_indices": [triplet_idx] if triplet_idx is not None else [],
#                     "source_triplets": [source_triplet] if source_triplet is not None else [],
#                     "provenance_root_ids": list(provenance_roots),
#                     "triplet_support_count": 1 if source_triplet is not None else 0,
#                     "metadata_support_count": 0 if source_triplet is not None else 1,
#                 }
#             else:
#                 if triplet_idx is not None:
#                     bucket[key]["support_triplet_indices"].append(triplet_idx)
#                 if source_triplet is not None:
#                     bucket[key]["source_triplets"].append(source_triplet)
#                     bucket[key]["triplet_support_count"] = int(bucket[key].get("triplet_support_count", 0)) + 1
#                 else:
#                     bucket[key]["metadata_support_count"] = int(bucket[key].get("metadata_support_count", 0)) + 1
#                 if doc_id and doc_id not in bucket[key]["source_doc_ids"]:
#                     bucket[key]["source_doc_ids"].append(doc_id)
#                 if scale and scale not in bucket[key]["source_scales"]:
#                     bucket[key]["source_scales"].append(scale)
#                 for rid in provenance_roots:
#                     if rid not in bucket[key]["provenance_root_ids"]:
#                         bucket[key]["provenance_root_ids"].append(rid)

#         for idx, tri in enumerate(episodic_triples):
#             candidate = self._promote_triplet(tri, scale)
#             if candidate is None:
#                 continue
#             absorb(candidate, tri, idx)

#         for candidate in self._metadata_topic_candidates(unit):
#             absorb(candidate, None, None)

#         semantic_candidates = []
#         for cand in bucket.values():
#             if cand["relation"] == "related_to":
#                 triplet_support = int(cand.get("triplet_support_count", 0))
#                 metadata_support = int(cand.get("metadata_support_count", 0))
#                 total_support = triplet_support + metadata_support
#                 tail = cand.get("tail", "")
#                 if tail in WEAK_TOPIC_FAMILIES:
#                     if scale.lower() not in {"30s", "30sec"}:
#                         continue
#                     if total_support < 2 and triplet_support == 0:
#                         continue
#             semantic_candidates.append(cand)
#         semantic_candidates.sort(
#             key=lambda x: (
#                 x["head_type"],
#                 x["head"],
#                 x["relation"],
#                 x["tail_family"],
#                 x["tail_type"],
#             )
#         )

#         return {
#             "doc_id": doc_id,
#             "date": date,
#             "start_time": start_time,
#             "end_time": end_time,
#             "scale": scale,
#             "provenance_root_ids": provenance_roots,
#             "semantic_candidates": semantic_candidates,
#         }

#     def save_results(self, results: Dict[str, Any], output_dir: str = "."):
#         os.makedirs(output_dir, exist_ok=True)
#         out_path = os.path.join(output_dir, f"semantic_candidate_results_{self.model_name}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Semantic candidate results saved to: {out_path}")

#     def batch_semantic_extraction(
#         self,
#         episodic_units: List[Dict[str, Any]],
#         output_dir: str = ".",
#         max_workers: int = 1,
#         log_every: int = 200,
#     ) -> Dict[str, Any]:
#         """
#         Process episodic units and produce semantic candidates.
#         """
#         if max_workers <= 1:
#             results = []
#             for idx, unit in enumerate(tqdm(episodic_units, desc="semantic_extraction"), start=1):
#                 results.append(self.semantic_extraction(unit))
#                 if idx % max(1, log_every) == 0:
#                     logger.info("semantic_extraction progress: %d/%d units", idx, len(episodic_units))
#         else:
#             results_map: Dict[int, Dict[str, Any]] = {}
#             with ThreadPoolExecutor(max_workers=max_workers) as ex:
#                 futures = {
#                     ex.submit(self.semantic_extraction, unit): idx
#                     for idx, unit in enumerate(episodic_units)
#                 }
#                 for done_idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="semantic_extraction"), start=1):
#                     idx = futures[future]
#                     results_map[idx] = future.result()
#                     if done_idx % max(1, log_every) == 0:
#                         logger.info("semantic_extraction progress: %d/%d units", done_idx, len(episodic_units))
#             results = [results_map[i] for i in sorted(results_map)]

#         combined_results = {"units": results}
#         self.save_results(combined_results, output_dir)
#         return combined_results



import json
import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from .utils import SemanticRawOutput, SemanticOutput
from ...llm import LLMModel, PromptTemplateManager
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SemanticExtraction:
    """
    LLM-based semantic extraction from episodic triples.

    Keeps metadata from episodic units, including provenance_root_ids,
    so downstream semantic facts can project back to 30s roots.

    Added robustness:
    - structured retries
    - raw-text fallback
    - JSON repair / malformed-entry cleanup
    """

    def __init__(
        self,
        llm_model: LLMModel,
        max_retries: int = 2,
    ):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model
        self.max_retries = max_retries

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def semantic_extraction(self, chunk_key: str, episodic_triples: List[List[str]]) -> SemanticOutput:
        formatted_triples = "\n".join(f"{i}. {triple}" for i, triple in enumerate(episodic_triples))
        base_messages = self.prompt_template_manager.render(
            name="semantic_extraction",
            episodic_triples=formatted_triples,
        )

        if not isinstance(base_messages, list):
            logger.warning("Prompt render for %s did not return chat messages; got %s", chunk_key, type(base_messages))
            return SemanticOutput(
                chunk_id=chunk_key,
                semantic_triples=[],
                episodic_evidence=[],
            )

        # 1) Structured attempts with retries
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            messages = base_messages if attempt == 0 else self._build_retry_messages(base_messages, attempt, last_error)

            try:
                response = self.llm_model.generate(messages, text_format=SemanticRawOutput)

                repaired = self._repair_payload(
                    {
                        "semantic_triples": getattr(response, "semantic_triples", []),
                        "episodic_evidence": getattr(response, "episodic_evidence", []),
                    },
                    num_input_triples=len(episodic_triples),
                )

                return SemanticOutput(
                    chunk_id=chunk_key,
                    semantic_triples=repaired["semantic_triples"],
                    episodic_evidence=repaired["episodic_evidence"],
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Structured semantic extraction failed for %s on attempt %d/%d: %s",
                    chunk_key,
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )

        # 2) Raw-text fallback + repair
        try:
            repaired = self._generate_with_raw_fallback(
                base_messages=base_messages,
                num_input_triples=len(episodic_triples),
                last_error=last_error,
            )
            if repaired["semantic_triples"] or repaired["episodic_evidence"]:
                logger.info("Recovered semantic extraction for %s via raw fallback", chunk_key)
                return SemanticOutput(
                    chunk_id=chunk_key,
                    semantic_triples=repaired["semantic_triples"],
                    episodic_evidence=repaired["episodic_evidence"],
                )
        except Exception as e:
            logger.warning("Raw fallback semantic extraction failed for %s: %s", chunk_key, e)

        logger.warning("Semantic extraction failed for %s; returning empty output", chunk_key)
        return SemanticOutput(
            chunk_id=chunk_key,
            semantic_triples=[],
            episodic_evidence=[],
        )

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def batch_semantic_extraction(
        self,
        episodic_triples_batch: Dict[str, List[List[str]]],
        output_dir: str = ".",
    ) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[List[int]]]]:
        payload_batch = {
            chunk_key: {"triples": triples, "metadata": {}}
            for chunk_key, triples in episodic_triples_batch.items()
        }
        combined_results = self.batch_semantic_extraction_with_metadata(
            payload_batch,
            output_dir=output_dir,
        )
        return combined_results["semantic_triples"], combined_results["episodic_evidence"]

    def batch_semantic_extraction_with_metadata(
        self,
        episodic_payload_batch: Dict[str, Dict[str, Any]],
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        results: List[Tuple[str, SemanticOutput]] = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.semantic_extraction, chunk_key, payload["triples"]): chunk_key
                for chunk_key, payload in episodic_payload_batch.items()
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Extracting semantic triples",
                leave=True,
            )

            for future in pbar:
                chunk_key = futures[future]
                result = future.result()
                results.append((chunk_key, result))

        ordered_keys = list(episodic_payload_batch.keys())
        semantic_triples_map = {chunk_key: res.semantic_triples for chunk_key, res in results}
        episodic_evidence_map = {chunk_key: res.episodic_evidence for chunk_key, res in results}

        items: List[Dict[str, Any]] = []
        for chunk_key in ordered_keys:
            payload = episodic_payload_batch[chunk_key]
            metadata = dict(payload.get("metadata", {}))
            triples = payload.get("triples", [])
            item = {
                "chunk_id": chunk_key,
                **metadata,
                "source_triples": triples,
                "semantic_triples": semantic_triples_map.get(chunk_key, []),
                "episodic_evidence": episodic_evidence_map.get(chunk_key, []),
            }
            items.append(item)

        combined_results: Dict[str, Any] = {
            "items": items,
            "semantic_triples": {
                item["chunk_id"]: item["semantic_triples"] for item in items
            },
            "episodic_evidence": {
                item["chunk_id"]: item["episodic_evidence"] for item in items
            },
            "metadata": {
                item["chunk_id"]: {
                    k: v
                    for k, v in item.items()
                    if k not in {"semantic_triples", "episodic_evidence", "source_triples"}
                }
                for item in items
            },
        }

        output_path = os.path.join(
            output_dir,
            f"semantic_extraction_results_{self.llm_model.model_name}.json",
        )
        self.save_results(combined_results, output_path)
        logger.info("Saved semantic extraction results to %s", output_path)
        return combined_results

    # -------------------------------------------------------------------------
    # Retry helpers
    # -------------------------------------------------------------------------
    def _build_retry_messages(
        self,
        base_messages: List[Dict[str, str]],
        attempt: int,
        last_error: Exception | None,
    ) -> List[Dict[str, str]]:
        retry_instruction = (
            "Your previous output was invalid.\n"
            "Return ONLY valid JSON with exactly two keys: semantic_triples and episodic_evidence.\n"
            "Rules:\n"
            "- Never output an empty item like [] inside semantic_triples.\n"
            "- Every semantic triple must be exactly 3 non-empty strings: [subject, predicate, object].\n"
            "- Never output 2-slot triples such as ['Jake', 'holds_phone_with_both_hands'].\n"
            "- If a candidate triple is malformed, OMIT it instead of keeping it.\n"
            "- episodic_evidence must align exactly with semantic_triples.\n"
            "- Every evidence list must contain only valid 0-based integer indices.\n"
        )
        if last_error is not None:
            retry_instruction += f"\nPrevious validation error:\n{str(last_error)}\n"

        return list(base_messages) + [
            {"role": "user", "content": retry_instruction}
        ]

    # -------------------------------------------------------------------------
    # Raw fallback
    # -------------------------------------------------------------------------
    def _generate_with_raw_fallback(
        self,
        base_messages: List[Dict[str, str]],
        num_input_triples: int,
        last_error: Exception | None,
    ) -> Dict[str, List]:
        messages = self._build_retry_messages(base_messages, attempt=self.max_retries + 1, last_error=last_error)

        raw_response = self.llm_model.generate(messages)
        raw_text = self._extract_text_from_response(raw_response)
        parsed = self._parse_json_from_text(raw_text)

        return self._repair_payload(parsed, num_input_triples=num_input_triples)

    def _extract_text_from_response(self, raw_response: Any) -> str:
        """
        Best-effort extraction of text from different possible SDK wrappers.
        Adjust here if your LLM wrapper returns a different shape.
        """
        if raw_response is None:
            raise ValueError("Raw response is None")

        if isinstance(raw_response, str):
            return raw_response

        if isinstance(raw_response, dict):
            for key in ("output_text", "text", "content", "response", "raw_text"):
                value = raw_response.get(key)
                if isinstance(value, str):
                    return value

        for attr in ("output_text", "text", "content", "response", "raw_text"):
            if hasattr(raw_response, attr):
                value = getattr(raw_response, attr)
                if isinstance(value, str):
                    return value

        # last resort
        return str(raw_response)

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Empty raw text from model")

        cleaned = text.strip()

        # Remove fenced code block wrappers if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # First try direct parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Then try extracting the first top-level JSON object substring
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Could not parse a JSON object from raw model output")

    # -------------------------------------------------------------------------
    # Output repair
    # -------------------------------------------------------------------------
    def _repair_payload(
        self,
        payload: Dict[str, Any],
        num_input_triples: int,
    ) -> Dict[str, List]:
        semantic_triples = payload.get("semantic_triples", [])
        episodic_evidence = payload.get("episodic_evidence", [])

        if not isinstance(semantic_triples, list):
            semantic_triples = []
        if not isinstance(episodic_evidence, list):
            episodic_evidence = []

        repaired_triples: List[List[str]] = []
        repaired_evidence: List[List[int]] = []

        pair_count = min(len(semantic_triples), len(episodic_evidence))
        dropped_count = 0

        for i in range(pair_count):
            triple = semantic_triples[i]
            evidence = episodic_evidence[i]

            repaired_triple = self._repair_triple(triple)
            repaired_ev = self._repair_evidence(evidence, num_input_triples=num_input_triples)

            if repaired_triple is None:
                dropped_count += 1
                continue
            if not repaired_ev:
                dropped_count += 1
                continue

            repaired_triples.append(repaired_triple)
            repaired_evidence.append(repaired_ev)

        if len(semantic_triples) != len(episodic_evidence):
            logger.warning(
                "semantic_triples and episodic_evidence length mismatch: %d vs %d; truncated to %d",
                len(semantic_triples),
                len(episodic_evidence),
                pair_count,
            )

        if dropped_count > 0:
            logger.warning("Dropped %d malformed semantic entries during repair", dropped_count)

        return {
            "semantic_triples": repaired_triples,
            "episodic_evidence": repaired_evidence,
        }

    def _repair_triple(self, triple: Any) -> List[str] | None:
        if not isinstance(triple, list):
            return None

        # valid case
        if len(triple) == 3 and all(isinstance(x, str) for x in triple):
            cleaned = [self._clean_text(x) for x in triple]
            if all(cleaned):
                return cleaned
            return None

        # malformed cases: drop rather than guess too much
        # e.g. [] or ['Jake', 'holds_phone_with_both_hands']
        return None

    def _repair_evidence(self, evidence: Any, num_input_triples: int) -> List[int]:
        if not isinstance(evidence, list):
            return []

        repaired: List[int] = []
        seen = set()

        for item in evidence:
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < num_input_triples and idx not in seen:
                seen.add(idx)
                repaired.append(idx)

        repaired.sort()
        return repaired

    def _clean_text(self, text: str) -> str:
        return " ".join(text.strip().split())

