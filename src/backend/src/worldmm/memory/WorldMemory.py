"""
WorldMemory: unified event-centric memory system.

This version implements:
- multiscale episodic retrieval with dense RAG + graph-aware rerank
- soft-grouped event candidate pool (trigger / antecedent / broader-context)
- LLM event selector over a coarse candidate pool
- semantic memory as support only (not primary event routing)
- visual evidence only for final selected event anchors (keyframes)
"""

import time
import copy
import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from ..embedding import EmbeddingModel
from ..llm import LLMModel, PromptTemplateManager
from .episodic.EpisodicMemory_rag import CaptionEntryRAG, EpisodicMemoryRAG
from .semantic import SemanticMemory, SemanticTripleEntry
from .utils import *
from .visual import VisualMemory

logger = logging.getLogger(__name__)


STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "this", "that", "these", "those", "it", "its"
}


def _structured_value_to_text(value: Any) -> str:
    """Convert structured visual evidence into compact prompt-safe text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred_keys = (
            "name",
            "label",
            "object",
            "entity",
            "actor",
            "action",
            "attribute",
            "before",
            "after",
            "location",
            "time",
            "confidence",
        )
        parts: List[str] = []
        for key in preferred_keys:
            if key not in value:
                continue
            text = _structured_value_to_text(value.get(key))
            if text:
                parts.append(f"{key}={text}")
        if parts:
            return "; ".join(parts)
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        parts = [_structured_value_to_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", str(value)).strip()


def _structured_values_to_list(value: Any, limit: Optional[int] = None) -> List[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    texts = [_structured_value_to_text(item) for item in values]
    texts = [text for text in texts if text]
    return texts[:limit] if limit is not None else texts


class WorldMemory:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever_llm_model: LLMModel,
        respond_llm_model: Optional[LLMModel] = None,
        prompt_template_manager: Optional[PromptTemplateManager] = None,
        episodic_granularities: Optional[List[str]] = None,
        episodic_cache_tag: Optional[str] = None,
        max_rounds: int = 5,
        max_errors: int = 5,
    ):
        self.embedding_model = embedding_model
        self.retriever_llm_model = retriever_llm_model
        self.respond_llm_model = respond_llm_model or retriever_llm_model
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.max_rounds = max_rounds
        self.max_errors = max_errors

        self.episodic_memory = EpisodicMemoryRAG(
            embedding_model=embedding_model,
            llm_model=retriever_llm_model,
            prompt_template_manager=self.prompt_template_manager,
            granularities=episodic_granularities,
            cache_tag=episodic_cache_tag,
        )
        self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
        self.visual_memory = VisualMemory(embedding_model=embedding_model)

        self.indexed_time: int = 0

        self.episodic_top_k: int = 3
        self.semantic_top_k: int = 10
        self.visual_top_k: int = 3

        # anchor projection weights across granularities
        self.anchor_weight_30s = 1.00
        self.anchor_weight_3min = 0.65
        self.anchor_weight_10min = 0.45
        self.anchor_weight_1h = 0.30

        # soft-group selector pool sizes
        self.selector_global_top_n = 10
        self.selector_trigger_top_n = 4
        self.selector_antecedent_top_n = 4
        self.selector_broader_top_n = 3
        self.selector_max_candidates = 12

    # -----------------------------------------------------
    # query formatting
    # -----------------------------------------------------

    def _build_query_with_time(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time: Optional[int] = None,
    ) -> str:
        lines = [f"Query: {query}"]
        if until_time is not None:
            lines.append(f"Query Time: {transform_timestamp(str(until_time))}")
            lines.append(
                "Important: Interpret all relative temporal expressions "
                '(e.g. "before", "after", "earlier", "later", "recently", '
                '"a few hours ago", "first", "last") relative to Query Time.'
            )
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            lines.append(f"Choices: {choices_str}")
        return "\n".join(lines)

    # -----------------------------------------------------
    # loading
    # -----------------------------------------------------

    def load_episodic_captions(
        self,
        caption_files: Optional[Dict[str, str]] = None,
        caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        if caption_files:
            self.episodic_memory.load_captions_from_files(caption_files)
        if caption_data:
            self.episodic_memory.load_captions_from_data(caption_data)

    def load_episodic_sidecar(
        self,
        triplet_files: Optional[Dict[str, str]] = None,
        graph_files: Optional[Dict[str, str]] = None,
        triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
        graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if triplet_files or graph_files:
            self.episodic_memory.load_sidecar_from_files(
                triplet_files=triplet_files,
                graph_files=graph_files,
            )
        if triplet_data or graph_data:
            self.episodic_memory.load_sidecar_from_data(
                triplet_data=triplet_data,
                graph_data=graph_data,
            )

    def load_semantic_triples(
        self,
        file_path: Optional[str] = None,
        data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if file_path:
            self.semantic_memory.load_triples_from_file(file_path)
        if data:
            self.semantic_memory.load_triples_from_data(data)

    def load_visual_clips(
        self,
        embeddings_path: Optional[str] = None,
        clips_path: Optional[str] = None,
        clips_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if embeddings_path:
            self.visual_memory.load_embeddings_from_file(embeddings_path)
        if clips_path:
            self.visual_memory.load_clips_from_file(clips_path)
        if clips_data:
            self.visual_memory.load_clips_from_data(clips_data)

    def prepare_episodic_dense_index(self, force_rebuild: bool = False) -> None:
        if hasattr(self.episodic_memory, "build_dense_index"):
            self.episodic_memory.build_dense_index(force_rebuild=force_rebuild)

    # -----------------------------------------------------
    # indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
            return

        logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")

        if hasattr(self.episodic_memory, "build_dense_index"):
            self.episodic_memory.build_dense_index(force_rebuild=False)

        self.episodic_memory.index(until_time)
        self.semantic_memory.index(until_time)
        self.visual_memory.index(until_time)
        self.indexed_time = until_time
        logger.info("Indexing complete for all memory types")

    # -----------------------------------------------------
    # helpers
    # -----------------------------------------------------

    def _tokenize(self, text: str) -> Set[str]:
        toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
        return {t for t in toks if len(t) > 1 and t not in STOPWORDS}

    def _normalize_dict(self, score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        mn, mx = min(values), max(values)
        if abs(mx - mn) < 1e-8:
            return {k: 1.0 for k in score_map}
        return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}

    def _episodic_weight_for_granularity(self, granularity: str) -> float:
        if granularity == "30sec":
            return self.anchor_weight_30s
        if granularity == "3min":
            return self.anchor_weight_3min
        if granularity == "10min":
            return self.anchor_weight_10min
        if granularity == "1h":
            return self.anchor_weight_1h
        return 0.30

    def _parse_timestamp_int(self, ts: int) -> Tuple[int, int, int, int]:
        ts_str = str(ts)
        day = int(ts_str[0])
        hh = int(ts_str[1:3])
        mm = int(ts_str[3:5])
        ss = int(ts_str[5:7])
        return day, hh, mm, ss

    def _timestamp_to_seconds(self, ts: int) -> int:
        day, hh, mm, ss = self._parse_timestamp_int(ts)
        return day * 86400 + hh * 3600 + mm * 60 + ss

    def _entry_center_seconds(self, entry: CaptionEntryRAG) -> float:
        start_ts, end_ts = entry.timestamp_int
        return 0.5 * (self._timestamp_to_seconds(start_ts) + self._timestamp_to_seconds(end_ts))

    def _overlap_ratio(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a))

    def _event_tokens(self, doc_id: str) -> Set[str]:
        toks: Set[str] = set()
        entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
        if entry is None:
            return toks
        toks.update(self._tokenize(entry.text))
        toks.update(self._tokenize(entry.visual_summary))
        for line in entry.metadata.get("critical_speech_lines", []) or []:
            toks.update(self._tokenize(str(line)))

        parent = None
        if hasattr(self.episodic_memory, "get_parent_caption"):
            parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
        if parent is not None:
            toks.update(self._tokenize(parent.text))
            toks.update(self._tokenize(parent.visual_summary))
            for line in parent.metadata.get("critical_speech_lines", []) or []:
                toks.update(self._tokenize(str(line)))

        for tri in self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")[:8]:
            if isinstance(tri, list) and len(tri) == 3:
                toks.update(self._tokenize(" ".join(map(str, tri))))

        visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
        if visual_entry is not None:
            toks.update(self._tokenize(getattr(visual_entry, "keyframe_caption", "")))
            scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
            if isinstance(scene_summary, dict):
                toks.update(self._tokenize(json.dumps(scene_summary, ensure_ascii=False)))
            for obj in getattr(visual_entry, "visual_objects", []) or []:
                toks.update(self._tokenize(str(obj)))

        return toks

    def _project_episodic_candidates_to_30s(
        self,
        candidates: List[Tuple[CaptionEntryRAG, float]],
    ) -> Dict[str, float]:
        projected = defaultdict(float)
        if not candidates:
            return projected

        for rank, (entry, score) in enumerate(candidates):
            rank_bonus = 1.0 / (rank + 1)
            base = float(score) * rank_bonus * self._episodic_weight_for_granularity(entry.granularity)
            target_doc_ids = self.episodic_memory.expand_entry_to_30s_doc_ids(entry)
            target_doc_ids = [
                doc_id for doc_id in target_doc_ids
                if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
            ]
            if not target_doc_ids:
                continue
            denom = math.sqrt(len(target_doc_ids))
            for doc_id in target_doc_ids:
                projected[doc_id] += base / denom
        return projected

    def _project_semantic_to_30s(
        self,
        semantic_entries: List[SemanticTripleEntry],
    ) -> Tuple[Dict[str, float], Dict[str, List[SemanticTripleEntry]]]:
        projected = defaultdict(float)
        support_map: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)
        if not semantic_entries:
            return projected, support_map

        for rank, entry in enumerate(semantic_entries):
            candidate_roots = (
                list(getattr(entry, "provenance_root_ids", []) or [])
                or list(getattr(entry, "source_doc_ids", []) or [])
                or list(entry.evidence_event_ids or [])
            )
            if not candidate_roots:
                continue

            rank_bonus = 1.0 / (rank + 1)
            support_factor = 1.0 + 0.15 * min(int(entry.support_count), 5)
            conf_factor = 0.7 + 0.3 * float(entry.confidence)
            base = rank_bonus * support_factor * conf_factor

            valid_doc_ids: List[str] = []
            for root_id in candidate_roots:
                if self.episodic_memory.get_caption_by_doc_id(root_id, "30sec") is not None:
                    valid_doc_ids.append(root_id)
                    continue
                root_entry = self.episodic_memory.get_caption_by_doc_id(root_id, granularity=None)
                if root_entry is not None:
                    valid_doc_ids.extend(self.episodic_memory.expand_entry_to_30s_doc_ids(root_entry))

            valid_doc_ids = [
                doc_id for doc_id in dict.fromkeys(valid_doc_ids)
                if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
            ]
            if not valid_doc_ids:
                continue

            denom = math.sqrt(len(valid_doc_ids))
            for doc_id in valid_doc_ids:
                projected[doc_id] += base / denom
                support_map[doc_id].append(entry)

        return projected, support_map

    def _build_semantic_context(self, semantic_entries: List[SemanticTripleEntry], top_n: int = 5) -> str:
        if not semantic_entries:
            return ""
        lines = ["Semantic Facts:"]
        for entry in semantic_entries[:top_n]:
            lines.append(f"- {entry.to_display_str()}")
        return "\n".join(lines)

    def _build_event_packet(
        self,
        doc_id: str,
        score: float,
        supporting_facts: Optional[List[SemanticTripleEntry]] = None,
    ) -> str:
        entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
        if entry is None:
            return ""

        visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
        triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
        supporting_facts = supporting_facts or []
        parent_3min = None
        if hasattr(self.episodic_memory, "get_parent_caption"):
            parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min")

        lines = []
        lines.append(f"Event Anchor: {doc_id}")
        lines.append(f"Relevance Score: {score:.4f}")
        lines.append(entry.to_display_str(include_visual_summary=True))

        critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
        if critical_lines:
            lines.append("Critical Speech:")
            for line in critical_lines[:3]:
                if str(line).strip():
                    lines.append(f"- {line}")

        if parent_3min is not None and parent_3min.doc_id != doc_id:
            p_start, p_end = parent_3min.timestamp_int
            lines.append(
                f"3min Context [{transform_timestamp(str(p_start))} - {transform_timestamp(str(p_end))}]: {parent_3min.text}"
            )
            if parent_3min.visual_summary:
                lines.append(f"3min Visual: {parent_3min.visual_summary}")
            parent_critical_lines = list(parent_3min.metadata.get("critical_speech_lines", []) or [])
            if parent_critical_lines:
                lines.append("3min Critical Speech:")
                for line in parent_critical_lines[:3]:
                    if str(line).strip():
                        lines.append(f"- {line}")

        if visual_entry is not None:
            if getattr(visual_entry, "keyframe_caption", ""):
                lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
            visual_objects = getattr(visual_entry, "visual_objects", []) or []
            visual_object_texts = _structured_values_to_list(visual_objects, limit=8)
            if visual_object_texts:
                lines.append("Visual Objects: " + ", ".join(visual_object_texts))
            scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
            if isinstance(scene_summary, dict):
                dominant_scene = scene_summary.get("dominant_scene", "")
                if dominant_scene:
                    lines.append(f"Scene: {dominant_scene}")

        if triplets:
            lines.append("Episodic Triplets:")
            for tri in triplets[:6]:
                if isinstance(tri, list) and len(tri) == 3:
                    lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

        if supporting_facts:
            lines.append("Supporting Semantic Facts:")
            for fact in supporting_facts[:3]:
                lines.append(f"- {fact.to_display_str()}")

        return "\n".join(lines)

    def _build_round_history(
        self,
        query: str,
        top_doc_ids: List[str],
        semantic_entries: List[SemanticTripleEntry],
    ) -> List[Dict[str, Any]]:
        return [{
            "round_num": 1,
            "decision": "search",
            "memory_type": "episodic+semantic",
            "search_query": query,
            "retrieved_content": (
                f"Top events: {top_doc_ids}\n"
                f"Top semantic facts: {[e.id for e in semantic_entries[:5]]}"
            ),
        }]

    def _render_retrieved_items_for_qa(self, retrieved_items: List[RetrievedItem]) -> List[Dict[str, Any]]:
        messages = []
        for item in retrieved_items:
            if item.memory_type in ("episodic", "semantic"):
                messages.append({"type": "text", "text": item.content})
            elif item.memory_type == "visual":
                if isinstance(item.content, list):
                    for img in item.content:
                        if isinstance(img, Image.Image):
                            messages.append({"type": "image", "image": img})
                        elif isinstance(img, dict) and "image" in img:
                            messages.append({"type": "image", "image": img["image"]})
        return messages

    # -----------------------------------------------------
    # soft-group selector
    # -----------------------------------------------------

    def _compute_event_role_scores(
        self,
        query: str,
        episodic_norm: Dict[str, float],
        semantic_norm: Dict[str, float],
        semantic_support_map: Dict[str, List[SemanticTripleEntry]],
    ) -> Dict[str, Dict[str, float]]:
        if not episodic_norm:
            return {}

        query_tokens = self._tokenize(query)
        seed_doc_ids = [doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])[:3]]
        seed_entries = [self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") for doc_id in seed_doc_ids]
        seed_entries = [e for e in seed_entries if e is not None]
        seed_tokens = {doc_id: self._event_tokens(doc_id) for doc_id in seed_doc_ids}
        seed_centers = {
            doc_id: self._entry_center_seconds(self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec"))
            for doc_id in seed_doc_ids
            if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
        }

        trigger_centroid = 0.0
        if seed_centers:
            trigger_centroid = sum(seed_centers.values()) / len(seed_centers)
        earliest_seed = min(seed_centers.values()) if seed_centers else None

        parent_counts: Dict[str, int] = defaultdict(int)
        for doc_id in episodic_norm:
            parent = None
            if hasattr(self.episodic_memory, "get_parent_caption"):
                parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
            if parent is not None:
                parent_counts[parent.doc_id] += 1

        role_scores: Dict[str, Dict[str, float]] = {}
        for doc_id, ep_score in episodic_norm.items():
            entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
            if entry is None:
                continue

            center_sec = self._entry_center_seconds(entry)
            toks = self._event_tokens(doc_id)
            query_overlap = self._overlap_ratio(query_tokens, toks)
            seed_overlap = 0.0
            if seed_tokens:
                seed_overlap = max(self._overlap_ratio(toks, x) for x in seed_tokens.values())

            if trigger_centroid > 0.0:
                delta = abs(center_sec - trigger_centroid)
                temporal_proximity = math.exp(-delta / 600.0)
            else:
                temporal_proximity = 0.0
            trigger_score = 0.60 * ep_score + 0.20 * query_overlap + 0.20 * temporal_proximity

            earlierness = 0.0
            if earliest_seed is not None and center_sec < earliest_seed:
                gap = earliest_seed - center_sec
                earlierness = min(1.0, gap / 1800.0)
            support_presence = 1.0 if semantic_support_map.get(doc_id) else 0.0
            antecedent_score = 0.45 * earlierness + 0.35 * seed_overlap + 0.20 * support_presence

            broader_score = 0.0
            parent = None
            if hasattr(self.episodic_memory, "get_parent_caption"):
                parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
            if parent is not None:
                coverage = parent_counts.get(parent.doc_id, 0)
                broader_coverage = min(1.0, coverage / 3.0)
                broader_score = 0.60 * broader_coverage + 0.25 * query_overlap + 0.15 * ep_score

            role_scores[doc_id] = {
                "trigger": float(trigger_score),
                "antecedent": float(antecedent_score),
                "broader": float(broader_score),
                "semantic": float(semantic_norm.get(doc_id, 0.0)),
            }

        for role_name in ["trigger", "antecedent", "broader", "semantic"]:
            normed = self._normalize_dict({k: v[role_name] for k, v in role_scores.items()})
            for doc_id in role_scores:
                role_scores[doc_id][role_name] = normed.get(doc_id, 0.0)

        return role_scores

    def _build_event_selector_candidates(
        self,
        query: str,
        episodic_norm: Dict[str, float],
        semantic_norm: Dict[str, float],
        semantic_support_map: Dict[str, List[SemanticTripleEntry]],
    ) -> List[Dict[str, Any]]:
        role_scores = self._compute_event_role_scores(
            query=query,
            episodic_norm=episodic_norm,
            semantic_norm=semantic_norm,
            semantic_support_map=semantic_support_map,
        )
        if not role_scores:
            return []

        global_sorted = [doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])[: self.selector_global_top_n]]
        trigger_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["trigger"])[: self.selector_trigger_top_n]]
        antecedent_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["antecedent"])[: self.selector_antecedent_top_n]]
        broader_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["broader"])[: self.selector_broader_top_n]]

        ordered_doc_ids: List[str] = []
        for group in [global_sorted, trigger_sorted, antecedent_sorted, broader_sorted]:
            for doc_id in group:
                if doc_id not in ordered_doc_ids:
                    ordered_doc_ids.append(doc_id)
                if len(ordered_doc_ids) >= self.selector_max_candidates:
                    break
            if len(ordered_doc_ids) >= self.selector_max_candidates:
                break

        logger.info(
            "Selector pool groups | global=%s | trigger=%s | antecedent=%s | broader=%s",
            global_sorted,
            trigger_sorted,
            antecedent_sorted,
            broader_sorted,
        )

        candidates: List[Dict[str, Any]] = []
        for idx, doc_id in enumerate(ordered_doc_ids, start=1):
            entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
            if entry is None:
                continue
            parent = None
            if hasattr(self.episodic_memory, "get_parent_caption"):
                parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
            triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")[:4]
            support_facts = semantic_support_map.get(doc_id, [])[:2]
            primary_role = max(
                ["trigger", "antecedent", "broader"],
                key=lambda r: role_scores[doc_id].get(r, 0.0),
            )

            visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
            candidates.append({
                "index": idx,
                "doc_id": doc_id,
                "start_time": transform_timestamp(str(entry.timestamp_int[0])),
                "end_time": transform_timestamp(str(entry.timestamp_int[1])),
                "caption": entry.text,
                "visual_summary": entry.visual_summary,
                "critical_speech_lines": list(entry.metadata.get("critical_speech_lines", []) or [])[:4],
                "episodic_score": round(float(episodic_norm.get(doc_id, 0.0)), 4),
                "semantic_score": round(float(semantic_norm.get(doc_id, 0.0)), 4),
                "trigger_score": round(float(role_scores[doc_id].get("trigger", 0.0)), 4),
                "antecedent_score": round(float(role_scores[doc_id].get("antecedent", 0.0)), 4),
                "broader_score": round(float(role_scores[doc_id].get("broader", 0.0)), 4),
                "primary_role": primary_role,
                "triplets": triplets,
                "keyframe_caption": getattr(visual_entry, "keyframe_caption", "") if visual_entry is not None else "",
                "parent_3min_doc_id": parent.doc_id if parent is not None else None,
                "parent_3min_caption": parent.text if parent is not None else "",
                "parent_3min_visual_summary": parent.visual_summary if parent is not None else "",
                "parent_3min_critical_speech": list(parent.metadata.get("critical_speech_lines", []) or [])[:4] if parent is not None else [],
                "semantic_support": [fact.to_display_str() for fact in support_facts],
            })

        return candidates

    def _parse_event_selector_response(
        self,
        response: str,
        valid_doc_ids: List[str],
        num_candidates: int,
    ) -> List[str]:
        valid_doc_id_set = set(valid_doc_ids)
        selected: List[str] = []

        try:
            json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)

            if isinstance(parsed, dict):
                for key in ["selected_doc_ids", "doc_ids", "selected"]:
                    if key in parsed and isinstance(parsed[key], list):
                        for x in parsed[key]:
                            if isinstance(x, str) and x in valid_doc_id_set and x not in selected:
                                selected.append(x)
                for key in ["selected_indices", "indices"]:
                    if key in parsed and isinstance(parsed[key], list):
                        for x in parsed[key]:
                            try:
                                idx = int(x)
                            except Exception:
                                continue
                            if 1 <= idx <= len(valid_doc_ids):
                                doc_id = valid_doc_ids[idx - 1]
                                if doc_id not in selected:
                                    selected.append(doc_id)
            elif isinstance(parsed, list):
                for x in parsed:
                    if isinstance(x, str) and x in valid_doc_id_set and x not in selected:
                        selected.append(x)
                    else:
                        try:
                            idx = int(x)
                        except Exception:
                            continue
                        if 1 <= idx <= len(valid_doc_ids):
                            doc_id = valid_doc_ids[idx - 1]
                            if doc_id not in selected:
                                selected.append(doc_id)
        except Exception:
            pass

        if not selected:
            for doc_id in re.findall(r"DAY\d_[0-9]{8}_[0-9]{8}(?:_[A-Za-z0-9]+)?", response):
                if doc_id in valid_doc_id_set and doc_id not in selected:
                    selected.append(doc_id)

        if not selected:
            for m in re.findall(r"\b(?:candidate|index|idx)?\s*#?\s*(\d{1,2})\b", response, re.IGNORECASE):
                try:
                    idx = int(m)
                except Exception:
                    continue
                if 1 <= idx <= num_candidates:
                    doc_id = valid_doc_ids[idx - 1]
                    if doc_id not in selected:
                        selected.append(doc_id)

        return selected

    def _extract_selector_reason(self, response: str) -> str:
        try:
            json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
            parsed = json.loads(json_match.group()) if json_match else json.loads(response)
            if isinstance(parsed, dict):
                for key in ["reason", "rationale", "summary", "explanation"]:
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        except Exception:
            pass
        response = str(response).strip()
        return response[:2000] if response else ""

    def _extract_selector_metadata(self, response: str) -> Dict[str, str]:
        meta = {"question_family": "", "reason": ""}
        try:
            json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
            parsed = json.loads(json_match.group()) if json_match else json.loads(response)
            if isinstance(parsed, dict):
                qf = parsed.get("question_family", "")
                if isinstance(qf, str):
                    meta["question_family"] = qf.strip()
                for key in ["reason", "rationale", "summary", "explanation"]:
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        meta["reason"] = value.strip()
                        break
        except Exception:
            pass
        return meta

    def _select_top_events_with_llm(
        self,
        query: str,
        choices: Optional[Dict[str, str]],
        until_time: Optional[int],
        selector_candidates: List[Dict[str, Any]],
        final_top_k: int,
    ) -> Tuple[List[str], str]:
        if not selector_candidates:
            return [], ""
        if len(selector_candidates) <= final_top_k:
            return [c["doc_id"] for c in selector_candidates], "Selector shortcut: number of candidates <= final_top_k."

        query_with_time = self._build_query_with_time(query=query, choices=choices, until_time=until_time)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are selecting event packets for a long-video QA system.\n"
                    "Your job is NOT to choose events that are merely topically related. "
                    "Your job is to choose events whose evidence matches the exact predicate asked by the question.\n\n"

                    "You must do two things:\n"
                    "Step 1: infer the question family from the question.\n"
                    "Step 2: choose a small, complementary set of event packets that best supports the answer.\n\n"

                    "Use one of these question families:\n"
                    "1) action-owner\n"
                    "2) source-trace\n"
                    "3) participant-membership\n"
                    "4) plan-intention-decision\n"
                    "5) temporal-recall\n"
                    "6) habit-preference\n"
                    "7) attribute-content-purpose\n\n"

                    "Core principle:\n"
                    "- Prefer explicit evidence over weak implication.\n"
                    "- Prefer predicate-aligned evidence over broad contextual relevance.\n"
                    "- Do not over-select near-duplicate local events.\n"
                    "- Always return valid candidate indices and/or valid doc_ids from the provided list only.\n\n"

                    "[action-owner]\n"
                    "Question intent: identify who performed an action, who assisted, or who acted first.\n"
                    "Strong evidence:\n"
                    "- explicit actor + explicit queried action\n"
                    "- explicit cooperation, transfer, or assistance evidence when the question is about helping\n"
                    "- earliest valid explicit action when the question is about who acted first\n"
                    "Weak evidence:\n"
                    "- nearby presence\n"
                    "- interaction with related objects without the queried action\n"
                    "- later result scenes without explicit action evidence\n"
                    "Do NOT:\n"
                    "- infer the actor only from scene participation\n"
                    "- replace explicit action evidence with general topic-related context\n\n"

                    "[source-trace]\n"
                    "Question intent: identify where an object was before, where it came from, or how it was transferred.\n"
                    "Strong evidence:\n"
                    "- explicit prior location\n"
                    "- explicit transfer path\n"
                    "- explicit retrieval, carrying, bringing, taking, placing, or movement-between-locations evidence\n"
                    "- earlier events that directly establish previous location\n"
                    "Weak evidence:\n"
                    "- current-use scenes\n"
                    "- current location alone\n"
                    "- generic earlier background context without explicit source grounding\n"
                    "Do NOT:\n"
                    "- treat holding, using, or interacting with an object as sufficient evidence of prior location\n"
                    "- answer a previous-location question using only current-scene context\n"
                    "- omit a source-establishing event if one exists\n\n"

                    "[participant-membership]\n"
                    "Question intent: identify who joined, who helped, who was part of the activity, or who was absent.\n"
                    "Strong evidence:\n"
                    "- explicit participation in the shared activity\n"
                    "- explicit join/help/presence evidence in the relevant action chain\n"
                    "- contrastive evidence for absence or mismatch across time\n"
                    "Weak evidence:\n"
                    "- later co-presence in the same room\n"
                    "- nearby observer or bystander context\n"
                    "Do NOT:\n"
                    "- infer participation only from later appearance\n"
                    "- confuse bystanders with core participants\n\n"

                    "[plan-intention-decision]\n"
                    "Question intent: identify a plan, intention, decision, next step, proposal, or commitment.\n"
                    "Strong evidence:\n"
                    "- explicit plan, intention, decision, proposal, assignment, or commitment\n"
                    "- agent-specific future commitment\n"
                    "- final-decision evidence\n"
                    "Weak evidence:\n"
                    "- related discussion\n"
                    "- explanation, recommendation, or evaluation\n"
                    "- general topic proximity\n"
                    "- observation statements without commitment\n"
                    "- offer or suggestion unless it clearly implies the agent's own intended action\n"
                    "Do NOT:\n"
                    "- infer intention from discussion alone\n"
                    "- infer a personal plan from explanation or recommendation alone\n"
                    "- confuse proposal, observation, ownership, or topic relevance with intention\n\n"

                    "[temporal-recall]\n"
                    "Question intent: identify the last time, first time, previous occurrence, or temporally constrained event.\n"
                    "Strong evidence:\n"
                    "- event whose timestamp best satisfies the temporal constraint\n"
                    "- closest valid earlier or later occurrence that truly matches the queried event or topic\n"
                    "Weak evidence:\n"
                    "- semantically similar event at the wrong time\n"
                    "- salient but temporally invalid event\n"
                    "Do NOT:\n"
                    "- ignore first/last/before/after constraints\n"
                    "- choose a more relevant-looking event if its time is wrong\n\n"

                    "[habit-preference]\n"
                    "Question intent: identify a repeated behavior, usual pattern, stable preference, or dislike.\n"
                    "Strong evidence:\n"
                    "- repeated evidence across multiple events\n"
                    "- explicit preference statements\n"
                    "- aggregate frequency patterns\n"
                    "Weak evidence:\n"
                    "- one-off action\n"
                    "- isolated or accidental occurrence\n"
                    "Do NOT:\n"
                    "- infer a habit from only one weak event if stronger repeated evidence exists\n"
                    "- confuse temporary behavior with stable preference\n\n"

                    "[attribute-content-purpose]\n"
                    "Question intent: identify ownership, contents, identity, purpose, attribute, or category.\n"
                    "Strong evidence:\n"
                    "- direct statement of ownership, contents, identity, purpose, or queried attribute\n"
                    "- explicit visual or textual grounding of the queried property\n"
                    "Weak evidence:\n"
                    "- nearby action context\n"
                    "- related discussion without direct attribute grounding\n"
                    "Do NOT:\n"
                    "- replace a direct attribute question with surrounding activity\n"
                    "- infer ownership, content, purpose, or identity from loose association alone\n\n"

                    "Global anti-error rules:\n"
                    "- Do not infer agent ownership from scene participation alone.\n"
                    "- Do not infer intention from topic discussion alone.\n"
                    "- Do not infer source from current location alone.\n"
                    "- Do not infer habits from a single weak event if stronger repeated evidence exists.\n"
                    "- Do not infer attributes from nearby actions when direct grounding exists.\n"
                    "- When direct evidence and broad contextual evidence conflict, prefer direct evidence.\n"
                    "- Use role scores as hints, not hard constraints.\n"
                    "- Prefer a smaller set of directly relevant events over a larger set of vaguely related events.\n"
                    "- If a question has a critical constraint (actor, source, time, intention, ownership, identity), at least one selected event should directly ground that constraint."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{query_with_time}\n\n"
                    f"Candidate Event Packets:\n{json.dumps(selector_candidates, ensure_ascii=False, indent=2)}\n\n"
                    f"Select the best {final_top_k} candidates.\n\n"

                    "Selection goals:\n"
                    "- Choose complementary evidence, not repetitive evidence.\n"
                    "- Retain at least one event that directly grounds the core predicate of the question.\n"
                    "- If the question requires prior-state or source evidence, retain the event that directly establishes that prior state, even if it is earlier and less salient.\n"
                    "- If the question requires intention or decision evidence, retain explicit commitment or decision evidence rather than topic-related discussion.\n"
                    "- If the question requires identifying an actor, retain explicit actor evidence.\n"
                    "- If the question requires temporal comparison, enforce the temporal constraint strictly.\n"
                    "- If the question requires a stable habit or preference, prefer repeated or aggregate evidence over one-off evidence.\n"
                    "- If the question requires ownership, contents, identity, purpose, or attribute, prefer direct grounding over surrounding context.\n\n"

                    "Output requirements:\n"
                    "- Infer the correct question_family first.\n"
                    "- Then select the best candidates.\n"
                    "- The reason must explain why the selected events satisfy the core predicate better than merely related events.\n\n"

                    "Return ONLY JSON in this format:\n"
                    "{"
                    "\"question_family\": \"...\", "
                    "\"selected_indices\": [..], "
                    "\"selected_doc_ids\": [..], "
                    "\"reason\": \"...\""
                    "}"
                ),
            },
        ]

        try:
            response = self.respond_llm_model.generate(prompt)
            logger.info("LLM event selector raw response: %s", response)
        except Exception as e:
            logger.error(f"LLM event selector failed: {e}")
            return [], ""

        valid_doc_ids = [c["doc_id"] for c in selector_candidates]
        selected = self._parse_event_selector_response(
            response=response,
            valid_doc_ids=valid_doc_ids,
            num_candidates=len(selector_candidates),
        )
        meta = self._extract_selector_metadata(response)
        logger.info("LLM event selector question_family: %s", meta.get("question_family", ""))
        selector_reason = self._extract_selector_reason(response)
        return selected[:final_top_k], selector_reason

    # -----------------------------------------------------
    # direct fusion answer pipeline
    # -----------------------------------------------------

    # def answer(
    #     self,
    #     query: str,
    #     choices: Optional[Dict[str, str]] = None,
    #     until_time: Optional[int] = None,
    # ) -> QAResult:
    #     if until_time and until_time > self.indexed_time:
    #         self.index(until_time)

    #     full_query = self._build_query_with_time(
    #         query=query,
    #         choices=choices,
    #         until_time=until_time,
    #     )

    #     episodic_ranked = self.episodic_memory.retrieve_ranked(
    #         query=query,
    #         top_k_per_granularity={
    #             "30sec": max(self.episodic_top_k * 4, 10),
    #             "3min": max(self.episodic_top_k * 3, 6),
    #             "10min": max(self.episodic_top_k * 2, 5),
    #             "1h": max(self.episodic_top_k, 3),
    #         },
    #         dedup_by_doc_id=True,
    #     )
    #     semantic_entries = self.semantic_memory.retrieve(
    #         query=query,
    #         top_k=max(self.semantic_top_k, self.episodic_top_k * 3),
    #         as_context=False,
    #     )
    #     if isinstance(semantic_entries, str):
    #         semantic_entries = []

    #     logger.info(
    #         "Retrieved %d episodic candidates and %d semantic facts",
    #         len(episodic_ranked),
    #         len(semantic_entries),
    #     )

    #     if episodic_ranked:
    #         logger.info(
    #             "Top episodic candidates: %s",
    #             [
    #                 {
    #                     "doc_id": entry.doc_id,
    #                     "granularity": entry.granularity,
    #                     "score": round(score, 4),
    #                 }
    #                 for entry, score in episodic_ranked[:8]
    #             ],
    #         )

    #     if semantic_entries:
    #         logger.info(
    #             "Top semantic facts: %s",
    #             [
    #                 {
    #                     "fact_id": entry.id,
    #                     "triple": entry.triple,
    #                     "support_count": entry.support_count,
    #                     "confidence": round(float(entry.confidence), 4),
    #                 }
    #                 for entry in semantic_entries[:8]
    #             ],
    #         )

    #     episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
    #     semantic_projected, semantic_support_map = self._project_semantic_to_30s(semantic_entries)

    #     candidate_doc_ids = set(episodic_projected.keys())
    #     if not candidate_doc_ids:
    #         logger.warning("No candidate events found from episodic retrieval")
    #         candidate_doc_ids = set()
    #         for entry, _ in episodic_ranked[: self.episodic_top_k]:
    #             for doc_id in self.episodic_memory.expand_entry_to_30s_doc_ids(entry):
    #                 if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None:
    #                     candidate_doc_ids.add(doc_id)
    #         if not candidate_doc_ids:
    #             return QAResult(
    #                 question=query,
    #                 answer="Unable to retrieve relevant evidence",
    #                 retrieved_items=[],
    #                 round_history=[],
    #                 num_rounds=1,
    #             )

    #     episodic_norm = self._normalize_dict({doc_id: episodic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})
    #     semantic_norm = self._normalize_dict({doc_id: semantic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})

    #     anchor_scores: Dict[str, float] = {doc_id: episodic_norm.get(doc_id, 0.0) for doc_id in candidate_doc_ids}
    #     ranked_doc_ids = [doc_id for doc_id, _ in sorted(anchor_scores.items(), key=lambda x: -x[1])]

    #     logger.info(
    #         "Top episodic anchor scores before selector: %s",
    #         [
    #             {
    #                 "doc_id": doc_id,
    #                 "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
    #                 "sem": round(semantic_norm.get(doc_id, 0.0), 4),
    #             }
    #             for doc_id in ranked_doc_ids[:8]
    #         ],
    #     )

    #     selector_candidates = self._build_event_selector_candidates(
    #         query=query,
    #         episodic_norm=episodic_norm,
    #         semantic_norm=semantic_norm,
    #         semantic_support_map=semantic_support_map,
    #     )
    #     logger.info(
    #         "Built %d selector candidates: %s",
    #         len(selector_candidates),
    #         [
    #             {
    #                 "index": c["index"],
    #                 "doc_id": c["doc_id"],
    #                 "primary_role": c["primary_role"],
    #                 "ep": c["episodic_score"],
    #                 "tr": c["trigger_score"],
    #                 "ant": c["antecedent_score"],
    #                 "bro": c["broader_score"],
    #             }
    #             for c in selector_candidates
    #         ],
    #     )

    #     selected_doc_ids, selector_reason = self._select_top_events_with_llm(
    #         query=query,
    #         choices=choices,
    #         until_time=until_time,
    #         selector_candidates=selector_candidates,
    #         final_top_k=max(self.episodic_top_k, 1),
    #     )

    #     if not selected_doc_ids:
    #         logger.info("LLM event selector returned no valid doc_ids, fallback to coarse ranking")
    #         top_doc_ids = ranked_doc_ids[: max(self.episodic_top_k, 1)]
    #         selector_reason = (
    #             "Selector fallback: no valid doc_ids were parsed from the selector output. "
    #             "Coarse episodic ranking was used instead."
    #         )
    #     else:
    #         top_doc_ids = []
    #         for doc_id in selected_doc_ids:
    #             if doc_id not in top_doc_ids:
    #                 top_doc_ids.append(doc_id)
    #         if len(top_doc_ids) < max(self.episodic_top_k, 1):
    #             for doc_id in ranked_doc_ids:
    #                 if doc_id not in top_doc_ids:
    #                     top_doc_ids.append(doc_id)
    #                 if len(top_doc_ids) >= max(self.episodic_top_k, 1):
    #                     break

    #     logger.info("Selector reason summary: %s", selector_reason)

    #     logger.info(
    #         "Final selected event anchors: %s",
    #         [
    #             {
    #                 "doc_id": doc_id,
    #                 "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
    #                 "sem": round(semantic_norm.get(doc_id, 0.0), 4),
    #             }
    #             for doc_id in top_doc_ids
    #         ],
    #     )

    #     event_packets = []
    #     for doc_id in top_doc_ids:
    #         packet = self._build_event_packet(
    #             doc_id=doc_id,
    #             score=anchor_scores.get(doc_id, 0.0),
    #             supporting_facts=semantic_support_map.get(doc_id, []),
    #         )
    #         if packet:
    #             event_packets.append(packet)

    #     logger.info("Built %d event packets", len(event_packets))
    #     for doc_id in top_doc_ids:
    #         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
    #         if entry is not None:
    #             logger.info(
    #                 "Event packet anchor %s | time=%s-%s | text=%s",
    #                 doc_id,
    #                 entry.start_time,
    #                 entry.end_time,
    #                 entry.text[:120].replace("\n", " "),
    #             )

    #     semantic_context = self._build_semantic_context(semantic_entries, top_n=min(5, self.semantic_top_k))

    #     retrieved_items: List[RetrievedItem] = []
    #     if event_packets:
    #         retrieved_items.append(
    #             RetrievedItem(
    #                 memory_type="episodic",
    #                 content="\n\n".join(event_packets),
    #                 query=query,
    #                 round_num=1,
    #             )
    #         )
    #     if semantic_context:
    #         retrieved_items.append(
    #             RetrievedItem(
    #                 memory_type="semantic",
    #                 content=semantic_context,
    #                 query=query,
    #                 round_num=1,
    #             )
    #         )

    #     event_images = self.visual_memory.get_event_images(
    #         top_doc_ids,
    #         max_images_per_event=max(self.visual_top_k, 1),
    #     )

    #     if event_images:
    #         num_event_with_images = len(event_images)
    #         num_total_images = sum(len(v) for v in event_images.values())
    #         logger.info(
    #             "Loaded visual evidence for %d events, %d images total",
    #             num_event_with_images,
    #             num_total_images,
    #         )
    #         for doc_id in top_doc_ids:
    #             logger.info("Visual images for %s: %d", doc_id, len(event_images.get(doc_id, [])))

    #         all_images = []
    #         for doc_id in top_doc_ids:
    #             all_images.extend(event_images.get(doc_id, []))

    #         if all_images:
    #             logger.info("Sending %d images to QA", len(all_images))
    #             retrieved_items.append(
    #                 RetrievedItem(
    #                     memory_type="visual",
    #                     content=all_images,
    #                     query=query,
    #                     round_num=1,
    #                 )
    #             )
    #     else:
    #         logger.info("No visual evidence found for final event anchors")

    #     round_history = self._build_round_history(query, top_doc_ids, semantic_entries)

    #     try:
    #         qa_prompt = self.prompt_template_manager.render("qa_egolife")
    #     except Exception as e:
    #         logger.error(f"Failed to load qa_egolife template: {e}")
    #         raise

    #     qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
    #     qa_content.append({
    #         "type": "text",
    #         "text": (
    #             "Selector summary:\n"
    #             f"Chosen event anchors: {top_doc_ids}\n"
    #             f"Selector reason: {selector_reason}\n"
    #             "The selected event anchors were chosen because they form the strongest evidence chain for this question.\n"
    #             "Use these selected events as the primary basis for answering.\n"
    #             "Do not override a clearly supported conclusion from the selected evidence with a weaker alternative."
    #         )
    #     })
    #     qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
    #     if choices:
    #         grounding_lines = []
    #         narrator_labels = []
    #         for k, v in sorted(choices.items()):
    #             v_norm = str(v).strip().lower()
    #             if v_norm in {"me", "myself", "self", "narrator", "the narrator", "speaker"}:
    #                 narrator_labels.append(k)

    #         if narrator_labels:
    #             grounding_lines.append(
    #                 "Important grounding: in this egocentric first-person video, the pronouns 'I', 'me', 'my', and 'myself' refer to the narrator / camera wearer."
    #             )
    #             grounding_lines.append(
    #                 f"If the evidence says the narrator ('I') performed the action, prefer the corresponding choice(s): {', '.join(narrator_labels)}."
    #             )

    #         grounding_lines.append(
    #             "Answer selection rule: choose the option best supported by the retrieved evidence and the selector summary above."
    #         )
    #         grounding_lines.append(
    #             "If the selector reason and selected events clearly support a specific option, do not override it with a weaker alternative."
    #         )
    #         grounding_lines.append(
    #             "Please provide only the final answer from the choices given (e.g., A, B, C, or D)."
    #         )

    #         qa_content.append({"type": "text", "text": "\n" + "\n".join(grounding_lines)})

    #     num_text_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "text")
    #     num_image_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "image")
    #     logger.info(
    #         "QA payload prepared: %d text blocks, %d image blocks, %d retrieved items",
    #         num_text_blocks,
    #         num_image_blocks,
    #         len(retrieved_items),
    #     )

    #     qa_messages = copy.deepcopy(qa_prompt)
    #     qa_messages.append({"role": "user", "content": qa_content})

    #     try:
    #         answer = self.respond_llm_model.generate(qa_messages)
    #     except Exception as e:
    #         logger.error(f"Answer generation failed: {e}")
    #         answer = "Unable to generate answer"

    #     return QAResult(
    #         question=query,
    #         answer=answer,
    #         retrieved_items=retrieved_items,
    #         round_history=round_history,
    #         num_rounds=1,
    #     )

    def answer(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time: Optional[int] = None,
        answer_mode: str = "auto",
        use_image_evidence: bool = True,
        max_image_frames: int = 4,
        stream_handler: Any = None,
        prompt_context: Optional[str] = None,
        generate_answer: bool = True,
    ) -> QAResult:
        if answer_mode not in {"auto", "open_ended", "multiple_choice"}:
            raise ValueError(f"Unsupported answer_mode: {answer_mode}")

        effective_answer_mode = answer_mode
        if effective_answer_mode == "auto":
            effective_answer_mode = "multiple_choice" if choices else "open_ended"
        elif effective_answer_mode == "multiple_choice" and not choices:
            logger.warning("multiple_choice answer_mode requested without choices; falling back to open_ended")
            effective_answer_mode = "open_ended"

        if until_time and until_time > self.indexed_time:
            self.index(until_time)

        query_for_prompt = query
        if prompt_context:
            query_for_prompt = f"{prompt_context.strip()}\n\nUser question:\n{query}"

        full_query = self._build_query_with_time(
            query=query_for_prompt,
            choices=choices,
            until_time=until_time,
        )

        episodic_ranked = self.episodic_memory.retrieve_ranked(
            query=query,
            top_k_per_granularity={
                "30sec": max(self.episodic_top_k * 4, 10),
                "3min": max(self.episodic_top_k * 3, 6),
                "10min": max(self.episodic_top_k * 2, 5),
                "1h": max(self.episodic_top_k, 3),
            },
            dedup_by_doc_id=True,
        )
        semantic_entries = self.semantic_memory.retrieve(
            query=query,
            top_k=max(self.semantic_top_k, self.episodic_top_k * 3),
            as_context=False,
        )
        if isinstance(semantic_entries, str):
            semantic_entries = []

        logger.info(
            "Retrieved %d episodic candidates and %d semantic facts",
            len(episodic_ranked),
            len(semantic_entries),
        )

        if episodic_ranked:
            logger.info(
                "Top episodic candidates: %s",
                [
                    {
                        "doc_id": entry.doc_id,
                        "granularity": entry.granularity,
                        "score": round(score, 4),
                    }
                    for entry, score in episodic_ranked[:8]
                ],
            )

        if semantic_entries:
            logger.info(
                "Top semantic facts: %s",
                [
                    {
                        "fact_id": entry.id,
                        "triple": entry.triple,
                        "support_count": entry.support_count,
                        "confidence": round(float(entry.confidence), 4),
                    }
                    for entry in semantic_entries[:8]
                ],
            )

        episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
        semantic_projected, semantic_support_map = self._project_semantic_to_30s(semantic_entries)

        candidate_doc_ids = set(episodic_projected.keys())
        if not candidate_doc_ids:
            logger.warning("No candidate events found from episodic retrieval")
            candidate_doc_ids = set()
            for entry, _ in episodic_ranked[: self.episodic_top_k]:
                for doc_id in self.episodic_memory.expand_entry_to_30s_doc_ids(entry):
                    if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None:
                        candidate_doc_ids.add(doc_id)
            if not candidate_doc_ids:
                return QAResult(
                    question=query,
                    answer="Unable to retrieve relevant evidence",
                    retrieved_items=[],
                    round_history=[],
                    num_rounds=1,
                    answer_mode=effective_answer_mode,
                )

        episodic_norm = self._normalize_dict(
            {doc_id: episodic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids}
        )
        semantic_norm = self._normalize_dict(
            {doc_id: semantic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids}
        )

        anchor_scores: Dict[str, float] = {
            doc_id: episodic_norm.get(doc_id, 0.0) for doc_id in candidate_doc_ids
        }
        ranked_doc_ids = [doc_id for doc_id, _ in sorted(anchor_scores.items(), key=lambda x: -x[1])]

        logger.info(
            "Top episodic anchor scores before selector: %s",
            [
                {
                    "doc_id": doc_id,
                    "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
                    "sem": round(semantic_norm.get(doc_id, 0.0), 4),
                }
                for doc_id in ranked_doc_ids[:8]
            ],
        )

        retrieval_start = time.perf_counter()
        selector_candidates = self._build_event_selector_candidates(
            query=query,
            episodic_norm=episodic_norm,
            semantic_norm=semantic_norm,
            semantic_support_map=semantic_support_map,
        )
        logger.info(
            "Built %d selector candidates: %s",
            len(selector_candidates),
            [
                {
                    "index": c["index"],
                    "doc_id": c["doc_id"],
                    "primary_role": c["primary_role"],
                    "ep": c["episodic_score"],
                    "tr": c["trigger_score"],
                    "ant": c["antecedent_score"],
                    "bro": c["broader_score"],
                }
                for c in selector_candidates
            ],
        )

        selector_start = time.perf_counter()
        retrieval_ms = int(round((selector_start - retrieval_start) * 1000))
        selected_doc_ids, selector_reason = self._select_top_events_with_llm(
            query=query,
            choices=choices,
            until_time=until_time,
            selector_candidates=selector_candidates,
            final_top_k=max(self.episodic_top_k, 1),
        )
        selector_ms = int(round((time.perf_counter() - selector_start) * 1000))

        if not selected_doc_ids:
            logger.info("LLM event selector returned no valid doc_ids, fallback to coarse ranking")
            top_doc_ids = ranked_doc_ids[: max(self.episodic_top_k, 1)]
            selector_reason = (
                "Selector fallback: no valid doc_ids were parsed from the selector output. "
                "Coarse episodic ranking was used instead."
            )
        else:
            # Keep selector output only. Do NOT globally pad from coarse ranking.
            top_doc_ids = []
            for doc_id in selected_doc_ids:
                if doc_id not in top_doc_ids:
                    top_doc_ids.append(doc_id)

            top_doc_ids = top_doc_ids[: max(self.episodic_top_k, 1)]

            logger.info(
                "Selector returned %d valid doc_ids; keeping selector-only events without global padding: %s",
                len(top_doc_ids),
                top_doc_ids,
            )

        logger.info("Selector reason summary: %s", selector_reason)

        logger.info(
            "Final selected event anchors: %s",
            [
                {
                    "doc_id": doc_id,
                    "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
                    "sem": round(semantic_norm.get(doc_id, 0.0), 4),
                }
                for doc_id in top_doc_ids
            ],
        )

        pack_start = time.perf_counter()
        event_packets = []
        for doc_id in top_doc_ids:
            packet = self._build_event_packet(
                doc_id=doc_id,
                score=anchor_scores.get(doc_id, 0.0),
                supporting_facts=semantic_support_map.get(doc_id, []),
            )
            if packet:
                event_packets.append(packet)

        logger.info("Built %d event packets", len(event_packets))
        for doc_id in top_doc_ids:
            entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
            if entry is not None:
                logger.info(
                    "Event packet anchor %s | time=%s-%s | text=%s",
                    doc_id,
                    entry.start_time,
                    entry.end_time,
                    entry.text[:120].replace("\n", " "),
                )

        logger.info(
            "QA evidence packaging | num_selected_events=%d | selector_only=%s | global_semantic_context=%s",
            len(top_doc_ids),
            True,
            False,
        )

        retrieved_items: List[RetrievedItem] = []
        if event_packets:
            retrieved_items.append(
                RetrievedItem(
                    memory_type="episodic",
                    content="\n\n".join(event_packets),
                    query=query,
                    round_num=1,
                )
            )

        event_images: Dict[str, List[Any]] = {}
        if use_image_evidence and generate_answer:
            try:
                image_limit = max(1, int(max_image_frames or self.visual_top_k or 1))
            except Exception:
                image_limit = max(self.visual_top_k, 1)
            event_images = self.visual_memory.get_event_images(
                top_doc_ids,
                max_images_per_event=image_limit,
            )

        if event_images:
            num_event_with_images = len(event_images)
            num_total_images = sum(len(v) for v in event_images.values())
            logger.info(
                "Loaded visual evidence for %d events, %d images total",
                num_event_with_images,
                num_total_images,
            )
            for doc_id in top_doc_ids:
                logger.info("Visual images for %s: %d", doc_id, len(event_images.get(doc_id, [])))

            all_images = []
            for doc_id in top_doc_ids:
                all_images.extend(event_images.get(doc_id, []))

            if all_images:
                logger.info("Sending %d images to QA", len(all_images))
                retrieved_items.append(
                    RetrievedItem(
                        memory_type="visual",
                        content=all_images,
                        query=query,
                        round_num=1,
                    )
                )
        else:
            logger.info("No visual evidence found for final event anchors")

        round_history = self._build_round_history(query, top_doc_ids, semantic_entries)

        pack_ms = int(round((time.perf_counter() - pack_start) * 1000))
        answer = ""
        model_response_text = ""
        error_debug = ""
        fallback_used = False
        llm_debug: Dict[str, Any] = {}
        generation_ms = 0
        qa_template_name = "qa_egolife" if effective_answer_mode == "multiple_choice" else "qa_egolife_open"
        if generate_answer:
            # qa_template_name = "qa_egolife" if effective_answer_mode == "multiple_choice" else "qa_egolife_open"
            try:
                qa_prompt = self.prompt_template_manager.render(qa_template_name)
            except Exception as e:
                logger.error(f"Failed to load {qa_template_name} template: {e}")
                raise

            qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
            qa_content.append({
                "type": "text",
                "text": (
                    "Selector summary:\n"
                    f"Chosen event anchors: {top_doc_ids}\n"
                    f"Selector reason: {selector_reason}\n"
                    "The selected event anchors were chosen because they form the strongest evidence chain for this question.\n"
                    "Use these selected events as the primary basis for answering.\n"
                    "Do not override a clearly supported conclusion from the selected evidence with a weaker alternative."
                )
            })
            qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))

            if effective_answer_mode == "multiple_choice" and choices:
                grounding_lines = []
                narrator_labels = []
                for k, v in sorted(choices.items()):
                    v_norm = str(v).strip().lower()
                    if v_norm in {"me", "myself", "self", "narrator", "the narrator", "speaker"}:
                        narrator_labels.append(k)

                if narrator_labels:
                    grounding_lines.append(
                        "Important grounding: in this egocentric first-person video, the pronouns 'I', 'me', 'my', and 'myself' refer to the narrator / camera wearer."
                    )
                    grounding_lines.append(
                        f"If the evidence says the narrator ('I') performed the action, prefer the corresponding choice(s): {', '.join(narrator_labels)}."
                    )

                grounding_lines.append(
                    "Answer selection rule: choose the option best supported by the retrieved evidence and the selector summary above."
                )
                grounding_lines.append(
                    "If the selector reason and selected events clearly support a specific option, do not override it with a weaker alternative."
                )
                grounding_lines.append(
                    "Please provide only the final answer from the choices given (e.g., A, B, C, or D)."
                )

                qa_content.append({"type": "text", "text": "\n" + "\n".join(grounding_lines)})

            num_text_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "text")
            num_image_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "image")
            logger.info(
                "QA payload prepared: %d text blocks, %d image blocks, %d retrieved items",
                num_text_blocks,
                num_image_blocks,
                len(retrieved_items),
            )

            qa_messages = copy.deepcopy(qa_prompt)
            qa_messages.append({"role": "user", "content": qa_content})
            generation_start = time.perf_counter()
            try:
                answer = self.respond_llm_model.generate(qa_messages)
                model_response_text = "" if answer is None else str(answer).strip()
                llm_debug = dict(getattr(self.respond_llm_model, "last_debug", {}) or {})
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer = "Unable to generate answer"
                error_debug = f"{type(e).__name__}: {e}"
            generation_ms = int(round((time.perf_counter() - generation_start) * 1000))
        else:
            answer = ""

        semantic_fact_ids: List[str] = []
        seen_fact_ids = set()
        for doc_id in top_doc_ids:
            for fact in semantic_support_map.get(doc_id, []) or []:
                fact_id = str(getattr(fact, "id", "") or "")
                if fact_id and fact_id not in seen_fact_ids:
                    seen_fact_ids.add(fact_id)
                    semantic_fact_ids.append(fact_id)
        if not semantic_fact_ids:
            for fact in semantic_entries[: min(len(semantic_entries), max(self.semantic_top_k, 5))]:
                fact_id = str(getattr(fact, "id", "") or "")
                if fact_id and fact_id not in seen_fact_ids:
                    seen_fact_ids.add(fact_id)
                    semantic_fact_ids.append(fact_id)

        visual_event_image_counts = {doc_id: len(event_images.get(doc_id, [])) for doc_id in top_doc_ids if event_images.get(doc_id, [])}

        return QAResult(
            question=query,
            answer=answer,
            retrieved_items=retrieved_items,
            round_history=round_history,
            num_rounds=1,
            answer_mode=effective_answer_mode,
            qa_template_name=qa_template_name,
            selected_doc_ids=top_doc_ids,
            selector_reason=selector_reason,
            semantic_fact_ids=semantic_fact_ids,
            visual_event_image_counts=visual_event_image_counts,
            model_response_text=model_response_text,
            error_debug=error_debug,
            fallback_used=fallback_used,
            llm_debug=llm_debug,
            timing_ms={
                "retrieval_ms": retrieval_ms,
                "selector_ms": selector_ms,
                "pack_ms": pack_ms,
                "generation_ms": generation_ms if generate_answer else None,
                "generate_answer": bool(generate_answer),
            },
        )

    # -----------------------------------------------------
    # lifecycle helpers
    # -----------------------------------------------------

    def reset_index(self) -> None:
        self.episodic_memory.reset_index()
        self.semantic_memory.reset_index()
        self.visual_memory.reset_index()
        self.indexed_time = 0
        logger.info("All memory indices reset")

    def cleanup(self) -> None:
        self.semantic_memory.cleanup()
        self.visual_memory.cleanup()
        logger.info("Memory cleanup complete")

    def get_indexed_time(self) -> str:
        return transform_timestamp(str(self.indexed_time))

    def set_retrieval_top_k(
        self,
        episodic: Optional[int] = None,
        semantic: Optional[int] = None,
        visual: Optional[int] = None,
    ) -> None:
        if episodic is not None:
            self.episodic_top_k = episodic
        if semantic is not None:
            self.semantic_top_k = semantic
        if visual is not None:
            self.visual_top_k = visual


# """
# WorldMemory: unified event-centric memory system.

# This version implements:
# - multiscale episodic retrieval with dense RAG + graph-aware rerank
# - soft-grouped event candidate pool (trigger / antecedent / broader-context)
# - LLM event selector over a coarse candidate pool
# - semantic memory as support only (not primary event routing)
# - visual evidence only for final selected event anchors (keyframes)
# """

# import copy
# import json
# import logging
# import math
# import re
# from collections import defaultdict
# from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# from PIL import Image

# from ..embedding import EmbeddingModel
# from ..llm import LLMModel, PromptTemplateManager
# from .episodic.EpisodicMemory_rag import CaptionEntryRAG, EpisodicMemoryRAG
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .utils import *
# from .visual import VisualMemory

# logger = logging.getLogger(__name__)


# STOPWORDS = {
#     "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
#     "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
#     "what", "which", "who", "whom", "when", "where", "why", "how",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
#     "this", "that", "these", "those", "it", "its"
# }


# class WorldMemory:
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         retriever_llm_model: LLMModel,
#         respond_llm_model: Optional[LLMModel] = None,
#         prompt_template_manager: Optional[PromptTemplateManager] = None,
#         episodic_granularities: Optional[List[str]] = None,
#         episodic_cache_tag: Optional[str] = None,
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors

#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=embedding_model,
#             llm_model=retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=episodic_granularities,
#             cache_tag=episodic_cache_tag,
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=embedding_model)

#         self.indexed_time: int = 0

#         self.episodic_top_k: int = 3
#         self.semantic_top_k: int = 10
#         self.visual_top_k: int = 3

#         # anchor projection weights across granularities
#         self.anchor_weight_30s = 1.00
#         self.anchor_weight_3min = 0.65
#         self.anchor_weight_10min = 0.45
#         self.anchor_weight_1h = 0.30

#         # soft-group selector pool sizes
#         # self.selector_global_top_n = 10
#         # self.selector_trigger_top_n = 4
#         # self.selector_antecedent_top_n = 4
#         # self.selector_broader_top_n = 3
#         # self.selector_max_candidates = 12
#         self.selector_global_top_n = 12
#         self.selector_trigger_top_n = 4
#         self.selector_antecedent_top_n = 4
#         self.selector_broader_top_n = 3
#         self.selector_max_candidates = 16

#         # refusal / invalid-output retry controls
#         self.selector_refusal_retry_max = 2
#         self.answer_refusal_retry_max = 2

#     # -----------------------------------------------------
#     # query formatting
#     # -----------------------------------------------------

#     def _build_query_with_time(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> str:
#         lines = [f"Query: {query}"]
#         if until_time is not None:
#             lines.append(f"Query Time: {transform_timestamp(str(until_time))}")
#             lines.append(
#                 "Important: Interpret all relative temporal expressions "
#                 '(e.g. "before", "after", "earlier", "later", "recently", '
#                 '"a few hours ago", "first", "last") relative to Query Time.'
#             )
#         if choices:
#             choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
#             lines.append(f"Choices: {choices_str}")
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # loading
#     # -----------------------------------------------------

#     def load_episodic_captions(
#         self,
#         caption_files: Optional[Dict[str, str]] = None,
#         caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
#     ) -> None:
#         if caption_files:
#             self.episodic_memory.load_captions_from_files(caption_files)
#         if caption_data:
#             self.episodic_memory.load_captions_from_data(caption_data)

#     def load_episodic_sidecar(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_files or graph_files:
#             self.episodic_memory.load_sidecar_from_files(
#                 triplet_files=triplet_files,
#                 graph_files=graph_files,
#             )
#         if triplet_data or graph_data:
#             self.episodic_memory.load_sidecar_from_data(
#                 triplet_data=triplet_data,
#                 graph_data=graph_data,
#             )

#     def load_semantic_triples(
#         self,
#         file_path: Optional[str] = None,
#         data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if file_path:
#             self.semantic_memory.load_triples_from_file(file_path)
#         if data:
#             self.semantic_memory.load_triples_from_data(data)

#     def load_visual_clips(
#         self,
#         embeddings_path: Optional[str] = None,
#         clips_path: Optional[str] = None,
#         clips_data: Optional[List[Dict[str, Any]]] = None,
#     ) -> None:
#         if embeddings_path:
#             self.visual_memory.load_embeddings_from_file(embeddings_path)
#         if clips_path:
#             self.visual_memory.load_clips_from_file(clips_path)
#         if clips_data:
#             self.visual_memory.load_clips_from_data(clips_data)

#     def prepare_episodic_dense_index(self, force_rebuild: bool = False) -> None:
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=force_rebuild)

#     # -----------------------------------------------------
#     # indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
#             return

#         logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")

#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=False)

#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
#         self.indexed_time = until_time
#         logger.info("Indexing complete for all memory types")

#     # -----------------------------------------------------
#     # helpers
#     # -----------------------------------------------------

#     def _tokenize(self, text: str) -> Set[str]:
#         toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
#         return {t for t in toks if len(t) > 1 and t not in STOPWORDS}

#     def _normalize_dict(self, score_map: Dict[str, float]) -> Dict[str, float]:
#         if not score_map:
#             return {}
#         values = list(score_map.values())
#         mn, mx = min(values), max(values)
#         if abs(mx - mn) < 1e-8:
#             return {k: 1.0 for k in score_map}
#         return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}

#     def _episodic_weight_for_granularity(self, granularity: str) -> float:
#         if granularity == "30sec":
#             return self.anchor_weight_30s
#         if granularity == "3min":
#             return self.anchor_weight_3min
#         if granularity == "10min":
#             return self.anchor_weight_10min
#         if granularity == "1h":
#             return self.anchor_weight_1h
#         return 0.30

#     def _parse_timestamp_int(self, ts: int) -> Tuple[int, int, int, int]:
#         ts_str = str(ts)
#         day = int(ts_str[0])
#         hh = int(ts_str[1:3])
#         mm = int(ts_str[3:5])
#         ss = int(ts_str[5:7])
#         return day, hh, mm, ss

#     def _timestamp_to_seconds(self, ts: int) -> int:
#         day, hh, mm, ss = self._parse_timestamp_int(ts)
#         return day * 86400 + hh * 3600 + mm * 60 + ss

#     def _entry_center_seconds(self, entry: CaptionEntryRAG) -> float:
#         start_ts, end_ts = entry.timestamp_int
#         return 0.5 * (self._timestamp_to_seconds(start_ts) + self._timestamp_to_seconds(end_ts))

#     def _overlap_ratio(self, a: Set[str], b: Set[str]) -> float:
#         if not a or not b:
#             return 0.0
#         return len(a & b) / max(1, len(a))

#     def _event_tokens(self, doc_id: str) -> Set[str]:
#         toks: Set[str] = set()
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return toks
#         toks.update(self._tokenize(entry.text))
#         toks.update(self._tokenize(entry.visual_summary))
#         for line in entry.metadata.get("critical_speech_lines", []) or []:
#             toks.update(self._tokenize(str(line)))

#         parent = None
#         if hasattr(self.episodic_memory, "get_parent_caption"):
#             parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#         if parent is not None:
#             toks.update(self._tokenize(parent.text))
#             toks.update(self._tokenize(parent.visual_summary))
#             for line in parent.metadata.get("critical_speech_lines", []) or []:
#                 toks.update(self._tokenize(str(line)))

#         for tri in self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")[:8]:
#             if isinstance(tri, list) and len(tri) == 3:
#                 toks.update(self._tokenize(" ".join(map(str, tri))))

#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         if visual_entry is not None:
#             toks.update(self._tokenize(getattr(visual_entry, "keyframe_caption", "")))
#             scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#             if isinstance(scene_summary, dict):
#                 toks.update(self._tokenize(json.dumps(scene_summary, ensure_ascii=False)))
#             for obj in getattr(visual_entry, "visual_objects", []) or []:
#                 toks.update(self._tokenize(str(obj)))

#         return toks

#     def _project_episodic_candidates_to_30s(
#         self,
#         candidates: List[Tuple[CaptionEntryRAG, float]],
#     ) -> Dict[str, float]:
#         projected = defaultdict(float)
#         if not candidates:
#             return projected

#         for rank, (entry, score) in enumerate(candidates):
#             rank_bonus = 1.0 / (rank + 1)
#             base = float(score) * rank_bonus * self._episodic_weight_for_granularity(entry.granularity)
#             target_doc_ids = self.episodic_memory.expand_entry_to_30s_doc_ids(entry)
#             target_doc_ids = [
#                 doc_id for doc_id in target_doc_ids
#                 if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
#             ]
#             if not target_doc_ids:
#                 continue
#             denom = math.sqrt(len(target_doc_ids))
#             for doc_id in target_doc_ids:
#                 projected[doc_id] += base / denom
#         return projected

#     def _project_semantic_to_30s(
#         self,
#         semantic_entries: List[SemanticTripleEntry],
#     ) -> Tuple[Dict[str, float], Dict[str, List[SemanticTripleEntry]]]:
#         projected = defaultdict(float)
#         support_map: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)
#         if not semantic_entries:
#             return projected, support_map

#         for rank, entry in enumerate(semantic_entries):
#             candidate_roots = (
#                 list(getattr(entry, "provenance_root_ids", []) or [])
#                 or list(getattr(entry, "source_doc_ids", []) or [])
#                 or list(entry.evidence_event_ids or [])
#             )
#             if not candidate_roots:
#                 continue

#             rank_bonus = 1.0 / (rank + 1)
#             support_factor = 1.0 + 0.15 * min(int(entry.support_count), 5)
#             conf_factor = 0.7 + 0.3 * float(entry.confidence)
#             base = rank_bonus * support_factor * conf_factor

#             valid_doc_ids: List[str] = []
#             for root_id in candidate_roots:
#                 if self.episodic_memory.get_caption_by_doc_id(root_id, "30sec") is not None:
#                     valid_doc_ids.append(root_id)
#                     continue
#                 root_entry = self.episodic_memory.get_caption_by_doc_id(root_id, granularity=None)
#                 if root_entry is not None:
#                     valid_doc_ids.extend(self.episodic_memory.expand_entry_to_30s_doc_ids(root_entry))

#             valid_doc_ids = [
#                 doc_id for doc_id in dict.fromkeys(valid_doc_ids)
#                 if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
#             ]
#             if not valid_doc_ids:
#                 continue

#             denom = math.sqrt(len(valid_doc_ids))
#             for doc_id in valid_doc_ids:
#                 projected[doc_id] += base / denom
#                 support_map[doc_id].append(entry)

#         return projected, support_map

#     def _build_semantic_context(self, semantic_entries: List[SemanticTripleEntry], top_n: int = 5) -> str:
#         if not semantic_entries:
#             return ""
#         lines = ["Semantic Facts:"]
#         for entry in semantic_entries[:top_n]:
#             lines.append(f"- {entry.to_display_str()}")
#         return "\n".join(lines)

#     def _build_event_packet(
#         self,
#         doc_id: str,
#         score: float,
#         supporting_facts: Optional[List[SemanticTripleEntry]] = None,
#     ) -> str:
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return ""

#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         supporting_facts = supporting_facts or []
#         parent_3min = None
#         if hasattr(self.episodic_memory, "get_parent_caption"):
#             parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min")

#         lines = []
#         lines.append(f"Event Anchor: {doc_id}")
#         lines.append(f"Relevance Score: {score:.4f}")
#         lines.append(entry.to_display_str(include_visual_summary=True))

#         critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#         if critical_lines:
#             lines.append("Critical Speech:")
#             for line in critical_lines[:3]:
#                 if str(line).strip():
#                     lines.append(f"- {line}")

#         if parent_3min is not None and parent_3min.doc_id != doc_id:
#             p_start, p_end = parent_3min.timestamp_int
#             lines.append(
#                 f"3min Context [{transform_timestamp(str(p_start))} - {transform_timestamp(str(p_end))}]: {parent_3min.text}"
#             )
#             if parent_3min.visual_summary:
#                 lines.append(f"3min Visual: {parent_3min.visual_summary}")
#             parent_critical_lines = list(parent_3min.metadata.get("critical_speech_lines", []) or [])
#             if parent_critical_lines:
#                 lines.append("3min Critical Speech:")
#                 for line in parent_critical_lines[:3]:
#                     if str(line).strip():
#                         lines.append(f"- {line}")

#         if visual_entry is not None:
#             if getattr(visual_entry, "keyframe_caption", ""):
#                 lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
#             visual_objects = getattr(visual_entry, "visual_objects", []) or []
#             if visual_objects:
#                 lines.append("Visual Objects: " + ", ".join(visual_objects[:8]))
#             scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#             if isinstance(scene_summary, dict):
#                 dominant_scene = scene_summary.get("dominant_scene", "")
#                 if dominant_scene:
#                     lines.append(f"Scene: {dominant_scene}")

#         if triplets:
#             lines.append("Episodic Triplets:")
#             for tri in triplets[:6]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if supporting_facts:
#             lines.append("Supporting Semantic Facts:")
#             for fact in supporting_facts[:3]:
#                 lines.append(f"- {fact.to_display_str()}")

#         return "\n".join(lines)

#     def _build_round_history(
#         self,
#         query: str,
#         top_doc_ids: List[str],
#         semantic_entries: List[SemanticTripleEntry],
#     ) -> List[Dict[str, Any]]:
#         return [{
#             "round_num": 1,
#             "decision": "search",
#             "memory_type": "episodic+semantic",
#             "search_query": query,
#             "retrieved_content": (
#                 f"Top events: {top_doc_ids}\n"
#                 f"Top semantic facts: {[e.id for e in semantic_entries[:5]]}"
#             ),
#         }]

#     def _render_retrieved_items_for_qa(self, retrieved_items: List[RetrievedItem]) -> List[Dict[str, Any]]:
#         messages = []
#         for item in retrieved_items:
#             if item.memory_type in ("episodic", "semantic"):
#                 messages.append({"type": "text", "text": item.content})
#             elif item.memory_type == "visual":
#                 if isinstance(item.content, list):
#                     for img in item.content:
#                         if isinstance(img, Image.Image):
#                             messages.append({"type": "image", "image": img})
#                         elif isinstance(img, dict) and "image" in img:
#                             messages.append({"type": "image", "image": img["image"]})
#         return messages

#     def _normalize_llm_response_text(self, response: Any) -> str:
#         if response is None:
#             return ""
#         if isinstance(response, str):
#             return response.strip()
#         try:
#             return str(response).strip()
#         except Exception:
#             return ""

#     def _is_refusal_like_response(self, response: Any) -> bool:
#         text = self._normalize_llm_response_text(response).lower()
#         if not text:
#             return False

#         refusal_patterns = [
#             r"\bi['’]?m sorry\b",
#             r"\bi am sorry\b",
#             r"\bcannot assist\b",
#             r"\bcan['’]?t assist\b",
#             r"\bcannot help\b",
#             r"\bcan['’]?t help\b",
#             r"\bunable to help\b",
#             r"\bnot able to assist\b",
#             r"\bcannot comply\b",
#             r"\bcan['’]?t comply\b",
#             r"\bmust refuse\b",
#             r"\bcan['’]?t answer\b",
#             r"\bcannot answer\b",
#             r"\bcan['’]?t provide\b",
#             r"\bcannot provide\b",
#             r"\bwon['’]?t be able to\b",
#         ]
#         return any(re.search(pattern, text, re.IGNORECASE) for pattern in refusal_patterns)

#     def _build_retry_instruction(
#         self,
#         stage: str,
#         previous_response: str,
#         failure_reason: str,
#         choices: Optional[Dict[str, str]] = None,
#     ) -> str:
#         base_lines = [
#             f"Your previous {stage} response was unusable because it was flagged as: {failure_reason}.",
#             "This is a benign long-video QA task over user-provided captions, metadata, and optional images.",
#             "Do not refuse or apologize.",
#             "Answer directly using only the provided evidence.",
#         ]

#         if stage == "selector":
#             base_lines.extend([
#                 "Return ONLY valid JSON.",
#                 "Every selected index or doc_id must come from the provided candidate list.",
#                 "Do not output any prose before or after the JSON.",
#             ])
#         elif stage == "answer":
#             if choices:
#                 base_lines.extend([
#                     "Return only the final answer from the provided choices.",
#                     "Do not include explanation or extra words.",
#                 ])
#             else:
#                 base_lines.extend([
#                     "Give a concise, evidence-grounded answer.",
#                     "If evidence is limited, give the best supported answer rather than refusing.",
#                 ])

#         prev = self._normalize_llm_response_text(previous_response)
#         if prev:
#             base_lines.append(f"Previous response: {prev[:500]}")

#         return "\n".join(base_lines)

#     def _append_retry_instruction_to_prompt(self, prompt: Any, retry_instruction: str) -> Any:
#         if isinstance(prompt, list):
#             updated_prompt = list(prompt)
#             updated_prompt.append({"role": "user", "content": retry_instruction})
#             return updated_prompt
#         if isinstance(prompt, str):
#             return prompt + "\n\n" + retry_instruction
#         return prompt

#     def _generate_with_refusal_retry(
#         self,
#         model: LLMModel,
#         prompt: Any,
#         stage: str,
#         max_retries: int,
#         validator: Optional[Callable[[str], bool]] = None,
#         choices: Optional[Dict[str, str]] = None,
#     ) -> str:
#         current_prompt = prompt
#         last_response = ""

#         for attempt in range(max_retries + 1):
#             try:
#                 response = model.generate(current_prompt)
#                 response_text = self._normalize_llm_response_text(response)
#                 last_response = response_text
#                 logger.info("%s raw response (attempt %d/%d): %s", stage, attempt + 1, max_retries + 1, response_text)
#             except Exception as e:
#                 logger.error("%s failed on attempt %d/%d: %s", stage, attempt + 1, max_retries + 1, e)
#                 response_text = ""
#                 last_response = ""

#             refusal_like = self._is_refusal_like_response(response_text)
#             valid = validator(response_text) if validator is not None else bool(response_text)

#             if not refusal_like and valid:
#                 return response_text

#             if attempt >= max_retries:
#                 failure_bits = []
#                 if refusal_like:
#                     failure_bits.append("refusal-like response")
#                 if not valid:
#                     failure_bits.append("invalid or empty output")
#                 logger.warning(
#                     "%s exhausted retries; final failure reasons: %s",
#                     stage,
#                     ", ".join(failure_bits) if failure_bits else "unknown",
#                 )
#                 break

#             failure_bits = []
#             if refusal_like:
#                 failure_bits.append("refusal-like response")
#             if not valid:
#                 failure_bits.append("invalid or empty output")
#             failure_reason = ", ".join(failure_bits) if failure_bits else "unknown issue"

#             logger.warning(
#                 "%s retrying after %s (attempt %d/%d)",
#                 stage,
#                 failure_reason,
#                 attempt + 1,
#                 max_retries + 1,
#             )
#             retry_instruction = self._build_retry_instruction(
#                 stage=stage,
#                 previous_response=response_text,
#                 failure_reason=failure_reason,
#                 choices=choices,
#             )
#             current_prompt = self._append_retry_instruction_to_prompt(current_prompt, retry_instruction)

#         return last_response

#     # -----------------------------------------------------
#     # soft-group selector
#     # -----------------------------------------------------

#     def _compute_event_role_scores(
#         self,
#         query: str,
#         episodic_norm: Dict[str, float],
#         semantic_norm: Dict[str, float],
#         semantic_support_map: Dict[str, List[SemanticTripleEntry]],
#     ) -> Dict[str, Dict[str, float]]:
#         if not episodic_norm:
#             return {}

#         query_tokens = self._tokenize(query)
#         seed_doc_ids = [doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])[:3]]
#         seed_entries = [self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") for doc_id in seed_doc_ids]
#         seed_entries = [e for e in seed_entries if e is not None]
#         seed_tokens = {doc_id: self._event_tokens(doc_id) for doc_id in seed_doc_ids}
#         seed_centers = {
#             doc_id: self._entry_center_seconds(self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec"))
#             for doc_id in seed_doc_ids
#             if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
#         }

#         trigger_centroid = 0.0
#         if seed_centers:
#             trigger_centroid = sum(seed_centers.values()) / len(seed_centers)
#         earliest_seed = min(seed_centers.values()) if seed_centers else None

#         parent_counts: Dict[str, int] = defaultdict(int)
#         for doc_id in episodic_norm:
#             parent = None
#             if hasattr(self.episodic_memory, "get_parent_caption"):
#                 parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#             if parent is not None:
#                 parent_counts[parent.doc_id] += 1

#         role_scores: Dict[str, Dict[str, float]] = {}
#         for doc_id, ep_score in episodic_norm.items():
#             entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#             if entry is None:
#                 continue

#             center_sec = self._entry_center_seconds(entry)
#             toks = self._event_tokens(doc_id)
#             query_overlap = self._overlap_ratio(query_tokens, toks)
#             seed_overlap = 0.0
#             if seed_tokens:
#                 seed_overlap = max(self._overlap_ratio(toks, x) for x in seed_tokens.values())

#             if trigger_centroid > 0.0:
#                 delta = abs(center_sec - trigger_centroid)
#                 temporal_proximity = math.exp(-delta / 600.0)
#             else:
#                 temporal_proximity = 0.0
#             trigger_score = 0.60 * ep_score + 0.20 * query_overlap + 0.20 * temporal_proximity

#             earlierness = 0.0
#             if earliest_seed is not None and center_sec < earliest_seed:
#                 gap = earliest_seed - center_sec
#                 earlierness = min(1.0, gap / 1800.0)
#             support_presence = 1.0 if semantic_support_map.get(doc_id) else 0.0
#             antecedent_score = 0.45 * earlierness + 0.35 * seed_overlap + 0.20 * support_presence

#             broader_score = 0.0
#             parent = None
#             if hasattr(self.episodic_memory, "get_parent_caption"):
#                 parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#             if parent is not None:
#                 coverage = parent_counts.get(parent.doc_id, 0)
#                 broader_coverage = min(1.0, coverage / 3.0)
#                 broader_score = 0.60 * broader_coverage + 0.25 * query_overlap + 0.15 * ep_score

#             role_scores[doc_id] = {
#                 "trigger": float(trigger_score),
#                 "antecedent": float(antecedent_score),
#                 "broader": float(broader_score),
#                 "semantic": float(semantic_norm.get(doc_id, 0.0)),
#             }

#         for role_name in ["trigger", "antecedent", "broader", "semantic"]:
#             normed = self._normalize_dict({k: v[role_name] for k, v in role_scores.items()})
#             for doc_id in role_scores:
#                 role_scores[doc_id][role_name] = normed.get(doc_id, 0.0)

#         return role_scores

#     # def _build_event_selector_candidates(
#     #     self,
#     #     query: str,
#     #     episodic_norm: Dict[str, float],
#     #     semantic_norm: Dict[str, float],
#     #     semantic_support_map: Dict[str, List[SemanticTripleEntry]],
#     # ) -> List[Dict[str, Any]]:
#     #     role_scores = self._compute_event_role_scores(
#     #         query=query,
#     #         episodic_norm=episodic_norm,
#     #         semantic_norm=semantic_norm,
#     #         semantic_support_map=semantic_support_map,
#     #     )
#     #     if not role_scores:
#     #         return []

#     #     global_sorted = [doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])[: self.selector_global_top_n]]
#     #     trigger_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["trigger"])[: self.selector_trigger_top_n]]
#     #     antecedent_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["antecedent"])[: self.selector_antecedent_top_n]]
#     #     broader_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["broader"])[: self.selector_broader_top_n]]

#     #     ordered_doc_ids: List[str] = []
#     #     for group in [global_sorted, trigger_sorted, antecedent_sorted, broader_sorted]:
#     #         for doc_id in group:
#     #             if doc_id not in ordered_doc_ids:
#     #                 ordered_doc_ids.append(doc_id)
#     #             if len(ordered_doc_ids) >= self.selector_max_candidates:
#     #                 break
#     #         if len(ordered_doc_ids) >= self.selector_max_candidates:
#     #             break

#     #     logger.info(
#     #         "Selector pool groups | global=%s | trigger=%s | antecedent=%s | broader=%s",
#     #         global_sorted,
#     #         trigger_sorted,
#     #         antecedent_sorted,
#     #         broader_sorted,
#     #     )

#     #     candidates: List[Dict[str, Any]] = []
#     #     for idx, doc_id in enumerate(ordered_doc_ids, start=1):
#     #         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#     #         if entry is None:
#     #             continue
#     #         parent = None
#     #         if hasattr(self.episodic_memory, "get_parent_caption"):
#     #             parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#     #         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")[:4]
#     #         support_facts = semantic_support_map.get(doc_id, [])[:2]
#     #         primary_role = max(
#     #             ["trigger", "antecedent", "broader"],
#     #             key=lambda r: role_scores[doc_id].get(r, 0.0),
#     #         )

#     #         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#     #         candidates.append({
#     #             "index": idx,
#     #             "doc_id": doc_id,
#     #             "start_time": transform_timestamp(str(entry.timestamp_int[0])),
#     #             "end_time": transform_timestamp(str(entry.timestamp_int[1])),
#     #             "caption": entry.text,
#     #             "visual_summary": entry.visual_summary,
#     #             "critical_speech_lines": list(entry.metadata.get("critical_speech_lines", []) or [])[:4],
#     #             "episodic_score": round(float(episodic_norm.get(doc_id, 0.0)), 4),
#     #             "semantic_score": round(float(semantic_norm.get(doc_id, 0.0)), 4),
#     #             "trigger_score": round(float(role_scores[doc_id].get("trigger", 0.0)), 4),
#     #             "antecedent_score": round(float(role_scores[doc_id].get("antecedent", 0.0)), 4),
#     #             "broader_score": round(float(role_scores[doc_id].get("broader", 0.0)), 4),
#     #             "primary_role": primary_role,
#     #             "triplets": triplets,
#     #             "keyframe_caption": getattr(visual_entry, "keyframe_caption", "") if visual_entry is not None else "",
#     #             "parent_3min_doc_id": parent.doc_id if parent is not None else None,
#     #             "parent_3min_caption": parent.text if parent is not None else "",
#     #             "parent_3min_visual_summary": parent.visual_summary if parent is not None else "",
#     #             "parent_3min_critical_speech": list(parent.metadata.get("critical_speech_lines", []) or [])[:4] if parent is not None else [],
#     #             "semantic_support": [fact.to_display_str() for fact in support_facts],
#     #         })

#     #     return candidates

#     def _build_event_selector_candidates(
#         self,
#         query: str,
#         episodic_norm: Dict[str, float],
#         semantic_norm: Dict[str, float],
#         semantic_support_map: Dict[str, List[SemanticTripleEntry]],
#     ) -> List[Dict[str, Any]]:
#         role_scores = self._compute_event_role_scores(
#             query=query,
#             episodic_norm=episodic_norm,
#             semantic_norm=semantic_norm,
#             semantic_support_map=semantic_support_map,
#         )
#         if not role_scores:
#             return []

#         global_sorted = [doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])[: self.selector_global_top_n]]
#         trigger_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["trigger"])[: self.selector_trigger_top_n]]
#         antecedent_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["antecedent"])[: self.selector_antecedent_top_n]]
#         broader_sorted = [doc_id for doc_id, _ in sorted(role_scores.items(), key=lambda x: -x[1]["broader"])[: self.selector_broader_top_n]]

#         ordered_doc_ids: List[str] = []
#         for group in [global_sorted, trigger_sorted, antecedent_sorted, broader_sorted]:
#             for doc_id in group:
#                 if doc_id not in ordered_doc_ids:
#                     ordered_doc_ids.append(doc_id)
#                 if len(ordered_doc_ids) >= self.selector_max_candidates:
#                     break
#             if len(ordered_doc_ids) >= self.selector_max_candidates:
#                 break

#         logger.info(
#             "Selector pool groups | global=%s | trigger=%s | antecedent=%s | broader=%s",
#             global_sorted,
#             trigger_sorted,
#             antecedent_sorted,
#             broader_sorted,
#         )

#         candidates: List[Dict[str, Any]] = []
#         for doc_id in ordered_doc_ids:
#             entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#             if entry is None:
#                 continue
#             parent = None
#             if hasattr(self.episodic_memory, "get_parent_caption"):
#                 parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#             triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")[:4]
#             support_facts = semantic_support_map.get(doc_id, [])[:2]
#             primary_role = max(
#                 ["trigger", "antecedent", "broader"],
#                 key=lambda r: role_scores[doc_id].get(r, 0.0),
#             )

#             visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#             candidates.append({
#                 # Placeholder index; renumber after chronological sorting.
#                 "index": -1,
#                 "doc_id": doc_id,
#                 "start_time": transform_timestamp(str(entry.timestamp_int[0])),
#                 "end_time": transform_timestamp(str(entry.timestamp_int[1])),
#                 "caption": entry.text,
#                 "visual_summary": entry.visual_summary,
#                 "critical_speech_lines": list(entry.metadata.get("critical_speech_lines", []) or [])[:4],
#                 "episodic_score": round(float(episodic_norm.get(doc_id, 0.0)), 4),
#                 "semantic_score": round(float(semantic_norm.get(doc_id, 0.0)), 4),
#                 "trigger_score": round(float(role_scores[doc_id].get("trigger", 0.0)), 4),
#                 "antecedent_score": round(float(role_scores[doc_id].get("antecedent", 0.0)), 4),
#                 "broader_score": round(float(role_scores[doc_id].get("broader", 0.0)), 4),
#                 "primary_role": primary_role,
#                 "triplets": triplets,
#                 "keyframe_caption": getattr(visual_entry, "keyframe_caption", "") if visual_entry is not None else "",
#                 "parent_3min_doc_id": parent.doc_id if parent is not None else None,
#                 "parent_3min_caption": parent.text if parent is not None else "",
#                 "parent_3min_visual_summary": parent.visual_summary if parent is not None else "",
#                 "parent_3min_critical_speech": list(parent.metadata.get("critical_speech_lines", []) or [])[:4] if parent is not None else [],
#                 "semantic_support": [fact.to_display_str() for fact in support_facts],
#             })

#         # -------------------------------------------------
#         # Sort chronologically before sending candidates to the LLM selector.
#         # -------------------------------------------------
#         def _candidate_time_key(cand: Dict[str, Any]) -> Tuple[int, int]:
#             entry = self.episodic_memory.get_caption_by_doc_id(cand["doc_id"], "30sec")
#             if entry is None:
#                 return (10**18, 10**18)
#             start_ts, end_ts = entry.timestamp_int
#             return (
#                 self._timestamp_to_seconds(start_ts),
#                 self._timestamp_to_seconds(end_ts),
#             )

#         candidates.sort(key=_candidate_time_key)

#         # Renumber after sorting so the LLM sees stable candidate indices.
#         for idx, cand in enumerate(candidates, start=1):
#             cand["index"] = idx

#         logger.info(
#             "Selector candidates reordered chronologically: %s",
#             [
#                 {
#                     "index": c["index"],
#                     "doc_id": c["doc_id"],
#                     "start_time": c["start_time"],
#                     "end_time": c["end_time"],
#                 }
#                 for c in candidates
#             ],
#         )

#         return candidates

#     def _parse_event_selector_response(
#         self,
#         response: str,
#         valid_doc_ids: List[str],
#         num_candidates: int,
#     ) -> List[str]:
#         valid_doc_id_set = set(valid_doc_ids)
#         selected: List[str] = []

#         try:
#             json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
#             if json_match:
#                 parsed = json.loads(json_match.group())
#             else:
#                 parsed = json.loads(response)

#             if isinstance(parsed, dict):
#                 for key in ["selected_doc_ids", "doc_ids", "selected"]:
#                     if key in parsed and isinstance(parsed[key], list):
#                         for x in parsed[key]:
#                             if isinstance(x, str) and x in valid_doc_id_set and x not in selected:
#                                 selected.append(x)
#                 for key in ["selected_indices", "indices"]:
#                     if key in parsed and isinstance(parsed[key], list):
#                         for x in parsed[key]:
#                             try:
#                                 idx = int(x)
#                             except Exception:
#                                 continue
#                             if 1 <= idx <= len(valid_doc_ids):
#                                 doc_id = valid_doc_ids[idx - 1]
#                                 if doc_id not in selected:
#                                     selected.append(doc_id)
#             elif isinstance(parsed, list):
#                 for x in parsed:
#                     if isinstance(x, str) and x in valid_doc_id_set and x not in selected:
#                         selected.append(x)
#                     else:
#                         try:
#                             idx = int(x)
#                         except Exception:
#                             continue
#                         if 1 <= idx <= len(valid_doc_ids):
#                             doc_id = valid_doc_ids[idx - 1]
#                             if doc_id not in selected:
#                                 selected.append(doc_id)
#         except Exception:
#             pass

#         if not selected:
#             for doc_id in re.findall(r"DAY\d_[0-9]{8}_[0-9]{8}(?:_[A-Za-z0-9]+)?", response):
#                 if doc_id in valid_doc_id_set and doc_id not in selected:
#                     selected.append(doc_id)

#         if not selected:
#             for m in re.findall(r"\b(?:candidate|index|idx)?\s*#?\s*(\d{1,2})\b", response, re.IGNORECASE):
#                 try:
#                     idx = int(m)
#                 except Exception:
#                     continue
#                 if 1 <= idx <= num_candidates:
#                     doc_id = valid_doc_ids[idx - 1]
#                     if doc_id not in selected:
#                         selected.append(doc_id)

#         return selected

#     def _extract_selector_reason(self, response: str) -> str:
#         try:
#             json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
#             parsed = json.loads(json_match.group()) if json_match else json.loads(response)
#             if isinstance(parsed, dict):
#                 for key in ["reason", "rationale", "summary", "explanation"]:
#                     value = parsed.get(key)
#                     if isinstance(value, str) and value.strip():
#                         return value.strip()
#         except Exception:
#             pass
#         response = str(response).strip()
#         return response[:2000] if response else ""

#     def _extract_selector_metadata(self, response: str) -> Dict[str, str]:
#         meta = {"question_family": "", "reason": ""}
#         try:
#             json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
#             parsed = json.loads(json_match.group()) if json_match else json.loads(response)
#             if isinstance(parsed, dict):
#                 qf = parsed.get("question_family", "")
#                 if isinstance(qf, str):
#                     meta["question_family"] = qf.strip()
#                 for key in ["reason", "rationale", "summary", "explanation"]:
#                     value = parsed.get(key)
#                     if isinstance(value, str) and value.strip():
#                         meta["reason"] = value.strip()
#                         break
#         except Exception:
#             pass
#         return meta

#     def _select_top_events_with_llm(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]],
#         until_time: Optional[int],
#         selector_candidates: List[Dict[str, Any]],
#         final_top_k: int,
#     ) -> Tuple[List[str], str]:
#         if not selector_candidates:
#             return [], ""
#         if len(selector_candidates) <= final_top_k:
#             return [c["doc_id"] for c in selector_candidates], "Selector shortcut: number of candidates <= final_top_k."

#         query_with_time = self._build_query_with_time(query=query, choices=choices, until_time=until_time)

#         prompt = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are selecting event packets for a long-video QA system.\n"
#                     "Your job is NOT to choose events that are merely topically related. "
#                     "Your job is to choose events whose evidence matches the exact predicate asked by the question.\n\n"

#                     "You must do two things:\n"
#                     "Step 1: infer the question family from the question.\n"
#                     "Step 2: choose a small, complementary set of event packets that best supports the answer.\n\n"

#                     "Use one of these question families:\n"
#                     "1) action-owner\n"
#                     "2) source-trace\n"
#                     "3) participant-membership\n"
#                     "4) plan-intention-decision\n"
#                     "5) temporal-recall\n"
#                     "6) habit-preference\n"
#                     "7) attribute-content-purpose\n\n"

#                     "Core principle:\n"
#                     "- Prefer explicit evidence over weak implication.\n"
#                     "- Prefer predicate-aligned evidence over broad contextual relevance.\n"
#                     "- Do not over-select near-duplicate local events.\n"
#                     "- Always return valid candidate indices and/or valid doc_ids from the provided list only.\n\n"

#                     "[action-owner]\n"
#                     "Question intent: identify who performed an action, who assisted, or who acted first.\n"
#                     "Strong evidence:\n"
#                     "- explicit actor + explicit queried action\n"
#                     "- explicit cooperation, transfer, or assistance evidence when the question is about helping\n"
#                     "- earliest valid explicit action when the question is about who acted first\n"
#                     "Weak evidence:\n"
#                     "- nearby presence\n"
#                     "- interaction with related objects without the queried action\n"
#                     "- later result scenes without explicit action evidence\n"
#                     "Do NOT:\n"
#                     "- infer the actor only from scene participation\n"
#                     "- replace explicit action evidence with general topic-related context\n\n"

#                     "[source-trace]\n"
#                     "Question intent: identify where an object was before, where it came from, or how it was transferred.\n"
#                     "Strong evidence:\n"
#                     "- explicit prior location\n"
#                     "- explicit transfer path\n"
#                     "- explicit retrieval, carrying, bringing, taking, placing, or movement-between-locations evidence\n"
#                     "- earlier events that directly establish previous location\n"
#                     "Weak evidence:\n"
#                     "- current-use scenes\n"
#                     "- current location alone\n"
#                     "- generic earlier background context without explicit source grounding\n"
#                     "Do NOT:\n"
#                     "- treat holding, using, or interacting with an object as sufficient evidence of prior location\n"
#                     "- answer a previous-location question using only current-scene context\n"
#                     "- omit a source-establishing event if one exists\n\n"

#                     "[participant-membership]\n"
#                     "Question intent: identify who joined, who helped, who was part of the activity, or who was absent.\n"
#                     "Strong evidence:\n"
#                     "- explicit participation in the shared activity\n"
#                     "- explicit join/help/presence evidence in the relevant action chain\n"
#                     "- contrastive evidence for absence or mismatch across time\n"
#                     "Weak evidence:\n"
#                     "- later co-presence in the same room\n"
#                     "- nearby observer or bystander context\n"
#                     "Do NOT:\n"
#                     "- infer participation only from later appearance\n"
#                     "- confuse bystanders with core participants\n\n"

#                     "[plan-intention-decision]\n"
#                     "Question intent: identify a plan, intention, decision, next step, proposal, or commitment.\n"
#                     "Strong evidence:\n"
#                     "- explicit plan, intention, decision, proposal, assignment, or commitment\n"
#                     "- agent-specific future commitment\n"
#                     "- final-decision evidence\n"
#                     "Weak evidence:\n"
#                     "- related discussion\n"
#                     "- explanation, recommendation, or evaluation\n"
#                     "- general topic proximity\n"
#                     "- observation statements without commitment\n"
#                     "- offer or suggestion unless it clearly implies the agent's own intended action\n"
#                     "Do NOT:\n"
#                     "- infer intention from discussion alone\n"
#                     "- infer a personal plan from explanation or recommendation alone\n"
#                     "- confuse proposal, observation, ownership, or topic relevance with intention\n\n"

#                     "[temporal-recall]\n"
#                     "Question intent: identify the last time, first time, previous occurrence, or temporally constrained event.\n"
#                     "Strong evidence:\n"
#                     "- event whose timestamp best satisfies the temporal constraint\n"
#                     "- closest valid earlier or later occurrence that truly matches the queried event or topic\n"
#                     "Weak evidence:\n"
#                     "- semantically similar event at the wrong time\n"
#                     "- salient but temporally invalid event\n"
#                     "Do NOT:\n"
#                     "- ignore first/last/before/after constraints\n"
#                     "- choose a more relevant-looking event if its time is wrong\n\n"

#                     "[habit-preference]\n"
#                     "Question intent: identify a repeated behavior, usual pattern, stable preference, or dislike.\n"
#                     "Strong evidence:\n"
#                     "- repeated evidence across multiple events\n"
#                     "- explicit preference statements\n"
#                     "- aggregate frequency patterns\n"
#                     "Weak evidence:\n"
#                     "- one-off action\n"
#                     "- isolated or accidental occurrence\n"
#                     "Do NOT:\n"
#                     "- infer a habit from only one weak event if stronger repeated evidence exists\n"
#                     "- confuse temporary behavior with stable preference\n\n"

#                     "[attribute-content-purpose]\n"
#                     "Question intent: identify ownership, contents, identity, purpose, attribute, or category.\n"
#                     "Strong evidence:\n"
#                     "- direct statement of ownership, contents, identity, purpose, or queried attribute\n"
#                     "- explicit visual or textual grounding of the queried property\n"
#                     "Weak evidence:\n"
#                     "- nearby action context\n"
#                     "- related discussion without direct attribute grounding\n"
#                     "Do NOT:\n"
#                     "- replace a direct attribute question with surrounding activity\n"
#                     "- infer ownership, content, purpose, or identity from loose association alone\n\n"

#                     "Global anti-error rules:\n"
#                     "- Do not infer agent ownership from scene participation alone.\n"
#                     "- Do not infer intention from topic discussion alone.\n"
#                     "- Do not infer source from current location alone.\n"
#                     "- Do not infer habits from a single weak event if stronger repeated evidence exists.\n"
#                     "- Do not infer attributes from nearby actions when direct grounding exists.\n"
#                     "- When direct evidence and broad contextual evidence conflict, prefer direct evidence.\n"
#                     "- Use role scores as hints, not hard constraints.\n"
#                     "- Prefer a smaller set of directly relevant events over a larger set of vaguely related events.\n"
#                     "- If a question has a critical constraint (actor, source, time, intention, ownership, identity), at least one selected event should directly ground that constraint."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"{query_with_time}\n\n"
#                     f"Candidate Event Packets:\n{json.dumps(selector_candidates, ensure_ascii=False, indent=2)}\n\n"
#                     f"Select the best {final_top_k} candidates.\n\n"

#                     "Selection goals:\n"
#                     "- Choose complementary evidence, not repetitive evidence.\n"
#                     "- Retain at least one event that directly grounds the core predicate of the question.\n"
#                     "- If the question requires prior-state or source evidence, retain the event that directly establishes that prior state, even if it is earlier and less salient.\n"
#                     "- If the question requires intention or decision evidence, retain explicit commitment or decision evidence rather than topic-related discussion.\n"
#                     "- If the question requires identifying an actor, retain explicit actor evidence.\n"
#                     "- If the question requires temporal comparison, enforce the temporal constraint strictly.\n"
#                     "- If the question requires a stable habit or preference, prefer repeated or aggregate evidence over one-off evidence.\n"
#                     "- If the question requires ownership, contents, identity, purpose, or attribute, prefer direct grounding over surrounding context.\n\n"

#                     "Output requirements:\n"
#                     "- Infer the correct question_family first.\n"
#                     "- Then select the best candidates.\n"
#                     "- The reason must explain why the selected events satisfy the core predicate better than merely related events.\n\n"

#                     "Return ONLY JSON in this format:\n"
#                     "{"
#                     "\"question_family\": \"...\", "
#                     "\"selected_indices\": [..], "
#                     "\"selected_doc_ids\": [..], "
#                     "\"reason\": \"...\""
#                     "}"
#                 ),
#             },
#         ]

#         valid_doc_ids = [c["doc_id"] for c in selector_candidates]

#         def _selector_validator(response_text: str) -> bool:
#             parsed = self._parse_event_selector_response(
#                 response=response_text,
#                 valid_doc_ids=valid_doc_ids,
#                 num_candidates=len(selector_candidates),
#             )
#             return len(parsed) > 0

#         response = self._generate_with_refusal_retry(
#             model=self.respond_llm_model,
#             prompt=prompt,
#             stage="selector",
#             max_retries=max(self.selector_refusal_retry_max, 0),
#             validator=_selector_validator,
#             choices=choices,
#         )

#         selected = self._parse_event_selector_response(
#             response=response,
#             valid_doc_ids=valid_doc_ids,
#             num_candidates=len(selector_candidates),
#         )
#         meta = self._extract_selector_metadata(response)
#         logger.info("LLM event selector question_family: %s", meta.get("question_family", ""))
#         selector_reason = self._extract_selector_reason(response)
#         return selected[:final_top_k], selector_reason

#     # -----------------------------------------------------
#     # direct fusion answer pipeline
#     # -----------------------------------------------------

#     def answer(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> QAResult:
#         if until_time and until_time > self.indexed_time:
#             self.index(until_time)

#         full_query = self._build_query_with_time(
#             query=query,
#             choices=choices,
#             until_time=until_time,
#         )

#         episodic_ranked = self.episodic_memory.retrieve_ranked(
#             query=query,
#             top_k_per_granularity={
#                 "30sec": max(self.episodic_top_k * 4, 10),
#                 "3min": max(self.episodic_top_k * 3, 6),
#                 "10min": max(self.episodic_top_k * 2, 5),
#                 "1h": max(self.episodic_top_k, 3),
#             },
#             dedup_by_doc_id=True,
#         )
#         semantic_entries = self.semantic_memory.retrieve(
#             query=query,
#             top_k=max(self.semantic_top_k, self.episodic_top_k * 3),
#             as_context=False,
#         )
#         if isinstance(semantic_entries, str):
#             semantic_entries = []

#         logger.info(
#             "Retrieved %d episodic candidates and %d semantic facts",
#             len(episodic_ranked),
#             len(semantic_entries),
#         )

#         if episodic_ranked:
#             logger.info(
#                 "Top episodic candidates: %s",
#                 [
#                     {
#                         "doc_id": entry.doc_id,
#                         "granularity": entry.granularity,
#                         "score": round(score, 4),
#                     }
#                     for entry, score in episodic_ranked[:8]
#                 ],
#             )

#         if semantic_entries:
#             logger.info(
#                 "Top semantic facts: %s",
#                 [
#                     {
#                         "fact_id": entry.id,
#                         "triple": entry.triple,
#                         "support_count": entry.support_count,
#                         "confidence": round(float(entry.confidence), 4),
#                     }
#                     for entry in semantic_entries[:8]
#                 ],
#             )

#         episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
#         semantic_projected, semantic_support_map = self._project_semantic_to_30s(semantic_entries)

#         candidate_doc_ids = set(episodic_projected.keys())
#         if not candidate_doc_ids:
#             logger.warning("No candidate events found from episodic retrieval")
#             candidate_doc_ids = set()
#             for entry, _ in episodic_ranked[: self.episodic_top_k]:
#                 for doc_id in self.episodic_memory.expand_entry_to_30s_doc_ids(entry):
#                     if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None:
#                         candidate_doc_ids.add(doc_id)
#             if not candidate_doc_ids:
#                 return QAResult(
#                     question=query,
#                     answer="Unable to retrieve relevant evidence",
#                     retrieved_items=[],
#                     round_history=[],
#                     num_rounds=1,
#                 )

#         episodic_norm = self._normalize_dict({doc_id: episodic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})
#         semantic_norm = self._normalize_dict({doc_id: semantic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})

#         anchor_scores: Dict[str, float] = {doc_id: episodic_norm.get(doc_id, 0.0) for doc_id in candidate_doc_ids}
#         ranked_doc_ids = [doc_id for doc_id, _ in sorted(anchor_scores.items(), key=lambda x: -x[1])]

#         logger.info(
#             "Top episodic anchor scores before selector: %s",
#             [
#                 {
#                     "doc_id": doc_id,
#                     "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
#                     "sem": round(semantic_norm.get(doc_id, 0.0), 4),
#                 }
#                 for doc_id in ranked_doc_ids[:8]
#             ],
#         )

#         selector_candidates = self._build_event_selector_candidates(
#             query=query,
#             episodic_norm=episodic_norm,
#             semantic_norm=semantic_norm,
#             semantic_support_map=semantic_support_map,
#         )
#         logger.info(
#             "Built %d selector candidates: %s",
#             len(selector_candidates),
#             [
#                 {
#                     "index": c["index"],
#                     "doc_id": c["doc_id"],
#                     "primary_role": c["primary_role"],
#                     "ep": c["episodic_score"],
#                     "tr": c["trigger_score"],
#                     "ant": c["antecedent_score"],
#                     "bro": c["broader_score"],
#                 }
#                 for c in selector_candidates
#             ],
#         )

#         selected_doc_ids, selector_reason = self._select_top_events_with_llm(
#             query=query,
#             choices=choices,
#             until_time=until_time,
#             selector_candidates=selector_candidates,
#             final_top_k=max(self.episodic_top_k, 1),
#         )

#         if not selected_doc_ids:
#             logger.info("LLM event selector returned no valid doc_ids, fallback to coarse ranking")
#             top_doc_ids = ranked_doc_ids[: max(self.episodic_top_k, 1)]
#             selector_reason = (
#                 "Selector fallback: no valid doc_ids were parsed from the selector output. "
#                 "Coarse episodic ranking was used instead."
#             )
#         else:
#             top_doc_ids = []
#             for doc_id in selected_doc_ids:
#                 if doc_id not in top_doc_ids:
#                     top_doc_ids.append(doc_id)
#             if len(top_doc_ids) < max(self.episodic_top_k, 1):
#                 for doc_id in ranked_doc_ids:
#                     if doc_id not in top_doc_ids:
#                         top_doc_ids.append(doc_id)
#                     if len(top_doc_ids) >= max(self.episodic_top_k, 1):
#                         break

#         logger.info("Selector reason summary: %s", selector_reason)

#         logger.info(
#             "Final selected event anchors: %s",
#             [
#                 {
#                     "doc_id": doc_id,
#                     "anchor": round(anchor_scores.get(doc_id, 0.0), 4),
#                     "sem": round(semantic_norm.get(doc_id, 0.0), 4),
#                 }
#                 for doc_id in top_doc_ids
#             ],
#         )

#         event_packets = []
#         for doc_id in top_doc_ids:
#             packet = self._build_event_packet(
#                 doc_id=doc_id,
#                 score=anchor_scores.get(doc_id, 0.0),
#                 supporting_facts=semantic_support_map.get(doc_id, []),
#             )
#             if packet:
#                 event_packets.append(packet)

#         logger.info("Built %d event packets", len(event_packets))
#         for doc_id in top_doc_ids:
#             entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#             if entry is not None:
#                 logger.info(
#                     "Event packet anchor %s | time=%s-%s | text=%s",
#                     doc_id,
#                     entry.start_time,
#                     entry.end_time,
#                     entry.text[:120].replace("\n", " "),
#                 )

#         semantic_context = self._build_semantic_context(semantic_entries, top_n=min(5, self.semantic_top_k))

#         retrieved_items: List[RetrievedItem] = []
#         if event_packets:
#             retrieved_items.append(
#                 RetrievedItem(
#                     memory_type="episodic",
#                     content="\n\n".join(event_packets),
#                     query=query,
#                     round_num=1,
#                 )
#             )
#         if semantic_context:
#             retrieved_items.append(
#                 RetrievedItem(
#                     memory_type="semantic",
#                     content=semantic_context,
#                     query=query,
#                     round_num=1,
#                 )
#             )

#         event_images = self.visual_memory.get_event_images(
#             top_doc_ids,
#             max_images_per_event=max(self.visual_top_k, 1),
#         )

#         if event_images:
#             num_event_with_images = len(event_images)
#             num_total_images = sum(len(v) for v in event_images.values())
#             logger.info(
#                 "Loaded visual evidence for %d events, %d images total",
#                 num_event_with_images,
#                 num_total_images,
#             )
#             for doc_id in top_doc_ids:
#                 logger.info("Visual images for %s: %d", doc_id, len(event_images.get(doc_id, [])))

#             all_images = []
#             for doc_id in top_doc_ids:
#                 all_images.extend(event_images.get(doc_id, []))

#             if all_images:
#                 logger.info("Sending %d images to QA", len(all_images))
#                 retrieved_items.append(
#                     RetrievedItem(
#                         memory_type="visual",
#                         content=all_images,
#                         query=query,
#                         round_num=1,
#                     )
#                 )
#         else:
#             logger.info("No visual evidence found for final event anchors")

#         round_history = self._build_round_history(query, top_doc_ids, semantic_entries)

#         try:
#             qa_prompt = self.prompt_template_manager.render("qa_egolife")
#         except Exception as e:
#             logger.error(f"Failed to load qa_egolife template: {e}")
#             raise

#         qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
#         qa_content.append({
#             "type": "text",
#             "text": (
#                 "Selector summary:\n"
#                 f"Chosen event anchors: {top_doc_ids}\n"
#                 f"Selector reason: {selector_reason}\n"
#                 "The selected event anchors were chosen because they form the strongest evidence chain for this question.\n"
#                 "Use these selected events as the primary basis for answering.\n"
#                 "Do not override a clearly supported conclusion from the selected evidence with a weaker alternative."
#             )
#         })
#         qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
#         if choices:
#             grounding_lines = []
#             narrator_labels = []
#             for k, v in sorted(choices.items()):
#                 v_norm = str(v).strip().lower()
#                 if v_norm in {"me", "myself", "self", "narrator", "the narrator", "speaker"}:
#                     narrator_labels.append(k)

#             if narrator_labels:
#                 grounding_lines.append(
#                     "Important grounding: in this egocentric first-person video, the pronouns 'I', 'me', 'my', and 'myself' refer to the narrator / camera wearer."
#                 )
#                 grounding_lines.append(
#                     f"If the evidence says the narrator ('I') performed the action, prefer the corresponding choice(s): {', '.join(narrator_labels)}."
#                 )

#             grounding_lines.append(
#                 "Answer selection rule: choose the option best supported by the retrieved evidence and the selector summary above."
#             )
#             grounding_lines.append(
#                 "If the selector reason and selected events clearly support a specific option, do not override it with a weaker alternative."
#             )
#             grounding_lines.append(
#                 "Please provide only the final answer from the choices given (e.g., A, B, C, or D)."
#             )

#             qa_content.append({"type": "text", "text": "\n" + "\n".join(grounding_lines)})

#         num_text_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "text")
#         num_image_blocks = sum(1 for x in qa_content if isinstance(x, dict) and x.get("type") == "image")
#         logger.info(
#             "QA payload prepared: %d text blocks, %d image blocks, %d retrieved items",
#             num_text_blocks,
#             num_image_blocks,
#             len(retrieved_items),
#         )

#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({"role": "user", "content": qa_content})

#         answer = self._generate_with_refusal_retry(
#             model=self.respond_llm_model,
#             prompt=qa_messages,
#             stage="answer",
#             max_retries=max(self.answer_refusal_retry_max, 0),
#             choices=choices,
#         )
#         if not answer:
#             answer = "Unable to generate answer"

#         return QAResult(
#             question=query,
#             answer=answer,
#             retrieved_items=retrieved_items,
#             round_history=round_history,
#             num_rounds=1,
#         )

#     # -----------------------------------------------------
#     # lifecycle helpers
#     # -----------------------------------------------------

#     def reset_index(self) -> None:
#         self.episodic_memory.reset_index()
#         self.semantic_memory.reset_index()
#         self.visual_memory.reset_index()
#         self.indexed_time = 0
#         logger.info("All memory indices reset")

#     def cleanup(self) -> None:
#         self.semantic_memory.cleanup()
#         self.visual_memory.cleanup()
#         logger.info("Memory cleanup complete")

#     def get_indexed_time(self) -> str:
#         return transform_timestamp(str(self.indexed_time))

#     def set_retrieval_top_k(
#         self,
#         episodic: Optional[int] = None,
#         semantic: Optional[int] = None,
#         visual: Optional[int] = None,
#     ) -> None:
#         if episodic is not None:
#             self.episodic_top_k = episodic
#         if semantic is not None:
#             self.semantic_top_k = semantic
#         if visual is not None:
#             self.visual_top_k = visual

# """
# WorldMemory: retrieval-agent-based heterogeneous memory retrieval with unified evidence packets.

# This version uses:
# - LLM-only question-family classification (no heuristic routing)
# - retrieval-agent outer loop for heterogeneous memory selection
# - family-aware episodic retrieval with graph refinement handled inside EpisodicMemoryRAG
# - semantic and visual retrieval converted into unified evidence packets
# """

# import copy
# import json
# import logging
# import re
# from typing import Any, Dict, List, Optional, Set, Tuple

# from PIL import Image

# from ..embedding import EmbeddingModel
# from ..llm import LLMModel, PromptTemplateManager
# from .episodic.EpisodicMemory_rag import CaptionEntryRAG, EpisodicMemoryRAG
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .utils import *
# from .visual import VisualMemory

# logger = logging.getLogger(__name__)


# class WorldMemory:
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         retriever_llm_model: LLMModel,
#         respond_llm_model: Optional[LLMModel] = None,
#         prompt_template_manager: Optional[PromptTemplateManager] = None,
#         episodic_granularities: Optional[List[str]] = None,
#         episodic_cache_tag: Optional[str] = None,
#         qa_template_name: str = "qa_egolife",
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors
#         self.qa_template_name = qa_template_name

#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=embedding_model,
#             llm_model=retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=episodic_granularities,
#             cache_tag=episodic_cache_tag,
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=embedding_model)

#         self.indexed_time: int = 0
#         self.episodic_top_k: int = 4
#         self.semantic_top_k: int = 6
#         self.visual_top_k: int = 3

#     def set_retrieval_top_k(
#         self,
#         episodic: Optional[int] = None,
#         semantic: Optional[int] = None,
#         visual: Optional[int] = None,
#     ) -> None:
#         """
#         Backward-compatible helper for eval scripts.

#         Allows the evaluation script to set retrieval top-k values after
#         constructing WorldMemory.
#         """
#         if episodic is not None:
#             self.episodic_top_k = int(episodic)
#         if semantic is not None:
#             self.semantic_top_k = int(semantic)
#         if visual is not None:
#             self.visual_top_k = int(visual)

#         logger.info(
#             "Set retrieval top-k: episodic=%s, semantic=%s, visual=%s",
#             self.episodic_top_k,
#             self.semantic_top_k,
#             self.visual_top_k,
#         )


#     def cleanup(self) -> None:
#         """
#         Best-effort cleanup hook for multiprocessing eval.

#         Safe to call even if some submodules do not expose cleanup().
#         """
#         try:
#             if hasattr(self.episodic_memory, "cleanup"):
#                 self.episodic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Episodic cleanup failed: %s", e)

#         try:
#             if hasattr(self.semantic_memory, "cleanup"):
#                 self.semantic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Semantic cleanup failed: %s", e)

#         try:
#             if hasattr(self.visual_memory, "cleanup"):
#                 self.visual_memory.cleanup()
#         except Exception as e:
#             logger.warning("Visual cleanup failed: %s", e)

#         try:
#             import torch
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         except Exception:
#             pass

#         logger.info("WorldMemory cleanup complete")
#     # -----------------------------------------------------
#     # loading / indexing
#     # -----------------------------------------------------

#     def load_episodic_captions(
#         self,
#         caption_files: Optional[Dict[str, str]] = None,
#         caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
#     ) -> None:
#         if caption_files:
#             self.episodic_memory.load_captions_from_files(caption_files)
#         if caption_data:
#             self.episodic_memory.load_captions_from_data(caption_data)

#     def load_episodic_sidecar(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_files or graph_files:
#             self.episodic_memory.load_sidecar_from_files(
#                 triplet_files=triplet_files,
#                 graph_files=graph_files,
#             )
#         if triplet_data or graph_data:
#             self.episodic_memory.load_sidecar_from_data(
#                 triplet_data=triplet_data,
#                 graph_data=graph_data,
#             )

#     def load_semantic_triples(
#         self,
#         file_path: Optional[str] = None,
#         data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if file_path:
#             self.semantic_memory.load_triples_from_file(file_path)
#         if data:
#             self.semantic_memory.load_triples_from_data(data)

#     def load_visual_clips(
#         self,
#         embeddings_path: Optional[str] = None,
#         clips_path: Optional[str] = None,
#         clips_data: Optional[List[Dict[str, Any]]] = None,
#     ) -> None:
#         if embeddings_path:
#             self.visual_memory.load_embeddings_from_file(embeddings_path)
#         if clips_path:
#             self.visual_memory.load_clips_from_file(clips_path)
#         if clips_data:
#             self.visual_memory.load_clips_from_data(clips_data)

#     def prepare_episodic_dense_index(self, force_rebuild: bool = False) -> None:
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=force_rebuild)

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug("Already indexed up to %s, skipping", self.indexed_time)
#             return

#         logger.info("Indexing all memories up to %s", transform_timestamp(str(until_time)))
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=False)
#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
#         self.indexed_time = until_time
#         logger.info("Indexing complete for all memory types")

#     # -----------------------------------------------------
#     # query / parsing helpers
#     # -----------------------------------------------------

#     def _build_query_with_time(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> str:
#         lines = [f"Query: {query}"]
#         if until_time is not None:
#             lines.append(f"Query Time: {transform_timestamp(str(until_time))}")
#             lines.append(
#                 "Important: Interpret all relative temporal expressions "
#                 '(e.g. "before", "after", "earlier", "later", "recently", '
#                 '"a few hours ago", "first", "last") relative to Query Time.'
#             )
#         if choices:
#             choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
#             lines.append(f"Choices: {choices_str}")
#         return "\n".join(lines)

#     def _parse_json_object(self, response: str) -> Dict[str, Any]:
#         try:
#             json_match = re.search(r"\{.*\}", response, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group())
#             return json.loads(response)
#         except Exception:
#             return {}

#     def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
#         try:
#             data = self._parse_json_object(response)
#             decision = str(data.get("decision", "answer")).lower()
#             reason = data.get("reason")
#             selected_memory = None
#             if decision == "search" and "selected_memory" in data:
#                 mem_data = data["selected_memory"]
#                 selected_memory = MemorySearchOutput(
#                     memory_type=str(mem_data.get("memory_type", "")).lower(),
#                     search_query=str(mem_data.get("search_query", "")),
#                 )
#             return ReasoningOutput(decision=decision, selected_memory=selected_memory, reason=reason)
#         except Exception as e:
#             logger.warning("Failed to parse reasoning response: %s", e)
#             return ReasoningOutput(decision="answer")

#     def _format_round_history(self, rounds: List[Dict[str, Any]]) -> str:
#         if not rounds:
#             return "[]"
#         lines = []
#         for r in rounds:
#             lines.append(
#                 f"### Round {r['round_num']}\n"
#                 f"Decision: {r['decision']}\n"
#                 f"Memory: {r['memory_type']}\n"
#                 f"Search Query: {r['search_query']}\n"
#                 f"Retrieved:\n{r['retrieved_content']}"
#             )
#         return "\n\n".join(lines)

#     def _render_retrieved_items_for_qa(self, retrieved_items: List[RetrievedItem]) -> List[Dict[str, Any]]:
#         messages: List[Dict[str, Any]] = []
#         for item in retrieved_items:
#             if item.memory_type in ("episodic", "semantic"):
#                 messages.append({"type": "text", "text": item.content})
#             elif item.memory_type == "visual":
#                 if isinstance(item.content, list):
#                     for img in item.content:
#                         if isinstance(img, Image.Image):
#                             messages.append({"type": "image", "image": img})
#                         elif isinstance(img, dict) and "image" in img:
#                             messages.append({"type": "image", "image": img["image"]})
#         return messages

#     def _clean_text(self, text: Any) -> str:
#         if text is None:
#             return ""
#         text = str(text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

#     def _short_text(self, text: Any, max_chars: int = 180) -> str:
#         text = self._clean_text(text)
#         if len(text) <= max_chars:
#             return text
#         return text[: max_chars - 3].rstrip() + "..."

#     def _build_episodic_history_summary(
#         self,
#         selected: List[Tuple[CaptionEntryRAG, float]],
#         max_items: int = 5,
#     ) -> str:
#         if not selected:
#             return "[No episodic results]"

#         lines = ["Retrieved episodic evidence:"]
#         for entry, score in selected[:max_items]:
#             start_ts, end_ts = entry.timestamp_int
#             time_span = f"{transform_timestamp(str(start_ts))} - {transform_timestamp(str(end_ts))}"

#             main_text = self._clean_text(entry.text)

#             visual_bits: List[str] = []
#             if entry.visual_summary:
#                 visual_bits.append(self._clean_text(entry.visual_summary))

#             visual_entry = self.visual_memory.get_clip_by_doc_id(entry.doc_id)
#             if visual_entry is not None:
#                 keyframe_caption = self._clean_text(getattr(visual_entry, "keyframe_caption", ""))
#                 if keyframe_caption and not visual_bits:
#                     visual_bits.append(keyframe_caption)

#                 scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#                 if isinstance(scene_summary, dict):
#                     dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                     if dominant_scene:
#                         visual_bits.append(f"scene={dominant_scene}")

#                 visual_objects = list(getattr(visual_entry, "visual_objects", []) or [])
#                 if visual_objects:
#                     visual_bits.append("objects=" + ", ".join(map(str, visual_objects[:8])))

#             critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#             speech_text = ""
#             if critical_lines:
#                 speech_text = self._clean_text(" | ".join([str(x) for x in critical_lines if str(x).strip()]))

#             line = f"- {time_span}: {main_text}"
#             if visual_bits:
#                 line += " | Visual: " + " ; ".join(visual_bits)
#             if speech_text:
#                 line += f" | Speech: {speech_text}"
#             line += f" | score={score:.4f}"
#             lines.append(line)

#         return "\n".join(lines)

#     def _build_semantic_history_summary(
#         self,
#         selected: List[SemanticTripleEntry],
#         max_items: int = 3,
#     ) -> str:
#         if not selected:
#             return "[No semantic results]"

#         lines = ["Retrieved semantic evidence:"]
#         for entry in selected[:max_items]:
#             fact_text = self._clean_text(entry.to_display_str())

#             support_text = ""
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#                 if support_ids:
#                     support_text = f" | support_events={support_ids}"

#             conf_text = ""
#             conf = getattr(entry, "confidence", None)
#             if conf is not None:
#                 try:
#                     conf_text = f" | confidence={float(conf):.4f}"
#                 except Exception:
#                     pass

#             lines.append(f"- {fact_text}{support_text}{conf_text}")

#         return "\n".join(lines)

#     def _build_visual_history_summary_from_doc_ids(
#         self,
#         doc_ids: List[str],
#         max_items: int = 3,
#     ) -> str:
#         if not doc_ids:
#             return "[No visual results]"

#         lines = ["Retrieved visual evidence:"]
#         kept = 0
#         for doc_id in doc_ids:
#             clip = self.visual_memory.get_clip_by_doc_id(doc_id)
#             if clip is None:
#                 continue

#             span_text = doc_id
#             if hasattr(clip, "timestamp_int"):
#                 try:
#                     s, e = clip.timestamp_int
#                     span_text = f"{transform_timestamp(str(s))} - {transform_timestamp(str(e))}"
#                 except Exception:
#                     span_text = doc_id

#             keyframe_caption = self._clean_text(getattr(clip, "keyframe_caption", ""))

#             scene_summary = getattr(clip, "scene_summary", {}) or {}
#             dominant_scene = ""
#             scene_desc = ""
#             if isinstance(scene_summary, dict):
#                 dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                 scene_desc = self._clean_text(scene_summary.get("scene_description", ""))

#             visual_objects = list(getattr(clip, "visual_objects", []) or [])
#             objects_text = ", ".join(map(str, visual_objects[:10])) if visual_objects else ""

#             parts = []
#             if keyframe_caption:
#                 parts.append(keyframe_caption)
#             if dominant_scene:
#                 parts.append(f"scene={dominant_scene}")
#             if scene_desc:
#                 parts.append(f"scene_description={scene_desc}")
#             if objects_text:
#                 parts.append(f"objects={objects_text}")

#             if not parts:
#                 parts.append("visual evidence available")

#             lines.append(f"- {span_text}: " + " | ".join(parts))
#             kept += 1
#             if kept >= max_items:
#                 break

#         if kept == 0:
#             return "[Visual images retrieved, but no textual visual summary available]"
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # LLM-only question-family classifier
#     # -----------------------------------------------------

#     def _default_family_info(self) -> Dict[str, Any]:
#         return {
#             "question_family": "event",
#             "graph_mode": "default",
#             "time_bias": "none",
#             "need_visual_followup": False,
#         }

#     def _llm_family_classifier(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         system_prompt = (
#             "You classify long-video QA questions into exactly one question family and output JSON only. "
#             "Valid families: source-trace, temporal-recall, action-owner, participant-membership, "
#             "plan-intention-decision, habit-preference, attribute-content-purpose, event. "
#             "Also output graph_mode, time_bias (forward/backward/none), and need_visual_followup (true/false). "
#             "Be conservative and choose the simplest valid family."
#         )
#         family_guide = (
#             "Family guidance:\n"
#             "- source-trace: prior location, transfer path, brought from, placed before, where X was before. graph_mode=backtrack_object_source, time_bias=backward, need_visual_followup=false\n"
#             "- temporal-recall: first/last/before/after/earlier/later. graph_mode=temporal_walk, time_bias=backward or forward or none, need_visual_followup=false\n"
#             "- action-owner: who did/used/moved/picked/brought something. graph_mode=actor_action_refine, time_bias=none, need_visual_followup=false\n"
#             "- participant-membership: who was present / with whom / who joined or left. graph_mode=participant_cooccurrence_refine, time_bias=none, need_visual_followup=false\n"
#             "- plan-intention-decision: plans, rationale, commitments, intended next step. graph_mode=topic_commitment_refine, time_bias=none, need_visual_followup=false\n"
#             "- habit-preference: repeated patterns, often/usually/prefer. graph_mode=habit_support_only, time_bias=none, need_visual_followup=false\n"
#             "- attribute-content-purpose: color, appearance, what was inside, visual identity, purpose requiring direct grounding. graph_mode=anchor_refine_then_visual, time_bias=none, need_visual_followup=true\n"
#             "- event: generic specific-event questions that do not strongly fit the above. graph_mode=default, time_bias=none, need_visual_followup=false"
#         )
#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         user_prompt = (
#             f"{full_query}\n\n"
#             f"{family_guide}\n\n"
#             "Return JSON with keys: question_family, graph_mode, time_bias, need_visual_followup"
#         )
#         fallback = self._default_family_info()
#         try:
#             response = self.respond_llm_model.generate([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ])
#             data = self._parse_json_object(response)
#             if not data:
#                 return fallback
#             return {
#                 "question_family": str(data.get("question_family", fallback["question_family"])).strip().lower(),
#                 "graph_mode": str(data.get("graph_mode", fallback["graph_mode"])).strip().lower(),
#                 "time_bias": str(data.get("time_bias", fallback["time_bias"])).strip().lower(),
#                 "need_visual_followup": bool(data.get("need_visual_followup", fallback["need_visual_followup"])),
#             }
#         except Exception as e:
#             logger.warning("LLM family classifier failed, fallback to default family info: %s", e)
#             return fallback

#     def _classify_question_family(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         return self._llm_family_classifier(query=query, choices=choices, until_time=until_time)

#     def _family_retrieval_prior(self, family_info: Dict[str, Any]) -> str:
#         family = family_info.get("question_family", "event")
#         mapping = {
#             "source-trace": "Search for earlier source-establishing events rather than merely the current location.",
#             "temporal-recall": "Search for the event that best satisfies first/last/before/after constraints.",
#             "action-owner": "Search for explicit actor-action evidence, not just nearby participation.",
#             "participant-membership": "Search for explicit participation/co-presence evidence across relevant event windows.",
#             "plan-intention-decision": "Search for explicit commitments, decisions, or rationale statements, often in speech or topic context.",
#             "habit-preference": "Search for repeated or aggregate evidence; semantic memory is often useful.",
#             "attribute-content-purpose": "Search for direct grounding of the queried property; visual follow-up may help.",
#             "event": "Search for the most predicate-aligned event evidence first.",
#         }
#         return mapping.get(family, mapping["event"])

#     def _family_hint_block(self, family_info: Dict[str, Any]) -> str:
#         return (
#             "Question family hint:\n"
#             f"- question_family: {family_info.get('question_family', 'event')}\n"
#             f"- graph_mode_if_episodic: {family_info.get('graph_mode', 'default')}\n"
#             f"- time_bias: {family_info.get('time_bias', 'none')}\n"
#             f"- need_visual_followup: {str(bool(family_info.get('need_visual_followup', False))).lower()}\n\n"
#             "Family-specific retrieval prior:\n"
#             f"- {self._family_retrieval_prior(family_info)}\n"
#             "Use this as a prior, but revise based on round history if needed."
#         )

#     # -----------------------------------------------------
#     # packet builders
#     # -----------------------------------------------------

#     def _build_event_packet(
#         self,
#         doc_id: str,
#         score: float,
#         supporting_facts: Optional[List[SemanticTripleEntry]] = None,
#     ) -> str:
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return ""

#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         supporting_facts = supporting_facts or []
#         parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min") if hasattr(self.episodic_memory, "get_parent_caption") else None

#         lines: List[str] = []
#         lines.append(f"Event Anchor: {doc_id}")
#         lines.append(f"Relevance Score: {score:.4f}")
#         lines.append(entry.to_display_str(include_visual_summary=True))

#         critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#         if critical_lines:
#             lines.append("Critical Speech:")
#             for line in critical_lines[:3]:
#                 if str(line).strip():
#                     lines.append(f"- {line}")

#         if parent_3min is not None and parent_3min.doc_id != doc_id:
#             p_start, p_end = parent_3min.timestamp_int
#             lines.append(
#                 f"3min Context [{transform_timestamp(str(p_start))} - {transform_timestamp(str(p_end))}]: {parent_3min.text}"
#             )
#             if parent_3min.visual_summary:
#                 lines.append(f"3min Visual: {parent_3min.visual_summary}")

#         if visual_entry is not None:
#             if getattr(visual_entry, "keyframe_caption", ""):
#                 lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
#             visual_objects = getattr(visual_entry, "visual_objects", []) or []
#             if visual_objects:
#                 lines.append("Visual Objects: " + ", ".join(visual_objects[:8]))
#             scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#             if isinstance(scene_summary, dict) and scene_summary.get("dominant_scene"):
#                 lines.append(f"Scene: {scene_summary.get('dominant_scene')}")

#         if triplets:
#             lines.append("Episodic Triplets:")
#             for tri in triplets[:6]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if supporting_facts:
#             lines.append("Supporting Semantic Facts:")
#             for fact in supporting_facts[:3]:
#                 lines.append(f"- {fact.to_display_str()}")

#         return "\n".join(lines)

#     def _build_semantic_packet(self, entry: SemanticTripleEntry) -> str:
#         lines = [f"Semantic Packet: {entry.to_display_str()}"]
#         support_ids: List[str] = []
#         if hasattr(self.semantic_memory, "get_support_event_ids"):
#             support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#         if support_ids:
#             lines.append("Support Event IDs: " + ", ".join(support_ids))
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # retrieval helpers
#     # -----------------------------------------------------

#     def retrieve_from_episodic_packets(
#         self,
#         query: str,
#         family_info: Dict[str, Any],
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str]]:
#         top_k = top_k or self.episodic_top_k
#         retrieved_set = retrieved_set or set()

#         ranked = self.episodic_memory.retrieve_ranked_with_family(
#             query=query,
#             family_info=family_info,
#         )
#         if not ranked:
#             return [], "[No episodic results]", retrieved_set, []

#         selected: List[Tuple[CaptionEntryRAG, float]] = []
#         for entry, score in ranked:
#             key = f"episodic:{entry.doc_id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append((entry, score))
#             if len(selected) >= top_k:
#                 break

#         if not selected:
#             return [], "[No new episodic results]", retrieved_set, []

#         packets: List[str] = []
#         top_doc_ids: List[str] = []
#         for entry, score in selected:
#             top_doc_ids.append(entry.doc_id)
#             packet = self._build_event_packet(doc_id=entry.doc_id, score=score)
#             if packet:
#                 packets.append(packet)

#         summary = self._build_episodic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="episodic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set, top_doc_ids

#     def retrieve_from_semantic_packets(
#         self,
#         query: str,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str]]:
#         top_k = top_k or self.semantic_top_k
#         retrieved_set = retrieved_set or set()
#         entries = self.semantic_memory.retrieve(query=query, top_k=top_k * 2, as_context=False)
#         if not entries:
#             return [], "[No semantic results]", retrieved_set, []

#         selected: List[SemanticTripleEntry] = []
#         support_doc_ids: List[str] = []
#         for entry in entries:
#             key = f"semantic:{entry.id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append(entry)
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 support_doc_ids.extend(self.semantic_memory.get_support_event_ids(entry, limit=2))
#             if len(selected) >= top_k:
#                 break

#         packets = [self._build_semantic_packet(entry) for entry in selected]
#         packets = [p for p in packets if p]
#         summary = self._build_semantic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="semantic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )

#         dedup_support: List[str] = []
#         seen: Set[str] = set()
#         for x in support_doc_ids:
#             if x and x not in seen:
#                 seen.add(x)
#                 dedup_support.append(x)
#         return items, summary, retrieved_set, dedup_support

#     def retrieve_from_visual_packets(
#         self,
#         query: str,
#         anchor_doc_ids: Optional[List[str]] = None,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str]]:
#         top_k = top_k or self.visual_top_k
#         retrieved_set = retrieved_set or set()
#         all_images: List[Image.Image] = []
#         summary = "[No visual results]"

#         if anchor_doc_ids:
#             doc_ids = [x for x in anchor_doc_ids if x][:top_k]
#             event_images = self.visual_memory.get_event_images(
#                 doc_ids,
#                 max_images_per_event=max(self.visual_top_k, 1),
#             )
#             if event_images:
#                 kept_doc_ids: List[str] = []
#                 for doc_id in doc_ids:
#                     images = event_images.get(doc_id, [])
#                     if not images:
#                         continue
#                     key = f"visual:{doc_id}"
#                     if key in retrieved_set:
#                         continue
#                     retrieved_set.add(key)
#                     kept_doc_ids.append(doc_id)
#                     all_images.extend(images)
#                 if kept_doc_ids and all_images:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)
#         else:
#             result = self.visual_memory.retrieve(query=query, top_k=top_k, as_context=True)
#             if isinstance(result, dict) and result:
#                 kept_doc_ids: List[str] = []
#                 for key, images in result.items():
#                     if f"visual:{key}" in retrieved_set:
#                         continue
#                     retrieved_set.add(f"visual:{key}")
#                     kept_doc_ids.append(key)
#                     all_images.extend(images)
#                 if kept_doc_ids:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)

#         items: List[RetrievedItem] = []
#         if all_images:
#             items.append(
#                 RetrievedItem(
#                     memory_type="visual",
#                     content=all_images,
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set

#     # -----------------------------------------------------
#     # main answer loop
#     # -----------------------------------------------------

#     def answer(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> QAResult:
#         if until_time and until_time > self.indexed_time:
#             self.index(until_time)

#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         family_info = self._classify_question_family(query=query, choices=choices, until_time=until_time)

#         retrieved_set: Set[str] = set()
#         retrieved_items: List[RetrievedItem] = []
#         round_history: List[Dict[str, Any]] = []
#         latest_anchor_doc_ids: List[str] = []

#         reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
#         round_num = 0
#         err_count = 0

#         while round_num < self.max_rounds and err_count < self.max_errors:
#             round_num += 1
#             logger.info("Reasoning round %d", round_num)

#             history_str = self._format_round_history(round_history)
#             user_content = (
#                 f"{full_query}\n\n"
#                 f"{self._family_hint_block(family_info)}\n\n"
#                 f"Round History:\n{history_str}\n\n"
#                 "Task:\n"
#                 'Step 1: Decide whether to "search" or "answer".\n'
#                 "Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."
#             )
#             reasoning_messages = copy.deepcopy(reasoning_prompt)
#             reasoning_messages.append({"role": "user", "content": user_content})

#             try:
#                 response = self.respond_llm_model.generate(reasoning_messages)
#                 reasoning_output = self._parse_reasoning_response(response)
#             except Exception as e:
#                 logger.error("Reasoning failed: %s", e)
#                 err_count += 1
#                 continue

#             logger.info("Decision: %s", reasoning_output.decision)
#             if reasoning_output.decision == "answer":
#                 break

#             if reasoning_output.decision != "search" or not reasoning_output.selected_memory:
#                 logger.warning("Invalid search decision payload")
#                 err_count += 1
#                 continue

#             memory_type = reasoning_output.selected_memory.memory_type
#             search_query = reasoning_output.selected_memory.search_query or query
#             logger.info("Searching %s: %s", memory_type, search_query)

#             new_items: List[RetrievedItem] = []
#             summary = "[No results]"
#             anchor_doc_ids: List[str] = []

#             try:
#                 if memory_type == "episodic":
#                     new_items, summary, retrieved_set, anchor_doc_ids = self.retrieve_from_episodic_packets(
#                         search_query,
#                         family_info=family_info,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                 elif memory_type == "semantic":
#                     new_items, summary, retrieved_set, anchor_doc_ids = self.retrieve_from_semantic_packets(
#                         search_query,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                 elif memory_type == "visual":
#                     new_items, summary, retrieved_set = self.retrieve_from_visual_packets(
#                         search_query,
#                         anchor_doc_ids=latest_anchor_doc_ids if latest_anchor_doc_ids else None,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                 else:
#                     logger.warning("Unknown memory type: %s", memory_type)
#                     err_count += 1
#                     continue
#             except Exception as e:
#                 logger.error("Retrieval from %s failed: %s", memory_type, e)
#                 err_count += 1
#                 continue

#             retrieved_items.extend(new_items)
#             round_history.append({
#                 "round_num": round_num,
#                 "decision": "search",
#                 "memory_type": memory_type,
#                 "search_query": search_query,
#                 "retrieved_content": summary,
#             })

#         logger.info("Generating answer from accumulated context")
#         qa_prompt = self.prompt_template_manager.render(self.qa_template_name)
#         qa_content: List[Dict[str, Any]] = [{
#             "type": "text",
#             "text": (
#                 f"{full_query}\n\n"
#                 "Important for answering:\n"
#                 "- Treat Query Time as the reference point for all relative temporal expressions.\n"
#                 "- Interpret 'before', 'after', 'earlier', 'later', 'recently', 'a few hours ago', 'first', and 'last' relative to Query Time.\n"
#                 "- Prefer evidence that satisfies the temporal constraint exactly.\n\n"
#                 "Context:\n"
#             )
#         }]
#         qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
#         qa_content.append({
#             "type": "text",
#             "text": "\n\nRetrieval history summary:\n" + self._format_round_history(round_history),
#         })
#         if choices:
#             qa_content.append({
#                 "type": "text",
#                 "text": "\nPlease provide only the final answer from the choices given (e.g. A, B, C, or D).",
#             })
#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({"role": "user", "content": qa_content})

#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error("Answer generation failed: %s", e)
#             answer = "Unable to generate answer"

#         return QAResult(
#             question=query,
#             answer=answer,
#             retrieved_items=retrieved_items,
#             round_history=round_history,
#             num_rounds=round_num,
#         )

#     def reset_index(self) -> None:
#         self.episodic_memory.reset_index()
#         self.semantic_memory.reset_index()
#         self.visual_memory.reset_index()
#         self.indexed_time = 0
#         logger.info("All memory indices reset")

#     def reset(self) -> None:
#         self.reset_index()
#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=self.embedding_model,
#             llm_model=self.retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=self.episodic_memory.granularities,
#             cache_tag=getattr(self.episodic_memory, "cache_tag", None),
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=self.embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=self.embedding_model)
#         logger.info("WorldMemory reset complete")


# """
# WorldMemory: retrieval-agent-based heterogeneous memory retrieval with unified evidence packets.

# This version uses:
# - LLM-only question-family classification (no heuristic routing)
# - retrieval-agent outer loop for heterogeneous memory selection
# - family-aware episodic retrieval with graph refinement handled inside EpisodicMemoryRAG
# - semantic and visual retrieval converted into unified evidence packets
# """

# import copy
# import json
# import logging
# import re
# from typing import Any, Dict, List, Optional, Set, Tuple

# from PIL import Image

# from ..embedding import EmbeddingModel
# from ..llm import LLMModel, PromptTemplateManager
# from .episodic.EpisodicMemory_rag import CaptionEntryRAG, EpisodicMemoryRAG
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .utils import *
# from .visual import VisualMemory

# logger = logging.getLogger(__name__)


# class WorldMemory:
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         retriever_llm_model: LLMModel,
#         respond_llm_model: Optional[LLMModel] = None,
#         prompt_template_manager: Optional[PromptTemplateManager] = None,
#         episodic_granularities: Optional[List[str]] = None,
#         episodic_cache_tag: Optional[str] = None,
#         qa_template_name: str = "qa_egolife",
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors
#         self.qa_template_name = qa_template_name

#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=embedding_model,
#             llm_model=retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=episodic_granularities,
#             cache_tag=episodic_cache_tag,
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=embedding_model)

#         self.indexed_time: int = 0
#         self.episodic_top_k: int = 4
#         self.semantic_top_k: int = 6
#         self.visual_top_k: int = 3

#     def set_retrieval_top_k(
#         self,
#         episodic: Optional[int] = None,
#         semantic: Optional[int] = None,
#         visual: Optional[int] = None,
#     ) -> None:
#         """
#         Backward-compatible helper for eval scripts.

#         Allows the evaluation script to set retrieval top-k values after
#         constructing WorldMemory.
#         """
#         if episodic is not None:
#             self.episodic_top_k = int(episodic)
#         if semantic is not None:
#             self.semantic_top_k = int(semantic)
#         if visual is not None:
#             self.visual_top_k = int(visual)

#         logger.info(
#             "Set retrieval top-k: episodic=%s, semantic=%s, visual=%s",
#             self.episodic_top_k,
#             self.semantic_top_k,
#             self.visual_top_k,
#         )


#     def cleanup(self) -> None:
#         """
#         Best-effort cleanup hook for multiprocessing eval.

#         Safe to call even if some submodules do not expose cleanup().
#         """
#         try:
#             if hasattr(self.episodic_memory, "cleanup"):
#                 self.episodic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Episodic cleanup failed: %s", e)

#         try:
#             if hasattr(self.semantic_memory, "cleanup"):
#                 self.semantic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Semantic cleanup failed: %s", e)

#         try:
#             if hasattr(self.visual_memory, "cleanup"):
#                 self.visual_memory.cleanup()
#         except Exception as e:
#             logger.warning("Visual cleanup failed: %s", e)

#         try:
#             import torch
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         except Exception:
#             pass

#         logger.info("WorldMemory cleanup complete")
#     # -----------------------------------------------------
#     # loading / indexing
#     # -----------------------------------------------------

#     def load_episodic_captions(
#         self,
#         caption_files: Optional[Dict[str, str]] = None,
#         caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
#     ) -> None:
#         if caption_files:
#             self.episodic_memory.load_captions_from_files(caption_files)
#         if caption_data:
#             self.episodic_memory.load_captions_from_data(caption_data)

#     def load_episodic_sidecar(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_files or graph_files:
#             self.episodic_memory.load_sidecar_from_files(
#                 triplet_files=triplet_files,
#                 graph_files=graph_files,
#             )
#         if triplet_data or graph_data:
#             self.episodic_memory.load_sidecar_from_data(
#                 triplet_data=triplet_data,
#                 graph_data=graph_data,
#             )

#     def load_semantic_triples(
#         self,
#         file_path: Optional[str] = None,
#         data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if file_path:
#             self.semantic_memory.load_triples_from_file(file_path)
#         if data:
#             self.semantic_memory.load_triples_from_data(data)

#     def load_visual_clips(
#         self,
#         embeddings_path: Optional[str] = None,
#         clips_path: Optional[str] = None,
#         clips_data: Optional[List[Dict[str, Any]]] = None,
#     ) -> None:
#         if embeddings_path:
#             self.visual_memory.load_embeddings_from_file(embeddings_path)
#         if clips_path:
#             self.visual_memory.load_clips_from_file(clips_path)
#         if clips_data:
#             self.visual_memory.load_clips_from_data(clips_data)

#     def prepare_episodic_dense_index(self, force_rebuild: bool = False) -> None:
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=force_rebuild)

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug("Already indexed up to %s, skipping", self.indexed_time)
#             return

#         logger.info("Indexing all memories up to %s", transform_timestamp(str(until_time)))
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=False)
#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
#         self.indexed_time = until_time
#         logger.info("Indexing complete for all memory types")

#     # -----------------------------------------------------
#     # query / parsing helpers
#     # -----------------------------------------------------

#     def _build_query_with_time(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> str:
#         lines = [f"Query: {query}"]
#         if until_time is not None:
#             lines.append(f"Query Time: {transform_timestamp(str(until_time))}")
#             lines.append(
#                 "Important: Interpret all relative temporal expressions "
#                 '(e.g. "before", "after", "earlier", "later", "recently", '
#                 '"a few hours ago", "first", "last") relative to Query Time.'
#             )
#         if choices:
#             choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
#             lines.append(f"Choices: {choices_str}")
#         return "\n".join(lines)

#     def _parse_json_object(self, response: str) -> Dict[str, Any]:
#         try:
#             json_match = re.search(r"\{.*\}", response, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group())
#             return json.loads(response)
#         except Exception:
#             return {}

#     def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
#         try:
#             data = self._parse_json_object(response)
#             decision = str(data.get("decision", "answer")).lower()
#             reason = data.get("reason")
#             selected_memory = None
#             if decision == "search" and "selected_memory" in data:
#                 mem_data = data["selected_memory"]
#                 selected_memory = MemorySearchOutput(
#                     memory_type=str(mem_data.get("memory_type", "")).lower(),
#                     search_query=str(mem_data.get("search_query", "")),
#                 )
#             return ReasoningOutput(decision=decision, selected_memory=selected_memory, reason=reason)
#         except Exception as e:
#             logger.warning("Failed to parse reasoning response: %s", e)
#             return ReasoningOutput(decision="answer")

#     def _format_round_history(self, rounds: List[Dict[str, Any]]) -> str:
#         if not rounds:
#             return "[]"
#         lines = []
#         for r in rounds:
#             lines.append(
#                 f"### Round {r['round_num']}\n"
#                 f"Decision: {r['decision']}\n"
#                 f"Memory: {r['memory_type']}\n"
#                 f"Search Query: {r['search_query']}\n"
#                 f"Retrieved:\n{r['retrieved_content']}"
#             )
#         return "\n\n".join(lines)

#     def _render_retrieved_items_for_qa(self, retrieved_items: List[RetrievedItem]) -> List[Dict[str, Any]]:
#         messages: List[Dict[str, Any]] = []
#         for item in retrieved_items:
#             if item.memory_type in ("episodic", "semantic"):
#                 messages.append({"type": "text", "text": item.content})
#             elif item.memory_type == "visual":
#                 if isinstance(item.content, list):
#                     for img in item.content:
#                         if isinstance(img, Image.Image):
#                             messages.append({"type": "image", "image": img})
#                         elif isinstance(img, dict) and "image" in img:
#                             messages.append({"type": "image", "image": img["image"]})
#         return messages

#     def _clean_text(self, text: Any) -> str:
#         if text is None:
#             return ""
#         text = str(text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

#     def _short_text(self, text: Any, max_chars: int = 180) -> str:
#         text = self._clean_text(text)
#         if len(text) <= max_chars:
#             return text
#         return text[: max_chars - 3].rstrip() + "..."

#     def _build_episodic_history_summary(
#         self,
#         selected: List[Tuple[CaptionEntryRAG, float]],
#         max_items: int = 5,
#     ) -> str:
#         if not selected:
#             return "[No episodic results]"

#         lines = ["Retrieved episodic evidence:"]
#         for entry, score in selected[:max_items]:
#             start_ts, end_ts = entry.timestamp_int
#             time_span = f"{transform_timestamp(str(start_ts))} - {transform_timestamp(str(end_ts))}"

#             main_text = self._clean_text(entry.text)

#             visual_bits: List[str] = []
#             if entry.visual_summary:
#                 visual_bits.append(self._clean_text(entry.visual_summary))

#             visual_entry = self.visual_memory.get_clip_by_doc_id(entry.doc_id)
#             if visual_entry is not None:
#                 keyframe_caption = self._clean_text(getattr(visual_entry, "keyframe_caption", ""))
#                 if keyframe_caption and not visual_bits:
#                     visual_bits.append(keyframe_caption)

#                 scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#                 if isinstance(scene_summary, dict):
#                     dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                     if dominant_scene:
#                         visual_bits.append(f"scene={dominant_scene}")

#                 visual_objects = list(getattr(visual_entry, "visual_objects", []) or [])
#                 if visual_objects:
#                     visual_bits.append("objects=" + ", ".join(map(str, visual_objects[:8])))

#             critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#             speech_text = ""
#             if critical_lines:
#                 speech_text = self._clean_text(" | ".join([str(x) for x in critical_lines if str(x).strip()]))

#             line = f"- {time_span}: {main_text}"
#             if visual_bits:
#                 line += " | Visual: " + " ; ".join(visual_bits)
#             if speech_text:
#                 line += f" | Speech: {speech_text}"
#             line += f" | score={score:.4f}"
#             lines.append(line)

#         return "\n".join(lines)

#     def _build_semantic_history_summary(
#         self,
#         selected: List[SemanticTripleEntry],
#         max_items: int = 3,
#     ) -> str:
#         if not selected:
#             return "[No semantic results]"

#         lines = ["Retrieved semantic evidence:"]
#         for entry in selected[:max_items]:
#             fact_text = self._clean_text(entry.to_display_str())

#             support_text = ""
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#                 if support_ids:
#                     support_text = f" | support_events={support_ids}"

#             conf_text = ""
#             conf = getattr(entry, "confidence", None)
#             if conf is not None:
#                 try:
#                     conf_text = f" | confidence={float(conf):.4f}"
#                 except Exception:
#                     pass

#             lines.append(f"- {fact_text}{support_text}{conf_text}")

#         return "\n".join(lines)

#     def _build_visual_history_summary_from_doc_ids(
#         self,
#         doc_ids: List[str],
#         max_items: int = 3,
#     ) -> str:
#         if not doc_ids:
#             return "[No visual results]"

#         lines = ["Retrieved visual evidence:"]
#         kept = 0
#         for doc_id in doc_ids:
#             clip = self.visual_memory.get_clip_by_doc_id(doc_id)
#             if clip is None:
#                 continue

#             span_text = doc_id
#             if hasattr(clip, "timestamp_int"):
#                 try:
#                     s, e = clip.timestamp_int
#                     span_text = f"{transform_timestamp(str(s))} - {transform_timestamp(str(e))}"
#                 except Exception:
#                     span_text = doc_id

#             keyframe_caption = self._clean_text(getattr(clip, "keyframe_caption", ""))

#             scene_summary = getattr(clip, "scene_summary", {}) or {}
#             dominant_scene = ""
#             scene_desc = ""
#             if isinstance(scene_summary, dict):
#                 dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                 scene_desc = self._clean_text(scene_summary.get("scene_description", ""))

#             visual_objects = list(getattr(clip, "visual_objects", []) or [])
#             objects_text = ", ".join(map(str, visual_objects[:10])) if visual_objects else ""

#             parts = []
#             if keyframe_caption:
#                 parts.append(keyframe_caption)
#             if dominant_scene:
#                 parts.append(f"scene={dominant_scene}")
#             if scene_desc:
#                 parts.append(f"scene_description={scene_desc}")
#             if objects_text:
#                 parts.append(f"objects={objects_text}")

#             if not parts:
#                 parts.append("visual evidence available")

#             lines.append(f"- {span_text}: " + " | ".join(parts))
#             kept += 1
#             if kept >= max_items:
#                 break

#         if kept == 0:
#             return "[Visual images retrieved, but no textual visual summary available]"
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # LLM-only question-family classifier
#     # -----------------------------------------------------

#     def _default_family_info(self) -> Dict[str, Any]:
#         return {
#             "question_family": "event",
#             "graph_mode": "default",
#             "time_bias": "none",
#             "need_visual_followup": False,
#         }

#     def _llm_family_classifier(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         system_prompt = (
#             "You classify long-video QA questions into exactly one question family and output JSON only. "
#             "Valid families: source-trace, temporal-recall, action-owner, participant-membership, "
#             "plan-intention-decision, habit-preference, attribute-content-purpose, event. "
#             "Also output graph_mode, time_bias (forward/backward/none), and need_visual_followup (true/false). "
#             "Be conservative and choose the simplest valid family."
#         )
#         family_guide = (
#             "Family guidance:\n"
#             "- source-trace: prior location, transfer path, brought from, placed before, where X was before. graph_mode=backtrack_object_source, time_bias=backward, need_visual_followup=false\n"
#             "- temporal-recall: first/last/before/after/earlier/later. graph_mode=temporal_walk, time_bias=backward or forward or none, need_visual_followup=false\n"
#             "- action-owner: who did/used/moved/picked/brought something. graph_mode=actor_action_refine, time_bias=none, need_visual_followup=false\n"
#             "- participant-membership: who was present / with whom / who joined or left. graph_mode=participant_cooccurrence_refine, time_bias=none, need_visual_followup=false\n"
#             "- plan-intention-decision: plans, rationale, commitments, intended next step. graph_mode=topic_commitment_refine, time_bias=none, need_visual_followup=false\n"
#             "- habit-preference: repeated patterns, often/usually/prefer. graph_mode=habit_support_only, time_bias=none, need_visual_followup=false\n"
#             "- attribute-content-purpose: color, appearance, what was inside, visual identity, purpose requiring direct grounding. graph_mode=anchor_refine_then_visual, time_bias=none, need_visual_followup=true\n"
#             "- event: generic specific-event questions that do not strongly fit the above. graph_mode=default, time_bias=none, need_visual_followup=false"
#         )
#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         user_prompt = (
#             f"{full_query}\n\n"
#             f"{family_guide}\n\n"
#             "Return JSON with keys: question_family, graph_mode, time_bias, need_visual_followup"
#         )
#         fallback = self._default_family_info()
#         try:
#             response = self.respond_llm_model.generate([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ])
#             data = self._parse_json_object(response)
#             if not data:
#                 return fallback
#             return {
#                 "question_family": str(data.get("question_family", fallback["question_family"])).strip().lower(),
#                 "graph_mode": str(data.get("graph_mode", fallback["graph_mode"])).strip().lower(),
#                 "time_bias": str(data.get("time_bias", fallback["time_bias"])).strip().lower(),
#                 "need_visual_followup": bool(data.get("need_visual_followup", fallback["need_visual_followup"])),
#             }
#         except Exception as e:
#             logger.warning("LLM family classifier failed, fallback to default family info: %s", e)
#             return fallback

#     def _classify_question_family(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         return self._llm_family_classifier(query=query, choices=choices, until_time=until_time)

#     def _family_retrieval_prior(self, family_info: Dict[str, Any]) -> str:
#         family = family_info.get("question_family", "event")
#         mapping = {
#             "source-trace": "Search for earlier source-establishing events rather than merely the current location.",
#             "temporal-recall": "Search for the event that best satisfies first/last/before/after constraints.",
#             "action-owner": "Search for explicit actor-action evidence, not just nearby participation.",
#             "participant-membership": "Search for explicit participation/co-presence evidence across relevant event windows.",
#             "plan-intention-decision": "Search for explicit commitments, decisions, or rationale statements, often in speech or topic context.",
#             "habit-preference": "Search for repeated or aggregate evidence; semantic memory is often useful.",
#             "attribute-content-purpose": "Search for direct grounding of the queried property; visual follow-up may help.",
#             "event": "Search for the most predicate-aligned event evidence first.",
#         }
#         return mapping.get(family, mapping["event"])

#     def _family_hint_block(self, family_info: Dict[str, Any]) -> str:
#         return (
#             "Question family hint:\n"
#             f"- question_family: {family_info.get('question_family', 'event')}\n"
#             f"- graph_mode_if_episodic: {family_info.get('graph_mode', 'default')}\n"
#             f"- time_bias: {family_info.get('time_bias', 'none')}\n"
#             f"- need_visual_followup: {str(bool(family_info.get('need_visual_followup', False))).lower()}\n\n"
#             "Family-specific retrieval prior:\n"
#             f"- {self._family_retrieval_prior(family_info)}\n"
#             "Use this as a prior, but revise based on round history if needed."
#         )

#     # -----------------------------------------------------
#     # packet builders
#     # -----------------------------------------------------

#     def _build_event_packet(
#         self,
#         doc_id: str,
#         score: float,
#         supporting_facts: Optional[List[SemanticTripleEntry]] = None,
#     ) -> str:
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return ""

#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         raw_triplets = []
#         if hasattr(self.episodic_memory, "get_raw_triplets_by_doc_id"):
#             raw_triplets = self.episodic_memory.get_raw_triplets_by_doc_id(doc_id, "30sec")
#         supporting_facts = supporting_facts or []
#         parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min") if hasattr(self.episodic_memory, "get_parent_caption") else None

#         lines: List[str] = []
#         lines.append(f"Event Anchor: {doc_id}")
#         lines.append(f"Relevance Score: {score:.4f}")
#         lines.append(entry.to_display_str(include_visual_summary=True))

#         critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#         if critical_lines:
#             lines.append("Critical Speech:")
#             for line in critical_lines[:3]:
#                 if str(line).strip():
#                     lines.append(f"- {line}")

#         if parent_3min is not None and parent_3min.doc_id != doc_id:
#             p_start, p_end = parent_3min.timestamp_int
#             lines.append(
#                 f"3min Context [{transform_timestamp(str(p_start))} - {transform_timestamp(str(p_end))}]: {parent_3min.text}"
#             )
#             if parent_3min.visual_summary:
#                 lines.append(f"3min Visual: {parent_3min.visual_summary}")

#         if visual_entry is not None:
#             if getattr(visual_entry, "keyframe_caption", ""):
#                 lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
#             visual_objects = getattr(visual_entry, "visual_objects", []) or []
#             if visual_objects:
#                 lines.append("Visual Objects: " + ", ".join(visual_objects[:8]))
#             scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#             if isinstance(scene_summary, dict) and scene_summary.get("dominant_scene"):
#                 lines.append(f"Scene: {scene_summary.get('dominant_scene')}")

#         if raw_triplets:
#             lines.append("Raw Triplets:")
#             for tri in raw_triplets[:8]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if triplets:
#             lines.append("Canonical Episodic Triplets:")
#             for tri in triplets[:6]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if supporting_facts:
#             lines.append("Supporting Semantic Facts:")
#             for fact in supporting_facts[:3]:
#                 lines.append(f"- {fact.to_display_str()}")

#         return "\n".join(lines)

#     def _build_semantic_packet(self, entry: SemanticTripleEntry) -> str:
#         lines = [f"Semantic Packet: {entry.to_display_str()}"]
#         support_ids: List[str] = []
#         if hasattr(self.semantic_memory, "get_support_event_ids"):
#             support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#         if support_ids:
#             lines.append("Support Event IDs: " + ", ".join(support_ids))
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # retrieval helpers
#     # -----------------------------------------------------

#     def retrieve_from_episodic_packets(
#         self,
#         query: str,
#         family_info: Dict[str, Any],
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str]]:
#         top_k = top_k or self.episodic_top_k
#         retrieved_set = retrieved_set or set()

#         ranked = self.episodic_memory.retrieve_ranked_with_family(
#             query=query,
#             family_info=family_info,
#         )
#         if not ranked:
#             return [], "[No episodic results]", retrieved_set, []

#         selected: List[Tuple[CaptionEntryRAG, float]] = []
#         for entry, score in ranked:
#             key = f"episodic:{entry.doc_id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append((entry, score))
#             if len(selected) >= top_k:
#                 break

#         if not selected:
#             return [], "[No new episodic results]", retrieved_set, []

#         packets: List[str] = []
#         top_doc_ids: List[str] = []
#         for entry, score in selected:
#             top_doc_ids.append(entry.doc_id)
#             packet = self._build_event_packet(doc_id=entry.doc_id, score=score)
#             if packet:
#                 packets.append(packet)

#         summary = self._build_episodic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="episodic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set, top_doc_ids

#     def retrieve_from_semantic_packets(
#         self,
#         query: str,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str]]:
#         top_k = top_k or self.semantic_top_k
#         retrieved_set = retrieved_set or set()
#         entries = self.semantic_memory.retrieve(query=query, top_k=top_k * 2, as_context=False)
#         if not entries:
#             return [], "[No semantic results]", retrieved_set, []

#         selected: List[SemanticTripleEntry] = []
#         support_doc_ids: List[str] = []
#         for entry in entries:
#             key = f"semantic:{entry.id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append(entry)
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 support_doc_ids.extend(self.semantic_memory.get_support_event_ids(entry, limit=2))
#             if len(selected) >= top_k:
#                 break

#         packets = [self._build_semantic_packet(entry) for entry in selected]
#         packets = [p for p in packets if p]
#         summary = self._build_semantic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="semantic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )

#         dedup_support: List[str] = []
#         seen: Set[str] = set()
#         for x in support_doc_ids:
#             if x and x not in seen:
#                 seen.add(x)
#                 dedup_support.append(x)
#         return items, summary, retrieved_set, dedup_support

#     def retrieve_from_visual_packets(
#         self,
#         query: str,
#         anchor_doc_ids: Optional[List[str]] = None,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str]]:
#         top_k = top_k or self.visual_top_k
#         retrieved_set = retrieved_set or set()
#         all_images: List[Image.Image] = []
#         summary = "[No visual results]"

#         if anchor_doc_ids:
#             doc_ids = [x for x in anchor_doc_ids if x][:top_k]
#             event_images = self.visual_memory.get_event_images(
#                 doc_ids,
#                 max_images_per_event=max(self.visual_top_k, 1),
#             )
#             if event_images:
#                 kept_doc_ids: List[str] = []
#                 for doc_id in doc_ids:
#                     images = event_images.get(doc_id, [])
#                     if not images:
#                         continue
#                     key = f"visual:{doc_id}"
#                     if key in retrieved_set:
#                         continue
#                     retrieved_set.add(key)
#                     kept_doc_ids.append(doc_id)
#                     all_images.extend(images)
#                 if kept_doc_ids and all_images:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)
#         else:
#             result = self.visual_memory.retrieve(query=query, top_k=top_k, as_context=True)
#             if isinstance(result, dict) and result:
#                 kept_doc_ids: List[str] = []
#                 for key, images in result.items():
#                     if f"visual:{key}" in retrieved_set:
#                         continue
#                     retrieved_set.add(f"visual:{key}")
#                     kept_doc_ids.append(key)
#                     all_images.extend(images)
#                 if kept_doc_ids:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)

#         items: List[RetrievedItem] = []
#         if all_images:
#             items.append(
#                 RetrievedItem(
#                     memory_type="visual",
#                     content=all_images,
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set

#     # -----------------------------------------------------
#     # main answer loop
#     # -----------------------------------------------------

#     def answer(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> QAResult:
#         if until_time and until_time > self.indexed_time:
#             self.index(until_time)

#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         family_info = self._classify_question_family(query=query, choices=choices, until_time=until_time)

#         retrieved_set: Set[str] = set()
#         retrieved_items: List[RetrievedItem] = []
#         round_history: List[Dict[str, Any]] = []
#         latest_anchor_doc_ids: List[str] = []

#         reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
#         round_num = 0
#         err_count = 0

#         while round_num < self.max_rounds and err_count < self.max_errors:
#             round_num += 1
#             logger.info("Reasoning round %d", round_num)

#             history_str = self._format_round_history(round_history)
#             user_content = (
#                 f"{full_query}\n\n"
#                 f"{self._family_hint_block(family_info)}\n\n"
#                 f"Round History:\n{history_str}\n\n"
#                 "Task:\n"
#                 'Step 1: Decide whether to "search" or "answer".\n'
#                 "Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."
#             )
#             reasoning_messages = copy.deepcopy(reasoning_prompt)
#             reasoning_messages.append({"role": "user", "content": user_content})

#             try:
#                 response = self.respond_llm_model.generate(reasoning_messages)
#                 reasoning_output = self._parse_reasoning_response(response)
#             except Exception as e:
#                 logger.error("Reasoning failed: %s", e)
#                 err_count += 1
#                 continue

#             logger.info("Decision: %s", reasoning_output.decision)
#             if reasoning_output.decision == "answer":
#                 break

#             if reasoning_output.decision != "search" or not reasoning_output.selected_memory:
#                 logger.warning("Invalid search decision payload")
#                 err_count += 1
#                 continue

#             memory_type = reasoning_output.selected_memory.memory_type
#             search_query = reasoning_output.selected_memory.search_query or query
#             logger.info("Searching %s: %s", memory_type, search_query)

#             new_items: List[RetrievedItem] = []
#             summary = "[No results]"
#             anchor_doc_ids: List[str] = []

#             try:
#                 if memory_type == "episodic":
#                     new_items, summary, retrieved_set, anchor_doc_ids = self.retrieve_from_episodic_packets(
#                         search_query,
#                         family_info=family_info,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                 elif memory_type == "semantic":
#                     new_items, summary, retrieved_set, anchor_doc_ids = self.retrieve_from_semantic_packets(
#                         search_query,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                 elif memory_type == "visual":
#                     new_items, summary, retrieved_set = self.retrieve_from_visual_packets(
#                         search_query,
#                         anchor_doc_ids=latest_anchor_doc_ids if latest_anchor_doc_ids else None,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                 else:
#                     logger.warning("Unknown memory type: %s", memory_type)
#                     err_count += 1
#                     continue
#             except Exception as e:
#                 logger.error("Retrieval from %s failed: %s", memory_type, e)
#                 err_count += 1
#                 continue

#             retrieved_items.extend(new_items)
#             round_history.append({
#                 "round_num": round_num,
#                 "decision": "search",
#                 "memory_type": memory_type,
#                 "search_query": search_query,
#                 "retrieved_content": summary,
#             })

#         logger.info("Generating answer from accumulated context")
#         qa_prompt = self.prompt_template_manager.render(self.qa_template_name)
#         qa_content: List[Dict[str, Any]] = [{
#             "type": "text",
#             "text": (
#                 f"{full_query}\n\n"
#                 "Important for answering:\n"
#                 "- Treat Query Time as the reference point for all relative temporal expressions.\n"
#                 "- Interpret 'before', 'after', 'earlier', 'later', 'recently', 'a few hours ago', 'first', and 'last' relative to Query Time.\n"
#                 "- Prefer evidence that satisfies the temporal constraint exactly.\n\n"
#                 "Context:\n"
#             )
#         }]
#         qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
#         qa_content.append({
#             "type": "text",
#             "text": "\n\nRetrieval history summary:\n" + self._format_round_history(round_history),
#         })
#         if choices:
#             qa_content.append({
#                 "type": "text",
#                 "text": "\nPlease provide only the final answer from the choices given (e.g. A, B, C, or D).",
#             })
#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({"role": "user", "content": qa_content})

#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error("Answer generation failed: %s", e)
#             answer = "Unable to generate answer"

#         return QAResult(
#             question=query,
#             answer=answer,
#             retrieved_items=retrieved_items,
#             round_history=round_history,
#             num_rounds=round_num,
#         )

#     def reset_index(self) -> None:
#         self.episodic_memory.reset_index()
#         self.semantic_memory.reset_index()
#         self.visual_memory.reset_index()
#         self.indexed_time = 0
#         logger.info("All memory indices reset")

#     def reset(self) -> None:
#         self.reset_index()
#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=self.embedding_model,
#             llm_model=self.retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=self.episodic_memory.granularities,
#             cache_tag=getattr(self.episodic_memory, "cache_tag", None),
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=self.embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=self.embedding_model)
#         logger.info("WorldMemory reset complete")


# """
# WorldMemory: retrieval-agent-based heterogeneous memory retrieval with unified evidence packets.

# This version uses:
# - LLM-only question-family classification (no heuristic routing)
# - retrieval-agent outer loop for heterogeneous memory selection
# - family-aware episodic retrieval with graph refinement handled inside EpisodicMemoryRAG
# - semantic and visual retrieval converted into unified evidence packets
# """

# import copy
# import json
# import logging
# import re
# from typing import Any, Dict, List, Optional, Set, Tuple
# from collections import defaultdict

# from PIL import Image

# from ..embedding import EmbeddingModel
# from ..llm import LLMModel, PromptTemplateManager
# from .episodic.EpisodicMemory_rag import CaptionEntryRAG, EpisodicMemoryRAG
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .utils import *
# from .visual import VisualMemory

# logger = logging.getLogger(__name__)


# class WorldMemory:
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         retriever_llm_model: LLMModel,
#         respond_llm_model: Optional[LLMModel] = None,
#         prompt_template_manager: Optional[PromptTemplateManager] = None,
#         episodic_granularities: Optional[List[str]] = None,
#         episodic_cache_tag: Optional[str] = None,
#         qa_template_name: str = "qa_egolife",
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors
#         self.qa_template_name = qa_template_name

#         self.episodic_memory = EpisodicMemoryRAG(
#             embedding_model=embedding_model,
#             llm_model=retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=episodic_granularities,
#             cache_tag=episodic_cache_tag,
#         )
#         self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
#         self.visual_memory = VisualMemory(embedding_model=embedding_model)

#         self.indexed_time: int = 0
#         self.episodic_top_k: int = 4
#         self.semantic_top_k: int = 6
#         self.visual_top_k: int = 3
#         self.selector_top_k: int = 4
#         self.semantic_bridge_top_k: int = 6
#         self.semantic_bridge_history_top_k: int = 3

#     def set_retrieval_top_k(
#         self,
#         episodic: Optional[int] = None,
#         semantic: Optional[int] = None,
#         visual: Optional[int] = None,
#     ) -> None:
#         """
#         Backward-compatible helper for eval scripts.

#         Allows the evaluation script to set retrieval top-k values after
#         constructing WorldMemory.
#         """
#         if episodic is not None:
#             self.episodic_top_k = int(episodic)
#         if semantic is not None:
#             self.semantic_top_k = int(semantic)
#         if visual is not None:
#             self.visual_top_k = int(visual)

#         logger.info(
#             "Set retrieval top-k: episodic=%s, semantic=%s, visual=%s",
#             self.episodic_top_k,
#             self.semantic_top_k,
#             self.visual_top_k,
#         )


#     def cleanup(self) -> None:
#         """
#         Best-effort cleanup hook for multiprocessing eval.

#         Safe to call even if some submodules do not expose cleanup().
#         """
#         try:
#             if hasattr(self.episodic_memory, "cleanup"):
#                 self.episodic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Episodic cleanup failed: %s", e)

#         try:
#             if hasattr(self.semantic_memory, "cleanup"):
#                 self.semantic_memory.cleanup()
#         except Exception as e:
#             logger.warning("Semantic cleanup failed: %s", e)

#         try:
#             if hasattr(self.visual_memory, "cleanup"):
#                 self.visual_memory.cleanup()
#         except Exception as e:
#             logger.warning("Visual cleanup failed: %s", e)

#         try:
#             import torch
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         except Exception:
#             pass

#         logger.info("WorldMemory cleanup complete")
#     # -----------------------------------------------------
#     # loading / indexing
#     # -----------------------------------------------------

#     def load_episodic_captions(
#         self,
#         caption_files: Optional[Dict[str, str]] = None,
#         caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
#     ) -> None:
#         if caption_files:
#             self.episodic_memory.load_captions_from_files(caption_files)
#         if caption_data:
#             self.episodic_memory.load_captions_from_data(caption_data)

#     def load_episodic_sidecar(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_files or graph_files:
#             self.episodic_memory.load_sidecar_from_files(
#                 triplet_files=triplet_files,
#                 graph_files=graph_files,
#             )
#         if triplet_data or graph_data:
#             self.episodic_memory.load_sidecar_from_data(
#                 triplet_data=triplet_data,
#                 graph_data=graph_data,
#             )

#     def load_semantic_triples(
#         self,
#         file_path: Optional[str] = None,
#         data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if file_path:
#             self.semantic_memory.load_triples_from_file(file_path)
#         if data:
#             self.semantic_memory.load_triples_from_data(data)

#     def load_visual_clips(
#         self,
#         embeddings_path: Optional[str] = None,
#         clips_path: Optional[str] = None,
#         clips_data: Optional[List[Dict[str, Any]]] = None,
#     ) -> None:
#         if embeddings_path:
#             self.visual_memory.load_embeddings_from_file(embeddings_path)
#         if clips_path:
#             self.visual_memory.load_clips_from_file(clips_path)
#         if clips_data:
#             self.visual_memory.load_clips_from_data(clips_data)

#     def prepare_episodic_dense_index(self, force_rebuild: bool = False) -> None:
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=force_rebuild)

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug("Already indexed up to %s, skipping", self.indexed_time)
#             return

#         logger.info("Indexing all memories up to %s", transform_timestamp(str(until_time)))
#         if hasattr(self.episodic_memory, "build_dense_index"):
#             self.episodic_memory.build_dense_index(force_rebuild=False)
#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
#         self.indexed_time = until_time
#         logger.info("Indexing complete for all memory types")

#     # -----------------------------------------------------
#     # query / parsing helpers
#     # -----------------------------------------------------

#     def _build_query_with_time(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> str:
#         lines = [f"Query: {query}"]
#         if until_time is not None:
#             lines.append(f"Query Time: {transform_timestamp(str(until_time))}")
#             lines.append(
#                 "Important: Interpret all relative temporal expressions "
#                 '(e.g. "before", "after", "earlier", "later", "recently", '
#                 '"a few hours ago", "first", "last") relative to Query Time.'
#             )
#         if choices:
#             choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
#             lines.append(f"Choices: {choices_str}")
#         return "\n".join(lines)

#     def _parse_json_object(self, response: str) -> Dict[str, Any]:
#         try:
#             json_match = re.search(r"\{.*\}", response, re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group())
#             return json.loads(response)
#         except Exception:
#             return {}

#     def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
#         try:
#             data = self._parse_json_object(response)
#             decision = str(data.get("decision", "answer")).lower()
#             reason = data.get("reason")
#             selected_memory = None
#             if decision == "search" and "selected_memory" in data:
#                 mem_data = data["selected_memory"]
#                 selected_memory = MemorySearchOutput(
#                     memory_type=str(mem_data.get("memory_type", "")).lower(),
#                     search_query=str(mem_data.get("search_query", "")),
#                 )
#             return ReasoningOutput(decision=decision, selected_memory=selected_memory, reason=reason)
#         except Exception as e:
#             logger.warning("Failed to parse reasoning response: %s", e)
#             return ReasoningOutput(decision="answer")

#     def _format_round_history(self, rounds: List[Dict[str, Any]]) -> str:
#         if not rounds:
#             return "[]"
#         lines = []
#         for r in rounds:
#             lines.append(
#                 f"### Round {r['round_num']}\n"
#                 f"Decision: {r['decision']}\n"
#                 f"Memory: {r['memory_type']}\n"
#                 f"Search Query: {r['search_query']}\n"
#                 f"Retrieved:\n{r['retrieved_content']}"
#             )
#         return "\n\n".join(lines)

#     def _render_retrieved_items_for_qa(self, retrieved_items: List[RetrievedItem]) -> List[Dict[str, Any]]:
#         messages: List[Dict[str, Any]] = []
#         for item in retrieved_items:
#             if item.memory_type in ("episodic", "semantic"):
#                 messages.append({"type": "text", "text": item.content})
#             elif item.memory_type == "visual":
#                 if isinstance(item.content, list):
#                     for img in item.content:
#                         if isinstance(img, Image.Image):
#                             messages.append({"type": "image", "image": img})
#                         elif isinstance(img, dict) and "image" in img:
#                             messages.append({"type": "image", "image": img["image"]})
#         return messages

#     def _clean_text(self, text: Any) -> str:
#         if text is None:
#             return ""
#         text = str(text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text

#     def _short_text(self, text: Any, max_chars: int = 180) -> str:
#         text = self._clean_text(text)
#         if len(text) <= max_chars:
#             return text
#         return text[: max_chars - 3].rstrip() + "..."

#     def _build_episodic_history_summary(
#         self,
#         selected: List[Tuple[CaptionEntryRAG, float]],
#         max_items: int = 5,
#     ) -> str:
#         if not selected:
#             return "[No episodic results]"

#         lines = ["Retrieved episodic evidence:"]
#         for entry, score in selected[:max_items]:
#             start_ts, end_ts = entry.timestamp_int
#             time_span = f"{transform_timestamp(str(start_ts))} - {transform_timestamp(str(end_ts))}"

#             main_text = self._clean_text(entry.text)

#             visual_bits: List[str] = []
#             if entry.visual_summary:
#                 visual_bits.append(self._clean_text(entry.visual_summary))

#             visual_entry = self.visual_memory.get_clip_by_doc_id(entry.doc_id)
#             if visual_entry is not None:
#                 keyframe_caption = self._clean_text(getattr(visual_entry, "keyframe_caption", ""))
#                 if keyframe_caption and not visual_bits:
#                     visual_bits.append(keyframe_caption)

#                 scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#                 if isinstance(scene_summary, dict):
#                     dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                     if dominant_scene:
#                         visual_bits.append(f"scene={dominant_scene}")

#                 visual_objects = list(getattr(visual_entry, "visual_objects", []) or [])
#                 if visual_objects:
#                     visual_bits.append("objects=" + ", ".join(map(str, visual_objects[:8])))

#             critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#             speech_text = ""
#             if critical_lines:
#                 speech_text = self._clean_text(" | ".join([str(x) for x in critical_lines if str(x).strip()]))

#             line = f"- {time_span}: {main_text}"
#             if visual_bits:
#                 line += " | Visual: " + " ; ".join(visual_bits)
#             if speech_text:
#                 line += f" | Speech: {speech_text}"
#             line += f" | score={score:.4f}"
#             lines.append(line)

#         return "\n".join(lines)

#     def _build_semantic_history_summary(
#         self,
#         selected: List[SemanticTripleEntry],
#         max_items: int = 3,
#     ) -> str:
#         if not selected:
#             return "[No semantic results]"

#         lines = ["Retrieved semantic evidence:"]
#         for entry in selected[:max_items]:
#             fact_text = self._clean_text(entry.to_display_str())

#             support_text = ""
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#                 if support_ids:
#                     support_text = f" | support_events={support_ids}"

#             conf_text = ""
#             conf = getattr(entry, "confidence", None)
#             if conf is not None:
#                 try:
#                     conf_text = f" | confidence={float(conf):.4f}"
#                 except Exception:
#                     pass

#             lines.append(f"- {fact_text}{support_text}{conf_text}")

#         return "\n".join(lines)

#     def _build_visual_history_summary_from_doc_ids(
#         self,
#         doc_ids: List[str],
#         max_items: int = 3,
#     ) -> str:
#         if not doc_ids:
#             return "[No visual results]"

#         lines = ["Retrieved visual evidence:"]
#         kept = 0
#         for doc_id in doc_ids:
#             clip = self.visual_memory.get_clip_by_doc_id(doc_id)
#             if clip is None:
#                 continue

#             span_text = doc_id
#             if hasattr(clip, "timestamp_int"):
#                 try:
#                     s, e = clip.timestamp_int
#                     span_text = f"{transform_timestamp(str(s))} - {transform_timestamp(str(e))}"
#                 except Exception:
#                     span_text = doc_id

#             keyframe_caption = self._clean_text(getattr(clip, "keyframe_caption", ""))

#             scene_summary = getattr(clip, "scene_summary", {}) or {}
#             dominant_scene = ""
#             scene_desc = ""
#             if isinstance(scene_summary, dict):
#                 dominant_scene = self._clean_text(scene_summary.get("dominant_scene", ""))
#                 scene_desc = self._clean_text(scene_summary.get("scene_description", ""))

#             visual_objects = list(getattr(clip, "visual_objects", []) or [])
#             objects_text = ", ".join(map(str, visual_objects[:10])) if visual_objects else ""

#             parts = []
#             if keyframe_caption:
#                 parts.append(keyframe_caption)
#             if dominant_scene:
#                 parts.append(f"scene={dominant_scene}")
#             if scene_desc:
#                 parts.append(f"scene_description={scene_desc}")
#             if objects_text:
#                 parts.append(f"objects={objects_text}")

#             if not parts:
#                 parts.append("visual evidence available")

#             lines.append(f"- {span_text}: " + " | ".join(parts))
#             kept += 1
#             if kept >= max_items:
#                 break

#         if kept == 0:
#             return "[Visual images retrieved, but no textual visual summary available]"
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # LLM-only question-family classifier
#     # -----------------------------------------------------

#     def _default_family_info(self) -> Dict[str, Any]:
#         return {
#             "question_family": "event",
#             "graph_mode": "default",
#             "time_bias": "none",
#             "need_visual_followup": False,
#         }

#     def _llm_family_classifier(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         system_prompt = (
#             "You classify long-video QA questions into exactly one question family and output JSON only. "
#             "Valid families: source-trace, temporal-recall, action-owner, participant-membership, "
#             "plan-intention-decision, habit-preference, attribute-content-purpose, event. "
#             "Also output graph_mode, time_bias (forward/backward/none), and need_visual_followup (true/false). "
#             "Be conservative and choose the simplest valid family."
#         )
#         family_guide = (
#             "Family guidance:\n"
#             "- source-trace: prior location, transfer path, brought from, placed before, where X was before. graph_mode=backtrack_object_source, time_bias=backward, need_visual_followup=false\n"
#             "- temporal-recall: first/last/before/after/earlier/later. graph_mode=temporal_walk, time_bias=backward or forward or none, need_visual_followup=false\n"
#             "- action-owner: who did/used/moved/picked/brought something. graph_mode=actor_action_refine, time_bias=none, need_visual_followup=false\n"
#             "- participant-membership: who was present / with whom / who joined or left. graph_mode=participant_cooccurrence_refine, time_bias=none, need_visual_followup=false\n"
#             "- plan-intention-decision: plans, rationale, commitments, intended next step. graph_mode=topic_commitment_refine, time_bias=none, need_visual_followup=false\n"
#             "- habit-preference: repeated patterns, often/usually/prefer. graph_mode=habit_support_only, time_bias=none, need_visual_followup=false\n"
#             "- attribute-content-purpose: color, appearance, what was inside, visual identity, purpose requiring direct grounding. graph_mode=anchor_refine_then_visual, time_bias=none, need_visual_followup=true\n"
#             "- event: generic specific-event questions that do not strongly fit the above. graph_mode=default, time_bias=none, need_visual_followup=false"
#         )
#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         user_prompt = (
#             f"{full_query}\n\n"
#             f"{family_guide}\n\n"
#             "Return JSON with keys: question_family, graph_mode, time_bias, need_visual_followup"
#         )
#         fallback = self._default_family_info()
#         try:
#             response = self.respond_llm_model.generate([
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ])
#             data = self._parse_json_object(response)
#             if not data:
#                 return fallback
#             return {
#                 "question_family": str(data.get("question_family", fallback["question_family"])).strip().lower(),
#                 "graph_mode": str(data.get("graph_mode", fallback["graph_mode"])).strip().lower(),
#                 "time_bias": str(data.get("time_bias", fallback["time_bias"])).strip().lower(),
#                 "need_visual_followup": bool(data.get("need_visual_followup", fallback["need_visual_followup"])),
#             }
#         except Exception as e:
#             logger.warning("LLM family classifier failed, fallback to default family info: %s", e)
#             return fallback

#     def _classify_question_family(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> Dict[str, Any]:
#         return self._llm_family_classifier(query=query, choices=choices, until_time=until_time)

#     def _family_retrieval_prior(self, family_info: Dict[str, Any]) -> str:
#         family = family_info.get("question_family", "event")
#         mapping = {
#             "source-trace": "Search for earlier source-establishing events rather than merely the current location.",
#             "temporal-recall": "Search for the event that best satisfies first/last/before/after constraints.",
#             "action-owner": "Search for explicit actor-action evidence, not just nearby participation.",
#             "participant-membership": "Search for explicit participation/co-presence evidence across relevant event windows.",
#             "plan-intention-decision": "Search for explicit commitments, decisions, or rationale statements, often in speech or topic context.",
#             "habit-preference": "Search for repeated or aggregate evidence; semantic memory is often useful.",
#             "attribute-content-purpose": "Search for direct grounding of the queried property; visual follow-up may help.",
#             "event": "Search for the most predicate-aligned event evidence first.",
#         }
#         return mapping.get(family, mapping["event"])

#     def _family_hint_block(self, family_info: Dict[str, Any]) -> str:
#         return (
#             "Question family hint:\n"
#             f"- question_family: {family_info.get('question_family', 'event')}\n"
#             f"- graph_mode_if_episodic: {family_info.get('graph_mode', 'default')}\n"
#             f"- time_bias: {family_info.get('time_bias', 'none')}\n"
#             f"- need_visual_followup: {str(bool(family_info.get('need_visual_followup', False))).lower()}\n\n"
#             "Family-specific retrieval prior:\n"
#             f"- {self._family_retrieval_prior(family_info)}\n"
#             "Use this as a prior, but revise based on round history if needed."
#         )

#     # -----------------------------------------------------
#     # packet builders
#     # -----------------------------------------------------

#     def _build_event_packet(
#         self,
#         doc_id: str,
#         score: float,
#         supporting_facts: Optional[List[SemanticTripleEntry]] = None,
#     ) -> str:
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return ""

#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         raw_triplets = self.episodic_memory.get_raw_triplets_by_doc_id(doc_id, "30sec") if hasattr(self.episodic_memory, "get_raw_triplets_by_doc_id") else []
#         supporting_facts = supporting_facts or []
#         parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min") if hasattr(self.episodic_memory, "get_parent_caption") else None

#         lines: List[str] = []
#         lines.append(f"Event Anchor: {doc_id}")
#         lines.append(f"Relevance Score: {score:.4f}")
#         lines.append(entry.to_display_str(include_visual_summary=True))

#         critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#         if critical_lines:
#             lines.append("Critical Speech:")
#             for line in critical_lines[:3]:
#                 if str(line).strip():
#                     lines.append(f"- {line}")

#         if parent_3min is not None and parent_3min.doc_id != doc_id:
#             p_start, p_end = parent_3min.timestamp_int
#             lines.append(
#                 f"3min Context [{transform_timestamp(str(p_start))} - {transform_timestamp(str(p_end))}]: {parent_3min.text}"
#             )
#             if parent_3min.visual_summary:
#                 lines.append(f"3min Visual: {parent_3min.visual_summary}")

#         if visual_entry is not None:
#             if getattr(visual_entry, "keyframe_caption", ""):
#                 lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
#             visual_objects = getattr(visual_entry, "visual_objects", []) or []
#             if visual_objects:
#                 lines.append("Visual Objects: " + ", ".join(visual_objects[:8]))
#             scene_summary = getattr(visual_entry, "scene_summary", {}) or {}
#             if isinstance(scene_summary, dict) and scene_summary.get("dominant_scene"):
#                 lines.append(f"Scene: {scene_summary.get('dominant_scene')}")

#         if raw_triplets:
#             lines.append("Raw Triplets:")
#             for tri in raw_triplets[:8]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if triplets:
#             lines.append("Canonical Episodic Triplets:")
#             for tri in triplets[:6]:
#                 if isinstance(tri, list) and len(tri) == 3:
#                     lines.append(f"- ({tri[0]}, {tri[1]}, {tri[2]})")

#         if supporting_facts:
#             lines.append("Supporting Semantic Facts:")
#             for fact in supporting_facts[:3]:
#                 lines.append(f"- {fact.to_display_str()}")

#         return "\n".join(lines)

#     def _build_semantic_packet(self, entry: SemanticTripleEntry) -> str:
#         lines = [f"Semantic Packet: {entry.to_display_str()}"]
#         support_ids: List[str] = []
#         if hasattr(self.semantic_memory, "get_support_event_ids"):
#             support_ids = self.semantic_memory.get_support_event_ids(entry, limit=3)
#         if support_ids:
#             lines.append("Support Event IDs: " + ", ".join(support_ids))
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # retrieval helpers
#     # -----------------------------------------------------

#     def retrieve_from_episodic_packets(
#         self,
#         query: str,
#         family_info: Dict[str, Any],
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str]]:
#         top_k = top_k or self.episodic_top_k
#         retrieved_set = retrieved_set or set()

#         ranked = self.episodic_memory.retrieve_ranked_with_family(
#             query=query,
#             family_info=family_info,
#         )
#         if not ranked:
#             return [], "[No episodic results]", retrieved_set, []

#         selected: List[Tuple[CaptionEntryRAG, float]] = []
#         for entry, score in ranked:
#             key = f"episodic:{entry.doc_id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append((entry, score))
#             if len(selected) >= top_k:
#                 break

#         if not selected:
#             return [], "[No new episodic results]", retrieved_set, []

#         packets: List[str] = []
#         top_doc_ids: List[str] = []
#         for entry, score in selected:
#             top_doc_ids.append(entry.doc_id)
#             packet = self._build_event_packet(doc_id=entry.doc_id, score=score)
#             if packet:
#                 packets.append(packet)

#         summary = self._build_episodic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="episodic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set, top_doc_ids

#     def retrieve_from_semantic_packets(
#         self,
#         query: str,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str], List[str], Dict[str, List[SemanticTripleEntry]], List[SemanticTripleEntry]]:
#         top_k = top_k or self.semantic_top_k
#         retrieved_set = retrieved_set or set()
#         entries = self.semantic_memory.retrieve(query=query, top_k=top_k * 2, as_context=False)
#         if not entries:
#             return [], "[No semantic results]", retrieved_set, [], {}, []

#         selected: List[SemanticTripleEntry] = []
#         support_doc_ids: List[str] = []
#         support_fact_map: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)
#         for entry in entries:
#             key = f"semantic:{entry.id}"
#             if key in retrieved_set:
#                 continue
#             retrieved_set.add(key)
#             selected.append(entry)
#             if hasattr(self.semantic_memory, "get_support_event_ids"):
#                 cur_support = self.semantic_memory.get_support_event_ids(entry, limit=4)
#                 support_doc_ids.extend(cur_support)
#                 for doc_id in cur_support:
#                     support_fact_map[str(doc_id)].append(entry)
#             if len(selected) >= top_k:
#                 break

#         packets = [self._build_semantic_packet(entry) for entry in selected]
#         packets = [p for p in packets if p]
#         summary = self._build_semantic_history_summary(selected)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="semantic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=round_num,
#                 )
#             )

#         dedup_support: List[str] = []
#         seen: Set[str] = set()
#         for x in support_doc_ids:
#             if x and x not in seen:
#                 seen.add(x)
#                 dedup_support.append(x)
#         return items, summary, retrieved_set, dedup_support, dict(support_fact_map), selected

#     def retrieve_from_visual_packets(
#         self,
#         query: str,
#         anchor_doc_ids: Optional[List[str]] = None,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#         round_num: int = 1,
#     ) -> Tuple[List[RetrievedItem], str, Set[str]]:
#         top_k = top_k or self.visual_top_k
#         retrieved_set = retrieved_set or set()
#         all_images: List[Image.Image] = []
#         summary = "[No visual results]"

#         if anchor_doc_ids:
#             doc_ids = [x for x in anchor_doc_ids if x][:top_k]
#             event_images = self.visual_memory.get_event_images(
#                 doc_ids,
#                 max_images_per_event=max(self.visual_top_k, 1),
#             )
#             if event_images:
#                 kept_doc_ids: List[str] = []
#                 for doc_id in doc_ids:
#                     images = event_images.get(doc_id, [])
#                     if not images:
#                         continue
#                     key = f"visual:{doc_id}"
#                     if key in retrieved_set:
#                         continue
#                     retrieved_set.add(key)
#                     kept_doc_ids.append(doc_id)
#                     all_images.extend(images)
#                 if kept_doc_ids and all_images:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)
#         else:
#             result = self.visual_memory.retrieve(query=query, top_k=top_k, as_context=True)
#             if isinstance(result, dict) and result:
#                 kept_doc_ids: List[str] = []
#                 for key, images in result.items():
#                     if f"visual:{key}" in retrieved_set:
#                         continue
#                     retrieved_set.add(f"visual:{key}")
#                     kept_doc_ids.append(key)
#                     all_images.extend(images)
#                 if kept_doc_ids:
#                     summary = self._build_visual_history_summary_from_doc_ids(kept_doc_ids)

#         items: List[RetrievedItem] = []
#         if all_images:
#             items.append(
#                 RetrievedItem(
#                     memory_type="visual",
#                     content=all_images,
#                     query=query,
#                     round_num=round_num,
#                 )
#             )
#         return items, summary, retrieved_set

#     def _should_ground_semantic_bridge(
#         self,
#         family_info: Dict[str, Any],
#         support_doc_ids: List[str],
#         round_num: int,
#     ) -> bool:
#         if not support_doc_ids:
#             return False
#         family = str(family_info.get("question_family", "event"))
#         if family in {"plan-intention-decision", "temporal-recall", "participant-membership", "habit-preference", "source-trace", "action-owner"}:
#             return True
#         return round_num <= 2 and len(support_doc_ids) <= 8

#     def _ground_semantic_to_episodic(
#         self,
#         query: str,
#         family_info: Dict[str, Any],
#         support_doc_ids: List[str],
#         support_fact_map: Optional[Dict[str, List[SemanticTripleEntry]]] = None,
#         top_k: Optional[int] = None,
#     ) -> Tuple[List[RetrievedItem], str, List[str], Dict[str, List[SemanticTripleEntry]]]:
#         top_k = top_k or self.semantic_bridge_top_k
#         support_fact_map = support_fact_map or {}
#         ranked = self.episodic_memory.retrieve_ranked_from_doc_id_pool(
#             query=query,
#             doc_ids=support_doc_ids,
#             family_info=family_info,
#             final_top_k=top_k,
#             neighbor_radius=2,
#             max_candidates=48,
#         )
#         if not ranked:
#             return [], "[No semantic-grounded episodic support]", [], {}

#         packets: List[str] = []
#         selected: List[Tuple[CaptionEntryRAG, float]] = []
#         doc_ids: List[str] = []
#         per_doc_support: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)
#         for entry, score in ranked[:top_k]:
#             doc_ids.append(entry.doc_id)
#             selected.append((entry, score))
#             local_support = list(support_fact_map.get(entry.doc_id, []) or [])
#             if not local_support:
#                 # allow root-level / parent-level support fallback
#                 for raw_support_doc_id, facts in (support_fact_map or {}).items():
#                     if raw_support_doc_id == entry.doc_id:
#                         local_support.extend(facts)
#             if local_support:
#                 per_doc_support[entry.doc_id].extend(local_support)
#             packet = self._build_event_packet(
#                 doc_id=entry.doc_id,
#                 score=score,
#                 supporting_facts=per_doc_support.get(entry.doc_id, []),
#             )
#             if packet:
#                 packets.append(packet)

#         summary = self._build_episodic_history_summary(selected, max_items=self.semantic_bridge_history_top_k)
#         items: List[RetrievedItem] = []
#         if packets:
#             items.append(
#                 RetrievedItem(
#                     memory_type="episodic",
#                     content="\n\n".join(packets),
#                     query=query,
#                     round_num=0,
#                 )
#             )
#         return items, summary, doc_ids, dict(per_doc_support)

#     def _make_selector_candidate(
#         self,
#         idx: int,
#         doc_id: str,
#         score: float,
#         source_tags: Set[str],
#         support_facts: Optional[List[SemanticTripleEntry]] = None,
#     ) -> Dict[str, Any]:
#         entry = self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec")
#         if entry is None:
#             return {}
#         start_ts, end_ts = entry.timestamp_int
#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         raw_triplets = self.episodic_memory.get_raw_triplets_by_doc_id(doc_id, "30sec") if hasattr(self.episodic_memory, "get_raw_triplets_by_doc_id") else []
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         critical_lines = list(entry.metadata.get("critical_speech_lines", []) or [])
#         candidate = {
#             "idx": idx,
#             "doc_id": doc_id,
#             "score": round(float(score), 4),
#             "source_tags": sorted(list(source_tags)),
#             "time_span": f"{transform_timestamp(str(start_ts))} - {transform_timestamp(str(end_ts))}",
#             "text": self._short_text(entry.text, 420),
#             "visual_summary": self._short_text(entry.visual_summary or getattr(visual_entry, "keyframe_caption", ""), 260),
#             "critical_speech_lines": [self._short_text(x, 160) for x in critical_lines[:4] if str(x).strip()],
#             "raw_triplets": raw_triplets[:8],
#             "episodic_triplets": triplets[:6],
#             "supporting_semantic_facts": [fact.to_display_str() for fact in (support_facts or [])[:3]],
#         }
#         return candidate

#     def _build_selector_candidates(
#         self,
#         candidate_scores: Dict[str, float],
#         candidate_sources: Dict[str, Set[str]],
#         candidate_support_facts: Dict[str, List[SemanticTripleEntry]],
#         max_candidates: int = 14,
#     ) -> List[Dict[str, Any]]:
#         sorted_doc_ids = [doc_id for doc_id, _ in sorted(candidate_scores.items(), key=lambda x: -x[1])]
#         selector_candidates: List[Dict[str, Any]] = []
#         for doc_id in sorted_doc_ids[:max_candidates]:
#             cand = self._make_selector_candidate(
#                 idx=len(selector_candidates),
#                 doc_id=doc_id,
#                 score=candidate_scores.get(doc_id, 0.0),
#                 source_tags=candidate_sources.get(doc_id, set()),
#                 support_facts=candidate_support_facts.get(doc_id, []),
#             )
#             if cand:
#                 selector_candidates.append(cand)
#         return selector_candidates

#     def _parse_selector_response(self, response: str, selector_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
#         data = self._parse_json_object(response)
#         if not data:
#             return {}
#         valid_doc_ids = {str(c.get("doc_id", "")) for c in selector_candidates}
#         selected_doc_ids: List[str] = []
#         for x in data.get("selected_doc_ids", []) or []:
#             x = str(x)
#             if x in valid_doc_ids and x not in selected_doc_ids:
#                 selected_doc_ids.append(x)
#         if not selected_doc_ids:
#             indices = data.get("selected_indices", []) or []
#             for idx in indices:
#                 try:
#                     idx = int(idx)
#                 except Exception:
#                     continue
#                 if 0 <= idx < len(selector_candidates):
#                     doc_id = str(selector_candidates[idx].get("doc_id", ""))
#                     if doc_id and doc_id not in selected_doc_ids:
#                         selected_doc_ids.append(doc_id)
#         return {
#             "question_family": str(data.get("question_family", "")).strip().lower(),
#             "selected_doc_ids": selected_doc_ids,
#             "reason": str(data.get("reason", "")).strip(),
#         }


#     def _run_llm_selector_on_final_pool(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]],
#         until_time: Optional[int],
#         selector_candidates: List[Dict[str, Any]],
#         final_top_k: int,
#     ) -> Dict[str, Any]:
#         if not selector_candidates:
#             return {}
#         query_with_time = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         prompt = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are selecting event packets for a long-video QA system.\n"
#                     "Your job is NOT to choose events that are merely topically related. "
#                     "Your job is to choose events whose evidence matches the exact predicate asked by the question.\n\n"
#                     "You must do two things:\n"
#                     "Step 1: infer the question family from the question.\n"
#                     "Step 2: choose a small, complementary set of event packets that best supports the answer.\n\n"
#                     "Use one of these question families:\n"
#                     "1) action-owner\n"
#                     "2) source-trace\n"
#                     "3) participant-membership\n"
#                     "4) plan-intention-decision\n"
#                     "5) temporal-recall\n"
#                     "6) habit-preference\n"
#                     "7) attribute-content-purpose\n\n"
#                     "Core principle:\n"
#                     "- Prefer explicit evidence over weak implication.\n"
#                     "- Prefer predicate-aligned evidence over broad contextual relevance.\n"
#                     "- Do not over-select near-duplicate local events.\n"
#                     "- Always return valid candidate indices and/or valid doc_ids from the provided list only.\n\n"
#                     "Global anti-error rules:\n"
#                     "- Do not infer agent ownership from scene participation alone.\n"
#                     "- Do not infer intention from topic discussion alone.\n"
#                     "- Do not infer source from current location alone.\n"
#                     "- Do not infer habits from a single weak event if stronger repeated evidence exists.\n"
#                     "- Do not infer attributes from nearby actions when direct grounding exists.\n"
#                     "- When direct evidence and broad contextual evidence conflict, prefer direct evidence.\n"
#                     "- Prefer a smaller set of directly relevant events over a larger set of vaguely related events.\n"
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"{query_with_time}\n\n"
#                     f"Candidate Event Packets:\n{json.dumps(selector_candidates, ensure_ascii=False, indent=2)}\n\n"
#                     f"Select the best {final_top_k} candidates.\n\n"
#                     "Selection goals:\n"
#                     "- Choose complementary evidence, not repetitive evidence.\n"
#                     "- Retain at least one event that directly grounds the core predicate of the question.\n"
#                     "- If the question requires prior-state or source evidence, retain the event that directly establishes that prior state, even if it is earlier and less salient.\n"
#                     "- If the question requires intention or decision evidence, retain explicit commitment or decision evidence rather than topic-related discussion.\n"
#                     "- If the question requires identifying an actor, retain explicit actor evidence.\n"
#                     "- If the question requires temporal comparison, enforce the temporal constraint strictly.\n"
#                     "- If the question requires a stable habit or preference, prefer repeated or aggregate evidence over one-off evidence.\n"
#                     "- If the question requires ownership, contents, identity, purpose, or attribute, prefer direct grounding over surrounding context.\n\n"
#                     "Output requirements:\n"
#                     "- Infer the correct question_family first.\n"
#                     "- Then select the best candidates.\n"
#                     "- The reason must explain why the selected events satisfy the core predicate better than merely related events.\n\n"
#                     "Return ONLY JSON in this format:\n"
#                     '{"question_family": ".", "selected_indices": [.], "selected_doc_ids": [.], "reason": "."}'
#                 ),
#             },
#         ]
#         try:
#             response = self.respond_llm_model.generate(prompt)
#             return self._parse_selector_response(response, selector_candidates)
#         except Exception as e:
#             logger.warning("LLM selector failed: %s", e)
#             return {}

#         # -----------------------------------------------------
#         # main answer loop
#         # -----------------------------------------------------


#     def answer(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> QAResult:
#         if until_time and until_time > self.indexed_time:
#             self.index(until_time)

#         full_query = self._build_query_with_time(query=query, choices=choices, until_time=until_time)
#         family_info = self._classify_question_family(query=query, choices=choices, until_time=until_time)

#         retrieved_set: Set[str] = set()
#         retrieved_items: List[RetrievedItem] = []
#         round_history: List[Dict[str, Any]] = []
#         latest_anchor_doc_ids: List[str] = []

#         candidate_scores: Dict[str, float] = defaultdict(float)
#         candidate_sources: Dict[str, Set[str]] = defaultdict(set)
#         candidate_support_facts: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)

#         reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
#         round_num = 0
#         err_count = 0

#         while round_num < self.max_rounds and err_count < self.max_errors:
#             round_num += 1
#             logger.info("Reasoning round %d", round_num)

#             history_str = self._format_round_history(round_history)
#             user_content = (
#                 f"{full_query}\n\n"
#                 f"{self._family_hint_block(family_info)}\n\n"
#                 f"Round History:\n{history_str}\n\n"
#                 "Task:\n"
#                 'Step 1: Decide whether to "search" or "answer".\n'
#                 "Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."
#             )
#             reasoning_messages = copy.deepcopy(reasoning_prompt)
#             reasoning_messages.append({"role": "user", "content": user_content})

#             try:
#                 response = self.respond_llm_model.generate(reasoning_messages)
#                 reasoning_output = self._parse_reasoning_response(response)
#             except Exception as e:
#                 logger.error("Reasoning failed: %s", e)
#                 err_count += 1
#                 continue

#             logger.info("Decision: %s", reasoning_output.decision)
#             if reasoning_output.decision == "answer":
#                 break

#             if reasoning_output.decision != "search" or not reasoning_output.selected_memory:
#                 logger.warning("Invalid search decision payload")
#                 err_count += 1
#                 continue

#             memory_type = reasoning_output.selected_memory.memory_type
#             search_query = reasoning_output.selected_memory.search_query or query
#             logger.info("Searching %s: %s", memory_type, search_query)

#             new_items: List[RetrievedItem] = []
#             summary = "[No results]"
#             anchor_doc_ids: List[str] = []

#             try:
#                 if memory_type == "episodic":
#                     new_items, summary, retrieved_set, anchor_doc_ids = self.retrieve_from_episodic_packets(
#                         search_query,
#                         family_info=family_info,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                         for rank, doc_id in enumerate(anchor_doc_ids):
#                             candidate_scores[doc_id] += 1.0 / (rank + 1)
#                             candidate_sources[doc_id].add(f"episodic_r{round_num}")
#                 elif memory_type == "semantic":
#                     new_items, summary, retrieved_set, anchor_doc_ids, support_fact_map, semantic_entries = self.retrieve_from_semantic_packets(
#                         search_query,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                     if anchor_doc_ids:
#                         latest_anchor_doc_ids = anchor_doc_ids
#                     if self._should_ground_semantic_bridge(family_info, anchor_doc_ids, round_num):
#                         grounded_items, grounded_summary, grounded_doc_ids, per_doc_support = self._ground_semantic_to_episodic(
#                             search_query,
#                             family_info=family_info,
#                             support_doc_ids=anchor_doc_ids,
#                             support_fact_map=support_fact_map,
#                             top_k=self.semantic_bridge_top_k,
#                         )
#                         if grounded_items:
#                             new_items.extend(grounded_items)
#                             summary = summary + "\n\nSemantic-grounded episodic support:\n" + grounded_summary
#                             latest_anchor_doc_ids = grounded_doc_ids or latest_anchor_doc_ids
#                             for rank, doc_id in enumerate(grounded_doc_ids):
#                                 candidate_scores[doc_id] += 1.10 / (rank + 1)
#                                 candidate_sources[doc_id].add(f"semantic_bridge_r{round_num}")
#                                 for fact in per_doc_support.get(doc_id, [])[:4]:
#                                     if fact not in candidate_support_facts[doc_id]:
#                                         candidate_support_facts[doc_id].append(fact)
#                 elif memory_type == "visual":
#                     new_items, summary, retrieved_set = self.retrieve_from_visual_packets(
#                         search_query,
#                         anchor_doc_ids=latest_anchor_doc_ids if latest_anchor_doc_ids else None,
#                         retrieved_set=retrieved_set,
#                         round_num=round_num,
#                     )
#                 else:
#                     logger.warning("Unknown memory type: %s", memory_type)
#                     err_count += 1
#                     continue
#             except Exception as e:
#                 logger.error("Retrieval from %s failed: %s", memory_type, e)
#                 err_count += 1
#                 continue

#             retrieved_items.extend(new_items)
#             round_history.append({
#                 "round_num": round_num,
#                 "decision": "search",
#                 "memory_type": memory_type,
#                 "search_query": search_query,
#                 "retrieved_content": summary,
#             })

#         selector_candidates = self._build_selector_candidates(
#             candidate_scores=candidate_scores,
#             candidate_sources=candidate_sources,
#             candidate_support_facts=candidate_support_facts,
#             max_candidates=14,
#         )
#         selector_result: Dict[str, Any] = {}
#         selected_doc_ids: List[str] = []
#         if selector_candidates:
#             selector_result = self._run_llm_selector_on_final_pool(
#                 query=query,
#                 choices=choices,
#                 until_time=until_time,
#                 selector_candidates=selector_candidates,
#                 final_top_k=min(self.selector_top_k, max(1, len(selector_candidates))),
#             )
#             selected_doc_ids = list(selector_result.get("selected_doc_ids", []) or [])
#             if not selected_doc_ids:
#                 selected_doc_ids = [c["doc_id"] for c in selector_candidates[: min(self.selector_top_k, len(selector_candidates))]]
#             round_history.append({
#                 "round_num": round_num + 1,
#                 "decision": "selector",
#                 "memory_type": "selector",
#                 "search_query": query,
#                 "retrieved_content": json.dumps({
#                     "question_family": selector_result.get("question_family", family_info.get("question_family", "event")),
#                     "selected_doc_ids": selected_doc_ids,
#                     "reason": selector_result.get("reason", ""),
#                 }, ensure_ascii=False, indent=2),
#             })

#         logger.info("Generating answer from selected evidence packets")
#         qa_prompt = self.prompt_template_manager.render(self.qa_template_name)
#         qa_content: List[Dict[str, Any]] = [{
#             "type": "text",
#             "text": (
#                 f"{full_query}\n\n"
#                 "Important for answering:\n"
#                 "- Treat Query Time as the reference point for all relative temporal expressions.\n"
#                 "- Interpret 'before', 'after', 'earlier', 'later', 'recently', 'a few hours ago', 'first', and 'last' relative to Query Time.\n"
#                 "- Prefer evidence that satisfies the temporal constraint exactly.\n"
#                 "- Use the selected evidence packets below as the primary grounding evidence.\n\n"
#                 "Selected Evidence Packets:\n"
#             )
#         }]

#         selected_items: List[RetrievedItem] = []
#         if selected_doc_ids:
#             packet_texts: List[str] = []
#             for doc_id in selected_doc_ids:
#                 packet = self._build_event_packet(
#                     doc_id=doc_id,
#                     score=candidate_scores.get(doc_id, 0.0),
#                     supporting_facts=candidate_support_facts.get(doc_id, []),
#                 )
#                 if packet:
#                     packet_texts.append(packet)
#             if packet_texts:
#                 selected_items.append(
#                     RetrievedItem(
#                         memory_type="episodic",
#                         content="\n\n".join(packet_texts),
#                         query=query,
#                         round_num=round_num + 1,
#                     )
#                 )
#         if not selected_items:
#             selected_items = retrieved_items

#         qa_content.extend(self._render_retrieved_items_for_qa(selected_items))
#         if selector_result.get("reason"):
#             qa_content.append({
#                 "type": "text",
#                 "text": "\nSelector Reasoning Chain:\n" + selector_result.get("reason", ""),
#             })
#         qa_content.append({
#             "type": "text",
#             "text": "\n\nRetrieval history summary:\n" + self._format_round_history(round_history),
#         })
#         qa_content.append({
#             "type": "text",
#             "text": (
#                 "\nThe selected event anchors were chosen because they form the strongest evidence chain for this question.\n"
#                 "Use these selected events as the primary basis for answering.\n"
#                 "Do not override a clearly supported conclusion from the selected evidence with a weaker alternative.\n\n"
#             )
#         })
#         if choices:
#             qa_content.append({
#                 "type": "text",
#                 "text": "\nPlease provide only the final answer from the choices given (e.g. A, B, C, or D).",
#             })
#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({"role": "user", "content": qa_content})

#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error("Answer generation failed: %s", e)
#             answer = "Unable to generate answer"

#         return QAResult(
#             question=query,
#             answer=answer,
#             retrieved_items=selected_items,
#             round_history=round_history,
#             num_rounds=round_num,
#         )
