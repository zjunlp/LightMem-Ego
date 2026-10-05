# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import json
import logging
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch

from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class CaptionEntry_videomme:
    """Represents a single caption entry with its metadata."""
    id: str
    text: str
    visual_summary: str
    start_time: str
    end_time: str
    date: str
    granularity: str

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        """Convert start and end times to integer format (day + time.zfill(8))."""
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts

    def to_display_str(self) -> str:
        """Format caption for display with time range."""
        start_ts, end_ts = self.timestamp_int
        return f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\ntext: {self.text}\nvisual_summary: {self.visual_summary}"


def _transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


def _load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class MemoryCell_videomme:
    """
    Episodic Memory module that implements multiscale retrieval using dense embeddings.
    
    This class manages episodic captions at multiple temporal granularities
    and provides retrieval functionality using dense vector similarity.
    """

    GRANULARITY_ORDER = ["10sec", "30sec", "3min", "10min"]

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        granularities: Optional[List[str]] = None,
    ):
        """
        Initialize MemoryCell.
        
        Args:
            embedding_model: Embedding model for dense retrieval
            granularities: List of granularity levels to use
        """
        self.embedding_model = embedding_model
        self.granularities = granularities or self.GRANULARITY_ORDER

        # Storage for captions
        self.captions: Dict[str, List[CaptionEntry_videomme]] = {g: [] for g in self.granularities}
        self.caption_id_to_entry: Dict[str, CaptionEntry_videomme] = {}

        # Mapping from caption text to CaptionEntry (kept for compatibility)
        self.text_to_entry: Dict[str, CaptionEntry_videomme] = {}

        # Dense index state
        self.doc_embeddings: Dict[str, Optional[np.ndarray]] = {g: None for g in self.granularities}
        self.doc_end_times: Dict[str, Optional[np.ndarray]] = {g: None for g in self.granularities}
        self.doc_retrieval_texts: Dict[str, List[str]] = {g: [] for g in self.granularities}
        self.dense_index_built: bool = False
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

        # Indexed entries
        self.indexed_entries: Dict[str, List[CaptionEntry_videomme]] = {g: [] for g in self.granularities}
        self.indexed_time: int = 0

        # Dense encoding batch size
        self.dense_encode_batch_size = 8
        self.dense_encode_min_batch_size = 1

    # -----------------------------------------------------
    # Loading captions
    # -----------------------------------------------------

    def load_captions_from_files(
        self,
        caption_files: Dict[str, str],
    ) -> None:
        """
        Load captions from JSON files for each granularity level.
        """
        for granularity, file_path in caption_files.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
                continue
            try:
                data = _load_json(file_path)
                self._process_caption_data(data, granularity)
                logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
            except Exception as e:
                logger.error(f"Failed to load captions from {file_path}: {e}")
        self.dense_index_built = False

    def load_captions_from_data(
        self,
        caption_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """
        Load captions from in-memory data for each granularity level.
        """
        for granularity, data in caption_data.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
                continue
            self._process_caption_data(data, granularity)
            logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
        self.dense_index_built = False

    def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
        """Process raw caption data and create CaptionEntry objects."""
        for idx, entry in enumerate(data):
            caption_id = f"{granularity}_{idx}"
            caption_entry = CaptionEntry_videomme(
                id=caption_id,
                text=entry.get("text", ""),
                visual_summary=entry.get("visual_summary", ""),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=str(entry.get("date", "")),
                granularity=granularity,
            )
            self.captions[granularity].append(caption_entry)
            self.caption_id_to_entry[caption_id] = caption_entry
            self.text_to_entry[caption_entry.text] = caption_entry

    # -----------------------------------------------------
    # Dense indexing
    # -----------------------------------------------------

    def _normalize_embedding_matrix(self, embs: np.ndarray) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None, :]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return embs / norms

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into dense embeddings."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        batch_size = max(self.dense_encode_min_batch_size, self.dense_encode_batch_size)
        total = len(texts)
        i = 0
        all_embs: List[np.ndarray] = []

        while i < total:
            cur_bs = min(batch_size, total - i)
            batch_texts = texts[i:i + cur_bs]

            try:
                if hasattr(self.embedding_model, "encode_text"):
                    embs = self.embedding_model.encode_text(batch_texts, batch_size=cur_bs)
                elif hasattr(self.embedding_model, "encode"):
                    embs = self.embedding_model.encode(batch_texts)
                else:
                    raise AttributeError("EmbeddingModel must expose encode_text() or encode().")

                embs = np.asarray(embs, dtype=np.float32)
                if embs.ndim == 1:
                    embs = embs[None, :]
                all_embs.append(embs)
                i += cur_bs

            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_size <= self.dense_encode_min_batch_size:
                    raise
                batch_size = max(self.dense_encode_min_batch_size, batch_size // 2)
                logger.warning("Reduced batch size to %d due to OOM", batch_size)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if batch_size <= self.dense_encode_min_batch_size:
                        raise
                    batch_size = max(self.dense_encode_min_batch_size, batch_size // 2)
                    logger.warning("Reduced batch size to %d due to OOM", batch_size)
                else:
                    raise

        embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        return self._normalize_embedding_matrix(embs)

    def _entry_embedding_text(self, entry: CaptionEntry_videomme) -> str:
        """Build the text used for embedding (text + visual_summary)."""
        parts = [entry.text]
        if entry.visual_summary:
            parts.append(f"Visual: {entry.visual_summary}")
        return "\n".join(parts)

    def _build_dense_index(self) -> None:
        """Build dense embeddings for all loaded captions."""
        if self.dense_index_built:
            return

        for granularity in self.granularities:
            entries = self.captions.get(granularity, [])
            if not entries:
                self.doc_embeddings[granularity] = None
                self.doc_end_times[granularity] = np.asarray([], dtype=np.int64)
                self.doc_retrieval_texts[granularity] = []
                continue

            texts = [self._entry_embedding_text(e) for e in entries]
            logger.info("Building dense episodic index for %s: %d docs", granularity, len(entries))
            embeddings = self._encode_texts(texts)
            self.doc_embeddings[granularity] = embeddings
            self.doc_end_times[granularity] = np.asarray([e.timestamp_int[1] for e in entries], dtype=np.int64)
            self.doc_retrieval_texts[granularity] = texts

        self.dense_index_built = True

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get cached or newly encoded query embedding."""
        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]
        emb = self._encode_texts([query])[0]
        self._query_embedding_cache[query] = emb
        return emb

    def _lookup_entry_from_index(self, granularity: str, idx: int) -> Optional[CaptionEntry_videomme]:
        entries = self.captions.get(granularity, [])
        if idx < 0 or idx >= len(entries):
            return None
        return entries[idx]

    def _get_prefix_size(self, granularity: str) -> int:
        """Number of entries in this granularity with end_time <= indexed_time."""
        end_times = self.doc_end_times.get(granularity)
        if end_times is None:
            return 0
        return int(np.searchsorted(end_times, self.indexed_time, side="right"))

    # -----------------------------------------------------
    # Indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        """
        Index all captions up to until_time.
        Dense index is built once on first call.
        """
        if not self.dense_index_built:
            self._build_dense_index()
        self.indexed_time = until_time
        # Update indexed_entries
        for granularity in self.granularities:
            prefix_size = self._get_prefix_size(granularity)
            self.indexed_entries[granularity] = self.captions[granularity][:prefix_size]
        logger.info("Dense episodic index ready up to %s", _transform_timestamp(str(until_time)))

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve_captions_as_str(self, entries: List[CaptionEntry_videomme]) -> str:
        """Format a list of caption entries as context string."""
        return "\n\n".join(entry.to_display_str() for entry in entries)

    def retrieve(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = {
            "10sec": 10,
            "30sec": 5,
            "3min": 5,
            "10min": 3
        },
        final_top_k: int = 3,
        as_context: bool = True
    ) -> Union[List[CaptionEntry_videomme], str]:
        """
        Retrieve relevant captions using dense retrieval.
        
        Args:
            query: The search query
            top_k_per_granularity: Number of candidates to retrieve per granularity level.
            final_top_k: Final number of results to return.
            as_context: Whether to return results as context strings instead of CaptionEntry objects.
            
        Returns:
            List of CaptionEntry objects or formatted context string.
        """
        if self.indexed_time == 0:
            logger.warning("No captions indexed. Call index(until_time) before retrieve().")
            return [] if not as_context else ""

        if not self.dense_index_built:
            self._build_dense_index()

        query_emb = self._get_query_embedding(query)
        all_candidates: List[Tuple[CaptionEntry_videomme, float]] = []

        for granularity in self.granularities:
            emb = self.doc_embeddings.get(granularity)
            if emb is None or emb.shape[0] == 0:
                continue

            prefix_size = self._get_prefix_size(granularity)
            if prefix_size <= 0:
                continue

            active_emb = emb[:prefix_size]
            scores = np.dot(active_emb, query_emb)   # cosine similarity (embeddings are normalized)
            if scores.size == 0:
                continue

            # Determine top_k for this granularity
            if isinstance(top_k_per_granularity, dict):
                top_k = top_k_per_granularity.get(granularity, 5)
            else:
                top_k = top_k_per_granularity

            top_k = min(prefix_size, max(1, top_k))
            if top_k >= scores.size:
                indices = np.argsort(-scores)[:top_k]
            else:
                partial = np.argpartition(-scores, top_k - 1)[:top_k]
                indices = partial[np.argsort(-scores[partial])]

            for idx in indices:
                entry = self._lookup_entry_from_index(granularity, idx)
                if entry is not None:
                    all_candidates.append((entry, float(scores[idx])))

        # Deduplicate by entry id, keep highest score
        best_by_id: Dict[str, Tuple[CaptionEntry_videomme, float]] = {}
        for entry, score in all_candidates:
            if entry.id not in best_by_id or score > best_by_id[entry.id][1]:
                best_by_id[entry.id] = (entry, score)

        sorted_candidates = sorted(best_by_id.values(), key=lambda x: -x[1])
        result_entries = [entry for entry, _ in sorted_candidates[:final_top_k]]

        if as_context:
            return self.retrieve_captions_as_str(result_entries)
        return result_entries

    # -----------------------------------------------------
    # Utility methods
    # -----------------------------------------------------

    def reset_index(self) -> None:
        """Reset the indexed state, clearing dense index and cache."""
        self._query_embedding_cache.clear()
        for g in self.granularities:
            self.doc_embeddings[g] = None
            self.doc_end_times[g] = None
            self.doc_retrieval_texts[g] = []
            self.indexed_entries[g] = []
        self.dense_index_built = False
        self.indexed_time = 0
        logger.info("Dense episodic index reset")

    def get_indexed_time(self) -> str:
        """Get the current indexed time boundary."""
        if self.indexed_time <= 0:
            return "DAY0 00:00:00"
        return _transform_timestamp(str(self.indexed_time))

    def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry_videomme]:
        """Get a caption entry by its ID."""
        return self.caption_id_to_entry.get(caption_id)