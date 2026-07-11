# """
# Episodic Memory module for WorldMM.
# """

# import json
# import logging
# from typing import Dict, List, Any, Optional, Tuple, Union
# from dataclasses import dataclass

# from ...llm import LLMModel, PromptTemplateManager
# from ...embedding import EmbeddingModel

# from hipporag import HippoRAG

# logger = logging.getLogger(__name__)


# @dataclass
# class CaptionEntry:
#     """Represents a single caption entry with its metadata."""
#     id: str
#     text: str
#     start_time: str
#     end_time: str
#     date: str
#     granularity: str
#     video_path: Optional[str] = None
    
#     @property
#     def timestamp_int(self) -> Tuple[int, int]:
#         """Convert start and end times to integer format (day + time.zfill(8))."""
#         day = self.date.replace('DAY', '').replace('Day', '')
#         start_ts = int(day + self.start_time.zfill(8))
#         end_ts = int(day + self.end_time.zfill(8))
#         return start_ts, end_ts
    
#     def to_display_str(self) -> str:
#         """Format caption for display with time range."""
#         start_ts, end_ts = self.timestamp_int
#         return f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"


# def _transform_timestamp(ts_str: str) -> str:
#     """Transform timestamp string to human-readable format."""
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# def _load_json(file_path: str) -> Any:
#     """Load JSON file."""
#     with open(file_path, 'r') as f:
#         return json.load(f)


# class EpisodicMemory:
#     """
#     Episodic Memory module that implements multiscale retrieval and filtering.
    
#     This class manages episodic captions at multiple temporal granularities
#     (30sec, 3min, 10min, 1h) and provides retrieval functionality using
#     HippoRAG for indexing/retrieval with LLM-based multiscale filtering.
    
#     The retrieval process:
#     1. Index captions up to a given timestamp using HippoRAG
#     2. Retrieve top-k candidates from each granularity level using HippoRAG
#     3. Use LLM with multiscale_filter template to filter and rank the most relevant captions
#     4. Return the filtered captions in ranked order
    
#     Attributes:
#         granularities: List of granularity levels to use
#         captions: Dictionary mapping granularity -> list of CaptionEntry
#         hipporag: Dictionary mapping granularity -> HippoRAG instance
#         llm_model: Language model for filtering
#         prompt_template_manager: Manager for prompt templates
#     """
    
#     GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]
    
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         llm_model: LLMModel,
#         prompt_template_manager: PromptTemplateManager,
#         granularities: Optional[List[str]] = None,
#     ):
#         """
#         Initialize EpisodicMemory.
        
#         Args:
#             embedding_model: Embedding model for HippoRAG
#             llm_model: LLM model for filtering
#             prompt_template_manager: Prompt template manager
#             granularities: List of granularity levels to use (default: all)
#         """
#         self.embedding_model = embedding_model
#         self.llm_model = llm_model
#         self.prompt_template_manager = prompt_template_manager
#         self.granularities = granularities or self.GRANULARITY_ORDER
        
#         # Storage for captions
#         self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
        
#         # Mapping from caption text to CaptionEntry for reverse lookup after HippoRAG retrieval
#         self.text_to_entry: Dict[str, CaptionEntry] = {}
        
#         # HippoRAG instance for each granularity
#         self.hipporag: Dict[str, HippoRAG] = {}
        
#         # Track indexed entries (entries that have been indexed up to indexed_time)
#         self.indexed_entries: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.indexed_time: int = 0  # 0 means nothing indexed yet
    
#     def _get_or_create_hipporag(self, granularity: str) -> HippoRAG:
#         """Get or create HippoRAG instance for a granularity level."""
#         if granularity not in self.hipporag:
#             self.hipporag[granularity] = HippoRAG(
#                 save_dir=f".cache/episodic_memory/{granularity}",
#                 llm_model=self.llm_model,
#                 embedding_model=self.embedding_model)
#         return self.hipporag[granularity]
    
#     def load_captions_from_files(
#         self,
#         caption_files: Dict[str, str],
#     ) -> None:
#         """
#         Load captions from JSON files for each granularity level.
        
#         Args:
#             caption_files: Dict mapping granularity -> JSON file path
#         """
#         for granularity, file_path in caption_files.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
#                 continue
            
#             try:
#                 data = _load_json(file_path)
#                 self._process_caption_data(data, granularity)
#                 logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
#             except Exception as e:
#                 logger.error(f"Failed to load captions from {file_path}: {e}")
    
#     def load_captions_from_data(
#         self,
#         caption_data: Dict[str, List[Dict[str, Any]]],
#     ) -> None:
#         """
#         Load captions from in-memory data for each granularity level.
        
#         Args:
#             caption_data: Dict mapping granularity -> list of caption dicts
#         """
#         for granularity, data in caption_data.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
#                 continue
            
#             self._process_caption_data(data, granularity)
#             logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
    
#     def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
#         """Process raw caption data and create CaptionEntry objects."""
#         for idx, entry in enumerate(data):
#             caption_id = f"{granularity}_{idx}"
#             caption_entry = CaptionEntry(
#                 id=caption_id,
#                 text=entry.get("text", ""),
#                 start_time=str(entry.get("start_time", "")),
#                 end_time=str(entry.get("end_time", "")),
#                 date=str(entry.get("date", "")),
#                 granularity=granularity,
#                 video_path=entry.get("video_path"),
#             )
#             self.captions[granularity].append(caption_entry)
#             self.caption_id_to_entry[caption_id] = caption_entry
#             self.text_to_entry[caption_entry.text] = caption_entry
    
#     def index(self, until_time: int) -> None:
#         """
#         Index captions up to the specified timestamp using HippoRAG.
        
#         This method provides all captions with end_time <= until_time to HippoRAG
#         for indexing. Embeddings are computed inside HippoRAG.
        
#         If captions have already been indexed up to a later time, this is a no-op.
#         If called with a later time than previously indexed, it will re-index
#         with the expanded set of captions.
        
#         Args:
#             until_time: Timestamp in integer format (day + time.zfill(8)) - index all 
#                        captions with end_time <= this value
#         """
#         # If already indexed beyond this time, no need to recompute
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
#             return
        
#         for granularity in self.granularities:
#             if not self.captions[granularity]:
#                 logger.warning(f"No captions loaded for granularity {granularity}")
#                 continue
            
#             # Get entries that should be indexed (end_time <= until_time)
#             entries_to_index = [
#                 entry for entry in self.captions[granularity]
#                 if entry.timestamp_int[1] <= until_time
#             ]
            
#             if not entries_to_index:
#                 logger.debug(f"No entries to index for granularity {granularity} up to {until_time}")
#                 continue
            
#             # Get caption texts for HippoRAG
#             caption_texts = [entry.text for entry in entries_to_index]
            
#             # Get or create HippoRAG instance and update index
#             hipporag = self._get_or_create_hipporag(granularity)
#             hipporag.update(docs=caption_texts)
            
#             # Update indexed entries
#             self.indexed_entries[granularity] = entries_to_index
            
#             logger.info(f"Indexed {len(entries_to_index)} captions for granularity {granularity}")
        
#         self.indexed_time = until_time

#     def retrieve_captions_as_str(self, entries: List[CaptionEntry]) -> str:
#         """
#         Format a list of caption entries as context string.
        
#         Args:
#             entries: List of CaptionEntry objects
            
#         Returns:
#             Formatted context string
#         """
#         return "\n\n".join(entry.to_display_str() for entry in entries)
    
#     def retrieve(
#         self,
#         query: str,
#         top_k_per_granularity: Union[int, Dict[str, int]] = {
#             "30sec": 10,
#             "3min": 5,
#             "10min": 5,
#             "1h": 3
#         },
#         final_top_k: int = 3,
#         as_context: bool = True
#     ) -> Union[List[CaptionEntry], str]:
#         """
#         Retrieve relevant captions using HippoRAG and multiscale filtering.
        
#         This method retrieves from the indexed captions using HippoRAG.
#         Make sure to call index(until_time) before calling retrieve().
        
#         The retrieval process:
#         1. Retrieves top-k candidates from each granularity level using HippoRAG
#         2. Uses LLM with multiscale_filter template to filter and rank results
        
#         Args:
#             query: The search query
#             top_k_per_granularity: Number of candidates to retrieve per granularity level.
#                 Can be an int (same for all granularities) or a dict mapping granularity -> top_k.
#             final_top_k: Final number of results to return after filtering
#             as_context: Whether to return results as context strings instead of CaptionEntry objects
            
#         Returns:
#             List of CaptionEntry objects in ranked order
#         """
#         if self.indexed_time == 0:
#             logger.warning("No captions indexed. Call index(until_time) before retrieve().")
#             return []
        
#         # Retrieve from each granularity level using HippoRAG
#         all_candidates: List[Tuple[CaptionEntry, float]] = []
        
#         for granularity in self.granularities:
#             if granularity not in self.hipporag:
#                 continue
            
#             # Get top_k for this granularity
#             if isinstance(top_k_per_granularity, dict):
#                 granularity_top_k = top_k_per_granularity.get(granularity, 5)  # default to 5 if not specified
#             else:
#                 granularity_top_k = top_k_per_granularity
            
#             hipporag = self.hipporag[granularity]
            
#             retrieval_result = hipporag.retrieve(
#                 queries=[query], 
#                 num_to_retrieve=granularity_top_k
#             )
            
#             if not retrieval_result or not retrieval_result[0].docs:
#                 continue
            
#             # Convert retrieved docs back to CaptionEntry
#             retrieved_docs = retrieval_result[0].docs
#             retrieved_scores = retrieval_result[0].doc_scores if hasattr(retrieval_result[0], 'doc_scores') else [1.0] * len(retrieved_docs)
            
#             count = 0
#             for doc_text, score in zip(retrieved_docs, retrieved_scores):
#                 if count >= granularity_top_k:
#                     break
                
#                 # Look up the CaptionEntry from text
#                 entry = self.text_to_entry.get(doc_text)
#                 if entry is None:
#                     logger.warning(f"Could not find CaptionEntry for retrieved text: {doc_text[:50]}...")
#                     continue
                
#                 all_candidates.append((entry, score))
#                 count += 1
        
#         if not all_candidates:
#             logger.warning("No candidates retrieved from any granularity level")
#             return [] if not as_context else ""
        
#         # Use LLM to filter and rank candidates
#         filtered_entries = self._filter_with_llm(
#             query=query,
#             candidates=all_candidates,
#             final_top_k=final_top_k,
#         )
        
#         if as_context:
#             return self.retrieve_captions_as_str(filtered_entries)
        
#         return filtered_entries
    
#     def _filter_with_llm(
#         self,
#         query: str,
#         candidates: List[Tuple[CaptionEntry, float]],
#         final_top_k: int,
#     ) -> List[CaptionEntry]:
#         """
#         Use LLM to filter and rank candidates using multiscale_filter template.
        
#         Args:
#             query: Original search query
#             candidates: List of (CaptionEntry, score) tuples from all granularities
#             final_top_k: Number of results to return
            
#         Returns:
#             Filtered and ranked list of CaptionEntry objects
#         """
#         if len(candidates) <= final_top_k:
#             # No need to filter if we have fewer candidates than requested
#             return [entry for entry, _ in candidates]
        
#         # Format candidates for the LLM
#         caption_list = []
#         id_to_entry = {}
#         for entry, score in candidates:
#             start_ts, end_ts = entry.timestamp_int
#             caption_info = {
#                 "id": entry.id,
#                 "granularity": entry.granularity,
#                 "start_time": _transform_timestamp(str(start_ts)),
#                 "end_time": _transform_timestamp(str(end_ts)),
#                 "text": entry.text,
#             }
#             caption_list.append(caption_info)
#             id_to_entry[entry.id] = entry
        
#         # Build prompt using template
#         try:
#             prompt = self.prompt_template_manager.render("multiscale_filter")
#         except Exception as e:
#             logger.error(f"Failed to render multiscale_filter template: {e}")
#             # Fallback: return top candidates by score
#             return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]
        
#         # Add the query and candidates to the prompt
#         filter_message = {
#             "role": "user",
#             "content": f"""Question: {query}

# Retrieved Captions:
# {json.dumps(caption_list, indent=2)}

# Select the top {final_top_k} most relevant caption IDs to answer the question.
# Return ONLY a JSON array of caption IDs in order of relevance (most relevant first)."""
#         }
#         prompt.append(filter_message)
        
#         try:
#             response = self.llm_model.generate(prompt)
            
#             # Parse response to get selected IDs
#             selected_ids = self._parse_filter_response(response, set(id_to_entry.keys()))
            
#             # Return entries in the order specified by LLM
#             result = []
#             for cap_id in selected_ids[:final_top_k]:
#                 if cap_id in id_to_entry:
#                     result.append(id_to_entry[cap_id])
            
#             # If LLM returned fewer than requested, fill with top-scoring candidates
#             if len(result) < final_top_k:
#                 existing_ids = {e.id for e in result}
#                 for entry, _ in sorted(candidates, key=lambda x: -x[1]):
#                     if entry.id not in existing_ids:
#                         result.append(entry)
#                         if len(result) >= final_top_k:
#                             break
            
#             return result
            
#         except Exception as e:
#             logger.error(f"LLM filtering failed: {e}")
#             # Fallback: return top candidates by score
#             return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]
    
#     def _parse_filter_response(self, response: str, valid_ids: set) -> List[str]:
#         """
#         Parse LLM response to extract selected caption IDs.
        
#         Args:
#             response: LLM response string
#             valid_ids: Set of valid caption IDs
            
#         Returns:
#             List of caption IDs
#         """
#         import re
        
#         # Try to extract JSON array from response
#         try:
#             match = re.search(r'\[.*?\]', response, re.DOTALL)
#             if match:
#                 ids = json.loads(match.group())
#                 if isinstance(ids, list):
#                     return [str(id) for id in ids if str(id) in valid_ids]
#         except json.JSONDecodeError:
#             pass
        
#         # Fallback: look for ID patterns in response
#         found_ids = []
#         for valid_id in valid_ids:
#             if valid_id in response:
#                 found_ids.append(valid_id)
        
#         return found_ids
    
#     def reset_index(self) -> None:
#         """Reset the indexed state, clearing HippoRAG instances and indexed entries."""
#         self.hipporag.clear()
#         for g in self.granularities:
#             self.indexed_entries[g] = []
#         self.indexed_time = 0
#         logger.info("Index reset - all HippoRAG instances and indexed entries cleared")
    
#     def get_indexed_time(self) -> str:
#         """Get the current indexed time boundary."""
#         return _transform_timestamp(str(self.indexed_time))
    
#     def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
#         """Get a caption entry by its ID."""
#         return self.caption_id_to_entry.get(caption_id)


# """
# Episodic Memory module for WorldMM.
# Hybrid version:
# - HippoRAG for coarse caption retrieval
# - sidecar episodic graph / triplets for graph-aware rerank
# """

# import json
# import logging
# import os
# import re
# from typing import Dict, List, Any, Optional, Tuple, Union
# from dataclasses import dataclass, field

# from ...llm import LLMModel, PromptTemplateManager
# from ...embedding import EmbeddingModel

# from hipporag import HippoRAG

# logger = logging.getLogger(__name__)


# STOPWORDS = {
#     "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
#     "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
#     "what", "which", "who", "whom", "when", "where", "why", "how",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
#     "this", "that", "these", "those", "it", "its"
# }


# @dataclass
# class CaptionEntry:
#     id: str
#     doc_id: str
#     text: str
#     start_time: str
#     end_time: str
#     date: str
#     granularity: str
#     video_path: Optional[str] = None
#     visual_summary: str = ""
#     metadata: Dict[str, Any] = field(default_factory=dict)

#     @property
#     def timestamp_int(self) -> Tuple[int, int]:
#         day = self.date.replace('DAY', '').replace('Day', '')
#         start_ts = int(day + self.start_time.zfill(8))
#         end_ts = int(day + self.end_time.zfill(8))
#         return start_ts, end_ts

#     def to_display_str(self, include_visual_summary: bool = True) -> str:
#         start_ts, end_ts = self.timestamp_int
#         base = f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"
#         if include_visual_summary and self.visual_summary:
#             base += f"\nVisual: {self.visual_summary}"
#         return base


# @dataclass
# class GraphEventSidecar:
#     event_id: str
#     doc_id: str
#     granularity: str
#     entity_labels: List[str] = field(default_factory=list)
#     relation_types: List[str] = field(default_factory=list)
#     triplet_strings: List[str] = field(default_factory=list)
#     prev_doc_id: Optional[str] = None
#     next_doc_id: Optional[str] = None
#     graph_tokens: set = field(default_factory=set)


# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# def _load_json(file_path: str) -> Any:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def _tokenize(text: str) -> set:
#     toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
#     return {t for t in toks if len(t) > 1 and t not in STOPWORDS}


# class EpisodicMemory:
#     GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]

#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         llm_model: LLMModel,
#         prompt_template_manager: PromptTemplateManager,
#         granularities: Optional[List[str]] = None,
#     ):
#         self.embedding_model = embedding_model
#         self.llm_model = llm_model
#         self.prompt_template_manager = prompt_template_manager
#         self.granularities = granularities or self.GRANULARITY_ORDER

#         self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
#         self.doc_id_to_entry: Dict[str, Dict[str, CaptionEntry]] = {g: {} for g in self.granularities}
#         self.text_to_entries: Dict[str, Dict[str, List[CaptionEntry]]] = {g: {} for g in self.granularities}

#         self.hipporag: Dict[str, HippoRAG] = {}
#         self.indexed_entries: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.indexed_time: int = 0

#         # sidecar graph/triplets
#         self.triplets_by_doc: Dict[str, Dict[str, List[List[str]]]] = {g: {} for g in self.granularities}
#         self.graph_sidecar: Dict[str, Dict[str, GraphEventSidecar]] = {g: {} for g in self.granularities}

#         # weights for graph-aware rerank
#         self.graph_score_weight = 0.25
#         self.metadata_score_weight = 0.15

#     def _get_or_create_hipporag(self, granularity: str) -> HippoRAG:
#         if granularity not in self.hipporag:
#             self.hipporag[granularity] = HippoRAG(
#                 save_dir=f".cache/episodic_memory/{granularity}",
#                 llm_model=self.llm_model,
#                 embedding_model=self.embedding_model,
#             )
#         return self.hipporag[granularity]

#     # -----------------------------------------------------
#     # Load captions
#     # -----------------------------------------------------

#     def load_captions_from_files(self, caption_files: Dict[str, str]) -> None:
#         for granularity, file_path in caption_files.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not configured")
#                 continue
#             try:
#                 data = _load_json(file_path)
#                 self._process_caption_data(data, granularity)
#                 logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")
#             except Exception as e:
#                 logger.error(f"Failed to load captions from {file_path}: {e}")

#     def load_captions_from_data(self, caption_data: Dict[str, List[Dict[str, Any]]]) -> None:
#         for granularity, data in caption_data.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not configured")
#                 continue
#             self._process_caption_data(data, granularity)
#             logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")

#     def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
#         self.captions[granularity] = []
#         self.doc_id_to_entry[granularity] = {}
#         self.text_to_entries[granularity] = {}

#         for idx, entry in enumerate(data):
#             doc_id = str(entry.get("doc_id", f"{granularity}_{idx}"))
#             caption_id = doc_id

#             metadata = {
#                 "action_threads": entry.get("action_threads", entry.get("main_actions", [])),
#                 "object_threads": entry.get("object_threads", entry.get("salient_objects", [])),
#                 "topic_threads": entry.get("topic_threads", entry.get("conversation_focus", [])),
#                 "speaker_stats": entry.get("speaker_stats", entry.get("speakers", [])),
#                 "scene_summary": entry.get("scene_summary", {}),
#                 "visual_object_threads": entry.get("visual_object_threads", entry.get("visual_objects", [])),
#             }

#             caption_entry = CaptionEntry(
#                 id=caption_id,
#                 doc_id=doc_id,
#                 text=entry.get("text", entry.get("fine_caption", "")),
#                 start_time=str(entry.get("start_time", "")),
#                 end_time=str(entry.get("end_time", "")),
#                 date=str(entry.get("date", "")),
#                 granularity=granularity,
#                 video_path=entry.get("video_path"),
#                 visual_summary=entry.get("visual_summary", ""),
#                 metadata=metadata,
#             )

#             self.captions[granularity].append(caption_entry)
#             self.caption_id_to_entry[caption_id] = caption_entry
#             self.doc_id_to_entry[granularity][doc_id] = caption_entry

#             self.text_to_entries[granularity].setdefault(caption_entry.text, []).append(caption_entry)

#     # -----------------------------------------------------
#     # Load sidecar triplets / graph
#     # -----------------------------------------------------

#     def load_sidecar_from_files(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#     ) -> None:
#         if triplet_files:
#             for granularity, file_path in triplet_files.items():
#                 if granularity not in self.granularities:
#                     continue
#                 data = _load_json(file_path)
#                 self._process_triplet_data(data, granularity)

#         if graph_files:
#             for granularity, file_path in graph_files.items():
#                 if granularity not in self.granularities:
#                     continue
#                 data = _load_json(file_path)
#                 self._process_graph_data(data, granularity)

#     def load_sidecar_from_data(
#         self,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_data:
#             for granularity, data in triplet_data.items():
#                 if granularity not in self.granularities:
#                     continue
#                 self._process_triplet_data(data, granularity)

#         if graph_data:
#             for granularity, data in graph_data.items():
#                 if granularity not in self.granularities:
#                     continue
#                 self._process_graph_data(data, granularity)

#     def _process_triplet_data(self, data: Dict[str, Any], granularity: str) -> None:
#         self.triplets_by_doc[granularity] = dict(data.get("triplet_map", {}))
#         logger.info(f"Loaded sidecar triplets for {granularity}: {len(self.triplets_by_doc[granularity])} docs")

#     def _process_graph_data(self, data: Dict[str, Any], granularity: str) -> None:
#         nodes = {node["id"]: node for node in data.get("nodes", [])}
#         edges = data.get("edges", [])
#         doc_id_to_event_id = data.get("doc_id_to_event_id", {})

#         # initialize sidecar from event nodes
#         sidecar = {}
#         for doc_id, event_id in doc_id_to_event_id.items():
#             sidecar[doc_id] = GraphEventSidecar(
#                 event_id=event_id,
#                 doc_id=doc_id,
#                 granularity=granularity,
#             )

#         event_id_to_doc_id = {event_id: doc_id for doc_id, event_id in doc_id_to_event_id.items()}

#         # parse edges
#         for edge in edges:
#             source = edge["source"]
#             target = edge["target"]
#             edge_type = edge["type"]
#             event_id = edge.get("event_id")

#             # temporal
#             if edge_type == "before" and source in event_id_to_doc_id and target in event_id_to_doc_id:
#                 src_doc = event_id_to_doc_id[source]
#                 tgt_doc = event_id_to_doc_id[target]
#                 if src_doc in sidecar:
#                     sidecar[src_doc].next_doc_id = tgt_doc
#                 if tgt_doc in sidecar:
#                     sidecar[tgt_doc].prev_doc_id = src_doc
#                 continue

#             if event_id not in event_id_to_doc_id:
#                 continue

#             doc_id = event_id_to_doc_id[event_id]
#             info = sidecar[doc_id]

#             src_node = nodes.get(source, {})
#             tgt_node = nodes.get(target, {})

#             src_type = src_node.get("type")
#             tgt_type = tgt_node.get("type")
#             src_label = src_node.get("label", "")
#             tgt_label = tgt_node.get("label", "")

#             # attachment edges
#             if source == event_id and tgt_type != "Event":
#                 if tgt_label:
#                     info.entity_labels.append(tgt_label)
#                 if edge_type:
#                     info.relation_types.append(edge_type)
#                 continue

#             # triplet edges
#             if src_type != "Event" and tgt_type != "Event":
#                 if src_label:
#                     info.entity_labels.append(src_label)
#                 if tgt_label:
#                     info.entity_labels.append(tgt_label)
#                 if edge_type:
#                     info.relation_types.append(edge_type)
#                 if src_label and tgt_label and edge_type:
#                     info.triplet_strings.append(f"{src_label} {edge_type} {tgt_label}")

#         # merge event-node attrs into tokens
#         for doc_id, info in sidecar.items():
#             event_node = nodes.get(info.event_id, {})
#             token_source = []

#             token_source.extend(info.entity_labels)
#             token_source.extend(info.relation_types)
#             token_source.extend(info.triplet_strings)

#             if event_node.get("text"):
#                 token_source.append(event_node["text"])
#             if event_node.get("visual_summary"):
#                 token_source.append(event_node["visual_summary"])

#             for field in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
#                 val = event_node.get(field, [])
#                 if isinstance(val, list):
#                     token_source.extend([json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in val])

#             token_source.append(json.dumps(event_node.get("scene_summary", {}), ensure_ascii=False))

#             info.graph_tokens = set()
#             for item in token_source:
#                 info.graph_tokens.update(_tokenize(item))

#             # dedup
#             info.entity_labels = sorted(set(info.entity_labels))
#             info.relation_types = sorted(set(info.relation_types))
#             info.triplet_strings = sorted(set(info.triplet_strings))

#         self.graph_sidecar[granularity] = sidecar
#         logger.info(f"Loaded sidecar graph for {granularity}: {len(sidecar)} event nodes")

#     # -----------------------------------------------------
#     # Indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
#             return

#         for granularity in self.granularities:
#             if not self.captions[granularity]:
#                 logger.warning(f"No captions loaded for granularity {granularity}")
#                 continue

#             entries_to_index = [
#                 entry for entry in self.captions[granularity]
#                 if entry.timestamp_int[1] <= until_time
#             ]

#             if not entries_to_index:
#                 continue

#             caption_texts = [entry.text for entry in entries_to_index]

#             hipporag = self._get_or_create_hipporag(granularity)
#             hipporag.update(docs=caption_texts)

#             self.indexed_entries[granularity] = entries_to_index
#             logger.info(f"Indexed {len(entries_to_index)} captions for {granularity}")

#         self.indexed_time = until_time

#     # -----------------------------------------------------
#     # Scoring helpers
#     # -----------------------------------------------------

#     def _lookup_entry_from_text(self, granularity: str, text: str) -> Optional[CaptionEntry]:
#         candidates = self.text_to_entries[granularity].get(text, [])
#         if not candidates:
#             return None
#         return candidates[0]

#     def _entry_metadata_tokens(self, entry: CaptionEntry) -> set:
#         toks = set()
#         toks.update(_tokenize(entry.text))
#         toks.update(_tokenize(entry.visual_summary))

#         metadata = entry.metadata or {}
#         for k in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
#             val = metadata.get(k, [])
#             if isinstance(val, list):
#                 for x in val:
#                     toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))

#         scene_summary = metadata.get("scene_summary", {})
#         if scene_summary:
#             toks.update(_tokenize(json.dumps(scene_summary, ensure_ascii=False)))

#         speaker_stats = metadata.get("speaker_stats", [])
#         if isinstance(speaker_stats, list):
#             for x in speaker_stats:
#                 toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))

#         return toks

#     def _normalize_scores(self, scores: List[float]) -> List[float]:
#         if not scores:
#             return []
#         mn, mx = min(scores), max(scores)
#         if mx - mn < 1e-8:
#             return [1.0 for _ in scores]
#         return [(s - mn) / (mx - mn) for s in scores]

#     def _overlap_score(self, query_tokens: set, candidate_tokens: set) -> float:
#         if not query_tokens or not candidate_tokens:
#             return 0.0
#         inter = query_tokens & candidate_tokens
#         return len(inter) / max(1, len(query_tokens))

#     def _graph_aware_rerank(
#         self,
#         granularity: str,
#         query: str,
#         candidates: List[Tuple[CaptionEntry, float]],
#     ) -> List[Tuple[CaptionEntry, float]]:
#         if not candidates:
#             return []

#         query_tokens = _tokenize(query)
#         base_scores = [score for _, score in candidates]
#         base_scores = self._normalize_scores(base_scores)

#         reranked = []
#         for (entry, _), base_score in zip(candidates, base_scores):
#             metadata_score = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))

#             graph_score = 0.0
#             sidecar = self.graph_sidecar[granularity].get(entry.doc_id)
#             if sidecar:
#                 graph_score = self._overlap_score(query_tokens, sidecar.graph_tokens)

#             final_score = (
#                 base_score
#                 + self.graph_score_weight * graph_score
#                 + self.metadata_score_weight * metadata_score
#             )

#             reranked.append((entry, final_score))

#         reranked.sort(key=lambda x: -x[1])
#         return reranked

#     # -----------------------------------------------------
#     # Retrieval
#     # -----------------------------------------------------

#     def retrieve_captions_as_str(self, entries: List[CaptionEntry], include_visual_summary: bool = True) -> str:
#         return "\n\n".join(entry.to_display_str(include_visual_summary=include_visual_summary) for entry in entries)

#     def retrieve(
#         self,
#         query: str,
#         top_k_per_granularity: Union[int, Dict[str, int]] = {
#             "30sec": 10,
#             "3min": 5,
#             "10min": 5,
#             "1h": 3
#         },
#         final_top_k: int = 3,
#         as_context: bool = True
#     ) -> Union[List[CaptionEntry], str]:
#         if self.indexed_time == 0:
#             logger.warning("No captions indexed. Call index(until_time) before retrieve().")
#             return [] if not as_context else ""

#         all_candidates: List[Tuple[CaptionEntry, float]] = []

#         for granularity in self.granularities:
#             if granularity not in self.hipporag:
#                 continue

#             if isinstance(top_k_per_granularity, dict):
#                 granularity_top_k = top_k_per_granularity.get(granularity, 5)
#             else:
#                 granularity_top_k = top_k_per_granularity

#             hipporag = self.hipporag[granularity]
#             retrieval_result = hipporag.retrieve(
#                 queries=[query],
#                 num_to_retrieve=granularity_top_k * 2
#             )

#             if not retrieval_result or not retrieval_result[0].docs:
#                 continue

#             retrieved_docs = retrieval_result[0].docs
#             retrieved_scores = retrieval_result[0].doc_scores if hasattr(retrieval_result[0], 'doc_scores') else [1.0] * len(retrieved_docs)

#             raw_candidates = []
#             for doc_text, score in zip(retrieved_docs, retrieved_scores):
#                 entry = self._lookup_entry_from_text(granularity, doc_text)
#                 if entry is None:
#                     logger.warning(f"Could not find CaptionEntry for retrieved text in {granularity}: {doc_text[:50]}...")
#                     continue
#                 raw_candidates.append((entry, score))

#             reranked_candidates = self._graph_aware_rerank(
#                 granularity=granularity,
#                 query=query,
#                 candidates=raw_candidates,
#             )

#             all_candidates.extend(reranked_candidates[:granularity_top_k])

#         if not all_candidates:
#             logger.warning("No candidates retrieved from any granularity level")
#             return [] if not as_context else ""

#         filtered_entries = self._filter_with_llm(
#             query=query,
#             candidates=all_candidates,
#             final_top_k=final_top_k,
#         )

#         if as_context:
#             return self.retrieve_captions_as_str(filtered_entries, include_visual_summary=True)

#         return filtered_entries

#     def _filter_with_llm(
#         self,
#         query: str,
#         candidates: List[Tuple[CaptionEntry, float]],
#         final_top_k: int,
#     ) -> List[CaptionEntry]:
#         if len(candidates) <= final_top_k:
#             return [entry for entry, _ in candidates]

#         caption_list = []
#         id_to_entry = {}
#         for entry, score in candidates:
#             start_ts, end_ts = entry.timestamp_int
#             caption_info = {
#                 "id": entry.id,
#                 "doc_id": entry.doc_id,
#                 "granularity": entry.granularity,
#                 "start_time": _transform_timestamp(str(start_ts)),
#                 "end_time": _transform_timestamp(str(end_ts)),
#                 "text": entry.text,
#                 "visual_summary": entry.visual_summary,
#                 "score": round(score, 4),
#             }
#             caption_list.append(caption_info)
#             id_to_entry[entry.id] = entry

#         try:
#             prompt = self.prompt_template_manager.render("multiscale_filter")
#         except Exception as e:
#             logger.error(f"Failed to render multiscale_filter template: {e}")
#             return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]

#         filter_message = {
#             "role": "user",
#             "content": f"""Question: {query}

# Retrieved Captions:
# {json.dumps(caption_list, indent=2, ensure_ascii=False)}

# Select the top {final_top_k} most relevant caption IDs to answer the question.
# Return ONLY a JSON array of caption IDs in order of relevance (most relevant first)."""
#         }
#         prompt.append(filter_message)

#         try:
#             response = self.llm_model.generate(prompt)
#             selected_ids = self._parse_filter_response(response, set(id_to_entry.keys()))

#             result = []
#             for cap_id in selected_ids[:final_top_k]:
#                 if cap_id in id_to_entry:
#                     result.append(id_to_entry[cap_id])

#             if len(result) < final_top_k:
#                 existing_ids = {e.id for e in result}
#                 for entry, _ in sorted(candidates, key=lambda x: -x[1]):
#                     if entry.id not in existing_ids:
#                         result.append(entry)
#                         if len(result) >= final_top_k:
#                             break

#             return result

#         except Exception as e:
#             logger.error(f"LLM filtering failed: {e}")
#             return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]

#     def _parse_filter_response(self, response: str, valid_ids: set) -> List[str]:
#         try:
#             match = re.search(r'\[.*?\]', response, re.DOTALL)
#             if match:
#                 ids = json.loads(match.group())
#                 if isinstance(ids, list):
#                     return [str(id) for id in ids if str(id) in valid_ids]
#         except json.JSONDecodeError:
#             pass

#         found_ids = []
#         for valid_id in valid_ids:
#             if valid_id in response:
#                 found_ids.append(valid_id)
#         return found_ids

#     def reset_index(self) -> None:
#         self.hipporag.clear()
#         for g in self.granularities:
#             self.indexed_entries[g] = []
#         self.indexed_time = 0
#         logger.info("Index reset - HippoRAG instances and indexed entries cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
#         return self.caption_id_to_entry.get(caption_id)

# """
# Episodic Memory module for WorldMM.

# Hybrid version:
# - HippoRAG for coarse multiscale caption retrieval
# - sidecar episodic graph / triplets for graph-aware rerank
# - lightweight temporal graph expansion over retrieved candidates
# - object/entity-linked event expansion over retrieved seed events
# - visual_summary is used only in retrieval/rerank text, not in HippoRAG OpenIE graph construction
# - exposes ranked multiscale candidates so WorldMemory can project them to 30s anchors
# """

# import os
# import json
# import logging
# import re
# from collections import defaultdict
# from typing import Dict, List, Any, Optional, Tuple, Union, Set
# from dataclasses import dataclass, field

# from ...llm import LLMModel, PromptTemplateManager
# from ...embedding import EmbeddingModel
# from hipporag import HippoRAG

# logger = logging.getLogger(__name__)


# STOPWORDS = {
#     "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
#     "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
#     "what", "which", "who", "whom", "when", "where", "why", "how",
#     "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
#     "this", "that", "these", "those", "it", "its"
# }


# @dataclass
# class CaptionEntry:
#     id: str
#     doc_id: str
#     text: str
#     start_time: str
#     end_time: str
#     date: str
#     granularity: str
#     video_path: Optional[str] = None
#     visual_summary: str = ""
#     metadata: Dict[str, Any] = field(default_factory=dict)

#     @property
#     def timestamp_int(self) -> Tuple[int, int]:
#         day = self.date.replace('DAY', '').replace('Day', '')
#         start_ts = int(day + self.start_time.zfill(8))
#         end_ts = int(day + self.end_time.zfill(8))
#         return start_ts, end_ts

#     def to_display_str(self, include_visual_summary: bool = True) -> str:
#         start_ts, end_ts = self.timestamp_int
#         base = f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"
#         if include_visual_summary and self.visual_summary:
#             base += f"\nVisual: {self.visual_summary}"
#         return base


# @dataclass
# class GraphEventSidecar:
#     event_id: str
#     doc_id: str
#     granularity: str
#     entity_labels: List[str] = field(default_factory=list)
#     relation_types: List[str] = field(default_factory=list)
#     triplet_strings: List[str] = field(default_factory=list)
#     prev_doc_id: Optional[str] = None
#     next_doc_id: Optional[str] = None
#     graph_tokens: set = field(default_factory=set)


# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"



# def _load_json(file_path: str) -> Any:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)



# def _tokenize(text: str) -> set:
#     toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
#     return {t for t in toks if len(t) > 1 and t not in STOPWORDS}



# def _safe_cache_tag(x: Optional[str]) -> str:
#     if not x:
#         return "default"
#     x = str(x).strip()
#     x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
#     return x or "default"



# def _normalize_phrase(text: str) -> Optional[str]:
#     text = str(text).strip().lower()
#     if not text:
#         return None
#     text = re.sub(r"[_/\\-]+", " ", text)
#     text = re.sub(r"\s+", " ", text).strip(" ,.;:!?\"'`()[]{}")
#     if not text or text in STOPWORDS:
#         return None
#     if re.fullmatch(r"[\d\s:.-]+", text):
#         return None
#     if len(text) < 2:
#         return None
#     return text



# def _timestamp_to_seconds(ts_int: int) -> int:
#     ts_str = str(ts_int)
#     if len(ts_str) < 9:
#         ts_str = ts_str.zfill(9)
#     day = int(ts_str[0])
#     hh = int(ts_str[1:3])
#     mm = int(ts_str[3:5])
#     ss = int(ts_str[5:7])
#     return day * 86400 + hh * 3600 + mm * 60 + ss


# class EpisodicMemory:
#     GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]

#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         llm_model: LLMModel,
#         prompt_template_manager: PromptTemplateManager,
#         granularities: Optional[List[str]] = None,
#         cache_tag: Optional[str] = None,
#     ):
#         self.embedding_model = embedding_model
#         self.llm_model = llm_model
#         self.prompt_template_manager = prompt_template_manager
#         self.granularities = granularities or self.GRANULARITY_ORDER
#         self.cache_tag = _safe_cache_tag(cache_tag)

#         self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
#         self.doc_id_to_entry: Dict[str, Dict[str, CaptionEntry]] = {g: {} for g in self.granularities}
#         self.text_to_entries: Dict[str, Dict[str, List[CaptionEntry]]] = {g: {} for g in self.granularities}

#         self.hipporag: Dict[str, HippoRAG] = {}
#         self.indexed_entries: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
#         self.indexed_time: int = 0

#         self.triplets_by_doc: Dict[str, Dict[str, List[List[str]]]] = {g: {} for g in self.granularities}
#         self.graph_sidecar: Dict[str, Dict[str, GraphEventSidecar]] = {g: {} for g in self.granularities}

#         # lazy inverse index for object/entity-linked event expansion
#         self.link_entity_to_doc_ids: Dict[str, Dict[str, Set[str]]] = {g: {} for g in self.granularities}
#         self.link_entity_index_built: Dict[str, bool] = {g: False for g in self.granularities}

#         # graph-aware rerank weights
#         self.retrieval_text_score_weight = 0.20
#         self.graph_score_weight = 0.25
#         self.metadata_score_weight = 0.15

#         # visual-summary-aware retrieval text only for finer granularities.
#         # NOTE: this is used only at retrieval/rerank time, NOT passed into HippoRAG OpenIE graph building.
#         self.visual_summary_retrieval_granularities = {"30sec", "3min"}

#         # temporal graph expansion over event neighbors
#         self.graph_expand_top_n = 4
#         self.graph_expand_hops = 1
#         self.graph_expand_decay = 0.60

#         # object/entity-linked event expansion
#         self.entity_expand_top_n = 4
#         self.entity_expand_limit_per_seed = 6
#         self.entity_expand_decay = 0.75

#     def _get_or_create_hipporag(self, granularity: str) -> HippoRAG:
#         if granularity not in self.hipporag:
#             save_dir = os.path.join(".cache", "episodic_memory", self.cache_tag, granularity)
#             logger.info(f"Using HippoRAG cache dir for {granularity}: {save_dir}")
#             self.hipporag[granularity] = HippoRAG(
#                 save_dir=save_dir,
#                 llm_model=self.llm_model,
#                 embedding_model=self.embedding_model,
#             )
#         return self.hipporag[granularity]

#     # -----------------------------------------------------
#     # Loading captions
#     # -----------------------------------------------------

#     def load_captions_from_files(self, caption_files: Dict[str, str]) -> None:
#         for granularity, file_path in caption_files.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not configured")
#                 continue
#             try:
#                 data = _load_json(file_path)
#                 self._process_caption_data(data, granularity)
#                 logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")
#             except Exception as e:
#                 logger.error(f"Failed to load captions from {file_path}: {e}")

#     def load_captions_from_data(self, caption_data: Dict[str, List[Dict[str, Any]]]) -> None:
#         for granularity, data in caption_data.items():
#             if granularity not in self.granularities:
#                 logger.warning(f"Skipping granularity {granularity} - not configured")
#                 continue
#             self._process_caption_data(data, granularity)
#             logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")

#     def _make_doc_id(self, entry: Dict[str, Any], granularity: str, idx: int) -> str:
#         if entry.get("doc_id"):
#             return str(entry["doc_id"])
#         date = str(entry.get("date", ""))
#         start_time = str(entry.get("start_time", "")).zfill(8)
#         end_time = str(entry.get("end_time", "")).zfill(8)
#         if date and start_time and end_time:
#             suffix = "" if granularity == "30sec" else f"_{granularity}"
#             return f"{date}_{start_time}_{end_time}{suffix}"
#         return f"{granularity}_{idx}"

#     def _use_visual_summary_in_retrieval(self, granularity: str) -> bool:
#         return granularity in self.visual_summary_retrieval_granularities

#     def _entry_retrieval_text(self, entry: CaptionEntry) -> str:
#         if self._use_visual_summary_in_retrieval(entry.granularity) and entry.visual_summary:
#             return f"{entry.text}\nVisual: {entry.visual_summary}"
#         return entry.text

#     def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
#         self.captions[granularity] = []
#         self.doc_id_to_entry[granularity] = {}
#         self.text_to_entries[granularity] = {}
#         self.link_entity_index_built[granularity] = False
#         self.link_entity_to_doc_ids[granularity] = {}

#         for idx, entry in enumerate(data):
#             doc_id = self._make_doc_id(entry, granularity, idx)
#             caption_id = doc_id

#             metadata = {
#                 "action_threads": entry.get("action_threads", entry.get("main_actions", [])),
#                 "object_threads": entry.get("object_threads", entry.get("salient_objects", [])),
#                 "topic_threads": entry.get("topic_threads", entry.get("conversation_focus", [])),
#                 "speaker_stats": entry.get("speaker_stats", entry.get("speakers", [])),
#                 "scene_summary": entry.get("scene_summary", {}),
#                 "visual_object_threads": entry.get("visual_object_threads", entry.get("visual_objects", [])),
#                 "source_doc_ids": list(entry.get("source_doc_ids", []) or []),
#                 "child_ids": list(entry.get("child_ids", []) or []),
#             }

#             caption_entry = CaptionEntry(
#                 id=caption_id,
#                 doc_id=doc_id,
#                 text=entry.get("text", entry.get("fine_caption", "")),
#                 start_time=str(entry.get("start_time", "")),
#                 end_time=str(entry.get("end_time", "")),
#                 date=str(entry.get("date", "")),
#                 granularity=granularity,
#                 video_path=entry.get("video_path"),
#                 visual_summary=entry.get("visual_summary", ""),
#                 metadata=metadata,
#             )

#             self.captions[granularity].append(caption_entry)
#             self.caption_id_to_entry[caption_id] = caption_entry
#             self.doc_id_to_entry[granularity][doc_id] = caption_entry
#             self.text_to_entries[granularity].setdefault(caption_entry.text, []).append(caption_entry)

#     # -----------------------------------------------------
#     # Sidecar loading
#     # -----------------------------------------------------

#     def load_sidecar_from_files(
#         self,
#         triplet_files: Optional[Dict[str, str]] = None,
#         graph_files: Optional[Dict[str, str]] = None,
#     ) -> None:
#         if triplet_files:
#             for granularity, file_path in triplet_files.items():
#                 if granularity not in self.granularities:
#                     continue
#                 data = _load_json(file_path)
#                 self._process_triplet_data(data, granularity)

#         if graph_files:
#             for granularity, file_path in graph_files.items():
#                 if granularity not in self.granularities:
#                     continue
#                 data = _load_json(file_path)
#                 self._process_graph_data(data, granularity)

#     def load_sidecar_from_data(
#         self,
#         triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
#         graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> None:
#         if triplet_data:
#             for granularity, data in triplet_data.items():
#                 if granularity not in self.granularities:
#                     continue
#                 self._process_triplet_data(data, granularity)

#         if graph_data:
#             for granularity, data in graph_data.items():
#                 if granularity not in self.granularities:
#                     continue
#                 self._process_graph_data(data, granularity)

#     def _process_triplet_data(self, data: Dict[str, Any], granularity: str) -> None:
#         self.triplets_by_doc[granularity] = dict(data.get("triplet_map", {}))
#         logger.info(f"Loaded sidecar triplets for {granularity}: {len(self.triplets_by_doc[granularity])} docs")
#         self.link_entity_index_built[granularity] = False

#     def _process_graph_data(self, data: Dict[str, Any], granularity: str) -> None:
#         nodes = {node["id"]: node for node in data.get("nodes", [])}
#         edges = data.get("edges", [])
#         doc_id_to_event_id = data.get("doc_id_to_event_id", {})

#         sidecar = {}
#         for doc_id, event_id in doc_id_to_event_id.items():
#             sidecar[doc_id] = GraphEventSidecar(
#                 event_id=event_id,
#                 doc_id=doc_id,
#                 granularity=granularity,
#             )

#         event_id_to_doc_id = {event_id: doc_id for doc_id, event_id in doc_id_to_event_id.items()}

#         for edge in edges:
#             source = edge["source"]
#             target = edge["target"]
#             edge_type = edge["type"]
#             event_id = edge.get("event_id")

#             if edge_type == "before" and source in event_id_to_doc_id and target in event_id_to_doc_id:
#                 src_doc = event_id_to_doc_id[source]
#                 tgt_doc = event_id_to_doc_id[target]
#                 if src_doc in sidecar:
#                     sidecar[src_doc].next_doc_id = tgt_doc
#                 if tgt_doc in sidecar:
#                     sidecar[tgt_doc].prev_doc_id = src_doc
#                 continue

#             if event_id not in event_id_to_doc_id:
#                 continue

#             doc_id = event_id_to_doc_id[event_id]
#             info = sidecar[doc_id]

#             src_node = nodes.get(source, {})
#             tgt_node = nodes.get(target, {})
#             src_type = src_node.get("type")
#             tgt_type = tgt_node.get("type")
#             src_label = src_node.get("label", "")
#             tgt_label = tgt_node.get("label", "")

#             if source == event_id and tgt_type != "Event":
#                 if tgt_label:
#                     info.entity_labels.append(tgt_label)
#                 if edge_type:
#                     info.relation_types.append(edge_type)
#                 continue

#             if src_type != "Event" and tgt_type != "Event":
#                 if src_label:
#                     info.entity_labels.append(src_label)
#                 if tgt_label:
#                     info.entity_labels.append(tgt_label)
#                 if edge_type:
#                     info.relation_types.append(edge_type)
#                 if src_label and tgt_label and edge_type:
#                     info.triplet_strings.append(f"{src_label} {edge_type} {tgt_label}")

#         for doc_id, info in sidecar.items():
#             event_node = nodes.get(info.event_id, {})
#             token_source = []
#             token_source.extend(info.entity_labels)
#             token_source.extend(info.relation_types)
#             token_source.extend(info.triplet_strings)
#             if event_node.get("text"):
#                 token_source.append(event_node["text"])
#             if event_node.get("visual_summary"):
#                 token_source.append(event_node["visual_summary"])
#             for field in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
#                 val = event_node.get(field, [])
#                 if isinstance(val, list):
#                     token_source.extend([
#                         json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
#                         for x in val
#                     ])
#             token_source.append(json.dumps(event_node.get("scene_summary", {}), ensure_ascii=False))

#             info.graph_tokens = set()
#             for item in token_source:
#                 info.graph_tokens.update(_tokenize(item))

#             info.entity_labels = sorted(set(info.entity_labels))
#             info.relation_types = sorted(set(info.relation_types))
#             info.triplet_strings = sorted(set(info.triplet_strings))

#         self.graph_sidecar[granularity] = sidecar
#         self.link_entity_index_built[granularity] = False
#         logger.info(f"Loaded sidecar graph for {granularity}: {len(sidecar)} event nodes")

#     # -----------------------------------------------------
#     # Indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
#             return

#         for granularity in self.granularities:
#             if not self.captions[granularity]:
#                 logger.warning(f"No captions loaded for granularity {granularity}")
#                 continue

#             entries_to_index = [
#                 entry for entry in self.captions[granularity]
#                 if entry.timestamp_int[1] <= until_time
#             ]
#             if not entries_to_index:
#                 continue

#             # IMPORTANT:
#             # Only raw caption text is given to HippoRAG, so visual_summary does NOT enter OpenIE / fact graph construction.
#             caption_texts = [entry.text for entry in entries_to_index]
#             hipporag = self._get_or_create_hipporag(granularity)
#             hipporag.update(docs=caption_texts)
#             self.indexed_entries[granularity] = entries_to_index
#             logger.info(f"Indexed {len(entries_to_index)} captions for {granularity}")

#         self.indexed_time = until_time

#     # -----------------------------------------------------
#     # Helpers
#     # -----------------------------------------------------

#     def _lookup_entry_from_text(self, granularity: str, text: str) -> Optional[CaptionEntry]:
#         candidates = self.text_to_entries[granularity].get(text, [])
#         if not candidates:
#             return None
#         return candidates[0]

#     def _entry_retrieval_tokens(self, entry: CaptionEntry) -> set:
#         return _tokenize(self._entry_retrieval_text(entry))

#     def _entry_metadata_tokens(self, entry: CaptionEntry) -> set:
#         toks = set()
#         toks.update(_tokenize(entry.text))
#         toks.update(_tokenize(entry.visual_summary))
#         metadata = entry.metadata or {}
#         for k in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
#             val = metadata.get(k, [])
#             if isinstance(val, list):
#                 for x in val:
#                     toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
#         scene_summary = metadata.get("scene_summary", {})
#         if scene_summary:
#             toks.update(_tokenize(json.dumps(scene_summary, ensure_ascii=False)))
#         speaker_stats = metadata.get("speaker_stats", [])
#         if isinstance(speaker_stats, list):
#             for x in speaker_stats:
#                 toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
#         return toks

#     def _normalize_scores(self, scores: List[float]) -> List[float]:
#         if not scores:
#             return []
#         mn, mx = min(scores), max(scores)
#         if mx - mn < 1e-8:
#             return [1.0 for _ in scores]
#         return [(s - mn) / (mx - mn) for s in scores]

#     def _overlap_score(self, query_tokens: set, candidate_tokens: set) -> float:
#         if not query_tokens or not candidate_tokens:
#             return 0.0
#         inter = query_tokens & candidate_tokens
#         return len(inter) / max(1, len(query_tokens))

#     def _graph_aware_rerank(
#         self,
#         granularity: str,
#         query: str,
#         candidates: List[Tuple[CaptionEntry, float]],
#     ) -> List[Tuple[CaptionEntry, float]]:
#         if not candidates:
#             return []

#         query_tokens = _tokenize(query)
#         base_scores = [score for _, score in candidates]
#         base_scores = self._normalize_scores(base_scores)

#         reranked = []
#         for (entry, _), base_score in zip(candidates, base_scores):
#             retrieval_text_score = self._overlap_score(query_tokens, self._entry_retrieval_tokens(entry))
#             metadata_score = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))
#             graph_score = 0.0
#             sidecar = self.graph_sidecar[granularity].get(entry.doc_id)
#             if sidecar:
#                 graph_score = self._overlap_score(query_tokens, sidecar.graph_tokens)
#             final_score = (
#                 base_score
#                 + self.retrieval_text_score_weight * retrieval_text_score
#                 + self.graph_score_weight * graph_score
#                 + self.metadata_score_weight * metadata_score
#             )
#             reranked.append((entry, final_score))

#         reranked.sort(key=lambda x: -x[1])
#         return reranked

#     def _collect_link_phrases(self, value: Any) -> List[str]:
#         phrases: List[str] = []
#         if value is None:
#             return phrases
#         if isinstance(value, str):
#             norm = _normalize_phrase(value)
#             if norm:
#                 phrases.append(norm)
#             return phrases
#         if isinstance(value, dict):
#             for key in ["label", "name", "object", "entity", "item", "value", "mention", "text"]:
#                 if key in value:
#                     phrases.extend(self._collect_link_phrases(value[key]))
#             return phrases
#         if isinstance(value, list):
#             for item in value:
#                 phrases.extend(self._collect_link_phrases(item))
#             return phrases
#         return phrases

#     def _get_link_entities_for_entry(self, entry: CaptionEntry, granularity: str) -> Set[str]:
#         labels: Set[str] = set()

#         sidecar = self.graph_sidecar.get(granularity, {}).get(entry.doc_id)
#         if sidecar is not None:
#             for label in sidecar.entity_labels:
#                 norm = _normalize_phrase(label)
#                 if norm:
#                     labels.add(norm)
#             for triplet_str in sidecar.triplet_strings:
#                 # heuristic split: "a relation b" -> keep whole phrase too noisy; rely on entity_labels first
#                 parts = [p.strip() for p in re.split(r"\s+(?:is|are|was|were|has|have|had|at|in|on|with|to|from|of)\s+", triplet_str, maxsplit=1)]
#                 for part in parts:
#                     norm = _normalize_phrase(part)
#                     if norm and len(norm.split()) <= 4:
#                         labels.add(norm)

#         metadata = entry.metadata or {}
#         for field in ["object_threads", "visual_object_threads"]:
#             labels.update(self._collect_link_phrases(metadata.get(field, [])))

#         return {x for x in labels if x}

#     def _ensure_link_entity_index(self, granularity: str) -> None:
#         if self.link_entity_index_built.get(granularity, False):
#             return

#         index: Dict[str, Set[str]] = defaultdict(set)
#         for entry in self.captions.get(granularity, []):
#             for label in self._get_link_entities_for_entry(entry, granularity):
#                 index[label].add(entry.doc_id)

#         self.link_entity_to_doc_ids[granularity] = dict(index)
#         self.link_entity_index_built[granularity] = True
#         logger.info(
#             "Built object/entity link index for %s: %d labels",
#             granularity,
#             len(self.link_entity_to_doc_ids[granularity]),
#         )

#     def _expand_temporal_neighbors(
#         self,
#         granularity: str,
#         ranked_candidates: List[Tuple[CaptionEntry, float]],
#     ) -> List[Tuple[CaptionEntry, float]]:
#         if not ranked_candidates:
#             return []

#         expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
#         seeds = ranked_candidates[: max(1, self.graph_expand_top_n)]

#         for entry, seed_score in seeds:
#             current_doc_ids = [entry.doc_id]
#             decay = float(seed_score)

#             for _ in range(self.graph_expand_hops):
#                 next_doc_ids: List[str] = []
#                 decay *= self.graph_expand_decay
#                 for current_doc_id in current_doc_ids:
#                     sidecar = self.graph_sidecar.get(granularity, {}).get(current_doc_id)
#                     if sidecar is None:
#                         continue
#                     for neighbor_doc_id in [sidecar.prev_doc_id, sidecar.next_doc_id]:
#                         if not neighbor_doc_id:
#                             continue
#                         neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
#                         if neighbor_entry is None:
#                             continue
#                         if neighbor_entry.timestamp_int[1] > self.indexed_time:
#                             continue
#                         prev = expanded.get(neighbor_doc_id)
#                         if prev is None or decay > prev[1]:
#                             expanded[neighbor_doc_id] = (neighbor_entry, decay)
#                         next_doc_ids.append(neighbor_doc_id)
#                 current_doc_ids = next_doc_ids
#                 if not current_doc_ids:
#                     break

#         return sorted(expanded.values(), key=lambda x: -x[1])

#     def _expand_entity_neighbors(
#         self,
#         granularity: str,
#         ranked_candidates: List[Tuple[CaptionEntry, float]],
#     ) -> List[Tuple[CaptionEntry, float]]:
#         if not ranked_candidates:
#             return []

#         self._ensure_link_entity_index(granularity)
#         index = self.link_entity_to_doc_ids.get(granularity, {})
#         if not index:
#             return []

#         expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
#         seeds = ranked_candidates[: max(1, self.entity_expand_top_n)]

#         for seed_entry, seed_score in seeds:
#             seed_labels = self._get_link_entities_for_entry(seed_entry, granularity)
#             if not seed_labels:
#                 continue

#             candidate_overlap_counts: Dict[str, int] = defaultdict(int)
#             for label in seed_labels:
#                 for neighbor_doc_id in index.get(label, set()):
#                     if neighbor_doc_id == seed_entry.doc_id:
#                         continue
#                     candidate_overlap_counts[neighbor_doc_id] += 1

#             seed_time = _timestamp_to_seconds(seed_entry.timestamp_int[0])
#             scored_neighbors: List[Tuple[CaptionEntry, float]] = []

#             for neighbor_doc_id, overlap_count in candidate_overlap_counts.items():
#                 neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
#                 if neighbor_entry is None:
#                     continue
#                 if neighbor_entry.timestamp_int[1] > self.indexed_time:
#                     continue

#                 neighbor_time = _timestamp_to_seconds(neighbor_entry.timestamp_int[0])
#                 time_gap = abs(neighbor_time - seed_time)
#                 temporal_proximity = 1.0 / (1.0 + time_gap / 300.0)  # 5-minute scale
#                 overlap_strength = min(overlap_count, 3) / 3.0

#                 score = float(seed_score) * self.entity_expand_decay * (0.55 + 0.45 * overlap_strength) * temporal_proximity
#                 scored_neighbors.append((neighbor_entry, score))

#             scored_neighbors.sort(key=lambda x: -x[1])
#             for neighbor_entry, score in scored_neighbors[: self.entity_expand_limit_per_seed]:
#                 prev = expanded.get(neighbor_entry.doc_id)
#                 if prev is None or score > prev[1]:
#                     expanded[neighbor_entry.doc_id] = (neighbor_entry, score)

#         return sorted(expanded.values(), key=lambda x: -x[1])

#     def expand_entry_to_30s_doc_ids(self, entry: CaptionEntry) -> List[str]:
#         if entry.granularity == "30sec":
#             return [entry.doc_id]
#         source_doc_ids = list(entry.metadata.get("source_doc_ids", []) or [])
#         child_ids = list(entry.metadata.get("child_ids", []) or [])
#         candidate_ids = source_doc_ids or child_ids
#         if candidate_ids:
#             return [str(x) for x in candidate_ids]
#         return [entry.doc_id]

#     def get_caption_by_doc_id(self, doc_id: str, granularity: Optional[str] = None) -> Optional[CaptionEntry]:
#         if granularity is not None:
#             return self.doc_id_to_entry.get(granularity, {}).get(doc_id)
#         for g in self.granularities:
#             if doc_id in self.doc_id_to_entry.get(g, {}):
#                 return self.doc_id_to_entry[g][doc_id]
#         return None

#     def get_triplets_by_doc_id(self, doc_id: str, granularity: str = "30sec") -> List[List[str]]:
#         return self.triplets_by_doc.get(granularity, {}).get(doc_id, [])

#     def get_parent_caption(self, doc_id: str, parent_granularity: str = "3min") -> Optional[CaptionEntry]:
#         child_entry = self.get_caption_by_doc_id(doc_id, "30sec")
#         if child_entry is None:
#             return None
#         if parent_granularity not in self.doc_id_to_entry:
#             return None

#         child_start, child_end = child_entry.timestamp_int
#         best_parent: Optional[CaptionEntry] = None
#         best_span: Optional[int] = None

#         for parent in self.captions.get(parent_granularity, []):
#             if parent.date != child_entry.date:
#                 continue
#             parent_start, parent_end = parent.timestamp_int
#             if parent_start <= child_start and parent_end >= child_end:
#                 span = parent_end - parent_start
#                 if best_parent is None or best_span is None or span < best_span:
#                     best_parent = parent
#                     best_span = span

#         return best_parent

#     # -----------------------------------------------------
#     # Retrieval
#     # -----------------------------------------------------

#     def retrieve_captions_as_str(self, entries: List[CaptionEntry], include_visual_summary: bool = True) -> str:
#         return "\n\n".join(entry.to_display_str(include_visual_summary=include_visual_summary) for entry in entries)

#     def retrieve_ranked(
#         self,
#         query: str,
#         top_k_per_granularity: Union[int, Dict[str, int]] = None,
#         dedup_by_doc_id: bool = True,
#     ) -> List[Tuple[CaptionEntry, float]]:
#         if top_k_per_granularity is None:
#             top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

#         if self.indexed_time == 0:
#             logger.warning("No captions indexed. Call index(until_time) before retrieve().")
#             return []

#         all_candidates: List[Tuple[CaptionEntry, float]] = []
#         for granularity in self.granularities:
#             if granularity not in self.hipporag:
#                 continue

#             if isinstance(top_k_per_granularity, dict):
#                 granularity_top_k = top_k_per_granularity.get(granularity, 5)
#             else:
#                 granularity_top_k = top_k_per_granularity

#             hipporag = self.hipporag[granularity]
#             retrieval_result = hipporag.retrieve(
#                 queries=[query],
#                 num_to_retrieve=granularity_top_k * 2,
#             )
#             if not retrieval_result or not retrieval_result[0].docs:
#                 continue

#             retrieved_docs = retrieval_result[0].docs
#             retrieved_scores = (
#                 retrieval_result[0].doc_scores
#                 if hasattr(retrieval_result[0], 'doc_scores')
#                 else [1.0] * len(retrieved_docs)
#             )

#             raw_candidates = []
#             for doc_text, score in zip(retrieved_docs, retrieved_scores):
#                 entry = self._lookup_entry_from_text(granularity, doc_text)
#                 if entry is None:
#                     logger.warning(
#                         f"Could not find CaptionEntry for retrieved text in {granularity}: {doc_text[:50]}..."
#                     )
#                     continue
#                 raw_candidates.append((entry, float(score)))

#             reranked_candidates = self._graph_aware_rerank(
#                 granularity=granularity,
#                 query=query,
#                 candidates=raw_candidates,
#             )

#             selected_base = reranked_candidates[:granularity_top_k]
#             expanded_temporal = self._expand_temporal_neighbors(granularity, selected_base)
#             expanded_entity = self._expand_entity_neighbors(granularity, selected_base)

#             logger.info(
#                 "Episodic %s: base=%d temporal_expanded=%d entity_expanded=%d",
#                 granularity,
#                 len(selected_base),
#                 len(expanded_temporal),
#                 len(expanded_entity),
#             )

#             all_candidates.extend(selected_base)
#             all_candidates.extend(expanded_temporal)
#             all_candidates.extend(expanded_entity)

#         if dedup_by_doc_id:
#             best_by_doc: Dict[str, Tuple[CaptionEntry, float]] = {}
#             for entry, score in all_candidates:
#                 prev = best_by_doc.get(entry.doc_id)
#                 if prev is None or score > prev[1]:
#                     best_by_doc[entry.doc_id] = (entry, score)
#             all_candidates = list(best_by_doc.values())

#         all_candidates.sort(key=lambda x: -x[1])
#         return all_candidates

#     def retrieve(
#         self,
#         query: str,
#         top_k_per_granularity: Union[int, Dict[str, int]] = None,
#         final_top_k: int = 3,
#         as_context: bool = True,
#     ) -> Union[List[CaptionEntry], str]:
#         if top_k_per_granularity is None:
#             top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

#         ranked = self.retrieve_ranked(
#             query=query,
#             top_k_per_granularity=top_k_per_granularity,
#             dedup_by_doc_id=True,
#         )
#         if not ranked:
#             return [] if not as_context else ""

#         result_entries = [entry for entry, _ in ranked[:final_top_k]]
#         if as_context:
#             return self.retrieve_captions_as_str(result_entries, include_visual_summary=True)
#         return result_entries

#     def reset_index(self) -> None:
#         self.hipporag.clear()
#         for g in self.granularities:
#             self.indexed_entries[g] = []
#             self.link_entity_index_built[g] = False
#             self.link_entity_to_doc_ids[g] = {}
#         self.indexed_time = 0
#         logger.info("Index reset - HippoRAG instances and indexed entries cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
#         return self.caption_id_to_entry.get(caption_id)



"""
Episodic Memory module for WorldMM.

Hybrid version:
- HippoRAG for coarse multiscale caption retrieval
- sidecar episodic graph / triplets for graph-aware rerank
- lightweight temporal graph expansion over retrieved candidates
- object/entity-linked event expansion over retrieved seed events
- visual_summary is used only in retrieval/rerank text, not in HippoRAG OpenIE graph construction
- exposes ranked multiscale candidates so WorldMemory can project them to 30s anchors

Revised on top of the user's full original version:
- keep the original retrieval / expansion structure intact
- add compatibility for newer multiscale fields such as critical_speech_lines
- let rerank / retrieval text consume critical speech evidence without pushing it into HippoRAG graph construction
- let sidecar graph tokens consume critical_speech_lines when present
"""

import os
import json
import logging
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field

from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel
from hipporag import HippoRAG

logger = logging.getLogger(__name__)


STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "this", "that", "these", "those", "it", "its"
}


@dataclass
class CaptionEntry:
    id: str
    doc_id: str
    text: str
    start_time: str
    end_time: str
    date: str
    granularity: str
    video_path: Optional[str] = None
    visual_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts

    def to_display_str(self, include_visual_summary: bool = True) -> str:
        start_ts, end_ts = self.timestamp_int
        base = f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"
        if include_visual_summary and self.visual_summary:
            base += f"\nVisual: {self.visual_summary}"
        return base


@dataclass
class GraphEventSidecar:
    event_id: str
    doc_id: str
    granularity: str
    entity_labels: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    triplet_strings: List[str] = field(default_factory=list)
    prev_doc_id: Optional[str] = None
    next_doc_id: Optional[str] = None
    graph_tokens: set = field(default_factory=set)


def _transform_timestamp(ts_str: str) -> str:
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"



def _load_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _tokenize(text: str) -> set:
    toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
    return {t for t in toks if len(t) > 1 and t not in STOPWORDS}



def _safe_cache_tag(x: Optional[str]) -> str:
    if not x:
        return "default"
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x or "default"



def _normalize_phrase(text: str) -> Optional[str]:
    text = str(text).strip().lower()
    if not text:
        return None
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?\"'`()[]{}")
    if not text or text in STOPWORDS:
        return None
    if re.fullmatch(r"[\d\s:.-]+", text):
        return None
    if len(text) < 2:
        return None
    return text



def _timestamp_to_seconds(ts_int: int) -> int:
    ts_str = str(ts_int)
    if len(ts_str) < 9:
        ts_str = ts_str.zfill(9)
    day = int(ts_str[0])
    hh = int(ts_str[1:3])
    mm = int(ts_str[3:5])
    ss = int(ts_str[5:7])
    return day * 86400 + hh * 3600 + mm * 60 + ss


class EpisodicMemory:
    GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        prompt_template_manager: PromptTemplateManager,
        granularities: Optional[List[str]] = None,
        cache_tag: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.prompt_template_manager = prompt_template_manager
        self.granularities = granularities or self.GRANULARITY_ORDER
        self.cache_tag = _safe_cache_tag(cache_tag)

        self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
        self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
        self.doc_id_to_entry: Dict[str, Dict[str, CaptionEntry]] = {g: {} for g in self.granularities}
        self.text_to_entries: Dict[str, Dict[str, List[CaptionEntry]]] = {g: {} for g in self.granularities}

        self.hipporag: Dict[str, HippoRAG] = {}
        self.indexed_entries: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
        self.indexed_time: int = 0

        self.triplets_by_doc: Dict[str, Dict[str, List[List[str]]]] = {g: {} for g in self.granularities}
        self.graph_sidecar: Dict[str, Dict[str, GraphEventSidecar]] = {g: {} for g in self.granularities}

        # lazy inverse index for object/entity-linked event expansion
        self.link_entity_to_doc_ids: Dict[str, Dict[str, Set[str]]] = {g: {} for g in self.granularities}
        self.link_entity_index_built: Dict[str, bool] = {g: False for g in self.granularities}

        # graph-aware rerank weights
        self.retrieval_text_score_weight = 0.20
        self.graph_score_weight = 0.25
        self.metadata_score_weight = 0.15

        # visual-summary-aware retrieval text only for finer granularities.
        # NOTE: this is used only at retrieval/rerank time, NOT passed into HippoRAG OpenIE graph building.
        self.visual_summary_retrieval_granularities = {"30sec", "3min"}

        # newer multiscale outputs may include critical speech lines.
        # We only use them at retrieval / rerank time, not for HippoRAG graph construction.
        self.critical_speech_retrieval_granularities = {"30sec", "3min", "10min", "1h"}
        self.max_critical_speech_lines_in_retrieval = 4

        # temporal graph expansion over event neighbors
        self.graph_expand_top_n = 4
        self.graph_expand_hops = 1
        self.graph_expand_decay = 0.60

        # object/entity-linked event expansion
        self.entity_expand_top_n = 4
        self.entity_expand_limit_per_seed = 6
        self.entity_expand_decay = 0.75
        self.last_retrieval_debug: Dict[str, Any] = {}

    def _get_or_create_hipporag(self, granularity: str) -> HippoRAG:
        if granularity not in self.hipporag:
            save_dir = os.path.join(".cache", "episodic_memory", self.cache_tag, granularity)
            logger.info(f"Using HippoRAG cache dir for {granularity}: {save_dir}")
            self.hipporag[granularity] = HippoRAG(
                save_dir=save_dir,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
            )
        return self.hipporag[granularity]

    def _load_cached_hipporag_for_retrieval(self, hipporag: HippoRAG, expected_docs: int, granularity: str) -> bool:
        """Prepare an existing HippoRAG cache without running OpenIE again.

        Query workers should not rebuild graph/OpenIE state for memory_ready sessions.
        If the cache is incomplete, return False and let the original update path handle it.
        """
        cache_only = str(os.getenv("WORLDMM_QUERY_USE_CACHED_HIPPORAG", "1")).lower() not in {"0", "false", "no"}
        if not cache_only:
            return False
        strict_load_only = str(os.getenv("WORLDMM_QUERY_STRICT_LOAD_ONLY", "1")).lower() not in {"0", "false", "no"}
        try:
            passage_count = len(hipporag.chunk_embedding_store.get_all_ids())
            entity_count = len(hipporag.entity_embedding_store.get_all_ids())
            fact_count = len(hipporag.fact_embedding_store.get_all_ids())
            graph_nodes = hipporag.graph.vcount()
            if passage_count != expected_docs or graph_nodes == 0 or (entity_count == 0 and fact_count == 0):
                logger.info(
                    "HippoRAG cache incomplete or mismatched for %s: passages=%s/%s entities=%s facts=%s graph_nodes=%s",
                    granularity,
                    passage_count,
                    expected_docs,
                    entity_count,
                    fact_count,
                    graph_nodes,
                )
                if strict_load_only:
                    self.last_retrieval_debug.setdefault("cache_warnings", []).append(
                        {
                            "granularity": granularity,
                            "reason": "hipporag_cache_incomplete_or_mismatched",
                            "passage_count": passage_count,
                            "expected_docs": expected_docs,
                            "entity_count": entity_count,
                            "fact_count": fact_count,
                            "graph_nodes": graph_nodes,
                        }
                    )
                    # Do not rebuild in query strict load-only mode. Keep loading so
                    # retrieval can fall back to already loaded captions if cached
                    # passage texts no longer match the active 30s source.
                    return True
                return False
            if not getattr(hipporag, "ready_to_retrieve", False):
                hipporag.prepare_retrieval_objects()
            logger.info(
                "Loaded cached HippoRAG retrieval objects for %s: passages=%s entities=%s facts=%s graph_nodes=%s",
                granularity,
                passage_count,
                entity_count,
                fact_count,
                graph_nodes,
            )
            return True
        except Exception as exc:
            if strict_load_only:
                raise RuntimeError(f"strict load-only HippoRAG cache load failed for {granularity}: {exc}") from exc
            logger.warning("Failed to load cached HippoRAG for %s; falling back to update: %s", granularity, exc)
            return False

    def _summary_like_query(self, query: str) -> bool:
        text = str(query or "").lower()
        return any(
            marker in text
            for marker in [
                "总结",
                "概括",
                "主要发生",
                "整个视频",
                "全过程",
                "从头到尾",
                "summary",
                "summarize",
                "overall",
                "entire video",
                "whole video",
            ]
        )

    def _caption_fallback_rank(
        self,
        query: str,
        granularity: str,
        top_k: int,
        reason: str,
    ) -> List[Tuple[CaptionEntry, float]]:
        entries = list(self.indexed_entries.get(granularity) or self.captions.get(granularity) or [])
        if not entries:
            return []
        query_tokens = _tokenize(query)
        summary_like = self._summary_like_query(query)
        scored: List[Tuple[CaptionEntry, float]] = []
        total = max(len(entries), 1)
        for idx, entry in enumerate(entries):
            token_score = self._overlap_score(query_tokens, self._entry_retrieval_tokens(entry)) if query_tokens else 0.0
            if token_score <= 0.0 and (summary_like or not query_tokens):
                token_score = 0.55
            if token_score <= 0.0:
                continue
            chronological_bonus = max(0.0, 1.0 - idx / total) * 0.05
            scored.append((entry, float(token_score + chronological_bonus)))
        scored.sort(key=lambda x: -x[1])
        selected = scored[: max(top_k, 1)]
        if selected:
            self.last_retrieval_debug.setdefault("fallback_events", []).append(
                {
                    "granularity": granularity,
                    "reason": reason,
                    "candidate_count": len(entries),
                    "selected_count": len(selected),
                    "doc_ids": [entry.doc_id for entry, _ in selected],
                }
            )
        return selected

    # -----------------------------------------------------
    # Loading captions
    # -----------------------------------------------------

    def load_captions_from_files(self, caption_files: Dict[str, str]) -> None:
        for granularity, file_path in caption_files.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not configured")
                continue
            try:
                data = _load_json(file_path)
                self._process_caption_data(data, granularity)
                logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")
            except Exception as e:
                logger.error(f"Failed to load captions from {file_path}: {e}")

    def load_captions_from_data(self, caption_data: Dict[str, List[Dict[str, Any]]]) -> None:
        for granularity, data in caption_data.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not configured")
                continue
            self._process_caption_data(data, granularity)
            logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")

    def _make_doc_id(self, entry: Dict[str, Any], granularity: str, idx: int) -> str:
        if entry.get("doc_id"):
            return str(entry["doc_id"])
        date = str(entry.get("date", ""))
        start_time = str(entry.get("start_time", "")).zfill(8)
        end_time = str(entry.get("end_time", "")).zfill(8)
        if date and start_time and end_time:
            suffix = "" if granularity == "30sec" else f"_{granularity}"
            return f"{date}_{start_time}_{end_time}{suffix}"
        return f"{granularity}_{idx}"

    def _use_visual_summary_in_retrieval(self, granularity: str) -> bool:
        return granularity in self.visual_summary_retrieval_granularities

    def _use_critical_speech_in_retrieval(self, granularity: str) -> bool:
        return granularity in self.critical_speech_retrieval_granularities

    def _get_critical_speech_lines(self, entry: CaptionEntry) -> List[str]:
        metadata = entry.metadata or {}
        lines = metadata.get("critical_speech_lines", []) or []
        if not isinstance(lines, list):
            return []
        clean_lines: List[str] = []
        for line in lines:
            text = str(line).strip()
            if text:
                clean_lines.append(text)
        return clean_lines

    def _entry_retrieval_text(self, entry: CaptionEntry) -> str:
        parts = [entry.text]

        if self._use_visual_summary_in_retrieval(entry.granularity) and entry.visual_summary:
            parts.append(f"Visual: {entry.visual_summary}")

        if self._use_critical_speech_in_retrieval(entry.granularity):
            critical_lines = self._get_critical_speech_lines(entry)
            if critical_lines:
                clipped = critical_lines[: self.max_critical_speech_lines_in_retrieval]
                parts.append("Critical speech: " + " | ".join(clipped))

        return "\n".join([p for p in parts if p])

    def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
        self.captions[granularity] = []
        self.doc_id_to_entry[granularity] = {}
        self.text_to_entries[granularity] = {}
        self.link_entity_index_built[granularity] = False
        self.link_entity_to_doc_ids[granularity] = {}

        for idx, entry in enumerate(data):
            doc_id = self._make_doc_id(entry, granularity, idx)
            caption_id = doc_id

            metadata = {
                "action_threads": entry.get("action_threads", entry.get("main_actions", [])),
                "object_threads": entry.get("object_threads", entry.get("salient_objects", [])),
                "topic_threads": entry.get("topic_threads", entry.get("conversation_focus", [])),
                "speaker_stats": entry.get("speaker_stats", entry.get("speakers", [])),
                "scene_summary": entry.get("scene_summary", {}),
                "visual_object_threads": entry.get("visual_object_threads", entry.get("visual_objects", [])),
                "critical_speech_lines": list(entry.get("critical_speech_lines", []) or []),
                "source_doc_ids": list(entry.get("source_doc_ids", []) or []),
                "child_ids": list(entry.get("child_ids", []) or []),
            }

            caption_entry = CaptionEntry(
                id=caption_id,
                doc_id=doc_id,
                text=entry.get("text", entry.get("fine_caption", "")),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=str(entry.get("date", "")),
                granularity=granularity,
                video_path=entry.get("video_path"),
                visual_summary=entry.get("visual_summary", ""),
                metadata=metadata,
            )

            self.captions[granularity].append(caption_entry)
            self.caption_id_to_entry[caption_id] = caption_entry
            self.doc_id_to_entry[granularity][doc_id] = caption_entry
            self.text_to_entries[granularity].setdefault(caption_entry.text, []).append(caption_entry)

    # -----------------------------------------------------
    # Sidecar loading
    # -----------------------------------------------------

    def load_sidecar_from_files(
        self,
        triplet_files: Optional[Dict[str, str]] = None,
        graph_files: Optional[Dict[str, str]] = None,
    ) -> None:
        if triplet_files:
            for granularity, file_path in triplet_files.items():
                if granularity not in self.granularities:
                    continue
                data = _load_json(file_path)
                self._process_triplet_data(data, granularity)

        if graph_files:
            for granularity, file_path in graph_files.items():
                if granularity not in self.granularities:
                    continue
                data = _load_json(file_path)
                self._process_graph_data(data, granularity)

    def load_sidecar_from_data(
        self,
        triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
        graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if triplet_data:
            for granularity, data in triplet_data.items():
                if granularity not in self.granularities:
                    continue
                self._process_triplet_data(data, granularity)

        if graph_data:
            for granularity, data in graph_data.items():
                if granularity not in self.granularities:
                    continue
                self._process_graph_data(data, granularity)

    def _process_triplet_data(self, data: Dict[str, Any], granularity: str) -> None:
        self.triplets_by_doc[granularity] = dict(data.get("triplet_map", {}))
        logger.info(f"Loaded sidecar triplets for {granularity}: {len(self.triplets_by_doc[granularity])} docs")
        self.link_entity_index_built[granularity] = False

    def _process_graph_data(self, data: Dict[str, Any], granularity: str) -> None:
        nodes = {node["id"]: node for node in data.get("nodes", [])}
        edges = data.get("edges", [])
        doc_id_to_event_id = data.get("doc_id_to_event_id", {})

        sidecar = {}
        for doc_id, event_id in doc_id_to_event_id.items():
            sidecar[doc_id] = GraphEventSidecar(
                event_id=event_id,
                doc_id=doc_id,
                granularity=granularity,
            )

        event_id_to_doc_id = {event_id: doc_id for doc_id, event_id in doc_id_to_event_id.items()}

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge["type"]
            event_id = edge.get("event_id")

            if edge_type == "before" and source in event_id_to_doc_id and target in event_id_to_doc_id:
                src_doc = event_id_to_doc_id[source]
                tgt_doc = event_id_to_doc_id[target]
                if src_doc in sidecar:
                    sidecar[src_doc].next_doc_id = tgt_doc
                if tgt_doc in sidecar:
                    sidecar[tgt_doc].prev_doc_id = src_doc
                continue

            if event_id not in event_id_to_doc_id:
                continue

            doc_id = event_id_to_doc_id[event_id]
            info = sidecar[doc_id]

            src_node = nodes.get(source, {})
            tgt_node = nodes.get(target, {})
            src_type = src_node.get("type")
            tgt_type = tgt_node.get("type")
            src_label = src_node.get("label", "")
            tgt_label = tgt_node.get("label", "")

            if source == event_id and tgt_type != "Event":
                if tgt_label:
                    info.entity_labels.append(tgt_label)
                if edge_type:
                    info.relation_types.append(edge_type)
                continue

            if src_type != "Event" and tgt_type != "Event":
                if src_label:
                    info.entity_labels.append(src_label)
                if tgt_label:
                    info.entity_labels.append(tgt_label)
                if edge_type:
                    info.relation_types.append(edge_type)
                if src_label and tgt_label and edge_type:
                    info.triplet_strings.append(f"{src_label} {edge_type} {tgt_label}")

        for doc_id, info in sidecar.items():
            event_node = nodes.get(info.event_id, {})
            token_source = []
            token_source.extend(info.entity_labels)
            token_source.extend(info.relation_types)
            token_source.extend(info.triplet_strings)
            if event_node.get("text"):
                token_source.append(event_node["text"])
            if event_node.get("visual_summary"):
                token_source.append(event_node["visual_summary"])
            if event_node.get("critical_speech_lines"):
                token_source.extend([str(x) for x in (event_node.get("critical_speech_lines") or [])])
            for field in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
                val = event_node.get(field, [])
                if isinstance(val, list):
                    token_source.extend([
                        json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
                        for x in val
                    ])
            token_source.append(json.dumps(event_node.get("scene_summary", {}), ensure_ascii=False))

            info.graph_tokens = set()
            for item in token_source:
                info.graph_tokens.update(_tokenize(item))

            info.entity_labels = sorted(set(info.entity_labels))
            info.relation_types = sorted(set(info.relation_types))
            info.triplet_strings = sorted(set(info.triplet_strings))

        self.graph_sidecar[granularity] = sidecar
        self.link_entity_index_built[granularity] = False
        logger.info(f"Loaded sidecar graph for {granularity}: {len(sidecar)} event nodes")

    # -----------------------------------------------------
    # Indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
            return

        for granularity in self.granularities:
            if not self.captions[granularity]:
                logger.warning(f"No captions loaded for granularity {granularity}")
                continue

            entries_to_index = [
                entry for entry in self.captions[granularity]
                if entry.timestamp_int[1] <= until_time
            ]
            if not entries_to_index:
                continue

            # IMPORTANT:
            # Only raw caption text is given to HippoRAG, so visual_summary / critical_speech_lines do NOT enter
            # HippoRAG OpenIE / fact graph construction. They are only used later for rerank / selector-style matching.
            caption_texts = [entry.text for entry in entries_to_index]
            hipporag = self._get_or_create_hipporag(granularity)
            if not self._load_cached_hipporag_for_retrieval(hipporag, len(caption_texts), granularity):
                hipporag.update(docs=caption_texts)
            self.indexed_entries[granularity] = entries_to_index
            logger.info(f"Indexed {len(entries_to_index)} captions for {granularity}")

        self.indexed_time = until_time

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _lookup_entry_from_text(self, granularity: str, text: str) -> Optional[CaptionEntry]:
        candidates = self.text_to_entries[granularity].get(text, [])
        if not candidates:
            return None
        return candidates[0]

    def _entry_retrieval_tokens(self, entry: CaptionEntry) -> set:
        return _tokenize(self._entry_retrieval_text(entry))

    def _entry_metadata_tokens(self, entry: CaptionEntry) -> set:
        toks = set()
        toks.update(_tokenize(entry.text))
        toks.update(_tokenize(entry.visual_summary))
        for line in self._get_critical_speech_lines(entry):
            toks.update(_tokenize(line))
        metadata = entry.metadata or {}
        for k in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
            val = metadata.get(k, [])
            if isinstance(val, list):
                for x in val:
                    toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
        scene_summary = metadata.get("scene_summary", {})
        if scene_summary:
            toks.update(_tokenize(json.dumps(scene_summary, ensure_ascii=False)))
        speaker_stats = metadata.get("speaker_stats", [])
        if isinstance(speaker_stats, list):
            for x in speaker_stats:
                toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
        return toks

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-8:
            return [1.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    def _overlap_score(self, query_tokens: set, candidate_tokens: set) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        inter = query_tokens & candidate_tokens
        return len(inter) / max(1, len(query_tokens))

    def _graph_aware_rerank(
        self,
        granularity: str,
        query: str,
        candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        base_scores = [score for _, score in candidates]
        base_scores = self._normalize_scores(base_scores)

        reranked = []
        for (entry, _), base_score in zip(candidates, base_scores):
            retrieval_text_score = self._overlap_score(query_tokens, self._entry_retrieval_tokens(entry))
            metadata_score = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))
            graph_score = 0.0
            sidecar = self.graph_sidecar[granularity].get(entry.doc_id)
            if sidecar:
                graph_score = self._overlap_score(query_tokens, sidecar.graph_tokens)
            final_score = (
                base_score
                + self.retrieval_text_score_weight * retrieval_text_score
                + self.graph_score_weight * graph_score
                + self.metadata_score_weight * metadata_score
            )
            reranked.append((entry, final_score))

        reranked.sort(key=lambda x: -x[1])
        return reranked

    def _collect_link_phrases(self, value: Any) -> List[str]:
        phrases: List[str] = []
        if value is None:
            return phrases
        if isinstance(value, str):
            norm = _normalize_phrase(value)
            if norm:
                phrases.append(norm)
            return phrases
        if isinstance(value, dict):
            for key in ["label", "name", "object", "entity", "item", "value", "mention", "text"]:
                if key in value:
                    phrases.extend(self._collect_link_phrases(value[key]))
            return phrases
        if isinstance(value, list):
            for item in value:
                phrases.extend(self._collect_link_phrases(item))
            return phrases
        return phrases

    def _get_link_entities_for_entry(self, entry: CaptionEntry, granularity: str) -> Set[str]:
        labels: Set[str] = set()

        sidecar = self.graph_sidecar.get(granularity, {}).get(entry.doc_id)
        if sidecar is not None:
            for label in sidecar.entity_labels:
                norm = _normalize_phrase(label)
                if norm:
                    labels.add(norm)
            for triplet_str in sidecar.triplet_strings:
                # heuristic split: "a relation b" -> keep whole phrase too noisy; rely on entity_labels first
                parts = [p.strip() for p in re.split(r"\s+(?:is|are|was|were|has|have|had|at|in|on|with|to|from|of)\s+", triplet_str, maxsplit=1)]
                for part in parts:
                    norm = _normalize_phrase(part)
                    if norm and len(norm.split()) <= 4:
                        labels.add(norm)

        metadata = entry.metadata or {}
        for field in ["object_threads", "visual_object_threads"]:
            labels.update(self._collect_link_phrases(metadata.get(field, [])))

        return {x for x in labels if x}

    def _ensure_link_entity_index(self, granularity: str) -> None:
        if self.link_entity_index_built.get(granularity, False):
            return

        index: Dict[str, Set[str]] = defaultdict(set)
        for entry in self.captions.get(granularity, []):
            for label in self._get_link_entities_for_entry(entry, granularity):
                index[label].add(entry.doc_id)

        self.link_entity_to_doc_ids[granularity] = dict(index)
        self.link_entity_index_built[granularity] = True
        logger.info(
            "Built object/entity link index for %s: %d labels",
            granularity,
            len(self.link_entity_to_doc_ids[granularity]),
        )

    def _expand_temporal_neighbors(
        self,
        granularity: str,
        ranked_candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not ranked_candidates:
            return []

        expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
        seeds = ranked_candidates[: max(1, self.graph_expand_top_n)]

        for entry, seed_score in seeds:
            current_doc_ids = [entry.doc_id]
            decay = float(seed_score)

            for _ in range(self.graph_expand_hops):
                next_doc_ids: List[str] = []
                decay *= self.graph_expand_decay
                for current_doc_id in current_doc_ids:
                    sidecar = self.graph_sidecar.get(granularity, {}).get(current_doc_id)
                    if sidecar is None:
                        continue
                    for neighbor_doc_id in [sidecar.prev_doc_id, sidecar.next_doc_id]:
                        if not neighbor_doc_id:
                            continue
                        neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
                        if neighbor_entry is None:
                            continue
                        if neighbor_entry.timestamp_int[1] > self.indexed_time:
                            continue
                        prev = expanded.get(neighbor_doc_id)
                        if prev is None or decay > prev[1]:
                            expanded[neighbor_doc_id] = (neighbor_entry, decay)
                        next_doc_ids.append(neighbor_doc_id)
                current_doc_ids = next_doc_ids
                if not current_doc_ids:
                    break

        return sorted(expanded.values(), key=lambda x: -x[1])

    def _expand_entity_neighbors(
        self,
        granularity: str,
        ranked_candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not ranked_candidates:
            return []

        self._ensure_link_entity_index(granularity)
        index = self.link_entity_to_doc_ids.get(granularity, {})
        if not index:
            return []

        expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
        seeds = ranked_candidates[: max(1, self.entity_expand_top_n)]

        for seed_entry, seed_score in seeds:
            seed_labels = self._get_link_entities_for_entry(seed_entry, granularity)
            if not seed_labels:
                continue

            candidate_overlap_counts: Dict[str, int] = defaultdict(int)
            for label in seed_labels:
                for neighbor_doc_id in index.get(label, set()):
                    if neighbor_doc_id == seed_entry.doc_id:
                        continue
                    candidate_overlap_counts[neighbor_doc_id] += 1

            seed_time = _timestamp_to_seconds(seed_entry.timestamp_int[0])
            scored_neighbors: List[Tuple[CaptionEntry, float]] = []

            for neighbor_doc_id, overlap_count in candidate_overlap_counts.items():
                neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
                if neighbor_entry is None:
                    continue
                if neighbor_entry.timestamp_int[1] > self.indexed_time:
                    continue

                neighbor_time = _timestamp_to_seconds(neighbor_entry.timestamp_int[0])
                time_gap = abs(neighbor_time - seed_time)
                temporal_proximity = 1.0 / (1.0 + time_gap / 300.0)  # 5-minute scale
                overlap_strength = min(overlap_count, 3) / 3.0

                score = float(seed_score) * self.entity_expand_decay * (0.55 + 0.45 * overlap_strength) * temporal_proximity
                scored_neighbors.append((neighbor_entry, score))

            scored_neighbors.sort(key=lambda x: -x[1])
            for neighbor_entry, score in scored_neighbors[: self.entity_expand_limit_per_seed]:
                prev = expanded.get(neighbor_entry.doc_id)
                if prev is None or score > prev[1]:
                    expanded[neighbor_entry.doc_id] = (neighbor_entry, score)

        return sorted(expanded.values(), key=lambda x: -x[1])

    def expand_entry_to_30s_doc_ids(self, entry: CaptionEntry) -> List[str]:
        if entry.granularity == "30sec":
            return [entry.doc_id]
        source_doc_ids = list(entry.metadata.get("source_doc_ids", []) or [])
        child_ids = list(entry.metadata.get("child_ids", []) or [])
        candidate_ids = source_doc_ids or child_ids
        if candidate_ids:
            return [str(x) for x in candidate_ids]
        return [entry.doc_id]

    def get_caption_by_doc_id(self, doc_id: str, granularity: Optional[str] = None) -> Optional[CaptionEntry]:
        if granularity is not None:
            return self.doc_id_to_entry.get(granularity, {}).get(doc_id)
        for g in self.granularities:
            if doc_id in self.doc_id_to_entry.get(g, {}):
                return self.doc_id_to_entry[g][doc_id]
        return None

    def get_triplets_by_doc_id(self, doc_id: str, granularity: str = "30sec") -> List[List[str]]:
        return self.triplets_by_doc.get(granularity, {}).get(doc_id, [])

    def get_parent_caption(self, doc_id: str, parent_granularity: str = "3min") -> Optional[CaptionEntry]:
        child_entry = self.get_caption_by_doc_id(doc_id, "30sec")
        if child_entry is None:
            return None
        if parent_granularity not in self.doc_id_to_entry:
            return None

        child_start, child_end = child_entry.timestamp_int
        best_parent: Optional[CaptionEntry] = None
        best_span: Optional[int] = None

        for parent in self.captions.get(parent_granularity, []):
            if parent.date != child_entry.date:
                continue
            parent_start, parent_end = parent.timestamp_int
            if parent_start <= child_start and parent_end >= child_end:
                span = parent_end - parent_start
                if best_parent is None or best_span is None or span < best_span:
                    best_parent = parent
                    best_span = span

        return best_parent

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve_captions_as_str(self, entries: List[CaptionEntry], include_visual_summary: bool = True) -> str:
        return "\n\n".join(entry.to_display_str(include_visual_summary=include_visual_summary) for entry in entries)

    def retrieve_ranked(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = None,
        dedup_by_doc_id: bool = True,
    ) -> List[Tuple[CaptionEntry, float]]:
        self.last_retrieval_debug = {
            "query": query,
            "fallback_used": False,
            "granularities": {},
            "cache_warnings": list(self.last_retrieval_debug.get("cache_warnings", [])),
            "fallback_events": [],
        }
        if top_k_per_granularity is None:
            top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

        if self.indexed_time == 0:
            logger.warning("No captions indexed. Call index(until_time) before retrieve().")
            return []

        all_candidates: List[Tuple[CaptionEntry, float]] = []
        for granularity in self.granularities:
            if granularity not in self.hipporag:
                continue

            if isinstance(top_k_per_granularity, dict):
                granularity_top_k = top_k_per_granularity.get(granularity, 5)
            else:
                granularity_top_k = top_k_per_granularity

            hipporag = self.hipporag[granularity]
            gran_debug = self.last_retrieval_debug["granularities"].setdefault(
                granularity,
                {
                    "indexed_entries": len(self.indexed_entries.get(granularity, [])),
                    "hipporag_docs": 0,
                    "matched_candidates": 0,
                    "unmatched_docs": 0,
                },
            )
            try:
                retrieval_result = hipporag.retrieve(
                    queries=[query],
                    num_to_retrieve=granularity_top_k * 2,
                )
            except Exception as exc:
                logger.warning("HippoRAG retrieval failed for %s; using loaded-caption fallback: %s", granularity, exc)
                gran_debug["error"] = f"{type(exc).__name__}: {exc}"
                fallback = self._caption_fallback_rank(query, granularity, granularity_top_k, "hipporag_retrieval_failed")
                if fallback:
                    self.last_retrieval_debug["fallback_used"] = True
                    all_candidates.extend(fallback)
                continue
            if not retrieval_result or not retrieval_result[0].docs:
                fallback = self._caption_fallback_rank(query, granularity, granularity_top_k, "hipporag_returned_no_docs")
                if fallback:
                    self.last_retrieval_debug["fallback_used"] = True
                    all_candidates.extend(fallback)
                continue

            retrieved_docs = retrieval_result[0].docs
            gran_debug["hipporag_docs"] = len(retrieved_docs)
            retrieved_scores = (
                retrieval_result[0].doc_scores
                if hasattr(retrieval_result[0], 'doc_scores')
                else [1.0] * len(retrieved_docs)
            )

            raw_candidates = []
            unmatched_docs = 0
            for doc_text, score in zip(retrieved_docs, retrieved_scores):
                entry = self._lookup_entry_from_text(granularity, doc_text)
                if entry is None:
                    unmatched_docs += 1
                    logger.warning(
                        f"Could not find CaptionEntry for retrieved text in {granularity}: {doc_text[:50]}..."
                    )
                    continue
                raw_candidates.append((entry, float(score)))
            gran_debug["matched_candidates"] = len(raw_candidates)
            gran_debug["unmatched_docs"] = unmatched_docs
            if not raw_candidates:
                fallback = self._caption_fallback_rank(query, granularity, granularity_top_k, "hipporag_docs_unmatched_active_captions")
                if fallback:
                    self.last_retrieval_debug["fallback_used"] = True
                    all_candidates.extend(fallback)
                continue

            reranked_candidates = self._graph_aware_rerank(
                granularity=granularity,
                query=query,
                candidates=raw_candidates,
            )

            selected_base = reranked_candidates[:granularity_top_k]
            expanded_temporal = self._expand_temporal_neighbors(granularity, selected_base)
            expanded_entity = self._expand_entity_neighbors(granularity, selected_base)

            logger.info(
                "Episodic %s: base=%d temporal_expanded=%d entity_expanded=%d",
                granularity,
                len(selected_base),
                len(expanded_temporal),
                len(expanded_entity),
            )

            all_candidates.extend(selected_base)
            all_candidates.extend(expanded_temporal)
            all_candidates.extend(expanded_entity)

        if dedup_by_doc_id:
            best_by_doc: Dict[str, Tuple[CaptionEntry, float]] = {}
            for entry, score in all_candidates:
                prev = best_by_doc.get(entry.doc_id)
                if prev is None or score > prev[1]:
                    best_by_doc[entry.doc_id] = (entry, score)
            all_candidates = list(best_by_doc.values())

        all_candidates.sort(key=lambda x: -x[1])
        return all_candidates

    def retrieve(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = None,
        final_top_k: int = 3,
        as_context: bool = True,
    ) -> Union[List[CaptionEntry], str]:
        if top_k_per_granularity is None:
            top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

        ranked = self.retrieve_ranked(
            query=query,
            top_k_per_granularity=top_k_per_granularity,
            dedup_by_doc_id=True,
        )
        if not ranked:
            return [] if not as_context else ""

        result_entries = [entry for entry, _ in ranked[:final_top_k]]
        if as_context:
            return self.retrieve_captions_as_str(result_entries, include_visual_summary=True)
        return result_entries

    def reset_index(self) -> None:
        self.hipporag.clear()
        for g in self.granularities:
            self.indexed_entries[g] = []
            self.link_entity_index_built[g] = False
            self.link_entity_to_doc_ids[g] = {}
        self.indexed_time = 0
        logger.info("Index reset - HippoRAG instances and indexed entries cleared")

    def get_indexed_time(self) -> str:
        return _transform_timestamp(str(self.indexed_time))

    def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
        return self.caption_id_to_entry.get(caption_id)
