# """
# WorldMemory: Unified memory system integrating episodic, semantic, and visual memories
# with iterative reasoning for long-term video reasoning.
# """

# import copy
# import json
# import logging
# import re
# from typing import Any, Dict, List, Optional, Set, Tuple
# from PIL import Image

# from ..llm import LLMModel, PromptTemplateManager
# from ..embedding import EmbeddingModel

# from .episodic import EpisodicMemory, CaptionEntry
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .visual import VisualMemory
# from .utils import *

# logger = logging.getLogger(__name__)


# class WorldMemory:
#     """
#     Unified memory system for WorldMM that integrates episodic, semantic, 
#     and visual memories with iterative reasoning.
    
#     The system implements a multi-round retrieval process:
#     1. Given a query, the reasoning agent decides whether to search or answer
#     2. If searching, it selects a memory type and forms a search query
#     3. Retrieved context is accumulated across rounds
#     4. When the agent decides to answer, the QA model uses all accumulated context
    
#     Memory Types:
#     - Episodic: Specific events/actions using HippoRAG for retrieval
#     - Semantic: Entity/relationship knowledge using PPR graph retrieval  
#     - Visual: Scene/setting snapshots using embedding similarity
    
#     Attributes:
#         episodic_memory: EpisodicMemory instance
#         semantic_memory: SemanticMemory instance
#         visual_memory: VisualMemory instance
#         retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
#         respond_llm_model: LLM for iterative reasoning and generating answers
#         prompt_template_manager: Manager for prompt templates
#         max_rounds: Maximum retrieval rounds
#         max_errors: Maximum errors before forcing answer
#     """
    
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#         retriever_llm_model: LLMModel,
#         respond_llm_model: Optional[LLMModel] = None,
#         prompt_template_manager: Optional[PromptTemplateManager] = None,
#         episodic_granularities: Optional[List[str]] = None,
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         """
#         Initialize WorldMemory with all memory subsystems.
        
#         Args:
#             embedding_model: Embedding model for all memory types
#             retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
#             respond_llm_model: LLM for iterative reasoning and generating answers (defaults to retriever_llm_model)
#             prompt_template_manager: Manager for prompt templates (creates default if None)
#             episodic_granularities: Granularity levels for episodic memory
#             max_rounds: Maximum retrieval rounds before forcing answer
#             max_errors: Maximum errors before forcing answer
#         """
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors
        
#         # Initialize memory subsystems
#         self.episodic_memory = EpisodicMemory(
#             embedding_model=embedding_model,
#             llm_model=retriever_llm_model,
#             prompt_template_manager=self.prompt_template_manager,
#             granularities=episodic_granularities,
#         )
        
#         self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
        
#         self.visual_memory = VisualMemory(embedding_model=embedding_model)
        
#         # Track indexed time
#         self.indexed_time: int = 0
        
#         # Retrieval configuration
#         self.episodic_top_k: int = 3
#         self.semantic_top_k: int = 10
#         self.visual_top_k: int = 3
        
#     def load_episodic_captions(
#         self,
#         caption_files: Optional[Dict[str, str]] = None,
#         caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
#     ) -> None:
#         """
#         Load episodic captions from files or data.
        
#         Args:
#             caption_files: Dict mapping granularity -> JSON file path
#             caption_data: Dict mapping granularity -> list of caption dicts
#         """
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
#         """
#         Load semantic triples from file or data.
        
#         Args:
#             file_path: Path to JSON file with semantic triples
#             data: In-memory dict with semantic triples
#         """
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
#         """
#         Load visual clips and embeddings.
        
#         Args:
#             embeddings_path: Path to pickle file with precomputed embeddings
#             clips_path: Path to JSON file with clip metadata
#             clips_data: In-memory list of clip metadata dicts
#         """
#         if embeddings_path:
#             self.visual_memory.load_embeddings_from_file(embeddings_path)
#         if clips_path:
#             self.visual_memory.load_clips_from_file(clips_path)
#         if clips_data:
#             self.visual_memory.load_clips_from_data(clips_data)
    
#     def index(self, until_time: int) -> None:
#         """
#         Index all memory types up to the specified timestamp.
        
#         This should be called before any retrieval to ensure memories
#         are indexed up to the query time.
        
#         Args:
#             until_time: Timestamp in integer format (day + time.zfill(8))
#         """
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
#             return
        
#         logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")
        
#         # Index each memory type
#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
        
#         self.indexed_time = until_time
#         logger.info(f"Indexing complete for all memory types")
    
#     def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
#         """
#         Parse the reasoning agent's JSON response.
        
#         Args:
#             response: JSON string from the reasoning LLM
            
#         Returns:
#             ReasoningOutput with decision and optional memory selection
#         """
#         try:
#             # Try to extract JSON from response
#             json_match = re.search(r'\{.*\}', response, re.DOTALL)
#             if json_match:
#                 data = json.loads(json_match.group())
#             else:
#                 data = json.loads(response)
            
#             decision = data.get("decision", "answer").lower()
#             reason = data.get("reason")
            
#             selected_memory = None
#             if decision == "search" and "selected_memory" in data:
#                 mem_data = data["selected_memory"]
#                 selected_memory = MemorySearchOutput(
#                     memory_type=mem_data.get("memory_type", "").lower(),
#                     search_query=mem_data.get("search_query", ""),
#                 )
            
#             return ReasoningOutput(
#                 decision=decision,
#                 selected_memory=selected_memory,
#                 reason=reason,
#             )
            
#         except (json.JSONDecodeError, KeyError, TypeError) as e:
#             logger.warning(f"Failed to parse reasoning response: {e}")
#             # Default to answer if parsing fails
#             return ReasoningOutput(decision="answer")
    
#     def _format_round_history(self, rounds: List[Dict[str, Any]]) -> str:
#         """
#         Format the round history for the reasoning prompt.
        
#         Args:
#             rounds: List of round information dicts
            
#         Returns:
#             Formatted string for the prompt
#         """
#         if not rounds:
#             return "[]"
        
#         lines = []
#         for r in rounds:
#             round_str = f"""### Round {r['round_num']}
# Decision: {r['decision']}
# Memory: {r['memory_type']}
# Search Query: {r['search_query']}
# Retrieved:
# {r['retrieved_content']}"""
#             lines.append(round_str)
        
#         return "\n\n".join(lines)
    
#     def _render_retrieved_items_for_qa(
#         self, 
#         retrieved_items: List[RetrievedItem]
#     ) -> List[Dict[str, Any]]:
#         """
#         Render retrieved items for the QA prompt.
        
#         Args:
#             retrieved_items: List of RetrievedItem objects
            
#         Returns:
#             List of message content dicts for the LLM
#         """
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
    
#     def retrieve_from_episodic(
#         self, 
#         query: str, 
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#     ) -> Tuple[str, Set[str]]:
#         """
#         Retrieve from episodic memory.
        
#         Args:
#             query: Search query
#             top_k: Number of results to retrieve
#             retrieved_set: Set of already retrieved items to avoid duplicates
            
#         Returns:
#             Tuple of (formatted content string, updated retrieved set)
#         """
#         top_k = top_k or self.episodic_top_k
#         retrieved_set = retrieved_set or set()
        
#         # Retrieve from episodic memory
#         result = self.episodic_memory.retrieve(
#             query=query,
#             final_top_k=top_k * 2,  # Get extra to filter duplicates
#             as_context=False,
#         )
        
#         if not result:
#             return "", retrieved_set
        
#         # Result is List[CaptionEntry] when as_context=False
#         if isinstance(result, str):
#             return result, retrieved_set
        
#         # Filter out already retrieved items
#         new_items: List[CaptionEntry] = []
#         for entry in result:
#             if entry.text not in retrieved_set:
#                 new_items.append(entry)
#                 retrieved_set.add(entry.text)
#             if len(new_items) >= top_k:
#                 break
        
#         # Format as context string
#         content = self.episodic_memory.retrieve_captions_as_str(new_items)
#         return content, retrieved_set
    
#     def retrieve_from_semantic(
#         self,
#         query: str,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#     ) -> Tuple[str, Set[str]]:
#         """
#         Retrieve from semantic memory.
        
#         Args:
#             query: Search query
#             top_k: Number of results to retrieve
#             retrieved_set: Set of already retrieved items to avoid duplicates
            
#         Returns:
#             Tuple of (formatted content string, updated retrieved set)
#         """
#         top_k = top_k or self.semantic_top_k
#         retrieved_set = retrieved_set or set()
        
#         # Retrieve from semantic memory
#         result = self.semantic_memory.retrieve(
#             query=query,
#             top_k=top_k * 2,  # Get extra to filter duplicates
#             as_context=False,
#         )
        
#         if not result:
#             return "", retrieved_set
        
#         # Result is List[SemanticTripleEntry] when as_context=False
#         if isinstance(result, str):
#             return result, retrieved_set
        
#         # Filter out already retrieved items
#         new_items: List[SemanticTripleEntry] = []
#         for entry in result:
#             if entry.id not in retrieved_set:
#                 new_items.append(entry)
#                 retrieved_set.add(entry.id)
#             if len(new_items) >= top_k:
#                 break
        
#         # Format as context string
#         content = self.semantic_memory.retrieve_triples_as_str(new_items)
#         return content, retrieved_set
    
#     def retrieve_from_visual(
#         self,
#         query: str,
#         top_k: Optional[int] = None,
#         retrieved_set: Optional[Set[str]] = None,
#     ) -> Tuple[Dict[str, List[Any]], Set[str]]:
#         """
#         Retrieve from visual memory.
        
#         Args:
#             query: Search query (text or time range)
#             top_k: Number of clips to retrieve
#             retrieved_set: Set of already retrieved items to avoid duplicates
            
#         Returns:
#             Tuple of (content dict with images, updated retrieved set)
#         """
#         top_k = top_k or self.visual_top_k
#         retrieved_set = retrieved_set or set()
        
#         # Retrieve from visual memory
#         result = self.visual_memory.retrieve(
#             query=query,
#             top_k=top_k,
#             as_context=True,
#         )
        
#         if not result:
#             return {}, retrieved_set
        
#         # Result should be Dict[str, List[Image]] when as_context=True
#         if isinstance(result, dict):
#             # Track retrieved clips by their display keys
#             for key in result.keys():
#                 retrieved_set.add(key)
#             return result, retrieved_set
        
#         # Fallback for unexpected return type
#         return {}, retrieved_set
    
#     def answer(
#         self,
#         query: str,
#         choices: Optional[Dict[str, str]] = None,
#         until_time: Optional[int] = None,
#     ) -> QAResult:
#         """
#         Answer a question using iterative memory retrieval.
        
#         This is the main entry point for the WorldMM pipeline:
#         1. Index memories up to the query time
#         2. Iteratively retrieve from memories based on reasoning
#         3. Answer the question using accumulated context
        
#         Args:
#             query: The question to answer
#             choices: Optional dict of answer choices (e.g., {"A": "...", "B": "..."})
#             until_time: Timestamp to index up to (uses current indexed time if None)
            
#         Returns:
#             QAResult with the answer and retrieval history
#         """
#         # Index if needed
#         if until_time and until_time > self.indexed_time:
#             self.index(until_time)
        
#         # Format query with choices if provided
#         full_query = f"Query: {query}"
#         if choices:
#             choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
#             full_query += f"\nChoices: {choices_str}"
        
#         # Initialize retrieval state
#         retrieved_set: Set[str] = set()
#         retrieved_items: List[RetrievedItem] = []
#         round_history: List[Dict[str, Any]] = []
        
#         # Get reasoning prompt template
#         reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
        
#         round_num = 0
#         err_count = 0
        
#         while round_num < self.max_rounds and err_count < self.max_errors:
#             round_num += 1
#             logger.info(f"Reasoning round {round_num}")
            
#             # Build the user message for reasoning
#             history_str = self._format_round_history(round_history)
            
#             user_content = f"""{full_query}

# Round History:
# {history_str}

# Task:
# Step 1: Decide whether to "search" or "answer".
# Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."""
            
#             # Get reasoning decision
#             reasoning_messages = copy.deepcopy(reasoning_prompt)
#             reasoning_messages.append({
#                 "role": "user",
#                 "content": user_content,
#             })
            
#             try:
#                 response = self.respond_llm_model.generate(reasoning_messages)
#                 reasoning_output = self._parse_reasoning_response(response)
#             except Exception as e:
#                 logger.error(f"Reasoning failed: {e}")
#                 err_count += 1
#                 continue
            
#             logger.info(f"Decision: {reasoning_output.decision}")
            
#             # Handle decision
#             if reasoning_output.decision == "answer":
#                 break
            
#             if reasoning_output.decision == "search":
#                 if not reasoning_output.selected_memory:
#                     logger.warning("Search decision but no memory selected")
#                     err_count += 1
#                     continue
                
#                 memory_type = reasoning_output.selected_memory.memory_type
#                 search_query = reasoning_output.selected_memory.search_query
                
#                 logger.info(f"Searching {memory_type}: {search_query}")
                
#                 # Retrieve from selected memory
#                 content = ""
#                 images = None
                
#                 if memory_type == "episodic":
#                     content, retrieved_set = self.retrieve_from_episodic(
#                         search_query, 
#                         retrieved_set=retrieved_set
#                     )
                    
#                 elif memory_type == "semantic":
#                     content, retrieved_set = self.retrieve_from_semantic(
#                         search_query,
#                         retrieved_set=retrieved_set
#                     )
                    
#                 elif memory_type == "visual":
#                     images, retrieved_set = self.retrieve_from_visual(
#                         search_query,
#                         retrieved_set=retrieved_set
#                     )
#                     # Format visual content for round history
#                     if images:
#                         content = f"[{len(sum(images.values(), []))} images from {len(images)} clips]"
#                         # Flatten images for retrieved items
#                         all_images = []
#                         for clip_images in images.values():
#                             all_images.extend(clip_images)
#                         retrieved_items.append(RetrievedItem(
#                             memory_type="visual",
#                             content=all_images,
#                             query=search_query,
#                             round_num=round_num,
#                         ))
#                 else:
#                     logger.warning(f"Unknown memory type: {memory_type}")
#                     err_count += 1
#                     continue
                
#                 # Add to retrieved items (text memories)
#                 if memory_type in ("episodic", "semantic") and content:
#                     retrieved_items.append(RetrievedItem(
#                         memory_type=memory_type,
#                         content=content,
#                         query=search_query,
#                         round_num=round_num,
#                     ))
                
#                 # Add to round history
#                 round_history.append({
#                     "round_num": round_num,
#                     "decision": "search",
#                     "memory_type": memory_type,
#                     "search_query": search_query,
#                     "retrieved_content": content if content else "[No results]",
#                 })
        
#         # Generate final answer
#         logger.info("Generating answer from accumulated context")
        
#         try:
#             qa_prompt = self.prompt_template_manager.render("qa_egolife")
#         except Exception as e:
#             logger.error(f"Failed to load qa_egolife template: {e}")
#             raise
        
#         # Build QA message with all retrieved context
#         qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
#         qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
        
#         if choices:
#             qa_content.append({
#                 "type": "text", 
#                 "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
#             })
        
#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({
#             "role": "user",
#             "content": qa_content,
#         })
        
#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error(f"Answer generation failed: {e}")
#             answer = "Unable to generate answer"
#             print(f"qa prompt: {qa_messages}")
        
#         return QAResult(
#             question=query,
#             answer=answer,
#             retrieved_items=retrieved_items,
#             round_history=round_history,
#             num_rounds=round_num,
#         )
    
#     def reset_index(self) -> None:
#         """Reset all indexed state across all memory types."""
#         self.episodic_memory.reset_index()
#         self.semantic_memory.reset_index()
#         self.visual_memory.reset_index()
#         self.indexed_time = 0
#         logger.info("All memory indices reset")
    
#     def cleanup(self) -> None:
#         """Release GPU memory and other resources."""
#         self.semantic_memory.cleanup()
#         self.visual_memory.cleanup()
#         logger.info("Memory cleanup complete")
    
#     def get_indexed_time(self) -> str:
#         """Get the current indexed time as human-readable string."""
#         return transform_timestamp(str(self.indexed_time))
    
#     def set_retrieval_top_k(
#         self,
#         episodic: Optional[int] = None,
#         semantic: Optional[int] = None,
#         visual: Optional[int] = None,
#     ) -> None:
#         """
#         Configure the number of items to retrieve from each memory type.
        
#         Args:
#             episodic: Top-k for episodic memory
#             semantic: Top-k for semantic memory
#             visual: Top-k for visual memory
#         """
#         if episodic is not None:
#             self.episodic_top_k = episodic
#         if semantic is not None:
#             self.semantic_top_k = semantic
#         if visual is not None:
#             self.visual_top_k = visual


# """
# WorldMemory: unified event-centric memory system.

# This version implements a lightweight retrieval pipeline:
# - multiscale episodic retrieval with HippoRAG + graph-aware rerank
# - lightweight temporal graph expansion over episodic candidates
# - visual-summary-aware initial retrieval for 30sec / 3min episodic layers
# - semantic retrieval used only as semantic support for final episodic anchors
# - visual evidence only for final top event anchors (keyframes), no global visual recall
# """

# import copy
# import logging
# import math
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple
# from PIL import Image

# from ..llm import LLMModel, PromptTemplateManager
# from ..embedding import EmbeddingModel

# from .episodic import EpisodicMemory, CaptionEntry
# from .semantic import SemanticMemory, SemanticTripleEntry
# from .visual import VisualMemory
# from .utils import *

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
#         max_rounds: int = 5,
#         max_errors: int = 5,
#     ):
#         self.embedding_model = embedding_model
#         self.retriever_llm_model = retriever_llm_model
#         self.respond_llm_model = respond_llm_model or retriever_llm_model
#         self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
#         self.max_rounds = max_rounds
#         self.max_errors = max_errors

#         self.episodic_memory = EpisodicMemory(
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

#         self.anchor_weight_30s = 1.00
#         self.anchor_weight_3min = 0.65
#         self.anchor_weight_10min = 0.45
#         self.anchor_weight_1h = 0.30

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

#     # -----------------------------------------------------
#     # indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
#             return

#         logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")
#         self.episodic_memory.index(until_time)
#         self.semantic_memory.index(until_time)
#         self.visual_memory.index(until_time)
#         self.indexed_time = until_time
#         logger.info("Indexing complete for all memory types")

#     # -----------------------------------------------------
#     # scoring helpers
#     # -----------------------------------------------------

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

#     def _project_episodic_candidates_to_30s(
#         self,
#         candidates: List[Tuple[CaptionEntry, float]],
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
#     ) -> Dict[str, List[SemanticTripleEntry]]:
#         support_map: Dict[str, List[SemanticTripleEntry]] = defaultdict(list)
#         if not semantic_entries:
#             return support_map

#         for entry in semantic_entries:
#             if not entry.evidence_event_ids:
#                 continue
#             valid_doc_ids = [
#                 doc_id for doc_id in entry.evidence_event_ids
#                 if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None
#             ]
#             for doc_id in valid_doc_ids:
#                 support_map[doc_id].append(entry)

#         return support_map

#     def _collect_supporting_semantic_facts(
#         self,
#         top_doc_ids: List[str],
#         semantic_support_map: Dict[str, List[SemanticTripleEntry]],
#         max_facts: int,
#     ) -> List[SemanticTripleEntry]:
#         collected: List[SemanticTripleEntry] = []
#         seen = set()
#         for doc_id in top_doc_ids:
#             for fact in semantic_support_map.get(doc_id, []):
#                 if fact.id in seen:
#                     continue
#                 seen.add(fact.id)
#                 collected.append(fact)
#                 if len(collected) >= max_facts:
#                     return collected
#         return collected

#     def _build_semantic_context(self, semantic_entries: List[SemanticTripleEntry], top_n: int = 5) -> str:
#         if not semantic_entries:
#             return ""
#         lines = ["Semantic Support Facts:"]
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

#         parent_3min = self.episodic_memory.get_parent_caption(doc_id, "3min")
#         visual_entry = self.visual_memory.get_clip_by_doc_id(doc_id)
#         triplets = self.episodic_memory.get_triplets_by_doc_id(doc_id, "30sec")
#         supporting_facts = supporting_facts or []

#         lines = []
#         lines.append(f"Event Anchor: {doc_id}")
#         lines.append(f"Relevance Score: {score:.4f}")
#         lines.append("30s Evidence:")
#         lines.append(entry.to_display_str(include_visual_summary=True))

#         if parent_3min is not None and parent_3min.doc_id != doc_id:
#             lines.append("3min Context:")
#             lines.append(parent_3min.to_display_str(include_visual_summary=True))

#         if visual_entry is not None:
#             if visual_entry.keyframe_caption:
#                 lines.append(f"Keyframe Caption: {visual_entry.keyframe_caption}")
#             if visual_entry.visual_objects:
#                 lines.append("Visual Objects: " + ", ".join(visual_entry.visual_objects[:8]))
#             if visual_entry.scene_summary:
#                 dominant_scene = visual_entry.scene_summary.get("dominant_scene", "")
#                 if dominant_scene:
#                     lines.append(f"Scene: {dominant_scene}")

#         if triplets:
#             lines.append("Episodic Triplets:")
#             for tri in triplets[:6]:
#                 if len(tri) == 3:
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
#             "memory_type": "episodic+semantic_support",
#             "search_query": query,
#             "retrieved_content": (
#                 f"Top events: {top_doc_ids}\n"
#                 f"Supporting semantic facts: {[e.id for e in semantic_entries[:5]]}"
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

#         # 1) parallel episodic retrieval + semantic support retrieval
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
#                 ]
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
#                 ]
#             )

#         # 2) episodic-only ranking on 30s anchors; semantic only used as support
#         episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
#         semantic_support_map = self._project_semantic_to_30s(semantic_entries)

#         candidate_doc_ids = set(episodic_projected.keys())
#         if not candidate_doc_ids:
#             logger.warning("No candidate events found from episodic retrieval")
#             candidate_doc_ids = set()
#             for entry, _ in episodic_ranked[:self.episodic_top_k]:
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

#         episodic_norm = self._normalize_dict({
#             doc_id: episodic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids
#         })

#         ranked_doc_ids = [
#             doc_id for doc_id, _ in sorted(episodic_norm.items(), key=lambda x: -x[1])
#         ]
#         top_doc_ids = ranked_doc_ids[:max(self.episodic_top_k, 1)]

#         logger.info(
#             "Top episodic event anchors: %s",
#             [
#                 {
#                     "doc_id": doc_id,
#                     "ep": round(episodic_norm.get(doc_id, 0.0), 4),
#                     "semantic_support_count": len(semantic_support_map.get(doc_id, [])),
#                 }
#                 for doc_id in top_doc_ids
#             ]
#         )

#         # 3) build event packets
#         event_packets = []
#         for doc_id in top_doc_ids:
#             packet = self._build_event_packet(
#                 doc_id=doc_id,
#                 score=episodic_norm.get(doc_id, 0.0),
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

#         selected_support_facts = self._collect_supporting_semantic_facts(
#             top_doc_ids=top_doc_ids,
#             semantic_support_map=semantic_support_map,
#             max_facts=self.semantic_top_k,
#         )
#         semantic_context = self._build_semantic_context(
#             selected_support_facts,
#             top_n=min(5, len(selected_support_facts)),
#         )

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

#         # 4) final visual evidence only for selected event anchors
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

#         round_history = self._build_round_history(query, top_doc_ids, selected_support_facts)

#         # 5) generate answer
#         try:
#             qa_prompt = self.prompt_template_manager.render("qa_egolife")
#         except Exception as e:
#             logger.error(f"Failed to load qa_egolife template: {e}")
#             raise

#         qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
#         qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
#         if choices:
#             qa_content.append({
#                 "type": "text",
#                 "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
#             })

#         num_text_blocks = sum(
#             1 for x in qa_content
#             if isinstance(x, dict) and x.get("type") == "text"
#         )
#         num_image_blocks = sum(
#             1 for x in qa_content
#             if isinstance(x, dict) and x.get("type") == "image"
#         )

#         logger.info(
#             "QA payload prepared: %d text blocks, %d image blocks, %d retrieved items",
#             num_text_blocks,
#             num_image_blocks,
#             len(retrieved_items),
#         )

#         qa_messages = copy.deepcopy(qa_prompt)
#         qa_messages.append({
#             "role": "user",
#             "content": qa_content,
#         })

#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error(f"Answer generation failed: {e}")
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
# WorldMemory: unified event-centric memory system.

# This version implements:
# - multiscale episodic retrieval with HippoRAG + graph-aware rerank
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
# from typing import Any, Dict, List, Optional, Set, Tuple

# from PIL import Image

# from ..embedding import EmbeddingModel
# from ..llm import LLMModel, PromptTemplateManager
# from .episodic import CaptionEntry, EpisodicMemory
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

#         self.episodic_memory = EpisodicMemory(
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
#         self.selector_global_top_n = 10
#         self.selector_trigger_top_n = 4
#         self.selector_antecedent_top_n = 4
#         self.selector_broader_top_n = 3
#         self.selector_max_candidates = 12

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

#     # -----------------------------------------------------
#     # indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         if self.indexed_time >= until_time:
#             logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
#             return

#         logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")
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

#     def _entry_center_seconds(self, entry: CaptionEntry) -> float:
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

#         parent = None
#         if hasattr(self.episodic_memory, "get_parent_caption"):
#             parent = self.episodic_memory.get_parent_caption(doc_id, "3min")
#         if parent is not None:
#             toks.update(self._tokenize(parent.text))
#             toks.update(self._tokenize(parent.visual_summary))

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
#         candidates: List[Tuple[CaptionEntry, float]],
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
#             if not entry.evidence_event_ids:
#                 continue
#             rank_bonus = 1.0 / (rank + 1)
#             support_factor = 1.0 + 0.15 * min(int(entry.support_count), 5)
#             conf_factor = 0.7 + 0.3 * float(entry.confidence)
#             base = rank_bonus * support_factor * conf_factor

#             valid_doc_ids = [
#                 doc_id for doc_id in entry.evidence_event_ids
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
#         seed_centers = {doc_id: self._entry_center_seconds(self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec"))
#                         for doc_id in seed_doc_ids
#                         if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None}

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

#             # trigger: high episodic score + near the main cluster + query alignment
#             if trigger_centroid > 0.0:
#                 delta = abs(center_sec - trigger_centroid)
#                 temporal_proximity = math.exp(-delta / 600.0)  # ~10min decay
#             else:
#                 temporal_proximity = 0.0
#             trigger_score = 0.60 * ep_score + 0.20 * query_overlap + 0.20 * temporal_proximity

#             # antecedent: earlier than trigger seeds + object/entity continuity + support presence
#             earlierness = 0.0
#             if earliest_seed is not None and center_sec < earliest_seed:
#                 gap = earliest_seed - center_sec
#                 earlierness = min(1.0, gap / 1800.0)  # saturate at 30min earlier
#             support_presence = 1.0 if semantic_support_map.get(doc_id) else 0.0
#             antecedent_score = 0.45 * earlierness + 0.35 * seed_overlap + 0.20 * support_presence

#             # broader-context: belongs to a dense 3min parent and still relevant to query
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

#         # normalize each role across candidates
#         for role_name in ["trigger", "antecedent", "broader", "semantic"]:
#             normed = self._normalize_dict({k: v[role_name] for k, v in role_scores.items()})
#             for doc_id in role_scores:
#                 role_scores[doc_id][role_name] = normed.get(doc_id, 0.0)

#         return role_scores

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
#         for idx, doc_id in enumerate(ordered_doc_ids, start=1):
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

#             candidates.append({
#                 "index": idx,
#                 "doc_id": doc_id,
#                 "start_time": transform_timestamp(str(entry.timestamp_int[0])),
#                 "end_time": transform_timestamp(str(entry.timestamp_int[1])),
#                 "caption": entry.text,
#                 "visual_summary": entry.visual_summary,
#                 "episodic_score": round(float(episodic_norm.get(doc_id, 0.0)), 4),
#                 "semantic_score": round(float(semantic_norm.get(doc_id, 0.0)), 4),
#                 "trigger_score": round(float(role_scores[doc_id].get("trigger", 0.0)), 4),
#                 "antecedent_score": round(float(role_scores[doc_id].get("antecedent", 0.0)), 4),
#                 "broader_score": round(float(role_scores[doc_id].get("broader", 0.0)), 4),
#                 "primary_role": primary_role,
#                 "triplets": triplets,
#                 "parent_3min_doc_id": parent.doc_id if parent is not None else None,
#                 "parent_3min_caption": parent.text if parent is not None else "",
#                 "semantic_support": [fact.to_display_str() for fact in support_facts],
#             })

#         return candidates

#     def _parse_event_selector_response(
#         self,
#         response: str,
#         valid_doc_ids: List[str],
#         num_candidates: int,
#     ) -> List[str]:
#         valid_doc_id_set = set(valid_doc_ids)
#         selected: List[str] = []

#         # 1) Try JSON object / array first
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

#         # 2) Regex doc_ids
#         if not selected:
#             for doc_id in re.findall(r"DAY\d_[0-9]{8}_[0-9]{8}(?:_[A-Za-z0-9]+)?", response):
#                 if doc_id in valid_doc_id_set and doc_id not in selected:
#                     selected.append(doc_id)

#         # 3) Regex indices if still empty
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

#         # prompt = [
#         #     {
#         #         "role": "system",
#         #         "content": (
#         #             "You are selecting event packets for a long-video QA system.\n"
#         #             "Your job is NOT to choose events that are merely topically related. "
#         #             "Your job is to choose events whose evidence matches the exact predicate asked by the question.\n\n"

#         #             "Step 1: infer the question family from the question.\n"
#         #             "Use one of these families:\n"
#         #             "1) action-owner\n"
#         #             "2) source-trace\n"
#         #             "3) participant-membership\n"
#         #             "4) plan-intention-decision\n"
#         #             "5) temporal-recall\n"
#         #             "6) habit-preference\n"
#         #             "7) attribute-content-purpose\n\n"

#         #             "Step 2: choose a small, complementary set of event packets that best supports the answer.\n"
#         #             "Prefer explicit evidence over weak implication.\n"
#         #             "Do not over-select near-duplicate local events.\n"
#         #             "Always return valid candidate indices and/or valid doc_ids from the provided list only.\n\n"

#         #             "Question-family rules:\n\n"

#         #             "[action-owner]\n"
#         #             "- Prefer events with an explicit actor performing the queried action.\n"
#         #             "- For 'helped', prefer explicit cooperation / transfer / assistance evidence.\n"
#         #             "- For 'first', prefer the earliest valid explicit action, not later related scenes.\n"
#         #             "- Nearby presence, related objects, or later results are weaker than an explicit action.\n\n"

#         #             "[source-trace]\n"
#         #             "- Prefer evidence that explicitly shows where the object came from, where it was before, or how it was transferred.\n"
#         #             "- Source-location evidence is stronger than generic earlier context.\n"
#         #             "- Current location does not answer previous location.\n"
#         #             "- Prefer carry / retrieve / bring / take-from / upstairs-downstairs chains.\n\n"

#         #             "[participant-membership]\n"
#         #             "- Prefer events that explicitly show who joined, helped, or was present in the activity.\n"
#         #             "- Do not infer participation only from later co-presence in the same room.\n"
#         #             "- Distinguish core participants from bystanders.\n\n"

#         #             "[plan-intention-decision]\n"
#         #             "- Prefer explicit plans, proposals, assignments, intentions, or final decisions.\n"
#         #             "- Do NOT infer intention from related discussion, observation, or topic proximity.\n"
#         #             "- 'talking about flowers' is not the same as 'plans to grow flowers'.\n"
#         #             "- 'I will watch it grow' is weaker than 'I plan to grow it' or 'I bought/planned it for growing'.\n\n"

#         #             "[temporal-recall]\n"
#         #             "- Respect 'last time', 'first time', 'before', and relative temporal constraints strictly.\n"
#         #             "- Prefer the closest valid earlier occurrence that truly matches the queried event/topic.\n"
#         #             "- A semantically similar event at the wrong time is not sufficient.\n\n"

#         #             "[habit-preference]\n"
#         #             "- Prefer repeated or aggregate evidence across multiple events.\n"
#         #             "- For 'usually', 'always', 'most', 'likes', 'doesn't like', repeated evidence or explicit preference statements are stronger than one-off actions.\n\n"

#         #             "[attribute-content-purpose]\n"
#         #             "- Prefer direct evidence about ownership, contents, identity, or purpose.\n"
#         #             "- Do not replace a direct attribute question with nearby action context.\n\n"

#         #             "Global anti-error rules:\n"
#         #             "- Do not infer agent ownership from scene participation alone.\n"
#         #             "- Do not infer intention from topic discussion alone.\n"
#         #             "- Do not infer source from current location alone.\n"
#         #             "- Prefer explicit predicate-aligned evidence over broad contextual relevance.\n"
#         #             "- Use role scores as hints, not hard constraints."
#         #         ),
#         #     },
#         #     {
#         #         "role": "user",
#         #         "content": (
#         #             f"{query_with_time}\n\n"
#         #             f"Candidate Event Packets:\n{json.dumps(selector_candidates, ensure_ascii=False, indent=2)}\n\n"
#         #             f"Select the best {final_top_k} candidates.\n\n"
#         #             "Selection goals:\n"
#         #             "- Choose complementary evidence, not repetitive evidence.\n"
#         #             "- If the question is about a source / previous location, retain the source-establishing event even if it is temporally earlier and less salient.\n"
#         #             "- If the question is about a plan / decision, retain explicit intention / proposal / final-decision evidence, not just related discussion.\n"
#         #             "- If the question is about who did something, retain explicit actor evidence.\n"
#         #             "- If the question is about last time / first time, enforce the temporal constraint strictly.\n\n"
#         #             "Return ONLY JSON in this format:\n"
#         #             '{'
#         #             '"question_family": "...", '
#         #             '"selected_indices": [..], '
#         #             '"selected_doc_ids": [..], '
#         #             '"reason": "..."'
#         #             '}'
#         #         ),
#         #     },
#         # ]

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

#                     "Question-family rules:\n\n"

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

#         try:
#             response = self.respond_llm_model.generate(prompt)
#             logger.info("LLM event selector raw response: %s", response)
#         except Exception as e:
#             logger.error(f"LLM event selector failed: {e}")
#             return [], ""

#         valid_doc_ids = [c["doc_id"] for c in selector_candidates]
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

#         # 1) parallel episodic + semantic retrieval
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

#         # 2) project all evidence to 30s anchors
#         episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
#         semantic_projected, semantic_support_map = self._project_semantic_to_30s(semantic_entries)

#         # semantic is support only: event-anchor selection is driven by episodic anchors
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

#         # 3) soft-group selector pool + LLM event selector
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

#         # 4) build event packets
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

#         # 5) final visual evidence only for selected event anchors
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

#         # 6) generate answer
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

#         try:
#             answer = self.respond_llm_model.generate(qa_messages)
#         except Exception as e:
#             logger.error(f"Answer generation failed: {e}")
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



"""
WorldMemory: unified event-centric memory system.

This version implements:
- multiscale episodic retrieval with HippoRAG + graph-aware rerank
- soft-grouped event candidate pool (trigger / antecedent / broader-context)
- LLM event selector over a coarse candidate pool
- semantic memory as support only (not primary event routing)
- visual evidence only for final selected event anchors (keyframes)
"""

import copy
import json
import logging
import math
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from ..embedding import EmbeddingModel
from ..llm import LLMModel, PromptTemplateManager
from .episodic import CaptionEntry, EpisodicMemory
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
    """Convert MST/WorldMM structured evidence into compact prompt-safe text."""
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


def _format_exception_traceback(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}", traceback.format_exc()]
    last_attempt = getattr(exc, "last_attempt", None)
    if last_attempt is not None:
        try:
            inner = last_attempt.exception()
        except Exception:
            inner = None
        if inner is not None and inner is not exc:
            parts.append(f"last_attempt_exception={type(inner).__name__}: {inner}")
            inner_tb = getattr(inner, "__traceback__", None)
            if inner_tb is not None:
                parts.append("".join(traceback.format_exception(type(inner), inner, inner_tb)))
    if getattr(exc, "__cause__", None) is not None:
        cause = exc.__cause__
        parts.append(f"cause={type(cause).__name__}: {cause}")
    if getattr(exc, "__context__", None) is not None:
        context = exc.__context__
        parts.append(f"context={type(context).__name__}: {context}")
    return "\n".join(part for part in parts if part).strip()


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

        self.episodic_memory = EpisodicMemory(
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

    # -----------------------------------------------------
    # indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
            return

        logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")
        self.episodic_memory.index(until_time)
        skip_semantic_index = os.getenv("WORLDMM_QUERY_SKIP_REINDEX", "0").strip().lower() in {"1", "true", "yes", "on"}
        if skip_semantic_index:
            logger.info("Skipping semantic embedding index in query strict load-only mode")
            try:
                self.semantic_memory.indexed_time = until_time
            except Exception:
                pass
        else:
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

    def _entry_center_seconds(self, entry: CaptionEntry) -> float:
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
        candidates: List[Tuple[CaptionEntry, float]],
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
        seed_centers = {doc_id: self._entry_center_seconds(self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec"))
                        for doc_id in seed_doc_ids
                        if self.episodic_memory.get_caption_by_doc_id(doc_id, "30sec") is not None}

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

            # trigger: high episodic score + near the main cluster + query alignment
            if trigger_centroid > 0.0:
                delta = abs(center_sec - trigger_centroid)
                temporal_proximity = math.exp(-delta / 600.0)  # ~10min decay
            else:
                temporal_proximity = 0.0
            trigger_score = 0.60 * ep_score + 0.20 * query_overlap + 0.20 * temporal_proximity

            # antecedent: earlier than trigger seeds + object/entity continuity + support presence
            earlierness = 0.0
            if earliest_seed is not None and center_sec < earliest_seed:
                gap = earliest_seed - center_sec
                earlierness = min(1.0, gap / 1800.0)  # saturate at 30min earlier
            support_presence = 1.0 if semantic_support_map.get(doc_id) else 0.0
            antecedent_score = 0.45 * earlierness + 0.35 * seed_overlap + 0.20 * support_presence

            # broader-context: belongs to a dense 3min parent and still relevant to query
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

        # normalize each role across candidates
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

        # 1) Try JSON object / array first
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

        # 2) Regex doc_ids
        if not selected:
            for doc_id in re.findall(r"DAY\d_[0-9]{8}_[0-9]{8}(?:_[A-Za-z0-9]+)?", response):
                if doc_id in valid_doc_id_set and doc_id not in selected:
                    selected.append(doc_id)

        # 3) Regex indices if still empty
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

        # prompt = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are selecting event packets for a long-video QA system.\n"
        #             "Your job is NOT to choose events that are merely topically related. "
        #             "Your job is to choose events whose evidence matches the exact predicate asked by the question.\n\n"

        #             "Step 1: infer the question family from the question.\n"
        #             "Use one of these families:\n"
        #             "1) action-owner\n"
        #             "2) source-trace\n"
        #             "3) participant-membership\n"
        #             "4) plan-intention-decision\n"
        #             "5) temporal-recall\n"
        #             "6) habit-preference\n"
        #             "7) attribute-content-purpose\n\n"

        #             "Step 2: choose a small, complementary set of event packets that best supports the answer.\n"
        #             "Prefer explicit evidence over weak implication.\n"
        #             "Do not over-select near-duplicate local events.\n"
        #             "Always return valid candidate indices and/or valid doc_ids from the provided list only.\n\n"

        #             "Question-family rules:\n\n"

        #             "[action-owner]\n"
        #             "- Prefer events with an explicit actor performing the queried action.\n"
        #             "- For 'helped', prefer explicit cooperation / transfer / assistance evidence.\n"
        #             "- For 'first', prefer the earliest valid explicit action, not later related scenes.\n"
        #             "- Nearby presence, related objects, or later results are weaker than an explicit action.\n\n"

        #             "[source-trace]\n"
        #             "- Prefer evidence that explicitly shows where the object came from, where it was before, or how it was transferred.\n"
        #             "- Source-location evidence is stronger than generic earlier context.\n"
        #             "- Current location does not answer previous location.\n"
        #             "- Prefer carry / retrieve / bring / take-from / upstairs-downstairs chains.\n\n"

        #             "[participant-membership]\n"
        #             "- Prefer events that explicitly show who joined, helped, or was present in the activity.\n"
        #             "- Do not infer participation only from later co-presence in the same room.\n"
        #             "- Distinguish core participants from bystanders.\n\n"

        #             "[plan-intention-decision]\n"
        #             "- Prefer explicit plans, proposals, assignments, intentions, or final decisions.\n"
        #             "- Do NOT infer intention from related discussion, observation, or topic proximity.\n"
        #             "- 'talking about flowers' is not the same as 'plans to grow flowers'.\n"
        #             "- 'I will watch it grow' is weaker than 'I plan to grow it' or 'I bought/planned it for growing'.\n\n"

        #             "[temporal-recall]\n"
        #             "- Respect 'last time', 'first time', 'before', and relative temporal constraints strictly.\n"
        #             "- Prefer the closest valid earlier occurrence that truly matches the queried event/topic.\n"
        #             "- A semantically similar event at the wrong time is not sufficient.\n\n"

        #             "[habit-preference]\n"
        #             "- Prefer repeated or aggregate evidence across multiple events.\n"
        #             "- For 'usually', 'always', 'most', 'likes', 'doesn't like', repeated evidence or explicit preference statements are stronger than one-off actions.\n\n"

        #             "[attribute-content-purpose]\n"
        #             "- Prefer direct evidence about ownership, contents, identity, or purpose.\n"
        #             "- Do not replace a direct attribute question with nearby action context.\n\n"

        #             "Global anti-error rules:\n"
        #             "- Do not infer agent ownership from scene participation alone.\n"
        #             "- Do not infer intention from topic discussion alone.\n"
        #             "- Do not infer source from current location alone.\n"
        #             "- Prefer explicit predicate-aligned evidence over broad contextual relevance.\n"
        #             "- Use role scores as hints, not hard constraints."
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"{query_with_time}\n\n"
        #             f"Candidate Event Packets:\n{json.dumps(selector_candidates, ensure_ascii=False, indent=2)}\n\n"
        #             f"Select the best {final_top_k} candidates.\n\n"
        #             "Selection goals:\n"
        #             "- Choose complementary evidence, not repetitive evidence.\n"
        #             "- If the question is about a source / previous location, retain the source-establishing event even if it is temporally earlier and less salient.\n"
        #             "- If the question is about a plan / decision, retain explicit intention / proposal / final-decision evidence, not just related discussion.\n"
        #             "- If the question is about who did something, retain explicit actor evidence.\n"
        #             "- If the question is about last time / first time, enforce the temporal constraint strictly.\n\n"
        #             "Return ONLY JSON in this format:\n"
        #             '{'
        #             '"question_family": "...", '
        #             '"selected_indices": [..], '
        #             '"selected_doc_ids": [..], '
        #             '"reason": "..."'
        #             '}'
        #         ),
        #     },
        # ]

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

                    "Question-family rules:\n\n"

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

        # 1) parallel episodic + semantic retrieval
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

        # 2) project all evidence to 30s anchors
        episodic_projected = self._project_episodic_candidates_to_30s(episodic_ranked)
        semantic_projected, semantic_support_map = self._project_semantic_to_30s(semantic_entries)

        # semantic is support only: event-anchor selection is driven by episodic anchors
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

        episodic_norm = self._normalize_dict({doc_id: episodic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})
        semantic_norm = self._normalize_dict({doc_id: semantic_projected.get(doc_id, 0.0) for doc_id in candidate_doc_ids})

        anchor_scores: Dict[str, float] = {doc_id: episodic_norm.get(doc_id, 0.0) for doc_id in candidate_doc_ids}
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

        # 3) soft-group selector pool + LLM event selector
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

        selected_doc_ids, selector_reason = self._select_top_events_with_llm(
            query=query,
            choices=choices,
            until_time=until_time,
            selector_candidates=selector_candidates,
            final_top_k=max(self.episodic_top_k, 1),
        )

        if not selected_doc_ids:
            logger.info("LLM event selector returned no valid doc_ids, fallback to coarse ranking")
            top_doc_ids = ranked_doc_ids[: max(self.episodic_top_k, 1)]
            selector_reason = (
                "Selector fallback: no valid doc_ids were parsed from the selector output. "
                "Coarse episodic ranking was used instead."
            )
        else:
            top_doc_ids = []
            for doc_id in selected_doc_ids:
                if doc_id not in top_doc_ids:
                    top_doc_ids.append(doc_id)
            if len(top_doc_ids) < max(self.episodic_top_k, 1):
                for doc_id in ranked_doc_ids:
                    if doc_id not in top_doc_ids:
                        top_doc_ids.append(doc_id)
                    if len(top_doc_ids) >= max(self.episodic_top_k, 1):
                        break

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

        # 4) build event packets
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

        semantic_context = self._build_semantic_context(semantic_entries, top_n=min(5, self.semantic_top_k))

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
        if semantic_context:
            retrieved_items.append(
                RetrievedItem(
                    memory_type="semantic",
                    content=semantic_context,
                    query=query,
                    round_num=1,
                )
            )

        # 5) optional final visual evidence only for the top selected event anchor.
        # Keep this gated because loading/base64-encoding images is expensive for
        # the online text-only query path.
        event_images = {}
        image_paths_used: List[str] = []
        if use_image_evidence:
            max_image_frames = max(0, int(max_image_frames or 0))
            event_images = self.visual_memory.get_event_images(
                top_doc_ids[:1],
                max_images_per_event=max_image_frames,
                total_max_images=max_image_frames,
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
                image_paths_used = [
                    str(getattr(img, "filename", ""))
                    for img in all_images
                    if getattr(img, "filename", "")
                ]
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
            if use_image_evidence:
                logger.info("No visual evidence found for final event anchors")
            else:
                logger.info("Image evidence disabled for this query")

        round_history = self._build_round_history(query, top_doc_ids, semantic_entries)

        # 6) generate answer
        qa_template_name = "qa_egolife" if effective_answer_mode == "multiple_choice" else "qa_egolife_open"
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

        model_response_text = ""
        error_debug = ""
        fallback_used = False
        llm_debug: Dict[str, Any] = {}
        answer_generation_start = time.perf_counter()
        try:
            chunks: list[str] = []

            def _on_chunk(text: str) -> None:
                if not text:
                    return
                chunks.append(text)
                if callable(stream_handler):
                    try:
                        stream_handler(
                            {
                                "type": "delta",
                                "stage": "answer",
                                "delta": text,
                            }
                        )
                    except Exception:
                        pass

            answer_raw = self.respond_llm_model.stream_generate(qa_messages, on_chunk=_on_chunk) if callable(stream_handler) else self.respond_llm_model.generate(qa_messages)
            llm_debug = dict(getattr(self.respond_llm_model, "last_debug", {}) or {})
            llm_debug["answer_generation_ms"] = int(round((time.perf_counter() - answer_generation_start) * 1000))
            model_response_text = "" if answer_raw is None else str(answer_raw).strip()
            if not model_response_text:
                raise RuntimeError("Answer generation returned empty text")
            if callable(stream_handler):
                try:
                    stream_handler(
                        {
                            "type": "final",
                            "stage": "answer",
                            "text": model_response_text,
                            "answer": model_response_text,
                        }
                    )
                except Exception:
                    pass
            answer = model_response_text
        except Exception as e:
            primary_answer_generation_ms = int(round((time.perf_counter() - answer_generation_start) * 1000))
            traceback_text = _format_exception_traceback(e)
            error_debug = f"image_or_primary_answer_error={type(e).__name__}: {e}\n{traceback_text}"
            llm_debug["traceback"] = traceback_text
            llm_debug["primary_answer_generation_ms"] = primary_answer_generation_ms
            logger.error(f"Answer generation failed: {e}")
            if use_image_evidence and num_image_blocks > 0:
                logger.warning("Image-mode answer failed; retrying text-only QA fallback")
                text_only_qa_content = [
                    block
                    for block in qa_content
                    if not (isinstance(block, dict) and block.get("type") == "image")
                ]
                text_only_messages = copy.deepcopy(qa_prompt)
                text_only_messages.append({"role": "user", "content": text_only_qa_content})
                fallback_generation_start = time.perf_counter()
                try:
                    fallback_raw = self.respond_llm_model.generate(text_only_messages)
                    fallback_debug = dict(getattr(self.respond_llm_model, "last_debug", {}) or {})
                    fallback_debug["answer_generation_ms"] = int(round((time.perf_counter() - fallback_generation_start) * 1000))
                    fallback_debug["primary_answer_generation_ms"] = primary_answer_generation_ms
                    model_response_text = "" if fallback_raw is None else str(fallback_raw).strip()
                    if not model_response_text:
                        raise RuntimeError("Text-only fallback returned empty text")
                    answer = model_response_text
                    fallback_used = True
                    if llm_debug:
                        fallback_debug["primary_image_debug"] = llm_debug
                    llm_debug = fallback_debug
                except Exception as fallback_e:
                    fallback_generation_ms = int(round((time.perf_counter() - fallback_generation_start) * 1000))
                    fallback_traceback = _format_exception_traceback(fallback_e)
                    fallback_debug = dict(getattr(self.respond_llm_model, "last_debug", {}) or {})
                    fallback_debug["answer_generation_ms"] = fallback_generation_ms
                    if fallback_debug:
                        llm_debug["text_only_fallback_debug"] = fallback_debug
                    llm_debug["text_only_fallback_traceback"] = fallback_traceback
                    llm_debug["text_only_fallback_generation_ms"] = fallback_generation_ms
                    error_debug += f"\ntext_only_fallback_error={type(fallback_e).__name__}: {fallback_e}\n{fallback_traceback}"
                    logger.error(f"Text-only fallback answer generation failed: {fallback_e}")
                    answer = "Unable to generate answer"
            else:
                answer = "Unable to generate answer"

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
            image_evidence_enabled=use_image_evidence,
            image_paths_used=image_paths_used,
            model_response_text=model_response_text,
            error_debug=error_debug,
            fallback_used=fallback_used,
            llm_debug=llm_debug,
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
