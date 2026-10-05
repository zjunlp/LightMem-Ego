# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import copy
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

from ..llm import LLMModel, PromptTemplateManager
from ..embedding import EmbeddingModel

from .multimodal_memory_cell import MemoryCell_videomme as MemoryCell, CaptionEntry_videomme as CaptionEntry
from .semantic_graph import SemanticMemory_videomme as SemanticMemory, SemanticTripleEntry_videomme as SemanticTripleEntry
from .visual import VisualMemory_videomme as VisualMemory
from .utils import *

logger = logging.getLogger(__name__)


class EM2Memory_videomme:
    """
    Unified memory system for EM2Memory that integrates episodic, semantic, 
    and visual memories with iterative reasoning.
    
    The system implements a multi-round retrieval process:
    1. Given a query, the reasoning agent decides whether to search or answer
    2. If searching, it selects a memory type and forms a search query
    3. Retrieved context is accumulated across rounds
    4. When the agent decides to answer, the QA model uses all accumulated context
    
    Memory Types:
    - Episodic: Specific events/actions for retrieval
    - Semantic: Entity/relationship knowledge using PPR graph retrieval  
    - Visual: Scene/setting snapshots using embedding similarity
    
    Attributes:
        episodic_memory: MemoryCell instance
        semantic_memory: SemanticMemory instance
        visual_memory: VisualMemory instance
        retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
        respond_llm_model: LLM for iterative reasoning and generating answers
        prompt_template_manager: Manager for prompt templates
        max_rounds: Maximum retrieval rounds
        max_errors: Maximum errors before forcing answer
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever_llm_model: LLMModel,
        respond_llm_model: Optional[LLMModel] = None,
        prompt_template_manager: Optional[PromptTemplateManager] = None,
        episodic_granularities: Optional[List[str]] = None,
        qa_template_name: str = "qa_egolife",
        max_rounds: int = 5,
        max_errors: int = 5,
    ):
        """
        Initialize EM2Memory with all memory subsystems.
        
        Args:
            embedding_model: Embedding model for all memory types
            retriever_llm_model: LLM for retrieval operations (NER, OpenIE)
            respond_llm_model: LLM for iterative reasoning and generating answers (defaults to retriever_llm_model)
            prompt_template_manager: Manager for prompt templates (creates default if None)
            episodic_granularities: Granularity levels for episodic memory
            max_rounds: Maximum retrieval rounds before forcing answer
            max_errors: Maximum errors before forcing answer
        """
        self.embedding_model = embedding_model
        self.retriever_llm_model = retriever_llm_model
        self.respond_llm_model = respond_llm_model or retriever_llm_model
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.qa_template_name = qa_template_name
        
        # Initialize memory subsystems
        self.episodic_memory = MemoryCell(
            embedding_model=embedding_model,
            granularities=episodic_granularities,
        )
        
        self.semantic_memory = SemanticMemory(embedding_model=embedding_model)
        
        self.visual_memory = VisualMemory(embedding_model=embedding_model)
        
        # Track indexed time
        self.indexed_time: int = 0
        
        # Retrieval configuration
        self.episodic_top_k: int = 3
        self.semantic_top_k: int = 10
        self.visual_top_k: int = 3
        
    def load_episodic_captions(
        self,
        caption_files: Optional[Dict[str, str]] = None,
        caption_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Load episodic captions from files or data.
        
        Args:
            caption_files: Dict mapping granularity -> JSON file path
            caption_data: Dict mapping granularity -> list of caption dicts
        """
        if caption_files:
            self.episodic_memory.load_captions_from_files(caption_files)
        if caption_data:
            self.episodic_memory.load_captions_from_data(caption_data)
    
    def load_semantic_triples(
        self,
        file_path: Optional[str] = None,
        data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Load semantic triples from file or data.
        
        Args:
            file_path: Path to JSON file with semantic triples
            data: In-memory dict with semantic triples
        """
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
        """
        Load visual clips and embeddings.
        
        Args:
            embeddings_path: Path to pickle file with precomputed embeddings
            clips_path: Path to JSON file with clip metadata
            clips_data: In-memory list of clip metadata dicts
        """
        if embeddings_path:
            self.visual_memory.load_embeddings_from_file(embeddings_path)
        if clips_path:
            self.visual_memory.load_clips_from_file(clips_path)
        if clips_data:
            self.visual_memory.load_clips_from_data(clips_data)
    
    def index(self, until_time: int) -> None:
        """
        Index all memory types up to the specified timestamp.
        
        This should be called before any retrieval to ensure memories
        are indexed up to the query time.
        
        Args:
            until_time: Timestamp in integer format (day + time.zfill(8))
        """
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping")
            return
        
        logger.info(f"Indexing all memories up to {transform_timestamp(str(until_time))}")
        
        # Index each memory type
        self.episodic_memory.index(until_time)
        self.semantic_memory.index(until_time)
        self.visual_memory.index(until_time)
        
        self.indexed_time = until_time
        logger.info(f"Indexing complete for all memory types")
    
    def _parse_reasoning_response(self, response: str) -> ReasoningOutput:
        """
        Parse the reasoning agent's JSON response.
        
        Args:
            response: JSON string from the reasoning LLM
            
        Returns:
            ReasoningOutput with decision and optional memory selection
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            decision = data.get("decision", "answer").lower()
            reason = data.get("reason")
            
            selected_memory = None
            if decision == "search" and "selected_memory" in data:
                mem_data = data["selected_memory"]
                selected_memory = MemorySearchOutput(
                    memory_type=mem_data.get("memory_type", "").lower(),
                    search_query=mem_data.get("search_query", ""),
                )
            
            return ReasoningOutput(
                decision=decision,
                selected_memory=selected_memory,
                reason=reason,
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            # Default to answer if parsing fails
            return ReasoningOutput(decision="answer")
    
    def _format_round_history(self, rounds: List[Dict[str, Any]]) -> str:
        """
        Format the round history for the reasoning prompt.
        
        Args:
            rounds: List of round information dicts
            
        Returns:
            Formatted string for the prompt
        """
        if not rounds:
            return "[]"
        
        lines = []
        for r in rounds:
            round_str = f"""### Round {r['round_num']}
Decision: {r['decision']}
Memory: {r['memory_type']}
Search Query: {r['search_query']}
Retrieved:
{r['retrieved_content']}"""
            lines.append(round_str)
        
        return "\n\n".join(lines)
    
    def _render_retrieved_items_for_qa(
        self, 
        retrieved_items: List[RetrievedItem],
        max_images: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Render retrieved items for the QA prompt, limiting the total number of images.
        
        Args:
            retrieved_items: List of RetrievedItem objects
            max_images: Maximum number of images to include in the output.
            
        Returns:
            List of message content dicts for the LLM
        """
        messages = []
        image_count = 0
        for item in retrieved_items:
            if item.memory_type in ("episodic", "semantic"):
                messages.append({"type": "text", "text": item.content})
            elif item.memory_type == "visual":
                if isinstance(item.content, list):
                    for img in item.content:
                        if image_count >= max_images:
                            logger.warning(f"Reached max images ({max_images}), skipping remaining images.")
                            break
                        if isinstance(img, Image.Image):
                            messages.append({"type": "image", "image": img})
                        elif isinstance(img, dict) and "image" in img:
                            messages.append({"type": "image", "image": img["image"]})
                        image_count += 1
                elif image_count < max_images:
                    if isinstance(item.content, Image.Image):
                        messages.append({"type": "image", "image": item.content})
                        image_count += 1
                    elif isinstance(item.content, dict) and "image" in item.content:
                        messages.append({"type": "image", "image": item.content["image"]})
                        image_count += 1
        return messages
    
    def retrieve_from_episodic(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[str, Set[str]]:
        """
        Retrieve from episodic memory.

        Modified:
        - duplicate filtering uses entry.id instead of entry.text
        - larger candidate pool for dense retrieval
        """

        top_k = top_k or self.episodic_top_k
        retrieved_set = retrieved_set or set()

        result = self.episodic_memory.retrieve(
            query=query,
            final_top_k=top_k*2,
            as_context=False,
        )

        if not result:
            return "", retrieved_set

        if isinstance(result, str):
            return result, retrieved_set

        new_items: List[CaptionEntry] = []

        for entry in result:
            if entry.id not in retrieved_set:
                new_items.append(entry)
                retrieved_set.add(entry.id)

            if len(new_items) >= top_k:
                break

        content = self.episodic_memory.retrieve_captions_as_str(
            new_items
        )

        return content, retrieved_set
    
    def retrieve_from_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[str, Set[str]]:
        """
        Retrieve from semantic memory.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            retrieved_set: Set of already retrieved items to avoid duplicates
            
        Returns:
            Tuple of (formatted content string, updated retrieved set)
        """
        top_k = top_k or self.semantic_top_k
        retrieved_set = retrieved_set or set()
        
        # Retrieve from semantic memory
        result = self.semantic_memory.retrieve(
            query=query,
            top_k=top_k * 2,  # Get extra to filter duplicates
            as_context=False,
        )
        
        if not result:
            return "", retrieved_set
        
        # Result is List[SemanticTripleEntry] when as_context=False
        if isinstance(result, str):
            return result, retrieved_set
        
        # Filter out already retrieved items
        new_items: List[SemanticTripleEntry] = []
        for entry in result:
            if entry.id not in retrieved_set:
                new_items.append(entry)
                retrieved_set.add(entry.id)
            if len(new_items) >= top_k:
                break
        
        # Format as context string
        content = self.semantic_memory.retrieve_triples_as_str(new_items)
        return content, retrieved_set
    
    def retrieve_from_visual(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieved_set: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, List[Any]], Set[str]]:
        """
        Retrieve from visual memory.
        
        Args:
            query: Search query (text or time range)
            top_k: Number of clips to retrieve
            retrieved_set: Set of already retrieved items to avoid duplicates
            
        Returns:
            Tuple of (content dict with images, updated retrieved set)
        """
        top_k = top_k or self.visual_top_k
        retrieved_set = retrieved_set or set()
        
        # Retrieve from visual memory
        result = self.visual_memory.retrieve(
            query=query,
            top_k=top_k,
            as_context=True,
        )
        
        if not result:
            return {}, retrieved_set
        
        # Result should be Dict[str, List[Image]] when as_context=True
        if isinstance(result, dict):
            # Track retrieved clips by their display keys
            for key in result.keys():
                retrieved_set.add(key)
            return result, retrieved_set
        
        # Fallback for unexpected return type
        return {}, retrieved_set
    
    def answer(
        self,
        query: str,
        choices: Optional[Dict[str, str]] = None,
        until_time: Optional[int] = None,
    ) -> QAResult:
        """
        Answer a question using iterative memory retrieval.
        
        This is the main entry point for the EM2Memory pipeline:
        1. Index memories up to the query time
        2. Iteratively retrieve from memories based on reasoning
        3. Answer the question using accumulated context
        
        Args:
            query: The question to answer
            choices: Optional dict of answer choices (e.g., {"A": "...", "B": "..."})
            until_time: Timestamp to index up to (uses current indexed time if None)
            
        Returns:
            QAResult with the answer and retrieval history
        """
        # Index if needed
        if until_time and until_time > self.indexed_time:
            self.index(until_time)
        
        # Format query with choices if provided
        full_query = f"Query: {query}"
        if choices:
            choices_str = " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
            full_query += f"\nChoices: {choices_str}"
        
        # Initialize retrieval state
        retrieved_set: Set[str] = set()
        retrieved_items: List[RetrievedItem] = []
        round_history: List[Dict[str, Any]] = []
        
        # Get reasoning prompt template
        reasoning_prompt = self.prompt_template_manager.render("memory_reasoning")
        
        round_num = 0
        err_count = 0
        
        while round_num < self.max_rounds and err_count < self.max_errors:
            round_num += 1
            logger.info(f"Reasoning round {round_num}")
            
            # Build the user message for reasoning
            history_str = self._format_round_history(round_history)
            
            user_content = f"""{full_query}

Round History:
{history_str}

Task:
Step 1: Decide whether to "search" or "answer".
Step 2 (only if search): Pick one memory type (episodic/semantic/visual) and form a search query."""
            
            # Get reasoning decision
            reasoning_messages = copy.deepcopy(reasoning_prompt)
            reasoning_messages.append({
                "role": "user",
                "content": user_content,
            })
            
            try:
                response = self.respond_llm_model.generate(reasoning_messages)
                reasoning_output = self._parse_reasoning_response(response)
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                err_count += 1
                continue
            
            logger.info(f"Decision: {reasoning_output.decision}")
            
            # Handle decision
            if reasoning_output.decision == "answer":
                break
            
            if reasoning_output.decision == "search":
                if not reasoning_output.selected_memory:
                    logger.warning("Search decision but no memory selected")
                    err_count += 1
                    continue
                
                memory_type = reasoning_output.selected_memory.memory_type
                search_query = reasoning_output.selected_memory.search_query
                
                logger.info(f"Searching {memory_type}: {search_query}")
                
                # Retrieve from selected memory
                content = ""
                images = None
                
                if memory_type == "episodic":
                    content, retrieved_set = self.retrieve_from_episodic(
                        search_query, 
                        retrieved_set=retrieved_set
                    )
                    
                elif memory_type == "semantic":
                    content, retrieved_set = self.retrieve_from_semantic(
                        search_query,
                        retrieved_set=retrieved_set
                    )
                    
                elif memory_type == "visual":
                    images, retrieved_set = self.retrieve_from_visual(
                        search_query,
                        retrieved_set=retrieved_set
                    )
                    # Format visual content for round history
                    if images:
                        content = f"[{len(sum(images.values(), []))} images from {len(images)} clips]"
                        # Flatten images for retrieved items
                        all_images = []
                        for clip_images in images.values():
                            all_images.extend(clip_images)
                        retrieved_items.append(RetrievedItem(
                            memory_type="visual",
                            content=all_images,
                            query=search_query,
                            round_num=round_num,
                        ))
                else:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    err_count += 1
                    continue
                
                # Add to retrieved items (text memories)
                if memory_type in ("episodic", "semantic") and content:
                    retrieved_items.append(RetrievedItem(
                        memory_type=memory_type,
                        content=content,
                        query=search_query,
                        round_num=round_num,
                    ))
                
                # Add to round history
                round_history.append({
                    "round_num": round_num,
                    "decision": "search",
                    "memory_type": memory_type,
                    "search_query": search_query,
                    "retrieved_content": content if content else "[No results]",
                })
        
        # Generate final answer
        logger.info("Generating answer from accumulated context")
        
        try:
            qa_prompt = self.prompt_template_manager.render(self.qa_template_name)
        except Exception as e:
            logger.error(f"Failed to load {self.qa_template_name} template: {e}")
            raise
        
        # Build QA message with all retrieved context
        qa_content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
        qa_content.extend(self._render_retrieved_items_for_qa(retrieved_items))
        
        if choices:
            qa_content.append({
                "type": "text", 
                "text": "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."
            })
        
        qa_messages = copy.deepcopy(qa_prompt)
        qa_messages.append({
            "role": "user",
            "content": qa_content,
        })
        
        try:
            answer = self.respond_llm_model.generate(qa_messages)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "Unable to generate answer"
        
        return QAResult(
            question=query,
            answer=answer,
            retrieved_items=retrieved_items,
            round_history=round_history,
            num_rounds=round_num,
        )
    
    def reset_index(self) -> None:
        """Reset all indexed state across all memory types."""
        self.episodic_memory.reset_index()
        self.semantic_memory.reset_index()
        self.visual_memory.reset_index()
        self.indexed_time = 0
        logger.info("All memory indices reset")
    
    def reset(self) -> None:
        """Reset all state including loaded data for per-video processing."""
        self.reset_index()
        self.episodic_memory.captions = {g: [] for g in self.episodic_memory.granularities}
        self.episodic_memory.caption_id_to_entry.clear()
        if hasattr(self.episodic_memory, "text_to_entry"):
            self.episodic_memory.text_to_entry.clear()
        self.semantic_memory.triple_id_to_entry.clear()
        self.semantic_memory.timestamp_to_triples.clear()
        self.semantic_memory.available_timestamps.clear()
        self.visual_memory.clips.clear()
        self.visual_memory.clip_id_to_entry.clear()
        self.visual_memory.embedding_lookup.clear()
        logger.info("All memory data and indices reset")
    
    def cleanup(self) -> None:
        """Release GPU memory and other resources."""
        self.semantic_memory.cleanup()
        self.visual_memory.cleanup()
        logger.info("Memory cleanup complete")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time as human-readable string."""
        return transform_timestamp(str(self.indexed_time))
    
    def set_retrieval_top_k(
        self,
        episodic: Optional[int] = None,
        semantic: Optional[int] = None,
        visual: Optional[int] = None,
    ) -> None:
        """
        Configure the number of items to retrieve from each memory type.
        
        Args:
            episodic: Top-k for episodic memory
            semantic: Top-k for semantic memory
            visual: Top-k for visual memory
        """
        if episodic is not None:
            self.episodic_top_k = episodic
        if semantic is not None:
            self.semantic_top_k = semantic
        if visual is not None:
            self.visual_top_k = visual
