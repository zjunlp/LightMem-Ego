# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import json
import logging
import re
import torch
import torch.nn.functional as F
import igraph as ig
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class SemanticTripleEntry_videomme:
    """Represents a single semantic triple entry with its metadata."""
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int  # The timestamp when this triple was consolidated
    
    @property
    def triple(self) -> List[str]:
        """Return the triple as a list."""
        return [self.subject, self.predicate, self.object]
    
    @property
    def text(self) -> str:
        """Return the triple as a joined text string for embedding."""
        return " ".join(self.triple)
    
    def to_display_str(self) -> str:
        """Format triple for display."""
        return f"({self.subject}, {self.predicate}, {self.object})"


def _transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


class SemanticMemory_videomme:
    """
    Semantic memory for general knowledge using Personalized PageRank.
    
    This class manages semantic triples (subject, predicate, object) that represent
    consolidated knowledge. It uses a graph-based retrieval approach where:
    - Entities (subjects and objects) are vertices in the graph
    - Triples define edges between entities
    - Retrieval uses Personalized PageRank (PPR) to find relevant triples
    
    The retrieval process:
    1. Index triples up to a given timestamp, building a graph and embeddings
    2. For a query, find top-k similar triples using embedding similarity
    3. Extract entities from those triples for PPR personalization
    4. Run PPR on the entity graph
    5. Score triples by summing PPR scores of their subject and object entities
    6. Return top-k triples by PPR-based score
    
    Attributes:
        embedding_model: Model for computing triple embeddings
        timestamp_to_triples: Dict mapping timestamp to list of triples at that timestamp
        indexed_entries: List of entries from the closest timestamp before indexed_time
        indexed_time: Timestamp boundary for indexed triples
        indexed_timestamp: The specific timestamp of the indexed triples
        graph: igraph Graph with entities as vertices
        embeddings: Tensor of triple embeddings for indexed entries
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize SemanticMemory.
        
        Args:
            embedding_model: Embedding model for computing triple embeddings
        """
        self.embedding_model = embedding_model
        
        # Storage for triples
        self.triple_id_to_entry: Dict[str, SemanticTripleEntry_videomme] = {}
        self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry_videomme]] = {}
        self.available_timestamps: List[int] = []  # Sorted list of timestamps
        
        # Indexed state
        self.indexed_entries: List[SemanticTripleEntry_videomme] = []
        self.indexed_time: int = 0
        self.indexed_timestamp: int = 0  # The specific timestamp that was indexed
        
        # Graph and embeddings for retrieval
        self.graph: Optional[ig.Graph] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
    
    def load_triples_from_file(self, file_path: str) -> None:
        """
        Load semantic triples from a JSON file.
        
        Expected format:
        {
            "timestamp1": {
                "consolidated_semantic_triples": [[subj, pred, obj], ...]
            },
            ...
        }
        
        Args:
            file_path: Path to JSON file containing consolidated semantic triples
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.load_triples_from_data(data)
    
    def load_triples_from_data(self, data: Dict[str, Any]) -> None:
        if "facts" not in data or "timeline" not in data:
            raise ValueError("Expected builder output format with 'facts' and 'timeline' keys")
        
        facts_by_id = {fact["fact_id"]: fact["triple"] for fact in data["facts"]}
        timeline = data["timeline"]
        for chunk_id, entry in timeline.items():
            end_time_str = entry.get("end_time", "")
            if not end_time_str:
                continue

            date_str = entry.get("date", "")
            if not date_str:
                continue
            match = re.search(r'\d+', date_str)
            if not match:
                continue
            day_num = match.group()
            
            timestamp_str = day_num + end_time_str
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                logger.warning(f"Invalid timestamp from {date_str}+{end_time_str}")
                continue
            
            active_fact_ids = entry.get("active_fact_ids", [])
            triples = []
            for fid in active_fact_ids:
                triple = facts_by_id.get(fid)
                if triple and len(triple) >= 3:
                    triples.append(triple)
                else:
                    logger.warning(f"Fact ID {fid} not found or invalid triple")
            
            if not triples:
                continue
            
            timestamp_entries = []
            for idx, triple in enumerate(triples):
                triple_id = f"semantic_{timestamp}_{idx}"
                entry_obj = SemanticTripleEntry_videomme(
                    id=triple_id,
                    subject=triple[0],
                    predicate=triple[1],
                    object=triple[2],
                    timestamp=timestamp,
                )
                self.triple_id_to_entry[triple_id] = entry_obj
                timestamp_entries.append(entry_obj)
            
            self.timestamp_to_triples[timestamp] = timestamp_entries
        
        self.available_timestamps = sorted(self.timestamp_to_triples.keys())
        logger.info(f"Loaded semantic triples across {len(self.available_timestamps)} timestamps")
        

    def index(self, until_time: int) -> None:
        """
        Index semantic triples from the closest timestamp before or at the specified time.
        
        This builds the entity graph and computes embeddings for triples
        from the most recent consolidated semantic memory timestamp <= until_time.
        
        Args:
            until_time: Timestamp boundary - index triples from closest timestamp <= this value
        """
        # Find the closest timestamp before or at until_time
        closest_timestamp = None
        for ts in reversed(self.available_timestamps):
            if ts <= until_time:
                closest_timestamp = ts
                break
        
        if closest_timestamp is None:
            logger.debug(f"No timestamp found up to {until_time}")
            return
        
        # Skip if already indexed this exact timestamp
        if self.indexed_timestamp == closest_timestamp:
            logger.debug(f"Already indexed timestamp {closest_timestamp}, skipping")
            return
        
        # Get entries from this specific timestamp
        entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
        
        if not entries_to_index:
            logger.debug(f"No entries at timestamp {closest_timestamp}")
            return
        
        # Collect all unique entities
        all_entities: Set[str] = set()
        self.triple_to_entities = {}
        
        for entry in entries_to_index:
            subj, obj = entry.subject, entry.object
            if subj:
                all_entities.add(subj)
            if obj:
                all_entities.add(obj)
            self.triple_to_entities[entry.id] = (subj, obj)
        
        # Build graph with entities as vertices
        self.graph = ig.Graph()
        entity_list = list(all_entities)
        self.graph.add_vertices(entity_list)
        entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
        
        # Add edges for each triple (connecting subject to object)
        edges_to_add = []
        for entry in entries_to_index:
            subj, obj = self.triple_to_entities[entry.id]
            if subj and obj and subj in entity_to_vertex and obj in entity_to_vertex:
                subj_vertex = entity_to_vertex[subj]
                obj_vertex = entity_to_vertex[obj]
                if subj_vertex != obj_vertex:
                    edges_to_add.append((subj_vertex, obj_vertex))
        
        if edges_to_add:
            self.graph.add_edges(edges_to_add)
        
        # Compute embeddings for triples
        all_texts = [entry.text for entry in entries_to_index]
        all_embeddings = self.embedding_model.encode_text(all_texts)
        
        self.embeddings = torch.tensor(
            all_embeddings, 
            dtype=torch.float32, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        self.indexed_timestamp = closest_timestamp
        
        logger.info(f"Indexed {len(entries_to_index)} semantic triples from timestamp {closest_timestamp} (query time: {until_time})")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        as_context: bool = True,
    ) -> Union[List[SemanticTripleEntry_videomme], str]:
        """
        Retrieve top-k semantic triples using Personalized PageRank.
        
        The retrieval process:
        1. Compute query embedding and find top-k similar triples
        2. Extract entities from those triples for PPR personalization
        3. Run PPR on the entity graph
        4. Score triples by summing PPR scores of subject and object entities
        5. Return top-k triples by PPR score
        
        Args:
            query: Search query text
            top_k: Number of triples to retrieve
            as_context: If True, return formatted string instead of entries
            
        Returns:
            List of SemanticTripleEntry objects or formatted context string
        """
        if not self.indexed_entries or self.embeddings is None or self.graph is None:
            logger.warning("No triples indexed. Call index(until_time) before retrieve().")
            return "" if as_context else []
        
        device = self.embeddings.device
        
        # Encode query
        query_embedding = self.embedding_model.encode_text(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)
        
        # Compute similarities with triple embeddings
        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        
        # Get top-k similar triples for personalization
        num_available = len(self.indexed_entries)
        top_k_sim = min(top_k, num_available)
        top_values, top_pos_indices = torch.topk(similarities, top_k_sim)
        
        top_sim_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]
        
        # Extract entities from top similar triples for PPR personalization
        personalization_entities: Set[str] = set()
        for entry in top_sim_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj:
                personalization_entities.add(subj)
            if obj:
                personalization_entities.add(obj)
        
        if not personalization_entities:
            # Fallback: return top similar triples by embedding similarity
            if as_context:
                return self.retrieve_triples_as_str(top_sim_entries)
            return top_sim_entries
        
        # Set reset vector for PPR (personalize on entities from top similar triples)
        num_entities = self.graph.vcount()
        entity_list = [self.graph.vs[i]['name'] for i in range(num_entities)]
        reset = [
            1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
            for entity in entity_list
        ]
        
        # Run Personalized PageRank on entities
        ppr_scores = self.graph.personalized_pagerank(
            directed=False,
            damping=0.85,
            reset=reset,
            implementation='prpack'
        )
        
        # Create entity to PPR score mapping
        entity_to_ppr = {entity_list[i]: ppr_scores[i] for i in range(num_entities)}
        
        # Score triples as sum of subject and object PPR scores
        triple_scores: Dict[str, float] = {}
        for entry in self.indexed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            subj_score = entity_to_ppr.get(subj, 0.0) if subj else 0.0
            obj_score = entity_to_ppr.get(obj, 0.0) if obj else 0.0
            triple_scores[entry.id] = subj_score + obj_score
        
        # Get top-k triples by PPR score
        sorted_entries = sorted(
            self.indexed_entries,
            key=lambda e: triple_scores.get(e.id, 0.0),
            reverse=True
        )[:top_k]
        
        if as_context:
            return self.retrieve_triples_as_str(sorted_entries)
        
        return sorted_entries
    
    def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry_videomme]) -> str:
        """
        Format a list of triple entries as context string.
        
        Args:
            entries: List of SemanticTripleEntry objects
            
        Returns:
            Formatted context string
        """
        lines = []
        for entry in entries:
            lines.append(entry.to_display_str())
        return "\n".join(lines)
    
    def cleanup(self) -> None:
        """Explicitly free GPU memory."""
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_index(self) -> None:
        """Reset the indexed state, clearing graph and embeddings."""
        self.graph = None
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.indexed_timestamp = 0
        self.triple_to_entities = {}
        logger.info("Index reset - graph and embeddings cleared")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time boundary as human-readable string."""
        return _transform_timestamp(str(self.indexed_time))
    
    def get_indexed_timestamp(self) -> str:
        """Get the specific timestamp that was indexed as human-readable string."""
        return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"
    
    def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry_videomme]:
        """Get a triple entry by its ID."""
        return self.triple_id_to_entry.get(triple_id)
    
    def get_indexed_count(self) -> int:
        """Get the number of indexed triples."""
        return len(self.indexed_entries)
