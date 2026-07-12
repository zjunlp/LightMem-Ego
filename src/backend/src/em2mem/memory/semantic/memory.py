# """
# Semantic Memory module for Em2Mem.
# """

# import json
# import logging
# import torch
# import torch.nn.functional as F
# import igraph as ig
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from dataclasses import dataclass

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     """Represents a single semantic triple entry with its metadata."""
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int  # The timestamp when this triple was consolidated
    
#     @property
#     def triple(self) -> List[str]:
#         """Return the triple as a list."""
#         return [self.subject, self.predicate, self.object]
    
#     @property
#     def text(self) -> str:
#         """Return the triple as a joined text string for embedding."""
#         return " ".join(self.triple)
    
#     def to_display_str(self) -> str:
#         """Format triple for display."""
#         return f"({self.subject}, {self.predicate}, {self.object})"


# def _transform_timestamp(ts_str: str) -> str:
#     """Transform timestamp string to human-readable format."""
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     """
#     Semantic memory for general knowledge using Personalized PageRank.
    
#     This class manages semantic triples (subject, predicate, object) that represent
#     consolidated knowledge. It uses a graph-based retrieval approach where:
#     - Entities (subjects and objects) are vertices in the graph
#     - Triples define edges between entities
#     - Retrieval uses Personalized PageRank (PPR) to find relevant triples
    
#     The retrieval process:
#     1. Index triples up to a given timestamp, building a graph and embeddings
#     2. For a query, find top-k similar triples using embedding similarity
#     3. Extract entities from those triples for PPR personalization
#     4. Run PPR on the entity graph
#     5. Score triples by summing PPR scores of their subject and object entities
#     6. Return top-k triples by PPR-based score
    
#     Attributes:
#         embedding_model: Model for computing triple embeddings
#         timestamp_to_triples: Dict mapping timestamp to list of triples at that timestamp
#         indexed_entries: List of entries from the closest timestamp before indexed_time
#         indexed_time: Timestamp boundary for indexed triples
#         indexed_timestamp: The specific timestamp of the indexed triples
#         graph: igraph Graph with entities as vertices
#         embeddings: Tensor of triple embeddings for indexed entries
#     """
    
#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#     ):
#         """
#         Initialize SemanticMemory.
        
#         Args:
#             embedding_model: Embedding model for computing triple embeddings
#         """
#         self.embedding_model = embedding_model
        
#         # Storage for triples
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []  # Sorted list of timestamps
        
#         # Indexed state
#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0  # The specific timestamp that was indexed
        
#         # Graph and embeddings for retrieval
#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
    
#     def load_triples_from_file(self, file_path: str) -> None:
#         """
#         Load semantic triples from a JSON file.
        
#         Expected format:
#         {
#             "timestamp1": {
#                 "consolidated_semantic_triples": [[subj, pred, obj], ...]
#             },
#             ...
#         }
        
#         Args:
#             file_path: Path to JSON file containing consolidated semantic triples
#         """
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#         self.load_triples_from_data(data)
    
#     def load_triples_from_data(
#         self,
#         data: Dict[str, Dict[str, Any]],
#     ) -> None:
#         """
#         Load semantic triples from in-memory data.
        
#         Args:
#             data: Dict mapping timestamp -> {consolidated_semantic_triples}
#         """
#         for timestamp_str, content in data.items():
#             timestamp = int(timestamp_str)
#             triples = content.get("consolidated_semantic_triples", [])
            
#             timestamp_entries = []
#             for idx, triple in enumerate(triples):
#                 if len(triple) < 3:
#                     logger.warning(f"Skipping invalid triple at {timestamp_str}[{idx}]: {triple}")
#                     continue
                
#                 triple_id = f"semantic_{timestamp}_{idx}"
                
#                 entry = SemanticTripleEntry(
#                     id=triple_id,
#                     subject=triple[0],
#                     predicate=triple[1],
#                     object=triple[2] if len(triple) > 2 else "",
#                     timestamp=timestamp,
#                 )
#                 self.triple_id_to_entry[triple_id] = entry
#                 timestamp_entries.append(entry)
            
#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries
            
#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic triples across {len(self.available_timestamps)} timestamps")
    
#     def index(self, until_time: int) -> None:
#         """
#         Index semantic triples from the closest timestamp before or at the specified time.
        
#         This builds the entity graph and computes embeddings for triples
#         from the most recent consolidated semantic memory timestamp <= until_time.
        
#         Args:
#             until_time: Timestamp boundary - index triples from closest timestamp <= this value
#         """
#         # Find the closest timestamp before or at until_time
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break
        
#         if closest_timestamp is None:
#             logger.debug(f"No timestamp found up to {until_time}")
#             return
        
#         # Skip if already indexed this exact timestamp
#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed timestamp {closest_timestamp}, skipping")
#             return
        
#         # Get entries from this specific timestamp
#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
        
#         if not entries_to_index:
#             logger.debug(f"No entries at timestamp {closest_timestamp}")
#             return
        
#         # Collect all unique entities
#         all_entities: Set[str] = set()
#         self.triple_to_entities = {}
        
#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 all_entities.add(subj)
#             if obj:
#                 all_entities.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)
        
#         # Build graph with entities as vertices
#         self.graph = ig.Graph()
#         entity_list = list(all_entities)
#         self.graph.add_vertices(entity_list)
#         entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
        
#         # Add edges for each triple (connecting subject to object)
#         edges_to_add = []
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities[entry.id]
#             if subj and obj and subj in entity_to_vertex and obj in entity_to_vertex:
#                 subj_vertex = entity_to_vertex[subj]
#                 obj_vertex = entity_to_vertex[obj]
#                 if subj_vertex != obj_vertex:
#                     edges_to_add.append((subj_vertex, obj_vertex))
        
#         if edges_to_add:
#             self.graph.add_edges(edges_to_add)
        
#         # Compute embeddings for triples
#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)
        
#         self.embeddings = torch.tensor(
#             all_embeddings, 
#             dtype=torch.float32, 
#             device="cuda" if torch.cuda.is_available() else "cpu"
#         )
#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp
        
#         logger.info(f"Indexed {len(entries_to_index)} semantic triples from timestamp {closest_timestamp} (query time: {until_time})")
    
#     def retrieve(
#         self,
#         query: str,
#         top_k: int = 10,
#         as_context: bool = True,
#     ) -> Union[List[SemanticTripleEntry], str]:
#         """
#         Retrieve top-k semantic triples using Personalized PageRank.
        
#         The retrieval process:
#         1. Compute query embedding and find top-k similar triples
#         2. Extract entities from those triples for PPR personalization
#         3. Run PPR on the entity graph
#         4. Score triples by summing PPR scores of subject and object entities
#         5. Return top-k triples by PPR score
        
#         Args:
#             query: Search query text
#             top_k: Number of triples to retrieve
#             as_context: If True, return formatted string instead of entries
            
#         Returns:
#             List of SemanticTripleEntry objects or formatted context string
#         """
#         if not self.indexed_entries or self.embeddings is None or self.graph is None:
#             logger.warning("No triples indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []
        
#         device = self.embeddings.device
        
#         # Encode query
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)
        
#         # Compute similarities with triple embeddings
#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        
#         # Get top-k similar triples for personalization
#         num_available = len(self.indexed_entries)
#         top_k_sim = min(top_k, num_available)
#         top_values, top_pos_indices = torch.topk(similarities, top_k_sim)
        
#         top_sim_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]
        
#         # Extract entities from top similar triples for PPR personalization
#         personalization_entities: Set[str] = set()
#         for entry in top_sim_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)
        
#         if not personalization_entities:
#             # Fallback: return top similar triples by embedding similarity
#             if as_context:
#                 return self.retrieve_triples_as_str(top_sim_entries)
#             return top_sim_entries
        
#         # Set reset vector for PPR (personalize on entities from top similar triples)
#         num_entities = self.graph.vcount()
#         entity_list = [self.graph.vs[i]['name'] for i in range(num_entities)]
#         reset = [
#             1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
#             for entity in entity_list
#         ]
        
#         # Run Personalized PageRank on entities
#         ppr_scores = self.graph.personalized_pagerank(
#             directed=False,
#             damping=0.85,
#             reset=reset,
#             implementation='prpack'
#         )
        
#         # Create entity to PPR score mapping
#         entity_to_ppr = {entity_list[i]: ppr_scores[i] for i in range(num_entities)}
        
#         # Score triples as sum of subject and object PPR scores
#         triple_scores: Dict[str, float] = {}
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             subj_score = entity_to_ppr.get(subj, 0.0) if subj else 0.0
#             obj_score = entity_to_ppr.get(obj, 0.0) if obj else 0.0
#             triple_scores[entry.id] = subj_score + obj_score
        
#         # Get top-k triples by PPR score
#         sorted_entries = sorted(
#             self.indexed_entries,
#             key=lambda e: triple_scores.get(e.id, 0.0),
#             reverse=True
#         )[:top_k]
        
#         if as_context:
#             return self.retrieve_triples_as_str(sorted_entries)
        
#         return sorted_entries
    
#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         """
#         Format a list of triple entries as context string.
        
#         Args:
#             entries: List of SemanticTripleEntry objects
            
#         Returns:
#             Formatted context string
#         """
#         lines = []
#         for entry in entries:
#             lines.append(entry.to_display_str())
#         return "\n".join(lines)
    
#     def cleanup(self) -> None:
#         """Explicitly free GPU memory."""
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     def reset_index(self) -> None:
#         """Reset the indexed state, clearing graph and embeddings."""
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         logger.info("Index reset - graph and embeddings cleared")
    
#     def get_indexed_time(self) -> str:
#         """Get the current indexed time boundary as human-readable string."""
#         return _transform_timestamp(str(self.indexed_time))
    
#     def get_indexed_timestamp(self) -> str:
#         """Get the specific timestamp that was indexed as human-readable string."""
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"
    
#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         """Get a triple entry by its ID."""
#         return self.triple_id_to_entry.get(triple_id)
    
#     def get_indexed_count(self) -> int:
#         """Get the number of indexed triples."""
#         return len(self.indexed_entries)



# """
# Semantic Memory module for Em2Mem.

# This version reads timestamped semantic fact snapshots produced by the new
# semantic consolidation pipeline.

# It preserves the original retrieval spirit:
# - embedding similarity over semantic facts
# - entity graph retrieval with Personalized PageRank (PPR)

# But the indexed unit is now a richer semantic fact rather than a bare triple.
# """

# import json
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# import igraph as ig

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     """
#     Backward-compatible semantic entry type.

#     This now represents a semantic fact instead of an old bare triple.
#     """
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int

#     subject_type: str = ""
#     object_type: str = ""
#     semantic_summary: str = ""
#     support_count: int = 1
#     support_days: List[str] = field(default_factory=list)
#     support_scales: List[str] = field(default_factory=list)
#     confidence: float = 0.5
#     habit_strength: str = "low"
#     evidence_event_ids: List[str] = field(default_factory=list)

#     @property
#     def triple(self) -> List[str]:
#         return [self.subject, self.predicate, self.object]

#     @property
#     def text(self) -> str:
#         if self.semantic_summary:
#             return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
#         return " ".join(self.triple)

#     def to_display_str(self) -> str:
#         base = f"({self.subject}, {self.predicate}, {self.object})"
#         extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
#         if self.semantic_summary:
#             return f"{self.semantic_summary} {base} {extra}"
#         return f"{base} {extra}"


# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     """
#     Semantic memory for consolidated long-term knowledge using Personalized PageRank.

#     New expected input format:
#     {
#       "11110000": {
#         "facts": [...],
#         "stats": {...}
#       },
#       ...
#     }

#     Each fact entry should include:
#     - fact_id
#     - head / head_type
#     - relation
#     - tail / tail_type
#     - semantic_summary
#     - support_count
#     - support_days
#     - support_scales
#     - confidence
#     - habit_strength
#     - evidence_event_ids
#     """

#     def __init__(
#         self,
#         embedding_model: EmbeddingModel,
#     ):
#         self.embedding_model = embedding_model

#         # storage
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []

#         # indexed state
#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0

#         # retrieval state
#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
#         self.entity_to_vertex: Dict[str, int] = {}

#     # -----------------------------------------------------
#     # loading
#     # -----------------------------------------------------

#     def load_triples_from_file(self, file_path: str) -> None:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.load_triples_from_data(data)

#     def load_triples_from_data(
#         self,
#         data: Dict[str, Dict[str, Any]],
#     ) -> None:
#         """
#         Supports:
#         1) new schema:
#            {
#              "11110000": {"facts": [...]}
#            }

#         2) backward-compatible old schema:
#            {
#              "11110000": {"consolidated_semantic_triples": [[s,p,o], ...]}
#            }
#         """
#         self.triple_id_to_entry.clear()
#         self.timestamp_to_triples.clear()
#         self.available_timestamps = []

#         for timestamp_str, content in data.items():
#             try:
#                 timestamp = int(timestamp_str)
#             except Exception:
#                 logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
#                 continue

#             timestamp_entries: List[SemanticTripleEntry] = []

#             # New format
#             if isinstance(content, dict) and "facts" in content:
#                 facts = content.get("facts", []) or []
#                 for idx, fact in enumerate(facts):
#                     if not isinstance(fact, dict):
#                         continue

#                     subject = str(fact.get("head", "")).strip()
#                     predicate = str(fact.get("relation", "")).strip()
#                     obj = str(fact.get("tail", "")).strip()
#                     if not subject or not predicate or not obj:
#                         continue

#                     fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"

#                     entry = SemanticTripleEntry(
#                         id=fact_id,
#                         subject=subject,
#                         predicate=predicate,
#                         object=obj,
#                         timestamp=timestamp,
#                         subject_type=str(fact.get("head_type", "")).strip(),
#                         object_type=str(fact.get("tail_type", "")).strip(),
#                         semantic_summary=str(fact.get("semantic_summary", "")).strip(),
#                         support_count=int(fact.get("support_count", 1)),
#                         support_days=list(fact.get("support_days", []) or []),
#                         support_scales=list(fact.get("support_scales", []) or []),
#                         confidence=float(fact.get("confidence", 0.5)),
#                         habit_strength=str(fact.get("habit_strength", "low")).strip(),
#                         evidence_event_ids=list(fact.get("evidence_event_ids", []) or []),
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             # Old format fallback
#             elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
#                 triples = content.get("consolidated_semantic_triples", []) or []
#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]),
#                         predicate=str(triple[1]),
#                         object=str(triple[2]),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         evidence_event_ids=[],
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries

#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic facts across {len(self.available_timestamps)} timestamps")

#     # -----------------------------------------------------
#     # indexing
#     # -----------------------------------------------------

#     def index(self, until_time: int) -> None:
#         """
#         Index semantic facts from the closest timestamp <= until_time.
#         """
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break

#         if closest_timestamp is None:
#             logger.debug(f"No semantic timestamp found up to {until_time}")
#             return

#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
#             return

#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
#         if not entries_to_index:
#             logger.debug(f"No semantic entries at timestamp {closest_timestamp}")
#             return

#         self.triple_to_entities = {}
#         entity_set: Set[str] = set()

#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 entity_set.add(subj)
#             if obj:
#                 entity_set.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)

#         entity_list = sorted(entity_set)
#         self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}

#         self.graph = ig.Graph()
#         self.graph.add_vertices(entity_list)

#         # aggregate weighted entity edges
#         pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if not subj or not obj or subj == obj:
#                 continue

#             a, b = sorted([subj, obj])
#             weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
#             pair_weights[(a, b)] += weight

#         if pair_weights:
#             edges = []
#             weights = []
#             for (a, b), w in pair_weights.items():
#                 if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
#                     continue
#                 edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
#                 weights.append(w)
#             if edges:
#                 self.graph.add_edges(edges)
#                 self.graph.es["weight"] = weights

#         # embeddings
#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp

#         logger.info(
#             f"Indexed {len(entries_to_index)} semantic facts from timestamp {closest_timestamp} "
#             f"(query time: {until_time})"
#         )

#     # -----------------------------------------------------
#     # retrieval
#     # -----------------------------------------------------

#     def _min_max_norm(self, values: List[float]) -> List[float]:
#         if not values:
#             return []
#         vmin = min(values)
#         vmax = max(values)
#         if abs(vmax - vmin) < 1e-8:
#             return [1.0 for _ in values]
#         return [(v - vmin) / (vmax - vmin) for v in values]

#     def retrieve(
#         self,
#         query: str,
#         top_k: int = 10,
#         as_context: bool = True,
#     ) -> Union[List[SemanticTripleEntry], str]:
#         """
#         Retrieve top-k semantic facts.

#         Retrieval pipeline:
#         1. embedding similarity over fact texts / summaries
#         2. select top semantic seeds
#         3. run PPR on entity graph using seed entities
#         4. rank facts using combined embedding + PPR + confidence score
#         """
#         if not self.indexed_entries or self.embeddings is None:
#             logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []

#         device = self.embeddings.device

#         # encode query
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

#         # similarity
#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
#         sim_values = similarities.detach().cpu().tolist()

#         # top seed facts by similarity
#         num_available = len(self.indexed_entries)
#         top_seed_k = min(max(top_k * 2, 8), num_available)
#         top_values, top_pos_indices = torch.topk(similarities, top_seed_k)
#         top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

#         # if graph is unavailable or empty, fallback to embedding ranking
#         if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
#             sorted_entries = sorted(
#                 zip(self.indexed_entries, sim_values),
#                 key=lambda x: x[1],
#                 reverse=True,
#             )[:top_k]
#             result = [entry for entry, _ in sorted_entries]
#             if as_context:
#                 return self.retrieve_triples_as_str(result)
#             return result

#         # seed entities for PPR
#         personalization_entities: Set[str] = set()
#         for entry in top_seed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)

#         if not personalization_entities:
#             result = top_seed_entries[:top_k]
#             if as_context:
#                 return self.retrieve_triples_as_str(result)
#             return result

#         entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
#         reset = [
#             1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
#             for entity in entity_list
#         ]

#         try:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
#                 implementation="prpack",
#             )
#         except Exception:
#             # safe fallback without weights
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 implementation="prpack",
#             )

#         entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

#         # score facts
#         fact_ppr_scores = []
#         fact_conf_scores = []
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
#             fact_ppr_scores.append(ppr_score)
#             fact_conf_scores.append(float(entry.confidence))

#         sim_norm = self._min_max_norm(sim_values)
#         ppr_norm = self._min_max_norm(fact_ppr_scores)
#         conf_norm = self._min_max_norm(fact_conf_scores)

#         combined = []
#         for idx, entry in enumerate(self.indexed_entries):
#             score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
#             combined.append((entry, score))

#         combined.sort(key=lambda x: x[1], reverse=True)
#         result = [entry for entry, _ in combined[:top_k]]

#         if as_context:
#             return self.retrieve_triples_as_str(result)
#         return result

#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         lines = []
#         for entry in entries:
#             lines.append(entry.to_display_str())
#         return "\n".join(lines)

#     # -----------------------------------------------------
#     # lifecycle helpers
#     # -----------------------------------------------------

#     def cleanup(self) -> None:
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def reset_index(self) -> None:
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         self.entity_to_vertex = {}
#         logger.info("Semantic index reset - graph and embeddings cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_indexed_timestamp(self) -> str:
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         return self.triple_id_to_entry.get(triple_id)

#     def get_indexed_count(self) -> int:
#         return len(self.indexed_entries)

# """
# Semantic Memory module for Em2Mem.

# This version reads timestamped semantic fact snapshots produced by the new
# semantic consolidation pipeline.

# It preserves the retrieval spirit we discussed:
# - embedding similarity over semantic facts
# - entity graph retrieval with Personalized PageRank (PPR)
# - fact ranking combines semantic match + graph relevance + confidence
# """

# import os
# import json
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# import igraph as ig

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int

#     subject_type: str = ""
#     object_type: str = ""
#     semantic_summary: str = ""
#     support_count: int = 1
#     support_days: List[str] = field(default_factory=list)
#     support_scales: List[str] = field(default_factory=list)
#     confidence: float = 0.5
#     habit_strength: str = "low"
#     evidence_event_ids: List[str] = field(default_factory=list)

#     @property
#     def triple(self) -> List[str]:
#         return [self.subject, self.predicate, self.object]

#     @property
#     def text(self) -> str:
#         if self.semantic_summary:
#             return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
#         return " ".join(self.triple)

#     def to_display_str(self) -> str:
#         base = f"({self.subject}, {self.predicate}, {self.object})"
#         extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
#         if self.semantic_summary:
#             return f"{self.semantic_summary} {base} {extra}"
#         return f"{base} {extra}"



# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     def __init__(self, embedding_model: EmbeddingModel):
#         self.embedding_model = embedding_model
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []

#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0

#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
#         self.entity_to_vertex: Dict[str, int] = {}

#     def load_triples_from_file(self, file_path: str) -> None:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.load_triples_from_data(data)

#     def load_triples_from_data(self, data: Dict[str, Dict[str, Any]]) -> None:
#         self.triple_id_to_entry.clear()
#         self.timestamp_to_triples.clear()
#         self.available_timestamps = []

#         for timestamp_str, content in data.items():
#             try:
#                 timestamp = int(timestamp_str)
#             except Exception:
#                 logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
#                 continue

#             timestamp_entries: List[SemanticTripleEntry] = []

#             if isinstance(content, dict) and "facts" in content:
#                 facts = content.get("facts", []) or []
#                 for idx, fact in enumerate(facts):
#                     if not isinstance(fact, dict):
#                         continue
#                     subject = str(fact.get("head", "")).strip()
#                     predicate = str(fact.get("relation", "")).strip()
#                     obj = str(fact.get("tail", "")).strip()
#                     if not subject or not predicate or not obj:
#                         continue
#                     fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=fact_id,
#                         subject=subject,
#                         predicate=predicate,
#                         object=obj,
#                         timestamp=timestamp,
#                         subject_type=str(fact.get("head_type", "")).strip(),
#                         object_type=str(fact.get("tail_type", "")).strip(),
#                         semantic_summary=str(fact.get("semantic_summary", "")).strip(),
#                         support_count=int(fact.get("support_count", 1)),
#                         support_days=list(fact.get("support_days", []) or []),
#                         support_scales=list(fact.get("support_scales", []) or []),
#                         confidence=float(fact.get("confidence", 0.5)),
#                         habit_strength=str(fact.get("habit_strength", "low")).strip(),
#                         evidence_event_ids=list(fact.get("evidence_event_ids", []) or []),
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
#                 triples = content.get("consolidated_semantic_triples", []) or []
#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]),
#                         predicate=str(triple[1]),
#                         object=str(triple[2]),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         evidence_event_ids=[],
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries

#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic facts across {len(self.available_timestamps)} timestamps")

#     def index(self, until_time: int) -> None:
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break

#         if closest_timestamp is None:
#             logger.debug(f"No semantic timestamp found up to {until_time}")
#             return
#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
#             return

#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
#         if not entries_to_index:
#             return

#         self.triple_to_entities = {}
#         entity_set: Set[str] = set()
#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 entity_set.add(subj)
#             if obj:
#                 entity_set.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)

#         entity_list = sorted(entity_set)
#         self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
#         self.graph = ig.Graph()
#         self.graph.add_vertices(entity_list)

#         pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if not subj or not obj or subj == obj:
#                 continue
#             a, b = sorted([subj, obj])
#             weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
#             pair_weights[(a, b)] += weight

#         if pair_weights:
#             edges = []
#             weights = []
#             for (a, b), w in pair_weights.items():
#                 if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
#                     continue
#                 edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
#                 weights.append(w)
#             if edges:
#                 self.graph.add_edges(edges)
#                 self.graph.es["weight"] = weights

#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp
#         logger.info(
#             f"Indexed {len(entries_to_index)} semantic facts from timestamp {closest_timestamp} "
#             f"(query time: {until_time})"
#         )

#     def _min_max_norm(self, values: List[float]) -> List[float]:
#         if not values:
#             return []
#         vmin = min(values)
#         vmax = max(values)
#         if abs(vmax - vmin) < 1e-8:
#             return [1.0 for _ in values]
#         return [(v - vmin) / (vmax - vmin) for v in values]

#     def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
#         if not self.indexed_entries or self.embeddings is None:
#             logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []

#         device = self.embeddings.device
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
#         sim_values = similarities.detach().cpu().tolist()

#         num_available = len(self.indexed_entries)
#         top_seed_k = min(max(top_k * 2, 8), num_available)
#         _, top_pos_indices = torch.topk(similarities, top_seed_k)
#         top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

#         if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
#             sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
#             result = [entry for entry, _ in sorted_entries]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         personalization_entities: Set[str] = set()
#         for entry in top_seed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)

#         if not personalization_entities:
#             result = top_seed_entries[:top_k]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
#         reset = [1.0 / len(personalization_entities) if entity in personalization_entities else 0.0 for entity in entity_list]

#         try:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
#                 implementation="prpack",
#             )
#         except Exception:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 implementation="prpack",
#             )

#         entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

#         fact_ppr_scores = []
#         fact_conf_scores = []
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
#             fact_ppr_scores.append(ppr_score)
#             fact_conf_scores.append(float(entry.confidence))

#         sim_norm = self._min_max_norm(sim_values)
#         ppr_norm = self._min_max_norm(fact_ppr_scores)
#         conf_norm = self._min_max_norm(fact_conf_scores)

#         combined = []
#         for idx, entry in enumerate(self.indexed_entries):
#             score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
#             combined.append((entry, score))

#         combined.sort(key=lambda x: x[1], reverse=True)
#         result = [entry for entry, _ in combined[:top_k]]
#         return self.retrieve_triples_as_str(result) if as_context else result

#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         return "\n".join(entry.to_display_str() for entry in entries)

#     def cleanup(self) -> None:
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def reset_index(self) -> None:
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         self.entity_to_vertex = {}
#         logger.info("Semantic index reset - graph and embeddings cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_indexed_timestamp(self) -> str:
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         return self.triple_id_to_entry.get(triple_id)

#     def get_indexed_count(self) -> int:
#         return len(self.indexed_entries)


# """
# Semantic Memory module for Em2Mem.

# This version reads timestamped semantic fact snapshots produced by the new
# semantic consolidation pipeline.

# It preserves the retrieval spirit we discussed:
# - embedding similarity over semantic facts
# - entity graph retrieval with Personalized PageRank (PPR)
# - fact ranking combines semantic match + graph relevance + confidence
# """

# import json
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# import igraph as ig

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int

#     subject_type: str = ""
#     object_type: str = ""
#     semantic_summary: str = ""
#     support_count: int = 1
#     support_days: List[str] = field(default_factory=list)
#     support_scales: List[str] = field(default_factory=list)
#     confidence: float = 0.5
#     habit_strength: str = "low"
#     raw_support_count: int = 1
#     evidence_event_ids: List[str] = field(default_factory=list)
#     provenance_root_ids: List[str] = field(default_factory=list)
#     source_doc_ids: List[str] = field(default_factory=list)

#     @property
#     def triple(self) -> List[str]:
#         return [self.subject, self.predicate, self.object]

#     @property
#     def text(self) -> str:
#         if self.semantic_summary:
#             return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
#         return " ".join(self.triple)

#     def to_display_str(self) -> str:
#         base = f"({self.subject}, {self.predicate}, {self.object})"
#         extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
#         if self.semantic_summary:
#             return f"{self.semantic_summary} {base} {extra}"
#         return f"{base} {extra}"



# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     def __init__(self, embedding_model: EmbeddingModel):
#         self.embedding_model = embedding_model
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []

#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0

#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
#         self.entity_to_vertex: Dict[str, int] = {}

#     def load_triples_from_file(self, file_path: str) -> None:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.load_triples_from_data(data)

#     def load_triples_from_data(self, data: Dict[str, Dict[str, Any]]) -> None:
#         self.triple_id_to_entry.clear()
#         self.timestamp_to_triples.clear()
#         self.available_timestamps = []

#         for timestamp_str, content in data.items():
#             try:
#                 timestamp = int(timestamp_str)
#             except Exception:
#                 logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
#                 continue

#             timestamp_entries: List[SemanticTripleEntry] = []

#             if isinstance(content, dict) and "facts" in content:
#                 facts = content.get("facts", []) or []
#                 for idx, fact in enumerate(facts):
#                     if not isinstance(fact, dict):
#                         continue
#                     subject = str(fact.get("head", "")).strip()
#                     predicate = str(fact.get("relation", "")).strip()
#                     obj = str(fact.get("tail", "")).strip()
#                     if not subject or not predicate or not obj:
#                         continue
#                     fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
#                     evidence_event_ids = list(fact.get("evidence_event_ids", []) or [])
#                     source_doc_ids = list(fact.get("source_doc_ids", []) or [])
#                     provenance_root_ids = list(fact.get("provenance_root_ids", []) or [])
#                     if not evidence_event_ids and source_doc_ids:
#                         evidence_event_ids = list(source_doc_ids)
#                     if not provenance_root_ids and source_doc_ids:
#                         provenance_root_ids = list(source_doc_ids)

#                     entry = SemanticTripleEntry(
#                         id=fact_id,
#                         subject=subject,
#                         predicate=predicate,
#                         object=obj,
#                         timestamp=timestamp,
#                         subject_type=str(fact.get("head_type", "")).strip(),
#                         object_type=str(fact.get("tail_type", "")).strip(),
#                         semantic_summary=str(fact.get("semantic_summary", "")).strip(),
#                         support_count=int(fact.get("support_count", 1)),
#                         support_days=list(fact.get("support_days", []) or []),
#                         support_scales=list(fact.get("support_scales", []) or []),
#                         confidence=float(fact.get("confidence", 0.5)),
#                         habit_strength=str(fact.get("habit_strength", "low")).strip(),
#                         raw_support_count=int(fact.get("raw_support_count", fact.get("support_count", 1))),
#                         evidence_event_ids=evidence_event_ids,
#                         provenance_root_ids=provenance_root_ids,
#                         source_doc_ids=source_doc_ids,
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
#                 triples = content.get("consolidated_semantic_triples", []) or []
#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]),
#                         predicate=str(triple[1]),
#                         object=str(triple[2]),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         raw_support_count=1,
#                         evidence_event_ids=[],
#                         provenance_root_ids=[],
#                         source_doc_ids=[],
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries

#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic facts across {len(self.available_timestamps)} timestamps")

#     def index(self, until_time: int) -> None:
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break

#         if closest_timestamp is None:
#             logger.debug(f"No semantic timestamp found up to {until_time}")
#             return
#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
#             return

#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
#         if not entries_to_index:
#             return

#         self.triple_to_entities = {}
#         entity_set: Set[str] = set()
#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 entity_set.add(subj)
#             if obj:
#                 entity_set.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)

#         entity_list = sorted(entity_set)
#         self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
#         self.graph = ig.Graph()
#         self.graph.add_vertices(entity_list)

#         pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if not subj or not obj or subj == obj:
#                 continue
#             a, b = sorted([subj, obj])
#             weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
#             pair_weights[(a, b)] += weight

#         if pair_weights:
#             edges = []
#             weights = []
#             for (a, b), w in pair_weights.items():
#                 if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
#                     continue
#                 edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
#                 weights.append(w)
#             if edges:
#                 self.graph.add_edges(edges)
#                 self.graph.es["weight"] = weights

#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp
#         logger.info(
#             f"Indexed {len(entries_to_index)} semantic facts from timestamp {closest_timestamp} "
#             f"(query time: {until_time})"
#         )

#     def _min_max_norm(self, values: List[float]) -> List[float]:
#         if not values:
#             return []
#         vmin = min(values)
#         vmax = max(values)
#         if abs(vmax - vmin) < 1e-8:
#             return [1.0 for _ in values]
#         return [(v - vmin) / (vmax - vmin) for v in values]

#     def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
#         if not self.indexed_entries or self.embeddings is None:
#             logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []

#         device = self.embeddings.device
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
#         sim_values = similarities.detach().cpu().tolist()

#         num_available = len(self.indexed_entries)
#         top_seed_k = min(max(top_k * 2, 8), num_available)
#         _, top_pos_indices = torch.topk(similarities, top_seed_k)
#         top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

#         if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
#             sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
#             result = [entry for entry, _ in sorted_entries]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         personalization_entities: Set[str] = set()
#         for entry in top_seed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)

#         if not personalization_entities:
#             result = top_seed_entries[:top_k]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
#         reset = [1.0 / len(personalization_entities) if entity in personalization_entities else 0.0 for entity in entity_list]

#         try:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
#                 implementation="prpack",
#             )
#         except Exception:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 implementation="prpack",
#             )

#         entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

#         fact_ppr_scores = []
#         fact_conf_scores = []
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
#             fact_ppr_scores.append(ppr_score)
#             fact_conf_scores.append(float(entry.confidence))

#         sim_norm = self._min_max_norm(sim_values)
#         ppr_norm = self._min_max_norm(fact_ppr_scores)
#         conf_norm = self._min_max_norm(fact_conf_scores)

#         combined = []
#         for idx, entry in enumerate(self.indexed_entries):
#             score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
#             combined.append((entry, score))

#         combined.sort(key=lambda x: x[1], reverse=True)
#         result = [entry for entry, _ in combined[:top_k]]
#         return self.retrieve_triples_as_str(result) if as_context else result

#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         return "\n".join(entry.to_display_str() for entry in entries)

#     def cleanup(self) -> None:
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def reset_index(self) -> None:
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         self.entity_to_vertex = {}
#         logger.info("Semantic index reset - graph and embeddings cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_indexed_timestamp(self) -> str:
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         return self.triple_id_to_entry.get(triple_id)

#     def get_indexed_count(self) -> int:
#         return len(self.indexed_entries)


# """
# Semantic Memory module for Em2Mem.

# This version reads timestamped semantic fact snapshots produced by the new
# semantic consolidation pipeline.

# It preserves the retrieval spirit we discussed:
# - embedding similarity over semantic facts
# - entity graph retrieval with Personalized PageRank (PPR)
# - fact ranking combines semantic match + graph relevance + confidence
# """

# import json
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# import igraph as ig

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int

#     subject_type: str = ""
#     object_type: str = ""
#     semantic_summary: str = ""
#     support_count: int = 1
#     support_days: List[str] = field(default_factory=list)
#     support_scales: List[str] = field(default_factory=list)
#     confidence: float = 0.5
#     habit_strength: str = "low"
#     raw_support_count: int = 1
#     evidence_event_ids: List[str] = field(default_factory=list)
#     provenance_root_ids: List[str] = field(default_factory=list)
#     source_doc_ids: List[str] = field(default_factory=list)

#     @property
#     def triple(self) -> List[str]:
#         return [self.subject, self.predicate, self.object]

#     @property
#     def text(self) -> str:
#         if self.semantic_summary:
#             return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
#         return " ".join(self.triple)

#     def to_display_str(self) -> str:
#         base = f"({self.subject}, {self.predicate}, {self.object})"
#         extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
#         if self.semantic_summary:
#             return f"{self.semantic_summary} {base} {extra}"
#         return f"{base} {extra}"



# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     def __init__(self, embedding_model: EmbeddingModel):
#         self.embedding_model = embedding_model
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []

#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0

#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
#         self.entity_to_vertex: Dict[str, int] = {}

#     def load_triples_from_file(self, file_path: str) -> None:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.load_triples_from_data(data)

#     def load_triples_from_data(self, data: Dict[str, Dict[str, Any]]) -> None:
#         self.triple_id_to_entry.clear()
#         self.timestamp_to_triples.clear()
#         self.available_timestamps = []

#         for timestamp_str, content in data.items():
#             try:
#                 timestamp = int(timestamp_str)
#             except Exception:
#                 logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
#                 continue

#             timestamp_entries: List[SemanticTripleEntry] = []

#             if isinstance(content, dict) and "facts" in content:
#                 facts = content.get("facts", []) or []
#                 for idx, fact in enumerate(facts):
#                     if not isinstance(fact, dict):
#                         continue
#                     subject = str(fact.get("head", "")).strip()
#                     predicate = str(fact.get("relation", "")).strip()
#                     obj = str(fact.get("tail", "")).strip()
#                     if not subject or not predicate or not obj:
#                         continue
#                     fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
#                     evidence_event_ids = list(fact.get("evidence_event_ids", []) or [])
#                     source_doc_ids = list(fact.get("source_doc_ids", []) or [])
#                     provenance_root_ids = list(fact.get("provenance_root_ids", []) or [])
#                     if not evidence_event_ids and source_doc_ids:
#                         evidence_event_ids = list(source_doc_ids)
#                     if not provenance_root_ids and source_doc_ids:
#                         provenance_root_ids = list(source_doc_ids)

#                     entry = SemanticTripleEntry(
#                         id=fact_id,
#                         subject=subject,
#                         predicate=predicate,
#                         object=obj,
#                         timestamp=timestamp,
#                         subject_type=str(fact.get("head_type", "")).strip(),
#                         object_type=str(fact.get("tail_type", "")).strip(),
#                         semantic_summary=str(fact.get("semantic_summary", "")).strip(),
#                         support_count=int(fact.get("support_count", 1)),
#                         support_days=list(fact.get("support_days", []) or []),
#                         support_scales=list(fact.get("support_scales", []) or []),
#                         confidence=float(fact.get("confidence", 0.5)),
#                         habit_strength=str(fact.get("habit_strength", "low")).strip(),
#                         raw_support_count=int(fact.get("raw_support_count", fact.get("support_count", 1))),
#                         evidence_event_ids=evidence_event_ids,
#                         provenance_root_ids=provenance_root_ids,
#                         source_doc_ids=source_doc_ids,
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
#                 triples = content.get("consolidated_semantic_triples", []) or []
#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]),
#                         predicate=str(triple[1]),
#                         object=str(triple[2]),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         raw_support_count=1,
#                         evidence_event_ids=[],
#                         provenance_root_ids=[],
#                         source_doc_ids=[],
#                     )
#                     self.triple_id_to_entry[entry.id] = entry
#                     timestamp_entries.append(entry)

#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries

#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic facts across {len(self.available_timestamps)} timestamps")

#     def index(self, until_time: int) -> None:
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break

#         if closest_timestamp is None:
#             logger.debug(f"No semantic timestamp found up to {until_time}")
#             return
#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
#             return

#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
#         if not entries_to_index:
#             return

#         self.triple_to_entities = {}
#         entity_set: Set[str] = set()
#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 entity_set.add(subj)
#             if obj:
#                 entity_set.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)

#         entity_list = sorted(entity_set)
#         self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
#         self.graph = ig.Graph()
#         self.graph.add_vertices(entity_list)

#         pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if not subj or not obj or subj == obj:
#                 continue
#             a, b = sorted([subj, obj])
#             weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
#             pair_weights[(a, b)] += weight

#         if pair_weights:
#             edges = []
#             weights = []
#             for (a, b), w in pair_weights.items():
#                 if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
#                     continue
#                 edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
#                 weights.append(w)
#             if edges:
#                 self.graph.add_edges(edges)
#                 self.graph.es["weight"] = weights

#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp
#         logger.info(
#             f"Indexed {len(entries_to_index)} semantic facts from timestamp {closest_timestamp} "
#             f"(query time: {until_time})"
#         )

#     def _min_max_norm(self, values: List[float]) -> List[float]:
#         if not values:
#             return []
#         vmin = min(values)
#         vmax = max(values)
#         if abs(vmax - vmin) < 1e-8:
#             return [1.0 for _ in values]
#         return [(v - vmin) / (vmax - vmin) for v in values]

#     def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
#         if not self.indexed_entries or self.embeddings is None:
#             logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []

#         device = self.embeddings.device
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
#         sim_values = similarities.detach().cpu().tolist()

#         num_available = len(self.indexed_entries)
#         top_seed_k = min(max(top_k * 2, 8), num_available)
#         _, top_pos_indices = torch.topk(similarities, top_seed_k)
#         top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

#         if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
#             sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
#             result = [entry for entry, _ in sorted_entries]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         personalization_entities: Set[str] = set()
#         for entry in top_seed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)

#         if not personalization_entities:
#             result = top_seed_entries[:top_k]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
#         reset = [1.0 / len(personalization_entities) if entity in personalization_entities else 0.0 for entity in entity_list]

#         try:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
#                 implementation="prpack",
#             )
#         except Exception:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 implementation="prpack",
#             )

#         entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

#         fact_ppr_scores = []
#         fact_conf_scores = []
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
#             fact_ppr_scores.append(ppr_score)
#             fact_conf_scores.append(float(entry.confidence))

#         sim_norm = self._min_max_norm(sim_values)
#         ppr_norm = self._min_max_norm(fact_ppr_scores)
#         conf_norm = self._min_max_norm(fact_conf_scores)

#         combined = []
#         for idx, entry in enumerate(self.indexed_entries):
#             score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
#             combined.append((entry, score))

#         combined.sort(key=lambda x: x[1], reverse=True)
#         result = [entry for entry, _ in combined[:top_k]]
#         return self.retrieve_triples_as_str(result) if as_context else result

#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         return "\n".join(entry.to_display_str() for entry in entries)

#     def get_support_event_ids(self, entry: SemanticTripleEntry, limit: int = 2) -> List[str]:
#         ids = list(getattr(entry, "evidence_event_ids", []) or [])
#         if not ids:
#             ids = list(getattr(entry, "source_doc_ids", []) or [])
#         if not ids:
#             ids = list(getattr(entry, "provenance_root_ids", []) or [])
#         deduped: List[str] = []
#         seen = set()
#         for x in ids:
#             x = str(x)
#             if x and x not in seen:
#                 seen.add(x)
#                 deduped.append(x)
#             if len(deduped) >= limit:
#                 break
#         return deduped

#     def build_packet_text(self, entry: SemanticTripleEntry, support_event_limit: int = 2) -> str:
#         lines = [f"Semantic Fact: {entry.to_display_str()}"]
#         support_ids = self.get_support_event_ids(entry, limit=support_event_limit)
#         if support_ids:
#             lines.append("Support Event IDs: " + ", ".join(support_ids))
#         if entry.support_days:
#             lines.append("Support Days: " + ", ".join(entry.support_days[:5]))
#         if entry.support_scales:
#             lines.append("Support Scales: " + ", ".join(entry.support_scales[:5]))
#         return "\n".join(lines)

#     def retrieve_packets(
#         self,
#         query: str,
#         top_k: int = 5,
#         support_event_limit: int = 2,
#     ) -> List[Dict[str, Any]]:
#         entries = self.retrieve(query=query, top_k=top_k, as_context=False)
#         packets: List[Dict[str, Any]] = []
#         for entry in entries:
#             packets.append({
#                 "packet_type": "semantic",
#                 "fact_id": entry.id,
#                 "text": self.build_packet_text(entry, support_event_limit=support_event_limit),
#                 "support_event_ids": self.get_support_event_ids(entry, limit=support_event_limit),
#                 "confidence": float(entry.confidence),
#                 "support_count": int(entry.support_count),
#             })
#         return packets

#     def cleanup(self) -> None:
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def reset_index(self) -> None:
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         self.entity_to_vertex = {}
#         logger.info("Semantic index reset - graph and embeddings cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_indexed_timestamp(self) -> str:
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         return self.triple_id_to_entry.get(triple_id)

#     def get_indexed_count(self) -> int:
#         return len(self.indexed_entries)


# """
# Semantic Memory module for Em2Mem.

# This version reads timestamped semantic fact snapshots produced by the new
# semantic consolidation pipeline.

# It preserves the retrieval spirit we discussed:
# - embedding similarity over semantic facts
# - entity graph retrieval with Personalized PageRank (PPR)
# - fact ranking combines semantic match + graph relevance + confidence
# """

# import io
# import json
# import logging
# import os
# import zipfile
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Set, Tuple, Union
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# import igraph as ig

# from ...embedding import EmbeddingModel

# logger = logging.getLogger(__name__)


# @dataclass
# class SemanticTripleEntry:
#     id: str
#     subject: str
#     predicate: str
#     object: str
#     timestamp: int

#     subject_type: str = ""
#     object_type: str = ""
#     semantic_summary: str = ""
#     support_count: int = 1
#     support_days: List[str] = field(default_factory=list)
#     support_scales: List[str] = field(default_factory=list)
#     confidence: float = 0.5
#     habit_strength: str = "low"
#     raw_support_count: int = 1
#     evidence_event_ids: List[str] = field(default_factory=list)
#     provenance_root_ids: List[str] = field(default_factory=list)
#     source_doc_ids: List[str] = field(default_factory=list)

#     @property
#     def triple(self) -> List[str]:
#         return [self.subject, self.predicate, self.object]

#     @property
#     def text(self) -> str:
#         if self.semantic_summary:
#             return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
#         return " ".join(self.triple)

#     def to_display_str(self) -> str:
#         base = f"({self.subject}, {self.predicate}, {self.object})"
#         extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
#         if self.semantic_summary:
#             return f"{self.semantic_summary} {base} {extra}"
#         return f"{base} {extra}"



# def _transform_timestamp(ts_str: str) -> str:
#     day = ts_str[0]
#     time_str = ts_str[1:]
#     hh = time_str[0:2]
#     mm = time_str[2:4]
#     ss = time_str[4:6]
#     return f"DAY{day} {hh}:{mm}:{ss}"


# class SemanticMemory:
#     def __init__(self, embedding_model: EmbeddingModel):
#         self.embedding_model = embedding_model
#         self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
#         self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
#         self.available_timestamps: List[int] = []

#         self.indexed_entries: List[SemanticTripleEntry] = []
#         self.indexed_time: int = 0
#         self.indexed_timestamp: int = 0

#         self.graph: Optional[ig.Graph] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
#         self.entity_to_vertex: Dict[str, int] = {}

#     def load_triples_from_file(self, file_path: str) -> None:
#         if str(file_path).lower().endswith(".zip"):
#             with zipfile.ZipFile(file_path, "r") as zf:
#                 json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
#                 if not json_names:
#                     raise ValueError(f"No JSON file found inside semantic zip: {file_path}")
#                 target_name = json_names[0]
#                 with zf.open(target_name) as f:
#                     data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
#         else:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#         self.load_triples_from_data(data)

#     def load_triples_from_data(self, data: Dict[str, Dict[str, Any]]) -> None:
#         self.triple_id_to_entry.clear()
#         self.timestamp_to_triples.clear()
#         self.available_timestamps = []

#         if not isinstance(data, dict):
#             logger.warning("Semantic data is not a dict; skipping load")
#             return

#         if "semantic_triples" in data and isinstance(data.get("semantic_triples"), dict):
#             semantic_triples = data.get("semantic_triples", {}) or {}
#             episodic_evidence = data.get("episodic_evidence", {}) or {}
#             normalized_data: Dict[str, Dict[str, Any]] = {}
#             for timestamp_str, triples in semantic_triples.items():
#                 normalized_data[str(timestamp_str)] = {
#                     "consolidated_semantic_triples": triples or [],
#                     "consolidated_episodic_evidence": episodic_evidence.get(str(timestamp_str), []) if isinstance(episodic_evidence, dict) else [],
#                 }
#             data = normalized_data

#         def _safe_int(x: Any, default: int) -> int:
#             try:
#                 return int(x)
#             except Exception:
#                 return default

#         def _safe_float(x: Any, default: float) -> float:
#             try:
#                 return float(x)
#             except Exception:
#                 return default

#         def _append_entry(timestamp: int, entry: SemanticTripleEntry, bucket: List[SemanticTripleEntry]) -> None:
#             self.triple_id_to_entry[entry.id] = entry
#             bucket.append(entry)

#         for timestamp_str, content in data.items():
#             try:
#                 timestamp = int(timestamp_str)
#             except Exception:
#                 logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
#                 continue

#             timestamp_entries: List[SemanticTripleEntry] = []

#             if isinstance(content, dict) and "facts" in content:
#                 facts = content.get("facts", []) or []
#                 for idx, fact in enumerate(facts):
#                     if not isinstance(fact, dict):
#                         continue
#                     subject = str(fact.get("head", "")).strip()
#                     predicate = str(fact.get("relation", "")).strip()
#                     obj = str(fact.get("tail", "")).strip()
#                     if not subject or not predicate or not obj:
#                         continue
#                     fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
#                     evidence_event_ids = list(fact.get("evidence_event_ids", []) or [])
#                     source_doc_ids = list(fact.get("source_doc_ids", []) or [])
#                     provenance_root_ids = list(fact.get("provenance_root_ids", []) or [])
#                     if not evidence_event_ids and source_doc_ids:
#                         evidence_event_ids = list(source_doc_ids)
#                     if not provenance_root_ids and source_doc_ids:
#                         provenance_root_ids = list(source_doc_ids)

#                     entry = SemanticTripleEntry(
#                         id=fact_id,
#                         subject=subject,
#                         predicate=predicate,
#                         object=obj,
#                         timestamp=timestamp,
#                         subject_type=str(fact.get("head_type", "")).strip(),
#                         object_type=str(fact.get("tail_type", "")).strip(),
#                         semantic_summary=str(fact.get("semantic_summary", "")).strip(),
#                         support_count=_safe_int(fact.get("support_count", 1), 1),
#                         support_days=list(fact.get("support_days", []) or []),
#                         support_scales=list(fact.get("support_scales", []) or []),
#                         confidence=_safe_float(fact.get("confidence", 0.5), 0.5),
#                         habit_strength=str(fact.get("habit_strength", "low")).strip(),
#                         raw_support_count=_safe_int(fact.get("raw_support_count", fact.get("support_count", 1)), 1),
#                         evidence_event_ids=evidence_event_ids,
#                         provenance_root_ids=provenance_root_ids,
#                         source_doc_ids=source_doc_ids,
#                     )
#                     _append_entry(timestamp, entry, timestamp_entries)

#             elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
#                 triples = content.get("consolidated_semantic_triples", []) or []
#                 raw_support = content.get("consolidated_episodic_evidence", []) or []
#                 flattened_support: List[str] = []
#                 for item in raw_support:
#                     if isinstance(item, list):
#                         flattened_support.extend([str(x) for x in item if str(x).strip()])
#                     elif str(item).strip():
#                         flattened_support.append(str(item))

#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]).strip(),
#                         predicate=str(triple[1]).strip(),
#                         object=str(triple[2]).strip(),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=max(1, len(flattened_support)) if flattened_support else 1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         raw_support_count=max(1, len(flattened_support)) if flattened_support else 1,
#                         evidence_event_ids=[],
#                         provenance_root_ids=flattened_support,
#                         source_doc_ids=[],
#                     )
#                     if entry.subject and entry.predicate and entry.object:
#                         _append_entry(timestamp, entry, timestamp_entries)

#             elif isinstance(content, list):
#                 triples = content
#                 for idx, triple in enumerate(triples):
#                     if not isinstance(triple, list) or len(triple) < 3:
#                         continue
#                     entry_id = f"semantic_{timestamp}_{idx}"
#                     entry = SemanticTripleEntry(
#                         id=entry_id,
#                         subject=str(triple[0]).strip(),
#                         predicate=str(triple[1]).strip(),
#                         object=str(triple[2]).strip(),
#                         timestamp=timestamp,
#                         semantic_summary="",
#                         support_count=1,
#                         confidence=0.5,
#                         habit_strength="low",
#                         raw_support_count=1,
#                         evidence_event_ids=[],
#                         provenance_root_ids=[],
#                         source_doc_ids=[],
#                     )
#                     if entry.subject and entry.predicate and entry.object:
#                         _append_entry(timestamp, entry, timestamp_entries)

#             if timestamp_entries:
#                 self.timestamp_to_triples[timestamp] = timestamp_entries

#         self.available_timestamps = sorted(self.timestamp_to_triples.keys())
#         logger.info(f"Loaded semantic facts across {len(self.available_timestamps)} timestamps")

#     def index(self, until_time: int) -> None:
#         closest_timestamp = None
#         for ts in reversed(self.available_timestamps):
#             if ts <= until_time:
#                 closest_timestamp = ts
#                 break

#         if closest_timestamp is None:
#             logger.debug(f"No semantic timestamp found up to {until_time}")
#             return
#         if self.indexed_timestamp == closest_timestamp:
#             logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
#             return

#         entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])
#         if not entries_to_index:
#             return

#         self.triple_to_entities = {}
#         entity_set: Set[str] = set()
#         for entry in entries_to_index:
#             subj, obj = entry.subject, entry.object
#             if subj:
#                 entity_set.add(subj)
#             if obj:
#                 entity_set.add(obj)
#             self.triple_to_entities[entry.id] = (subj, obj)

#         entity_list = sorted(entity_set)
#         self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
#         self.graph = ig.Graph()
#         self.graph.add_vertices(entity_list)

#         pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
#         for entry in entries_to_index:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if not subj or not obj or subj == obj:
#                 continue
#             a, b = sorted([subj, obj])
#             weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
#             pair_weights[(a, b)] += weight

#         if pair_weights:
#             edges = []
#             weights = []
#             for (a, b), w in pair_weights.items():
#                 if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
#                     continue
#                 edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
#                 weights.append(w)
#             if edges:
#                 self.graph.add_edges(edges)
#                 self.graph.es["weight"] = weights

#         all_texts = [entry.text for entry in entries_to_index]
#         all_embeddings = self.embedding_model.encode_text(all_texts)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

#         self.indexed_entries = entries_to_index
#         self.indexed_time = until_time
#         self.indexed_timestamp = closest_timestamp
#         logger.info(
#             f"Indexed {len(entries_to_index)} semantic facts from timestamp {closest_timestamp} "
#             f"(query time: {until_time})"
#         )

#     def _min_max_norm(self, values: List[float]) -> List[float]:
#         if not values:
#             return []
#         vmin = min(values)
#         vmax = max(values)
#         if abs(vmax - vmin) < 1e-8:
#             return [1.0 for _ in values]
#         return [(v - vmin) / (vmax - vmin) for v in values]

#     def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
#         if not self.indexed_entries or self.embeddings is None:
#             logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
#             return "" if as_context else []

#         device = self.embeddings.device
#         query_embedding = self.embedding_model.encode_text(query)
#         if len(query_embedding.shape) == 1:
#             query_embedding = query_embedding.reshape(1, -1)
#         query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

#         similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
#         sim_values = similarities.detach().cpu().tolist()

#         num_available = len(self.indexed_entries)
#         top_seed_k = min(max(top_k * 2, 8), num_available)
#         _, top_pos_indices = torch.topk(similarities, top_seed_k)
#         top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

#         if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
#             sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
#             result = [entry for entry, _ in sorted_entries]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         personalization_entities: Set[str] = set()
#         for entry in top_seed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             if subj:
#                 personalization_entities.add(subj)
#             if obj:
#                 personalization_entities.add(obj)

#         if not personalization_entities:
#             result = top_seed_entries[:top_k]
#             return self.retrieve_triples_as_str(result) if as_context else result

#         entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
#         reset = [1.0 / len(personalization_entities) if entity in personalization_entities else 0.0 for entity in entity_list]

#         try:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
#                 implementation="prpack",
#             )
#         except Exception:
#             ppr_scores = self.graph.personalized_pagerank(
#                 directed=False,
#                 damping=0.85,
#                 reset=reset,
#                 implementation="prpack",
#             )

#         entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

#         fact_ppr_scores = []
#         fact_conf_scores = []
#         for entry in self.indexed_entries:
#             subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
#             ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
#             fact_ppr_scores.append(ppr_score)
#             fact_conf_scores.append(float(entry.confidence))

#         sim_norm = self._min_max_norm(sim_values)
#         ppr_norm = self._min_max_norm(fact_ppr_scores)
#         conf_norm = self._min_max_norm(fact_conf_scores)

#         combined = []
#         for idx, entry in enumerate(self.indexed_entries):
#             score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
#             combined.append((entry, score))

#         combined.sort(key=lambda x: x[1], reverse=True)
#         result = [entry for entry, _ in combined[:top_k]]
#         return self.retrieve_triples_as_str(result) if as_context else result

#     def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
#         return "\n".join(entry.to_display_str() for entry in entries)

#     def get_support_event_ids(self, entry: SemanticTripleEntry, limit: int = 2) -> List[str]:
#         ids = list(getattr(entry, "evidence_event_ids", []) or [])
#         if not ids:
#             ids = list(getattr(entry, "source_doc_ids", []) or [])
#         if not ids:
#             ids = list(getattr(entry, "provenance_root_ids", []) or [])
#         deduped: List[str] = []
#         seen = set()
#         for x in ids:
#             x = str(x)
#             if x and x not in seen:
#                 seen.add(x)
#                 deduped.append(x)
#             if len(deduped) >= limit:
#                 break
#         return deduped

#     def build_packet_text(self, entry: SemanticTripleEntry, support_event_limit: int = 2) -> str:
#         lines = [f"Semantic Fact: {entry.to_display_str()}"]
#         support_ids = self.get_support_event_ids(entry, limit=support_event_limit)
#         if support_ids:
#             lines.append("Support Event IDs: " + ", ".join(support_ids))
#         if entry.support_days:
#             lines.append("Support Days: " + ", ".join(entry.support_days[:5]))
#         if entry.support_scales:
#             lines.append("Support Scales: " + ", ".join(entry.support_scales[:5]))
#         return "\n".join(lines)

#     def retrieve_packets(
#         self,
#         query: str,
#         top_k: int = 5,
#         support_event_limit: int = 2,
#     ) -> List[Dict[str, Any]]:
#         entries = self.retrieve(query=query, top_k=top_k, as_context=False)
#         packets: List[Dict[str, Any]] = []
#         for entry in entries:
#             packets.append({
#                 "packet_type": "semantic",
#                 "fact_id": entry.id,
#                 "text": self.build_packet_text(entry, support_event_limit=support_event_limit),
#                 "support_event_ids": self.get_support_event_ids(entry, limit=support_event_limit),
#                 "confidence": float(entry.confidence),
#                 "support_count": int(entry.support_count),
#             })
#         return packets

#     def cleanup(self) -> None:
#         if self.embeddings is not None:
#             del self.embeddings
#             self.embeddings = None
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def reset_index(self) -> None:
#         self.graph = None
#         self.embeddings = None
#         self.indexed_entries = []
#         self.indexed_time = 0
#         self.indexed_timestamp = 0
#         self.triple_to_entities = {}
#         self.entity_to_vertex = {}
#         logger.info("Semantic index reset - graph and embeddings cleared")

#     def get_indexed_time(self) -> str:
#         return _transform_timestamp(str(self.indexed_time))

#     def get_indexed_timestamp(self) -> str:
#         return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

#     def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
#         return self.triple_id_to_entry.get(triple_id)

#     def get_indexed_count(self) -> int:
#         return len(self.indexed_entries)


"""
Semantic Memory module for Em2Mem.

This version supports both:
1) timestamped semantic snapshot files
2) flat top-level semantic memory files with {"facts": [...], "timeline": ...}

Key fixes:
- top-level `facts` are normalized into timestamp buckets
- flat fact files use cumulative indexing (all facts with ts <= until_time)
- snapshot-style files keep closest-timestamp indexing behavior
"""

import io
import hashlib
import json
import logging
import os
import re
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import igraph as ig

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


def _semantic_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred = []
        for key in ("name", "label", "subject", "predicate", "object", "entity", "relation", "value"):
            if key in value:
                text = _semantic_value_to_text(value.get(key))
                if text:
                    preferred.append(f"{key}={text}")
        if preferred:
            return "; ".join(preferred)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (list, tuple, set)):
        return " ".join(text for text in (_semantic_value_to_text(item) for item in value) if text)
    return re.sub(r"\s+", " ", str(value)).strip()


@dataclass
class SemanticTripleEntry:
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int

    subject_type: str = ""
    object_type: str = ""
    semantic_summary: str = ""
    support_count: int = 1
    support_days: List[str] = field(default_factory=list)
    support_scales: List[str] = field(default_factory=list)
    confidence: float = 0.5
    habit_strength: str = "low"
    raw_support_count: int = 1
    evidence_event_ids: List[str] = field(default_factory=list)
    provenance_root_ids: List[str] = field(default_factory=list)
    source_doc_ids: List[str] = field(default_factory=list)

    @property
    def triple(self) -> List[str]:
        return [self.subject, self.predicate, self.object]

    @property
    def text(self) -> str:
        if self.semantic_summary:
            return (
                f"{_semantic_value_to_text(self.subject)} "
                f"{_semantic_value_to_text(self.predicate)} "
                f"{_semantic_value_to_text(self.object)}. "
                f"{_semantic_value_to_text(self.semantic_summary)}"
            ).strip()
        return " ".join(_semantic_value_to_text(item) for item in self.triple if _semantic_value_to_text(item))

    def to_display_str(self) -> str:
        base = (
            f"({_semantic_value_to_text(self.subject)}, "
            f"{_semantic_value_to_text(self.predicate)}, "
            f"{_semantic_value_to_text(self.object)})"
        )
        extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
        if self.semantic_summary:
            return f"{_semantic_value_to_text(self.semantic_summary)} {base} {extra}"
        return f"{base} {extra}"


def _transform_timestamp(ts_str: str) -> str:
    ts_str = str(ts_str)
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


class SemanticMemory:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

        self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
        self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
        self.available_timestamps: List[int] = []

        # "snapshot": use closest timestamp bucket
        # "flat_facts": cumulative all buckets <= until_time
        self.index_mode: str = "snapshot"

        self.indexed_entries: List[SemanticTripleEntry] = []
        self.indexed_time: int = 0
        self.indexed_timestamp: int = 0

        self.graph: Optional[ig.Graph] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
        self.entity_to_vertex: Dict[str, int] = {}
        self.embedding_cache_dir: Optional[Path] = None
        self.embedding_cache_enabled: bool = os.getenv("EM2MEM_SEMANTIC_EMBED_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.embedding_cache_status: Dict[str, Any] = {"enabled": self.embedding_cache_enabled}
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

    # -----------------------------------------------------
    # loading
    # -----------------------------------------------------

    def load_triples_from_file(self, file_path: str) -> None:
        source_path = Path(file_path)
        configured_cache_dir = os.getenv("EM2MEM_SEMANTIC_EMBED_CACHE_DIR", "").strip()
        self.embedding_cache_dir = Path(configured_cache_dir) if configured_cache_dir else source_path.parent / ".semantic_embedding_cache"
        if str(file_path).lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zf:
                json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
                if not json_names:
                    raise ValueError(f"No JSON file found inside semantic zip: {file_path}")
                target_name = json_names[0]
                with zf.open(target_name) as f:
                    data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self.load_triples_from_data(data)

    def _embedding_model_id(self) -> str:
        model_name = getattr(self.embedding_model, "text_model_name", None)
        if model_name:
            return str(model_name)
        text_model = getattr(self.embedding_model, "_text_model", None)
        if text_model is not None:
            nested_name = getattr(text_model, "model_name", None)
            if nested_name:
                return str(nested_name)
        return type(self.embedding_model).__name__

    def _semantic_cache_key(
        self,
        *,
        entries: List[SemanticTripleEntry],
        until_time: int,
        closest_timestamp: int,
    ) -> tuple[str, str]:
        hasher = hashlib.sha256()
        model_id = self._embedding_model_id()
        hasher.update(f"model={model_id}\n".encode("utf-8"))
        hasher.update(f"mode={self.index_mode}\n".encode("utf-8"))
        hasher.update(f"until={until_time}\nclosest={closest_timestamp}\n".encode("utf-8"))
        for entry in entries:
            hasher.update(str(entry.id).encode("utf-8", errors="ignore"))
            hasher.update(b"\0")
            hasher.update(str(entry.timestamp).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(entry.text.encode("utf-8", errors="ignore"))
            hasher.update(b"\0")
        return hasher.hexdigest(), model_id

    def _semantic_cache_path(self, cache_key: str) -> Optional[Path]:
        if not self.embedding_cache_enabled:
            return None
        cache_dir = self.embedding_cache_dir
        if cache_dir is None:
            configured_cache_dir = os.getenv("EM2MEM_SEMANTIC_EMBED_CACHE_DIR", "").strip()
            if configured_cache_dir:
                cache_dir = Path(configured_cache_dir)
        if cache_dir is None:
            return None
        return cache_dir / f"semantic_embeddings_{cache_key[:24]}.npz"

    def _load_embedding_cache(
        self,
        *,
        cache_path: Optional[Path],
        cache_key: str,
        expected_count: int,
    ) -> Optional[np.ndarray]:
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                manifest_raw = data["manifest"] if "manifest" in data.files else None
                manifest = json.loads(str(manifest_raw.item())) if manifest_raw is not None else {}
                if manifest.get("cache_key") != cache_key:
                    raise ValueError("semantic embedding cache key mismatch")
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
                raise ValueError(f"semantic embedding cache shape mismatch: {embeddings.shape}")
            self.embedding_cache_status = {
                "enabled": self.embedding_cache_enabled,
                "hit": True,
                "path": str(cache_path),
                "cache_key": cache_key,
                "entry_count": expected_count,
            }
            logger.info("Loaded semantic embedding cache: %s entries=%d", cache_path, expected_count)
            return embeddings
        except Exception as exc:
            self.embedding_cache_status = {
                "enabled": self.embedding_cache_enabled,
                "hit": False,
                "path": str(cache_path),
                "cache_key": cache_key,
                "entry_count": expected_count,
                "error": str(exc),
            }
            logger.warning("Failed to load semantic embedding cache %s: %s", cache_path, exc)
            return None

    def _save_embedding_cache(
        self,
        *,
        cache_path: Optional[Path],
        cache_key: str,
        model_id: str,
        embeddings: np.ndarray,
        until_time: int,
        closest_timestamp: int,
    ) -> None:
        if cache_path is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "cache_key": cache_key,
                "model_id": model_id,
                "index_mode": self.index_mode,
                "until_time": until_time,
                "closest_timestamp": closest_timestamp,
                "entry_count": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            }
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            with tmp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                    manifest=np.array(json.dumps(manifest, ensure_ascii=False)),
                )
            tmp_path.replace(cache_path)
            self.embedding_cache_status = {
                "enabled": self.embedding_cache_enabled,
                "hit": False,
                "saved": True,
                "path": str(cache_path),
                "cache_key": cache_key,
                "entry_count": int(embeddings.shape[0]),
            }
            logger.info("Saved semantic embedding cache: %s entries=%d", cache_path, embeddings.shape[0])
        except Exception as exc:
            self.embedding_cache_status = {
                "enabled": self.embedding_cache_enabled,
                "hit": False,
                "saved": False,
                "path": str(cache_path),
                "cache_key": cache_key,
                "entry_count": int(embeddings.shape[0]) if getattr(embeddings, "ndim", 0) else 0,
                "error": str(exc),
            }
            logger.warning("Failed to save semantic embedding cache %s: %s", cache_path, exc)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        query_embedding = self.embedding_model.encode_text(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        self._query_embedding_cache[query] = query_embedding
        return query_embedding

    def _safe_int(self, x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _safe_float(self, x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _append_entry(self, timestamp: int, entry: SemanticTripleEntry, bucket: List[SemanticTripleEntry]) -> None:
        self.triple_id_to_entry[entry.id] = entry
        bucket.append(entry)

    def _normalize_top_level_facts(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a flat semantic memory file like:
            {"facts": [...], "timeline": ...}
        into:
            {timestamp_str: {"facts": [...]}, ...}
        """
        self.index_mode = "flat_facts"
        normalized_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"facts": []})

        facts_obj = data.get("facts", [])
        if isinstance(facts_obj, list):
            facts_iter = facts_obj
        elif isinstance(facts_obj, dict):
            facts_iter = list(facts_obj.values())
        else:
            facts_iter = []

        logger.info(
            "Normalizing top-level semantic facts: type=%s count=%s",
            type(facts_obj).__name__,
            len(facts_iter),
        )

        for fact in facts_iter:
            if not isinstance(fact, dict):
                continue

            triple = fact.get("triple", []) or []
            if not isinstance(triple, list) or len(triple) < 3:
                continue

            last_seen = fact.get("last_seen") or fact.get("first_seen") or {}
            if not isinstance(last_seen, dict):
                last_seen = {}

            date = str(last_seen.get("date", "DAY9"))
            end_time = str(last_seen.get("end_time", last_seen.get("start_time", "00000000"))).zfill(8)
            m = re.search(r"(\d+)", date)
            day = m.group(1) if m else "9"
            ts_key = f"{int(day)}{end_time}"

            normalized_data[ts_key]["facts"].append({
                "fact_id": fact.get("fact_id"),
                "head": triple[0],
                "relation": triple[1],
                "tail": triple[2],
                "head_type": fact.get("head_type", ""),
                "tail_type": fact.get("tail_type", ""),
                "semantic_summary": fact.get("semantic_summary", ""),
                "support_count": fact.get("support_count", 1),
                "support_days": fact.get("support_days", []),
                "support_scales": fact.get("support_scales", []),
                "confidence": fact.get(
                    "confidence",
                    min(0.95, 0.45 + 0.04 * min(self._safe_int(fact.get("support_count", 1), 1), 8))
                ),
                "habit_strength": fact.get("habit_strength", "low"),
                "raw_support_count": fact.get("raw_support_count", fact.get("support_count", 1)),
                "evidence_event_ids": fact.get("support_docs", []) or fact.get("evidence_event_ids", []),
                "provenance_root_ids": fact.get("provenance_root_ids", []) or fact.get("support_docs", []),
                "source_doc_ids": fact.get("support_docs", []) or fact.get("source_doc_ids", []),
            })

        return dict(normalized_data)

    def _normalize_old_semantic_extraction(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert old extraction results:
            {"semantic_triples": {...}, "episodic_evidence": {...}}
        into timestamp buckets.
        """
        semantic_triples = data.get("semantic_triples", {}) or {}
        episodic_evidence = data.get("episodic_evidence", {}) or {}
        normalized_data: Dict[str, Dict[str, Any]] = {}

        for timestamp_str, triples in semantic_triples.items():
            normalized_data[str(timestamp_str)] = {
                "consolidated_semantic_triples": triples or [],
                "consolidated_episodic_evidence": episodic_evidence.get(str(timestamp_str), [])
                if isinstance(episodic_evidence, dict) else [],
            }
        return normalized_data

    def load_triples_from_data(self, data: Dict[str, Any]) -> None:
        self.triple_id_to_entry.clear()
        self.timestamp_to_triples.clear()
        self.available_timestamps = []
        self.index_mode = "snapshot"

        if not isinstance(data, dict):
            logger.warning("Semantic data is not a dict; skipping load")
            return

        logger.info("Semantic load top-level keys: %s", list(data.keys())[:10])

        # Case 1: top-level semantic memory file with facts + timeline
        if "facts" in data:
            data = self._normalize_top_level_facts(data)

        # Case 2: old extraction-style format
        elif "semantic_triples" in data and isinstance(data.get("semantic_triples"), dict):
            data = self._normalize_old_semantic_extraction(data)

        for timestamp_str, content in data.items():
            try:
                timestamp = int(timestamp_str)
            except Exception:
                logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
                continue

            timestamp_entries: List[SemanticTripleEntry] = []

            if isinstance(content, dict) and "facts" in content:
                facts = content.get("facts", []) or []
                for idx, fact in enumerate(facts):
                    if not isinstance(fact, dict):
                        continue

                    subject = str(fact.get("head", "")).strip()
                    predicate = str(fact.get("relation", "")).strip()
                    obj = str(fact.get("tail", "")).strip()
                    if not subject or not predicate or not obj:
                        continue

                    fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
                    evidence_event_ids = list(fact.get("evidence_event_ids", []) or [])
                    source_doc_ids = list(fact.get("source_doc_ids", []) or [])
                    provenance_root_ids = list(fact.get("provenance_root_ids", []) or [])

                    if not evidence_event_ids and source_doc_ids:
                        evidence_event_ids = list(source_doc_ids)
                    if not provenance_root_ids and source_doc_ids:
                        provenance_root_ids = list(source_doc_ids)

                    entry = SemanticTripleEntry(
                        id=fact_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        timestamp=timestamp,
                        subject_type=str(fact.get("head_type", "")).strip(),
                        object_type=str(fact.get("tail_type", "")).strip(),
                        semantic_summary=str(fact.get("semantic_summary", "")).strip(),
                        support_count=self._safe_int(fact.get("support_count", 1), 1),
                        support_days=list(fact.get("support_days", []) or []),
                        support_scales=list(fact.get("support_scales", []) or []),
                        confidence=self._safe_float(fact.get("confidence", 0.5), 0.5),
                        habit_strength=str(fact.get("habit_strength", "low")).strip(),
                        raw_support_count=self._safe_int(
                            fact.get("raw_support_count", fact.get("support_count", 1)), 1
                        ),
                        evidence_event_ids=evidence_event_ids,
                        provenance_root_ids=provenance_root_ids,
                        source_doc_ids=source_doc_ids,
                    )
                    self._append_entry(timestamp, entry, timestamp_entries)

            elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
                triples = content.get("consolidated_semantic_triples", []) or []
                raw_support = content.get("consolidated_episodic_evidence", []) or []

                flattened_support: List[str] = []
                for item in raw_support:
                    if isinstance(item, list):
                        flattened_support.extend([str(x) for x in item if str(x).strip()])
                    elif str(item).strip():
                        flattened_support.append(str(item))

                for idx, triple in enumerate(triples):
                    if not isinstance(triple, list) or len(triple) < 3:
                        continue
                    entry_id = f"semantic_{timestamp}_{idx}"
                    entry = SemanticTripleEntry(
                        id=entry_id,
                        subject=str(triple[0]).strip(),
                        predicate=str(triple[1]).strip(),
                        object=str(triple[2]).strip(),
                        timestamp=timestamp,
                        semantic_summary="",
                        support_count=max(1, len(flattened_support)) if flattened_support else 1,
                        confidence=0.5,
                        habit_strength="low",
                        raw_support_count=max(1, len(flattened_support)) if flattened_support else 1,
                        evidence_event_ids=[],
                        provenance_root_ids=flattened_support,
                        source_doc_ids=[],
                    )
                    if entry.subject and entry.predicate and entry.object:
                        self._append_entry(timestamp, entry, timestamp_entries)

            elif isinstance(content, list):
                triples = content
                for idx, triple in enumerate(triples):
                    if not isinstance(triple, list) or len(triple) < 3:
                        continue
                    entry_id = f"semantic_{timestamp}_{idx}"
                    entry = SemanticTripleEntry(
                        id=entry_id,
                        subject=str(triple[0]).strip(),
                        predicate=str(triple[1]).strip(),
                        object=str(triple[2]).strip(),
                        timestamp=timestamp,
                        semantic_summary="",
                        support_count=1,
                        confidence=0.5,
                        habit_strength="low",
                        raw_support_count=1,
                        evidence_event_ids=[],
                        provenance_root_ids=[],
                        source_doc_ids=[],
                    )
                    if entry.subject and entry.predicate and entry.object:
                        self._append_entry(timestamp, entry, timestamp_entries)

            if timestamp_entries:
                self.timestamp_to_triples[timestamp] = timestamp_entries

        self.available_timestamps = sorted(self.timestamp_to_triples.keys())
        logger.info(
            "Loaded semantic facts across %d timestamps (mode=%s)",
            len(self.available_timestamps),
            self.index_mode,
        )

    # -----------------------------------------------------
    # indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        closest_timestamp = None
        for ts in reversed(self.available_timestamps):
            if ts <= until_time:
                closest_timestamp = ts
                break

        if closest_timestamp is None:
            logger.debug(f"No semantic timestamp found up to {until_time}")
            self.indexed_entries = []
            self.indexed_time = until_time
            self.indexed_timestamp = 0
            return

        if self.index_mode == "flat_facts":
            if self.indexed_time == until_time and self.indexed_entries:
                logger.debug(f"Already indexed cumulative semantic facts up to {until_time}, skipping")
                return

            entries_to_index: List[SemanticTripleEntry] = []
            for ts in self.available_timestamps:
                if ts > until_time:
                    break
                entries_to_index.extend(self.timestamp_to_triples.get(ts, []))
        else:
            if self.indexed_timestamp == closest_timestamp:
                logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
                return
            entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])

        if not entries_to_index:
            self.indexed_entries = []
            self.indexed_time = until_time
            self.indexed_timestamp = closest_timestamp
            logger.debug("No semantic entries available after indexing filter")
            return

        self.triple_to_entities = {}
        entity_set: Set[str] = set()
        for entry in entries_to_index:
            subj, obj = entry.subject, entry.object
            if subj:
                entity_set.add(subj)
            if obj:
                entity_set.add(obj)
            self.triple_to_entities[entry.id] = (subj, obj)

        entity_list = sorted(entity_set)
        self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
        self.graph = ig.Graph()
        self.graph.add_vertices(entity_list)

        pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        for entry in entries_to_index:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if not subj or not obj or subj == obj:
                continue
            a, b = sorted([subj, obj])
            weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
            pair_weights[(a, b)] += weight

        if pair_weights:
            edges = []
            weights = []
            for (a, b), w in pair_weights.items():
                if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
                    continue
                edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
                weights.append(w)
            if edges:
                self.graph.add_edges(edges)
                self.graph.es["weight"] = weights

        cache_key, model_id = self._semantic_cache_key(
            entries=entries_to_index,
            until_time=until_time,
            closest_timestamp=closest_timestamp,
        )
        cache_path = self._semantic_cache_path(cache_key)
        all_embeddings = self._load_embedding_cache(
            cache_path=cache_path,
            cache_key=cache_key,
            expected_count=len(entries_to_index),
        )
        if all_embeddings is None:
            all_texts = [entry.text for entry in entries_to_index]
            all_embeddings = self.embedding_model.encode_text(all_texts)
            self._save_embedding_cache(
                cache_path=cache_path,
                cache_key=cache_key,
                model_id=model_id,
                embeddings=all_embeddings,
                until_time=until_time,
                closest_timestamp=closest_timestamp,
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        self.indexed_timestamp = closest_timestamp

        if self.index_mode == "flat_facts":
            logger.info(
                "Indexed %d cumulative semantic facts up to timestamp %s (query time: %s)",
                len(entries_to_index),
                closest_timestamp,
                until_time,
            )
        else:
            logger.info(
                "Indexed %d semantic facts from timestamp %s (query time: %s)",
                len(entries_to_index),
                closest_timestamp,
                until_time,
            )

    # -----------------------------------------------------
    # retrieval
    # -----------------------------------------------------

    def _min_max_norm(self, values: List[float]) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-8:
            return [1.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
        if not self.indexed_entries or self.embeddings is None:
            logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
            return "" if as_context else []

        device = self.embeddings.device
        query_embedding = self._get_query_embedding(query)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        sim_values = similarities.detach().cpu().tolist()

        num_available = len(self.indexed_entries)
        top_seed_k = min(max(top_k * 2, 8), num_available)
        _, top_pos_indices = torch.topk(similarities, top_seed_k)
        top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

        if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
            sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
            result = [entry for entry, _ in sorted_entries]
            return self.retrieve_triples_as_str(result) if as_context else result

        personalization_entities: Set[str] = set()
        for entry in top_seed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj:
                personalization_entities.add(subj)
            if obj:
                personalization_entities.add(obj)

        if not personalization_entities:
            result = top_seed_entries[:top_k]
            return self.retrieve_triples_as_str(result) if as_context else result

        entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
        reset = [
            1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
            for entity in entity_list
        ]

        try:
            ppr_scores = self.graph.personalized_pagerank(
                directed=False,
                damping=0.85,
                reset=reset,
                weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
                implementation="prpack",
            )
        except Exception:
            ppr_scores = self.graph.personalized_pagerank(
                directed=False,
                damping=0.85,
                reset=reset,
                implementation="prpack",
            )

        entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

        fact_ppr_scores = []
        fact_conf_scores = []
        for entry in self.indexed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
            fact_ppr_scores.append(ppr_score)
            fact_conf_scores.append(float(entry.confidence))

        sim_norm = self._min_max_norm(sim_values)
        ppr_norm = self._min_max_norm(fact_ppr_scores)
        conf_norm = self._min_max_norm(fact_conf_scores)

        combined = []
        for idx, entry in enumerate(self.indexed_entries):
            score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
            combined.append((entry, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        result = [entry for entry, _ in combined[:top_k]]
        return self.retrieve_triples_as_str(result) if as_context else result

    def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
        return "\n".join(entry.to_display_str() for entry in entries)

    # -----------------------------------------------------
    # support / packets
    # -----------------------------------------------------

    def get_support_event_ids(self, entry: SemanticTripleEntry, limit: int = 2) -> List[str]:
        ids = list(getattr(entry, "evidence_event_ids", []) or [])
        if not ids and hasattr(entry, "support_docs"):
            ids = list(getattr(entry, "support_docs", []) or [])
        if not ids:
            ids = list(getattr(entry, "source_doc_ids", []) or [])
        if not ids:
            ids = list(getattr(entry, "provenance_root_ids", []) or [])

        deduped: List[str] = []
        seen = set()
        for x in ids:
            x = str(x)
            if x and x not in seen:
                seen.add(x)
                deduped.append(x)
            if len(deduped) >= limit:
                break
        return deduped

    def build_packet_text(self, entry: SemanticTripleEntry, support_event_limit: int = 2) -> str:
        lines = [f"Semantic Fact: {entry.to_display_str()}"]
        support_ids = self.get_support_event_ids(entry, limit=support_event_limit)
        if support_ids:
            lines.append("Support Event IDs: " + ", ".join(support_ids))
        if entry.support_days:
            lines.append("Support Days: " + ", ".join(entry.support_days[:5]))
        if entry.support_scales:
            lines.append("Support Scales: " + ", ".join(entry.support_scales[:5]))
        return "\n".join(lines)

    def retrieve_packets(
        self,
        query: str,
        top_k: int = 5,
        support_event_limit: int = 2,
    ) -> List[Dict[str, Any]]:
        entries = self.retrieve(query=query, top_k=top_k, as_context=False)
        packets: List[Dict[str, Any]] = []
        for entry in entries:
            packets.append({
                "packet_type": "semantic",
                "fact_id": entry.id,
                "text": self.build_packet_text(entry, support_event_limit=support_event_limit),
                "support_event_ids": self.get_support_event_ids(entry, limit=support_event_limit),
                "confidence": float(entry.confidence),
                "support_count": int(entry.support_count),
            })
        return packets

    # -----------------------------------------------------
    # misc
    # -----------------------------------------------------

    def cleanup(self) -> None:
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_index(self) -> None:
        self.graph = None
        self.embeddings = None
        self._query_embedding_cache = {}
        self.indexed_entries = []
        self.indexed_time = 0
        self.indexed_timestamp = 0
        self.triple_to_entities = {}
        self.entity_to_vertex = {}
        logger.info("Semantic index reset - graph and embeddings cleared")

    def get_indexed_time(self) -> str:
        return _transform_timestamp(str(self.indexed_time))

    def get_indexed_timestamp(self) -> str:
        return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

    def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
        return self.triple_id_to_entry.get(triple_id)

    def get_indexed_count(self) -> int:
        return len(self.indexed_entries)
