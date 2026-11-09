"""Reuse query work without sharing mutable results or changing evidence."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/HippoRAG/src"))
from em2mem.embedding.embedding_wrapper import EmbeddingModel
from em2mem.memory.episodic.EpisodicMemory_rag import EpisodicMemoryRAG
from em2mem.memory.semantic.memory import SemanticMemory
from online_evidence_content import query_terms, query_focus, target_signals


class CountingEncoder:
    def __init__(self):
        self.calls = []

    def encode_text(self, query, **kwargs):
        self.calls.append(query)
        return np.array([[3.0, 4.0]], dtype=np.float32)


def wrapper():
    embedding = EmbeddingModel(text_model_name="offline-query-test")
    encoder = CountingEncoder()
    embedding._text_model = encoder
    return embedding, encoder


def test_episodic_and_semantic_search_share_encoding_but_keep_their_normalization():
    embedding, encoder = wrapper()
    episodic = EpisodicMemoryRAG(embedding, None, None)
    semantic = SemanticMemory(embedding)
    np.testing.assert_allclose(episodic._get_query_embedding("Where is the bag?"), [.6, .8])
    np.testing.assert_array_equal(semantic._get_query_embedding("Where is the bag?"), [[3, 4]])
    assert encoder.calls == ["Where is the bag?"]
    semantic._get_query_embedding("Who held the bag?")
    assert encoder.calls == ["Where is the bag?", "Who held the bag?"]


def test_query_vectors_cannot_be_mutated_by_a_consumer_and_invalidate_on_model_change():
    embedding, encoder = wrapper()
    vector = embedding.encode_query("question")
    vector[:] = 0
    np.testing.assert_array_equal(embedding.encode_query("question"), [[3, 4]])
    assert len(encoder.calls) == 1
    embedding.text_model_name = "another-offline-model"
    embedding.encode_query("question")
    assert len(encoder.calls) == 2


def test_query_vector_storage_is_bounded_and_keeps_recent_queries():
    embedding, encoder = wrapper()
    embedding._query_vectors_limit = 2
    for question in ("a", "b", "a", "c", "b"):
        embedding.encode_query(question)
    assert encoder.calls == ["a", "b", "c", "b"]
    assert len(embedding._query_vectors) == 2


def test_text_helpers_return_independent_sets_and_observation_updates_stay_visible():
    terms = query_terms("戴着墨镜")
    terms.clear()
    assert "sunglasses" in query_terms("戴着墨镜")
    focus = query_focus("他戴着墨镜吗？")
    focus.clear()
    assert "sunglasses" in query_focus("他戴着墨镜吗？")
    observation = {"description": "The man wears sunglasses."}
    assert target_signals("他戴着墨镜吗？", observation)["sunglasses"] == "confirmed"
    observation["description"] = "The man does not wear sunglasses."
    assert target_signals("他戴着墨镜吗？", observation)["sunglasses"] == "negative"
