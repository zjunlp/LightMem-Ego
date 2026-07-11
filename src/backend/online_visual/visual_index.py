from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class VisualSearchIndex:
    def __init__(self, index: Any, backend: str, embeddings: np.ndarray | None = None) -> None:
        self.index = index
        self.backend = backend
        self.embeddings = embeddings

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = np.asarray(query_vectors, dtype="float32")
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if self.backend == "faiss":
            scores, indices = self.index.search(query_vectors, top_k)
            return scores, indices
        if self.embeddings is None or len(self.embeddings) == 0:
            return np.zeros((len(query_vectors), 0), dtype="float32"), np.zeros((len(query_vectors), 0), dtype="int64")
        scores = query_vectors @ self.embeddings.T
        k = min(top_k, self.embeddings.shape[0])
        order = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores.astype("float32"), order.astype("int64")


def save_visual_index(path: Path, embeddings: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        tmp = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(index, str(tmp))
        tmp.replace(path)
        return "faiss"
    except Exception:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"backend": "numpy_flat_ip", "embeddings": embeddings}, f)
        tmp.replace(path)
        return "numpy_flat_ip"


def append_visual_index(path: Path, existing_embeddings: np.ndarray, new_embeddings: np.ndarray) -> str:
    """Append new rows to an existing visual index with checkpointed writes.

    If a FAISS index already exists and dimensions match, this uses native
    `index.add()`. Otherwise it falls back to rebuilding a flat numpy/FAISS
    index from the concatenated embedding matrix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_embeddings = np.asarray(existing_embeddings, dtype="float32")
    new_embeddings = np.asarray(new_embeddings, dtype="float32")
    if new_embeddings.size == 0:
        return load_visual_index(path).backend if path.exists() else save_visual_index(path, existing_embeddings)
    if existing_embeddings.size == 0:
        return save_visual_index(path, new_embeddings)
    all_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
    try:
        import faiss  # type: ignore

        if path.exists():
            index = faiss.read_index(str(path))
            if int(index.d) == int(new_embeddings.shape[1]):
                index.add(new_embeddings)
                tmp = path.with_suffix(path.suffix + ".tmp")
                faiss.write_index(index, str(tmp))
                tmp.replace(path)
                return "faiss"
    except Exception:
        pass
    return save_visual_index(path, all_embeddings)


def load_visual_index(path: Path) -> VisualSearchIndex:
    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(path))
        return VisualSearchIndex(index=index, backend="faiss")
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
        embeddings = np.asarray(obj.get("embeddings"), dtype="float32")
        return VisualSearchIndex(index=None, backend=str(obj.get("backend") or "numpy_flat_ip"), embeddings=embeddings)
