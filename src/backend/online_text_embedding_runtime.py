from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
HIPPO_SRC_ROOT = PROJECT_ROOT / "src" / "HippoRAG" / "src"
for _path in (SRC_ROOT, HIPPO_SRC_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _is_placeholder_path(value: str | None) -> bool:
    if not value:
        return True
    text = str(value).strip()
    return text in {
        "",
        "/path/to/qwen3-embedding-4b",
        "/path/to/Qwen3-Embedding-4B",
        "path/to/qwen3-embedding-4b",
    }


def _resolve_model_path(explicit: str | None = None) -> str:
    for value in (
        explicit,
        os.getenv("EM2MEM_TEXT_EMBED_MODEL"),
        str(PROJECT_ROOT / "models" / "Qwen3-Embedding-4B"),
        "Qwen/Qwen3-Embedding-4B",
    ):
        if not _is_placeholder_path(value):
            return str(value).strip()
    return str(PROJECT_ROOT / "models" / "Qwen3-Embedding-4B")


def _configure_hf_local_first() -> None:
    allow_download = os.getenv("EM2MEM_ALLOW_HF_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    if allow_download:
        return
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return vectors
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return vectors / denom


class Qwen3TextEmbeddingRuntime:
    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
    ) -> None:
        self.model_path = _resolve_model_path(model_path)
        self.device = device or os.getenv("EM2MEM_TEXT_EMBED_DEVICE") or os.getenv("EM2MEM_EMBEDDING_DEVICE") or "cuda"
        self.batch_size = int(batch_size or os.getenv("EM2MEM_TEXT_EMBED_BATCH_SIZE") or 256)
        self.normalize = normalize
        self._embedding_model = None

    @property
    def model(self):
        if self._embedding_model is None:
            _configure_hf_local_first()
            from em2mem.embedding.qwen3_embedding import Qwen3EmbeddingModel

            self._embedding_model = Qwen3EmbeddingModel(model_name=self.model_path, device=self.device)
        return self._embedding_model

    def encode_texts(self, texts: list[str], batch_size: int | None = None, normalize: bool | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        effective_batch_size = max(1, int(batch_size or self.batch_size or 256))
        vectors = np.asarray(self.model.encode_text(texts, batch_size=effective_batch_size), dtype="float32")
        should_normalize = self.normalize if normalize is None else bool(normalize)
        if should_normalize:
            vectors = l2_normalize(vectors)
        return vectors

    def info(self) -> dict:
        return {
            "backend": "qwen3",
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
        }
