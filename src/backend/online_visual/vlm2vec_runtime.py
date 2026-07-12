from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
HIPPO_SRC_ROOT = PROJECT_ROOT / "src" / "HippoRAG" / "src"
for _path in (SRC_ROOT, HIPPO_SRC_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _is_placeholder_path(value: str | None) -> bool:
    if not value:
        return True
    text = str(value).strip()
    return text in {"", "/path/to/vlm2vec-v2", "/path/to/VLM2Vec-V2.0", "path/to/vlm2vec-v2"}


def _resolve_model_path(explicit: str | None = None) -> str:
    for value in (
        explicit,
        os.getenv("EM2MEM_VLM2VEC_MODEL_PATH"),
        os.getenv("EM2MEM_VIS_EMBED_MODEL"),
        str(PROJECT_ROOT / "models" / "VLM2Vec-V2.0"),
    ):
        if not _is_placeholder_path(value):
            return str(value).strip()
    return str(PROJECT_ROOT / "models" / "VLM2Vec-V2.0")


def _configure_hf_local_first() -> None:
    """Prefer local HuggingFace files; allow network only when explicitly requested."""
    allow_download = os.getenv("EM2MEM_ALLOW_HF_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}
    if allow_download:
        return
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return vectors / denom


class VLM2VecRuntime:
    """Reusable visual embedding runtime.

    `mock` is deterministic and lightweight for structure tests.
    `vlm2vec` reuses Em2Mem's EmbeddingModel wrapper and loads the model once.
    `remote` calls a long-running VLM2Vec embedding HTTP service.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
        mock_dim: int = 256,
    ) -> None:
        self.backend = (backend or os.getenv("EM2MEM_VISUAL_BACKEND") or "vlm2vec").strip().lower()
        self.model_path = _resolve_model_path(model_path)
        self.device = device or os.getenv("EM2MEM_VLM2VEC_DEVICE") or "cuda"
        self.dtype = dtype or os.getenv("EM2MEM_VLM2VEC_DTYPE") or "float16"
        self.batch_size = int(batch_size or os.getenv("EM2MEM_VISUAL_BATCH_SIZE") or 8)
        self.normalize = normalize
        self.mock_dim = mock_dim
        self.remote_url = (
            os.getenv("EM2MEM_VLM2VEC_EMBED_URL")
            or f"http://{os.getenv('EM2MEM_VLM2VEC_EMBED_HOST', '127.0.0.1')}:{os.getenv('EM2MEM_VLM2VEC_EMBED_PORT', '18091')}"
        ).rstrip("/")
        self.remote_timeout = float(os.getenv("EM2MEM_VLM2VEC_EMBED_TIMEOUT_SECONDS", "300") or 300)
        self._embedding_model = None
        self._session: requests.Session | None = None

        if self.backend not in {"mock", "vlm2vec", "remote"}:
            raise ValueError(f"Unsupported visual backend: {self.backend}")
        if self.backend == "vlm2vec" and not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"VLM2Vec model path not found: {self.model_path}. "
                "Set EM2MEM_VLM2VEC_MODEL_PATH or use --backend mock."
            )

    @property
    def model(self):
        if self._embedding_model is None:
            _configure_hf_local_first()
            from em2mem.embedding import EmbeddingModel

            self._embedding_model = EmbeddingModel(vis_model_name=self.model_path, device=self.device)
            self._embedding_model.load_model(model_type="vision")
        return self._embedding_model

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _remote_post(self, path: str, payload: dict) -> np.ndarray:
        url = f"{self.remote_url}{path}"
        response = self.session.post(url, json=payload, timeout=self.remote_timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:2000]
            raise RuntimeError(f"Embedding service request failed: {url} status={response.status_code} detail={detail}") from exc
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Invalid embedding service response from {url}: missing embeddings")
        return np.asarray(embeddings, dtype="float32")

    def ping_remote(self) -> dict:
        import time

        deadline = time.monotonic() + float(os.getenv("EM2MEM_VLM2VEC_EMBED_READY_TIMEOUT_SECONDS", "900") or 900)
        last_error: BaseException | None = None
        while True:
            try:
                response = self.session.get(f"{self.remote_url}/health", timeout=min(self.remote_timeout, 30.0))
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(float(os.getenv("EM2MEM_VLM2VEC_EMBED_READY_POLL_SECONDS", "2") or 2))
        raise RuntimeError(f"VLM2Vec embedding service is not ready at {self.remote_url}: {last_error}")

    def _mock_vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=(self.mock_dim,)).astype("float32")
        return vec

    def _mock_encode(self, values: Iterable[str]) -> np.ndarray:
        vectors = np.vstack([self._mock_vector(v) for v in values]).astype("float32")
        return l2_normalize(vectors) if self.normalize else vectors

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, self.mock_dim), dtype="float32")
        if self.backend == "mock":
            return self._mock_encode([f"image::{p}" for p in image_paths])
        if self.backend == "remote":
            outputs = []
            for start in range(0, len(image_paths), self.batch_size):
                batch = image_paths[start:start + self.batch_size]
                outputs.append(self._remote_post("/embed/images", {"image_paths": batch, "normalize": self.normalize}))
            vectors = np.vstack(outputs).astype("float32") if outputs else np.zeros((0, self.mock_dim), dtype="float32")
            return l2_normalize(vectors) if self.normalize else vectors

        outputs = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = image_paths[start:start + self.batch_size]
            emb = self.model.encode_image(batch, batch_size=len(batch))
            outputs.append(np.asarray(emb, dtype="float32"))
        vectors = np.vstack(outputs).astype("float32")
        return l2_normalize(vectors) if self.normalize else vectors

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.mock_dim), dtype="float32")
        if self.backend == "mock":
            return self._mock_encode([f"text::{t}" for t in texts])
        if self.backend == "remote":
            outputs = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                outputs.append(self._remote_post("/embed/texts", {"texts": batch, "normalize": self.normalize}))
            vectors = np.vstack(outputs).astype("float32") if outputs else np.zeros((0, self.mock_dim), dtype="float32")
            return l2_normalize(vectors) if self.normalize else vectors

        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            emb = self.model.encode_vis_query(batch, batch_size=len(batch))
            outputs.append(np.asarray(emb, dtype="float32"))
        vectors = np.vstack(outputs).astype("float32")
        return l2_normalize(vectors) if self.normalize else vectors

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "remote_url": self.remote_url if self.backend == "remote" else None,
        }


_GLOBAL_RUNTIME: VLM2VecRuntime | None = None
_GLOBAL_RUNTIME_KEY: tuple | None = None


def get_global_vlm2vec_runtime(
    backend: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    batch_size: int | None = None,
    normalize: bool = True,
) -> VLM2VecRuntime:
    global _GLOBAL_RUNTIME, _GLOBAL_RUNTIME_KEY
    key = (
        backend or os.getenv("EM2MEM_VISUAL_BACKEND") or "vlm2vec",
        _resolve_model_path(model_path),
        device or os.getenv("EM2MEM_VLM2VEC_DEVICE") or "cuda",
        dtype or os.getenv("EM2MEM_VLM2VEC_DTYPE") or "float16",
        int(batch_size or os.getenv("EM2MEM_VISUAL_BATCH_SIZE") or 8),
        normalize,
    )
    if _GLOBAL_RUNTIME is None or _GLOBAL_RUNTIME_KEY != key:
        _GLOBAL_RUNTIME = VLM2VecRuntime(
            backend=key[0],
            model_path=key[1],
            device=key[2],
            dtype=key[3],
            batch_size=key[4],
            normalize=normalize,
        )
        _GLOBAL_RUNTIME_KEY = key
    return _GLOBAL_RUNTIME
