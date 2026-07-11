from __future__ import annotations

import os
import time
from typing import List, Union

import numpy as np
import requests


def _text_embed_url() -> str:
    return (
        os.getenv("WORLDMM_TEXT_EMBED_URL")
        or f"http://{os.getenv('WORLDMM_TEXT_EMBED_HOST', '127.0.0.1')}:{os.getenv('WORLDMM_TEXT_EMBED_PORT', '18096')}"
    ).rstrip("/")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


class RemoteTextEmbeddingModel:
    """HTTP client for a long-running Qwen3 text embedding service."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        url: str | None = None,
        timeout_seconds: float | None = None,
        default_batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> None:
        del device
        self.model_name = model_name or os.getenv("WORLDMM_TEXT_EMBED_MODEL") or ""
        self.remote_url = (url or _text_embed_url()).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else _env_float("WORLDMM_TEXT_EMBED_TIMEOUT_SECONDS", 300.0)
        )
        self.default_batch_size = int(
            default_batch_size
            if default_batch_size is not None
            else _env_int("WORLDMM_TEXT_EMBED_BATCH_SIZE", 256)
        )
        if normalize is None:
            normalize = os.getenv("WORLDMM_TEXT_EMBED_NORMALIZE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.normalize = bool(normalize)
        self._session: requests.Session | None = None
        self._ready_checked = False

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def ping_remote(self) -> dict:
        deadline = time.monotonic() + _env_float("WORLDMM_TEXT_EMBED_READY_TIMEOUT_SECONDS", 900.0)
        poll_seconds = max(0.2, _env_float("WORLDMM_TEXT_EMBED_READY_POLL_SECONDS", 2.0))
        last_error: BaseException | None = None
        while True:
            try:
                response = self.session.get(f"{self.remote_url}/health", timeout=min(self.timeout_seconds, 30.0))
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok" and data.get("model_loaded"):
                    self._ready_checked = True
                    return data
                last_error = RuntimeError(f"service not loaded yet: {data}")
            except Exception as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_seconds)
        raise RuntimeError(f"Qwen3 text embedding service is not ready at {self.remote_url}: {last_error}")

    def _ensure_ready(self) -> None:
        if not self._ready_checked:
            self.ping_remote()

    def _remote_post(self, texts: list[str], batch_size: int) -> np.ndarray:
        self._ensure_ready()
        response = self.session.post(
            f"{self.remote_url}/embed/texts",
            json={
                "texts": texts,
                "batch_size": batch_size,
                "normalize": self.normalize,
            },
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:2000]
            raise RuntimeError(
                f"Qwen3 text embedding request failed: status={response.status_code} detail={detail}"
            ) from exc
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Invalid Qwen3 text embedding response: missing embeddings")
        return np.asarray(embeddings, dtype="float32")

    def encode_text(self, texts: Union[str, List[str]], batch_size: int | None = None, **_: object) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        effective_batch_size = max(1, int(batch_size or self.default_batch_size or 256))
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start:start + effective_batch_size]
            outputs.append(self._remote_post(batch, effective_batch_size))
        return np.vstack(outputs).astype("float32") if outputs else np.zeros((0, 0), dtype="float32")

    def encode(self, content: Union[str, List[str]], **kwargs: object) -> np.ndarray:
        return self.encode_text(content, **kwargs)
