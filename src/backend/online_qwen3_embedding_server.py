from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from online_text_embedding_runtime import Qwen3TextEmbeddingRuntime


PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class TextEmbeddingRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    batch_size: int | None = None
    normalize: bool | None = None


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str | None = None
    encoding_format: str = "float"


def _vectors_payload(vectors: np.ndarray, elapsed_ms: int) -> dict[str, Any]:
    vectors = np.asarray(vectors, dtype="float32")
    dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] else 0
    return {
        "status": "ok",
        "count": int(vectors.shape[0]) if vectors.ndim >= 1 else 0,
        "dim": dim,
        "elapsed_ms": elapsed_ms,
        "embeddings": vectors.tolist(),
    }


def create_app(runtime: Qwen3TextEmbeddingRuntime, *, preload: bool) -> FastAPI:
    app = FastAPI(title="WorldMM Qwen3 Text Embedding Service")

    if preload:
        _ = runtime.model

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": runtime._embedding_model is not None,
            "runtime": runtime.info(),
        }

    @app.post("/embed/texts")
    def embed_texts(request: TextEmbeddingRequest) -> dict[str, Any]:
        if not request.texts:
            return _vectors_payload(np.zeros((0, 0), dtype="float32"), 0)
        start = time.perf_counter()
        try:
            vectors = runtime.encode_texts(
                request.texts,
                batch_size=request.batch_size,
                normalize=request.normalize,
            )
        except Exception as exc:
            logger.exception("embed_texts failed count=%d", len(request.texts))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        return _vectors_payload(vectors, elapsed_ms)

    @app.post("/v1/embeddings")
    def openai_embeddings(request: OpenAIEmbeddingRequest) -> dict[str, Any]:
        texts = [request.input] if isinstance(request.input, str) else list(request.input)
        start = time.perf_counter()
        try:
            vectors = runtime.encode_texts(texts)
        except Exception as exc:
            logger.exception("openai_embeddings failed count=%d", len(texts))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        return {
            "object": "list",
            "model": request.model or runtime.model_path,
            "data": [
                {
                    "object": "embedding",
                    "embedding": np.asarray(vector, dtype="float32").tolist(),
                    "index": index,
                }
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> None:
    logging.basicConfig(level=os.getenv("WORLDMM_TEXT_EMBED_LOG_LEVEL", "INFO").upper())
    parser = argparse.ArgumentParser(description="WorldMM Qwen3 text embedding HTTP service.")
    parser.add_argument("--host", default=os.getenv("WORLDMM_TEXT_EMBED_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WORLDMM_TEXT_EMBED_PORT", "18096")))
    parser.add_argument("--model-path", default=os.getenv("WORLDMM_TEXT_EMBED_MODEL"))
    parser.add_argument("--device", default=os.getenv("WORLDMM_TEXT_EMBED_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("WORLDMM_TEXT_EMBED_BATCH_SIZE", "256")))
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=os.getenv("WORLDMM_TEXT_EMBED_NORMALIZE", "0").strip().lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    runtime = Qwen3TextEmbeddingRuntime(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    app = create_app(runtime, preload=not args.no_preload)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("WORLDMM_TEXT_EMBED_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
