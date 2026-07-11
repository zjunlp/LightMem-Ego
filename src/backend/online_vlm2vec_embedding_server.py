from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from online_visual.vlm2vec_runtime import VLM2VecRuntime


PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class TextEmbeddingRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    normalize: bool = True


class ImageEmbeddingRequest(BaseModel):
    image_paths: list[str] = Field(default_factory=list)
    normalize: bool = True


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


def create_app(runtime: VLM2VecRuntime, *, preload: bool) -> FastAPI:
    app = FastAPI(title="WorldMM VLM2Vec Embedding Service")

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
            vectors = runtime.encode_texts(request.texts)
        except Exception as exc:
            logger.exception("embed_texts failed count=%d", len(request.texts))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not request.normalize:
            vectors = np.asarray(vectors, dtype="float32")
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        return _vectors_payload(vectors, elapsed_ms)

    @app.post("/embed/images")
    def embed_images(request: ImageEmbeddingRequest) -> dict[str, Any]:
        if not request.image_paths:
            return _vectors_payload(np.zeros((0, 0), dtype="float32"), 0)
        start = time.perf_counter()
        try:
            vectors = runtime.encode_images(request.image_paths)
        except Exception as exc:
            logger.exception("embed_images failed count=%d first_path=%s", len(request.image_paths), request.image_paths[0] if request.image_paths else "")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not request.normalize:
            vectors = np.asarray(vectors, dtype="float32")
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        return _vectors_payload(vectors, elapsed_ms)

    return app


def main() -> None:
    logging.basicConfig(level=os.getenv("WORLDMM_VLM2VEC_EMBED_LOG_LEVEL", "INFO").upper())
    parser = argparse.ArgumentParser(description="WorldMM VLM2Vec embedding HTTP service.")
    parser.add_argument("--host", default=os.getenv("WORLDMM_VLM2VEC_EMBED_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WORLDMM_VLM2VEC_EMBED_PORT", "18091")))
    parser.add_argument("--model-path", default=os.getenv("WORLDMM_VLM2VEC_MODEL_PATH"))
    parser.add_argument("--device", default=os.getenv("WORLDMM_VLM2VEC_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.getenv("WORLDMM_VLM2VEC_DTYPE", "float16"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("WORLDMM_VISUAL_BATCH_SIZE", "8")))
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    runtime = VLM2VecRuntime(
        backend="vlm2vec",
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        normalize=True,
    )
    app = create_app(runtime, preload=not args.no_preload)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=os.getenv("WORLDMM_VLM2VEC_EMBED_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
