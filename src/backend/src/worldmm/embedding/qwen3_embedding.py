import os
from pathlib import Path
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


def _default_attn_implementation() -> str:
    requested = (
        os.getenv("WORLDMM_TEXT_EMBED_ATTN_IMPLEMENTATION")
        or os.getenv("WORLDMM_QWEN3_ATTN_IMPLEMENTATION")
        or ""
    ).strip()
    if requested:
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _resolve_embedding_device(device: str) -> str:
    requested = (
        os.getenv("WORLDMM_TEXT_EMBED_DEVICE")
        or os.getenv("WORLDMM_EMBEDDING_DEVICE")
        or device
        or "auto"
    ).strip()
    if requested in {"", "auto"}:
        return "auto"
    if requested.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return requested
        except Exception:
            pass
        return "cpu"
    return requested


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "./models/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = _resolve_embedding_device(device)
        allow_download = os.getenv("WORLDMM_ALLOW_HF_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}
        local_files_only = Path(model_name).exists() and not allow_download
        attn_implementation = _default_attn_implementation()

        model_kwargs = {"attn_implementation": attn_implementation, "dtype": "auto"}
        if self.device != "auto":
            model_kwargs["device_map"] = self.device

        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
            local_files_only=local_files_only,
        )
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
