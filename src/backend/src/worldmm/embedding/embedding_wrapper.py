import os
import numpy as np
from typing import Union, List, Optional
from PIL import Image
from pathlib import Path


def _resolve_default_model_path(env_name: str, relative_parts: List[str], repo_id: str) -> str:
    """Prefer an explicit env var, then a repo-local model directory, else fall back to repo id."""
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value

    repo_root = Path(__file__).resolve().parents[3]
    local_path = repo_root.joinpath(*relative_parts)
    if local_path.exists():
        return str(local_path)

    return repo_id


class EmbeddingModel:
    """Universal embedding wrapper that routes different modalities to appropriate models"""
    
    def __init__(self, 
                text_model_name: Optional[str] = None,
                vis_model_name: Optional[str] = None,
                device: str = "cuda"):
        """
        Initialize embedding models for different modalities
        
        Args:
            text_model_name: Model name for text embeddings (defaults to Qwen3-Embedding-4B)
            vis_model_name: Model name for visual embeddings (defaults to VLM2Vec V2.0)
            device: Device to run models on
        """
        self.device = device
        
        # Initialize models lazily
        self._text_model = None
        self._vis_model = None
        self.text_backend = os.getenv("WORLDMM_TEXT_EMBED_BACKEND", "local").strip().lower()
        self.text_model_name = text_model_name or _resolve_default_model_path(
            env_name="WORLDMM_TEXT_EMBED_MODEL",
            relative_parts=["models", "Qwen3-Embedding-4B"],
            repo_id="Qwen/Qwen3-Embedding-4B",
        )
        self.vis_model_name = vis_model_name or _resolve_default_model_path(
            env_name="WORLDMM_VIS_EMBED_MODEL",
            relative_parts=["models", "VLM2Vec-V2.0"],
            repo_id="VLM2Vec/VLM2Vec-V2.0",
        )

        # Eagerly load models
        # self._text_model = self.text_model
        # self._vis_model = self.vis_model

    @property
    def text_model(self):
        """Lazy loading of text model"""
        if self._text_model is None:
            if self.text_backend == "remote":
                from .remote_text_embedding import RemoteTextEmbeddingModel as TextEmbeddingModel
            elif self.text_backend in {"", "local", "qwen3"}:
                from .qwen3_embedding import Qwen3EmbeddingModel as TextEmbeddingModel
            else:
                raise ValueError(f"Unsupported text embedding backend: {self.text_backend}")
            self._text_model = TextEmbeddingModel(
                model_name=self.text_model_name,
                device=self.device
            )
        return self._text_model
    
    @property
    def vis_model(self):
        """Lazy loading of visual model"""
        if self._vis_model is None:
            from .vlm2vecv2 import VLM2VecV2EmbeddingModel as VisEmbeddingModel
            self._vis_model = VisEmbeddingModel(
                model_name=self.vis_model_name,
                device=self.device
            )
        return self._vis_model

    def load_model(self, model_type: Optional[str] = None):
        """Load embedding models based on specified type"""
        if model_type is None:
            _ = self.text_model
            _ = self.vis_model
        elif model_type == "text":
            _ = self.text_model
        elif model_type == "vision":
            _ = self.vis_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from None, 'text', or 'vision'")

    def encode_text(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode text using Qwen3 model"""
        return self.text_model.encode_text(texts, **kwargs)

    def encode_vis_query(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode visual query using VLM2VecV2 model"""
        return self.vis_model.encode_text(texts, **kwargs)

    def encode_image(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> np.ndarray:
        """Encode images using VLM2VecV2 model"""
        return self.vis_model.encode_image(images, **kwargs)
    
    def encode_video(self, video_paths: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode videos using VLM2VecV2 model"""
        return self.vis_model.encode_video(video_paths, **kwargs)
    
    def encode(self, content: Union[str, List[str], Image.Image, List[Image.Image]], 
               modality: str = "text", **kwargs) -> np.ndarray:
        """
        Universal encode method that routes to appropriate model based on modality
        
        Args:
            content: Content to encode (text, images, or video paths)
            modality: Type of content ('text', 'image', 'video', 'vis_query')
            **kwargs: Additional arguments for specific encoders
        
        Returns:
            numpy array of embeddings
        """
        if modality == "text":
            return self.encode_text(content, **kwargs)  # type: ignore
        elif modality == "image":
            return self.encode_image(content, **kwargs)  # type: ignore
        elif modality == "video":
            return self.encode_video(content, **kwargs)  # type: ignore
        elif modality == "vis_query":
            return self.encode_vis_query(content, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported modality: {modality}. Choose from 'text', 'image', 'video', 'vis_query'")
