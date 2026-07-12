"""
OpenAI GPT Model Wrapper with comprehensive video and image processing capabilities.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import functools
import io
import json
import logging
import os
import sqlite3
import hashlib
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic_core import PydanticUndefined

from filelock import FileLock

import cv2
from decord import VideoReader, cpu
from openai import AsyncOpenAI, OpenAI
from PIL import Image
from tqdm.asyncio import tqdm as tqdm_asyncio

from .utils import dynamic_retry_decorator

# Configure logging
logger = logging.getLogger(__name__)
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
#     logger.addHandler(handler)
#     logger.setLevel(logging.WARNING)

# Model configuration
MODEL_DICT = {
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.3": "gpt-5.3",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.5": "gpt-5.5",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
    "gpt-5-nano": "gpt-5-nano-2025-08-07",
    "gpt-4o-mini":"gpt-4o-mini",
}

# Global cache configuration
_CACHE: OrderedDict[Tuple, Any] = OrderedDict()
_MAX_CACHE_SIZE = 500


def _truncate_debug(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _to_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _reject_html_response(text: str) -> str:
    stripped = text.strip()
    lowered = stripped[:512].lower()
    if lowered.startswith("<!doctype html") or lowered.startswith("<html") or "<title>sub2api" in lowered:
        raise OpenAIModelError(
            "OpenAI-compatible endpoint returned an HTML page instead of model JSON/text. "
            "Check OPENAI_BASE_URL, usually it must include /v1."
        )
    return stripped


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _debug_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _debug_duration_ms(start: float) -> int:
    return int(round(max(0.0, time.perf_counter() - start) * 1000))


ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _normalize_reasoning_effort(value: Any) -> str:
    effort = str(value or "none").strip().lower()
    if effort not in ALLOWED_REASONING_EFFORTS:
        logger.warning("Unsupported reasoning_effort=%r; using none", value)
        return "none"
    return effort


def _reasoning_effort_kwargs() -> Dict[str, Any]:
    if _env_bool("EM2MEM_OPENAI_DISABLE_REASONING", True):
        return {"reasoning_effort": "none"}
    effort = (
        os.getenv("EM2MEM_CHAT_REASONING_EFFORT")
        or os.getenv("EM2MEM_OPENAI_REASONING_EFFORT")
        or "none"
    )
    return {"reasoning_effort": _normalize_reasoning_effort(effort)}


def _looks_like_unsupported_reasoning(exc: Exception) -> bool:
    text = repr(exc).lower()
    has_reasoning_field = "reasoning_effort" in text or "reasoning" in text
    has_unsupported_signal = (
        "unsupported" in text
        or "unrecognized" in text
        or "unknown" in text
        or "unexpected" in text
    )
    return has_reasoning_field and has_unsupported_signal


def class_str(cls):
    fields = []
    for name, field in cls.model_fields.items():
        annotation = str(field.annotation).replace('typing.', '')
        if field.default is not PydanticUndefined:
            fields.append(f"{name}: {annotation} = {field.default}")
        else:
            fields.append(f"{name}: {annotation}")
    return f"{cls.__name__}({', '.join(fields)})"


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get prompt/messages from args
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        # get additional parameters from kwargs or self attributes
        model = getattr(self, "model_name", None)
        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "prompt": prompt,  # prompt requires JSON serializable
            "model": model,
            "text_format": class_str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message_dict = json.loads(row[0])
                text_format = args[1] if len(args) > 1 else kwargs.get("text_format")
                if text_format and isinstance(message_dict, dict):
                    message = text_format(**message_dict)
                else:
                    message = message_dict
                return message

        # if cache miss, call the original function to get the result
        message = func(self, *args, **kwargs)

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """)
            message_str = json.dumps(message, default=lambda o: o.model_dump() if hasattr(o, 'model_dump') else str(o))
            c.execute("INSERT OR REPLACE INTO cache (key, message) VALUES (?, ?)",
                      (key_hash, message_str))
            conn.commit()
            conn.close()

        return message

    return wrapper


def async_cache_response(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # get prompt/messages from args
        if args:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("Missing required 'prompt' parameter for caching.")

        # get additional parameters from kwargs or self attributes
        model = getattr(self, "model_name", None)
        text_format = args[1] if len(args) > 1 else kwargs.get("text_format")

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "prompt": prompt,  # prompt requires JSON serializable
            "model": model,
            "text_format": str(text_format) if text_format else None,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message_dict = json.loads(row[0])
                text_format = args[1] if len(args) > 1 else kwargs.get("text_format")
                if text_format and isinstance(message_dict, dict):
                    message = text_format(**message_dict)
                else:
                    message = message_dict
                return message

        # if cache miss, call the original function to get the result
        message = await func(self, *args, **kwargs)

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT
                )
            """)
            message_str = json.dumps(message, default=lambda o: o.model_dump() if hasattr(o, 'model_dump') else str(o))
            c.execute("INSERT OR REPLACE INTO cache (key, message) VALUES (?, ?)",
                      (key_hash, message_str))
            conn.commit()
            conn.close()

        return message

    return wrapper


class OpenAIModelError(Exception):
    """Custom exception for OpenAI model operations."""
    pass


class OpenAIModel:
    """
    OpenAI GPT model wrapper with video and image processing capabilities.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        max_size: Tuple[int, int] = (512, 512),
        max_size_video: Tuple[int, int] = (256, 256),
        quality: int = 85,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI model wrapper.
        
        Args:
            model_name: Name of the OpenAI model to use
            max_retries: Maximum number of retry attempts
            max_size: Maximum size for image thumbnails
            max_size_video: Maximum size for video frames
            quality: JPEG quality for encoding (1-100)
            fps: Frames per second for video sampling
            nframes: Number of frames to sample from video
            api_key: OpenAI API key (uses env var if not provided)
            cache_dir: Directory to store cache file (defaults to current directory)
            **kwargs: Additional arguments passed to OpenAI API
            
        Raises:
            OpenAIModelError: If model initialization fails
            ValueError: If both fps and nframes are provided
        """
        # Validate parameters
        if fps is not None and nframes is not None:
            raise ValueError("Cannot provide both 'fps' and 'nframes'. Please choose one for video sampling.")
            
        if model_name not in MODEL_DICT:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_DICT.keys())}")

        # Initialize API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIModelError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_API_URL")
        )

        print(f"Using OpenAI base URL: {base_url}")
        timeout = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = _env_float("EM2MEM_OPENAI_TIMEOUT_SECONDS", 120.0)
        # Initialize OpenAI clients
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
            self.sync_client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise OpenAIModelError("Failed to initialize OpenAI client") from e

        # Set instance attributes
        self.model_name = MODEL_DICT[model_name]
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))  # Clamp quality between 1-100
        self.fps = fps
        self.nframes = nframes
        self.kwargs = kwargs
        self.last_debug: Dict[str, Any] = {}
        
        # Initialize cache file path in current directory
        self.cache_file_name = os.path.join(cache_dir or ".cache", f"openai_cache_{model_name.replace('-', '_')}.db")

        logger.info(f"Initialized OpenAIModel with {self.model_name}")

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and convert file path to Path object."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _image_cache_identifier(self, img: Image.Image) -> str:
        """Return a stable identifier for a PIL Image for caching.

        If the image was loaded from a file, use its filename. Otherwise
        compute an MD5 of a PNG-serialized bytes representation.
        """
        # Prefer filename when available
        filename = getattr(img, "filename", None)
        if filename:
            return str(filename)

        # Compute md5 of PNG bytes for an in-memory image
        try:
            buf = io.BytesIO()
            img_rgb = img.convert("RGB")
            img_rgb.save(buf, format="PNG", optimize=True)
            return hashlib.md5(buf.getvalue()).hexdigest()
        except Exception:
            # Fallback to object id if anything goes wrong
            return f"pil-id-{id(img)}"

    def _manage_cache(self, key: Tuple, value: Any) -> None:
        """Manage cache with LRU eviction policy."""
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        
        if len(_CACHE) > _MAX_CACHE_SIZE:
            _CACHE.popitem(last=False)
            logger.debug(f"Cache evicted oldest entry. Current size: {len(_CACHE)}")

    def _preprocess_prompt(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess prompt by handling images and videos in parallel."""
        prompt_copy = copy.deepcopy(prompt)
        
        # Collect items with content to process in parallel
        content_items = [(i, item) for i, item in enumerate(prompt_copy)]
            
        # Process content items in parallel using ThreadPoolExecutor
        max_workers = min(len(content_items), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_content, item["content"]): i 
                for i, item in content_items
            }
            
            # Wait for all content processing to complete
            for future in as_completed(future_to_index):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    logger.error(f"Failed to process content item: {e}")
                    
        return prompt_copy
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Encode image to base64 with caching and optimization.
        
        Args:
            image: Path to the image file or a PIL Image object

        Returns:
            Base64 encoded image string
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            OpenAIModelError: If image processing fails
        """
        # Accept either a PIL Image or a path-like object
        try:
            if isinstance(image, Image.Image):
                img = image
                identifier = self._image_cache_identifier(img)
            else:
                path = self._validate_file_path(image)
                identifier = str(path)
                # Open image from path
                img = Image.open(path)

            cache_key = (identifier, self.max_size, self.quality)

            # Check cache first
            if cache_key in _CACHE:
                _CACHE.move_to_end(cache_key)
                logger.debug(f"Cache hit for image: {identifier}")
                return _CACHE[cache_key]

            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize maintaining aspect ratio
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)

            # Encode to JPEG
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=self.quality, optimize=True)
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Cache the result
            self._manage_cache(cache_key, encoded)
            logger.debug(f"Encoded and cached image: {identifier}")

            # If we opened the image from path, close the file-handle
            if not isinstance(image, Image.Image):
                try:
                    img.close()
                except Exception:
                    pass
            
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode image {image}: {e}")
            raise OpenAIModelError(f"Failed to encode image: {e}") from e

    def encode_video(self, video_path: Union[str, Path]) -> List[str]:
        """
        Encode video frames to base64 with intelligent sampling and caching.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of base64 encoded frame strings
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            OpenAIModelError: If video processing fails
        """
        video_path = self._validate_file_path(video_path)
        cache_key = (str(video_path), self.max_size_video, self.quality, self.fps, self.nframes)
        
        # Check cache first
        if cache_key in _CACHE:
            _CACHE.move_to_end(cache_key)
            logger.debug(f"Cache hit for video: {video_path}")
            return _CACHE[cache_key]

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                raise OpenAIModelError(f"Video file appears to be empty or corrupted: {video_path}")

            # Determine sampling strategy
            sample_indices = self._calculate_sample_indices(vr, total_frames)
            
            # Extract and encode frames
            base64_frames = []
            for idx in sample_indices:
                try:
                    frame = vr[idx].asnumpy()  # RGB format from decord
                    
                    # Convert to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Resize frame
                    frame_resized = cv2.resize(frame_bgr, self.max_size_video, interpolation=cv2.INTER_LANCZOS4)
                    
                    # Encode to JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    success, buffer = cv2.imencode(".jpg", frame_resized, encode_param)
                    
                    if not success:
                        logger.warning(f"Failed to encode frame {idx} from {video_path}")
                        continue
                        
                    base64_frame = base64.b64encode(buffer).decode("utf-8")
                    base64_frames.append(base64_frame)
                    
                except Exception as e:
                    logger.warning(f"Failed to process frame {idx} from {video_path}: {e}")
                    continue

            if not base64_frames:
                raise OpenAIModelError(f"No frames could be extracted from video: {video_path}")

            # Cache the result
            self._manage_cache(cache_key, base64_frames)
            logger.info(f"Encoded {len(base64_frames)} frames from video: {video_path}")
            return base64_frames
            
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}")
            raise OpenAIModelError(f"Failed to encode video: {e}") from e

    def _calculate_sample_indices(self, vr: VideoReader, total_frames: int) -> List[int]:
        """Calculate which frames to sample from the video."""
        sample_indices = []
        
        if self.fps is not None:
            # Sample at specified FPS
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise OpenAIModelError("Cannot determine video FPS")
                
            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))
            
        elif self.nframes is not None:
            # Sample fixed number of frames
            if self.nframes <= 0:
                raise ValueError("nframes must be a positive integer")
                
            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                # Evenly distribute frames across video duration
                indices = [int(i * (total_frames - 1) / (self.nframes - 1)) for i in range(self.nframes)]
                sample_indices = sorted(list(set(indices)))
                
        else:
            # Default: sample at 1 FPS
            logger.warning("No fps or nframes specified, defaulting to 1 FPS sampling")
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))

        # Ensure we have at least the first and last frame
        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
            
        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))
        
        logger.debug(f"Sampling {len(sample_indices)} frames from {total_frames} total frames")
        return sample_indices

    def _process_content(self, content: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Process content by converting image and video references to base64 format in parallel."""
        if isinstance(content, str):
            return

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            raise ValueError("content must be a str, dict, or list")

        # Collect image/video tasks; handle text synchronously to avoid mixing
        media_tasks: List[Tuple[int, str, Any]] = []
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be a dict, got {type(item)}")

            t = item.get("type")
            if t == "text" and "text" in item:
                content[i] = {"type": "input_text", "text": item["text"]}
            elif t == "image" and "image" in item:
                media_tasks.append((i, "image", item["image"]))
            elif t == "video" and "video" in item:
                media_tasks.append((i, "video", item["video"]))
            else:
                raise ValueError(f"Unsupported media item at index {i}: {item}")

        if not media_tasks:
            return

        # Process image/video tasks in parallel
        max_workers = min(len(media_tasks), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in media_tasks:
                i, media_type, file = task
                if media_type == "image":
                    fut = executor.submit(self.encode_image, file)
                elif media_type == "video":
                    fut = executor.submit(self.encode_video, file)
                else:
                    raise ValueError(f"Unsupported media type at index {i}: {media_type}")
                future_to_task[fut] = task

            # Collect results in a list to maintain order
            results = []
            for fut in as_completed(future_to_task):
                i, media_type, _ = future_to_task[fut]
                try:
                    if media_type == "image":
                        image_base64 = fut.result()
                        results.append((i, "image", image_base64))
                    elif media_type == "video":
                        video_frames = fut.result()
                        results.append((i, "video", video_frames))
                except Exception as e:
                    logger.error(f"Failed to process {media_type} content at index {i}: {e}")
                    results.append((i, "error", None))

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])

            # Apply results to content with proper shifting
            shift = 0
            for i, media_type, result in results:
                if media_type == "error":
                    continue
                elif media_type == "image":
                    content[i + shift] = {"type": "input_image", "image_url": f"data:image/jpeg;base64,{result}"}
                elif media_type == "video":
                    image_items = [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame}"}
                                   for frame in result]
                    content[i + shift:i + shift + 1] = image_items
                    shift += len(image_items) - 1

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Coerce string prompts into a single user message."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    def _prompt_has_image(self, prompt: List[Dict[str, Any]]) -> bool:
        for message in prompt:
            content = message.get("content")
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image", "input_image", "image_url"}:
                    return True
        return False

    def _payload_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"num_messages": len(messages), "messages": []}
        for message in messages:
            content = message.get("content")
            item: Dict[str, Any] = {"role": message.get("role", "user")}
            if isinstance(content, str):
                item["content_type"] = "text"
                item["text_chars"] = len(content)
            elif isinstance(content, list):
                item["content_type"] = "array"
                item["text_blocks"] = sum(1 for x in content if isinstance(x, dict) and x.get("type") == "text")
                item["image_url_blocks"] = sum(1 for x in content if isinstance(x, dict) and x.get("type") == "image_url")
                item["other_blocks"] = len(content) - item["text_blocks"] - item["image_url_blocks"]
            else:
                item["content_type"] = type(content).__name__
            summary["messages"].append(item)
        return summary

    def _record_debug(
        self,
        request_path: str,
        messages: List[Dict[str, Any]] | None = None,
        response: Any = None,
        error: Exception | None = None,
        api_timing: Dict[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "request_path": request_path,
            "model": self.model_name,
        }
        if api_timing is not None:
            payload["api_timing"] = api_timing
        if messages is not None:
            payload["payload_summary"] = self._payload_summary(messages)
        if response is not None:
            payload["raw_response_preview"] = _truncate_debug(response, 1000)
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"
        self.last_debug = payload

    @cache_response
    @dynamic_retry_decorator
    def generate(self, prompt: Union[str, List[Dict[str, Any]]], text_format: Optional[type] = None, **kwargs) -> Any:
        """
        Generate completion for a single prompt.
        
        Args:
            prompt: Conversation prompt (list of role/content dicts or raw string)
            text_format (optional): Pydantic model or other structure for parsing the response

        Returns:
            Generated response string
        """
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        processed_prompt = self._preprocess_prompt(prompt_copy)
        if self._prompt_has_image(processed_prompt):
            return self._generate_with_chat_fallback(
                processed_prompt,
                text_format=text_format,
                force_request_path="chat.completions:image",
                **kwargs,
            )
        if _env_bool("EM2MEM_OPENAI_FORCE_CHAT_COMPLETIONS", True):
            return self._generate_with_chat_fallback(
                processed_prompt,
                text_format=text_format,
                force_request_path="chat.completions:forced",
                **kwargs,
            )
        
        try:

            # If text_format is provided (e.g. a Pydantic model), use the parse endpoint
            if text_format is not None:
                request_started_at = _debug_utc_now_iso()
                request_start = time.perf_counter()
                response = self.sync_client.responses.parse(
                    model=self.model_name,
                    input=processed_prompt,
                    text_format=text_format,
                    **self.kwargs,
                    **kwargs
                )
                self._record_debug(
                    "responses.parse",
                    messages=processed_prompt,
                    response=response,
                    api_timing={
                        "api_request_started_at": request_started_at,
                        "api_response_finished_at": _debug_utc_now_iso(),
                        "api_duration_ms": _debug_duration_ms(request_start),
                        "request_path": "responses.parse",
                        "model": self.model_name,
                        "attempt": 1,
                        "attempt_count": 1,
                    },
                )

                # responses.parse should expose the parsed output on `output_parsed`
                return getattr(response, "output_parsed", None)

            # Default unstructured behavior
            request_started_at = _debug_utc_now_iso()
            request_start = time.perf_counter()
            response = self.sync_client.responses.create(
                model=self.model_name,
                input=processed_prompt,
                **self.kwargs,
                **kwargs
            )
            self._record_debug(
                "responses.create",
                messages=processed_prompt,
                response=response,
                api_timing={
                    "api_request_started_at": request_started_at,
                    "api_response_finished_at": _debug_utc_now_iso(),
                    "api_duration_ms": _debug_duration_ms(request_start),
                    "request_path": "responses.create",
                    "model": self.model_name,
                    "attempt": 1,
                    "attempt_count": 1,
                },
            )
            
            # return response.output_text.strip()
            return self._extract_text_from_response(response)
            
        except Exception as e:
            self._record_debug("responses.create", messages=processed_prompt, error=e)
            logger.warning(f"Responses API path failed, retrying with chat.completions: {e}")
            try:
                return self._generate_with_chat_fallback(
                    processed_prompt,
                    text_format=text_format,
                    **kwargs,
                )
            except Exception as chat_e:
                logger.error(f"OpenAI API error: {chat_e}")
                raise OpenAIModelError(
                    f"Failed to get completion. responses_error={e}; chat_error={chat_e}"
                ) from chat_e

    def stream_generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        on_chunk: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Stream a completion from chat.completions and optionally emit chunks as they arrive.

        This keeps the legacy generate() path intact while providing a true
        incremental path for SSE / progressive UI updates.
        """
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        processed_prompt = self._preprocess_prompt(prompt_copy)
        return self._stream_with_chat_fallback(
            processed_prompt,
            text_format=text_format,
            on_chunk=on_chunk,
            force_request_path="chat.completions:stream",
            **kwargs,
        )

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], text_format: Optional[type] = None) -> List[Any]:
        """
        Process multiple prompts in batch with async generation and preprocessing.
        
        Args:
            batch_prompts: List of conversation prompts
            text_format (optional): Pydantic model or other structure for parsing the response

        Returns:
            List of response strings
        """
        # Normalize and preprocess
        batch_prompts_copy = [copy.deepcopy(self._normalize_prompt(prompt)) for prompt in batch_prompts]

        if not batch_prompts_copy:
            return []
        
        # Process prompts in parallel using ThreadPoolExecutor
        max_workers = min(len(batch_prompts_copy), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_prompts = list(executor.map(self._preprocess_prompt, batch_prompts_copy))
        
        # Use async generation for batch processing
        try:
            return asyncio.run(self.async_generation(processed_prompts, text_format=text_format))
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise OpenAIModelError(f"Batch generation failed: {e}") from e

    def _flatten_content(self, content: Any) -> str:
        """Flatten different content formats into plain text."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    # OpenAI / chat-style content chunks
                    text = item.get("text")
                    if text is None and "content" in item:
                        text = item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if text is None:
                        text = getattr(item, "content", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join([p for p in parts if p]).strip()

        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"].strip()
            if isinstance(content.get("content"), str):
                return content["content"].strip()

        return str(content).strip()


    def _extract_text_from_response(self, response: Any) -> str:
        """Support multiple proxy / SDK response formats."""
        if response is None:
            raise OpenAIModelError("Empty response from model.")

        # Case 1: proxy returns raw string directly
        if isinstance(response, str):
            return _reject_html_response(response)

        # Case 2: proxy returns dict
        if isinstance(response, dict):
            if isinstance(response.get("output_text"), str):
                return _reject_html_response(response["output_text"])

            if isinstance(response.get("text"), str):
                return _reject_html_response(response["text"])

            choices = response.get("choices")
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    if isinstance(first.get("text"), str):
                        return _reject_html_response(first["text"])

                    message = first.get("message", {})
                    content = message.get("content")
                    text = self._flatten_content(content)
                    if text:
                        return _reject_html_response(text)

            raise OpenAIModelError(f"Unsupported dict response format: {response}")

        # Case 3: standard Responses API object
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return _reject_html_response(output_text)

        # Case 4: Chat Completions style object
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]

            text = getattr(first, "text", None)
            if isinstance(text, str) and text.strip():
                return _reject_html_response(text)

            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                text = self._flatten_content(content)
                if text:
                    return _reject_html_response(text)

            # dict-like choice fallback
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return _reject_html_response(first["text"])
                message = first.get("message", {})
                content = message.get("content")
                text = self._flatten_content(content)
                if text:
                    return _reject_html_response(text)

        raise OpenAIModelError(f"Unsupported response type: {type(response)}")

    def _convert_content_to_chat_format(self, content: Any) -> Any:
        """Convert Responses-API style content blocks to Chat Completions format."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            return content

        chat_content: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                chat_content.append({"type": "text", "text": str(item)})
                continue

            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                chat_content.append({"type": "text", "text": _to_prompt_text(item.get("text", ""))})
            elif item_type in {"image", "input_image", "image_url"}:
                image_url = item.get("image_url") or item.get("image")
                if isinstance(image_url, dict):
                    chat_content.append({"type": "image_url", "image_url": image_url})
                else:
                    chat_content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                chat_content.append(item)

        return chat_content

    def _convert_prompt_to_chat_messages(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert a normalized prompt into Chat Completions messages."""
        messages: List[Dict[str, Any]] = []
        for message in prompt:
            messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": self._convert_content_to_chat_format(message.get("content")),
                }
            )
        return messages

    def _parse_structured_text(self, text: str, text_format: type) -> Any:
        """Parse a JSON string into the requested Pydantic schema."""
        if hasattr(text_format, "model_validate_json"):
            return text_format.model_validate_json(text)
        if hasattr(text_format, "parse_raw"):
            return text_format.parse_raw(text)

        data = json.loads(text)
        return text_format(**data)

    def _generate_with_chat_fallback(
        self,
        processed_prompt: List[Dict[str, Any]],
        text_format: Optional[type] = None,
        force_request_path: str = "chat.completions",
        **kwargs: Any,
    ) -> Any:
        """Fallback path for proxies that support chat.completions but not responses."""
        chat_messages = self._convert_prompt_to_chat_messages(processed_prompt)
        request_kwargs = {**self.kwargs, **_reasoning_effort_kwargs(), **kwargs}
        try:
            max_attempts = max(1, int(os.getenv("EM2MEM_CHAT_COMPLETIONS_FALLBACK_ATTEMPTS", "2") or 2))
        except ValueError:
            max_attempts = 2

        def _timed_chat_create(path: str, attempt: int, **create_kwargs: Any) -> Any:
            request_started_at = _debug_utc_now_iso()
            request_start = time.perf_counter()
            try:
                response = self.sync_client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                self._record_debug(
                    path,
                    messages=chat_messages,
                    error=exc,
                    api_timing={
                        "api_request_started_at": request_started_at,
                        "api_response_finished_at": _debug_utc_now_iso(),
                        "api_duration_ms": _debug_duration_ms(request_start),
                        "request_path": path,
                        "model": self.model_name,
                        "attempt": attempt,
                        "attempt_count": max_attempts,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            self._record_debug(
                path,
                messages=chat_messages,
                response=response,
                api_timing={
                    "api_request_started_at": request_started_at,
                    "api_response_finished_at": _debug_utc_now_iso(),
                    "api_duration_ms": _debug_duration_ms(request_start),
                    "request_path": path,
                    "model": self.model_name,
                    "attempt": attempt,
                    "attempt_count": max_attempts,
                },
            )
            return response

        if text_format is not None:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(
                        "[QUERY-LLM-REQUEST] path=%s attempt=%s/%s model=%s messages=%s text_format=true",
                        force_request_path,
                        attempt,
                        max_attempts,
                        self.model_name,
                        len(chat_messages),
                    )
                    try:
                        response = _timed_chat_create(
                            force_request_path,
                            attempt,
                            model=self.model_name,
                            messages=chat_messages,
                            response_format={"type": "json_object"},
                            **request_kwargs,
                        )
                    except Exception:
                        if "reasoning_effort" in request_kwargs or "reasoning" in request_kwargs:
                            request_kwargs = dict(request_kwargs)
                            request_kwargs.pop("reasoning_effort", None)
                            request_kwargs.pop("reasoning", None)
                        response = _timed_chat_create(
                            f"{force_request_path}:without_reasoning",
                            attempt,
                            model=self.model_name,
                            messages=chat_messages,
                            **request_kwargs,
                        )

                    return self._parse_structured_text(
                        self._extract_text_from_response(response),
                        text_format,
                    )
                except Exception as e:
                    last_error = e
                    if not (
                        isinstance(getattr(self, "last_debug", None), dict)
                        and isinstance(self.last_debug.get("api_timing"), dict)
                    ):
                        self._record_debug(force_request_path, messages=chat_messages, error=e)
                    logger.warning(
                        "[QUERY-LLM-ERROR] path=%s attempt=%s/%s model=%s error=%s",
                        force_request_path,
                        attempt,
                        max_attempts,
                        self.model_name,
                        f"{type(e).__name__}: {e}",
                    )
            raise OpenAIModelError(
                f"Failed to get structured completion after {max_attempts} chat.completions attempts: {last_error}"
            )

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "[QUERY-LLM-REQUEST] path=%s attempt=%s/%s model=%s messages=%s text_format=false",
                    force_request_path,
                    attempt,
                    max_attempts,
                    self.model_name,
                    len(chat_messages),
                )
                try:
                    response = _timed_chat_create(
                        force_request_path,
                        attempt,
                        model=self.model_name,
                        messages=chat_messages,
                        **request_kwargs,
                    )
                except Exception as e:
                    if ("reasoning_effort" not in request_kwargs and "reasoning" not in request_kwargs) or not _looks_like_unsupported_reasoning(e):
                        raise
                    request_kwargs = dict(request_kwargs)
                    request_kwargs.pop("reasoning_effort", None)
                    request_kwargs.pop("reasoning", None)
                    logger.info(
                        "[QUERY-LLM-REQUEST] path=%s:without_reasoning attempt=%s/%s model=%s messages=%s text_format=false",
                        force_request_path,
                        attempt,
                        max_attempts,
                        self.model_name,
                        len(chat_messages),
                    )
                    response = _timed_chat_create(
                        f"{force_request_path}:without_reasoning",
                        attempt,
                        model=self.model_name,
                        messages=chat_messages,
                        **request_kwargs,
                    )
                return self._extract_text_from_response(response)
            except Exception as e:
                last_error = e
                if not (
                    isinstance(getattr(self, "last_debug", None), dict)
                    and isinstance(self.last_debug.get("api_timing"), dict)
                ):
                    self._record_debug(force_request_path, messages=chat_messages, error=e)
                logger.warning(
                    "[QUERY-LLM-ERROR] path=%s attempt=%s/%s model=%s error=%s",
                    force_request_path,
                    attempt,
                    max_attempts,
                    self.model_name,
                    f"{type(e).__name__}: {e}",
                )
        raise OpenAIModelError(
            f"Failed to get completion after {max_attempts} chat.completions attempts: {last_error}"
        )

    async def _generate_with_chat_fallback_async(
        self,
        processed_prompt: List[Dict[str, Any]],
        text_format: Optional[type] = None,
        **kwargs: Any,
    ) -> Any:
        """Async fallback path for proxies that support chat.completions but not responses."""
        chat_messages = self._convert_prompt_to_chat_messages(processed_prompt)
        request_kwargs = {**self.kwargs, **_reasoning_effort_kwargs(), **kwargs}

        if text_format is not None:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=chat_messages,
                    response_format={"type": "json_object"},
                    **request_kwargs,
                )
            except Exception:
                if "reasoning_effort" in request_kwargs or "reasoning" in request_kwargs:
                    request_kwargs = dict(request_kwargs)
                    request_kwargs.pop("reasoning_effort", None)
                    request_kwargs.pop("reasoning", None)
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=chat_messages,
                    **request_kwargs,
                )

            return self._parse_structured_text(
                self._extract_text_from_response(response),
                text_format,
            )

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                **request_kwargs,
            )
        except Exception as e:
            if ("reasoning_effort" not in request_kwargs and "reasoning" not in request_kwargs) or not _looks_like_unsupported_reasoning(e):
                raise
            request_kwargs = dict(request_kwargs)
            request_kwargs.pop("reasoning_effort", None)
            request_kwargs.pop("reasoning", None)
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                **request_kwargs,
            )
        return self._extract_text_from_response(response)

    def _extract_stream_delta_text(self, chunk: Any) -> str:
        """Best-effort extraction of streamed token text from a chat completion chunk."""
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    delta = first.get("delta") or {}
                    if isinstance(delta, dict):
                        text = delta.get("content")
                        if isinstance(text, str):
                            return text
                        return self._flatten_content(text)
                    text = first.get("text")
                    if isinstance(text, str):
                        return text
            text = chunk.get("output_text")
            if isinstance(text, str):
                return text
            text = chunk.get("text")
            if isinstance(text, str):
                return text
            return ""
        choices = getattr(chunk, "choices", None)
        if choices:
            first = choices[0]
            delta = getattr(first, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                text = self._flatten_content(content)
                if text:
                    return text
                text = getattr(delta, "text", None)
                if isinstance(text, str):
                    return text
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
        text = getattr(chunk, "output_text", None)
        if isinstance(text, str):
            return text
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            return text
        return ""

    def _stream_with_chat_fallback(
        self,
        processed_prompt: List[Dict[str, Any]],
        text_format: Optional[type] = None,
        on_chunk: Any = None,
        force_request_path: str = "chat.completions:stream",
        **kwargs: Any,
    ) -> Any:
        """Streaming fallback path for proxies that support chat.completions streaming."""
        chat_messages = self._convert_prompt_to_chat_messages(processed_prompt)
        request_kwargs = {**self.kwargs, **_reasoning_effort_kwargs(), **kwargs}
        if text_format is not None:
            request_kwargs = dict(request_kwargs)
            request_kwargs["response_format"] = {"type": "json_object"}

        def _emit(text: str) -> None:
            if not text or on_chunk is None:
                return
            try:
                on_chunk(text)
            except Exception as cb_exc:
                logger.debug("stream chunk callback failed: %s", cb_exc)

        try:
            request_started_at = _debug_utc_now_iso()
            request_start = time.perf_counter()
            first_token_at: str | None = None
            first_token_ms: int | None = None
            try:
                stream = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=chat_messages,
                    stream=True,
                    **request_kwargs,
                )
            except Exception as e:
                if "reasoning_effort" not in request_kwargs and "reasoning" not in request_kwargs or not _looks_like_unsupported_reasoning(e):
                    raise
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop("reasoning_effort", None)
                request_kwargs.pop("reasoning", None)
                request_started_at = _debug_utc_now_iso()
                request_start = time.perf_counter()
                first_token_at = None
                first_token_ms = None
                stream = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=chat_messages,
                    stream=True,
                    **request_kwargs,
                )

            chunks: list[str] = []
            for chunk in stream:
                text = self._extract_stream_delta_text(chunk)
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = _debug_utc_now_iso()
                    first_token_ms = _debug_duration_ms(request_start)
                chunks.append(text)
                _emit(text)

            final_text = "".join(chunks).strip()
            if not final_text:
                raise OpenAIModelError("Empty streaming response from model.")
            response_finished_at = _debug_utc_now_iso()
            api_duration_ms = _debug_duration_ms(request_start)
            self._record_debug(
                force_request_path,
                messages=chat_messages,
                response={
                    "streaming": True,
                    "chunk_count": len(chunks),
                    "output_text_preview": final_text[:1000],
                },
                api_timing={
                    "api_request_started_at": request_started_at,
                    "api_first_token_at": first_token_at,
                    "api_response_finished_at": response_finished_at,
                    "api_duration_ms": api_duration_ms,
                    "stream_first_token_ms": first_token_ms,
                    "request_path": force_request_path,
                    "model": self.model_name,
                    "attempt": 1,
                    "attempt_count": 1,
                },
            )
            if text_format is not None:
                return self._parse_structured_text(final_text, text_format)
            return final_text
        except Exception as e:
            self._record_debug(
                force_request_path,
                messages=chat_messages,
                error=e,
                api_timing={
                    "api_request_started_at": locals().get("request_started_at"),
                    "api_first_token_at": locals().get("first_token_at"),
                    "api_response_finished_at": _debug_utc_now_iso(),
                    "api_duration_ms": _debug_duration_ms(locals().get("request_start", time.perf_counter())),
                    "stream_first_token_ms": locals().get("first_token_ms"),
                    "request_path": force_request_path,
                    "model": self.model_name,
                    "attempt": 1,
                    "attempt_count": 1,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise

    @async_cache_response
    @dynamic_retry_decorator
    async def _generate_single_prompt(self, prompt: Union[str, List[Dict[str, Any]]], text_format: Optional[type] = None) -> Any:
        """Generate completion for a single prompt asynchronously. Supports optional structured parsing via `text_format`."""
        try:
            prompt = self._normalize_prompt(prompt)
            if _env_bool("EM2MEM_OPENAI_FORCE_CHAT_COMPLETIONS", True):
                processed_prompt = self._preprocess_prompt(copy.deepcopy(prompt))
                return await self._generate_with_chat_fallback_async(processed_prompt, text_format=text_format)
            if text_format is not None:
                response = await self.async_client.responses.parse(
                    model=self.model_name,
                    input=prompt,
                    text_format=text_format,
                    **self.kwargs,
                )
                return getattr(response, "output_parsed", None)

            response = await self.async_client.responses.create(
                model=self.model_name,
                input=prompt,
                **self.kwargs,
            )
            # return response.output_text.strip()
            return self._extract_text_from_response(response)
        except Exception as e:
            logger.warning(f"Responses API async path failed, retrying with chat.completions: {e}")
            try:
                return await self._generate_with_chat_fallback_async(prompt, text_format=text_format)
            except Exception as chat_e:
                logger.error(f"Async OpenAI API error: {chat_e}")
                raise OpenAIModelError(
                    f"Failed to get async completion. responses_error={e}; chat_error={chat_e}"
                ) from chat_e

    async def async_generation(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], chunk_size: int = 50, text_format: Optional[type] = None) -> List[Any]:
        """
        Generate completions for multiple prompts asynchronously with chunking.
        
        Args:
            batch_prompts: List of conversation prompts
            chunk_size: Number of concurrent requests per chunk
            text_format (optional): Pydantic model or other structure for parsing the response

        Returns:
            List of response strings
        """
        responses = []
        total_chunks = (len(batch_prompts) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(batch_prompts), chunk_size):
            chunk_num = (i // chunk_size) + 1
            batch = batch_prompts[i:i + chunk_size]
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(batch)} prompts)")
            
            tasks = [self._generate_single_prompt(prompt, text_format) for prompt in batch]
            try:
                batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Chunk {chunk_num}")
                responses.extend(batch_responses)
            except Exception as e:
                logger.error(f"Error in chunk {chunk_num}: {e}")
                raise

        return responses

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get current cache statistics."""
        return {
            "cache_size": len(_CACHE),
            "max_cache_size": _MAX_CACHE_SIZE,
            "cache_hit_rate": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }

    def clear_cache(self) -> None:
        """Clear the global cache."""
        _CACHE.clear()
        logger.info("Cache cleared")

    def __repr__(self) -> str:
        """String representation of the model instance."""
        return (f"OpenAIModel(model_name='{self.model_name}', "
                f"kwargs={self.kwargs})")


# Convenience function for backward compatibility
def create_openai_model(model_name: str, **kwargs) -> OpenAIModel:
    """Create an OpenAI model instance with convenient defaults."""
    return OpenAIModel(model_name=model_name, **kwargs)
