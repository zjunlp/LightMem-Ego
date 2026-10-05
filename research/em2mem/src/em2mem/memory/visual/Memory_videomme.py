# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import os
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class VideoClipEntry_videomme:
    """Represents a single video clip entry with its metadata."""
    id: str
    video_path: str
    start_time: str
    end_time: str
    date: str
    clip_start_sec: Optional[float] = None
    clip_end_sec: Optional[float] = None
    embedding: Optional[np.ndarray] = None  # Precomputed embedding
    
    @property
    def timestamp_int(self) -> Tuple[int, int]:
        """Convert start and end times to integer format (day + time.zfill(8))."""
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts
    
    def to_display_str(self) -> str:
        """Format video clip for display with time range."""
        start_ts, end_ts = self.timestamp_int
        return f"{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}"


@dataclass
class FrameEntry_videomme:
    """Represents a single frame from a video."""
    video_path: str
    frame_index: int
    timestamp_sec: float
    frame: Optional[Image.Image] = None


def _transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


def _load_json(file_path: str) -> Any:
    """Load JSON file."""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)


def _time_str_to_seconds(time_str: str) -> float:
    """Convert HHMMSSFF time string to seconds."""
    time_str = str(time_str).zfill(8)
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    return float(hours * 3600 + minutes * 60 + seconds)


def _parse_time_range(time_range: str) -> Tuple[int, int]:
    """
    Parse a time range string into start and end timestamps.
    
    Expected format: "DAY X HH:MM:SS - DAY Y HH:MM:SS"
    Example: "DAY1 11:09:43 - DAY1 11:09:58"
    
    Args:
        time_range: Time range string
        
    Returns:
        Tuple of (start_timestamp, end_timestamp) as integers
    """
    import re
    
    # Pattern to match "DAY X HH:MM:SS"
    pattern = r'DAY\s*(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})'
    matches = re.findall(pattern, time_range, re.IGNORECASE)
    
    if len(matches) < 2:
        raise ValueError(f"Invalid time range format: {time_range}. Expected 'DAY X HH:MM:SS - DAY Y HH:MM:SS'")
    
    # Parse start time
    start_day, start_hh, start_mm, start_ss = matches[0]
    start_ts = int(f"{start_day}{start_hh.zfill(2)}{start_mm.zfill(2)}{start_ss.zfill(2)}00")
    
    # Parse end time
    end_day, end_hh, end_mm, end_ss = matches[1]
    end_ts = int(f"{end_day}{end_hh.zfill(2)}{end_mm.zfill(2)}{end_ss.zfill(2)}00")
    
    return start_ts, end_ts


def _is_time_range_query(query: str) -> bool:
    """
    Check if a query string is a time range query.
    
    Args:
        query: Query string to check
        
    Returns:
        True if query matches time range format
    """
    import re
    pattern = r'DAY\s*\d+\s+\d{1,2}:\d{2}:\d{2}\s*-\s*DAY\s*\d+\s+\d{1,2}:\d{2}:\d{2}'
    return bool(re.search(pattern, query, re.IGNORECASE))


class VisualMemory_videomme:
    """
    Visual Memory module that implements embedding-based video retrieval.
    
    This class manages precomputed visual embeddings for 30-second video clips
    and provides retrieval functionality based on embedding similarity.
    
    Two retrieval modes:
    1. Query-based (text): Given a text query, find the most similar video clips
       using precomputed embeddings (for 30-sec clips)
    2. Query-based (timestamp): Given a timestamp, retrieve 1fps frames from
       the corresponding video
    
    Attributes:
        embedding_model: Model for computing query embeddings
        clips: List of all VideoClipEntry objects
        indexed_entries: List of entries indexed up to indexed_time
        indexed_time: Timestamp boundary for indexed clips
        embeddings: Tensor of precomputed embeddings for indexed entries
        video_path_to_embedding: Dict mapping video_path to precomputed embedding
    """
    
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize VisualMemory.
        
        Args:
            embedding_model: Embedding model for computing query embeddings (optional)
        """
        self.embedding_model = embedding_model
        
        # Storage for video clips
        self.clips: List[VideoClipEntry_videomme] = []
        self.clip_id_to_entry: Dict[str, VideoClipEntry_videomme] = {}
        
        # Precomputed embeddings from pickle file
        self.embedding_lookup: Dict[str, np.ndarray] = {}
        
        # Indexed state
        self.indexed_entries: List[VideoClipEntry_videomme] = []
        self.indexed_time: int = 0
        self.embeddings: Optional[torch.Tensor] = None
        self.index_to_pos: Dict[str, int] = {}
    
    def load_embeddings_from_file(self, embeddings_path: str) -> None:
        """
        Load precomputed video embeddings from a pickle file.
        
        Expected format: Dict[video_path, np.ndarray]
        
        Args:
            embeddings_path: Path to pickle file containing video embeddings
        """
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        with open(embeddings_path, 'rb') as f:
            self.embedding_lookup = pickle.load(f)
        
        logger.info(f"Loaded {len(self.embedding_lookup)} video embeddings from {embeddings_path}")
    
    def load_clips_from_file(self, clips_path: str) -> None:
        """
        Load video clip metadata from a JSON file.
        
        Expected format: List of dicts with keys: start_time, end_time, date, video_path, text (optional)
        
        Args:
            clips_path: Path to JSON file containing clip metadata
        """
        data = _load_json(clips_path)
        self.load_clips_from_data(data)
    
    def load_clips_from_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Load video clip metadata from in-memory data.
        
        Args:
            data: List of dicts with keys: start_time, end_time, date, video_path, text (optional)
        """
        for idx, entry in enumerate(data):
            clip_id = entry.get("clip_id") or entry.get("id") or f"visual_{idx}"
            video_path = entry.get("source_video_path") or entry.get("video_path", "")
            
            # Get precomputed embedding if available (try clip_id, video_path, start_time)
            embedding = self.embedding_lookup.get(clip_id)
            if embedding is None:
                embedding = self.embedding_lookup.get(video_path)
            if embedding is None:
                start_time_key = str(entry.get("start_time", ""))
                embedding = self.embedding_lookup.get(start_time_key)
            
            clip_start_sec = entry.get("start_sec")
            clip_end_sec = entry.get("end_sec")
            if clip_start_sec is None and entry.get("start_time"):
                clip_start_sec = _time_str_to_seconds(str(entry["start_time"]))
            if clip_end_sec is None and entry.get("end_time"):
                clip_end_sec = _time_str_to_seconds(str(entry["end_time"]))

            clip_entry = VideoClipEntry_videomme(
                id=clip_id,
                video_path=video_path,
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=entry.get("date", ""),
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                embedding=embedding,
            )
            self.clips.append(clip_entry)
            self.clip_id_to_entry[clip_id] = clip_entry
        
        # Sort clips by timestamp for efficient indexing
        self.clips.sort(key=lambda c: c.timestamp_int[0])
        logger.info(f"Loaded {len(self.clips)} video clips")
    
    def index(self, until_time: int) -> None:
        """
        Index video clips up to the specified timestamp.
        
        This builds the embedding tensor for clips with end_time <= until_time.
        Only clips with precomputed embeddings are included.
        
        Args:
            until_time: Timestamp boundary - index all clips with end_time <= this value
        """
        # Skip if already indexed beyond this time
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
            return
        
        # Get entries to index (with valid embeddings)
        entries_to_index = [
            entry for entry in self.clips
            if entry.timestamp_int[1] <= until_time and entry.embedding is not None
        ]
        
        if not entries_to_index:
            logger.debug(f"No entries to index up to {until_time}")
            return
        
        # Build embedding tensor
        all_embeddings = []
        self.index_to_pos = {}
        
        for pos, entry in enumerate(entries_to_index):
            self.index_to_pos[entry.id] = pos
            all_embeddings.append(entry.embedding)
        
        self.embeddings = torch.tensor(
            np.array(all_embeddings),
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        
        logger.info(f"Indexed {len(entries_to_index)} video clips up to {until_time}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        fps: float = 1.0,
        max_frames: int = 50,
        as_context: bool = True,
    ) -> Union[List[VideoClipEntry_videomme], List[FrameEntry_videomme], Dict[str, List[Image.Image]]]:
        """
        Retrieve visual content based on query type.
        
        - If query is a time range (e.g., "DAY1 11:09:43 - DAY1 11:09:58"): 
          Retrieve frames from videos within that time range
        - If query is natural language: Retrieve similar video clips using embedding similarity
        
        Args:
            query: Either a time range string or natural language query
            top_k: Number of clips to retrieve (for similarity search)
            fps: Frames per second to extract (for time range query, default: 1fps)
            max_frames: Maximum number of frames (for time range query, default: 64)
            as_context: If True, return Dict[str, List[Image.Image]] where key is display time range
            
        Returns:
            For time range query: List of FrameEntry objects (or Dict if as_context=True)
            For text query: List of VideoClipEntry objects (or Dict if as_context=True)
        """
        # Check if query is a time range
        if _is_time_range_query(query):
            frames = self._retrieve_by_time_range(
                time_range=query,
                fps=fps,
                max_frames=max_frames,
            )
            if as_context:
                # Apply uniform sampling if needed
                if len(frames) > max_frames:
                    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
                    frames = [frames[i] for i in indices]
                images = [f.frame for f in frames if f.frame is not None]
                return {query: images}
            return frames
        
        # Similarity-based retrieval for natural language query
        return self._retrieve_by_similarity(
            query=query,
            top_k=top_k,
            fps=fps,
            max_frames=max_frames,
            as_context=as_context,
        )
    
    def _retrieve_by_similarity(
        self,
        query: str,
        top_k: int = 5,
        fps: float = 1.0,
        max_frames: int = 50,
        as_context: bool = False,
    ) -> Union[List[VideoClipEntry_videomme], Dict[str, List[Image.Image]]]:
        """
        Retrieve top-k similar video clips using embedding similarity.
        
        Args:
            query: Text query to encode (uses embedding_model.encode_vis_query)
            top_k: Number of clips to retrieve
            fps: Frames per second to extract (when as_context=True)
            max_frames: Maximum number of frames (when as_context=True)
            as_context: If True, return Dict[str, List[Image.Image]]
            
        Returns:
            List of VideoClipEntry objects or Dict[str, List[Image.Image]]
        """
        if not self.indexed_entries or self.embeddings is None:
            logger.warning("No clips indexed. Call index(until_time) before retrieve().")
            return {} if as_context else []
        
        if self.embedding_model is None:
            raise ValueError("embedding_model is required for similarity-based retrieval")
        
        device = self.embeddings.device
        
        # Use visual query encoding for cross-modal retrieval
        q_emb = self.embedding_model.encode_vis_query(query)
        
        # Ensure proper shape
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        
        query_tensor = torch.tensor(q_emb, dtype=torch.float32, device=device)
        
        # Compute similarities
        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        
        # Get top-k
        num_available = len(self.indexed_entries)
        k = min(top_k, num_available)
        top_values, top_indices = torch.topk(similarities, k)
        
        results = [self.indexed_entries[idx] for idx in top_indices.cpu().tolist()]
        
        if as_context:
            # Extract frames from all retrieved clips, organized by clip
            frames_by_clip: Dict[str, List[FrameEntry_videomme]] = {}
            for clip in results:
                frames = self._extract_frames(
                    clip.video_path,
                    fps=fps,
                    max_frames=None,
                    start_sec=clip.clip_start_sec,
                    end_sec=clip.clip_end_sec,
                )
                display_key = clip.to_display_str()
                frames_by_clip[display_key] = frames
            
            return self._frames_to_context_dict(frames_by_clip, max_frames)
        
        return results
    
    def _retrieve_by_time_range(
        self,
        time_range: str,
        fps: float = 1.0,
        max_frames: int = 50,
    ) -> List[FrameEntry_videomme]:
        """
        Retrieve frames from videos within the given time range.
        
        Only extracts frames that fall within the query time range,
        even if the video clip is longer (e.g., 30-second clips).
        
        Args:
            time_range: Time range string (e.g., "DAY1 11:09:43 - DAY1 11:09:58")
            fps: Frames per second to extract (default: 1fps)
            max_frames: Maximum number of frames (default: 64, uses uniform sampling if exceeded)
            
        Returns:
            List of FrameEntry objects with extracted frames
        """
        try:
            start_ts, end_ts = _parse_time_range(time_range)
        except ValueError as e:
            logger.error(str(e))
            return []
        
        # Find all clips that overlap with the time range
        matching_clips = []
        for clip in self.clips:
            clip_start, clip_end = clip.timestamp_int
            # Check if clip overlaps with the query time range
            if clip_start <= end_ts and clip_end >= start_ts:
                matching_clips.append(clip)
        
        if not matching_clips:
            logger.warning(f"No clips found for time range {time_range}")
            return []
        
        # Sort clips by start time
        matching_clips.sort(key=lambda c: c.timestamp_int[0])
        
        # Extract frames from all matching clips, only within the query time range
        all_frames: List[FrameEntry_videomme] = []
        for clip in matching_clips:
            clip_start, clip_end = clip.timestamp_int
            
            # Calculate the overlap between query range and clip range
            overlap_start = max(start_ts, clip_start)
            overlap_end = min(end_ts, clip_end)
            
            # Convert to seconds relative to clip start
            # Timestamp format: DHHMMSS00 (day + HHMMSS + 00)
            # We need to calculate seconds from clip start
            start_sec = self._timestamp_diff_seconds(clip_start, overlap_start)
            end_sec = self._timestamp_diff_seconds(clip_start, overlap_end)
            
            frames = self._extract_frames(
                clip.video_path,
                fps=fps,
                max_frames=None,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            all_frames.extend(frames)
        
        # Apply uniform sampling if total frames exceed max_frames
        if max_frames is not None and len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int).tolist()
            all_frames = [all_frames[i] for i in indices]
        
        return all_frames
    
    def _timestamp_diff_seconds(self, ts_from: int, ts_to: int) -> float:
        """
        Calculate the difference in seconds between two timestamps.
        
        Timestamp format: DHHMMSS00 (day digit + HHMMSS + 00)
        
        Args:
            ts_from: Start timestamp
            ts_to: End timestamp
            
        Returns:
            Difference in seconds
        """
        def parse_ts(ts: int) -> Tuple[int, int, int, int]:
            ts_str = str(ts)
            day = int(ts_str[0])
            hh = int(ts_str[1:3])
            mm = int(ts_str[3:5])
            ss = int(ts_str[5:7])
            return day, hh, mm, ss
        
        d1, h1, m1, s1 = parse_ts(ts_from)
        d2, h2, m2, s2 = parse_ts(ts_to)
        
        total_sec_from = d1 * 86400 + h1 * 3600 + m1 * 60 + s1
        total_sec_to = d2 * 86400 + h2 * 3600 + m2 * 60 + s2
        
        return float(total_sec_to - total_sec_from)
    
    def _frames_to_context_dict(
        self,
        frames_by_clip: Dict[str, List[FrameEntry_videomme]],
        max_frames: int,
    ) -> Dict[str, List[Image.Image]]:
        """
        Convert frames by clip to context dict with PIL Images.
        
        Applies uniform sampling if total frames exceed max_frames.
        
        Args:
            frames_by_clip: Dict mapping display key to list of FrameEntry
            max_frames: Maximum total number of frames
            
        Returns:
            Dict mapping display time range to list of PIL Images
        """
        # Count total frames
        total_frames = sum(len(frames) for frames in frames_by_clip.values())
        
        # Calculate sampling ratio if needed
        if total_frames > max_frames:
            sample_ratio = max_frames / total_frames
        else:
            sample_ratio = 1.0
        
        result: Dict[str, List[Image.Image]] = {}
        for display_key, frames in frames_by_clip.items():
            if sample_ratio < 1.0:
                # Apply proportional sampling to each clip
                num_to_keep = max(1, int(len(frames) * sample_ratio))
                indices = np.linspace(0, len(frames) - 1, num_to_keep, dtype=int).tolist()
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = frames
            
            images = [f.frame for f in sampled_frames if f.frame is not None]
            if images:
                result[display_key] = images
        
        return result
    
    def _extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: Optional[int] = 50,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> List[FrameEntry_videomme]:
        """
        Extract frames from a video at the specified fps.
        
        If the number of frames at the given fps exceeds max_frames,
        uniformly samples max_frames from the video instead.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (default: 1fps)
            max_frames: Maximum number of frames (default: 64, None for no limit)
            start_sec: Start time in seconds (None for beginning of video)
            end_sec: End time in seconds (None for end of video)
            
        Returns:
            List of FrameEntry objects
        """
        try:
            from decord import VideoReader, cpu
        except ImportError as e:
            raise ImportError("decord is required for frame extraction.") from e
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []
        
        frames = []
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            total_frames = len(vr)
            video_duration = total_frames / video_fps if video_fps > 0 else 0
            
            # Determine frame range based on start_sec and end_sec
            if start_sec is None:
                start_sec = 0.0
            if end_sec is None:
                end_sec = video_duration
            
            # Clamp to valid range
            start_sec = max(0.0, min(start_sec, video_duration))
            end_sec = max(start_sec, min(end_sec, video_duration))
            
            # Convert to frame indices
            start_frame = int(start_sec * video_fps)
            end_frame = int(end_sec * video_fps)
            end_frame = min(end_frame, total_frames - 1)
            
            if start_frame >= end_frame:
                return []
            
            # Calculate frame interval for target fps
            frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
            frame_interval = max(1, frame_interval)
            
            # Calculate frame indices at target fps within the range
            frame_indices = list(range(start_frame, end_frame + 1, frame_interval))
            
            # If exceeds max_frames, use uniform sampling instead
            if max_frames is not None and len(frame_indices) > max_frames:
                # Uniform sampling across the specified range
                frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int).tolist()
            
            if not frame_indices:
                return []
            
            # Batch read frames using decord
            video_frames = vr.get_batch(frame_indices).asnumpy()
            
            for i, frame_idx in enumerate(frame_indices):
                # decord returns RGB frames directly
                pil_frame = Image.fromarray(video_frames[i])
                
                timestamp_sec = frame_idx / video_fps if video_fps > 0 else 0
                
                frames.append(FrameEntry_videomme(
                    video_path=video_path,
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    frame=pil_frame,
                ))
        
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return []
        
        logger.debug(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def get_clip_by_id(self, clip_id: str) -> Optional[VideoClipEntry_videomme]:
        """Get a clip entry by its ID."""
        return self.clip_id_to_entry.get(clip_id)
    
    def get_clip_by_video_path(self, video_path: str) -> Optional[VideoClipEntry_videomme]:
        """Get a clip entry by its video path."""
        for clip in self.clips:
            if clip.video_path == video_path:
                return clip
        return None
    
    def cleanup(self) -> None:
        """Explicitly free GPU memory."""
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_index(self) -> None:
        """Reset the indexed state, clearing embeddings."""
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.index_to_pos = {}
        logger.info("Index reset - embeddings cleared")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time boundary as human-readable string."""
        return _transform_timestamp(str(self.indexed_time))
    
    def get_clips_count(self) -> int:
        """Get the total number of loaded clips."""
        return len(self.clips)
    
    def get_indexed_count(self) -> int:
        """Get the number of indexed clips."""
        return len(self.indexed_entries)
    
    def get_clips_with_embeddings_count(self) -> int:
        """Get the number of clips that have precomputed embeddings."""
        return sum(1 for clip in self.clips if clip.embedding is not None)
