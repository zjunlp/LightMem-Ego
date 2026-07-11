from __future__ import annotations

from pathlib import Path
from typing import Any

from online_current.mcur_store import MCurStore
from online_current.schemas import env_float
from online_short_term.frame_diff_detector import FrameDiffEventDetector
from online_short_term.stream_chunk_manager import discover_chunks
from online_short_term.transcript_aligner import TranscriptAligner


class MCurUpdater:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.store = MCurStore(self.session_dir)
        self.detector = FrameDiffEventDetector(
            self.session_dir,
            candidate_fps=env_float("WORLDMM_MCUR_CANDIDATE_FPS", 1.0),
        )
        self.aligner = TranscriptAligner(self.session_dir)

    def update_from_stream_chunk(
        self,
        session_id: str | None = None,
        chunk_path: Path | str | None = None,
        chunk_start: float | None = None,
        chunk_end: float | None = None,
        chunk_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del session_id
        if chunk_info is None:
            if chunk_path is None:
                raise ValueError("chunk_path or chunk_info is required")
            chunk_abs = Path(chunk_path)
            if not chunk_abs.is_absolute():
                chunk_abs = self.session_dir / chunk_abs
            start = float(chunk_start or 0.0)
            end = float(chunk_end if chunk_end is not None else start)
            chunk_info = {
                "chunk_id": chunk_abs.stem,
                "start_time": start,
                "end_time": end,
                "duration": max(0.0, end - start),
                "path": str(chunk_abs.relative_to(self.session_dir)) if chunk_abs.is_relative_to(self.session_dir) else str(chunk_abs),
            }
        chunk_abs = self.session_dir / str(chunk_info.get("path", ""))
        if not chunk_abs.exists():
            raise FileNotFoundError(f"chunk not found: {chunk_abs}")
        chunk_id = str(chunk_info.get("chunk_id") or chunk_abs.stem)
        start = float(chunk_info.get("start_time") or 0.0)
        end = float(chunk_info.get("end_time") or start + float(chunk_info.get("duration") or 0.0))
        detected = self.detector.detect(
            chunk_path=chunk_abs,
            chunk_id=chunk_id,
            chunk_global_start_time=start,
            chunk_global_end_time=end,
        )
        frames = list(detected.get("frames", []) or [])
        window_start = max(0.0, end - self.store.window_seconds)
        aligned = self.aligner.align(window_start, end)
        diff_scores = [float(frame.get("diff_score") or 0.0) for frame in frames]
        diff_stats = {
            "max_diff": max(diff_scores) if diff_scores else 0.0,
            "mean_diff": sum(diff_scores) / len(diff_scores) if diff_scores else 0.0,
            "last_diff": diff_scores[-1] if diff_scores else 0.0,
        }
        return self.store.update_from_chunk(
            chunk_info={**chunk_info, "start_time": start, "end_time": end},
            frames=frames,
            transcript_segments=aligned.get("transcript_segments", []),
            diff_stats=diff_stats,
        )

    def update_from_existing_stream(self, force: bool = False, limit_chunks: int | None = None) -> dict[str, Any]:
        if force:
            self.store.clear()
        chunks = discover_chunks(self.session_dir)
        if limit_chunks is not None:
            chunks = chunks[: max(0, int(limit_chunks))]
        state: dict[str, Any] = self.store.get_state()
        for chunk in chunks:
            state = self.update_from_stream_chunk(chunk_info=chunk)
        return state
