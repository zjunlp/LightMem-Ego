from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from online_current.mcur_store import MCurStore
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_preprocess.task_queue import enqueue_mst_refine_task
from online_short_term.frame_diff_detector import FrameDiffEventDetector
from online_short_term.mst_store import MSTStore
from online_short_term.schemas import build_retrieval_text, mst_event_stub, rel_to_session
from online_short_term.stream_chunk_manager import StreamChunkManager
from online_short_term.transcript_aligner import TranscriptAligner


class MicroEventBuilder:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_id = session_dir.name
        self.detector = FrameDiffEventDetector(session_dir)
        self.aligner = TranscriptAligner(session_dir)
        self.store = MSTStore(session_dir)

    def process_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        chunk_path = self.session_dir / str(chunk.get("path", ""))
        if not chunk_path.exists():
            raise FileNotFoundError(f"chunk not found: {chunk_path}")
        chunk_id = str(chunk.get("chunk_id") or chunk_path.stem)
        start = float(chunk.get("start_time", 0.0))
        end = float(chunk.get("end_time", start + float(chunk.get("duration", 0.0))))
        detected = self.detector.detect(
            chunk_path=chunk_path,
            chunk_id=chunk_id,
            chunk_global_start_time=start,
            chunk_global_end_time=end,
        )
        events = []
        for boundary_index, boundary in enumerate(detected.get("events", []) or []):
            event = mst_event_stub(
                session_id=self.session_id,
                chunk_id_value=chunk_id,
                start_time=float(boundary["start_time"]),
                end_time=float(boundary["end_time"]),
                boundary_reason=str(boundary.get("boundary_reason") or "chunk_end"),
                diff_score=float(boundary.get("diff_score") or 0.0),
                diff_stats=dict(boundary.get("diff_stats") or {}),
                chunk_path=rel_to_session(self.session_dir, chunk_path),
                boundary_index=boundary_index,
            )
            aligned = self.aligner.align(event["start_time"], event["end_time"])
            event["transcript"] = aligned["transcript"]
            event["transcript_segments"] = aligned["transcript_segments"]
            event["keyframes"] = boundary.get("keyframes", [])
            event["retrieval_text"] = build_retrieval_text(event)
            events.append(event)
        return self.store.append_events(events)

    def process_stream_chunk(
        self,
        chunk: dict[str, Any],
        *,
        project_root: Path | None = None,
        enqueue_refine: bool = True,
    ) -> dict[str, Any]:
        manager = StreamChunkManager(self.session_dir)
        chunk_path = self.session_dir / str(chunk.get("path") or chunk.get("chunk_path") or "")
        if not chunk_path.exists():
            raise FileNotFoundError(f"chunk not found: {chunk_path}")
        chunk_id = str(chunk.get("chunk_id") or chunk_path.stem)
        if chunk_id.startswith("upload_") or str(chunk.get("path") or chunk.get("chunk_path") or "").startswith("stream/upload_chunks/"):
            raise RuntimeError(f"process_stream_chunk requires a processing chunk, got upload chunk: {chunk_id}")
        if not chunk_id.startswith("proc_") and "processing_chunk_id" not in chunk:
            raise RuntimeError(f"process_stream_chunk requires proc_* timeline chunk, got: {chunk_id}")
        start = float(chunk.get("start_time", 0.0) or 0.0)
        end = float(chunk.get("end_time", start + float(chunk.get("duration", 0.0) or 0.0)) or start)
        event_state = manager.load_event_state()
        stream_state = manager.load_stream_state(default={})
        event_state.setdefault("stream_id", stream_state.get("stream_id"))
        detected = self.detector.detect_stream_chunk(
            chunk_path=chunk_path,
            chunk_id=chunk_id,
            chunk_global_start_time=start,
            chunk_global_end_time=end,
            previous_state=event_state,
        )
        frames = list(detected.get("candidate_frames", []) or [])
        diff_records = list(detected.get("diff_records", []) or [])
        self._append_diff_records(diff_records)

        window_start = max(0.0, end - MCurStore(self.session_dir).window_seconds)
        aligned_current = self.aligner.align(window_start, end)
        diff_scores = [float(frame.get("diff_score") or 0.0) for frame in frames]
        diff_stats = {
            "max_diff": max(diff_scores) if diff_scores else 0.0,
            "mean_diff": sum(diff_scores) / len(diff_scores) if diff_scores else 0.0,
            "last_diff": diff_scores[-1] if diff_scores else 0.0,
        }
        mcur_store = MCurStore(self.session_dir)
        mcur_state = mcur_store.update_from_chunk(
            chunk_info={**chunk, "chunk_id": chunk_id, "start_time": start, "end_time": end, "path": rel_to_session(self.session_dir, chunk_path)},
            frames=frames,
            transcript_segments=aligned_current.get("transcript_segments", []),
            diff_stats=diff_stats,
        )

        new_event_state = dict(detected.get("new_event_state") or {})
        new_event_state.setdefault("session_id", self.session_id)
        new_event_state.setdefault("stream_id", stream_state.get("stream_id"))
        manager.save_event_state(new_event_state)
        self._attach_stream_open_event_to_mcur(new_event_state.get("open_event"))

        events = self._closed_boundaries_to_events(
            boundaries=list(detected.get("closed_events", []) or []),
            chunk_id=chunk_id,
            chunk_path=chunk_path,
        )
        appended = self.store.append_events(events)
        refine_task_path = None
        if appended and enqueue_refine and project_root is not None:
            refine_task_path = enqueue_mst_refine_task(
                project_root=Path(project_root),
                session_id=self.session_id,
                backend=os.getenv("EM2MEM_MST_REFINE_BACKEND", "openai"),
                limit_events=int(os.getenv("EM2MEM_MST_REFINE_LIMIT_EVENTS", "10")),
                force_refine=False,
            )
        return {
            "session_id": self.session_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk.get("chunk_index"),
            "proc_index": chunk.get("proc_index", chunk.get("chunk_index")),
            "upload_chunk_index": chunk.get("upload_chunk_index") or chunk.get("source_upload_chunk_index"),
            "candidate_frame_count": len(frames),
            "diff_record_count": len(diff_records),
            "closed_event_count": len(appended),
            "closed_event_ids": [event.get("event_id") for event in appended],
            "has_open_event": bool(new_event_state.get("open_event")),
            "mcur_state": mcur_state,
            "refine_task_path": str(refine_task_path) if refine_task_path else None,
            "reused_frame_extraction": True,
        }

    def close_stream_open_event(
        self,
        *,
        project_root: Path | None = None,
        enqueue_refine: bool = True,
        reason: str = "stream_end",
    ) -> dict[str, Any]:
        manager = StreamChunkManager(self.session_dir)
        event_state = manager.load_event_state()
        closed, new_state = self.detector.close_stream_open_event(event_state, reason=reason)
        new_state.setdefault("session_id", self.session_id)
        new_state.setdefault("stream_id", event_state.get("stream_id"))
        manager.save_event_state(new_state)
        self._attach_stream_open_event_to_mcur(None)
        events = self._closed_boundaries_to_events(
            boundaries=closed,
            chunk_id=self._source_chunks_label(closed[0].get("source_chunks", [])) if closed else "stream_end",
            chunk_path=None,
        )
        appended = self.store.append_events(events)
        refine_task_path = None
        if appended and enqueue_refine and project_root is not None:
            refine_task_path = enqueue_mst_refine_task(
                project_root=Path(project_root),
                session_id=self.session_id,
                backend=os.getenv("EM2MEM_MST_REFINE_BACKEND", "openai"),
                limit_events=int(os.getenv("EM2MEM_MST_REFINE_LIMIT_EVENTS", "10")),
                force_refine=False,
            )
        return {
            "session_id": self.session_id,
            "closed_event_count": len(appended),
            "closed_event_ids": [event.get("event_id") for event in appended],
            "has_open_event": False,
            "refine_task_path": str(refine_task_path) if refine_task_path else None,
            "boundary_reason": reason,
        }

    def _closed_boundaries_to_events(
        self,
        *,
        boundaries: list[dict[str, Any]],
        chunk_id: str,
        chunk_path: Path | None,
    ) -> list[dict[str, Any]]:
        events = []
        chunk_rel = rel_to_session(self.session_dir, chunk_path) if chunk_path is not None else ""
        for idx, boundary in enumerate(boundaries):
            source_chunks = list(boundary.get("source_chunks") or [])
            chunk_id_value = self._source_chunks_label(source_chunks) if source_chunks else chunk_id
            boundary_index = int(boundary.get("boundary_index", idx) or idx)
            event = mst_event_stub(
                session_id=self.session_id,
                chunk_id_value=chunk_id_value,
                start_time=float(boundary["start_time"]),
                end_time=float(boundary["end_time"]),
                boundary_reason=str(boundary.get("boundary_reason") or "visual_change"),
                diff_score=float(boundary.get("diff_score") or 0.0),
                diff_stats=dict(boundary.get("diff_stats") or {}),
                chunk_path=chunk_rel,
                boundary_index=boundary_index,
            )
            aligned = self.aligner.align(event["start_time"], event["end_time"])
            event["transcript"] = aligned["transcript"]
            event["transcript_segments"] = aligned["transcript_segments"]
            event["keyframes"] = boundary.get("keyframes", [])
            event["source_chunks"] = source_chunks
            event["cross_chunk"] = bool(boundary.get("cross_chunk") or len(source_chunks) > 1)
            event["source"] = {
                **dict(event.get("source") or {}),
                "type": "stream_frame_diff",
                "chunk_path": chunk_rel,
                "source_chunks": source_chunks,
                "cross_chunk": event["cross_chunk"],
            }
            event["retrieval_text"] = build_retrieval_text(event)
            events.append(event)
        return events

    def _append_diff_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        path = self.session_dir / "stream" / "diff_records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                payload = {**record, "session_id": self.session_id, "created_at": now}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _attach_stream_open_event_to_mcur(self, open_event: Any) -> None:
        path = self.session_dir / "current" / "open_event.json"
        current = read_json(path, default={})
        if not isinstance(current, dict):
            current = {}
        if open_event:
            current["stream_open_event"] = open_event
        else:
            current.pop("stream_open_event", None)
        if current:
            write_json_atomic(path, current)

    def _source_chunks_label(self, source_chunks: list[Any]) -> str:
        chunks = [str(item) for item in source_chunks if item]
        if not chunks:
            return "stream"
        if len(chunks) == 1:
            return chunks[0]
        return f"{chunks[0]}..{chunks[-1]}"

    def _build_retrieval_text(self, event: dict[str, Any]) -> str:
        return build_retrieval_text(event)
