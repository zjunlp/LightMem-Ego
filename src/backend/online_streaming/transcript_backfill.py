from __future__ import annotations

import json
import math
import os
import hashlib
from pathlib import Path
from typing import Any

from online_current.mcur_store import MCurStore
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_preprocess.task_queue import enqueue_mst_consolidation_task, enqueue_mst_refine_task
from online_short_term.mst_store import MSTStore
from online_short_term.refine_status import write_refine_status
from online_short_term.schemas import build_retrieval_text
from online_short_term.stream_chunk_manager import StreamChunkManager


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return max(float(start_a), float(start_b)) <= min(float(end_a), float(end_b))


def transcript_dirty_windows_path(session_dir: Path) -> Path:
    return Path(session_dir) / "short_term" / "transcript_dirty_windows.json"


def _window_for_event(event: dict[str, Any], window_seconds: float = 30.0) -> dict[str, Any]:
    start = float(event.get("start_time", 0.0) or 0.0)
    window_start = math.floor(start / window_seconds) * window_seconds
    window_end = window_start + window_seconds
    return {
        "window_id": f"win_{int(window_start):06d}_{int(window_end):06d}",
        "start_time": round(window_start, 3),
        "end_time": round(window_end, 3),
    }


def load_transcript_dirty_windows(session_dir: Path) -> dict[str, Any]:
    path = transcript_dirty_windows_path(session_dir)
    state = read_json(path, default={})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("session_id", Path(session_dir).name)
    state.setdefault("dirty_version", 0)
    state.setdefault("windows", [])
    return state


def mark_transcript_dirty_windows(session_dir: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    path = transcript_dirty_windows_path(session_dir)
    state = load_transcript_dirty_windows(session_dir)
    by_id = {
        str(item.get("window_id")): dict(item)
        for item in state.get("windows", []) or []
        if isinstance(item, dict) and item.get("window_id")
    }
    now = utc_now_iso()
    changed: list[dict[str, Any]] = []
    for event in events:
        window = _window_for_event(event)
        existing = by_id.get(window["window_id"], {})
        event_ids = list(existing.get("event_ids") or [])
        event_id = str(event.get("event_id") or "")
        if event_id and event_id not in event_ids:
            event_ids.append(event_id)
        payload = {
            **existing,
            **window,
            "event_ids": event_ids,
            "status": "dirty",
            "reason": "transcript_backfill",
            "updated_at": now,
        }
        by_id[window["window_id"]] = payload
        changed.append(payload)
    state["windows"] = sorted(by_id.values(), key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("window_id", ""))))
    state["dirty_version"] = int(state.get("dirty_version", 0) or 0) + 1
    state["updated_at"] = now
    write_json_atomic(path, state)
    return changed


def mark_transcript_dirty_window_queued(session_dir: Path, window_start: float, window_end: float, task_id: str | None) -> None:
    state = load_transcript_dirty_windows(session_dir)
    now = utc_now_iso()
    for item in state.get("windows", []) or []:
        if not isinstance(item, dict):
            continue
        if abs(float(item.get("start_time", 0.0) or 0.0) - float(window_start)) < 1e-6 and abs(float(item.get("end_time", 0.0) or 0.0) - float(window_end)) < 1e-6:
            item["status"] = "queued"
            item["consolidation_task_id"] = task_id
            item["queued_at"] = now
            item["updated_at"] = now
    state["updated_at"] = now
    write_json_atomic(transcript_dirty_windows_path(session_dir), state)


def mark_transcript_dirty_window_consolidated(session_dir: Path, window_start: float, window_end: float, task_id: str | None = None) -> None:
    state = load_transcript_dirty_windows(session_dir)
    now = utc_now_iso()
    for item in state.get("windows", []) or []:
        if not isinstance(item, dict):
            continue
        if abs(float(item.get("start_time", 0.0) or 0.0) - float(window_start)) < 1e-6 and abs(float(item.get("end_time", 0.0) or 0.0) - float(window_end)) < 1e-6:
            item["status"] = "consolidated"
            item["consolidated_at"] = now
            item["consolidation_task_id"] = task_id or item.get("consolidation_task_id")
            item["updated_at"] = now
    state["updated_at"] = now
    write_json_atomic(transcript_dirty_windows_path(session_dir), state)


class TranscriptBackfiller:
    def __init__(
        self,
        session_dir: Path,
        *,
        project_root: Path | None = None,
        enqueue_refine: bool = True,
        enqueue_consolidation: bool = True,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.session_id = self.session_dir.name
        self.project_root = Path(project_root) if project_root is not None else self.session_dir.parents[1] if len(self.session_dir.parents) > 1 else None
        self.enqueue_refine = enqueue_refine
        self.enqueue_consolidation = enqueue_consolidation

    def backfill_segments(self, segments: list[dict[str, Any]], *, reason: str = "stream_asr") -> dict[str, Any]:
        normalized = [dict(item) for item in segments if isinstance(item, dict) and str(item.get("text") or "").strip()]
        if not normalized:
            return {
                "backfilled_current": False,
                "backfilled_event_count": 0,
                "updated_event_ids": [],
                "re_refine_event_count": 0,
                "dirty_window_count": 0,
            }
        current_result = MCurStore(self.session_dir).update_transcript_segments(normalized, source=reason)
        event_state_result = self._backfill_event_state(normalized)
        store = MSTStore(self.session_dir)
        mst_result = store.backfill_transcript_segments(normalized, reason=reason)
        audio_only_created = []
        if int(mst_result.get("backfilled_event_count", 0) or 0) == 0 and _is_audio_backfill_reason(reason):
            audio_only_created = self._append_audio_only_micro_events(store, normalized, reason=reason)
            if audio_only_created:
                mst_result = store.backfill_transcript_segments(normalized, reason=reason)
                if int(mst_result.get("backfilled_event_count", 0) or 0) == 0:
                    mst_result = {
                        **mst_result,
                        "backfilled_event_count": len(audio_only_created),
                        "updated_event_ids": [str(event.get("event_id")) for event in audio_only_created if event.get("event_id")],
                        "needs_refine_event_ids": [str(event.get("event_id")) for event in audio_only_created if event.get("event_id") and event.get("needs_refine")],
                        "updated_events": audio_only_created,
                        "audio_only_created": True,
                    }
        write_refine_status(store)
        refine_tasks = self._enqueue_refine_tasks(mst_result.get("needs_refine_event_ids", []), reason=reason)
        dirty_events = [event for event in mst_result.get("updated_events", []) or [] if event.get("needs_reconsolidation")]
        dirty_windows = mark_transcript_dirty_windows(self.session_dir, dirty_events)
        consolidation_tasks = self._enqueue_dirty_consolidation_tasks(dirty_windows, reason=reason)
        return {
            "backfilled_current": bool(current_result.get("backfilled_current")),
            "current_segment_count": int(current_result.get("segment_count", 0) or 0),
            "backfilled_open_event": bool(event_state_result.get("backfilled_open_event")),
            "backfilled_event_count": int(mst_result.get("backfilled_event_count", 0) or 0),
            "updated_event_ids": mst_result.get("updated_event_ids", []),
            "audio_only_event_count": len(audio_only_created),
            "audio_only_event_ids": [str(event.get("event_id")) for event in audio_only_created if event.get("event_id")],
            "re_refine_event_count": len(refine_tasks),
            "refine_task_paths": [str(path) for path in refine_tasks],
            "dirty_window_count": len(dirty_windows),
            "dirty_windows_due_to_transcript": dirty_windows,
            "consolidation_task_paths": [str(path) for path in consolidation_tasks],
        }

    def _append_audio_only_micro_events(self, store: MSTStore, segments: list[dict[str, Any]], *, reason: str) -> list[dict[str, Any]]:
        if not segments:
            return []
        starts = [float(seg.get("start", 0.0) or 0.0) for seg in segments]
        ends = [float(seg.get("end", seg.get("start", 0.0)) or 0.0) for seg in segments]
        start = min(starts) if starts else 0.0
        end = max(ends) if ends else start
        if end < start:
            end = start
        text = _segments_text(segments)
        digest = hashlib.sha1(f"{start:.3f}:{end:.3f}:{text}".encode("utf-8")).hexdigest()[:8]
        now = utc_now_iso()
        event = {
            "event_id": f"mst_audio_{self.session_id}_{int(round(start * 1000)):09d}_{int(round(end * 1000)):09d}_{digest}",
            "session_id": self.session_id,
            "chunk_id": "audio_chunk_asr",
            "source": {"type": "audio_chunk_asr", "reason": reason},
            "input_source": "audio_chunk_asr",
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "available_at": round(end, 3),
            "status": "provisional",
            "version": 1,
            "boundary_reason": "audio_asr_backfill",
            "diff_score": 0.0,
            "diff_stats": {},
            "event_caption_placeholder": f"Audio transcript is available between {start:.1f}s and {end:.1f}s.",
            "event_caption_fast": "",
            "event_caption_refined": None,
            "caption_source": "placeholder",
            "transcript": text,
            "transcript_segments": segments,
            "transcript_version": 1,
            "transcript_source": reason,
            "transcript_updated_at": now,
            "keyframes": [],
            "evidence_frames": [],
            "source_frame_indices": [],
            "entities": [],
            "visual_objects": [],
            "main_actions": [],
            "state_changes": [],
            "needs_refine": True,
            "refined_stale": False,
            "stale_reason": None,
            "needs_reconsolidation": False,
            "dirty_reason": None,
            "dirty_window_id": None,
            "dirty_time_range": None,
            "merged_to_long_term": False,
            "merged_episode_id": None,
            "merged_at": None,
            "confidence": 0.6,
            "created_at": now,
            "updated_at": now,
        }
        event["retrieval_text"] = build_retrieval_text(event)
        created = store.append_events([event])
        try:
            append_timeline_event(
                self.session_dir,
                "audio_transcript_mst_event_created",
                metadata={"event_id": event["event_id"], "segment_count": len(segments), "reason": reason},
            )
        except Exception as exc:
            print(f"[transcript_backfill] timeline append failed session_id={self.session_id}: {exc}", flush=True)
        return created

    def _backfill_event_state(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        manager = StreamChunkManager(self.session_dir)
        state = manager.load_event_state()
        open_event = state.get("open_event") if isinstance(state, dict) else None
        if not isinstance(open_event, dict):
            return {"backfilled_open_event": False, "segment_count": 0}
        start = float(open_event.get("start_time", 0.0) or 0.0)
        end = float(open_event.get("last_update_time", open_event.get("end_time", start)) or start)
        matched = [
            dict(seg)
            for seg in segments
            if _overlaps(start, end, float(seg.get("start", 0.0) or 0.0), float(seg.get("end", seg.get("start", 0.0)) or 0.0))
        ]
        if not matched:
            return {"backfilled_open_event": False, "segment_count": 0}
        merged = _merge_segments(list(open_event.get("transcript_segments", []) or []), matched)
        open_event["transcript_segments"] = merged
        open_event["transcript"] = _segments_text(merged)
        open_event["transcript_source"] = "stream_asr"
        open_event["transcript_updated_at"] = utc_now_iso()
        state["open_event"] = open_event
        manager.save_event_state(state)
        return {"backfilled_open_event": True, "segment_count": len(matched)}

    def _enqueue_refine_tasks(self, event_ids: list[str], *, reason: str) -> list[Path]:
        if not self.enqueue_refine or self.project_root is None:
            return []
        normalized_event_ids: list[str] = []
        for event_id in event_ids:
            text = str(event_id or "").strip()
            if text and text not in normalized_event_ids:
                normalized_event_ids.append(text)
        if not normalized_event_ids:
            return []
        task_path = enqueue_mst_refine_task(
            project_root=self.project_root,
            session_id=self.session_id,
            backend=os.getenv("EM2MEM_MST_REFINE_BACKEND", "openai"),
            limit_events=len(normalized_event_ids),
            event_ids=normalized_event_ids,
            force_refine=True,
            reason=reason,
        )
        append_timeline_event(
            self.session_dir,
            "refine_queued",
            metadata={
                "event_ids": normalized_event_ids,
                "event_count": len(normalized_event_ids),
                "reason": reason,
                "task_id": task_path.stem,
            },
        )
        return [task_path]

    def _enqueue_dirty_consolidation_tasks(self, windows: list[dict[str, Any]], *, reason: str) -> list[Path]:
        if not self.enqueue_consolidation or self.project_root is None:
            return []
        paths: list[Path] = []
        for window in windows:
            try:
                start = float(window.get("start_time"))
                end = float(window.get("end_time"))
            except Exception:
                continue
            task_path = enqueue_mst_consolidation_task(
                project_root=self.project_root,
                session_id=self.session_id,
                backend=os.getenv("EM2MEM_MST_EPISODIC_BACKEND", "openai"),
                update_em2mem=os.getenv("EM2MEM_MST_CONSOLIDATE_UPDATE_EM2MEM", "1").strip().lower() in {"1", "true", "yes", "on"},
                force=True,
                limit_windows=1,
                window_start=start,
                window_end=end,
                reason=reason,
            )
            mark_transcript_dirty_window_queued(self.session_dir, start, end, task_path.stem)
            paths.append(task_path)
        return paths


def _segment_key(segment: dict[str, Any]) -> str:
    explicit = str(segment.get("segment_id") or "").strip()
    if explicit:
        return explicit
    start = int(round(float(segment.get("start", 0.0) or 0.0) * 1000))
    end = int(round(float(segment.get("end", segment.get("start", 0.0)) or 0.0) * 1000))
    text = str(segment.get("text") or "").strip()
    return f"{start:09d}_{end:09d}_{text}"


def _merge_segments(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {_segment_key(item): dict(item) for item in existing if isinstance(item, dict)}
    for segment in incoming:
        if isinstance(segment, dict):
            by_key[_segment_key(segment)] = {**by_key.get(_segment_key(segment), {}), **dict(segment)}
    return sorted(by_key.values(), key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0), str(item.get("segment_id") or "")))


def _segments_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(seg.get("text") or "").strip() for seg in segments if seg.get("text")).strip()


def _is_audio_backfill_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    return text in {"audio_asr_backfill", "audio_chunk_asr"} or "audio" in text
