from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_frame_timestamp(value: Any, path: str | None = None) -> float | None:
    try:
        timestamp = float(value)
    except Exception:
        return None
    path_text = str(path or "")
    token = ""
    if path_text:
        match = re.search(r"kf_(\d{7,})(?:\D|$)", Path(path_text).stem)
        token = match.group(1) if match else ""
    is_stream_keyframe = "stream/keyframes" in path_text.replace("\\", "/")
    if timestamp >= 1000.0 and (is_stream_keyframe or len(token) >= 7):
        timestamp = timestamp / 1000.0
    return round(timestamp, 3)


def _timestamp_from_frame_path(path: str) -> float | None:
    stem = Path(path).stem
    for part in reversed(stem.split("_")):
        try:
            return _normalize_frame_timestamp(int(part), path)
        except Exception:
            continue
    return None


class EvidencePacker:
    """Trim retrieved evidence before prompt/image assembly and API return."""

    def __init__(self, session_dir: Path | None = None, runtime_session_dir: Path | None = None) -> None:
        self.session_dir = session_dir
        self.runtime_session_dir = runtime_session_dir or session_dir

    def _image_path_runtime_scoped(self, path: Any) -> bool:
        text = str(path or "").replace("\\", "/").lstrip("/")
        if not text:
            return False
        if text.startswith("stream/day_assets/"):
            return False
        return text.startswith(
            (
                "current/",
                "stream/",
                "short_term/",
                "mcur/",
                "frame_stream/",
                "audio_stream/",
            )
        )

    def _path_exists(self, path: str) -> bool:
        if self.session_dir is None:
            return True
        base_dir = self.runtime_session_dir if self._image_path_runtime_scoped(path) else self.session_dir
        return bool(base_dir and (base_dir / path).exists())

    def pack(
        self,
        query: str,
        route_decision: dict[str, Any],
        retrieval_result: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del query
        cache_context = cache_context or {}
        text_limit = int(route_decision.get("text_top_k") or 5)
        final_limit = int(route_decision.get("final_evidence_k") or 4)
        frame_limit = int(route_decision.get("evidence_frames_k") or 5)
        use_image = bool(route_decision.get("use_image_evidence", False))
        max_images = int(route_decision.get("max_image_evidence") or 0)

        text_results = list(retrieval_result.get("text_results", []) or [])[:text_limit]
        fused_results = list(retrieval_result.get("fused_results", []) or [])[:final_limit]
        visual_results = list(retrieval_result.get("visual_results", []) or [])
        short_term_results = list(retrieval_result.get("short_term_results", []) or [])
        final_evidence = list(retrieval_result.get("final_evidence", []) or [])
        fusion_summary = dict(retrieval_result.get("fusion_summary", {}) or {})
        current_context = retrieval_result.get("current_context") or {}
        current_selection = retrieval_result.get("current_selection") or {}
        current_frames = list(current_selection.get("evidence_frames", []) or [])

        ranked_frames = self._rank_frames(
            fused_results=fused_results,
            evidence_frames=current_frames + list(retrieval_result.get("evidence_frames", []) or []),
            short_term_results=short_term_results,
            final_evidence=final_evidence,
            cache_context=cache_context,
        )
        selected_frames, dedup_removed = self._dedup_frames(ranked_frames, limit=frame_limit)
        selected_images = self._select_images(selected_frames, use_image=use_image, max_images=max_images)
        preferred_current_images = [
            str(path)
            for path in (current_selection.get("selected_image_paths_for_mllm", []) or [])
            if path
        ]
        if use_image and preferred_current_images:
            selected_images = self._merge_preferred_images(preferred_current_images, selected_images, max_images)

        prompt_visual = []
        seen_visual = set()
        for fused in fused_results:
            for item in fused.get("visual_items", []) or []:
                visual_id = str(item.get("visual_id") or item.get("image_path") or "")
                if not visual_id or visual_id in seen_visual:
                    continue
                seen_visual.add(visual_id)
                prompt_visual.append({
                    "visual_id": item.get("visual_id"),
                    "segment_id": item.get("segment_id"),
                    "canonical_segment_id": item.get("canonical_segment_id") or fused.get("canonical_segment_id"),
                    "timestamp": (
                        _normalize_frame_timestamp(item.get("timestamp"), item.get("image_path"))
                        if item.get("timestamp") is not None
                        else _timestamp_from_frame_path(str(item.get("image_path") or ""))
                    ),
                    "image_path": item.get("image_path"),
                    "keyframe_caption": item.get("keyframe_caption"),
                    "visual_score": item.get("visual_score") or item.get("score"),
                    "fused_score": fused.get("fused_score"),
                })

        summary = {
            "text_evidence_count": len(text_results),
            "visual_frame_count": len(selected_frames),
            "sent_image_count": len(selected_images),
            "dedup_removed": dedup_removed,
            "use_image_evidence": use_image,
            "max_image_evidence": max_images,
            "final_evidence_count": len(fused_results),
            "visual_results_count": len(visual_results),
            "short_term_evidence_count": len(short_term_results[:final_limit]),
            "short_term_provisional_count": sum(1 for item in short_term_results[:final_limit] if item.get("status") == "provisional"),
            "current_evidence_count": len(current_frames),
            "current_sent_image_count": len([path for path in selected_images if str(path).startswith("current/")]),
            "current_window": [
                current_context.get("window_start_time"),
                current_context.get("window_end_time"),
            ] if current_context else None,
            "current_stale": bool(current_context.get("is_stale")) if current_context else False,
            "cache_used": bool(cache_context.get("is_followup")),
            "cache_evidence_count": int(cache_context.get("cache_evidence_count") or 0),
            "cache_referenced_entities": cache_context.get("referenced_entities", []),
            "cache_referenced_time_ranges": cache_context.get("referenced_time_ranges", []),
            "final_text_evidence_count": len(final_evidence[:final_limit]) if final_evidence else len(text_results),
            "final_evidence_frame_count": len(selected_frames),
            "selected_memory_sources": fusion_summary.get("selected_memory_sources", []),
            "image_selection_reason": (
                "selected from final fused evidence"
                if final_evidence and selected_images
                else ("image evidence disabled" if not use_image else "selected from retrieval frames")
            ),
        }
        return {
            "prompt_text_evidence": text_results,
            "prompt_visual_evidence_text": prompt_visual,
            "selected_evidence_frames": selected_frames,
            "selected_image_paths_for_mllm": selected_images,
            "selected_evidence": final_evidence[:final_limit],
            "packed_text_results": text_results,
            "packed_fused_results": fused_results,
            "evidence_pack_summary": summary,
            "image_selection_summary": {
                "selected_image_paths": selected_images,
                "candidate_frames": len(ranked_frames),
                "dedup_removed": dedup_removed,
            },
        }

    def _rank_frames(
        self,
        fused_results: list[dict[str, Any]],
        evidence_frames: list[dict[str, Any]],
        short_term_results: list[dict[str, Any]] | None = None,
        final_evidence: list[dict[str, Any]] | None = None,
        cache_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        frames = []
        cache_context = cache_context or {}
        short_term_results = short_term_results or []
        final_evidence = final_evidence or []
        for ev in final_evidence:
            paths = list(ev.get("keyframe_paths", []) or [])
            if not paths and ev.get("path"):
                paths = [ev.get("path")]
            if not paths and ev.get("image_path"):
                paths = [ev.get("image_path")]
            for path in paths:
                path = str(path or "")
                if not path:
                    continue
                frames.append({
                    "path": path,
                    "timestamp": (
                        _normalize_frame_timestamp(ev.get("timestamp"), path)
                        if ev.get("timestamp") is not None
                        else _timestamp_from_frame_path(path)
                    ),
                    "caption": ev.get("caption") or ev.get("transcript") or "",
                    "segment_id": ev.get("evidence_id"),
                    "canonical_segment_id": ev.get("evidence_id"),
                    "source": ev.get("source_memory") or ev.get("source") or "final_evidence",
                    "source_type": ev.get("source_type"),
                    "final_score": ev.get("final_score"),
                    "_rank_score": max(_safe_float(ev.get("final_score")), _safe_float(ev.get("retrieval_score")), 0.35),
                })
        for frame in evidence_frames:
            item = dict(frame)
            path = str(item.get("path") or item.get("image_path") or "")
            item["timestamp"] = (
                _normalize_frame_timestamp(item.get("timestamp"), path)
                if item.get("timestamp") is not None
                else _timestamp_from_frame_path(path)
            )
            item.setdefault("source", item.get("source") or "retrieval")
            item["_rank_score"] = max(
                _safe_float(item.get("fused_score")),
                _safe_float(item.get("visual_score")),
                _safe_float(item.get("text_score")),
            )
            frames.append(item)
        for fused in fused_results:
            for visual in fused.get("visual_items", []) or []:
                path = str(visual.get("image_path") or "")
                if not path:
                    continue
                frames.append({
                    "path": path,
                    "timestamp": (
                        _normalize_frame_timestamp(visual.get("timestamp"), path)
                        if visual.get("timestamp") is not None
                        else _timestamp_from_frame_path(path)
                    ),
                    "caption": visual.get("keyframe_caption", ""),
                    "visual_score": visual.get("visual_score") or visual.get("score"),
                    "fused_score": fused.get("fused_score"),
                    "segment_id": visual.get("segment_id") or fused.get("segment_id"),
                    "canonical_segment_id": visual.get("canonical_segment_id") or fused.get("canonical_segment_id"),
                    "source": "packed_hybrid",
                    "_rank_score": max(_safe_float(fused.get("fused_score")), _safe_float(visual.get("visual_score") or visual.get("score"))),
                })
        for event in short_term_results:
            for frame in event.get("keyframes", []) or []:
                if not isinstance(frame, dict):
                    continue
                path = str(frame.get("path") or "")
                if not path:
                    continue
                frames.append({
                    "path": path,
                    "timestamp": (
                        _normalize_frame_timestamp(frame.get("timestamp"), path)
                        if frame.get("timestamp") is not None
                        else _timestamp_from_frame_path(path)
                    ),
                    "caption": (
                        event.get("event_caption_refined")
                        or event.get("event_caption_fast")
                        or event.get("event_caption_placeholder")
                        or ""
                    ),
                    "short_term_score": event.get("score"),
                    "segment_id": event.get("event_id"),
                    "canonical_segment_id": event.get("event_id"),
                    "source": "M_st",
                    "status": event.get("status"),
                    "caption_source": event.get("caption_source"),
                    "_rank_score": max(_safe_float(event.get("score")), 0.4),
                })
        cache_count = 0
        if cache_context.get("is_followup"):
            for frame in cache_context.get("referenced_evidence_frames", []) or []:
                if not isinstance(frame, dict):
                    continue
                path = str(frame.get("path") or "")
                if not path:
                    continue
                item = dict(frame)
                item["timestamp"] = (
                    _normalize_frame_timestamp(item.get("timestamp"), path)
                    if item.get("timestamp") is not None
                    else _timestamp_from_frame_path(path)
                )
                item.setdefault("source", "cache_context")
                item["_rank_score"] = max(_safe_float(item.get("score")), 0.35)
                frames.append(item)
                cache_count += 1
                if cache_count >= 2:
                    break
            cache_context["cache_evidence_count"] = cache_count
        frames.sort(key=lambda item: (-_safe_float(item.get("_rank_score")), _safe_float(item.get("timestamp"), 1e9)))
        return frames

    def _dedup_frames(self, frames: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], int]:
        selected: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        per_segment: dict[str, int] = {}
        removed = 0
        for frame in frames:
            path = str(frame.get("path") or "")
            if not path or path in seen_paths:
                removed += 1
                continue
            segment = str(frame.get("canonical_segment_id") or frame.get("segment_id") or "")
            if segment and per_segment.get(segment, 0) >= 2:
                removed += 1
                continue
            ts = _safe_float(frame.get("timestamp"), -9999.0)
            too_close = False
            for existing in selected:
                if segment and segment == str(existing.get("canonical_segment_id") or existing.get("segment_id") or ""):
                    if abs(ts - _safe_float(existing.get("timestamp"), 9999.0)) < 1.5:
                        too_close = True
                        break
            if too_close:
                removed += 1
                continue
            item = {k: v for k, v in frame.items() if not k.startswith("_")}
            selected.append(item)
            seen_paths.add(path)
            if segment:
                per_segment[segment] = per_segment.get(segment, 0) + 1
            if len(selected) >= limit:
                break
        return selected, removed

    def _select_images(self, frames: list[dict[str, Any]], use_image: bool, max_images: int) -> list[str]:
        if not use_image or max_images <= 0:
            return []
        selected = []
        for frame in frames:
            path = str(frame.get("path") or "")
            if not path:
                continue
            if not self._path_exists(path):
                continue
            selected.append(path)
            if len(selected) >= max_images:
                break
        return selected

    def _merge_preferred_images(self, preferred: list[str], existing: list[str], max_images: int) -> list[str]:
        selected: list[str] = []
        for path in preferred + existing:
            if not path or path in selected:
                continue
            if not self._path_exists(path):
                continue
            selected.append(path)
            if len(selected) >= max_images:
                break
        return selected
