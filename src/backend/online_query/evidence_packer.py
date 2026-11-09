from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any

from online_time import keyframe_path_seconds, seconds_value
from online_query.cache_schema import has_valid_binding, has_exact_frame_binding
from online_query.evidence_schema import evidence_frames, normalize_evidence
from online_evidence_content import relevance, visual_anchors, focus_match, content_fields, query_actor_text, query_focus, actor_binding_fit, observation_match, action_support, inspection_signals, inspection_terms, action_records, action_window, native_action_level, image_observation_records

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_frame_timestamp(value: Any, path: str | None = None) -> float | None:
    timestamp = seconds_value(value)
    return timestamp


def _timestamp_from_frame_path(path: str) -> float | None:
    return keyframe_path_seconds(path)


def _evidence_frame_metadata(value: Any) -> list[dict[str, Any]]:
    """Compatibility alias for the shared, lossless schema boundary."""
    return evidence_frames(value)


def image_evidence_label(path: str, events: list[dict[str, Any]], frames: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Resolve capture metadata from a single frame as well as event lists."""
    owner = next((event for event in events if any(frame["path"] == path for frame in evidence_frames(event))), {})
    native_records = _evidence_frame_metadata(owner)
    records = list(frames or []) + native_records
    frame = next((row for row in records if path == (row.get("path") or row.get("image_path")) and row.get("timestamp") is not None), {})
    capture = frame.get("timestamp")
    capture = _normalize_frame_timestamp(capture, path) if capture is not None else _timestamp_from_frame_path(path)
    label = {"image_path": path, "evidence_id": owner.get("evidence_id") or frame.get("evidence_id"),
            "source": owner.get("source_memory") or frame.get("source"),
            "capture_time_seconds": capture,
            "event_time_seconds": [owner.get("start_time"), owner.get("end_time")],
            "original_source": frame.get("original_source") or frame.get("source"),
            "cache_reused": owner.get("source_memory") == "M_cache"}
    native = next((row for row in native_records if row.get("path") == path), {})
    description = (frame.get("frame_description") or native.get("description")
                   or native.get("caption") or native.get("keyframe_caption"))
    if isinstance(description, str) and description.strip():
        label["frame_description"] = description.strip()
        for key in ("description_source", "source_micro_event_id", "source_micro_event_version", "source_episode_id"):
            value = native.get(key, frame.get(key))
            if value is not None:
                label[key] = value
    return label


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


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
        return bool(base_dir and (base_dir / path).is_file())

    def _resolve_frame_path(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Reuse the recorded original capture after a current alias expires."""
        path = str(frame.get("path") or frame.get("image_path") or "")
        if path and self._path_exists(path):
            return frame
        source_path = frame.get("source_path")
        if not isinstance(source_path, str) or not source_path or not self._path_exists(source_path):
            return frame
        base = self.runtime_session_dir if self._image_path_runtime_scoped(source_path) else self.session_dir
        if base is not None and not (base / source_path).resolve().is_relative_to(base.resolve()):
            return frame
        capture = _normalize_frame_timestamp(frame.get("timestamp"))
        source_capture = _timestamp_from_frame_path(source_path)
        if capture is not None and source_capture is not None and abs(capture - source_capture) > 0.01:
            return frame
        return {**frame, "path": source_path, "path_recovered_from": path}

    def pack(
        self,
        query: str,
        route_decision: dict[str, Any],
        retrieval_result: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_context = cache_context or {}
        text_limit = int(route_decision.get("text_top_k") or 5)
        final_limit = int(route_decision.get("final_evidence_k") or 4)
        use_image = bool(route_decision.get("use_image_evidence", False))
        max_images = int(route_decision.get("max_image_evidence") or 0)
        per_evidence_image_limit = max(0, _env_int("EM2MEM_IMAGES_PER_EVIDENCE_LIMIT", 3))
        configured_total_limit = max(0, _env_int("EM2MEM_QUERY_MAX_TOTAL_IMAGES", 9))
        total_image_limit = min(max_images, configured_total_limit) if max_images > 0 else 0

        text_results = list(retrieval_result.get("text_results", []) or [])[:text_limit]
        fused_results = list(retrieval_result.get("fused_results", []) or [])[:final_limit]
        visual_results = list(retrieval_result.get("visual_results", []) or [])
        short_term_results = list(retrieval_result.get("short_term_results", []) or [])
        authoritative = "final_evidence" in retrieval_result
        final_evidence = list(retrieval_result.get("final_evidence", []) or [])[:final_limit]
        if not authoritative and short_term_results and not text_results and not fused_results:
            # Use the same localized selection policy when long-term memory is
            # still building; the fallback must not revert to prefix images.
            authoritative = True
            final_evidence = [{"evidence_id": event.get("event_id"), "source_memory": "M_st", "start_time": event.get("start_time"), "end_time": event.get("end_time"), "final_score": event.get("score", 0), "caption": event.get("event_caption_refined") or event.get("retrieval_text") or "", "metadata": event, "keyframe_paths": [frame["path"] for frame in event.get("keyframes", []) if frame.get("path")]} for event in short_term_results[:final_limit]]
        final_evidence = [normalize_evidence(event) for event in final_evidence]
        for event in final_evidence:
            resolved = [self._resolve_frame_path(frame) for frame in evidence_frames(event)]
            event["evidence_frames"] = resolved
            replacements = {frame["path_recovered_from"]: frame["path"] for frame in resolved if frame.get("path_recovered_from")}
            event["keyframe_paths"] = [replacements.get(path, path) for path in event.get("keyframe_paths", [])]
        fusion_summary = dict(retrieval_result.get("fusion_summary", {}) or {})
        current_context = retrieval_result.get("current_context") or {}
        current_selection = retrieval_result.get("current_selection") or {}
        current_frames = list(current_selection.get("evidence_frames", []) or [])

        ranked_frames = self._rank_frames(
            fused_results=fused_results,
            evidence_frames=current_frames + list(retrieval_result.get("evidence_frames", []) or []),
            short_term_results=short_term_results,
            final_evidence=final_evidence if authoritative else None,
            cache_context=cache_context,
        )
        from online_query_constraints import overlaps, resolve_query_constraints, constraint_windows
        provided_constraints = route_decision.get("query_constraints")
        if provided_constraints is None:
            provided_constraints = cache_context.get("query_constraints")
        constraints = resolve_query_constraints(query, query_constraints=provided_constraints)
        initial_frame_count = len(ranked_frames)
        windows = constraint_windows(constraints)
        if windows:
            ranked_frames = [frame for frame in ranked_frames if any(overlaps(frame.get("timestamp"), frame.get("timestamp"), window) for window in windows)]
        require_current = bool(constraints.get("requires_current") or constraints.get("requires_current_tail")
                               or route_decision.get("requires_current") or route_decision.get("requires_current_tail"))
        reliable_same_time = (cache_context.get("temporal_relation") == "same_time"
                              and has_valid_binding(cache_context) and not require_current)
        soft_windows = []
        related_binding = (has_valid_binding(cache_context) and not require_current
                           and cache_context.get("temporal_relation") in {"same_time", "related"})
        if not windows and related_binding and cache_context.get("referenced_time_ranges"):
            soft_windows = [[_safe_float(window.get("start")), _safe_float(window.get("end"), _safe_float(window.get("start")))]
                            for window in cache_context["referenced_time_ranges"]]
        if not windows and reliable_same_time and has_exact_frame_binding(cache_context):
            windows = [[_safe_float(window.get("start")), _safe_float(window.get("end"), _safe_float(window.get("start")))]
                       for window in cache_context["referenced_time_ranges"]]
            ranked_frames = [frame for frame in ranked_frames if frame.get("timestamp") is not None
                             and any(overlaps(frame["timestamp"], frame["timestamp"], window) for window in windows)]
        scoped_frame_count = len(ranked_frames)
        required_current_frames = [frame for frame in ranked_frames if require_current and frame.get("source") == "M_cur"]
        if authoritative:
            # A resolver-bound object may guide image relevance, never time
            # parsing or evidence eligibility. Ordinary cache entities cannot.
            selection_query = query
            bound_focus = cache_context.get("bound_focus_entities") or []
            explicit_time = bool(constraints.get("time_explicit") or constraint_windows(constraints))
            if not explicit_time and len(bound_focus) == 1 and isinstance(bound_focus[0], dict):
                name = str(bound_focus[0].get("name") or "").strip()[:120]
                if name:
                    selection_query = f"{query}\n关注物体: {name}"
            ranked_frames = self._order_event_frames(selection_query, ranked_frames, final_evidence, per_evidence_image_limit,
                overview=bool(constraints.get("summary")) or route_decision.get("query_type") == "long_term_summary",
                total_images=total_image_limit, windows=windows, preferred_windows=soft_windows,
                preserve_initial=constraints.get("temporal_preference") == "earlier_and_later",
                query_constraints=constraints)
        if required_current_frames:
            retained_paths = {frame.get("path") for frame in ranked_frames}
            for frame in required_current_frames:
                if frame.get("path") not in retained_paths:
                    frame["selection_reason"] = "required unrefined current coverage; visual state unverified"
                    ranked_frames.append(frame)
        # Upstream selections can contain more images than this event's budget.
        # Reserve current core coverage before older rolling-window references;
        # truncating the upstream preference can otherwise lose the process.
        prefer_current = require_current or constraints.get("scope") == "current" or route_decision.get("query_type") == "current_perception"
        current_order = self._current_coverage_order(
            query, ranked_frames, current_context, current_selection, final_evidence,
        ) if prefer_current else []
        preferred_order = {path: index for index, path in enumerate(current_order)}
        ranked_frames.sort(key=lambda frame: preferred_order.get(str(frame.get("path") or ""), len(preferred_order)))
        if require_current and constraints.get("requires_history"):
            current_first = next((frame for frame in ranked_frames if frame.get("source") == "M_cur" and self._path_exists(str(frame.get("path") or ""))), None)
            history_first = next((frame for frame in ranked_frames if frame.get("source") in {"M_st", "M_lt", "M_cache"} and self._path_exists(str(frame.get("path") or ""))), None)
            reserved = [frame for frame in (current_first, history_first) if frame is not None]
            ranked_frames = reserved + [frame for frame in ranked_frames if frame not in reserved]
        selected_frames, dedup_removed = self._dedup_frames(
            ranked_frames,
            limit=None,
            per_segment_limit=max(1, per_evidence_image_limit),
            compare_content=bool(use_image and total_image_limit > 0 and per_evidence_image_limit > 0),
        )
        selected_images = self._select_images(
            selected_frames,
            use_image=use_image,
            per_evidence_limit=per_evidence_image_limit,
            total_limit=total_image_limit,
        )
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

        sent_sources = {frame.get("source") for frame in selected_frames if frame.get("path") in selected_images}
        missing_image_sources = []
        if require_current and "M_cur" not in sent_sources:
            missing_image_sources.append("current")
        if constraints.get("requires_history") and not sent_sources.intersection({"M_st", "M_lt", "M_cache"}):
            missing_image_sources.append("history")
        summary = {
            "required_image_sources_missing": missing_image_sources,
            "query_constraints": constraints,
            "sent_image_paths": selected_images,
            "text_evidence_count": len(text_results),
            "visual_frame_count": len(selected_frames),
            "sent_image_count": len(selected_images),
            "dedup_removed": dedup_removed,
            "use_image_evidence": use_image,
            "max_image_evidence": max_images,
            "per_evidence_image_limit": per_evidence_image_limit,
            "total_image_limit": total_image_limit,
            "final_evidence_count": len(final_evidence),
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
                "final event priority; query/action-time frame selection; event coverage"
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
                "candidate_frames_before_scope": initial_frame_count,
                "candidate_frames_after_scope": scoped_frame_count,
                "cache_time_policy": "exact_frame" if has_exact_frame_binding(cache_context) else "soft" if soft_windows else "none",
                "ambiguous_inspection_count": sum(frame.get("selection_reason", "").startswith("ambiguous target inspection") for frame in selected_frames if frame.get("path") in selected_images),
                "inspection_fallback_count": sum(frame.get("selection_reason", "").startswith("bounded inspection fallback") for frame in selected_frames if frame.get("path") in selected_images),
                "inspection_fallback_reasons": [frame.get("selection_reason") for frame in selected_frames if frame.get("path") in selected_images and frame.get("selection_reason", "").startswith("bounded inspection fallback")],
                "dedup_removed": dedup_removed,
                "event_order": [ev.get("evidence_id") for ev in final_evidence],
                "sent_frames": [{k: frame.get(k) for k in ("path", "timestamp", "evidence_id", "source", "selection_reason", "final_score", "duplicate_provenance")} for frame in selected_frames if frame.get("path") in selected_images],
            },
        }

    def _current_coverage_order(
        self, query: str, frames: list[dict[str, Any]], context: dict[str, Any],
        selection: dict[str, Any], events: list[dict[str, Any]],
    ) -> list[str]:
        """Order eligible current captures without adding or widening evidence."""
        preferred = {str(path): index for index, path in enumerate(
            selection.get("selected_image_paths_for_mllm", []) or []) if path}
        state = context.get("state") or {}
        now = next((seconds_value(value) for value in (
            context.get("current_time"), context.get("window_end_time"),
            state.get("current_time"), state.get("window_end_time"),
        ) if seconds_value(value) is not None), None)
        labeled_latest = [frame for frame in frames if frame.get("source") == "M_cur"
                          and frame.get("role") == "latest" and frame.get("timestamp") is not None]
        if now is None and not labeled_latest:
            # Legacy callers may provide only inferred path times. Those do
            # not establish NOW or authorize replacing their explicit order.
            return list(preferred)
        candidates = {}
        for frame in frames:
            path = str(frame.get("path") or "")
            if frame.get("source") == "M_cur" and frame.get("timestamp") is not None and self._path_exists(path):
                candidates.setdefault(path, frame)
        if not candidates:
            return []
        pending = list(candidates.values())
        latest = max(pending, key=lambda frame: float(frame["timestamp"]))
        if now is None:
            now = max(float(frame["timestamp"]) for frame in labeled_latest)
        core_seconds = max(0.0, _safe_float(state.get("core_seconds"), 10.0))
        core_start = now - core_seconds
        core = [frame for frame in pending if core_start <= float(frame["timestamp"]) <= now and frame is not latest]
        anchors = []
        for event in [context, context.get("open_event") or {}] + [
                event for event in events if event.get("source_memory") == "M_cur"]:
            anchors.extend(visual_anchors(query, event))

        def anchor_score(frame):
            timestamp = float(frame["timestamp"])
            return max((anchor["score"] if frame["path"] in anchor["paths"]
                        or anchor["start"] <= timestamp <= anchor["end"] else 0.0
                        for anchor in anchors), default=0.0)

        chosen = [latest]
        if core:
            # Localized observations lead; existing visual change scores are a
            # fallback, not a claim about the semantic contents of the image.
            process = max(core, key=lambda frame: (
                anchor_score(frame), _safe_float(frame.get("diff_score")),
                -abs(float(frame["timestamp"]) - (core_start + now) / 2),
                -preferred.get(frame["path"], len(preferred)),
            ))
            chosen.append(process)
            core.remove(process)
        if core:
            # Keep the earlier side of the selected process before spending a
            # third slot on an outer-window reference or another adjacent peak.
            first = min(core, key=lambda frame: float(frame["timestamp"]))
            chosen.append(first)
            core.remove(first)
        while core:
            frame = max(core, key=lambda frame: min(
                abs(float(frame["timestamp"]) - float(other["timestamp"])) for other in chosen))
            chosen.append(frame)
            core.remove(frame)
        seen = {frame["path"] for frame in chosen}
        older = sorted((frame for frame in pending if frame["path"] not in seen),
                       key=lambda frame: preferred.get(frame["path"], len(preferred)))
        return [frame["path"] for frame in chosen + older]

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
        authoritative = final_evidence is not None
        final_evidence = final_evidence or []
        # A fused event timestamp describes the event, not every image in it.
        # Reuse capture metadata even when the higher-scored candidate comes
        # from another retrieval path, before path deduplication can discard it.
        frame_by_path: dict[str, dict[str, Any]] = {}
        def identities(frame):
            return {str(frame[key]) for key in ("evidence_id", "canonical_segment_id", "segment_id")
                    if frame.get(key) is not None}
        selected_identities: dict[str, set[str]] = {}
        for value in final_evidence:
            for native in _evidence_frame_metadata(value):
                path = native["path"]
                selected_identities.setdefault(path, set()).update(identities(native))
                existing = frame_by_path.setdefault(path, dict(native))
                origins = existing.setdefault("frame_origins", [])
                for origin in native.get("frame_origins", []):
                    tagged = {**origin, "selection_association": "selected"}
                    if tagged not in origins:
                        origins.append(tagged)
        for value in short_term_results + fused_results + evidence_frames:
            for native in _evidence_frame_metadata(value):
                path = native["path"]
                existing = frame_by_path.get(path)
                if authoritative and path not in selected_identities:
                    continue
                compatible = not authoritative or bool(identities(native) & selected_identities.get(path, set()))
                association = "matching_selected_event" if authoritative and compatible else "unselected_candidate" if authoritative else "retrieval_candidate"
                if existing is None:
                    existing = frame_by_path[path] = dict(native)
                    existing["frame_origins"] = []
                # Selected capture metadata and captions remain authoritative.
                # A matching candidate can fill a missing/path-derived capture,
                # while a rejected candidate is retained only as tagged origin.
                if compatible:
                    if (existing.get("timestamp") is None or existing.get("capture_time_origin") == "path") and native.get("capture_time_origin") == "metadata":
                        for key in ("timestamp", "capture_time_seconds", "capture_time_origin"):
                            existing[key] = native.get(key)
                    for key, value in native.items():
                        if key != "frame_origins" and (not authoritative or existing.get(key) in (None, "")):
                            existing[key] = value
                origins = existing.setdefault("frame_origins", [])
                for origin in native.get("frame_origins", []):
                    tagged = {**origin, "selection_association": association}
                    if tagged not in origins:
                        origins.append(tagged)
        segment_by_path: dict[str, str] = {}
        for ev in final_evidence:
            metadata = ev.get("metadata") or {}
            segment = (ev.get("canonical_segment_id") or ev.get("segment_id")
                       or metadata.get("canonical_segment_id") or metadata.get("segment_id")
                       or ev.get("evidence_id"))
            for native_frame in _evidence_frame_metadata(ev):
                path = native_frame["path"]
                if segment:
                    segment_by_path.setdefault(path, str(segment))
                frames.append({
                    **native_frame,
                    "evidence_id": ev.get("evidence_id"),
                    "path": path,
                    "timestamp": native_frame.get("timestamp"),
                    "caption": native_frame.get("caption") or native_frame.get("keyframe_caption") or ev.get("caption") or ev.get("transcript") or "",
                    "segment_id": segment,
                    "canonical_segment_id": segment,
                    "source": ev.get("source_memory") or ev.get("source") or "final_evidence",
                    "source_type": ev.get("source_type"),
                    "final_score": ev.get("final_score"),
                    "_rank_score": _safe_float(ev.get("final_score")),
                })
        if authoritative:
            # Raw candidates may enrich capture metadata above, but cannot
            # re-enter the final evidence set through the image branch.
            evidence_frames = []
            fused_results = []
            short_term_results = []
            cache_context = {}
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
            if not item.get("segment_id") and not item.get("canonical_segment_id") and item.get("source") == "M_cur":
                item["segment_id"] = "M_cur"
                item["canonical_segment_id"] = "M_cur"
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
        for frame in frames:
            path = str(frame.get("path") or frame.get("image_path") or "")
            metadata = frame_by_path.get(path, {})
            if metadata.get("timestamp") is not None:
                frame["timestamp"] = _normalize_frame_timestamp(metadata["timestamp"], path)
            for key in ("original_source", "frame_origins", "capture_time_origin", "capture_time_seconds"):
                if key in metadata:
                    frame[key] = metadata[key]
            if metadata.get("caption") or metadata.get("keyframe_caption"):
                frame["caption"] = metadata.get("caption") or metadata.get("keyframe_caption")
            segment = (frame.get("canonical_segment_id") or frame.get("segment_id")
                       or segment_by_path.get(path) or metadata.get("canonical_segment_id") or metadata.get("segment_id"))
            if segment:
                frame["canonical_segment_id"] = segment
        frames.sort(key=lambda item: (-_safe_float(item.get("_rank_score")), _safe_float(item.get("timestamp"), 1e9)))
        return frames

    def _order_event_frames(self, query: str, frames: list[dict[str, Any]], events: list[dict[str, Any]], per_event: int, *, overview: bool = False, total_images: int = 0, windows: list[list[float]] | None = None, preferred_windows: list[list[float]] | None = None, preserve_initial: bool = False, query_constraints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fused event priority, local action relevance, then event coverage."""
        from online_query_constraints import actor_fit, overlaps
        groups = []
        unknown_inspection_pool = []
        initial_candidates = []
        windows = windows or []
        from online_query_constraints import resolve_query_constraints
        constraints = resolve_query_constraints(query, query_constraints=query_constraints)
        requirements = constraints.get("requirements") or {}
        targeted = bool(query_focus(query) or inspection_terms(query))
        # Scoped state verification needs the requested pictures even when
        # their observations say "no", "unclear", or describe another item.
        # Positive occurrence lookup still requires target inspection support.
        state_verification = bool(windows) and bool(re.search(
            r"是否|有没有|吗|什么|状态|区别|相同|变化|\b(?:what|whether|is|are|does|do|has|have)\b", query, re.I))
        state_overview = any(word in query.lower() for word in ["变化", "穿戴", "状态", "change", "appearance", "wear", "clothes"])
        for event_index, event in enumerate(events):
            candidates = [frame for frame in frames if frame.get("evidence_id") == event.get("evidence_id") and self._path_exists(str(frame.get("path") or ""))]
            event_start, event_end = _safe_float(event.get("start_time")), _safe_float(event.get("end_time"))
            if event.get("start_time") is not None and event.get("end_time") is not None:
                candidates = [frame for frame in candidates if frame.get("timestamp") is None
                              or event_start <= float(frame["timestamp"]) <= event_end]
            if not candidates:
                continue
            observations = image_observation_records(event)
            # A contradictory actor in an exact observation is useful negative
            # selection evidence. Missing positional wording is not a conflict.
            eligible = []
            for frame in candidates:
                exact_observations = []
                for observation in observations:
                    paths = observation.get("keyframe_paths") or []
                    direct_path = observation.get("path") or observation.get("image_path")
                    start, end = action_window(observation)
                    timestamp = frame.get("timestamp")
                    if (direct_path == frame["path"] or frame["path"] in paths
                            or (not direct_path and not paths and start is not None and timestamp is not None
                                and start <= float(timestamp) <= (end if end is not None else start))):
                        exact_observations.append(observation)
                # A selective caption can mention a different person while the
                # requested person is also visible in this same saved image.
                frame["_actor_conflict"] = bool(not overview and exact_observations and all(
                    actor_binding_fit(query, observation, event) == 0 for observation in exact_observations))
                frame["_direct_observations"] = exact_observations
                eligible.append(frame)
            candidates = eligible
            if not candidates:
                continue
            anchors = visual_anchors(query, event, records=observations)
            for frame in candidates:
                ts = frame.get("timestamp")
                # Event captions are shared by every frame and cannot serve as
                # per-frame relevance. Localized action evidence takes priority.
                local = 0.0 if anchors else relevance(query, frame.get("caption") or "")
                for anchor in anchors:
                    if frame["path"] in anchor["paths"]:
                        local = max(local, anchor["score"] * 2)
                    elif ts is not None:
                        center = (anchor["start"] + anchor["end"]) / 2
                        local = max(local, anchor["score"] / (1 + abs(float(ts)-center) / 3))
                # Ambiguous object names remain candidates to inspect, rather
                # than being treated as evidence that the target is absent.
                direct = frame.get("_direct_observations", [])
                if direct:
                    # A selective frame caption must not erase a localized
                    # action that was already matched to this same capture.
                    local = max(local, max(observation_match(query, observation, event) for observation in direct) * 2)
                signals = [inspection_signals(query, observation) for observation in direct]
                verification = max((0.15 * actor_fit(query, observation)
                                    for observation, values in zip(direct, signals)
                                    if "candidate" in values.values()), default=0.0)
                if not overview and signals and all(values and all(value in {"unknown", "negative"} for value in values.values()) for values in signals):
                    # Explicitly described other objects/actors must not acquire
                    # relevance from a shared event caption. Ambiguous devices
                    # are candidates (not unknown) in the target signal helper.
                    frame["_unrelated_direct"] = True
                # A description is selective: mentioning a bottle does not
                # establish that a phone is absent from the same saved image.
                # Unknown descriptions get a bounded inspection opportunity;
                # explicit negative target observations retain their polarity.
                generic_unknown = bool(signals) and all(values and all(value in {"unknown", "background"} for value in values.values()) for values in signals)
                # A selected current event may lack a caption by design. During
                # a grounded follow-up, allow one saved frame within its cited
                # span for inspection; unrelated current placeholders stay out.
                scoped_current = (frame.get("source") == "M_cur" and ts is not None
                    and any(overlaps(ts, ts, window) for window in preferred_windows or []))
                frame["_cache_scoped_current"] = scoped_current
                can_inspect = frame.get("source") != "M_cur" or scoped_current
                if generic_unknown and can_inspect:
                    unknown_inspection_pool.append(frame)
                frame["_inspection_unknown"] = bool((generic_unknown or not direct or frame.get("_actor_conflict")
                    or frame.get("_unrelated_direct")) and can_inspect)
                frame["_inspection_supported"] = bool(state_verification and direct) or (not frame.get("_actor_conflict") and (bool(verification) or any(
                    any(value in {"confirmed", "candidate"} for value in values.values()) for values in signals)))
                frame["_action_support_level"] = 0 if overview else max((native_action_level(query, observation)
                    for observation in direct if actor_binding_fit(query, observation, event) > 0), default=0)
                frame["_verification_signal"] = verification
                frame["_local_relevance"] = max(local, verification)
                if frame.get("_actor_conflict"):
                    frame["_local_relevance"] *= 0.25
                frame["event_score"] = frame.get("final_score")
                frame["frame_relevance_score"] = frame["_local_relevance"]
                frame["frame_score_method"] = "stored_description_and_action_time"
                descriptions = list(dict.fromkeys(str(row["description"]).strip() for row in direct
                    if row.get("description") and (row.get("path") or row.get("image_path")) == frame["path"]))
                if descriptions:
                    frame["frame_description"] = " ".join(descriptions)
                exact = list(direct)
                frame["_coverage_observations"] = exact
                frame["_target_fit"] = max(((action_support(query, observation) or {}).get("score", focus_match(query, observation) or 0)
                                             for observation in exact),
                                             default=focus_match(query, content_fields(event) or event.get("caption") or "") or 0)
                frame["selection_reason"] = "query-matched observation/action time" if anchors else "temporal coverage within selected event"
                if signals and any("candidate" in values.values() for values in signals) and not any("confirmed" in values.values() for values in signals):
                    frame["selection_reason"] = "ambiguous target inspection: localized candidate needs image verification"
            if preserve_initial:
                # Keep the requested initial state even when the target is
                # absent/unclear there. It is a comparison anchor, not support
                # for a positive occurrence. Actor/time/file checks still apply.
                initial_candidates.extend(frame for frame in candidates if frame.get("timestamp") is not None)
            # Text scores rank inspection candidates; weak or missing lexical
            # matches do not establish that an image lacks the requested object.
            ordered = []
            temporal = [float(frame["timestamp"]) for frame in candidates if frame.get("timestamp") is not None]
            span = max(temporal)-min(temporal) if temporal else 0
            has_local = bool(anchors) or len({round(frame["_local_relevance"], 6) for frame in candidates}) > 1
            if has_local:
                while candidates:
                    def score(frame):
                        diversity = min((abs(float(frame["timestamp"])-float(picked["timestamp"])) for picked in ordered if picked.get("timestamp") is not None), default=span) / max(span, 1) if frame.get("timestamp") is not None else 0
                        # Structured fields are a small hint, not a separate
                        # priority class above equivalent natural-language facts.
                        return frame["_local_relevance"] + 0.05 * frame.get("_action_support_level", 0) + 0.10 * diversity
                    frame = max(candidates, key=score)
                    ordered.append(frame)
                    candidates.remove(frame)
            else:
                candidates.sort(key=lambda frame: _safe_float(frame.get("timestamp"), 1e9))
                indices = list(dict.fromkeys(round(i*(len(candidates)-1)/max(1, per_event-1)) for i in range(max(1, per_event))))
                if overview and event_index > 0:
                    indices = (list(reversed(indices)) if state_overview or event_index == len(events)-1
                               else list(dict.fromkeys([round((len(candidates)-1)/2), 0, len(candidates)-1])))
                if overview:
                    for frame in candidates:
                        frame["selection_reason"] = "overview timeline coverage; initial, representative actions, and latest state"
                ordered = [candidates[i] for i in indices] + [frame for i, frame in enumerate(candidates) if i not in indices]
            if not overview and targeted and any(frame.get("_coverage_observations") for frame in ordered):
                # Keep event/fusion ranking as the primary decision.  Within
                # an already selected event, put complementary stored
                # observations ahead of adjacent duplicate stills.
                from online_evidence_coverage import order_frame_candidates_by_complementarity
                ordered = order_frame_candidates_by_complementarity(query, ordered)
            groups.append(ordered)
        if overview and total_images > 0:
            from online_evidence_coverage import select_coverage
            flat = [frame for group in groups for frame in group]
            representatives = {id(group[0]) for group in groups if group}
            candidates = [{"timestamp": frame.get("timestamp"), "visual_observations": frame.get("_coverage_observations", []),
                           "score": frame.get("_local_relevance", 0) + float(id(frame) in representatives), "frame": frame} for frame in flat]
            chosen = [item["frame"] for item in select_coverage(query, candidates, total_images, preserve_middle=True)]
            timed = [frame for frame in flat if frame.get("timestamp") is not None]
            if timed and total_images >= 2:
                endpoints = [min(timed, key=lambda frame: float(frame["timestamp"])), max(timed, key=lambda frame: float(frame["timestamp"]))]
                # Preserve the best retrieved event's view before allocating
                # the remaining slots using approximate state coverage.
                historical = [group for group in groups if group[0].get("source") == "M_lt"] or groups
                leading = max(historical, key=lambda group: _safe_float(group[0].get("final_score")))[0] if historical else None
                protected = endpoints + ([leading] if leading is not None and leading not in endpoints else [])
                chosen = protected + [frame for frame in chosen if frame not in protected]
                chosen = chosen[:total_images]
                chosen.sort(key=lambda frame: _safe_float(frame.get("timestamp")))
            for frame in chosen:
                frame["selection_reason"] = "overview covers distinct observed states/actions and endpoints"
            return chosen + [frame for frame in flat if frame not in chosen]
        flat = [group[index] for index in range(max((len(g) for g in groups), default=0))
                for group in groups if index < len(group)]
        initial = min(initial_candidates, key=lambda frame: float(frame["timestamp"])) if initial_candidates else None
        if initial is not None:
            initial["selection_reason"] = "requested initial state for temporal comparison; verify target presence"
        if not overview and not targeted and any(frame.get("_direct_observations") for frame in flat):
            # With an open object question there is no named target to match.
            # Reuse observed state/action diversity across the candidate images
            # before spending every slot on the first frame of each event.
            from online_evidence_coverage import coverage_features
            pending = list(flat)
            # Local novelty must not erase the highest-priority fused event.
            chosen = [flat[0]] if flat and total_images else []
            pending = [frame for frame in pending if frame not in chosen]
            covered = set()
            counts = {chosen[0].get("evidence_id"): 1} if chosen else {}
            while pending and len(chosen) < total_images:
                def gain(frame):
                    observations = [{**observation, "description": query_actor_text(query, observation)}
                                    for observation in frame.get("_direct_observations", [])]
                    features = coverage_features(query, {"visual_observations": observations})
                    novel = features - covered
                    return (sum(2 if feature.startswith(("state:", "action:", "transition:")) else 1 for feature in novel),
                            frame.get("_local_relevance", 0))
                eligible = [frame for frame in pending if counts.get(frame.get("evidence_id"), 0) < max(1, per_event)]
                if not eligible:
                    break
                frame = max(eligible, key=gain)
                if gain(frame)[0] <= 0:
                    break
                observations = [{**observation, "description": query_actor_text(query, observation)}
                                for observation in frame.get("_direct_observations", [])]
                covered.update(coverage_features(query, {"visual_observations": observations}))
                frame["selection_reason"] = "open question covers distinct timed observed states/actions"
                chosen.append(frame)
                counts[frame.get("evidence_id")] = counts.get(frame.get("evidence_id"), 0) + 1
                pending.remove(frame)
            flat = chosen + [frame for frame in flat if frame not in chosen]
        if overview or not targeted:
            # Open-ended requests to inspect an event have no target whose
            # absence could justify restricting its temporal representatives.
            return ([initial] + [frame for frame in flat if frame is not initial]) if initial is not None else flat
        # Text co-mentions are candidates, not proof of actor/action/time
        # conjunction. Keep alternative inspection evidence for the model.
        priority = []
        seen_paths = set()
        per_event_counts = {}
        def reserve(frame):
            path = frame["path"]
            event_id = frame.get("evidence_id")
            if path in seen_paths or per_event_counts.get(event_id, 0) >= max(1, per_event):
                return
            priority.append(frame)
            seen_paths.add(path)
            per_event_counts[event_id] = per_event_counts.get(event_id, 0) + 1
        if initial is not None:
            reserve(initial)
        # Each requested window gets its own direct inspection opportunity.
        # A gap between windows is never used to satisfy either side.
        for window in windows or preferred_windows or []:
            matching = [frame for frame in flat if overlaps(frame.get("timestamp"), frame.get("timestamp"), window)]
            if matching:
                reserve(max(matching, key=lambda frame: frame.get("_local_relevance", 0) + 0.05 * frame.get("_action_support_level", 0)))
        # A cited period is not a single still. In particular its newest frame
        # can be unrefined and must be inspected before repeating older labels.
        for window in preferred_windows or []:
            matching = [frame for frame in flat if frame.get("timestamp") is not None
                        and overlaps(frame["timestamp"], frame["timestamp"], window)
                        and not frame.get("_actor_conflict")]
            if matching and len(priority) < total_images:
                latest = max(matching, key=lambda frame: float(frame["timestamp"]))
                latest["selection_reason"] = "end of referenced period; verify latest visible state"
                reserve(latest)
        timed = [frame for frame in flat if frame.get("timestamp") is not None and not frame.get("_actor_conflict")]
        positive = [frame for frame in timed if frame.get("_inspection_supported")]
        earlier = requirements.get("earliest") or constraints.get("temporal_preference") in {"earlier", "earlier_and_later"}
        if timed and not windows and (earlier or requirements.get("comparison")):
            first = initial or min(positive or timed, key=lambda frame: float(frame["timestamp"]))
            first["selection_reason"] = "earlier phase inspection; not proof of first occurrence"
            reserve(first)
            if requirements.get("comparison") and len(priority) < total_images:
                last = max(timed, key=lambda frame: float(frame["timestamp"]))
                last["selection_reason"] = "later phase inspection; starting conditions need not persist"
                reserve(last)
            elif earlier and len(priority) < total_images:
                # Another view of the earliest retrieved event helps distinguish
                # its partially visible beginning from a clear later capture.
                same_event = [frame for frame in timed if frame.get("evidence_id") == first.get("evidence_id")]
                if same_event:
                    reserve(max(same_event, key=lambda frame: float(frame["timestamp"])))
        time_anchor_count = len(priority)
        # A comparison's different targets each need a picture; event scores
        # alone can otherwise spend all slots on the more frequent action.
        if any(word in query.lower() for word in ("比较", "先", "还是", "分别", "before", "after", "compare")):
            focus_questions = {"phone": "手机", "sunglasses": "墨镜", "glasses": "眼镜", "bag": "背包",
                               "mask": "口罩", "coat": "外套", "hat": "帽子", "seatbelt": "安全带"}
            for focus in sorted(query_focus(query)):
                focus_query = focus_questions.get(focus)
                if not focus_query:
                    continue
                matching = [(frame, max((relevance(focus_query, observation) * actor_fit(query, observation)
                                        for observation in frame.get("_direct_observations", [])), default=0)) for frame in flat]
                matching = [(frame, score) for frame, score in matching if score >= 0.5]
                if matching:
                    reserve(max(matching, key=lambda row: row[1])[0])
        # Exact target observations outrank a bare event representative. Keep
        # distinct occurrences within an event, without spending every slot on
        # adjacent samples of the same action.
        strong = sorted([frame for frame in flat if frame.get("_coverage_observations") and frame.get("_inspection_supported")
                         and not frame.get("_actor_conflict")],
                        key=lambda frame: (-(frame.get("_local_relevance", 0) + 0.05 * frame.get("_action_support_level", 0)),
                                            frame.get("_complementarity_rank", 0) if frame.get("_complementarity_enabled") else 0,
                                            _safe_float(frame.get("timestamp"), 1e9)))
        strong_times = {}
        for frame in strong:
            event_id = frame.get("evidence_id")
            ts = frame.get("timestamp")
            previous = strong_times.setdefault(event_id, [])
            event = next((event for event in events if event.get("evidence_id") == event_id), {})
            selected_observations = frame.get("_coverage_observations", [])
            selected_bindings = [action_support(query, row) for row in selected_observations]
            selected_actions = {str(binding["action"]).strip().lower() for binding in selected_bindings if binding}
            selected_text = {re.sub(r"\W+", " ", query_actor_text(query, row).lower()).strip()
                             for row in selected_observations}
            breaks = []
            for observation in image_observation_records(event):
                if not isinstance(observation, dict) or actor_binding_fit(query, observation, event) == 0:
                    continue
                binding = action_support(query, observation)
                if selected_actions and binding:
                    changed = str(binding["action"]).strip().lower() not in selected_actions or binding["status"] == "negative"
                else:
                    # Legacy free-text observations have no explicit action
                    # fields. A distinct intervening observation is a candidate
                    # boundary, not a claim about a particular named action.
                    text = re.sub(r"\W+", " ", query_actor_text(query, observation).lower()).strip()
                    changed = bool(selected_text) and text not in selected_text
                point, _ = action_window(observation)
                if changed and point is not None:
                    breaks.append(point)
            if ts is not None and any(abs(float(ts)-value) < 8 and not any(min(float(ts), value) < float(point) < max(float(ts), value) for point in breaks) for value in previous):
                continue
            reserve(frame)
            if ts is not None:
                previous.append(float(ts))
        supported = [frame for frame in flat if frame.get("_inspection_supported")]
        for group in groups:
            candidate = next((frame for frame in group if frame.get("_inspection_supported")), None)
            if candidate is not None:
                reserve(candidate)
        # Only directly supported observations/ambiguous target candidates may
        # fill spare slots. A shared event caption is not a reason to fill six.
        tail = [frame for frame in supported if frame["path"] not in seen_paths]
        for frame in tail:
            if not frame.get("selection_reason", "").startswith("ambiguous target inspection"):
                frame["selection_reason"] = "direct observation supports target inspection"
        unknown = [frame for frame in flat + unknown_inspection_pool
                   if frame.get("_inspection_unknown") and frame["path"] not in seen_paths]
        event_order = {event.get("evidence_id"): i for i,event in enumerate(events)}
        unknown.sort(key=lambda frame: (-bool(frame.get("_cache_scoped_current")),
            event_order.get(frame.get("evidence_id"),len(events)), -bool(frame.get("_direct_observations")),
            -float(frame["timestamp"]) if frame.get("_cache_scoped_current") else -frame.get("_local_relevance",0)))
        fallback, seen_events, unknown_paths = [], set(), set()
        for frame in unknown:
            event_id, path = frame.get("evidence_id"), frame["path"]
            if event_id in seen_events or path in unknown_paths:
                continue
            # Keep one diverse inspection alternative per selected event.
            # Confirmed candidates stay ahead; unused final image capacity can
            # inspect more events instead of stopping at an arbitrary two.
            if len(fallback) >= max(0, total_images):
                break
            frame["selection_reason"] = ("bounded inspection fallback: saved current frame within cited span; content unverified"
                if frame.get("_cache_scoped_current") else "bounded inspection fallback: generic observation leaves target unverified"
                if frame.get("_direct_observations") else "bounded inspection fallback: selected event lacks localized labels")
            fallback.append(frame); seen_events.add(event_id); unknown_paths.add(path)
        # Do not let a richer ST annotation spend every slot before an already
        # selected LT/cache event gets even one look. Reserve at most one per
        # represented memory source; other image slots remain query-ranked.
        source_representatives = []
        represented_sources = set()
        for event in events:
            source = event.get("source_memory")
            if source not in {"M_lt","M_st","M_cache"} or source in represented_sources:
                continue
            candidate = next((frame for frame in priority + tail if frame.get("evidence_id") == event.get("evidence_id")), None)
            if candidate is not None:
                source_representatives.append(candidate); represented_sources.add(source)
        if len(source_representatives) > 1:
            fixed = priority[:time_anchor_count]
            priority = fixed + [frame for frame in source_representatives if frame not in fixed] + [
                frame for frame in priority[time_anchor_count:] if frame not in source_representatives]
            tail = [frame for frame in tail if frame not in source_representatives]
        # An unsupported top event still deserves one look ahead of repeated
        # supported frames. Its label is explicitly an inspection candidate.
        top_id = events[0].get("evidence_id") if events else None
        top_fallback = [frame for frame in unknown if frame.get("evidence_id") == top_id]
        if top_fallback and not any(frame.get("evidence_id") == top_id for frame in priority):
            reserve_count = min(per_event, 2 if total_images >= 4 else 1)
            ordered_top = sorted(top_fallback, key=lambda frame: _safe_float(frame.get("timestamp")))
            # With no localized observations, inspect both ends of the highest
            # ranked event instead of letting richly captioned lower events
            # consume every image. These remain explicitly unverified views.
            chosen_top = [ordered_top[0]]
            if reserve_count > 1 and ordered_top[-1]["path"] != ordered_top[0]["path"]:
                chosen_top.append(ordered_top[-1])
            for frame in chosen_top:
                frame["selection_reason"] = "bounded inspection fallback: highest-ranked sparse event temporal coverage; content unverified"
            priority = priority[:time_anchor_count] + chosen_top + priority[time_anchor_count:]
            fallback = [frame for frame in fallback if frame not in chosen_top]
        return priority + tail + fallback

    def _content_signature(self, path: str) -> tuple | None:
        """Exact content identity; never collapse similar-looking action states."""
        if self.session_dir is None:
            return None
        import hashlib
        base = self.runtime_session_dir if self._image_path_runtime_scoped(path) else self.session_dir
        try:
            raw = (base / path).read_bytes()
        except OSError:
            return None
        if not raw:
            return None  # Empty placeholders contain no image evidence to compare.
        try:
            from PIL import Image
            with Image.open(base / path) as image:
                pixels = image.convert("RGBA")
                return ("pixels", pixels.size, hashlib.sha256(pixels.tobytes()).digest())
        except (ImportError, OSError, ValueError):
            return ("bytes", hashlib.sha256(raw).digest())

    def _near_duplicate(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        signature = self._content_signature(str(left["path"]))
        return signature is not None and signature == self._content_signature(str(right["path"]))

    def _dedup_frames(
        self,
        frames: list[dict[str, Any]],
        limit: int | None,
        per_segment_limit: int = 2,
        *,
        compare_content: bool = True,
    ) -> tuple[list[dict[str, Any]], int]:
        selected: list[dict[str, Any]] = []
        seen_paths: dict[str, dict[str, Any]] = {}
        seen_content: dict[tuple, dict[str, Any]] = {}
        per_segment: dict[str, int] = {}
        removed = 0
        for frame in frames:
            frame = self._resolve_frame_path(frame)
            path = str(frame.get("path") or "")
            if path in seen_paths:
                seen_paths[path].setdefault("duplicate_provenance", []).append({k: frame.get(k) for k in ("path", "source", "evidence_id", "timestamp", "segment_id", "canonical_segment_id")})
                removed += 1
                continue
            if not path or not self._path_exists(path):
                removed += 1
                continue
            signature = self._content_signature(path) if compare_content else None
            if signature is not None:
                # Identical pixels can recur at distinct observed periods.
                # Only cross-source copies of the same capture share a slot.
                capture = (_normalize_frame_timestamp(frame.get("timestamp"), path)
                           if frame.get("timestamp") is not None else _timestamp_from_frame_path(path))
                signature = (capture, signature)
            if signature is not None and signature in seen_content:
                owner = seen_content[signature]
                owner.setdefault("duplicate_provenance", []).append({k: frame.get(k) for k in ("path", "source", "evidence_id", "timestamp", "segment_id", "canonical_segment_id")})
                removed += 1
                continue
            segment = str(frame.get("canonical_segment_id") or frame.get("segment_id") or "")
            if segment and per_segment.get(segment, 0) >= per_segment_limit:
                removed += 1
                continue
            item = {k: v for k, v in frame.items() if not k.startswith("_")}
            selected.append(item)
            if signature is not None:
                seen_content[signature] = item
            seen_paths[path] = item
            if segment:
                per_segment[segment] = per_segment.get(segment, 0) + 1
            if limit is not None and len(selected) >= limit:
                break
        return selected, removed

    def _select_images(
        self,
        frames: list[dict[str, Any]],
        use_image: bool,
        per_evidence_limit: int,
        total_limit: int | None,
    ) -> list[str]:
        if not use_image or per_evidence_limit <= 0 or (total_limit is not None and total_limit <= 0):
            return []
        selected = []
        per_segment: dict[str, int] = {}
        for frame in frames:
            path = str(frame.get("path") or "")
            if not path:
                continue
            segment = str(frame.get("canonical_segment_id") or frame.get("segment_id") or path)
            if per_segment.get(segment, 0) >= per_evidence_limit:
                continue
            if not self._path_exists(path):
                continue
            selected.append(path)
            per_segment[segment] = per_segment.get(segment, 0) + 1
            if total_limit is not None and len(selected) >= total_limit:
                break
        return selected

    def _merge_preferred_images(self, preferred: list[str], existing: list[str], max_images: int | None) -> list[str]:
        if max_images is not None and max_images <= 0:
            return []
        selected: list[str] = []
        for path in preferred + existing:
            if not path or path in selected:
                continue
            if not self._path_exists(path):
                continue
            selected.append(path)
            if max_images is not None and len(selected) >= max_images:
                break
        return selected
