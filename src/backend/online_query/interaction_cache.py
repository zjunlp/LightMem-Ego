from __future__ import annotations

import os
import re
import threading
import fcntl
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from online_query.frame_timestamps import frame_timestamp_seconds
from online_time import seconds_value
from .evidence_schema import evidence_frames, normalize_evidence
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def _canonical_segment_id(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"(seg_\d{6}_\d{6})(?:_\d{4})?", text)
    if match:
        return match.group(1)
    parts = text.split("_")
    if len(parts) == 5 and parts[0] == "seg" and all(p.isdigit() for p in parts[1:]):
        return "_".join(parts[:4])
    return text


def _normalize_frame_timestamp(value: Any, path: str | None = None) -> float | None:
    timestamp = seconds_value(value)
    return frame_timestamp_seconds(timestamp, path) if timestamp is not None else None


def _cache_root_for_session(session_dir: Path) -> Path:
    return session_dir / "em2mem" / "cache"


def generation_outcome(result: dict[str, Any]) -> str:
    raw = result.get("raw") or {}
    debug = raw.get("llm_debug") or {}
    answer_debug = raw.get("answer_debug") or {}
    if result.get("status") in {"error", "failed", "not_ready"}:
        return "failed"
    if str(result.get("answer", "")).strip() in {"Unable to generate answer", "Unable to retrieve relevant evidence"}:
        return "failed"
    if debug.get("local_fallback_used") or answer_debug.get("local_fallback_used") or (answer_debug.get("fallback_used") and answer_debug.get("error_debug")):
        return "local_evidence_summary"
    if debug.get("generation_status"):
        return debug["generation_status"]
    if answer_debug.get("generation_status"):
        return answer_debug["generation_status"]
    return "multimodal_success" if InteractionCache._sent_image_paths(None, result) else "text_success"


def unresolved_localization(question: str, answer: str) -> bool:
    if not re.search(r"什么时候|何时|几秒|最早|第一次|\bwhen\b|earliest|first time", question, re.I):
        return False
    uncertain = bool(re.search(r"无法确定|不能确定|不能确认|未能确认|不足以确认|未确认|没有明确|不确定是否|unable to (?:confirm|determine|locate)|cannot (?:confirm|determine)|not enough evidence|could not (?:confirm|locate)", answer, re.I))
    if uncertain and re.search(r"精确|准确|exact|precise", answer, re.I):
        # A verified sampled occurrence remains a valid reference even when its
        # exact onset between frames is unknown. Do not erase the 26s anchor
        # just because the answer correctly qualifies the start time.
        for clause in re.split(r"[。；;\n]", answer):
            if re.search(r"无法|不能|没有|未确认|不足|cannot|unable|\bnot\b", clause, re.I):
                continue
            direct = re.search(r"画面|可见|清楚|显示|看到|shows|visible|observed", clause, re.I)
            if re.search(r"贴耳|耳边|\bear\b", question, re.I):
                direct = direct and re.search(r"贴.{0,5}耳|耳.{0,8}手机|phone.{0,30}\bear\b", clause, re.I)
            if direct:
                return False
    return uncertain


class InteractionCache:
    """Session-level interaction and retrieval cache.

    This is separate from SessionEngineCache. It stores recent dialogue,
    entities, hot time ranges, and hot memories for follow-up resolution.
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        cache_path: Path | None = None,
        max_interactions: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self.cache_path = cache_path or (_cache_root_for_session(session_dir) / "interaction_cache.json")
        self.max_interactions = max_interactions or _env_int("EM2MEM_INTERACTION_CACHE_MAX_TURNS", 20)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else _env_int("EM2MEM_INTERACTION_CACHE_TTL_SECONDS", 3600)
        self.max_hot_entities = _env_int("EM2MEM_CACHE_MAX_HOT_ENTITIES", 50)
        self.max_hot_time_ranges = _env_int("EM2MEM_CACHE_MAX_HOT_TIME_RANGES", 20)
        self.max_hot_memories = _env_int("EM2MEM_CACHE_MAX_HOT_MEMORIES", 50)
        self._lock = threading.RLock()
        self._last_loaded_mtime = 0.0
        self.data = self._empty()
        with self._write_lock():
            self.load()

    @contextmanager
    def _write_lock(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _empty(self) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "session_id": self.session_id,
            "created_at": now,
            "updated_at": now,
            "recent_interactions": [],
            "next_turn_number": 1,
            "hot_time_ranges": [],
            "hot_entities": [],
            "hot_memories": [],
            "warnings": [],
        }

    def load(self) -> None:
        with self._lock:
            if not self.cache_path.exists():
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.flush()
                return
            payload = read_json(self.cache_path, default=None)
            if not isinstance(payload, dict):
                self.data = self._empty()
                self.data["warnings"].append("interaction cache file was invalid; reinitialized")
                self.flush()
                return
            self.data = self._normalize_payload(payload)
            self._last_loaded_mtime = self.cache_path.stat().st_mtime

    def reload_if_changed(self) -> None:
        with self._lock:
            if self.cache_path.exists():
                mtime = self.cache_path.stat().st_mtime
                if mtime > self._last_loaded_mtime:
                    self.load()
            # Expiration must also work while the file is unchanged. Reading
            # removes stale anchors in memory; the next locked write persists
            # that cleanup together with the new turn.
            self._prune_expired(self.data)

    def _prune_expired(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.ttl_seconds <= 0:
            return data
        cutoff = time.time() - self.ttl_seconds

        def fresh(item: Any, field: str) -> bool:
            if not isinstance(item, dict):
                return False
            value = item.get(field)
            if not value:
                return True  # Legacy records without a date cannot be aged reliably.
            try:
                stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                return stamp.timestamp() >= cutoff
            except (TypeError, ValueError, OverflowError):
                return True

        for key, field in (("recent_interactions", "created_at"),
                           ("hot_entities", "last_accessed_at"),
                           ("hot_time_ranges", "last_accessed_at"),
                           ("hot_memories", "last_accessed_at")):
            data[key] = [item for item in data.get(key, []) if fresh(item, field)]
        return data

    def flush(self) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.data["session_id"] = self.session_id
            self.data["updated_at"] = utc_now_iso()
            write_json_atomic(self.cache_path, self.data)
            if self.cache_path.exists():
                self._last_loaded_mtime = self.cache_path.stat().st_mtime

    def clear(self) -> dict[str, Any]:
        with self._lock, self._write_lock():
            self.data = self._empty()
            self.flush()
            return self.summary()

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._empty()
        data.update(payload)
        for key in ("recent_interactions", "hot_time_ranges", "hot_entities", "hot_memories", "warnings"):
            if not isinstance(data.get(key), list):
                data[key] = []
        # Allocate independently of the bounded/expiring turn list. Migrate
        # legacy files before trimming so an older high ID is not reused.
        try:
            next_number = max(1, int(data.get("next_turn_number", 1)))
        except (TypeError, ValueError, OverflowError):
            next_number = 1
        ids = [turn.get("turn_id") for turn in data["recent_interactions"] if isinstance(turn, dict)]
        for key, field in (("hot_entities", "supporting_turn_ids"),
                           ("hot_time_ranges", "source_turn_ids"), ("hot_memories", "source_turn_ids")):
            for row in data[key]:
                if isinstance(row, dict):
                    ids.extend(row.get(field) or [])
                    row[field] = list(row.get(field) or [])[-self.max_interactions:]
        for turn_id in ids:
            match = re.fullmatch(r"turn_(\d+)", str(turn_id or ""))
            if match:
                next_number = max(next_number, int(match[1]) + 1)
        data["next_turn_number"] = next_number
        data["recent_interactions"] = data["recent_interactions"][-self.max_interactions :]
        for interaction in data["recent_interactions"]:
            if isinstance(interaction, dict):
                interaction["evidence_frames"] = self._trim_evidence_frames(interaction.get("evidence_frames", []))
                # Legacy citation windows could have been enlarged by action
                # descriptions. Clip them to their own saved captures without
                # joining separate occurrences or changing other time sources.
                captures = [seconds_value(frame.get("timestamp")) for frame in interaction["evidence_frames"]]
                captures = [stamp for stamp in captures if stamp is not None]
                for window in interaction.get("time_ranges", []) or []:
                    if not isinstance(window, dict) or window.get("source") != "answer_cited_sent_frames":
                        continue
                    start, end = seconds_value(window.get("start")), seconds_value(window.get("end"))
                    if start is None or end is None:
                        continue
                    observed = [stamp for stamp in captures if start <= stamp <= end]
                    if observed:
                        window.update(start=min(observed), end=max(observed))
                interaction["focus_entities"] = [entity for entity in interaction.get("focus_entities", []) if isinstance(entity, dict)][:4]
                interaction["focus_evidence_refs"] = [
                    {"evidence_id": str(ref.get("evidence_id") or "")[:200],
                     "source_memory": str(ref.get("source_memory") or "")[:40],
                     "start_time": seconds_value(ref.get("start_time")),
                     "end_time": seconds_value(ref.get("end_time")),
                     "keyframe_paths": [str(path)[:500] for path in (ref.get("keyframe_paths") or [])[:4]]}
                    for ref in interaction.get("focus_evidence_refs", []) if isinstance(ref, dict)][:6]
        data["hot_entities"] = data["hot_entities"][-self.max_hot_entities :]
        data["hot_time_ranges"] = data["hot_time_ranges"][-self.max_hot_time_ranges :]
        data["hot_memories"] = data["hot_memories"][-self.max_hot_memories :]
        return self._prune_expired(data)

    def summary(self) -> dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            return {
                "session_id": self.session_id,
                "cache_path": str(self.cache_path),
                "recent_interactions_count": len(self.data.get("recent_interactions", [])),
                "hot_entities": list(self.data.get("hot_entities", []))[: self.max_hot_entities],
                "hot_time_ranges": list(self.data.get("hot_time_ranges", []))[: self.max_hot_time_ranges],
                "hot_memories_count": len(self.data.get("hot_memories", [])),
                "updated_at": self.data.get("updated_at"),
                "warnings": list(self.data.get("warnings", []))[-5:],
            }

    def recent_interactions(self, limit: int = 5) -> list[dict[str, Any]]:
        self.reload_if_changed()
        with self._lock:
            return list(self.data.get("recent_interactions", []))[-limit:]

    def latest_context(self) -> dict[str, Any]:
        self.reload_if_changed()
        with self._lock:
            recent = list(self.data.get("recent_interactions", []))
            latest = recent[-1] if recent else {}
            return {
                "latest_interaction": latest,
                "recent_interactions": recent,
                "hot_entities": list(self.data.get("hot_entities", [])),
                "hot_time_ranges": list(self.data.get("hot_time_ranges", [])),
                "hot_memories": list(self.data.get("hot_memories", [])),
            }

    def update_from_query_result(
        self,
        question: str,
        resolved_question: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        result.setdefault("question", question)
        outcome = generation_outcome(result)
        result["generation_status"] = outcome
        if outcome in {"failed", "local_evidence_summary"}:
            return {"updated": False, "reason": outcome}
        with self._lock, self._write_lock():
            # Another lane/process may have completed a turn since this
            # instance was loaded. Append to the current on-disk turn list.
            self.load()
            interactions = list(self.data.get("recent_interactions", []))
            next_number = self.data["next_turn_number"]
            turn_id = f"turn_{next_number:06d}"
            self.data["next_turn_number"] = next_number + 1
            entities = self._extract_entities(result)
            time_ranges = self._extract_time_ranges(result)
            action_time_ranges = self._extract_action_time_ranges(result)
            retrieved_memory_ids = self._extract_memory_ids(result)
            retrieved_segment_ids = self._extract_segment_ids(result)
            visual_ids = self._extract_visual_ids(result)
            evidence_frames = list(result.get("evidence_frames", []) or [])
            sent_paths = self._sent_image_paths(result)
            if sent_paths is not None:
                evidence_frames = [frame for frame in evidence_frames if (frame.get("path") or frame.get("image_path")) in set(sent_paths)]
            cited = self._answer_cited_frames(result)
            if cited:
                evidence_frames = cited
            displayed_frames = list(evidence_frames)
            evidence_frames = self._trim_evidence_frames(evidence_frames)
            unresolved = unresolved_localization(question, str(result.get("answer") or ""))
            if unresolved:
                time_ranges = []
                action_time_ranges = []
                retrieved_memory_ids = retrieved_segment_ids = visual_ids = []
                evidence_frames = []
            confidence = self._estimate_confidence(result, entities, time_ranges)
            turn = {
                "turn_id": turn_id,
                "created_at": utc_now_iso(),
                "question": question,
                "resolved_question": resolved_question,
                "answer": result.get("answer", ""),
                "query_type": result.get("query_type"),
                "route_decision": result.get("route_decision", {}),
                "time_ranges": time_ranges,
                "action_time_ranges": action_time_ranges,
                "entities": entities,
                "focus_entities": self._extract_focus_entities(result, entities),
                "focus_evidence_refs": self._focus_evidence_refs(result),
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_segment_ids": retrieved_segment_ids,
                "visual_ids": visual_ids,
                "evidence_frames": evidence_frames,
                "displayed_evidence_frames": displayed_frames,
                "sent_image_paths": sent_paths or [],
                "generation_status": outcome,
                "anchor_status": "unresolved" if unresolved else "confirmed",
                "confidence": confidence,
            }
            interactions.append(turn)
            self.data["recent_interactions"] = interactions[-self.max_interactions :]
            if not unresolved:
                self._merge_hot_time_ranges(time_ranges, turn_id)
                self._merge_hot_entities(entities, turn_id, retrieved_memory_ids, visual_ids, time_ranges)
                self._merge_hot_memories(self._selected_events(result) if self._selected_events(result) is not None else result.get("retrieved_memories", []), turn_id)
            self.flush()
            return {
                "updated": True,
                "turn_id": turn_id,
                "recent_interactions_count": len(self.data.get("recent_interactions", [])),
                "hot_entities_count": len(self.data.get("hot_entities", [])),
                "hot_time_ranges_count": len(self.data.get("hot_time_ranges", [])),
            }

    def _extract_time_ranges(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        cited = self._answer_cited_frames(result)
        times = [float(frame["timestamp"]) for frame in cited if frame.get("timestamp") is not None]
        if times:
            # These windows locate cited observations. An action's full span
            # does not establish the same visual state throughout that span.
            from online_evidence_content import content_fields, as_text
            from online_query_constraints import actor_fit
            intervals = [(time, time) for time in times]
            events = (result.get("raw") or {}).get("selected_evidence", []) or result.get("selected_evidence", []) or []
            question = str(result.get("question") or "")
            ear_question = bool(re.search(r"贴耳|耳边|\bear\b", question, re.I))
            breaks = []
            if ear_question:
                for event in events:
                    for observation in content_fields(event).get("visual_observations", []) or []:
                        text = as_text(observation)
                        if actor_fit(question, observation) > 0 and re.search(r"phone|手机", text, re.I) and re.search(r"lower|puts? (?:it |the phone )?down|放下|移开", text, re.I):
                            if observation.get("timestamp") is not None:
                                breaks.append(float(observation["timestamp"]))
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1] + 10 and not any(merged[-1][1] < time < start for time in breaks):
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            return [{"start": start, "end": end, "score": 0.9, "source": "answer_cited_sent_frames"} for start, end in merged]
        question = str(result.get("question") or "")
        if result.get("retrieval_mode") == "current" and re.search(r"现在|当前|此刻|\bnow\b|\bcurrently\b|at (?:the )?moment", question, re.I):
            raw = result.get("raw") or {}
            selection = raw.get("current_selection") or {}
            frames = (result.get("evidence_frames") or []) + (selection.get("evidence_frames") or selection.get("selected_frames") or [])
            sent = set(self._sent_image_paths(result) or [])
            observed = [seconds_value(frame.get("timestamp")) for frame in frames
                        if (frame.get("path") or frame.get("image_path")) in sent]
            observed = [stamp for stamp in observed if stamp is not None]
            if observed:
                latest = max(observed)
                return [{"start": latest, "end": latest, "score": 0.9, "source": "current_latest_sent_frame"}]
            return []
        ranges = []
        for item in result.get("timestamps", []) or []:
            start = _safe_float(item.get("start"))
            end = _safe_float(item.get("end"), start)
            if end < start:
                end = start
            ranges.append({"start": start, "end": end, "score": _safe_float(item.get("score"), 0.75)})
        return ranges[:5]

    def _extract_action_time_ranges(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Keep source action intervals as context, never as citation times."""
        from online_evidence_content import visual_anchors
        times = [seconds_value(frame.get("timestamp")) for frame in self._answer_cited_frames(result)]
        times = [stamp for stamp in times if stamp is not None]
        if not times:
            return []
        events = (result.get("raw") or {}).get("selected_evidence", []) or result.get("selected_evidence", []) or []
        ranges = []
        for event in events:
            for anchor in visual_anchors(str(result.get("question") or ""), event):
                start, end = anchor["start"], anchor["end"]
                if end > start and any(start <= stamp <= end for stamp in times):
                    window = {"start": start, "end": end, "source": "selected_event_action_interval",
                              "evidence_id": event.get("evidence_id") or event.get("doc_id") or event.get("event_id")}
                    if window not in ranges:
                        ranges.append(window)
        return ranges[:12]

    def _answer_cited_frames(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Focus follow-ups using citations to actually submitted images only."""
        answer = str(result.get("answer") or "")
        sent = set(self._sent_image_paths(result) or [])
        if not answer or not sent:
            return []
        frames = [frame for frame in result.get("evidence_frames", []) or [] if (frame.get("path") or frame.get("image_path")) in sent]
        # Display order belongs to the answer, not to retrieval/sending order.
        # A model may receive [70, 40] but display 40 first and then 70.
        indexed = []
        for frame in frames:
            path = str(frame.get("path") or frame.get("image_path") or "")
            name = Path(path).name
            found = re.search(re.escape(path), answer) or (re.search(re.escape(name), answer) if name else None)
            if found:
                indexed.append((found.start(), frame))
        cited = []
        seen_paths = set()
        for _, frame in sorted(indexed, key=lambda row: row[0]):
            path = frame.get("path") or frame.get("image_path")
            if path not in seen_paths:
                cited.append(frame)
                seen_paths.add(path)
        if cited:
            return cited
        mentions = [(match.start(), float(match[1]), float(match[1])) for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:秒|seconds?(?![a-z]))", answer, flags=re.IGNORECASE)]
        # A range cites the actually submitted captures within that interval;
        # do not turn its final endpoint into the only follow-up anchor.
        for match in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(秒|seconds?|分钟?|minutes?)?\s*(?:[~～—–-]|到|至|\bto\b)\s*(\d+(?:\.\d+)?)\s*(秒|seconds?(?![a-z])|分钟?|minutes?(?![a-z]))", answer, flags=re.IGNORECASE):
            scale = lambda unit: 60 if unit and ("分" in unit or unit.lower().startswith("min")) else 1
            lo = float(match[1]) * scale(match[2] or match[4])
            hi = float(match[3]) * scale(match[4])
            mentions.append((match.start(), min(lo, hi), max(lo, hi)))
        for group in re.finditer(r"(\d+(?:\.\d+)?(?:\s*[、,，和及与]\s*\d+(?:\.\d+)?)+)\s*(?:秒|seconds?\b)", answer, flags=re.IGNORECASE):
            mentions.extend((group.start(1) + match.start(), float(match[0]), float(match[0]))
                            for match in re.finditer(r"\d+(?:\.\d+)?", group[1]))
        for _, lo, hi in sorted(mentions):
            for frame in sorted(frames, key=lambda row: float(row.get("timestamp") or 0)):
                path = frame.get("path") or frame.get("image_path")
                if path not in seen_paths and frame.get("timestamp") is not None and lo - 0.01 < float(frame["timestamp"]) < hi + 0.01:
                    cited.append(frame)
                    seen_paths.add(path)
        return cited

    def _sent_image_paths(self, result: dict[str, Any]) -> list[str] | None:
        raw = result.get("raw") or {}
        if "image_paths_used" in raw:
            return list(raw.get("image_paths_used") or [])
        debug = raw.get("llm_debug") or {}
        if "actual_image_paths" in debug:
            return list(debug.get("actual_image_paths") or [])
        answer_debug = raw.get("answer_debug") or {}
        if "image_paths_used" in answer_debug:
            return list(answer_debug.get("image_paths_used") or [])
        return None

    def _selected_events(self, result: dict[str, Any]) -> list[dict[str, Any]] | None:
        """None denotes legacy output; an explicit empty final set stays empty."""
        raw = result.get("raw") or {}
        selected = result.get("selected_evidence", raw.get("selected_evidence"))
        if selected:
            return [normalize_evidence(event) for event in selected if isinstance(event, dict)]
        context = raw.get("current_context") or result.get("current_context") or {}
        selection = raw.get("current_selection") or {}
        if context and selection and (result.get("retrieval_mode") == "current" or result.get("current_used") or selected is None):
            frames = evidence_frames(selection, source="M_cur")
            sent = self._sent_image_paths(result)
            if sent is not None:
                frames = [frame for frame in frames if (frame.get("path") or frame.get("image_path")) in sent]
            times = [seconds_value(frame.get("timestamp")) for frame in frames]
            times = [stamp for stamp in times if stamp is not None]
            if times:
                opened = context.get("open_event") or {}
                return [normalize_evidence({"evidence_id": opened.get("open_event_id") or f"M_cur_{min(times):g}_{max(times):g}",
                         "source_memory": "M_cur", "start_time": min(times), "end_time": max(times),
                         "evidence_frames": frames,
                         "keyframe_paths": [frame.get("path") for frame in frames if frame.get("path")]})]
        return [] if selected is not None else None

    def _focus_evidence_refs(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        events = self._selected_events(result)
        if events is None:
            events = result.get("retrieved_memories", []) or []
        focus = self._extract_focus_entities(result, self._extract_entities(result))
        if focus:
            focused_events = []
            for event in events:
                structured = self._extract_entities({"selected_evidence": [event]})
                names = [str(entity.get("name") or "").lower() for entity in structured]
                supports_focus = any(
                    str(entity.get("name") or "").lower() in names
                    or any(self._focus_name_matches(str(entity.get("name") or ""), name) for name in names)
                    for entity in focus)
                # Unrefined current frames still ground the observed-time
                # reference; object-topic matching does not assert their pixels.
                unrefined_current = event.get("source_memory") == "M_cur" and not structured
                if supports_focus or unrefined_current:
                    focused_events.append(event)
            events = focused_events
        refs = []
        sent = self._sent_image_paths(result)
        for event in events[:6]:
            paths = event.get("keyframe_paths") or []
            if sent is not None:
                paths = [path for path in paths if path in sent]
            refs.append({"evidence_id": str(event.get("evidence_id") or event.get("memory_id") or event.get("evidence_doc_id") or "")[:200],
                         "source_memory": str(event.get("source_memory") or "legacy")[:40],
                         "start_time": seconds_value(event.get("start_time")),
                         "end_time": seconds_value(event.get("end_time")),
                         "keyframe_paths": [str(path)[:500] for path in paths[:4]]})
        return refs

    @staticmethod
    def _normalized_focus_name(text: str) -> str:
        from online_evidence_content import CONCEPT_GROUPS
        # Normalize established aliases, preserving arbitrary descriptive words
        # (small, plastic, striped...) rather than enumerating attributes.
        aliases = [(word, f" concept{index} ") for index, group in enumerate(CONCEPT_GROUPS[:12] + CONCEPT_GROUPS[16:21]) for word in group]
        normalized = text.lower()
        pattern = "|".join(re.escape(word) if not word.isascii() else r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])"
                           for word, _ in sorted(aliases, key=lambda row: -len(row[0])))
        mapping = dict(aliases)
        normalized = re.sub(pattern, lambda match: mapping[match.group(0)], normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @classmethod
    def _focus_name_matches(cls, target: str, candidate: str) -> bool:
        target_name = cls._normalized_focus_name(target)
        candidate_name = cls._normalized_focus_name(candidate)
        if not candidate_name:
            return False
        pattern = re.escape(candidate_name)
        if candidate_name[0].isascii():
            pattern = r"(?<![a-z0-9])" + pattern
        if candidate_name[-1].isascii():
            pattern += r"(?![a-z0-9])"
        return bool(re.search(pattern, target_name))

    def _extract_focus_entities(self, result: dict[str, Any], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from online_evidence_content import query_focus, CONCEPT_GROUPS
        # Historical/background clauses must not compete with the answer's subject.
        lead = re.split(r"[。；;\n]|(?<![0-9])\.(?![0-9])|此前|之前|earlier|previously", str(result.get("answer") or ""), maxsplit=1, flags=re.I)[0]
        concepts = query_focus(lead)
        # Conservative ambiguity guard for explicitly stated multiplicity;
        # these language cues do not estimate a count or assert presence.
        multiple = bool(re.search(r"\b(?:another|two|both|multiple)\b|另一个|另外一个|两个|两只|两件|多个|两种", lead, re.I))
        objects = [item for item in entities if item.get("type") not in {"person", "scene"}]
        named = [item for item in objects if self._focus_name_matches(lead, str(item.get("name") or ""))]
        named_concepts = set().union(*(query_focus(str(item.get("name") or "")) for item in named)) if named else set()
        matches = named + [item for item in objects if item not in named
                           and (query_focus(str(item.get("name") or "")) & concepts)
                           and (multiple or not (query_focus(str(item.get("name") or "")) & named_concepts))]
        if matches:
            events = self._selected_events(result) or []
            event_names = [set(str(entity.get("name") or "").lower()
                               for entity in self._extract_entities({"selected_evidence": [event]}))
                           for event in events]
            def contained_alias(item: dict[str, Any]) -> bool:
                name = self._normalized_focus_name(str(item.get("name") or ""))
                raw_name = str(item.get("name") or "").lower()
                containing = [names for names in event_names if raw_name in names]
                return bool(containing) and any(
                    (self._normalized_focus_name(str(other.get("name") or "")) != name
                     or (len(str(other.get("name") or "")), -matches.index(other)) > (len(raw_name), -matches.index(item)))
                    and self._focus_name_matches(str(other.get("name") or ""), str(item.get("name") or ""))
                    and all(str(other.get("name") or "").lower() in names for names in containing)
                    for other in matches if other is not item)
            if not multiple:
                matches = [item for item in matches if not contained_alias(item)]
            # Keep distinct equally salient objects; aliases of a generic topic
            # are only used when the structured evidence did not name it.
            uncertain = re.search(r"无法|不能|未能|未确认|没有|不确定|并未|没拿|未拿|没有拿|\b(?:not|no|cannot|unable|uncertain|maybe|possibly|may|might)\b", lead, re.I)
            if uncertain:
                return [dict(item, source="answer_topic_hint", last_state=None) for item in matches[:4]]
            return matches[:4]
        hints = []
        for group in CONCEPT_GROUPS[:12]:
            for term in group:
                if query_focus(term) & concepts:
                    hints.append({"name": term, "type": "object", "aliases": list(group),
                                  "last_state": None, "confidence": 0.5, "source": "answer_topic_hint"})
                    break
        return hints[:4]

    def _extract_memory_ids(self, result: dict[str, Any]) -> list[str]:
        ids = []
        selected = self._selected_events(result)
        if selected is not None:
            return list(dict.fromkeys(str(item.get("evidence_id") or item.get("memory_id")) for item in selected if item.get("evidence_id") or item.get("memory_id")))
        for item in result.get("retrieved_memories", []) or []:
            memory_id = item.get("memory_id") or item.get("evidence_doc_id")
            if memory_id:
                ids.append(str(memory_id))
        return list(dict.fromkeys(ids))

    def _extract_segment_ids(self, result: dict[str, Any]) -> list[str]:
        ids = []
        selected = self._selected_events(result)
        if selected is not None:
            return list(dict.fromkeys(_canonical_segment_id(item.get("canonical_segment_id") or item.get("segment_id") or item.get("evidence_id")) for item in selected if item.get("canonical_segment_id") or item.get("segment_id") or item.get("evidence_id")))
        for item in result.get("retrieved_memories", []) or []:
            segment_id = item.get("canonical_segment_id") or item.get("segment_id") or item.get("evidence_doc_id")
            if segment_id:
                ids.append(_canonical_segment_id(segment_id))
        for item in result.get("fused_results", []) or []:
            segment_id = item.get("canonical_segment_id") or item.get("segment_id")
            if segment_id:
                ids.append(_canonical_segment_id(segment_id))
        return list(dict.fromkeys(ids))

    def _extract_visual_ids(self, result: dict[str, Any]) -> list[str]:
        selected = self._selected_events(result)
        if selected is not None:
            return list(dict.fromkeys(str(item["visual_id"]) for item in selected if item.get("visual_id")))
        ids = []
        for item in result.get("visual_results", []) or []:
            if item.get("visual_id"):
                ids.append(str(item["visual_id"]))
        for fused in result.get("fused_results", []) or []:
            for item in fused.get("visual_items", []) or []:
                if item.get("visual_id"):
                    ids.append(str(item["visual_id"]))
        return list(dict.fromkeys(ids))

    def _extract_entities(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}

        def add(name: Any, entity_type: str = "object", confidence: float = 0.55, source: str = "unknown", state: str | None = None) -> None:
            text = str(name or "").strip()
            if not text or len(text) < 2:
                return
            key = _normalize_key(text)
            if not key or key in {"segment", "scene", "video", "none", "null"}:
                return
            item = candidates.setdefault(
                key,
                {
                    "name": text,
                    "type": entity_type,
                    "aliases": [text],
                    "last_state": state,
                    "confidence": confidence,
                    "source": source,
                },
            )
            item["confidence"] = max(_safe_float(item.get("confidence")), confidence)
            if state and not item.get("last_state"):
                item["last_state"] = state
            if text not in item["aliases"]:
                item["aliases"].append(text)

        selected = self._selected_events(result)
        if selected is not None:
            for event in selected:
                for visual in (event, event.get("metadata") or {}, event.get("content") or {}):
                    self._extract_structured_visual_entities(visual, add)
                    for entity in visual.get("entities", []) or []:
                        if isinstance(entity, dict):
                            add(entity.get("name") or entity.get("label"), entity.get("type", "object"), _safe_float(entity.get("confidence"), 0.6), "selected_entities")
                        else:
                            add(entity, "person" if re.search(r"\b(?:man|woman|person)\b|男子|女人", str(entity), re.I) else "object", 0.6, "selected_entities")
            return sorted(candidates.values(), key=lambda item: -_safe_float(item.get("confidence")))[:20]

        for fused in result.get("fused_results", []) or []:
            text = fused.get("text") or {}
            if isinstance(text, dict):
                caption = str(text.get("caption") or "")
                self._extract_named_terms_from_text(caption, add, "caption")
            for visual in fused.get("visual_items", []) or []:
                self._extract_structured_visual_entities(visual, add)
        for visual in result.get("visual_results", []) or []:
            self._extract_structured_visual_entities(visual, add)
        for fact in result.get("supporting_semantic_facts", []) or []:
            triple = fact.get("triple") if isinstance(fact, dict) else None
            if isinstance(triple, list):
                if len(triple) >= 1:
                    add(triple[0], "entity", _safe_float(fact.get("confidence"), 0.55), "semantic_fact")
                if len(triple) >= 3 and not self._looks_like_scene(triple[2]):
                    add(triple[2], "entity", _safe_float(fact.get("confidence"), 0.55), "semantic_fact")
        for memory in result.get("retrieved_memories", []) or []:
            self._extract_named_terms_from_text(str(memory.get("caption") or ""), add, "retrieved_memory")
        return sorted(candidates.values(), key=lambda item: -_safe_float(item.get("confidence")))[:20]

    def _extract_structured_visual_entities(self, visual: dict[str, Any], add: Any) -> None:
        for obj in visual.get("visual_objects", []) or []:
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("label") or obj.get("object")
                state = ", ".join(str(x) for x in obj.get("attributes", [])[:3]) if isinstance(obj.get("attributes"), list) else None
                add(name, "person" if re.search(r"\b(?:man|woman|person)\b|男子|女人", str(name), re.I) else "object", _safe_float(obj.get("confidence"), 0.65), "visual_objects", state)
            else:
                add(obj, "object", 0.6, "visual_objects")
        for action in visual.get("main_actions", []) or []:
            if isinstance(action, dict):
                add(action.get("actor"), "person", _safe_float(action.get("confidence"), 0.6), "main_actions", action.get("action"))
                for obj in action.get("objects", []) or []:
                    add(obj, "object", _safe_float(action.get("confidence"), 0.6), "main_actions", action.get("action"))
        for change in visual.get("state_changes", []) or []:
            if isinstance(change, dict):
                state = f"{change.get('before')} -> {change.get('after')}"
                add(change.get("entity"), "object", _safe_float(change.get("confidence"), 0.6), "state_changes", state)

    def _extract_named_terms_from_text(self, text: str, add: Any, source: str) -> None:
        common = [
            "手机", "屏幕", "桌子", "白板", "盒子", "包", "人", "杯子", "电脑", "笔记本",
            "phone", "smartphone", "screen", "table", "whiteboard", "box", "case", "bag",
            "person", "people", "laptop", "notebook", "tripod", "kitchen", "counter", "sink",
        ]
        lower = text.lower()
        for term in common:
            if term.lower() in lower:
                entity_type = "person" if term in {"人", "person", "people"} else "object"
                add(term, entity_type, 0.5, source)

    def _looks_like_scene(self, value: Any) -> bool:
        text = str(value or "").lower()
        return any(keyword in text for keyword in ("area", "room", "scene", "indoor", "outdoor", "厨房", "房间", "区域"))

    def _trim_evidence_frames(self, frames: Any) -> list[dict[str, Any]]:
        # Retain frame provenance through ordinary follow-ups, without storing
        # arbitrary nested event payloads inside each origin.
        summary_keys = ("path", "image_path", "source_path", "timestamp", "capture_time_seconds", "capture_time_origin",
                        "caption", "keyframe_caption", "segment_id", "canonical_segment_id", "evidence_id",
                        "event_id", "memory_id", "source", "source_memory", "original_source", "source_type",
                        "score", "fused_score", "visual_score", "semantic_score", "retrieval_score",
                        "raw_retrieval_score", "final_score", "score_policy", "role", "schema_path", "selection_association")
        def summary(row):
            output = {}
            for key in summary_keys:
                value = row.get(key)
                if isinstance(value, str):
                    output[key] = value[:1200 if key in {"caption", "keyframe_caption"} else 500]
                elif value is None or isinstance(value, (int, float, bool)):
                    if key in row:
                        output[key] = value
            return output

        output = []
        for frame in list(frames or [])[:8]:
            if not isinstance(frame, dict):
                continue
            normalized = evidence_frames(frame)
            if not normalized:
                continue
            record = normalized[0]
            trimmed = summary(record)
            trimmed["segment_id"] = _canonical_segment_id(record.get("canonical_segment_id") or record.get("segment_id") or record["path"])
            trimmed["score"] = next((trimmed[key] for key in ("fused_score", "visual_score", "score") if trimmed.get(key) is not None), None)
            trimmed["source"] = record.get("source") or "query_result"
            origins = []
            native_origins = [origin for origin in record.get("frame_origins", []) if isinstance(origin, dict)]
            representatives = []
            sources = set()
            for origin in native_origins:
                source_key = (str(origin.get("source")), str(origin.get("original_source")))
                if source_key not in sources:
                    representatives.append(origin)
                    sources.add(source_key)
            retained = representatives + [origin for origin in native_origins if origin not in representatives]
            for origin in retained[:4]:
                if isinstance(origin, dict):
                    entry = summary(origin)
                    if isinstance(origin.get("metadata"), dict):
                        entry["metadata"] = summary(origin["metadata"])
                    origins.append(entry)
            trimmed["frame_origins"] = origins
            output.append(trimmed)
        return output

    def _estimate_confidence(self, result: dict[str, Any], entities: list[dict[str, Any]], ranges: list[dict[str, Any]]) -> float:
        base = 0.55
        if result.get("retrieved_memories"):
            base += 0.12
        if entities:
            base += 0.08
        if ranges:
            base += 0.08
        if result.get("answer") and result.get("answer") != "Unable to generate answer":
            base += 0.1
        return min(base, 0.95)

    def _merge_hot_time_ranges(self, ranges: list[dict[str, Any]], turn_id: str) -> None:
        hot = list(self.data.get("hot_time_ranges", []))
        for item in ranges:
            start = round(_safe_float(item.get("start")), 2)
            end = round(_safe_float(item.get("end"), start), 2)
            found = None
            for existing in hot:
                if abs(_safe_float(existing.get("start")) - start) < 0.5 and abs(_safe_float(existing.get("end")) - end) < 0.5:
                    found = existing
                    break
            if found is None:
                found = {"start": start, "end": end, "source_turn_ids": [], "confidence": 0.0}
                hot.append(found)
            if turn_id not in found["source_turn_ids"]:
                found["source_turn_ids"].append(turn_id)
            found["source_turn_ids"] = found["source_turn_ids"][-self.max_interactions:]
            found["confidence"] = max(_safe_float(found.get("confidence")), _safe_float(item.get("score"), 0.7))
            found["last_accessed_at"] = utc_now_iso()
        hot.sort(key=lambda x: str(x.get("last_accessed_at", "")), reverse=True)
        self.data["hot_time_ranges"] = hot[: self.max_hot_time_ranges]

    def _merge_hot_entities(
        self,
        entities: list[dict[str, Any]],
        turn_id: str,
        memory_ids: list[str],
        visual_ids: list[str],
        ranges: list[dict[str, Any]],
    ) -> None:
        hot = {str(item.get("entity_key") or _normalize_key(item.get("canonical_name"))): item for item in self.data.get("hot_entities", [])}
        last_seen = ranges[0]["start"] if ranges else None
        for entity in entities:
            key = _normalize_key(entity.get("name"))
            if not key:
                continue
            item = hot.setdefault(
                key,
                {
                    "entity_key": key,
                    "canonical_name": entity.get("name"),
                    "aliases": [],
                    "entity_type": entity.get("type", "entity"),
                    "supporting_turn_ids": [],
                    "supporting_memory_ids": [],
                    "supporting_visual_ids": [],
                    "confidence": 0.0,
                },
            )
            for alias in entity.get("aliases", []) or [entity.get("name")]:
                if alias and alias not in item["aliases"]:
                    item["aliases"].append(alias)
            if turn_id not in item["supporting_turn_ids"]:
                item["supporting_turn_ids"].append(turn_id)
            item["supporting_turn_ids"] = item["supporting_turn_ids"][-self.max_interactions:]
            item["supporting_memory_ids"] = list(dict.fromkeys(list(item.get("supporting_memory_ids", [])) + memory_ids))[-10:]
            item["supporting_visual_ids"] = list(dict.fromkeys(list(item.get("supporting_visual_ids", [])) + visual_ids))[-10:]
            item["confidence"] = max(_safe_float(item.get("confidence")), _safe_float(entity.get("confidence"), 0.55))
            item["last_state"] = entity.get("last_state") or item.get("last_state")
            item["last_seen_time"] = last_seen if last_seen is not None else item.get("last_seen_time")
            item["last_accessed_at"] = utc_now_iso()
        values = sorted(hot.values(), key=lambda x: str(x.get("last_accessed_at", "")), reverse=True)
        self.data["hot_entities"] = values[: self.max_hot_entities]

    def _merge_hot_memories(self, memories: Any, turn_id: str) -> None:
        hot = {str(item.get("memory_id")): item for item in self.data.get("hot_memories", []) if item.get("memory_id")}
        for memory in list(memories or []):
            if not isinstance(memory, dict):
                continue
            memory_id = str(memory.get("memory_id") or memory.get("evidence_doc_id") or memory.get("evidence_id") or "")
            if not memory_id:
                continue
            item = hot.setdefault(
                memory_id,
                {
                    "memory_id": memory_id,
                    "segment_id": _canonical_segment_id(memory.get("canonical_segment_id") or memory.get("segment_id") or memory_id),
                    "source_turn_ids": [],
                    "confidence": 0.0,
                },
            )
            if turn_id not in item["source_turn_ids"]:
                item["source_turn_ids"].append(turn_id)
            item["source_turn_ids"] = item["source_turn_ids"][-self.max_interactions:]
            item["start"] = memory.get("start_time")
            item["end"] = memory.get("end_time")
            item["confidence"] = max(_safe_float(item.get("confidence")), _safe_float(memory.get("score"), 0.7))
            item["last_accessed_at"] = utc_now_iso()
        values = sorted(hot.values(), key=lambda x: str(x.get("last_accessed_at", "")), reverse=True)
        self.data["hot_memories"] = values[: self.max_hot_memories]


def get_interaction_cache_summary(session_dir: Path, session_id: str) -> dict[str, Any]:
    cache = InteractionCache(session_id=session_id, session_dir=session_dir)
    return cache.summary()


def clear_interaction_cache(session_dir: Path, session_id: str) -> dict[str, Any]:
    cache = InteractionCache(session_id=session_id, session_dir=session_dir)
    return cache.clear()
