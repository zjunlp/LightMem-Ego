from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

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


def _cache_root_for_session(session_dir: Path) -> Path:
    return session_dir / "em2mem" / "cache"


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
        self.ttl_seconds = ttl_seconds or _env_int("EM2MEM_INTERACTION_CACHE_TTL_SECONDS", 3600)
        self.max_hot_entities = _env_int("EM2MEM_CACHE_MAX_HOT_ENTITIES", 50)
        self.max_hot_time_ranges = _env_int("EM2MEM_CACHE_MAX_HOT_TIME_RANGES", 20)
        self.max_hot_memories = _env_int("EM2MEM_CACHE_MAX_HOT_MEMORIES", 50)
        self._lock = threading.RLock()
        self._last_loaded_mtime = 0.0
        self.data = self._empty()
        self.load()

    def _empty(self) -> dict[str, Any]:
        now = utc_now_iso()
        return {
            "session_id": self.session_id,
            "created_at": now,
            "updated_at": now,
            "recent_interactions": [],
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
            if not self.cache_path.exists():
                return
            mtime = self.cache_path.stat().st_mtime
            if mtime > self._last_loaded_mtime:
                self.load()

    def flush(self) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.data["session_id"] = self.session_id
            self.data["updated_at"] = utc_now_iso()
            write_json_atomic(self.cache_path, self.data)
            if self.cache_path.exists():
                self._last_loaded_mtime = self.cache_path.stat().st_mtime

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self.data = self._empty()
            self.flush()
            return self.summary()

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._empty()
        data.update(payload)
        for key in ("recent_interactions", "hot_time_ranges", "hot_entities", "hot_memories", "warnings"):
            if not isinstance(data.get(key), list):
                data[key] = []
        data["recent_interactions"] = data["recent_interactions"][-self.max_interactions :]
        for interaction in data["recent_interactions"]:
            if isinstance(interaction, dict):
                interaction["evidence_frames"] = self._trim_evidence_frames(interaction.get("evidence_frames", []))
        data["hot_entities"] = data["hot_entities"][-self.max_hot_entities :]
        data["hot_time_ranges"] = data["hot_time_ranges"][-self.max_hot_time_ranges :]
        data["hot_memories"] = data["hot_memories"][-self.max_hot_memories :]
        return data

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
        with self._lock:
            interactions = list(self.data.get("recent_interactions", []))
            turn_id = f"turn_{len(interactions) + 1:06d}"
            entities = self._extract_entities(result)
            time_ranges = self._extract_time_ranges(result)
            retrieved_memory_ids = self._extract_memory_ids(result)
            retrieved_segment_ids = self._extract_segment_ids(result)
            visual_ids = self._extract_visual_ids(result)
            evidence_frames = self._trim_evidence_frames(result.get("evidence_frames", []))
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
                "entities": entities,
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_segment_ids": retrieved_segment_ids,
                "visual_ids": visual_ids,
                "evidence_frames": evidence_frames,
                "confidence": confidence,
            }
            interactions.append(turn)
            self.data["recent_interactions"] = interactions[-self.max_interactions :]
            self._merge_hot_time_ranges(time_ranges, turn_id)
            self._merge_hot_entities(entities, turn_id, retrieved_memory_ids, visual_ids, time_ranges)
            self._merge_hot_memories(result.get("retrieved_memories", []), turn_id)
            self.flush()
            return {
                "updated": True,
                "turn_id": turn_id,
                "recent_interactions_count": len(self.data.get("recent_interactions", [])),
                "hot_entities_count": len(self.data.get("hot_entities", [])),
                "hot_time_ranges_count": len(self.data.get("hot_time_ranges", [])),
            }

    def _extract_time_ranges(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        ranges = []
        for item in result.get("timestamps", []) or []:
            start = _safe_float(item.get("start"))
            end = _safe_float(item.get("end"), start)
            if end < start:
                end = start
            ranges.append({"start": start, "end": end, "score": _safe_float(item.get("score"), 0.75)})
        return ranges[:5]

    def _extract_memory_ids(self, result: dict[str, Any]) -> list[str]:
        ids = []
        for item in result.get("retrieved_memories", []) or []:
            memory_id = item.get("memory_id") or item.get("evidence_doc_id")
            if memory_id:
                ids.append(str(memory_id))
        return list(dict.fromkeys(ids))

    def _extract_segment_ids(self, result: dict[str, Any]) -> list[str]:
        ids = []
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
                add(name, "object", _safe_float(obj.get("confidence"), 0.65), "visual_objects", state)
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
        output = []
        for frame in list(frames or [])[:8]:
            if not isinstance(frame, dict):
                continue
            output.append({
                "path": frame.get("path"),
                "timestamp": _normalize_frame_timestamp(frame.get("timestamp"), str(frame.get("path") or frame.get("image_path") or "")),
                "caption": frame.get("caption"),
                "segment_id": _canonical_segment_id(frame.get("canonical_segment_id") or frame.get("segment_id") or frame.get("path")),
                "score": frame.get("fused_score") or frame.get("visual_score") or frame.get("score"),
                "source": frame.get("source", "query_result"),
            })
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
            memory_id = str(memory.get("memory_id") or memory.get("evidence_doc_id") or "")
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
