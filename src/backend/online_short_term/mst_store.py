from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_short_term.schemas import (
    DEFAULT_SESSIONS_ROOT,
    build_retrieval_text,
    effective_caption,
    env_float,
    env_int,
    event_time_id,
    rel_to_session,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class MSTStore:
    def __init__(
        self,
        session_dir: Path,
        recent_window_seconds: float | None = None,
        max_events: int | None = None,
        archive_max_events: int | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.session_id = session_dir.name
        self.short_term_dir = session_dir / "short_term"
        self.events_path = self.short_term_dir / "micro_events.jsonl"
        self.state_path = self.short_term_dir / "mst_state.json"
        self.index_path = self.short_term_dir / "mst_index.json"
        self.id_mapping_path = self.short_term_dir / "mst_id_mapping.json"
        self.archive_dir = self.short_term_dir / "archive"
        self.archive_events_path = self.archive_dir / "micro_events_all.jsonl"
        self.archive_state_path = self.archive_dir / "archive_state.json"
        self.refine_dir = self.short_term_dir / "refine"
        self.refine_state_path = self.refine_dir / "refine_state.json"
        self.ready_windows_path = self.refine_dir / "refined_ready_windows.json"
        self.recent_window_seconds = recent_window_seconds or env_float("EM2MEM_MST_RECENT_WINDOW_SECONDS", 1800.0)
        self.max_events = max_events or env_int("EM2MEM_MST_MAX_EVENTS", 1000)
        self.archive_max_events = archive_max_events if archive_max_events is not None else env_int("EM2MEM_MST_ARCHIVE_MAX_EVENTS", 0)

    @classmethod
    def from_session(cls, session_id: str, sessions_root: str | Path = DEFAULT_SESSIONS_ROOT) -> "MSTStore":
        root = Path(sessions_root)
        return cls(root / session_id)

    def is_ready(self) -> bool:
        state = self.get_state()
        return bool(state.get("short_term_ready")) and self.events_path.exists()

    def load_events(self) -> list[dict[str, Any]]:
        return [self.normalize_event(event) for event in _load_jsonl(self.events_path)]

    def load_archive_events(self) -> list[dict[str, Any]]:
        return [self.normalize_event(event) for event in _load_jsonl(self.archive_events_path)]

    def save_events(self, events: list[dict[str, Any]]) -> None:
        events = [self.normalize_event(event) for event in events]
        events = self.evict_old_events(events)
        _write_jsonl(self.events_path, events)
        self._write_index(events)
        self._write_state(events)

    def save_archive_events(self, events: list[dict[str, Any]], bump_version: bool = True) -> None:
        events = [self.normalize_event(event) for event in events]
        events = sorted(events, key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("event_id", ""))))
        if self.archive_max_events and self.archive_max_events > 0 and len(events) > self.archive_max_events:
            events = events[-self.archive_max_events :]
        _write_jsonl(self.archive_events_path, events)
        self._write_archive_state(events, bump_version=bump_version)

    def append_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        self.short_term_dir.mkdir(parents=True, exist_ok=True)
        incoming = [self.normalize_event(event) for event in events]
        active = self._merge_by_event_id(self.load_events(), incoming)
        archive_error = None
        try:
            archive = self._merge_by_event_id(self.load_archive_events(), incoming)
            self.save_archive_events(archive, bump_version=True)
        except Exception as exc:  # active must stay usable even if archive fails
            archive_error = str(exc)
        self.save_events(active)
        if archive_error:
            state = self.get_state()
            pending = list(state.get("pending_archive_event_ids", []) or [])
            pending.extend(event["event_id"] for event in incoming)
            state["pending_archive_event_ids"] = list(dict.fromkeys(pending))[-100:]
            state["archive_error"] = archive_error
            write_json_atomic(self.state_path, state)
        return incoming

    def update_events(self, updates: list[dict[str, Any]]) -> dict[str, bool]:
        if not updates:
            return {"active_updated": False, "archive_updated": False}
        normalized = [self.normalize_event(event) for event in updates]
        active_before = self.load_events()
        archive_before = self.load_archive_events()
        active_ids = {str(event.get("event_id")) for event in active_before if event.get("event_id")}
        active_updates = [event for event in normalized if str(event.get("event_id")) in active_ids]
        active_after = self._merge_by_event_id(active_before, active_updates)
        archive_after = self._merge_by_event_id(archive_before, normalized)
        active_updated = json.dumps(active_before, sort_keys=True, ensure_ascii=False) != json.dumps(active_after, sort_keys=True, ensure_ascii=False)
        archive_updated = json.dumps(archive_before, sort_keys=True, ensure_ascii=False) != json.dumps(archive_after, sort_keys=True, ensure_ascii=False)
        if archive_updated:
            self.save_archive_events(archive_after, bump_version=True)
        if active_updated:
            self.save_events(active_after)
        return {"active_updated": active_updated, "archive_updated": archive_updated}

    def backfill_transcript_segments(
        self,
        segments: list[dict[str, Any]],
        reason: str = "stream_asr",
    ) -> dict[str, Any]:
        normalized_segments = [dict(seg) for seg in segments if isinstance(seg, dict) and str(seg.get("text") or "").strip()]
        if not normalized_segments:
            return {
                "backfilled_event_count": 0,
                "updated_event_ids": [],
                "needs_refine_event_ids": [],
                "needs_reconsolidation_windows": [],
                "updated_events": [],
            }
        archive_events = self.load_archive_events()
        active_ids = {str(event.get("event_id")) for event in self.load_events() if event.get("event_id")}
        updated_events: list[dict[str, Any]] = []
        needs_refine: list[str] = []
        recon_windows: list[dict[str, Any]] = []
        now = utc_now_iso()
        for event in archive_events:
            event_start = float(event.get("start_time", 0.0) or 0.0)
            event_end = float(event.get("end_time", event_start) or event_start)
            matched = [
                seg
                for seg in normalized_segments
                if max(event_start, float(seg.get("start", 0.0) or 0.0))
                <= min(event_end, float(seg.get("end", seg.get("start", 0.0)) or 0.0))
            ]
            if not matched:
                continue
            merged_segments = self._merge_transcript_segments(list(event.get("transcript_segments", []) or []), matched)
            updated = dict(event)
            updated["transcript_segments"] = merged_segments
            updated["transcript"] = " ".join(str(seg.get("text") or "").strip() for seg in merged_segments if seg.get("text")).strip()
            updated["transcript_version"] = int(updated.get("transcript_version", 0) or 0) + 1
            updated["transcript_updated_at"] = now
            updated["transcript_source"] = reason
            updated["version"] = int(updated.get("version", 1) or 1) + 1
            updated["updated_at"] = now
            status = str(updated.get("status") or "provisional")
            if status in {"provisional", "refine_failed", "refined", "final"}:
                updated["needs_refine"] = True
                if status in {"refined", "final"}:
                    updated["refined_stale"] = True
                    updated["stale_reason"] = "transcript_backfill"
                event_id = str(updated.get("event_id") or "")
                if event_id:
                    needs_refine.append(event_id)
            if bool(updated.get("merged_to_long_term")):
                updated["needs_reconsolidation"] = True
                updated["dirty_reason"] = "transcript_backfill"
                window = self._transcript_dirty_window_for_event(updated)
                updated["dirty_window_id"] = window["window_id"]
                updated["dirty_time_range"] = [window["start_time"], window["end_time"]]
                recon_windows.append(window)
            updated["retrieval_text"] = build_retrieval_text(updated)
            updated_events.append(updated)

        update_result = self.update_events(updated_events) if updated_events else {"active_updated": False, "archive_updated": False}
        return {
            "backfilled_event_count": len(updated_events),
            "updated_event_ids": [str(event.get("event_id")) for event in updated_events if event.get("event_id")],
            "active_updated_event_ids": [str(event.get("event_id")) for event in updated_events if str(event.get("event_id")) in active_ids],
            "needs_refine_event_ids": list(dict.fromkeys(needs_refine)),
            "needs_reconsolidation_windows": self._dedupe_windows(recon_windows),
            "updated_events": updated_events,
            "update_result": update_result,
        }

    def normalize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        now = utc_now_iso()
        start = float(item.get("start_time", 0.0) or 0.0)
        end = float(item.get("end_time", start) or start)
        if not item.get("event_id") or str(item.get("event_id", "")).startswith("mst_event_"):
            item["event_id"] = event_time_id(self.session_id, start, end, self._boundary_index_from_event(item))
        item.setdefault("session_id", self.session_id)
        item.setdefault("duration", max(0.0, round(end - start, 3)))
        item.setdefault("available_at", end)
        item.setdefault("status", "provisional")
        item.setdefault("version", 1)
        item.setdefault("event_caption_placeholder", item.get("event_caption_fast") or f"A provisional short-term video event occurs between {start:.1f}s and {end:.1f}s.")
        item.setdefault("event_caption_fast", "")
        item.setdefault("event_caption_refined", None)
        _caption, source = effective_caption(item)
        item["caption_source"] = item.get("caption_source") if item.get("event_caption_refined") is None else "refined"
        if item.get("event_caption_refined"):
            item["caption_source"] = "refined"
        elif not item.get("caption_source"):
            item["caption_source"] = source
        item.setdefault("transcript", "")
        item.setdefault("transcript_segments", [])
        item.setdefault("transcript_version", 0)
        item.setdefault("transcript_source", None)
        item.setdefault("transcript_updated_at", None)
        item.setdefault("keyframes", [])
        item.setdefault("entities", [])
        item.setdefault("visual_objects", [])
        item.setdefault("main_actions", [])
        item.setdefault("state_changes", [])
        item.setdefault("needs_refine", False)
        item.setdefault("refined_stale", False)
        item.setdefault("stale_reason", None)
        item.setdefault("refine_completed_at", None)
        item.setdefault("refine_speed", None)
        item.setdefault("needs_reconsolidation", False)
        item.setdefault("dirty_reason", None)
        item.setdefault("dirty_window_id", None)
        item.setdefault("dirty_time_range", None)
        item.setdefault("refine", {})
        refine = item["refine"] if isinstance(item.get("refine"), dict) else {}
        refine.setdefault("refine_attempts", 0)
        refine.setdefault("last_refine_at", None)
        refine.setdefault("last_refine_queued_at", None)
        refine.setdefault("last_refine_worker_started_at", None)
        refine.setdefault("last_refine_completed_at", None)
        refine.setdefault("last_refine_failed_at", None)
        refine.setdefault("last_refine_retry_queued_at", None)
        refine.setdefault("last_refine_error", None)
        refine.setdefault("backend", None)
        refine.setdefault("last_refine_task_id", None)
        refine.setdefault("last_refine_task_reason", None)
        refine.setdefault("refine_timeline", [])
        item["refine"] = refine
        item.setdefault("merged_to_long_term", False)
        item.setdefault("merged_episode_id", None)
        item.setdefault("merged_at", None)
        item.setdefault("confidence", 0.55)
        item.setdefault("created_at", now)
        item["updated_at"] = item.get("updated_at") or now
        item["retrieval_text"] = build_retrieval_text(item)
        return item

    def _boundary_index_from_event(self, event: dict[str, Any]) -> int:
        for key in ("boundary_index", "event_index"):
            try:
                return int(event.get(key))
            except Exception:
                pass
        return 0

    def _merge_by_event_id(self, existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {str(event.get("event_id")): event for event in existing if event.get("event_id")}
        for update in updates:
            by_id[str(update["event_id"])] = {**by_id.get(str(update["event_id"]), {}), **update}
        return sorted(by_id.values(), key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("event_id", ""))))

    def _merge_transcript_segments(self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(item: dict[str, Any]) -> str:
            explicit = str(item.get("segment_id") or "").strip()
            if explicit:
                return explicit
            start = int(round(float(item.get("start", 0.0) or 0.0) * 1000))
            end = int(round(float(item.get("end", item.get("start", 0.0)) or 0.0) * 1000))
            return f"{start:09d}_{end:09d}_{str(item.get('text') or '').strip()}"

        by_key = {key(item): dict(item) for item in existing if isinstance(item, dict)}
        for item in incoming:
            by_key[key(item)] = {**by_key.get(key(item), {}), **dict(item)}
        return sorted(by_key.values(), key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0), str(item.get("segment_id") or "")))

    def _transcript_dirty_window_for_event(self, event: dict[str, Any]) -> dict[str, Any]:
        start = float(event.get("start_time", 0.0) or 0.0)
        window_start = int(start // 30.0) * 30.0
        window_end = window_start + 30.0
        return {
            "window_id": f"win_{int(window_start):06d}_{int(window_end):06d}",
            "start_time": round(window_start, 3),
            "end_time": round(window_end, 3),
        }

    def _dedupe_windows(self, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {str(window.get("window_id")): dict(window) for window in windows if window.get("window_id")}
        return sorted(by_id.values(), key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("window_id", ""))))

    def evict_old_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        events = sorted(events, key=lambda item: float(item.get("end_time", item.get("start_time", 0.0))))
        latest = max(float(item.get("end_time", 0.0)) for item in events)
        window_start = latest - float(self.recent_window_seconds)
        kept = [event for event in events if float(event.get("end_time", 0.0)) >= window_start]
        if len(kept) > self.max_events:
            kept = kept[-self.max_events :]
        return kept

    def get_recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.load_events()[-limit:]

    def search_events(
        self,
        query: str,
        top_k: int = 5,
        cache_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from online_short_term.mst_retriever import MSTRetriever

        return MSTRetriever(self).search(query, top_k=top_k, cache_context=cache_context)

    def get_state(self) -> dict[str, Any]:
        state = read_json(self.state_path, default={})
        if not isinstance(state, dict):
            state = {}
        archive_state = self.get_archive_state()
        return {
            "session_id": self.session_id,
            "short_term_ready": bool(state.get("short_term_ready", False)),
            "mst_version": int(state.get("mst_version", 0) or 0),
            "archive_version": int(archive_state.get("archive_version", state.get("archive_version", 0)) or 0),
            "last_event_start": float(state.get("last_event_start", 0.0) or 0.0),
            "last_processed_time": float(state.get("last_processed_time", 0.0) or 0.0),
            "event_count": int(state.get("active_event_count", state.get("event_count", 0)) or 0),
            "active_event_count": int(state.get("active_event_count", state.get("event_count", 0)) or 0),
            "archive_event_count": int(archive_state.get("archive_event_count", state.get("archive_event_count", 0)) or 0),
            "recent_window_seconds": float(state.get("recent_window_seconds", self.recent_window_seconds)),
            "max_events": int(state.get("max_events", self.max_events)),
            "archive_max_events": int(state.get("archive_max_events", self.archive_max_events)),
            "active_path": state.get("active_path", rel_to_session(self.session_dir, self.events_path)),
            "archive_path": state.get("archive_path", rel_to_session(self.session_dir, self.archive_events_path)),
            "active_time_span": state.get("active_time_span", [0.0, 0.0]),
            "archive_time_span": archive_state.get("archive_time_span", state.get("archive_time_span", [0.0, 0.0])),
            "pending_archive_event_ids": state.get("pending_archive_event_ids", []),
            "archive_error": state.get("archive_error"),
            "updated_at": state.get("updated_at"),
        }

    def get_archive_state(self) -> dict[str, Any]:
        state = read_json(self.archive_state_path, default={})
        if not isinstance(state, dict):
            state = {}
        return {
            "session_id": self.session_id,
            "archive_version": int(state.get("archive_version", 0) or 0),
            "archive_event_count": int(state.get("archive_event_count", 0) or 0),
            "archive_time_span": state.get("archive_time_span", [0.0, 0.0]),
            "unrefined_event_count": int(state.get("unrefined_event_count", 0) or 0),
            "refined_event_count": int(state.get("refined_event_count", 0) or 0),
            "updated_at": state.get("updated_at"),
        }

    def clear(self, clear_archive: bool = False) -> dict[str, Any]:
        self.short_term_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.events_path, self.index_path, self.id_mapping_path):
            if path.exists():
                path.unlink()
        if clear_archive:
            for path in (self.archive_events_path, self.archive_state_path, self.refine_state_path, self.ready_windows_path):
                if path.exists():
                    path.unlink()
        state = {
            "session_id": self.session_id,
            "short_term_ready": False,
            "mst_version": 0,
            "archive_version": 0 if clear_archive else self.get_archive_state().get("archive_version", 0),
            "last_event_start": 0.0,
            "last_processed_time": 0.0,
            "event_count": 0,
            "active_event_count": 0,
            "archive_event_count": 0 if clear_archive else self.get_archive_state().get("archive_event_count", 0),
            "recent_window_seconds": self.recent_window_seconds,
            "max_events": self.max_events,
            "archive_max_events": self.archive_max_events,
            "active_path": rel_to_session(self.session_dir, self.events_path),
            "archive_path": rel_to_session(self.session_dir, self.archive_events_path),
            "active_time_span": [0.0, 0.0],
            "archive_time_span": [0.0, 0.0] if clear_archive else self.get_archive_state().get("archive_time_span", [0.0, 0.0]),
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.state_path, state)
        if clear_archive:
            self._write_archive_state([], bump_version=False)
        return state

    def _write_state(self, events: list[dict[str, Any]]) -> None:
        old_state = self.get_state()
        last_processed_time = max([float(event.get("end_time", 0.0)) for event in events] or [0.0])
        last_event_start = max([float(event.get("start_time", 0.0)) for event in events] or [0.0])
        active_time_span = self._time_span(events)
        archive_state = self.get_archive_state()
        state = {
            "session_id": self.session_id,
            "short_term_ready": bool(events),
            "mst_version": int(old_state.get("mst_version", 0)) + 1,
            "archive_version": int(archive_state.get("archive_version", 0) or 0),
            "last_event_start": round(last_event_start, 3),
            "last_processed_time": round(last_processed_time, 3),
            "event_count": len(events),
            "active_event_count": len(events),
            "archive_event_count": int(archive_state.get("archive_event_count", 0) or 0),
            "recent_window_seconds": self.recent_window_seconds,
            "max_events": self.max_events,
            "archive_max_events": self.archive_max_events,
            "active_path": rel_to_session(self.session_dir, self.events_path),
            "archive_path": rel_to_session(self.session_dir, self.archive_events_path),
            "active_time_span": active_time_span,
            "archive_time_span": archive_state.get("archive_time_span", [0.0, 0.0]),
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.state_path, state)

    def _write_archive_state(self, events: list[dict[str, Any]], bump_version: bool = True) -> None:
        old = self.get_archive_state()
        refined = sum(1 for event in events if event.get("status") in {"refined", "final"})
        unrefined = sum(1 for event in events if event.get("status") not in {"refined", "final"})
        state = {
            "session_id": self.session_id,
            "archive_version": int(old.get("archive_version", 0) or 0) + (1 if bump_version else 0),
            "archive_event_count": len(events),
            "archive_time_span": self._time_span(events),
            "unrefined_event_count": unrefined,
            "refined_event_count": refined,
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(self.archive_state_path, state)

    def _time_span(self, events: list[dict[str, Any]]) -> list[float]:
        if not events:
            return [0.0, 0.0]
        return [
            round(min(float(event.get("start_time", 0.0)) for event in events), 3),
            round(max(float(event.get("end_time", 0.0)) for event in events), 3),
        ]

    def _write_index(self, events: list[dict[str, Any]]) -> None:
        index = {
            "items": [
                {
                    "event_id": event.get("event_id"),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "text": event.get("retrieval_text", ""),
                }
                for event in events
            ],
            "updated_at": utc_now_iso(),
        }
        mapping = {
            str(i): event.get("event_id")
            for i, event in enumerate(events)
        }
        write_json_atomic(self.index_path, index)
        write_json_atomic(self.id_mapping_path, mapping)
