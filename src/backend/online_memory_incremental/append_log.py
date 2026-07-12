from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import ensure_dir, utc_now_iso, write_json_atomic


READY_STATUSES = {"appended_fast", "graph_semantic_pending", "fully_ready"}


@dataclass(frozen=True)
class AppendDecision:
    episode_id: str
    append_id: str
    should_append: bool
    reason: str
    latest: dict[str, Any] | None = None


class MemoryAppendLog:
    """Append-only status log for MST 30s episode -> Em2Mem memory updates."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
        return entries

    def latest_by_episode(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for entry in self.read_entries():
            episode_id = str(entry.get("episode_id") or "")
            if episode_id:
                latest[episode_id] = entry
        return latest

    def decide(self, episode: dict[str, Any], force: bool = False) -> AppendDecision:
        episode_id = str(episode.get("episode_id") or "")
        append_id = f"append_{episode_id}"
        if not episode_id:
            return AppendDecision(episode_id="", append_id=append_id, should_append=False, reason="missing_episode_id")
        latest = self.latest_by_episode().get(episode_id)
        if latest and latest.get("status") in READY_STATUSES and not force:
            return AppendDecision(
                episode_id=episode_id,
                append_id=str(latest.get("append_id") or append_id),
                should_append=False,
                reason="already_appended",
                latest=latest,
            )
        if latest and force:
            return AppendDecision(
                episode_id=episode_id,
                append_id=append_id,
                should_append=True,
                reason="force_rebuild",
                latest=latest,
            )
        return AppendDecision(episode_id=episode_id, append_id=append_id, should_append=True, reason="pending")

    def append_status(
        self,
        *,
        episode: dict[str, Any],
        status: str,
        fast_memory_version: int | None = None,
        semantic_memory_version: int | None = None,
        graph_version: int | None = None,
        visual_version: int | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        episode_id = str(episode.get("episode_id") or "")
        source = "mst_micro_events"
        if episode.get("episodic_source"):
            source = str(episode.get("episodic_source"))
        elif isinstance(episode.get("source"), dict) and episode.get("source", {}).get("type"):
            source = str(episode.get("source", {}).get("type"))
        payload: dict[str, Any] = {
            "append_id": f"append_{episode_id}",
            "session_id": episode.get("session_id"),
            "episode_id": episode_id,
            "segment_id": episode.get("segment_id"),
            "start_time": float(episode.get("start_time") or 0.0),
            "end_time": float(episode.get("end_time") or episode.get("start_time") or 0.0),
            "source": source,
            "source_micro_event_ids": episode.get("source_micro_event_ids") or [],
            "status": status,
            "fast_memory_version": fast_memory_version,
            "semantic_memory_version": semantic_memory_version,
            "graph_version": graph_version,
            "visual_version": visual_version,
            "created_at": episode.get("created_at") or utc_now_iso(),
            "updated_at": utc_now_iso(),
            "error": error,
        }
        if extra:
            payload.update(extra)
        ensure_dir(self.path.parent)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def write_state(self, path: Path, session_id: str) -> dict[str, Any]:
        entries = self.read_entries()
        latest = self.latest_by_episode()
        status_counts: dict[str, int] = {}
        for entry in latest.values():
            status = str(entry.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        state = {
            "session_id": session_id,
            "append_log_path": self.path.name,
            "append_entry_count": len(entries),
            "unique_episode_count": len(latest),
            "status_counts": status_counts,
            "pending_count": status_counts.get("pending", 0) + status_counts.get("appending", 0),
            "appended_count": sum(status_counts.get(x, 0) for x in READY_STATUSES),
            "failed_count": status_counts.get("failed", 0),
            "updated_at": utc_now_iso(),
        }
        write_json_atomic(path, state)
        return state
