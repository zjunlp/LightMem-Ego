from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalizer import MemoryNormalizer, episodic_sort_key, semantic_sort_key, visual_sort_key
from .repository import LongTermMemoryRepository, MemoryRepositoryError
from .schemas import EPISODIC_GRANULARITIES, MEMORY_SOURCE, SCHEMA_VERSION
from .session_labels import session_labels
from online_memory_edit.schemas import edit_policy


class MemoryViewError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class MemoryViewResult:
    payload: dict[str, Any]
    response_bytes: int


class LongTermMemoryViewService:
    def __init__(self, sessions_root: Path, session_id: str) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = session_id

    def load(self) -> MemoryViewResult:
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                return self._load_once()
            except FileNotFoundError as exc:
                raise MemoryViewError(404, "session_not_found", "Session 不存在") from exc
            except MemoryRepositoryError as exc:
                last_error = exc
                message = str(exc)
                if message in {"memory path escapes session directory", "invalid visual evidence"}:
                    raise MemoryViewError(500, "memory_view_failed", "长期记忆读取失败") from exc
        message = str(last_error or "")
        if "lagging" in message:
            raise MemoryViewError(503, "memory_component_lagging", "长期记忆组件仍在更新，请稍后刷新") from last_error
        if "changing" in message:
            raise MemoryViewError(503, "memory_snapshot_changing", "长期记忆正在更新，请稍后刷新") from last_error
        if "not ready" in message:
            raise MemoryViewError(503, "memory_not_ready", "长期记忆尚未就绪") from last_error
        raise MemoryViewError(500, "memory_view_failed", "长期记忆读取失败") from last_error

    def _load_once(self) -> MemoryViewResult:
        repository = LongTermMemoryRepository(self.sessions_root, self.session_id)
        active = repository.resolve_active_root()
        config_before = active.config
        signature_before = repository.version_signature(config_before)
        repository.ensure_stable_components(signature_before)

        episodic_source = repository.load_episodic(config_before)
        semantic_source = repository.load_semantic(config_before)
        visual_source = repository.load_visual(config_before, episodic_source)

        config_after = repository.reload_config(active)
        signature_after = repository.version_signature(config_after)
        repository.ensure_stable_components(signature_after)
        if signature_before != signature_after:
            raise MemoryRepositoryError("memory snapshot is changing")

        normalizer = MemoryNormalizer(self.session_id)
        episodic: dict[str, list[dict[str, Any]]] = {}
        for granularity in EPISODIC_GRANULARITIES:
            records = [normalizer.episodic(item, granularity) for item in episodic_source.get(granularity, [])]
            if granularity == "30sec":
                id_counts: dict[str, int] = {}
                for record in records:
                    record_id = str(record.get("id") or "")
                    id_counts[record_id] = id_counts.get(record_id, 0) + 1
                for record in records:
                    if id_counts.get(str(record.get("id") or ""), 0) > 1:
                        record["editable"] = False
                        record["editDisabledReason"] = "duplicate_stable_id"
                    elif active.kind != "em2mem":
                        record["editable"] = False
                        record["editDisabledReason"] = "unsupported_source_record"
            records.sort(key=episodic_sort_key)
            episodic[granularity] = records
        semantic = [normalizer.semantic(item) for item in semantic_source]
        semantic.sort(key=semantic_sort_key)
        visual = _deduplicate_visual([normalizer.visual(item) for item in visual_source])
        visual.sort(key=visual_sort_key)

        episodic_count = sum(len(episodic[key]) for key in EPISODIC_GRANULARITIES)
        counts = {
            "total": episodic_count + len(semantic) + len(visual),
            "episodic": episodic_count,
            "semantic": len(semantic),
            "visual": len(visual),
            "episodic_by_granularity": {key: len(episodic[key]) for key in EPISODIC_GRANULARITIES},
        }
        memory_version = signature_before.get("memory_version")
        semantic_version = signature_before.get("semantic_version")
        visual_version = signature_before.get("visual_version")
        if semantic and semantic_version is None:
            semantic_version = memory_version
        if visual and visual_version is None:
            visual_version = memory_version
        can_edit = active.kind == "em2mem" and any(
            record.get("editable") for record in episodic["30sec"]
        )
        edit_disabled_reason = None if can_edit else (
            "unsupported_source_record" if active.kind != "em2mem" else "no_editable_30sec"
        )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "session_id": self.session_id,
            **session_labels(repository.session_dir),
            "memory_source": MEMORY_SOURCE,
            "memory_version": memory_version,
            "memoryVersion": memory_version,
            "component_versions": {
                "episodic": memory_version,
                "semantic": semantic_version,
                "visual": visual_version,
            },
            "availability": {
                "episodic": "ready" if episodic_count else "not_available",
                "semantic": "ready" if semantic else "not_available",
                "visual": "ready" if visual else "not_available",
            },
            "active_root_kind": active.kind,
            "qa_aligned": active.kind == "em2mem",
            "complete_record_set": True,
            "field_truncation": normalizer.truncated_field_count > 0,
            "truncated_field_count": normalizer.truncated_field_count,
            "updated_at": config_after.get("updated_at") or config_after.get("last_ready_at"),
            "counts": counts,
            "episodic": episodic,
            "semantic": semantic,
            "visual": visual,
            "editPolicy": edit_policy(),
            "canEdit": can_edit,
            "editDisabledReason": edit_disabled_reason,
            "canRollback": _can_rollback(config_after, memory_version),
            "propagation": (
                config_after.get("memory_edit_propagation")
                if isinstance(config_after.get("memory_edit_propagation"), dict)
                else None
            ),
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        max_bytes = _max_response_bytes()
        if len(encoded) > max_bytes:
            raise MemoryViewError(
                413,
                "memory_response_too_large",
                "该 Session 的长期记忆数据超过安全展示上限",
                {"max_response_bytes": max_bytes},
            )
        return MemoryViewResult(payload=payload, response_bytes=len(encoded))


def _max_response_bytes() -> int:
    raw = os.getenv("EM2MEM_MEMORY_VIEW_MAX_RESPONSE_BYTES", "33554432")
    try:
        value = int(raw)
    except ValueError:
        value = 33554432
    return max(1_048_576, value)


def _can_rollback(config: dict[str, Any], memory_version: Any) -> bool:
    rollback = config.get("memory_edit_rollback")
    if not isinstance(rollback, dict) or not rollback.get("available"):
        return False
    try:
        return int(rollback.get("edit_version")) == int(memory_version)
    except (TypeError, ValueError):
        return False


def _deduplicate_visual(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("source_id") or record.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result
