from __future__ import annotations

import os
import shutil
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from online_memory.evidence_to_em2mem import (
    _load_triplet_map,
    _memory_generation_backend,
    write_caption_files,
    write_semantic_files,
    write_sidecar_files,
)
from online_memory.em2mem_layout import Em2MemOnlineLayout, ensure_em2mem_layout, hhmmssff_to_seconds
from online_memory_view.graph_service import memory_graph_etag
from online_memory_view.normalizer import (
    episodic_content_field,
    source_record_fingerprint,
    stable_episodic_id,
)
from online_memory_view.repository import LongTermMemoryRepository, MemoryRepositoryError
from online_memory_view.service import LongTermMemoryViewService
from online_preprocess.io_utils import read_json, relative_to_session, utc_now_iso, write_json_atomic

from .idempotency import IdempotencyError, IdempotencyStore
from .lock import session_memory_mutation_lock
from .schemas import MAX_CHARS_PER_RECORD, MAX_TOTAL_CHARS


class MemoryEditError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        return {
            "status": "error",
            "error": {"code": self.code, "message": self.message, "details": self.details},
        }


@dataclass(frozen=True)
class MemoryEditResult:
    status_code: int
    payload: dict[str, Any]
    replayed: bool = False


@dataclass
class _Candidate:
    root: Path
    config: dict[str, Any]
    config_path: Path
    version: int
    raw_30sec: list[dict[str, Any]]
    propagation: dict[str, Any]
    runtime_engine: Any = None


class MemoryEditService:
    def __init__(
        self,
        sessions_root: Path,
        session_id: str,
        *,
        project_root: Path | None = None,
        runtime_validator: Callable[[Path, dict[str, Any]], Any] | None = None,
        derivation_backend_override: str | None = None,
    ) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = session_id
        self.session_dir = self.sessions_root / session_id
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.runtime_validator = runtime_validator or self._validate_runtime
        self.derivation_backend_override = derivation_backend_override
        self.idempotency = IdempotencyStore(self.session_dir)

    def update(self, body: Any, idempotency_key: str) -> MemoryEditResult:
        return self._execute_idempotent("update_30sec", body, idempotency_key, self._update_locked)

    def rollback(self, body: Any, idempotency_key: str) -> MemoryEditResult:
        return self._execute_idempotent("rollback_30sec", body, idempotency_key, self._rollback_locked)

    def _execute_idempotent(
        self,
        operation: str,
        body: Any,
        key: str,
        action: Callable[[Any], dict[str, Any]],
    ) -> MemoryEditResult:
        if not self.session_dir.is_dir():
            raise MemoryEditError(404, "session_not_found", "Session 不存在")
        with session_memory_mutation_lock(self.session_dir):
            try:
                stored = self.idempotency.begin(key, operation, body)
            except IdempotencyError as exc:
                raise MemoryEditError(exc.status_code, exc.code, exc.message) from exc
            if stored is not None:
                return MemoryEditResult(stored.status_code, stored.payload, replayed=True)
            try:
                payload = action(body)
            except MemoryEditError as exc:
                try:
                    self.idempotency.finish(
                        key,
                        exc.status_code,
                        exc.payload(),
                        retryable=exc.code in {
                            "semantic_rebuild_unavailable",
                            "memory_derivation_failed",
                        },
                    )
                except Exception:
                    pass
                raise
            except Exception as exc:
                error = MemoryEditError(500, "memory_update_failed", "长期记忆更新失败")
                try:
                    self.idempotency.finish(key, error.status_code, error.payload())
                except Exception:
                    pass
                raise error from exc
            try:
                self.idempotency.finish(key, 200, payload)
            except Exception:
                # The memory publication is already committed. Do not report a
                # failed write after the authoritative version was switched.
                pass
            return MemoryEditResult(200, payload)

    def _update_locked(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != {"baseVersion", "records"}:
            raise _invalid_record(None, "request", "请求只能包含 baseVersion 和 records")
        active, config, current_version, raw_30sec = self._active_state()
        base_version = _positive_int(body.get("baseVersion"))
        if base_version is None:
            raise _invalid_record(None, "baseVersion", "基础版本号必须是正整数")
        if base_version != current_version:
            raise MemoryEditError(
                409,
                "memory_version_conflict",
                "长期记忆已被其他请求更新，请重新加载。",
                {"currentVersion": current_version},
            )
        requested = body.get("records")
        if not isinstance(requested, list) or not requested:
            raise _invalid_record(None, "records", "至少需要一条变化记录")

        index = self._editable_index(raw_30sec)
        seen: set[str] = set()
        changes: dict[str, str] = {}
        total_chars = 0
        for value in requested:
            if not isinstance(value, dict) or set(value) != {"id", "content"}:
                raise _invalid_record(None, "records", "每条记录只能包含 id 和 content")
            record_id = value.get("id")
            content = value.get("content")
            if not isinstance(record_id, str) or not record_id:
                raise _invalid_record(None, "id", "记录 ID 无效")
            if record_id in seen:
                raise _invalid_record(record_id, "id", "记录 ID 重复")
            seen.add(record_id)
            entry = index.get(record_id)
            if entry is None:
                raise _invalid_record(record_id, "id", "记录不存在或不可编辑")
            if not isinstance(content, str) or not content.strip():
                raise _invalid_record(record_id, "content", "记忆内容不能为空")
            length = len(content)
            if length > MAX_CHARS_PER_RECORD:
                raise MemoryEditError(
                    422,
                    "memory_too_long",
                    "单条记忆内容超过长度限制",
                    {"recordId": record_id, "field": "content", "limit": MAX_CHARS_PER_RECORD},
                )
            total_chars += length
            if total_chars > MAX_TOTAL_CHARS:
                raise MemoryEditError(
                    422,
                    "memory_too_long",
                    "本次修改内容超过总长度限制",
                    {"recordId": record_id, "field": "records", "limit": MAX_TOTAL_CHARS},
                )
            source = raw_30sec[entry["recordIndex"]]
            if source_record_fingerprint(source) != entry["sourceFingerprint"]:
                raise MemoryEditError(
                    409,
                    "memory_version_conflict",
                    "长期记忆已被其他请求更新，请重新加载。",
                    {"currentVersion": current_version},
                )
            if content != source.get(entry["contentField"]):
                changes[record_id] = content
        if not changes:
            raise _invalid_record(None, "records", "没有实际变化的记录")

        updated = [dict(record) for record in raw_30sec]
        changed_source_ids: set[str] = set()
        for record_id, content in changes.items():
            entry = index[record_id]
            updated[entry["recordIndex"]][entry["contentField"]] = content
            changed_source_ids.add(entry["sourceId"])

        target_version = current_version + 1
        candidate = self._build_candidate(
            config,
            updated,
            target_version,
            rollback_available=True,
            changed_source_ids=changed_source_ids,
        )
        backup: Path | None = None
        try:
            backup = self._stage_rollback_backup(config, current_version, target_version, raw_30sec)
            candidate.config["memory_edit_rollback"]["backup_path"] = relative_to_session(backup, self.session_dir)
            write_json_atomic(candidate.config_path, candidate.config)
            self._publish(candidate, config)
            payload = self._success_payload(candidate, "当前记忆已修改，请刷新页面。", can_rollback=True)
            self._cleanup_superseded_backup(config, keep=backup)
        except Exception:
            self._restore_previous_config(config, candidate)
            self._discard_candidate(candidate)
            if backup is not None:
                self._discard_rollback_backup(backup)
            raise
        return payload

    def _rollback_locked(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != {"currentVersion"}:
            raise _invalid_record(None, "request", "请求只能包含 currentVersion")
        _active, config, current_version, _raw = self._active_state()
        requested_version = _positive_int(body.get("currentVersion"))
        if requested_version is None:
            raise _invalid_record(None, "currentVersion", "当前版本号必须是正整数")
        if requested_version != current_version:
            raise MemoryEditError(
                409,
                "memory_version_conflict",
                "长期记忆已被其他请求更新，请重新加载。",
                {"currentVersion": current_version},
            )
        rollback_state = config.get("memory_edit_rollback")
        backup_path = self._configured_backup_path(rollback_state)
        backup = read_json(backup_path, default={}) if backup_path else {}
        if (
            not isinstance(rollback_state, dict)
            or not rollback_state.get("available")
            or int(rollback_state.get("edit_version") or 0) != current_version
            or not isinstance(backup, dict)
            or backup.get("consumed")
            or int(backup.get("editVersion") or 0) != current_version
            or (
                self._backup_snapshot_path(backup) is None
                and not isinstance(backup.get("records"), list)
            )
        ):
            raise MemoryEditError(409, "rollback_unavailable", "没有可撤回的长期记忆修改")

        target_version = current_version + 1
        candidate = self._build_rollback_candidate(config, backup, target_version)
        try:
            self._publish(candidate, config)
            payload = self._success_payload(candidate, "已撤回上一次记忆修改，请刷新页面。", can_rollback=False)
            backup["consumed"] = True
            backup["consumedAt"] = utc_now_iso()
            backup["rollbackVersion"] = target_version
            if backup_path is not None:
                write_json_atomic(backup_path, backup)
        except Exception:
            self._restore_previous_config(config, candidate)
            self._discard_candidate(candidate)
            raise
        return payload

    def _active_state(self) -> tuple[Any, dict[str, Any], int, list[dict[str, Any]]]:
        repository = LongTermMemoryRepository(self.sessions_root, self.session_id)
        try:
            active = repository.resolve_active_root()
        except FileNotFoundError as exc:
            raise MemoryEditError(404, "session_not_found", "Session 不存在") from exc
        except MemoryRepositoryError as exc:
            raise MemoryEditError(503, "memory_not_ready", "长期记忆尚未就绪") from exc
        if active.kind != "em2mem":
            raise MemoryEditError(422, "invalid_thirty_second_memory", "该 Session 的记忆格式不支持编辑")
        signature = repository.version_signature(active.config)
        try:
            repository.ensure_stable_components(signature)
        except MemoryRepositoryError as exc:
            raise MemoryEditError(503, "memory_component_lagging", "长期记忆组件仍在更新，请稍后重试") from exc
        version = signature.get("memory_version")
        if not isinstance(version, int) or version <= 0:
            raise MemoryEditError(503, "memory_not_ready", "长期记忆尚未就绪")
        raw = repository.load_episodic(active.config).get("30sec", [])
        if not raw:
            raise MemoryEditError(503, "memory_not_ready", "30 秒长期记忆尚未生成")
        return active, dict(active.config), version, [dict(item) for item in raw]

    def _editable_index(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        candidates: list[tuple[str, int, dict[str, Any], str]] = []
        counts: dict[str, int] = {}
        for index, record in enumerate(records):
            record_id = stable_episodic_id(self.session_id, "30sec", record)
            counts[record_id] = counts.get(record_id, 0) + 1
            content_field = episodic_content_field(record)
            if content_field:
                candidates.append((record_id, index, record, content_field))
        return {
            record_id: {
                "sourceId": str(record.get("doc_id") or record.get("evidence_doc_id") or record.get("segment_id")),
                "granularity": "30sec",
                "recordIndex": index,
                "contentField": content_field,
                "sourceFingerprint": source_record_fingerprint(record),
            }
            for record_id, index, record, content_field in candidates
            if counts.get(record_id) == 1
            and str(record.get("doc_id") or record.get("evidence_doc_id") or record.get("segment_id") or "").strip()
        }

    def _build_candidate(
        self,
        previous_config: dict[str, Any],
        records: list[dict[str, Any]],
        version: int,
        *,
        rollback_available: bool,
        changed_source_ids: set[str] | None = None,
    ) -> _Candidate:
        temp_root, target_root = self._snapshot_paths(version)
        layout = Em2MemOnlineLayout(temp_root, self.session_id)
        ensure_em2mem_layout(layout)
        self._copy_configured_components(previous_config, layout, include_derived=False)

        prepared = [_prepare_caption_record(item, self.session_id) for item in records]
        model = str((previous_config.get("query_rag_args") or {}).get("retriever_model") or os.getenv("EM2MEM_MEMORY_MODEL") or "gpt-5.4")
        multiscale_backend = self._multiscale_backend(previous_config)
        derivation_backend = self._derivation_backend()
        self._derive_component(
            "multiscale",
            lambda: write_caption_files(
                layout,
                prepared,
                model_name=model,
                generation_backend=multiscale_backend,
            ),
            cleanup_root=temp_root,
        )
        caption_by_scale = {
            "30sec": prepared,
            "3min": _json_records(layout.caption_3min_path),
            "10min": _json_records(layout.caption_10min_path),
            "1h": _json_records(layout.caption_1h_path),
        }
        sidecars = self._derive_component(
            "episodic_graph",
            lambda: write_sidecar_files(
                layout,
                model,
                caption_by_scale,
                generation_backend=derivation_backend,
            ),
            semantic_backend=derivation_backend,
            cleanup_root=temp_root,
        )
        semantic_result = self._derive_component(
            "semantic",
            lambda: write_semantic_files(
                layout,
                model,
                prepared,
                generation_backend=derivation_backend,
                triplet_map=_load_triplet_map(sidecars["30sec"]["triplets"]),
            ),
            semantic_backend=derivation_backend,
            cleanup_root=temp_root,
        )
        write_json_atomic(layout.root / "editable_records.json", self._editable_index(prepared))

        updated_at = utc_now_iso()
        backends = {
            "multiscale": multiscale_backend,
            "triplets": "llm_openie" if derivation_backend == "llm" else "rule",
            "semantic": "llm_semantic_extraction_consolidation" if derivation_backend == "llm" else "rule",
        }
        previous_semantic = self._derive_component(
            "semantic_diff",
            lambda: LongTermMemoryRepository(self.sessions_root, self.session_id).load_semantic(previous_config),
            cleanup_root=temp_root,
        )
        semantic_memory_path = semantic_result[1]
        semantic_facts = _semantic_facts(semantic_memory_path)
        propagation = self._propagation_payload(
            version,
            caption_by_scale,
            backends,
            changed_source_ids=changed_source_ids or set(),
            changed_facts=_changed_items(previous_semantic, semantic_facts, "fact_id"),
        )
        config = self._snapshot_config(
            previous_config,
            layout,
            target_root,
            version,
            model=model,
            rollback_available=rollback_available,
            mode="manual_30sec_edit" if rollback_available else "manual_30sec_rollback",
            backends=backends,
            propagation=propagation,
            updated_at=updated_at,
        )
        return self._finalize_candidate(
            temp_root,
            target_root,
            layout,
            config,
            version,
            prepared,
            propagation,
        )

    def _build_rollback_candidate(
        self,
        current_config: dict[str, Any],
        backup: dict[str, Any],
        version: int,
    ) -> _Candidate:
        snapshot_path = self._backup_snapshot_path(backup)
        if snapshot_path is None:
            records = backup.get("records")
            if isinstance(records, list):
                return self._build_candidate(
                    current_config,
                    [dict(item) for item in records if isinstance(item, dict)],
                    version,
                    rollback_available=False,
                )
            raise MemoryEditError(409, "rollback_unavailable", "没有可撤回的长期记忆修改")

        restored_config = backup.get("config")
        if not isinstance(restored_config, dict):
            raise MemoryEditError(409, "rollback_unavailable", "没有可撤回的长期记忆修改")
        temp_root, target_root = self._snapshot_paths(version)
        layout = Em2MemOnlineLayout(temp_root, self.session_id)
        try:
            shutil.copytree(snapshot_path, layout.root)
        except Exception as exc:
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "长期记忆完整快照恢复失败",
                {"component": "rollback_snapshot"},
            ) from exc
        prepared = _json_records(layout.caption_30sec_path)
        caption_by_scale = {
            "30sec": prepared,
            "3min": _json_records(layout.caption_3min_path),
            "10min": _json_records(layout.caption_10min_path),
            "1h": _json_records(layout.caption_1h_path),
        }
        model = str((restored_config.get("query_rag_args") or {}).get("retriever_model") or os.getenv("EM2MEM_MEMORY_MODEL") or "gpt-5.4")
        backends = self._reported_backends(restored_config)
        current_semantic = LongTermMemoryRepository(self.sessions_root, self.session_id).load_semantic(current_config)
        restored_semantic = _semantic_facts_from_root(layout.semantic_root, model)
        propagation = self._propagation_payload(
            version,
            caption_by_scale,
            backends,
            changed_source_ids=set(),
            changed_facts=_changed_items(current_semantic, restored_semantic, "fact_id"),
            restored=True,
        )
        config = self._snapshot_config(
            restored_config,
            layout,
            target_root,
            version,
            model=model,
            rollback_available=False,
            mode="manual_30sec_rollback",
            backends=backends,
            propagation=propagation,
            updated_at=utc_now_iso(),
        )
        return self._finalize_candidate(
            temp_root,
            target_root,
            layout,
            config,
            version,
            prepared,
            propagation,
        )

    def _snapshot_paths(self, version: int) -> tuple[Path, Path]:
        snapshots_root = self.session_dir / "em2mem" / "edit_snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        return (
            snapshots_root / f".v{version:06d}_{token}.tmp",
            snapshots_root / f"v{version:06d}_{token}",
        )

    def _copy_configured_components(
        self,
        config: dict[str, Any],
        layout: Em2MemOnlineLayout,
        *,
        include_derived: bool,
    ) -> None:
        repository = LongTermMemoryRepository(self.sessions_root, self.session_id)
        repository.resolve_active_root()
        query_args = config.get("query_rag_args") if isinstance(config.get("query_rag_args"), dict) else {}
        mappings: list[tuple[Any, Path]] = []
        if include_derived:
            mappings.extend(
                (
                    (query_args.get("episodic_caption_root") or config.get("caption_root"), layout.caption_root),
                    (query_args.get("episodic_sidecar_root") or config.get("sidecar_root"), layout.sidecar_root),
                    (query_args.get("semantic_root") or config.get("semantic_root"), layout.semantic_root),
                    (config.get("memory_edit_index_path"), layout.root / "editable_records.json"),
                )
            )
        mappings.extend((
            (config.get("visual_root"), layout.visual_root),
            (query_args.get("visual_root"), layout.embeddings_root),
            (config.get("visual_embedding_path"), layout.visual_embedding_path),
            (config.get("visual_items_path"), layout.root / "visual" / "visual_items.jsonl"),
        ))
        for configured, destination in mappings:
            if not configured:
                continue
            source = repository._config_path(configured)
            if source is None or not source.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    def _multiscale_backend(self, config: dict[str, Any]) -> str:
        for value in (
            config.get("multiscale_generation_backend"),
            config.get("memory_generation_backend"),
            config.get("generation_backend"),
        ):
            normalized = _normalized_generation_backend(value)
            if normalized:
                return normalized
        snapshot_backend = _normalized_generation_backend(
            self._snapshot_backends(config).get("multiscale")
        )
        if snapshot_backend:
            return snapshot_backend
        return _memory_generation_backend(None)

    def _derivation_backend(self) -> str:
        if self.derivation_backend_override is None:
            return "llm"
        backend = _normalized_generation_backend(self.derivation_backend_override)
        if backend is None:
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "长期记忆派生后端配置无效",
                {"component": "configuration"},
            )
        return backend

    def _derive_component(
        self,
        component: str,
        action: Callable[[], Any],
        *,
        semantic_backend: str | None = None,
        cleanup_root: Path | None = None,
    ) -> Any:
        try:
            return action()
        except MemoryEditError:
            raise
        except Exception as exc:
            if cleanup_root is not None:
                shutil.rmtree(cleanup_root, ignore_errors=True)
            if semantic_backend == "llm":
                raise MemoryEditError(
                    503,
                    "semantic_rebuild_unavailable",
                    "Semantic 记忆重建服务暂不可用，请稍后重试",
                    {"component": component},
                ) from exc
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "长期记忆派生组件构建失败",
                {"component": component},
            ) from exc

    def _reported_backends(self, config: dict[str, Any]) -> dict[str, str]:
        multiscale = self._multiscale_backend(config)
        snapshot_backends = self._snapshot_backends(config)
        triplets = str(
            config.get("episodic_triplet_generation_backend")
            or snapshot_backends.get("triplets")
            or ""
        ).strip()
        semantic = str(
            config.get("semantic_generation_backend")
            or snapshot_backends.get("semantic")
            or ""
        ).strip()
        derivation = self._derivation_backend()
        return {
            "multiscale": multiscale,
            "triplets": triplets or ("llm_openie" if derivation == "llm" else "rule"),
            "semantic": semantic or (
                "llm_semantic_extraction_consolidation" if derivation == "llm" else "rule"
            ),
        }

    def _snapshot_backends(self, config: dict[str, Any]) -> dict[str, Any]:
        value = config.get("latest_snapshot_path")
        if value is None or not str(value).strip():
            return {}
        path = Path(str(value))
        if path.is_absolute() or any(part == ".." for part in path.parts):
            return {}
        metadata_path = (self.session_dir / path / "snapshot_meta.json").resolve()
        try:
            metadata_path.relative_to(self.session_dir.resolve())
        except ValueError:
            return {}
        metadata = read_json(metadata_path, default={})
        if not isinstance(metadata, dict):
            return {}
        propagation = metadata.get("propagation")
        if not isinstance(propagation, dict):
            return {}
        backends = propagation.get("backends")
        return backends if isinstance(backends, dict) else {}

    def _propagation_payload(
        self,
        version: int,
        caption_by_scale: dict[str, list[dict[str, Any]]],
        backends: dict[str, str],
        *,
        changed_source_ids: set[str],
        changed_facts: int,
        restored: bool = False,
    ) -> dict[str, Any]:
        derived: dict[str, dict[str, Any]] = {}
        for scale in ("3min", "10min", "1h"):
            records = caption_by_scale.get(scale, [])
            affected = sum(
                1
                for record in records
                if changed_source_ids.intersection(_record_source_ids(record))
            )
            state: dict[str, Any] = {
                "status": "ready",
                "recomputedRecords": 0 if restored else len(records),
                "affectedRecords": affected,
            }
            if restored:
                state["restoredRecords"] = len(records)
            derived[scale] = state
        return {
            "status": "ready",
            "sourceScale": "30sec",
            "targetMemoryVersion": version,
            "operation": "rollback" if restored else "update",
            "derivedScales": derived,
            "semantic": {
                "status": "ready",
                "version": version,
                "changedFacts": max(0, int(changed_facts)),
            },
            "graphs": {
                "episodic": {"status": "ready", "version": version},
                "semantic": {"status": "ready", "version": version},
            },
            "backends": dict(backends),
        }

    def _snapshot_config(
        self,
        base_config: dict[str, Any],
        layout: Em2MemOnlineLayout,
        target_root: Path,
        version: int,
        *,
        model: str,
        rollback_available: bool,
        mode: str,
        backends: dict[str, str],
        propagation: dict[str, Any],
        updated_at: str,
    ) -> dict[str, Any]:
        relative_root = relative_to_session(target_root / "em2mem", self.session_dir)
        config = dict(base_config)
        query_args = dict(config.get("query_rag_args") or {})
        config.update(
            {
                "status": "memory_ready",
                "memory_version": version,
                "memoryVersion": version,
                "latest_ready_memory_version": version,
                "latest_fast_ready_version": version,
                "latest_graph_ready_version": version,
                "graph_version": version,
                "latest_semantic_ready_version": version,
                "semantic_version": version,
                "latest_visual_ready_version": version,
                "visual_version": version,
                "building_memory_version": None,
                "building_versions": {"fast": None, "visual": None, "graph": None, "semantic": None},
                "readiness": {
                    **(config.get("readiness") if isinstance(config.get("readiness"), dict) else {}),
                    "episodic_ready": True,
                    "graph_ready": True,
                    "semantic_ready": True,
                    "long_term_partial_ready": True,
                    "long_term_full_ready": True,
                },
                "lag": {
                    "semantic_lagging": False,
                    "graph_lagging": False,
                    "visual_lagging": False,
                    "semantic_lag_versions": 0,
                    "graph_lag_versions": 0,
                    "visual_lag_versions": 0,
                },
                "memory_build_state": "ready",
                "long_term_partial_ready": True,
                "long_term_full_ready": True,
                "semantic_memory_ready": True,
                "caption_root": f"{relative_root}/caption_root",
                "sidecar_root": f"{relative_root}/sidecar_root",
                "semantic_root": f"{relative_root}/semantic_root",
                "visual_root": f"{relative_root}/visual_root",
                "latest_snapshot_version": version,
                "latest_snapshot_path": relative_to_session(target_root, self.session_dir),
                "memory_edit_index_path": f"{relative_root}/editable_records.json",
                "memory_edit_rollback": {
                    "available": rollback_available,
                    "edit_version": version if rollback_available else None,
                },
                "memory_generation_backend": backends["multiscale"],
                "multiscale_generation_backend": backends["multiscale"],
                "episodic_triplet_generation_backend": backends["triplets"],
                "semantic_generation_backend": backends["semantic"],
                "memory_edit_propagation": propagation,
                "em2mem_update_mode": mode,
                "last_ready_at": updated_at,
                "updated_at": updated_at,
            }
        )
        visual_items = layout.root / "visual" / "visual_items.jsonl"
        if visual_items.exists():
            config["visual_items_path"] = f"{relative_root}/visual/visual_items.jsonl"
        else:
            config.pop("visual_items_path", None)
        if layout.visual_embedding_path.exists():
            config["visual_embedding_path"] = f"{relative_root}/embeddings/visual_embeddings.pkl"
        else:
            config.pop("visual_embedding_path", None)
        query_args.update(
            {
                "subject": self.session_id,
                "retriever_model": model,
                "episodic_caption_root": f"{relative_root}/caption_root",
                "episodic_sidecar_root": f"{relative_root}/sidecar_root",
                "semantic_root": f"{relative_root}/semantic_root",
                "visual_root": f"{relative_root}/embeddings",
                "visual_evidence_file": f"{relative_root}/visual_root/session_visual_evidence.json",
            }
        )
        config["query_rag_args"] = query_args
        return config

    def _finalize_candidate(
        self,
        temp_root: Path,
        target_root: Path,
        layout: Em2MemOnlineLayout,
        config: dict[str, Any],
        version: int,
        records: list[dict[str, Any]],
        propagation: dict[str, Any],
    ) -> _Candidate:
        try:
            write_json_atomic(layout.memory_config_path, config)
            write_json_atomic(
                temp_root / "snapshot_meta.json",
                {
                    "session_id": self.session_id,
                    "snapshot_version": version,
                    "components": {"episodic": version, "graph": version, "semantic": version, "visual": version},
                    "propagation": propagation,
                    "created_at": config.get("updated_at"),
                    "source": config["em2mem_update_mode"],
                },
            )
            temp_root.replace(target_root)
            final_config_path = target_root / "em2mem" / "memory_config.json"
            write_json_atomic(final_config_path, config)
        except Exception as exc:
            shutil.rmtree(temp_root, ignore_errors=True)
            shutil.rmtree(target_root, ignore_errors=True)
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "候选快照写入失败",
                {"component": "snapshot_files"},
            ) from exc
        candidate = _Candidate(
            target_root,
            config,
            final_config_path,
            version,
            records,
            propagation,
        )
        try:
            self._validate_candidate(candidate)
        except Exception:
            self._discard_candidate(candidate)
            raise
        return candidate

    def _backup_snapshot_path(self, backup: dict[str, Any]) -> Path | None:
        value = backup.get("snapshotPath")
        if value is None or not str(value).strip():
            return None
        candidate = (self.session_dir / str(value)).resolve()
        try:
            candidate.relative_to((self.session_dir / "memory" / "rollback").resolve())
        except ValueError:
            return None
        required = (
            candidate / "caption_root" / f"{self.session_id}_30sec.json",
            candidate / "caption_root" / f"{self.session_id}_3min.json",
            candidate / "caption_root" / f"{self.session_id}_10min.json",
            candidate / "caption_root" / f"{self.session_id}_1h.json",
            candidate / "sidecar_root" / "30s",
            candidate / "semantic_root",
        )
        return candidate if candidate.is_dir() and all(path.exists() for path in required) else None

    def _validate_candidate(self, candidate: _Candidate) -> None:
        config = candidate.config
        version_fields = (
            "memory_version",
            "latest_ready_memory_version",
            "latest_graph_ready_version",
            "latest_semantic_ready_version",
        )
        if any(_positive_int(config.get(key)) != candidate.version for key in version_fields):
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "候选记忆版本校验失败",
                {"component": "version_manifest"},
            )
        root = candidate.root / "em2mem"
        required = (
            root / "caption_root" / f"{self.session_id}_30sec.json",
            root / "caption_root" / f"{self.session_id}_3min.json",
            root / "caption_root" / f"{self.session_id}_10min.json",
            root / "caption_root" / f"{self.session_id}_1h.json",
            root / "sidecar_root" / "30s",
            root / "semantic_root",
            root / "editable_records.json",
        )
        if (
            not all(path.exists() for path in required)
            or not list((root / "sidecar_root" / "30s").glob("episodic_graph_30s_*.json"))
            or not list((root / "semantic_root").glob("semantic_memory_*.json"))
        ):
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "候选记忆文件校验失败",
                {"component": "snapshot_files"},
            )
        try:
            candidate.runtime_engine = self.runtime_validator(candidate.config_path, config)
        except Exception as exc:
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "候选查询运行时加载失败",
                {"component": "query_runtime"},
            ) from exc

    def _validate_runtime(self, config_path: Path, config: dict[str, Any]) -> Any:
        from online_query.query_engine import load_query_engine

        engine = load_query_engine(
            self.session_id,
            sessions_root=self.sessions_root,
            memory_config_override=(config_path, config),
        )
        return engine

    def _publish(self, candidate: _Candidate, previous_config: dict[str, Any]) -> None:
        active_config_path = self.session_dir / "em2mem" / "memory_config.json"
        active_pointer = self.session_dir / "memory" / "active.json"
        from online_query import GLOBAL_SESSION_ENGINE_CACHE
        try:
            write_json_atomic(active_config_path, candidate.config)
            # Keep an explicit active pointer for readers that support
            # immutable snapshot publication.  Legacy readers continue to use
            # memory_config.json, which is updated in the same critical section.
            write_json_atomic(
                active_pointer,
                {
                    "sessionId": self.session_id,
                    "version": candidate.version,
                    "snapshotPath": candidate.config.get("latest_snapshot_path"),
                    "updatedAt": candidate.config.get("updated_at"),
                },
            )
            if candidate.runtime_engine is not None:
                # Test/dry-run validators may return a lightweight sentinel;
                # only install fully fledged query engines in the shared cache.
                if hasattr(candidate.runtime_engine, "memory_config_path"):
                    candidate.runtime_engine.memory_config_path = active_config_path
                    candidate.runtime_engine.memory_config_mtime = active_config_path.stat().st_mtime
                    GLOBAL_SESSION_ENGINE_CACHE.install(candidate.runtime_engine)
                else:
                    GLOBAL_SESSION_ENGINE_CACHE.invalidate(self.session_id)
            else:
                GLOBAL_SESSION_ENGINE_CACHE.invalidate(self.session_id)
        except Exception as exc:
            try:
                write_json_atomic(active_config_path, previous_config)
            except Exception:
                pass
            if candidate.runtime_engine is not None:
                try:
                    candidate.runtime_engine.close()
                except Exception:
                    pass
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "查询运行时切换失败",
                {"component": "query_runtime"},
            ) from exc

    def _restore_previous_config(self, previous_config: dict[str, Any], candidate: _Candidate) -> None:
        active_config_path = self.session_dir / "em2mem" / "memory_config.json"
        current = read_json(active_config_path, default={})
        restored = False
        if isinstance(current, dict) and _positive_int(current.get("memory_version")) == candidate.version:
            write_json_atomic(active_config_path, previous_config)
            restored = True
        if not restored:
            return
        try:
            pointer = self.session_dir / "memory" / "active.json"
            write_json_atomic(pointer, {"sessionId": self.session_id, "version": previous_config.get("memory_version"), "snapshotPath": previous_config.get("latest_snapshot_path"), "updatedAt": previous_config.get("updated_at")})
            from online_query import GLOBAL_SESSION_ENGINE_CACHE

            GLOBAL_SESSION_ENGINE_CACHE.invalidate(self.session_id)
        except Exception:
            pass

    def _success_payload(self, candidate: _Candidate, message: str, *, can_rollback: bool) -> dict[str, Any]:
        memory = LongTermMemoryViewService(self.sessions_root, self.session_id).load().payload
        component_versions = memory.get("component_versions")
        if (
            memory.get("memoryVersion") != candidate.version
            or not isinstance(component_versions, dict)
            or component_versions.get("episodic") != candidate.version
            or component_versions.get("semantic") != candidate.version
        ):
            raise MemoryEditError(
                500,
                "memory_derivation_failed",
                "发布后的长期记忆版本不一致",
                {"component": "response_consistency"},
            )
        signature = {
            "memory_version": candidate.version,
            "graph_version": candidate.version,
            "semantic_version": candidate.version,
            "updated_at": candidate.config.get("updated_at"),
        }
        etag = memory_graph_etag(self.session_id, "30sec", signature)
        return {
            "status": "ok",
            "success": True,
            "message": message,
            "sessionId": self.session_id,
            "memoryVersion": candidate.version,
            "memory": memory,
            "graphUpdate": {
                "status": "ready",
                "scale": "30sec",
                "memoryVersion": candidate.version,
                "componentVersions": {
                    "episodic": candidate.version,
                    "graph": candidate.version,
                    "semantic": candidate.version,
                },
                "graphVersions": {
                    "episodicGraph": candidate.version,
                    "semanticGraph": candidate.version,
                },
                "etag": etag,
            },
            "propagation": candidate.propagation,
            "canRollback": can_rollback,
        }

    def _stage_rollback_backup(
        self,
        config: dict[str, Any],
        source_version: int,
        edit_version: int,
        records: list[dict[str, Any]],
    ) -> Path:
        root = self.session_dir / "memory" / "rollback"
        root.mkdir(parents=True, exist_ok=True)
        backup_root = root / f"backup_v{edit_version:06d}_{uuid4().hex}"
        layout = Em2MemOnlineLayout(backup_root / "snapshot", self.session_id)
        metadata_path = backup_root / "backup.json"
        try:
            ensure_em2mem_layout(layout)
            self._copy_configured_components(config, layout, include_derived=True)
            if not layout.caption_30sec_path.exists():
                write_json_atomic(layout.caption_30sec_path, records)
            if not (layout.root / "editable_records.json").exists():
                write_json_atomic(layout.root / "editable_records.json", self._editable_index(records))
            write_json_atomic(layout.memory_config_path, config)
            write_json_atomic(
                metadata_path,
                {
                    "sessionId": self.session_id,
                    "sourceVersion": source_version,
                    "editVersion": edit_version,
                    "snapshotPath": relative_to_session(layout.root, self.session_dir),
                    "config": config,
                    "records": records,
                    "consumed": False,
                    "createdAt": utc_now_iso(),
                },
            )
        except Exception:
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
        return metadata_path

    def _configured_backup_path(self, rollback_state: Any) -> Path | None:
        if not isinstance(rollback_state, dict) or not rollback_state.get("backup_path"):
            return None
        candidate = (self.session_dir / str(rollback_state["backup_path"])).resolve()
        try:
            candidate.relative_to((self.session_dir / "memory" / "rollback").resolve())
        except ValueError:
            return None
        return candidate

    def _cleanup_superseded_backup(self, previous_config: dict[str, Any], *, keep: Path) -> None:
        previous = self._configured_backup_path(previous_config.get("memory_edit_rollback"))
        if previous is None or previous == keep:
            return
        self._discard_rollback_backup(previous)

    def _discard_rollback_backup(self, path: Path) -> None:
        rollback_root = (self.session_dir / "memory" / "rollback").resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(rollback_root)
        except ValueError:
            return
        parent = resolved.parent
        if parent != rollback_root and parent.name.startswith("backup_v"):
            shutil.rmtree(parent, ignore_errors=True)
            return
        try:
            resolved.unlink()
        except OSError:
            pass

    @staticmethod
    def _discard_candidate(candidate: _Candidate) -> None:
        if candidate.root.exists():
            shutil.rmtree(candidate.root, ignore_errors=True)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalized_generation_backend(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "rule" or text.startswith("rule_"):
        return "rule"
    if text == "llm" or text.startswith("llm_"):
        return "llm"
    return None


def _record_source_ids(record: dict[str, Any]) -> set[str]:
    values = record.get("source_doc_ids") or record.get("child_ids") or []
    if not isinstance(values, list):
        values = []
    return {str(value).strip() for value in values if str(value).strip()}


def _semantic_facts(path: Path) -> list[dict[str, Any]]:
    value = read_json(path, default={})
    if isinstance(value, dict):
        value = value.get("facts", [])
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _semantic_facts_from_root(root: Path, model: str) -> list[dict[str, Any]]:
    exact = root / f"semantic_memory_{model}.json"
    if exact.is_file():
        return _semantic_facts(exact)
    candidates = sorted(root.glob("semantic_memory_*.json"))
    return _semantic_facts(candidates[0]) if len(candidates) == 1 else []


def _changed_items(before: list[dict[str, Any]], after: list[dict[str, Any]], id_key: str) -> int:
    def keyed(items: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for index, item in enumerate(items):
            raw_key = str(item.get(id_key) or "").strip()
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            result[raw_key or f"__index_{index}"] = encoded
        return result

    left = keyed(before)
    right = keyed(after)
    return sum(1 for key in set(left) | set(right) if left.get(key) != right.get(key))


def _invalid_record(record_id: str | None, field: str, message: str) -> MemoryEditError:
    return MemoryEditError(
        422,
        "invalid_thirty_second_memory",
        message,
        {"recordId": record_id, "field": field, "limit": None},
    )


def _json_records(path: Path) -> list[dict[str, Any]]:
    value = read_json(path, default=[])
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _prepare_caption_record(value: dict[str, Any], session_id: str) -> dict[str, Any]:
    record = dict(value)
    if record.get("start") is None:
        record["start"] = hhmmssff_to_seconds(record.get("start_time") or 0)
    if record.get("end") is None:
        record["end"] = hhmmssff_to_seconds(record.get("end_time") or record.get("start_time") or 0)
    record.setdefault("session_id", session_id)
    record.setdefault("date", "DAY1")
    record.setdefault("start_time", "00000000")
    record.setdefault("end_time", record["start_time"])
    return record
