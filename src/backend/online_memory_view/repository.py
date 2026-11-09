from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_memory.content import effective_episodic_content
from online_preprocess.io_utils import read_json
from online_time import frame_capture_seconds
from .session_labels import session_labels


@dataclass(frozen=True)
class ActiveMemoryRoot:
    kind: str
    root: Path
    config_path: Path
    config: dict[str, Any]


class MemoryRepositoryError(RuntimeError):
    pass


class LongTermMemoryRepository:
    def __init__(self, sessions_root: Path, session_id: str) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = session_id
        self.session_dir = self.sessions_root / session_id
        self._active_root: ActiveMemoryRoot | None = None

    def resolve_active_root(self) -> ActiveMemoryRoot:
        if not self.session_dir.exists() or not self.session_dir.is_dir():
            raise FileNotFoundError("session not found")
        for kind in ("em2mem", "lightmem_ego"):
            config_path = self.session_dir / kind / "memory_config.json"
            config = read_json(config_path, default={})
            if isinstance(config, dict) and config.get("status") == "memory_ready":
                active = ActiveMemoryRoot(
                    kind=kind,
                    root=config_path.parent,
                    config_path=config_path,
                    config=config,
                )
                self._active_root = active
                return active
        raise MemoryRepositoryError("memory is not ready")

    def reload_config(self, active: ActiveMemoryRoot) -> dict[str, Any]:
        config = read_json(active.config_path, default={})
        if not isinstance(config, dict) or config.get("status") != "memory_ready":
            raise MemoryRepositoryError("memory is not ready")
        return config

    def version_signature(self, config: dict[str, Any]) -> dict[str, Any]:
        lag = config.get("lag") if isinstance(config.get("lag"), dict) else {}
        building = config.get("building_versions") if isinstance(config.get("building_versions"), dict) else {}
        return {
            "memory_version": _positive_int(
                config.get("latest_ready_memory_version") or config.get("memory_version")
            ),
            "semantic_version": _positive_int(
                config.get("latest_semantic_ready_version") or config.get("semantic_version")
            ),
            "graph_version": _positive_int(
                config.get("latest_graph_ready_version") or config.get("graph_version")
            ),
            "visual_version": _positive_int(
                config.get("latest_visual_ready_version") or config.get("visual_version")
            ),
            "building_memory_version": config.get("building_memory_version"),
            "building_semantic_version": building.get("semantic"),
            "building_graph_version": building.get("graph"),
            "building_visual_version": building.get("visual"),
            "semantic_lagging": bool(lag.get("semantic_lagging")),
            "graph_lagging": bool(lag.get("graph_lagging")),
            "visual_lagging": bool(config.get("visual_lagging") or lag.get("visual_lagging")),
            "updated_at": config.get("updated_at"),
        }

    def catalog_metadata(self) -> dict[str, Any]:
        """Return stable memory versions and display metadata for the catalog."""
        active = self.resolve_active_root()
        signature = self.version_signature(active.config)
        self.ensure_stable_components(signature)
        memory_version = signature.get("memory_version")
        if not isinstance(memory_version, int) or isinstance(memory_version, bool) or memory_version <= 0:
            raise MemoryRepositoryError("invalid memory version")
        self.validate_config_paths(active.config)
        return {
            "session_id": self.session_id,
            **session_labels(self.session_dir),
            "memory_version": memory_version,
            "component_versions": {
                "episodic": memory_version,
                "semantic": signature.get("semantic_version"),
                "visual": signature.get("visual_version"),
            },
            "updated_at": active.config.get("updated_at") or active.config.get("last_ready_at"),
            "active_root_kind": active.kind,
            "qa_aligned": active.kind == "em2mem",
        }

    def validate_config_paths(self, config: dict[str, Any]) -> None:
        """Validate configured paths without opening any memory content files."""
        query_args = _query_args(config)
        values = (
            config.get("caption_root"),
            config.get("semantic_root"),
            config.get("visual_root"),
            config.get("visual_items_path"),
            query_args.get("episodic_caption_root"),
            query_args.get("semantic_root"),
            query_args.get("visual_evidence_file"),
        )
        for value in values:
            if value is not None and str(value).strip():
                self._config_path(value)

    def ensure_stable_components(self, signature: dict[str, Any]) -> None:
        if signature.get("building_memory_version") is not None:
            raise MemoryRepositoryError("memory snapshot is changing")
        if signature.get("building_semantic_version") is not None or signature.get("building_visual_version") is not None:
            raise MemoryRepositoryError("memory component is lagging")
        if signature.get("building_graph_version") is not None or signature.get("graph_lagging"):
            raise MemoryRepositoryError("memory component is lagging")
        if signature.get("semantic_lagging") or signature.get("visual_lagging"):
            raise MemoryRepositoryError("memory component is lagging")
        memory_version = signature.get("memory_version")
        semantic_version = signature.get("semantic_version")
        graph_version = signature.get("graph_version")
        visual_version = signature.get("visual_version")
        if memory_version and semantic_version and semantic_version < memory_version:
            raise MemoryRepositoryError("memory component is lagging")
        if memory_version and graph_version and graph_version < memory_version:
            raise MemoryRepositoryError("memory component is lagging")
        if memory_version and visual_version and visual_version < memory_version:
            raise MemoryRepositoryError("memory component is lagging")

    def load_episodic(self, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        query_args = _query_args(config)
        subject = str(query_args.get("subject") or self.session_id)
        caption_root = self._config_path(query_args.get("episodic_caption_root") or config.get("caption_root"))
        if caption_root is None or not caption_root.exists():
            return {key: [] for key in ("30sec", "3min", "10min", "1h")}
        file_map = _episodic_file_map(caption_root, subject)
        result: dict[str, list[dict[str, Any]]] = {}
        for granularity in ("30sec", "3min", "10min", "1h"):
            path_value = file_map.get(granularity)
            if not path_value:
                result[granularity] = []
                continue
            path = self._ensure_session_path(Path(path_value))
            value = read_json(path, default=[])
            result[granularity] = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
        return result

    def load_semantic(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        query_args = _query_args(config)
        semantic_root = self._config_path(query_args.get("semantic_root") or config.get("semantic_root"))
        if semantic_root is None or not semantic_root.exists():
            return []
        model = str(query_args.get("retriever_model") or "")
        try:
            semantic_path = self._ensure_session_path(_resolve_semantic_path(semantic_root, model))
        except FileNotFoundError:
            return []
        value = read_json(semantic_path, default=[])
        if isinstance(value, dict):
            value = value.get("facts", [])
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def load_visual(self, config: dict[str, Any], episodic: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        visual_items_path = self._config_path(config.get("visual_items_path"))
        if visual_items_path is not None and visual_items_path.exists():
            return _read_visual_items(visual_items_path)

        query_args = _query_args(config)
        evidence_path = self._config_path(query_args.get("visual_evidence_file"))
        if evidence_path is None or not evidence_path.exists():
            caption_root = self._config_path(query_args.get("episodic_caption_root") or config.get("caption_root"))
            subject = str(query_args.get("subject") or self.session_id)
            if caption_root is not None:
                inferred = _infer_visual_evidence_file(caption_root, subject)
                if inferred:
                    evidence_path = self._ensure_session_path(inferred)
        if evidence_path is not None and evidence_path.exists():
            return _build_visual_items(self.session_dir, evidence_path)
        return _visual_items_from_episodic(self.session_id, episodic.get("30sec", []))

    def _config_path(self, value: Any) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = self.session_dir / path
        resolved = self._ensure_session_path(path)
        if resolved.exists() or self._active_root is None:
            return resolved
        relative = Path(str(value))
        if not relative.is_absolute() and len(relative.parts) > 1 and relative.parts[0] in {
            "worldmm",
            "em2mem",
            "lightmem_ego",
        }:
            legacy_candidate = self._ensure_session_path(self._active_root.root.joinpath(*relative.parts[1:]))
            if legacy_candidate.exists():
                return legacy_candidate
        return resolved

    def _ensure_session_path(self, path: Path) -> Path:
        session_root = self.session_dir.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(session_root)
        except ValueError as exc:
            raise MemoryRepositoryError("memory path escapes session directory") from exc
        return resolved


def _query_args(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("query_rag_args")
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _visual_items_from_episodic(session_id: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for doc in docs:
        visual_id = str(doc.get("evidence_doc_id") or doc.get("doc_id") or doc.get("segment_id") or "")
        if not visual_id:
            continue
        items.append(
            {
                "visual_id": f"vis_{visual_id}",
                "session_id": session_id,
                "segment_id": doc.get("segment_id"),
                "evidence_doc_id": doc.get("evidence_doc_id") or doc.get("doc_id"),
                "start_time": doc.get("start_time") or doc.get("start"),
                "end_time": doc.get("end_time") or doc.get("end"),
                "timestamp": doc.get("start_time") or doc.get("start"),
                "keyframe_caption": doc.get("keyframe_caption"),
                "segment_caption": effective_episodic_content(doc),
                "scene": doc.get("scene"),
                "visual_objects": doc.get("visual_objects") or [],
                "main_actions": doc.get("main_actions") or [],
                "state_changes": doc.get("state_changes") or [],
                "conversation_focus": doc.get("conversation_focus"),
                "linked_memory_ids": [visual_id],
            }
        )
    return items


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def _first_glob(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _episodic_file_map(root: Path, subject: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for granularity, glob_patterns in {
        "30sec": ["*30sec.json", "*30s.json"],
        "3min": ["*3min.json"],
        "10min": ["*10min.json"],
        "1h": ["*1h.json"],
    }.items():
        names = [
            f"{subject}_EVIDENCE_{granularity}.json",
            f"{subject}_{granularity}.json",
            f"{subject}_EVIDENCE_SMOKE_{granularity}.json",
        ]
        path = _first_existing([root / name for name in names]) or _first_glob(root, glob_patterns)
        if path is not None:
            result[granularity] = path
    return result


def _resolve_semantic_path(root: Path, model: str) -> Path:
    candidates = []
    if model:
        candidates.extend(
            [
                root / f"semantic_memory_{model}.json",
                root / f"semantic_consolidation_results_{model}.json",
            ]
        )
    path = _first_existing(candidates)
    if path is None:
        path = _first_glob(root, ["semantic_memory_*.json", "semantic_consolidation_results_*.json"])
    if path is None:
        raise FileNotFoundError("semantic memory file not found")
    return path


def _infer_visual_evidence_file(root: Path, subject: str) -> Path | None:
    return _first_existing(
        [
            root / f"{subject}_evidence.json",
            root / f"{subject}_EVIDENCE_30sec.json",
            root / f"{subject}_30sec.json",
        ]
    ) or _first_glob(root, ["*evidence*.json", "*30sec.json", "*30s.json"])


def _read_visual_items(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                items.append(value)
    return items


def _build_visual_items(session_dir: Path, evidence_path: Path) -> list[dict[str, Any]]:
    docs = read_json(evidence_path, default=[])
    if not isinstance(docs, list):
        raise MemoryRepositoryError("invalid visual evidence")
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        caption_by_path: dict[str, dict[str, Any]] = {}
        for key in ("keyframe_captions", "keyframe_caption"):
            for item in doc.get(key, []) or []:
                if isinstance(item, dict) and item.get("path"):
                    caption_by_path[str(item["path"])] = item
        raw_paths = doc.get("keyframe_paths") or list(caption_by_path)
        paths = [str(value) for value in raw_paths if value]
        for keyframe_path in paths:
            if keyframe_path in seen_paths:
                continue
            seen_paths.add(keyframe_path)
            frame_meta = caption_by_path.get(keyframe_path, {})
            segment_id = str(doc.get("segment_id") or Path(keyframe_path).parent.name or "segment_unknown")
            visual_id = "vis_" + _safe_id(f"{segment_id}_{Path(keyframe_path).stem}")
            evidence_doc_id = str(
                doc.get("doc_id")
                or doc.get("evidence_doc_id")
                or f"session_{session_dir.name}_{segment_id}"
            )
            items.append(
                {
                    "visual_id": visual_id,
                    "segment_id": segment_id,
                    "evidence_doc_id": evidence_doc_id,
                    "start_time": doc.get("start_time"),
                    "end_time": doc.get("end_time"),
                    "timestamp": frame_capture_seconds(keyframe_path, frame_meta),
                    "keyframe_caption": str(frame_meta.get("caption") or ""),
                    "segment_caption": effective_episodic_content(doc),
                    "scene": doc.get("scene"),
                    "visual_objects": _as_list(doc.get("visual_objects")),
                    "main_actions": _as_list(doc.get("main_actions")),
                    "state_changes": _as_list(doc.get("state_changes")),
                    "conversation_focus": doc.get("conversation_focus"),
                    "linked_memory_ids": [value for value in (evidence_doc_id, segment_id) if value],
                }
            )
    return items


def _safe_id(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    return cleaned.strip("_") or "unknown"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
