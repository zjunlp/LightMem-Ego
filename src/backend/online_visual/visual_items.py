from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import ensure_dir, read_json


def _safe_id(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def _infer_segment_id(doc: dict[str, Any], keyframe_path: str) -> str:
    segment_id = str(doc.get("segment_id") or "").strip()
    if segment_id and segment_id.lower() not in {"none", "null"}:
        return segment_id
    parent = Path(keyframe_path).parent.name
    return parent if parent else "segment_unknown"


def _keyframe_caption_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in doc.get("keyframe_captions", []) or []:
        if isinstance(item, dict) and item.get("path"):
            result[str(item["path"])] = item
    for item in doc.get("keyframe_caption", []) or []:
        if isinstance(item, dict) and item.get("path"):
            result[str(item["path"])] = item
    return result


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_visual_items(session_dir: Path, limit_items: int | None = None, evidence_path: Path | None = None) -> list[dict[str, Any]]:
    evidence_path = evidence_path or session_dir / "evidence" / "session_evidence.json"
    evidence_docs = read_json(evidence_path, default=[])
    if not isinstance(evidence_docs, list):
        raise ValueError(f"Invalid evidence file: {evidence_path}")

    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for doc in evidence_docs:
        if not isinstance(doc, dict):
            continue
        caption_by_path = _keyframe_caption_map(doc)
        keyframe_paths = list(doc.get("keyframe_paths", []) or [])
        if not keyframe_paths:
            keyframe_paths = list(caption_by_path.keys())
        for keyframe_path in keyframe_paths:
            keyframe_path = str(keyframe_path)
            if not keyframe_path or keyframe_path in seen_paths:
                continue
            seen_paths.add(keyframe_path)
            frame_meta = caption_by_path.get(keyframe_path, {})
            segment_id = _infer_segment_id(doc, keyframe_path)
            visual_id = "vis_" + _safe_id(f"{segment_id}_{Path(keyframe_path).stem}")
            evidence_doc_id = str(doc.get("doc_id") or doc.get("evidence_doc_id") or "")
            if not evidence_doc_id or evidence_doc_id.lower().endswith("_none") or evidence_doc_id.lower() in {"none", "null"}:
                evidence_doc_id = f"session_{session_dir.name}_{segment_id}"
            item = {
                "visual_id": visual_id,
                "session_id": str(doc.get("session_id") or session_dir.name),
                "segment_id": segment_id,
                "evidence_doc_id": evidence_doc_id,
                "start_time": doc.get("start_time"),
                "end_time": doc.get("end_time"),
                "timestamp": frame_meta.get("timestamp", doc.get("start_time")),
                "image_path": keyframe_path,
                "keyframe_caption": str(frame_meta.get("caption") or ""),
                "segment_caption": str(doc.get("fine_caption") or doc.get("caption") or ""),
                "scene": doc.get("scene"),
                "visual_objects": _as_list(doc.get("visual_objects")),
                "main_actions": _as_list(doc.get("main_actions")),
                "state_changes": _as_list(doc.get("state_changes")),
                "conversation_focus": doc.get("conversation_focus"),
                "linked_memory_ids": [x for x in [evidence_doc_id, segment_id] if x],
                "clip_path": doc.get("clip_path"),
                "source": "keyframe",
            }
            items.append(item)
            if limit_items is not None and len(items) >= limit_items:
                return items
    return items


def write_visual_items(path: Path, items: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_visual_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
    return items
