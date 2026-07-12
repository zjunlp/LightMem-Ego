from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, write_json

from .em2mem_layout import Em2MemOnlineLayout, hhmmssff_to_seconds, seconds_to_hhmmssff

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "HippoRAG" / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _memory_generation_backend(value: str | None = None) -> str:
    backend = (value or os.getenv("EM2MEM_MEMORY_GENERATION_BACKEND") or "llm").strip().lower()
    if backend not in {"llm", "rule"}:
        raise ValueError("EM2MEM_MEMORY_GENERATION_BACKEND must be 'llm' or 'rule'")
    return backend


def _build_llm(model_name: str) -> Any:
    from em2mem.llm import LLMModel

    return LLMModel(model_name=model_name, max_retries=_env_int("EM2MEM_MEMORY_LLM_RETRIES", 3))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:80] or "item"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _object_name(obj: Any) -> str:
    if isinstance(obj, dict):
        return _clean_text(obj.get("name") or obj.get("label") or obj.get("object") or obj.get("entity"))
    return _clean_text(obj)


def _attribute_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_attribute_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            key_text = _clean_text(key)
            item_text = _attribute_text(item)
            if key_text and item_text:
                parts.append(f"{key_text}={item_text}")
            elif key_text:
                parts.append(key_text)
        return ", ".join(parts)
    return _clean_text(value)


def _object_to_text(obj: Any) -> str:
    if isinstance(obj, dict):
        name = _object_name(obj)
        attrs = obj.get("attributes") or obj.get("attribute") or []
        attr_text = _attribute_text(attrs)
        if name and attr_text:
            return f"{name} ({attr_text})"
        if name:
            return name
        return _clean_text(json.dumps(obj, ensure_ascii=False))
    return _clean_text(obj)


def _normalize_visual_objects(value: Any) -> list[str]:
    objects = []
    for item in _as_list(value):
        text = _object_to_text(item)
        if text:
            objects.append(text)
    return list(dict.fromkeys(objects))


def _action_text(action: Any) -> str:
    if isinstance(action, dict):
        return _clean_text(action.get("action") or action.get("relation") or action.get("text"))
    return _clean_text(action)


def _action_objects(action: Any) -> list[str]:
    if not isinstance(action, dict):
        return []
    objects = action.get("objects") or action.get("object") or []
    return [_clean_text(x) for x in _as_list(objects) if _clean_text(x)]


def _keyframe_caption_text(doc: dict[str, Any]) -> str:
    captions = []
    for item in doc.get("keyframe_captions", []) or []:
        caption = _clean_text(item.get("caption") if isinstance(item, dict) else item)
        if caption:
            captions.append(caption)
    return " ".join(captions)


def _make_segment_id(doc: dict[str, Any], idx: int) -> str:
    raw = _clean_text(doc.get("segment_id"))
    if raw and raw.lower() != "none":
        return raw
    _start_time, _end_time, start, end = _caption_time_fields(doc)
    return f"seg_{int(round(start)):06d}_{int(round(end)):06d}_{idx:04d}"


def _make_doc_id(session_id: str, doc: dict[str, Any], segment_id: str) -> str:
    raw = _clean_text(doc.get("doc_id"))
    if raw and not raw.endswith("_None"):
        return raw
    return f"session_{session_id}_{segment_id}"


def _caption_text(doc: dict[str, Any]) -> str:
    fine_caption = _clean_text(doc.get("fine_caption"))
    scene = _clean_text(doc.get("scene"))
    transcript = _clean_text(doc.get("transcript"))
    actions = "; ".join(_action_text(x) for x in doc.get("main_actions", []) or [] if _action_text(x))
    state_changes = []
    for item in doc.get("state_changes", []) or []:
        if not isinstance(item, dict):
            continue
        entity = _clean_text(item.get("entity"))
        attr = _clean_text(item.get("attribute"))
        before = _clean_text(item.get("before"))
        after = _clean_text(item.get("after"))
        if entity and (before or after):
            state_changes.append(f"{entity} {attr}: {before} -> {after}".strip())
    parts = []
    if fine_caption:
        parts.append(fine_caption)
    elif scene:
        parts.append(scene)
    if actions:
        parts.append(f"Actions: {actions}")
    if transcript:
        parts.append(f"Speech: {transcript}")
    if state_changes:
        parts.append("State changes: " + "; ".join(state_changes))
    return " ".join(parts).strip() or "No grounded caption is available for this segment."


def _critical_speech_lines(doc: dict[str, Any]) -> list[str]:
    lines = []
    for item in doc.get("transcript_segments", []) or []:
        if isinstance(item, dict):
            text = _clean_text(item.get("text"))
            if text:
                lines.append(text)
    return lines


def _topic_threads(doc: dict[str, Any]) -> list[str]:
    focus = doc.get("conversation_focus")
    if isinstance(focus, list):
        return [_clean_text(x) for x in focus if _clean_text(x)]
    text = _clean_text(focus)
    return [text] if text else []


def _scene_summary(doc: dict[str, Any]) -> dict[str, Any]:
    scene = _clean_text(doc.get("scene"))
    return {
        "dominant_scene": scene or None,
        "scene": scene or None,
        "source": "online_evidence",
    }


def _is_hhmmssff_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not re.fullmatch(r"\d{8}", text):
        return False
    try:
        minutes = int(text[2:4])
        seconds = int(text[4:6])
        frames = int(text[6:8])
    except ValueError:
        return False
    return minutes < 60 and seconds < 60 and frames < 100


def _seconds_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _caption_time_fields(doc: dict[str, Any]) -> tuple[str, str, float, float]:
    raw_start_time = doc.get("start_time")
    raw_end_time = doc.get("end_time")
    start_is_display_code = _is_hhmmssff_text(raw_start_time)
    end_is_display_code = _is_hhmmssff_text(raw_end_time)

    if start_is_display_code:
        start_time = str(raw_start_time).strip()
    else:
        start_time = seconds_to_hhmmssff(_seconds_value(raw_start_time, 0.0))

    if end_is_display_code:
        end_time = str(raw_end_time).strip()
    else:
        fallback_end = raw_start_time if raw_end_time is None else raw_end_time
        end_time = seconds_to_hhmmssff(_seconds_value(fallback_end, 0.0))

    if doc.get("local_start_time") is not None:
        start_sec = _seconds_value(doc.get("local_start_time"), 0.0)
    elif doc.get("start") is not None:
        start_sec = _seconds_value(doc.get("start"), 0.0)
    elif start_is_display_code:
        start_sec = hhmmssff_to_seconds(start_time)
    else:
        start_sec = _seconds_value(raw_start_time, 0.0)

    if doc.get("local_end_time") is not None:
        end_sec = _seconds_value(doc.get("local_end_time"), start_sec)
    elif doc.get("end") is not None:
        end_sec = _seconds_value(doc.get("end"), start_sec)
    elif end_is_display_code:
        end_sec = hhmmssff_to_seconds(end_time)
    else:
        fallback_end = raw_start_time if raw_end_time is None else raw_end_time
        end_sec = _seconds_value(fallback_end, start_sec)

    if end_sec < start_sec:
        end_sec = start_sec
    return start_time, end_time, start_sec, end_sec


def evidence_doc_to_caption_item(session_id: str, doc: dict[str, Any], idx: int) -> dict[str, Any]:
    segment_id = _make_segment_id(doc, idx)
    doc_id = _make_doc_id(session_id, doc, segment_id)
    start_time, end_time, start_sec, end_sec = _caption_time_fields(doc)
    keyframe_captions = doc.get("keyframe_captions", []) or []
    keyframe_paths = list(doc.get("keyframe_paths", []) or [])
    clip_path = _clean_text(doc.get("clip_path"))
    video_path = clip_path or _clean_text(doc.get("source_video_path")) or "input.mp4"

    visual_objects = _normalize_visual_objects(doc.get("visual_objects", []) or [])
    main_actions = doc.get("main_actions", []) or []
    fine_caption = _clean_text(doc.get("fine_caption")) or _caption_text(doc)
    date = _clean_text(doc.get("date") or doc.get("day_label")) or "DAY1"

    item = {
        "doc_id": doc_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "start": start_sec,
        "end": end_sec,
        "duration": round(max(0.0, end_sec - start_sec), 3),
        "video_path": video_path,
        "clip_path": clip_path,
        "source_video_path": _clean_text(doc.get("source_video_path")) or "input.mp4",
        "transcript": _clean_text(doc.get("transcript")),
        "transcript_text": _clean_text(doc.get("transcript")),
        "transcript_segments": doc.get("transcript_segments", []) or [],
        "text": _caption_text(doc),
        "caption": fine_caption,
        "fine_caption": fine_caption,
        "visual_summary": fine_caption,
        "scene": _clean_text(doc.get("scene")) or None,
        "scene_summary": _scene_summary(doc),
        "keyframe_caption": _keyframe_caption_text(doc),
        "keyframe_captions": keyframe_captions,
        "keyframe_paths": keyframe_paths,
        "visual_objects": visual_objects,
        "visual_object_threads": visual_objects,
        "main_actions": main_actions,
        "action_threads": main_actions,
        "state_changes": doc.get("state_changes", []) or [],
        "conversation_focus": doc.get("conversation_focus"),
        "topic_threads": _topic_threads(doc),
        "speakers": doc.get("speakers", []) or [],
        "speaker_stats": doc.get("speakers", []) or [],
        "critical_speech_lines": _critical_speech_lines(doc),
        "evidence_doc_id": doc_id,
        "source_doc_ids": [doc_id],
        "child_ids": [],
        "confidence": doc.get("confidence"),
        "status": doc.get("status", "final"),
    }
    for key in (
        "parent_session_id",
        "child_session_id",
        "source_child_session_id",
        "day_label",
        "local_start_time",
        "local_end_time",
        "display_date",
        "display_start_time",
        "display_end_time",
        "display_time_range",
        "display_datetime_start",
        "display_datetime_end",
        "display_iso_start",
        "display_iso_end",
        "timezone",
        "time_source",
    ):
        if doc.get(key) is not None:
            item[key] = doc[key]
    return item


def load_online_evidence(session_dir: Path, evidence_filename: str = "session_evidence.json") -> list[dict[str, Any]]:
    evidence_path = session_dir / "evidence" / evidence_filename
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing online evidence file: {evidence_path}")
    data = read_json(evidence_path, default=[])
    if not isinstance(data, list):
        raise ValueError(f"evidence/{evidence_filename} must be a JSON list")
    return data


def build_caption_items(session_id: str, evidence_docs: list[dict[str, Any]], limit_segments: int | None = None) -> list[dict[str, Any]]:
    docs = evidence_docs[:limit_segments] if limit_segments is not None else evidence_docs
    items = [evidence_doc_to_caption_item(session_id, doc, idx) for idx, doc in enumerate(docs)]
    return sorted(items, key=lambda x: (x["date"], x["start_time"], x["end_time"]))


def _aggregate_caption_group(group: list[dict[str, Any]], scale: str, idx: int) -> dict[str, Any]:
    first = group[0]
    last = group[-1]
    date = str(first.get("date") or "DAY1")
    doc_id = f"{first['session_id']}_{date}_{scale}_{idx:04d}_{first['start_time']}_{last['end_time']}"
    text_parts = []
    for item in group:
        if item.get("fine_caption"):
            text_parts.append(str(item["fine_caption"]))
        elif item.get("text"):
            text_parts.append(str(item["text"]))
    keyframes = []
    for item in group:
        keyframes.extend(item.get("keyframe_paths", []) or [])
    item = {
        "doc_id": doc_id,
        "session_id": first["session_id"],
        "date": date,
        "start_time": first["start_time"],
        "end_time": last["end_time"],
        "start": first["start"],
        "end": last["end"],
        "duration": round(float(last["end"]) - float(first["start"]), 3),
        "video_path": first.get("video_path"),
        "text": " ".join(text_parts),
        "caption": " ".join(text_parts),
        "fine_caption": " ".join(text_parts),
        "visual_summary": " ".join(text_parts),
        "scene_summary": first.get("scene_summary", {}),
        "keyframe_paths": keyframes,
        "action_threads": [x for item in group for x in item.get("action_threads", []) or []],
        "object_threads": [x for item in group for x in item.get("visual_objects", []) or []],
        "visual_object_threads": [x for item in group for x in item.get("visual_objects", []) or []],
        "topic_threads": [x for item in group for x in item.get("topic_threads", []) or []],
        "critical_speech_lines": [x for item in group for x in item.get("critical_speech_lines", []) or []],
        "source_doc_ids": [item["doc_id"] for item in group],
        "child_ids": [item["doc_id"] for item in group],
        "level": scale,
    }
    passthrough_first_keys = (
        "parent_session_id",
        "child_session_id",
        "source_child_session_id",
        "day_label",
        "display_date",
        "display_start_time",
        "display_datetime_start",
        "display_iso_start",
        "timezone",
        "time_source",
    )
    passthrough_last_keys = (
        "display_end_time",
        "display_datetime_end",
        "display_iso_end",
    )
    for key in passthrough_first_keys:
        if first.get(key) is not None:
            item[key] = first[key]
    for key in passthrough_last_keys:
        if last.get(key) is not None:
            item[key] = last[key]
    if first.get("local_start_time") is not None:
        item["local_start_time"] = first["local_start_time"]
    if last.get("local_end_time") is not None:
        item["local_end_time"] = last["local_end_time"]
    if item.get("display_start_time") and item.get("display_end_time"):
        item["display_time_range"] = f"{item['display_start_time']}-{item['display_end_time']}"
    return item


def build_multiscale_caption_items(caption_30s: list[dict[str, Any]], window_seconds: int, scale: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in caption_30s:
        date = str(item.get("date") or "DAY1")
        bucket = int(float(item["start"]) // window_seconds)
        buckets[(date, bucket)].append(item)
    output = []
    for idx, key in enumerate(sorted(buckets)):
        group = sorted(buckets[key], key=lambda x: x["start"])
        if group:
            output.append(_aggregate_caption_group(group, scale, idx))
    return output


def _triplets_for_item(item: dict[str, Any]) -> list[list[str]]:
    triplets: list[list[str]] = []
    scene = _clean_text(item.get("scene"))
    if scene:
        triplets.append(["segment", "occurs_in", scene])

    for obj in item.get("visual_objects", []) or []:
        name = _object_name(obj)
        if name:
            triplets.append([name, "appears_in", scene or "segment"])

    for action in item.get("main_actions", []) or []:
        action_text = _action_text(action)
        if not action_text:
            continue
        actor = _clean_text(action.get("actor")) if isinstance(action, dict) else "person"
        objects = _action_objects(action)
        if objects:
            for obj in objects:
                triplets.append([actor or "person", action_text, obj])
        else:
            triplets.append([actor or "person", "does", action_text])

    for change in item.get("state_changes", []) or []:
        if not isinstance(change, dict):
            continue
        entity = _clean_text(change.get("entity"))
        attr = _clean_text(change.get("attribute")) or "state"
        after = _clean_text(change.get("after"))
        before = _clean_text(change.get("before"))
        if entity and after:
            triplets.append([entity, f"{attr}_changed_to", after])
        if entity and before:
            triplets.append([entity, f"{attr}_changed_from", before])

    for topic in _topic_threads(item):
        triplets.append(["conversation", "discusses", topic])

    return _dedupe_triplets(triplets)


def _dedupe_triplets(triplets: list[list[str]]) -> list[list[str]]:
    seen = set()
    out = []
    for tri in triplets:
        if len(tri) != 3:
            continue
        clean = [_clean_text(x) for x in tri]
        if not all(clean):
            continue
        key = tuple(x.lower() for x in clean)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _load_caption_file(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    return data if isinstance(data, list) else []


def _item_text_for_openie(item: dict[str, Any]) -> str:
    parts = [
        _clean_text(item.get("text")),
        _clean_text(item.get("fine_caption")),
        _clean_text(item.get("transcript")),
        _clean_text(item.get("keyframe_caption")),
    ]
    return "\n".join(part for part in parts if part) or _clean_text(item)


def _openie_triplets_for_items(items: list[dict[str, Any]], model_name: str, output_dir: Path) -> dict[str, list[list[str]]]:
    from em2mem.memory.episodic.openie import OpenIE

    chunks = {
        str(item["doc_id"]): {"content": _item_text_for_openie(item)}
        for item in items
        if item.get("doc_id")
    }
    if not chunks:
        return {}
    openie = OpenIE(llm_model=_build_llm(model_name))
    _ner_map, triple_map = openie.batch_openie(
        chunks=chunks,
        output_dir=str(output_dir),
        max_workers=_env_int("EM2MEM_OPENIE_MAX_WORKERS", 4),
    )
    return {str(key): _dedupe_triplets(value or []) for key, value in (triple_map or {}).items()}


def _write_llm_multiscale_caption_files(
    layout: Em2MemOnlineLayout,
    caption_30s: list[dict[str, Any]],
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from em2mem.memory.episodic.gen_multiscale import gen_multiscale

    llm = _build_llm(model_name)
    write_json(layout.caption_30sec_path, caption_30s)
    if os.getenv("EM2MEM_MEMORY_FORCE_LLM_REGENERATE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        for path in (layout.caption_3min_path, layout.caption_10min_path, layout.caption_1h_path):
            if path.exists():
                path.unlink()
    gen_multiscale(
        input_json=str(layout.caption_30sec_path),
        save_dir=str(layout.caption_root),
        llm=llm,
        windows=(180, 600, 3600),
        granularity_names=(
            f"{layout.subject}_3min",
            f"{layout.subject}_10min",
            f"{layout.subject}_1h",
        ),
        perspective=os.getenv("EM2MEM_MULTISCALE_PERSPECTIVE", "egocentric"),
        default_date="DAY1",
    )
    return (
        _load_caption_file(layout.caption_3min_path),
        _load_caption_file(layout.caption_10min_path),
        _load_caption_file(layout.caption_1h_path),
    )


def write_caption_files(
    layout: Em2MemOnlineLayout,
    caption_30s: list[dict[str, Any]],
    model_name: str | None = None,
    generation_backend: str | None = None,
) -> dict[str, Path]:
    backend = _memory_generation_backend(generation_backend)
    if backend == "llm":
        if not model_name:
            raise ValueError("model_name is required for LLM multiscale caption generation")
        caption_3min, caption_10min, caption_1h = _write_llm_multiscale_caption_files(
            layout=layout,
            caption_30s=caption_30s,
            model_name=model_name,
        )
    else:
        caption_3min = build_multiscale_caption_items(caption_30s, 180, "3min")
        caption_10min = build_multiscale_caption_items(caption_30s, 600, "10min")
        caption_1h = build_multiscale_caption_items(caption_30s, 3600, "1h")
        write_json(layout.caption_30sec_path, caption_30s)
        write_json(layout.caption_3min_path, caption_3min)
        write_json(layout.caption_10min_path, caption_10min)
        write_json(layout.caption_1h_path, caption_1h)

    write_json(layout.caption_30sec_path, caption_30s)
    write_json(layout.caption_3min_path, caption_3min)
    write_json(layout.caption_10min_path, caption_10min)
    write_json(layout.caption_1h_path, caption_1h)
    write_json(layout.visual_evidence_path, caption_30s)
    return {
        "30sec": layout.caption_30sec_path,
        "3min": layout.caption_3min_path,
        "10min": layout.caption_10min_path,
        "1h": layout.caption_1h_path,
    }


def write_sidecar_files(
    layout: Em2MemOnlineLayout,
    model_name: str,
    caption_by_scale: dict[str, list[dict[str, Any]]],
    generation_backend: str | None = None,
) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    folder_by_scale = {"30sec": "30s", "3min": "3min", "10min": "10min", "1h": "1h"}
    filename_scale_by_scale = {"30sec": "30s", "3min": "3min", "10min": "10min", "1h": "1h"}
    backend = _memory_generation_backend(generation_backend)

    for scale, items in caption_by_scale.items():
        folder = folder_by_scale[scale]
        filename_scale = filename_scale_by_scale[scale]
        out_dir = layout.sidecar_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        if backend == "llm":
            triplet_map = _openie_triplets_for_items(
                items=items,
                model_name=model_name,
                output_dir=out_dir,
            )
        else:
            triplet_map = {str(item["doc_id"]): _triplets_for_item(item) for item in items}

        units = []
        for idx, item in enumerate(items):
            triplets = triplet_map.get(str(item["doc_id"]), [])
            unit = {
                "doc_id": item["doc_id"],
                "date": item["date"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "text": item.get("text", ""),
                "fine_caption": item.get("fine_caption", ""),
                "video_path": item.get("video_path", ""),
                "source_doc_ids": item.get("source_doc_ids", []),
                "child_ids": item.get("child_ids", []),
                "openie_results": triplets,
                "raw_triplets": triplets,
                "episodic_triplets": triplets,
                "idx": idx,
            }
            units.append(unit)

        triplet_payload = {
            "scale": filename_scale,
            "units": units,
            "triplet_map": triplet_map,
            "source": "online_evidence_adapter_llm_openie" if backend == "llm" else "online_evidence_adapter_rule",
        }

        graph = _build_graph_payload(items, triplet_map, scale)
        triplet_path = out_dir / f"episodic_triplets_{filename_scale}_{model_name}.json"
        graph_path = out_dir / f"episodic_graph_{filename_scale}_{model_name}.json"
        write_json(triplet_path, triplet_payload)
        write_json(graph_path, graph)
        outputs[scale] = {"triplets": triplet_path, "graph": graph_path}
    return outputs


def _build_graph_payload(items: list[dict[str, Any]], triplet_map: dict[str, list[list[str]]], scale: str) -> dict[str, Any]:
    nodes = []
    edges = []
    doc_id_to_event_id = {}
    entity_ids: dict[str, str] = {}

    def entity_node(label: str) -> str:
        key = _slug(label)
        node_id = entity_ids.get(key)
        if node_id:
            return node_id
        node_id = f"entity_{key}"
        entity_ids[key] = node_id
        nodes.append({"id": node_id, "type": "Entity", "label": label})
        return node_id

    prev_event_id = None
    for item in items:
        doc_id = item["doc_id"]
        event_id = f"event_{_slug(doc_id)}"
        doc_id_to_event_id[doc_id] = event_id
        nodes.append({
            "id": event_id,
            "type": "Event",
            "label": doc_id,
            "text": item.get("text", ""),
            "visual_summary": item.get("visual_summary", ""),
            "action_threads": item.get("action_threads", []),
            "object_threads": item.get("object_threads", []),
            "topic_threads": item.get("topic_threads", []),
            "visual_object_threads": item.get("visual_object_threads", []),
            "critical_speech_lines": item.get("critical_speech_lines", []),
            "scene_summary": item.get("scene_summary", {}),
        })
        if prev_event_id:
            edges.append({"source": prev_event_id, "target": event_id, "type": "before"})
        prev_event_id = event_id

        for tri in triplet_map.get(doc_id, []):
            h, r, t = tri
            h_id = entity_node(h)
            t_id = entity_node(t)
            edges.append({"source": event_id, "target": h_id, "type": "mentions", "event_id": event_id})
            edges.append({"source": event_id, "target": t_id, "type": "mentions", "event_id": event_id})
            edges.append({"source": h_id, "target": t_id, "type": r, "event_id": event_id})

    return {
        "graph_type": "event_centric_episodic_graph",
        "scale": scale,
        "nodes": nodes,
        "edges": edges,
        "doc_id_to_event_id": doc_id_to_event_id,
        "source": "online_evidence_adapter",
    }


def _load_triplet_map(path: Path) -> dict[str, list[list[str]]]:
    data = read_json(path, default={})
    if not isinstance(data, dict):
        return {}
    raw = data.get("triplet_map") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): _dedupe_triplets(value or []) for key, value in raw.items()}


def _semantic_facts_from_consolidated(
    caption_30s: list[dict[str, Any]],
    consolidated_triples: list[list[str]],
    consolidated_evidence: list[list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    item_by_doc = {str(item.get("doc_id")): item for item in caption_30s}
    facts: list[dict[str, Any]] = []
    timeline_map: dict[str, list[str]] = defaultdict(list)
    candidate_lines: list[str] = []

    for idx, triple in enumerate(consolidated_triples):
        if not isinstance(triple, list) or len(triple) < 3:
            continue
        clean = [_clean_text(x) for x in triple[:3]]
        if not all(clean):
            continue
        support_docs = []
        if idx < len(consolidated_evidence):
            support_docs = [str(x) for x in _as_list(consolidated_evidence[idx]) if str(x).strip()]
        support_docs = list(dict.fromkeys(support_docs))
        support_items = [item_by_doc[x] for x in support_docs if x in item_by_doc]
        if not support_items and caption_30s:
            support_items = [caption_30s[0]]
        first = min(support_items, key=lambda x: (str(x.get("date", "DAY1")), str(x.get("start_time", "")))) if support_items else {}
        last = max(support_items, key=lambda x: (str(x.get("date", "DAY1")), str(x.get("end_time", "")))) if support_items else first
        support_days = sorted({str(item.get("date") or "DAY1") for item in support_items}) or ["DAY1"]
        first_date = str(first.get("date") or support_days[0])
        last_date = str(last.get("date") or support_days[-1])
        try:
            last_day_number = int(last_date.upper().replace("DAY", ""))
        except Exception:
            last_day_number = 1
        timestamp = f"{last_day_number}{last.get('end_time', '00000000')}"
        fact_id = f"fact_llm_{idx:05d}_{_slug('_'.join(clean))[:40]}"
        text = f"{clean[0]} {clean[1]} {clean[2]}"
        fact = {
            "fact_id": fact_id,
            "triple": clean,
            "head": clean[0],
            "relation": clean[1],
            "tail": clean[2],
            "head_type": "entity",
            "tail_type": "entity",
            "semantic_summary": text,
            "support_count": max(1, len(support_docs)),
            "support_days": support_days,
            "support_scales": ["30sec"],
            "confidence": 0.75,
            "habit_strength": "medium" if len(support_docs) > 1 else "low",
            "raw_support_count": max(1, len(support_docs)),
            "support_docs": support_docs,
            "evidence_event_ids": support_docs,
            "provenance_root_ids": support_docs,
            "source_doc_ids": support_docs,
            "first_seen": {
                "date": first_date,
                "start_time": first.get("start_time"),
                "end_time": first.get("end_time"),
            },
            "last_seen": {
                "date": last_date,
                "start_time": last.get("start_time"),
                "end_time": last.get("end_time"),
            },
        }
        facts.append(fact)
        timeline_map[timestamp].append(fact_id)
        candidate_lines.append(json.dumps({
            "fact_id": fact_id,
            "fact_type": "llm_semantic_consolidation",
            "text": text,
            "entities": [clean[0], clean[2]],
            "evidence_doc_ids": support_docs,
            "start_time": first.get("start"),
            "end_time": last.get("end"),
        }, ensure_ascii=False))

    timeline = [
        {"timestamp": ts, "fact_ids": fact_ids}
        for ts, fact_ids in sorted(timeline_map.items())
    ]
    return facts, timeline, candidate_lines


def _write_semantic_files_with_llm(
    layout: Em2MemOnlineLayout,
    model_name: str,
    caption_30s: list[dict[str, Any]],
    triplet_map: dict[str, list[list[str]]],
) -> tuple[Path, Path, int]:
    from em2mem.embedding import EmbeddingModel
    from em2mem.memory.semantic import SemanticConsolidation, SemanticExtraction

    candidate_path = layout.semantic_root / "semantic_candidates.jsonl"
    memory_path = layout.semantic_root / f"semantic_memory_{model_name}.json"
    extraction = SemanticExtraction(llm_model=_build_llm(model_name), max_retries=_env_int("EM2MEM_SEMANTIC_LLM_RETRIES", 2))
    payload_batch = {}
    ordered_items = sorted(caption_30s, key=lambda x: (str(x.get("date", "DAY1")), str(x.get("start_time", "")), str(x.get("end_time", ""))))
    for item in ordered_items:
        doc_id = str(item["doc_id"])
        payload_batch[doc_id] = {
            "triples": triplet_map.get(doc_id, []),
            "metadata": {
                "doc_id": doc_id,
                "timestamp": f"1{item['end_time']}",
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
            },
        }
    extraction_results = extraction.batch_semantic_extraction_with_metadata(
        episodic_payload_batch=payload_batch,
        output_dir=str(layout.semantic_root),
    )

    consolidation = SemanticConsolidation(
        llm_model=_build_llm(model_name),
        embedding_model=EmbeddingModel(),
    )
    accumulated_triples: list[list[str]] = []
    accumulated_evidence: list[list[str]] = []
    for item in ordered_items:
        doc_id = str(item["doc_id"])
        new_triples = (extraction_results.get("semantic_triples") or {}).get(doc_id, []) or []
        new_evidence = [[doc_id] for _ in new_triples]
        consolidated, evidence, remove_items = consolidation.batch_semantic_consolidation(
            existing_semantic_results=(accumulated_triples, accumulated_evidence),
            new_semantic_results=(new_triples, new_evidence),
        )
        for removed_triple, removed_evidence in remove_items:
            for ridx, (old_triple, old_evidence) in enumerate(list(zip(accumulated_triples, accumulated_evidence))):
                if old_triple == removed_triple and old_evidence == removed_evidence:
                    accumulated_triples.pop(ridx)
                    accumulated_evidence.pop(ridx)
                    break
        accumulated_triples.extend(consolidated)
        accumulated_evidence.extend(evidence)

    facts, timeline, candidate_lines = _semantic_facts_from_consolidated(
        caption_30s=caption_30s,
        consolidated_triples=accumulated_triples,
        consolidated_evidence=accumulated_evidence,
    )
    candidate_path.write_text("\n".join(candidate_lines) + ("\n" if candidate_lines else ""), encoding="utf-8")
    write_json(memory_path, {
        "facts": facts,
        "timeline": timeline,
        "source": "em2mem_llm_semantic_extraction_consolidation",
        "semantic_memory_ready": True,
        "semantic_generation_backend": "llm",
        "semantic_extraction_results": f"semantic_extraction_results_{model_name}.json",
    })
    return candidate_path, memory_path, len(facts)


def write_semantic_files(
    layout: Em2MemOnlineLayout,
    model_name: str,
    caption_30s: list[dict[str, Any]],
    generation_backend: str | None = None,
    triplet_map: dict[str, list[list[str]]] | None = None,
) -> tuple[Path, Path, int]:
    backend = _memory_generation_backend(generation_backend)
    if backend == "llm":
        return _write_semantic_files_with_llm(
            layout=layout,
            model_name=model_name,
            caption_30s=caption_30s,
            triplet_map=triplet_map or {},
        )

    candidate_path = layout.semantic_root / "semantic_candidates.jsonl"
    memory_path = layout.semantic_root / f"semantic_memory_{model_name}.json"
    facts = []
    timeline = []
    candidate_lines = []

    for item in caption_30s:
        doc_id = item["doc_id"]
        triples = _triplets_for_item(item)
        date = str(item.get("date") or "DAY1")
        try:
            day_number = int(date.upper().replace("DAY", ""))
        except Exception:
            day_number = 1
        timestamp = f"{day_number}{item['end_time']}"
        fact_ids = []
        for idx, triple in enumerate(triples):
            fact_id = f"fact_{_slug(doc_id)}_{idx:03d}"
            text = f"{triple[0]} {triple[1]} {triple[2]}"
            fact = {
                "fact_id": fact_id,
                "triple": triple,
                "head_type": "entity",
                "tail_type": "entity",
                "semantic_summary": text,
                "support_count": 1,
                "support_days": [date],
                "support_scales": ["30sec"],
                "confidence": 0.55,
                "habit_strength": "low",
                "raw_support_count": 1,
                "support_docs": [doc_id],
                "evidence_event_ids": [doc_id],
                "provenance_root_ids": [doc_id],
                "source_doc_ids": [doc_id],
                "first_seen": {"date": date, "start_time": item["start_time"], "end_time": item["end_time"]},
                "last_seen": {"date": date, "start_time": item["start_time"], "end_time": item["end_time"]},
            }
            facts.append(fact)
            fact_ids.append(fact_id)
            candidate_lines.append(json.dumps({
                "fact_id": fact_id,
                "session_id": item["session_id"],
                "segment_id": item["segment_id"],
                "start_time": item["start"],
                "end_time": item["end"],
                "fact_type": "evidence_triplet",
                "text": text,
                "entities": [triple[0], triple[2]],
                "evidence_doc_id": doc_id,
                "keyframe_paths": item.get("keyframe_paths", []),
            }, ensure_ascii=False))
        timeline.append({"timestamp": timestamp, "doc_id": doc_id, "fact_ids": fact_ids})

    candidate_path.write_text("\n".join(candidate_lines) + ("\n" if candidate_lines else ""), encoding="utf-8")
    write_json(memory_path, {
        "facts": facts,
        "timeline": timeline,
        "source": "online_evidence_adapter_rule",
        "semantic_memory_ready": True,
        "semantic_generation_backend": "rule",
    })
    return candidate_path, memory_path, len(facts)
