"""Regenerate missing visual summaries and publish a validated memory snapshot."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from online_memory.visual_summary import aggregate_visual_summaries
from online_memory.frame_captions import inherit_frame_captions, merge_frame_captions
from online_memory.em2mem_layout import Em2MemOnlineLayout, ensure_em2mem_layout
from online_memory_edit.lock import session_memory_mutation_lock
from online_memory_edit.service import MemoryEditService
from online_memory_incremental.snapshot_manager import pin_query_snapshot
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
from online_time import event_time_seconds


def record_id(record):
    return str(record.get("doc_id") or record.get("evidence_doc_id") or record.get("episode_id"))


def record_fingerprint(record):
    content = {key: record.get(key) for key in ("doc_id", "start", "end", "start_time", "end_time",
               "text", "caption", "fine_caption", "keyframe_paths")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def active_records(session_dir: Path):
    config = read_json(session_dir / "em2mem/memory_config.json", default={})
    pinned = pin_query_snapshot(session_dir, config)
    root = session_dir / pinned["caption_root"]
    path = root / f"{session_dir.name}_30sec.json"
    return config, read_json(path, default=[])


def generate_missing(session_dir: Path, draft_path: Path, *, builder=None, on_progress=None):
    """Persist each successful result so retries never repay completed work."""
    session_dir = Path(session_dir)
    _, records = active_records(session_dir)
    draft = read_json(draft_path, default={}) or {}
    builder = builder or MSTEpisodicWindowBuilder(backend="openai")
    for index, record in enumerate(records):
        if record.get("visual_summary"):
            continue
        identity, fingerprint = record_id(record), record_fingerprint(record)
        saved = draft.get(identity) or {}
        if saved.get("fingerprint") == fingerprint and saved.get("visual_summary"):
            continue
        start = event_time_seconds(record.get("start_time", record.get("start")))
        end = event_time_seconds(record.get("end_time", record.get("end")))
        summary, status, debug = builder._build_visual_summary(
            {"start_time": start, "end_time": end}, record,
            list(record.get("keyframe_paths") or []), list(record.get("keyframe_captions") or []),
            str(record.get("transcript") or ""), session_dir)
        draft[identity] = {"fingerprint": fingerprint, "visual_summary": summary,
            "visual_summary_status": status, "visual_summary_debug": debug, "generated_at": utc_now_iso()}
        write_json_atomic(draft_path, draft)
        if on_progress:
            on_progress({"session_id": session_dir.name, "record": identity, "index": index+1,
                         "total": len(records), "status": status})
    return draft


def _apply_visual(record, result):
    if record_fingerprint(record) != result.get("fingerprint") or not result.get("visual_summary"):
        return record
    return {**record, **{key: result[key] for key in (
        "visual_summary", "visual_summary_status", "visual_summary_debug") if key in result},
        "keyframe_captions": merge_frame_captions(list(record.get("keyframe_captions") or [])
            + list((result.get("visual_summary_debug") or {}).get("keyframe_captions") or [])),
        "visual_summary_repaired_at": result.get("generated_at")}


def _bounds(record):
    return (event_time_seconds(record.get("start_time", record.get("start"))),
            event_time_seconds(record.get("end_time", record.get("end"))))


def publish_repair(session_dir: Path, draft: dict[str, Any], *, runtime_validator=None,
                   frame_caption_sources: list[dict[str, Any]] | None = None):
    """Keep the previous publication live until the complete candidate loads.

    Visual summaries and optional capture labels change. Narrative graphs and
    semantic facts are carried forward; caption vectors are validated by the
    same runtime validator used for memory editing.
    """
    session_dir = Path(session_dir)
    service = MemoryEditService(session_dir.parent, session_dir.name, runtime_validator=runtime_validator)
    with session_memory_mutation_lock(session_dir):
        config, records = active_records(session_dir)
        pinned = pin_query_snapshot(session_dir, config)
        updated = [_apply_visual(record, draft.get(record_id(record), {})) if not record.get("visual_summary") else record
                   for record in records]
        if frame_caption_sources is not None:
            updated = [inherit_frame_captions(record, frame_caption_sources) for record in updated]
        changed = [record_id(before) for before, after in zip(records, updated) if before != after]
        summary_changed = any(before.get("visual_summary") != after.get("visual_summary")
                              for before, after in zip(records, updated))
        if not changed:
            return {"status": "unchanged", "session_id": session_dir.name, "repaired": 0}
        version = int(config.get("latest_ready_memory_version") or config["memory_version"]) + 1
        temp, target = service._snapshot_paths(version)
        layout = Em2MemOnlineLayout(temp, session_dir.name)
        ensure_em2mem_layout(layout)
        service._copy_configured_components(pinned, layout, include_derived=True)
        source_visual_file = (pinned.get("query_rag_args") or {}).get("visual_evidence_file")
        if source_visual_file and (session_dir / source_visual_file).is_file():
            shutil.copy2(session_dir / source_visual_file, layout.visual_root / "session_visual_evidence.json")
        write_json_atomic(layout.caption_30sec_path, updated)
        for scale, path in [("3min", layout.caption_3min_path), ("10min", layout.caption_10min_path), ("1h", layout.caption_1h_path)]:
            parents = read_json(path, default=[]) or []
            for parent in parents if summary_changed else []:
                start, end = _bounds(parent)
                children = [child for child in updated if start is not None and end is not None
                    and _bounds(child)[0] is not None and _bounds(child)[1] is not None
                    and _bounds(child)[0] < end and _bounds(child)[1] > start]
                parent["visual_summary"] = aggregate_visual_summaries(children)
                parent["visual_summary_status"] = "derived" if parent["visual_summary"] else "unavailable"
            write_json_atomic(path, parents)
        # The visual evidence catalogue is metadata over these same captures.
        visual_file = layout.visual_root / "session_visual_evidence.json"
        visual = read_json(visual_file, default=[])
        lookup = {record_id(record): record for record in updated}
        def patch_catalog(value):
            if isinstance(value, list):
                return [patch_catalog(row) for row in value]
            if isinstance(value, dict):
                row = {key: patch_catalog(child) if isinstance(child, (dict,list)) else child for key,child in value.items()}
                source = lookup.get(record_id(row))
                if source:
                    row.update({key: source[key] for key in ("visual_summary", "visual_summary_status", "visual_summary_debug", "keyframe_captions") if key in source})
                return row
            return value
        if visual_file.exists():
            write_json_atomic(visual_file, patch_catalog(visual))
        write_json_atomic(layout.root / "editable_records.json", service._editable_index(updated))
        operation = "frame_caption_repair" if frame_caption_sources is not None else "visual_summary_repair"
        propagation = {"operation": operation, "changed_source_ids": changed,
                       "graph_and_semantic": "carried forward; source narrative text is unchanged"}
        backends = {"multiscale": str(config.get("multiscale_generation_backend") or "rule"),
                    "triplets": str(config.get("episodic_triplet_generation_backend") or "preserved"),
                    "semantic": str(config.get("semantic_generation_backend") or "preserved")}
        candidate_config = service._snapshot_config(config, layout, target, version,
            model=str((config.get("query_rag_args") or {}).get("retriever_model") or "gpt-6.1-sol"),
            rollback_available=False, mode=operation, backends=backends,
            propagation=propagation, updated_at=utc_now_iso())
        if draft:
            candidate_config["visual_summary_repair"] = {"repaired_doc_ids": changed, "previous_version": version-1}
        if frame_caption_sources is not None:
            candidate_config["frame_caption_repair"] = {"repaired_doc_ids": changed, "previous_version": version-1,
                "description_source": "existing_short_term_capture_observations", "additional_generation_model_calls": 0}
        if config.get("memory_edit_rollback"):
            candidate_config["memory_edit_rollback"] = dict(config["memory_edit_rollback"])
        candidate = service._finalize_candidate(temp, target, layout, candidate_config, version, updated, propagation)
        try:
            service._publish(candidate, config)
        except Exception:
            service._discard_candidate(candidate)
            raise
        # Keep append/consolidation source files aligned so a later append does
        # not reintroduce a failed summary. Published older snapshots stay intact.
        episode_lookup = {str(row.get("episode_id")): row for row in updated if row.get("episode_id")}
        for key in ("mst_episodic_path", "mst_captioned_30sec_path", "mst_evidence_path"):
            value = config.get(key)
            if not value:
                continue
            path = (session_dir / str(value)).resolve()
            if not path.is_relative_to(session_dir.resolve()) or not path.is_file():
                continue
            data = read_json(path, default=[])
            rows = data if isinstance(data, list) else data.get("events", data.get("episodes", [])) if isinstance(data,dict) else []
            for row in rows:
                source = lookup.get(record_id(row)) or episode_lookup.get(str(row.get("episode_id")))
                if source:
                    row.update({key: source[key] for key in ("visual_summary", "visual_summary_status", "visual_summary_debug", "keyframe_captions") if key in source})
            write_json_atomic(path, data)
        return {"status": "published", "session_id": session_dir.name, "version": version,
                "previous_version": version-1, "repaired": len(changed), "snapshot": str(target)}


def publish_frame_caption_repair(session_dir: Path, *, runtime_validator=None):
    """Publish saved ST observations as LT frame metadata, without generation."""
    from online_short_term.mst_store import MSTStore
    sources = MSTStore(Path(session_dir)).load_archive_events()
    return publish_repair(session_dir, {}, runtime_validator=runtime_validator,
                          frame_caption_sources=sources)
