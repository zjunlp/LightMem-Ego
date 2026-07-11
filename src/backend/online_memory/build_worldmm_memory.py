from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from online_preprocess.io_utils import read_json, relative_to_session, utc_now_iso, write_json, write_status

from .evidence_to_worldmm import (
    build_caption_items,
    load_online_evidence,
    _load_triplet_map,
    _memory_generation_backend,
    write_caption_files,
    write_semantic_files,
    write_sidecar_files,
)
from .worldmm_layout import WorldMMOnlineLayout, ensure_worldmm_layout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "HippoRAG" / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_caption_scale(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    return data if isinstance(data, list) else []


def _day_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "DAY1").strip().upper()
    if text.startswith("DAY"):
        try:
            return int(text[3:]), text
        except Exception:
            return 1, text
    return 1, text


def _latest_caption_day(caption_30s: list[dict[str, Any]]) -> str:
    days = [str(item.get("date") or "DAY1") for item in caption_30s if isinstance(item, dict)]
    if not days:
        return "DAY1"
    return max(days, key=_day_sort_key)


def _build_visual_embeddings(
    session_dir: Path,
    caption_30s: list[dict[str, Any]],
    output_path: Path,
    num_frames: int = 16,
) -> bool:
    try:
        from worldmm.embedding import EmbeddingModel
    except Exception as exc:
        raise RuntimeError(f"Failed to import WorldMM EmbeddingModel: {exc}") from exc

    model_name = os.getenv("WORLDMM_VIS_EMBED_MODEL") or os.getenv("WORLDMM_VLM2VEC_MODEL_PATH") or "VLM2Vec/VLM2Vec-V2.0"
    embedding_model = EmbeddingModel(vis_model_name=model_name)
    embedding_model.load_model(model_type="vision")

    video_keys = []
    abs_paths = []
    for item in caption_30s:
        video_key = str(item.get("video_path") or item.get("clip_path") or "").strip()
        if not video_key:
            continue
        video_path = session_dir / video_key
        if not video_path.exists():
            continue
        video_keys.append(video_key)
        abs_paths.append(str(video_path))

    if not abs_paths:
        return False

    embeddings = embedding_model.encode_video(abs_paths, num_frames=num_frames, batch_size=1)
    embeddings_dict = {key: embeddings[idx] for idx, key in enumerate(video_keys)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(embeddings_dict, f)
    return True


def build_online_worldmm_memory(
    session_id: str,
    sessions_root: Path,
    force: bool = False,
    skip_visual_embedding: bool = False,
    skip_semantic: bool = False,
    dry_run: bool = False,
    limit_segments: int | None = None,
    model_name: str | None = None,
    source: str = "auto",
    verbose: bool = False,
) -> Path:
    session_dir = sessions_root / session_id
    layout = WorldMMOnlineLayout(session_dir=session_dir, session_id=session_id)
    model = model_name or os.getenv("OPENAI_MODEL") or os.getenv("WORLDMM_MEMORY_MODEL") or "gpt-5"
    log_path = layout.logs_root / "worldmm_memory_adapter.log"
    pipeline_mode = os.getenv("WORLDMM_PIPELINE_MODE", "mst").strip().lower()
    if pipeline_mode not in {"mst", "legacy", "hybrid"}:
        pipeline_mode = "mst"
    requested_source = (source or "auto").strip().lower()
    if requested_source in {"legacy", "legacy_evidence"}:
        requested_source = "online_evidence"
    if requested_source in {"mst", "mst_micro_events"}:
        requested_source = "mst_episodic"

    mst_caption_path = session_dir / "captions" / "mst_session_30sec_captioned.json"
    mst_evidence_path = session_dir / "evidence" / "mst_session_evidence.json"
    mst_episodes_path = session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json"
    legacy_caption_path = session_dir / "captions" / "session_30sec_captioned.json"
    legacy_evidence_path = session_dir / "evidence" / "session_evidence.json"
    mst_ready = mst_caption_path.exists() and mst_evidence_path.exists() and mst_episodes_path.exists()
    legacy_ready = legacy_caption_path.exists() and legacy_evidence_path.exists()
    legacy_fallback_allowed = os.getenv("WORLDMM_ALLOW_LEGACY_EVIDENCE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
    legacy_fallback_used = False

    if requested_source == "auto":
        if pipeline_mode == "legacy":
            source = "online_evidence"
        elif pipeline_mode == "hybrid":
            if mst_ready:
                source = "mst_episodic"
            elif legacy_ready:
                source = "online_evidence"
                legacy_fallback_used = True
            else:
                source = "mst_episodic"
        else:
            if mst_ready:
                source = "mst_episodic"
            elif legacy_fallback_allowed and legacy_ready:
                source = "online_evidence"
                legacy_fallback_used = True
            else:
                source = "mst_episodic"
    else:
        source = requested_source

    if source == "mst_episodic":
        evidence_filename = "mst_session_evidence.json"
        captioned_30sec_rel = "captions/mst_session_30sec_captioned.json"
        evidence_rel = "evidence/mst_session_evidence.json"
        episodic_source = "mst_micro_events"
        worldmm_update_mode = "full_rebuild_fallback"
    elif source == "online_evidence":
        evidence_filename = "session_evidence.json"
        captioned_30sec_rel = "captions/session_30sec_captioned.json"
        evidence_rel = "evidence/session_evidence.json"
        episodic_source = "legacy_evidence"
        worldmm_update_mode = "full_rebuild"
    else:
        raise ValueError("source must be auto, online_evidence/legacy_evidence, or mst_episodic")
    memory_generation_backend = _memory_generation_backend(os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND"))

    if source == "mst_episodic":
        # Stage 5B2 uses refined M_st micro-events as the 30s source.
        # It must not depend on the legacy preprocess/session_30sec.json or segments_30s.json files.
        required = [
            session_dir / evidence_rel,
            session_dir / captioned_30sec_rel,
            session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json",
        ]
    else:
        required = [
            session_dir / evidence_rel,
            session_dir / captioned_30sec_rel,
            session_dir / "preprocess" / "session_30sec.json",
            session_dir / "preprocess" / "segments_30s.json",
            session_dir / "input.mp4",
        ]
    missing = [path for path in required if not path.exists()]
    if missing:
        if not dry_run:
            previous_config = read_json(layout.memory_config_path, default={})
            if not isinstance(previous_config, dict):
                previous_config = {}
            waiting_for = "mst_consolidation" if source == "mst_episodic" else "legacy_evidence"
            waiting_config = dict(previous_config)
            waiting_config.update(
                {
                    "session_id": session_id,
                    "pipeline_mode": pipeline_mode,
                    "requested_30s_source": requested_source,
                    "active_30s_source": Path(captioned_30sec_rel).stem,
                    "episodic_source": episodic_source,
                    "mst_episodic_ready": mst_ready,
                    "legacy_evidence_available": legacy_ready,
                    "legacy_evidence_fallback_used": legacy_fallback_used,
                    "legacy_evidence_used": source == "online_evidence",
                    "memory_build_state": "waiting",
                    "memory_build_waiting_for": waiting_for,
                    "last_build_error": None,
                    "last_waiting_reason": "Missing required memory adapter inputs: " + ", ".join(str(p) for p in missing),
                    "updated_at": utc_now_iso(),
                }
            )
            write_json(layout.memory_config_path, waiting_config)
        raise FileNotFoundError("Missing required memory adapter inputs: " + ", ".join(str(p) for p in missing))

    if dry_run:
        if verbose:
            print(f"dry-run ok: session={session_id} required_inputs={len(required)}")
        return layout.memory_config_path

    if layout.memory_config_path.exists() and not force:
        existing = read_json(layout.memory_config_path, default={})
        if isinstance(existing, dict) and existing.get("status") == "memory_ready" and not existing.get("latest_ready_memory_version"):
            existing["latest_ready_memory_version"] = int(existing.get("memory_version") or 1)
            existing["memory_version"] = int(existing.get("memory_version") or existing["latest_ready_memory_version"])
            existing["memory_build_state"] = existing.get("memory_build_state") or "ready"
            existing["episodic_index_ready"] = existing.get("episodic_index_ready", True)
            existing["hipporag_cache_ready"] = existing.get("hipporag_cache_ready", True)
            existing["last_ready_at"] = existing.get("last_ready_at") or existing.get("created_at") or utc_now_iso()
        if isinstance(existing, dict):
            existing.setdefault("pipeline_mode", pipeline_mode)
            existing.setdefault("active_30s_source", Path(captioned_30sec_rel).stem)
            existing.setdefault("episodic_source", episodic_source)
            existing.setdefault("legacy_evidence_available", legacy_ready)
            existing.setdefault("mst_episodic_ready", mst_ready)
            existing.setdefault("legacy_evidence_used", existing.get("active_30s_source") == "session_30sec_captioned")
            existing.setdefault("legacy_evidence_fallback_used", False)
            write_json(layout.memory_config_path, existing)
        return layout.memory_config_path

    ensure_worldmm_layout(layout)
    previous_config = read_json(layout.memory_config_path, default={})
    if not isinstance(previous_config, dict):
        previous_config = {}
    previous_ready_version = int(
        previous_config.get("latest_ready_memory_version")
        or previous_config.get("memory_version")
        or (1 if previous_config.get("status") == "memory_ready" else 0)
        or 0
    )
    building_memory_version = previous_ready_version + 1
    building_config = dict(previous_config)
    building_config.update(
        {
            "session_id": session_id,
            "pipeline_mode": pipeline_mode,
            "requested_30s_source": requested_source,
            "active_30s_source": Path(captioned_30sec_rel).stem,
            "episodic_source": episodic_source,
            "mst_episodic_ready": mst_ready,
            "legacy_evidence_available": legacy_ready,
            "legacy_evidence_fallback_used": legacy_fallback_used,
            "legacy_evidence_used": source == "online_evidence",
            "building_memory_version": building_memory_version,
            "memory_build_state": "building",
            "last_build_started_at": utc_now_iso(),
            "status": previous_config.get("status", "memory_building"),
        }
    )
    write_json(layout.memory_config_path, building_config)
    write_status(session_dir, session_id, status="processing", stage="memory_building", progress=92, error=None)
    _append_log(log_path, f"build start session={session_id} model={model} source={source}")

    evidence_docs = load_online_evidence(session_dir, evidence_filename=evidence_filename)
    caption_30s = build_caption_items(session_id=session_id, evidence_docs=evidence_docs, limit_segments=limit_segments)
    caption_paths = write_caption_files(
        layout,
        caption_30s,
        model_name=model,
        generation_backend=memory_generation_backend,
    )
    caption_by_scale = {
        "30sec": caption_30s,
        "3min": _load_caption_scale(layout.caption_3min_path),
        "10min": _load_caption_scale(layout.caption_10min_path),
        "1h": _load_caption_scale(layout.caption_1h_path),
    }
    _append_log(log_path, f"caption files written: 30sec={len(caption_30s)}")

    episodic_item_count = sum(len(items) for items in caption_by_scale.values())
    skip_episodic_sidecar = _env_bool("WORLDMM_SKIP_EPISODIC_SIDECAR", False)
    sidecar_paths = {}
    if skip_episodic_sidecar:
        _append_log(log_path, "episodic sidecar skipped by WORLDMM_SKIP_EPISODIC_SIDECAR")
    else:
        write_status(session_dir, session_id, status="processing", stage="memory_building", progress=94, error=None)
        sidecar_paths = write_sidecar_files(
            layout=layout,
            model_name=model,
            caption_by_scale=caption_by_scale,
            generation_backend=memory_generation_backend,
        )
        _append_log(log_path, f"sidecar files written scales={list(sidecar_paths)}")

    semantic_candidates_path = None
    semantic_memory_path = None
    semantic_fact_count = 0
    semantic_memory_ready = False
    if not skip_semantic:
        write_status(session_dir, session_id, status="processing", stage="memory_building", progress=96, error=None)
        semantic_candidates_path, semantic_memory_path, semantic_fact_count = write_semantic_files(
            layout=layout,
            model_name=model,
            caption_30s=caption_30s,
            generation_backend=memory_generation_backend,
            triplet_map=_load_triplet_map(sidecar_paths.get("30sec", {}).get("triplets")) if sidecar_paths.get("30sec") else None,
        )
        semantic_memory_ready = True
        _append_log(log_path, f"semantic files written facts={semantic_fact_count}")

    visual_embedding_ready = False
    visual_embedding_error = None
    if not skip_visual_embedding:
        write_status(session_dir, session_id, status="processing", stage="memory_building", progress=98, error=None)
        try:
            visual_embedding_ready = _build_visual_embeddings(
                session_dir=session_dir,
                caption_30s=caption_30s,
                output_path=layout.visual_embedding_path,
            )
        except Exception as exc:
            visual_embedding_error = str(exc)
            _append_log(log_path, f"visual embedding fallback: {exc}")

    latest_day = _latest_caption_day(caption_30s)
    max_end_time = max((str(item.get("end_time", "00000000")) for item in caption_30s if str(item.get("date") or "DAY1") == latest_day), default="23595999")
    config = {
        "session_id": session_id,
        "status": "memory_ready",
        "memory_version": building_memory_version,
        "latest_ready_memory_version": building_memory_version,
        "building_memory_version": None,
        "memory_build_state": "ready",
        "episodic_index_ready": True,
        "hipporag_cache_ready": False,
        "long_term_partial_ready": True,
        "long_term_full_ready": bool(semantic_memory_ready and visual_embedding_ready),
        "generation_backend": memory_generation_backend,
        "semantic_backend": "llm" if memory_generation_backend == "llm" and semantic_memory_ready else ("rule" if semantic_memory_ready else None),
        "last_ready_at": utc_now_iso(),
        "last_build_started_at": building_config.get("last_build_started_at"),
        "caption_root": relative_to_session(layout.caption_root, session_dir),
        "sidecar_root": relative_to_session(layout.sidecar_root, session_dir),
        "semantic_root": relative_to_session(layout.semantic_root, session_dir),
        "visual_root": relative_to_session(layout.visual_root, session_dir),
        "visual_embedding_path": relative_to_session(layout.visual_embedding_path, session_dir),
        "visual_embedding_ready": visual_embedding_ready,
        "visual_embedding_error": visual_embedding_error,
        "semantic_memory_ready": semantic_memory_ready,
        "semantic_candidates_path": relative_to_session(semantic_candidates_path, session_dir) if semantic_candidates_path else None,
        "semantic_memory_path": relative_to_session(semantic_memory_path, session_dir) if semantic_memory_path else None,
        "evidence_path": evidence_rel,
        "captioned_30sec_path": captioned_30sec_rel,
        "pipeline_mode": pipeline_mode,
        "requested_30s_source": requested_source,
        "active_30s_source": Path(captioned_30sec_rel).stem,
        "episodic_source": episodic_source,
        "worldmm_30s_input_source": Path(captioned_30sec_rel).stem,
        "worldmm_update_mode": worldmm_update_mode,
        "legacy_evidence_available": legacy_ready,
        "legacy_evidence_path": "evidence/session_evidence.json" if legacy_ready else None,
        "legacy_captioned_30sec_path": "captions/session_30sec_captioned.json" if legacy_ready else None,
        "legacy_evidence_fallback_used": legacy_fallback_used,
        "legacy_evidence_used": source == "online_evidence",
        "memory_generation_backend": memory_generation_backend,
        "multiscale_generation_backend": memory_generation_backend,
        "episodic_sidecar_enabled": not skip_episodic_sidecar,
        "episodic_triplet_generation_backend": (
            "disabled"
            if skip_episodic_sidecar
            else ("llm_openie" if memory_generation_backend == "llm" else "rule")
        ),
        "semantic_generation_backend": "llm_semantic_extraction_consolidation" if memory_generation_backend == "llm" else "rule",
        "created_at": utc_now_iso(),
        "source": {
            "input_video": "input.mp4",
            "adapter": "online_worldmm_memory_adapter",
            "episodic_source": episodic_source,
        },
        "counts": {
            "caption_30sec": len(caption_30s),
            "caption_multiscale_total": episodic_item_count,
            "semantic_facts": semantic_fact_count,
        },
        "query_rag_args": {
            "subject": session_id,
            "retriever_model": model,
            "respond_model": os.getenv("WORLDMM_RESPOND_MODEL", model),
            "until_date": latest_day,
            "until_time": max_end_time,
            "episodic_caption_root": relative_to_session(layout.caption_root, session_dir),
            "episodic_sidecar_root": relative_to_session(layout.sidecar_root, session_dir),
            "semantic_root": relative_to_session(layout.semantic_root, session_dir),
            "visual_root": relative_to_session(layout.embeddings_root, session_dir),
            "visual_evidence_file": relative_to_session(layout.visual_evidence_path, session_dir),
        },
        "worldmm_files": {
            "caption_30sec": relative_to_session(caption_paths["30sec"], session_dir),
            "caption_3min": relative_to_session(caption_paths["3min"], session_dir),
            "caption_10min": relative_to_session(caption_paths["10min"], session_dir),
            "caption_1h": relative_to_session(caption_paths["1h"], session_dir),
            "visual_evidence": relative_to_session(layout.visual_evidence_path, session_dir),
        },
    }
    if source == "mst_episodic":
        config.update(
            {
                "mst_episodic_ready": True,
                "mst_episodic_path": "worldmm/mst_episodic/mst_30sec_episodes.json",
                "mst_captioned_30sec_path": captioned_30sec_rel,
                "mst_evidence_path": evidence_rel,
                "last_mst_worldmm_update_at": utc_now_iso(),
            }
        )
    else:
        config.update(
            {
                "mst_episodic_ready": mst_ready,
                "mst_episodic_path": "worldmm/mst_episodic/mst_30sec_episodes.json" if mst_episodes_path.exists() else None,
                "mst_captioned_30sec_path": "captions/mst_session_30sec_captioned.json" if mst_caption_path.exists() else None,
                "mst_evidence_path": "evidence/mst_session_evidence.json" if mst_evidence_path.exists() else None,
            }
        )
    write_json(layout.memory_config_path, config)
    write_status(
        session_dir,
        session_id,
        status="done",
        stage="memory_ready",
        progress=100,
        error=None,
        outputs={"memory_config": relative_to_session(layout.memory_config_path, session_dir)},
    )
    _append_log(log_path, f"memory ready config={layout.memory_config_path}")
    return layout.memory_config_path
