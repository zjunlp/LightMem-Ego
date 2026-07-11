from __future__ import annotations

from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _component_versions_path(session_dir: Path) -> Path:
    return Path(session_dir) / "worldmm" / "incremental" / "component_versions.json"


def _merge_component(existing: dict[str, Any], key: str, patch: dict[str, Any]) -> None:
    current = existing.get(key) if isinstance(existing.get(key), dict) else {}
    merged = dict(current)
    merged.update(patch)
    existing[key] = merged


def merge_component_versions(session_dir: Path, patches: dict[str, dict[str, Any]], *, reconcile: bool = True) -> dict[str, Any]:
    session_dir = Path(session_dir)
    path = _component_versions_path(session_dir)
    versions = read_json(path, default={})
    if not isinstance(versions, dict):
        versions = {}
    versions["session_id"] = session_dir.name
    for key, patch in patches.items():
        if isinstance(patch, dict):
            _merge_component(versions, key, patch)
        else:
            versions[key] = patch
    versions["updated_at"] = utc_now_iso()
    write_json_atomic(path, versions)
    if reconcile:
        return reconcile_component_versions(session_dir)
    return versions


def reconcile_component_versions(session_dir: Path) -> dict[str, Any]:
    session_dir = Path(session_dir)
    versions_path = _component_versions_path(session_dir)
    versions = read_json(versions_path, default={})
    if not isinstance(versions, dict):
        versions = {}
    config_path = session_dir / "worldmm" / "memory_config.json"
    config = read_json(config_path, default={})
    if not isinstance(config, dict):
        config = {}
    graph_state = read_json(session_dir / "worldmm" / "incremental" / "graph" / "graph_state.json", default={})
    semantic_state = read_json(session_dir / "worldmm" / "incremental" / "semantic" / "semantic_state.json", default={})
    if not isinstance(graph_state, dict):
        graph_state = {}
    if not isinstance(semantic_state, dict):
        semantic_state = {}

    fast_v = _safe_int(config.get("latest_fast_ready_version") or config.get("latest_ready_memory_version") or config.get("memory_version"), 0)
    visual_v = _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0)
    graph_v = _safe_int(config.get("latest_graph_ready_version") or graph_state.get("latest_graph_ready_version") or config.get("graph_version"), 0)
    semantic_v = _safe_int(config.get("latest_semantic_ready_version") or semantic_state.get("latest_semantic_ready_version") or config.get("semantic_version"), 0)

    lag = config.get("lag") if isinstance(config.get("lag"), dict) else {}
    readiness = config.get("readiness") if isinstance(config.get("readiness"), dict) else {}

    checkpoint_dir = session_dir / "worldmm" / "incremental" / "visual" / "checkpoints" / f"v{visual_v:06d}"
    visual_checkpoint_ready = (
        visual_v > 0
        and checkpoint_dir.exists()
        and (checkpoint_dir / "visual.faiss").exists()
        and (checkpoint_dir / "visual_embeddings.pkl").exists()
        and (checkpoint_dir / "visual_id_mapping.json").exists()
        and (checkpoint_dir / "visual_items.jsonl").exists()
    )
    visual_ready = _as_bool(config.get("visual_embedding_ready"), False) and visual_v > 0 and (
        visual_checkpoint_ready or (session_dir / str(config.get("visual_faiss_path") or "worldmm/visual/visual.faiss")).exists()
    )
    graph_ready = graph_v > 0 and (
        _as_bool(readiness.get("graph_ready"), False)
        or _safe_int(graph_state.get("latest_graph_ready_version"), 0) >= graph_v
        or _safe_int(graph_state.get("graph_version"), 0) >= graph_v
    )
    semantic_ready = semantic_v > 0 and (
        _as_bool(config.get("semantic_memory_ready"), False)
        or _as_bool(readiness.get("semantic_ready"), False)
        or _safe_int(semantic_state.get("latest_semantic_ready_version"), 0) >= semantic_v
        or _safe_int(semantic_state.get("semantic_version"), 0) >= semantic_v
    )
    fast_ready = fast_v > 0

    versions["session_id"] = session_dir.name
    fast_patch = {"latest_ready_version": fast_v, "building_version": None}
    if fast_v:
        fast_patch["active_query_version"] = max(_safe_int((versions.get("fast") or {}).get("active_query_version"), 0), fast_v)
    _merge_component(versions, "fast", fast_patch)
    _merge_component(versions, "episodic", {"version": fast_v, "ready": fast_ready})
    _merge_component(versions, "visual", {"version": visual_v, "ready": visual_ready, "lagging": bool(fast_v and visual_v < fast_v), "building_version": None if visual_ready else (fast_v or None)})
    _merge_component(versions, "graph", {"version": graph_v, "ready": graph_ready, "lagging": bool(fast_v and graph_v < fast_v), "building_version": None if graph_ready else (fast_v or None)})
    _merge_component(versions, "semantic", {"version": semantic_v, "ready": semantic_ready, "lagging": bool(fast_v and semantic_v < fast_v), "building_version": None if semantic_ready else (fast_v or None)})

    full_v = min([fast_v, visual_v, graph_v, semantic_v]) if fast_ready and visual_ready and graph_ready and semantic_ready else 0
    full_ready = bool(full_v and full_v >= fast_v)
    _merge_component(versions, "full", {"latest_full_ready_version": full_v, "long_term_full_ready": full_ready})
    if full_ready:
        versions["active_query_memory_version"] = _safe_int(versions.get("active_query_memory_version"), full_v) or full_v
        versions["active_query_updated_at"] = versions.get("active_query_updated_at") or utc_now_iso()

    versions["hipporag_cache_lagging"] = not _as_bool(config.get("hipporag_cache_ready"), False)
    if versions["hipporag_cache_lagging"]:
        versions["hipporag_cache_warning"] = "HippoRAG cache is lagging or missing; graph/semantic component readiness is tracked separately."
    versions["updated_at"] = utc_now_iso()
    write_json_atomic(versions_path, versions)

    if full_ready and (not _as_bool(config.get("long_term_full_ready"), False) or (isinstance(readiness, dict) and not _as_bool(readiness.get("long_term_full_ready"), False))):
        readiness = dict(readiness)
        readiness["long_term_full_ready"] = True
        readiness["visual_ready"] = visual_ready
        readiness["graph_ready"] = graph_ready
        readiness["semantic_ready"] = semantic_ready
        lag = dict(lag)
        lag["visual_lagging"] = False
        lag["graph_lagging"] = False
        lag["semantic_lagging"] = False
        config["readiness"] = readiness
        config["lag"] = lag
        config["long_term_full_ready"] = True
        config["updated_at"] = utc_now_iso()
        write_json_atomic(config_path, config)
    return versions
