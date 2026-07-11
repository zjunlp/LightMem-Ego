from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from online_preprocess.io_utils import read_json, utc_now_iso, write_json, write_json_atomic
from online_memory_incremental.component_versions import merge_component_versions, reconcile_component_versions
from online_visual.visual_index import append_visual_index, save_visual_index
from online_visual.visual_items import build_visual_items, read_visual_items, write_visual_items
from online_visual.vlm2vec_runtime import get_global_vlm2vec_runtime, l2_normalize


PROJECT_ROOT = Path(__file__).resolve().parent


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _update_status_outputs(session_dir: Path, visual_ready: bool) -> None:
    status_path = session_dir / "status.json"
    status = read_json(status_path, default={})
    if not isinstance(status, dict):
        status = {}
    outputs = dict(status.get("outputs", {}) or {})
    outputs["visual_embedding_ready"] = visual_ready
    status.update({
        "updated_at": utc_now_iso(),
        "outputs": outputs,
    })
    if visual_ready:
        if status.get("stage") != "memory_ready":
            status["stage"] = "visual_embedding_ready"
        status["visual_stage"] = "visual_embedding_ready"
        status["progress"] = max(int(status.get("progress") or 0), 100)
    write_json(status_path, status)


def _write_failure_config(memory_config_path: Path, error: str) -> None:
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        config = {}
    config["visual_embedding_ready"] = False
    config["visual_embedding_error"] = error
    config["visual_updated_at"] = utc_now_iso()
    write_json(memory_config_path, config)


def _write_visual_checkpoint(
    *,
    session_dir: Path,
    visual_root: Path,
    visual_version: int,
    index_backend: str,
    item_count: int,
) -> str:
    checkpoint_dir = session_dir / "worldmm" / "incremental" / "visual" / "checkpoints" / f"v{visual_version:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in ("visual.faiss", "visual_embeddings.pkl", "visual_id_mapping.json", "visual_items.jsonl"):
        src = visual_root / name
        if src.exists():
            dst = checkpoint_dir / name
            shutil.copy2(src, dst)
            copied[name] = dst.relative_to(session_dir).as_posix()
    write_json(
        checkpoint_dir / "checkpoint_meta.json",
        {
            "session_id": session_dir.name,
            "visual_version": visual_version,
            "index_backend": index_backend,
            "item_count": item_count,
            "files": copied,
            "created_at": utc_now_iso(),
        },
    )
    return checkpoint_dir.relative_to(session_dir).as_posix()


def build_visual_embeddings(
    session_id: str,
    sessions_root: Path,
    backend: str = "vlm2vec",
    force: bool = False,
    limit_items: int | None = None,
    batch_size: int = 8,
    normalize: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    session_dir = sessions_root / session_id
    memory_config_path = session_dir / "worldmm" / "memory_config.json"
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        config = {}
    configured_evidence = config.get("evidence_path") or config.get("mst_evidence_path")
    evidence_path = session_dir / str(configured_evidence) if configured_evidence else session_dir / "evidence" / "session_evidence.json"
    if not evidence_path.exists() and (session_dir / "evidence" / "mst_session_evidence.json").exists():
        evidence_path = session_dir / "evidence" / "mst_session_evidence.json"
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing evidence file for session {session_id}: {evidence_path}")
    if not memory_config_path.exists():
        raise FileNotFoundError(f"Missing worldmm/memory_config.json for session {session_id}")

    visual_root = session_dir / "worldmm" / "visual"
    items_path = visual_root / "visual_items.jsonl"
    mapping_path = visual_root / "visual_id_mapping.json"
    embeddings_path = visual_root / "visual_embeddings.pkl"
    faiss_path = visual_root / "visual.faiss"

    if dry_run:
        items = build_visual_items(session_dir=session_dir, limit_items=limit_items, evidence_path=evidence_path)
        if verbose:
            print(f"dry-run ok: session={session_id} visual_items={len(items)} visual_root={visual_root}")
        return visual_root

    if embeddings_path.exists() and faiss_path.exists() and mapping_path.exists() and not force:
        if verbose:
            print(f"visual index already exists: {visual_root}")
        return visual_root

    visual_root.mkdir(parents=True, exist_ok=True)
    items = build_visual_items(session_dir=session_dir, limit_items=limit_items, evidence_path=evidence_path)
    if not items:
        error = "No keyframes found in session_evidence.json"
        _write_failure_config(memory_config_path, error)
        raise RuntimeError(error)
    write_visual_items(items_path, items)

    image_paths = []
    valid_items = []
    for item in items:
        image_path = session_dir / str(item["image_path"])
        if image_path.exists():
            image_paths.append(str(image_path))
            valid_items.append(item)
        elif verbose:
            print(f"skip missing image: {image_path}")
    if not image_paths:
        error = "No existing keyframe image files found"
        _write_failure_config(memory_config_path, error)
        raise RuntimeError(error)

    try:
        runtime = get_global_vlm2vec_runtime(backend=backend, batch_size=batch_size, normalize=normalize)
        embeddings = runtime.encode_images(image_paths)
        embeddings = np.asarray(embeddings, dtype="float32")
        if normalize:
            embeddings = l2_normalize(embeddings)
        index_backend = save_visual_index(faiss_path, embeddings)
    except Exception as exc:
        _write_failure_config(memory_config_path, str(exc))
        raise

    mapping = {
        "index_backend": index_backend,
        "row_to_visual_id": {str(i): item["visual_id"] for i, item in enumerate(valid_items)},
        "visual_id_to_row": {item["visual_id"]: i for i, item in enumerate(valid_items)},
        "visual_id_to_item": {item["visual_id"]: item for item in valid_items},
    }
    write_json(mapping_path, mapping)

    payload = {
        "model": runtime.model_path,
        "backend": runtime.backend,
        "dim": int(embeddings.shape[1]),
        "normalized": bool(normalize),
        "items": [{"visual_id": item["visual_id"], "embedding_index": i} for i, item in enumerate(valid_items)],
        "embeddings": embeddings,
    }
    with embeddings_path.open("wb") as f:
        pickle.dump(payload, f)

    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        config = {}
    visual_version = int(config.get("visual_version") or 0) + 1
    checkpoint_path = _write_visual_checkpoint(
        session_dir=session_dir,
        visual_root=visual_root,
        visual_version=visual_version,
        index_backend=index_backend,
        item_count=len(valid_items),
    )
    config.update({
        "visual_embedding_ready": True,
        "visual_embedding_error": None,
        "visual_root": _relative(visual_root, session_dir),
        "visual_items_path": _relative(items_path, session_dir),
        "visual_id_mapping_path": _relative(mapping_path, session_dir),
        "visual_embedding_path": _relative(embeddings_path, session_dir),
        "visual_faiss_path": _relative(faiss_path, session_dir),
        "visual_embedding_model": runtime.model_path,
        "visual_embedding_backend": runtime.backend,
        "visual_index_backend": index_backend,
        "visual_checkpoint_path": checkpoint_path,
        "visual_embedding_dim": int(embeddings.shape[1]),
        "visual_item_count": len(valid_items),
        "visual_created_at": utc_now_iso(),
        "visual_version": visual_version,
        "latest_visual_ready_version": visual_version,
        "visual_lagging": False,
        "building_versions": {
            **(config.get("building_versions") if isinstance(config.get("building_versions"), dict) else {}),
            "visual": None,
        },
    })
    write_json_atomic(memory_config_path, config)
    merge_component_versions(
        session_dir,
        {"visual": {"version": visual_version, "ready": True, "lagging": False, "building_version": None}},
    )
    _update_status_outputs(session_dir, visual_ready=True)
    if verbose:
        print(f"visual index built: items={len(valid_items)} dim={embeddings.shape[1]} backend={index_backend}")
    return visual_root


def append_visual_embeddings(
    session_id: str,
    sessions_root: Path,
    keyframe_paths: list[str] | None = None,
    episode_ids: list[str] | None = None,
    target_visual_version: int | None = None,
    backend: str = "vlm2vec",
    batch_size: int = 8,
    normalize: bool = True,
    verbose: bool = False,
) -> Path:
    session_dir = sessions_root / session_id
    memory_config_path = session_dir / "worldmm" / "memory_config.json"
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        config = {}
    evidence_path = session_dir / str(config.get("evidence_path") or config.get("mst_evidence_path") or "evidence/mst_session_evidence.json")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing evidence file for visual append: {evidence_path}")

    visual_root = session_dir / "worldmm" / "visual"
    items_path = visual_root / "visual_items.jsonl"
    mapping_path = visual_root / "visual_id_mapping.json"
    embeddings_path = visual_root / "visual_embeddings.pkl"
    faiss_path = visual_root / "visual.faiss"
    append_log_path = session_dir / "worldmm" / "incremental" / "visual" / "visual_append_log.jsonl"
    memory_append_log_path = session_dir / "worldmm" / "incremental" / "memory_append_log.jsonl"
    visual_root.mkdir(parents=True, exist_ok=True)
    append_log_path.parent.mkdir(parents=True, exist_ok=True)

    existing_items = read_visual_items(items_path)
    existing_paths = {str(item.get("image_path")) for item in existing_items if item.get("image_path")}
    wanted_paths = {str(p) for p in (keyframe_paths or []) if str(p).strip()}
    candidates = build_visual_items(session_dir=session_dir, evidence_path=evidence_path)
    if wanted_paths:
        candidates = [item for item in candidates if str(item.get("image_path")) in wanted_paths]
    new_items = [item for item in candidates if str(item.get("image_path")) not in existing_paths]

    if not new_items:
        visual_version = int(target_visual_version or config.get("latest_visual_ready_version") or config.get("visual_version") or 0)
        lag = dict(config.get("lag") or {}) if isinstance(config.get("lag"), dict) else {}
        lag["visual_lagging"] = False
        lag["visual_lag_versions"] = 0
        readiness = dict(config.get("readiness") or {}) if isinstance(config.get("readiness"), dict) else {}
        readiness["visual_ready"] = bool(config.get("visual_embedding_ready"))
        readiness["long_term_full_ready"] = bool(readiness.get("visual_ready") and readiness.get("graph_ready") and readiness.get("semantic_ready"))
        config.update(
            {
                "visual_embedding_ready": bool(config.get("visual_embedding_ready")),
                "visual_version": visual_version,
                "latest_visual_ready_version": visual_version,
                "visual_lagging": False,
                "lag": lag,
                "readiness": readiness,
                "building_versions": {
                    **(config.get("building_versions") if isinstance(config.get("building_versions"), dict) else {}),
                    "visual": None,
                },
                "long_term_full_ready": bool(readiness.get("long_term_full_ready")),
                "visual_updated_at": utc_now_iso(),
            }
        )
        write_json_atomic(memory_config_path, config)
        merge_component_versions(
            session_dir,
            {"visual": {"version": visual_version, "ready": bool(config.get("visual_embedding_ready")), "lagging": False, "building_version": None}},
        )
        import json

        with append_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "session_id": session_id,
                "episode_ids": episode_ids or [],
                "target_visual_version": target_visual_version,
                "status": "skipped_no_new_images",
                "new_item_count": 0,
                "updated_at": utc_now_iso(),
            }, ensure_ascii=False) + "\n")
        _append_memory_visual_ready(memory_append_log_path, session_dir, episode_ids or [], visual_version)
        return visual_root

    image_paths = []
    valid_new_items = []
    for item in new_items:
        image_path = session_dir / str(item["image_path"])
        if image_path.exists():
            image_paths.append(str(image_path))
            valid_new_items.append(item)
        elif verbose:
            print(f"skip missing image: {image_path}")
    if not valid_new_items:
        raise RuntimeError("visual append found no existing new keyframe image files")

    runtime = get_global_vlm2vec_runtime(backend=backend, batch_size=batch_size, normalize=normalize)
    new_embeddings = runtime.encode_images(image_paths)
    new_embeddings = np.asarray(new_embeddings, dtype="float32")
    if normalize:
        new_embeddings = l2_normalize(new_embeddings)

    old_embeddings = np.zeros((0, new_embeddings.shape[1]), dtype="float32")
    old_payload: dict[str, Any] = {}
    if embeddings_path.exists():
        with embeddings_path.open("rb") as f:
            old_payload = pickle.load(f)
        old_embeddings = np.asarray(old_payload.get("embeddings"), dtype="float32")
        if old_embeddings.ndim != 2 or old_embeddings.shape[1] != new_embeddings.shape[1]:
            old_embeddings = np.zeros((0, new_embeddings.shape[1]), dtype="float32")

    all_items = existing_items + valid_new_items
    all_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
    index_backend = append_visual_index(faiss_path, old_embeddings, new_embeddings)
    write_visual_items(items_path, all_items)

    mapping = {
        "index_backend": index_backend,
        "row_to_visual_id": {str(i): item["visual_id"] for i, item in enumerate(all_items)},
        "visual_id_to_row": {item["visual_id"]: i for i, item in enumerate(all_items)},
        "visual_id_to_item": {item["visual_id"]: item for item in all_items},
    }
    write_json(mapping_path, mapping)

    payload = {
        "model": runtime.model_path,
        "backend": runtime.backend,
        "dim": int(all_embeddings.shape[1]),
        "normalized": bool(normalize),
        "items": [{"visual_id": item["visual_id"], "embedding_index": i} for i, item in enumerate(all_items)],
        "embeddings": all_embeddings,
    }
    with embeddings_path.open("wb") as f:
        pickle.dump(payload, f)

    visual_version = int(target_visual_version or (int(config.get("visual_version") or 0) + 1))
    checkpoint_path = _write_visual_checkpoint(
        session_dir=session_dir,
        visual_root=visual_root,
        visual_version=visual_version,
        index_backend=index_backend,
        item_count=len(all_items),
    )
    config.update(
        {
            "visual_embedding_ready": True,
            "visual_embedding_error": None,
            "visual_root": _relative(visual_root, session_dir),
            "visual_items_path": _relative(items_path, session_dir),
            "visual_id_mapping_path": _relative(mapping_path, session_dir),
            "visual_embedding_path": _relative(embeddings_path, session_dir),
            "visual_faiss_path": _relative(faiss_path, session_dir),
            "visual_embedding_model": runtime.model_path,
            "visual_embedding_backend": runtime.backend,
            "visual_index_backend": index_backend,
            "visual_checkpoint_path": checkpoint_path,
            "visual_embedding_dim": int(all_embeddings.shape[1]),
            "visual_item_count": len(all_items),
            "visual_updated_at": utc_now_iso(),
            "visual_version": visual_version,
            "latest_visual_ready_version": visual_version,
            "visual_lagging": False,
            "lag": {
                **(config.get("lag") if isinstance(config.get("lag"), dict) else {}),
                "visual_lagging": False,
                "visual_lag_versions": 0,
            },
            "readiness": {
                **(config.get("readiness") if isinstance(config.get("readiness"), dict) else {}),
                "visual_ready": True,
                "long_term_full_ready": bool(
                    (config.get("readiness") if isinstance(config.get("readiness"), dict) else {}).get("graph_ready", True)
                    and (config.get("readiness") if isinstance(config.get("readiness"), dict) else {}).get("semantic_ready", True)
                ),
            },
            "building_versions": {
                **(config.get("building_versions") if isinstance(config.get("building_versions"), dict) else {}),
                "visual": None,
            },
        }
    )
    write_json_atomic(memory_config_path, config)
    merge_component_versions(
        session_dir,
        {"visual": {"version": visual_version, "ready": True, "lagging": False, "building_version": None}},
    )
    reconcile_component_versions(session_dir)
    _update_status_outputs(session_dir, visual_ready=True)
    import json

    with append_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "session_id": session_id,
            "episode_ids": episode_ids or [],
            "target_visual_version": target_visual_version,
            "status": "appended",
            "new_item_count": len(valid_new_items),
            "total_item_count": len(all_items),
            "updated_at": utc_now_iso(),
        }, ensure_ascii=False) + "\n")
    _append_memory_visual_ready(memory_append_log_path, session_dir, episode_ids or [], visual_version)
    if verbose:
        print(f"visual append done: new_items={len(valid_new_items)} total={len(all_items)} version={visual_version}")
    return visual_root


def _append_memory_visual_ready(path: Path, session_dir: Path, episode_ids: list[str], visual_version: int) -> None:
    if not episode_ids:
        return
    episodes_path = session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json"
    episodes = read_json(episodes_path, default=[])
    if isinstance(episodes, dict):
        episodes = episodes.get("episodes") or []
    by_id = {str(ep.get("episode_id")): ep for ep in episodes if isinstance(ep, dict)}
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("a", encoding="utf-8") as f:
        for episode_id in episode_ids:
            ep = by_id.get(str(episode_id), {})
            f.write(json.dumps({
                "append_id": f"append_{episode_id}",
                "session_id": session_dir.name,
                "episode_id": episode_id,
                "segment_id": ep.get("segment_id"),
                "start_time": ep.get("start_time"),
                "end_time": ep.get("end_time"),
                "source": "visual_append",
                "source_micro_event_ids": ep.get("source_micro_event_ids") or [],
                "status": "fully_ready",
                "fast_memory_version": visual_version,
                "semantic_memory_version": visual_version,
                "graph_version": visual_version,
                "visual_version": visual_version,
                "created_at": ep.get("created_at") or utc_now_iso(),
                "updated_at": utc_now_iso(),
                "error": None,
            }, ensure_ascii=False) + "\n")
    try:
        from online_memory_incremental.append_log import MemoryAppendLog

        MemoryAppendLog(path).write_state(session_dir / "worldmm" / "incremental" / "append_state.json", session_dir.name)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build online visual embedding index from session keyframes.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default="online_sessions")
    parser.add_argument("--backend", default=None, choices=["vlm2vec", "mock"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = args.backend or "vlm2vec"
    build_visual_embeddings(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        backend=backend,
        force=args.force,
        limit_items=args.limit_items,
        batch_size=args.batch_size,
        normalize=args.normalize,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
