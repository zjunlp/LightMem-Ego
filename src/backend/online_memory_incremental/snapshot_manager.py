from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from online_preprocess.io_utils import ensure_dir, read_json, relative_to_session, utc_now_iso, write_json_atomic
from online_memory_edit.lock import session_memory_mutation_lock


def pin_query_snapshot(session_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve one published generation; never read the next writer's roots.

    Incremental snapshots put components directly in the snapshot directory;
    memory-edit snapshots put them under ``em2mem``. Component lag is retained.
    """
    value = config.get("latest_snapshot_path")
    if not value:
        return dict(config)  # Legacy sessions without snapshot publication.
    session_dir = Path(session_dir)
    snapshot = session_dir / str(value)
    if not snapshot.resolve().is_relative_to(session_dir.resolve()):
        raise ValueError("published memory snapshot is outside its session")
    component_root = snapshot / "em2mem" if (snapshot / "em2mem" / "caption_root").is_dir() else snapshot
    if not (component_root / "caption_root").is_dir():
        raise FileNotFoundError(f"published memory snapshot is unavailable: {snapshot}")
    root = relative_to_session(component_root, session_dir)
    result = dict(config)
    result["lag"] = dict(config.get("lag") or {})
    result["readiness"] = dict(config.get("readiness") or {})
    args = dict(config.get("query_rag_args") or {})
    args.update(episodic_caption_root=f"{root}/caption_root",
                episodic_sidecar_root=f"{root}/sidecar_root", semantic_root=f"{root}/semantic_root",
                visual_root=f"{root}/embeddings", visual_evidence_file=f"{root}/visual_root/session_visual_evidence.json")
    result.update(query_rag_args=args, query_snapshot_path=str(value),
                  caption_root=f"{root}/caption_root", sidecar_root=f"{root}/sidecar_root",
                  semantic_root=f"{root}/semantic_root")
    files = dict(config.get("em2mem_files") or {})
    for key, value in files.items():
        if value and (key.startswith("caption_") or key == "visual_evidence"):
            folder = "caption_root" if key.startswith("caption_") else "visual_root"
            files[key] = f"{root}/{folder}/{Path(value).name}"
    result["em2mem_files"] = files
    meta = read_json(snapshot / "snapshot_meta.json", default={}) or {}
    components = meta.get("components") or {}
    version = meta.get("snapshot_version")
    if version is not None:
        advertised = config.get("latest_ready_memory_version") or config.get("memory_version")
        if advertised is not None and int(version) != int(advertised):
            raise RuntimeError("published memory version and snapshot generation disagree")
        result.update(latest_ready_memory_version=version, latest_fast_ready_version=version)
        for name in ("graph", "semantic", "visual"):
            if name in components:
                result[f"latest_{name}_ready_version"] = components[name]
                result[f"{name}_version"] = components[name]
        lag = dict(config.get("lag") or {})
        ready = dict(config.get("readiness") or {})
        for name in ("graph", "semantic", "visual"):
            behind = int(components.get(name) or 0) < int(version)
            if name == "visual" and not (config.get("visual_embedding_ready") or lag.get("visual_lagging")):
                behind = False  # An optional, disabled image index is not pending work.
            lag[f"{name}_lagging"] = behind
            lag[f"{name}_lag_versions"] = max(0, int(version) - int(components.get(name) or 0))
            ready[f"{name}_ready"] = not behind
        ready["visual_ready"] = bool(config.get("visual_embedding_ready"))
        result.update(lag=lag, readiness=ready)
    # Visual embeddings publish independently. Their checkpoint is also an
    # immutable generation, so they need not wait for another text snapshot.
    visual_root = component_root / "embeddings"
    checkpoint_value = config.get("visual_checkpoint_path")
    if checkpoint_value:
        checkpoint = session_dir / str(checkpoint_value)
        if not checkpoint.resolve().is_relative_to(session_dir.resolve()):
            raise ValueError("published visual checkpoint is outside its session")
        checkpoint_meta = read_json(checkpoint / "checkpoint_meta.json", default={}) or {}
        visual_version = checkpoint_meta.get("visual_version")
        if (checkpoint.resolve().is_relative_to(session_dir.resolve()) and checkpoint.is_dir()
                and visual_version is not None and (version is None or int(visual_version) <= int(version))):
            visual_root = checkpoint
            result.update(visual_version=visual_version, latest_visual_ready_version=visual_version)
            if version is not None:
                behind = int(visual_version) < int(version)
                result["lag"].update(visual_lagging=behind, visual_lag_versions=max(0, int(version)-int(visual_version)))
                result["readiness"]["visual_ready"] = bool(config.get("visual_embedding_ready"))
    visual_rel = relative_to_session(visual_root, session_dir)
    result["visual_root"] = visual_rel
    args["visual_root"] = visual_rel
    for key, filename in (("visual_items_path", "visual_items.jsonl"), ("visual_id_mapping_path", "visual_id_mapping.json"),
                          ("visual_embedding_path", "visual_embeddings.pkl"), ("visual_faiss_path", "visual.faiss")):
        if config.get(key):
            result[key] = f"{visual_rel}/{filename}"
    if config.get("visual_embedding_ready") and not (visual_root / "visual_embeddings.pkl").is_file():
        result["visual_embedding_ready"] = False
        result.setdefault("lag", {})["visual_lagging"] = True
        result.setdefault("readiness", {})["visual_ready"] = False
    result["long_term_full_ready"] = not any((result.get("lag") or {}).get(f"{name}_lagging") for name in ("graph", "semantic", "visual"))
    result.setdefault("readiness", {})["long_term_full_ready"] = result["long_term_full_ready"]
    return result


class SnapshotManager:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.root = self.session_dir / "em2mem" / "incremental" / "snapshots"

    def _copy_if_exists(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)

    def build_fast_snapshot(
        self,
        version: int,
        *,
        components: dict[str, Any],
        long_term_partial_ready: bool,
        long_term_full_ready: bool,
        semantic_lagging: bool,
        graph_lagging: bool,
    ) -> Path:
        with session_memory_mutation_lock(self.session_dir):
            return self._build_fast_snapshot_unlocked(
                version,
                components=components,
                long_term_partial_ready=long_term_partial_ready,
                long_term_full_ready=long_term_full_ready,
                semantic_lagging=semantic_lagging,
                graph_lagging=graph_lagging,
            )

    def _build_fast_snapshot_unlocked(
        self,
        version: int,
        *,
        components: dict[str, Any],
        long_term_partial_ready: bool,
        long_term_full_ready: bool,
        semantic_lagging: bool,
        graph_lagging: bool,
    ) -> Path:
        ensure_dir(self.root)
        # A component refresh can publish the same memory version twice. Each
        # publication has its own immutable directory; readers keep the old
        # generation until its pointer is replaced after successful validation.
        target = self.root / f"v{version:06d}_{uuid4().hex}"
        tmp = self.root / f".{target.name}.tmp"
        ensure_dir(tmp)
        try:
            return self._populate_snapshot(
                tmp, target, version, components=components,
                long_term_partial_ready=long_term_partial_ready,
                long_term_full_ready=long_term_full_ready,
                semantic_lagging=semantic_lagging, graph_lagging=graph_lagging,
            )
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def _populate_snapshot(
        self, tmp: Path, target: Path, version: int, *,
        components: dict[str, Any], long_term_partial_ready: bool,
        long_term_full_ready: bool, semantic_lagging: bool, graph_lagging: bool,
    ) -> Path:
        em2mem = self.session_dir / "em2mem"
        for name in ("caption_root", "sidecar_root", "semantic_root", "visual_root", "visual", "embeddings"):
            self._copy_if_exists(em2mem / name, tmp / name)
        self._copy_if_exists(em2mem / "memory_config.json", tmp / "memory_config.json")

        caption_30s = tmp / "caption_root" / f"{self.session_dir.name}_30sec.json"
        if not caption_30s.exists():
            raise FileNotFoundError(f"snapshot missing required 30sec caption file: {caption_30s}")

        meta = {
            "session_id": self.session_dir.name,
            "snapshot_version": version,
            "components": components,
            "long_term_partial_ready": bool(long_term_partial_ready),
            "long_term_full_ready": bool(long_term_full_ready),
            "semantic_lagging": bool(semantic_lagging),
            "graph_lagging": bool(graph_lagging),
            "created_at": utc_now_iso(),
        }
        write_json_atomic(tmp / "snapshot_meta.json", meta)
        tmp.replace(target)
        return target

    def update_config_snapshot_fields(self, config_path: Path, snapshot_path: Path, version: int) -> dict[str, Any]:
        config = read_json(config_path, default={})
        if not isinstance(config, dict):
            config = {}
        config.update(
            {
                "latest_snapshot_version": version,
                "latest_snapshot_path": relative_to_session(snapshot_path, self.session_dir),
                "last_snapshot_ready_at": utc_now_iso(),
            }
        )
        write_json_atomic(config_path, config)
        return config
