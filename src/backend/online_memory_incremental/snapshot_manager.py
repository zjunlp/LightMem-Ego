from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import ensure_dir, read_json, relative_to_session, utc_now_iso, write_json_atomic


class SnapshotManager:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.root = self.session_dir / "worldmm" / "incremental" / "snapshots"

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
        ensure_dir(self.root)
        target = self.root / f"v{version:06d}"
        tmp = self.root / f"v{version:06d}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(tmp)

        worldmm = self.session_dir / "worldmm"
        for name in ("caption_root", "sidecar_root", "semantic_root", "visual_root", "visual", "embeddings"):
            self._copy_if_exists(worldmm / name, tmp / name)
        self._copy_if_exists(worldmm / "memory_config.json", tmp / "memory_config.json")

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
