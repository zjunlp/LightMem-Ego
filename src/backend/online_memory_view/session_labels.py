"""Shared display labels for the existing session catalog and media views."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def session_labels(session_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    session_dir = Path(session_dir)
    if metadata is None:
        try:
            path = session_dir / "metadata.json"
            path.resolve().relative_to(session_dir.resolve())
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RuntimeError):
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    binding = metadata.get("demo_media")
    binding = binding if isinstance(binding, dict) else {}
    title = next((value.strip() for value in (
        metadata.get("title"), binding.get("title"), metadata.get("original_filename")
    ) if isinstance(value, str) and value.strip()), session_dir.name)
    revision = metadata.get("title_revision", 0)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        revision = 0
    updated_at = metadata.get("title_updated_at")
    return {
        "title": title,
        "title_revision": revision,
        "title_updated_at": updated_at if isinstance(updated_at, str) else None,
    }
