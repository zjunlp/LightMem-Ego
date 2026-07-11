from __future__ import annotations

from pathlib import Path
from typing import Any

from online_preprocess.task_queue import enqueue_visual_task


def enqueue_visual_append_task(
    *,
    project_root: Path,
    session_id: str,
    episode_ids: list[str],
    keyframe_paths: list[str],
    target_visual_version: int,
) -> Path | None:
    unique_paths = list(dict.fromkeys(str(p) for p in keyframe_paths if str(p).strip()))
    if not unique_paths:
        return None
    return enqueue_visual_task(
        project_root=project_root,
        session_id=session_id,
        force=False,
        backend=None,
        limit_items=None,
        task_type="visual_append",
        episode_ids=episode_ids,
        keyframe_paths=unique_paths,
        target_visual_version=target_visual_version,
    )
