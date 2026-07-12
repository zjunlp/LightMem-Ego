from __future__ import annotations

from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json


def load_memory_config(session_id: str, sessions_root: Path = Path("online_sessions")) -> dict[str, Any]:
    path = sessions_root / session_id / "em2mem" / "memory_config.json"
    data = read_json(path, default={})
    return data if isinstance(data, dict) else {}


def is_memory_ready(session_id: str, sessions_root: Path = Path("online_sessions")) -> bool:
    return load_memory_config(session_id, sessions_root).get("status") == "memory_ready"
