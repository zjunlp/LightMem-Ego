from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


class OnlinePreprocessError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_remove(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    replace_attempted = False
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not tmp_path.exists():
            raise FileNotFoundError(f"atomic temp file was not created: {tmp_path}")
        try:
            replace_attempted = True
            os.replace(tmp_path, path)
        except Exception as exc:
            print(
                "[write_json_atomic] replace failed "
                f"target={path} tmp={tmp_path} tmp_exists={tmp_path.exists()} "
                f"target_parent_exists={path.parent.exists()} error={exc}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            raise
    except Exception:
        if tmp_path.exists() and not replace_attempted:
            # Keep failed replace temp files for diagnostics; remove only temp
            # files that failed before they could become a valid candidate.
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_tool_binary(tool_name: str, env_name: str) -> str:
    explicit = os.getenv(env_name)
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return str(explicit_path)
        found = shutil.which(explicit)
        if found:
            return found
        raise OnlinePreprocessError(
            f"{tool_name} executable was not found at {env_name}={explicit!r}. "
            f"Set {env_name} to a valid absolute path or add {tool_name} to PATH."
        )
    ffmpeg_home = os.getenv("FFMPEG_HOME")
    if ffmpeg_home:
        candidate = Path(ffmpeg_home) / tool_name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(tool_name)
    if found:
        return found
    raise OnlinePreprocessError(
        f"{tool_name} executable was not found. Set {env_name}=<absolute path> "
        f"or add {tool_name} to PATH."
    )


def ffmpeg_bin() -> str:
    return resolve_tool_binary("ffmpeg", "EM2MEM_FFMPEG_BIN")


def ffprobe_bin() -> str:
    return resolve_tool_binary("ffprobe", "EM2MEM_FFPROBE_BIN")


def relative_to_session(path: Path, session_dir: Path) -> str:
    return path.relative_to(session_dir).as_posix()


def run_command(
    cmd: Sequence[str],
    description: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        executable = str(cmd[0]) if cmd else "<empty command>"
        raise OnlinePreprocessError(
            f"{description} failed because executable was not found: {executable}. "
            "If this is ffmpeg/ffprobe, set EM2MEM_FFMPEG_BIN / EM2MEM_FFPROBE_BIN."
        ) from exc
    if result.returncode != 0:
        raise OnlinePreprocessError(
            f"{description} failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout.strip()}\n"
            f"STDERR:\n{result.stderr.strip()}"
        )
    return result


def write_status(
    session_dir: Path,
    session_id: str,
    status: str,
    stage: str,
    progress: int,
    error: str | None = None,
    outputs: dict[str, Any] | None = None,
) -> None:
    status_path = session_dir / "status.json"
    existing = read_json(status_path, default={})
    if not isinstance(existing, dict):
        existing = {}
    payload = {
        "session_id": session_id,
        "status": status,
        "stage": stage,
        "progress": progress,
        "error": error,
        "updated_at": utc_now_iso(),
    }
    existing_outputs = existing.get("outputs") if isinstance(existing.get("outputs"), dict) else {}
    if outputs is not None:
        payload["outputs"] = {**existing_outputs, **outputs}
    elif existing_outputs:
        payload["outputs"] = existing_outputs
    write_json(status_path, payload)
