"""Portable text-memory imports and exports; never restore client file paths."""
from __future__ import annotations

import fcntl
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic

MAX_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 1000
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-import")
_jobs: dict[tuple[str, str], Any] = {}
_jobs_lock = threading.Lock()


class MemoryTransferError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.details: dict[str, Any] = {}


def portable(value: Any, depth: int = 0) -> Any:
    """Export record data, excluding local resource pointers and credentials."""
    if depth > 40:
        raise MemoryTransferError(400, "invalid_import_format", "Memory nesting is too deep.")
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            if (name.endswith(("_path", "_paths", "_root", "_url")) or name in {
                "path", "url", "embedding", "embeddings", "token", "api_key", "password",
                "authorization", "credentials", "query_rag_args", "memory_config",
            }):
                continue
            result[key] = portable(item, depth + 1)
        return result
    if isinstance(value, list):
        return [portable(item, depth + 1) for item in value]
    return value


def parse_memory(data: bytes, filename: str, format: str = "auto") -> list[dict[str, Any]]:
    if len(data) > MAX_BYTES:
        raise MemoryTransferError(413, "memory_import_too_large", "The file must be 10 MiB or smaller.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".json", ".txt"}:
        raise MemoryTransferError(400, "unsupported_import_extension", "Only JSON and TXT are supported.")
    if format not in {"auto", "json", "txt"} or (format != "auto" and suffix != "." + format):
        raise MemoryTransferError(400, "invalid_import_format", "File format does not match its extension.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MemoryTransferError(400, "memory_import_invalid_utf8", "Use UTF-8 encoding.") from exc
    if suffix == ".txt":
        rows = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    else:
        try:
            raw = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite number")))
            if isinstance(raw, dict):
                if "record" in raw:
                    raw = [raw["record"]]
                elif "records" in raw:
                    raw = raw["records"]
                elif isinstance(raw.get("episodic"), dict):
                    raw = raw["episodic"].get("30sec", [])
                elif "memories" in raw:
                    raw = raw["memories"]
                else:
                    raw = [raw]
            rows = raw if isinstance(raw, list) else [raw]
        except (ValueError, RecursionError) as exc:
            raise MemoryTransferError(400, "memory_import_invalid_json", "Invalid JSON memory file.") from exc
    if len(rows) > MAX_RECORDS:
        raise MemoryTransferError(413, "memory_import_record_limit", "At most 1000 records can be imported.")
    records = []
    for row in rows:
        if isinstance(row, str):
            row = {"content": row}
        if not isinstance(row, dict):
            raise MemoryTransferError(400, "invalid_import_format", "Each memory must be text or an object.")
        row = portable(row)
        fields = {**(row.get("data") if isinstance(row.get("data"), dict) else {}), **row}
        content = next((fields[key].strip() for key in ("content", "text", "fine_caption", "caption", "semantic_summary", "visual_summary", "keyframe_caption", "segment_caption")
                        if isinstance(fields.get(key), str) and fields[key].strip()), "")
        if not content and fields.get("triple"):
            triple = fields["triple"]
            content = " ".join(map(str, triple)) if isinstance(triple, list) else str(triple)
        if not content and fields.get("head") and fields.get("tail"):
            content = " ".join(str(fields.get(key) or "") for key in ("head", "relation", "tail")).strip()
        if not content:
            if not fields:
                continue
            raise MemoryTransferError(400, "invalid_import_format", "A record has no memory text.")
        times = fields.get("event_time") or fields.get("eventTime") or {}
        if not isinstance(times, dict):
            times = {}
        start = fields.get("start_time", fields.get("start", times.get("start_seconds", times.get("startSeconds"))))
        end = fields.get("end_time", fields.get("end", times.get("end_seconds", times.get("endSeconds", start))))
        from online_time import event_time_seconds
        known = start is not None and fields.get("timestamp_known") is not False
        if known:
            try:
                start, end = event_time_seconds(start), event_time_seconds(end)
                if start is None or end is None or not all(math.isfinite(x) and x >= 0 for x in (start, end)) or end < start:
                    raise ValueError("Invalid time range")
            except (ValueError, TypeError, OverflowError) as exc:
                raise MemoryTransferError(400, "invalid_import_format", "Invalid memory time range.") from exc
        else:
            start = end = 0.0  # Indexing convention only, explicitly marked as untimed.
        for key in ("main_actions", "state_changes", "visual_objects", "entities", "visual_observations", "keyframe_captions"):
            if key in fields and not isinstance(fields[key], list):
                raise MemoryTransferError(400, "invalid_import_format", f"{key} must be a list.")
        record = {key: fields[key] for key in (
            "visual_summary", "main_actions", "state_changes", "visual_objects", "entities",
            "transcript", "scene", "visual_observations", "keyframe_captions", "confidence",
        ) if key in fields}
        # Imported observations are text annotations, never attached local images.
        from online_memory.em2mem_layout import seconds_to_hhmmssff
        date = str(fields.get("date") or "DAY1")
        record.update(text=content, caption=content, fine_caption=content,
                      start_time=seconds_to_hhmmssff(start), end_time=seconds_to_hhmmssff(end), start=start, end=end,
                      date=date if re.fullmatch(r"DAY\d+", date) else "DAY1", timestamp_known=known,
                      source_type="imported_memory", keyframe_paths=[],
                      imported_record=fields.get("imported_record", row))
        records.append(record)
    if not records:
        raise MemoryTransferError(400, "memory_import_empty", "No memory text was found.")
    return records


def build_imported_memory(session: Path, root: Path, project: Path, report) -> dict[str, Any]:
    """Run the normal memory builder and index initialization in isolation."""
    report("building_memory", 25)
    script = (
        "from pathlib import Path; from online_memory import build_online_em2mem_memory; "
        "import sys; build_online_em2mem_memory(sys.argv[1], Path(sys.argv[2]), "
        "source='imported_memory', force=True, skip_visual_embedding=True, skip_semantic=False)"
    )
    for command, stage, progress in (
        ([sys.executable, "-c", script, session.name, str(root)], "building_memory", 30),
        ([sys.executable, str(project / "memory_worker_subprocess.py"), "build_retrieval_artifacts",
          "--session-id", session.name, "--sessions-root", str(root)], "building_retrieval", 75),
    ):
        report(stage, progress)
        result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=3600)
        with (session / "import_build.log").open("a", encoding="utf-8") as log:
            log.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see the server import build log.")
    report("warming_query", 95)
    from online_query.warmup import warm_query_session
    warm = warm_query_session(session_id=session.name, sessions_root=root, wait_for_memory=False, reason="memory_import")
    if warm.get("status") != "ready":
        raise RuntimeError("Imported memory query initialization did not become ready.")
    return {"memory_version": read_json(session / "em2mem/memory_config.json", {}).get("memory_version")}


class MemoryImportService:
    def __init__(self, sessions_root: Path, project_root: Path, *, builder=None, schedule=True):
        self.root, self.project = Path(sessions_root).resolve(), Path(project_root).resolve()
        self.builder, self.schedule = builder or build_imported_memory, schedule

    def _session(self, task_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise MemoryTransferError(404, "memory_import_task_not_found", "Import task not found.")
        path = self.root / ("import_" + task_id)
        if path.is_symlink():
            raise MemoryTransferError(404, "memory_import_task_not_found", "Import task not found.")
        return path

    def receive(self, file: BinaryIO, filename: str, format="auto", title="") -> dict[str, Any]:
        data = file.read(MAX_BYTES + 1)
        records = parse_memory(data, filename, format)
        task_id = uuid4().hex
        session = self._session(task_id)
        session.mkdir(parents=True, exist_ok=False)
        now = utc_now_iso()
        for i, record in enumerate(records):
            record.update(doc_id=f"session_{session.name}_import_{i:06d}", segment_id=f"import_{i:06d}", session_id=session.name)
        state = dict(task_id=task_id, session_id=session.name, state="queued", stage="queued", progress=0,
                     file_name=Path(filename).name, record_count=len(records), query_ready=False,
                     created_at=now, updated_at=now, error=None)
        try:
            write_json_atomic(session / "captions/imported_30sec_captioned.json", records)
            write_json_atomic(session / "evidence/imported_memory_evidence.json", records)
            write_json_atomic(session / "metadata.json", {
                "session_id": session.name, "session_kind": "imported_memory", "interaction_mode": "text",
                "title": (str(title).strip() or Path(filename).stem)[:200], "title_revision": 0,
                "original_filename": Path(filename).name, "created_at": now, "updated_at": now,
            })
            self._write(session, state)
        except Exception:
            shutil.rmtree(session)
            raise
        self.ensure_scheduled(task_id)
        return {**state, "status": "queued"}

    def _write(self, session, state):
        state["updated_at"] = utc_now_iso()
        write_json_atomic(session / "memory_import.json", state)
        write_json_atomic(session / "status.json", {**state, "status": "ready" if state["state"] == "ready" else state["state"],
            "session_kind": "imported_memory", "interaction_mode": "text", "query_allowed": state.get("query_ready") is True})

    def status(self, task_id):
        session = self._session(task_id)
        state = read_json(session / "memory_import.json", {})
        if not state:
            raise MemoryTransferError(404, "memory_import_task_not_found", "Import task not found.")
        return {**state, "status": "ok"}

    def ensure_scheduled(self, task_id):
        if not self.schedule:
            return
        key = (str(self.root), task_id)
        with _jobs_lock:
            # Drop completed jobs so the process does not retain old futures.
            for old in [k for k, job in _jobs.items() if job.done()]:
                _jobs.pop(old, None)
            if key not in _jobs:
                _jobs[key] = _executor.submit(self.run, task_id)

    def run(self, task_id):
        session = self._session(task_id)
        with (session / "memory_import.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            state = read_json(session / "memory_import.json", {})
            if not state or state["state"] == "ready":
                return
            def report(stage, progress):
                state.update(state="processing", stage=stage, progress=progress, query_ready=False, error=None)
                self._write(session, state)
            try:
                report("normalizing", 10)
                result = self.builder(session, self.root, self.project, report)
                config = read_json(session / "em2mem/memory_config.json", {})
                if config.get("status") != "memory_ready" or config.get("retrieval_artifacts_ready") is not True:
                    raise RuntimeError("Published memory and retrieval indexes are required.")
                state.update(result, state="ready", stage="ready", progress=100, query_ready=True, error=None)
            except Exception as exc:
                state.update(state="failed", query_ready=False,
                             error={"code": "memory_import_failed", "message": str(exc)[:500]})
            self._write(session, state)

    def recover(self):
        for path in self.root.glob("import_*/memory_import.json"):
            state = read_json(path, {})
            if state.get("state") in {"queued", "processing"}:
                task_id = str(state.get("task_id") or "")
                try:
                    if self._session(task_id) == path.parent:
                        self.ensure_scheduled(task_id)
                except MemoryTransferError:
                    continue


def export_memory(sessions_root: Path, session_id: str, memory_id: str) -> dict[str, Any]:
    from online_memory_view.repository import LongTermMemoryRepository
    from online_memory_view.normalizer import MemoryNormalizer
    from online_memory_view.service import MemoryViewError
    if not session_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", session_id):
        raise MemoryTransferError(400, "invalid_session_id", "Invalid session ID.")
    repo = LongTermMemoryRepository(sessions_root, session_id)
    try:
        active = repo.resolve_active_root()
    except FileNotFoundError as exc:
        raise MemoryTransferError(404, "session_not_found", "Session not found.") from exc
    signature = repo.version_signature(active.config)
    repo.ensure_stable_components(signature)
    episodic = repo.load_episodic(active.config)
    normalizer = MemoryNormalizer(session_id)
    groups = [("episodic", scale, records) for scale, records in episodic.items()]
    groups += [("semantic", None, repo.load_semantic(active.config)),
               ("visual", None, repo.load_visual(active.config, episodic))]
    for kind, scale, records in groups:
        for record in records:
            normalized = normalizer.episodic(record, scale) if kind == "episodic" else getattr(normalizer, kind)(record)
            if normalized["id"] != memory_id:
                continue
            current = repo.version_signature(repo.reload_config(active))
            if current != signature:
                raise MemoryViewError(503, "memory_snapshot_changing", "Memory is being updated; retry export.")
            return {"schema_version": 1, "format": "lightmem-memory", "session_id": session_id,
                    "memory_version": signature.get("memory_version"), "memory_id": memory_id,
                    "memory_type": kind, "granularity": scale, "exported_at": utc_now_iso(),
                    "record": portable(record)}
    raise MemoryTransferError(404, "memory_not_found", "Memory record not found.")
