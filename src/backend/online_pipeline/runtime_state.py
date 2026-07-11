from __future__ import annotations

import os
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic
from online_preprocess.task_queue import ensure_queue_dirs, get_queue_dirs


WORKER_NAMES = (
    "stream",
    "preprocess",
    "evidence",
    "refine",
    "consolidation",
    "visual",
    "memory",
    "rokid_day_merge",
    "query",
    "live_ingest",
)

PIPELINE_MODES = {"mst", "legacy", "hybrid"}


def get_pipeline_mode() -> str:
    mode = os.getenv("WORLDMM_PIPELINE_MODE", "mst").strip().lower()
    return mode if mode in PIPELINE_MODES else "mst"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_worker_role(worker: str, mode: str) -> str:
    if worker == "evidence":
        if mode == "legacy":
            return "main"
        if mode == "hybrid":
            return "optional_legacy"
        return "legacy_optional"
    if worker in {"stream", "refine", "consolidation"} and mode == "legacy":
        return "optional"
    return "main"


def _default_worker_enabled(worker: str, mode: str) -> bool:
    if worker == "evidence":
        return mode in {"legacy", "hybrid"} or _env_bool("WORLDMM_ENABLE_LEGACY_EVIDENCE_WORKER", False)
    if worker == "stream":
        return mode != "legacy"
    if worker in {"refine", "consolidation"}:
        return mode != "legacy"
    return True


def pipeline_summary(mode: str | None = None) -> dict[str, Any]:
    mode = mode or get_pipeline_mode()
    if mode == "legacy":
        main_workers = ["preprocess", "evidence", "visual", "memory", "query"]
        optional_legacy_workers: list[str] = []
        main_path = "preprocess -> legacy evidence -> memory -> visual -> query"
    elif mode == "hybrid":
        main_workers = ["stream", "live_ingest", "preprocess", "refine", "consolidation", "visual", "memory", "query"]
        optional_legacy_workers = ["evidence"]
        main_path = "stream/live_ingest/preprocess -> M_cur/M_st -> MST refine -> MST consolidation -> memory -> visual -> query"
    else:
        main_workers = ["stream", "live_ingest", "preprocess", "refine", "consolidation", "visual", "memory", "query"]
        optional_legacy_workers = ["evidence"]
        main_path = "stream/live_ingest/preprocess -> M_cur/M_st -> MST refine -> MST consolidation -> memory -> visual -> query"
    return {
        "pipeline_mode": mode,
        "main_path": main_path,
        "main_workers": main_workers,
        "optional_legacy_workers": optional_legacy_workers,
        "legacy_evidence_worker_required": mode == "legacy",
    }


def _as_int(value: Any, default: int = 0) -> int:
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


def _as_span(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [round(float(value[0]), 3), round(float(value[1]), 3)]
        except Exception:
            pass
    return [0.0, 0.0]


def runtime_dir(project_root: Path) -> Path:
    return Path(project_root) / "runtime" / "workers"


def _parse_utc_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _heartbeat_age_seconds(value: Any) -> float | None:
    dt = _parse_utc_iso(value)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def _pid_alive(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except Exception:
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _read_pid_file(root: Path, worker: str) -> int | None:
    try:
        text = (root / f"{worker}.pid").read_text(encoding="utf-8").strip()
        pid = int(text)
        return pid if pid > 0 else None
    except Exception:
        return None


def _worker_runtime_paths(root: Path, worker: str) -> list[Path]:
    paths: list[Path] = []
    primary = root / f"{worker}.json"
    if primary.exists():
        paths.append(primary)
    paths.extend(sorted(root.glob(f"{worker}_*.json")))
    return paths


def _runtime_sort_key(payload: dict[str, Any]) -> tuple[int, float]:
    alive = 1 if _as_bool(payload.get("pid_alive")) else 0
    heartbeat = _parse_utc_iso(payload.get("last_heartbeat"))
    timestamp = heartbeat.timestamp() if heartbeat is not None else 0.0
    return alive, timestamp


def _aggregate_worker_payload(root: Path, worker: str, payloads: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not payloads:
        payload = {
            "worker": worker,
            "runtime_file_exists": False,
            "role": _default_worker_role(worker, mode),
            "enabled_by_default": _default_worker_enabled(worker, mode),
            "pipeline_mode": mode,
            "instance_count": 0,
            "alive_instance_count": 0,
            "instances": [],
        }
        if worker == "evidence" and mode == "mst":
            payload.setdefault("status", "disabled")
            payload.setdefault("reason", "MST-first pipeline uses refined micro-events to build 30s episodic evidence")
        return _apply_worker_liveness(payload, root, worker)

    primary = max(payloads, key=_runtime_sort_key)
    instances = sorted(payloads, key=lambda item: str(item.get("worker_instance") or ""))
    alive_count = sum(1 for item in instances if item.get("pid_alive"))
    aggregate = dict(primary)
    aggregate["worker"] = worker
    aggregate["pipeline_mode"] = mode
    aggregate["runtime_file_exists"] = True
    aggregate["instance_count"] = len(instances)
    aggregate["alive_instance_count"] = alive_count
    aggregate["instances"] = [
        {
            "worker_instance": item.get("worker_instance"),
            "status": item.get("status"),
            "pid": item.get("pid"),
            "pid_alive": bool(item.get("pid_alive")),
            "last_heartbeat": item.get("last_heartbeat"),
            "heartbeat_age_seconds": item.get("heartbeat_age_seconds"),
            "last_task_id": item.get("last_task_id"),
            "last_error": item.get("last_error"),
            "queue_pending": item.get("queue_pending"),
        }
        for item in instances
    ]
    aggregate["pid_alive"] = alive_count > 0
    aggregate["queue_pending"] = max(int(item.get("queue_pending") or 0) for item in instances)
    aggregate["last_error"] = next((item.get("last_error") for item in instances if item.get("last_error")), None)
    if any(item.get("status") == "busy" and item.get("pid_alive") for item in instances):
        aggregate["status"] = "busy"
    elif any(item.get("status") == "ready" and item.get("pid_alive") for item in instances):
        aggregate["status"] = "ready"
    elif any(item.get("status") == "failed" for item in instances):
        aggregate["status"] = "failed"
    elif alive_count == 0:
        aggregate["status"] = primary.get("status") or "stale"
    if len(instances) > 1:
        aggregate["reason"] = f"{alive_count}/{len(instances)} instances alive"
    return aggregate


def _worker_stale_threshold_seconds() -> float:
    try:
        return float(os.getenv("WORLDMM_WORKER_STALE_SECONDS", "60"))
    except Exception:
        return 60.0


def _apply_worker_liveness(payload: dict[str, Any], root: Path, worker: str) -> dict[str, Any]:
    """Overlay process/heartbeat liveness on top of the last written runtime JSON.

    runtime/workers/*.json is a heartbeat snapshot, not the source of truth for
    whether the worker process is still alive.  Without this check a killed
    worker can keep showing status=ready forever.
    """
    enabled = bool(payload.get("enabled_by_default"))
    runtime_file_exists = bool(payload.get("runtime_file_exists", True))
    if not runtime_file_exists:
        payload["pid_alive"] = False
        payload["heartbeat_age_seconds"] = None
        payload["stale"] = False
        payload["status"] = "missing" if enabled else "disabled"
        payload["reason"] = "worker runtime file is missing"
        return payload
    status = str(payload.get("status") or "").strip().lower()
    if not status:
        status = "disabled" if not enabled else "unknown"
        payload["status"] = status

    if status in {"disabled", "stopped"}:
        payload["pid_alive"] = False
        payload["stale"] = False
        return payload
    if status in {"error", "failed"}:
        payload["status"] = "failed"
        age = _heartbeat_age_seconds(payload.get("last_heartbeat"))
        payload["pid_alive"] = _pid_alive(payload.get("pid"))
        payload["heartbeat_age_seconds"] = round(age, 3) if age is not None else None
        payload["heartbeat_stale_threshold_seconds"] = _worker_stale_threshold_seconds()
        payload["stale"] = bool(age is None or age > _worker_stale_threshold_seconds())
        return payload

    if payload.get("pid") is None:
        pid_from_file = _read_pid_file(root, worker)
        if pid_from_file is not None:
            payload["pid"] = pid_from_file

    age = _heartbeat_age_seconds(payload.get("last_heartbeat"))
    alive = _pid_alive(payload.get("pid"))
    threshold = _worker_stale_threshold_seconds()
    heartbeat_stale = age is None or age > threshold

    payload["pid_alive"] = alive
    payload["heartbeat_age_seconds"] = round(age, 3) if age is not None else None
    payload["heartbeat_stale_threshold_seconds"] = threshold
    payload["stale"] = bool((not alive) or heartbeat_stale)

    if not alive:
        payload["status"] = "stale"
        payload["reason"] = "worker pid is missing or no longer alive; runtime JSON is stale"
    elif heartbeat_stale:
        payload["status"] = "stale"
        payload["reason"] = "worker heartbeat exceeded stale threshold"
    return payload


def write_worker_runtime(
    project_root: Path,
    worker: str,
    *,
    status: str,
    role: str | None = None,
    pipeline_mode: str | None = None,
    enabled_by_default: bool | None = None,
    model_name: str | None = None,
    model_path: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    model_loaded: bool | None = None,
    client_loaded: bool | None = None,
    warmup_done: bool | None = None,
    queue_pending: int | None = None,
    last_task_id: str | None = None,
    last_error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    mode = pipeline_mode or get_pipeline_mode()
    role = role or _default_worker_role(worker, mode)
    if enabled_by_default is None:
        enabled_by_default = _default_worker_enabled(worker, mode)
    model_or_backend = model_name or backend or model_path
    payload: dict[str, Any] = {
        "worker": worker,
        "role": role,
        "pipeline_mode": mode,
        "enabled_by_default": bool(enabled_by_default),
        "model_or_backend": model_or_backend,
        "status": status,
        "pid": os.getpid(),
        "backend": backend,
        "model_name": model_name,
        "model_path": model_path,
        "device": device,
        "model_loaded": bool(model_loaded) if model_loaded is not None else False,
        "client_loaded": bool(client_loaded) if client_loaded is not None else False,
        "warmup_done": bool(warmup_done) if warmup_done is not None else False,
        "queue_pending": int(queue_pending or 0),
        "last_heartbeat": utc_now_iso(),
        "last_task_id": last_task_id,
        "last_error": last_error,
    }
    if extra:
        payload.update(extra)
    path = runtime_dir(project_root) / f"{worker}.json"
    write_json_atomic(path, payload)
    return path


def _runtime_heartbeat_interval_seconds(env_name: str | None = None) -> float:
    if env_name:
        value = os.getenv(env_name)
        if value not in {None, ""}:
            try:
                return max(1.0, float(value))
            except Exception:
                pass
    stale_seconds = max(3.0, _worker_stale_threshold_seconds())
    return max(1.0, min(15.0, stale_seconds / 3.0))


class WorkerTaskHeartbeat:
    """Refresh worker runtime while a long task is in progress."""

    def __init__(
        self,
        project_root: Path,
        worker: str,
        *,
        task: dict[str, Any] | None = None,
        claimed_path: Path | None = None,
        status: str = "busy",
        backend: str | None = None,
        model_name: str | None = None,
        model_path: str | None = None,
        device: str | None = None,
        model_loaded: bool | None = None,
        client_loaded: bool | None = None,
        warmup_done: bool | None = True,
        queue_pending: int | Callable[[], int] | None = None,
        last_error: str | None = None,
        extra: dict[str, Any] | None = None,
        extra_fn: Callable[[], dict[str, Any]] | None = None,
        interval_env: str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.worker = worker
        self.task = task or {}
        self.claimed_path = Path(claimed_path) if claimed_path is not None else None
        self.status = status
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model_loaded = model_loaded
        self.client_loaded = client_loaded
        self.warmup_done = warmup_done
        self.queue_pending = queue_pending
        self.last_error = last_error
        self.extra = dict(extra or {})
        self.extra_fn = extra_fn
        self.interval_seconds = _runtime_heartbeat_interval_seconds(interval_env)
        self.task_id = str(self.task.get("task_id") or (self.claimed_path.stem if self.claimed_path else f"{worker}_task"))
        self.session_id = str(
            self.task.get("session_id")
            or self.task.get("child_session_id")
            or self.task.get("parent_session_id")
            or ""
        )
        self.task_type = str(self.task.get("task_type") or worker)
        self.started_at = str(self.task.get("claimed_at") or self.task.get("updated_at") or utc_now_iso())
        self.started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{worker}-heartbeat-{self.task_id}", daemon=True)

    def __enter__(self) -> "WorkerTaskHeartbeat":
        self._write()
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _queue_pending(self) -> int | None:
        if self.queue_pending is None:
            return None
        if callable(self.queue_pending):
            try:
                return int(self.queue_pending())
            except Exception:
                return None
        try:
            return int(self.queue_pending)
        except Exception:
            return None

    def _current_task(self) -> dict[str, Any]:
        current: dict[str, Any] = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "started_at": self.started_at,
            "running_seconds": round(max(0.0, time.monotonic() - self.started_monotonic), 3),
        }
        if self.session_id:
            current["session_id"] = self.session_id
        if self.claimed_path is not None:
            current["task_path"] = str(self.claimed_path)
        for key in (
            "reason",
            "backend",
            "source",
            "update_mode",
            "force",
            "limit_events",
            "limit_windows",
            "limit_segments",
            "event_id",
            "window_start",
            "window_end",
            "chunk_id",
            "chunk_index",
            "upload_chunk_id",
            "upload_chunk_index",
            "parent_session_id",
            "child_session_id",
            "day_label",
            "day_index",
        ):
            if key in self.task:
                current[key] = self.task.get(key)
        return current

    def _extra(self) -> dict[str, Any]:
        payload = dict(self.extra)
        if self.extra_fn is not None:
            try:
                payload.update(self.extra_fn())
            except Exception as exc:
                payload["heartbeat_extra_error"] = str(exc)
        current_task = self._current_task()
        payload.update(
            {
                "current_task": current_task,
                "current_task_id": self.task_id,
                "current_task_type": self.task_type,
            }
        )
        if self.session_id:
            payload["current_session_id"] = self.session_id
        if self.task.get("reason") is not None:
            payload["current_task_reason"] = self.task.get("reason")
        return payload

    def _write(self) -> None:
        write_worker_runtime(
            self.project_root,
            self.worker,
            status=self.status,
            backend=self.backend,
            model_name=self.model_name,
            model_path=self.model_path,
            device=self.device,
            model_loaded=self.model_loaded,
            client_loaded=self.client_loaded,
            warmup_done=self.warmup_done,
            queue_pending=self._queue_pending(),
            last_task_id=self.task_id,
            last_error=self.last_error,
            extra=self._extra(),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write()
            except Exception as exc:
                print(f"[{self.worker}_worker] heartbeat update failed task={self.task_id}: {exc}", flush=True)


def queue_counts(project_root: Path) -> dict[str, int]:
    dirs = ensure_queue_dirs(Path(project_root))
    counts: dict[str, int] = {}
    for key, path in dirs.items():
        counts[key] = len(list(path.glob("*.json"))) if path.exists() else 0
    return counts


def collect_worker_runtime(project_root: Path) -> dict[str, Any]:
    root = runtime_dir(project_root)
    mode = get_pipeline_mode()
    workers: dict[str, Any] = {}
    for name in WORKER_NAMES:
        payloads: list[dict[str, Any]] = []
        for runtime_path in _worker_runtime_paths(root, name):
            payload = read_json(runtime_path, default={})
            if not isinstance(payload, dict):
                payload = {}
            payload["runtime_file_exists"] = True
            payload["worker"] = name
            payload["worker_instance"] = runtime_path.stem
            stale_mode = payload.get("pipeline_mode") not in {None, mode}
            if stale_mode:
                payload["role"] = _default_worker_role(name, mode)
                payload["enabled_by_default"] = _default_worker_enabled(name, mode)
            else:
                payload.setdefault("role", _default_worker_role(name, mode))
                payload.setdefault("enabled_by_default", _default_worker_enabled(name, mode))
            payload["pipeline_mode"] = mode
            payloads.append(_apply_worker_liveness(payload, root, runtime_path.stem))
        workers[name] = _aggregate_worker_payload(root, name, payloads, mode)
    return workers


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return read_json(path, default=default)
    except Exception:
        return default


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _count_cross_chunk_diff_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict) and payload.get("is_cross_chunk"):
                    count += 1
    except Exception:
        return 0
    return count


def _count_events_with_flag(path: Path, flag: str) -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict) and payload.get(flag):
                    count += 1
    except Exception:
        return 0
    return count


def _ready_windows_count(session_dir: Path) -> int:
    windows = _safe_json(session_dir / "short_term" / "refine" / "refined_ready_windows.json", [])
    if not isinstance(windows, list):
        return 0
    return sum(
        1
        for item in windows
        if isinstance(item, dict)
        and item.get("ready_for_30s_episodic")
        and item.get("is_closed_window")
    )


def _pending_ready_windows_count(session_dir: Path) -> int:
    windows = _safe_json(session_dir / "short_term" / "refine" / "refined_ready_windows.json", [])
    consolidation = _safe_json(session_dir / "short_term" / "consolidation_state.json", {})
    consolidated = set()
    if isinstance(consolidation, dict):
        consolidated = set((consolidation.get("window_to_episode") or {}).keys())
    if not isinstance(windows, list):
        return 0
    return sum(
        1
        for item in windows
        if isinstance(item, dict)
        and item.get("ready_for_30s_episodic")
        and item.get("is_closed_window")
        and str(item.get("window_id") or "") not in consolidated
    )


def _derive_memory_version(config: dict[str, Any]) -> int | None:
    for key in ("latest_ready_memory_version", "memory_version", "version"):
        value = config.get(key)
        if value is not None:
            parsed = _as_int(value, 0)
            if parsed > 0:
                return parsed
    if config.get("status") == "memory_ready":
        return 1
    return None


def build_session_pipeline_state(session_dir: Path) -> dict[str, Any]:
    session_dir = Path(session_dir)
    session_id = session_dir.name
    mode = get_pipeline_mode()
    current_state = _safe_json(session_dir / "current" / "current_state.json", {})
    mst_state = _safe_json(session_dir / "short_term" / "mst_state.json", {})
    archive_state = _safe_json(session_dir / "short_term" / "archive" / "archive_state.json", {})
    refine_state = _safe_json(session_dir / "short_term" / "refine" / "refine_state.json", {})
    episodic_state = _safe_json(session_dir / "worldmm" / "mst_episodic" / "mst_episodic_state.json", {})
    consolidation_state = _safe_json(session_dir / "short_term" / "consolidation_state.json", {})
    memory_config = _safe_json(session_dir / "worldmm" / "memory_config.json", {})
    component_versions = _safe_json(session_dir / "worldmm" / "incremental" / "component_versions.json", {})
    append_state = _safe_json(session_dir / "worldmm" / "incremental" / "append_state.json", {})
    stream_state = _safe_json(session_dir / "stream" / "stream_state.json", {})
    event_state = _safe_json(session_dir / "stream" / "event_state.json", {})
    partial_transcript_state = _safe_json(session_dir / "stream" / "transcript" / "partial_transcript_state.json", {})
    transcript_dirty_state = _safe_json(session_dir / "short_term" / "transcript_dirty_windows.json", {})

    if not isinstance(current_state, dict):
        current_state = {}
    if not isinstance(mst_state, dict):
        mst_state = {}
    if not isinstance(archive_state, dict):
        archive_state = {}
    if not isinstance(refine_state, dict):
        refine_state = {}
    if not isinstance(episodic_state, dict):
        episodic_state = {}
    if not isinstance(consolidation_state, dict):
        consolidation_state = {}
    if not isinstance(memory_config, dict):
        memory_config = {}
    if not isinstance(component_versions, dict):
        component_versions = {}
    if not isinstance(append_state, dict):
        append_state = {}
    if not isinstance(stream_state, dict):
        stream_state = {}
    if not isinstance(event_state, dict):
        event_state = {}
    if not isinstance(partial_transcript_state, dict):
        partial_transcript_state = {}
    if not isinstance(transcript_dirty_state, dict):
        transcript_dirty_state = {}

    legacy_caption_path = session_dir / "captions" / "session_30sec_captioned.json"
    legacy_evidence_path = session_dir / "evidence" / "session_evidence.json"
    mst_caption_path = session_dir / "captions" / "mst_session_30sec_captioned.json"
    mst_evidence_path = session_dir / "evidence" / "mst_session_evidence.json"
    legacy_evidence_available = legacy_caption_path.exists() and legacy_evidence_path.exists()
    mst_episodic_ready = mst_caption_path.exists() and mst_evidence_path.exists() and (
        session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json"
    ).exists()
    active_30s_source = (
        memory_config.get("active_30s_source")
        or memory_config.get("worldmm_30s_input_source")
        or ("mst_session_30sec_captioned" if memory_config.get("episodic_source") == "mst_micro_events" else None)
        or ("session_30sec_captioned" if memory_config.get("episodic_source") in {"legacy_evidence", "online_evidence"} else None)
    )
    legacy_evidence_used = active_30s_source == "session_30sec_captioned"

    latest_ready = memory_config.get("latest_ready_memory_version")
    if latest_ready is None:
        latest_ready = _derive_memory_version(memory_config)
    building = memory_config.get("building_memory_version")
    active_query = memory_config.get("active_query_memory_version") or latest_ready
    build_state = memory_config.get("memory_build_state") or ("ready" if latest_ready else "idle")
    if memory_config.get("status") == "memory_ready" and latest_ready and not building and build_state in {"building", "failed"}:
        build_state = "ready_with_warnings" if memory_config.get("last_build_error") else "ready"
        memory_config["memory_build_state"] = build_state
        memory_config["building_memory_version"] = None
        write_json_atomic(session_dir / "worldmm" / "memory_config.json", memory_config)

    visual_ready = _as_bool(memory_config.get("visual_embedding_ready"), False)
    semantic_ready = _as_bool(memory_config.get("semantic_memory_ready"), False)
    long_term_partial_ready = bool(latest_ready) and (
        _as_bool(memory_config.get("episodic_index_ready"), True)
        or _as_bool(memory_config.get("hipporag_cache_ready"), True)
    )
    long_term_full_ready = bool(latest_ready) and semantic_ready and visual_ready
    lag_state = memory_config.get("lag") if isinstance(memory_config.get("lag"), dict) else {}
    latest_fast = memory_config.get("latest_fast_ready_version") or latest_ready
    latest_visual = memory_config.get("latest_visual_ready_version") or memory_config.get("visual_version")
    latest_graph = memory_config.get("latest_graph_ready_version") or memory_config.get("graph_version")
    latest_semantic = memory_config.get("latest_semantic_ready_version") or memory_config.get("semantic_version")

    episode_file_count = 0
    episodes_path = session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json"
    episodes_payload = _safe_json(episodes_path, [])
    if isinstance(episodes_payload, list):
        episode_file_count = len(episodes_payload)
    elif isinstance(episodes_payload, dict):
        episode_file_count = len(episodes_payload.get("episodes") or [])

    state = {
        "session_id": session_id,
        "pipeline_mode": memory_config.get("pipeline_mode") or mode,
        "stream": {
            "status": stream_state.get("status", "not_started"),
            "stream_id": stream_state.get("stream_id"),
            "received_chunk_count": len(stream_state.get("upload_chunks", stream_state.get("received_chunks", [])) or []),
            "received_upload_chunk_count": len(stream_state.get("upload_chunks", stream_state.get("received_chunks", [])) or []),
            "generated_processing_chunk_count": len(stream_state.get("processing_chunks", []) or []),
            "processed_chunk_count": sum(
                1
                for item in stream_state.get("processing_chunks", []) or []
                if isinstance(item, dict) and str(item.get("status") or "") == "processed"
            )
            if stream_state.get("processing_chunks")
            else len(stream_state.get("processed_chunks", []) or []),
            "processed_processing_chunk_count": sum(
                1
                for item in stream_state.get("processing_chunks", []) or []
                if isinstance(item, dict) and str(item.get("status") or "") == "processed"
            ),
            "next_expected_chunk_index": _as_int(stream_state.get("next_expected_chunk_index"), 0),
            "next_expected_upload_chunk_index": _as_int(stream_state.get("next_expected_upload_chunk_index"), 0),
            "next_expected_proc_index": _as_int(stream_state.get("next_expected_proc_index"), 0),
            "last_processed_chunk_index": _as_int(stream_state.get("last_processed_chunk_index"), -1),
            "last_processed_proc_index": _as_int(stream_state.get("last_processed_proc_index"), -1),
            "missing_chunks": stream_state.get("missing_chunks", []),
            "duplicate_chunks": stream_state.get("duplicate_chunks", []),
            "has_open_event": isinstance(event_state.get("open_event"), dict) and bool(event_state.get("open_event")),
            "open_event_start": (event_state.get("open_event") or {}).get("start_time") if isinstance(event_state.get("open_event"), dict) else None,
            "open_event_end": (event_state.get("open_event") or {}).get("last_update_time") if isinstance(event_state.get("open_event"), dict) else None,
            "last_candidate_frame_time": (event_state.get("last_candidate_frame") or {}).get("timestamp") if isinstance(event_state.get("last_candidate_frame"), dict) else None,
            "diff_record_count": _count_lines(session_dir / "stream" / "diff_records.jsonl"),
            "cross_chunk_diff_count": _count_cross_chunk_diff_records(session_dir / "stream" / "diff_records.jsonl"),
            "latency": stream_state.get("latency", {}) if isinstance(stream_state.get("latency"), dict) else {},
        },
        "stream_asr": {
            "enabled": _as_bool(os.getenv("WORLDMM_STREAM_ASR_ENABLED"), True),
            "backend": os.getenv("WORLDMM_STREAM_ASR_BACKEND", "whisperx"),
            "queue_pending": queue_counts(session_dir.parents[1]).get("stream_asr_queued", 0) if len(session_dir.parents) > 1 else 0,
            "partial_transcript_version": _as_int(partial_transcript_state.get("partial_transcript_version"), 0),
            "partial_transcript_segment_count": _as_int(partial_transcript_state.get("segment_count"), _count_lines(session_dir / "stream" / "transcript" / "partial_transcript.jsonl")),
            "time_span": _as_span(partial_transcript_state.get("time_span")),
            "processed_asr_chunks": partial_transcript_state.get("processed_asr_chunks", []),
            "failed_asr_chunks": partial_transcript_state.get("failed_asr_chunks", []),
            "last_asr_chunk_index": partial_transcript_state.get("last_asr_chunk_index"),
        },
        "transcript_backfill": {
            "last_backfill_at": partial_transcript_state.get("updated_at"),
            "pending_refine_due_to_transcript": _count_events_with_flag(session_dir / "short_term" / "archive" / "micro_events_all.jsonl", "needs_refine"),
            "dirty_windows_due_to_transcript": sum(
                1
                for item in transcript_dirty_state.get("windows", []) or []
                if isinstance(item, dict) and str(item.get("status") or "") in {"dirty", "queued"}
            ),
            "dirty_windows": transcript_dirty_state.get("windows", []),
        },
        "source": {
            "active_30s_source": active_30s_source,
            "episodic_source": memory_config.get("episodic_source"),
            "mst_episodic_ready": _as_bool(memory_config.get("mst_episodic_ready"), mst_episodic_ready),
            "mst_captioned_30sec_path": memory_config.get("mst_captioned_30sec_path") or ("captions/mst_session_30sec_captioned.json" if mst_caption_path.exists() else None),
            "mst_evidence_path": memory_config.get("mst_evidence_path") or ("evidence/mst_session_evidence.json" if mst_evidence_path.exists() else None),
            "legacy_evidence_available": _as_bool(memory_config.get("legacy_evidence_available"), legacy_evidence_available),
            "legacy_evidence_path": "evidence/session_evidence.json" if legacy_evidence_path.exists() else None,
            "legacy_captioned_30sec_path": "captions/session_30sec_captioned.json" if legacy_caption_path.exists() else None,
            "legacy_evidence_used": bool(memory_config.get("legacy_evidence_used") or memory_config.get("legacy_evidence_fallback_used") or legacy_evidence_used),
            "legacy_evidence_fallback_used": bool(memory_config.get("legacy_evidence_fallback_used", False)),
        },
        "current": {
            "ready": _as_bool(current_state.get("mcur_ready"), False),
            "version": _as_int(current_state.get("mcur_version"), 0),
            "time_span": _as_span(
                current_state.get("time_span")
                or current_state.get("current_time_span")
                or [current_state.get("window_start_time"), current_state.get("window_end_time")]
            ),
        },
        "short_term": {
            "ready": _as_bool(mst_state.get("short_term_ready"), False),
            "mst_version": _as_int(mst_state.get("mst_version"), 0),
            "active_event_count": _as_int(mst_state.get("active_event_count") or mst_state.get("event_count"), 0),
            "archive_event_count": _as_int(archive_state.get("archive_event_count") or mst_state.get("archive_event_count"), 0),
            "time_span": _as_span(mst_state.get("active_time_span")),
        },
        "refine": {
            "refine_version": _as_int(refine_state.get("refine_version") or archive_state.get("archive_version"), 0),
            "pending_event_count": _as_int(refine_state.get("pending_event_count"), 0),
            "refined_event_count": _as_int(refine_state.get("refined_event_count"), 0),
            "ready_30s_window_count": _as_int(refine_state.get("ready_30s_window_count"), _ready_windows_count(session_dir)),
        },
        "episodic_30s": {
            "ready": bool((session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json").exists()),
            "version": _as_int(episodic_state.get("mst_episodic_version") or episodic_state.get("version") or episodic_state.get("episode_count"), episode_file_count),
            "generated_episode_count": _as_int(
                episodic_state.get("episode_count")
                or episodic_state.get("generated_episode_count")
                or consolidation_state.get("generated_episode_count"),
                episode_file_count,
            ),
            "pending_ready_window_count": _as_int(
                consolidation_state.get("pending_ready_window_count"),
                _pending_ready_windows_count(session_dir),
            ),
        },
        "long_term": {
            "latest_ready_memory_version": _as_int(latest_ready, 0) if latest_ready is not None else None,
            "building_memory_version": _as_int(building, 0) if building is not None else None,
            "active_query_memory_version": _as_int(active_query, 0) if active_query is not None else None,
            "worldmm_update_mode": memory_config.get("worldmm_update_mode"),
            "latest_fast_ready_version": _as_int(latest_fast, 0) if latest_fast is not None else None,
            "latest_visual_ready_version": _as_int(latest_visual, 0) if latest_visual is not None else None,
            "latest_graph_ready_version": _as_int(latest_graph, 0) if latest_graph is not None else None,
            "latest_semantic_ready_version": _as_int(latest_semantic, 0) if latest_semantic is not None else None,
            "graph_lagging": bool(lag_state.get("graph_lagging")),
            "semantic_lagging": bool(lag_state.get("semantic_lagging")),
            "visual_lagging": bool(lag_state.get("visual_lagging")),
            "component_versions": component_versions,
            "append_pending_count": _as_int(append_state.get("pending_count"), 0),
            "append_appended_count": _as_int(append_state.get("appended_count"), 0),
            "append_failed_count": _as_int(append_state.get("failed_count"), 0),
            "build_state": build_state,
            "long_term_partial_ready": long_term_partial_ready,
            "long_term_full_ready": long_term_full_ready,
            "using_stale_while_building": bool(latest_ready and building and _as_int(building, 0) > _as_int(latest_ready, 0)),
        },
        "visual": {
            "visual_embedding_ready": visual_ready,
            "visual_version": _as_int(memory_config.get("visual_version"), 0),
            "latest_visual_ready_version": _as_int(latest_visual, 0) if latest_visual is not None else None,
            "visual_lagging": bool(lag_state.get("visual_lagging")),
        },
        "semantic": {
            "semantic_memory_ready": semantic_ready,
            "semantic_version": _as_int(memory_config.get("semantic_version"), 0),
            "latest_semantic_ready_version": _as_int(latest_semantic, 0) if latest_semantic is not None else None,
            "latest_graph_ready_version": _as_int(latest_graph, 0) if latest_graph is not None else None,
            "semantic_lagging": bool(lag_state.get("semantic_lagging")),
            "graph_lagging": bool(lag_state.get("graph_lagging")),
            "semantic_pending_count": _as_int(memory_config.get("semantic_pending_count"), 0),
        },
        "query": {
            "query_ready": bool(latest_ready or _as_bool(mst_state.get("short_term_ready"), False) or _as_bool(current_state.get("mcur_ready"), False)),
            "strict_load_only": _as_bool(os.getenv("WORLDMM_QUERY_STRICT_LOAD_ONLY"), True),
        },
        "updated_at": utc_now_iso(),
    }
    return state


def refresh_session_pipeline_state(session_dir: Path) -> dict[str, Any]:
    state = build_session_pipeline_state(Path(session_dir))
    write_json_atomic(Path(session_dir) / "pipeline_state.json", state)
    return state


def collect_session_states(sessions_root: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    root = Path(sessions_root)
    if session_id:
        session_dirs = [root / session_id]
    else:
        session_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:20] if root.exists() else []
    states = []
    for session_dir in session_dirs:
        if not session_dir.exists():
            continue
        states.append(refresh_session_pipeline_state(session_dir))
    return states


def collect_pipeline_runtime(project_root: Path, sessions_root: Path, session_id: str | None = None) -> dict[str, Any]:
    project_root = Path(project_root)
    sessions_root = Path(sessions_root)
    mode = get_pipeline_mode()
    workers = collect_worker_runtime(project_root)
    query_runtime = _safe_json(project_root / "online_tasks" / "query_runtime.json", {})
    sessions = collect_session_states(sessions_root, session_id=session_id)
    try:
        from online_pipeline.backpressure import compute_backpressure
        from online_pipeline.stream_timeline import read_timeline_events

        for state in sessions:
            stream = state.get("stream") if isinstance(state.get("stream"), dict) else {}
            state["backpressure"] = compute_backpressure(project_root=project_root, stream_latency=stream.get("latency") if isinstance(stream, dict) else None)
            state["recent_timeline_events"] = read_timeline_events(sessions_root / str(state.get("session_id")), limit=10)
    except Exception:
        pass
    evidence_runtime = workers.get("evidence") or {}
    legacy_evidence = {
        "role": evidence_runtime.get("role") or _default_worker_role("evidence", mode),
        "enabled": bool(evidence_runtime.get("status") not in {None, "", "disabled"}) and bool(evidence_runtime.get("enabled_by_default", _default_worker_enabled("evidence", mode))),
        "enabled_by_default": _default_worker_enabled("evidence", mode),
        "status": evidence_runtime.get("status") or ("disabled" if mode == "mst" else "unknown"),
        "reason": evidence_runtime.get("reason") or (
            "MST-first pipeline uses refined micro-events to build 30s episodic evidence"
            if mode == "mst"
            else "legacy evidence worker is the main 30s evidence generator"
            if mode == "legacy"
            else "hybrid mode keeps legacy evidence as optional fallback/ablation"
        ),
    }
    return {
        "status": "ok",
        "updated_at": utc_now_iso(),
        "pipeline_mode": mode,
        "pipeline": pipeline_summary(mode),
        "legacy_evidence": legacy_evidence,
        "workers": workers,
        "queue_counts": queue_counts(project_root),
        "sessions": sessions,
        "query_runtime": query_runtime if isinstance(query_runtime, dict) else {},
    }
