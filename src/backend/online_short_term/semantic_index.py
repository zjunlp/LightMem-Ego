"""Incremental fine-event vectors, prepared outside the question path."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from online_preprocess.io_utils import utc_now_iso, write_json_atomic

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mst-semantic")
_LOCK = threading.RLock()
_PENDING: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
_RUNNING: set[str] = set()
_INDEXES: dict[str, Any] = {}
_ENCODER = None


def enabled() -> bool:
    default = "1" if os.getenv("EM2MEM_TEXT_EMBED_BACKEND") == "remote" else "0"
    return os.getenv("EM2MEM_MST_SEMANTIC_ENABLED", default).lower() in {"1", "true", "yes", "on"}


def event_text(event: dict[str, Any]) -> str:
    # retrieval_text already repeats the structured fields in newer stores.
    # Encode their substantive values once, without paths/confidence/schema.
    values = [event.get("event_caption_refined") or event.get("retrieval_text")
              or event.get("event_caption_fast"), event.get("transcript")]
    for field in ("visual_objects", "main_actions", "state_changes", "visual_observations"):
        for record in event.get(field) or []:
            if isinstance(record, dict):
                parts = [record.get(key) for key in ("actor", "action", "objects", "entity", "attribute",
                         "before", "after", "name", "attributes", "description") if record.get(key)]
                values.append(" ".join(part if isinstance(part, str) else json.dumps(part, ensure_ascii=False) for part in parts))
    return "\n".join(dict.fromkeys(str(value).strip() for value in values if value and str(value).strip()))


def _encoder():
    global _ENCODER
    with _LOCK:
        if _ENCODER is None:
            import online_text_embedding_runtime  # Sets up the local source path.
            from em2mem.embedding import EmbeddingModel
            _ENCODER = EmbeddingModel()
        return _ENCODER


def _normalized(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid short-term embedding matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class ShortTermSemanticIndex:
    def __init__(self, session_dir: Path, encoder=None, *, namespace: str | None = None):
        self.encoder = encoder
        identity = {key: os.getenv(key, "") for key in (
            "EM2MEM_TEXT_EMBED_BACKEND", "EM2MEM_TEXT_EMBED_MODEL", "EM2MEM_TEXT_EMBED_URL",
            "EM2MEM_TEXT_EMBED_NORMALIZE", "EM2MEM_TEXT_EMBED_REVISION")}
        identity["schema"] = "fine-event-v1"
        self.namespace = namespace or hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        self.root = Path(session_dir) / "short_term" / "semantic_index" / self.namespace
        self.path = self.root / "vectors.npz"
        self.state_path = self.root / "state.json"
        self._signature = None
        self._vectors: dict[str, np.ndarray] = {}
        self.last_search: dict[str, Any] = {"status": "not_ready"}
        self._prepared_signature = None
        self._prepared_at = 0.0

    def _key(self, event):
        text = event_text(event)
        return hashlib.sha256((str(event.get("event_id")) + "\0" + text).encode()).hexdigest()

    def _read(self):
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != self._signature:
                with np.load(self.path, allow_pickle=False) as data:
                    if str(data["namespace"].item()) != self.namespace:
                        return {}
                    self._vectors = {str(key): vector for key, vector in zip(data["keys"], data["vectors"])}
                self._signature = signature
            return dict(self._vectors)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return {}

    def _write(self, vectors):
        temporary = self.root / f".{uuid4().hex}.npz"
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, namespace=np.array(self.namespace),
                    keys=np.array(list(vectors)), vectors=np.stack(list(vectors.values())))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def build(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Encode changed substantive records only; publish each completed batch."""
        docs = {self._key(event): event_text(event) for event in events
                if event.get("event_id") and event_text(event)
                and (event.get("event_caption_refined") or event.get("status") in {"refined", "final"})}
        if not docs:
            return {"status": "empty", "encoded": 0}
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "build.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            vectors = self._read()
            # A prefix-time query or partial retry may supply only a subset.
            # Retain content-addressed vectors outside it: eligibility is checked
            # against the current candidate records, not cache membership.
            missing = [key for key in docs if key not in vectors]
            encoded_count = 0
            batch_size = max(1, int(os.getenv("EM2MEM_MST_EMBED_BATCH_SIZE", "2")))
            for offset in range(0, len(missing), batch_size):
                write_json_atomic(self.state_path, {"status": "building", "ready": sum(key in vectors for key in docs),
                    "total": len(docs), "updated_at": utc_now_iso()})
                keys = missing[offset:offset + batch_size]
                encoder = self.encoder or _encoder()
                encoded = _normalized(encoder.encode_text([docs[key] for key in keys], batch_size=batch_size))
                if len(encoded) != len(keys):
                    raise ValueError("Short-term embedding count mismatch")
                vectors.update(zip(keys, encoded))
                encoded_count += len(keys)
                self._write(vectors)
            if not missing and vectors:
                self._write(vectors)
            result = {"status": "ready", "ready": len(docs), "total": len(docs), "cached_versions": len(vectors),
                      "encoded": encoded_count, "updated_at": utc_now_iso()}
            write_json_atomic(self.state_path, result)
            return result

    def prepare(self, events):
        """Coalesce updates without making readers wait for a builder lock."""
        key = str(self.path)
        signature = tuple(self._key(event) for event in events)
        with _LOCK:
            if signature == self._prepared_signature and time.monotonic()-self._prepared_at < 30:
                return
            self._prepared_signature, self._prepared_at = signature, time.monotonic()
            _PENDING[key] = (self, deepcopy(events))
            if key not in _RUNNING:
                _RUNNING.add(key)
                _EXECUTOR.submit(_build_pending, key)

    def scores(self, question: str, events: list[dict[str, Any]]) -> dict[str, float]:
        vectors = self._read()
        available = [(event, vectors[self._key(event)]) for event in events if self._key(event) in vectors]
        self.last_search = {"status": "ready" if len(available) == len(events) else "partial" if available else "not_ready",
                            "indexed_candidates": len(available), "eligible_candidates": len(events)}
        if not available:
            return {}
        try:
            encoder = self.encoder or _encoder()
            query = encoder.encode_query(question) if hasattr(encoder, "encode_query") else encoder.encode_text([question])
            query = _normalized(query)[0]
            return {str(event["event_id"]): float(np.clip(np.dot(vector, query), 0.0, 1.0))
                    for event, vector in available}
        except Exception as exc:
            self.last_search.update(status="fallback", error=f"{type(exc).__name__}: {exc}")
            return {}


def _build_pending(key):
    while True:
        with _LOCK:
            item = _PENDING.pop(key, None)
            if item is None:
                _RUNNING.discard(key)
                return
        index, events = item
        try:
            index.build(events)
        except Exception as exc:
            index.root.mkdir(parents=True, exist_ok=True)
            write_json_atomic(index.state_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                               "updated_at": utc_now_iso()})


def get_index(session_dir: Path, encoder=None):
    candidate = ShortTermSemanticIndex(session_dir, encoder)
    key = str(candidate.path)
    with _LOCK:
        index = _INDEXES.setdefault(key, candidate)
        if encoder is not None:
            index.encoder = encoder
        return index


def schedule_semantic_index(session_dir: Path, events):
    if enabled():
        get_index(session_dir).prepare(events)
