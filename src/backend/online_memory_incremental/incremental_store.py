from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from online_memory.evidence_to_worldmm import (
    build_multiscale_caption_items,
    evidence_doc_to_caption_item,
    load_online_evidence,
)
from online_memory.worldmm_layout import WorldMMOnlineLayout, ensure_worldmm_layout, seconds_to_hhmmssff
from online_preprocess.io_utils import ensure_dir, read_json, relative_to_session, utc_now_iso, write_json_atomic

from .append_log import MemoryAppendLog
from .component_versions import merge_component_versions, reconcile_component_versions
from .dirty_windows import DirtyWindowManager, infer_multiscale_levels
from .graph_delta import generate_graph_delta
from .hipporag_cache import inspect_hipporag_cache_health, refresh_hipporag_cache
from .semantic_delta import generate_semantic_delta
from .snapshot_manager import SnapshotManager
from .visual_task_writer import enqueue_visual_append_task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "src" / "HippoRAG" / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass
class IncrementalAppendResult:
    status: str
    session_id: str
    worldmm_update_mode: str
    fast_memory_version: int | None
    appended_episode_ids: list[str]
    skipped_episode_ids: list[str]
    dirty_window_count: int
    graph_lagging: bool
    semantic_lagging: bool
    visual_lagging: bool
    snapshot_path: str | None
    append_state_path: str
    component_versions_path: str
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        return [x for x in data["episodes"] if isinstance(x, dict)]
    return []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _log_stage(session_id: str, stage: str, **fields: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    print(f"[incremental_memory][stage] session={session_id} stage={stage} {suffix}".rstrip(), flush=True)


def _max_end_hhmmss(caption_items: list[dict[str, Any]]) -> str:
    max_end = max((float(item.get("end") or 0.0) for item in caption_items), default=0.0)
    return seconds_to_hhmmssff(max_end)


def _extract_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _clean_str_list(value: Any, limit: int = 12) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("name") or item.get("entity") or item.get("action") or json.dumps(item, ensure_ascii=False)
            else:
                text = item
            text = re.sub(r"\s+", " ", str(text or "")).strip()
            if text and text not in out:
                out.append(text)
            if len(out) >= limit:
                break
        return out
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return [text] if text else []


class IncrementalMemoryAppender:
    def __init__(
        self,
        *,
        session_id: str,
        sessions_root: Path,
        project_root: Path | None = None,
        model_name: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.session_id = session_id
        self.sessions_root = Path(sessions_root)
        self.session_dir = self.sessions_root / session_id
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.model_name = model_name or os.getenv("WORLDMM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"
        self.verbose = verbose
        self.layout = WorldMMOnlineLayout(session_dir=self.session_dir, session_id=session_id)
        self.incremental_root = self.session_dir / "worldmm" / "incremental"
        self.append_log = MemoryAppendLog(self.incremental_root / "memory_append_log.jsonl")
        self.dirty_manager = DirtyWindowManager(self.session_dir)

    @property
    def append_state_path(self) -> Path:
        return self.incremental_root / "append_state.json"

    @property
    def component_versions_path(self) -> Path:
        return self.incremental_root / "component_versions.json"

    def load_versions(self) -> dict[str, Any]:
        data = read_json(self.component_versions_path, default={})
        if isinstance(data, dict) and data:
            return data
        config = self.load_memory_config()
        fast_version = _safe_int(config.get("latest_fast_ready_version") or config.get("latest_ready_memory_version") or config.get("memory_version"), 0)
        semantic_version = _safe_int(config.get("latest_semantic_ready_version") or config.get("semantic_version"), fast_version if config.get("semantic_memory_ready") else 0)
        graph_version = _safe_int(config.get("latest_graph_ready_version") or config.get("graph_version"), fast_version if config.get("hipporag_cache_ready") else 0)
        visual_version = _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0)
        return {
            "session_id": self.session_id,
            "fast": {"latest_ready_version": fast_version, "building_version": None, "active_query_version": _safe_int(config.get("active_query_fast_version"), fast_version)},
            "episodic": {"version": fast_version, "ready": bool(fast_version)},
            "visual": {"version": visual_version, "ready": bool(config.get("visual_embedding_ready")), "lagging": bool(fast_version and visual_version < fast_version)},
            "graph": {"version": graph_version, "building_version": None, "ready": bool(graph_version), "lagging": bool(fast_version and graph_version < fast_version)},
            "semantic": {"version": semantic_version, "building_version": None, "ready": bool(config.get("semantic_memory_ready")), "lagging": bool(fast_version and semantic_version < fast_version)},
            "full": {"latest_full_ready_version": min([v for v in [fast_version, visual_version, graph_version, semantic_version] if v] or [0]), "long_term_full_ready": bool(config.get("long_term_full_ready"))},
            "updated_at": utc_now_iso(),
        }

    def write_versions(self, versions: dict[str, Any]) -> None:
        patches = {
            key: value
            for key, value in versions.items()
            if key in {"fast", "episodic", "visual", "graph", "semantic", "full"} and isinstance(value, dict)
        }
        merged = merge_component_versions(self.session_dir, patches, reconcile=False)
        for key, value in versions.items():
            if key not in patches and key not in {"session_id", "updated_at"}:
                merged[key] = value
        merged["session_id"] = self.session_id
        merged["updated_at"] = utc_now_iso()
        write_json_atomic(self.component_versions_path, merged)
        reconcile_component_versions(self.session_dir)

    def load_memory_config(self) -> dict[str, Any]:
        data = read_json(self.layout.memory_config_path, default={})
        return data if isinstance(data, dict) else {}

    def _ensure_base_layout(self) -> None:
        ensure_worldmm_layout(self.layout)
        ensure_dir(self.incremental_root)
        ensure_dir(self.incremental_root / "graph" / "deltas")
        ensure_dir(self.incremental_root / "semantic" / "deltas")
        ensure_dir(self.incremental_root / "visual")

    def load_mst_episodes(self) -> list[dict[str, Any]]:
        return _load_json_list(self.session_dir / "worldmm" / "mst_episodic" / "mst_30sec_episodes.json")

    def _episode_filter(self, episode_ids: list[str] | None) -> set[str] | None:
        if not episode_ids:
            return None
        return {str(x) for x in episode_ids if str(x).strip()}

    def plan_ready_episodes(self, *, episode_ids: list[str] | None = None, force: bool = False) -> dict[str, Any]:
        wanted = self._episode_filter(episode_ids)
        episodes = []
        skipped = []
        for episode in self.load_mst_episodes():
            episode_id = str(episode.get("episode_id") or "")
            if wanted is not None and episode_id not in wanted:
                continue
            decision = self.append_log.decide(episode, force=force)
            if decision.should_append:
                episodes.append({"episode_id": episode_id, "reason": decision.reason, "start_time": episode.get("start_time"), "end_time": episode.get("end_time")})
            else:
                skipped.append({"episode_id": episode_id, "reason": decision.reason, "latest_status": (decision.latest or {}).get("status")})
        return {"session_id": self.session_id, "append_candidates": episodes, "skipped": skipped, "candidate_count": len(episodes)}

    def _evidence_by_episode(self) -> dict[str, dict[str, Any]]:
        docs = load_online_evidence(self.session_dir, evidence_filename="mst_session_evidence.json")
        by_episode: dict[str, dict[str, Any]] = {}
        for doc in docs:
            episode_id = str(doc.get("episode_id") or doc.get("doc_id") or "")
            if episode_id:
                by_episode[episode_id] = doc
        return by_episode

    def _caption_item_for_episode(self, episode: dict[str, Any], evidence_doc: dict[str, Any], idx: int) -> dict[str, Any]:
        item = evidence_doc_to_caption_item(self.session_id, evidence_doc, idx)
        item["episode_id"] = episode.get("episode_id")
        item["source_episode_id"] = episode.get("episode_id")
        item["source_micro_event_ids"] = episode.get("source_micro_event_ids") or evidence_doc.get("source_micro_event_ids") or []
        item["episodic_source"] = "mst_micro_events"
        item["incremental_source"] = "stage8b_incremental_append"
        return item

    def _load_caption_30s(self) -> list[dict[str, Any]]:
        return _load_json_list(self.layout.caption_30sec_path)

    def _upsert_caption_items(self, existing: list[dict[str, Any]], new_items: list[dict[str, Any]], force: bool) -> list[dict[str, Any]]:
        by_doc = {str(item.get("doc_id") or item.get("episode_id") or ""): dict(item) for item in existing}
        for item in new_items:
            doc_id = str(item.get("doc_id") or item.get("episode_id") or "")
            if not doc_id:
                continue
            if doc_id in by_doc and not force:
                continue
            by_doc[doc_id] = dict(item)
        return sorted(by_doc.values(), key=lambda x: (str(x.get("date", "DAY1")), float(x.get("start") or 0.0), float(x.get("end") or 0.0)))

    def _update_dirty_multiscale(self, caption_30s: list[dict[str, Any]], dirty_windows: list[dict[str, Any]]) -> set[str]:
        levels = infer_multiscale_levels(self.session_dir)
        dirty_by_level = {}
        for window in dirty_windows:
            dirty_by_level.setdefault(str(window.get("level")), []).append(window)

        clean_ids: set[str] = set()
        for level, seconds in levels.items():
            path = self.layout.caption_root / f"{self.session_id}_{level}.json"
            existing = _load_json_list(path)
            rebuilt_all = build_multiscale_caption_items(caption_30s, seconds, level)
            dirty = dirty_by_level.get(level, [])
            if not dirty:
                continue
            dirty_ids = {str(item["window_id"]) for item in dirty}

            def item_window_id(item: dict[str, Any]) -> str:
                start = int(float(item.get("start") or 0.0) // seconds) * seconds
                end = start + seconds
                return f"{level}_{start:06d}_{end:06d}"

            rebuilt_dirty = {item_window_id(item): item for item in rebuilt_all if item_window_id(item) in dirty_ids}
            self._maybe_rewrite_dirty_windows_with_llm(rebuilt_dirty, caption_30s, level)
            kept = [item for item in existing if item_window_id(item) not in dirty_ids]
            merged = kept + list(rebuilt_dirty.values())
            merged = sorted(merged, key=lambda x: (float(x.get("start") or 0.0), float(x.get("end") or 0.0)))
            write_json_atomic(path, merged)
            clean_ids.update(rebuilt_dirty.keys())
        return clean_ids

    def _maybe_rewrite_dirty_windows_with_llm(self, rebuilt_dirty: dict[str, dict[str, Any]], caption_30s: list[dict[str, Any]], level: str) -> None:
        backend = (os.getenv("WORLDMM_INCREMENTAL_DIRTY_MULTISCALE_BACKEND") or os.getenv("WORLDMM_MEMORY_GENERATION_BACKEND") or "llm").strip().lower()
        if backend != "llm" or not rebuilt_dirty:
            for item in rebuilt_dirty.values():
                item["generation_backend"] = item.get("generation_backend") or "rule_incremental_dirty_window"
            return
        max_windows = _safe_int(os.getenv("WORLDMM_INCREMENTAL_DIRTY_LLM_MAX_WINDOWS"), 4)
        if max_windows <= 0:
            return
        from worldmm.llm import LLMModel

        llm = LLMModel(model_name=self.model_name, max_retries=_safe_int(os.getenv("WORLDMM_MEMORY_LLM_RETRIES"), 2))
        for idx, item in enumerate(rebuilt_dirty.values()):
            if idx >= max_windows:
                item["generation_backend"] = item.get("generation_backend") or "rule_incremental_dirty_window_over_limit"
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or 0.0)
            children = [
                {
                    "doc_id": child.get("doc_id"),
                    "start": child.get("start"),
                    "end": child.get("end"),
                    "text": child.get("text") or child.get("fine_caption") or child.get("caption"),
                    "transcript": child.get("transcript"),
                    "visual_objects": child.get("visual_objects") or [],
                    "main_actions": child.get("main_actions") or [],
                    "state_changes": child.get("state_changes") or [],
                    "entities": child.get("entities") or [],
                }
                for child in caption_30s
                if float(child.get("start") or 0.0) < end and float(child.get("end") or 0.0) > start
            ]
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You rewrite one dirty multiscale episodic memory window for a long-video memory system. "
                        "Use only the provided child 30-second memories. Preserve temporal order and do not invent identities. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "level": level,
                            "window": {"start": start, "end": end},
                            "current_rule_item": item,
                            "child_30s_memories": children,
                            "return_fields": [
                                "text",
                                "topic_threads",
                                "key_observations",
                                "involved_entities",
                                "salient_objects",
                                "events",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            try:
                response = llm.generate(prompt, max_tokens=_safe_int(os.getenv("WORLDMM_INCREMENTAL_DIRTY_LLM_MAX_TOKENS"), 900))
                data = _extract_json_object(response)
                rewritten_text = re.sub(r"\s+", " ", str(data.get("text") or "")).strip()
                if rewritten_text:
                    item["text"] = rewritten_text
                    item["caption"] = rewritten_text
                    item["fine_caption"] = rewritten_text
                for key in ("topic_threads", "key_observations", "involved_entities", "salient_objects", "events"):
                    values = _clean_str_list(data.get(key), limit=16)
                    if values:
                        item[key] = values
                item["generation_backend"] = "llm_incremental_dirty_window"
                item["generation_model"] = self.model_name
                item["generation_updated_at"] = utc_now_iso()
            except Exception as exc:
                item["generation_backend"] = "rule_incremental_dirty_window_fallback"
                item["generation_error"] = f"{type(exc).__name__}: {exc}"

    def _write_caption_roots(self, caption_30s: list[dict[str, Any]]) -> None:
        write_json_atomic(self.layout.caption_30sec_path, caption_30s)
        write_json_atomic(self.layout.visual_evidence_path, caption_30s)

    def _write_visual_task(self, *, episode_ids: list[str], keyframe_paths: list[str], target_version: int) -> Path | None:
        return enqueue_visual_append_task(
            project_root=self.project_root,
            session_id=self.session_id,
            episode_ids=episode_ids,
            keyframe_paths=keyframe_paths,
            target_visual_version=target_version,
        )

    def _refresh_hipporag_cache_isolated(self, *, force: bool = False) -> dict[str, Any]:
        if not _env_bool("WORLDMM_INCREMENTAL_REFRESH_HIPPORAG_CACHE", True):
            return {"status": "disabled", "healthy": None}
        if not _env_bool("WORLDMM_INCREMENTAL_HIPPORAG_REFRESH_SUBPROCESS", True):
            return refresh_hipporag_cache(
                session_dir=self.session_dir,
                project_root=self.project_root,
                model_name=self.model_name,
                force=force,
            )
        cmd = [
            sys.executable,
            str(self.project_root / "incremental_append_memory.py"),
            "--session-id",
            self.session_id,
            "--sessions-root",
            str(self.sessions_root),
            "--refresh-hipporag-cache-only",
        ]
        if self.model_name:
            cmd.extend(["--model", self.model_name])
        if force:
            cmd.append("--force")
        proc = subprocess.run(cmd, cwd=str(self.project_root), text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            return {
                "status": "failed",
                "healthy": False,
                "returncode": proc.returncode,
                "error": f"hipporag refresh subprocess failed rc={proc.returncode}",
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        try:
            return json.loads(proc.stdout)
        except Exception:
            return {
                "status": "done",
                "healthy": None,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }

    def _update_memory_config(
        self,
        *,
        version: int,
        caption_30s: list[dict[str, Any]],
        graph_ready: bool,
        semantic_ready: bool,
        visual_task_path: Path | None,
        snapshot_path: Path,
    ) -> dict[str, Any]:
        config = self.load_memory_config()
        semantic_version = version if semantic_ready else _safe_int(config.get("latest_semantic_ready_version") or config.get("semantic_version"), 0)
        graph_version = version if graph_ready else _safe_int(config.get("latest_graph_ready_version") or config.get("graph_version"), 0)
        visual_version = _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0)
        visual_lagging = bool(visual_task_path)
        semantic_lagging = semantic_version < version
        graph_lagging = graph_version < version
        config.update(
            {
                "session_id": self.session_id,
                "status": "memory_ready",
                "memory_version": version,
                "latest_ready_memory_version": version,
                "latest_fast_ready_version": version,
                "building_memory_version": None,
                "memory_build_state": "ready",
                "worldmm_update_mode": "incremental_append",
                "pipeline_mode": os.getenv("WORLDMM_PIPELINE_MODE", "mst"),
                "active_30s_source": "mst_session_30sec_captioned",
                "worldmm_30s_input_source": "mst_session_30sec_captioned",
                "episodic_source": "mst_micro_events",
                "mst_episodic_ready": True,
                "legacy_evidence_used": False,
                "legacy_evidence_fallback_used": False,
                "caption_root": "worldmm/caption_root",
                "sidecar_root": "worldmm/sidecar_root",
                "semantic_root": "worldmm/semantic_root",
                "visual_root": config.get("visual_root") or "worldmm/visual",
                "evidence_path": "evidence/mst_session_evidence.json",
                "captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
                "mst_captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
                "mst_evidence_path": "evidence/mst_session_evidence.json",
                "episodic_index_ready": True,
                "long_term_partial_ready": True,
                "long_term_full_ready": bool(not visual_lagging and semantic_ready and graph_ready),
                "latest_semantic_ready_version": semantic_version,
                "latest_graph_ready_version": graph_version,
                "latest_visual_ready_version": visual_version,
                "semantic_version": semantic_version,
                "graph_version": graph_version,
                "building_versions": {
                    "fast": None,
                    "visual": version if visual_lagging else None,
                    "graph": None if graph_ready else version,
                    "semantic": None if semantic_ready else version,
                },
                "readiness": {
                    "long_term_partial_ready": True,
                    "long_term_full_ready": bool(not visual_lagging and semantic_ready and graph_ready),
                    "episodic_ready": True,
                    "visual_ready": not visual_lagging and bool(config.get("visual_embedding_ready")),
                    "graph_ready": graph_ready,
                    "semantic_ready": semantic_ready,
                },
                "lag": {
                    "semantic_lagging": semantic_lagging,
                    "graph_lagging": graph_lagging,
                    "visual_lagging": visual_lagging,
                    "semantic_lag_versions": max(0, version - semantic_version),
                    "graph_lag_versions": max(0, version - graph_version),
                    "visual_lag_versions": max(0, version - visual_version),
                },
                "query_rag_args": {
                    **(config.get("query_rag_args") or {}),
                    "subject": self.session_id,
                    "retriever_model": config.get("query_rag_args", {}).get("retriever_model") if isinstance(config.get("query_rag_args"), dict) else self.model_name,
                    "respond_model": config.get("query_rag_args", {}).get("respond_model") if isinstance(config.get("query_rag_args"), dict) else self.model_name,
                    "until_date": "DAY1",
                    "until_time": _max_end_hhmmss(caption_30s),
                    "episodic_caption_root": "worldmm/caption_root",
                    "episodic_sidecar_root": "worldmm/sidecar_root",
                    "semantic_root": "worldmm/semantic_root",
                    "visual_root": config.get("query_rag_args", {}).get("visual_root", "worldmm/embeddings") if isinstance(config.get("query_rag_args"), dict) else "worldmm/embeddings",
                    "visual_evidence_file": "worldmm/visual_root/session_visual_evidence.json",
                },
                "worldmm_files": {
                    "caption_30sec": relative_to_session(self.layout.caption_30sec_path, self.session_dir),
                    "caption_3min": relative_to_session(self.layout.caption_3min_path, self.session_dir) if self.layout.caption_3min_path.exists() else None,
                    "caption_10min": relative_to_session(self.layout.caption_10min_path, self.session_dir) if self.layout.caption_10min_path.exists() else None,
                    "caption_1h": relative_to_session(self.layout.caption_1h_path, self.session_dir) if self.layout.caption_1h_path.exists() else None,
                    "visual_evidence": relative_to_session(self.layout.visual_evidence_path, self.session_dir),
                },
                "counts": {
                    **(config.get("counts") if isinstance(config.get("counts"), dict) else {}),
                    "caption_30sec": len(caption_30s),
                },
                "latest_snapshot_version": version,
                "latest_snapshot_path": relative_to_session(snapshot_path, self.session_dir),
                "last_incremental_append_at": utc_now_iso(),
                "last_ready_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        )
        write_json_atomic(self.layout.memory_config_path, config)
        return config

    def append_ready_episodes(
        self,
        *,
        episode_ids: list[str] | None = None,
        force: bool = False,
        dry_run: bool = False,
        skip_graph_semantic: bool = False,
    ) -> IncrementalAppendResult:
        self._ensure_base_layout()
        plan = self.plan_ready_episodes(episode_ids=episode_ids, force=force)
        candidates = {item["episode_id"] for item in plan["append_candidates"]}
        skipped_ids = [item["episode_id"] for item in plan["skipped"]]
        if dry_run:
            return IncrementalAppendResult(
                status="dry_run",
                session_id=self.session_id,
                worldmm_update_mode="incremental_append",
                fast_memory_version=None,
                appended_episode_ids=sorted(candidates),
                skipped_episode_ids=skipped_ids,
                dirty_window_count=0,
                graph_lagging=False,
                semantic_lagging=False,
                visual_lagging=False,
                snapshot_path=None,
                append_state_path=relative_to_session(self.append_state_path, self.session_dir),
                component_versions_path=relative_to_session(self.component_versions_path, self.session_dir),
                message="dry-run: no files modified",
            )
        if not candidates:
            state = self.append_log.write_state(self.append_state_path, self.session_id)
            hipporag_health = inspect_hipporag_cache_health(self.session_dir, self.project_root)
            if not hipporag_health.get("healthy") and _env_bool("WORLDMM_INCREMENTAL_REFRESH_HIPPORAG_CACHE_ON_NOOP", False):
                try:
                    hipporag_health = self._refresh_hipporag_cache_isolated(force=False).get("after", hipporag_health)
                except Exception as exc:
                    hipporag_health["repair_error"] = f"{type(exc).__name__}: {exc}"
            return IncrementalAppendResult(
                status="ok",
                session_id=self.session_id,
                worldmm_update_mode="incremental_append",
                fast_memory_version=_safe_int(self.load_memory_config().get("latest_fast_ready_version") or self.load_memory_config().get("latest_ready_memory_version"), 0),
                appended_episode_ids=[],
                skipped_episode_ids=skipped_ids,
                dirty_window_count=0,
                graph_lagging=False,
                semantic_lagging=False,
                visual_lagging=False,
                snapshot_path=None,
                append_state_path=relative_to_session(self.append_state_path, self.session_dir),
                component_versions_path=relative_to_session(self.component_versions_path, self.session_dir),
                message=(
                    f"no new episodes to append; append_state={state.get('appended_count', 0)} appended; "
                    f"hipporag_cache_healthy={bool(hipporag_health.get('healthy'))}"
                ),
            )

        config = self.load_memory_config()
        current_version = max(
            _safe_int(config.get("latest_fast_ready_version"), 0),
            _safe_int(config.get("latest_ready_memory_version"), 0),
            _safe_int(config.get("memory_version"), 0),
            _safe_int(self.load_versions().get("fast", {}).get("latest_ready_version"), 0),
        )
        target_version = current_version + 1
        _log_stage(self.session_id, "fast_append", target_version=target_version, candidate_count=len(candidates))
        episodes = [ep for ep in self.load_mst_episodes() if str(ep.get("episode_id") or "") in candidates]
        evidence_by_episode = self._evidence_by_episode()
        existing_caption_30s = self._load_caption_30s()
        new_caption_items: list[dict[str, Any]] = []
        dirty_windows: list[dict[str, Any]] = []
        keyframe_paths: list[str] = []

        for episode in episodes:
            episode_id = str(episode.get("episode_id") or "")
            evidence_doc = evidence_by_episode.get(episode_id)
            if not evidence_doc:
                self.append_log.append_status(episode=episode, status="failed", error=f"missing MST evidence doc for {episode_id}")
                continue
            self.append_log.append_status(episode=episode, status="appending", fast_memory_version=target_version, extra={"force": force})
            item = self._caption_item_for_episode(episode, evidence_doc, len(existing_caption_30s) + len(new_caption_items))
            new_caption_items.append(item)
            dirty_windows.extend(self.dirty_manager.mark_dirty(episode))
            keyframe_paths.extend(str(x) for x in item.get("keyframe_paths", []) or [])

        caption_30s = self._upsert_caption_items(existing_caption_30s, new_caption_items, force=force)
        self._write_caption_roots(caption_30s)
        clean_ids = self._update_dirty_multiscale(caption_30s, dirty_windows)
        self.dirty_manager.mark_clean(clean_ids)

        visual_task_path = None
        if _env_bool("WORLDMM_AUTO_VISUAL_EMBEDDING", True):
            _log_stage(self.session_id, "visual_enqueue", target_version=target_version, keyframe_count=len(keyframe_paths))
            visual_task_path = self._write_visual_task(
                episode_ids=[str(ep.get("episode_id")) for ep in episodes],
                keyframe_paths=keyframe_paths,
                target_version=target_version,
            )
            _log_stage(
                self.session_id,
                "visual_build",
                delegated_to="online_visual_worker",
                visual_task_path=str(visual_task_path) if visual_task_path else None,
            )
        else:
            _log_stage(self.session_id, "visual_skip", reason="WORLDMM_AUTO_VISUAL_EMBEDDING=0", target_version=target_version, keyframe_count=len(keyframe_paths))
        visual_lagging = visual_task_path is not None

        # Publish the fast episodic snapshot before slow graph/semantic/cache work.
        fast_components = {
            "episodic": target_version,
            "visual": _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0),
            "graph": _safe_int(config.get("latest_graph_ready_version"), 0),
            "semantic": _safe_int(config.get("latest_semantic_ready_version"), 0),
        }
        snapshot_path = SnapshotManager(self.session_dir).build_fast_snapshot(
            target_version,
            components=fast_components,
            long_term_partial_ready=True,
            long_term_full_ready=False,
            semantic_lagging=True,
            graph_lagging=True,
        )
        self._update_memory_config(
            version=target_version,
            caption_30s=caption_30s,
            graph_ready=False,
            semantic_ready=False,
            visual_task_path=visual_task_path,
            snapshot_path=snapshot_path,
        )
        _log_stage(self.session_id, "component_versions_update", target_version=target_version, fast_ready=True)
        config = self.load_memory_config()

        graph_state = {"graph_lagging": True, "latest_graph_ready_version": _safe_int(config.get("latest_graph_ready_version"), 0)}
        semantic_state = {"semantic_lagging": True, "latest_semantic_ready_version": _safe_int(config.get("latest_semantic_ready_version"), 0)}
        if new_caption_items and not skip_graph_semantic:
            try:
                _log_stage(self.session_id, "triple_extraction", target_version=target_version, item_count=len(new_caption_items))
                graph_state = generate_graph_delta(
                    session_dir=self.session_dir,
                    version=target_version,
                    new_caption_items=new_caption_items,
                    all_caption_items=caption_30s,
                    model_name=self.model_name,
                )
                _log_stage(self.session_id, "semantic_ner", target_version=target_version)
                graph_delta_path = self.session_dir / str(graph_state["delta_path"])
                semantic_state = generate_semantic_delta(
                    session_dir=self.session_dir,
                    version=target_version,
                    graph_delta_path=graph_delta_path,
                    model_name=self.model_name,
                )
            except Exception as exc:
                _log_stage(self.session_id, "semantic_ner", status="lagging", error=f"{type(exc).__name__}: {exc}")
                graph_state["error"] = str(exc)
                semantic_state["error"] = str(exc)
        graph_ready = not bool(graph_state.get("graph_lagging")) and _safe_int(graph_state.get("latest_graph_ready_version"), 0) >= target_version
        semantic_ready = not bool(semantic_state.get("semantic_lagging")) and _safe_int(semantic_state.get("latest_semantic_ready_version"), 0) >= target_version

        components = {
            "episodic": target_version,
            "visual": _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0),
            "graph": target_version if graph_ready else _safe_int(config.get("latest_graph_ready_version"), 0),
            "semantic": target_version if semantic_ready else _safe_int(config.get("latest_semantic_ready_version"), 0),
        }
        snapshot_path = SnapshotManager(self.session_dir).build_fast_snapshot(
            target_version,
            components=components,
            long_term_partial_ready=True,
            long_term_full_ready=bool(not visual_lagging and graph_ready and semantic_ready),
            semantic_lagging=not semantic_ready,
            graph_lagging=not graph_ready,
        )
        self._update_memory_config(
            version=target_version,
            caption_30s=caption_30s,
            graph_ready=graph_ready,
            semantic_ready=semantic_ready,
            visual_task_path=visual_task_path,
            snapshot_path=snapshot_path,
        )
        try:
            _log_stage(self.session_id, "graph_build", mode="hipporag_cache_subprocess", target_version=target_version)
            hipporag_refresh_result = self._refresh_hipporag_cache_isolated(force=False)
            _log_stage(
                self.session_id,
                "graph_build",
                status=hipporag_refresh_result.get("status"),
                returncode=hipporag_refresh_result.get("returncode"),
            )
        except Exception as exc:
            hipporag_refresh_result = {
                "status": "failed",
                "healthy": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if hipporag_refresh_result.get("status") == "failed" or hipporag_refresh_result.get("healthy") is False:
            cache_config = self.load_memory_config()
            cache_config["hipporag_cache_ready"] = False
            cache_config["hipporag_cache_lagging"] = True
            cache_config["hipporag_cache_warning"] = "HippoRAG cache refresh failed or is missing; graph/semantic component readiness remains independent."
            cache_config["hipporag_cache_error"] = hipporag_refresh_result.get("error") or hipporag_refresh_result.get("stderr_tail")
            cache_config["updated_at"] = utc_now_iso()
            write_json_atomic(self.layout.memory_config_path, cache_config)

        config = self.load_memory_config()
        versions = self.load_versions()
        versions.update(
            {
                "fast": {"latest_ready_version": target_version, "building_version": None, "active_query_version": _safe_int(config.get("active_query_fast_version"), current_version)},
                "episodic": {"version": target_version, "ready": True},
                "visual": {
                    "version": _safe_int(config.get("latest_visual_ready_version") or config.get("visual_version"), 0),
                    "building_version": target_version if visual_lagging else None,
                    "ready": bool(config.get("visual_embedding_ready")),
                    "lagging": visual_lagging,
                },
                "graph": {
                    "version": target_version if graph_ready else _safe_int(config.get("latest_graph_ready_version"), 0),
                    "building_version": None if graph_ready else target_version,
                    "ready": bool(graph_ready),
                    "lagging": not graph_ready,
                },
                "semantic": {
                    "version": target_version if semantic_ready else _safe_int(config.get("latest_semantic_ready_version"), 0),
                    "building_version": None if semantic_ready else target_version,
                    "ready": bool(semantic_ready),
                    "lagging": not semantic_ready,
                },
                "full": {
                    "latest_full_ready_version": target_version if (not visual_lagging and graph_ready and semantic_ready) else current_version,
                    "long_term_full_ready": bool(not visual_lagging and graph_ready and semantic_ready),
                },
            }
        )
        self.write_versions(versions)

        appended_ids = [str(ep.get("episode_id")) for ep in episodes]
        final_append_status = (
            "fully_ready"
            if graph_ready and semantic_ready and not visual_lagging
            else "appended_fast"
            if graph_ready and semantic_ready
            else "graph_semantic_pending"
        )
        for episode in episodes:
            self.append_log.append_status(
                episode=episode,
                status=final_append_status,
                fast_memory_version=target_version,
                semantic_memory_version=target_version if semantic_ready else None,
                graph_version=target_version if graph_ready else None,
                visual_version=None,
                extra={
                    "snapshot_path": relative_to_session(snapshot_path, self.session_dir),
                    "visual_task_path": str(visual_task_path) if visual_task_path else None,
                    "hipporag_cache_status": hipporag_refresh_result.get("status"),
                    "hipporag_cache_healthy": hipporag_refresh_result.get("healthy"),
                },
            )
        self.append_log.write_state(self.append_state_path, self.session_id)

        return IncrementalAppendResult(
            status="ok",
            session_id=self.session_id,
            worldmm_update_mode="incremental_append",
            fast_memory_version=target_version,
            appended_episode_ids=appended_ids,
            skipped_episode_ids=skipped_ids,
            dirty_window_count=len(clean_ids),
            graph_lagging=not graph_ready,
            semantic_lagging=not semantic_ready,
            visual_lagging=visual_lagging,
            snapshot_path=relative_to_session(snapshot_path, self.session_dir),
            append_state_path=relative_to_session(self.append_state_path, self.session_dir),
            component_versions_path=relative_to_session(self.component_versions_path, self.session_dir),
        )
