from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from online_memory.em2mem_layout import seconds_to_hhmmssff
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
HIPPO_SRC_ROOT = PROJECT_ROOT / "src" / "HippoRAG" / "src"
for _path in (SRC_ROOT, HIPPO_SRC_ROOT):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


GRANULARITIES = ("30sec", "3min", "10min", "1h")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        return [x for x in data["episodes"] if isinstance(x, dict)]
    return []


def _caption_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("fine_caption") or item.get("caption") or "").strip()


def _parquet_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0


def _parquet_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        text_col = next((col for col in ("content", "text", "document", "passage", "chunk") if col in df.columns), None)
        if text_col is None:
            object_cols = [col for col in df.columns if str(df[col].dtype) == "object"]
            text_col = object_cols[0] if object_cols else None
        if text_col is None:
            return []
        return [str(x) for x in df[text_col].values.tolist() if str(x).strip()]
    except Exception:
        return []


def _first_model_cache_dir(granularity_cache_dir: Path) -> Path | None:
    if not granularity_cache_dir.exists():
        return None
    for child in sorted(granularity_cache_dir.iterdir()):
        if child.is_dir() and (child / "chunk_embeddings" / "vdb_chunk.parquet").exists():
            return child
    return None


def _cache_counts(project_root: Path, session_id: str, granularity: str, cache_tag: str | None = None) -> dict[str, Any]:
    tag = cache_tag or f"online_{session_id}"
    granularity_dir = Path(project_root) / ".cache" / "episodic_memory" / tag / granularity
    model_dir = _first_model_cache_dir(granularity_dir)
    counts: dict[str, Any] = {
        "granularity": granularity,
        "cache_dir": str(granularity_dir),
        "model_cache_dir": str(model_dir) if model_dir else None,
        "passages": 0,
        "entities": 0,
        "facts": 0,
        "graph_pickle_exists": False,
    }
    if model_dir is None:
        return counts
    chunk_path = model_dir / "chunk_embeddings" / "vdb_chunk.parquet"
    cached_texts = _parquet_texts(chunk_path)
    counts["passages"] = _parquet_row_count(chunk_path)
    counts["passage_texts"] = cached_texts
    counts["entities"] = _parquet_row_count(model_dir / "entity_embeddings" / "vdb_entity.parquet")
    counts["facts"] = _parquet_row_count(model_dir / "fact_embeddings" / "vdb_fact.parquet")
    counts["graph_pickle_exists"] = (model_dir / "graph.pickle").exists()
    return counts


def inspect_hipporag_cache_health(session_dir: Path, project_root: Path | None = None, cache_tag: str | None = None) -> dict[str, Any]:
    session_dir = Path(session_dir)
    project_root = Path(project_root) if project_root else PROJECT_ROOT
    session_id = session_dir.name
    levels: dict[str, dict[str, Any]] = {}
    healthy = True
    for granularity in GRANULARITIES:
        caption_path = session_dir / "em2mem" / "caption_root" / f"{session_id}_{granularity}.json"
        captions = _load_json_list(caption_path)
        active_texts = list(dict.fromkeys([_caption_text(item) for item in captions if _caption_text(item)]))
        expected_docs = len(active_texts)
        counts = _cache_counts(project_root, session_id, granularity, cache_tag=cache_tag)
        cached_texts = [str(x) for x in counts.get("passage_texts", []) or [] if str(x).strip()]
        cached_set = set(cached_texts)
        active_set = set(active_texts)
        missing_texts = [text for text in active_texts if text not in cached_set]
        stale_texts = [text for text in cached_texts if text not in active_set]
        passages = _safe_int(counts.get("passages"), 0)
        entities = _safe_int(counts.get("entities"), 0)
        facts = _safe_int(counts.get("facts"), 0)
        graph_ok = bool(counts.get("graph_pickle_exists"))
        reason = "ok"
        action = "none"
        level_healthy = True
        if expected_docs == 0:
            reason = "no_active_captions"
        elif passages == 0:
            reason = "missing_cache"
            action = "append_or_build"
            level_healthy = False
        elif stale_texts:
            reason = "cache_has_stale_passages"
            action = "reconcile_dirty_window"
            level_healthy = False
        elif missing_texts:
            reason = "cache_missing_new_passages"
            action = "append_missing"
            level_healthy = False
        elif not graph_ok:
            reason = "graph_pickle_missing"
            action = "reconcile_dirty_window"
            level_healthy = False
        elif entities == 0 and facts == 0:
            reason = "entity_fact_cache_empty"
            action = "reconcile_dirty_window"
            level_healthy = False
        levels[granularity] = {
            **counts,
            "passage_texts": None,
            "caption_path": str(caption_path),
            "expected_docs": expected_docs,
            "missing_doc_count": len(missing_texts),
            "stale_doc_count": len(stale_texts),
            "appendable_in_place": bool(missing_texts and not stale_texts and graph_ok and not (entities == 0 and facts == 0)),
            "healthy": level_healthy,
            "reason": reason,
            "recommended_action": action,
        }
        healthy = healthy and level_healthy
    return {
        "session_id": session_id,
        "cache_tag": cache_tag or f"online_{session_id}",
        "healthy": healthy,
        "levels": levels,
        "checked_at": utc_now_iso(),
    }


@contextmanager
def _temporary_env(updates: dict[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _caption_file_map(session_dir: Path) -> dict[str, str]:
    session_id = session_dir.name
    result: dict[str, str] = {}
    for granularity in GRANULARITIES:
        path = session_dir / "em2mem" / "caption_root" / f"{session_id}_{granularity}.json"
        if path.exists():
            result[granularity] = str(path)
    return result


def _active_caption_texts(session_dir: Path, granularity: str) -> list[str]:
    path = session_dir / "em2mem" / "caption_root" / f"{session_dir.name}_{granularity}.json"
    return list(dict.fromkeys([_caption_text(item) for item in _load_json_list(path) if _caption_text(item)]))


def _max_end_timestamp_int(session_dir: Path) -> int:
    session_id = session_dir.name
    max_end = 0.0
    for granularity in GRANULARITIES:
        path = session_dir / "em2mem" / "caption_root" / f"{session_id}_{granularity}.json"
        for item in _load_json_list(path):
            try:
                max_end = max(max_end, float(item.get("end") or 0.0))
            except Exception:
                continue
    # Em2Mem CaptionEntry.timestamp_int includes a DAY1 offset. Query code
    # builds the same shape via query_rag.build_until_timestamp("DAY1", ...).
    return 100000000 + int(seconds_to_hhmmssff(max_end))


def _granularities_to_refresh(health: dict[str, Any], force: bool) -> list[str]:
    selected: list[str] = []
    for granularity, info in (health.get("levels") or {}).items():
        expected_docs = _safe_int(info.get("expected_docs"), 0)
        if expected_docs <= 0:
            continue
        action = str(info.get("recommended_action") or "")
        if not force and action not in {"rebuild_granularity", "append_missing", "append_or_build", "reconcile_dirty_window"}:
            continue
        selected.append(str(granularity))
    return selected


def _replace_granularity_cache_dirs(
    *,
    project_root: Path,
    session_id: str,
    source_tag: str,
    target_tag: str,
    granularities: list[str],
) -> list[str]:
    replaced: list[str] = []
    cache_root = project_root / ".cache" / "episodic_memory"
    for granularity in granularities:
        src = cache_root / source_tag / granularity
        dst = cache_root / target_tag / granularity
        if not src.exists():
            continue
        tmp_dst = dst.with_name(f"{dst.name}.swap")
        old_dst = dst.with_name(f"{dst.name}.old")
        if tmp_dst.exists():
            shutil.rmtree(tmp_dst)
        if old_dst.exists():
            shutil.rmtree(old_dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp_dst)
        if dst.exists():
            dst.rename(old_dst)
        tmp_dst.rename(dst)
        if old_dst.exists():
            shutil.rmtree(old_dst)
        replaced.append(granularity)
    return replaced


def _copy_granularity_cache_dirs(
    *,
    project_root: Path,
    source_tag: str,
    target_tag: str,
    granularities: list[str],
) -> list[str]:
    copied: list[str] = []
    cache_root = project_root / ".cache" / "episodic_memory"
    for granularity in granularities:
        src = cache_root / source_tag / granularity
        dst = cache_root / target_tag / granularity
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        copied.append(granularity)
    return copied


def _build_em2mem_memory_for_cache(
    *,
    session_dir: Path,
    cache_tag: str,
    model_name: str | None,
):
    from em2mem.embedding import EmbeddingModel
    from em2mem.llm import LLMModel, PromptTemplateManager
    from em2mem.memory import EM2Memory

    embedding_model = EmbeddingModel()
    llm = LLMModel(model_name=model_name or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4")
    em2mem_memory = EM2Memory(
        embedding_model=embedding_model,
        retriever_llm_model=llm,
        respond_llm_model=llm,
        prompt_template_manager=PromptTemplateManager(),
        max_rounds=1,
        max_errors=3,
        episodic_cache_tag=cache_tag,
    )
    em2mem_memory.load_episodic_captions(caption_files=_caption_file_map(session_dir))
    return em2mem_memory


def _sync_granularity_in_tmp(
    *,
    em2mem_memory: Any,
    session_dir: Path,
    granularity: str,
    info: dict[str, Any],
    tmp_tag: str,
    project_root: Path,
) -> dict[str, Any]:
    active_docs = _active_caption_texts(session_dir, granularity)
    cached_docs = _cache_counts(project_root, session_dir.name, granularity, cache_tag=tmp_tag).get("passage_texts") or []
    cached_docs = [str(x) for x in cached_docs if str(x).strip()]
    stale_docs = [text for text in cached_docs if text not in set(active_docs)]
    missing_docs = [text for text in active_docs if text not in set(cached_docs)]
    action = str(info.get("recommended_action") or "")
    result = {
        "granularity": granularity,
        "action": action,
        "active_docs": len(active_docs),
        "stale_docs": len(stale_docs),
        "missing_docs": len(missing_docs),
        "mode": "copy_on_write_append",
    }
    force_rebuild = action in {"append_or_build", "reconcile_dirty_window"} and (
        str(info.get("reason")) in {"entity_fact_cache_empty", "graph_pickle_missing"}
    )
    if force_rebuild:
        cache_dir = project_root / ".cache" / "episodic_memory" / tmp_tag / granularity
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        getattr(em2mem_memory.episodic_memory, "hipporag", {}).pop(granularity, None)
        hipporag = em2mem_memory.episodic_memory._get_or_create_hipporag(granularity)
        hipporag.update(docs=active_docs)
        hipporag.prepare_retrieval_objects()
        result["mode"] = "copy_on_write_rebuild_corrupt_level"
        return result

    hipporag = em2mem_memory.episodic_memory._get_or_create_hipporag(granularity)

    if stale_docs:
        try:
            hipporag.delete(stale_docs)
            result["mode"] = "copy_on_write_delete_stale_then_append"
        except Exception:
            cache_dir = project_root / ".cache" / "episodic_memory" / tmp_tag / granularity
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            getattr(em2mem_memory.episodic_memory, "hipporag", {}).pop(granularity, None)
            hipporag = em2mem_memory.episodic_memory._get_or_create_hipporag(granularity)
            hipporag.update(docs=active_docs)
            hipporag.prepare_retrieval_objects()
            result["mode"] = "copy_on_write_rebuild_after_delete_failed"
            return result

    if missing_docs:
        hipporag.update(docs=missing_docs)
    elif not stale_docs and action in {"append_or_build", "reconcile_dirty_window"}:
        hipporag.update(docs=active_docs)
    hipporag.prepare_retrieval_objects()
    return result


def refresh_hipporag_cache(
    *,
    session_dir: Path,
    project_root: Path | None = None,
    model_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Append/reconcile HippoRAG's own parquet/vector/graph cache from active captions.

    This is intended for memory worker / CLI execution, never query worker.
    It performs copy-on-write append for missing passages and only rebuilds
    affected granularity caches when stale/corrupt state prevents native append.
    """
    session_dir = Path(session_dir)
    project_root = Path(project_root) if project_root else PROJECT_ROOT
    if not _env_bool("EM2MEM_INCREMENTAL_REFRESH_HIPPORAG_CACHE", True):
        return {"status": "disabled", "session_id": session_dir.name, "healthy": None}

    target_tag = f"online_{session_dir.name}"
    before = inspect_hipporag_cache_health(session_dir, project_root, cache_tag=target_tag)
    if before.get("healthy") and not force:
        return {
            "status": "already_healthy",
            "session_id": session_dir.name,
            "healthy": True,
            "before": before,
            "after": before,
            "update_mode": "incremental_append",
            "operation_results": [],
            "replaced_granularities": [],
        }

    refresh_granularities = _granularities_to_refresh(before, force)
    if not refresh_granularities:
        return {
            "status": "no_refresh_needed",
            "session_id": session_dir.name,
            "healthy": bool(before.get("healthy")),
            "before": before,
            "after": before,
            "update_mode": "incremental_append",
            "operation_results": [],
            "replaced_granularities": [],
        }

    tmp_tag = f"{target_tag}_tmp_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '').replace('+', '')}"
    tmp_root = project_root / ".cache" / "episodic_memory" / tmp_tag
    old_cwd = Path.cwd()
    os.chdir(project_root)
    operation_results: list[dict[str, Any]] = []
    replaced_granularities: list[str] = []
    try:
        with _temporary_env(
            {
                "EM2MEM_QUERY_STRICT_LOAD_ONLY": "0",
                "EM2MEM_QUERY_USE_CACHED_HIPPORAG": "0",
                "EM2MEM_QUERY_SKIP_REINDEX": "0",
            }
        ):
            _copy_granularity_cache_dirs(
                project_root=project_root,
                source_tag=target_tag,
                target_tag=tmp_tag,
                granularities=refresh_granularities,
            )
            world_memory = _build_em2mem_memory_for_cache(session_dir=session_dir, cache_tag=tmp_tag, model_name=model_name)
            for granularity in refresh_granularities:
                info = (before.get("levels") or {}).get(granularity) or {}
                operation_results.append(
                    _sync_granularity_in_tmp(
                        em2mem_memory=em2mem_memory,
                        session_dir=session_dir,
                        granularity=granularity,
                        info=info,
                        tmp_tag=tmp_tag,
                        project_root=project_root,
                    )
                )
            try:
                em2mem_memory.cleanup()
            except Exception:
                pass
        tmp_health = inspect_hipporag_cache_health(session_dir, project_root, cache_tag=tmp_tag)
        unhealthy_tmp = [
            granularity
            for granularity in refresh_granularities
            if not ((tmp_health.get("levels") or {}).get(granularity) or {}).get("healthy")
        ]
        if unhealthy_tmp:
            raise RuntimeError(f"temporary HippoRAG cache is still unhealthy for: {', '.join(unhealthy_tmp)}")
        replaced_granularities = _replace_granularity_cache_dirs(
            project_root=project_root,
            session_id=session_dir.name,
            source_tag=tmp_tag,
            target_tag=target_tag,
            granularities=refresh_granularities,
        )
    finally:
        os.chdir(old_cwd)
        if tmp_root.exists():
            shutil.rmtree(tmp_root)

    after = inspect_hipporag_cache_health(session_dir, project_root, cache_tag=target_tag)
    state_path = session_dir / "em2mem" / "incremental" / "hipporag_cache_state.json"
    payload = {
        "session_id": session_dir.name,
        "status": "healthy" if after.get("healthy") else "degraded",
        "healthy": bool(after.get("healthy")),
        "update_mode": "copy_on_write_append_reconcile",
        "operation_results": operation_results,
        "appended_granularities": [
            item.get("granularity")
            for item in operation_results
            if "append" in str(item.get("mode") or "") and _safe_int(item.get("missing_docs"), 0) > 0
        ],
        "reconciled_granularities": [
            item.get("granularity")
            for item in operation_results
            if item.get("mode") in {"copy_on_write_delete_stale_then_append", "copy_on_write_rebuild_corrupt_level", "copy_on_write_rebuild_after_delete_failed"}
        ],
        "replaced_granularities": replaced_granularities,
        "before": before,
        "after": after,
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(state_path, payload)

    config_path = session_dir / "em2mem" / "memory_config.json"
    config = read_json(config_path, default={})
    if isinstance(config, dict):
        config["hipporag_cache_ready"] = bool(after.get("healthy"))
        config["hipporag_cache_health"] = after
        config["hipporag_cache_update_mode"] = "copy_on_write_append_reconcile"
        config["hipporag_cache_state_path"] = "em2mem/incremental/hipporag_cache_state.json"
        config["updated_at"] = utc_now_iso()
        write_json_atomic(config_path, config)
    return payload
