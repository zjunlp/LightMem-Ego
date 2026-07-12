from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from online_memory.evidence_to_em2mem import (
    _build_graph_payload,
    _dedupe_triplets,
    _openie_triplets_for_items,
    _triplets_for_item,
)
from online_memory.em2mem_layout import Em2MemOnlineLayout
from online_preprocess.io_utils import ensure_dir, read_json, utc_now_iso, write_json_atomic


def _model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"


def _generation_backend(value: str | None = None) -> str:
    backend = (value or os.getenv("EM2MEM_INCREMENTAL_GRAPH_BACKEND") or os.getenv("EM2MEM_MEMORY_GENERATION_BACKEND") or "llm").strip().lower()
    return backend if backend in {"llm", "rule"} else "llm"


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_triplet_payload(path: Path) -> dict[str, Any]:
    data = read_json(path, default={})
    return data if isinstance(data, dict) else {}


def _merge_active_30s_sidecar(
    *,
    layout: Em2MemOnlineLayout,
    model_name: str,
    caption_30s: list[dict[str, Any]],
    triplet_map_delta: dict[str, list[list[str]]],
    backend: str,
) -> dict[str, str]:
    sidecar_dir = layout.sidecar_root / "30s"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    triplet_path = sidecar_dir / f"episodic_triplets_30s_{model_name}.json"
    graph_path = sidecar_dir / f"episodic_graph_30s_{model_name}.json"
    payload = _load_triplet_payload(triplet_path)
    triplet_map = payload.get("triplet_map") if isinstance(payload.get("triplet_map"), dict) else {}
    triplet_map = {str(k): _dedupe_triplets(v or []) for k, v in triplet_map.items()}
    for doc_id, triples in triplet_map_delta.items():
        existing = triplet_map.get(str(doc_id), [])
        triplet_map[str(doc_id)] = _dedupe_triplets(existing + (triples or []))

    units = []
    for idx, item in enumerate(caption_30s):
        doc_id = str(item.get("doc_id") or "")
        triples = triplet_map.get(doc_id, [])
        units.append(
            {
                "doc_id": doc_id,
                "date": item.get("date", "DAY1"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "text": item.get("text", ""),
                "fine_caption": item.get("fine_caption", ""),
                "video_path": item.get("video_path", ""),
                "source_doc_ids": item.get("source_doc_ids", []),
                "child_ids": item.get("child_ids", []),
                "openie_results": triples,
                "raw_triplets": triples,
                "episodic_triplets": triples,
                "idx": idx,
            }
        )
    write_json_atomic(
        triplet_path,
        {
            "scale": "30s",
            "units": units,
            "triplet_map": triplet_map,
            "source": f"online_incremental_{backend}_openie",
            "updated_at": utc_now_iso(),
        },
    )
    write_json_atomic(graph_path, _build_graph_payload(caption_30s, triplet_map, "30sec"))
    return {"triplets": str(triplet_path), "graph": str(graph_path)}


def generate_graph_delta(
    *,
    session_dir: Path,
    version: int,
    new_caption_items: list[dict[str, Any]],
    all_caption_items: list[dict[str, Any]],
    model_name: str | None = None,
    generation_backend: str | None = None,
) -> dict[str, Any]:
    model = _model_name(model_name)
    backend = _generation_backend(generation_backend)
    layout = Em2MemOnlineLayout(session_dir=session_dir, session_id=session_dir.name)
    delta_dir = session_dir / "em2mem" / "incremental" / "graph" / "deltas"
    state_path = session_dir / "em2mem" / "incremental" / "graph" / "graph_state.json"
    delta_path = delta_dir / f"graph_delta_v{version:06d}.jsonl"
    error: str | None = None

    try:
        if backend == "llm":
            triplet_map = _openie_triplets_for_items(new_caption_items, model, delta_dir)
        else:
            triplet_map = {str(item["doc_id"]): _triplets_for_item(item) for item in new_caption_items}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        backend = "rule"
        triplet_map = {str(item["doc_id"]): _triplets_for_item(item) for item in new_caption_items}

    rows: list[dict[str, Any]] = []
    for item in new_caption_items:
        doc_id = str(item.get("doc_id") or "")
        triples = _dedupe_triplets(triplet_map.get(doc_id, []))
        entities = sorted({x for tri in triples for x in (tri[0], tri[2]) if x})
        rows.append(
            {
                "graph_version": version,
                "source_episode_id": item.get("episode_id") or item.get("episode_ids") or doc_id,
                "source_doc_id": doc_id,
                "new_entities": entities,
                "merged_entities": [],
                "new_facts": triples,
                "new_edges": [{"head": h, "relation": r, "tail": t} for h, r, t in triples],
                "start_time": item.get("start"),
                "end_time": item.get("end"),
                "confidence": float(item.get("confidence") or 0.75),
                "backend": backend,
                "created_at": utc_now_iso(),
            }
        )
    _jsonl_write(delta_path, rows)
    active_paths = _merge_active_30s_sidecar(
        layout=layout,
        model_name=model,
        caption_30s=all_caption_items,
        triplet_map_delta=triplet_map,
        backend=backend,
    )
    state = {
        "session_id": session_dir.name,
        "graph_version": version,
        "latest_graph_ready_version": version,
        "building_graph_version": None,
        "graph_lagging": False,
        "delta_path": delta_path.relative_to(session_dir).as_posix(),
        "active_sidecar_paths": {k: str(Path(v).relative_to(session_dir)) for k, v in active_paths.items()},
        "new_episode_count": len(new_caption_items),
        "new_fact_count": sum(len(row.get("new_facts") or []) for row in rows),
        "backend": backend,
        "error": error,
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(state_path, state)
    return state
