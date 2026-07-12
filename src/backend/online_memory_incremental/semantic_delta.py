from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from online_memory.evidence_to_em2mem import _dedupe_triplets
from online_memory.em2mem_layout import seconds_to_hhmmssff
from online_preprocess.io_utils import ensure_dir, read_json, utc_now_iso, write_json_atomic


def _model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("EM2MEM_MEMORY_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.4"


def _slug(value: Any) -> str:
    import re

    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text[:80] or "item"


def _load_graph_delta(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm(value: Any) -> str:
    import re

    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _fact_key(fact: dict[str, Any]) -> tuple[str, str, str]:
    triple = fact.get("triple") if isinstance(fact.get("triple"), list) else None
    if triple and len(triple) >= 3:
        return (_norm(triple[0]), _norm(triple[1]), _norm(triple[2]))
    return (_norm(fact.get("head")), _norm(fact.get("relation")), _norm(fact.get("tail")))


def _merge_unique(left: Any, right: Any) -> list[Any]:
    values = []
    seen = set()
    for item in list(left or []) + list(right or []):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        values.append(item)
    return values


def _time_key(value: dict[str, Any] | None, default: str) -> str:
    if not isinstance(value, dict):
        return default
    return str(value.get("date", "DAY1")) + str(value.get("start_time", "")) + str(value.get("end_time", ""))


def _consolidate_semantic_facts(facts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    removed = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        key = _fact_key(fact)
        if not any(key):
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = dict(fact)
            continue
        removed += 1
        existing["support_docs"] = _merge_unique(existing.get("support_docs"), fact.get("support_docs"))
        existing["source_doc_ids"] = _merge_unique(existing.get("source_doc_ids"), fact.get("source_doc_ids"))
        existing["evidence_event_ids"] = _merge_unique(existing.get("evidence_event_ids"), fact.get("evidence_event_ids"))
        existing["provenance_root_ids"] = _merge_unique(existing.get("provenance_root_ids"), fact.get("provenance_root_ids"))
        existing["support_days"] = _merge_unique(existing.get("support_days"), fact.get("support_days"))
        existing["support_scales"] = _merge_unique(existing.get("support_scales"), fact.get("support_scales"))
        existing["raw_support_count"] = int(existing.get("raw_support_count") or 1) + int(fact.get("raw_support_count") or 1)
        existing["support_count"] = len(existing.get("support_docs") or existing.get("source_doc_ids") or [])
        existing["confidence"] = max(float(existing.get("confidence") or 0.0), float(fact.get("confidence") or 0.0))
        first_existing = existing.get("first_seen") if isinstance(existing.get("first_seen"), dict) else None
        first_new = fact.get("first_seen") if isinstance(fact.get("first_seen"), dict) else None
        last_existing = existing.get("last_seen") if isinstance(existing.get("last_seen"), dict) else None
        last_new = fact.get("last_seen") if isinstance(fact.get("last_seen"), dict) else None
        if first_new and _time_key(first_new, "zz") < _time_key(first_existing, "zz"):
            existing["first_seen"] = first_new
        if last_new and _time_key(last_new, "") > _time_key(last_existing, ""):
            existing["last_seen"] = last_new
        versions = [int(v) for v in [existing.get("semantic_version"), fact.get("semantic_version")] if str(v).isdigit()]
        if versions:
            existing["semantic_version"] = max(versions)
        existing["consolidated_at"] = utc_now_iso()
    return list(by_key.values()), {"input_count": len(facts), "output_count": len(by_key), "dedup_removed": removed}


def generate_semantic_delta(
    *,
    session_dir: Path,
    version: int,
    graph_delta_path: Path,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = _model_name(model_name)
    rows = _load_graph_delta(graph_delta_path)
    delta_dir = session_dir / "em2mem" / "incremental" / "semantic" / "deltas"
    delta_path = delta_dir / f"semantic_delta_v{version:06d}.jsonl"
    state_path = session_dir / "em2mem" / "incremental" / "semantic" / "semantic_state.json"
    semantic_root = session_dir / "em2mem" / "semantic_root"
    candidate_path = semantic_root / "semantic_candidates.jsonl"
    memory_path = semantic_root / f"semantic_memory_{model}.json"

    facts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        doc_id = str(row.get("source_doc_id") or "")
        episode_id = str(row.get("source_episode_id") or doc_id)
        start_code = seconds_to_hhmmssff(float(row.get("start_time") or 0.0))
        end_code = seconds_to_hhmmssff(float(row.get("end_time") or row.get("start_time") or 0.0))
        triples = _dedupe_triplets(row.get("new_facts") or [])
        for idx, triple in enumerate(triples):
            fact_id = f"fact_delta_v{version:06d}_{_slug(doc_id)}_{idx:03d}"
            text = f"{triple[0]} {triple[1]} {triple[2]}"
            fact = {
                "fact_id": fact_id,
                "triple": triple,
                "head": triple[0],
                "relation": triple[1],
                "tail": triple[2],
                "head_type": "entity",
                "tail_type": "entity",
                "semantic_summary": text,
                "support_count": 1,
                "support_days": ["DAY1"],
                "support_scales": ["30sec"],
                "confidence": row.get("confidence", 0.7),
                "habit_strength": "low",
                "raw_support_count": 1,
                "support_docs": [doc_id],
                "evidence_event_ids": [doc_id],
                "provenance_root_ids": [doc_id],
                "source_doc_ids": [doc_id],
                "source_episode_id": episode_id,
                "first_seen": {"date": "DAY1", "start_time": start_code, "end_time": end_code},
                "last_seen": {"date": "DAY1", "start_time": start_code, "end_time": end_code},
                "semantic_version": version,
                "created_at": utc_now_iso(),
            }
            facts.append(fact)
            candidates.append(
                {
                    "fact_id": fact_id,
                    "session_id": session_dir.name,
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "fact_type": "incremental_graph_delta",
                    "text": text,
                    "entities": [triple[0], triple[2]],
                    "evidence_doc_id": doc_id,
                    "source_episode_id": episode_id,
                    "semantic_version": version,
                }
            )

    ensure_dir(delta_path.parent)
    with delta_path.open("w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    if candidates:
        _append_jsonl(candidate_path, candidates)

    memory = read_json(memory_path, default={})
    if not isinstance(memory, dict):
        memory = {"facts": [], "timeline": []}
    existing_facts = memory.get("facts") if isinstance(memory.get("facts"), list) else []
    new_ids = {str(fact.get("fact_id")) for fact in facts}
    current_doc_ids = {str(row.get("source_doc_id") or "") for row in rows if row.get("source_doc_id")}
    merged_facts = []
    for fact in existing_facts:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("fact_id") or "")
        support_docs = {str(x) for x in (fact.get("support_docs") or fact.get("source_doc_ids") or [])}
        if fact_id in new_ids:
            continue
        if fact_id.startswith("fact_delta_v") and support_docs.intersection(current_doc_ids):
            continue
        merged_facts.append(fact)
    merged_facts.extend(facts)
    merged_facts, consolidation_summary = _consolidate_semantic_facts(merged_facts)
    timeline = memory.get("timeline") if isinstance(memory.get("timeline"), list) else []
    timeline = [
        item for item in timeline
        if not (
            isinstance(item, dict)
            and any((fid in new_ids or str(fid).startswith("fact_delta_v")) for fid in (item.get("fact_ids") or []))
        )
    ]
    for fact in facts:
        ts = f"1{str(fact.get('last_seen', {}).get('end_time') or '').replace(':', '')}"
        timeline.append({"timestamp": ts, "fact_ids": [fact["fact_id"]], "source": "incremental_delta"})
    memory.update(
        {
            "facts": merged_facts,
            "timeline": timeline,
            "source": "em2mem_incremental_semantic_delta",
            "semantic_memory_ready": True,
            "semantic_generation_backend": "incremental_graph_delta",
            "semantic_consolidation": "deterministic_incremental_global_dedupe",
            "semantic_consolidation_summary": consolidation_summary,
            "latest_semantic_ready_version": version,
            "updated_at": utc_now_iso(),
        }
    )
    write_json_atomic(memory_path, memory)
    consolidation_state_path = session_dir / "em2mem" / "incremental" / "semantic" / "semantic_consolidation_state.json"
    write_json_atomic(
        consolidation_state_path,
        {
            "session_id": session_dir.name,
            "semantic_version": version,
            "backend": "deterministic_incremental_global_dedupe",
            **consolidation_summary,
            "semantic_memory_path": memory_path.relative_to(session_dir).as_posix(),
            "updated_at": utc_now_iso(),
        },
    )

    state = {
        "session_id": session_dir.name,
        "semantic_version": version,
        "latest_semantic_ready_version": version,
        "building_semantic_version": None,
        "semantic_lagging": False,
        "delta_path": delta_path.relative_to(session_dir).as_posix(),
        "semantic_memory_path": memory_path.relative_to(session_dir).as_posix(),
        "semantic_candidate_path": candidate_path.relative_to(session_dir).as_posix(),
        "new_fact_count": len(facts),
        "semantic_consolidation_summary": consolidation_summary,
        "updated_at": utc_now_iso(),
    }
    write_json_atomic(state_path, state)
    return state
