from __future__ import annotations

import argparse
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from online_short_term.micro_event_refiner import MicroEventRefiner
from online_short_term.refine_lease import acquire_refine_event_lease
from online_short_term.mst_store import MSTStore
from online_short_term.refine_status import write_refine_status
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT, build_retrieval_text


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def refine_attempts(event: dict[str, Any]) -> int:
    refine = event.get("refine") if isinstance(event.get("refine"), dict) else {}
    try:
        return int(refine.get("refine_attempts", 0) or 0)
    except Exception:
        return 0


def is_auto_refine_eligible(
    event: dict[str, Any],
    *,
    now: datetime | None = None,
    max_attempts: int | None = None,
    retry_backoff_seconds: float | None = None,
) -> bool:
    status = str(event.get("status") or "provisional")
    stale = bool(event.get("needs_refine") or event.get("refined_stale"))
    if status == "provisional":
        return True
    if status in {"refined", "final"} and stale:
        return True
    if status != "refine_failed":
        return False
    refine = event.get("refine") if isinstance(event.get("refine"), dict) else {}
    error = str(refine.get("last_refine_error") or "").lower()
    if re.search(r"authentication|permissiondenied|invalid api.key|\b401\b|\b403\b", error):
        return False
    max_attempts = _env_int("EM2MEM_MST_REFINE_MAX_ATTEMPTS", 3) if max_attempts is None else int(max_attempts)
    if max_attempts >= 0 and refine_attempts(event) >= max_attempts:
        return False
    transient = bool(re.search(r"connection|timeout|timed out|ratelimit|rate.limit|\b429\b|\b50[234]\b|empty.*response", error))
    default_backoff = min(120.0, 15.0 * 2 ** max(0, min(3, refine_attempts(event)-1))) if transient else 300.0
    retry_backoff_seconds = (
        _env_float("EM2MEM_MST_REFINE_RETRY_BACKOFF_SECONDS", default_backoff)
        if retry_backoff_seconds is None
        else float(retry_backoff_seconds)
    )
    refine = event.get("refine") if isinstance(event.get("refine"), dict) else {}
    last_refine_at = _parse_timestamp(refine.get("last_refine_failed_at") or refine.get("last_refine_at"))
    if last_refine_at is None or retry_backoff_seconds <= 0:
        return True
    now = now or datetime.now(timezone.utc)
    return (now - last_refine_at).total_seconds() >= retry_backoff_seconds


def _select_events(
    events: list[dict],
    *,
    event_id: str | None,
    event_ids: list[str] | None,
    limit_events: int,
    force_refine: bool,
) -> list[dict]:
    allowed_event_ids = {str(value) for value in (event_ids or []) if str(value or "").strip()}
    selected = []
    for event in events:
        current_event_id = str(event.get("event_id") or "")
        if event_id and current_event_id != event_id:
            continue
        if allowed_event_ids and current_event_id not in allowed_event_ids:
            continue
        if not force_refine and not is_auto_refine_eligible(event):
            continue
        selected.append(event)
        if event_id:
            break
        if allowed_event_ids and len(selected) >= len(allowed_event_ids):
            break
        if limit_events > 0 and len(selected) >= limit_events:
            break
    return selected


def _load_event_by_id(store: MSTStore, event_id: str) -> dict[str, Any] | None:
    event_id = str(event_id or "")
    for events in (store.load_archive_events(), store.load_events()):
        for event in events:
            if str(event.get("event_id") or "") == event_id:
                return event
    return None


_REFINE_RESULT_FIELDS = {
    "event_caption_refined",
    "visual_objects",
    "main_actions",
    "state_changes",
    "visual_observations",
    "entities",
    "confidence",
    "caption_source",
    "refine_completed_at",
    "refine_speed",
    "refine",
}


def _merge_refine_result(
    source_event: dict[str, Any],
    latest_event: dict[str, Any],
    refined_event: dict[str, Any],
) -> dict[str, Any]:
    """Apply model output without overwriting transcript updates made in flight."""

    merged = dict(latest_event)
    for field in _REFINE_RESULT_FIELDS:
        if field in refined_event:
            merged[field] = refined_event[field]

    source_transcript_version = int(source_event.get("transcript_version", 0) or 0)
    latest_transcript_version = int(latest_event.get("transcript_version", 0) or 0)
    transcript_changed = latest_transcript_version != source_transcript_version
    if str(refined_event.get("status") or "") == "refine_failed":
        merged["status"] = "refine_failed"
        merged["needs_refine"] = True
    else:
        merged["status"] = "refined"
        merged["needs_refine"] = bool(transcript_changed)
        merged["refined_stale"] = bool(transcript_changed)
        merged["stale_reason"] = "transcript_changed_during_refine" if transcript_changed else None

    merged["version"] = max(
        int(latest_event.get("version", 1) or 1) + 1,
        int(refined_event.get("version", 1) or 1),
    )
    merged["updated_at"] = refined_event.get("updated_at") or latest_event.get("updated_at")
    merged["retrieval_text"] = build_retrieval_text(merged)
    return merged


def refine_session(
    *,
    session_id: str,
    sessions_root: Path,
    backend: str,
    limit_events: int,
    event_id: str | None = None,
    event_ids: list[str] | None = None,
    force_refine: bool = False,
    only_active: bool = False,
    only_archive: bool = False,
    task_id: str | None = None,
    task_queued_at: str | None = None,
    task_worker_started_at: str | None = None,
    task_reason: str | None = None,
    verbose: bool = False,
) -> dict:
    session_dir = sessions_root / session_id
    store = MSTStore(session_dir)
    with store.mutation_lock():
        source_events = store.load_events() if only_active else store.load_archive_events()
        if not source_events and not only_archive:
            source_events = store.load_events()
    selected = _select_events(
        source_events,
        event_id=event_id,
        event_ids=event_ids,
        limit_events=limit_events,
        force_refine=force_refine,
    )
    refined = []
    skipped_already_refined: list[str] = []
    skipped_lease_timeout: list[str] = []
    update_results: list[dict[str, bool]] = []
    max_concurrency = max(1, int(os.getenv("EM2MEM_REFINE_MAX_CONCURRENCY", "1") or 1))

    def _refine_one(event: dict) -> tuple[str, str, dict[str, Any] | None, dict[str, bool] | None]:
        event_id_value = str(event.get("event_id") or "")
        lease_owner = f"{task_id or 'manual'}:{event_id_value}"
        lease_wait_seconds = (
            _env_float("EM2MEM_MST_REFINE_FORCE_LEASE_WAIT_SECONDS", 150.0)
            if force_refine
            else _env_float("EM2MEM_MST_REFINE_LEASE_WAIT_SECONDS", 2.0)
        )
        with acquire_refine_event_lease(
            session_dir,
            event_id_value,
            owner=lease_owner,
            wait_seconds=lease_wait_seconds,
        ) as lease:
            if lease is None:
                return "lease_timeout", event_id_value, None, None

            with store.mutation_lock():
                latest = _load_event_by_id(store, event_id_value)
            if latest is None:
                return "missing", event_id_value, None, None

            selected_version = int(event.get("version", 1) or 1)
            latest_version = int(latest.get("version", 1) or 1)
            completed_by_other_task = (
                latest_version > selected_version
                and not latest.get("needs_refine")
                and not latest.get("refined_stale")
                and str(latest.get("status") or "") in {"refined", "final"}
            )
            if completed_by_other_task or (not force_refine and not is_auto_refine_eligible(latest)):
                return "already_refined", event_id_value, None, None

            updated = MicroEventRefiner(backend=backend).refine_event(
                latest,
                session_dir,
                task_id=task_id,
                task_queued_at=task_queued_at,
                task_worker_started_at=task_worker_started_at,
                task_reason=task_reason,
            )
            with store.mutation_lock():
                latest_before_write = _load_event_by_id(store, event_id_value) or latest
                persisted = _merge_refine_result(latest, latest_before_write, updated)
                if only_active:
                    active = store._merge_by_event_id(store.load_events(), [persisted])
                    store.save_events(active)
                    update_result = {"active_updated": True, "archive_updated": False}
                elif only_archive:
                    archive = store._merge_by_event_id(store.load_archive_events(), [persisted])
                    store.save_archive_events(archive, bump_version=True)
                    update_result = {"active_updated": False, "archive_updated": True}
                else:
                    update_result = store.update_events([persisted])
            return "refined", event_id_value, persisted, update_result

    def _record_result(result: tuple[str, str, dict[str, Any] | None, dict[str, bool] | None]) -> None:
        status, event_id_value, updated, update_result = result
        if status == "refined" and updated is not None:
            refined.append(updated)
            if update_result is not None:
                update_results.append(update_result)
            if verbose:
                print(f"[mst_refine] {updated.get('event_id')} status={updated.get('status')} source={updated.get('caption_source')}")
        elif status == "already_refined" and updated is None:
            skipped_already_refined.append(event_id_value)
        elif status == "lease_timeout":
            skipped_lease_timeout.append(event_id_value)

    if max_concurrency > 1 and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(selected))) as executor:
            futures = [executor.submit(_refine_one, event) for event in selected]
            for future in as_completed(futures):
                _record_result(future.result())
    else:
        for event in selected:
            _record_result(_refine_one(event))
    update_result = {
        "active_updated": any(item.get("active_updated") for item in update_results),
        "archive_updated": any(item.get("archive_updated") for item in update_results),
    }
    with store.mutation_lock():
        windows_path, refine_state_path = write_refine_status(store)
    return {
        "session_id": session_id,
        "backend": backend,
        "selected_event_count": len(selected),
        "refined_event_count": len(refined),
        "selected_event_ids": [event.get("event_id") for event in selected],
        "refined_event_ids": [event.get("event_id") for event in refined if event.get("status") in {"refined", "final"}],
        "skipped_already_refined_count": len(skipped_already_refined),
        "skipped_already_refined_ids": skipped_already_refined,
        "skipped_lease_timeout_count": len(skipped_lease_timeout),
        "skipped_lease_timeout_ids": skipped_lease_timeout,
        "max_concurrency": max_concurrency,
        "update_result": update_result,
        "mst_state": store.get_state(),
        "archive_state": store.get_archive_state(),
        "refine_state_path": str(refine_state_path),
        "refined_ready_windows_path": str(windows_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine provisional M_st micro-events asynchronously.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--backend", default=os.getenv("EM2MEM_MST_REFINE_BACKEND", "openai"), choices=["mock", "openai"])
    parser.add_argument("--limit-events", type=int, default=10)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--force-refine", action="store_true")
    parser.add_argument("--only-active", action="store_true")
    parser.add_argument("--only-archive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.only_active and args.only_archive:
        raise SystemExit("--only-active and --only-archive are mutually exclusive")
    result = refine_session(
        session_id=args.session_id,
        sessions_root=Path(args.sessions_root),
        backend=args.backend,
        limit_events=args.limit_events,
        event_id=args.event_id,
        force_refine=args.force_refine,
        only_active=args.only_active,
        only_archive=args.only_archive,
        verbose=args.verbose,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
