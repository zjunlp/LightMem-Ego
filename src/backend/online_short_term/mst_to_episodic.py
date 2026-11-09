from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, relative_to_session, utc_now_iso, write_json, write_json_atomic
from online_short_term.consolidation_state import load_consolidation_state, memory_handoff, write_consolidation_state
from online_short_term.episodic_window_builder import MSTEpisodicWindowBuilder
from online_short_term.mst_store import MSTStore
from online_short_term.refine_status import write_refine_status
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT
from online_pipeline.rokid_day import apply_demo_day_fields
from online_memory_edit.lock import session_memory_mutation_lock


REFINED_STATUSES = {"refined", "final"}


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _window_episode_id(window: dict[str, Any]) -> str:
    start = int(round(float(window.get("start_time", 0.0) or 0.0)))
    end = int(round(float(window.get("end_time", 0.0) or 0.0)))
    return f"mst_ep_{start:06d}_{end:06d}"


def _window_segment_id(window: dict[str, Any]) -> str:
    start = int(round(float(window.get("start_time", 0.0) or 0.0)))
    end = int(round(float(window.get("end_time", 0.0) or 0.0)))
    return f"seg_{start:06d}_{end:06d}"


def _event_ids_for_window(window: dict[str, Any]) -> list[str]:
    ids = window.get("event_ids", []) or window.get("source_micro_event_ids", []) or []
    return [str(event_id) for event_id in ids if event_id]


def _load_existing_episodes(path: Path) -> list[dict[str, Any]]:
    data = read_json(path, default=[])
    return data if isinstance(data, list) else []


def _episode_version(existing: list[dict[str, Any]], episode_id: str, force: bool) -> int:
    versions = [int(item.get("version", 1) or 1) for item in existing if item.get("episode_id") == episode_id]
    if not versions:
        return 1
    # An already-completed window is skipped before generation unless forced
    # or reopened by source invalidation. Either replacement is a new source
    # revision, including the ordinary retry of a reopened window.
    return max(versions) + 1


def _replace_or_append_episode(existing: list[dict[str, Any]], episode: dict[str, Any]) -> list[dict[str, Any]]:
    episode_id = episode.get("episode_id")
    kept = [item for item in existing if item.get("episode_id") != episode_id]
    kept.append(episode)
    return sorted(kept, key=lambda item: (float(item.get("start_time", 0.0)), str(item.get("episode_id", ""))))


def _caption_doc_from_episode(episode: dict[str, Any]) -> dict[str, Any]:
    output = {
        "episode_id": episode.get("episode_id"),
        "version": int(episode.get("version", 1) or 1),
        "doc_id": episode.get("episode_id"),
        "segment_id": episode.get("segment_id"),
        "session_id": episode.get("session_id"),
        "start_time": episode.get("start_time"),
        "end_time": episode.get("end_time"),
        "video_path": "input.mp4",
        "clip_path": "",
        "caption": episode.get("caption"),
        "fine_caption": episode.get("fine_caption"),
        "visual_summary": episode.get("visual_summary", ""),
        "visual_summary_status": episode.get("visual_summary_status"),
        "visual_summary_debug": episode.get("visual_summary_debug", {}),
        "visual_observations": episode.get("visual_observations", []),
        "transcript": episode.get("transcript"),
        "transcript_segments": episode.get("transcript_segments", []),
        "scene": episode.get("scene"),
        "main_actions": episode.get("main_actions", []),
        "state_changes": episode.get("state_changes", []),
        "visual_objects": episode.get("visual_objects", []),
        "keyframe_paths": episode.get("keyframe_paths", []),
        "keyframe_captions": episode.get("keyframe_captions", []),
        "source_micro_event_ids": episode.get("source_micro_event_ids", []),
        "refined_event_count": episode.get("refined_event_count", 0),
        "completeness_score": episode.get("completeness_score", 0.0),
        "episodic_source": "mst_micro_events",
        "status": "complete",
        "confidence": episode.get("confidence"),
    }
    for key in (
        "date",
        "day_label",
        "weekday_label",
        "weekday_label_en",
        "display_day_label",
        "relative_day_start_ms",
        "local_start_time",
        "local_end_time",
    ):
        if episode.get(key) is not None:
            output[key] = episode[key]
    return output


def _evidence_doc_from_episode(episode: dict[str, Any]) -> dict[str, Any]:
    start = int(round(float(episode.get("start_time", 0.0) or 0.0)))
    end = int(round(float(episode.get("end_time", 0.0) or 0.0)))
    doc_id = f"mst_evd_{start:06d}_{end:06d}"
    output = {
        "doc_id": doc_id,
        "version": int(episode.get("version", 1) or 1),
        "evidence_doc_id": doc_id,
        "episode_id": episode.get("episode_id"),
        "session_id": episode.get("session_id"),
        "segment_id": episode.get("segment_id"),
        "start_time": episode.get("start_time"),
        "end_time": episode.get("end_time"),
        "caption": episode.get("caption"),
        "fine_caption": episode.get("fine_caption"),
        "visual_summary": episode.get("visual_summary", ""),
        "visual_summary_status": episode.get("visual_summary_status"),
        "visual_summary_debug": episode.get("visual_summary_debug", {}),
        "visual_observations": episode.get("visual_observations", []),
        "scene": episode.get("scene"),
        "transcript": episode.get("transcript"),
        "transcript_segments": episode.get("transcript_segments", []),
        "transcript_summary": episode.get("transcript_summary"),
        "keyframe_captions": episode.get("keyframe_captions", []),
        "visual_objects": episode.get("visual_objects", []),
        "main_actions": episode.get("main_actions", []),
        "state_changes": episode.get("state_changes", []),
        "entities": episode.get("entities", []),
        "source_micro_event_ids": episode.get("source_micro_event_ids", []),
        "source_micro_events": episode.get("source_micro_events", []),
        "keyframe_paths": episode.get("keyframe_paths", []),
        "source_video_path": "input.mp4",
        "clip_path": "",
        "confidence": episode.get("confidence", 1.0),
        "completeness_score": episode.get("completeness_score", 1.0),
        "refined_event_count": episode.get("refined_event_count", 0),
        "status": "complete",
        "episodic_source": "mst_micro_events",
    }
    for key in (
        "date",
        "day_label",
        "weekday_label",
        "weekday_label_en",
        "display_day_label",
        "relative_day_start_ms",
        "local_start_time",
        "local_end_time",
    ):
        if episode.get(key) is not None:
            output[key] = episode[key]
    return output


def _write_outputs(session_dir: Path, episodes: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Path]:
    em2mem_dir = session_dir / "em2mem" / "mst_episodic"
    captions_dir = session_dir / "captions"
    evidence_dir = session_dir / "evidence"
    episodes_path = em2mem_dir / "mst_30sec_episodes.json"
    episodes_jsonl_path = em2mem_dir / "mst_30sec_episodes.jsonl"
    mapping_path = em2mem_dir / "mst_to_episode_mapping.json"
    state_path = em2mem_dir / "mst_episodic_state.json"
    captioned_path = captions_dir / "mst_session_30sec_captioned.json"
    evidence_path = evidence_dir / "mst_session_evidence.json"

    write_json(episodes_path, episodes)
    _write_jsonl(episodes_jsonl_path, episodes)
    mapping = {
        "window_to_episode": {f"{item.get('start_time')}-{item.get('end_time')}": item.get("episode_id") for item in episodes},
        "event_to_episode": {
            event_id: item.get("episode_id")
            for item in episodes
            for event_id in item.get("source_micro_event_ids", []) or []
        },
    }
    write_json(mapping_path, mapping)
    write_json(captioned_path, [_caption_doc_from_episode(item) for item in episodes])
    write_json(evidence_path, [_evidence_doc_from_episode(item) for item in episodes])
    state = {
        "session_id": result.get("session_id"),
        "status": "ready" if episodes else "empty",
        "episode_count": len(episodes),
        "generated_episode_count": result.get("generated_episode_count", 0),
        "generated_episode_ids": result.get("generated_episode_ids", []),
        "backend": result.get("backend"),
        "updated_at": utc_now_iso(),
        "episodes_path": relative_to_session(episodes_path, session_dir),
        "captioned_30sec_path": relative_to_session(captioned_path, session_dir),
        "evidence_path": relative_to_session(evidence_path, session_dir),
    }
    write_json(state_path, state)
    return {
        "episodes": episodes_path,
        "episodes_jsonl": episodes_jsonl_path,
        "mapping": mapping_path,
        "state": state_path,
        "captioned": captioned_path,
        "evidence": evidence_path,
    }


def _update_memory_config_metadata(session_dir: Path, update_mode: str | None = None) -> None:
    # Share the publisher's lock for the complete read/modify/write. Callers
    # must release M_st's mutation lock before waiting for memory publication.
    with session_memory_mutation_lock(session_dir):
        _update_memory_config_metadata_unlocked(session_dir, update_mode)


def _update_memory_config_metadata_unlocked(session_dir: Path, update_mode: str | None = None) -> None:
    memory_config_path = session_dir / "em2mem" / "memory_config.json"
    if not memory_config_path.exists() and update_mode is None:
        return
    config = read_json(memory_config_path, default={})
    if not isinstance(config, dict):
        config = {}
    now = utc_now_iso()
    config.update(
        {
            "episodic_source": "mst_micro_events",
            "mst_episodic_ready": True,
            "mst_episodic_path": "em2mem/mst_episodic/mst_30sec_episodes.json",
            "mst_captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
            "mst_evidence_path": "evidence/mst_session_evidence.json",
            "em2mem_30s_input_source": "mst_session_30sec_captioned",
            "last_mst_episodic_build_at": now,
        }
    )
    if update_mode:
        config["em2mem_update_mode"] = update_mode
        config["last_mst_em2mem_update_at"] = now
    write_json_atomic(memory_config_path, config)


def _update_status_outputs(session_dir: Path, session_id: str) -> None:
    status_path = session_dir / "status.json"
    status = read_json(status_path, default={})
    if not isinstance(status, dict):
        status = {}
    outputs = dict(status.get("outputs") or {})
    outputs.update(
        {
            "mst_episodic_ready": True,
            "mst_episodic_path": "em2mem/mst_episodic/mst_30sec_episodes.json",
            "mst_captioned_30sec_path": "captions/mst_session_30sec_captioned.json",
            "mst_evidence_path": "evidence/mst_session_evidence.json",
        }
    )
    status.setdefault("session_id", session_id)
    status["outputs"] = outputs
    status["updated_at"] = utc_now_iso()
    write_json_atomic(status_path, status)


def _select_ready_windows(
    windows: list[dict[str, Any]],
    *,
    window_start: float | None = None,
    window_end: float | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        start = float(window.get("start_time", 0.0) or 0.0)
        end = float(window.get("end_time", start) or start)
        if window_start is not None and start < float(window_start):
            continue
        if window_end is not None and end > float(window_end):
            continue
        selected.append(window)
    return selected


def build_episodic_from_mst(
    *,
    session_id: str,
    sessions_root: Path = DEFAULT_SESSIONS_ROOT,
    backend: str = "openai",
    force: bool = False,
    dry_run: bool = False,
    queue_memory_handoff: bool = False,
    window_start: float | None = None,
    window_end: float | None = None,
    limit_windows: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    session_dir = Path(sessions_root) / session_id
    store = MSTStore(session_dir)
    windows_path, _ = write_refine_status(store)
    windows = read_json(windows_path, default=[])
    if not isinstance(windows, list):
        windows = []
    windows = _select_ready_windows(windows, window_start=window_start, window_end=window_end)
    archive_events = store.load_archive_events()
    events_by_id = {str(event.get("event_id")): event for event in archive_events if event.get("event_id")}
    state = load_consolidation_state(session_dir, session_id)
    window_to_episode = dict(state.get("window_to_episode", {}) or {})
    existing_path = session_dir / "em2mem" / "mst_episodic" / "mst_30sec_episodes.json"
    existing_episodes = _load_existing_episodes(existing_path)

    ready_windows = []
    skipped = []
    for window in windows:
        window_id = str(window.get("window_id") or f"win_{int(window.get('start_time', 0)):06d}_{int(window.get('end_time', 0)):06d}")
        if not window.get("is_closed_window"):
            skipped.append({"window_id": window_id, "reason": "window is not closed"})
            continue
        if not window.get("ready_for_30s_episodic"):
            skipped.append({"window_id": window_id, "reason": "window is not refined-ready"})
            continue
        if int(window.get("event_count", 0) or 0) <= 0:
            skipped.append({"window_id": window_id, "reason": "window has no events"})
            continue
        if window_id in window_to_episode and not force:
            skipped.append({"window_id": window_id, "reason": "already consolidated", "episode_id": window_to_episode[window_id]})
            continue
        event_ids = _event_ids_for_window(window)
        missing = [event_id for event_id in event_ids if event_id not in events_by_id]
        if missing:
            skipped.append({"window_id": window_id, "reason": "missing archive events", "event_ids": missing})
            continue
        events = [events_by_id[event_id] for event_id in event_ids]
        invalid = [event.get("event_id") for event in events if event.get("status") not in REFINED_STATUSES]
        if invalid:
            skipped.append({"window_id": window_id, "reason": "not all micro-events are refined", "event_ids": invalid})
            continue
        ready_windows.append((window, events))
        if limit_windows is not None and len(ready_windows) >= int(limit_windows):
            break

    plan_ids = [_window_episode_id(window) for window, _ in ready_windows]
    result: dict[str, Any] = {
        "status": "ok",
        "session_id": session_id,
        "backend": backend,
        "dry_run": dry_run,
        "ready_window_count": len(ready_windows),
        "planned_episode_ids": plan_ids,
        "generated_episode_count": 0,
        "generated_episode_ids": [],
        "skipped_windows": skipped,
        "updated_em2mem": False,
        "em2mem_update_mode": None,
        "outputs": {},
    }
    if dry_run:
        return result

    builder = MSTEpisodicWindowBuilder(backend=backend)
    prepared = []
    for window, events in ready_windows:
        episode = builder.build_episode(
            window,
            events,
            session_id=session_id,
            session_dir=session_dir,
            version=_episode_version(existing_episodes, _window_episode_id(window), force=force),
        )
        apply_demo_day_fields(
            episode,
            session_dir=session_dir,
            start_seconds=episode.get("start_time"),
            end_seconds=episode.get("end_time"),
        )
        prepared.append((window, events, episode))

    # Model generation stays outside the lock. Source validation, rebasing onto
    # the latest episode/state files, publication, and merge flags form one
    # serialized commit shared with ASR and other consolidation workers.
    with store.mutation_lock():
        latest_by_id = {str(item.get("event_id")): item for item in store.load_archive_events()}
        state = load_consolidation_state(session_dir, session_id)
        window_to_episode = dict(state.get("window_to_episode", {}) or {})
        episodes = _load_existing_episodes(existing_path)
        generated = []
        updates = []
        for window, events, episode in prepared:
            latest_window = {
                event_id: event for event_id, event in latest_by_id.items()
                if float(event.get("start_time", 0.0)) < float(window["end_time"])
                and float(event.get("end_time", 0.0)) > float(window["start_time"])
            }
            source_ids = {str(event.get("event_id")) for event in events}
            membership_changed = set(latest_window) != source_ids
            changed = [
                latest_by_id.get(str(event.get("event_id")))
                for event in events
                if latest_by_id.get(str(event.get("event_id"))) is None
                or any(
                    latest_by_id[str(event.get("event_id"))].get(field) != event.get(field)
                    for field in ("version", "transcript_version", "needs_refine", "refined_stale")
                )
            ]
            changed.extend(event for event_id, event in latest_window.items() if event_id not in source_ids)
            if changed or membership_changed:
                from online_streaming.transcript_backfill import mark_transcript_dirty_windows

                current_episode = next((item for item in episodes if item.get("episode_id") == episode["episode_id"]), {})
                published_versions = {str(item.get("event_id")): item.get("version")
                                      for item in current_episode.get("source_micro_events", [])}
                if (current_episode and set(published_versions) == set(latest_window)
                        and all(published_versions[event_id] == latest.get("version")
                                and not any(latest.get(field) for field in
                                            ("needs_refine", "refined_stale", "needs_reconsolidation"))
                                for event_id, latest in latest_window.items())):
                    skipped.append({"window_id": window.get("window_id"), "reason": "newer events already consolidated"})
                    continue
                dirty_events = []
                for latest in changed:
                    if latest is None:
                        continue
                    # Another worker may already have published this newer
                    # event. Rejecting our old result must not dirty it again.
                    if (published_versions.get(str(latest.get("event_id"))) == latest.get("version")
                            and not any(latest.get(field) for field in
                                        ("needs_refine", "refined_stale", "needs_reconsolidation"))):
                        continue
                    dirty = dict(latest)
                    dirty["needs_reconsolidation"] = True
                    dirty["dirty_reason"] = "event_changed_during_consolidation"
                    dirty_events.append(dirty)
                updates.extend(dirty_events)
                dirty_ranges = list(dirty_events)
                if membership_changed:
                    dirty_ranges.append({"start_time": window["start_time"], "end_time": window["end_time"]})
                mark_transcript_dirty_windows(session_dir, dirty_ranges)
                # A normal refine-ready task may already be queued. Removing
                # this obsolete completion lets it retry without force=True.
                window_to_episode.pop(str(window.get("window_id")), None)
                skipped.append({"window_id": window.get("window_id"), "reason": "events changed during consolidation"})
                continue

            window_id = str(window.get("window_id") or f"{window.get('start_time')}-{window.get('end_time')}")
            if window_id in window_to_episode and not force:
                skipped.append({"window_id": window_id, "reason": "already consolidated"})
                continue
            episode["version"] = _episode_version(episodes, episode["episode_id"], force=force)
            episodes = _replace_or_append_episode(episodes, episode)
            generated.append(episode)
            window_to_episode[window_id] = episode["episode_id"]
            for event in events:
                updated = dict(latest_by_id[str(event.get("event_id"))])
                updated["merged_to_long_term"] = True
                updated["merged_episode_id"] = episode["episode_id"]
                updated["merged_at"] = utc_now_iso()
                updated["merged_version"] = episode["version"]
                updated["needs_reconsolidation"] = False
                updated["dirty_reason"] = None
                updated["dirty_window_id"] = None
                updated["dirty_window_ids"] = []
                updated["dirty_time_range"] = None
                updates.append(updated)
            if verbose:
                print(f"[mst_episodic] built {episode['episode_id']} events={len(events)} version={episode['version']}")

        generated_ids = [episode["episode_id"] for episode in generated]
        pending_handoffs = dict(state.get("pending_memory_handoffs", {}) or {})
        for episode in generated:
            previous = pending_handoffs.get(episode["episode_id"], {})
            pending_handoffs[episode["episode_id"]] = memory_handoff(
                episode, force=force,
                requested=bool(queue_memory_handoff or previous.get("requested")),
            )
        output_paths = _write_outputs(session_dir, episodes, {
            **result, "generated_episode_count": len(generated), "generated_episode_ids": generated_ids,
        })
        new_state = {
            **state,
            "session_id": session_id,
            "last_consolidated_window_end": max([float(ep.get("end_time", 0.0)) for ep in episodes]
                                                 + [state.get("last_consolidated_window_end", 0.0)]),
            "generated_episode_count": len(episodes),
            "generated_episode_ids": [item.get("episode_id") for item in episodes],
            "window_to_episode": window_to_episode,
            "pending_memory_handoffs": pending_handoffs,
            "pending_ready_window_count": max(0, len([w for w in windows if isinstance(w, dict) and w.get("ready_for_30s_episodic")]) - len(window_to_episode)),
            "not_ready_window_count": len([w for w in windows if isinstance(w, dict) and not w.get("ready_for_30s_episodic")]),
            "skipped_windows": skipped,
            "last_run_at": utc_now_iso(),
        }
        consolidation_path = write_consolidation_state(session_dir, new_state)
        update_result = store.update_events(updates) if updates else {"active_updated": False, "archive_updated": False}

    _update_memory_config_metadata(session_dir)
    _update_status_outputs(session_dir, session_id)

    result.update(
        {
            "generated_episode_count": len(generated),
            "generated_episode_ids": generated_ids,
            "total_episode_count": len(episodes),
            "update_result": update_result,
            "outputs": {key: relative_to_session(path, session_dir) for key, path in output_paths.items()},
            "consolidation_state_path": relative_to_session(consolidation_path, session_dir),
        }
    )
    return result


def update_mst_em2mem_metadata(session_dir: Path, update_mode: str) -> None:
    _update_memory_config_metadata(session_dir, update_mode=update_mode)
    _update_status_outputs(session_dir, session_dir.name)
