from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from online_pipeline.runtime_state import collect_pipeline_runtime


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "online_sessions"


def _short(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _age(value: Any) -> str:
    try:
        seconds = float(value)
    except Exception:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _print_table(runtime: dict[str, Any], *, show_timeline: bool = False) -> None:
    print(f"Em2Mem Online Pipeline Runtime | updated_at={runtime.get('updated_at')}")
    pipeline = runtime.get("pipeline") or {}
    legacy = runtime.get("legacy_evidence") or {}
    print(f"Pipeline mode: {runtime.get('pipeline_mode', pipeline.get('pipeline_mode', 'mst'))}")
    print(f"Main path: {pipeline.get('main_path', '-')}")
    print(
        "Legacy evidence: "
        f"{legacy.get('status', 'unknown')} role={legacy.get('role', '-')} "
        f"enabled={legacy.get('enabled', False)}"
    )
    print("")
    print("Workers")
    print("worker          inst  role             status        pid  age    warmup  model/client                         queue  last_error")
    workers = runtime.get("workers") or {}
    for name in ("stream", "live_ingest", "preprocess", "evidence", "refine", "consolidation", "visual", "memory", "rokid_day_merge", "query"):
        item = workers.get(name) or {}
        model = item.get("model_name") or item.get("backend") or "-"
        loaded = item.get("model_loaded") or item.get("client_loaded")
        print(
            f"{name:<15} {int(item.get('instance_count') or 1):<5} {_short(item.get('role')):<16} {_short(item.get('status')):<13} "
            f"{str(bool(item.get('pid_alive'))):<4} "
            f"{_age(item.get('heartbeat_age_seconds')):<6} "
            f"{str(bool(item.get('warmup_done'))):<7} "
            f"{_short(model)[:36]:<36} "
            f"{int(item.get('queue_pending') or 0):<6} "
            f"{_short(item.get('last_error'))[:80]}"
        )
    print("")
    print("Queues")
    counts = runtime.get("queue_counts") or {}
    groups = [
        ("stream", "stream_chunk_queued", "stream_chunk_in_progress", "stream_chunk_done", "stream_chunk_failed"),
        ("live_ingest", "live_ingest_queued", "live_ingest_in_progress", "live_ingest_done", "live_ingest_failed"),
        ("stream_asr", "stream_asr_queued", "stream_asr_in_progress", "stream_asr_done", "stream_asr_failed"),
        ("rokid_day_merge", "rokid_day_merge_queued", "rokid_day_merge_in_progress", "rokid_day_merge_done", "rokid_day_merge_failed"),
        ("preprocess", "queued", "in_progress", "done", "failed"),
        ("evidence", "evidence_queued", "evidence_in_progress", "evidence_done", "evidence_failed"),
        ("refine", "mst_refine_queued", "mst_refine_in_progress", "mst_refine_done", "mst_refine_failed"),
        ("consolidation", "mst_consolidation_queued", "mst_consolidation_in_progress", "mst_consolidation_done", "mst_consolidation_failed"),
        ("visual", "visual_queued", "visual_in_progress", "visual_done", "visual_failed"),
        ("memory", "memory_queued", "memory_in_progress", "memory_done", "memory_failed"),
        ("query", "query_queued", "query_in_progress", "query_done", "query_failed"),
    ]
    print("queue           pending  running  done    failed")
    for label, q, r, d, f in groups:
        print(f"{label:<15} {counts.get(q, 0):<8} {counts.get(r, 0):<8} {counts.get(d, 0):<7} {counts.get(f, 0):<7}")
    print("")
    print("Backpressure / Latency")
    print("session              bp_level action       retry  last_mcur_ms last_asr_ms avg_mcur_ms avg_asr_ms")
    for state in runtime.get("sessions") or []:
        bp = state.get("backpressure") or {}
        stream = state.get("stream") or {}
        latency = stream.get("latency") if isinstance(stream.get("latency"), dict) else {}
        print(
            f"{state.get('session_id','-'):<20} "
            f"{_short(bp.get('level')):<8} "
            f"{_short(bp.get('recommended_action')):<12} "
            f"{_short(bp.get('retry_after_seconds')):<6} "
            f"{_short(latency.get('last_chunk_upload_to_mcur_ms')):<12} "
            f"{_short(latency.get('last_chunk_upload_to_asr_ms')):<11} "
            f"{_short(latency.get('avg_upload_to_mcur_ms')):<11} "
            f"{_short(latency.get('avg_upload_to_asr_ms')):<10}"
        )
    print("")
    print("Stream ASR / Transcript Backfill")
    print("session              asr_on backend    segs ver span             asr_done/failed pending_refine dirty_windows")
    for state in runtime.get("sessions") or []:
        asr = state.get("stream_asr") or {}
        backfill = state.get("transcript_backfill") or {}
        span = asr.get("time_span") or [0.0, 0.0]
        processed = asr.get("processed_asr_chunks") or []
        failed = asr.get("failed_asr_chunks") or []
        print(
            f"{state.get('session_id','-'):<20} "
            f"{str(bool(asr.get('enabled'))):<6} "
            f"{_short(asr.get('backend'))[:10]:<10} "
            f"{int(asr.get('partial_transcript_segment_count') or 0):<4} "
            f"{int(asr.get('partial_transcript_version') or 0):<3} "
            f"{str(span)[:16]:<16} "
            f"{len(processed)}/{len(failed):<14} "
            f"{int(backfill.get('pending_refine_due_to_transcript') or 0):<14} "
            f"{int(backfill.get('dirty_windows_due_to_transcript') or 0):<5}"
        )
    print("")
    print("Sessions")
    print("session              stream    up/proc  pgen nextU nextP open last_fr xdiff diff  cur  mcur  mst  ref_ready  epi30  ready/build/active  fast graph sem  lag(G/S/V) mode                 active_30s_source          query")
    for state in runtime.get("sessions") or []:
        stream = state.get("stream") or {}
        cur = state.get("current") or {}
        st = state.get("short_term") or {}
        ref = state.get("refine") or {}
        epi = state.get("episodic_30s") or {}
        lt = state.get("long_term") or {}
        sem = state.get("semantic") or {}
        vis = state.get("visual") or {}
        qry = state.get("query") or {}
        src = state.get("source") or {}
        versions = f"{lt.get('latest_ready_memory_version')}/{lt.get('building_memory_version')}/{lt.get('active_query_memory_version')}"
        lag = f"{int(bool(lt.get('graph_lagging')))}{int(bool(lt.get('semantic_lagging')))}{int(bool(lt.get('visual_lagging')))}"
        print(
            f"{state.get('session_id','-'):<20} "
            f"{_short(stream.get('status'))[:8]:<8} "
            f"{stream.get('received_upload_chunk_count', stream.get('received_chunk_count',0))}/{stream.get('processed_processing_chunk_count', stream.get('processed_chunk_count',0)):<5} "
            f"{stream.get('generated_processing_chunk_count',0):<4} "
            f"{stream.get('next_expected_upload_chunk_index',0):<5} "
            f"{stream.get('next_expected_proc_index', stream.get('next_expected_chunk_index',0)):<5} "
            f"{str(bool(stream.get('has_open_event'))):<4} "
            f"{_short(stream.get('last_candidate_frame_time')):<7} "
            f"{stream.get('cross_chunk_diff_count',0):<5} "
            f"{stream.get('diff_record_count',0):<5} "
            f"{str(bool(cur.get('ready'))):<4} {cur.get('version',0):<5} "
            f"{st.get('mst_version',0):<4} {ref.get('ready_30s_window_count',0):<10} "
            f"{epi.get('generated_episode_count', epi.get('version',0)):<6} {versions:<19} "
            f"{_short(lt.get('latest_fast_ready_version')):<5} "
            f"{_short(lt.get('latest_graph_ready_version')):<5} "
            f"{_short(lt.get('latest_semantic_ready_version')):<4} "
            f"{lag:<10} "
            f"{_short(lt.get('em2mem_update_mode'))[:20]:<20} "
            f"{_short(src.get('active_30s_source'))[:26]:<26} "
            f"{str(bool(qry.get('query_ready'))):<5}"
        )
    query_runtime = runtime.get("query_runtime") or {}
    loaded = query_runtime.get("loaded_sessions") or ((query_runtime.get("cache") or {}).get("loaded_sessions")) or []
    print("")
    print(f"Query loaded sessions: {len(loaded)}")
    for item in loaded[:8]:
        print(
            f"  {item.get('session_id')} active={item.get('active_query_memory_version')} "
            f"latest={item.get('latest_ready_memory_version')} building={item.get('building_memory_version')} "
            f"strict={item.get('strict_load_only')} preload={item.get('preload_status')} "
            f"fast={(item.get('memory_component_versions') or {}).get('fast')} "
            f"source={item.get('active_30s_source') or item.get('episodic_source')}"
        )
    if show_timeline:
        print("")
        print("Recent Timeline")
        for state in runtime.get("sessions") or []:
            events = state.get("recent_timeline_events") or []
            if not events:
                continue
            print(f"session={state.get('session_id')}")
            for event in events[-10:]:
                print(
                    f"  {event.get('timestamp')} "
                    f"{_short(event.get('event_type')):<24} "
                    f"chunk={_short(event.get('chunk_index')):<4} "
                    f"lat={_short(event.get('stage_latency_ms')):<6} "
                    f"{json.dumps(event.get('metadata') or {}, ensure_ascii=False)[:120]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor online Em2Mem worker, queue, and session pipeline state.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--show-timeline", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    while True:
        runtime = collect_pipeline_runtime(Path(args.project_root), Path(args.sessions_root), session_id=args.session_id)
        if args.json:
            print(json.dumps(runtime, ensure_ascii=False, indent=2))
        else:
            if args.watch:
                os.system("clear")
            _print_table(runtime, show_timeline=args.show_timeline)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
