from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from online_current.mcur_store import MCurStore
from online_current.mcur_updater import MCurUpdater
from online_short_term.micro_event_builder import MicroEventBuilder
from online_short_term.mst_store import MSTStore
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT
from online_short_term.stream_chunk_manager import StreamChunkManager, prepare_output_session
from online_pipeline.runtime_state import refresh_session_pipeline_state
from online_preprocess.task_queue import enqueue_mst_refine_task


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate pseudo-streaming chunks from an existing input.mp4.")
    parser.add_argument("--session-id", required=True, help="Source session id.")
    parser.add_argument("--output-session-id", default=None, help="Optional target session id, e.g. <session_id>chunk.")
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--process", action="store_true", help="Immediately process each chunk into M_st.")
    parser.add_argument("--update-mcur", action="store_true", help="Update M_cur rolling current memory for each chunk.")
    parser.add_argument("--enqueue-stream-tasks", action="store_true", help="Simulate real stream upload by enqueueing stream_chunk tasks instead of direct processing.")
    parser.add_argument("--use-stream-api-mode", action="store_true", help="Alias for --enqueue-stream-tasks.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clear-archive", action="store_true", help="With --force, also clear append-only M_st archive.")
    parser.add_argument("--enqueue-refine", action=argparse.BooleanOptionalAction, default=_env_bool("EM2MEM_AUTO_MST_REFINE", True))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sessions_root = Path(args.sessions_root)
    target_session_id, session_dir = prepare_output_session(
        sessions_root=sessions_root,
        source_session_id=args.session_id,
        output_session_id=args.output_session_id,
        force=args.force,
    )
    manager = StreamChunkManager(session_dir)
    chunks = manager.split_input_video(
        chunk_seconds=args.chunk_seconds,
        max_chunks=args.max_chunks,
        force=args.force,
    )
    if args.enqueue_stream_tasks or args.use_stream_api_mode:
        task_path = manager.enqueue_ready_chunk(PROJECT_ROOT)
        refresh_session_pipeline_state(session_dir)
        print(
            f"[stream] queued first stream task target_session={target_session_id} "
            f"chunks={len(chunks)} task_path={task_path}"
        )
        return
    if args.force and args.process:
        MSTStore(session_dir).clear(clear_archive=args.clear_archive)
    if args.force and args.update_mcur:
        MCurStore(session_dir).clear()
    builder = MicroEventBuilder(session_dir) if args.process else None
    mcur_updater = MCurUpdater(session_dir) if args.update_mcur else None
    total_events = 0
    mcur_state = {}
    for chunk in chunks:
        event_count = 0
        if builder is not None:
            events = builder.process_chunk(chunk)
            event_count = len(events)
            total_events += event_count
            manager.append_processed_chunk({**chunk, "event_count": event_count})
        if mcur_updater is not None:
            mcur_state = mcur_updater.update_from_stream_chunk(chunk_info=chunk)
        if args.verbose:
            print(
                f"[stream] chunk={chunk['chunk_id']} path={chunk['path']} "
                f"events={event_count} mcur_version={mcur_state.get('mcur_version') if mcur_state else None}"
            )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    print(
        f"[stream] done source_session={args.session_id} target_session={target_session_id} "
        f"chunks={len(chunks)} events={total_events} mcur_ready={mcur_state.get('mcur_ready') if mcur_state else False}"
    )
    refresh_session_pipeline_state(session_dir)
    if total_events > 0 and args.enqueue_refine:
        task_path = enqueue_mst_refine_task(
            project_root=PROJECT_ROOT,
            session_id=target_session_id,
            backend=os.getenv("EM2MEM_MST_REFINE_BACKEND", "openai"),
            limit_events=int(os.getenv("EM2MEM_MST_REFINE_LIMIT_EVENTS", "10")),
            force_refine=False,
        )
        print(f"[stream] refine_task={task_path}")


if __name__ == "__main__":
    main()
