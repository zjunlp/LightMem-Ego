from __future__ import annotations

import argparse
import os
from pathlib import Path

from online_current.mcur_store import MCurStore
from online_current.mcur_updater import MCurUpdater
from online_short_term.micro_event_builder import MicroEventBuilder
from online_short_term.mst_store import MSTStore
from online_short_term.schemas import DEFAULT_SESSIONS_ROOT
from online_short_term.stream_chunk_manager import StreamChunkManager, discover_chunks
from online_pipeline.runtime_state import refresh_session_pipeline_state
from online_preprocess.task_queue import enqueue_mst_refine_task


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M_st micro-event FIFO from existing stream chunks.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--sessions-root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clear-archive", action="store_true", help="With --force, also clear append-only M_st archive.")
    parser.add_argument("--update-mcur", action="store_true", help="Also rebuild/update M_cur from the processed chunks.")
    parser.add_argument("--enqueue-refine", action=argparse.BooleanOptionalAction, default=_env_bool("WORLDMM_AUTO_MST_REFINE", True))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    session_dir = Path(args.sessions_root) / args.session_id
    if args.force:
        MSTStore(session_dir).clear(clear_archive=args.clear_archive)
        if args.update_mcur:
            MCurStore(session_dir).clear()
        processed = session_dir / "stream" / "processed_chunks.jsonl"
        if processed.exists():
            processed.unlink()
    chunks = discover_chunks(session_dir)
    builder = MicroEventBuilder(session_dir)
    manager = StreamChunkManager(session_dir)
    mcur_updater = MCurUpdater(session_dir) if args.update_mcur else None
    total = 0
    mcur_state = {}
    for chunk in chunks:
        events = builder.process_chunk(chunk)
        total += len(events)
        if mcur_updater is not None:
            mcur_state = mcur_updater.update_from_stream_chunk(chunk_info=chunk)
        manager.append_processed_chunk({**chunk, "event_count": len(events)})
        if args.verbose:
            print(
                f"[mst] processed {chunk['chunk_id']} events={len(events)} "
                f"mcur_version={mcur_state.get('mcur_version') if mcur_state else None}"
            )
    state = builder.store.get_state()
    print(
        f"[mst] done session={args.session_id} chunks={len(chunks)} appended_events={total} "
        f"state={state} mcur_ready={mcur_state.get('mcur_ready') if mcur_state else False}"
    )
    refresh_session_pipeline_state(session_dir)
    if total > 0 and args.enqueue_refine:
        task_path = enqueue_mst_refine_task(
            project_root=PROJECT_ROOT,
            session_id=args.session_id,
            backend=os.getenv("WORLDMM_MST_REFINE_BACKEND", "openai"),
            limit_events=int(os.getenv("WORLDMM_MST_REFINE_LIMIT_EVENTS", "10")),
            force_refine=False,
        )
        print(f"[mst] refine_task={task_path}")


if __name__ == "__main__":
    main()
