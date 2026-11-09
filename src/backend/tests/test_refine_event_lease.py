import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import refine_mst_micro_events as refine_module
from online_short_term.mst_store import MSTStore


def _append_test_event(store: MSTStore, session_id: str, event_id: str) -> None:
    store.append_events(
        [
            {
                "event_id": event_id,
                "session_id": session_id,
                "start_time": 0.0,
                "end_time": 2.0,
                "status": "provisional",
                "needs_refine": True,
                "event_caption_placeholder": "provisional event",
                "keyframes": [],
            }
        ]
    )


def test_concurrent_refine_calls_only_invoke_model_once() -> None:
    with TemporaryDirectory() as temp_dir:
        sessions_root = Path(temp_dir)
        session_id = "lease_session"
        store = MSTStore(sessions_root / session_id)
        event_id = "mst_event_lease_test"
        _append_test_event(store, session_id, event_id)

        call_count = 0
        call_count_lock = threading.Lock()
        start_barrier = threading.Barrier(2)

        class FakeRefiner:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            def refine_event(self, event, session_dir, **kwargs):
                nonlocal call_count
                with call_count_lock:
                    call_count += 1
                time.sleep(0.2)
                updated = dict(event)
                updated.update(
                    {
                        "status": "refined",
                        "needs_refine": False,
                        "refined_stale": False,
                        "caption_source": "refined",
                        "event_caption_refined": "refined once",
                        "version": int(event.get("version", 1) or 1) + 1,
                    }
                )
                return updated

        def run(task_id: str):
            start_barrier.wait()
            return refine_module.refine_session(
                session_id=session_id,
                sessions_root=sessions_root,
                backend="mock",
                limit_events=1,
                task_id=task_id,
            )

        with (
            patch.object(refine_module, "MicroEventRefiner", FakeRefiner),
            patch.dict(
                "os.environ",
                {
                    "EM2MEM_REFINE_MAX_CONCURRENCY": "1",
                    "EM2MEM_MST_REFINE_LEASE_WAIT_SECONDS": "2",
                },
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(run, ("task_a", "task_b")))

        assert call_count == 1
        assert sum(result["refined_event_count"] for result in results) == 1
        assert sum(result["skipped_already_refined_count"] for result in results) == 1
        saved = store.load_archive_events()[0]
        assert saved["status"] == "refined"
        assert saved["event_caption_refined"] == "refined once"


def test_transcript_backfill_during_refine_is_preserved() -> None:
    with TemporaryDirectory() as temp_dir:
        sessions_root = Path(temp_dir)
        session_id = "transcript_race_session"
        store = MSTStore(sessions_root / session_id)
        event_id = "mst_event_transcript_race"
        _append_test_event(store, session_id, event_id)
        model_started = threading.Event()
        allow_model_finish = threading.Event()

        class FakeRefiner:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            def refine_event(self, event, session_dir, **kwargs):
                model_started.set()
                assert allow_model_finish.wait(timeout=2)
                updated = dict(event)
                updated.update(
                    {
                        "status": "refined",
                        "needs_refine": False,
                        "refined_stale": False,
                        "caption_source": "refined",
                        "event_caption_refined": "caption from old transcript",
                        "version": int(event.get("version", 1) or 1) + 1,
                    }
                )
                return updated

        def run_refine():
            return refine_module.refine_session(
                session_id=session_id,
                sessions_root=sessions_root,
                backend="mock",
                limit_events=1,
                task_id="race_task",
            )

        with patch.object(refine_module, "MicroEventRefiner", FakeRefiner):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_refine)
                assert model_started.wait(timeout=2)
                store.backfill_transcript_segments(
                    [{"segment_id": "seg1", "start": 0.5, "end": 1.5, "text": "new transcript"}],
                    reason="stream_asr",
                )
                allow_model_finish.set()
                result = future.result(timeout=2)

        assert result["refined_event_count"] == 1
        saved = store.load_archive_events()[0]
        assert saved["transcript"] == "new transcript"
        assert saved["needs_refine"] is True
        assert saved["refined_stale"] is True
        assert saved["stale_reason"] == "transcript_changed_during_refine"


def test_mst_store_tolerates_historical_invalid_utf8() -> None:
    with TemporaryDirectory() as temp_dir:
        session_dir = Path(temp_dir) / "invalid_utf8_session"
        store = MSTStore(session_dir)
        store.archive_events_path.parent.mkdir(parents=True, exist_ok=True)
        store.archive_events_path.write_bytes(
            b'{"event_id":"event1","start_time":0,"end_time":1,'
            b'"event_caption_placeholder":"bad \xff text"}\n'
        )

        events = store.load_archive_events()

        assert len(events) == 1
        assert events[0]["event_id"] == "event1"
        store.save_archive_events(events)
        store.archive_events_path.read_text(encoding="utf-8")


def test_refine_session_persists_timed_observations_to_active_and_archive() -> None:
    with TemporaryDirectory() as temp_dir:
        sessions_root = Path(temp_dir)
        session_id = "observation_persistence"
        store = MSTStore(sessions_root / session_id)
        _append_test_event(store, session_id, "observation_event")
        observation = {
            "timestamp": 1.0,
            "path": "stream/frames/frame_000001.jpg",
            "description": "The front-right man holds a phone to his ear.",
        }

        class FakeRefiner:
            def __init__(self, backend: str) -> None:
                self.backend = backend

            def refine_event(self, event, session_dir, **kwargs):
                return {
                    **event,
                    "status": "refined",
                    "event_caption_refined": "A man uses a phone.",
                    "visual_observations": [observation],
                    "refine": {"observed_frames": [
                        {"timestamp": 1.0, "path": observation["path"]}
                    ]},
                }

        with patch.object(refine_module, "MicroEventRefiner", FakeRefiner):
            result = refine_module.refine_session(
                session_id=session_id,
                sessions_root=sessions_root,
                backend="mock",
                limit_events=1,
            )

        assert result["refined_event_count"] == 1
        for saved in (store.load_events()[0], store.load_archive_events()[0]):
            assert saved["visual_observations"] == [observation]
            assert "holds a phone to his ear" in saved["retrieval_text"]
            assert saved["refine"]["observed_frames"][0]["timestamp"] == 1.0
