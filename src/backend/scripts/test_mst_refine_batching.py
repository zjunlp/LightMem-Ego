from __future__ import annotations

import json
import shutil
import sys
import types
from uuid import uuid4
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("fcntl", types.SimpleNamespace(LOCK_EX=1, LOCK_UN=2, flock=lambda *args, **kwargs: None))
TEST_TMP_ROOT = PROJECT_ROOT.parent / "work" / "lightmem_ego_test_tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class TestWorkspace:
    def __enter__(self) -> str:
        self.path = TEST_TMP_ROOT / ("case_" + uuid4().hex)
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            shutil.rmtree(self.path)
        except Exception:
            pass
        return None


def temp_directory():
    return TestWorkspace()

from online_mst_refine_worker import (  # noqa: E402
    FRAME_STREAM_BATCH_REASON,
    _enqueue_refine_followup_if_needed,
)
import online_preprocess.task_queue as task_queue_module  # noqa: E402
from online_preprocess.task_queue import (  # noqa: E402
    enqueue_mst_consolidation_task,
    enqueue_mst_refine_task,
)
from refine_mst_micro_events import _select_events, is_auto_refine_eligible  # noqa: E402


def write_json_for_test(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


task_queue_module.write_json_atomic = write_json_for_test


def create_in_progress_task(
    project_root: Path,
    *,
    queue_key: str,
    task_type: str,
    session_id: str = "s1",
    reason: str | None = None,
    event_id: str | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
) -> Path:
    dirs = task_queue_module.ensure_queue_dirs(project_root)
    task_id = session_id + "_inprogress"
    target = dirs[queue_key] / (task_id + ".json")
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "session_id": session_id,
        "status": "in_progress",
        "reason": reason,
        "event_id": event_id,
        "window_start": window_start,
        "window_end": window_end,
    }
    write_json_for_test(target, payload)
    return target


class MSTRefineBatchingTest(unittest.TestCase):
    def test_refine_batch_reuses_queued_but_not_in_progress(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            first = enqueue_mst_refine_task(project_root, "s1", event_id=None, reason=FRAME_STREAM_BATCH_REASON)
            second = enqueue_mst_refine_task(project_root, "s1", event_id=None, reason=FRAME_STREAM_BATCH_REASON)
            self.assertEqual(first, second)


        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            claimed_path = create_in_progress_task(
                project_root,
                queue_key="mst_refine_in_progress",
                task_type="mst_refine",
                reason=FRAME_STREAM_BATCH_REASON,
            )
            followup = enqueue_mst_refine_task(project_root, "s1", event_id=None, reason=FRAME_STREAM_BATCH_REASON)
            self.assertNotEqual(claimed_path, followup)
            self.assertEqual(followup.parent.name, "mst_refine")

    def test_consolidation_reuses_queued_but_allows_in_progress_successor(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            first = enqueue_mst_consolidation_task(project_root, "s1", reason="mst_refine_ready_batch")
            second = enqueue_mst_consolidation_task(project_root, "s1", reason="mst_refine_ready_batch")
            self.assertEqual(first, second)


        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            claimed_path = create_in_progress_task(
                project_root,
                queue_key="mst_consolidation_in_progress",
                task_type="mst_consolidation",
                reason="mst_refine_ready_batch",
            )
            followup = enqueue_mst_consolidation_task(project_root, "s1", reason="mst_refine_ready_batch")
            self.assertNotEqual(claimed_path, followup)
            self.assertEqual(followup.parent.name, "mst_consolidation")

    def test_transcript_backfill_window_dedupe_still_reuses_in_progress(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            claimed_path = create_in_progress_task(
                project_root,
                queue_key="mst_consolidation_in_progress",
                task_type="mst_consolidation",
                reason="transcript_backfill",
                window_start=0.0,
                window_end=30.0,
            )
            second = enqueue_mst_consolidation_task(
                project_root,
                "s1",
                reason="transcript_backfill",
                window_start=0.0,
                window_end=30.0,
            )
            self.assertEqual(second, claimed_path)

    def test_refine_failed_respects_attempt_limit_and_backoff(self) -> None:
        now = datetime.now(timezone.utc)
        old = (now - timedelta(seconds=600)).isoformat()
        fresh = now.isoformat()
        self.assertTrue(
            is_auto_refine_eligible(
                {"status": "refine_failed", "refine": {"refine_attempts": 2, "last_refine_at": old}},
                now=now,
                max_attempts=3,
                retry_backoff_seconds=300,
            )
        )
        self.assertFalse(
            is_auto_refine_eligible(
                {"status": "refine_failed", "refine": {"refine_attempts": 3, "last_refine_at": old}},
                now=now,
                max_attempts=3,
                retry_backoff_seconds=300,
            )
        )
        self.assertFalse(
            is_auto_refine_eligible(
                {"status": "refine_failed", "refine": {"refine_attempts": 2, "last_refine_at": fresh}},
                now=now,
                max_attempts=3,
                retry_backoff_seconds=300,
            )
        )

    def test_force_refine_bypasses_automatic_attempt_limit(self) -> None:
        event = {"event_id": "e1", "status": "refine_failed", "refine": {"refine_attempts": 99}}
        self.assertEqual(_select_events([event], event_id=None, event_ids=None, limit_events=10, force_refine=False), [])
        self.assertEqual(_select_events([event], event_id=None, event_ids=None, limit_events=10, force_refine=True), [event])

    def test_refine_batch_merges_event_ids_for_same_reason(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            first = enqueue_mst_refine_task(
                project_root,
                "s1",
                event_ids=["e1"],
                limit_events=1,
                force_refine=True,
                reason="audio_asr_backfill",
            )
            second = enqueue_mst_refine_task(
                project_root,
                "s1",
                event_ids=["e2", "e1"],
                limit_events=2,
                force_refine=True,
                reason="audio_asr_backfill",
            )
            self.assertEqual(first, second)
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("event_ids"), ["e1", "e2"])
            self.assertEqual(payload.get("limit_events"), 2)

    def test_worker_followup_only_when_archive_has_eligible_pending(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            session_dir = Path(tmp) / "sessions" / "s1"
            archive_path = session_dir / "short_term" / "archive" / "micro_events_all.jsonl"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            events = [
                {"event_id": "e1", "session_id": "s1", "start_time": 0.0, "end_time": 1.0, "status": "provisional"},
                {"event_id": "e2", "session_id": "s1", "start_time": 1.0, "end_time": 2.0, "status": "provisional"},
            ]
            archive_path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")

            followup = _enqueue_refine_followup_if_needed(
                project_root=project_root,
                session_dir=session_dir,
                backend="mock",
                limit_events=10,
                event_id=None,
                force_refine=False,
                task_reason=FRAME_STREAM_BATCH_REASON,
            )
            self.assertIsNotNone(followup)
            second = _enqueue_refine_followup_if_needed(
                project_root=project_root,
                session_dir=session_dir,
                backend="mock",
                limit_events=10,
                event_id=None,
                force_refine=False,
                task_reason=FRAME_STREAM_BATCH_REASON,
            )
            self.assertEqual(followup, second)

    def test_worker_followup_skips_maxed_failed_events(self) -> None:
        with temp_directory() as tmp:
            project_root = Path(tmp) / "project"
            session_dir = Path(tmp) / "sessions" / "s1"
            archive_path = session_dir / "short_term" / "archive" / "micro_events_all.jsonl"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "event_id": "e1",
                "session_id": "s1",
                "start_time": 0.0,
                "end_time": 1.0,
                "status": "refine_failed",
                "refine": {"refine_attempts": 3, "last_refine_at": "2026-01-01T00:00:00+00:00"},
            }
            archive_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            followup = _enqueue_refine_followup_if_needed(
                project_root=project_root,
                session_dir=session_dir,
                backend="mock",
                limit_events=10,
                event_id=None,
                force_refine=False,
                task_reason=FRAME_STREAM_BATCH_REASON,
            )
            self.assertIsNone(followup)

    def test_frame_stream_uses_batch_reason_in_both_close_paths(self) -> None:
        source = (PROJECT_ROOT / "online_short_term" / "frame_stream_event_builder.py").read_text(encoding="utf-8")
        self.assertNotIn('reason="frame_stream_closed"', source)
        self.assertGreaterEqual(source.count('reason="frame_stream_batch"'), 2)
        self.assertNotIn("limit_events=1", source)


if __name__ == "__main__":
    unittest.main()
