from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
TEST_TMP_ROOT = PROJECT_ROOT.parent / "work" / "omnispark_test_tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from online_short_term.micro_event_refiner import MicroEventRefiner, _apply_refine_timing  # noqa: E402


class FailingMicroEventRefiner(MicroEventRefiner):
    def _refine_mock(self, event: dict) -> dict:
        raise RuntimeError("boom")

class TestWorkspace:
    def __enter__(self) -> Path:
        self.path = TEST_TMP_ROOT / ("case_" + uuid4().hex)
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            shutil.rmtree(self.path)
        except Exception:
            pass
        return None


def write_json_for_test(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class MSTRefineTimingTest(unittest.TestCase):
    def test_refine_timing_uses_stream_start_wall_clock(self) -> None:
        with TestWorkspace() as tmp:
            session_dir = tmp / "sessions" / "s1"
            write_json_for_test(
                session_dir / "stream" / "stream_state.json",
                {"created_at": "2026-01-01T00:00:00+00:00"},
            )
            event = {
                "event_id": "e1",
                "available_at": 10.0,
                "end_time": 10.0,
                "created_at": "2026-01-01T00:00:10+00:00",
                "refine": {},
            }

            updated = _apply_refine_timing(
                event,
                session_dir,
                completed_at_iso="2026-01-01T00:00:25+00:00",
            )

            self.assertEqual(updated["refine_completed_at"], 25.0)
            self.assertEqual(updated["refine_speed"], 15.0)
            self.assertEqual(updated["refine"]["last_refine_completed_at"], "2026-01-01T00:00:25+00:00")

    def test_refine_timing_falls_back_to_event_created_at(self) -> None:
        with TestWorkspace() as tmp:
            session_dir = tmp / "sessions" / "s1"
            event = {
                "event_id": "e1",
                "available_at": 10.0,
                "end_time": 10.0,
                "created_at": "2026-01-01T00:00:10+00:00",
                "refine": {},
            }

            updated = _apply_refine_timing(
                event,
                session_dir,
                completed_at_iso="2026-01-01T00:00:25+00:00",
            )

            self.assertEqual(updated["refine_completed_at"], 25.0)
            self.assertEqual(updated["refine_speed"], 15.0)



    def test_refine_timeline_records_queue_and_worker_times(self) -> None:
        with TestWorkspace() as tmp:
            session_dir = tmp / "sessions" / "s1"
            write_json_for_test(
                session_dir / "stream" / "stream_state.json",
                {"created_at": "2026-01-01T00:00:00+00:00"},
            )
            event = {
                "event_id": "e1",
                "available_at": 10.0,
                "end_time": 10.0,
                "created_at": "2026-01-01T00:00:10+00:00",
                "refine": {},
            }

            updated = MicroEventRefiner(backend="mock").refine_event(
                event,
                session_dir,
                task_id="task-1",
                task_queued_at="2026-01-01T00:00:11+00:00",
                task_worker_started_at="2026-01-01T00:00:12+00:00",
                task_reason="unit_test",
            )

            refine = updated["refine"]
            self.assertEqual(updated["status"], "refined")
            self.assertEqual(refine["last_refine_task_id"], "task-1")
            self.assertEqual(refine["last_refine_queued_at"], "2026-01-01T00:00:11+00:00")
            self.assertEqual(refine["last_refine_worker_started_at"], "2026-01-01T00:00:12+00:00")
            self.assertEqual(refine["last_refine_task_reason"], "unit_test")
            self.assertEqual(len(refine["refine_timeline"]), 1)
            attempt = refine["refine_timeline"][0]
            self.assertEqual(attempt["attempt"], 1)
            self.assertFalse(attempt["is_retry"])
            self.assertEqual(attempt["queued_at"], "2026-01-01T00:00:11+00:00")
            self.assertEqual(attempt["worker_started_at"], "2026-01-01T00:00:12+00:00")
            self.assertEqual(attempt["status"], "refined")
            self.assertIsNotNone(attempt["refine_completed_at"])

    def test_refine_timeline_records_retry_failure_time(self) -> None:
        with TestWorkspace() as tmp:
            session_dir = tmp / "sessions" / "s1"
            event = {
                "event_id": "e1",
                "available_at": 10.0,
                "end_time": 10.0,
                "created_at": "2026-01-01T00:00:10+00:00",
                "refine": {
                    "refine_attempts": 1,
                    "refine_timeline": [{"attempt": 1, "status": "refine_failed"}],
                },
            }

            updated = FailingMicroEventRefiner(backend="mock").refine_event(
                event,
                session_dir,
                task_id="task-2",
                task_queued_at="2026-01-01T00:01:00+00:00",
                task_worker_started_at="2026-01-01T00:01:02+00:00",
                task_reason="retry_test",
            )

            refine = updated["refine"]
            self.assertEqual(updated["status"], "refine_failed")
            self.assertEqual(refine["refine_attempts"], 2)
            self.assertEqual(refine["last_refine_retry_queued_at"], "2026-01-01T00:01:00+00:00")
            self.assertIsNotNone(refine["last_refine_failed_at"])
            self.assertIn("RuntimeError: boom", refine["last_refine_error"])
            self.assertEqual(len(refine["refine_timeline"]), 2)
            attempt = refine["refine_timeline"][-1]
            self.assertEqual(attempt["attempt"], 2)
            self.assertTrue(attempt["is_retry"])
            self.assertEqual(attempt["status"], "refine_failed")
            self.assertIsNotNone(attempt["refine_failed_at"])
            self.assertIn("RuntimeError: boom", attempt["error"])

if __name__ == "__main__":
    unittest.main()
