import json
from pathlib import Path

from online_pipeline.active_session import clear_old_session_tasks, write_active_session
from online_preprocess.task_queue import claim_query_task, enqueue_query_task, finish_query_task, get_queue_dirs


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_allow_inactive_query_task_claims_and_finishes_when_session_is_not_active(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDMM_SINGLE_ACTIVE_SESSION", "true")
    write_active_session(tmp_path, "active_session")
    queued_path = enqueue_query_task(
        tmp_path,
        "old_session",
        "what happened?",
        allow_inactive_session=True,
        task_source="session_history_api",
    )

    claimed = claim_query_task(tmp_path, queued_path)

    assert claimed is not None
    claimed_path, task = claimed
    assert task["allow_inactive_session"] is True
    done_path = finish_query_task(
        tmp_path,
        claimed_path,
        task,
        "done",
        result={"status": "ok", "answer": "answer"},
    )

    payload = _read_json(done_path)
    assert payload["status"] == "done"
    assert payload["allow_inactive_session"] is True
    assert payload["task_source"] == "session_history_api"


def test_normal_inactive_query_task_is_still_cancelled_on_claim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDMM_SINGLE_ACTIVE_SESSION", "true")
    write_active_session(tmp_path, "active_session")
    queued_path = enqueue_query_task(tmp_path, "old_session", "what happened?")

    claimed = claim_query_task(tmp_path, queued_path)

    assert claimed is None
    dirs = get_queue_dirs(tmp_path)
    failed_payloads = list(dirs["query_failed"].glob("*.json"))
    assert len(failed_payloads) == 1
    payload = _read_json(failed_payloads[0])
    assert payload["status"] == "cancelled"
    assert payload["error_type"] == "cancelled"


def test_clear_old_session_tasks_preserves_history_query_tasks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLDMM_SINGLE_ACTIVE_SESSION", "true")
    write_active_session(tmp_path, "active_session")
    preserved_path = enqueue_query_task(
        tmp_path,
        "old_session",
        "history question",
        allow_inactive_session=True,
        task_source="session_history_api",
    )
    cancelled_path = enqueue_query_task(tmp_path, "other_old_session", "normal question")

    result = clear_old_session_tasks(tmp_path, keep_session_id="active_session")

    assert result["aborted"] == 1
    assert preserved_path.exists()
    assert not cancelled_path.exists()
    preserved_payload = _read_json(preserved_path)
    assert preserved_payload["allow_inactive_session"] is True
