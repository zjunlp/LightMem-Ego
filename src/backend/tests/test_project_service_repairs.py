"""Offline regressions for service scheduling, queue races, and runtime views."""
import asyncio
import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading

import pytest

from online_preprocess.io_utils import read_json, write_json_atomic
from online_preprocess import task_queue as tq
from online_pipeline import runtime_state as rs


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Network forbidden in service repair regression")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setenv("EM2MEM_SINGLE_ACTIVE_SESSION", "0")
    monkeypatch.setenv("EM2MEM_ENABLE_DEMO_ROUTES", "0")
    monkeypatch.setenv("EM2MEM_WORKER_STALE_SECONDS", "60")


@pytest.mark.parametrize("disappear_at", ["read", "stat"])
def test_query_listing_survives_peer_claim(tmp_path, monkeypatch, disappear_at):
    queued = tq.enqueue_query_task(tmp_path, "s", "What is visible?", memory_mode="current")
    survivor = tq.enqueue_query_task(tmp_path, "s", "Earlier activity?", memory_mode="long_term")
    original = tq.read_json
    claimed = []
    def read_with_peer_claim(path, default=None):
        if path != queued or claimed:
            return original(path, default)
        if disappear_at == "stat":
            payload = original(path, default)
        with monkeypatch.context() as peer:
            peer.setattr(tq, "read_json", original)
            claimed.append(tq.claim_query_task(tmp_path, queued))
        if disappear_at == "read":
            # A real read after read_json's exists check observes this error.
            raise FileNotFoundError(queued)
        return payload
    monkeypatch.setattr(tq, "read_json", read_with_peer_claim)
    assert tq.list_queued_query_tasks(tmp_path) == [survivor]
    assert claimed[0] is not None
    assert tq.claim_query_task(tmp_path, queued) is None


def test_current_classifier_ignores_peer_claim_read_race(tmp_path, monkeypatch):
    import online_query_worker as worker
    queued = tq.enqueue_query_task(tmp_path, "s", "What is visible?", memory_mode="current")
    original = Path.read_text
    def peer_claim_before_read(path, *args, **kwargs):
        if path == queued:
            monkeypatch.setattr(Path, "read_text", original)
            assert tq.claim_query_task(tmp_path, queued) is not None
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", peer_claim_before_read)
    assert worker._is_current_lane_task(queued) is False


def test_claim_returns_none_after_peer_claim_between_check_and_read(tmp_path, monkeypatch):
    queued = tq.enqueue_query_task(tmp_path, "s", "Earlier activity?")
    original = Path.read_text
    def peer_claim_before_read(path, *args, **kwargs):
        if path == queued:
            monkeypatch.setattr(Path, "read_text", original)
            assert tq.claim_query_task(tmp_path, queued) is not None
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", peer_claim_before_read)
    assert tq.claim_query_task(tmp_path, queued) is None


@pytest.mark.parametrize("disappear_at", ["check", "read"])
def test_polling_task_lookup_survives_peer_claim_read_race(tmp_path, monkeypatch, disappear_at):
    import api_server as api
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    queued = tq.enqueue_query_task(tmp_path, "s", "Earlier activity?")
    original = Path.read_text
    def peer_claim_before_read(path, *args, **kwargs):
        if path == queued:
            monkeypatch.setattr(Path, "read_text", original)
            assert tq.claim_query_task(tmp_path, queued) is not None
        return original(path, *args, **kwargs)
    if disappear_at == "read":
        monkeypatch.setattr(Path, "read_text", peer_claim_before_read)
    else:
        def peer_claim_before_check(path, default=None):
            if path == queued:
                assert tq.claim_query_task(tmp_path, queued) is not None
            return read_json(path, default)
        monkeypatch.setattr(api, "read_json", peer_claim_before_check)
    state, path, task = api._find_query_task(queued.stem)
    assert state == "query_in_progress"
    assert path.name == queued.name and task["status"] == "in_progress"


@pytest.fixture
def sync_api(tmp_path, monkeypatch):
    import api_server as api
    import online_query
    sessions = tmp_path / "online_sessions"
    write_json_atomic(sessions / "s" / "em2mem" / "memory_config.json",
                      {"status": "memory_ready", "latest_ready_memory_version": 1})
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "ONLINE_SESSIONS_DIR", sessions)
    monkeypatch.setenv("EM2MEM_ENABLE_SYNC_ASK", "1")
    monkeypatch.setattr(online_query, "query_session", lambda **kwargs: {"status": "ok", "answer": "local"})
    return api, online_query


@pytest.mark.parametrize("stage", ["query", "tts"])
def test_sync_query_and_tts_leave_event_loop_responsive(sync_api, monkeypatch, stage):
    api, online_query = sync_api
    entered = threading.Event()
    released = threading.Event()
    timed_out = []
    def blocking_stage(*args, **kwargs):
        entered.set()
        timed_out.append(not released.wait(timeout=0.3))
        return kwargs.get("result") or (args[0] if args else {"status": "ok", "answer": "local"})
    if stage == "query":
        monkeypatch.setattr(online_query, "query_session", blocking_stage)
    else:
        monkeypatch.setattr(api, "_attach_answer_audio_for_response", blocking_stage)
    async def run():
        request = asyncio.create_task(api._handle_ask_session("s", api.AskRequest(question="test", mode="sync", answer_tts=True)))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        await api.ping()
        released.set()
        response = await request
        assert response.status_code == 200
    asyncio.run(run())
    assert timed_out == [False]


def test_sync_cancellation_keeps_running_work_serial(sync_api, monkeypatch):
    api, online_query = sync_api
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []
    submissions = []
    original_submit = api._SYNC_ASK_EXECUTOR.submit
    def record_submission(work):
        submissions.append(work)
        return original_submit(work)
    monkeypatch.setattr(api._SYNC_ASK_EXECUTOR, "submit", record_submission)
    def local_query(**kwargs):
        calls.append(kwargs["question"])
        if kwargs["question"] == "first":
            entered.set()
            assert release.wait(timeout=2)
            completed.set()
        else:
            assert completed.is_set()
        return {"status": "ok", "answer": kwargs["question"]}
    monkeypatch.setattr(online_query, "query_session", local_query)
    async def run():
        first = asyncio.create_task(api._handle_ask_session("s", api.AskRequest(question="first", mode="sync")))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(api._handle_ask_session("s", api.AskRequest(question="second", mode="sync")))
        await asyncio.sleep(0.025)
        assert calls == ["first"]
        assert len(submissions) == 1  # Waiting requests do not grow executor's queue.
        release.set()
        response = await second
        assert response.status_code == 200
        assert json.loads(response.body)["answer"] == "second"
    try:
        asyncio.run(asyncio.wait_for(run(), timeout=3))
    finally:
        release.set()
    assert calls == ["first", "second"]


def test_pipeline_state_repair_is_only_a_view_and_keeps_new_publication(tmp_path, monkeypatch):
    from online_memory_edit.lock import session_memory_mutation_lock
    session = tmp_path / "online_sessions" / "s"
    config_path = session / "em2mem" / "memory_config.json"
    old = {"status": "memory_ready", "latest_ready_memory_version": 1,
           "building_memory_version": None, "memory_build_state": "failed", "last_build_error": "old"}
    new = {"status": "memory_ready", "latest_ready_memory_version": 2,
           "building_memory_version": 3, "memory_build_state": "building", "new_root": "immutable/v2"}
    write_json_atomic(config_path, old)
    original = rs._safe_json
    def read_then_publish(path, default):
        value = original(path, default)
        if path == config_path:
            with session_memory_mutation_lock(session):
                write_json_atomic(config_path, new)
        return value
    monkeypatch.setattr(rs, "_safe_json", read_then_publish)
    assert rs.build_session_pipeline_state(session)["long_term"]["build_state"] == "ready_with_warnings"
    assert read_json(config_path) == new


@pytest.mark.parametrize("worker_status", ["failed", "stopped", "ready"])
def test_asr_health_has_distinct_families_without_prefix_double_count(tmp_path, worker_status):
    rs.write_worker_runtime(tmp_path, "asr", status=worker_status, extra={"task_filter": "voice_question"})
    rs.write_worker_runtime(tmp_path, "asr_stream", status="ready", extra={"task_filter": "stream"})
    rs.write_worker_runtime(tmp_path, "asr_stream_2", status="ready", extra={"task_filter": "stream"})
    workers = rs.collect_worker_runtime(tmp_path)
    assert workers["asr"]["instance_count"] == 1
    assert workers["asr"]["status"] == ("failed" if worker_status == "failed" else worker_status)
    assert workers["asr_stream"]["instance_count"] == 2
    assert workers["asr"]["task_filter"] == "voice_question"
    assert {item["worker_instance"] for item in workers["asr_stream"]["instances"]} == {"asr_stream", "asr_stream_2"}


def test_asr_health_missing_and_stale_are_visible(tmp_path):
    assert rs.collect_worker_runtime(tmp_path)["asr"]["status"] == "missing"
    path = rs.write_worker_runtime(tmp_path, "asr_stream", status="ready")
    data = read_json(path)
    data["last_heartbeat"] = "2000-01-01T00:00:00+00:00"
    write_json_atomic(path, data)
    assert rs.collect_worker_runtime(tmp_path)["asr_stream"]["status"] == "stale"
    data.update(pid=-1, last_heartbeat=rs.utc_now_iso())
    write_json_atomic(path, data)
    dead = rs.collect_worker_runtime(tmp_path)["asr_stream"]
    assert dead["status"] == "stale" and not dead["pid_alive"]


def test_monitor_uses_runtime_worker_registry(tmp_path, capsys):
    import monitor_online_pipeline as monitor
    rs.write_worker_runtime(tmp_path, "asr", status="failed", extra={"task_filter": "voice_question"})
    rs.write_worker_runtime(tmp_path, "asr_stream", status="ready", extra={"task_filter": "stream"})
    monitor._print_table({"workers": rs.collect_worker_runtime(tmp_path)})
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("asr ") and "failed" in line and "voice_question" in line for line in lines)
    assert any(line.startswith("asr_stream ") for line in lines)


@pytest.mark.parametrize("explicit_backend", [None, "local", "remote"])
def test_launcher_embedding_defaults_and_explicit_values_agree(tmp_path, explicit_backend):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # Only copied shell files run; no real service, credentials, or network.
    for name in ("start_online_all_workers.sh", "start_online_query_worker.sh", "llm_profile.sh", "text_embedding_profile.sh"):
        source = root / "scripts" / name
        if source.exists():
            shutil.copyfile(source, scripts / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text('#!/bin/sh\nif [ "$1" = "online_query_worker.py" ]; then\n  printf "%s" "$EM2MEM_TEXT_EMBED_BACKEND" > "$REPAIR_CAPTURE"\n  exit 0\nfi\nexit 1\n')
    stub.chmod(0o755)
    capture = tmp_path / "capture"
    env = {"PATH": str(bin_dir) + ":/usr/bin:/bin", "HOME": str(tmp_path),
           "PYTHON_BIN": str(stub), "EM2MEM_LLM_PROFILE_OVERRIDE": "remote", "REPAIR_CAPTURE": str(capture)}
    if explicit_backend is not None:
        env["EM2MEM_TEXT_EMBED_BACKEND"] = explicit_backend
    master = subprocess.run(["bash", str(scripts / "start_online_all_workers.sh"), "--dry-run"],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    subprocess.run(["bash", str(scripts / "start_online_query_worker.sh")],
                   env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    expected = explicit_backend or "remote"
    assert capture.read_text() == expected
    assert ("starting qwen3_embedding:" in master.stdout) is (expected == "remote")
