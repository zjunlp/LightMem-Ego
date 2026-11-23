import asyncio
import io
import json
import threading
import time

import pytest
from starlette.requests import Request
from starlette.datastructures import UploadFile

from online_memory_transfer import MemoryImportService, MemoryTransferError, export_memory, parse_memory
from online_preprocess.io_utils import read_json, write_json_atomic


def publish(session, root, project, report):
    from online_memory import build_online_em2mem_memory
    report("building_memory", 25)
    config_path = build_online_em2mem_memory(session.name, root, source="imported_memory", skip_visual_embedding=True,
                                           model_name="import-test", force=True)
    config = read_json(config_path, {})
    config["retrieval_artifacts_ready"] = True
    write_json_atomic(config_path, config)
    report("warming_query", 95)
    return {"memory_version": config["memory_version"]}


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("EM2MEM_MEMORY_GENERATION_BACKEND", "rule")
    return MemoryImportService(tmp_path, tmp_path, builder=publish, schedule=False)


@pytest.mark.parametrize("data,name,code", [
    (b"", "memory.txt", "memory_import_empty"),
    (b"abc", "memory.zip", "unsupported_import_extension"),
    (b"\xff", "memory.txt", "memory_import_invalid_utf8"),
    (b"{broken", "memory.json", "memory_import_invalid_json"),
    (b'{"text":NaN}', "memory.json", "memory_import_invalid_json"),
    (b'{"not_memory":1}', "memory.json", "invalid_import_format"),
    (b'{"text":"memory","start":20,"end":10}', "memory.json", "invalid_import_format"),
    (b'{"text":"memory","main_actions":"bad"}', "memory.json", "invalid_import_format"),
])
def test_invalid_upload_does_not_create_a_session(service, data, name, code):
    with pytest.raises(MemoryTransferError) as exc:
        service.receive(io.BytesIO(data), name)
    assert exc.value.code == code
    assert not list(service.root.iterdir())


def test_upload_limit_and_record_limit(service, monkeypatch):
    import online_memory_transfer as transfer
    monkeypatch.setattr(transfer, "MAX_BYTES", 20)
    with pytest.raises(MemoryTransferError) as exc:
        service.receive(io.BytesIO(b"x" * 21), "memory.txt")
    assert exc.value.status_code == 413
    monkeypatch.setattr(transfer, "MAX_RECORDS", 1)
    with pytest.raises(MemoryTransferError) as exc:
        parse_memory(b'["a","b"]', "memory.json")
    assert exc.value.code == "memory_import_record_limit"


def test_txt_import_builds_real_memory_without_requiring_a_video(service):
    from online_memory_view.service import LongTermMemoryViewService
    from online_memory_view.catalog import LongTermMemoryCatalogService
    receipt = service.receive(io.BytesIO("钥匙放在副驾。\n\n背包在后排。".encode()), "记忆.txt", title="导入测试")
    sid, task = receipt["session_id"], receipt["task_id"]
    assert receipt["status"] == receipt["state"] == "queued"
    assert receipt["query_ready"] is False
    assert LongTermMemoryCatalogService(service.root).load().payload["items"] == []
    service.run(task)
    result = service.status(task)
    assert result["state"] == "ready", result.get("error")
    assert result["record_count"] == 2 and result["query_ready"] is True
    memory = LongTermMemoryViewService(service.root, sid).load().payload
    assert memory["title"] == "导入测试" and memory["session_kind"] == "imported_memory"
    records = memory["episodic"]["30sec"]
    assert {r["content"] for r in records} == {"钥匙放在副驾。", "背包在后排。"}
    assert all(r["event_time"]["start_seconds"] is None for r in records)
    assert LongTermMemoryCatalogService(service.root).load().payload["total_ready_sessions"] == 1
    assert not (service.root / sid / "input.mp4").exists()


def test_export_import_round_trip_retains_full_text_and_real_times(service):
    from online_memory_view.service import LongTermMemoryViewService
    content = "手机贴耳。" * 2000
    payload = [{"text": content, "start_time": "00004300", "end_time": "00004500",
                "visual_summary": "男子朝向车窗。", "main_actions": [{"actor": "男子", "action": "举手机", "time_span": [43, 45]}],
                "keyframe_paths": ["/private/other-session.jpg"], "api_key": "must-not-be-restored"}]
    first = service.receive(io.BytesIO(json.dumps(payload).encode()), "memory.json")
    service.run(first["task_id"])
    view = LongTermMemoryViewService(service.root, first["session_id"]).load().payload
    record = view["episodic"]["30sec"][0]
    exported = export_memory(service.root, first["session_id"], record["id"])
    encoded = json.dumps(exported)
    assert "must-not-be-restored" not in encoded and "/private/" not in encoded
    assert exported["record"]["text"] == content
    second = service.receive(io.BytesIO(encoded.encode()), "export.json")
    service.run(second["task_id"])
    reread = LongTermMemoryViewService(service.root, second["session_id"]).load().payload["episodic"]["30sec"][0]
    assert reread["content"] == content
    assert reread["event_time"]["start_seconds"] == 43
    assert reread["event_time"]["end_seconds"] == 45
    assert first["session_id"] != second["session_id"]


def test_failed_or_unindexed_memory_never_becomes_query_ready(service):
    service.builder = lambda *args: {}
    receipt = service.receive(io.BytesIO(b"A red bag is on the seat."), "memory.txt")
    service.run(receipt["task_id"])
    result = service.status(receipt["task_id"])
    assert result["state"] == "failed" and result["query_ready"] is False
    assert result["error"]["code"] == "memory_import_failed"


def test_unknown_task_and_traversal_are_rejected(service):
    for task in ("../another-session", "missing", "0" * 32):
        with pytest.raises(MemoryTransferError) as exc:
            service.status(task)
        assert exc.value.status_code == 404


def test_restart_recovers_queued_import_exactly_once(service):
    receipt = service.receive(io.BytesIO(b"A blue bag."), "memory.txt")
    scheduled = []
    service.ensure_scheduled = scheduled.append
    service.recover()
    assert scheduled == [receipt["task_id"]]
    service.run(receipt["task_id"])
    service.run(receipt["task_id"])
    assert service.status(receipt["task_id"])["memory_version"] == 1


def test_api_upload_poll_and_export_contract(service, monkeypatch):
    import api_server as api
    from online_memory_view.service import LongTermMemoryViewService
    monkeypatch.setattr(api, "ONLINE_SESSIONS_DIR", service.root)
    monkeypatch.setattr(api, "_memory_import_service", lambda: service)
    request = Request({"type": "http", "headers": [(b"x-request-id", b"test-transfer")]})
    file = UploadFile(file=io.BytesIO(b"Keys on passenger seat."), filename="memory.txt")
    response = asyncio.run(api.create_memory_import(request, file=file, format="auto", title="Keys"))
    assert response.status_code == 202 and response.headers["x-request-id"] == "test-transfer"
    result = json.loads(response.body)
    service.run(result["task_id"])
    polled = asyncio.run(api.memory_import_status(result["task_id"], request))
    assert json.loads(polled.body)["state"] == "ready"
    record = LongTermMemoryViewService(service.root, result["session_id"]).load().payload["episodic"]["30sec"][0]
    exported = asyncio.run(api.export_session_memory(result["session_id"], record["id"], request))
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    assert json.loads(exported.body)["record"]["text"] == "Keys on passenger seat."
    status = json.loads(asyncio.run(api.session_status(result["session_id"])).body)
    assert status["query_allowed"] is True and status["query_ready"] is True


def test_import_runs_off_the_api_event_loop(service, monkeypatch):
    import api_server as api
    main_thread = threading.get_ident()
    threads = []
    def receive(*args):
        threads.append(threading.get_ident())
        time.sleep(.02)
        return {"status": "queued"}
    monkeypatch.setattr(service, "receive", receive)
    monkeypatch.setattr(api, "_memory_import_service", lambda: service)
    file = UploadFile(file=io.BytesIO(b"abc"), filename="a.txt")
    asyncio.run(api.create_memory_import(Request({"type": "http", "headers": []}), file, "auto", ""))
    assert threads and threads[0] != main_thread


def test_untimed_prompt_does_not_present_an_observed_zero_second():
    from online_evidence_content import prompt_evidence
    result, _ = prompt_evidence("Where are the keys?", [{"caption": "Keys are on the seat.", "start_time": 0, "end_time": 0,
        "metadata": {"timestamp_known": False}}])
    assert result[0]["time"] == [None, None]
    assert "no observed capture time" in result[0]["source_note"]


def test_real_http_multipart_and_download_match_frontend(service, monkeypatch):
    import httpx
    import api_server as api
    monkeypatch.setattr(api, "ONLINE_SESSIONS_DIR", service.root)
    monkeypatch.setattr(api, "_memory_import_service", lambda: service)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            uploaded = await client.post("/memory-imports", files={"file": ("a.txt", b"A red backpack.", "text/plain")},
                                         data={"format": "auto", "title": "Backpack"})
            assert uploaded.status_code == 202, uploaded.text
            receipt = uploaded.json()
            pending = await client.get("/memory-imports/" + receipt["task_id"])
            assert pending.json()["state"] == "queued"
            not_ready = await client.post("/session/" + receipt["session_id"] + "/ask", json={"question": "Which bag?"})
            assert not_ready.status_code == 409
            await asyncio.to_thread(service.run, receipt["task_id"])
            view = (await client.get("/session/" + receipt["session_id"] + "/memories")).json()
            mid = view["episodic"]["30sec"][0]["id"]
            exported = await client.get(f'/session/{receipt["session_id"]}/memories/{mid}/export?format=json')
            assert exported.status_code == 200
            assert exported.json()["record"]["text"] == "A red backpack."
            assert "attachment" in exported.headers["content-disposition"]
            invalid = await client.post("/memory-imports", files={"file": ("bad.json", b"{bad", "application/json")})
            assert invalid.status_code == 400
            assert invalid.json()["error"]["code"] == "memory_import_invalid_json"
            missing = await client.get(f'/session/{receipt["session_id"]}/memories/missing/export')
            assert missing.status_code == 404
    asyncio.run(run())


def test_export_uses_latest_edited_snapshot(service):
    from online_memory_view.service import LongTermMemoryViewService
    from online_memory_edit.service import MemoryEditService
    receipt = service.receive(io.BytesIO(b"A red bag is on the seat."), "a.txt")
    service.run(receipt["task_id"])
    sid = receipt["session_id"]
    before = LongTermMemoryViewService(service.root, sid).load().payload
    mid = before["episodic"]["30sec"][0]["id"]
    editor = MemoryEditService(service.root, sid, runtime_validator=lambda *args: None,
                               derivation_backend_override="rule")
    editor.update({"baseVersion": 1, "records": [{"id": mid, "content": "A blue bag is on the seat."}]}, "test-export-current")
    exported = export_memory(service.root, sid, mid)
    assert exported["memory_version"] == 2
    assert exported["record"]["text"] == "A blue bag is on the seat."


def test_bad_recovery_record_does_not_prevent_other_imports(service):
    bad = service.root / "import_bad" / "memory_import.json"
    write_json_atomic(bad, {"state": "queued", "task_id": "../bad"})
    receipt = service.receive(io.BytesIO(b"Some valid memory."), "memory.txt")
    scheduled = []
    service.ensure_scheduled = scheduled.append
    service.recover()
    assert scheduled == [receipt["task_id"]]
