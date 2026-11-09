"""Native website editor contracts, publication, and nonblocking API writes."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from online_memory.evidence_to_em2mem import (
    evidence_doc_to_caption_item, write_caption_files, write_sidecar_files,
    write_semantic_files, _load_triplet_map,
)
from online_memory.em2mem_layout import Em2MemOnlineLayout, ensure_em2mem_layout
from online_memory_edit.service import MemoryEditError, MemoryEditService
from online_memory_view.service import LongTermMemoryViewService
from online_preprocess.io_utils import read_json, write_json_atomic


@pytest.fixture
def memory_session(tmp_path):
    sid = "editor_test"
    layout = Em2MemOnlineLayout(tmp_path / sid, sid)
    ensure_em2mem_layout(layout)
    record = evidence_doc_to_caption_item(sid, {
        "doc_id": "event_a", "start": 0, "end": 30,
        "start_time": 0, "end_time": 30, "fine_caption": "A red bag rests on the seat.",
    }, 0)
    write_caption_files(layout, [record], model_name="fake", generation_backend="rule")
    scales = {s: read_json(layout.caption_root / f"{sid}_{s}.json", default=[]) for s in ("30sec", "3min", "10min", "1h")}
    sides = write_sidecar_files(layout, "fake", scales, generation_backend="rule")
    write_semantic_files(layout, "fake", [record], generation_backend="rule", triplet_map=_load_triplet_map(sides["30sec"]["triplets"]))
    write_json_atomic(layout.memory_config_path, {
        "status": "memory_ready", "memory_version": 1, "latest_ready_memory_version": 1,
        "semantic_version": 1, "graph_version": 1, "visual_version": 1,
        "caption_root": "em2mem/caption_root", "sidecar_root": "em2mem/sidecar_root",
        "semantic_root": "em2mem/semantic_root", "visual_root": "em2mem/visual_root",
        "memory_generation_backend": "rule", "updated_at": "2026-11-05T00:00:00Z",
        "query_rag_args": {"subject": sid, "retriever_model": "fake", "episodic_caption_root": "em2mem/caption_root",
                           "episodic_sidecar_root": "em2mem/sidecar_root", "semantic_root": "em2mem/semantic_root"},
    })
    return tmp_path, sid, layout


def test_native_editor_can_edit_save_retry_conflict_and_undo(memory_session):
    root, sid, layout = memory_session
    before = LongTermMemoryViewService(root, sid).load().payload
    assert before["canEdit"] is True and before["editDisabledReason"] is None
    item = before["episodic"]["30sec"][0]
    service = MemoryEditService(root, sid, runtime_validator=lambda *a: None, derivation_backend_override="rule")
    body = {"baseVersion": 1, "records": [{"id": item["id"], "content": "A blue bag rests on the seat."}]}
    saved = service.update(body, "native-editor-save")
    memory = saved.payload["memory"]
    assert memory["canEdit"] is True
    assert memory["canRollback"] is True
    assert memory["memory_version"] == 2
    assert memory["episodic"]["30sec"][0]["content"] == body["records"][0]["content"]
    assert saved.payload["propagation"]["targetMemoryVersion"] == 2
    assert service.update(body, "native-editor-save").replayed is True
    with pytest.raises(MemoryEditError) as exc:
        service.update(body, "native-editor-stale")
    assert exc.value.code == "memory_version_conflict"
    undone = service.rollback({"currentVersion": 2}, "native-editor-undo").payload["memory"]
    assert undone["memory_version"] == 3 and undone["canRollback"] is False
    assert undone["episodic"]["30sec"][0]["content"] == item["content"]


def test_no_editable_records_reports_disabled_reason(memory_session):
    root, sid, layout = memory_session
    write_json_atomic(layout.caption_30sec_path, [])
    result = LongTermMemoryViewService(root, sid).load().payload
    assert result["canEdit"] is False
    assert result["editDisabledReason"] == "no_editable_30sec"


@pytest.mark.parametrize("rollback", [False, True])
def test_memory_mutation_does_not_block_api_event_loop(monkeypatch, rollback):
    import api_server as api
    import online_memory_edit.service as module
    called = []
    main_thread = threading.get_ident()
    class Service:
        def __init__(self, *args, **kwargs): pass
        def update(self, body, key):
            called.append(threading.get_ident())
            time.sleep(.03)
            return SimpleNamespace(status_code=200, payload={"status": "ok"})
        rollback = update
    monkeypatch.setattr(module, "MemoryEditService", Service)
    request = Request({"type": "http", "headers": []})
    handler = api.rollback_session_memories_30sec if rollback else api.update_session_memories_30sec
    response = asyncio.run(handler("editor_test", request, body={}, idempotency_key="nonblocking-test"))
    assert json.loads(response.body)["status"] == "ok"
    assert called and called[0] != main_thread


def test_unchanged_graph_304_has_no_json_body(monkeypatch):
    import api_server as api
    import online_memory_view.graph_service as module
    class GraphService:
        def __init__(self, *args): pass
        def load(self, scale, etag):
            assert etag == '"version-1"'
            return SimpleNamespace(not_modified=True, payload=None, etag=etag)
    monkeypatch.setattr(module, "LongTermMemoryGraphService", GraphService)
    request = Request({"type": "http", "headers": [(b"if-none-match", b'"version-1"')]})
    response = asyncio.run(api.get_session_memory_graph("editor_test", request, scale="30sec"))
    assert response.status_code == 304
    assert response.body == b""
    assert response.headers["etag"] == '"version-1"'
