import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_lightweight_stubs(monkeypatch) -> None:
    online_preprocess = types.ModuleType("online_preprocess")
    online_preprocess.__path__ = []
    io_utils = types.ModuleType("online_preprocess.io_utils")
    io_utils.read_json = lambda path, default=None: json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default
    io_utils.utc_now_iso = lambda: "2026-07-08T00:00:00+00:00"
    io_utils.write_json_atomic = lambda path, data: _write_json(Path(path), data)
    monkeypatch.setitem(sys.modules, "online_preprocess", online_preprocess)
    monkeypatch.setitem(sys.modules, "online_preprocess.io_utils", io_utils)

    runtime_state = types.ModuleType("online_pipeline.runtime_state")
    runtime_state.queue_counts = lambda project_root: {}
    monkeypatch.setitem(sys.modules, "online_pipeline.runtime_state", runtime_state)


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_day_child_metadata(child_dir: Path, *, parent_id: str, child_id: str, day_index: int = 2) -> None:
    payload = {
        "session_id": child_id,
        "parent_session_id": parent_id,
        "child_session_id": child_id,
        "is_rokid_day_child": True,
        "day_label": f"DAY{day_index}",
        "day_index": day_index,
        "run_id": "run-1",
    }
    _write_json(child_dir / "metadata.json", payload)


def test_warmup_loads_parent_long_term_for_rokid_child(monkeypatch, tmp_path: Path) -> None:
    _install_lightweight_stubs(monkeypatch)
    parent_id = "parent"
    child_id = "parent__day0002"
    parent_dir = tmp_path / parent_id
    child_dir = tmp_path / child_id
    _write_day_child_metadata(child_dir, parent_id=parent_id, child_id=child_id)
    _write_json(parent_dir / "em2mem" / "memory_config.json", {"status": "memory_ready", "latest_ready_memory_version": 1})
    _write_json(child_dir / "em2mem" / "memory_config.json", {"status": "memory_ready", "latest_ready_memory_version": 2})

    loaded: list[str] = []

    class FakeCache:
        def get_or_load(self, *, session_id: str, long_term_retrieval_scheme: str, loader):
            loaded.append(session_id)
            return loader(session_id), False, 7

    fake_visual = types.ModuleType("online_visual.vlm2vec_runtime")
    fake_visual.get_global_vlm2vec_runtime = lambda: types.SimpleNamespace(
        info=lambda: {"backend": "local"},
        backend="local",
    )
    fake_query_package = types.ModuleType("online_query")
    fake_query_package.__path__ = [str(ROOT / "online_query")]
    fake_query_cache = types.ModuleType("online_query.query_cache")
    fake_query_cache.GLOBAL_SESSION_ENGINE_CACHE = FakeCache()
    fake_query_engine = types.ModuleType("online_query.query_engine")
    fake_query_engine._get_short_term_answer_model = lambda: object()
    fake_query_engine.load_query_engine = lambda session_id, **kwargs: {"session_id": session_id}
    monkeypatch.setitem(sys.modules, "online_query", fake_query_package)
    monkeypatch.setitem(sys.modules, "online_query.query_cache", fake_query_cache)
    monkeypatch.setitem(sys.modules, "online_query.query_engine", fake_query_engine)
    monkeypatch.setitem(sys.modules, "online_visual", types.ModuleType("online_visual"))
    monkeypatch.setitem(sys.modules, "online_visual.vlm2vec_runtime", fake_visual)

    warmup = _load_module("warmup_under_test", "online_query/warmup.py")

    monkeypatch.setattr(warmup, "_env_bool", lambda name, default: False)

    result = warmup.warm_query_session(
        session_id=child_id,
        sessions_root=tmp_path,
        cache=FakeCache(),
        wait_for_memory=False,
    )

    assert result["status"] == "ready"
    assert result["session_id"] == child_id
    assert result["requested_session_id"] == child_id
    assert result["long_term_session_id"] == parent_id
    assert result["parent_session_id"] == parent_id
    assert result["is_rokid_day_child"] is True
    assert loaded == [parent_id]

    state = json.loads((child_dir / "em2mem" / "query_warmup_state.json").read_text(encoding="utf-8"))
    assert state["session_id"] == child_id
    assert state["long_term_session_id"] == parent_id
    assert state["steps"][-1]["loaded_session_id"] == parent_id


def test_warmup_falls_back_to_child_long_term_when_parent_not_ready(monkeypatch, tmp_path: Path) -> None:
    _install_lightweight_stubs(monkeypatch)
    parent_id = "parent"
    child_id = "parent__day0002"
    parent_dir = tmp_path / parent_id
    child_dir = tmp_path / child_id
    parent_dir.mkdir(parents=True)
    _write_day_child_metadata(child_dir, parent_id=parent_id, child_id=child_id)
    _write_json(child_dir / "em2mem" / "memory_config.json", {"status": "memory_ready", "latest_ready_memory_version": 2})

    loaded: list[str] = []

    class FakeCache:
        def get_or_load(self, *, session_id: str, long_term_retrieval_scheme: str, loader):
            loaded.append(session_id)
            return loader(session_id), False, 7

    fake_visual = types.ModuleType("online_visual.vlm2vec_runtime")
    fake_visual.get_global_vlm2vec_runtime = lambda: types.SimpleNamespace(
        info=lambda: {"backend": "local"},
        backend="local",
    )
    fake_query_package = types.ModuleType("online_query")
    fake_query_package.__path__ = [str(ROOT / "online_query")]
    fake_query_cache = types.ModuleType("online_query.query_cache")
    fake_query_cache.GLOBAL_SESSION_ENGINE_CACHE = FakeCache()
    fake_query_engine = types.ModuleType("online_query.query_engine")
    fake_query_engine._get_short_term_answer_model = lambda: object()
    fake_query_engine.load_query_engine = lambda session_id, **kwargs: {"session_id": session_id}
    monkeypatch.setitem(sys.modules, "online_query", fake_query_package)
    monkeypatch.setitem(sys.modules, "online_query.query_cache", fake_query_cache)
    monkeypatch.setitem(sys.modules, "online_query.query_engine", fake_query_engine)
    monkeypatch.setitem(sys.modules, "online_visual", types.ModuleType("online_visual"))
    monkeypatch.setitem(sys.modules, "online_visual.vlm2vec_runtime", fake_visual)

    warmup = _load_module("warmup_child_fallback_under_test", "online_query/warmup.py")

    monkeypatch.setattr(warmup, "_env_bool", lambda name, default: False)

    result = warmup.warm_query_session(
        session_id=child_id,
        sessions_root=tmp_path,
        cache=FakeCache(),
        wait_for_memory=False,
    )

    assert result["status"] == "ready"
    assert result["long_term_session_id"] == child_id
    assert result["long_term_selection"]["selected_role"] == "current_child"
    assert loaded == [child_id]

    state = json.loads((child_dir / "em2mem" / "query_warmup_state.json").read_text(encoding="utf-8"))
    assert state["long_term_session_id"] == child_id
    assert state["steps"][-1]["loaded_session_id"] == child_id


def test_stream_query_context_uses_parent_for_long_term_readiness(monkeypatch, tmp_path: Path) -> None:
    _install_lightweight_stubs(monkeypatch)
    parent_id = "parent"
    child_id = "parent__day0002"
    parent_dir = tmp_path / parent_id
    child_dir = tmp_path / child_id
    _write_day_child_metadata(child_dir, parent_id=parent_id, child_id=child_id)

    _write_json(child_dir / "stream" / "stream_state.json", {"status": "running", "upload_chunks": [{"chunk_index": 3}]})
    _write_json(child_dir / "pipeline_state.json", {"current": {"ready": True}, "short_term": {"ready": True}, "long_term": {"long_term_partial_ready": False}})
    _write_json(child_dir / "em2mem" / "memory_config.json", {"latest_ready_memory_version": 9})
    _write_json(child_dir / "stream" / "transcript" / "partial_transcript_state.json", {"segment_count": 1, "last_asr_chunk_index": 3})
    _write_json(child_dir / "stream" / "frame_state.json", {"ready": True})
    _write_json(child_dir / "stream" / "frame_event_state.json", {})
    _write_json(child_dir / "current" / "current_state.json", {"current_text_ready": True})

    _write_json(parent_dir / "pipeline_state.json", {"long_term": {"long_term_partial_ready": False, "long_term_full_ready": False}})
    _write_json(parent_dir / "em2mem" / "memory_config.json", {"latest_ready_memory_version": 1, "latest_fast_ready_version": 1})
    _write_json(tmp_path / "online_tasks" / "query_runtime.json", {"loaded_sessions": [{"session_id": parent_id, "active_query_memory_version": 1}]})

    stream_query_context = _load_module("stream_query_context_under_test", "online_query/stream_query_context.py")

    monkeypatch.setattr(stream_query_context, "queue_counts", lambda project_root: {})

    context = stream_query_context.load_stream_query_context(
        child_id,
        sessions_root=tmp_path,
        project_root=tmp_path,
        question="summarize what happened yesterday",
    )

    assert context["is_stream_session"] is True
    assert context["is_rokid_day_child"] is True
    assert context["parent_session_id"] == parent_id
    assert context["long_term_session_id"] == parent_id
    assert context["current_ready"] is True
    assert context["short_term_ready"] is True
    assert context["long_term_partial_ready"] is True
    assert context["latest_fast_ready_version"] == 1
    assert context["active_query_memory_version"] == 1


def test_stream_query_context_uses_child_when_parent_not_ready(monkeypatch, tmp_path: Path) -> None:
    _install_lightweight_stubs(monkeypatch)
    parent_id = "parent"
    child_id = "parent__day0002"
    parent_dir = tmp_path / parent_id
    child_dir = tmp_path / child_id
    parent_dir.mkdir(parents=True)
    _write_day_child_metadata(child_dir, parent_id=parent_id, child_id=child_id)

    _write_json(child_dir / "stream" / "stream_state.json", {"status": "running", "upload_chunks": [{"chunk_index": 3}]})
    _write_json(child_dir / "pipeline_state.json", {"current": {"ready": True}, "short_term": {"ready": True}})
    _write_json(child_dir / "em2mem" / "memory_config.json", {"latest_ready_memory_version": 9, "latest_fast_ready_version": 9})
    _write_json(child_dir / "stream" / "transcript" / "partial_transcript_state.json", {"segment_count": 1, "last_asr_chunk_index": 3})
    _write_json(child_dir / "stream" / "frame_state.json", {"ready": True})
    _write_json(child_dir / "stream" / "frame_event_state.json", {})
    _write_json(child_dir / "current" / "current_state.json", {"current_text_ready": True})
    _write_json(tmp_path / "online_tasks" / "query_runtime.json", {"loaded_sessions": [{"session_id": child_id, "active_query_memory_version": 9}]})

    stream_query_context = _load_module("stream_query_context_child_fallback_under_test", "online_query/stream_query_context.py")

    monkeypatch.setattr(stream_query_context, "queue_counts", lambda project_root: {})

    context = stream_query_context.load_stream_query_context(
        child_id,
        sessions_root=tmp_path,
        project_root=tmp_path,
        question="summarize what happened today",
    )

    assert context["is_stream_session"] is True
    assert context["is_rokid_day_child"] is True
    assert context["parent_session_id"] == parent_id
    assert context["long_term_session_id"] == child_id
    assert context["long_term_selection"]["selected_role"] == "target_child"
    assert context["long_term_partial_ready"] is True
    assert context["active_query_memory_version"] == 9
