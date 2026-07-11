from pathlib import Path
import asyncio
import json
import os
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_api_import_stubs() -> None:
    os.environ.setdefault("WORLDMM_ENABLE_DEMO_ROUTES", "0")

    online_preprocess = types.ModuleType("online_preprocess")
    online_preprocess.__path__ = [str(ROOT / "online_preprocess")]
    sys.modules["online_preprocess"] = online_preprocess

    online_pipeline = types.ModuleType("online_pipeline")
    online_pipeline.__path__ = [str(ROOT / "online_pipeline")]
    sys.modules["online_pipeline"] = online_pipeline
    stream_timeline = types.ModuleType("online_pipeline.stream_timeline")
    stream_timeline.append_timeline_event = lambda *args, **kwargs: None
    sys.modules["online_pipeline.stream_timeline"] = stream_timeline

    fastapi = types.ModuleType("fastapi")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def on_event(self, *args, **kwargs):
            return lambda fn: fn

    class _DummyUploadFile:
        pass

    def _param(*args, **kwargs):
        return kwargs.get("default")

    fastapi.Body = _param
    fastapi.FastAPI = _DummyFastAPI
    fastapi.File = _param
    fastapi.Form = _param
    fastapi.Header = _param
    fastapi.Request = object
    fastapi.UploadFile = _DummyUploadFile
    sys.modules["fastapi"] = fastapi

    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors

    responses = types.ModuleType("fastapi.responses")

    class _DummyJSONResponse:
        def __init__(self, status_code=200, content=None, **kwargs):
            self.status_code = status_code
            self.content = content or {}
            self.body = json.dumps(self.content).encode("utf-8")

    class _DummyResponse:
        def __init__(self, *args, **kwargs):
            pass

    responses.FileResponse = _DummyResponse
    responses.JSONResponse = _DummyJSONResponse
    responses.StreamingResponse = _DummyResponse
    sys.modules["fastapi.responses"] = responses

    pydantic = types.ModuleType("pydantic")

    class _DummyBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic.BaseModel = _DummyBaseModel
    sys.modules["pydantic"] = pydantic


_install_api_import_stubs()

import api_server  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_augments_day_asset_frames_with_parent_file_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_server, "ONLINE_SESSIONS_DIR", tmp_path)
    _touch(tmp_path / "parent1" / "stream" / "day_assets" / "DAY1" / "stream" / "frames" / "frame_000074.jpg")
    _touch(tmp_path / "parent1__day0002" / "stream" / "frames" / "frame_000051.jpg")
    payload = {
        "session_id": "parent1__day0002",
        "result": {
            "session_id": "parent1__day0002",
            "stream_context": {
                "long_term_session_id": "parent1",
                "parent_session_id": "parent1",
            },
            "evidence_frames": [
                {"path": "stream/day_assets/DAY1/stream/frames/frame_000074.jpg"},
                {"path": "stream/frames/frame_000051.jpg"},
            ],
        },
        "evidence_frames": [
            {"path": "stream/day_assets/DAY1/stream/frames/frame_000074.jpg"},
            {"path": "stream/frames/frame_000051.jpg"},
        ],
    }

    enhanced = api_server._augment_evidence_frames_for_response(payload, "parent1__day0002")

    frames = enhanced["result"]["evidence_frames"]
    assert enhanced["evidence_frames"] == frames
    assert frames[0]["owner_session_id"] == "parent1"
    assert frames[0]["file_url"] == "/session/parent1/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fframe_000074.jpg"
    assert frames[0]["relative_file_url"] == "/session/parent1/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fframe_000074.jpg"
    assert frames[0]["image_url"] == frames[0]["file_url"]
    assert frames[0]["url"] == frames[0]["file_url"]
    assert frames[0]["src"] == frames[0]["file_url"]
    assert frames[0]["file_available"] is True
    assert frames[1]["owner_session_id"] == "parent1__day0002"
    assert frames[1]["file_url"] == "/session/parent1__day0002/file?path=stream%2Fframes%2Fframe_000051.jpg"
    assert frames[1]["file_available"] is True


def test_existing_owner_session_is_preserved_and_missing_file_is_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_server, "ONLINE_SESSIONS_DIR", tmp_path)
    _touch(tmp_path / "explicit_owner" / "stream" / "day_assets" / "DAY1" / "stream" / "frames" / "frame_000074.jpg")
    payload = {
        "stream_context": {"long_term_session_id": "parent1"},
        "evidence_frames": [
            {
                "path": "stream/day_assets/DAY1/stream/frames/frame_000074.jpg",
                "owner_session_id": "explicit_owner",
            },
            {"path": "stream/day_assets/DAY1/stream/frames/missing.jpg"},
        ],
    }

    enhanced = api_server._augment_evidence_frames_for_response(payload, "parent1__day0002")

    frames = enhanced["evidence_frames"]
    assert frames[0]["owner_session_id"] == "explicit_owner"
    assert frames[0]["file_url"] == "/session/explicit_owner/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fframe_000074.jpg"
    assert frames[0]["file_available"] is True
    assert frames[1]["owner_session_id"] == "parent1"
    assert frames[1]["file_url"] == "/session/parent1/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fmissing.jpg"
    assert frames[1]["file_available"] is False


def test_api_base_url_makes_browser_ready_absolute_image_urls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_server, "ONLINE_SESSIONS_DIR", tmp_path)
    _touch(tmp_path / "parent1" / "stream" / "day_assets" / "DAY1" / "stream" / "frames" / "frame_000074.jpg")
    payload = {
        "stream_context": {"long_term_session_id": "parent1"},
        "evidence_frames": [
            {"path": "stream/day_assets/DAY1/stream/frames/frame_000074.jpg"},
        ],
    }

    enhanced = api_server._augment_evidence_frames_for_response(
        payload,
        "parent1__day0002",
        api_base_url="http://127.0.0.1:8000",
    )

    frame = enhanced["evidence_frames"][0]
    assert frame["relative_file_url"] == "/session/parent1/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fframe_000074.jpg"
    assert frame["file_url"] == "http://127.0.0.1:8000/session/parent1/file?path=stream%2Fday_assets%2FDAY1%2Fstream%2Fframes%2Fframe_000074.jpg"
    assert frame["image_url"] == frame["file_url"]
    assert frame["url"] == frame["file_url"]
    assert frame["src"] == frame["file_url"]
    assert frame["thumbnail_url"] == frame["file_url"]
    assert frame["file_available"] is True


def test_historical_ask_bypasses_single_active_session_without_changing_legacy_ask(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    sessions_root = project_root / "online_sessions"
    monkeypatch.setenv("WORLDMM_SINGLE_ACTIVE_SESSION", "true")
    monkeypatch.setattr(api_server, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(api_server, "ONLINE_SESSIONS_DIR", sessions_root)
    _write_json(project_root / "runtime" / "active_session.json", {"active_session_id": "active_session"})
    _write_json(sessions_root / "old_session" / "worldmm" / "memory_config.json", {"status": "memory_ready"})
    _write_json(sessions_root / "old_session" / "status.json", {"stage": "memory_ready", "progress": 100})

    legacy_request = api_server.AskRequest(question="what happened?", mode="async")
    legacy_response = asyncio.run(api_server.ask_session("old_session", legacy_request))

    assert legacy_response.status_code == 409
    assert legacy_response.content["status"] == "inactive_session"

    historical_request = api_server.AskRequest(question="what happened?", mode="async")
    historical_response = asyncio.run(api_server.ask_historical_session("old_session", historical_request))

    assert historical_response.status_code == 202
    assert historical_response.content["status"] == "queued"
    task_id = historical_response.content["task_id"]
    task_payload = json.loads((project_root / "online_tasks" / "query" / f"{task_id}.json").read_text(encoding="utf-8"))
    assert task_payload["session_id"] == "old_session"
    assert task_payload["allow_inactive_session"] is True
    assert task_payload["task_source"] == "session_history_api"
