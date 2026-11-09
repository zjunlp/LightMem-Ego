"""Read-only session video metadata and actual HTTP byte-range serving."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_server


@pytest.fixture
def media_client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server, "ONLINE_SESSIONS_DIR", tmp_path)
    return TestClient(api_server.app), tmp_path


def make_session(root, metadata=None):
    session = root / "demo_case"
    session.mkdir()
    if metadata is not None:
        (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return session


def test_bound_media_metadata_and_actual_byte_range(media_client):
    client, root = media_client
    session = make_session(root, {"demo_media": {
        "path": "media/front.mp4", "title": "道路前视",
        "duration_seconds": 120, "time_origin": "session_relative_seconds", "video_start_seconds": 0,
        "view_description": "前视；本地时间从零开始",
    }})
    video = session / "media/front.mp4"
    video.parent.mkdir()
    content = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256))
    video.write_bytes(content)
    response = client.get("/session/demo_case/demo-media")
    assert response.status_code == 200
    version = response.json()["video_version"]
    assert response.json() == {
        "status": "ok", "session_id": "demo_case", "title": "道路前视",
        "available": True, "video_url": f"/session/demo_case/demo-video?v={version}", "video_version": version,
        "duration_seconds": 120, "time_origin": "session_relative_seconds",
        "video_start_seconds": 0, "time_alignment_known": True, "memory_until_seconds": None,
        "view_description": "前视；本地时间从零开始", "reason": None,
    }
    response = client.get("/session/demo_case/demo-video", headers={"Range": "bytes=12-31"})
    assert response.status_code == 206
    assert response.content == content[12:32]
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-range"] == f"bytes 12-31/{len(content)}"
    assert response.headers["accept-ranges"] == "bytes"
    assert client.get("/session/demo_case/demo-video").content == content
    assert client.get("/session/demo_case/file?path=media/front.mp4").status_code == 403


def test_playback_version_never_mixes_ranges_from_replaced_files(media_client):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "input.mp4"}})
    (session / "input.mp4").write_bytes(b"old recording data")
    old = client.get("/session/demo_case/demo-media").json()
    assert client.get(old["video_url"], headers={"Range": "bytes=4-8"}).content == b"recor"
    (session / "input.mp4").write_bytes(b"new compressed recording has different byte offsets")
    new = client.get("/session/demo_case/demo-media").json()
    assert old["video_url"] != new["video_url"]
    assert client.get(old["video_url"], headers={"Range": "bytes=4-8"}).status_code == 409
    fresh = client.get(new["video_url"], headers={"Range": "bytes=4-13"})
    assert fresh.status_code == 206 and fresh.content == b"compressed"
    assert "immutable" in fresh.headers["cache-control"]
    assert client.get("/session/demo_case/demo-video").headers["cache-control"] == "no-cache"


def test_existing_input_fallback_ignores_stale_absolute_saved_path(media_client):
    client, root = media_client
    session = make_session(root, {"original_filename": "原视频.mp4", "saved_video_path": "/elsewhere/input.mp4"})
    (session / "input.mp4").write_bytes(b"video")
    result = client.get("/session/demo_case/demo-media").json()
    assert result["available"] is True
    assert result["title"] == "原视频.mp4"
    assert result["duration_seconds"] is None
    assert result["time_origin"] == "session_relative_seconds"
    assert result["time_alignment_known"] is False
    assert result["video_start_seconds"] is None


def test_missing_video_keeps_session_metadata_available(media_client):
    client, root = media_client
    make_session(root, {"original_filename": "Original.mp4"})
    response = client.get("/session/demo_case/demo-media")
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["available"] is False
    assert result["video_url"] is None
    assert result["title"] == "Original.mp4"
    assert result["reason"]
    assert client.get("/session/demo_case/demo-video").status_code == 404


def test_catalog_memory_and_player_share_the_explicit_session_title(media_client):
    client, root = media_client
    session = make_session(root, {
        "title": "车内日常（2分钟）", "title_revision": 2,
        "demo_media": {"title": "旧测试标题", "path": "input.mp4"},
    })
    (session / "em2mem").mkdir()
    (session / "em2mem/memory_config.json").write_text(json.dumps({
        "status": "memory_ready", "memory_version": 1,
    }))
    catalog = client.get("/memories/sessions")
    memories = client.get("/session/demo_case/memories")
    media = client.get("/session/demo_case/demo-media")
    assert catalog.status_code == memories.status_code == media.status_code == 200
    assert catalog.json()["items"][0]["title"] == memories.json()["title"] == media.json()["title"] == "车内日常（2分钟）"
    assert catalog.json()["items"][0]["title_revision"] == memories.json()["title_revision"] == 2


def test_bad_optional_metadata_does_not_hide_ready_memory(media_client):
    client, root = media_client
    session = make_session(root)
    (session / "metadata.json").write_text("{incomplete")
    (session / "em2mem").mkdir()
    (session / "em2mem/memory_config.json").write_text(json.dumps({
        "status": "memory_ready", "memory_version": 1,
    }))
    catalog = client.get("/memories/sessions")
    assert catalog.status_code == 200
    assert catalog.json()["items"][0]["title"] == "demo_case"
    assert client.get("/session/demo_case/memories").status_code == 200


@pytest.mark.parametrize("path", ["../outside.mp4", "/tmp/outside.mp4", "media/../../outside.mp4", "secret.txt", "", None])
def test_invalid_binding_cannot_serve_or_fallback(media_client, path):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": path}})
    (session / "input.mp4").write_bytes(b"fallback must not hide an invalid binding")
    result = client.get("/session/demo_case/demo-media").json()
    assert result["available"] is False
    assert result["video_url"] is None
    assert client.get("/session/demo_case/demo-video").status_code == 404


def test_external_video_symlink_is_not_served(media_client):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "media.mp4"}})
    outside = root / "outside.mp4"
    outside.write_bytes(b"outside")
    (session / "media.mp4").symlink_to(outside)
    assert client.get("/session/demo_case/demo-media").json()["available"] is False
    assert client.get("/session/demo_case/demo-video").status_code == 404


def test_mp4_symlink_cannot_disguise_other_session_file_types(media_client):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "media.mp4"}})
    (session / "secret.txt").write_bytes(b"private non-video file")
    (session / "media.mp4").symlink_to(session / "secret.txt")
    assert client.get("/session/demo_case/demo-media").json()["available"] is False
    assert client.get("/session/demo_case/demo-video").status_code == 404


def test_video_symlink_loop_is_unavailable(media_client):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "loop.mp4"}})
    (session / "loop.mp4").symlink_to("loop.mp4")
    assert client.get("/session/demo_case/demo-media").json()["available"] is False
    assert client.get("/session/demo_case/demo-video").status_code == 404


def test_external_session_symlink_is_rejected(media_client, tmp_path):
    client, root = media_client
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    try:
        (root / "demo_case").symlink_to(outside, target_is_directory=True)
        assert client.get("/session/demo_case/demo-media").status_code == 400
        assert client.get("/session/demo_case/demo-video").status_code == 400
    finally:
        outside.rmdir()


@pytest.mark.parametrize("endpoint", ["demo-media", "demo-video"])
def test_invalid_and_missing_sessions(media_client, endpoint):
    client, _ = media_client
    assert client.get(f"/session/bad.session/{endpoint}").status_code == 400
    assert client.get(f"/session/missing/{endpoint}").status_code == 404


@pytest.mark.parametrize("duration", [True, -1, "nan", float("inf"), {"bad": "value"}])
def test_invalid_duration_is_unknown_not_invalid_json(media_client, duration):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "input.mp4", "duration_seconds": duration}})
    (session / "input.mp4").write_bytes(b"video")
    response = client.get("/session/demo_case/demo-media")
    assert response.status_code == 200
    assert response.json()["duration_seconds"] is None


def test_external_metadata_symlink_is_not_read(media_client):
    client, root = media_client
    session = make_session(root)
    outside = root / "outside.json"
    outside.write_text(json.dumps({"demo_media": {"path": "input.mp4", "title": "secret"}}))
    (session / "metadata.json").symlink_to(outside)
    (session / "input.mp4").write_bytes(b"video")
    result = client.get("/session/demo_case/demo-media").json()
    assert result["available"] is False
    assert result["title"] != "secret"


@pytest.mark.parametrize("offset", [None, True, "0", float("nan"), float("inf"), float("-inf")])
def test_unknown_or_nonfinite_offset_never_claims_alignment(media_client, offset):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "input.mp4", "time_origin": "session_relative_seconds", "video_start_seconds": offset}})
    (session / "input.mp4").write_bytes(b"video")
    result = client.get("/session/demo_case/demo-media").json()
    assert result["available"] is True
    assert result["video_start_seconds"] is None
    assert result["time_alignment_known"] is False


@pytest.mark.parametrize("offset", [0, 12.5, -5])
def test_explicit_finite_offset_and_session_clock_are_known(media_client, offset):
    client, root = media_client
    session = make_session(root, {"demo_media": {"path": "input.mp4", "time_origin": "session_relative_seconds", "video_start_seconds": offset}})
    (session / "input.mp4").write_bytes(b"video")
    result = client.get("/session/demo_case/demo-media").json()
    assert result["video_start_seconds"] == offset
    assert result["time_alignment_known"] is True


def test_other_clock_cannot_claim_session_alignment(media_client):
    client, root = media_client
    make_session(root, {"demo_media": {"time_origin": "merged_source_seconds", "video_start_seconds": 120}})
    result = client.get("/session/demo_case/demo-media").json()
    assert result["time_alignment_known"] is False
    assert result["video_start_seconds"] is None


@pytest.mark.parametrize("milliseconds", [0, 98123])
def test_frame_state_reports_memory_coverage_independently_from_video(media_client, milliseconds):
    client, root = media_client
    session = make_session(root, {"demo_media": {"duration_seconds": 120, "memory_until_seconds": 1}})
    (session / "stream").mkdir()
    (session / "stream/frame_state.json").write_text(json.dumps({"latest_relative_ts_ms": milliseconds}))
    result = client.get("/session/demo_case/demo-media").json()
    assert result["available"] is False
    assert result["memory_until_seconds"] == milliseconds / 1000


@pytest.mark.parametrize("value", [True, -1, "1000", float("nan"), float("inf"), None])
def test_invalid_frame_coverage_is_unknown(media_client, value):
    client, root = media_client
    session = make_session(root)
    (session / "stream").mkdir()
    (session / "stream/frame_state.json").write_text(json.dumps({"latest_relative_ts_ms": value}))
    result = client.get("/session/demo_case/demo-media").json()
    assert result["memory_until_seconds"] is None


def test_explicit_coverage_fallback(media_client):
    client, root = media_client
    make_session(root, {"demo_media": {"memory_until_seconds": 98}})
    assert client.get("/session/demo_case/demo-media").json()["memory_until_seconds"] == 98


def test_external_frame_state_does_not_leak_coverage(media_client):
    client, root = media_client
    session = make_session(root)
    outside = root / "outside_state.json"
    outside.write_text(json.dumps({"latest_relative_ts_ms": 90000}))
    (session / "stream").mkdir()
    (session / "stream/frame_state.json").symlink_to(outside)
    assert client.get("/session/demo_case/demo-media").json()["memory_until_seconds"] is None
