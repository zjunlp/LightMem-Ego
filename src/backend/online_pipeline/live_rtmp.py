from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Any

from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic


SAFE_STREAM_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except Exception:
        return default


def live_rtmp_config() -> dict[str, Any]:
    scheme = _env_str("EM2MEM_LIVE_RTMP_SCHEME", "rtmp").lower()
    domain = _env_str("EM2MEM_LIVE_RTMP_DOMAIN", "localhost")
    public_port = _env_int("EM2MEM_LIVE_RTMP_PUBLIC_PORT", 1935)
    app = _env_str("EM2MEM_LIVE_RTMP_APP", "live").strip("/")
    internal_pull_base = _env_str("EM2MEM_LIVE_RTMP_INTERNAL_PULL_BASE", f"{scheme}://{domain}/{app}").rstrip("/")
    return {
        "enabled": _env_bool("EM2MEM_LIVE_RTMP_ENABLED", False),
        "scheme": scheme,
        "domain": domain,
        "public_port": public_port,
        "app": app or "live",
        "internal_pull_base": internal_pull_base,
    }


def webrtc_whip_config() -> dict[str, Any]:
    scheme = _env_str("EM2MEM_WEBRTC_WHIP_SCHEME", "http").lower()
    domain = _env_str("EM2MEM_WEBRTC_WHIP_DOMAIN", "localhost")
    public_port = _env_int("EM2MEM_WEBRTC_WHIP_PUBLIC_PORT", 443)
    path = _env_str("EM2MEM_WEBRTC_WHIP_PATH", "/rtc/v1/whip/")
    app = _env_str("EM2MEM_WEBRTC_WHIP_APP", "live").strip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    preferred_video_codec = _env_str("EM2MEM_WEBRTC_PREFERRED_VIDEO_CODEC", "h264")
    preferred_audio_codec = _env_str("EM2MEM_WEBRTC_PREFERRED_AUDIO_CODEC", "opus")
    return {
        "enabled": _env_bool("EM2MEM_WEBRTC_WHIP_ENABLED", False),
        "scheme": scheme,
        "domain": domain,
        "public_port": public_port,
        "path": path,
        "app": app or "live",
        "preferred_video_codec": preferred_video_codec,
        "preferred_audio_codec": preferred_audio_codec,
    }

def webrtc_play_config() -> dict[str, Any]:
    scheme = _env_str("EM2MEM_WEBRTC_PLAY_SCHEME", "https").lower()
    domain = _env_str("EM2MEM_WEBRTC_PLAY_DOMAIN", "localhost")
    public_port = _env_int("EM2MEM_WEBRTC_PLAY_PUBLIC_PORT", 443)
    path = _env_str("EM2MEM_WEBRTC_PLAY_PATH", "/rtc/v1/play/")
    app = _env_str("EM2MEM_WEBRTC_PLAY_APP", _env_str("EM2MEM_LIVE_RTMP_APP", "live")).strip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    return {
        "enabled": _env_bool("EM2MEM_WEBRTC_PLAY_ENABLED", False),
        "scheme": scheme,
        "domain": domain,
        "public_port": public_port,
        "path": path,
        "app": app or "live",
    }


def _public_rtmp_base(config: dict[str, Any]) -> str:
    scheme = str(config.get("scheme") or "rtmp")
    domain = str(config.get("domain") or "localhost")
    app = str(config.get("app") or "live").strip("/")
    public_port = int(config.get("public_port") or 1935)
    default_port = (scheme == "rtmp" and public_port == 1935) or (scheme == "rtmps" and public_port == 443)
    port_part = "" if default_port else f":{public_port}"
    return f"{scheme}://{domain}{port_part}/{app}"


def _public_http_base(config: dict[str, Any]) -> str:
    scheme = str(config.get("scheme") or "http")
    domain = str(config.get("domain") or "localhost")
    public_port = int(config.get("public_port") or 443)
    default_port = (scheme == "https" and public_port == 443) or (scheme == "http" and public_port == 80)
    port_part = "" if default_port else f":{public_port}"
    return f"{scheme}://{domain}{port_part}"


def _build_whip_url(config: dict[str, Any], stream_name: str) -> str:
    base = _public_http_base(config)
    path = str(config.get("path") or "/rtc/v1/whip/")
    app = str(config.get("app") or "live")
    separator = "&" if "?" in path else "?"
    return f"{base}{path}{separator}app={app}&stream={stream_name}"

def _build_webrtc_play_url(config: dict[str, Any], stream_name: str) -> str:
    if not bool(config.get("enabled")):
        return ""
    base = _public_http_base(config)
    path = str(config.get("path") or "/rtc/v1/play/")
    app = str(config.get("app") or "live")
    separator = "&" if "?" in path else "?"
    return f"{base}{path}{separator}app={app}&stream={stream_name}"


def make_stream_name(session_id: str) -> str:
    safe_session_id = SAFE_STREAM_NAME_RE.sub("_", str(session_id or "").strip())
    safe_session_id = safe_session_id.strip("_") or "session"
    return f"{safe_session_id}_{secrets.token_hex(6)}"


def build_live_source(
    *,
    session_id: str,
    stream_id: str,
    stream_name: str,
    input_mode: str = "live_pusher_rtmp",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(config or live_rtmp_config())
    public_base = _public_rtmp_base(cfg)
    stream_name = SAFE_STREAM_NAME_RE.sub("_", stream_name)
    now = utc_now_iso()
    push_url = f"{public_base}/{stream_name}"
    pull_url_internal = f"{str(cfg.get('internal_pull_base') or public_base).rstrip('/')}/{stream_name}"
    play_url = _build_webrtc_play_url(webrtc_play_config(), stream_name)
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "input_mode": input_mode,
        "protocol": str(cfg.get("scheme") or "rtmp"),
        "app": str(cfg.get("app") or "live"),
        "stream_name": stream_name,
        "push_url_public": push_url,
        "pull_url_public": push_url,
        "pull_url_internal": pull_url_internal,
        "webrtc_play_url_public": play_url,
        "webrtc_play_url_available": bool(play_url),
        "srs_host": str(cfg.get("domain") or "localhost"),
        "srs_port": int(cfg.get("public_port") or 1935),
        "ingest_status": "not_started",
        "last_frame_at": None,
        "last_audio_at": None,
        "srs_checked": False,
        "srs_online": None,
        "created_at": now,
        "updated_at": now,
    }


def build_webrtc_whip_live_source(
    *,
    session_id: str,
    stream_id: str,
    stream_name: str,
    whip_config: dict[str, Any] | None = None,
    rtmp_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    whip_cfg = dict(whip_config or webrtc_whip_config())
    rtmp_cfg = dict(rtmp_config or live_rtmp_config())
    stream_name = SAFE_STREAM_NAME_RE.sub("_", stream_name)
    now = utc_now_iso()
    whip_url = _build_whip_url(whip_cfg, stream_name)
    rtmp_public_base = _public_rtmp_base(rtmp_cfg)
    pull_url_public = f"{rtmp_public_base}/{stream_name}"
    pull_url_internal = f"{str(rtmp_cfg.get('internal_pull_base') or rtmp_public_base).rstrip('/')}/{stream_name}"
    play_url = _build_webrtc_play_url(webrtc_play_config(), stream_name)
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "input_mode": "web_webrtc_whip",
        "protocol": "webrtc",
        "signaling_protocol": "whip",
        "app": str(whip_cfg.get("app") or "live"),
        "stream_name": stream_name,
        "whip_url_public": whip_url,
        "push_url_public": whip_url,
        "push_protocol": "webrtc_whip",
        "pull_url_public": pull_url_public,
        "pull_url_internal": pull_url_internal,
        "webrtc_play_url_public": play_url,
        "webrtc_play_url_available": bool(play_url),
        "rtmp_fallback_push_url": pull_url_public,
        "preferred_video_codec": str(whip_cfg.get("preferred_video_codec") or "h264"),
        "preferred_audio_codec": str(whip_cfg.get("preferred_audio_codec") or "opus"),
        "srs_host": str(whip_cfg.get("domain") or "localhost"),
        "srs_port": int(whip_cfg.get("public_port") or 443),
        "rtmp_pull_host": str(rtmp_cfg.get("domain") or "localhost"),
        "rtmp_pull_port": int(rtmp_cfg.get("public_port") or 1935),
        "ingest_status": "not_started",
        "last_frame_at": None,
        "last_audio_at": None,
        "srs_checked": False,
        "srs_online": None,
        "published": None,
        "last_publish_at": None,
        "created_at": now,
        "updated_at": now,
    }


def live_source_path(session_dir: Path) -> Path:
    return Path(session_dir) / "stream" / "live_source.json"


def save_live_source(session_dir: Path, live_source: dict[str, Any]) -> None:
    path = live_source_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, dict(live_source))


def load_live_source(session_dir: Path) -> dict[str, Any]:
    payload = read_json(live_source_path(session_dir), default={})
    return payload if isinstance(payload, dict) else {}


def stream_state_live_block(live_source: dict[str, Any]) -> dict[str, Any]:
    if not live_source:
        return {
            "enabled": False,
            "input_mode": "chunk",
            "push_url_available": False,
            "ingest_status": None,
            "last_frame_at": None,
            "last_audio_at": None,
            "srs_checked": False,
            "srs_online": None,
        }
    return {
        "enabled": True,
        "input_mode": live_source.get("input_mode", "live_pusher_rtmp"),
        "protocol": live_source.get("protocol", "rtmp"),
        "signaling_protocol": live_source.get("signaling_protocol"),
        "app": live_source.get("app", "live"),
        "stream_name": live_source.get("stream_name"),
        "push_url_public": live_source.get("push_url_public"),
        "whip_url_public": live_source.get("whip_url_public"),
        "webrtc_play_url_public": live_source.get("webrtc_play_url_public"),
        "pull_url_public": live_source.get("pull_url_public"),
        "pull_url_internal": live_source.get("pull_url_internal"),
        "push_url_available": bool(live_source.get("push_url_public")),
        "whip_url_available": bool(live_source.get("whip_url_public")),
        "webrtc_play_url_available": bool(live_source.get("webrtc_play_url_public")),
        "push_protocol": live_source.get("push_protocol"),
        "ingest_status": live_source.get("ingest_status", "not_started"),
        "last_frame_at": live_source.get("last_frame_at"),
        "last_audio_at": live_source.get("last_audio_at"),
        "srs_checked": bool(live_source.get("srs_checked", False)),
        "srs_online": live_source.get("srs_online"),
        "updated_at": live_source.get("updated_at"),
    }


def public_live_status_block(session_dir: Path, stream_state: dict[str, Any] | None = None) -> dict[str, Any]:
    live_source = load_live_source(session_dir)
    try:
        from online_pipeline.live_ingest import load_live_ingest_state

        live_ingest_state = load_live_ingest_state(session_dir)
    except Exception:
        live_ingest_state = {}
    state_live = stream_state.get("live") if isinstance(stream_state, dict) and isinstance(stream_state.get("live"), dict) else {}
    source = live_source or state_live
    if not source:
        cfg = live_rtmp_config()
        return {
            "enabled": bool(cfg.get("enabled")),
            "input_mode": (stream_state or {}).get("input_mode", "chunk") if isinstance(stream_state, dict) else "chunk",
            "protocol": str(cfg.get("scheme") or "rtmp"),
            "signaling_protocol": None,
            "stream_name": None,
            "push_url_available": False,
            "whip_url_available": False,
            "webrtc_play_url_available": False,
            "ingest_status": None,
            "last_frame_at": None,
            "last_audio_at": None,
            "srs_checked": False,
            "srs_online": None,
        }
    payload = {
        "enabled": True,
        "input_mode": source.get("input_mode", "live_pusher_rtmp"),
        "protocol": source.get("protocol", "rtmp"),
        "signaling_protocol": source.get("signaling_protocol"),
        "stream_name": source.get("stream_name"),
        "push_url_public": source.get("push_url_public"),
        "whip_url_public": source.get("whip_url_public"),
        "webrtc_play_url_public": source.get("webrtc_play_url_public"),
        "push_url_available": bool(source.get("push_url_public") or source.get("push_url_available")),
        "whip_url_available": bool(source.get("whip_url_public")),
        "webrtc_play_url_available": bool(source.get("webrtc_play_url_public")),
        "push_protocol": source.get("push_protocol"),
        "pull_url_public": source.get("pull_url_public"),
        "pull_url_internal": source.get("pull_url_internal"),
        "ingest_status": live_ingest_state.get("status") or source.get("ingest_status", "not_started"),
        "last_frame_at": live_ingest_state.get("last_frame_at") or source.get("last_frame_at"),
        "last_audio_at": live_ingest_state.get("last_audio_at") or source.get("last_audio_at"),
        "srs_checked": bool(source.get("srs_checked", False)),
        "srs_online": source.get("srs_online"),
    }
    if live_ingest_state:
        payload.update(
            {
                "frames_ingested": live_ingest_state.get("frames_ingested", 0),
                "audio_chunks_ingested": live_ingest_state.get("audio_chunks_ingested", 0),
                "frame_fps": live_ingest_state.get("frame_fps"),
                "audio_segment_ms": live_ingest_state.get("audio_segment_ms"),
                "source": live_ingest_state.get("source"),
                "last_error": live_ingest_state.get("last_error"),
                "latest_frame_relative_ts_ms": live_ingest_state.get(
                    "latest_frame_relative_ts_ms",
                    live_ingest_state.get("last_frame_relative_ts_ms"),
                ),
                "latest_audio_relative_ts_ms": live_ingest_state.get(
                    "latest_audio_relative_ts_ms",
                    live_ingest_state.get("last_audio_relative_ts_ms"),
                ),
                "latest_audio_end_relative_ts_ms": live_ingest_state.get("latest_audio_end_relative_ts_ms"),
                "av_skew_ms": live_ingest_state.get("av_skew_ms"),
                "timestamp_mode": live_ingest_state.get("timestamp_mode"),
                "sync_status": live_ingest_state.get("sync_status"),
                "timeline_version": live_ingest_state.get("timeline_version"),
            }
        )
    return payload


def public_webrtc_status_block(session_dir: Path, stream_state: dict[str, Any] | None = None) -> dict[str, Any]:
    live_source = load_live_source(session_dir)
    state_live = stream_state.get("live") if isinstance(stream_state, dict) and isinstance(stream_state.get("live"), dict) else {}
    source = live_source or state_live
    if not source or source.get("input_mode") != "web_webrtc_whip":
        return {
            "enabled": False,
            "whip_url_available": False,
            "webrtc_play_url_available": False,
            "app": None,
            "stream_name": None,
            "published": None,
            "last_publish_at": None,
        }
    return {
        "enabled": True,
        "protocol": "webrtc",
        "signaling_protocol": "whip",
        "whip_url": source.get("whip_url_public"),
        "whip_url_public": source.get("whip_url_public"),
        "webrtc_play_url_public": source.get("webrtc_play_url_public"),
        "whip_url_available": bool(source.get("whip_url_public")),
        "webrtc_play_url_available": bool(source.get("webrtc_play_url_public")),
        "app": source.get("app", "live"),
        "stream_name": source.get("stream_name"),
        "preferred_video_codec": source.get("preferred_video_codec"),
        "preferred_audio_codec": source.get("preferred_audio_codec"),
        "published": source.get("published"),
        "last_publish_at": source.get("last_publish_at"),
    }
