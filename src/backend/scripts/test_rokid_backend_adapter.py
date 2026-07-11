#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online_preprocess.io_utils import read_json  # noqa: E402


def _post_json(server: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{server.rstrip('/')}{path}", json=payload, timeout=30)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: {response.status_code} {json.dumps(body, ensure_ascii=False)}")
    return body


def _post_multipart(server: str, path: str, *, file_field: str, file_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    with file_path.open("rb") as handle:
        files = {file_field: (file_path.name, handle)}
        response = requests.post(f"{server.rstrip('/')}{path}", files=files, data={k: str(v) for k, v in data.items() if v is not None}, timeout=60)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: {response.status_code} {json.dumps(body, ensure_ascii=False)}")
    return body


def _get_json(server: str, path: str) -> dict[str, Any]:
    response = requests.get(f"{server.rstrip('/')}{path}", timeout=30)
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"GET {path} failed: {response.status_code} {json.dumps(body, ensure_ascii=False)}")
    return body


def _make_test_jpg(path: Path) -> Path:
    try:
        from PIL import Image, ImageDraw

        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (640, 360), (24, 28, 34))
        draw = ImageDraw.Draw(image)
        draw.rectangle((28, 28, 612, 332), outline=(38, 180, 130), width=6)
        draw.text((48, 48), "Rokid backend adapter test", fill=(240, 244, 248))
        image.save(path, format="JPEG", quality=75)
        return path
    except Exception as exc:
        raise RuntimeError("Pillow is required to auto-generate --frame; pass --frame /path/to/test.jpg") from exc


def _make_test_wav(path: Path, *, sample_rate: int = 16000, duration_ms: int = 1000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(sample_rate * duration_ms / 1000)
    amplitude = 900
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = bytearray()
        for idx in range(frames):
            value = int(amplitude * math.sin(2 * math.pi * 440 * idx / sample_rate))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(samples))
    return path


def _session_file(session_id: str, rel_path: str | None) -> bool:
    return bool(rel_path) and (PROJECT_ROOT / "online_sessions" / session_id / str(rel_path)).exists()


def run(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = PROJECT_ROOT / ".cache" / "rokid_adapter_test"
    frame_path = args.frame or _make_test_jpg(temp_dir / "rokid_test_frame.jpg")
    audio_path = args.audio or _make_test_wav(temp_dir / "rokid_test_audio.wav", duration_ms=args.audio_duration_ms)

    start_payload = {
        "input_mode": "rokid_frame_audio",
        "metadata": {
            "source": "rokid_glass",
            "device_type": "rokid",
            "transport": "phone_sdk",
            "sdk": "rokid",
            "sdk_version": args.sdk_version,
            "device_id": args.device_id,
            "glass_sn": args.glass_sn,
            "phone_model": args.phone_model,
        },
    }
    start_payload["metadata"] = {k: v for k, v in start_payload["metadata"].items() if v is not None}
    start = _post_json(args.server, "/stream/start", start_payload)
    session_id = str(start.get("session_id") or "")
    if not session_id:
        raise RuntimeError(f"/stream/start did not return session_id: {start}")

    frame_result = _post_multipart(
        args.server,
        f"/stream/{session_id}/frame",
        file_field="frame",
        file_path=frame_path,
        data={
            "frame_index": args.frame_index,
            "relative_ts_ms": args.frame_relative_ts_ms,
            "client_ts_ms": args.client_ts_ms,
            "format": args.frame_format or frame_path.suffix.lstrip("."),
            "source": "rokid_sdk_video",
        },
    )
    audio_result = _post_multipart(
        args.server,
        f"/stream/{session_id}/audio_chunk",
        file_field="audio",
        file_path=audio_path,
        data={
            "audio_index": args.audio_index,
            "relative_ts_ms": args.audio_relative_ts_ms,
            "client_ts_ms": args.client_ts_ms,
            "duration_ms": args.audio_duration_ms,
            "format": args.audio_format or audio_path.suffix.lstrip("."),
            "sample_rate": args.sample_rate,
            "channels": args.channels,
            "source": "rokid_sdk_audio",
        },
    )
    status = _get_json(args.server, f"/stream/{session_id}/status")
    current_frames_path = PROJECT_ROOT / "online_sessions" / session_id / "current" / "current_frames.jsonl"
    frame_state = read_json(PROJECT_ROOT / "online_sessions" / session_id / "stream" / "frame_state.json", default={})
    audio_state = read_json(PROJECT_ROOT / "online_sessions" / session_id / "stream" / "audio_state.json", default={})
    if not isinstance(frame_state, dict):
        frame_state = {}
    if not isinstance(audio_state, dict):
        audio_state = {}
    checks = {
        "session_started": start.get("input_mode") == "rokid_frame_audio" and bool(start.get("frame_upload_url")) and bool(start.get("audio_upload_url")),
        "rokid_start_block": bool((start.get("rokid") or {}).get("enabled")),
        "frame_mcur_ready": bool(frame_result.get("mcur_ready") or status.get("memory", {}).get("current_ready")),
        "audio_accepted": str(audio_result.get("status")) == "audio_chunk_received",
        "status_has_rokid": bool((status.get("rokid") or {}).get("enabled")),
        "frame_state_updated": int(frame_state.get("received_count", 0) or 0) >= 1,
        "audio_state_updated": int(audio_state.get("received_count", 0) or 0) >= 1,
        "current_frames_jsonl_exists": current_frames_path.exists(),
        "frame_saved_exists": _session_file(session_id, frame_result.get("saved_path")),
        "audio_saved_exists": _session_file(session_id, audio_result.get("saved_path")),
    }
    return {
        "ok": all(checks.values()),
        "server": args.server,
        "session_id": session_id,
        "start": start,
        "frame_result": frame_result,
        "audio_result": audio_result,
        "status_rokid": status.get("rokid"),
        "frame_state": frame_state,
        "audio_state": audio_state,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP smoke test for the Rokid backend adapter.")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--audio-index", type=int, default=0)
    parser.add_argument("--frame-relative-ts-ms", type=int, default=0)
    parser.add_argument("--audio-relative-ts-ms", type=int, default=0)
    parser.add_argument("--audio-duration-ms", type=int, default=1000)
    parser.add_argument("--client-ts-ms", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--frame-format")
    parser.add_argument("--audio-format")
    parser.add_argument("--sdk-version")
    parser.add_argument("--device-id")
    parser.add_argument("--glass-sn")
    parser.add_argument("--phone-model")
    args = parser.parse_args()

    report = run(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"session_id: {report['session_id']}")
        for key, value in report["checks"].items():
            print(f"{key}: {value}")
        print(f"ok: {report['ok']}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
