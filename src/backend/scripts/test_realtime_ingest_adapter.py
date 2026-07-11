#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from online_pipeline.realtime_ingest import ingest_audio_chunk, ingest_frame  # noqa: E402


def _strip_internal(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("_http_status_code", None)
    return payload


def _print_json(title: str, payload: dict) -> None:
    print(f"==== {title} ====")
    print(json.dumps(_strip_internal(payload), ensure_ascii=False, indent=2))


def _exists(session_id: str, rel_path: str | None) -> bool:
    if not rel_path:
        return False
    return (PROJECT_ROOT / "online_sessions" / session_id / rel_path).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly test realtime ingest adapter without HTTP/FastAPI.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--audio-index", type=int, default=0)
    parser.add_argument("--frame-relative-ts-ms", type=int, default=0)
    parser.add_argument("--audio-relative-ts-ms", type=int, default=0)
    parser.add_argument("--audio-duration-ms", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--audio-mime")
    parser.add_argument("--disable-asr-enqueue", action="store_true")
    args = parser.parse_args()

    if not args.frame and not args.audio:
        raise SystemExit("Provide --frame and/or --audio.")

    if args.frame:
        frame_path = args.frame.resolve()
        result = ingest_frame(
            PROJECT_ROOT,
            args.session_id,
            frame_path.read_bytes(),
            frame_index=args.frame_index,
            relative_ts_ms=args.frame_relative_ts_ms,
            format=frame_path.suffix.lstrip("."),
            source="direct_adapter_test_frame",
            filename_hint=frame_path.name,
        )
        _print_json("frame_result", result)
        print("frame_saved_exists", _exists(args.session_id, result.get("saved_path")))
        print("current_frame_exists", _exists(args.session_id, result.get("current_frame_path")))

    if args.audio:
        audio_path = args.audio.resolve()
        result = ingest_audio_chunk(
            PROJECT_ROOT,
            args.session_id,
            audio_path.read_bytes(),
            audio_index=args.audio_index,
            relative_ts_ms=args.audio_relative_ts_ms,
            duration_ms=args.audio_duration_ms,
            sample_rate=args.sample_rate,
            channels=args.channels,
            format=audio_path.suffix.lstrip("."),
            content_type=args.audio_mime,
            source="direct_adapter_test_audio",
            filename_hint=audio_path.name,
            enqueue_asr=not args.disable_asr_enqueue,
        )
        _print_json("audio_result", result)
        print("audio_saved_exists", _exists(args.session_id, result.get("saved_path")))


if __name__ == "__main__":
    main()
