from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib import request


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def _probe_duration(path: Path) -> float:
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as resp:  # noqa: S310 - local dev helper
        return json.loads(resp.read().decode("utf-8"))


def _post_multipart(url: str, fields: dict[str, object], file_path: Path) -> dict:
    boundary = "----em2mem" + uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")
    mime = mimetypes.guess_type(file_path.name)[0] or "video/mp4"
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))},
    )
    with request.urlopen(req, timeout=180) as resp:  # noqa: S310 - local dev helper
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local MP4 as legal upload chunks through the stream API. "
            "The server normalizes arbitrary upload chunk durations into fixed processing chunks."
        )
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--chunk-seconds", type=float, default=7.0, help="Duration of each uploaded client chunk.")
    parser.add_argument(
        "--processing-chunk-seconds",
        type=float,
        default=5.0,
        help="Server-side processing chunk duration passed to /stream/start.",
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_video = Path(args.input_video).resolve()
    if not input_video.exists():
        raise FileNotFoundError(input_video)
    server = args.server.rstrip("/")
    start_resp = _post_json(
        f"{server}/stream/start",
        {
            "chunk_duration": args.processing_chunk_seconds,
            "metadata": {
                "source": "upload_chunks_test",
                "upload_chunk_seconds": args.chunk_seconds,
            },
        },
    )
    session_id = start_resp["session_id"]
    duration = _probe_duration(input_video)
    print(json.dumps({"stream_start": start_resp}, ensure_ascii=False, indent=2))

    with tempfile.TemporaryDirectory(prefix="em2mem_stream_chunks_") as tmp:
        tmp_dir = Path(tmp)
        start = 0.0
        idx = 0
        while start < duration - 1e-3:
            if args.max_chunks is not None and idx >= args.max_chunks:
                break
            end = min(duration, start + args.chunk_seconds)
            chunk_path = tmp_dir / f"chunk_{idx:06d}.mp4"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    str(input_video),
                    "-t",
                    f"{end - start:.3f}",
                    "-c",
                    "copy",
                    str(chunk_path),
                ]
            )
            resp = _post_multipart(
                f"{server}/stream/{session_id}/chunk",
                {
                    "chunk_index": idx,
                    "is_last": "false",
                },
                chunk_path,
            )
            if args.verbose:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
            start = end
            idx += 1

    end_resp = _post_json(f"{server}/stream/{session_id}/end", {"final_chunk_index": idx - 1, "close_open_event": True})
    print(json.dumps({"stream_end": end_resp, "session_id": session_id}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
