#!/usr/bin/env bash
set -euo pipefail

URL="${1:-rtmp://localhost/live/test_stream}"
OUTPUT="${2:-/tmp/em2mem_rtmp_frame.jpg}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not installed." >&2
  exit 1
fi

ffmpeg -y -i "$URL" -frames:v 1 -q:v 2 "$OUTPUT"
echo "Saved frame: $OUTPUT"
