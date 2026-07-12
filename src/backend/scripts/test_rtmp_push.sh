#!/usr/bin/env bash
set -euo pipefail

URL="${1:-rtmp://localhost/live/test_stream}"
INPUT="${2:-}"
DURATION="${EM2MEM_RTMP_TEST_DURATION:-30}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not installed." >&2
  exit 1
fi

if [[ -n "$INPUT" ]]; then
  ffmpeg -re -stream_loop -1 -i "$INPUT" \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -c:a aac -ar 44100 -b:a 128k \
    -f flv "$URL"
else
  ffmpeg -re \
    -f lavfi -i "testsrc=size=1280x720:rate=25" \
    -f lavfi -i "sine=frequency=1000:sample_rate=44100" \
    -t "$DURATION" \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -pix_fmt yuv420p \
    -c:a aac -ar 44100 -b:a 128k \
    -f flv "$URL"
fi
