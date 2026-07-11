#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-http://127.0.0.1:8000}"
SESSION_ID="${SESSION_ID:-}"
AUDIO_PATH="${AUDIO_PATH:-}"
AUDIO_INDEX="${AUDIO_INDEX:-0}"
CLIENT_TS_MS="${CLIENT_TS_MS:-1710000000000}"
RELATIVE_TS_MS="${RELATIVE_TS_MS:-0}"
DURATION_MS="${DURATION_MS:-1000}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
CHANNELS="${CHANNELS:-1}"
SOURCE="${SOURCE:-recorder_manager}"
FORMAT="${FORMAT:-}"
MIME="${MIME:-}"

if [[ -z "$SESSION_ID" ]]; then
  echo "SESSION_ID is required" >&2
  exit 2
fi

if [[ -z "$AUDIO_PATH" || ! -f "$AUDIO_PATH" ]]; then
  echo "AUDIO_PATH must point to an existing audio file" >&2
  exit 2
fi

if [[ -z "$FORMAT" ]]; then
  FORMAT="${AUDIO_PATH##*.}"
fi

if [[ -n "$MIME" ]]; then
  AUDIO_FIELD="audio=@${AUDIO_PATH};type=${MIME}"
else
  AUDIO_FIELD="audio=@${AUDIO_PATH}"
fi

curl -X POST "${SERVER}/stream/${SESSION_ID}/audio_chunk" \
  -F "${AUDIO_FIELD}" \
  -F "audio_index=${AUDIO_INDEX}" \
  -F "client_ts_ms=${CLIENT_TS_MS}" \
  -F "relative_ts_ms=${RELATIVE_TS_MS}" \
  -F "duration_ms=${DURATION_MS}" \
  -F "sample_rate=${SAMPLE_RATE}" \
  -F "channels=${CHANNELS}" \
  -F "format=${FORMAT}" \
  -F "source=${SOURCE}"
