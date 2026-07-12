#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/api runtime/api

PID_PATH="runtime/api/server.pid"

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[start_api] API server already running pid=${existing_pid}; reusing it"
    exit 0
  fi
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env_ffmpeg.sh"

LOG_PATH="logs/api/server.log"

nohup setsid python -m uvicorn api_server:app --host "${EM2MEM_API_HOST:-127.0.0.1}" --port "${EM2MEM_API_PORT:-8000}" > "$LOG_PATH" 2>&1 &
echo "$!" > "$PID_PATH"
echo "[start_api] API server pid=$(cat "$PID_PATH") log=${LOG_PATH}"
