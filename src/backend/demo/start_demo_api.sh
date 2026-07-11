#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

API_HOST="${WORLDMM_DEMO_API_HOST:-${WORLDMM_API_HOST:-127.0.0.1}}"
MAIN_API_PORT="${WORLDMM_API_PORT:-8000}"
if [[ -n "${WORLDMM_DEMO_API_PORT:-}" ]]; then
  API_PORT="$WORLDMM_DEMO_API_PORT"
elif [[ "$MAIN_API_PORT" =~ ^[0-9]+$ ]]; then
  API_PORT="$((MAIN_API_PORT + 1))"
else
  API_PORT="8001"
fi

if [[ "${WORLDMM_ALLOW_DEMO_SAME_PORT:-0}" != "1" && "$API_PORT" == "$MAIN_API_PORT" ]]; then
  echo "Refusing to start demo API on the same port as WORLDMM_API_PORT=$MAIN_API_PORT." >&2
  echo "Set WORLDMM_DEMO_API_PORT to another port, for example 8001." >&2
  exit 1
fi

echo "Starting WorldMM demo API on ${API_HOST}:${API_PORT} (main API port: ${MAIN_API_PORT})"
python -m uvicorn demo.api:app --host "$API_HOST" --port "$API_PORT"
