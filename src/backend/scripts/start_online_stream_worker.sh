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

export EM2MEM_PIPELINE_MODE="${EM2MEM_PIPELINE_MODE:-mst}"

echo "[start_online_stream_worker] EM2MEM_PIPELINE_MODE=${EM2MEM_PIPELINE_MODE}"
echo "[start_online_stream_worker] task_queue_root=online_tasks/stream_chunk"

exec python online_stream_worker.py "$@"
