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

export EM2MEM_MST_REFINE_BACKEND="${EM2MEM_MST_REFINE_BACKEND:-openai}"
export EM2MEM_REFINE_MAX_CONCURRENCY="${EM2MEM_REFINE_MAX_CONCURRENCY:-4}"
export EM2MEM_WORKER_INSTANCE_NAME="${EM2MEM_WORKER_INSTANCE_NAME:-refine}"

echo "[start_online_mst_refine_worker] EM2MEM_WORKER_INSTANCE_NAME=${EM2MEM_WORKER_INSTANCE_NAME}"
echo "[start_online_mst_refine_worker] EM2MEM_MST_REFINE_BACKEND=${EM2MEM_MST_REFINE_BACKEND}"
echo "[start_online_mst_refine_worker] EM2MEM_MST_REFINE_MODEL=${EM2MEM_MST_REFINE_MODEL:-${EM2MEM_VLM_MODEL:-${OPENAI_MODEL:-}}}"
echo "[start_online_mst_refine_worker] EM2MEM_REFINE_MAX_CONCURRENCY=${EM2MEM_REFINE_MAX_CONCURRENCY}"

exec python online_mst_refine_worker.py "$@"
