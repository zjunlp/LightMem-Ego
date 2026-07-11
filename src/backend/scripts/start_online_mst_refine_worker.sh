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

export WORLDMM_MST_REFINE_BACKEND="${WORLDMM_MST_REFINE_BACKEND:-openai}"
export WORLDMM_REFINE_MAX_CONCURRENCY="${WORLDMM_REFINE_MAX_CONCURRENCY:-4}"
export WORLDMM_WORKER_INSTANCE_NAME="${WORLDMM_WORKER_INSTANCE_NAME:-refine}"

echo "[start_online_mst_refine_worker] WORLDMM_WORKER_INSTANCE_NAME=${WORLDMM_WORKER_INSTANCE_NAME}"
echo "[start_online_mst_refine_worker] WORLDMM_MST_REFINE_BACKEND=${WORLDMM_MST_REFINE_BACKEND}"
echo "[start_online_mst_refine_worker] WORLDMM_MST_REFINE_MODEL=${WORLDMM_MST_REFINE_MODEL:-${WORLDMM_VLM_MODEL:-${OPENAI_MODEL:-}}}"
echo "[start_online_mst_refine_worker] WORLDMM_REFINE_MAX_CONCURRENCY=${WORLDMM_REFINE_MAX_CONCURRENCY}"

exec python online_mst_refine_worker.py "$@"
