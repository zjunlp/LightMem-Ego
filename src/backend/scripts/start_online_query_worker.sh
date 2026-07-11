#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${WORLDMM_QUERY_CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${WORLDMM_QUERY_CUDA_VISIBLE_DEVICES}"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="3"
fi

DEFAULT_VLM2VEC_MODEL_PATH="$PROJECT_ROOT/models/VLM2Vec-V2.0"
if [ -z "${WORLDMM_VLM2VEC_MODEL_PATH:-}" ] || [ "${WORLDMM_VLM2VEC_MODEL_PATH:-}" = "/path/to/vlm2vec-v2" ]; then
  export WORLDMM_VLM2VEC_MODEL_PATH="${WORLDMM_VIS_EMBED_MODEL:-$DEFAULT_VLM2VEC_MODEL_PATH}"
fi
export WORLDMM_PRELOAD_VLM2VEC="${WORLDMM_PRELOAD_VLM2VEC:-1}"
export WORLDMM_QUERY_CACHE_MAX_SESSIONS="${WORLDMM_QUERY_CACHE_MAX_SESSIONS:-1}"
export WORLDMM_PRELOAD_RECENT_MEMORY_READY="${WORLDMM_PRELOAD_RECENT_MEMORY_READY:-1}"
export WORLDMM_QUERY_STRICT_LOAD_ONLY="${WORLDMM_QUERY_STRICT_LOAD_ONLY:-1}"
export WORLDMM_QUERY_SKIP_REINDEX="${WORLDMM_QUERY_SKIP_REINDEX:-1}"
export WORLDMM_QUERY_USE_CACHED_HIPPORAG="${WORLDMM_QUERY_USE_CACHED_HIPPORAG:-1}"
export WORLDMM_ALLOW_HF_DOWNLOAD="${WORLDMM_ALLOW_HF_DOWNLOAD:-0}"
if [ "${WORLDMM_ALLOW_HF_DOWNLOAD}" != "1" ]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi

echo "[start_online_query_worker] PROJECT_ROOT=$PROJECT_ROOT"
echo "[start_online_query_worker] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[start_online_query_worker] WORLDMM_QUERY_CACHE_MAX_SESSIONS=${WORLDMM_QUERY_CACHE_MAX_SESSIONS}"
echo "[start_online_query_worker] WORLDMM_QUERY_CACHE_TTL_SECONDS=${WORLDMM_QUERY_CACHE_TTL_SECONDS:-3600}"
echo "[start_online_query_worker] WORLDMM_VISUAL_BACKEND=${WORLDMM_VISUAL_BACKEND:-vlm2vec}"
echo "[start_online_query_worker] WORLDMM_VLM2VEC_MODEL_PATH=${WORLDMM_VLM2VEC_MODEL_PATH}"
echo "[start_online_query_worker] WORLDMM_ALLOW_HF_DOWNLOAD=${WORLDMM_ALLOW_HF_DOWNLOAD}"
echo "[start_online_query_worker] WORLDMM_PRELOAD_VLM2VEC=${WORLDMM_PRELOAD_VLM2VEC}"
echo "[start_online_query_worker] WORLDMM_QUERY_STRICT_LOAD_ONLY=${WORLDMM_QUERY_STRICT_LOAD_ONLY}"
echo "[start_online_query_worker] WORLDMM_QUERY_SKIP_REINDEX=${WORLDMM_QUERY_SKIP_REINDEX}"
echo "[start_online_query_worker] WORLDMM_QUERY_USE_CACHED_HIPPORAG=${WORLDMM_QUERY_USE_CACHED_HIPPORAG}"
echo "[start_online_query_worker] WORLDMM_TEXT_EMBED_BACKEND=${WORLDMM_TEXT_EMBED_BACKEND:-local}"
echo "[start_online_query_worker] WORLDMM_TEXT_EMBED_URL=${WORLDMM_TEXT_EMBED_URL:-http://${WORLDMM_TEXT_EMBED_HOST:-127.0.0.1}:${WORLDMM_TEXT_EMBED_PORT:-18096}}"
echo "[start_online_query_worker] WORLDMM_PRELOAD_QUERY_SESSIONS=${WORLDMM_PRELOAD_QUERY_SESSIONS:-}"
echo "[start_online_query_worker] WORLDMM_PRELOAD_RECENT_MEMORY_READY=${WORLDMM_PRELOAD_RECENT_MEMORY_READY}"
echo "[start_online_query_worker] sessions_root=online_sessions"
echo "[start_online_query_worker] task_queue_root=online_tasks/query"

exec python online_query_worker.py "$@"
