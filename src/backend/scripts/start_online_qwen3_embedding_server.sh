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

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${WORLDMM_TEXT_EMBED_CUDA_VISIBLE_DEVICES:-3}"
fi

export WORLDMM_ALLOW_HF_DOWNLOAD="${WORLDMM_ALLOW_HF_DOWNLOAD:-0}"
if [[ "${WORLDMM_ALLOW_HF_DOWNLOAD}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi

DEFAULT_QWEN3_MODEL_PATH="$ROOT_DIR/models/Qwen3-Embedding-4B"
if [[ -z "${WORLDMM_TEXT_EMBED_MODEL:-}" || "${WORLDMM_TEXT_EMBED_MODEL:-}" == "/path/to/qwen3-embedding-4b" ]]; then
  export WORLDMM_TEXT_EMBED_MODEL="$DEFAULT_QWEN3_MODEL_PATH"
fi
export WORLDMM_TEXT_EMBED_HOST="${WORLDMM_TEXT_EMBED_HOST:-127.0.0.1}"
export WORLDMM_TEXT_EMBED_PORT="${WORLDMM_TEXT_EMBED_PORT:-18096}"
export WORLDMM_TEXT_EMBED_URL="${WORLDMM_TEXT_EMBED_URL:-http://${WORLDMM_TEXT_EMBED_HOST}:${WORLDMM_TEXT_EMBED_PORT}}"
export WORLDMM_TEXT_EMBED_DEVICE="${WORLDMM_TEXT_EMBED_DEVICE:-cuda:0}"

echo "[start_online_qwen3_embedding_server] WORLDMM_TEXT_EMBED_MODEL=${WORLDMM_TEXT_EMBED_MODEL}"
echo "[start_online_qwen3_embedding_server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[start_online_qwen3_embedding_server] WORLDMM_TEXT_EMBED_DEVICE=${WORLDMM_TEXT_EMBED_DEVICE}"
echo "[start_online_qwen3_embedding_server] WORLDMM_TEXT_EMBED_HOST=${WORLDMM_TEXT_EMBED_HOST}"
echo "[start_online_qwen3_embedding_server] WORLDMM_TEXT_EMBED_PORT=${WORLDMM_TEXT_EMBED_PORT}"

if python - "$WORLDMM_TEXT_EMBED_URL" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))
    if data.get("status") == "ok" and data.get("model_loaded"):
        raise SystemExit(0)
except SystemExit:
    raise
except Exception:
    pass
raise SystemExit(1)
PY
then
  echo "[start_online_qwen3_embedding_server] existing healthy service found at ${WORLDMM_TEXT_EMBED_URL}; reuse it"
  exit 0
fi

exec python online_qwen3_embedding_server.py "$@"
