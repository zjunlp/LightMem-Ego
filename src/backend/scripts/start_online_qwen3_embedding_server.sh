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
  export CUDA_VISIBLE_DEVICES="${EM2MEM_TEXT_EMBED_CUDA_VISIBLE_DEVICES:-3}"
fi

export EM2MEM_ALLOW_HF_DOWNLOAD="${EM2MEM_ALLOW_HF_DOWNLOAD:-0}"
if [[ "${EM2MEM_ALLOW_HF_DOWNLOAD}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi

DEFAULT_QWEN3_MODEL_PATH="$ROOT_DIR/models/Qwen3-Embedding-4B"
if [[ -z "${EM2MEM_TEXT_EMBED_MODEL:-}" || "${EM2MEM_TEXT_EMBED_MODEL:-}" == "/path/to/qwen3-embedding-4b" ]]; then
  export EM2MEM_TEXT_EMBED_MODEL="$DEFAULT_QWEN3_MODEL_PATH"
fi
export EM2MEM_TEXT_EMBED_HOST="${EM2MEM_TEXT_EMBED_HOST:-127.0.0.1}"
export EM2MEM_TEXT_EMBED_PORT="${EM2MEM_TEXT_EMBED_PORT:-18096}"
export EM2MEM_TEXT_EMBED_URL="${EM2MEM_TEXT_EMBED_URL:-http://${EM2MEM_TEXT_EMBED_HOST}:${EM2MEM_TEXT_EMBED_PORT}}"
export EM2MEM_TEXT_EMBED_DEVICE="${EM2MEM_TEXT_EMBED_DEVICE:-cuda:0}"

echo "[start_online_qwen3_embedding_server] EM2MEM_TEXT_EMBED_MODEL=${EM2MEM_TEXT_EMBED_MODEL}"
echo "[start_online_qwen3_embedding_server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[start_online_qwen3_embedding_server] EM2MEM_TEXT_EMBED_DEVICE=${EM2MEM_TEXT_EMBED_DEVICE}"
echo "[start_online_qwen3_embedding_server] EM2MEM_TEXT_EMBED_HOST=${EM2MEM_TEXT_EMBED_HOST}"
echo "[start_online_qwen3_embedding_server] EM2MEM_TEXT_EMBED_PORT=${EM2MEM_TEXT_EMBED_PORT}"

if python - "$EM2MEM_TEXT_EMBED_URL" <<'PY'
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
  echo "[start_online_qwen3_embedding_server] existing healthy service found at ${EM2MEM_TEXT_EMBED_URL}; reuse it"
  exit 0
fi

exec python online_qwen3_embedding_server.py "$@"
