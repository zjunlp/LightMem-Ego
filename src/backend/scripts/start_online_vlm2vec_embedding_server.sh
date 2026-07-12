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

export EM2MEM_ALLOW_HF_DOWNLOAD="${EM2MEM_ALLOW_HF_DOWNLOAD:-0}"
if [[ "${EM2MEM_ALLOW_HF_DOWNLOAD}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi

export EM2MEM_VLM2VEC_EMBED_HOST="${EM2MEM_VLM2VEC_EMBED_HOST:-127.0.0.1}"
export EM2MEM_VLM2VEC_EMBED_PORT="${EM2MEM_VLM2VEC_EMBED_PORT:-18091}"
export EM2MEM_VLM2VEC_EMBED_URL="${EM2MEM_VLM2VEC_EMBED_URL:-http://${EM2MEM_VLM2VEC_EMBED_HOST}:${EM2MEM_VLM2VEC_EMBED_PORT}}"

echo "[start_online_vlm2vec_embedding_server] EM2MEM_VLM2VEC_MODEL_PATH=${EM2MEM_VLM2VEC_MODEL_PATH:-}"
echo "[start_online_vlm2vec_embedding_server] EM2MEM_VLM2VEC_DEVICE=${EM2MEM_VLM2VEC_DEVICE:-cuda}"
echo "[start_online_vlm2vec_embedding_server] EM2MEM_VLM2VEC_EMBED_HOST=${EM2MEM_VLM2VEC_EMBED_HOST}"
echo "[start_online_vlm2vec_embedding_server] EM2MEM_VLM2VEC_EMBED_PORT=${EM2MEM_VLM2VEC_EMBED_PORT}"

if python - "$EM2MEM_VLM2VEC_EMBED_URL" <<'PY'
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
  echo "[start_online_vlm2vec_embedding_server] existing healthy service found at ${EM2MEM_VLM2VEC_EMBED_URL}; reuse it"
  exit 0
fi

exec python online_vlm2vec_embedding_server.py "$@"
