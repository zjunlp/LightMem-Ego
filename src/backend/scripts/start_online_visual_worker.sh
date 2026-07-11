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

export WORLDMM_CLEAN_CUDA_ENV="${WORLDMM_CLEAN_CUDA_ENV:-1}"
if [[ "${WORLDMM_CLEAN_CUDA_ENV}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat/lib"
  unset CUDA_HOME
fi
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
if [[ "${WORLDMM_CUDA_LAUNCH_BLOCKING:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export CUDA_LAUNCH_BLOCKING=1
else
  unset CUDA_LAUNCH_BLOCKING
fi

DEFAULT_VLM2VEC_MODEL_PATH="$ROOT_DIR/models/VLM2Vec-V2.0"
if [[ -z "${WORLDMM_VLM2VEC_MODEL_PATH:-}" || "${WORLDMM_VLM2VEC_MODEL_PATH:-}" == "/path/to/vlm2vec-v2" ]]; then
  export WORLDMM_VLM2VEC_MODEL_PATH="${WORLDMM_VIS_EMBED_MODEL:-$DEFAULT_VLM2VEC_MODEL_PATH}"
fi
export WORLDMM_ALLOW_HF_DOWNLOAD="${WORLDMM_ALLOW_HF_DOWNLOAD:-0}"
if [[ "${WORLDMM_ALLOW_HF_DOWNLOAD}" != "1" ]]; then
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi

echo "[start_online_visual_worker] WORLDMM_VISUAL_BACKEND=${WORLDMM_VISUAL_BACKEND:-vlm2vec}"
echo "[start_online_visual_worker] WORLDMM_CLEAN_CUDA_ENV=${WORLDMM_CLEAN_CUDA_ENV}"
echo "[start_online_visual_worker] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "[start_online_visual_worker] CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "[start_online_visual_worker] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-<unset>}"
echo "[start_online_visual_worker] WORLDMM_VLM2VEC_MODEL_PATH=${WORLDMM_VLM2VEC_MODEL_PATH}"
echo "[start_online_visual_worker] WORLDMM_ALLOW_HF_DOWNLOAD=${WORLDMM_ALLOW_HF_DOWNLOAD}"
if [[ "${WORLDMM_VISUAL_BACKEND:-vlm2vec}" == "remote" ]]; then
  python - <<'PY'
import sys
print(f"[start_online_visual_worker] python={sys.executable}", flush=True)
print("[start_online_visual_worker] skipping torch diagnostics for remote backend", flush=True)
PY
else
  python - <<'PY'
import sys
print(f"[start_online_visual_worker] python={sys.executable}", flush=True)
try:
    import torch
    print(f"[start_online_visual_worker] torch.__version__={torch.__version__}", flush=True)
    print(f"[start_online_visual_worker] torch.version.cuda={torch.version.cuda}", flush=True)
    print(f"[start_online_visual_worker] torch.__file__={torch.__file__}", flush=True)
except Exception as exc:
    print(f"[start_online_visual_worker] torch_info_error={type(exc).__name__}: {exc}", flush=True)
PY
fi

exec python online_visual_worker.py "$@"
