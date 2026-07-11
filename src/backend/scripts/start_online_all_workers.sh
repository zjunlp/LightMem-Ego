#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/workers runtime/workers

unset all_proxy && unset ALL_PROXY

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

REQUESTED_PIPELINE_MODE="${WORLDMM_PIPELINE_MODE:-}"
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi
if [[ -n "$REQUESTED_PIPELINE_MODE" ]]; then
  WORLDMM_PIPELINE_MODE="$REQUESTED_PIPELINE_MODE"
fi

PIPELINE_MODE="${WORLDMM_PIPELINE_MODE:-mst}"
PIPELINE_MODE="$(printf '%s' "$PIPELINE_MODE" | tr '[:upper:]' '[:lower:]')"
case "$PIPELINE_MODE" in
  mst|legacy|hybrid) ;;
  *) PIPELINE_MODE="mst" ;;
esac
export WORLDMM_PIPELINE_MODE="$PIPELINE_MODE"
export HF_ENDPOINT="https://hf-mirror.com"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

start_worker() {
  local name="$1"
  shift
  local log_path="logs/workers/${name}.log"
  local pid_path="runtime/workers/${name}.pid"
  local runtime_path="runtime/workers/${name}.json"
  echo "[start_online_all_workers] starting ${name}: $*"
  if [[ -f "$pid_path" ]]; then
    local existing_pid
    existing_pid="$(cat "$pid_path" 2>/dev/null || true)"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[start_online_all_workers] ${name} already running pid=${existing_pid}; reusing it"
      return
    fi
  fi
  if [[ -f "$runtime_path" ]]; then
    local runtime_pid
    runtime_pid="$("$PYTHON_BIN" - "$runtime_path" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    pid = data.get("pid")
    print(pid if pid else "")
except Exception:
    print("")
PY
)"
    if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
      echo "[start_online_all_workers] ${name} already running runtime_pid=${runtime_pid}; reusing it"
      return
    fi
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  nohup setsid "$@" > "$log_path" 2>&1 &
  echo "$!" > "$pid_path"
  echo "[start_online_all_workers] ${name} pid=$(cat "$pid_path") log=${log_path}"
}

is_enabled() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

worker_count() {
  local raw="${1:-1}"
  if [[ ! "$raw" =~ ^[0-9]+$ ]] || [[ "$raw" -lt 1 ]]; then
    echo 1
    return
  fi
  echo "$raw"
}

vlm2vec_embedding_ready() {
  local url="${WORLDMM_VLM2VEC_EMBED_URL:-http://${WORLDMM_VLM2VEC_EMBED_HOST:-127.0.0.1}:${WORLDMM_VLM2VEC_EMBED_PORT:-18091}}"
  "$PYTHON_BIN" - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("status") == "ok" and data.get("model_loaded"):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

qwen3_embedding_ready() {
  local url="${WORLDMM_TEXT_EMBED_URL:-http://${WORLDMM_TEXT_EMBED_HOST:-127.0.0.1}:${WORLDMM_TEXT_EMBED_PORT:-18096}}"
  "$PYTHON_BIN" - "$url" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("status") == "ok" and data.get("model_loaded"):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[start_online_all_workers] Pipeline mode: ${PIPELINE_MODE}"
if [[ "$PIPELINE_MODE" == "legacy" ]]; then
  echo "[start_online_all_workers] Main path: preprocess -> legacy evidence -> memory -> visual -> query"
else
  echo "[start_online_all_workers] Main path: stream/live_ingest/preprocess -> M_cur/M_st -> MST refine -> MST consolidation -> memory -> visual -> query"
fi

start_worker preprocess bash scripts/start_online_worker.sh
echo "[start_online_all_workers] preprocess worker also consumes stream_asr tasks with the same warm WhisperXRuntime"

if [[ "$PIPELINE_MODE" != "legacy" ]]; then
  start_worker stream bash scripts/start_online_stream_worker.sh
  start_worker live_ingest bash scripts/start_online_live_ingest_worker.sh
fi

if [[ "$PIPELINE_MODE" == "legacy" ]]; then
  echo "[start_online_all_workers] Legacy evidence worker: main"
  start_worker evidence bash scripts/start_online_evidence_worker.sh --backend openai
elif [[ "$PIPELINE_MODE" == "hybrid" ]]; then
  echo "[start_online_all_workers] Legacy evidence worker: optional legacy"
  start_worker evidence bash scripts/start_online_evidence_worker.sh --backend openai
else
  if [[ "${WORLDMM_ENABLE_LEGACY_EVIDENCE_WORKER:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    echo "[start_online_all_workers] Legacy evidence worker: explicitly enabled as legacy_optional"
    start_worker evidence bash scripts/start_online_evidence_worker.sh --backend openai
  else
    echo "[start_online_all_workers] Legacy evidence worker: disabled by default"
  fi
fi

if [[ "$PIPELINE_MODE" != "legacy" ]]; then
  REFINE_WORKER_COUNT="$(worker_count "${WORLDMM_MST_REFINE_WORKER_COUNT:-1}")"
  echo "[start_online_all_workers] refine worker count: ${REFINE_WORKER_COUNT}"
  if [[ "$REFINE_WORKER_COUNT" == "1" ]]; then
    start_worker refine bash scripts/start_online_mst_refine_worker.sh
  else
    for ((i = 1; i <= REFINE_WORKER_COUNT; i++)); do
      start_worker "refine_${i}" env WORLDMM_WORKER_INSTANCE_NAME="refine_${i}" bash scripts/start_online_mst_refine_worker.sh
    done
  fi
  start_worker consolidation bash scripts/start_online_mst_consolidation_worker.sh
fi

if is_enabled "${WORLDMM_AUTO_VISUAL_EMBEDDING:-1}" && [[ "${WORLDMM_VISUAL_BACKEND:-vlm2vec}" == "remote" ]]; then
  if vlm2vec_embedding_ready; then
    echo "[start_online_all_workers] VLM2Vec embedding service already healthy at ${WORLDMM_VLM2VEC_EMBED_URL:-http://${WORLDMM_VLM2VEC_EMBED_HOST:-127.0.0.1}:${WORLDMM_VLM2VEC_EMBED_PORT:-18091}}; reusing it"
  else
    start_worker vlm2vec_embedding bash scripts/start_online_vlm2vec_embedding_server.sh
  fi
fi

if [[ "${WORLDMM_TEXT_EMBED_BACKEND:-local}" == "remote" ]]; then
  if qwen3_embedding_ready; then
    echo "[start_online_all_workers] Qwen3 text embedding service already healthy at ${WORLDMM_TEXT_EMBED_URL:-http://${WORLDMM_TEXT_EMBED_HOST:-127.0.0.1}:${WORLDMM_TEXT_EMBED_PORT:-18096}}; reusing it"
  else
    CUDA_VISIBLE_DEVICES="${WORLDMM_TEXT_EMBED_CUDA_VISIBLE_DEVICES:-3}" WORLDMM_TEXT_EMBED_DEVICE="${WORLDMM_TEXT_EMBED_DEVICE:-cuda:0}" start_worker qwen3_embedding bash scripts/start_online_qwen3_embedding_server.sh
  fi
fi

if is_enabled "${WORLDMM_AUTO_VISUAL_EMBEDDING:-1}"; then
  start_worker visual bash scripts/start_online_visual_worker.sh
else
  echo "[start_online_all_workers] Visual embedding worker: disabled by WORLDMM_AUTO_VISUAL_EMBEDDING=0"
fi
start_worker memory bash scripts/start_online_memory_worker.sh
CUDA_VISIBLE_DEVICES=1 start_worker rokid_day_merge bash scripts/start_online_rokid_day_merge_worker.sh
CUDA_VISIBLE_DEVICES=3 start_worker query bash scripts/start_online_query_worker.sh

echo "[start_online_all_workers] started. Monitor with: python monitor_online_pipeline.py --watch"
