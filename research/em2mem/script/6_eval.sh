#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Evaluation Script
# Usage: ./script/6_eval.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" RETRIEVER_MODEL="gpt-5-mini" RESPOND_MODEL="gpt-5" 
EPISODIC_TOP_K="5" SEMANTIC_TOP_K="8" VISUAL_TOP_K="3" 
NUM_WORKERS="8" MAX_ROUNDS="3" MAX_ERRORS="3" OUTPUT_DIR="output"
GPU_LIST="0,1"

# ===== Paths =====
MULTIMODAL_MEMORY_CELL_DIR="output/metadata/multimodal_memory_cell/${PERSON}"
SEMANTIC_GRAPH_DIR="output/metadata/semantic_graph/${PERSON}"
VISUAL_DIR="output/metadata/visual_memory/${PERSON}"
VISUAL_EVIDENCE_FILE="${MULTIMODAL_MEMORY_CELL_DIR}/${PERSON}_record.json"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
        --respond-model) RESPOND_MODEL="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_TOP_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_TOP_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_TOP_K="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/eval/${PERSON}"
mkdir -p "$LOG_DIR"

# ===== Env =====
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export HF_ENDPOINT=https://hf-mirror.com

python eval/eval.py \
    --person "$PERSON" \
    --retriever-model "$RETRIEVER_MODEL" \
    --respond-model "$RESPOND_MODEL" \
    --episodic-top-k "$EPISODIC_TOP_K" \
    --semantic-top-k "$SEMANTIC_TOP_K" \
    --visual-top-k "$VISUAL_TOP_K" \
    --num-workers "$NUM_WORKERS" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-list "$GPU_LIST" \
    --memory-cell-dir "$MULTIMODAL_MEMORY_CELL_DIR" \
    --semantic-graph-dir "$SEMANTIC_GRAPH_DIR" \
    --visual-dir "$VISUAL_DIR" \
    --visual-evidence-file "$VISUAL_EVIDENCE_FILE" \
    2>&1 | tee "${LOG_DIR}/eval_${PERSON}_${TIMESTAMP}.log"