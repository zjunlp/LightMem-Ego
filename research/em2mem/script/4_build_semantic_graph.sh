#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Semantic Graph Construction Script
# Usage: ./script/4_build_semantic_graph.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" MODEL="gpt-5-mini"
SCALES=("30sec" "3min" "10min" "1h")

# ===== Paths =====
EPISODIC_FILE="output/metadata/multimodal_memory_cell/${PERSON}/30sec/episodic_triplets_30sec_${MODEL}.json"
SEMANTIC_DIR="output/metadata/semantic_graph/${PERSON}"
CAND_FILE="${SEMANTIC_DIR}/semantic_candidates_${MODEL}.json"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p output/metadata/semantic_graph/${PERSON}

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/semantic_graph/${PERSON}"
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

# ==== Build Semantic Graph =====
echo -e "${BLUE}Semantic Graph: Extracting semantic triples...${NC}"
python preprocess/semantic_graph/build_semantic_candidate.py \
    --episodic-file "$EPISODIC_FILE" \
    --output-dir "$SEMANTIC_DIR" \
    --model "$MODEL" 2>&1 | tee "$LOG_DIR/build_semantic_candidate_$TIMESTAMP.log"

echo -e "${BLUE}Semantic Graph: Building semantic graph...${NC}"
python preprocess/semantic_graph/consolidate_semantic_graph.py \
    --semantic-file "$CAND_FILE" \
    --output-dir "$SEMANTIC_DIR" \
    --model "$MODEL" 2>&1 | tee "$LOG_DIR/consolidate_semantic_graph_$TIMESTAMP.log"

echo -e "${BLUE}Semantic graph construction completed for ${PERSON}.${NC}"