#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Preprocessing Script
# Usage: ./script/2_preprocess.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE"

source .venv/bin/activate

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/preprocess/${PERSON}"
mkdir -p "$LOG_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export MAX_WORKERS_FILES="${MAX_WORKERS_FILES:-2}"
export MAX_WORKERS_LINES="${MAX_WORKERS_LINES:-8}"


echo -e "${BLUE}Translating DenseCaption...${NC}"
python data/EgoLife/utils/translate_densecap.py 2>&1 | tee "$LOG_DIR/translate_densecap_$TIMESTAMP.log"

echo -e "${BLUE}Generating Sync data...${NC}"
python data/EgoLife/utils/generate_sync.py 2>&1 | tee "$LOG_DIR/generate_sync_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done! Output: output/metadata/*_memory/${PERSON}/${NC}"
