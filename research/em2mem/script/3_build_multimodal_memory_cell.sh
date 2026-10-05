#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Multimodal Memory Cell Construction Script
# Usage: ./script/3_build_multimodal_memory_cell.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" MODEL="gpt-5-mini"
NUM_KEYFRAMES=3 MAX_WORKERS=8 DAY="" MAX_SYNC_FILES=""
SCALES=("30sec" "3min" "10min" "1h")

# ===== Paths =====
SYNC_DIR="data/EgoLife/EgoLifeCap/Sync" FINECAP_FILE="data/EgoLife/EgoLifeCap/${PERSON}/${PERSON}.json"
EVENT_RECORD_FILE="output/metadata/multimodal_memory_cell/${PERSON}/${PERSON}_record.json"
VIDEO_SEARCH_ROOT="data/EgoLife" FRAMES_DIR="tmp/egolife_keyframes_aug/${PERSON}"
TEMPORAL_CONTEXT_VIEWS_DIR="output/metadata/multimodal_memory_cell/${PERSON}/temporal_context_views"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --sync-dir) SYNC_DIR="$2"; shift 2 ;;
        --num-keyframes) NUM_KEYFRAMES="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --day) DAY="$2"; shift 2 ;;
        --max-sync-files) MAX_SYNC_FILES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p output/metadata/multimodal_memory_cell/${PERSON}

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/build_multimodal_memory_cell/${PERSON}"
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

# ==== Build Multimodal Memory Cell =====
echo -e "${BLUE}Multimodal Memory Cell: Generating fine captions...${NC}"
python preprocess/multimodal_memory_cell/generate_fine_caption.py \
    --sync-dir "${SYNC_DIR}" \
    --output "${FINECAP_FILE}" 2>&1 | tee "$LOG_DIR/generate_fine_caption_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell: Building multimodal event records...${NC}"
CMD=(
    python preprocess/multimodal_memory_cell/build_multimodal_event_record.py
    --input "$FINECAP_FILE"
    --output "$EVENT_RECORD_FILE"
    --sync-dir "$SYNC_DIR"
    --person "$PERSON"
    --video-search-root "$VIDEO_SEARCH_ROOT"
    --frames-dir "$FRAMES_DIR"
    --num-keyframes "$NUM_KEYFRAMES"
    --model "$MODEL"
    --max-workers "$MAX_WORKERS"
)
if [ -n "$DAY" ]; then
    CMD+=(--day "$DAY")
fi
if [ -n "$MAX_SYNC_FILES" ]; then
    CMD+=(--max-sync-files "$MAX_SYNC_FILES")
fi
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/build_multimodal_event_record_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell: Generating temporal context views...${NC}"
python preprocess/multimodal_memory_cell/generate_temporal_context_views.py \
    --person "$PERSON" \
    --json-path "$EVENT_RECORD_FILE" \
    --save-path "$TEMPORAL_CONTEXT_VIEWS_DIR" 2>&1 | tee "$LOG_DIR/generate_temporal_context_views_$TIMESTAMP.log"

for SCALE in "${SCALES[@]}"; do
    echo -e "${BLUE}Multimodal Memory Cell: Building episodic graph for scale ${SCALE}...${NC}"
    if [[ "$SCALE" == "30sec" ]]; then
        INPUT_FILE="${EVENT_RECORD_FILE}"
    else
        INPUT_FILE="${TEMPORAL_CONTEXT_VIEWS_DIR}/temporal_context_views_${SCALE}.json"
    fi
    OUTPUT_DIR="output/metadata/multimodal_memory_cell/${PERSON}/${SCALE}"
    python preprocess/multimodal_memory_cell/build_episodic_graph.py \
        --input-file "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --person "$PERSON" \
        --model "$MODEL" \
        --max-workers "$MAX_WORKERS" 2>&1 | tee "$LOG_DIR/build_episodic_graph_${SCALE}_$TIMESTAMP.log"
done

echo -e "${BLUE}Multimodal Memory Cell construction completed!${NC}"