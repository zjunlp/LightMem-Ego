#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Visual Feature Extraction Script
# Usage: ./script/5_extract_visual_features.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

PERSON="A1_JAKE" GPU_LIST="3" NUM_FRAMES=16


source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --person) PERSON="$2"; shift 2 ;;
        --gpu) GPU_LIST="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p output/metadata/visual_memory/${PERSON}

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/visual_memory/${PERSON}"
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

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_SPLITS=${#GPUS[@]}

if [ "$NUM_SPLITS" -eq 1 ]; then
    LOG_FILE="$LOG_DIR/visual_features_gpu${GPUS[0]}_$TIMESTAMP.log"
    echo -e "${BLUE}Extracting visual features on GPU ${GPUS[0]}...${NC}"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python preprocess/visual_memory/extract_visual_features.py \
        --person "$PERSON" \
        --num-frames "$NUM_FRAMES" 2>&1 | tee "$LOG_FILE"
else
    echo -e "${BLUE}Extracting visual features in parallel on GPUs: ${GPUS[*]}...${NC}"
    for i in "${!GPUS[@]}"; do
        GPU="${GPUS[i]}"
        LOG_FILE="$LOG_DIR/visual_features_gpu${GPU}_part$i_$TIMESTAMP.log"
        echo -e "${BLUE}Starting split $((i+1))/$NUM_SPLITS on GPU $GPU...${NC}"
        CUDA_VISIBLE_DEVICES=$GPU python preprocess/visual_memory/extract_visual_features.py \
            --person "$PERSON" \
            --split-id "$i" \
            --num-splits "$NUM_SPLITS" \
            --num-frames "$NUM_FRAMES" \
            > "$LOG_FILE" 2>&1 &
        
        pids+=($!)
        sleep 2
    done
    
    echo ""
    echo "All $NUM_SPLITS processes launched. Waiting for completion..."
    echo ""

    failed=0
    for i in "${!pids[@]}"; do
        pid=${pids[i]}
        if wait $pid; then
            echo -e "${BLUE}✓ Split $((i+1))/$NUM_SPLITS (PID $pid) completed successfully.${NC}"
        else
            echo -e "${BLUE}✗ Split $((i+1))/$NUM_SPLITS (PID $pid) failed.${NC}"
            failed=1
        fi
    done

    if [ "$failed" -eq 1 ]; then
        echo -e "${BLUE}Some splits failed.${NC}"
        exit 1
    fi

    echo -e "${BLUE}All splits completed successfully.${NC}"
    echo ""
    echo "Merging split files..."

    python preprocess/visual_memory/extract_visual_features.py \
        --person "$PERSON" \
        --num-splits "$NUM_SPLITS" \
        --merge 2>&1 | tee "$LOG_DIR/visual_features_merge_$TIMESTAMP.log"
fi

echo -e "${BLUE}Visual feature extraction completed for ${PERSON}.${NC}"