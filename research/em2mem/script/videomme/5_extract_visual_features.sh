#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# Usage: ./script/videomme/5_extract_visual_features.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

CAPTION_PATH="data/Video-MME/caption" OUTPUT_PATH="output/videomme/metadata/visual_memory" 
MODEL="gpt-5-mini" GPU_LIST="3" NUM_FRAMES=10

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --caption-path) CAPTION_PATH="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."
mkdir -p output/videomme/metadata/visual_memory

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/videomme/visual_memory"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Visual Memory: Extracting features...${NC}"
python preprocess/videomme/visual_memory/build_visual_memory.py \
    --caption-dir "$CAPTION_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --model "$MODEL" \
    --gpu "$GPU_LIST" \
    --num-frames "$NUM_FRAMES" \
    2>&1 | tee "$LOG_DIR/visual_features_$TIMESTAMP.log"

echo -e "${BLUE}Visual feature extraction completed.${NC}"