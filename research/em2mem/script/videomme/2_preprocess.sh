#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Preprocessing Script
# Usage: ./script/videomme/2_preprocess.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

VIDEO_PATH="data/Video-MME/data"
OUTPUT_PATH="data/Video-MME/transcript"
WHISPER_MODEL="distil-large-v3.5"
BATCH_SIZE="16"

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --model) WHISPER_MODEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."

source .venv/bin/activate

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/preprocess/videomme"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Generating Transcripts...${NC}"
python data/Video-MME/utils/transcribe.py --input-path "$VIDEO_PATH" --output-path "$OUTPUT_PATH" --model "$WHISPER_MODEL" --batch-size "$BATCH_SIZE" 2>&1 | tee "$LOG_DIR/transcribe_$TIMESTAMP.log"

echo -e "${BLUE}Preprocess Done!${NC}"
