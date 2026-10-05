#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# Usage: ./script/videomme/3_build_mulmodal_memory_cell.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

VIDEO_PATH="data/Video-MME/data" TRANSCRIPT_PATH="data/Video-MME/transcript" CAPTION_PATH="data/Video-MME/caption" 
OUTPUT_PATH="output/videomme/metadata/multimodal_memory_cell" FRAMES_DIR="tmp/keyframes_aug"
MODEL="gpt-5-mini" UNIT_TIME=10 NUM_KEYFRAMES=2 MAX_WORKERS=8 OVERWRITE=""

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --video-path) VIDEO_PATH="$2"; shift 2 ;;
        --transcript-path) TRANSCRIPT_PATH="$2"; shift 2 ;;
        --caption-path) CAPTION_PATH="$2"; shift 2 ;;
        --evidence-path) EVIDENCE_PATH="$2"; shift 2 ;;
        --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --unit-time) UNIT_TIME="$2"; shift 2 ;;
        --num-keyframes) NUM_KEYFRAMES="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --overwrite) OVERWRITE="--overwrite"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."
mkdir -p output/metadata/multimodal_memory_cell

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/videomme/multimodal_memory_cell"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Multimodal Memory Cell: Generating fine captions...${NC}"
python preprocess/videomme/multimodal_memory_cell/generate_fine_caption.py \
    --video-path "$VIDEO_PATH" \
    --transcript-path "$TRANSCRIPT_PATH" \
    --output-path "$CAPTION_PATH" \
    --model "$MODEL" \
    --unit-time 10 \
    $OVERWRITE 2>&1 | tee "$LOG_DIR/generate_fine_caption_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell: Building multimodal event records...${NC}"
python preprocess/videomme/multimodal_memory_cell/build_multimodal_record.py \
    --video-path "$VIDEO_PATH" \
    --transcript-path "$TRANSCRIPT_PATH" \
    --caption-path "$CAPTION_PATH" \
    --output-path "$OUTPUT_PATH" \
    --frames-dir "$FRAMES_DIR" \
    --num-keyframes "$NUM_KEYFRAMES" \
    --model "$MODEL" \
    2>&1 | tee "$LOG_DIR/build_multimodal_record_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell: Building multiscale temporal context views...${NC}"
python preprocess/videomme/multimodal_memory_cell/generate_temporal_context_views.py \
    --record-dir "$OUTPUT_PATH" \
    2>&1 | tee "$LOG_DIR/generate_temporal_context_views_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell: Building episodic graphs...${NC}"
python preprocess/videomme/multimodal_memory_cell/build_episodic_graph.py \
    --record-dir "$OUTPUT_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --model "$MODEL" \
    2>&1 | tee "$LOG_DIR/build_episodic_graph_$TIMESTAMP.log"

echo -e "${BLUE}Multimodal Memory Cell construction completed!${NC}"