#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# Usage: ./script/videomme/4_build_semantic_graph.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

EPISODIC_PATH="output/videomme/metadata/multimodal_memory_cell"
SEMANTIC_PATH="output/videomme/metadata/semantic_graph"
OUTPUT_PATH="output/videomme/metadata/semantic_graph"
MODEL="gpt-5-mini"
SOURCE_FIELD="openie_results"
PERIOD=10
MIN_GROUP_TRIPLES=1
GPU_LIST="3"

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --episodic-path) EPISODIC_PATH="$2"; shift 2 ;;
        --semantic-path) SEMANTIC_PATH="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --source-field) SOURCE_FIELD="$2"; shift 2 ;;
        --period) PERIOD="$2"; shift 2 ;;
        --min-group-triples) MIN_GROUP_TRIPLES="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."
mkdir -p output/videomme/metadata/semantic_graph

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/videomme/semantic_graph"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Semantic Graph: Building semantic candidates...${NC}"
python preprocess/videomme/semantic_graph/build_semantic_candidate.py \
    --episodic-dir "$EPISODIC_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --model "$MODEL" \
    --source-field "$SOURCE_FIELD" \
    --period "$PERIOD" \
    --min-group-triples "$MIN_GROUP_TRIPLES" \
    2>&1 | tee "$LOG_DIR/build_semantic_candidates_$TIMESTAMP.log"

echo -e "${BLUE}Semantic Graph: Consolidating semantic graph...${NC}"
IFS=',' read -ra gpu_array <<< "$GPU_LIST"
NUM_GPUS=${#gpu_array[@]}
echo "Using GPUs: ${gpu_array[*]} (total $NUM_GPUS GPUs)"

video_names=()
for dir in "$SEMANTIC_PATH"/*/; do
    [ -d "$dir" ] || continue
    video_name=$(basename "$dir")
    output_file="$OUTPUT_PATH/$video_name/semantic_graph_${MODEL}.json"
    if [ ! -f "$output_file" ]; then
        video_names+=("$video_name")
    fi
done

total_videos=${#video_names[@]}
if [ $total_videos -eq 0 ]; then
    echo "No videos found in $SEMANTIC_PATH, skipping consolidation."
    exit 0
fi

videos_per_gpu=$(( (total_videos + NUM_GPUS - 1) / NUM_GPUS ))

pids=()
for (( i=0; i<NUM_GPUS; i++ )); do
    start=$(( i * videos_per_gpu ))
    if [ $start -ge $total_videos ]; then
        break
    fi

    subset=("${video_names[@]:$start:$videos_per_gpu}")
    video_names_str=$(IFS=,; echo "${subset[*]}")
    gpu_id="${gpu_array[i]}"
    log_file="$LOG_DIR/consolidate_semantic_graph_gpu${gpu_id}_$TIMESTAMP.log"

    echo "Launching consolidation on GPU $gpu_id for videos: $video_names_str"
    CUDA_VISIBLE_DEVICES="$gpu_id" python preprocess/videomme/semantic_graph/consolidate_semantic_graph.py \
        --semantic-dir "$SEMANTIC_PATH" \
        --output-dir "$OUTPUT_PATH" \
        --model "$MODEL" \
        --video-names="$video_names_str" \
        2>&1 | tee -a "$log_file" &
    pids+=($!)
done

set +e
for pid in "${pids[@]}"; do
    wait $pid
done
set -e

echo -e "${BLUE}Semantic graph construction completed.${NC}"