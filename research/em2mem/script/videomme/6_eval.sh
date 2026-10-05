#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# Usage: ./script/videomme/6_eval.sh

RET_MODEL="gpt-5-mini"
RESP_MODEL="gpt-5"
DURATION="long"
GPU_LIST="3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        --retriever-model) RET_MODEL="$2"; shift 2 ;;
        --respond-model) RESP_MODEL="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/../.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/videomme/eval"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPU_IDS <<< "$GPU_LIST"
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: ${GPU_IDS[*]} (total $NUM_GPUS)"

VIDEO_IDS=$(python -c "
import json
data = json.load(open('data/Video-MME/videomme/test.json'))
ids = sorted(set(row['video_id'] for row in data if row.get('duration') == '$DURATION'))
print(' '.join(ids))
")

VIDEO_ARRAY=($VIDEO_IDS)
TOTAL_VIDEOS=${#VIDEO_ARRAY[@]}
VIDEOS_PER_GPU=$(( (TOTAL_VIDEOS + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total videos: $TOTAL_VIDEOS, videos per GPU: ~$VIDEOS_PER_GPU"

pids=()
for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    start=$((idx * VIDEOS_PER_GPU))
    if [ $start -ge $TOTAL_VIDEOS ]; then
        continue
    fi
    subset=("${VIDEO_ARRAY[@]:$start:$VIDEOS_PER_GPU}")
    if [ ${#subset[@]} -eq 0 ]; then
        continue
    fi
    video_ids_str=$(IFS=,; echo "${subset[*]}")
    log_file="$LOG_DIR/gpu_${gpu}_${TIMESTAMP}.log"

    echo "GPU $gpu will process: $video_ids_str (log: $log_file)"

    CUDA_VISIBLE_DEVICES=$gpu python eval/eval_videomme.py \
        --retriever-model "$RET_MODEL" \
        --respond-model "$RESP_MODEL" \
        --duration "$DURATION" \
        --video-ids="$video_ids_str" \
        > "$log_file" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All parallel evaluations finished."
python eval/summarize_eval.py > ".log/videomme/eval/summarize_eval_${TIMESTAMP}.log" 2>&1