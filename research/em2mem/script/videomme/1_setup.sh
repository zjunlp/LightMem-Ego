#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem setup script
# Usage: ./script/videomme/1_setup.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

DURATION="long"

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

case "$DURATION" in
    all|short|medium|long) ;;
    *)
        echo "Invalid --duration: $DURATION"
        echo "Expected one of: all, short, medium, long"
        exit 1
        ;;
esac

cd "$(dirname "$0")/../.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/setup/videomme"
mkdir -p "$LOG_DIR"

if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}uv could not be found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee "$LOG_DIR/uv_install_$TIMESTAMP.log"
fi

echo -e "${BLUE}Setting up virtual environment and installing dependencies...${NC}"
MAX_JOBS=8 uv sync 2>&1 | tee "$LOG_DIR/uv_sync_$TIMESTAMP.log"

source .venv/bin/activate

echo -e "${BLUE}Downloading Video-MME dataset...${NC}"
{
    hf download lmms-lab/Video-MME --repo-type=dataset --local-dir data/Video-MME

    find data/Video-MME -type f -name 'videos_chunked_*.zip' -print0 |
    while IFS= read -r -d '' zip; do
        unzip -o "$zip" -d "$(dirname "$zip")" && rm "$zip"
    done

    python data/Video-MME/utils/reformat.py --duration "$DURATION"
    python data/Video-MME/utils/filter_videos.py --duration "$DURATION"
    unzip data/Video-MME/caption.zip && rm data/Video-MME/caption.zip
} 2>&1 | tee "$LOG_DIR/hf_download_videomme_$TIMESTAMP.log"

echo -e "${BLUE}Setup Done! Dependencies installed and Video-MME downloaded for duration=${DURATION}.${NC}"
