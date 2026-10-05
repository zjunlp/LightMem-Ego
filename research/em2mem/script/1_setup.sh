#!/bin/bash
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

# EM2Mem Setup Script
# Usage: ./script/1_setup.sh

set -e
trap 'echo -e "\nInterrupted."; exit 130' INT TERM

cd "$(dirname "$0")/.."

BLUE='\033[1;34m' NC='\033[0m'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=".log/setup"
mkdir -p "$LOG_DIR"

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}uv could not be found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee "$LOG_DIR/uv_install_$TIMESTAMP.log"
fi

# Set up virtual environment and install dependencies
echo -e "${BLUE}Setting up virtual environment and installing dependencies...${NC}"
uv sync 2>&1 | tee "$LOG_DIR/uv_sync_$TIMESTAMP.log"

# Download EgoLife dataset
echo -e "${BLUE}Downloading EgoLife dataset...${NC}"
hf download lmms-lab/EgoLife --repo-type=dataset --local-dir data/EgoLife 2>&1 | tee "$LOG_DIR/hf_download_$TIMESTAMP.log"

echo -e "${BLUE}Setup Done! Dependencies installed and data downloaded.${NC}"
