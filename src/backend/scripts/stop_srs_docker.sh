#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${EM2MEM_SRS_CONTAINER_NAME:-em2mem-srs}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed." >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME"
echo "SRS container stopped: $CONTAINER_NAME"
