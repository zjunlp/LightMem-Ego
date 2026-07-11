#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_PATH="${WORLDMM_SRS_CONF:-$ROOT_DIR/deploy/srs/srs.conf}"
CONTAINER_NAME="${WORLDMM_SRS_CONTAINER_NAME:-worldmm-srs}"
IMAGE="${WORLDMM_SRS_DOCKER_IMAGE:-ossrs/srs:5}"
RTMP_PORT="${WORLDMM_SRS_RTMP_PORT:-1935}"
HTTP_API_PORT="${WORLDMM_SRS_HTTP_API_PORT:-1985}"
HTTP_SERVER_PORT="${WORLDMM_SRS_HTTP_SERVER_PORT:-8080}"
RTC_UDP_PORT="${WORLDMM_SRS_RTC_UDP_PORT:-8000}"

if [[ ! -f "$CONF_PATH" ]]; then
  echo "SRS config not found: $CONF_PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Install SRS manually or install Docker on the VPS." >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${RTMP_PORT}:1935" \
  -p "127.0.0.1:${HTTP_API_PORT}:1985" \
  -p "127.0.0.1:${HTTP_SERVER_PORT}:8080" \
  -p "${RTC_UDP_PORT}:8000/udp" \
  -v "$CONF_PATH:/usr/local/srs/conf/worldmm.conf:ro" \
  "$IMAGE" \
  ./objs/srs -c conf/worldmm.conf

echo "SRS container started: $CONTAINER_NAME"
echo "RTMP publish URL format: rtmp://localhost:${RTMP_PORT}/live/<stream_name>"
echo "WebRTC RTC UDP port: ${RTC_UDP_PORT}/udp"
docker logs --tail 50 "$CONTAINER_NAME"
