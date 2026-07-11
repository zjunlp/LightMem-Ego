#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv_whisperx/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv_whisperx/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env_ffmpeg.sh"

exec python online_worker.py "$@"
