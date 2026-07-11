#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"

DRY_RUN=0
FORCE=0
RESTART_WORKER=1
CLEAR_ACTIVE_MARKERS=1
TIMEOUT_SECONDS=8

usage() {
  cat <<'EOF'
Usage: scripts/cleanup_blocking_live_ingest.sh [--dry-run] [--force] [--no-restart] [--keep-active-markers] [--timeout SECONDS]

Clear stale Rokid RTMP live_ingest work that can block new glasses sessions.

This script:
  1. Stops online_live_ingest_worker.py and ffmpeg pull subprocesses recorded in runtime/state files.
  2. Removes queued and in-progress live_ingest task files.
  3. Marks affected live_ingest_state.json files as cleaned.
  4. Clears stale active session markers.
  5. Restarts the live_ingest worker unless --no-restart is set.

Do not run it while a glasses app session is intentionally streaming.

Options:
  --dry-run              Show what would be cleaned without changing anything.
  --force                Skip the confirmation prompt.
  --no-restart           Do not restart the live_ingest worker after cleanup.
  --keep-active-markers  Do not clear runtime/active_*.json files.
  --timeout SECONDS      Wait this many seconds after SIGTERM before SIGKILL.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-restart)
      RESTART_WORKER=0
      shift
      ;;
    --keep-active-markers)
      CLEAR_ACTIVE_MARKERS=0
      shift
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-}"
      if [[ ! "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
        echo "Error: --timeout requires a non-negative integer."
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unexpected argument '$1'"
      usage
      exit 1
      ;;
  esac
done

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "Error: no python interpreter found. Set PYTHON_BIN=/path/to/python."
  exit 1
fi

LIVE_INGEST_DIR="online_tasks/live_ingest"
LIVE_INGEST_IN_PROGRESS_DIR="online_tasks/live_ingest_in_progress"
RUNTIME_JSON="runtime/workers/live_ingest.json"
RUNTIME_PID="runtime/workers/live_ingest.pid"

mkdir -p "$LIVE_INGEST_DIR" "$LIVE_INGEST_IN_PROGRESS_DIR" logs/workers runtime/workers

declare -a TASK_FILES=()
declare -a SESSION_IDS=()
declare -a PID_CANDIDATES=()
declare -a UNIQUE_PIDS=()

add_task_file() {
  local path="$1"
  [[ -f "$path" ]] || return
  TASK_FILES+=("$path")
}

add_session_id() {
  local sid="${1:-}"
  [[ -n "$sid" ]] || return
  [[ "$sid" =~ ^[A-Za-z0-9_-]+$ ]] || return
  local existing
  for existing in "${SESSION_IDS[@]}"; do
    [[ "$existing" == "$sid" ]] && return
  done
  SESSION_IDS+=("$sid")
}

add_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  PID_CANDIDATES+=("$pid")
}

dedupe_pids() {
  local pid existing found
  UNIQUE_PIDS=()
  for pid in "${PID_CANDIDATES[@]}"; do
    found=0
    for existing in "${UNIQUE_PIDS[@]}"; do
      if [[ "$existing" == "$pid" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" == "0" ]]; then
      UNIQUE_PIDS+=("$pid")
    fi
  done
}

read_json_field() {
  local path="$1"
  local field="$2"
  "$PYTHON_BIN" - "$path" "$field" <<'PY'
import json
import sys

path, field = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    raise SystemExit(0)
if isinstance(payload, dict):
    value = payload.get(field)
    if value is not None:
        print(value)
PY
}

collect_task_files() {
  local file
  while IFS= read -r -d '' file; do
    add_task_file "$file"
  done < <(find "$LIVE_INGEST_DIR" -maxdepth 1 -type f -name '*.json' -print0)
  while IFS= read -r -d '' file; do
    add_task_file "$file"
  done < <(find "$LIVE_INGEST_IN_PROGRESS_DIR" -maxdepth 1 -type f -name '*.json' -print0)
}

collect_sessions_from_tasks() {
  local file sid
  for file in "${TASK_FILES[@]}"; do
    sid="$(read_json_field "$file" "session_id")"
    add_session_id "$sid"
  done
}

collect_worker_pids() {
  local pid
  if [[ -f "$RUNTIME_PID" ]]; then
    pid="$(cat "$RUNTIME_PID" 2>/dev/null || true)"
    add_pid "$pid"
  fi
  if [[ -f "$RUNTIME_JSON" ]]; then
    add_pid "$(read_json_field "$RUNTIME_JSON" "pid")"
    add_pid "$(read_json_field "$RUNTIME_JSON" "ffmpeg_video_pid")"
    add_pid "$(read_json_field "$RUNTIME_JSON" "ffmpeg_audio_pid")"
  fi
  while IFS= read -r pid; do
    add_pid "$pid"
  done < <(pgrep -f 'online_live_ingest_worker\.py' || true)
}

collect_ffmpeg_pids_from_sessions() {
  local sid state_path
  for sid in "${SESSION_IDS[@]}"; do
    state_path="online_sessions/${sid}/stream/live_ingest_state.json"
    [[ -f "$state_path" ]] || continue
    add_pid "$(read_json_field "$state_path" "ffmpeg_video_pid")"
    add_pid "$(read_json_field "$state_path" "ffmpeg_audio_pid")"
  done
}

mark_states_cleaned() {
  "$PYTHON_BIN" - "$@" <<'PY'
import datetime
import json
import sys
from pathlib import Path

session_ids = sys.argv[1:]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for sid in session_ids:
    path = Path("online_sessions") / sid / "stream" / "live_ingest_state.json"
    if not path.exists():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "status": "cleaned",
            "stop_requested": False,
            "stopped_at": now,
            "updated_at": now,
            "ffmpeg_video_pid": None,
            "ffmpeg_audio_pid": None,
            "last_error": "cleaned stale live_ingest task",
            "last_pull_error": "cleaned stale live_ingest task",
            "waiting_reason": None,
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

clear_active_markers() {
  "$PYTHON_BIN" <<'PY'
import datetime
import json
from pathlib import Path

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
for rel in ("runtime/active_rokid_session.json", "runtime/active_session.json"):
    path = Path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_session_id": None,
        "session_id": None,
        "cleared": True,
        "updated_at": now,
        "reason": "cleanup_blocking_live_ingest",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

stop_pids() {
  local pid
  [[ "${#UNIQUE_PIDS[@]}" -gt 0 ]] || return
  echo "[cleanup_blocking_live_ingest] Stopping pids: ${UNIQUE_PIDS[*]}"
  for pid in "${UNIQUE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep "$TIMEOUT_SECONDS"
  for pid in "${UNIQUE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[cleanup_blocking_live_ingest] SIGKILL pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

remove_task_files() {
  local file
  for file in "${TASK_FILES[@]}"; do
    if [[ "$file" != "$LIVE_INGEST_DIR"/*.json && "$file" != "$LIVE_INGEST_IN_PROGRESS_DIR"/*.json ]]; then
      echo "Refusing to remove unexpected path: $file"
      exit 1
    fi
    rm -f -- "$file"
  done
}

restart_worker() {
  if [[ "$RESTART_WORKER" != "1" ]]; then
    echo "[cleanup_blocking_live_ingest] Worker restart skipped by --no-restart."
    return
  fi
  echo "[cleanup_blocking_live_ingest] Restarting live_ingest worker..."
  nohup setsid bash scripts/start_online_live_ingest_worker.sh > logs/workers/live_ingest.log 2>&1 &
  echo "$!" > "$RUNTIME_PID"
}

print_summary() {
  echo "============================================"
  echo "  Cleanup Blocking Live Ingest"
  echo "  Root: $ROOT_DIR"
  echo "  Mode: $([[ "$DRY_RUN" == "1" ]] && echo DRY-RUN || echo APPLY)"
  echo "============================================"
  echo ""
  echo "Task files to remove: ${#TASK_FILES[@]}"
  local file sid pid
  for file in "${TASK_FILES[@]}"; do
    echo "  - $file"
  done
  echo ""
  echo "Affected sessions: ${#SESSION_IDS[@]}"
  for sid in "${SESSION_IDS[@]}"; do
    echo "  - $sid"
  done
  echo ""
  echo "Processes to stop: ${#UNIQUE_PIDS[@]}"
  for pid in "${UNIQUE_PIDS[@]}"; do
    echo "  - pid=$pid"
  done
  echo ""
}

collect_task_files
collect_sessions_from_tasks
collect_worker_pids
collect_ffmpeg_pids_from_sessions
dedupe_pids
print_summary

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$FORCE" != "1" ]]; then
  echo "This will stop the live_ingest worker and delete live_ingest queued/in-progress tasks."
  echo "Run only when no glasses session should currently be streaming."
  read -r -p "Continue? Type 'yes' to proceed: " answer
  if [[ "$answer" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

stop_pids
remove_task_files
if [[ "${#SESSION_IDS[@]}" -gt 0 ]]; then
  mark_states_cleaned "${SESSION_IDS[@]}"
fi
if [[ "$CLEAR_ACTIVE_MARKERS" == "1" ]]; then
  clear_active_markers
fi
restart_worker

sleep 1
echo "[cleanup_blocking_live_ingest] Done."
if [[ -f "$RUNTIME_JSON" ]]; then
  "$PYTHON_BIN" - "$RUNTIME_JSON" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    raise SystemExit(0)
print(
    "[cleanup_blocking_live_ingest] worker_status="
    + str(data.get("status"))
    + " queue_pending="
    + str(data.get("queue_pending"))
    + " active_sessions="
    + str(data.get("active_sessions"))
)
PY
fi
