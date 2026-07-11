#!/usr/bin/env bash
set -euo pipefail

PYTHON_WORKER_RE='(^|[[:space:]/])python([0-9.]*)?[[:space:]]+([^[:space:]]+[[:space:]]+)*online_(worker|stream_worker|live_ingest_worker|evidence_worker|mst_refine_worker|mst_consolidation_worker|visual_worker|memory_worker|query_worker|rokid_day_merge_worker)\.py([[:space:]]|$)'
UVICORN_RE='(^|[[:space:]/])python([0-9.]*)?[[:space:]]+-m[[:space:]]+uvicorn[[:space:]]+api_server:app([[:space:]]|$)|(^|[[:space:]/])uvicorn[[:space:]]+api_server:app([[:space:]]|$)'
WRAPPER_RE='(^|[[:space:]/])(bash|sh)[[:space:]]+([^[:space:]]+[[:space:]]+)*scripts/(start_api|start_online_worker|start_online_stream_worker|start_online_live_ingest_worker|start_online_evidence_worker|start_online_mst_refine_worker|start_online_mst_consolidation_worker|start_online_visual_worker|start_online_memory_worker|start_online_query_worker|start_online_rokid_day_merge_worker)\.sh([[:space:]]|$)'
PATTERN="${PYTHON_WORKER_RE}|${UVICORN_RE}|${WRAPPER_RE}"
PROTECTED_PATTERN='scripts/start_online_vlm2vec_embedding_server.sh|online_vlm2vec_embedding_server.py|scripts/start_online_qwen3_embedding_server.sh|online_qwen3_embedding_server.py'
TIMEOUT_SECONDS=8
DRY_RUN=0
FORCE=0
START_AFTER_STOP=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/stop_online_backend_processes.sh [--dry-run] [--force] [--timeout SECONDS] [--no-start]

Stops OmniSpark backend API and online worker processes matched by pgrep, then starts
scripts/start_online_all_workers.sh and scripts/start_api.sh.
The VLM2Vec and Qwen3 embedding APIs are protected and are not stopped by this script.

Options:
  --dry-run          Show matching processes without stopping them.
  --force           Send SIGKILL to processes that do not exit after timeout.
  --no-start         Stop matching processes without starting services again.
  --timeout SECONDS Wait this many seconds after normal kill before checking.
  -h, --help        Show this help.
EOF
}

start_services() {
  if [[ "$START_AFTER_STOP" != "1" ]]; then
    echo "[stop_online_backend_processes] --no-start was set; services were not restarted."
    exit 0
  fi

  cd "$ROOT_DIR"
  echo "[stop_online_backend_processes] Starting workers with scripts/start_online_all_workers.sh..."
  bash scripts/start_online_all_workers.sh
  echo "[stop_online_backend_processes] Starting API with scripts/start_api.sh..."
  bash scripts/start_api.sh
  exit 0
}

add_pid() {
  local pid="${1:-}"
  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ || "$pid" == "$$" || "$pid" == "$PPID" ]]; then
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  if is_protected_pid "$pid"; then
    return
  fi
  if ! is_target_pid "$pid"; then
    return
  fi
  TARGET_PIDS+=("$pid")
}

add_process_group_pid() {
  local pid="${1:-}"
  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ || "$pid" == "$$" || "$pid" == "$PPID" ]]; then
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi
  if is_protected_pid "$pid"; then
    return
  fi
  TARGET_PIDS+=("$pid")
}

is_protected_pid() {
  local pid="$1"
  local cmdline
  cmdline="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  [[ -n "$cmdline" && "$cmdline" =~ $PROTECTED_PATTERN ]]
}

is_target_pid() {
  local pid="$1"
  local cmdline
  cmdline="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  [[ -n "$cmdline" ]] || return 1
  [[ "$cmdline" =~ $PYTHON_WORKER_RE || "$cmdline" =~ $UVICORN_RE || "$cmdline" =~ $WRAPPER_RE ]]
}

is_protected_runtime_path() {
  local path="$1"
  local base
  base="$(basename "$path")"
  [[ "$base" == vlm2vec_embedding.pid || "$base" == vlm2vec_embedding.json || "$base" == qwen3_embedding.pid || "$base" == qwen3_embedding.json ]]
}

read_json_pid() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pid = payload.get("pid")
    print(pid if isinstance(pid, int) else "")
except Exception:
    print("")
PY
}

unique_pids() {
  local seen=" "
  local pid
  UNIQUE_PIDS=()
  for pid in "$@"; do
    if [[ "$seen" == *" $pid "* ]]; then
      continue
    fi
    seen+="$pid "
    UNIQUE_PIDS+=("$pid")
  done
}

collect_target_pids() {
  local pid_path pid runtime_path runtime_pid

  TARGET_PIDS=()

  mapfile -t MATCHED_PIDS < <(pgrep -f "$PATTERN" || true)
  for pid in "${MATCHED_PIDS[@]}"; do
    add_pid "$pid"
  done

  for pid_path in "$ROOT_DIR"/runtime/api/*.pid "$ROOT_DIR"/runtime/workers/*.pid; do
    [[ -f "$pid_path" ]] || continue
    if is_protected_runtime_path "$pid_path"; then
      continue
    fi
    pid="$(cat "$pid_path" 2>/dev/null || true)"
    add_pid "$pid"
  done

  for runtime_path in "$ROOT_DIR"/runtime/workers/*.json; do
    [[ -f "$runtime_path" ]] || continue
    if is_protected_runtime_path "$runtime_path"; then
      continue
    fi
    runtime_pid="$(read_json_pid "$runtime_path")"
    add_pid "$runtime_pid"
  done

  unique_pids "${TARGET_PIDS[@]}"
  TARGET_PIDS=("${UNIQUE_PIDS[@]}")
}

expand_process_groups() {
  local pid pgid member
  EXPANDED_PIDS=("${TARGET_PIDS[@]}")
  for pid in "${TARGET_PIDS[@]}"; do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] || continue
    mapfile -t GROUP_MEMBERS < <(ps -eo pid=,pgid= | awk -v pgid="$pgid" '$2 == pgid {print $1}')
    for member in "${GROUP_MEMBERS[@]}"; do
      add_process_group_pid "$member"
    done
  done
  unique_pids "${EXPANDED_PIDS[@]}" "${TARGET_PIDS[@]}"
  TARGET_PIDS=("${UNIQUE_PIDS[@]}")
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
    --no-start)
      START_AFTER_STOP=0
      shift
      ;;
    --timeout)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "[stop_online_backend_processes] --timeout requires a non-negative integer" >&2
        exit 2
      fi
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[stop_online_backend_processes] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[stop_online_backend_processes] Matching process command pattern:"
echo "  ${PATTERN}"
echo

if ! pgrep -af "$PATTERN"; then
  echo "[stop_online_backend_processes] No matching processes found."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[stop_online_backend_processes] Dry run only; services were not restarted."
    exit 0
  fi
fi

collect_target_pids
expand_process_groups

if [[ "${#TARGET_PIDS[@]}" -eq 0 ]]; then
  echo "[stop_online_backend_processes] No live target PIDs found after filtering."
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[stop_online_backend_processes] Dry run only; services were not restarted."
    exit 0
  fi
  start_services
fi

echo
echo "[stop_online_backend_processes] Target PIDs: ${TARGET_PIDS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[stop_online_backend_processes] Dry run only; no processes were stopped and services were not restarted."
  exit 0
fi

echo "[stop_online_backend_processes] Sending SIGTERM with kill..."
kill "${TARGET_PIDS[@]}" 2>/dev/null || true

if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
  sleep "$TIMEOUT_SECONDS"
fi

STILL_RUNNING=()
for pid in "${TARGET_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null && [[ "$(ps -o stat= -p "$pid" 2>/dev/null || true)" != Z* ]]; then
    STILL_RUNNING+=("$pid")
  fi
done

if [[ "${#STILL_RUNNING[@]}" -eq 0 ]]; then
  echo "[stop_online_backend_processes] All matched processes stopped."
  start_services
fi

echo "[stop_online_backend_processes] Still running after ${TIMEOUT_SECONDS}s: ${STILL_RUNNING[*]}"

if [[ "$FORCE" != "1" ]]; then
  echo "[stop_online_backend_processes] Re-run with --force to send SIGKILL to remaining processes."
  exit 1
fi

echo "[stop_online_backend_processes] Sending SIGKILL to remaining processes..."
kill -9 "${STILL_RUNNING[@]}" 2>/dev/null || true

FINAL_RUNNING=()
for pid in "${STILL_RUNNING[@]}"; do
  if kill -0 "$pid" 2>/dev/null && [[ "$(ps -o stat= -p "$pid" 2>/dev/null || true)" != Z* ]]; then
    FINAL_RUNNING+=("$pid")
  fi
done

if [[ "${#FINAL_RUNNING[@]}" -eq 0 ]]; then
  echo "[stop_online_backend_processes] All remaining processes were force-stopped."
  start_services
fi

echo "[stop_online_backend_processes] Failed to stop PIDs: ${FINAL_RUNNING[*]}" >&2
exit 1
