#!/usr/bin/env bash
# Delete all data for a given parent session on the backend server.
# Usage: ./cleanup_session.sh <session_id> [--dry-run] [--force]
#   --dry-run  List what would be deleted without actually deleting
#   --force    Skip confirmation prompt

set -euo pipefail

usage() {
    echo "Usage: $0 <session_id> [--dry-run] [--force]"
    echo "  session_id  Parent session ID to delete (e.g. 3d376dc97da2)"
    echo "  --dry-run   List what would be deleted without actually deleting"
    echo "  --force     Skip confirmation prompt"
    exit 1
}

# Parse arguments
SESSION_ID=""
DRY_RUN=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
        --help|-h) usage ;;
        *)
            if [ -z "$SESSION_ID" ]; then
                SESSION_ID="$arg"
            else
                echo "Error: unexpected extra argument '$arg'"
                usage
            fi
            ;;
    esac
done

if [ -z "$SESSION_ID" ]; then
    echo "Error: missing session_id"
    usage
fi

if [[ ! "$SESSION_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Error: invalid session_id '$SESSION_ID'"
    echo "Session IDs may only contain letters, numbers, '-' and '_'."
    exit 1
fi

# Backend project root
PROJECT_ROOT="${PROJECT_ROOT:-/zjunlp/chenyijun/worldmm-online-server-release}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: '$PYTHON_BIN' is required to inspect Rokid parent/child session metadata."
    echo "Set PYTHON_BIN=/path/to/python if python3 is not on PATH."
    exit 1
fi

echo "============================================"
echo "  Session Cleanup"
echo "  Parent Session ID: $SESSION_ID"
echo "  Server path: $PROJECT_ROOT"
if [ "$DRY_RUN" = true ]; then
    echo "  Mode: DRY-RUN (list only, no deletion)"
fi
echo "============================================"
echo ""

# ---- Helpers ----
declare -a SESSION_IDS=()
declare -a TO_DELETE=()

add_session_id() {
    local sid="$1"
    if [[ ! "$sid" =~ ^[A-Za-z0-9_-]+$ ]]; then
        return
    fi
    for existing in "${SESSION_IDS[@]}"; do
        if [ "$existing" = "$sid" ]; then
            return
        fi
    done
    SESSION_IDS+=("$sid")
}

add_delete_item() {
    local item="$1"
    if [ ! -e "$item" ] && [ ! -L "$item" ]; then
        return
    fi
    for existing in "${TO_DELETE[@]}"; do
        if [ "$existing" = "$item" ]; then
            return
        fi
    done
    TO_DELETE+=("$item")
}

json_matches_any_session_id() {
    local path="$1"
    shift
    "$PYTHON_BIN" -c '
import json
import sys

path = sys.argv[1]
session_ids = set(sys.argv[2:])
if not session_ids:
    sys.exit(1)

try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)

if not isinstance(payload, dict):
    sys.exit(1)

for field in ("session_id", "parent_session_id", "child_session_id", "active_session_id"):
    value = str(payload.get(field) or "").strip()
    if value in session_ids:
        sys.exit(0)

sys.exit(1)
' "$path" "$@"
}

ONLINE_SESSIONS_DIR="$PROJECT_ROOT/online_sessions"
PARENT_SESSION_DIR="$ONLINE_SESSIONS_DIR/$SESSION_ID"

# ---- Collect related parent/child sessions ----
# A Rokid parent session stores the cross-day memory. Each child session stores
# one Rokid DAY run, normally named {parent_session_id}__day0001.
add_session_id "$SESSION_ID"

DAY_STATE_PATH="$PARENT_SESSION_DIR/stream/day_state.json"
if [ -f "$DAY_STATE_PATH" ]; then
    while IFS= read -r child_session_id; do
        add_session_id "$child_session_id"
    done < <("$PYTHON_BIN" - "$DAY_STATE_PATH" <<'PY'
import json
import re
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(0)

if not isinstance(payload, dict):
    sys.exit(0)

runs = payload.get("runs")
if not isinstance(runs, dict):
    sys.exit(0)

seen = set()
for run in runs.values():
    if not isinstance(run, dict):
        continue
    child_id = str(run.get("child_session_id") or "").strip()
    if child_id and re.fullmatch(r"[A-Za-z0-9_-]+", child_id) and child_id not in seen:
        seen.add(child_id)
        print(child_id)
PY
)
fi

# Fallback for missing/stale day_state.json: scan conventional child dirs.
if [ -d "$ONLINE_SESSIONS_DIR" ]; then
    shopt -s nullglob
    for child_dir in "$ONLINE_SESSIONS_DIR/${SESSION_ID}__day"*; do
        if [ -d "$child_dir" ]; then
            add_session_id "$(basename "$child_dir")"
        fi
    done
    shopt -u nullglob
fi

echo "  Sessions to clean: ${SESSION_IDS[*]}"
echo ""

# 1) Parent and child session directories.
for sid in "${SESSION_IDS[@]}"; do
    add_delete_item "$ONLINE_SESSIONS_DIR/$sid"
done

# 2) Task files across online_tasks and online_tasks_aborted.
# Most task files are named {session_id}_*.json. Rokid merge tasks also carry
# parent_session_id and child_session_id in JSON, so inspect JSON fields too.
declare -a TASK_BASES=(
    "$PROJECT_ROOT/online_tasks"
    "$PROJECT_ROOT/online_tasks_aborted"
)

for task_base in "${TASK_BASES[@]}"; do
    if [ -d "$task_base" ]; then
        for sid in "${SESSION_IDS[@]}"; do
            while IFS= read -r -d '' f; do
                add_delete_item "$f"
            done < <(find "$task_base" -type f -name "${sid}_*.json" -print0 2>/dev/null || true)
        done
        while IFS= read -r -d '' f; do
            if json_matches_any_session_id "$f" "${SESSION_IDS[@]}"; then
                add_delete_item "$f"
            fi
        done < <(find "$task_base" -type f -name "*.json" -print0 2>/dev/null || true)
    fi
done

# 3) Runtime active-session markers, only if they point at the parent or one of
# its child sessions.
for runtime_file in \
    "$PROJECT_ROOT/runtime/active_session.json" \
    "$PROJECT_ROOT/runtime/active_rokid_session.json"
do
    if [ -f "$runtime_file" ] && json_matches_any_session_id "$runtime_file" "${SESSION_IDS[@]}"; then
        add_delete_item "$runtime_file"
    fi
done

# ---- Nothing to delete ----
if [ ${#TO_DELETE[@]} -eq 0 ]; then
    echo "No data found for parent session '$SESSION_ID'. It may have already been cleaned up, or the session ID is incorrect."
    exit 0
fi

# ---- List items ----
echo "The following ${#TO_DELETE[@]} items will be deleted:"
echo "--------------------------------------------"
for item in "${TO_DELETE[@]}"; do
    if [ -f "$item" ]; then
        size=$(stat -c%s "$item" 2>/dev/null || echo "?")
        echo "  [file] $item (${size} bytes)"
    elif [ -d "$item" ]; then
        count=$(find "$item" -type f 2>/dev/null | wc -l)
        size=$(du -sh "$item" 2>/dev/null | cut -f1 || echo "?")
        echo "  [dir]  $item (${count} files, ${size})"
    else
        echo "  [missing] $item"
    fi
done
echo "--------------------------------------------"
echo ""

# ---- Dry-run stops here ----
if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] No deletions performed. Remove --dry-run to execute."
    exit 0
fi

# ---- Confirm ----
if [ "$FORCE" != true ]; then
    read -r -p "Delete all of the above? This is irreversible. Type 'yes' to proceed: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
fi

# ---- Execute deletion ----
echo ""
echo "Deleting..."
DELETED=0
FAILED=0

for item in "${TO_DELETE[@]}"; do
    if [ ! -e "$item" ] && [ ! -L "$item" ]; then
        echo "  [skip] $item (does not exist)"
        continue
    fi

    if rm -rf "$item" 2>/dev/null; then
        echo "  [deleted] $item"
        DELETED=$((DELETED + 1))
    else
        echo "  [failed] $item (permission denied?)"
        FAILED=$((FAILED + 1))
    fi
done

# Remove empty task subdirectories.
for task_base in "${TASK_BASES[@]}"; do
    if [ -d "$task_base" ]; then
        find "$task_base" -type d -empty -delete 2>/dev/null || true
    fi
done

echo ""
echo "============================================"
echo "  Done: $DELETED deleted, $FAILED failed"
echo "============================================"
