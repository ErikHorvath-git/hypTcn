#!/usr/bin/env bash
# collect_for_training/scripts/collect_normal.sh
#
# Collects normal VM behavior frames for the hypTcn training dataset.
#
# Usage:
#   ./collect_for_training/scripts/collect_normal.sh [VM] [DURATION_SEC] [INTERVAL_MS] [SYSMAP]
#
# Defaults:
#   VM           = hyptcn-guest
#   DURATION_SEC = 300  (5 minutes)
#   INTERVAL_MS  = 200
#   SYSMAP       = configs/hyptcn-guest.sysmap  (if it exists)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VM="${1:-hyptcn-guest}"
DURATION="${2:-300}"
INTERVAL="${3:-200}"
SYSMAP="${4:-}"

# Auto-detect System.map if not specified.
if [ -z "$SYSMAP" ] && [ -f "configs/hyptcn-guest.sysmap" ]; then
    SYSMAP="configs/hyptcn-guest.sysmap"
fi

echo "[collect_normal] VM=$VM  duration=${DURATION}s  interval=${INTERVAL}ms"
[ -n "$SYSMAP" ] && echo "[collect_normal] sysmap=$SYSMAP"

# ── Start Python analyzer if not already running ───────────────────────────────
ANALYZER_PID=""
if ! pgrep -f "python.*analyzer.py" > /dev/null 2>&1; then
    echo "[collect_normal] Starting Python analyzer in background..."
    cd python
    ../".venv/bin/python" analyzer.py --socket /tmp/hyptcn.sock &
    ANALYZER_PID=$!
    cd "$REPO_ROOT"
    sleep 2
    echo "[collect_normal] Analyzer started (PID $ANALYZER_PID)"
fi

cleanup() {
    if [ -n "$ANALYZER_PID" ]; then
        kill "$ANALYZER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Build binary if needed ─────────────────────────────────────────────────────
if [ ! -f "bin/hyptcn" ]; then
    echo "[collect_normal] Building hyptcn..."
    make
fi

# ── Run collection ─────────────────────────────────────────────────────────────
SYSMAP_FLAG=()
[ -n "$SYSMAP" ] && SYSMAP_FLAG=(--sysmap "$SYSMAP")

echo "[collect_normal] Starting normal behavior collection..."
./bin/hyptcn \
    --vm "$VM" \
    --interval "$INTERVAL" \
    --collect \
    --collect-label normal \
    --collect-duration "$DURATION" \
    --collect-dir collect_for_training/ \
    "${SYSMAP_FLAG[@]+"${SYSMAP_FLAG[@]}"}"

# ── Report ─────────────────────────────────────────────────────────────────────
COUNT=$(find collect_for_training/normal/ -name '*.bin' 2>/dev/null | wc -l)
echo ""
echo "[collect_normal] Done. Total frames in collect_for_training/normal/: $COUNT"
echo "[collect_normal] Run 'collect_for_training/scripts/dataset_stats.sh' for full stats."
