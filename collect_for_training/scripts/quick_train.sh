#!/usr/bin/env bash
# collect_for_training/scripts/quick_train.sh
#
# Train the TCN on frames collected by the collect_* scripts.
# Wraps training/train.py --data-source collect.
#
# Usage:
#   ./collect_for_training/scripts/quick_train.sh [COLLECT_DIR] [EPOCHS] [BATCH_SIZE]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

COLLECT_DIR="${1:-collect_for_training}"
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-32}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "================================================================"
echo " hypTcn Quick Train"
echo " Data:       $COLLECT_DIR/"
echo " Epochs:     $EPOCHS"
echo " Batch size: $BATCH_SIZE"
echo " Python:     $PYTHON"
echo "================================================================"
echo ""

# Show dataset stats first
bash "$SCRIPT_DIR/dataset_stats.sh" "$COLLECT_DIR"
echo ""

# Check there is at least some data
TOTAL=$(find "$COLLECT_DIR" -maxdepth 2 -name '*.bin' 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No .bin frames found in $COLLECT_DIR/" >&2
    echo "       Run collect_normal.sh and collect_malware.sh first." >&2
    exit 1
fi

echo "[train] Starting training on $TOTAL frames..."
echo ""

"$PYTHON" training/train.py \
    --data-source collect \
    --collect-dir "$COLLECT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-dir models/

echo ""
echo "[train] Done. Weights saved to models/tcn_weights.pt"
echo "[train] Config saved to models/config.json"
