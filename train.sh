#!/usr/bin/env bash
# train.sh — preprocess data, train TCN, deploy weights
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PY="$REPO/.venv/bin/python"

echo "[1/3] Generating synthetic training data..."
"$PY" "$REPO/ml/pipeline/generate_data.py"

echo "[2/3] Training TCN..."
"$PY" "$REPO/ml/pipeline/train.py" \
    --data       "$REPO/ml/data/processed" \
    --output-dir "$REPO/ml/models" \
    --epochs 50

echo "[3/3] Deploying weights to python/model/weights/..."
mkdir -p "$REPO/python/model/weights"
cp "$REPO/ml/models/tcn_weights.pt" "$REPO/python/model/weights/tcn_weights.pt"
cp "$REPO/ml/models/tcn_weights.pt" "$REPO/models/tcn_weights.pt" 2>/dev/null || true

echo ""
echo "Done. Spusti pipeline: ./run.sh"
