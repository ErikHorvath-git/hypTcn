#!/usr/bin/env bash
# run.sh — spusti hypTcn pipeline (analyzer + scanner)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv/bin/python"
SOCKET="/tmp/hyptcn.sock"
VM="hyptcn-guest"
SYSMAP="$REPO/configs/hyptcn-guest.sysmap"
INTERVAL=500

echo "[run] Spúšťam Python analyzer..."
rm -f "$SOCKET"
"$VENV" "$REPO/python/analyzer.py" --socket "$SOCKET" &
ANALYZER_PID=$!
echo "[run] Analyzer PID: $ANALYZER_PID"

echo "[run] Čakám na socket..."
for i in $(seq 1 20); do
    [ -S "$SOCKET" ] && break
    sleep 0.5
done

if [ ! -S "$SOCKET" ]; then
    echo "[run] ERROR: socket sa nevytvoril"
    kill $ANALYZER_PID 2>/dev/null
    exit 1
fi

echo "[run] Spúšťam Go scanner (VM=$VM, interval=${INTERVAL}ms)..."
"$REPO/bin/hyptcn" --vm "$VM" --sysmap "$SYSMAP" --interval "$INTERVAL"

kill $ANALYZER_PID 2>/dev/null
