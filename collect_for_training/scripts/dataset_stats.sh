#!/usr/bin/env bash
# collect_for_training/scripts/dataset_stats.sh
#
# Print frame counts and estimated training sequences per label.
#
# Usage:
#   ./collect_for_training/scripts/dataset_stats.sh [COLLECT_DIR]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

COLLECT_DIR="${1:-collect_for_training}"

SEQUENCE_WINDOW=16
SEQUENCE_STRIDE=8

LABELS=(normal malware shellcode rootkit cryptominer ransomware)

echo "================================================================"
echo " hypTcn Dataset Statistics"
echo " Directory: $COLLECT_DIR/"
echo " Window=${SEQUENCE_WINDOW}  Stride=${SEQUENCE_STRIDE}"
echo "================================================================"
printf "  %-14s  %8s  %10s  %s\n" "Label" "Frames" "Sequences" "Status"
echo "  ──────────────────────────────────────────────────────────"

TOTAL_FRAMES=0
TOTAL_SEQS=0

for LABEL in "${LABELS[@]}"; do
    DIR="$COLLECT_DIR/$LABEL"
    if [ ! -d "$DIR" ]; then
        printf "  %-14s  %8s  %10s  %s\n" "$LABEL" "0" "0" "missing"
        continue
    fi

    COUNT=$(find "$DIR" -maxdepth 1 -name '*.bin' | wc -l)

    if [ "$COUNT" -ge "$SEQUENCE_WINDOW" ]; then
        SEQS=$(( (COUNT - SEQUENCE_WINDOW) / SEQUENCE_STRIDE + 1 ))
    else
        SEQS=0
    fi

    # Minimum thresholds (from README)
    case "$LABEL" in
        normal)      MIN=3000 ;;
        *)           MIN=1500 ;;
    esac

    if [ "$COUNT" -eq 0 ]; then
        STATUS="empty"
    elif [ "$COUNT" -lt "$MIN" ]; then
        STATUS="low (need $MIN)"
    else
        STATUS="ok"
    fi

    printf "  %-14s  %8d  %10d  %s\n" "$LABEL" "$COUNT" "$SEQS" "$STATUS"
    TOTAL_FRAMES=$(( TOTAL_FRAMES + COUNT ))
    TOTAL_SEQS=$(( TOTAL_SEQS + SEQS ))
done

echo "  ──────────────────────────────────────────────────────────"
printf "  %-14s  %8d  %10d\n" "TOTAL" "$TOTAL_FRAMES" "$TOTAL_SEQS"
echo ""

# Disk usage
if command -v du &>/dev/null; then
    DISK=$(du -sh "$COLLECT_DIR" 2>/dev/null | cut -f1)
    echo "  Disk usage: $DISK"
fi

echo ""
echo " Next step:  ./collect_for_training/scripts/quick_train.sh"
echo "================================================================"
