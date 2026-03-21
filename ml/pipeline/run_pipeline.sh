#!/usr/bin/env bash
# run_pipeline.sh — full hypTcn ML pipeline in one command
#
# Usage:
#   ./ml/pipeline/run_pipeline.sh [options]
#
# Options:
#   --vm <name>               KVM domain name (omit for mock mode)
#   --collect-duration <sec>  Collection duration per label (default: 300)
#   --interval <ms>           Sampling interval in ms (default: 100)
#   --sysmap <path>           System.map path (enables OS-layer features)
#   --epochs <n>              Training epochs (default: 50)
#   --batch-size <n>          Batch size (default: 32)
#   --skip-collect            Skip collection (use existing ml/data/raw/ data)
#   --skip-train              Skip training (use existing ml/models/ weights)
#
# Example (mock mode, quick test):
#   ./ml/pipeline/run_pipeline.sh --collect-duration 30 --epochs 5
#
# Example (live VM, full pipeline):
#   ./ml/pipeline/run_pipeline.sh --vm hyptcn-guest --collect-duration 300

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
BINARY="${REPO_ROOT}/bin/hyptcn"
RAW_DIR="${REPO_ROOT}/ml/data/raw"
PROCESSED_DIR="${REPO_ROOT}/ml/data/processed"
ML_MODELS_DIR="${REPO_ROOT}/ml/models"

# ── defaults ──────────────────────────────────────────────────────────────────
VM_NAME=""
COLLECT_DURATION=300
INTERVAL_MS=100
SYSMAP=""
EPOCHS=50
BATCH_SIZE=32
SKIP_COLLECT=false
SKIP_TRAIN=false

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vm)               VM_NAME="$2";            shift 2 ;;
        --collect-duration) COLLECT_DURATION="$2";   shift 2 ;;
        --interval)         INTERVAL_MS="$2";        shift 2 ;;
        --sysmap)           SYSMAP="$2";             shift 2 ;;
        --epochs)           EPOCHS="$2";             shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";         shift 2 ;;
        --skip-collect)     SKIP_COLLECT=true;       shift   ;;
        --skip-train)       SKIP_TRAIN=true;         shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
section() { echo; echo "═══════════════════════════════════════════════"; echo "  $*"; echo "═══════════════════════════════════════════════"; }
step()    { echo; echo "── $* ──"; }

check_binary() {
    if [[ ! -f "${BINARY}" ]]; then
        echo "ERROR: ${BINARY} not found. Run 'make' first." >&2
        exit 1
    fi
}

check_python() {
    if [[ ! -f "${PYTHON}" ]]; then
        echo "ERROR: ${PYTHON} not found. Run 'make deps' first." >&2
        exit 1
    fi
}

# ── STEP 1: Collect ───────────────────────────────────────────────────────────
if [[ "${SKIP_COLLECT}" == "false" ]]; then
    section "STEP 1: Collecting labeled frames"
    check_binary

    MOCK_FLAG=""
    VM_FLAG=""
    SYSMAP_FLAG=""
    [[ -z "${VM_NAME}" ]]  && MOCK_FLAG="--mock"
    [[ -n "${VM_NAME}" ]]  && VM_FLAG="--vm ${VM_NAME}"
    [[ -n "${SYSMAP}" ]]   && SYSMAP_FLAG="--sysmap ${SYSMAP}"

    LABELS=(normal shellcode rootkit cryptominer ransomware)
    for LABEL in "${LABELS[@]}"; do
        step "Collecting label: ${LABEL} (${COLLECT_DURATION}s)"
        # shellcheck disable=SC2086
        "${BINARY}" \
            ${MOCK_FLAG} ${VM_FLAG} ${SYSMAP_FLAG} \
            --collect \
            --collect-label "${LABEL}" \
            --collect-duration "${COLLECT_DURATION}" \
            --collect-dir "${RAW_DIR}" \
            --interval "${INTERVAL_MS}" \
            --quiet
        COUNT=$(find "${RAW_DIR}/${LABEL}" -name "*.bin" 2>/dev/null | wc -l)
        echo "  → ${COUNT} frames in ${RAW_DIR}/${LABEL}/"
    done
else
    section "STEP 1: Skipped (--skip-collect)"
fi

# ── STEP 2: Preprocess ────────────────────────────────────────────────────────
section "STEP 2: Preprocessing raw frames → numpy arrays"
check_python
"${PYTHON}" "${REPO_ROOT}/ml/pipeline/preprocess.py" \
    --input "${RAW_DIR}" \
    --output "${PROCESSED_DIR}"

# ── STEP 3: Train ─────────────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN}" == "false" ]]; then
    section "STEP 3: Training TCN"
    "${PYTHON}" "${REPO_ROOT}/ml/pipeline/train.py" \
        --data "${PROCESSED_DIR}" \
        --output-dir "${ML_MODELS_DIR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}"
else
    section "STEP 3: Skipped (--skip-train)"
fi

# ── STEP 4: Evaluate ──────────────────────────────────────────────────────────
section "STEP 4: Evaluating model"
"${PYTHON}" "${REPO_ROOT}/ml/pipeline/evaluate.py" \
    --data "${PROCESSED_DIR}" \
    --model-dir "${ML_MODELS_DIR}" \
    --n-latency 500

# ── STEP 5: Export ────────────────────────────────────────────────────────────
section "STEP 5: Exporting weights to python/model/weights/"
"${PYTHON}" "${REPO_ROOT}/ml/pipeline/export.py" \
    --model-dir "${ML_MODELS_DIR}" \
    --weights-dir "${REPO_ROOT}/python/model/weights"

# ── Done ──────────────────────────────────────────────────────────────────────
section "Pipeline complete. Model ready."
echo "  Run: make && make python-service"
echo
