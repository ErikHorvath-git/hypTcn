#!/usr/bin/env bash
# tools/get_sysmap.sh — copy System.map from a KVM guest to the host.
#
# Usage:
#   ./tools/get_sysmap.sh [GUEST_NAME]
#
# GUEST_NAME defaults to "hyptcn-guest".  The script tries SSH first
# (root@<guest>), then resolves the guest IP via 'virsh domifaddr' and
# retries SSH to that IP.
#
# The System.map is saved to configs/hyptcn-guest.sysmap so that the
# Go scanner can be invoked with:
#   ./bin/hyptcn --sysmap configs/hyptcn-guest.sysmap --vm hyptcn-guest
#
set -euo pipefail

GUEST="${1:-hyptcn-guest}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_DEST="${REPO_ROOT}/configs/hyptcn-guest.sysmap"

mkdir -p "$(dirname "$HOST_DEST")"

SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

# ── helper: copy via SSH given a hostname or IP ────────────────────────────────
_copy_via_ssh() {
    local target="$1"
    echo "[get_sysmap] connecting to ${target} …"
    local kernel_ver
    kernel_ver=$(ssh ${SSH_OPTS} "root@${target}" "uname -r" 2>/dev/null) || return 1
    local remote_path="/boot/System.map-${kernel_ver}"
    echo "[get_sysmap] copying ${remote_path} from ${target} …"
    scp ${SSH_OPTS} "root@${target}:${remote_path}" "${HOST_DEST}"
    return 0
}

# ── attempt 1: SSH directly to the guest domain name ──────────────────────────
if _copy_via_ssh "${GUEST}"; then
    echo "[get_sysmap] saved → ${HOST_DEST}"
    exit 0
fi

echo "[get_sysmap] direct SSH to '${GUEST}' failed; resolving IP via virsh …"

# ── attempt 2: resolve guest IP via virsh domifaddr ───────────────────────────
GUEST_IP=$(virsh domifaddr "${GUEST}" 2>/dev/null \
    | awk '/ipv4/{print $4}' \
    | cut -d/ -f1 \
    | head -1)

if [ -z "${GUEST_IP}" ]; then
    echo "[get_sysmap] ERROR: could not resolve IP for '${GUEST}' via virsh." >&2
    echo "             Start the guest and ensure it has network connectivity." >&2
    exit 1
fi

echo "[get_sysmap] guest IP: ${GUEST_IP}"

if _copy_via_ssh "${GUEST_IP}"; then
    echo "[get_sysmap] saved → ${HOST_DEST}"
    exit 0
fi

echo "[get_sysmap] ERROR: SSH to ${GUEST_IP} also failed." >&2
echo "             Ensure the guest is running and root SSH is enabled." >&2
exit 1
