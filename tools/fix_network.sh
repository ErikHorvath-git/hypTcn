#!/usr/bin/env bash
# tools/fix_network.sh — fix libvirt default network when virbr0 is orphaned
# Run as: bash tools/fix_network.sh
set -euo pipefail

GUEST="${1:-hyptcn-guest}"

echo "[fix_network] Stopping VM '$GUEST'..."
virsh destroy "$GUEST" 2>/dev/null || true

echo "[fix_network] Waiting for tap0 to be released..."
sleep 2

echo "[fix_network] Removing orphaned virbr0 bridge..."
if ip link show virbr0 &>/dev/null; then
    sudo ip link set virbr0 down 2>/dev/null || true
    sudo ip link delete virbr0 2>/dev/null || true
    echo "[fix_network] virbr0 removed."
else
    echo "[fix_network] virbr0 not found, skipping."
fi

echo "[fix_network] Starting libvirt default network..."
virsh net-start default

echo "[fix_network] Starting VM '$GUEST'..."
virsh start "$GUEST"

echo "[fix_network] Waiting for DHCP lease (up to 60s)..."
for i in $(seq 1 30); do
    IP=$(virsh domifaddr "$GUEST" 2>/dev/null | awk '/ipv4/{print $4}' | cut -d/ -f1 | head -1)
    if [ -n "$IP" ]; then
        echo "[fix_network] VM got IP: $IP"
        echo ""
        echo "Now run:"
        echo "  ./tools/get_sysmap.sh $GUEST"
        exit 0
    fi
    sleep 2
done

echo "[fix_network] ERROR: VM did not get an IP in 60s."
echo "  Try: virsh console $GUEST   (login: root / hyptcn)"
echo "  Then inside VM: dhclient -v"
exit 1
