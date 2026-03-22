#!/usr/bin/env bash
set -e

TMP_IMG="/tmp/hyptcn-guest-tmp.qcow2"

echo "[mount_vm] Copying image to /tmp (world-accessible)..."
cp /home/eh/vms/hyptcn-guest.qcow2 "$TMP_IMG"
chmod 666 "$TMP_IMG"

echo "[mount_vm] Resetting root password..."
LIBGUESTFS_BACKEND=direct guestfish --rw \
    -a "$TMP_IMG" \
    run : \
    mount /dev/sda1 / : \
    sh "echo 'root:hyptcn' | chpasswd"

echo "[mount_vm] Copying back..."
cp "$TMP_IMG" /home/eh/vms/hyptcn-guest.qcow2
rm "$TMP_IMG"

echo ""
echo "Done. Password is now: hyptcn"
echo "Start VM:  virsh --connect qemu:///session start hyptcn-guest"
