#!/usr/bin/env bash
#
# Grow swap from 2 GiB to 16 GiB and stop the kernel swapping out the desktop
# before it has to.
#
#   sudo ./scripts/setup_swap.sh
#
# Why this is not optional on this machine: a pick_place_milk_human run puts the
# learner at ~5.5 GiB of image buffers, and a browser plus VSCode were measured
# at ~18 GiB alongside it.  That fits in 30 GiB only just, and when it does not,
# the pages the kernel evicts are gnome-shell's -- the trainer keeps running
# while the *desktop* stalls, which is why this reads as "the machine froze"
# rather than as a training crash.  2 GiB of swap gives that no room to absorb a
# transient; it was measured 100% full with 1.07M pages swapped out.
#
# swappiness is lowered because the default 60 tells the kernel to evict anonymous
# pages (the desktop's) roughly as eagerly as it drops file cache.  The trainer
# writes buffer dumps at ~700 MB per 5000 steps, so there is always plenty of
# reclaimable page cache -- prefer dropping that over paging out an interactive UI.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

SWAPFILE=/swapfile
SIZE_GB=16

echo ">>> current swap:"
swapon --show || true
free -h | sed -n '1p;3p'

# A swapfile cannot be resized in place while in use.
if swapon --show=NAME --noheadings | grep -qx "${SWAPFILE}"; then
    echo ">>> disabling ${SWAPFILE} (this can take a minute if it is full --"
    echo "    every swapped-out page has to be read back into RAM first)"
    swapoff "${SWAPFILE}"
fi

echo ">>> allocating ${SIZE_GB}G at ${SWAPFILE}"
rm -f "${SWAPFILE}"
# fallocate leaves holes on some filesystems, which swapon rejects; dd is slower
# but always produces a file swapon accepts.
if ! fallocate -l "${SIZE_GB}G" "${SWAPFILE}" 2>/dev/null; then
    dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$((SIZE_GB * 1024)) status=progress
fi
chmod 600 "${SWAPFILE}"
mkswap "${SWAPFILE}"
swapon "${SWAPFILE}"

# Persist across reboots.
if ! grep -qE "^${SWAPFILE}[[:space:]]" /etc/fstab; then
    echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    echo ">>> added ${SWAPFILE} to /etc/fstab"
fi

# Prefer reclaiming the trainer's page cache over paging out the desktop.
sysctl -w vm.swappiness=10
if [[ -f /etc/sysctl.d/99-swappiness.conf ]]; then
    sed -i 's/^vm.swappiness.*/vm.swappiness = 10/' /etc/sysctl.d/99-swappiness.conf
else
    echo "vm.swappiness = 10" > /etc/sysctl.d/99-swappiness.conf
fi

echo
echo ">>> done:"
swapon --show
free -h | sed -n '1p;3p'
