#!/usr/bin/env bash
# Script to upgrade ROCm to 7.0 on Ubuntu 22.04 (Jammy) for GCN-Sim
# Requires root privileges (run with sudo).

set -euo pipefail

echo "=========================================================="
echo " Starting ROCm 7.0 (LLVM 20) Upgrade for GCN-Sim"
echo "=========================================================="

if [ "$(id -u)" -ne 0 ]; then
    echo "[-] Error: This script must be run with sudo or as root."
    echo "    Usage: sudo bash $0"
    exit 1
fi

ROCM_LIST="/etc/apt/sources.list.d/rocm.list"

echo "[1/4] Backing up existing repository configuration..."
if [ -f "$ROCM_LIST" ]; then
    cp "$ROCM_LIST" "${ROCM_LIST}.bak.$(date +%Y%m%d%H%M%S)"
fi

echo "[2/4] Configuring ROCm 7.0 repository in $ROCM_LIST..."
cat << 'EOF' > "$ROCM_LIST"
deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.0 jammy main
EOF

echo "[3/4] Updating APT package index..."
apt-get update -o Dir::Etc::sourcelist="sources.list.d/rocm.list" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" || apt-get update

echo "[4/4] Installing ROCm 7.0 LLVM compiler and HIP developer packages..."
apt-get install -y --no-install-recommends \
    rocm-core \
    rocm-llvm \
    rocm-device-libs \
    hip-dev \
    hip-runtime-amd

echo "----------------------------------------------------------"
echo " Verifying compiler installation..."
if [ -x "/opt/rocm-7.0.0/llvm/bin/clang" ]; then
    echo "[+] Found /opt/rocm-7.0.0/llvm/bin/clang"
    /opt/rocm-7.0.0/llvm/bin/clang --version | head -n 2
    
    # Update default /opt/rocm symlink if needed
    if [ -L "/opt/rocm" ] || [ ! -e "/opt/rocm" ]; then
        ln -snf /opt/rocm-7.0.0 /opt/rocm
        echo "[+] Updated /opt/rocm -> /opt/rocm-7.0.0"
    fi
elif [ -x "/opt/rocm/llvm/bin/clang" ]; then
    /opt/rocm/llvm/bin/clang --version | head -n 2
fi

echo "=========================================================="
echo "[+] ROCm 7.0 upgrade complete!"
echo "=========================================================="
