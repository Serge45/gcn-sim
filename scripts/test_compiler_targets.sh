#!/usr/bin/env bash
# Verify compiler support for gfx950 and gfx1200
set -euo pipefail

CLANG="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang"
LLVM_OBJDUMP="${ROCM_PATH:-/opt/rocm}/llvm/bin/llvm-objdump"
LLVM_READELF="${ROCM_PATH:-/opt/rocm}/llvm/bin/llvm-readelf"

if [ ! -x "$CLANG" ]; then
    echo "[-] Error: clang not found at $CLANG"
    exit 1
fi

echo "=== Compiler Version ==="
"$CLANG" --version | head -n 2

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo ""
echo "=== Testing gfx950 (CDNA4 / MI-350) Target ==="
cat << 'EOF' > "$TMP_DIR/kernel_gfx950.s"
    .text
    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
    .globl test_gfx950_kernel
    .p2align 8
    .type test_gfx950_kernel,@function
test_gfx950_kernel:
    s_load_dword s0, s[0:1], 0x0
    v_prng_b32 v0, v0
    s_waitcnt lgkmcnt(0)
    s_endpgm
.Lfunc_end0:
    .size test_gfx950_kernel, .Lfunc_end0 - test_gfx950_kernel
EOF

if "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx950 -x assembler "$TMP_DIR/kernel_gfx950.s" -c -o "$TMP_DIR/gfx950.o"; then
    echo "[+] gfx950 assembly assembled successfully!"
    "$LLVM_READELF" -h "$TMP_DIR/gfx950.o" | grep -E 'Class:|Machine:|Flags:'
    echo "[+] Disassembly:"
    "$LLVM_OBJDUMP" -d "$TMP_DIR/gfx950.o"
else
    echo "[-] gfx950 compilation failed"
fi

echo ""
echo "=== Testing gfx1200 (RDNA4) Target ==="
cat << 'EOF' > "$TMP_DIR/kernel_gfx1200.s"
    .text
    .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
    .globl test_gfx1200_kernel
    .p2align 8
    .type test_gfx1200_kernel,@function
test_gfx1200_kernel:
    s_load_b32 s0, s[0:1], 0x0
    s_wait_kmcnt 0
    s_endpgm
.Lfunc_end1:
    .size test_gfx1200_kernel, .Lfunc_end1 - test_gfx1200_kernel
EOF

if "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1200 -x assembler "$TMP_DIR/kernel_gfx1200.s" -c -o "$TMP_DIR/gfx1200.o"; then
    echo "[+] gfx1200 assembly assembled successfully!"
    "$LLVM_READELF" -h "$TMP_DIR/gfx1200.o" | grep -E 'Class:|Machine:|Flags:'
    echo "[+] Disassembly:"
    "$LLVM_OBJDUMP" -d "$TMP_DIR/gfx1200.o"
else
    echo "[-] gfx1200 compilation failed"
fi
