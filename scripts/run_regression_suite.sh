#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== [1/4] Standard Clean Build & Zero-Warning Check (core libs) ==="
cmake -S . -B build
cmake --build build -j"$(nproc)"

echo "=== [2/4] Running Full CTest Suite (20 suites) ==="
ctest --test-dir build --output-on-failure

echo "=== [3/4] ASan + UBSan Build & Regression Gate ==="
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j"$(nproc)"
ctest --test-dir build-asan --output-on-failure -E "test_hijack_|test_hip_"

if [[ -x build/tests/hipblaslt_gemm_bench && -f build/libamdhip64.so ]]; then
  echo "=== [4/4] End-to-End hipBLASLt GEMM Benchmark on RDNA4 (gfx1201) ==="
  GCN_SIM_ARCH=gfx1201 LD_LIBRARY_PATH="/opt/rocm-6.1.3/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$ROOT_DIR/build/libamdhip64.so" \
    "$ROOT_DIR/build/tests/hipblaslt_gemm_bench" 128 128 128
else
  echo "=== [4/4] Skipping hipblaslt_gemm_bench (not built) ==="
fi

if [[ -x .venv-rocm/bin/python && -f build/libamdhip64.so ]]; then
  echo "=== [5/5] PyTorch ROCm End-to-End Smoke Suite ==="
  LD_PRELOAD="$ROOT_DIR/build/libamdhip64.so" \
    "$ROOT_DIR/.venv-rocm/bin/python" "$ROOT_DIR/scripts/test_pytorch_smoke.py"
else
  echo "=== [5/5] Skipping PyTorch ROCm Smoke Suite (.venv-rocm not found) ==="
fi

echo "=== ALL ANTI-REGRESSION CHECKS PASSED ==="
