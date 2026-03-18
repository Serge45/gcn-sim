#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

// ----------------------------------------------------------------------------
// gemm_mfma: C = A * B  (row-major, float32, no transpose)
//
// Uses v_mfma_f32_4x4x1f32 with correct CDNA2 cross-lane outer-product
// semantics.  One 4-thread workgroup computes one 4x4 tile of C.
//
// 4x4x1 lane layout:
//   - 4 lanes form one block; lane t provides A[row=t] and B[col=t].
//   - The MFMA cross-lane outer product yields:
//       acc[gpr] for lane t = sum_k A[t][k] * B[k][gpr]
//   - Each lane writes 4 output elements: C[row=t][col=gpr] for gpr=0..3.
//
// Matrix dimensions:
//   A: M x K  (row-major)
//   B: K x N  (row-major)
//   C: M x N  (row-major), output is accumulated (C += A*B)
//
// Launch: grid=(N/4, M/4, 1) workgroups * block=(4, 1, 1) threads
//   => total grid threads: (N/4 * 4, M/4 * 1, 1)
// ----------------------------------------------------------------------------

using float32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

extern "C" __global__
void gemm_mfma_f32_4x4x1(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ C,
                          int M, int N, int K)
{
    // One workgroup = 4 threads = 1 MFMA block, covers one 4x4 tile of C.
    int wg_row = blockIdx.y;  // which 4-row tile
    int wg_col = blockIdx.x;  // which 4-col tile
    int lane   = threadIdx.x; // 0..3; lane t owns row=t and col=t

    // row = lane % 4 = lane (for a single 4-lane block)
    int c_row = wg_row * 4 + lane;  // global row this lane computes
    // col is determined by GPR index after MFMA; lane also owns column lane
    // for B reads: B[k][lane] is this lane's B column element.

    // Load existing C tile into accumulator.
    // acc[gpr] will hold C[c_row][wg_col*4 + gpr] after the MFMA loop.
    float32x4_t acc = {0.f, 0.f, 0.f, 0.f};
    if (c_row < M) {
        for (int g = 0; g < 4; g++) {
            int c_col = wg_col * 4 + g;
            if (c_col < N)
                acc[g] = C[c_row * N + c_col];
        }
    }

    // Accumulate across K dimension.
    // Each lane loads its own A row element and B column element.
    // MFMA cross-lane outer product: acc[gpr] += A[lane] * B[gpr]
    // (B[gpr] is src1 from lane gpr, read cross-lane by the hardware.)
    for (int k = 0; k < K; k++) {
        float a_val = (c_row < M)              ? A[c_row * K + k]          : 0.f;
        float b_val = (wg_col * 4 + lane < N) ? B[k * N + wg_col * 4 + lane] : 0.f;
        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, acc, 0, 0, 0);
    }

    // Write all 4 output columns for this lane's row back to C.
    if (c_row < M) {
        for (int g = 0; g < 4; g++) {
            int c_col = wg_col * 4 + g;
            if (c_col < N)
                C[c_row * N + c_col] = acc[g];
        }
    }
}
