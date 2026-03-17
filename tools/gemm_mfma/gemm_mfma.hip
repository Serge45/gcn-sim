#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp16.h>

// ----------------------------------------------------------------------------
// gemm_mfma: C = A * B  (row-major, float32, no transpose)
//
// Uses v_mfma_f32_4x4x1f32 to compute a 16x16 tile of C per workgroup.
// The workgroup is 16 threads (one 4x4 MFMA block uses 16 lanes).
// Each thread accumulates one row-slice across the K dimension.
//
// Matrix dimensions:
//   A: M x K  (row-major)
//   B: K x N  (row-major)
//   C: M x N  (row-major), output is accumulated (C += A*B)
//
// Launch: grid=(N/16, M/16, 1)  block=(16, 1, 1)
// ----------------------------------------------------------------------------

using float32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

extern "C" __global__
void gemm_mfma_f32_4x4x1(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ C,
                          int M, int N, int K)
{
    // Each workgroup computes a 4x4 tile of C.
    // blockDim.x == 16; each warp of 16 lanes maps to one 4x4 MFMA output.
    int wg_row = blockIdx.y;  // which 4-row tile
    int wg_col = blockIdx.x;  // which 4-col tile
    int lane   = threadIdx.x; // 0..15

    // Accumulator: 4x4 output stored in accvgpr via MFMA
    // We represent it as float32x4_t (4 floats = one column of the 4x4 output per lane)
    float32x4_t acc = {0.f, 0.f, 0.f, 0.f};

    // Load existing C tile into acc so we accumulate
    int c_row = wg_row * 4 + (lane / 4);  // row index into C
    int c_col = wg_col * 4 + (lane % 4);  // col index into C
    if (c_row < M && c_col < N) {
        // MFMA 4x4x1: each lane holds one element of the 4x4 output block.
        // acc[0] corresponds to the single output element for this lane.
        acc[0] = C[c_row * N + c_col];
    }

    // Accumulate across K
    for (int k = 0; k < K; k++) {
        // Each lane of the 16-lane group reads A[row][k] and B[k][col]
        // For 4x4x1 MFMA: src0 = a scalar from A (one per lane),
        //                  src1 = a scalar from B (one per lane)
        int a_row = wg_row * 4 + (lane / 4);
        int b_col = wg_col * 4 + (lane % 4);

        float a_val = (a_row < M) ? A[a_row * K + k] : 0.f;
        float b_val = (b_col < N) ? B[k * N + b_col] : 0.f;

        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, acc, 0, 0, 0);
    }

    // Write accumulator back to C
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = acc[0];
    }
}
