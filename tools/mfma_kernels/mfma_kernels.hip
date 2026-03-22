#include <hip/hip_runtime.h>

// ============================================================================
// MFMA GEMM kernels for testing all F32 MFMA variants.
// C = A * B  (row-major, float32, C is zeroed before launch).
//
// Lane layout (CDNA2):
//
//   16x16x1  — 16-thread WG (1 block of 16 lanes), 16 output AGPRs/lane
//     acc[gpr] at lane t = D[t][gpr]
//     WG writes C[wg_row*16+t][wg_col*16+gpr] for gpr=0..15
//
//   16x16x4  — 64-thread WG (4 K-groups of 16 lanes), 4 output AGPRs/lane
//     acc[gpr] at lane l = D[(l/16)*4+gpr][l%16]
//     One MFMA call covers K-indices k_base..k_base+3 (K must be multiple of 4)
//
//   32x32x1  — 64-thread WG (2 blocks of 32 lanes), 16 output AGPRs/lane
//     acc[gpr] at lane l = D[(l/32)*16+gpr][l%32]
//     src0 used: lanes with position < 16 within the block (t%16 = row within block)
//
//   32x32x2  — 64-thread WG (2 K-groups of 32 lanes), 16 output AGPRs/lane
//     acc[gpr] at lane l = D[(l/32)*16+gpr][l%32]
//     One MFMA call covers K-indices k_base, k_base+1 (K must be multiple of 2)
// ============================================================================

using float32x4_t  = __attribute__((__vector_size__( 4 * sizeof(float)))) float;
using float32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32_t = __attribute__((__vector_size__(32 * sizeof(float)))) float;

// ----------------------------------------------------------------------------
// 16x16x1: 16-thread workgroup, one 4x4 MFMA block, 16 output AGPRs per lane.
// Grid: block=(16,1,1), total_x=(N/16)*16, total_y=M/16
// ----------------------------------------------------------------------------
extern "C" __global__
void mfma_gemm_16x16x1f32(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K)
{
    int wg_row = blockIdx.y;
    int wg_col = blockIdx.x;
    int lane   = threadIdx.x;  // 0..15

    int c_row = wg_row * 16 + lane;

    float32x16_t acc = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int k = 0; k < K; k++) {
        float a = (c_row < M) ? A[c_row * K + k] : 0.f;
        float b = (wg_col * 16 + lane < N) ? B[k * N + wg_col * 16 + lane] : 0.f;
        acc = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, acc, 0, 0, 0);
    }

    // acc[gpr] = C[c_row][wg_col*16+gpr]
    if (c_row < M) {
        for (int g = 0; g < 16; g++) {
            int c_col = wg_col * 16 + g;
            if (c_col < N)
                C[c_row * N + c_col] = acc[g];
        }
    }
}

// ----------------------------------------------------------------------------
// 16x16x4: 64-thread workgroup, K=4 per MFMA call, 4 output AGPRs per lane.
// Grid: block=(64,1,1), total_x=(N/16)*64, total_y=M/16
// K must be a multiple of 4.
// ----------------------------------------------------------------------------
extern "C" __global__
void mfma_gemm_16x16x4f32(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K)
{
    int wg_row  = blockIdx.y;
    int wg_col  = blockIdx.x;
    int lane    = threadIdx.x;  // 0..63
    int kg_out  = lane / 16;    // K-group (determines output row block)
    int t       = lane % 16;    // position within K-group (= row and col index)

    float32x4_t acc = {0.f, 0.f, 0.f, 0.f};

    // K loop in steps of 4; each call covers K-indices k_base..k_base+3.
    // K-group kg within this call: K-index = k_base + kg.
    // Lane (kg*16+t): src0 = A[t][k_base+kg], src1 = B[k_base+kg][t]
    for (int k_base = 0; k_base < K; k_base += 4) {
        int kg = kg_out;               // this lane's K-group
        int k_idx = k_base + kg;
        int a_row = wg_row * 16 + t;
        int b_col = wg_col * 16 + t;

        float a = (a_row < M && k_idx < K) ? A[a_row * K + k_idx] : 0.f;
        float b = (k_idx < K && b_col < N) ? B[k_idx * N + b_col] : 0.f;
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
    }

    // acc[gpr] at lane l = D[(l/16)*4+gpr][l%16]
    int c_col = wg_col * 16 + t;
    for (int g = 0; g < 4; g++) {
        int c_row = wg_row * 16 + kg_out * 4 + g;
        if (c_row < M && c_col < N)
            C[c_row * N + c_col] = acc[g];
    }
}

// ----------------------------------------------------------------------------
// 32x32x1: 64-thread workgroup (2 blocks of 32 lanes), 16 output AGPRs/lane.
// Grid: block=(64,1,1), total_x=(N/32)*64, total_y=M/32
// ----------------------------------------------------------------------------
extern "C" __global__
void mfma_gemm_32x32x1f32(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K)
{
    int wg_row = blockIdx.y;
    int wg_col = blockIdx.x;
    int lane   = threadIdx.x;  // 0..63
    int b      = lane / 32;    // block (0 or 1)
    int t      = lane % 32;    // position within block

    // 32x32x1 accumulator: the intrinsic signature requires 32 elements on this
    // compiler version, but only acc[0..15] carry meaningful output per lane.
    // Each block independently holds its 16 output rows in acc[0..15]:
    //   block 0 (lanes 0..31): acc[g] = D[g][t]        (rows 0..15)
    //   block 1 (lanes 32..63): acc[g] = D[16+g][t]    (rows 16..31)
    float32x32_t acc = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // Lane (b*32+t): positions t=0..15 provide A[b*16+t][k] via src0.
    //               All positions t=0..31 provide B[k][t] via src1.
    for (int k = 0; k < K; k++) {
        int a_row = wg_row * 32 + b * 16 + (t % 16);
        int b_col = wg_col * 32 + t;
        float a  = (a_row < M) ? A[a_row * K + k] : 0.f;
        float bv = (b_col < N) ? B[k * N + b_col] : 0.f;
        acc = __builtin_amdgcn_mfma_f32_32x32x1f32(a, bv, acc, 0, 0, 0);
    }

    // Each lane writes acc[0..15] (not acc[b*16..b*16+15]).
    // The row offset b*16 is in the C index, not in the accumulator index.
    int c_col = wg_col * 32 + t;
    for (int g = 0; g < 16; g++) {
        int c_row = wg_row * 32 + b * 16 + g;
        if (c_row < M && c_col < N)
            C[c_row * N + c_col] = acc[g];
    }
}

// ----------------------------------------------------------------------------
// 32x32x2: 64-thread workgroup (2 K-groups of 32 lanes), 16 output AGPRs/lane.
// Grid: block=(64,1,1), total_x=(N/32)*64, total_y=M/32
// K must be a multiple of 2.
// ----------------------------------------------------------------------------
extern "C" __global__
void mfma_gemm_32x32x2f32(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K)
{
    int wg_row = blockIdx.y;
    int wg_col = blockIdx.x;
    int lane   = threadIdx.x;  // 0..63
    int kg_out = lane / 32;    // K-group (0 or 1)
    int t      = lane % 32;    // position within K-group

    // 32x32x2 accumulator: 16 elements (same as 32x32x1 per-lane AGPR count).
    // The two K-groups accumulate independently into the same 16 AGPRs.
    float32x16_t acc = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    // K loop in steps of 2; each call covers K-indices k_base, k_base+1.
    // Lane (kg*32+t): K-index = k_base + kg_out; src0=A[t][k_base+kg_out], src1=B[k_base+kg_out][t]
    for (int k_base = 0; k_base < K; k_base += 2) {
        int k_idx  = k_base + kg_out;
        int a_row  = wg_row * 32 + t;
        int b_col  = wg_col * 32 + t;
        float a  = (a_row < M && k_idx < K) ? A[a_row * K + k_idx] : 0.f;
        float bv = (k_idx < K && b_col < N) ? B[k_idx * N + b_col] : 0.f;
        acc = __builtin_amdgcn_mfma_f32_32x32x2f32(a, bv, acc, 0, 0, 0);
    }

    // CDNA2 lane mapping for v_mfma_f32_32x32x2f32:
    // D[i][j] at GPR g, lane l: row = 8*(g/4) + 4*(l/32) + (g%4), col = l%32
    int c_col = wg_col * 32 + t;
    for (int g = 0; g < 16; g++) {
        int c_row = wg_row * 32 + 8 * (g / 4) + 4 * kg_out + (g % 4);
        if (c_row < M && c_col < N)
            C[c_row * N + c_col] = acc[g];
    }
}
