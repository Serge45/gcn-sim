#include <gtest/gtest.h>
#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::string g_hsaco_path;

// Build kernarg from metadata
static std::vector<uint8_t> build_kernarg(const gcnsim::KernelInfo& ki,
                                          void** params, int n_params)
{
    uint32_t kernarg_size = ki.descriptor ? ki.descriptor->kernarg_size : 0;
    if (kernarg_size == 0) kernarg_size = 128;

    std::vector<uint8_t> buf(kernarg_size, 0);

    if (!ki.args.empty()) {
        size_t param_idx = 0;
        for (const auto& arg : ki.args) {
            if (arg.value_kind.size() >= 7 &&
                arg.value_kind.substr(0, 7) == "hidden_")
                continue;
            if (param_idx < (size_t)n_params &&
                params[param_idx] &&
                arg.offset + arg.size <= kernarg_size)
            {
                memcpy(buf.data() + arg.offset, params[param_idx], arg.size);
            }
            ++param_idx;
        }
    }
    return buf;
}

// CPU reference matching the Tensile-style kernel's actual computation.
//
// The kernel's LDS layout and store pattern produce:
//   D[r][c] = alpha * sum_k(A[k][c] * B[r][k]) + beta * C[r][c]
//
// This arises because:
// - LDS A feeds the MFMA src0 with A transposed (A_mfma[i][k] = A[k][i])
// - LDS B feeds the MFMA src1 with B transposed (B_mfma[k][j] = B[j][k])
// - The store transposes the MFMA output (D_mem[j][i] = MFMA_D[i][j])
// Net result: D_mem[r][c] = sum_k A[k][c] * B[r][k]
//
// A is M x K (row-major, stride LDA), B is K x N (stride LDB),
// C and D are M x N (strides LDC, LDD).
static void cpu_gemm_tensile(const float* A, const float* B, const float* C,
                              float* D, int M, int N, int K,
                              int LDA, int LDB, int LDC, int LDD,
                              float alpha, float beta)
{
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[k * LDA + c] * B[r * LDB + k];
            D[r * LDD + c] = alpha * s + beta * C[r * LDC + c];
        }
}

class TensileStyleGemmTest : public ::testing::Test {
protected:
    const gcnsim::KernelInfo* find_kernel(const gcnsim::HsacoModule& mod) {
        for (const auto& k : mod.kernels) {
            if (k.name.find("generated_gemm") != std::string::npos)
                return &k;
        }
        return nullptr;
    }
};

TEST_F(TensileStyleGemmTest, LoadAndFindKernel) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;
    EXPECT_GT(mod->kernels.size(), 0u);

    const auto* ki = find_kernel(*mod);
    EXPECT_NE(ki, nullptr) << "generated_gemm kernel not found in " << g_hsaco_path;
    if (ki) {
        EXPECT_EQ(ki->args.size(), 15u) << "Expected 15 kernel arguments";
    }
}

TEST_F(TensileStyleGemmTest, SingleTile_128x256x32) {
    // Smallest valid problem: one macro tile = 128 rows x 256 cols, K=32
    const int M = 128, N = 256, K = 32;
    const int LDA = K, LDB = N, LDC = N, LDD = N;
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> A(M * LDA), B(K * LDB), C(M * LDC, 0.f);
    std::vector<float> D_sim(M * LDD, 0.f), D_ref(M * LDD, 0.f);

    // Small integer values for exact float32 arithmetic
    for (int i = 0; i < M * LDA; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < K * LDB; i++) B[i] = (float)(i % 5 + 1);

    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;

    const auto* ki = find_kernel(*mod);
    ASSERT_NE(ki, nullptr) << "generated_gemm kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    float* pA = A.data();
    float* pB = B.data();
    float* pC = C.data();
    float* pD = D_sim.data();
    int pM = M, pN = N, pK = K;
    int pLDA = LDA, pLDB = LDB, pLDC = LDC, pLDD = LDD;
    float pAlpha = alpha, pBeta = beta;
    int numWgX = 1, numWgY = 1;

    void* params[] = {
        &pA, &pB, &pC, &pD,
        &pM, &pN, &pK,
        &pLDA, &pLDB, &pLDC, &pLDD,
        &pAlpha, &pBeta,
        &numWgX, &numWgY
    };
    auto kernarg = build_kernarg(*ki, params, 15);

    // Grid: 1 workgroup of 256 threads
    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = 256;    // 1 workgroup * 256 threads
    cfg.grid_y  = 1;
    cfg.grid_z  = 1;
    cfg.block_x = 256;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();

    std::string launch_err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, launch_err);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;

    // CPU reference matching the kernel's transposed computation
    cpu_gemm_tensile(A.data(), B.data(), C.data(), D_ref.data(),
                     M, N, K, LDA, LDB, LDC, LDD, alpha, beta);

    int failures = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int idx = i * LDD + j;
            float got      = D_sim[idx];
            float expected = D_ref[idx];
            if (std::abs(got - expected) > 1e-2f * std::abs(expected) + 1e-2f) {
                if (failures < 10)
                    EXPECT_NEAR(got, expected, 1e-2f)
                        << "Mismatch at [" << i << "][" << j << "]"
                        << ": expected " << expected << " got " << got;
                failures++;
            }
        }
    if (failures > 10)
        FAIL() << "Total element failures: " << failures << " / " << M * N;
}

TEST_F(TensileStyleGemmTest, AlphaBeta) {
    // Test alpha/beta scaling: D = 2.0 * (A*B) + 0.5 * C
    const int M = 128, N = 256, K = 32;
    const int LDA = K, LDB = N, LDC = N, LDD = N;
    const float alpha = 2.0f, beta = 0.5f;

    std::vector<float> A(M * LDA), B(K * LDB), C(M * LDC);
    std::vector<float> D_sim(M * LDD, 0.f), D_ref(M * LDD, 0.f);

    for (int i = 0; i < M * LDA; i++) A[i] = (float)(i % 3 + 1);
    for (int i = 0; i < K * LDB; i++) B[i] = (float)(i % 4 + 1);
    for (int i = 0; i < M * LDC; i++) C[i] = (float)(i % 5 + 1);

    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;

    const auto* ki = find_kernel(*mod);
    ASSERT_NE(ki, nullptr);

    float* pA = A.data();
    float* pB = B.data();
    float* pC = C.data();
    float* pD = D_sim.data();
    int pM = M, pN = N, pK = K;
    int pLDA = LDA, pLDB = LDB, pLDC = LDC, pLDD = LDD;
    float pAlpha = alpha, pBeta = beta;
    int numWgX = 1, numWgY = 1;

    void* params[] = {
        &pA, &pB, &pC, &pD,
        &pM, &pN, &pK,
        &pLDA, &pLDB, &pLDC, &pLDD,
        &pAlpha, &pBeta,
        &numWgX, &numWgY
    };
    auto kernarg = build_kernarg(*ki, params, 15);

    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = 256;
    cfg.grid_y  = 1;
    cfg.grid_z  = 1;
    cfg.block_x = 256;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();

    std::string launch_err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, launch_err);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;

    cpu_gemm_tensile(A.data(), B.data(), C.data(), D_ref.data(),
                     M, N, K, LDA, LDB, LDC, LDD, alpha, beta);

    int failures = 0;
    for (int i = 0; i < M * N; i++) {
        float got      = D_sim[i];
        float expected = D_ref[i];
        if (std::abs(got - expected) > 1e-2f * std::abs(expected) + 1e-2f) {
            if (failures < 10)
                EXPECT_NEAR(got, expected, 1e-2f)
                    << "Mismatch at element " << i
                    << " (row=" << i/N << " col=" << i%N << ")"
                    << ": expected " << expected << " got " << got;
            failures++;
        }
    }
    if (failures > 10)
        FAIL() << "Total failures: " << failures << " / " << M * N;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc)
            g_hsaco_path = argv[i + 1];
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
