#include <gtest/gtest.h>
#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::string g_hsaco_path;

// Build kernarg from metadata (same helper pattern as test_vector_add)
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
    } else {
        // Manual layout: A(ptr8), B(ptr8), C(ptr8), M(int4), N(int4), K(int4)
        size_t off = 0;
        size_t sizes[] = {8, 8, 8, 4, 4, 4};
        for (int i = 0; i < n_params; i++) {
            if (params[i] && off + sizes[i] <= kernarg_size)
                memcpy(buf.data() + off, params[i], sizes[i]);
            off += sizes[i];
        }
    }
    return buf;
}

// Naive CPU GEMM: C = A * B  (row-major, no transpose, accumulating)
static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] += s;
        }
}

class GemmMfmaTest : public ::testing::Test {
protected:
    // 16x16 matrices — 4 tiles of 4x4 in each dimension
    static constexpr int M = 16, N = 16, K = 16;

    std::vector<float> A, B, C_sim, C_ref;

    void SetUp() override {
        A.resize(M * K);
        B.resize(K * N);
        C_sim.resize(M * N, 0.f);
        C_ref.resize(M * N, 0.f);

        // Small integer values to keep reference arithmetic exact in float32
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 8 + 1);
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 4 + 1);
    }
};

TEST_F(GemmMfmaTest, LoadAndFindKernel) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;
    EXPECT_GT(mod->kernels.size(), 0u);

    bool found = false;
    for (const auto& ki : mod->kernels) {
        if (ki.name.find("gemm_mfma") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "gemm_mfma kernel not found in " << g_hsaco_path;
}

TEST_F(GemmMfmaTest, CorrectResults) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;

    const gcnsim::KernelInfo* ki = nullptr;
    for (const auto& k : mod->kernels) {
        if (k.name.find("gemm_mfma") != std::string::npos) {
            ki = &k;
            break;
        }
    }
    ASSERT_NE(ki, nullptr)    << "gemm_mfma kernel not found";
    ASSERT_NE(ki->code_ptr,   nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    float* pA = A.data();
    float* pB = B.data();
    float* pC = C_sim.data();
    int    pM = M, pN = N, pK = K;

    void* params[] = { &pA, &pB, &pC, &pM, &pN, &pK };
    auto kernarg = build_kernarg(*ki, params, 6);

    // Grid: (N/4 * 4, M/4 * 1, 1) total threads, block: (4, 1, 1)
    // Each workgroup is 4 threads (one MFMA block) and handles one 4x4 tile.
    // LaunchConfig.grid_x is total threads (not workgroup count).
    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = (N / 4) * 4;  // N/4 workgroups * 4 threads each
    cfg.grid_y  = (M / 4) * 1;  // M/4 workgroups * 1 thread each
    cfg.grid_z  = 1;
    cfg.block_x = 4;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();

    std::string launch_err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, launch_err);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;

    // Compute reference result on CPU
    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

    // Compare
    int failures = 0;
    for (int i = 0; i < M * N; i++) {
        float got      = C_sim[i];
        float expected = C_ref[i];
        if (std::abs(got - expected) > 1e-3f * std::abs(expected) + 1e-3f) {
            if (failures < 10)
                EXPECT_NEAR(got, expected, 1e-3f)
                    << "Mismatch at element " << i
                    << " (row=" << i/N << " col=" << i%N << ")"
                    << ": expected " << expected << " got " << got;
            failures++;
        }
    }
    if (failures > 10)
        FAIL() << "Total element failures: " << failures << " / " << M * N;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc)
            g_hsaco_path = argv[i + 1];
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
