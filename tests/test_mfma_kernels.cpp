#include <gtest/gtest.h>
#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static std::string g_hsaco_path;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static const gcnsim::KernelInfo*
find_kernel(const gcnsim::HsacoModule& mod, const std::string& name)
{
    for (const auto& ki : mod.kernels)
        if (ki.name.find(name) != std::string::npos)
            return &ki;
    return nullptr;
}

static bool run_kernel(const gcnsim::KernelInfo& ki,
                       const gcnsim::LaunchConfig& cfg,
                       void** params, int n_params,
                       std::string& err)
{
    auto kernarg = build_kernarg(ki, params, n_params);
    gcnsim::LaunchConfig c = cfg;
    c.kernarg      = kernarg.data();
    c.kernarg_size = (uint32_t)kernarg.size();
    return gcnsim::launch_kernel(ki, c, err);
}

static void check_close(const float* got, const float* ref, int n,
                        const char* label, float rtol = 1e-3f)
{
    int failures = 0;
    for (int i = 0; i < n; i++) {
        float tol = rtol * std::abs(ref[i]) + 1e-3f;
        if (std::abs(got[i] - ref[i]) > tol) {
            if (failures < 5)
                ADD_FAILURE() << label << " element " << i
                              << ": expected " << ref[i] << " got " << got[i];
            failures++;
        }
    }
    if (failures > 5)
        ADD_FAILURE() << label << ": total failures=" << failures << "/" << n;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class MfmaKernelsTest : public ::testing::Test {
protected:
    std::optional<gcnsim::HsacoModule> mod;
    std::string err;

    void SetUp() override {
        mod = gcnsim::load_hsaco(g_hsaco_path, err);
        ASSERT_TRUE(mod.has_value()) << "load_hsaco: " << err;
    }
};

// ---------------------------------------------------------------------------
// 16x16x1 — 16-thread WG, 16 output AGPRs per lane
// ---------------------------------------------------------------------------

TEST_F(MfmaKernelsTest, Gemm16x16x1) {
    constexpr int M = 32, N = 32, K = 16;

    const gcnsim::KernelInfo* ki = find_kernel(*mod, "mfma_gemm_16x16x1f32");
    ASSERT_NE(ki, nullptr) << "kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.f), C_ref(M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5 + 1);

    float *pA = A.data(), *pB = B.data(), *pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = {&pA, &pB, &pC, &pM, &pN, &pK};

    // 16-thread WG; grid_x = (N/16)*16 total, grid_y = M/16
    gcnsim::LaunchConfig cfg{};
    cfg.grid_x  = (N / 16) * 16;
    cfg.grid_y  = M / 16;
    cfg.grid_z  = 1;
    cfg.block_x = 16;
    cfg.block_y = 1;
    cfg.block_z = 1;

    ASSERT_TRUE(run_kernel(*ki, cfg, params, 6, err)) << err;

    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);
    check_close(C_sim.data(), C_ref.data(), M * N, "16x16x1");
}

// ---------------------------------------------------------------------------
// 16x16x4 — 64-thread WG, 4 output AGPRs per lane, K must be multiple of 4
// ---------------------------------------------------------------------------

TEST_F(MfmaKernelsTest, Gemm16x16x4) {
    constexpr int M = 32, N = 32, K = 16;

    const gcnsim::KernelInfo* ki = find_kernel(*mod, "mfma_gemm_16x16x4f32");
    ASSERT_NE(ki, nullptr) << "kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.f), C_ref(M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5 + 1);

    float *pA = A.data(), *pB = B.data(), *pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = {&pA, &pB, &pC, &pM, &pN, &pK};

    // 64-thread WG; grid_x = (N/16)*64 total, grid_y = M/16
    gcnsim::LaunchConfig cfg{};
    cfg.grid_x  = (N / 16) * 64;
    cfg.grid_y  = M / 16;
    cfg.grid_z  = 1;
    cfg.block_x = 64;
    cfg.block_y = 1;
    cfg.block_z = 1;

    ASSERT_TRUE(run_kernel(*ki, cfg, params, 6, err)) << err;

    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);
    check_close(C_sim.data(), C_ref.data(), M * N, "16x16x4");
}

// ---------------------------------------------------------------------------
// 32x32x1 — 64-thread WG (2 blocks of 32 lanes), 16 output AGPRs per lane
// ---------------------------------------------------------------------------

TEST_F(MfmaKernelsTest, Gemm32x32x1) {
    constexpr int M = 64, N = 64, K = 16;

    const gcnsim::KernelInfo* ki = find_kernel(*mod, "mfma_gemm_32x32x1f32");
    ASSERT_NE(ki, nullptr) << "kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.f), C_ref(M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5 + 1);

    float *pA = A.data(), *pB = B.data(), *pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = {&pA, &pB, &pC, &pM, &pN, &pK};

    // 64-thread WG; grid_x = (N/32)*64 total, grid_y = M/32
    gcnsim::LaunchConfig cfg{};
    cfg.grid_x  = (N / 32) * 64;
    cfg.grid_y  = M / 32;
    cfg.grid_z  = 1;
    cfg.block_x = 64;
    cfg.block_y = 1;
    cfg.block_z = 1;

    ASSERT_TRUE(run_kernel(*ki, cfg, params, 6, err)) << err;

    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);
    check_close(C_sim.data(), C_ref.data(), M * N, "32x32x1");
}

// ---------------------------------------------------------------------------
// 32x32x2 — 64-thread WG (2 K-groups of 32 lanes), 16 output AGPRs per lane
// ---------------------------------------------------------------------------

TEST_F(MfmaKernelsTest, Gemm32x32x2) {
    constexpr int M = 64, N = 64, K = 16;

    const gcnsim::KernelInfo* ki = find_kernel(*mod, "mfma_gemm_32x32x2f32");
    ASSERT_NE(ki, nullptr) << "kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.f), C_ref(M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5 + 1);

    float *pA = A.data(), *pB = B.data(), *pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = {&pA, &pB, &pC, &pM, &pN, &pK};

    // 64-thread WG; grid_x = (N/32)*64 total, grid_y = M/32
    gcnsim::LaunchConfig cfg{};
    cfg.grid_x  = (N / 32) * 64;
    cfg.grid_y  = M / 32;
    cfg.grid_z  = 1;
    cfg.block_x = 64;
    cfg.block_y = 1;
    cfg.block_z = 1;

    ASSERT_TRUE(run_kernel(*ki, cfg, params, 6, err)) << err;

    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);
    check_close(C_sim.data(), C_ref.data(), M * N, "32x32x2");
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc)
            g_hsaco_path = argv[++i];
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
