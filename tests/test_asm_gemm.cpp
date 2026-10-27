#include <gtest/gtest.h>
#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

namespace gcnsim {
WaveStatus run_wavefront(WavefrontState& wave);
}

static std::string g_hsaco_path;

// Build kernarg from metadata (same helper as test_gemm_mfma.cpp)
static std::vector<uint8_t> build_kernarg(const gcnsim::KernelInfo& ki,
                                          void** params, int n_params)
{
    uint32_t kernarg_size = ki.descriptor ? ki.descriptor->kernarg_size : 0;
    if (kernarg_size == 0) kernarg_size = 64;

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

// Naive CPU GEMM: C += A * B  (row-major, no transpose)
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

class AsmGemmMfmaTest : public ::testing::Test {
protected:
    static constexpr int M = 16, N = 16, K = 16;

    std::vector<float> A, B, C_sim, C_ref;

    void SetUp() override {
        A.resize(M * K);
        B.resize(K * N);
        C_sim.resize(M * N, 0.f);
        C_ref.resize(M * N, 0.f);

        // Small integer values to keep float32 arithmetic exact
        for (int i = 0; i < M * K; i++) A[i] = (float)(i % 8 + 1);
        for (int i = 0; i < K * N; i++) B[i] = (float)(i % 4 + 1);
    }
};

TEST_F(AsmGemmMfmaTest, LoadAndFindKernel) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;
    EXPECT_GT(mod->kernels.size(), 0u);

    bool found = false;
    for (const auto& ki : mod->kernels) {
        if (ki.name.find("asm_gemm_mfma") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "asm_gemm_mfma kernel not found in " << g_hsaco_path;
}

TEST_F(AsmGemmMfmaTest, CorrectResults) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;

    const gcnsim::KernelInfo* ki = nullptr;
    for (const auto& k : mod->kernels) {
        if (k.name.find("asm_gemm_mfma") != std::string::npos) {
            ki = &k;
            break;
        }
    }
    ASSERT_NE(ki, nullptr)    << "asm_gemm_mfma kernel not found";
    ASSERT_NE(ki->code_ptr,   nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    float* pA = A.data();
    float* pB = B.data();
    float* pC = C_sim.data();
    int    pM = M, pN = N, pK = K;

    void* params[] = { &pA, &pB, &pC, &pM, &pN, &pK };
    auto kernarg = build_kernarg(*ki, params, 6);

    // Grid: N/4 workgroups (x) * M/4 workgroups (y), block=(4,1,1)
    // LaunchConfig.grid_x/y are total thread counts.
    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = (N / 4) * 4;
    cfg.grid_y  = (M / 4) * 1;
    cfg.grid_z  = 1;
    cfg.block_x = 4;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();

    std::string launch_err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, launch_err);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;

    // CPU reference (C_ref starts at 0, same as C_sim)
    cpu_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

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

TEST_F(AsmGemmMfmaTest, LaunchOnGFX908_120CUs) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    const gcnsim::KernelInfo* ki = nullptr;
    for (const auto& k : mod->kernels) {
        if (k.name.find("asm_gemm_mfma") != std::string::npos) {
            ki = &k;
            break;
        }
    }
    ASSERT_NE(ki, nullptr);

    // Clone kernel and set target to GFX908 (CDNA1 / MI-100)
    gcnsim::KernelInfo ki_908 = *ki;
    ki_908.arch = gcnsim::TargetArch::GFX908;
    ki_908.is_gfx908 = true;
    ki_908.is_gfx942 = false;

    const int M = 8, N = 8, K = 8;
    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.0f), C_ref(M * N, 0.0f);
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;

    float* pA = A.data(); float* pB = B.data(); float* pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = { &pA, &pB, &pC, &pM, &pN, &pK };
    auto kernarg = build_kernarg(*ki, params, 6);

    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = (N / 4) * 4;
    cfg.grid_y  = (M / 4) * 1;
    cfg.grid_z  = 1;
    cfg.block_x = 4;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();
    cfg.enable_cycle_counting = true;

    uint64_t total_cycles = 0;
    std::string launch_err;
    bool ok = gcnsim::launch_kernel(ki_908, cfg, launch_err, &total_cycles);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;
    EXPECT_GT(total_cycles, 0u);

    // Verify numerical accuracy
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C_sim[i], 16.0f, 1e-4f); // 8 * 1.0 * 2.0 = 16.0
    }
}

TEST_F(AsmGemmMfmaTest, LaunchOnGFX950_256CUs) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    const gcnsim::KernelInfo* ki = nullptr;
    for (const auto& k : mod->kernels) {
        if (k.name.find("asm_gemm_mfma") != std::string::npos) {
            ki = &k;
            break;
        }
    }
    ASSERT_NE(ki, nullptr);

    // Clone kernel and set target to GFX950 (CDNA4 / MI-350 with 256 CUs)
    gcnsim::KernelInfo ki_950 = *ki;
    ki_950.arch = gcnsim::TargetArch::GFX950;
    ki_950.is_gfx950 = true;
    ki_950.is_gfx942 = false;
    ki_950.is_gfx908 = false;
    ki_950.accum_offset = 0;

    const int M = 8, N = 8, K = 8;
    std::vector<float> A(M * K), B(K * N), C_sim(M * N, 0.0f), C_ref(M * N, 0.0f);
    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;

    float* pA = A.data(); float* pB = B.data(); float* pC = C_sim.data();
    int pM = M, pN = N, pK = K;
    void* params[] = { &pA, &pB, &pC, &pM, &pN, &pK };
    auto kernarg = build_kernarg(*ki, params, 6);

    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = (N / 4) * 4;
    cfg.grid_y  = (M / 4) * 1;
    cfg.grid_z  = 1;
    cfg.block_x = 4;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg      = kernarg.data();
    cfg.kernarg_size = (uint32_t)kernarg.size();
    cfg.enable_cycle_counting = true;

    uint64_t total_cycles = 0;
    std::string launch_err;
    bool ok = gcnsim::launch_kernel(ki_950, cfg, launch_err, &total_cycles);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;
    EXPECT_GT(total_cycles, 0u);

    // Verify numerical accuracy: 8 * 1.0 * 2.0 = 16.0f
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C_sim[i], 16.0f, 1e-4f);
    }
}

TEST_F(AsmGemmMfmaTest, GFX950_MicroscalingFP4_InstructionKernel) {
    // Construct in-memory code object testing v_mfma_scale_f32_16x16x128_f8f6f4
    // w0: bits31_26 = 0x34, op = 1020, is_agpr = 1, dst = a0
    // w1: src0 = v0 (256 + 0), src1 = v4 (256 + 4), src2 = a0 (0)
    // Followed by s_endpgm (0xBF810000)
    std::vector<uint32_t> code = {
        (0x34u << 26) | (1020u << 16) | (1u << 15) | 0u,
        (0u << 18) | ((256u + 4u) << 9) | (256u + 0u),
        0xBF810000u // s_endpgm
    };

    auto ws = std::make_unique<gcnsim::WavefrontState>();
    memset((void*)ws.get(), 0, offsetof(gcnsim::WavefrontState, lgkm_pending));
    ws->exec = ~0ULL;
    ws->is_gfx950 = true;
    ws->kernel_code_ptr = reinterpret_cast<const uint8_t*>(code.data());
    ws->kernel_code_size = code.size() * sizeof(uint32_t);
    ws->count_cycles = true;

    // Initialize packed FP4 inputs in v0..v3 and v4..v7
    // FP4 E2M1: 1.0 is 0x2, 2.0 is 0x4
    uint32_t packed_1 = 0x22222222;
    uint32_t packed_2 = 0x44444444;
    for (int lane = 0; lane < 64; lane++) {
        for (int d = 0; d < 4; d++) {
            ws->vgpr[0 + d][lane] = packed_1;
            ws->vgpr[4 + d][lane] = packed_2;
        }
        for (int g = 0; g < 4; g++) {
            ws->agpr[g][lane] = 0;
        }
    }

    gcnsim::run_wavefront(*ws);

    // K=128 with A=1.0, B=2.0 -> dot product = 128 * 1.0 * 2.0 = 256.0f
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            float val;
            uint32_t bits = ws->agpr[gpr][lane];
            memcpy(&val, &bits, 4);
            EXPECT_FLOAT_EQ(val, 256.0f);
        }
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc)
            g_hsaco_path = argv[i + 1];
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
