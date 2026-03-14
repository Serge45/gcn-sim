#include <gtest/gtest.h>
#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>

static std::string g_hsaco_path;

// -------------------------------------------------------------------------
// Build kernarg buffer manually from arg metadata
// -------------------------------------------------------------------------
static std::vector<uint8_t> build_kernarg(const gcnsim::KernelInfo& ki,
                                           void** params) {
    uint32_t kernarg_size = ki.descriptor ? ki.descriptor->kernarg_size : 0;
    if (kernarg_size == 0) kernarg_size = 64; // fallback

    std::vector<uint8_t> buf(kernarg_size, 0);

    if (!ki.args.empty()) {
        // Only copy explicit user args; hidden args (value_kind starts with "hidden_")
        // are filled automatically by launch_kernel from the LaunchConfig.
        size_t param_idx = 0;
        for (size_t i = 0; i < ki.args.size(); i++) {
            const auto& arg = ki.args[i];
            if (arg.value_kind.size() >= 7 &&
                arg.value_kind.substr(0, 7) == "hidden_") {
                continue; // skip hidden args — launch_kernel fills these
            }
            if (params && params[param_idx] && arg.offset + arg.size <= kernarg_size) {
                memcpy(buf.data() + arg.offset, params[param_idx], arg.size);
            }
            ++param_idx;
        }
    } else {
        // No metadata — pack manually assuming:
        // arg0: const float* A (8 bytes at offset 0)
        // arg1: const float* B (8 bytes at offset 8)
        // arg2: float* C       (8 bytes at offset 16)
        // arg3: int N          (4 bytes at offset 24)
        uint32_t offset = 0;
        uint32_t sizes[] = {8, 8, 8, 4};
        for (int i = 0; i < 4 && params && params[i]; i++) {
            if (offset + sizes[i] <= kernarg_size) {
                memcpy(buf.data() + offset, params[i], sizes[i]);
            }
            // Align next arg to its natural size
            offset += sizes[i];
            // Align to next multiple of sizes[i+1] if applicable
        }
    }
    return buf;
}

class VectorAddTest : public ::testing::Test {
protected:
    static constexpr int N = 1024;
    std::vector<float> A, B, C;

    void SetUp() override {
        A.resize(N);
        B.resize(N);
        C.resize(N, 0.0f);
        for (int i = 0; i < N; i++) {
            A[i] = (float)(i + 1);
            B[i] = (float)(N - i);
        }
    }
};

TEST_F(VectorAddTest, LoadAndFindKernel) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;
    EXPECT_GT(mod->kernels.size(), 0u);

    bool found = false;
    for (const auto& ki : mod->kernels) {
        if (ki.name.find("vector_add") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "vector_add kernel not found";
}

TEST_F(VectorAddTest, CorrectResults) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;

    // Find vector_add kernel
    const gcnsim::KernelInfo* ki = nullptr;
    for (const auto& k : mod->kernels) {
        if (k.name.find("vector_add") != std::string::npos) {
            ki = &k;
            break;
        }
    }
    ASSERT_NE(ki, nullptr) << "vector_add kernel not found";
    ASSERT_NE(ki->code_ptr, nullptr);
    ASSERT_NE(ki->descriptor, nullptr);

    float* pA = A.data();
    float* pB = B.data();
    float* pC = C.data();
    int    pN = N;

    void* params[] = { &pA, &pB, &pC, &pN };
    std::vector<uint8_t> kernarg = build_kernarg(*ki, params);

    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = N;
    cfg.grid_y  = 1;
    cfg.grid_z  = 1;
    cfg.block_x = 256;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.kernarg  = kernarg.data();
    cfg.kernarg_size = kernarg.size();

    std::string launch_err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, launch_err);
    ASSERT_TRUE(ok) << "launch_kernel failed: " << launch_err;

    int failures = 0;
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        float got = C[i];
        if (std::abs(got - expected) > 1e-5f) {
            if (failures < 10) {
                EXPECT_NEAR(got, expected, 1e-5f)
                    << "Mismatch at index " << i
                    << ": expected " << expected << " got " << got;
            }
            failures++;
        }
    }
    if (failures > 10) {
        FAIL() << "Total failures: " << failures;
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc) {
            g_hsaco_path = argv[i + 1];
        }
    }
    if (g_hsaco_path.empty()) {
        // Try to find it relative to binary
        g_hsaco_path = "tools/vector_add/vector_add.hsaco";
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
