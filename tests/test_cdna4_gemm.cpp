#include <gtest/gtest.h>
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <string>

static std::string g_hsaco_path;

class Cdna4GemmTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(Cdna4GemmTest, LoadGfx950CodeObject) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "Failed to load HSACO: " << err;
    EXPECT_EQ(mod->arch, gcnsim::TargetArch::GFX950);
    EXPECT_TRUE(mod->is_gfx950);
    ASSERT_EQ(mod->kernels.size(), 1u);
    EXPECT_EQ(mod->kernels[0].name, "cdna4_gemm");
    EXPECT_TRUE(mod->kernels[0].is_gfx950);
}

TEST_F(Cdna4GemmTest, ExecuteDoubleRateFp16Mfma) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    const auto& kernel = mod->kernels[0];

    const int N = 64;
    std::vector<float> h_c(N, 0.0f);

    float* p_a = nullptr;
    float* p_b = nullptr;
    float* p_c = h_c.data();

    std::vector<uint8_t> buf(64, 0);
    memcpy(buf.data() + 0, &p_a, 8);
    memcpy(buf.data() + 8, &p_b, 8);
    memcpy(buf.data() + 16, &p_c, 8);

    gcnsim::LaunchConfig cfg;
    cfg.grid_x = 64; cfg.grid_y = 1; cfg.grid_z = 1;
    cfg.block_x = 64; cfg.block_y = 1; cfg.block_z = 1;
    cfg.kernarg = buf.data();
    cfg.kernarg_size = buf.size();
    cfg.enable_cycle_counting = true;
    cfg.num_cus = 256; // MI-350 / CDNA4: 256 active CUs

    uint64_t total_cycles = 0;
    bool ok = gcnsim::launch_kernel(kernel, cfg, err, &total_cycles);
    ASSERT_TRUE(ok) << "Kernel launch failed: " << err;
    EXPECT_GT(total_cycles, 0u);

    // Verify all 64 lanes computed 64.0f (32 * 1.0f * 2.0f)
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_c[i], 64.0f, 1e-4f) << "Mismatch at lane " << i;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc) {
            g_hsaco_path = argv[++i];
        }
    }
    return RUN_ALL_TESTS();
}
