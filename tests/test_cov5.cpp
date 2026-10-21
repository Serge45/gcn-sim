#include <gtest/gtest.h>
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include "gcnsim/memory_model.h"
#include <vector>
#include <cstring>
#include <string>

static std::string g_hsaco_path;

class Cov5Test : public ::testing::Test {
protected:
    std::string hsaco_path;

    void SetUp() override {
        hsaco_path = g_hsaco_path;
        if (hsaco_path.empty()) {
            hsaco_path = "tools/cov5_kernel/cov5_kernel.hsaco";
        }
        std::string err;
        auto mod = gcnsim::load_hsaco(hsaco_path, err);
        if (!mod) {
            hsaco_path = "build/tools/cov5_kernel/cov5_kernel.hsaco";
            mod = gcnsim::load_hsaco(hsaco_path, err);
        }
        if (!mod) {
            hsaco_path = "../tools/cov5_kernel/cov5_kernel.hsaco";
            mod = gcnsim::load_hsaco(hsaco_path, err);
        }
        ASSERT_TRUE(mod.has_value()) << "Failed to find or load cov5_kernel.hsaco: " << err;
        EXPECT_EQ(mod->cov_version, 5) << "Expected Code Object Version 5";
        ASSERT_FALSE(mod->kernels.empty()) << "No kernels found in COV5 binary";
    }
};

TEST_F(Cov5Test, EndToEndExecution) {
    std::string err;
    auto mod = gcnsim::load_hsaco(hsaco_path, err);
    ASSERT_TRUE(mod.has_value());
    const auto& kernel = mod->kernels[0];

    int out_val = 0;
    int a_val = 12345;
    int* p_out = &out_val;

    // Pack kernargs: out at offset 0 (8 bytes), a at offset 8 (4 bytes)
    size_t kernarg_size = 272;
    std::vector<uint8_t> kernargs(kernarg_size, 0);
    uint64_t ptr_val = reinterpret_cast<uint64_t>(p_out);
    std::memcpy(kernargs.data() + 0, &ptr_val, 8);
    std::memcpy(kernargs.data() + 8, &a_val, 4);

    gcnsim::LaunchConfig cfg{};
    cfg.grid_x = 1; cfg.grid_y = 1; cfg.grid_z = 1;
    cfg.block_x = 64; cfg.block_y = 1; cfg.block_z = 1;
    cfg.kernarg = kernargs.data();
    cfg.kernarg_size = kernargs.size();

    bool ok = gcnsim::launch_kernel(kernel, cfg, err);
    ASSERT_TRUE(ok) << "Kernel launch failed: " << err;
    EXPECT_EQ(out_val, 12345) << "Kernel calculation incorrect";
}

TEST_F(Cov5Test, MemoryFaultCaughtGracefully) {
    std::string err;
    auto mod = gcnsim::load_hsaco(hsaco_path, err);
    ASSERT_TRUE(mod.has_value());
    const auto& kernel = mod->kernels[0];

    // Out pointer is NULL
    int a_val = 999;
    size_t kernarg_size = 272;
    std::vector<uint8_t> kernargs(kernarg_size, 0);
    uint64_t ptr_val = 0; // null pointer
    std::memcpy(kernargs.data() + 0, &ptr_val, 8);
    std::memcpy(kernargs.data() + 8, &a_val, 4);

    gcnsim::LaunchConfig cfg{};
    cfg.grid_x = 1; cfg.grid_y = 1; cfg.grid_z = 1;
    cfg.block_x = 64; cfg.block_y = 1; cfg.block_z = 1;
    cfg.kernarg = kernargs.data();
    cfg.kernarg_size = kernargs.size();

    bool ok = gcnsim::launch_kernel(kernel, cfg, err);
    EXPECT_FALSE(ok) << "Expected kernel launch to fail on NULL pointer";
    EXPECT_NE(err.find("GPU Memory Fault: NULL pointer access"), std::string::npos)
        << "Expected memory fault message, got: " << err;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc) {
            g_hsaco_path = argv[i + 1];
        }
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
