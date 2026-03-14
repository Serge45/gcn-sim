#include <gtest/gtest.h>
#include "gcnsim/elf_loader.h"
#include <string>

static std::string g_hsaco_path;

class ElfLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ElfLoaderTest, LoadHsacoSucceeds) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << "load_hsaco failed: " << err;
}

TEST_F(ElfLoaderTest, AtLeastOneKernelFound) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    EXPECT_GT(mod->kernels.size(), 0u) << "No kernels found in HSACO";
}

TEST_F(ElfLoaderTest, KernelDescriptorValid) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    ASSERT_GT(mod->kernels.size(), 0u);

    const auto& ki = mod->kernels[0];
    EXPECT_NE(ki.descriptor, nullptr) << "Kernel descriptor is null";
    EXPECT_NE(ki.code_ptr, nullptr) << "Kernel code pointer is null";
    EXPECT_GT(ki.code_size, 0u) << "Kernel code size is 0";
}

TEST_F(ElfLoaderTest, KernelNameNotEmpty) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    ASSERT_GT(mod->kernels.size(), 0u);

    for (const auto& ki : mod->kernels) {
        EXPECT_FALSE(ki.name.empty()) << "Kernel has empty name";
    }
}

TEST_F(ElfLoaderTest, COVVersionDetected) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    EXPECT_GE(mod->cov_version, 3) << "Expected COV3 or higher";
    EXPECT_LE(mod->cov_version, 6) << "Unexpected COV version";
}

TEST_F(ElfLoaderTest, KernelCodeEntryOffsetReasonable) {
    std::string err;
    auto mod = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod.has_value()) << err;
    ASSERT_GT(mod->kernels.size(), 0u);

    for (const auto& ki : mod->kernels) {
        if (ki.descriptor) {
            // Entry offset for gfx9 should be 256 bytes
            EXPECT_GT(ki.descriptor->kernel_code_entry_byte_offset, 0)
                << "Kernel entry offset should be positive";
        }
    }
}

int main(int argc, char** argv) {
    // Parse --hsaco flag before passing to gtest
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc) {
            g_hsaco_path = argv[i + 1];
        }
    }
    if (g_hsaco_path.empty()) {
        g_hsaco_path = "/opt/rocm/bin/gfx90a/vector_add_memory_fault_kernels.hsaco";
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
