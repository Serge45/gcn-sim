#include <gtest/gtest.h>
#include "gcnsim/elf_loader.h"
#include <string>
#include <cstring>

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

TEST_F(ElfLoaderTest, GFX908_ArchitectureDetection) {
    std::string err;
    auto mod_base = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod_base.has_value()) << err;

    std::vector<uint8_t> gfx908_elf = mod_base->elf_data;
    ASSERT_GE(gfx908_elf.size(), 52u);

    // e_flags is at byte offset 48 (4 bytes) in standard 64-bit ELF header
    uint32_t flags = 0x530; // mach = 0x30 (GFX908 / MI-100)
    std::memcpy(gfx908_elf.data() + 48, &flags, 4);

    std::string load_err;
    auto mod908 = gcnsim::load_hsaco_from_memory(gfx908_elf.data(), gfx908_elf.size(), load_err);
    ASSERT_TRUE(mod908.has_value()) << load_err;
    EXPECT_EQ(mod908->arch, gcnsim::TargetArch::GFX908);
    EXPECT_TRUE(mod908->is_gfx908);
    EXPECT_FALSE(mod908->is_gfx942);
    if (!mod908->kernels.empty()) {
        EXPECT_EQ(mod908->kernels[0].arch, gcnsim::TargetArch::GFX908);
        EXPECT_TRUE(mod908->kernels[0].is_gfx908);
    }
}

TEST_F(ElfLoaderTest, GFX942_ArchitectureDetection) {
    std::string err;
    auto mod_base = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod_base.has_value()) << err;

    std::vector<uint8_t> gfx942_elf = mod_base->elf_data;
    ASSERT_GE(gfx942_elf.size(), 52u);

    uint32_t flags = 0x54c; // mach = 0x4c (GFX942 / MI-300)
    std::memcpy(gfx942_elf.data() + 48, &flags, 4);

    std::string load_err;
    auto mod942 = gcnsim::load_hsaco_from_memory(gfx942_elf.data(), gfx942_elf.size(), load_err);
    ASSERT_TRUE(mod942.has_value()) << load_err;
    EXPECT_EQ(mod942->arch, gcnsim::TargetArch::GFX942);
    EXPECT_TRUE(mod942->is_gfx942);
    EXPECT_FALSE(mod942->is_gfx908);
    EXPECT_FALSE(mod942->is_gfx950);
    if (!mod942->kernels.empty()) {
        EXPECT_EQ(mod942->kernels[0].arch, gcnsim::TargetArch::GFX942);
        EXPECT_TRUE(mod942->kernels[0].is_gfx942);
    }
}

TEST_F(ElfLoaderTest, GFX950_ArchitectureDetection) {
    std::string err;
    auto mod_base = gcnsim::load_hsaco(g_hsaco_path, err);
    ASSERT_TRUE(mod_base.has_value()) << err;

    std::vector<uint8_t> gfx950_elf = mod_base->elf_data;
    ASSERT_GE(gfx950_elf.size(), 52u);

    uint32_t flags = 0x54f; // mach = 0x4f (GFX950 / MI-350)
    std::memcpy(gfx950_elf.data() + 48, &flags, 4);

    std::string load_err;
    auto mod950 = gcnsim::load_hsaco_from_memory(gfx950_elf.data(), gfx950_elf.size(), load_err);
    ASSERT_TRUE(mod950.has_value()) << load_err;
    EXPECT_EQ(mod950->arch, gcnsim::TargetArch::GFX950);
    EXPECT_TRUE(mod950->is_gfx950);
    EXPECT_FALSE(mod950->is_gfx908);
    EXPECT_FALSE(mod950->is_gfx942);
    if (!mod950->kernels.empty()) {
        EXPECT_EQ(mod950->kernels[0].arch, gcnsim::TargetArch::GFX950);
        EXPECT_TRUE(mod950->kernels[0].is_gfx950);
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
