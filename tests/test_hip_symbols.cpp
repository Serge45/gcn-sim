#include <gtest/gtest.h>
#include "gcnsim/hip_hijack.h"
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>

class HipSymbolsAndModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        setenv("GCNSIM_HIJACK", "1", 1);
    }
};

TEST_F(HipSymbolsAndModuleTest, RegisterVarAndMemcpyToFromSymbol) {
    int host_var = 12345;
    const char* var_name = "test_device_scale";
    
    // Register variable using ABI symbol
    __hipRegisterVar(
        nullptr,
        reinterpret_cast<char*>(&host_var),
        nullptr,
        var_name,
        0,
        sizeof(int),
        0, // not constant
        1  // global
    );

    // 1. Get symbol address & size
    void* dev_ptr = nullptr;
    EXPECT_EQ(hipGetSymbolAddress(&dev_ptr, &host_var), gcnsim::hipSuccess);
    ASSERT_NE(dev_ptr, nullptr);

    size_t var_size = 0;
    EXPECT_EQ(hipGetSymbolSize(&var_size, &host_var), gcnsim::hipSuccess);
    EXPECT_EQ(var_size, sizeof(int));

    // Initially dev_ptr should have initial host value
    int initial_dev_val = 0;
    std::memcpy(&initial_dev_val, dev_ptr, sizeof(int));
    EXPECT_EQ(initial_dev_val, 12345);

    // 2. hipMemcpyToSymbol (by host pointer)
    int new_val = 98765;
    EXPECT_EQ(hipMemcpyToSymbol(&host_var, &new_val, sizeof(int), 0, gcnsim::hipMemcpyHostToDevice),
              gcnsim::hipSuccess);
    EXPECT_EQ(host_var, 98765);
    int readback_dev = 0;
    std::memcpy(&readback_dev, dev_ptr, sizeof(int));
    EXPECT_EQ(readback_dev, 98765);

    // 3. hipMemcpyFromSymbol
    int readback_val = 0;
    EXPECT_EQ(hipMemcpyFromSymbol(&readback_val, &host_var, sizeof(int), 0, gcnsim::hipMemcpyDeviceToHost),
              gcnsim::hipSuccess);
    EXPECT_EQ(readback_val, 98765);

    // 4. Invalid arguments
    EXPECT_EQ(hipGetSymbolAddress(nullptr, &host_var), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipGetSymbolAddress(&dev_ptr, nullptr), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipGetSymbolSize(nullptr, &host_var), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipGetSymbolSize(&var_size, nullptr), gcnsim::hipErrorInvalidValue);
}

TEST_F(HipSymbolsAndModuleTest, ModuleLoadDataFromMemory) {
    // Read vector_add.hsaco into memory buffer
    std::vector<std::string> search_paths = {
        "tools/vector_add/vector_add.hsaco",
        "build/tools/vector_add/vector_add.hsaco",
        "../tools/vector_add/vector_add.hsaco",
        "../build/tools/vector_add/vector_add.hsaco"
    };

    std::ifstream f;
    for (const auto& p : search_paths) {
        f.clear();
        f.open(p, std::ios::binary);
        if (f.is_open()) break;
    }
    ASSERT_TRUE(f.is_open()) << "Failed to find vector_add.hsaco";

    f.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    std::vector<uint8_t> hsaco_bytes(sz);
    f.read(reinterpret_cast<char*>(hsaco_bytes.data()), sz);

    // 1. hipModuleLoadData
    gcnsim::hipModule_t module = nullptr;
    EXPECT_EQ(hipModuleLoadData(&module, hsaco_bytes.data()), gcnsim::hipSuccess);
    ASSERT_NE(module, nullptr);

    // 2. hipModuleGetFunction
    gcnsim::hipFunction_t fn = nullptr;
    EXPECT_EQ(hipModuleGetFunction(&fn, module, "vector_add"), gcnsim::hipSuccess);
    ASSERT_NE(fn, nullptr);

    // 3. hipModuleUnload
    EXPECT_EQ(hipModuleUnload(module), gcnsim::hipSuccess);

    // 4. hipModuleLoadDataEx
    gcnsim::hipModule_t module_ex = nullptr;
    EXPECT_EQ(hipModuleLoadDataEx(&module_ex, hsaco_bytes.data(), 0, nullptr, nullptr),
              gcnsim::hipSuccess);
    ASSERT_NE(module_ex, nullptr);
    EXPECT_EQ(hipModuleUnload(module_ex), gcnsim::hipSuccess);

    // 5. Invalid image handling
    gcnsim::hipModule_t bad_mod = nullptr;
    uint8_t dummy_bytes[64] = {0};
    EXPECT_EQ(hipModuleLoadData(&bad_mod, dummy_bytes), gcnsim::hipErrorInvalidImage);
    EXPECT_EQ(hipModuleLoadData(nullptr, hsaco_bytes.data()), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipModuleLoadData(&bad_mod, nullptr), gcnsim::hipErrorInvalidValue);
}
