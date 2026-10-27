#include <gtest/gtest.h>
#include "gcnsim/hip_hijack.h"
#include <vector>
#include <cstring>
#include <cstdlib>

class HipRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Force hijack enabled
        setenv("GCNSIM_HIJACK", "1", 1);
    }
};

TEST_F(HipRuntimeTest, DeviceCountAndSwitching) {
    int count = 0;
    EXPECT_EQ(hipGetDeviceCount(&count), gcnsim::hipSuccess);
    EXPECT_GE(count, 1);

    int cur = -1;
    EXPECT_EQ(hipGetDevice(&cur), gcnsim::hipSuccess);
    EXPECT_EQ(cur, 0);

    // If multi-device (default 2 for MI200)
    if (count >= 2) {
        EXPECT_EQ(hipSetDevice(1), gcnsim::hipSuccess);
        EXPECT_EQ(hipGetDevice(&cur), gcnsim::hipSuccess);
        EXPECT_EQ(cur, 1);

        EXPECT_EQ(hipSetDevice(0), gcnsim::hipSuccess);
        EXPECT_EQ(hipGetDevice(&cur), gcnsim::hipSuccess);
        EXPECT_EQ(cur, 0);
    }

    // Invalid device IDs
    EXPECT_EQ(hipSetDevice(-1), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipSetDevice(999), gcnsim::hipErrorInvalidValue);
}

TEST_F(HipRuntimeTest, DevicePropertiesMI200) {
    unsetenv("GCN_SIM_ARCH");
    gcnsim::hipDeviceProp_t prop;
    EXPECT_EQ(hipGetDeviceProperties(&prop, 0), gcnsim::hipSuccess);
    
    // Verify MI250X properties
    EXPECT_STREQ(prop.gcnArchName, "gfx90a:sramecc+:xnack-");
    EXPECT_EQ(prop.major, 9);
    EXPECT_EQ(prop.minor, 0);
    EXPECT_EQ(prop.multiProcessorCount, 110);
    EXPECT_EQ(prop.warpSize, 64);
    EXPECT_EQ(prop.maxThreadsPerBlock, 1024);
    EXPECT_EQ(prop.totalGlobalMem, 64ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(prop.isMultiGpuBoard, 1);
    EXPECT_EQ(prop.canMapHostMemory, 1);

    // Invalid device ID
    EXPECT_EQ(hipGetDeviceProperties(&prop, -1), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipGetDeviceProperties(&prop, 100), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipGetDeviceProperties(nullptr, 0), gcnsim::hipErrorInvalidValue);
}

TEST_F(HipRuntimeTest, DevicePropertiesMI100) {
    setenv("GCN_SIM_ARCH", "gfx908", 1);
    gcnsim::hipDeviceProp_t prop;
    EXPECT_EQ(hipGetDeviceProperties(&prop, 0), gcnsim::hipSuccess);

    EXPECT_STREQ(prop.name, "AMD Instinct MI100");
    EXPECT_STREQ(prop.gcnArchName, "gfx908:sramecc+:xnack-");
    EXPECT_EQ(prop.major, 9);
    EXPECT_EQ(prop.minor, 0);
    EXPECT_EQ(prop.multiProcessorCount, 120);
    EXPECT_EQ(prop.warpSize, 64);
    EXPECT_EQ(prop.totalGlobalMem, 32ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(prop.l2CacheSize, 8 * 1024 * 1024);
    EXPECT_EQ(prop.memoryBusWidth, 4096);
    unsetenv("GCN_SIM_ARCH");
}

TEST_F(HipRuntimeTest, DevicePropertiesMI300) {
    setenv("GCN_SIM_ARCH", "gfx942", 1);
    gcnsim::hipDeviceProp_t prop;
    EXPECT_EQ(hipGetDeviceProperties(&prop, 0), gcnsim::hipSuccess);

    EXPECT_STREQ(prop.name, "AMD Instinct MI300X");
    EXPECT_STREQ(prop.gcnArchName, "gfx942:sramecc+:xnack-");
    EXPECT_EQ(prop.major, 9);
    EXPECT_EQ(prop.minor, 4);
    EXPECT_EQ(prop.multiProcessorCount, 304);
    EXPECT_EQ(prop.warpSize, 64);
    EXPECT_EQ(prop.totalGlobalMem, 192ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(prop.l2CacheSize, 32 * 1024 * 1024);
    EXPECT_EQ(prop.memoryBusWidth, 8192);
    unsetenv("GCN_SIM_ARCH");
}

TEST_F(HipRuntimeTest, DevicePropertiesMI350) {
    setenv("GCN_SIM_ARCH", "gfx950", 1);
    gcnsim::hipDeviceProp_t prop;
    EXPECT_EQ(hipGetDeviceProperties(&prop, 0), gcnsim::hipSuccess);

    EXPECT_STREQ(prop.name, "AMD Instinct MI350X");
    EXPECT_STREQ(prop.gcnArchName, "gfx950:sramecc+:xnack-");
    EXPECT_EQ(prop.major, 9);
    EXPECT_EQ(prop.minor, 5);
    EXPECT_EQ(prop.multiProcessorCount, 256);
    EXPECT_EQ(prop.warpSize, 64);
    EXPECT_EQ(prop.totalGlobalMem, 288ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(prop.l2CacheSize, 32 * 1024 * 1024);
    EXPECT_EQ(prop.memoryBusWidth, 8192);
    EXPECT_EQ(prop.sharedMemPerBlock, 163840);
    EXPECT_EQ(prop.maxSharedMemoryPerMultiProcessor, 163840);
    unsetenv("GCN_SIM_ARCH");
}

TEST_F(HipRuntimeTest, DeviceAttributes) {
    int val = 0;
    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeMaxThreadsPerBlock, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 1024);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeWarpSize, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 64);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeMultiprocessorCount, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 110);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeComputeCapabilityMajor, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 9);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeComputeCapabilityMinor, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 0);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeL2CacheSize, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 8 * 1024 * 1024);

    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeMaxRegistersPerBlock, 0), gcnsim::hipSuccess);
    EXPECT_EQ(val, 65536);

    // Invalid parameters
    EXPECT_EQ(hipDeviceGetAttribute(nullptr, gcnsim::hipDeviceAttributeWarpSize, 0), gcnsim::hipErrorInvalidValue);
    EXPECT_EQ(hipDeviceGetAttribute(&val, gcnsim::hipDeviceAttributeWarpSize, -1), gcnsim::hipErrorInvalidValue);
}

TEST_F(HipRuntimeTest, PeerAccess) {
    int canAccess = 0;
    EXPECT_EQ(hipDeviceCanAccessPeer(&canAccess, 0, 1), gcnsim::hipSuccess);
    EXPECT_EQ(canAccess, 1);

    EXPECT_EQ(hipDeviceCanAccessPeer(&canAccess, 0, 0), gcnsim::hipSuccess);
    EXPECT_EQ(canAccess, 0);

    EXPECT_EQ(hipDeviceEnablePeerAccess(1, 0), gcnsim::hipSuccess);
    EXPECT_EQ(hipDeviceDisablePeerAccess(1), gcnsim::hipSuccess);

    EXPECT_EQ(hipDeviceCanAccessPeer(nullptr, 0, 1), gcnsim::hipErrorInvalidValue);
}

TEST_F(HipRuntimeTest, EventManagementAndTiming) {
    gcnsim::hipEvent_t start = nullptr;
    gcnsim::hipEvent_t stop = nullptr;

    EXPECT_EQ(hipEventCreate(&start), gcnsim::hipSuccess);
    EXPECT_NE(start, nullptr);
    EXPECT_EQ(hipEventCreateWithFlags(&stop, 0), gcnsim::hipSuccess);
    EXPECT_NE(stop, nullptr);

    EXPECT_EQ(hipEventRecord(start, nullptr), gcnsim::hipSuccess);
    EXPECT_EQ(hipEventRecord(stop, nullptr), gcnsim::hipSuccess);

    EXPECT_EQ(hipEventSynchronize(stop), gcnsim::hipSuccess);
    EXPECT_EQ(hipEventQuery(stop), gcnsim::hipSuccess);

    float elapsed_ms = -1.0f;
    EXPECT_EQ(hipEventElapsedTime(&elapsed_ms, start, stop), gcnsim::hipSuccess);
    EXPECT_GE(elapsed_ms, 0.0f);

    EXPECT_EQ(hipStreamWaitEvent(nullptr, stop, 0), gcnsim::hipSuccess);

    EXPECT_EQ(hipEventDestroy(start), gcnsim::hipSuccess);
    EXPECT_EQ(hipEventDestroy(stop), gcnsim::hipSuccess);
}

TEST_F(HipRuntimeTest, MemsetOperations) {
    const size_t N = 128;
    void* d_ptr = nullptr;
    EXPECT_EQ(hipMalloc(&d_ptr, N * sizeof(uint32_t)), gcnsim::hipSuccess);
    ASSERT_NE(d_ptr, nullptr);

    // 1. hipMemset (byte)
    EXPECT_EQ(hipMemset(d_ptr, 0x7A, N * sizeof(uint32_t)), gcnsim::hipSuccess);
    std::vector<uint8_t> h_bytes(N * sizeof(uint32_t));
    EXPECT_EQ(hipMemcpy(h_bytes.data(), d_ptr, h_bytes.size(), gcnsim::hipMemcpyDeviceToHost), gcnsim::hipSuccess);
    for (size_t i = 0; i < h_bytes.size(); i++) {
        EXPECT_EQ(h_bytes[i], 0x7A);
    }

    // 2. hipMemsetD32
    EXPECT_EQ(hipMemsetD32(d_ptr, 0x12345678, N), gcnsim::hipSuccess);
    std::vector<uint32_t> h_words(N);
    EXPECT_EQ(hipMemcpy(h_words.data(), d_ptr, N * sizeof(uint32_t), gcnsim::hipMemcpyDeviceToHost), gcnsim::hipSuccess);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(h_words[i], 0x12345678u);
    }

    // 3. hipMemsetD16
    EXPECT_EQ(hipMemsetD16(d_ptr, 0xABCD, N * 2), gcnsim::hipSuccess);
    std::vector<uint16_t> h_shorts(N * 2);
    EXPECT_EQ(hipMemcpy(h_shorts.data(), d_ptr, N * 2 * sizeof(uint16_t), gcnsim::hipMemcpyDeviceToHost), gcnsim::hipSuccess);
    for (size_t i = 0; i < N * 2; i++) {
        EXPECT_EQ(h_shorts[i], 0xABCDu);
    }

    // 4. hipMemsetD8Async
    EXPECT_EQ(hipMemsetD8Async(d_ptr, 0x55, N * sizeof(uint32_t), nullptr), gcnsim::hipSuccess);
    EXPECT_EQ(hipMemcpy(h_bytes.data(), d_ptr, h_bytes.size(), gcnsim::hipMemcpyDeviceToHost), gcnsim::hipSuccess);
    for (size_t i = 0; i < h_bytes.size(); i++) {
        EXPECT_EQ(h_bytes[i], 0x55);
    }

    EXPECT_EQ(hipFree(d_ptr), gcnsim::hipSuccess);
}

TEST_F(HipRuntimeTest, HostAllocAndPointerAttributes) {
    void* h_ptr = nullptr;
    EXPECT_EQ(hipHostAlloc(&h_ptr, 512, 0), gcnsim::hipSuccess);
    ASSERT_NE(h_ptr, nullptr);

    // Can write directly to host-allocated memory
    std::memset(h_ptr, 0x42, 512);

    void* dev_ptr = nullptr;
    EXPECT_EQ(hipHostGetDevicePointer(&dev_ptr, h_ptr, 0), gcnsim::hipSuccess);
    EXPECT_EQ(dev_ptr, h_ptr);

    gcnsim::hipPointerAttribute_t attr;
    EXPECT_EQ(hipPointerGetAttributes(&attr, h_ptr), gcnsim::hipSuccess);
    EXPECT_EQ(attr.devicePointer, h_ptr);
    EXPECT_EQ(attr.hostPointer, h_ptr);

    EXPECT_EQ(hipHostFree(h_ptr), gcnsim::hipSuccess);
}

TEST_F(HipRuntimeTest, MemcpyPeerAnd2D) {
    const size_t count = 64;
    void* dev0 = nullptr;
    void* dev1 = nullptr;

    EXPECT_EQ(hipMalloc(&dev0, count * sizeof(int)), gcnsim::hipSuccess);
    EXPECT_EQ(hipMalloc(&dev1, count * sizeof(int)), gcnsim::hipSuccess);

    std::vector<int> src_data(count);
    for (size_t i = 0; i < count; i++) src_data[i] = static_cast<int>(i * 3 + 7);

    EXPECT_EQ(hipMemcpy(dev0, src_data.data(), count * sizeof(int), gcnsim::hipMemcpyHostToDevice), gcnsim::hipSuccess);

    // Peer memcpy dev0 (dev 0) -> dev1 (dev 1)
    EXPECT_EQ(hipMemcpyPeer(dev1, 1, dev0, 0, count * sizeof(int)), gcnsim::hipSuccess);

    std::vector<int> dst_data(count, 0);
    EXPECT_EQ(hipMemcpy(dst_data.data(), dev1, count * sizeof(int), gcnsim::hipMemcpyDeviceToHost), gcnsim::hipSuccess);
    for (size_t i = 0; i < count; i++) {
        EXPECT_EQ(dst_data[i], src_data[i]);
    }

    // Memcpy 2D
    const size_t width = 4 * sizeof(int);
    const size_t height = 4;
    const size_t spitch = 8 * sizeof(int);
    const size_t dpitch = 8 * sizeof(int);

    std::vector<int> mat_src(32, 0);
    std::vector<int> mat_dst(32, 0);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < 4; x++) {
            mat_src[y * 8 + x] = static_cast<int>(y * 10 + x);
        }
    }

    EXPECT_EQ(hipMemcpy2D(mat_dst.data(), dpitch, mat_src.data(), spitch, width, height, gcnsim::hipMemcpyHostToHost),
              gcnsim::hipSuccess);

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < 4; x++) {
            EXPECT_EQ(mat_dst[y * 8 + x], mat_src[y * 8 + x]);
        }
    }

    EXPECT_EQ(hipFree(dev0), gcnsim::hipSuccess);
    EXPECT_EQ(hipFree(dev1), gcnsim::hipSuccess);
}
