#include <gtest/gtest.h>
#include "gcnsim/hip_hijack.h"
#include "gcnsim/gcnsim.h"
#include <vector>
#include <fstream>
#include <cstring>
#include <cmath>
#include <string>
#include <cstdlib>

// Stub function representing the host-side compiled kernel address
static void vector_add_stub() {}

// Declaration of registration symbols implemented in gcnsim_hip
extern "C" void** __hipRegisterFatBinary(const void* data);
extern "C" void __hipRegisterFunction(
    void** fatBinHandle,
    const void* hostFun,
    char* deviceFun,
    const char* deviceName,
    int thread_limit,
    void* tid, void* bid, void* gdim, void* bdim,
    int* wsize
);

// Decalared functions from gcnsim_hip for compilation links
extern "C" gcnsim::hipError_t hipMalloc(void** devPtr, size_t size);
extern "C" gcnsim::hipError_t hipFree(void* devPtr);
extern "C" gcnsim::hipError_t hipMemcpy(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind);
extern "C" gcnsim::hipError_t hipStreamCreate(gcnsim::hipStream_t* stream);
extern "C" gcnsim::hipError_t hipStreamDestroy(gcnsim::hipStream_t stream);
extern "C" gcnsim::hipError_t hipStreamSynchronize(gcnsim::hipStream_t stream);
extern "C" gcnsim::hipError_t hipLaunchKernel(
    const void* function_address,
    gcnsim::dim3 numBlocks,
    gcnsim::dim3 dimBlocks,
    void** args,
    size_t sharedMemBytes,
    gcnsim::hipStream_t stream
);
extern "C" gcnsim::hipError_t hipModuleLoad(gcnsim::hipModule_t* module, const char* fname);
extern "C" gcnsim::hipError_t hipModuleGetFunction(gcnsim::hipFunction_t* function, gcnsim::hipModule_t module, const char* name);
extern "C" gcnsim::hipError_t hipModuleUnload(gcnsim::hipModule_t module);
extern "C" gcnsim::hipError_t hipModuleLaunchKernel(
    gcnsim::hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, gcnsim::hipStream_t stream,
    void** kernelParams, void** extra
);

static std::string g_hsaco_path;

class HijackVectorAddTest : public ::testing::Test {
protected:
    std::vector<uint8_t> hsaco_bytes;
    gcnsim::FatBinaryHeader fatbin_header;
    void** fatbin_handle = nullptr;

    void SetUp() override {
        // Load the vector_add.hsaco file into memory
        std::string actual_path = g_hsaco_path;
        std::ifstream f(actual_path, std::ios::binary);
        if (!f) {
            actual_path = "build/tools/vector_add/vector_add.hsaco";
            f.open(actual_path, std::ios::binary);
        }
        if (!f) {
            actual_path = "../tools/vector_add/vector_add.hsaco";
            f.open(actual_path, std::ios::binary);
        }
        if (!f) {
            actual_path = "tools/vector_add/vector_add.hsaco";
            f.open(actual_path, std::ios::binary);
        }
        ASSERT_TRUE(f.is_open()) << "Failed to find vector_add.hsaco";
        g_hsaco_path = actual_path;
        
        f.seekg(0, std::ios::end);
        auto sz = f.tellg();
        f.seekg(0, std::ios::beg);
        
        hsaco_bytes.resize((size_t)sz);
        f.read(reinterpret_cast<char*>(hsaco_bytes.data()), sz);
        ASSERT_GT(hsaco_bytes.size(), 0u);

        // Populate fat binary header
        fatbin_header.magic = 0x46624869; // dummy magic
        fatbin_header.version = 1;
        fatbin_header.binary = hsaco_bytes.data();
        fatbin_header.size = hsaco_bytes.size();

        // Register the fat binary and function stub
        fatbin_handle = __hipRegisterFatBinary(&fatbin_header);
        ASSERT_NE(fatbin_handle, nullptr);

        __hipRegisterFunction(
            fatbin_handle,
            reinterpret_cast<const void*>(vector_add_stub),
            const_cast<char*>("vector_add"),
            "vector_add",
            -1, nullptr, nullptr, nullptr, nullptr, nullptr
        );
    }
};

TEST_F(HijackVectorAddTest, EndToEndHijackLaunch) {
    // Enable hijacking
    setenv("GCNSIM_HIJACK", "1", 1);
    unsetenv("GCNSIM_HIJACK_FILTER");

    // Prepare inputs
    const int N = 256;
    std::vector<float> A(N), B(N), C(N, 0.0f);
    for (int i = 0; i < N; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(N - i);
    }

    // Allocate simulated device memory (routes to malloc)
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    gcnsim::hipError_t err;
    err = hipMalloc(reinterpret_cast<void**>(&d_A), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(d_A, nullptr);

    err = hipMalloc(reinterpret_cast<void**>(&d_B), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(d_B, nullptr);

    err = hipMalloc(reinterpret_cast<void**>(&d_C), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(d_C, nullptr);

    // Copy to dev (memcpy)
    err = hipMemcpy(d_A, A.data(), N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    err = hipMemcpy(d_B, B.data(), N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Create stream
    gcnsim::hipStream_t stream = nullptr;
    err = hipStreamCreate(&stream);
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(stream, nullptr);

    // Launch kernel
    gcnsim::dim3 blocks(1);
    gcnsim::dim3 threads(N);
    void* args[] = { &d_A, &d_B, &d_C, const_cast<int*>(&N) };

    err = hipLaunchKernel(
        reinterpret_cast<const void*>(vector_add_stub),
        blocks,
        threads,
        args,
        0,
        stream
    );
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Sync stream (no-op)
    err = hipStreamSynchronize(stream);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Copy back
    err = hipMemcpy(C.data(), d_C, N * sizeof(float), gcnsim::hipMemcpyDeviceToHost);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Verify calculations
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        EXPECT_NEAR(C[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipStreamDestroy(stream);
}

TEST_F(HijackVectorAddTest, DriverApiHijackLaunch) {
    setenv("GCNSIM_HIJACK", "1", 1);
    unsetenv("GCNSIM_HIJACK_FILTER");

    // Load module dynamically
    gcnsim::hipModule_t module = nullptr;
    gcnsim::hipError_t err = hipModuleLoad(&module, g_hsaco_path.c_str());
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(module, nullptr);

    // Get kernel function
    gcnsim::hipFunction_t function = nullptr;
    err = hipModuleGetFunction(&function, module, "vector_add");
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(function, nullptr);

    // Prepare inputs
    const int N = 128;
    std::vector<float> A(N), B(N), C(N, 0.0f);
    for (int i = 0; i < N; i++) {
        A[i] = (float)(i * 2);
        B[i] = (float)(N + i);
    }

    // Allocate & copy
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    err = hipMalloc(reinterpret_cast<void**>(&d_A), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMalloc(reinterpret_cast<void**>(&d_B), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMalloc(reinterpret_cast<void**>(&d_C), N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);

    err = hipMemcpy(d_A, A.data(), N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMemcpy(d_B, B.data(), N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Launch using hipModuleLaunchKernel (using Mode A: kernelParams)
    void* args[] = { &d_A, &d_B, &d_C, const_cast<int*>(&N) };
    err = hipModuleLaunchKernel(
        function,
        1, 1, 1,      // Grid size (blocks)
        N, 1, 1,      // Block size (threads)
        0, nullptr,
        args, nullptr
    );
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Copy back
    err = hipMemcpy(C.data(), d_C, N * sizeof(float), gcnsim::hipMemcpyDeviceToHost);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Verify
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        EXPECT_NEAR(C[i], expected, 1e-5f) << "Mismatch at index " << i;
    }

    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipModuleUnload(module);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--hsaco" && i + 1 < argc) {
            g_hsaco_path = argv[i + 1];
        }
    }
    if (g_hsaco_path.empty()) {
        g_hsaco_path = "tools/vector_add/vector_add.hsaco";
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
