#include <gtest/gtest.h>
#include "gcnsim/hip_hijack.h"
#include "gcnsim/gcnsim.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
#include <cstdlib>

// Forward declarations of hijacked APIs
extern "C" gcnsim::hipError_t hipMalloc(void** devPtr, size_t size);
extern "C" gcnsim::hipError_t hipFree(void* devPtr);
extern "C" gcnsim::hipError_t hipMemcpy(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind);
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

static std::string g_asm_co_path;

class HijackAsmGemmTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(HijackAsmGemmTest, DriverApiAssemblyGemmHijackLaunch) {
    setenv("GCNSIM_HIJACK", "1", 1);
    unsetenv("GCNSIM_HIJACK_FILTER");

    // Load assembly module dynamically
    gcnsim::hipModule_t module = nullptr;
    gcnsim::hipError_t err = hipModuleLoad(&module, g_asm_co_path.c_str());
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(module, nullptr);

    // Get kernel function
    gcnsim::hipFunction_t function = nullptr;
    err = hipModuleGetFunction(&function, module, "asm_gemm_mfma");
    ASSERT_EQ(err, gcnsim::hipSuccess);
    ASSERT_NE(function, nullptr);

    // Prepare GEMM parameters (M=16, N=16, K=16)
    const int M = 16, N = 16, K = 16;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.f), C_ref(M * N, 0.f);

    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 8 + 1);
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 4 + 1);

    // Allocate & copy
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    err = hipMalloc(reinterpret_cast<void**>(&d_A), M * K * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMalloc(reinterpret_cast<void**>(&d_B), K * N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMalloc(reinterpret_cast<void**>(&d_C), M * N * sizeof(float));
    ASSERT_EQ(err, gcnsim::hipSuccess);

    err = hipMemcpy(d_A, A.data(), M * K * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMemcpy(d_B, B.data(), K * N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);
    err = hipMemcpy(d_C, C.data(), M * N * sizeof(float), gcnsim::hipMemcpyHostToDevice);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Launch using hipModuleLaunchKernel
    void* args[] = { &d_A, &d_B, &d_C, const_cast<int*>(&M), const_cast<int*>(&N), const_cast<int*>(&K) };
    err = hipModuleLaunchKernel(
        function,
        N / 4, M / 4, 1,                  // Grid size (workgroups: 4x4x1)
        4, 1, 1,                          // Block size (threads: 4x1x1)
        0, nullptr,
        args, nullptr
    );
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // Copy back
    err = hipMemcpy(C.data(), d_C, M * N * sizeof(float), gcnsim::hipMemcpyDeviceToHost);
    ASSERT_EQ(err, gcnsim::hipSuccess);

    // CPU reference
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) {
                s += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = s;
        }
    }

    // Verify
    for (int i = 0; i < M * N; i++) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-3f) << "Mismatch at index " << i;
    }

    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipModuleUnload(module);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--asm-co" && i + 1 < argc) {
            g_asm_co_path = argv[i + 1];
        }
    }
    if (g_asm_co_path.empty()) {
        g_asm_co_path = "tools/asm_gemm_mfma/asm_gemm_mfma.co";
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
