#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

// CPU reference calculation matching the Tensile-style layout
static void cpu_gemm_tensile(const float* A, const float* B, const float* C,
                              float* D, int M, int N, int K,
                              int LDA, int LDB, int LDC, int LDD,
                              float alpha, float beta)
{
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) {
                s += A[k * LDA + c] * B[r * LDB + k];
            }
            D[r * LDD + c] = alpha * s + beta * C[r * LDC + c];
        }
    }
}

int main(int argc, char** argv) {
    std::string co_path = "build/tools/tensile_style_gemm/tensile_style_gemm.co";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--co" && i + 1 < argc) {
            co_path = argv[i + 1];
        }
    }

    std::cout << "[tensile_client] Loading code object from: " << co_path << std::endl;
    hipModule_t module = nullptr;
    hipError_t err = hipModuleLoad(&module, co_path.c_str());
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << hipGetErrorString(err) << " (code: " << err << ")" << std::endl;
        return 1;
    }

    hipFunction_t function = nullptr;
    err = hipModuleGetFunction(&function, module, "generated_gemm");
    if (err != hipSuccess) {
        std::cerr << "Failed to get function 'generated_gemm': " << hipGetErrorString(err) << std::endl;
        hipModuleUnload(module);
        return 1;
    }

    // Set matrix dimensions for a single tile 128x256x32 GEMM
    const int M = 128;
    const int N = 256;
    const int K = 32;

    const int LDA = K;
    const int LDB = N;
    const int LDC = N;
    const int LDD = N;

    float alpha = 1.0f;
    float beta = 0.0f;

    // A is indexed as A[k * LDA + c]. Max index: 31 * 32 + 255 = 1247. Size 4096 is safe.
    std::vector<float> A(M * LDA);
    // B is indexed as B[r * LDB + k]. Max index: 127 * 256 + 31 = 32543. Needs size 32768.
    std::vector<float> B(M * LDB);
    std::vector<float> C(M * LDC, 0.0f);
    std::vector<float> D_sim(M * LDD, 0.0f);
    std::vector<float> D_ref(M * LDD, 0.0f);

    // Initialize test data
    for (int i = 0; i < M * LDA; i++) A[i] = (float)(i % 7 + 1);
    for (int i = 0; i < M * LDB; i++) B[i] = (float)(i % 5 + 1);

    std::cout << "[tensile_client] Allocating device memory..." << std::endl;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_D = nullptr;

    hipMalloc(&d_A, A.size() * sizeof(float));
    hipMalloc(&d_B, B.size() * sizeof(float));
    hipMalloc(&d_C, C.size() * sizeof(float));
    hipMalloc(&d_D, D_sim.size() * sizeof(float));

    std::cout << "[tensile_client] Copying inputs to device..." << std::endl;
    hipMemcpy(d_A, A.data(), A.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), B.size() * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_C, C.data(), C.size() * sizeof(float), hipMemcpyHostToDevice);

    // Prepare parameters for the generated_gemm kernel
    float* pA = d_A;
    float* pB = d_B;
    float* pC = d_C;
    float* pD = d_D;
    int pM = M, pN = N, pK = K;
    int pLDA = LDA, pLDB = LDB, pLDC = LDC, pLDD = LDD;
    float pAlpha = alpha, pBeta = beta;
    int numWgX = 1, numWgY = 1;

    void* args[] = {
        &pA, &pB, &pC, &pD,
        &pM, &pN, &pK,
        &pLDA, &pLDB, &pLDC, &pLDD,
        &pAlpha, &pBeta,
        &numWgX, &numWgY
    };

    std::cout << "[tensile_client] Launching generated_gemm kernel..." << std::endl;
    err = hipModuleLaunchKernel(
        function,
        1, 1, 1,   // Grid size: exactly 1 block/workgroup
        256, 1, 1, // Block size: 256 threads
        0, nullptr,
        args, nullptr
    );

    if (err != hipSuccess) {
        std::cerr << "Failed to launch kernel: " << hipGetErrorString(err) << std::endl;
    } else {
        std::cout << "[tensile_client] Synchronizing..." << std::endl;
        hipDeviceSynchronize();

        std::cout << "[tensile_client] Copying results back..." << std::endl;
        hipMemcpy(D_sim.data(), d_D, D_sim.size() * sizeof(float), hipMemcpyDeviceToHost);

        // Verification
        std::cout << "[tensile_client] Verifying results..." << std::endl;
        cpu_gemm_tensile(A.data(), B.data(), C.data(), D_ref.data(),
                         M, N, K, LDA, LDB, LDC, LDD, alpha, beta);

        int errors = 0;
        for (int i = 0; i < M * N; i++) {
            float diff = std::abs(D_sim[i] - D_ref[i]);
            if (diff > 1e-2f * std::abs(D_ref[i]) + 1e-2f) {
                if (errors < 5) {
                    std::cerr << "Mismatch at " << i << ": got " << D_sim[i] 
                              << ", expected " << D_ref[i] << std::endl;
                }
                errors++;
            }
        }

        std::cout << "[tensile_client] Verification: " 
                  << (errors == 0 ? "PASSED" : "FAILED") 
                  << " (Errors: " << errors << ")" << std::endl;
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    hipModuleUnload(module);

    return 0;
}
