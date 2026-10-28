#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main(int argc, char* argv[]) {
    int64_t M = 64;
    int64_t N = 64;
    int64_t K = 64;

    if (argc >= 4) {
        M = std::atoll(argv[1]);
        N = std::atoll(argv[2]);
        K = std::atoll(argv[3]);
    }

    std::cout << "========================================================\n"
              << "       hipBLASLt GEMM Benchmark on gcn-sim             \n"
              << "========================================================\n"
              << " Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";

    // 1. Query Device Properties
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << " Target Device:     " << prop.name << "\n"
              << " GCN Architecture:  " << prop.gcnArchName << "\n"
              << " Compute Units:     " << prop.multiProcessorCount << " CUs\n"
              << " Clock Rate:        " << (prop.clockRate / 1000.0) << " MHz\n\n";

    // 2. Initialize hipBLASLt Handle
    hipblasLtHandle_t handle = nullptr;
    hipblasStatus_t status = hipblasLtCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "[-] Failed to initialize hipBLASLt handle (status=" << status << ")\n";
        return 1;
    }
    std::cout << "[+] hipblasLtCreate initialized successfully.\n";

    // 3. Allocate Host & Device Buffers (FP16 elements)
    size_t size_A = M * K * sizeof(uint16_t);
    size_t size_B = K * N * sizeof(uint16_t);
    size_t size_C = M * N * sizeof(uint16_t);
    size_t size_D = M * N * sizeof(uint16_t);

    std::vector<uint16_t> h_A(M * K, 0x3c00); // 1.0 in FP16
    std::vector<uint16_t> h_B(K * N, 0x3c00); // 1.0 in FP16
    std::vector<uint16_t> h_C(M * N, 0x0000); // 0.0 in FP16
    std::vector<uint16_t> h_D(M * N, 0x0000);

    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    hipMalloc(&d_D, size_D);
    std::cout << "[+] Buffers allocated: d_A=" << d_A << ", d_B=" << d_B << ", d_C=" << d_C << ", d_D=" << d_D << "\n";

    hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice);

    // 4. Create Matrix Layouts
    hipblasLtMatrixLayout_t matA = nullptr, matB = nullptr, matC = nullptr, matD = nullptr;
    hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, M, K, M);
    hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, K, N, K);
    hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, M, N, M);
    hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, M, N, M);

    // 5. Create Matmul Descriptor & Preference
    hipblasLtMatmulDesc_t matmulDesc = nullptr;
    hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulPreferenceCreate(&pref);
    size_t maxWorkspaceSize = 32 * 1024 * 1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &maxWorkspaceSize, sizeof(maxWorkspaceSize));

    void* d_workspace = nullptr;
    hipMalloc(&d_workspace, maxWorkspaceSize);
    hipMemset(d_workspace, 0, maxWorkspaceSize);

    // 6. Query Heuristic Algorithm
    hipblasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, matA, matB, matC, matD,
                                             pref, 1, &heuristicResult, &returnedResults);
    std::cout << "[+] Heuristic search: returned " << returnedResults << " candidates (status=" << status << ")\n";

    // 7. Setup Events for Timing
    hipEvent_t start = nullptr, stop = nullptr;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "[+] Launching hipblasLtMatmul...\n";
    hipEventRecord(start, 0);

    status = hipblasLtMatmul(
        handle,
        matmulDesc,
        &alpha,
        d_A, matA,
        d_B, matB,
        &beta,
        d_C, matC,
        d_D, matD,
        (returnedResults > 0) ? &heuristicResult.algo : nullptr,
        d_workspace,
        maxWorkspaceSize,
        0
    );

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "[+] hipblasLtMatmul execution status: " << status << "\n";

    // 8. Calculate Performance Metrics
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tflops = (elapsed_ms > 0.0f) ? (flops / (elapsed_ms * 1e-3 * 1e12)) : 0.0;

    std::cout << "\n================ Benchmark Results ================\n"
              << std::fixed << std::setprecision(4)
              << " Kernel Execution Time: " << elapsed_ms << " ms (" << elapsed_ms * 1000.0f << " us)\n"
              << " Total Floating Ops:    " << flops / 1e9 << " GFLOPs\n"
              << " Measured Performance:  " << tflops << " TFLOPs\n"
              << "====================================================\n";

    // 9. Verify Numerical Results
    hipMemcpy(h_D.data(), d_D, size_D, hipMemcpyDeviceToHost);

    auto half_to_float = [](uint16_t h) -> float {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h & 0x7c00) >> 10;
        uint32_t mant = (h & 0x03ff) << 13;
        if (exp == 0x1f) {
            exp = 0xff;
        } else if (exp == 0) {
            if (mant != 0) {
                float val = std::ldexp((float)mant / (1 << 23), -14);
                return (h & 0x8000) ? -val : val;
            }
            return (h & 0x8000) ? -0.0f : 0.0f;
        } else {
            exp = exp - 15 + 127;
        }
        uint32_t f_bits = sign | (exp << 23) | mant;
        float res;
        std::memcpy(&res, &f_bits, 4);
        return res;
    };

    std::cout << "\n[+] Verifying Numerical Correctness...\n";
    float expected = static_cast<float>(K); // since A[i,k]=1.0, B[k,j]=1.0, alpha=1.0, beta=0.0
    int errors = 0;
    float max_diff = 0.0f;
    for (int64_t i = 0; i < M * N; i++) {
        float val = half_to_float(h_D[i]);
        float diff = std::abs(val - expected);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) {
            if (errors < 5) {
                std::cerr << "[-] Mismatch at index " << i << ": got " << val 
                          << " (hex: 0x" << std::hex << h_D[i] << std::dec << "), expected " << expected << "\n";
            }
            errors++;
        }
    }
    if (errors == 0) {
        std::cout << "[+] Verification PASSED! All " << (M * N) << " elements match expected " 
                  << expected << " (Max diff: " << max_diff << ")\n";
    } else {
        std::cerr << "[-] Verification FAILED! Total errors: " << errors << " / " << (M * N) 
                  << " (Max diff: " << max_diff << ")\n";
    }

    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    hipFree(d_workspace);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatrixLayoutDestroy(matD);
    hipblasLtMatmulDescDestroy(matmulDesc);
    hipblasLtMatmulPreferenceDestroy(pref);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipblasLtDestroy(handle);

    return (status == HIPBLAS_STATUS_SUCCESS) ? 0 : 1;
}
