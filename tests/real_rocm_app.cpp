#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 256;
    std::vector<float> A(N), B(N), C(N, 0.0f);
    for (int i = 0; i < N; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(N - i);
    }

    std::cout << "[real_rocm_app] Allocating device memory..." << std::endl;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    std::cout << "[real_rocm_app] Copying inputs to device..." << std::endl;
    hipMemcpy(d_A, A.data(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), N * sizeof(float), hipMemcpyHostToDevice);

    std::cout << "[real_rocm_app] Launching vector_add kernel (N=" << N << ")..." << std::endl;
    hipLaunchKernelGGL(vector_add, dim3(1), dim3(N), 0, 0, d_A, d_B, d_C, N);

    std::cout << "[real_rocm_app] Synchronizing..." << std::endl;
    hipDeviceSynchronize();

    std::cout << "[real_rocm_app] Copying results back..." << std::endl;
    hipMemcpy(C.data(), d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = A[i] + B[i];
        if (std::abs(C[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::cerr << "Mismatch at " << i << ": got " << C[i] << ", expected " << expected << std::endl;
            }
            errors++;
        }
    }
    
    std::cout << "[real_rocm_app] Verification: " << (errors == 0 ? "PASSED" : "FAILED") 
              << " (Errors: " << errors << ")" << std::endl;

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return errors == 0 ? 0 : 1;
}
