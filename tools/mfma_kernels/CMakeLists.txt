cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping mfma_kernels.hsaco")
    return()
endif()

set(HSACO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mfma_kernels.hsaco)

add_custom_command(
    OUTPUT  ${HSACO_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -x hip --cuda-device-only --offload-arch=gfx90a
            --no-gpu-bundle-output -O2 -c
            -o ${HSACO_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/mfma_kernels.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mfma_kernels.hip
    COMMENT "Compiling mfma_kernels.hip → mfma_kernels.hsaco"
)

add_custom_target(mfma_kernels_hsaco ALL DEPENDS ${HSACO_OUTPUT})
