cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping gemm_mfma.hsaco")
    return()
endif()

set(HSACO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gemm_mfma.hsaco)

add_custom_command(
    OUTPUT  ${HSACO_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -x hip --cuda-device-only --offload-arch=gfx90a
            --no-gpu-bundle-output -O2 -c
            -o ${HSACO_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
    COMMENT "Compiling gemm_mfma.hip → gemm_mfma.hsaco"
)

add_custom_target(gemm_mfma_hsaco ALL DEPENDS ${HSACO_OUTPUT})

install(FILES ${HSACO_OUTPUT} DESTINATION share/gcnsim)
