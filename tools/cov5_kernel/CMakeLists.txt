cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    return()
endif()

set(HSACO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cov5_kernel.hsaco)

add_custom_command(
    OUTPUT  ${HSACO_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -x hip --cuda-device-only --offload-arch=gfx90a
            -mcode-object-version=5
            --no-gpu-bundle-output -O2 -g -gdwarf-4 -gz=none -c
            -o ${HSACO_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cov5_kernel.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cov5_kernel.hip
    COMMENT "Compiling cov5_kernel.hip → cov5_kernel.hsaco (COV5)"
)

add_custom_target(cov5_kernel_hsaco ALL DEPENDS ${HSACO_OUTPUT})
