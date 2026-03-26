cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping vector_add.hsaco")
    return()
endif()

set(HSACO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vector_add.hsaco)

add_custom_command(
    OUTPUT  ${HSACO_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -x hip --cuda-device-only --offload-arch=gfx90a
            --no-gpu-bundle-output -O2 -c
            -o ${HSACO_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
    COMMENT "Compiling vector_add.hip → vector_add.hsaco"
)

add_custom_target(vector_add_hsaco ALL DEPENDS ${HSACO_OUTPUT})

install(FILES ${HSACO_OUTPUT} DESTINATION share/gcnsim)
