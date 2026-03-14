cmake_minimum_required(VERSION 3.20)

find_program(HIPCC hipcc HINTS /opt/rocm/bin)
if(NOT HIPCC)
    message(WARNING "hipcc not found; skipping vector_add.hsaco")
    return()
endif()

find_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler
    HINTS /opt/rocm/llvm/bin)

set(HSACO_BUNDLE   ${CMAKE_CURRENT_BINARY_DIR}/vector_add_bundle.o)
set(HSACO_OUTPUT   ${CMAKE_CURRENT_BINARY_DIR}/vector_add.hsaco)

if(CLANG_OFFLOAD_BUNDLER)
    # Two-step build: hipcc generates a fat bundle, then we extract the AMDGPU ELF
    add_custom_command(
        OUTPUT  ${HSACO_BUNDLE}
        COMMAND ${HIPCC} --offload-arch=gfx90a -c --genco
                ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
                -o ${HSACO_BUNDLE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
        COMMENT "Building vector_add fat bundle for gfx90a"
    )
    add_custom_command(
        OUTPUT  ${HSACO_OUTPUT}
        COMMAND ${CLANG_OFFLOAD_BUNDLER}
                --unbundle --type=o
                --input=${HSACO_BUNDLE}
                --targets=hipv4-amdgcn-amd-amdhsa--gfx90a
                --output=${HSACO_OUTPUT}
        DEPENDS ${HSACO_BUNDLE}
        COMMENT "Extracting gfx90a ELF from fat bundle"
    )
else()
    # Fallback: rely on hipcc directly producing an HSACO (older ROCm)
    add_custom_command(
        OUTPUT  ${HSACO_OUTPUT}
        COMMAND ${HIPCC} --offload-arch=gfx90a -c --genco
                ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
                -o ${HSACO_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.hip
        COMMENT "Building vector_add.hsaco for gfx90a"
    )
endif()

add_custom_target(vector_add_hsaco ALL DEPENDS ${HSACO_OUTPUT})

install(FILES ${HSACO_OUTPUT} DESTINATION share/gcnsim)
