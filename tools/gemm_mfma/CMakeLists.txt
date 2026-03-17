cmake_minimum_required(VERSION 3.20)

find_program(HIPCC hipcc HINTS /opt/rocm/bin)
if(NOT HIPCC)
    message(WARNING "hipcc not found; skipping gemm_mfma.hsaco")
    return()
endif()

find_program(CLANG_OFFLOAD_BUNDLER clang-offload-bundler
    HINTS /opt/rocm/llvm/bin)

set(HSACO_BUNDLE  ${CMAKE_CURRENT_BINARY_DIR}/gemm_mfma_bundle.o)
set(HSACO_OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/gemm_mfma.hsaco)

if(CLANG_OFFLOAD_BUNDLER)
    add_custom_command(
        OUTPUT  ${HSACO_BUNDLE}
        COMMAND ${HIPCC} --offload-arch=gfx90a -c --genco
                ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
                -o ${HSACO_BUNDLE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
        COMMENT "Building gemm_mfma fat bundle for gfx90a"
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
    add_custom_command(
        OUTPUT  ${HSACO_OUTPUT}
        COMMAND ${HIPCC} --offload-arch=gfx90a -c --genco
                ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
                -o ${HSACO_OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_mfma.hip
        COMMENT "Building gemm_mfma.hsaco for gfx90a"
    )
endif()

add_custom_target(gemm_mfma_hsaco ALL DEPENDS ${HSACO_OUTPUT})

install(FILES ${HSACO_OUTPUT} DESTINATION share/gcnsim)
