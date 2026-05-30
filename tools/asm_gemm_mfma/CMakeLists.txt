cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping asm_gemm_mfma")
    return()
endif()

set(ASM_OBJ    ${CMAKE_CURRENT_BINARY_DIR}/asm_gemm_mfma.o)
set(ASM_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/asm_gemm_mfma.co)

add_custom_command(
    OUTPUT  ${ASM_OBJ}
    COMMAND ${AMDGPU_CLANG} -x assembler -target amdgcn-amd-amdhsa
            -mcode-object-version=4 -mcpu=gfx90a:xnack- -mwavefrontsize64
            -g -c -o ${ASM_OBJ} ${CMAKE_CURRENT_SOURCE_DIR}/asm_gemm_mfma.s
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/asm_gemm_mfma.s
    COMMENT "Assembling asm_gemm_mfma.s for gfx90a"
)

add_custom_command(
    OUTPUT  ${ASM_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -target amdgcn-amd-amdhsa -mcpu=gfx90a:xnack-
            ${ASM_OBJ} -o ${ASM_OUTPUT}
    DEPENDS ${ASM_OBJ}
    COMMENT "Linking asm_gemm_mfma.co"
)

add_custom_target(asm_gemm_mfma_co ALL DEPENDS ${ASM_OUTPUT})
