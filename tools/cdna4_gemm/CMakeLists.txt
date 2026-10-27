cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping cdna4_gemm")
    return()
endif()

set(CDNA4_ASM_OBJ    ${CMAKE_CURRENT_BINARY_DIR}/cdna4_gemm.o)
set(CDNA4_ASM_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cdna4_gemm.co)

add_custom_command(
    OUTPUT  ${CDNA4_ASM_OBJ}
    COMMAND ${AMDGPU_CLANG} -x assembler -target amdgcn-amd-amdhsa
            -mcpu=gfx950 -mwavefrontsize64
            -g -c -o ${CDNA4_ASM_OBJ} ${CMAKE_CURRENT_SOURCE_DIR}/cdna4_gemm.s
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cdna4_gemm.s
    COMMENT "Assembling cdna4_gemm.s for gfx950"
)

add_custom_command(
    OUTPUT  ${CDNA4_ASM_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -target amdgcn-amd-amdhsa -mcpu=gfx950
            ${CDNA4_ASM_OBJ} -o ${CDNA4_ASM_OUTPUT}
    DEPENDS ${CDNA4_ASM_OBJ}
    COMMENT "Linking cdna4_gemm.co"
)

add_custom_target(cdna4_gemm_co ALL DEPENDS ${CDNA4_ASM_OUTPUT})
