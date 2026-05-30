cmake_minimum_required(VERSION 3.20)

if(NOT AMDGPU_CLANG)
    message(WARNING "clang with amdgcn support not found; skipping tensile_style_gemm")
    return()
endif()

set(ASM_OBJ    ${CMAKE_CURRENT_BINARY_DIR}/tensile_style_gemm.o)
set(ASM_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tensile_style_gemm.co)

add_custom_command(
    OUTPUT  ${ASM_OBJ}
    COMMAND ${AMDGPU_CLANG} -x assembler -target amdgcn-amd-amdhsa
            -mcode-object-version=4 -mcpu=gfx90a -mwavefrontsize64
            -g -c -o ${ASM_OBJ} ${CMAKE_CURRENT_SOURCE_DIR}/tensile_style_gemm.s
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tensile_style_gemm.s
    COMMENT "Assembling tensile_style_gemm.s for gfx90a"
)

add_custom_command(
    OUTPUT  ${ASM_OUTPUT}
    COMMAND ${AMDGPU_CLANG} -target amdgcn-amd-amdhsa -mcpu=gfx90a
            ${ASM_OBJ} -o ${ASM_OUTPUT}
    DEPENDS ${ASM_OBJ}
    COMMENT "Linking tensile_style_gemm.co"
)

add_custom_target(tensile_style_gemm_co ALL DEPENDS ${ASM_OUTPUT})
