find_package(GTest REQUIRED)

# test_elf_loader
add_executable(test_elf_loader test_elf_loader.cpp)
target_link_libraries(test_elf_loader gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_elf_loader COMMAND test_elf_loader
         --hsaco ${CMAKE_BINARY_DIR}/tools/vector_add/vector_add.hsaco)

# test_isa_decoder
add_executable(test_isa_decoder test_isa_decoder.cpp)
target_link_libraries(test_isa_decoder gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_isa_decoder COMMAND test_isa_decoder)

# test_isa_executor
add_executable(test_isa_executor test_isa_executor.cpp)
target_link_libraries(test_isa_executor gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_isa_executor COMMAND test_isa_executor)

# test_wavefront
add_executable(test_wavefront test_wavefront.cpp)
target_link_libraries(test_wavefront gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_wavefront COMMAND test_wavefront)

# test_vector_add (integration test)
add_executable(test_vector_add test_vector_add.cpp)
target_link_libraries(test_vector_add gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_vector_add COMMAND test_vector_add
         --hsaco ${CMAKE_BINARY_DIR}/tools/vector_add/vector_add.hsaco)

# test_gemm_mfma (MFMA 4x4x1 integration test)
add_executable(test_gemm_mfma test_gemm_mfma.cpp)
target_link_libraries(test_gemm_mfma gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_gemm_mfma COMMAND test_gemm_mfma
         --hsaco ${CMAKE_BINARY_DIR}/tools/gemm_mfma/gemm_mfma.hsaco)

# test_mfma_kernels (16x16x1, 16x16x4, 32x32x1, 32x32x2 integration tests)
add_executable(test_mfma_kernels test_mfma_kernels.cpp)
target_link_libraries(test_mfma_kernels gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_mfma_kernels COMMAND test_mfma_kernels
         --hsaco ${CMAKE_BINARY_DIR}/tools/mfma_kernels/mfma_kernels.hsaco)

# test_asm_gemm (hand-written GFX90A assembly MFMA GEMM integration test)
add_executable(test_asm_gemm test_asm_gemm.cpp)
target_link_libraries(test_asm_gemm gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_asm_gemm COMMAND test_asm_gemm
         --hsaco ${CMAKE_BINARY_DIR}/tools/asm_gemm_mfma/asm_gemm_mfma.co)
add_dependencies(test_asm_gemm asm_gemm_mfma_co)

# test_cycle_model
add_executable(test_cycle_model test_cycle_model.cpp)
target_link_libraries(test_cycle_model gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_cycle_model COMMAND test_cycle_model)

# test_cache_model
add_executable(test_cache_model test_cache_model.cpp)
target_link_libraries(test_cache_model gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_cache_model COMMAND test_cache_model)

# test_tensile_gemm (Tensile-style MFMA 16x16x1 GEMM with LDS + barrier)
add_executable(test_tensile_gemm test_tensile_gemm.cpp)
target_link_libraries(test_tensile_gemm gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_tensile_gemm COMMAND test_tensile_gemm
         --hsaco ${CMAKE_BINARY_DIR}/tools/tensile_style_gemm/tensile_style_gemm.co)
add_dependencies(test_tensile_gemm tensile_style_gemm_co)

# test_hijack_vector_add
add_executable(test_hijack_vector_add test_hijack_vector_add.cpp)
target_link_libraries(test_hijack_vector_add gcnsim_hip gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_hijack_vector_add COMMAND test_hijack_vector_add
         --hsaco ${CMAKE_BINARY_DIR}/tools/vector_add/vector_add.hsaco)
add_dependencies(test_hijack_vector_add gcnsim_hip)

# test_hijack_asm_gemm
add_executable(test_hijack_asm_gemm test_hijack_asm_gemm.cpp)
target_link_libraries(test_hijack_asm_gemm gcnsim_hip gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_hijack_asm_gemm COMMAND test_hijack_asm_gemm
         --asm-co ${CMAKE_BINARY_DIR}/tools/asm_gemm_mfma/asm_gemm_mfma.co)
add_dependencies(test_hijack_asm_gemm gcnsim_hip asm_gemm_mfma_co)

# test_debugger
add_executable(test_debugger test_debugger.cpp)
target_link_libraries(test_debugger gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_debugger COMMAND test_debugger)

# test_cov5 (Code Object Version 5 & Memory Fault Handling)
add_executable(test_cov5 test_cov5.cpp)
target_link_libraries(test_cov5 gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_cov5 COMMAND test_cov5
         --hsaco ${CMAKE_BINARY_DIR}/tools/cov5_kernel/cov5_kernel.hsaco)
add_dependencies(test_cov5 cov5_kernel_hsaco)

# test_hip_runtime (HIP Runtime Expansion & Dual-GCD MI200)
add_executable(test_hip_runtime test_hip_runtime.cpp)
target_link_libraries(test_hip_runtime gcnsim_hip gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_hip_runtime COMMAND test_hip_runtime)
add_dependencies(test_hip_runtime gcnsim_hip)
