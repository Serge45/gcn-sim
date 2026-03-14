find_package(GTest REQUIRED)

# test_elf_loader
add_executable(test_elf_loader test_elf_loader.cpp)
target_link_libraries(test_elf_loader gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_elf_loader COMMAND test_elf_loader
         --hsaco /opt/rocm-7.3.0/bin/gfx90a/vector_add_memory_fault_kernels.hsaco)

# test_isa_decoder
add_executable(test_isa_decoder test_isa_decoder.cpp)
target_link_libraries(test_isa_decoder gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_isa_decoder COMMAND test_isa_decoder)

# test_wavefront
add_executable(test_wavefront test_wavefront.cpp)
target_link_libraries(test_wavefront gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_wavefront COMMAND test_wavefront)

# test_vector_add (integration test)
add_executable(test_vector_add test_vector_add.cpp)
target_link_libraries(test_vector_add gcnsim GTest::gtest GTest::gtest_main)
add_test(NAME test_vector_add COMMAND test_vector_add
         --hsaco ${CMAKE_BINARY_DIR}/tools/vector_add/vector_add.hsaco)
