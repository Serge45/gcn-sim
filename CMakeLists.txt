cmake_minimum_required(VERSION 3.20)
project(gcn-sim VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(yaml-cpp REQUIRED)

add_library(gcnsim SHARED
    src/elf_loader.cpp
    src/metadata_parser.cpp
    src/isa_decoder.cpp
    src/isa_executor.cpp
    src/wavefront.cpp
    src/execution_engine.cpp
    src/gcnsim_api.cpp)

target_include_directories(gcnsim
    PUBLIC  include
    PRIVATE /opt/rocm/include)

target_link_libraries(gcnsim PRIVATE yaml-cpp)
target_compile_options(gcnsim PRIVATE -Wall -Wextra)

option(GCNSIM_BUILD_TESTS "Build tests" ON)
if(GCNSIM_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(tools/vector_add)
add_subdirectory(tools/gemm_mfma)
add_subdirectory(tools/mfma_kernels)
add_subdirectory(tools/asm_gemm_mfma)
