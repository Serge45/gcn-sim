cmake_minimum_required(VERSION 3.20)
project(gcn-sim VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(yaml-cpp REQUIRED)

# Find a clang with amdgcn target support (ROCm clang or upstream LLVM with AMDGPU)
find_program(AMDGPU_CLANG clang HINTS /opt/rocm/llvm/bin)

add_library(gcnsim SHARED
    src/elf_loader.cpp
    src/metadata_parser.cpp
    src/isa_decoder.cpp
    src/isa_executor.cpp
    src/wavefront.cpp
    src/execution_engine.cpp
    src/gcnsim_api.cpp
    src/cache_model.cpp
    src/dwarf_parser.cpp
    src/gdb_mi_parser.cpp
    src/debugger_core.cpp)

target_include_directories(gcnsim
    PUBLIC  include)

target_link_libraries(gcnsim PRIVATE yaml-cpp)
target_compile_options(gcnsim PRIVATE -Wall -Wextra)

add_library(gcnsim_hip SHARED
    src/hijack/hip_hijack.cpp)

target_include_directories(gcnsim_hip
    PUBLIC  include)

target_link_libraries(gcnsim_hip PRIVATE gcnsim ${CMAKE_DL_LIBS})
target_link_options(gcnsim_hip PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/hijack/hip.map")
target_compile_options(gcnsim_hip PRIVATE -Wall -Wextra)

set_target_properties(gcnsim_hip PROPERTIES
    OUTPUT_NAME "amdhip64"
    SOVERSION 6
    VERSION 6.1.300
)

add_custom_command(TARGET gcnsim_hip POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:gcnsim_hip> ${CMAKE_BINARY_DIR}/libgcnsim_hip.so
)

option(GCNSIM_BUILD_TESTS "Build tests" ON)
if(GCNSIM_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build the gcndb debugger CLI executable
add_executable(gcndb
    tools/gcndb/gcndb.cpp)
target_link_libraries(gcndb PRIVATE gcnsim)


add_subdirectory(tools/vector_add)
add_subdirectory(tools/gemm_mfma)
add_subdirectory(tools/mfma_kernels)
add_subdirectory(tools/asm_gemm_mfma)
add_subdirectory(tools/tensile_style_gemm)
add_subdirectory(tools/cov5_kernel)
