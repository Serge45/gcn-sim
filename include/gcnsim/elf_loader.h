#pragma once
#include "kernel_descriptor.h"
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace gcnsim {

struct KernelArgInfo {
    std::string name;
    std::string type_name;
    uint32_t    size;    // bytes
    uint32_t    offset;  // byte offset within kernarg buffer
    std::string value_kind; // e.g. "global_buffer", "by_value"
};

struct KernelInfo {
    std::string            name;
    const KernelDescriptor* descriptor;    // points into elf_data
    const uint8_t*          code_ptr;      // first instruction
    size_t                  code_size;     // bytes in .text after entry
    std::vector<KernelArgInfo> args;
};

struct HsacoModule {
    std::vector<uint8_t>  elf_data;
    std::vector<KernelInfo> kernels;
    int cov_version;   // 3=COV3, 4=COV4, 5=COV5
};

// Load an HSACO ELF file. Returns nullopt on error (sets error_msg).
std::optional<HsacoModule> load_hsaco(const std::string& path, std::string& error_msg);

// Load an HSACO ELF from an in-memory buffer. Returns nullopt on error.
std::optional<HsacoModule> load_hsaco_from_memory(const uint8_t* data, size_t data_size, std::string& error_msg);

} // namespace gcnsim

