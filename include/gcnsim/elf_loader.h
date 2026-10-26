#pragma once
#include "kernel_descriptor.h"
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

namespace gcnsim {

enum class TargetArch {
    GFX908, // CDNA1 (MI-100)
    GFX90A, // CDNA2 (MI-200)
    GFX942, // CDNA3 (MI-300)
};

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
    uint64_t                code_vma;      // absolute Virtual Memory Address (from ELF)
    size_t                  code_size;     // bytes in .text after entry
    std::vector<KernelArgInfo> args;
    const uint8_t*          hsaco_data = nullptr; // pointer to start of HSACO ELF
    size_t                  hsaco_size = 0;       // total size of HSACO ELF
    uint32_t                accum_offset = 0;     // AGPR offset in unified VGPRs (from .amdhsa_accum_offset)
    uint32_t                elf_flags = 0;        // ELF e_flags
    TargetArch              arch = TargetArch::GFX90A;
    bool                    is_gfx908 = false;    // true if GFX908/CDNA1
    bool                    is_gfx942 = false;    // true if GFX942/CDNA3
};

struct HsacoModule {
    std::vector<uint8_t>  elf_data;
    std::vector<KernelInfo> kernels;
    int cov_version;   // 3=COV3, 4=COV4, 5=COV5
    uint32_t elf_flags = 0;
    TargetArch arch = TargetArch::GFX90A;
    bool     is_gfx908 = false;
    bool     is_gfx942 = false;
};

// Load an HSACO ELF file. Returns nullopt on error (sets error_msg).
std::optional<HsacoModule> load_hsaco(const std::string& path, std::string& error_msg);

// Load an HSACO ELF from an in-memory buffer. Returns nullopt on error.
std::optional<HsacoModule> load_hsaco_from_memory(const uint8_t* data, size_t data_size, std::string& error_msg);

} // namespace gcnsim
