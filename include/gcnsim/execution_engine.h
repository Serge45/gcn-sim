#pragma once
#include "elf_loader.h"
#include <cstdint>

namespace gcnsim {

struct LaunchConfig {
    uint32_t grid_x, grid_y, grid_z;    // total threads in each dim
    uint32_t block_x, block_y, block_z; // workgroup size
    const void* kernarg;                 // pointer to kernarg buffer
    size_t      kernarg_size;
};

// Run a kernel on the CPU simulator.
// Returns false on error (sets error_msg).
bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg);

} // namespace gcnsim
