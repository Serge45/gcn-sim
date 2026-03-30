#pragma once
#include "elf_loader.h"
#include <cstdint>

namespace gcnsim {

struct LaunchConfig {
    uint32_t grid_x, grid_y, grid_z;    // total threads in each dim
    uint32_t block_x, block_y, block_z; // workgroup size
    const void* kernarg;                 // pointer to kernarg buffer
    size_t      kernarg_size;
    bool enable_cycle_counting = false;  // if true, wavefronts accumulate cycle_count
    uint32_t global_mem_latency = 300;   // VMEM/FLAT/GLOBAL completion latency in cycles
    uint32_t smem_latency = 20;          // SMEM (scalar load) completion latency in cycles
    const char* trace_path = nullptr;    // if non-null, write Chrome Trace to this file
    bool enable_cu_interleaving = false; // interleave waves within workgroup on shared CU clock
};

// Run a kernel on the CPU simulator.
// Returns false on error (sets error_msg).
// If total_cycles is non-null and cfg.enable_cycle_counting is true:
//   - Without CU interleaving: sum of cycle_count across all wavefronts
//   - With CU interleaving: CU wall-clock time per workgroup (latency hiding reflected)
bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg,
                   uint64_t* total_cycles = nullptr);

} // namespace gcnsim
