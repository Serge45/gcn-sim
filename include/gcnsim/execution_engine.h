#pragma once
#include "elf_loader.h"
#include <cstdint>
#include <functional>
#include "wavefront.h"

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
    std::function<void(WavefrontState& wave)> debug_hook = nullptr; // hook callback for debugger
    uint32_t num_cus = 0;                // number of CUs (0 = auto: 256 for CDNA4 gfx950, 304 for CDNA3 gfx942, 104 for CDNA2 gfx90a, 120 for CDNA1 gfx908)
};

struct KernelStats {
    uint64_t total_cycles = 0;
    uint64_t total_mfma_cycles = 0;
    uint64_t total_valu_cycles = 0;
    uint64_t total_wave_cycles = 0;
    double mfma_utilization = 0.0;     // 0.0 to 1.0
    double valu_utilization = 0.0;     // 0.0 to 1.0
    double compute_intensity = 0.0;    // 0.0 to 1.0 (weighted MFMA + VALU)
    uint32_t effective_clock_khz = 0;  // Effective clock in kHz under active model
};

// Run a kernel on the CPU simulator.
// Returns false on error (sets error_msg).
// If total_cycles is non-null and cfg.enable_cycle_counting is true, CU wall-clock
// time is written (per-SIMD parallel execution with interleaving when >4 waves).
// If stats is non-null, detailed compute intensity & utilization statistics are populated.
bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg,
                   uint64_t* total_cycles = nullptr, KernelStats* stats = nullptr);

} // namespace gcnsim
