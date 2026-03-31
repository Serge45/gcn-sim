#pragma once
// Public C++ API for gcn-sim
#include "elf_loader.h"
#include "execution_engine.h"
#include <memory>
#include <string>
#include <optional>

namespace gcnsim {

struct SimModule {
    HsacoModule mod;
};

struct SimFunction {
    const KernelInfo* kernel;
    SimModule* owner;
};

std::optional<SimModule> module_load(const std::string& hsaco_path, std::string& err);

std::optional<SimFunction> module_get_function(SimModule& mod,
                                                const std::string& name,
                                                std::string& err);

// kernel_params: array of pointers to each argument value, in order.
// If total_cycles is non-null and enable_cycle_counting is true, CU wall-clock
// time is written (per-SIMD parallel execution with interleaving when >4 waves).
bool launch_kernel(SimFunction& fn,
                   uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                   uint32_t block_x, uint32_t block_y, uint32_t block_z,
                   void** kernel_params,
                   std::string& err,
                   bool enable_cycle_counting = false,
                   uint64_t* total_cycles = nullptr,
                   uint32_t global_mem_latency = 300,
                   const char* trace_path = nullptr);

} // namespace gcnsim
