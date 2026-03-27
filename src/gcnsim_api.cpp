#include "gcnsim/gcnsim.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/execution_engine.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace gcnsim {

std::optional<SimModule> module_load(const std::string& hsaco_path, std::string& err) {
    auto result = load_hsaco(hsaco_path, err);
    if (!result) return std::nullopt;
    SimModule sm;
    sm.mod = std::move(*result);
    return sm;
}

std::optional<SimFunction> module_get_function(SimModule& mod,
                                                const std::string& name,
                                                std::string& err)
{
    for (const KernelInfo& ki : mod.mod.kernels) {
        if (ki.name == name) {
            SimFunction sf;
            sf.kernel = &ki;
            sf.owner  = &mod;
            return sf;
        }
    }
    err = "Kernel not found: " + name;
    return std::nullopt;
}

bool launch_kernel(SimFunction& fn,
                   uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                   uint32_t block_x, uint32_t block_y, uint32_t block_z,
                   void** kernel_params,
                   std::string& err,
                   bool enable_cycle_counting,
                   uint64_t* total_cycles)
{
    if (!fn.kernel) {
        err = "SimFunction has null kernel";
        return false;
    }

    const KernelInfo& ki = *fn.kernel;
    uint32_t kernarg_size = ki.descriptor ? ki.descriptor->kernarg_size : 0;

    // Build kernarg buffer from kernel_params
    // Use arg info from metadata if available; otherwise use natural alignment packing
    std::vector<uint8_t> kernarg_buf;

    if (!ki.args.empty() && kernel_params) {
        // Use metadata offsets and sizes
        uint32_t total_size = kernarg_size;
        if (total_size == 0) {
            // Compute from args
            for (const auto& arg : ki.args) {
                uint32_t end = arg.offset + arg.size;
                if (end > total_size) total_size = end;
            }
        }
        kernarg_buf.resize(total_size, 0);
        for (size_t i = 0; i < ki.args.size(); i++) {
            const auto& arg = ki.args[i];
            if (kernel_params[i] && arg.offset + arg.size <= total_size) {
                memcpy(kernarg_buf.data() + arg.offset, kernel_params[i], arg.size);
            }
        }
    } else if (kernel_params && kernarg_size > 0) {
        // Fallback: pack args with 8-byte alignment for pointers, natural for others
        // We can't know sizes without metadata, so just copy the raw bytes
        // This requires kernarg_size to be correct in the descriptor
        kernarg_buf.resize(kernarg_size, 0);
        // Without metadata we can't properly pack — just zero-fill
        // This is a limitation when metadata is not available
    } else if (kernel_params) {
        // No size info — allocate a reasonable default buffer
        // and try to pack assuming all args are 8-byte (pointer-sized)
        kernarg_buf.resize(256, 0);
    }

    LaunchConfig cfg;
    cfg.grid_x  = grid_x;
    cfg.grid_y  = grid_y;
    cfg.grid_z  = grid_z;
    cfg.block_x = block_x;
    cfg.block_y = block_y;
    cfg.block_z = block_z;
    cfg.kernarg  = kernarg_buf.data();
    cfg.kernarg_size = kernarg_buf.size();
    cfg.enable_cycle_counting = enable_cycle_counting;

    return launch_kernel(*fn.kernel, cfg, err, total_cycles);
}

} // namespace gcnsim
