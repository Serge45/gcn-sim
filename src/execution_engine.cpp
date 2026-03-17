#include "gcnsim/execution_engine.h"
#include "gcnsim/wavefront.h"
#include "gcnsim/kernel_descriptor.h"
#include <string>
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>

namespace gcnsim {

// Forward declarations
void init_wavefront(WavefrontState& wave,
                    const KernelDescriptor& kd,
                    const uint8_t* kernarg_ptr,
                    const uint8_t* code_ptr,
                    size_t code_size,
                    uint32_t wg_id_x, uint32_t wg_id_y, uint32_t wg_id_z,
                    uint32_t wave_in_wg, uint32_t waves_per_wg,
                    uint32_t block_x, uint32_t block_y, uint32_t block_z);

void run_wavefront(WavefrontState& wave);

bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg) {
    if (!kernel.descriptor) {
        error_msg = "Kernel descriptor is null";
        return false;
    }
    if (!kernel.code_ptr) {
        error_msg = "Kernel code pointer is null";
        return false;
    }

    uint32_t block_x = std::max(1u, cfg.block_x);
    uint32_t block_y = std::max(1u, cfg.block_y);
    uint32_t block_z = std::max(1u, cfg.block_z);
    uint32_t grid_x  = std::max(1u, cfg.grid_x);
    uint32_t grid_y  = std::max(1u, cfg.grid_y);
    uint32_t grid_z  = std::max(1u, cfg.grid_z);

    uint32_t wg_count_x = (grid_x + block_x - 1) / block_x;
    uint32_t wg_count_y = (grid_y + block_y - 1) / block_y;
    uint32_t wg_count_z = (grid_z + block_z - 1) / block_z;

    uint32_t threads_per_wg = block_x * block_y * block_z;
    uint32_t waves_per_wg   = (threads_per_wg + WAVE_SIZE - 1) / WAVE_SIZE;

    // Build a mutable kernarg copy so we can fill in hidden args.
    // HIP/AMDHSA kernels have "hidden" implicit args after user args that describe
    // the launch configuration (block counts, group sizes, remainders, etc.).
    size_t kernarg_size = kernel.descriptor ? kernel.descriptor->kernarg_size : cfg.kernarg_size;
    if (kernarg_size == 0) kernarg_size = cfg.kernarg_size;
    std::vector<uint8_t> kernarg_buf(kernarg_size, 0);
    if (cfg.kernarg && cfg.kernarg_size > 0) {
        size_t copy_sz = std::min(cfg.kernarg_size, kernarg_size);
        memcpy(kernarg_buf.data(), cfg.kernarg, copy_sz);
    }

    // Fill in hidden args by scanning the arg metadata for known value_kinds
    for (const auto& arg : kernel.args) {
        if (arg.offset + arg.size > kernarg_size) continue;
        uint8_t* dst = kernarg_buf.data() + arg.offset;
        if (arg.value_kind == "hidden_block_count_x") {
            uint32_t v = wg_count_x; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_block_count_y") {
            uint32_t v = wg_count_y; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_block_count_z") {
            uint32_t v = wg_count_z; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_group_size_x") {
            uint32_t v = block_x; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_group_size_y") {
            uint32_t v = block_y; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_group_size_z") {
            uint32_t v = block_z; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_remainder_x") {
            // remainder = (grid_x % block_x) — how many threads in the last partial block
            uint32_t v = grid_x % block_x; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_remainder_y") {
            uint32_t v = grid_y % block_y; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_remainder_z") {
            uint32_t v = grid_z % block_z; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        }
        // hidden_global_offset_* and hidden_grid_dims left as 0 (no offset, not used by vector_add)
    }

    const uint8_t* kernarg = kernarg_buf.data();

    // Determine LDS size from kernel descriptor
    uint32_t lds_bytes = kernel.descriptor->group_segment_fixed_size;

    for (uint32_t wg_z = 0; wg_z < wg_count_z; wg_z++)
    for (uint32_t wg_y = 0; wg_y < wg_count_y; wg_y++)
    for (uint32_t wg_x = 0; wg_x < wg_count_x; wg_x++) {
        // Allocate per-workgroup LDS buffer; shared across all waves in the workgroup
        std::vector<uint8_t> lds_buf(lds_bytes, 0);
        uint8_t* lds_ptr = lds_bytes > 0 ? lds_buf.data() : nullptr;

        for (uint32_t wave = 0; wave < waves_per_wg; wave++) {
            // Allocate on heap — WavefrontState is large, too large for the stack
            auto ws = std::make_unique<WavefrontState>();
            init_wavefront(*ws, *kernel.descriptor,
                           kernarg,
                           kernel.code_ptr,
                           kernel.code_size,
                           wg_x, wg_y, wg_z,
                           wave, waves_per_wg,
                           block_x, block_y, block_z);
            ws->lds_ptr  = lds_ptr;
            ws->lds_size = lds_bytes;
            run_wavefront(*ws);
        }
    }
    return true;
}

} // namespace gcnsim
