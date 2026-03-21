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

WaveStatus run_wavefront(WavefrontState& wave);

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
            uint32_t v = grid_x % block_x; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_remainder_y") {
            uint32_t v = grid_y % block_y; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        } else if (arg.value_kind == "hidden_remainder_z") {
            uint32_t v = grid_z % block_z; memcpy(dst, &v, std::min((size_t)arg.size, sizeof(v)));
        }
    }

    const uint8_t* kernarg = kernarg_buf.data();

    // Determine LDS size from kernel descriptor
    uint32_t lds_bytes = kernel.descriptor->group_segment_fixed_size;

    // Determine scratch (private) size per lane from kernel descriptor
    uint32_t scratch_per_lane = kernel.descriptor->private_segment_fixed_size;
    uint32_t scratch_per_wave = scratch_per_lane * WAVE_SIZE;

    for (uint32_t wg_z = 0; wg_z < wg_count_z; wg_z++)
    for (uint32_t wg_y = 0; wg_y < wg_count_y; wg_y++)
    for (uint32_t wg_x = 0; wg_x < wg_count_x; wg_x++) {
        // Allocate per-workgroup LDS buffer; shared across all waves in the workgroup
        std::vector<uint8_t> lds_buf(lds_bytes, 0);
        uint8_t* lds_ptr = lds_bytes > 0 ? lds_buf.data() : nullptr;

        // Initialize all wavefronts for this workgroup
        std::vector<std::unique_ptr<WavefrontState>> waves(waves_per_wg);
        std::vector<std::vector<uint8_t>> scratch_bufs(waves_per_wg);
        std::vector<bool> ended(waves_per_wg, false);

        for (uint32_t w = 0; w < waves_per_wg; w++) {
            waves[w] = std::make_unique<WavefrontState>();
            init_wavefront(*waves[w], *kernel.descriptor,
                           kernarg,
                           kernel.code_ptr,
                           kernel.code_size,
                           wg_x, wg_y, wg_z,
                           w, waves_per_wg,
                           block_x, block_y, block_z);
            waves[w]->lds_ptr  = lds_ptr;
            waves[w]->lds_size = lds_bytes;

            scratch_bufs[w].resize(scratch_per_wave, 0);
            waves[w]->scratch_ptr      = scratch_per_wave > 0 ? scratch_bufs[w].data() : nullptr;
            waves[w]->scratch_size     = scratch_per_wave;
            waves[w]->scratch_per_lane = scratch_per_lane;
        }

        // Round-robin execution with barrier synchronization.
        // Run each wavefront until it hits s_barrier or s_endpgm.
        // When all active wavefronts have hit a barrier, resume them all.
        bool any_active = true;
        while (any_active) {
            uint32_t barrier_count = 0;
            uint32_t active_count  = 0;

            for (uint32_t w = 0; w < waves_per_wg; w++) {
                if (ended[w]) continue;
                active_count++;

                WaveStatus status = run_wavefront(*waves[w]);
                if (status == WaveStatus::ENDED) {
                    ended[w] = true;
                } else {
                    barrier_count++;
                }
            }

            // If all active waves hit barrier, they all proceed past it
            // If some ended and others hit barrier, the barrier is satisfied
            // (remaining waves continue)
            if (active_count == 0) {
                any_active = false;
            } else if (barrier_count == 0) {
                // All active waves ended this round — check if any remain
                any_active = false;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (!ended[w]) { any_active = true; break; }
                }
            }
            // else: barrier_count > 0, waves are waiting — loop resumes them
        }
    }
    return true;
}

} // namespace gcnsim
