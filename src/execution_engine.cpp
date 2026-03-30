#include "gcnsim/execution_engine.h"
#include "gcnsim/wavefront.h"
#include "gcnsim/kernel_descriptor.h"
#include <string>
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <cstdio>
#include "gcnsim/cache_model.h"

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
WaveStatus step_wavefront(WavefrontState& wave);

bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg,
                   uint64_t* total_cycles) {
    if (!kernel.descriptor) {
        error_msg = "Kernel descriptor is null";
        return false;
    }
    if (!kernel.code_ptr) {
        error_msg = "Kernel code pointer is null";
        return false;
    }

    if (total_cycles) *total_cycles = 0;

    // Open trace file if requested (Chrome Trace Event Format)
    FILE* trace_file = nullptr;
    if (cfg.trace_path && cfg.enable_cycle_counting) {
        trace_file = fopen(cfg.trace_path, "w");
        if (trace_file) {
            fprintf(trace_file, "{\"traceEvents\":[\n");
            fprintf(trace_file,
                "{\"ph\":\"M\",\"pid\":0,\"tid\":0,\"name\":\"process_name\","
                "\"args\":{\"name\":\"gcn-sim\"}}");
        }
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

    // Create shared L2 cache model (L1 is per-workgroup)
    std::unique_ptr<CacheModel> global_cache;
    if (cfg.enable_cycle_counting) {
        global_cache = std::make_unique<CacheModel>(CacheConfig{});
    }

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

            waves[w]->count_cycles = cfg.enable_cycle_counting;
            waves[w]->global_mem_latency = cfg.global_mem_latency;
            waves[w]->smem_latency = cfg.smem_latency;
            waves[w]->cache = global_cache.get();
            waves[w]->trace_file = trace_file;
            waves[w]->trace_wave_id = w;
            waves[w]->trace_wg_flat_id =
                wg_z * wg_count_y * wg_count_x + wg_y * wg_count_x + wg_x;
        }

        if (cfg.enable_cu_interleaving && cfg.enable_cycle_counting) {
            // Per-SIMD interleaved execution.
            // gfx90a: 4 SIMDs per CU, each 16-wide. A 64-lane wavefront takes
            // 4 cycles per instruction (16 lanes × 4 passes). Waves are assigned
            // to SIMDs round-robin. Interleaving only occurs within a SIMD when
            // it has 2+ waves; with 1 wave per SIMD there is no interleaving.
            // All 4 SIMDs run in parallel (lockstep outer loop).
            constexpr uint32_t NUM_SIMDS = 4;

            // Assign waves to SIMDs round-robin: wave w → SIMD (w % 4)
            std::vector<std::vector<uint32_t>> simd_waves(NUM_SIMDS);
            for (uint32_t w = 0; w < waves_per_wg; w++)
                simd_waves[w % NUM_SIMDS].push_back(w);

            uint64_t simd_cycle[NUM_SIMDS] = {};
            uint32_t next_on_simd[NUM_SIMDS] = {};
            std::vector<bool> at_barrier(waves_per_wg, false);

            for (;;) {
                // Check if any wave is still active
                uint32_t active = 0;
                bool all_at_barrier = true;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (ended[w]) continue;
                    active++;
                    if (!at_barrier[w]) all_at_barrier = false;
                }
                if (active == 0) break;

                // Release barriers when all active waves have reached one
                if (all_at_barrier) {
                    for (uint32_t w = 0; w < waves_per_wg; w++)
                        at_barrier[w] = false;
                    continue;
                }

                // Lockstep: try to issue one instruction per SIMD
                bool any_issued = false;
                for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                    auto& sw = simd_waves[s];
                    if (sw.empty()) continue;
                    uint32_t n = (uint32_t)sw.size();

                    // Scan from next_on_simd[s] for a ready wave on this SIMD
                    bool issued = false;
                    for (uint32_t i = 0; i < n; i++) {
                        uint32_t idx = (next_on_simd[s] + i) % n;
                        uint32_t w = sw[idx];
                        if (ended[w] || at_barrier[w]) continue;
                        if (waves[w]->cycle_count > simd_cycle[s]) continue;

                        waves[w]->cycle_count = std::max(waves[w]->cycle_count, simd_cycle[s]);
                        WaveStatus status = step_wavefront(*waves[w]);
                        simd_cycle[s] += 4;
                        next_on_simd[s] = (idx + 1) % n;

                        if (status == WaveStatus::ENDED)
                            ended[w] = true;
                        else if (status == WaveStatus::BARRIER_HIT)
                            at_barrier[w] = true;

                        issued = true;
                        any_issued = true;
                        break;
                    }

                    if (!issued) {
                        // Fast-forward this SIMD to its earliest ready wave
                        uint64_t min_ready = UINT64_MAX;
                        for (uint32_t idx = 0; idx < n; idx++) {
                            uint32_t w = sw[idx];
                            if (ended[w] || at_barrier[w]) continue;
                            min_ready = std::min(min_ready, waves[w]->cycle_count);
                        }
                        if (min_ready != UINT64_MAX && min_ready > simd_cycle[s])
                            simd_cycle[s] = min_ready;
                    }
                }

                if (!any_issued) {
                    // Check if all remaining waves are at barrier or ended
                    bool all_done = true;
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        if (!ended[w] && !at_barrier[w]) { all_done = false; break; }
                    }
                    if (all_done && !all_at_barrier) break;
                }
            }

            // CU wall-clock = max across all SIMDs
            uint64_t cu_wall = *std::max_element(simd_cycle, simd_cycle + NUM_SIMDS);
            if (total_cycles)
                *total_cycles += cu_wall;
            if (trace_file) {
                uint32_t wg_flat = wg_z * wg_count_y * wg_count_x +
                                   wg_y * wg_count_x + wg_x;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (waves[w]->cycle_count > 0) {
                        double cc = (double)waves[w]->cycle_count;
                        double mfma_util = (double)waves[w]->mfma_total_exec_cycles / cc * 100.0;
                        double valu_util = (double)waves[w]->valu_total_issue_cycles / cc * 100.0;
                        fprintf(trace_file,
                            ",\n{\"ph\":\"M\",\"pid\":%u,\"tid\":%u,"
                            "\"name\":\"thread_name\","
                            "\"args\":{\"name\":\"wave%u/SIMD%u (VALU %.1f%% MFMA %.1f%%)\"}}",
                            wg_flat, w, w, w % NUM_SIMDS, valu_util, mfma_util);
                    }
                }
            }
        } else {
            // Non-interleaved: run each wavefront to completion/barrier, then cycle.
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

                if (active_count == 0) {
                    any_active = false;
                } else if (barrier_count == 0) {
                    any_active = false;
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        if (!ended[w]) { any_active = true; break; }
                    }
                }
            }

            if (cfg.enable_cycle_counting) {
                uint32_t wg_flat = wg_z * wg_count_y * wg_count_x +
                                   wg_y * wg_count_x + wg_x;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (total_cycles)
                        *total_cycles += waves[w]->cycle_count;
                    if (trace_file && waves[w]->cycle_count > 0) {
                        double cc = (double)waves[w]->cycle_count;
                        double mfma_util = (double)waves[w]->mfma_total_exec_cycles / cc * 100.0;
                        double valu_util = (double)waves[w]->valu_total_issue_cycles / cc * 100.0;
                        fprintf(trace_file,
                            ",\n{\"ph\":\"M\",\"pid\":%u,\"tid\":%u,"
                            "\"name\":\"thread_name\","
                            "\"args\":{\"name\":\"wave%u (VALU %.1f%% MFMA %.1f%%)\"}}",
                            wg_flat, w, w, valu_util, mfma_util);
                    }
                }
            }
        }
    }

    if (trace_file) {
        fprintf(trace_file, "\n]}\n");
        fclose(trace_file);
    }
    return true;
}

} // namespace gcnsim
