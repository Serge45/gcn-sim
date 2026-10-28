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
#include "gcnsim/memory_model.h"

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

WaveStatus run_wavefront(WavefrontState& wave, uint64_t max_insns = 10000ULL);
WaveStatus step_wavefront(WavefrontState& wave);

bool launch_kernel(const KernelInfo& kernel, const LaunchConfig& cfg, std::string& error_msg,
                   uint64_t* total_cycles, KernelStats* stats) {
    if (!kernel.descriptor) {
        error_msg = "Kernel descriptor is null";
        return false;
    }
    if (!kernel.code_ptr) {
        error_msg = "Kernel code pointer is null";
        return false;
    }

    if (total_cycles) *total_cycles = 0;
    if (stats) *stats = KernelStats();

    uint64_t accum_mfma_cycles = 0;
    uint64_t accum_valu_cycles = 0;
    uint64_t accum_wave_cycles = 0;

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

    // Create shared cache model (L1 is per-workgroup, L2 shared, L3 Infinity Cache on CDNA3)
    std::unique_ptr<CacheModel> global_cache;
    if (cfg.enable_cycle_counting) {
        CacheConfig ccfg;
        if (kernel.is_gfx950 || kernel.is_gfx942) {
            ccfg.l1_size_bytes = 32768; // 32 KB per CU
            ccfg.l2_size_bytes = 32 * 1024 * 1024; // 32 MB total L2 across 8 XCDs
            ccfg.enable_l3 = true;
            ccfg.l3_size_bytes = 256 * 1024 * 1024; // 256 MB Infinity Cache
            ccfg.l3_hit_cycles = 180;
        } else if (kernel.is_gfx908) {
            ccfg.l1_size_bytes = 16384; // 16 KB per CU
            ccfg.l2_size_bytes = 8 * 1024 * 1024; // 8 MB L2
            ccfg.enable_l3 = false;
        }
        global_cache = std::make_unique<CacheModel>(ccfg);
    }

    // Determine LDS size from kernel descriptor
    uint32_t lds_bytes = kernel.descriptor->group_segment_fixed_size;

    // Determine scratch (private) size per lane from kernel descriptor
    uint32_t scratch_per_lane = kernel.descriptor->private_segment_fixed_size;
    uint32_t scratch_per_wave = scratch_per_lane * WAVE_SIZE;

    uint32_t num_cus = cfg.num_cus;
    if (num_cus == 0) {
        if (kernel.is_gfx950) num_cus = 256;
        else if (kernel.is_gfx942) num_cus = 304;
        else if (kernel.is_gfx908) num_cus = 120;
        else num_cus = 104;
    }
    std::vector<uint64_t> cu_wall_cycles(num_cus, 0);
    uint32_t current_wg = 0;
    uint32_t total_wgs = wg_count_x * wg_count_y * wg_count_z;
    constexpr uint32_t NUM_SIMDS = 4;

    try {
        if (total_wgs > 1 && waves_per_wg <= NUM_SIMDS) {
            struct ActiveWg {
                uint32_t wg_flat = 0;
                uint32_t wg_x = 0, wg_y = 0, wg_z = 0;
                uint32_t cu_id = 0;
                std::vector<uint8_t> lds_buf;
                std::vector<std::unique_ptr<WavefrontState>> waves;
                std::vector<std::vector<uint8_t>> scratch_bufs;
                std::vector<bool> ended;
                std::vector<bool> at_barrier;
                bool active = false;
            };

            auto init_active_wg = [&](ActiveWg& awg, uint32_t flat_idx) {
                awg.wg_flat = flat_idx;
                awg.wg_x = flat_idx % wg_count_x;
                awg.wg_y = (flat_idx / wg_count_x) % wg_count_y;
                awg.wg_z = flat_idx / (wg_count_x * wg_count_y);
                awg.cu_id = flat_idx % num_cus;
                awg.lds_buf.assign(lds_bytes, 0);
                uint8_t* lds_ptr = lds_bytes > 0 ? awg.lds_buf.data() : nullptr;

                awg.waves.resize(waves_per_wg);
                awg.scratch_bufs.resize(waves_per_wg);
                awg.ended.assign(waves_per_wg, false);
                awg.at_barrier.assign(waves_per_wg, false);

                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    awg.waves[w] = std::make_unique<WavefrontState>();
                    init_wavefront(*awg.waves[w], *kernel.descriptor,
                                   kernarg,
                                   kernel.code_ptr,
                                   kernel.code_size,
                                   awg.wg_x, awg.wg_y, awg.wg_z,
                                   w, waves_per_wg,
                                   block_x, block_y, block_z);
                    awg.waves[w]->lds_ptr  = lds_ptr;
                    awg.waves[w]->lds_size = lds_bytes;
                    awg.waves[w]->is_gfx908 = kernel.is_gfx908;
                    awg.waves[w]->is_gfx942 = kernel.is_gfx942;
                    awg.waves[w]->is_gfx950 = kernel.is_gfx950;
                    awg.waves[w]->accum_offset = kernel.accum_offset;

                    awg.scratch_bufs[w].resize(scratch_per_wave, 0);
                    awg.waves[w]->scratch_ptr      = scratch_per_wave > 0 ? awg.scratch_bufs[w].data() : nullptr;
                    awg.waves[w]->scratch_size     = scratch_per_wave;
                    awg.waves[w]->scratch_per_lane = scratch_per_lane;

                    awg.waves[w]->count_cycles = cfg.enable_cycle_counting;
                    awg.waves[w]->global_mem_latency = cfg.global_mem_latency;
                    awg.waves[w]->smem_latency = cfg.smem_latency;
                    awg.waves[w]->cache = global_cache.get();
                    awg.waves[w]->trace_file = trace_file;
                    awg.waves[w]->trace_wave_id = w;
                    awg.waves[w]->trace_wg_flat_id = awg.wg_flat;
                    awg.waves[w]->debug_hook = cfg.debug_hook;
                }
                awg.active = true;
            };

            uint32_t max_active = std::min(total_wgs, num_cus);
            std::vector<ActiveWg> active_wgs(max_active);
            uint32_t next_launch_idx = 0;
            for (uint32_t i = 0; i < max_active; i++) {
                init_active_wg(active_wgs[i], next_launch_idx++);
            }

            uint32_t completed_wgs = 0;
            while (completed_wgs < total_wgs) {
                for (uint32_t i = 0; i < max_active; i++) {
                    auto& awg = active_wgs[i];
                    if (!awg.active) continue;

                    uint32_t active_waves = 0;
                    uint32_t barrier_count = 0;
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        if (awg.ended[w]) continue;
                        active_waves++;
                        if (awg.at_barrier[w]) {
                            barrier_count++;
                            continue;
                        }
                        WaveStatus status = run_wavefront(*awg.waves[w], 10000ULL);
                        if (status == WaveStatus::ENDED) {
                            awg.ended[w] = true;
                        } else if (status == WaveStatus::BARRIER_HIT) {
                            awg.at_barrier[w] = true;
                            barrier_count++;
                        }
                    }

                    if (barrier_count > 0 && barrier_count == active_waves) {
                        for (uint32_t w = 0; w < waves_per_wg; w++) {
                            awg.at_barrier[w] = false;
                        }
                    }

                    bool all_done = true;
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        if (!awg.ended[w]) { all_done = false; break; }
                    }

                    if (all_done) {
                        if (cfg.enable_cycle_counting) {
                            uint64_t max_cc = 0;
                            for (uint32_t w = 0; w < waves_per_wg; w++) {
                                max_cc = std::max(max_cc, awg.waves[w]->cycle_count);
                            }
                            cu_wall_cycles[awg.cu_id] += max_cc;
                        }
                        for (uint32_t w = 0; w < waves_per_wg; w++) {
                            accum_mfma_cycles += awg.waves[w]->mfma_total_exec_cycles;
                            accum_valu_cycles += awg.waves[w]->valu_total_issue_cycles;
                            accum_wave_cycles += awg.waves[w]->cycle_count;
                            if (trace_file && awg.waves[w]->cycle_count > 0) {
                                double cc = (double)awg.waves[w]->cycle_count;
                                double mfma_util = (double)awg.waves[w]->mfma_total_exec_cycles / cc * 100.0;
                                double valu_util = (double)awg.waves[w]->valu_total_issue_cycles / cc * 100.0;
                                fprintf(trace_file,
                                    ",\n{\"ph\":\"M\",\"pid\":%u,\"tid\":%u,"
                                    "\"name\":\"thread_name\","
                                    "\"args\":{\"name\":\"wave%u/SIMD%u (VALU %.1f%% MFMA %.1f%%)\"}}",
                                    awg.wg_flat, w, w, w % NUM_SIMDS, valu_util, mfma_util);
                            }
                        }

                        completed_wgs++;
                        if (next_launch_idx < total_wgs) {
                            init_active_wg(awg, next_launch_idx++);
                        } else {
                            awg.active = false;
                        }
                    }
                }
            }
        } else {
        for (uint32_t wg_z = 0; wg_z < wg_count_z; wg_z++)
        for (uint32_t wg_y = 0; wg_y < wg_count_y; wg_y++)
        for (uint32_t wg_x = 0; wg_x < wg_count_x; wg_x++) {
        uint32_t cu_id = current_wg % num_cus;
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
            waves[w]->is_gfx908 = kernel.is_gfx908;
            waves[w]->is_gfx942 = kernel.is_gfx942;
            waves[w]->is_gfx950 = kernel.is_gfx950;
            waves[w]->accum_offset = kernel.accum_offset;

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
            waves[w]->debug_hook = cfg.debug_hook;
        }

        // Per-SIMD CU model: gfx90a has 4 SIMDs per CU, each 16-wide.
        // Waves assigned round-robin (wave w → SIMD w%4). Each SIMD runs
        // independently. Interleaving occurs within a SIMD with 2+ waves.
        constexpr uint32_t NUM_SIMDS = 4;
        bool need_interleave = cfg.enable_cycle_counting && waves_per_wg > NUM_SIMDS;

        if (need_interleave) {
            // >4 waves: per-SIMD lockstep scheduling with step_wavefront()
            std::vector<std::vector<uint32_t>> simd_waves(NUM_SIMDS);
            for (uint32_t w = 0; w < waves_per_wg; w++)
                simd_waves[w % NUM_SIMDS].push_back(w);

            uint64_t simd_cycle[NUM_SIMDS] = {};
            uint32_t next_on_simd[NUM_SIMDS] = {};
            std::vector<bool> at_barrier(waves_per_wg, false);

            for (;;) {
                uint32_t active = 0;
                bool all_at_barrier = true;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (ended[w]) continue;
                    active++;
                    if (!at_barrier[w]) all_at_barrier = false;
                }
                if (active == 0) break;

                if (all_at_barrier) {
                    for (uint32_t w = 0; w < waves_per_wg; w++)
                        at_barrier[w] = false;
                    continue;
                }

                bool any_issued = false;
                for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                    auto& sw = simd_waves[s];
                    if (sw.empty()) continue;
                    uint32_t n = (uint32_t)sw.size();

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
                    bool all_done = true;
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        if (!ended[w] && !at_barrier[w]) { all_done = false; break; }
                    }
                    if (all_done && !all_at_barrier) break;
                }
            }

            uint64_t cu_wall = *std::max_element(simd_cycle, simd_cycle + NUM_SIMDS);
            if (cfg.enable_cycle_counting) {
                cu_wall_cycles[cu_id] += cu_wall;
            }
            for (uint32_t w = 0; w < waves_per_wg; w++) {
                accum_mfma_cycles += waves[w]->mfma_total_exec_cycles;
                accum_valu_cycles += waves[w]->valu_total_issue_cycles;
                accum_wave_cycles += waves[w]->cycle_count;
            }
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
            // ≤4 waves (1 per SIMD): run each wave with run_wavefront() (fast path).
            // SIMDs run in parallel; CU wall-clock = max(wave cycle_counts).
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
                // CU wall-clock = max across all waves (each on its own SIMD)
                uint64_t max_cc = 0;
                uint32_t wg_flat = wg_z * wg_count_y * wg_count_x +
                                   wg_y * wg_count_x + wg_x;
                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    max_cc = std::max(max_cc, waves[w]->cycle_count);
                    accum_mfma_cycles += waves[w]->mfma_total_exec_cycles;
                    accum_valu_cycles += waves[w]->valu_total_issue_cycles;
                    accum_wave_cycles += waves[w]->cycle_count;
                    if (trace_file && waves[w]->cycle_count > 0) {
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
                cu_wall_cycles[cu_id] += max_cc;
            }
        }
        current_wg++;
    }
    }
    if (total_cycles && cfg.enable_cycle_counting) {
        *total_cycles = *std::max_element(cu_wall_cycles.begin(), cu_wall_cycles.end());
    }
    if (stats) {
        stats->total_cycles = (total_cycles && cfg.enable_cycle_counting) ? *total_cycles : 0;
        stats->total_mfma_cycles = accum_mfma_cycles;
        stats->total_valu_cycles = accum_valu_cycles;
        stats->total_wave_cycles = accum_wave_cycles;
        stats->mfma_utilization = (accum_wave_cycles > 0)
            ? static_cast<double>(accum_mfma_cycles) / static_cast<double>(accum_wave_cycles)
            : 0.0;
        stats->valu_utilization = (accum_wave_cycles > 0)
            ? static_cast<double>(accum_valu_cycles) / static_cast<double>(accum_wave_cycles)
            : 0.0;
        stats->compute_intensity = std::min(1.0, stats->mfma_utilization + 0.5 * stats->valu_utilization);
    }
    } catch (const MemoryFaultException& ex) {
        error_msg = ex.what();
        if (trace_file) fclose(trace_file);
        return false;
    } catch (const std::exception& ex) {
        error_msg = ex.what();
        if (trace_file) fclose(trace_file);
        return false;
    }

    if (trace_file) {
        fprintf(trace_file, "\n]}\n");
        fclose(trace_file);
    }
    return true;
}

} // namespace gcnsim
