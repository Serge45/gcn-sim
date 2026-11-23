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

MemoryManager& MemoryManager::Get() {
    static MemoryManager instance;
    return instance;
}

// Forward declarations
void init_wavefront(WavefrontState& wave,
                    const KernelDescriptor& kd,
                    const uint8_t* kernarg_ptr,
                    const uint8_t* code_ptr,
                    size_t code_size,
                    uint32_t wg_id_x, uint32_t wg_id_y, uint32_t wg_id_z,
                    uint32_t wave_in_wg, uint32_t waves_per_wg,
                    uint32_t block_x, uint32_t block_y, uint32_t block_z);

static void set_dispatch_ptr(WavefrontState& wave, const KernelDescriptor& kd, const uint8_t* dispatch_pkt) {
    if ((kd.kernel_code_properties & KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) && wave.scratch_ptr != nullptr) {
        // Scratch SRD using 48-bit sentinel base 0x0000F00200000000ULL (stride=0, non-canonical gap)
        // so mubuf_addr resolves per-lane private memory cleanly even if s[0:3] is copied.
        wave.sgpr[0] = 0x00000000u;
        wave.sgpr[1] = 0x0000F002u;
        wave.sgpr[2] = wave.scratch_per_lane;
        wave.sgpr[3] = 0x00020000u;
    }
    if (!(kd.kernel_code_properties & KCP_ENABLE_SGPR_DISPATCH_PTR)) return;
    int sgpr_idx = (kd.kernel_code_properties & KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) ? 4 : 0;
    uint64_t addr = reinterpret_cast<uint64_t>(dispatch_pkt);
    wave.sgpr[sgpr_idx]     = static_cast<uint32_t>(addr & 0xFFFFFFFFULL);
    wave.sgpr[sgpr_idx + 1] = static_cast<uint32_t>(addr >> 32);
}

WaveStatus run_wavefront(WavefrontState& wave, uint64_t max_insns = 10000ULL);
WaveStatus step_wavefront(WavefrontState& wave);

// Compute CDNA3 (gfx942 / MI300X) max concurrent workgroups per CU from public ISA/ABI limits:
//   - Unified VGPR+AGPR file: 512 regs/SIMD, 4 SIMDs/CU, 8-reg allocation granule in rsrc1[5:0]
//   - Max 8 waves/SIMD (32 waves/CU), max 16 WGs/CU, 64 KB LDS/CU
// Sources: AMD Instinct MI300 / CDNA 3 ISA Reference Guide & ROCm HSA Code Object ABI (public).
static uint32_t compute_cdna3_max_wgs_per_cu(const KernelInfo& kernel,
                                             uint32_t waves_per_wg,
                                             uint32_t lds_bytes) {
    uint32_t rsrc1 = kernel.descriptor ? kernel.descriptor->compute_pgm_rsrc1 : 0u;
    uint32_t vgprs = (rsrc1 != 0u) ? (((rsrc1 & 0x3Fu) + 1u) * 8u) : 64u;
    uint32_t max_waves_per_simd = std::min(8u, 512u / std::max(8u, vgprs));
    uint32_t max_waves_per_cu = std::max(1u, max_waves_per_simd * 4u);
    uint32_t max_wgs_by_waves = std::max(1u, max_waves_per_cu / std::max(1u, waves_per_wg));
    uint32_t max_wgs_by_lds = (lds_bytes > 0u) ? std::max(1u, 65536u / lds_bytes) : 16u;
    return std::min({16u, max_wgs_by_waves, max_wgs_by_lds});
}

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
    const char* trace_path = cfg.trace_path ? cfg.trace_path : std::getenv("GCNSIM_TRACE");
    if (trace_path && cfg.enable_cycle_counting) {
        trace_file = fopen(trace_path, "w");
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
    uint32_t wave_sz = kernel.wavefront_size ? kernel.wavefront_size : WAVE_SIZE;
    if (kernel.is_gfx1201) wave_sz = 32;
    uint32_t waves_per_wg   = (threads_per_wg + wave_sz - 1) / wave_sz;

    // Build a mutable kernarg copy so we can fill in hidden args.
    // Pad by 64 bytes so wide SMEM loads (s_load_dwordx4/x8/x16) near the tail don't OOB.
    size_t kernarg_size = kernel.descriptor ? kernel.descriptor->kernarg_size : cfg.kernarg_size;
    if (kernarg_size == 0) kernarg_size = cfg.kernarg_size;
    std::vector<uint8_t> kernarg_buf(kernarg_size + 64, 0);
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

    struct ScopedMemRegistrations {
        std::vector<uint64_t> bases;
        void add(const void* ptr, size_t sz, const char* tag) {
            if (ptr && sz > 0 && MemoryManager::Get().IsStrictMode()) {
                uint64_t b = reinterpret_cast<uint64_t>(ptr);
                MemoryManager::Get().RegisterRegion(b, sz, tag);
                bases.push_back(b);
            }
        }
        ~ScopedMemRegistrations() {
            for (uint64_t b : bases) {
                MemoryManager::Get().UnregisterRegion(b);
            }
        }
    } mem_guard;
    mem_guard.add(kernarg, kernarg_size, "kernarg");

    // Build HSA AQL kernel dispatch packet for kernels with KCP_ENABLE_SGPR_DISPATCH_PTR
    uint8_t dispatch_pkt[64] = {};
    {
        uint16_t header = 0x0102;
        uint16_t setup = 3;
        uint16_t wg_x = static_cast<uint16_t>(block_x);
        uint16_t wg_y = static_cast<uint16_t>(block_y);
        uint16_t wg_z = static_cast<uint16_t>(block_z);
        uint32_t priv_sz = kernel.descriptor->private_segment_fixed_size;
        uint32_t grp_sz = kernel.descriptor->group_segment_fixed_size;
        uint64_t ka_addr = reinterpret_cast<uint64_t>(kernarg);
        std::memcpy(dispatch_pkt + 0x00, &header, 2);
        std::memcpy(dispatch_pkt + 0x02, &setup, 2);
        std::memcpy(dispatch_pkt + 0x04, &wg_x, 2);
        std::memcpy(dispatch_pkt + 0x06, &wg_y, 2);
        std::memcpy(dispatch_pkt + 0x08, &wg_z, 2);
        std::memcpy(dispatch_pkt + 0x0C, &grid_x, 4);
        std::memcpy(dispatch_pkt + 0x10, &grid_y, 4);
        std::memcpy(dispatch_pkt + 0x14, &grid_z, 4);
        std::memcpy(dispatch_pkt + 0x18, &priv_sz, 4);
        std::memcpy(dispatch_pkt + 0x1C, &grp_sz, 4);
        std::memcpy(dispatch_pkt + 0x28, &ka_addr, 8);
    }
    mem_guard.add(dispatch_pkt, sizeof(dispatch_pkt), "dispatch_pkt");

    uint32_t num_cus = cfg.num_cus;
    if (num_cus == 0) {
        if (kernel.is_gfx950) num_cus = 256;
        else if (kernel.is_gfx942) num_cus = 304;
        else if (kernel.is_gfx908) num_cus = 120;
        else if (kernel.is_gfx1201) num_cus = 32; // 32 WGPs (64 CUs) on Radeon RX 9070 XT
        else num_cus = 104;
    }

    // Create cache model (per-CU L1, shared/per-XCD L2, L3 Infinity Cache / MALL)
    std::unique_ptr<CacheModel> global_cache;
    if (cfg.enable_cycle_counting) {
        CacheConfig ccfg;
        if (kernel.is_gfx942) {
            // CDNA3 (MI300X) public cache & memory hierarchy:
            //   - Official AMD CDNA 3 Whitepaper / MI300X Datasheet:
            //     304 CUs (8 XCDs x 38 CUs), 32 KB vL1D/CU, 4 MB L2/XCD (32 MB total),
            //     256 MB shared Infinity Cache (MALL), 5.3 TB/s HBM3 @ 2.1 GHz.
            //   - Unloaded latencies (L1=36, L2=160, L3=270, HBM=440 cyc) are approximate
            //     defaults from public MI300X microbenchmark reports (override HBM via
            //     GCN_SIM_HBM_LATENCY).
            ccfg.l1_size_bytes = 32768;
            ccfg.l1_ways = 4;
            ccfg.l1_line_bytes = 128;
            ccfg.l1_hit_cycles = 36;
            ccfg.num_xcds = 8;
            ccfg.l2_size_bytes = 4 * 1024 * 1024;
            ccfg.l2_ways = 16;
            ccfg.l2_line_bytes = 128;
            ccfg.l2_hit_cycles = 160;
            ccfg.enable_l3 = true;
            ccfg.l3_size_bytes = 256 * 1024 * 1024;
            ccfg.l3_ways = 16;
            ccfg.l3_line_bytes = 128;
            ccfg.l3_hit_cycles = 270;
            ccfg.enable_bw_model = true;
            uint32_t hbm_lat = (cfg.global_mem_latency == 300) ? 440 : cfg.global_mem_latency;
            if (const char* env_hbm = std::getenv("GCN_SIM_HBM_LATENCY")) {
                int v = std::atoi(env_hbm);
                if (v > 0) hbm_lat = static_cast<uint32_t>(v);
            }
            ccfg.hbm_base_cycles = hbm_lat;
        } else if (kernel.is_gfx950) {
            ccfg.l1_size_bytes = 32768; // 32 KB per CU
            ccfg.l2_size_bytes = 32 * 1024 * 1024; // 32 MB total L2 across 8 XCDs
            ccfg.enable_l3 = true;
            ccfg.l3_size_bytes = 256 * 1024 * 1024; // 256 MB Infinity Cache
            ccfg.l3_hit_cycles = 180;
        } else if (kernel.is_gfx1201) {
            ccfg.l1_size_bytes = 32768; // 32 KB L0/L1 per WGP
            ccfg.l2_size_bytes = 8 * 1024 * 1024; // 8 MB L2
            ccfg.enable_l3 = true;
            ccfg.l3_size_bytes = 64 * 1024 * 1024; // 64 MB 3rd-Gen Infinity Cache
            ccfg.l3_hit_cycles = 160;
        } else if (kernel.is_gfx908) {
            ccfg.l1_size_bytes = 16384; // 16 KB per CU
            ccfg.l2_size_bytes = 8 * 1024 * 1024; // 8 MB L2
            ccfg.enable_l3 = false;
        }
        global_cache = std::make_unique<CacheModel>(ccfg, num_cus);
    }

    // Determine LDS size from kernel descriptor + dynamic shared memory
    uint32_t lds_bytes = kernel.descriptor->group_segment_fixed_size + cfg.shared_mem_bytes;

    // Determine scratch (private) size per lane from kernel descriptor
    uint32_t scratch_per_lane = kernel.descriptor->private_segment_fixed_size;
    uint32_t scratch_per_wave = scratch_per_lane * wave_sz;

    std::vector<uint64_t> cu_wall_cycles(num_cus, 0);
    uint32_t current_wg = 0;
    uint32_t total_wgs = wg_count_x * wg_count_y * wg_count_z;
    constexpr uint32_t NUM_SIMDS = 4;

    auto sync_barrier_cycles = [&](std::vector<std::unique_ptr<WavefrontState>>& waves,
                                   const std::vector<bool>& ended) -> uint64_t {
        uint64_t max_barrier_cc = 0;
        for (uint32_t w = 0; w < waves_per_wg; w++) {
            if (!ended[w]) {
                max_barrier_cc = std::max(max_barrier_cc, waves[w]->cycle_count);
            }
        }
        if (cfg.enable_cycle_counting) {
            for (uint32_t w = 0; w < waves_per_wg; w++) {
                if (ended[w]) continue;
                if (max_barrier_cc > waves[w]->cycle_count) {
                    uint64_t stall_dur = max_barrier_cc - waves[w]->cycle_count;
                    if (trace_file) {
                        fprintf(trace_file,
                            ",\n{\"ph\":\"X\",\"pid\":%u,\"tid\":%u,"
                            "\"cat\":\"STALL\",\"name\":\"stall:barrier\","
                            "\"ts\":%llu,\"dur\":%llu,"
                            "\"args\":{\"pc\":\"0x%llx\"}}",
                            waves[w]->trace_wg_flat_id, waves[w]->trace_wave_id,
                            (unsigned long long)waves[w]->cycle_count,
                            (unsigned long long)stall_dur,
                            (unsigned long long)waves[w]->pc);
                    }
                    waves[w]->cycle_count = max_barrier_cc;
                }
            }
        }
        return max_barrier_cc;
    };

    uint32_t max_wgs_per_cu = kernel.is_gfx942
        ? compute_cdna3_max_wgs_per_cu(kernel, waves_per_wg, lds_bytes)
        : 1u;

    try {
        if (kernel.is_gfx942 && cfg.enable_cycle_counting) {
            // CDNA3 time-windowed concurrent multi-WG CU scheduler
            struct ResidentWg {
                uint32_t wg_flat = 0;
                uint32_t wg_x = 0, wg_y = 0, wg_z = 0;
                uint32_t cu_id = 0;
                uint64_t start_cycle = 0;
                std::vector<uint8_t> lds_buf;
                std::vector<std::unique_ptr<WavefrontState>> waves;
                std::vector<std::vector<uint8_t>> scratch_bufs;
                std::vector<bool> ended;
                std::vector<bool> at_barrier;
            };
            struct CuState {
                std::vector<uint32_t> queue;
                size_t q_head = 0;
                std::vector<ResidentWg> resident;
                uint64_t simd_cycle[NUM_SIMDS] = {};
                uint32_t simd_next[NUM_SIMDS] = {};
            };

            std::vector<CuState> cus(num_cus);
            const char* xcd_sched = std::getenv("GCN_SIM_XCD_SCHED");
            bool chunk_sched = (xcd_sched && std::string(xcd_sched) == "chunk");
            constexpr uint32_t NUM_XCDS = 8;
            uint32_t cus_per_xcd = std::max(1u, num_cus / NUM_XCDS);

            for (uint32_t wg = 0; wg < total_wgs; wg++) {
                uint32_t xcd = chunk_sched
                    ? ((wg * NUM_XCDS) / std::max(1u, total_wgs)) % NUM_XCDS
                    : (wg % NUM_XCDS);
                uint32_t cu = chunk_sched
                    ? (xcd * cus_per_xcd + (wg % cus_per_xcd)) % num_cus
                    : (xcd * cus_per_xcd + ((wg / NUM_XCDS) % cus_per_xcd)) % num_cus;
                cus[cu].queue.push_back(wg);
            }

            auto spawn_wg = [&](uint32_t cu_id, uint32_t flat_idx, uint64_t start_cycle) -> ResidentWg {
                ResidentWg rwg;
                rwg.wg_flat = flat_idx;
                rwg.wg_x = flat_idx % wg_count_x;
                rwg.wg_y = (flat_idx / wg_count_x) % wg_count_y;
                rwg.wg_z = flat_idx / (wg_count_x * wg_count_y);
                rwg.cu_id = cu_id;
                rwg.start_cycle = start_cycle;
                rwg.lds_buf.assign(lds_bytes, 0);
                uint8_t* lds_ptr = lds_bytes > 0 ? rwg.lds_buf.data() : nullptr;

                rwg.waves.resize(waves_per_wg);
                rwg.scratch_bufs.resize(waves_per_wg);
                rwg.ended.assign(waves_per_wg, false);
                rwg.at_barrier.assign(waves_per_wg, false);

                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    rwg.waves[w] = std::make_unique<WavefrontState>();
                    init_wavefront(*rwg.waves[w], *kernel.descriptor,
                                   kernarg,
                                   kernel.code_ptr,
                                   kernel.code_size,
                                   rwg.wg_x, rwg.wg_y, rwg.wg_z,
                                   w, waves_per_wg,
                                   block_x, block_y, block_z);
                    set_dispatch_ptr(*rwg.waves[w], *kernel.descriptor, dispatch_pkt);
                    rwg.waves[w]->lds_ptr = lds_ptr;
                    rwg.waves[w]->lds_size = lds_bytes;
                    rwg.waves[w]->is_gfx942 = true;
                    rwg.waves[w]->wave_size = wave_sz;
                    rwg.waves[w]->accum_offset = kernel.accum_offset;

                    rwg.scratch_bufs[w].resize(scratch_per_wave, 0);
                    rwg.waves[w]->scratch_ptr = scratch_per_wave > 0 ? rwg.scratch_bufs[w].data() : nullptr;
                    rwg.waves[w]->scratch_size = scratch_per_wave;
                    rwg.waves[w]->scratch_per_lane = scratch_per_lane;
                    set_dispatch_ptr(*rwg.waves[w], *kernel.descriptor, dispatch_pkt);

                    rwg.waves[w]->count_cycles = true;
                    rwg.waves[w]->cycle_count = start_cycle;
                    rwg.waves[w]->global_mem_latency = cfg.global_mem_latency;
                    rwg.waves[w]->smem_latency = cfg.smem_latency;
                    rwg.waves[w]->cache = global_cache.get();
                    rwg.waves[w]->cu_id = cu_id;
                    rwg.waves[w]->waves_per_wg = waves_per_wg;
                    rwg.waves[w]->trace_file = trace_file;
                    rwg.waves[w]->trace_wave_id = w;
                    rwg.waves[w]->trace_wg_flat_id = flat_idx;
                    rwg.waves[w]->debug_hook = cfg.debug_hook;
                }
                return rwg;
            };

            std::vector<uint32_t> active_cus;
            for (uint32_t c = 0; c < num_cus; c++) {
                while (cus[c].resident.size() < max_wgs_per_cu && cus[c].q_head < cus[c].queue.size()) {
                    cus[c].resident.push_back(spawn_wg(c, cus[c].queue[cus[c].q_head++], 0ULL));
                }
                if (!cus[c].resident.empty()) {
                    active_cus.push_back(c);
                }
            }

            uint32_t completed_wgs = 0;
            uint64_t sim_clock = 0;
            constexpr uint64_t WINDOW_CYCLES = 64ULL;

            while (completed_wgs < total_wgs) {
                uint64_t window_end = sim_clock + WINDOW_CYCLES;
                uint64_t next_global_ready = UINT64_MAX;
                bool any_progress = false;

                for (size_t ci = 0; ci < active_cus.size(); ) {
                    uint32_t c = active_cus[ci];
                    CuState& cu = cus[c];

                    for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                        if (cu.simd_cycle[s] < sim_clock) cu.simd_cycle[s] = sim_clock;
                    }

                    struct WaveSlot { size_t slot; uint32_t w; };
                    std::vector<WaveSlot> simd_waves[NUM_SIMDS];
                    for (size_t slot = 0; slot < cu.resident.size(); slot++) {
                        for (uint32_t w = 0; w < waves_per_wg; w++) {
                            if (cu.resident[slot].ended[w] || cu.resident[slot].at_barrier[w]) continue;
                            uint32_t s = static_cast<uint32_t>((slot * waves_per_wg + w) % NUM_SIMDS);
                            simd_waves[s].push_back({slot, w});
                        }
                    }

                    for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                        auto& sw = simd_waves[s];
                        if (sw.empty()) continue;
                        uint32_t n = static_cast<uint32_t>(sw.size());
                        while (cu.simd_cycle[s] < window_end) {
                            bool issued = false;
                            for (uint32_t i = 0; i < n; i++) {
                                uint32_t idx = (cu.simd_next[s] + i) % n;
                                auto& rwg = cu.resident[sw[idx].slot];
                                uint32_t w = sw[idx].w;
                                if (rwg.ended[w] || rwg.at_barrier[w]) continue;
                                auto& wave = *rwg.waves[w];
                                if (wave.cycle_count > cu.simd_cycle[s]) continue;

                                wave.cycle_count = std::max(wave.cycle_count, cu.simd_cycle[s]);
                                WaveStatus status = step_wavefront(wave);
                                cu.simd_cycle[s] += 4;
                                cu.simd_next[s] = (idx + 1) % n;
                                any_progress = true;
                                issued = true;

                                if (status == WaveStatus::ENDED) {
                                    rwg.ended[w] = true;
                                } else if (status == WaveStatus::BARRIER_HIT) {
                                    rwg.at_barrier[w] = true;
                                }
                                break;
                            }
                            if (!issued) {
                                uint64_t min_ready = UINT64_MAX;
                                for (uint32_t idx = 0; idx < n; idx++) {
                                    auto& rwg = cu.resident[sw[idx].slot];
                                    uint32_t w = sw[idx].w;
                                    if (rwg.ended[w] || rwg.at_barrier[w]) continue;
                                    min_ready = std::min(min_ready, rwg.waves[w]->cycle_count);
                                }
                                if (min_ready != UINT64_MAX) {
                                    if (min_ready < window_end && min_ready > cu.simd_cycle[s]) {
                                        cu.simd_cycle[s] = min_ready;
                                        any_progress = true;
                                    } else {
                                        next_global_ready = std::min(next_global_ready, min_ready);
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    for (size_t slot = 0; slot < cu.resident.size(); ) {
                        auto& rwg = cu.resident[slot];
                        uint32_t active_w = 0, barrier_w = 0;
                        for (uint32_t w = 0; w < waves_per_wg; w++) {
                            if (rwg.ended[w]) continue;
                            active_w++;
                            if (rwg.at_barrier[w]) barrier_w++;
                        }
                        if (barrier_w > 0 && barrier_w == active_w) {
                            uint64_t b_cc = sync_barrier_cycles(rwg.waves, rwg.ended);
                            for (uint32_t w = 0; w < waves_per_wg; w++) rwg.at_barrier[w] = false;
                            next_global_ready = std::min(next_global_ready, b_cc);
                            any_progress = true;
                        }
                        if (active_w == 0) {
                            uint64_t wg_end = 0;
                            for (uint32_t w = 0; w < waves_per_wg; w++) {
                                wg_end = std::max(wg_end, rwg.waves[w]->cycle_count);
                                uint64_t wave_dur = (rwg.waves[w]->cycle_count >= rwg.start_cycle)
                                    ? (rwg.waves[w]->cycle_count - rwg.start_cycle)
                                    : rwg.waves[w]->cycle_count;
                                accum_mfma_cycles += rwg.waves[w]->mfma_total_exec_cycles;
                                accum_valu_cycles += rwg.waves[w]->valu_total_issue_cycles;
                                accum_wave_cycles += wave_dur;
                                if (trace_file && wave_dur > 0) {
                                    double cc = static_cast<double>(wave_dur);
                                    double mfma_util = static_cast<double>(rwg.waves[w]->mfma_total_exec_cycles) / cc * 100.0;
                                    double valu_util = static_cast<double>(rwg.waves[w]->valu_total_issue_cycles) / cc * 100.0;
                                    fprintf(trace_file,
                                        ",\n{\"ph\":\"M\",\"pid\":%u,\"tid\":%u,"
                                        "\"name\":\"thread_name\","
                                        "\"args\":{\"name\":\"wave%u/SIMD%u (VALU %.1f%% MFMA %.1f%%)\"}}",
                                        rwg.wg_flat, w, w, static_cast<uint32_t>((slot * waves_per_wg + w) % NUM_SIMDS),
                                        valu_util, mfma_util);
                                }
                            }
                            cu_wall_cycles[c] = std::max(cu_wall_cycles[c], wg_end);
                            completed_wgs++;
                            any_progress = true;
                            cu.resident.erase(cu.resident.begin() + static_cast<ptrdiff_t>(slot));
                            if (cu.q_head < cu.queue.size()) {
                                uint64_t spawn_cc = std::max(sim_clock, *std::min_element(cu.simd_cycle, cu.simd_cycle + NUM_SIMDS));
                                cu.resident.push_back(spawn_wg(c, cu.queue[cu.q_head++], spawn_cc));
                            }
                            continue;
                        }
                        slot++;
                    }

                    if (cu.resident.empty() && cu.q_head >= cu.queue.size()) {
                        active_cus.erase(active_cus.begin() + static_cast<ptrdiff_t>(ci));
                    } else {
                        ci++;
                    }
                }

                if (any_progress) {
                    sim_clock = window_end;
                } else if (next_global_ready != UINT64_MAX && next_global_ready > sim_clock) {
                    sim_clock = next_global_ready;
                } else {
                    sim_clock = window_end;
                }
            }
        } else if (total_wgs > 1) {
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
                    set_dispatch_ptr(*awg.waves[w], *kernel.descriptor, dispatch_pkt);
                    awg.waves[w]->lds_ptr  = lds_ptr;
                    awg.waves[w]->lds_size = lds_bytes;
                    awg.waves[w]->is_gfx908 = kernel.is_gfx908;
                    awg.waves[w]->is_gfx942 = kernel.is_gfx942;
                    awg.waves[w]->is_gfx950 = kernel.is_gfx950;
                    awg.waves[w]->is_gfx1201 = kernel.is_gfx1201;
                    awg.waves[w]->wave_size = wave_sz;
                    awg.waves[w]->accum_offset = kernel.accum_offset;

                    awg.scratch_bufs[w].resize(scratch_per_wave, 0);
                    awg.waves[w]->scratch_ptr      = scratch_per_wave > 0 ? awg.scratch_bufs[w].data() : nullptr;
                    awg.waves[w]->scratch_size     = scratch_per_wave;
                    awg.waves[w]->scratch_per_lane = scratch_per_lane;
                    set_dispatch_ptr(*awg.waves[w], *kernel.descriptor, dispatch_pkt);

                    awg.waves[w]->count_cycles = cfg.enable_cycle_counting;
                    awg.waves[w]->global_mem_latency = cfg.global_mem_latency;
                    awg.waves[w]->smem_latency = cfg.smem_latency;
                    awg.waves[w]->cache = global_cache.get();
                    awg.waves[w]->cu_id = awg.cu_id;
                    awg.waves[w]->waves_per_wg = waves_per_wg;
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
                        sync_barrier_cycles(awg.waves, awg.ended);
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
                            if (waves_per_wg <= NUM_SIMDS) {
                                for (uint32_t w = 0; w < waves_per_wg; w++) {
                                    max_cc = std::max(max_cc, awg.waves[w]->cycle_count);
                                }
                            } else {
                                uint64_t simd_max_cc[NUM_SIMDS] = {};
                                uint64_t simd_issue_sum[NUM_SIMDS] = {};
                                for (uint32_t w = 0; w < waves_per_wg; w++) {
                                    uint32_t s = w % NUM_SIMDS;
                                    simd_max_cc[s] = std::max(simd_max_cc[s], awg.waves[w]->cycle_count);
                                    simd_issue_sum[s] += awg.waves[w]->insn_count * 4;
                                }
                                for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                                    max_cc = std::max(max_cc, std::max(simd_max_cc[s], simd_issue_sum[s]));
                                }
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
            set_dispatch_ptr(*waves[w], *kernel.descriptor, dispatch_pkt);
            waves[w]->lds_ptr  = lds_ptr;
            waves[w]->lds_size = lds_bytes;
            waves[w]->is_gfx908 = kernel.is_gfx908;
            waves[w]->is_gfx942 = kernel.is_gfx942;
            waves[w]->is_gfx950 = kernel.is_gfx950;
            waves[w]->is_gfx1201 = kernel.is_gfx1201;
            waves[w]->wave_size = wave_sz;
            waves[w]->accum_offset = kernel.accum_offset;

            scratch_bufs[w].resize(scratch_per_wave, 0);
            waves[w]->scratch_ptr      = scratch_per_wave > 0 ? scratch_bufs[w].data() : nullptr;
            waves[w]->scratch_size     = scratch_per_wave;
            waves[w]->scratch_per_lane = scratch_per_lane;
            set_dispatch_ptr(*waves[w], *kernel.descriptor, dispatch_pkt);

            waves[w]->count_cycles = cfg.enable_cycle_counting;
            waves[w]->global_mem_latency = cfg.global_mem_latency;
            waves[w]->smem_latency = cfg.smem_latency;
            waves[w]->cache = global_cache.get();
            waves[w]->cu_id = cu_id;
            waves[w]->waves_per_wg = waves_per_wg;
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
                    uint64_t max_b_cc = sync_barrier_cycles(waves, ended);
                    for (uint32_t s = 0; s < NUM_SIMDS; s++) {
                        simd_cycle[s] = std::max(simd_cycle[s], max_b_cc);
                    }
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
            std::vector<bool> at_barrier(waves_per_wg, false);
            for (;;) {
                uint32_t active_count  = 0;
                uint32_t barrier_count = 0;

                for (uint32_t w = 0; w < waves_per_wg; w++) {
                    if (ended[w]) continue;
                    active_count++;
                    if (at_barrier[w]) {
                        barrier_count++;
                        continue;
                    }

                    WaveStatus status = run_wavefront(*waves[w]);
                    if (status == WaveStatus::ENDED) {
                        ended[w] = true;
                    } else if (status == WaveStatus::BARRIER_HIT) {
                        at_barrier[w] = true;
                        barrier_count++;
                    }
                    // WaveStatus::RUNNING (hit max_insns quantum) continues on next iteration
                }

                if (active_count == 0) break;

                if (barrier_count > 0 && barrier_count == active_count) {
                    sync_barrier_cycles(waves, ended);
                    for (uint32_t w = 0; w < waves_per_wg; w++) {
                        at_barrier[w] = false;
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
    uint64_t max_cu_wall = *std::max_element(cu_wall_cycles.begin(), cu_wall_cycles.end());
    if (total_cycles && cfg.enable_cycle_counting) {
        *total_cycles = max_cu_wall;
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
        stats->max_wgs_per_cu = max_wgs_per_cu;
        if (max_cu_wall > 0 && num_cus > 0) {
            double total_wave_slots = static_cast<double>(max_cu_wall) * static_cast<double>(num_cus) * 32.0;
            stats->achieved_occupancy = std::min(1.0, static_cast<double>(accum_wave_cycles) / total_wave_slots);
        }
        if (global_cache) {
            stats->l1_hits   = global_cache->l1_hits();
            stats->l1_misses = global_cache->l1_misses();
            stats->l2_hits   = global_cache->l2_hits();
            stats->l2_misses = global_cache->l2_misses();
            stats->l3_hits   = global_cache->l3_hits();
            stats->l3_misses = global_cache->l3_misses();
            stats->l1_bytes  = global_cache->l1_bytes();
            stats->l2_bytes  = global_cache->l2_bytes();
            stats->l3_bytes  = global_cache->l3_bytes();
            stats->hbm_bytes = global_cache->hbm_bytes();
        }
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
