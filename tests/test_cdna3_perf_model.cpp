#include <gtest/gtest.h>
#include "gcnsim/cache_model.h"
#include "gcnsim/execution_engine.h"
#include "gcnsim/isa_decoder.h"
#include "gcnsim/kernel_descriptor.h"
#include "gcnsim/wavefront.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace gcnsim {
WaveStatus step_wavefront(WavefrontState& wave);
}

using namespace gcnsim;

static std::vector<uint8_t> build_words(std::initializer_list<uint32_t> words) {
    std::vector<uint8_t> prog(words.size() * 4);
    size_t i = 0;
    for (uint32_t w : words) {
        prog[i * 4 + 0] = static_cast<uint8_t>(w & 0xFFu);
        prog[i * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFFu);
        prog[i * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFFu);
        prog[i * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFFu);
        i++;
    }
    return prog;
}

static std::unique_ptr<WavefrontState> make_cdna3_wave(const std::vector<uint8_t>& prog,
                                                       bool is_gfx942 = true) {
    auto ws = std::make_unique<WavefrontState>();
    ws->reset_zero();
    ws->exec = ~0ULL;
    ws->wave_size = 64;
    ws->is_gfx942 = is_gfx942;
    ws->count_cycles = true;
    ws->global_mem_latency = 440;
    ws->smem_latency = 20;
    ws->kernel_code_ptr = prog.data();
    ws->kernel_code_size = prog.size();
    ws->pc = 0;
    return ws;
}

TEST(CDNA3PerfModel, PerXcdL2IsolationAndMallBacking) {
    CacheConfig cfg;
    cfg.l1_size_bytes = 32768;
    cfg.l1_ways = 4;
    cfg.l1_line_bytes = 128;
    cfg.l1_hit_cycles = 36;
    cfg.num_xcds = 8;
    cfg.l2_size_bytes = 4 * 1024 * 1024;
    cfg.l2_ways = 16;
    cfg.l2_line_bytes = 128;
    cfg.l2_hit_cycles = 160;
    cfg.enable_l3 = true;
    cfg.l3_size_bytes = 256 * 1024 * 1024;
    cfg.l3_ways = 16;
    cfg.l3_line_bytes = 128;
    cfg.l3_hit_cycles = 270;

    CacheModel cache(cfg, 304); // 38 CUs per XCD

    // Cold miss from CU 0 (XCD 0) -> HBM
    EXPECT_EQ(cache.access(0x100000ULL, 0), CacheHitLevel::HBM);
    // Same XCD (CU 1 -> XCD 0) misses CU 1 L1, hits XCD 0 L2
    EXPECT_EQ(cache.access(0x100000ULL, 1), CacheHitLevel::L2);
    // Cross XCD (CU 38 -> XCD 1) misses L1 & XCD 1 L2, hits shared 256 MB MALL (L3)
    EXPECT_EQ(cache.access(0x100000ULL, 38), CacheHitLevel::L3);
    // Repeat on CU 38 -> hits CU 38 L1
    EXPECT_EQ(cache.access(0x100000ULL, 38), CacheHitLevel::L1);
}

TEST(CDNA3PerfModel, MonotonicLatencyLadder) {
    CacheConfig cfg;
    cfg.l1_size_bytes = 32768;
    cfg.l1_line_bytes = 128;
    cfg.l1_hit_cycles = 36;
    cfg.num_xcds = 8;
    cfg.l2_size_bytes = 4 * 1024 * 1024;
    cfg.l2_line_bytes = 128;
    cfg.l2_hit_cycles = 160;
    cfg.enable_l3 = true;
    cfg.l3_size_bytes = 256 * 1024 * 1024;
    cfg.l3_line_bytes = 128;
    cfg.l3_hit_cycles = 270;
    cfg.enable_bw_model = true;
    cfg.hbm_base_cycles = 440;

    CacheModel cache(cfg, 304);
    auto r_hbm = cache.access_timed(0x200000ULL, 0, 100, 128, 300);
    auto r_l1  = cache.access_timed(0x200000ULL, 0, 100, 128, 300);
    auto r_l2  = cache.access_timed(0x200000ULL, 1, 100, 128, 300);
    auto r_l3  = cache.access_timed(0x200000ULL, 38, 100, 128, 300);

    EXPECT_EQ(r_hbm.level, CacheHitLevel::HBM);
    EXPECT_EQ(r_l1.level,  CacheHitLevel::L1);
    EXPECT_EQ(r_l2.level,  CacheHitLevel::L2);
    EXPECT_EQ(r_l3.level,  CacheHitLevel::L3);

    EXPECT_LT(r_l1.completion_cycle,  r_l2.completion_cycle);
    EXPECT_LT(r_l2.completion_cycle,  r_l3.completion_cycle);
    EXPECT_LT(r_l3.completion_cycle,  r_hbm.completion_cycle);
}

TEST(CDNA3PerfModel, Hbm3BandwidthCeilingAndQueuingContention) {
    CacheConfig cfg;
    cfg.l1_size_bytes = 32768;
    cfg.l1_line_bytes = 128;
    cfg.l1_hit_cycles = 36;
    cfg.num_xcds = 8;
    cfg.l2_size_bytes = 4 * 1024 * 1024;
    cfg.l2_line_bytes = 128;
    cfg.l2_hit_cycles = 160;
    cfg.enable_l3 = true;
    cfg.l3_size_bytes = 256 * 1024 * 1024;
    cfg.l3_line_bytes = 128;
    cfg.l3_hit_cycles = 270;
    cfg.enable_bw_model = true;
    cfg.hbm_bytes_per_cycle = 2524.0; // ~5.3 TB/s @ 2.1 GHz
    cfg.hbm_base_cycles = 440;

    CacheModel cache(cfg, 304);
    uint64_t max_comp = 0;
    constexpr int N_LINES = 2048; // 256 KB
    for (int i = 0; i < N_LINES; i++) {
        uint64_t addr = 0x1000000ULL + static_cast<uint64_t>(i) * 128ULL;
        auto r = cache.access_timed(addr, static_cast<uint32_t>(i % 304), 100, 128, 440);
        max_comp = std::max(max_comp, r.completion_cycle);
    }
    EXPECT_EQ(cache.hbm_bytes(), static_cast<uint64_t>(N_LINES) * 128ULL);
    // 262144 / 2524 ~= 104 cycles of HBM serialization delay on top of issue(100)+base(440)
    EXPECT_GE(max_comp, 100ULL + 440ULL + 100ULL);
}

TEST(CDNA3PerfModel, Cdna3MfmaThroughputAndRawScoreboard) {
    // 1. v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[0:3] -> 16 cycles on CDNA3 vs 32 on CDNA2
    auto mfma_f32_prog = build_words({
        0xD3C58000u, 0x04020300u, // v_mfma_f32_16x16x4_f32 a[0:3], v0, v1, a[0:3]
        0xBF810000u               // s_endpgm
    });
    auto w_942 = make_cdna3_wave(mfma_f32_prog, true);
    while (step_wavefront(*w_942) == WaveStatus::RUNNING) {}
    auto w_90a = make_cdna3_wave(mfma_f32_prog, false);
    while (step_wavefront(*w_90a) == WaveStatus::RUNNING) {}
    EXPECT_EQ(w_942->mfma_total_exec_cycles, 16u);
    EXPECT_EQ(w_90a->mfma_total_exec_cycles, 32u);
    EXPECT_EQ(w_942->cycle_count, 16u);

    // 2. RAW scoreboard: v_mfma_f32_16x16x16_f16 followed immediately by v_accvgpr_read_b32 v4, a0
    auto raw_prog = build_words({
        0xD3CD8000u, 0x04020500u, // v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]
        0xD3D84004u, 0x18000100u, // v_accvgpr_read_b32 v4, a0
        0xBF810000u               // s_endpgm
    });
    auto w_raw = make_cdna3_wave(raw_prog, true);
    while (step_wavefront(*w_raw) == WaveStatus::RUNNING) {}
    // MFMA starts at 0 (ready at 20), accvgpr_read stalls until 20 then takes 4 -> 24
    EXPECT_GE(w_raw->cycle_count, 24u);

    // 3. Independent VALU overlaps MFMA shadow without adding cycles when issued before read
    auto overlap_prog = build_words({
        0xD3CD8000u, 0x04020500u, // v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]
        0x02040101u,              // v_add_f32 v2, v1, v0 (independent)
        0x02040101u,              // v_add_f32 v2, v1, v0 (independent)
        0xD3D84004u, 0x18000100u, // v_accvgpr_read_b32 v4, a0
        0xBF810000u               // s_endpgm
    });
    auto w_ov = make_cdna3_wave(overlap_prog, true);
    while (step_wavefront(*w_ov) == WaveStatus::RUNNING) {}
    EXPECT_EQ(w_ov->cycle_count, w_raw->cycle_count);
}

TEST(CDNA3PerfModel, QuarterRateTranscendentalAndLdsBroadcast) {
    // v_rcp_f32 v1, v0 -> 16 cyc + RAW hazard before v_add_f32 v2, v1, v0
    auto rcp_prog = build_words({
        0x7E024500u, // v_rcp_f32 v1, v0
        0x02040101u, // v_add_f32 v2, v1, v0
        0xBF810000u  // s_endpgm
    });
    auto w_rcp = make_cdna3_wave(rcp_prog, true);
    while (step_wavefront(*w_rcp) == WaveStatus::RUNNING) {}
    EXPECT_GE(w_rcp->cycle_count, 20u);

    // ds_read_b32 v4, v0 with all 64 lanes reading address 0 (broadcast)
    auto ds_prog = build_words({
        0xD86C0000u, 0x04000000u, // ds_read_b32 v4, v0
        0xBF810000u               // s_endpgm
    });
    std::vector<uint8_t> lds(256, 0u);
    auto w_ds942 = make_cdna3_wave(ds_prog, true);
    w_ds942->lds_ptr = lds.data();
    w_ds942->lds_size = static_cast<uint32_t>(lds.size());
    while (step_wavefront(*w_ds942) == WaveStatus::RUNNING) {}

    auto w_ds90a = make_cdna3_wave(ds_prog, false);
    w_ds90a->lds_ptr = lds.data();
    w_ds90a->lds_size = static_cast<uint32_t>(lds.size());
    while (step_wavefront(*w_ds90a) == WaveStatus::RUNNING) {}

    // CDNA3 broadcast: 2 txns x 4 cyc = 8 cyc (+ endpgm 4 = 12) vs 256 cyc conflict on legacy
    EXPECT_EQ(w_ds942->cycle_count, 12u);
    EXPECT_GT(w_ds90a->cycle_count, 200u);
}

TEST(CDNA3PerfModel, MultiWgOccupancyHidesHbmLatency) {
    // Kernel: global_load_dword v4, v[0:1], off; s_waitcnt vmcnt(0); s_endpgm
    auto prog = build_words({
        0xDC508000u, 0x047F0000u, // global_load_dword v4, v[0:1], off
        0xBF8C0F70u,              // s_waitcnt vmcnt(0)
        0xBF810000u               // s_endpgm
    });

    std::vector<uint32_t> buf(4096, 42u);
    uint64_t ptr = reinterpret_cast<uint64_t>(buf.data());

    // Low occupancy: 512 VGPRs + 4 waves/WG (block_x=256) -> max 1 WG/CU
    KernelDescriptor kd_low{};
    kd_low.compute_pgm_rsrc1 = 63u; // (63+1)*8 = 512 VGPRs -> 1 wave/SIMD = 4 waves/CU
    KernelInfo ki_low{};
    ki_low.name = "mem_low_occ";
    ki_low.descriptor = &kd_low;
    ki_low.code_ptr = prog.data();
    ki_low.code_size = prog.size();
    ki_low.wavefront_size = 64;
    ki_low.arch = TargetArch::GFX942;
    ki_low.is_gfx942 = true;

    // High occupancy: 32 VGPRs + 4 waves/WG -> max 8 WGs/CU (32 waves/CU)
    KernelDescriptor kd_high = kd_low;
    kd_high.compute_pgm_rsrc1 = 3u; // (3+1)*8 = 32 VGPRs -> 8 waves/SIMD = 32 waves/CU
    KernelInfo ki_high = ki_low;
    ki_high.name = "mem_high_occ";
    ki_high.descriptor = &kd_high;

    LaunchConfig cfg{};
    cfg.grid_x = 256 * 256; // 256 WGs, 4 waves/WG -> 4 WGs per CU on 64 CUs
    cfg.grid_y = 1;
    cfg.grid_z = 1;
    cfg.block_x = 256;
    cfg.block_y = 1;
    cfg.block_z = 1;
    cfg.num_cus = 64;
    cfg.enable_cycle_counting = true;
    cfg.debug_hook = [&](WavefrontState& w) {
        if (w.pc == 0) {
            for (int l = 0; l < 64; l++) {
                uint64_t a = ptr + static_cast<uint64_t>(w.workgroup_id[0]) * 16ULL;
                w.vgpr[0][l] = static_cast<uint32_t>(a & 0xFFFFFFFFULL);
                w.vgpr[1][l] = static_cast<uint32_t>(a >> 32);
            }
        }
    };

    std::string err;
    uint64_t low_cycles = 0, high_cycles = 0;
    KernelStats low_stats{}, high_stats{};
    ASSERT_TRUE(launch_kernel(ki_low, cfg, err, &low_cycles, &low_stats)) << err;
    ASSERT_TRUE(launch_kernel(ki_high, cfg, err, &high_cycles, &high_stats)) << err;

    EXPECT_EQ(low_stats.max_wgs_per_cu, 1u);
    EXPECT_EQ(high_stats.max_wgs_per_cu, 8u);
    EXPECT_GT(high_stats.achieved_occupancy, low_stats.achieved_occupancy);
    EXPECT_LT(high_cycles, low_cycles);
}
