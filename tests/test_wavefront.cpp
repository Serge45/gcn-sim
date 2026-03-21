#include <gtest/gtest.h>
#include "gcnsim/wavefront.h"
#include "gcnsim/kernel_descriptor.h"
#include <cstring>

namespace gcnsim {
// Forward declaration from wavefront.cpp
void init_wavefront(WavefrontState& wave,
                    const KernelDescriptor& kd,
                    const uint8_t* kernarg_ptr,
                    const uint8_t* code_ptr,
                    size_t code_size,
                    uint32_t wg_id_x, uint32_t wg_id_y, uint32_t wg_id_z,
                    uint32_t wave_in_wg, uint32_t waves_per_wg,
                    uint32_t block_x, uint32_t block_y, uint32_t block_z);
}

using namespace gcnsim;

static KernelDescriptor make_kd(uint16_t kcp, uint32_t rsrc2) {
    KernelDescriptor kd{};
    kd.kernel_code_properties = kcp;
    kd.compute_pgm_rsrc2 = rsrc2;
    kd.kernarg_size = 32;
    kd.kernel_code_entry_byte_offset = 256;
    return kd;
}

TEST(Wavefront, FullWaveExecMask) {
    KernelDescriptor kd = make_kd(0, 0);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,   // wg_id
                   0, 1,      // wave_in_wg, waves_per_wg
                   64, 1, 1); // block dims

    EXPECT_EQ(ws.exec, ~0ULL) << "Full wavefront should have all lanes active";
}

TEST(Wavefront, PartialWaveExecMask32) {
    KernelDescriptor kd = make_kd(0, 0);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    // Block of 96 threads: first wave has 64 lanes, second has 32
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   1, 2,       // wave_in_wg=1 (second wave), waves_per_wg=2
                   96, 1, 1);

    uint64_t expected = 0x00000000FFFFFFFFULL;
    EXPECT_EQ(ws.exec, expected)
        << "Second wave of 96-thread block should have 32 active lanes";
}

TEST(Wavefront, VGPR0IsWorkitemIdX) {
    KernelDescriptor kd = make_kd(0, 0);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   0, 1,
                   64, 1, 1);

    for (int lane = 0; lane < 64; lane++) {
        EXPECT_EQ(ws.vgpr[0][lane], (uint32_t)lane)
            << "vgpr[0][" << lane << "] should be workitem_id.x = " << lane;
    }
}

TEST(Wavefront, KernargPtrInSGPR) {
    // Enable SGPR kernarg segment ptr (bit 3 of kcp)
    KernelDescriptor kd = make_kd(KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                                   RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   5, 0, 0,  // wg_id_x = 5
                   0, 1,
                   64, 1, 1);

    // With KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR, kernarg ptr goes into sgpr[0:1]
    uint64_t ka = reinterpret_cast<uint64_t>(dummy_kernarg);
    uint64_t sgpr_ka = ((uint64_t)ws.sgpr[1] << 32) | ws.sgpr[0];
    EXPECT_EQ(sgpr_ka, ka) << "SGPR[0:1] should contain kernarg pointer";

    // With RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, wg_id_x goes into sgpr[2]
    EXPECT_EQ(ws.sgpr[2], 5u) << "SGPR[2] should contain workgroup_id_x";
}

TEST(Wavefront, WorkitemIDLayout2D) {
    KernelDescriptor kd = make_kd(0, 0);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    // 8x8 block = 64 threads
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   0, 1,
                   8, 8, 1);

    for (int lane = 0; lane < 64; lane++) {
        uint32_t expected_x = lane % 8;
        uint32_t expected_y = lane / 8;
        EXPECT_EQ(ws.workitem_id[lane][0], expected_x);
        EXPECT_EQ(ws.workitem_id[lane][1], expected_y);
        EXPECT_EQ(ws.workitem_id[lane][2], 0u);
        // vgpr[0] should have workitem_id.x
        EXPECT_EQ(ws.vgpr[0][lane], expected_x);
    }
}

TEST(Wavefront, WorkgroupIDSet) {
    KernelDescriptor kd = make_kd(0, 0);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   3, 2, 1,  // wg_id
                   0, 1,
                   64, 1, 1);

    EXPECT_EQ(ws.workgroup_id[0], 3u);
    EXPECT_EQ(ws.workgroup_id[1], 2u);
    EXPECT_EQ(ws.workgroup_id[2], 1u);
}

// --- Phase 3: workitem_id.y/z in VGPRs ---
TEST(Wavefront, VGPR1IsWorkitemIdY) {
    // enable_vgpr_workitem_id >= 1: VGPR1 = workitem_id.y
    // compute_pgm_rsrc2 bits[12:11] = enable_vgpr_workitem_id
    // value 1 = X and Y
    uint32_t rsrc2 = (1u << 11); // enable_vgpr_workitem_id = 1
    KernelDescriptor kd = make_kd(0, rsrc2);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    // 8x8 block = 64 threads
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   0, 1,
                   8, 8, 1);

    for (int lane = 0; lane < 64; lane++) {
        uint32_t expected_x = lane % 8;
        uint32_t expected_y = lane / 8;
        EXPECT_EQ(ws.vgpr[0][lane], expected_x) << "vgpr[0] = workitem_id.x at lane " << lane;
        EXPECT_EQ(ws.vgpr[1][lane], expected_y) << "vgpr[1] = workitem_id.y at lane " << lane;
    }
}

TEST(Wavefront, VGPR2IsWorkitemIdZ) {
    // enable_vgpr_workitem_id = 2: VGPR0=X, VGPR1=Y, VGPR2=Z
    uint32_t rsrc2 = (2u << 11); // enable_vgpr_workitem_id = 2
    KernelDescriptor kd = make_kd(0, rsrc2);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    // 4x4x4 block = 64 threads
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   0, 1,
                   4, 4, 4);

    for (int lane = 0; lane < 64; lane++) {
        uint32_t expected_x = lane % 4;
        uint32_t expected_y = (lane / 4) % 4;
        uint32_t expected_z = lane / 16;
        EXPECT_EQ(ws.vgpr[0][lane], expected_x) << "vgpr[0] = workitem_id.x at lane " << lane;
        EXPECT_EQ(ws.vgpr[1][lane], expected_y) << "vgpr[1] = workitem_id.y at lane " << lane;
        EXPECT_EQ(ws.vgpr[2][lane], expected_z) << "vgpr[2] = workitem_id.z at lane " << lane;
    }
}

TEST(Wavefront, VGPR1NotSetWhenDisabled) {
    // enable_vgpr_workitem_id = 0: only VGPR0=X, VGPR1 should be zero
    uint32_t rsrc2 = 0; // enable_vgpr_workitem_id = 0
    KernelDescriptor kd = make_kd(0, rsrc2);
    uint8_t dummy_kernarg[32] = {};
    uint8_t dummy_code[64] = {};

    WavefrontState ws;
    init_wavefront(ws, kd, dummy_kernarg, dummy_code, 64,
                   0, 0, 0,
                   0, 1,
                   8, 8, 1);

    for (int lane = 0; lane < 64; lane++) {
        EXPECT_EQ(ws.vgpr[0][lane], (uint32_t)(lane % 8)) << "vgpr[0] = workitem_id.x";
        EXPECT_EQ(ws.vgpr[1][lane], 0u) << "vgpr[1] should be 0 when workitem_id.y disabled";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
