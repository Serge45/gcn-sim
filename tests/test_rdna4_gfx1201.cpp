#include <gtest/gtest.h>
#include "gcnsim/isa_decoder.h"
#include "gcnsim/wavefront.h"
#include "gcnsim/kernel_descriptor.h"
#include "gcnsim/hip_hijack.h"
#include <cstring>
#include <cmath>
#include <memory>
#include <vector>

namespace gcnsim {
void execute_instruction(WavefrontState& wave, const Instruction& inst);
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

static uint16_t f32_to_f16_test(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000u;
    int32_t exp = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (b >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float f16_to_f32_test(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) {
            float z; std::memcpy(&z, &sign, 4); return z;
        }
        float v = std::ldexp((float)mant, -24);
        return (sign ? -v : v);
    }
    uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

static std::unique_ptr<WavefrontState> make_gfx12_wave() {
    auto ws = std::make_unique<WavefrontState>();
    std::memset((void*)ws.get(), 0, offsetof(WavefrontState, lgkm_pending));
    ws->wave_size = 32;
    ws->is_gfx1201 = true;
    ws->exec = 0xFFFFFFFFULL;
    return ws;
}

TEST(RDNA4Gfx1201Test, Wave32StateInitAndArchitectedTtmp) {
    KernelDescriptor kd{};
    kd.kernel_code_properties = KCP_ENABLE_WAVEFRONT_SIZE32 | KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
    kd.compute_pgm_rsrc2 = (2 << RSRC2_USER_SGPR_COUNT_SHIFT) | RSRC2_ENABLE_SGPR_WORKGROUP_ID_X;

    auto ws = std::make_unique<WavefrontState>();
    uint64_t dummy_kernarg = 0x12345678ULL;
    uint8_t dummy_code[4] = {0};

    init_wavefront(*ws, kd, reinterpret_cast<const uint8_t*>(&dummy_kernarg),
                   dummy_code, sizeof(dummy_code),
                   /*wg_x=*/5, /*wg_y=*/3, /*wg_z=*/2,
                   /*wave_in_wg=*/1, /*waves_per_wg=*/2,
                   /*block_x=*/16, /*block_y=*/4, /*block_z=*/1);

    EXPECT_EQ(ws->wave_size, 32u);
    EXPECT_EQ(ws->exec, 0xFFFFFFFFULL);
    // Architected ttmp registers on GFX12:
    // ttmp7 = wg_y | (wg_z << 16)
    // ttmp8 = (wave_in_wg & 0x1F) << 25
    // ttmp9 = wg_x
    EXPECT_EQ(ws->sgpr[TTMP_BASE_IDX + 7], (3u | (2u << 16)));
    EXPECT_EQ(ws->sgpr[TTMP_BASE_IDX + 8], (1u << 25));
    EXPECT_EQ(ws->sgpr[TTMP_BASE_IDX + 9], 5u);

    // Wave 1 covers flat threads 32..63 -> tid_x = 0..15, tid_y = 2..3
    for (uint32_t lane = 0; lane < 32; lane++) {
        uint32_t flat = 32 + lane;
        uint32_t exp_x = flat % 16;
        uint32_t exp_y = flat / 16;
        EXPECT_EQ(ws->vgpr[0][lane] & 0x3FFu, exp_x);
        EXPECT_EQ((ws->vgpr[0][lane] >> 10) & 0x3FFu, exp_y);
    }
}

TEST(RDNA4Gfx1201Test, DevicePropertiesRx9070Xt) {
    setenv("GCN_SIM_ARCH", "gfx1201", 1);
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    EXPECT_STREQ(prop.gcnArchName, "gfx1201");
    EXPECT_EQ(prop.multiProcessorCount, 32);
    EXPECT_EQ(prop.warpSize, 32);
    EXPECT_EQ(prop.clockRate, 2970000);
    EXPECT_EQ(prop.sharedMemPerBlock, 65536u);
    EXPECT_EQ(prop.maxSharedMemoryPerMultiProcessor, 131072u);
    unsetenv("GCN_SIM_ARCH");
}

TEST(RDNA4Gfx1201Test, DecodeGfx12Formats) {
    // 1. s_wait_loadcnt 0x0002 -> 0xBF800002 | (0x40 << 16) = 0xBFC00002
    {
        uint32_t enc = 0xBFC00002u;
        Instruction inst = decode(reinterpret_cast<const uint8_t*>(&enc), false, false, true);
        EXPECT_EQ(inst.format, InstFormat::SOPP);
        EXPECT_EQ(inst.opcode, Opcode::S_WAIT_LOADCNT);
        EXPECT_EQ(inst.wait_vmcnt, 2u);
        EXPECT_EQ(inst.size_bytes, 4u);
    }
    // 2. 96-bit VBUFFER: buffer_load_d16_b16 v34, v8, s[48:51], null offen offset:16
    // enc = {0xC408007C, 0x40806022, (16u << 8) | 8u}
    {
        uint32_t enc[3] = {0xC408007Cu, 0x40806022u, (16u << 8) | 8u};
        Instruction inst = decode(reinterpret_cast<const uint8_t*>(enc), false, false, true);
        EXPECT_EQ(inst.format, InstFormat::MUBUF);
        EXPECT_EQ(inst.opcode, Opcode::BUFFER_LOAD_SHORT_D16);
        EXPECT_EQ(inst.size_bytes, 12u);
        EXPECT_EQ(inst.dst.index, 34u);
        EXPECT_EQ(inst.src[0].index, 8u);
        EXPECT_EQ(inst.src[1].index, 48u);
        EXPECT_EQ(inst.mubuf_offset, 16u);
        EXPECT_TRUE(inst.mubuf_offen);
    }
    // 3. VOPD: v_dual_fmac_f32 v6, v1, v2 :: v_dual_add_f32 v7, v3, v4
    // opx=0 (fmac_f32), opy=4 (add_f32)
    {
        uint32_t w0 = (0x32u << 26) | (0u << 22) | (4u << 17) | (2u << 9) | 257u;
        uint32_t w1 = (6u << 24) | (3u << 17) | (4u << 9) | 259u;
        uint32_t enc[2] = {w0, w1};
        Instruction inst = decode(reinterpret_cast<const uint8_t*>(enc), false, false, true);
        EXPECT_EQ(inst.format, InstFormat::VOPD);
        EXPECT_EQ(inst.opcode, Opcode::V_FMAC_F32);
        EXPECT_EQ(inst.vopd_opy, Opcode::V_ADD_F32);
        EXPECT_EQ(inst.dst.index, 6u);
        EXPECT_EQ(inst.vopd_dsty.index, 7u);
        EXPECT_EQ(inst.size_bytes, 8u);
    }
    // 4. RDNA4 WMMA: v_wmma_f32_16x16x16_f16 v[16:23], v[0:3], v[4:7], v[16:23]
    // enc = {0xCC408010, 0x1C400900}
    {
        uint32_t enc[2] = {0xCC408010u, 0x1C400900u};
        Instruction inst = decode(reinterpret_cast<const uint8_t*>(enc), false, false, true);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
        EXPECT_EQ(inst.opcode, Opcode::V_WMMA_F32_16X16X16_F16);
        EXPECT_EQ(inst.dst.index, 16u);
        EXPECT_EQ(inst.src[0].index, 0u);
        EXPECT_EQ(inst.src[1].index, 4u);
        EXPECT_EQ(inst.src[2].index, 16u);
        EXPECT_EQ(inst.size_bytes, 8u);
    }
}

TEST(RDNA4Gfx1201Test, VopdSimultaneousRegisterRead) {
    auto ws = make_gfx12_wave();
    for (int lane = 0; lane < 32; lane++) {
        ws->vgpr[0][lane] = 100 + lane;
        ws->vgpr[1][lane] = 10;
        ws->vgpr[2][lane] = 50; // old value of v2 before dual-issue
    }
    // v_dual_mov_b32 v2, v0 :: v_dual_add_nc_u32 v3, v2, v1
    // Y must read the OLD value of v2 (50), so v2 becomes 100+lane and v3 becomes 50+10=60.
    Instruction inst{};
    inst.format = InstFormat::VOPD;
    inst.opcode = Opcode::V_MOV_B32;
    inst.dst = {Operand::Kind::VGPR, 2, 0};
    inst.src[0] = {Operand::Kind::VGPR, 0, 0};

    inst.vopd_opy = Opcode::V_ADD_U32;
    inst.vopd_dsty = {Operand::Kind::VGPR, 3, 0};
    inst.vopd_srcy[0] = {Operand::Kind::VGPR, 2, 0};
    inst.vopd_srcy[1] = {Operand::Kind::VGPR, 1, 0};

    execute_instruction(*ws, inst);
    for (int lane = 0; lane < 32; lane++) {
        EXPECT_EQ(ws->vgpr[2][lane], (uint32_t)(100 + lane));
        EXPECT_EQ(ws->vgpr[3][lane], 60u);
    }
}

TEST(RDNA4Gfx1201Test, WmmaF32_16x16x16_F16_Execution) {
    auto ws = make_gfx12_wave();
    // Initialize A (16x16) in v[0:3] and B (16x16) in v[4:7]:
    // Set A[m][k] = (m + 1) * 0.5f, B[k][n] = (n + 1) * 0.25f, C[m][n] = 10.0f
    // Expected D[m][n] = 10.0f + 16 * (m + 1) * 0.5f * (n + 1) * 0.25f = 10.0f + 2.0f * (m + 1) * (n + 1)
    for (int m = 0; m < 16; m++) {
        for (int k = 0; k < 8; k++) {
            int reg = k / 2;
            uint16_t lo0 = f32_to_f16_test((m + 1) * 0.5f);
            uint16_t hi0 = f32_to_f16_test((m + 1) * 0.5f);
            if (k % 2 == 0) {
                ws->vgpr[0 + reg][m]      = (uint32_t)lo0 | ((uint32_t)hi0 << 16);
                ws->vgpr[0 + reg][m + 16] = (uint32_t)lo0 | ((uint32_t)hi0 << 16);
            }
        }
    }
    for (int n = 0; n < 16; n++) {
        for (int reg = 0; reg < 4; reg++) {
            uint16_t h = f32_to_f16_test((n + 1) * 0.25f);
            uint32_t packed = (uint32_t)h | ((uint32_t)h << 16);
            ws->vgpr[4 + reg][n]      = packed;
            ws->vgpr[4 + reg][n + 16] = packed;
        }
    }
    for (int lane = 0; lane < 32; lane++) {
        for (int r = 0; r < 8; r++) {
            float c_init = 10.0f;
            std::memcpy(&ws->vgpr[16 + r][lane], &c_init, 4);
        }
    }

    Instruction inst{};
    inst.format = InstFormat::MFMA;
    inst.opcode = Opcode::V_WMMA_F32_16X16X16_F16;
    inst.dst    = {Operand::Kind::VGPR, 16, 0};
    inst.src[0] = {Operand::Kind::VGPR, 0, 0};
    inst.src[1] = {Operand::Kind::VGPR, 4, 0};
    inst.src[2] = {Operand::Kind::VGPR, 16, 0};

    execute_instruction(*ws, inst);

    for (int lane = 0; lane < 32; lane++) {
        int col = lane % 16;
        int row_base = (lane / 16) * 8;
        for (int r = 0; r < 8; r++) {
            int row = row_base + r;
            float got = 0.0f;
            std::memcpy(&got, &ws->vgpr[16 + r][lane], 4);
            float expected = 10.0f + 2.0f * (row + 1) * (col + 1);
            EXPECT_FLOAT_EQ(got, expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST(RDNA4Gfx1201Test, WmmaI32_16x16x16_IU8_Execution) {
    auto ws = make_gfx12_wave();
    // A[m][k] = (m + 1), B[k][n] = (n + 1), C = 5
    // Expected D[m][n] = 5 + 16 * (m + 1) * (n + 1)
    for (int m = 0; m < 16; m++) {
        uint8_t v = (uint8_t)(m + 1);
        uint32_t packed = (uint32_t)v | ((uint32_t)v << 8) | ((uint32_t)v << 16) | ((uint32_t)v << 24);
        for (int reg = 0; reg < 2; reg++) {
            ws->vgpr[0 + reg][m]      = packed;
            ws->vgpr[0 + reg][m + 16] = packed;
        }
    }
    for (int n = 0; n < 16; n++) {
        uint8_t v = (uint8_t)(n + 1);
        uint32_t packed = (uint32_t)v | ((uint32_t)v << 8) | ((uint32_t)v << 16) | ((uint32_t)v << 24);
        for (int reg = 0; reg < 2; reg++) {
            ws->vgpr[4 + reg][n]      = packed;
            ws->vgpr[4 + reg][n + 16] = packed;
        }
    }
    for (int lane = 0; lane < 32; lane++) {
        for (int r = 0; r < 8; r++) {
            ws->vgpr[16 + r][lane] = 5u;
        }
    }

    Instruction inst{};
    inst.format = InstFormat::MFMA;
    inst.opcode = Opcode::V_WMMA_I32_16X16X16_IU8;
    inst.neg    = 0x3; // signed A and signed B
    inst.dst    = {Operand::Kind::VGPR, 16, 0};
    inst.src[0] = {Operand::Kind::VGPR, 0, 0};
    inst.src[1] = {Operand::Kind::VGPR, 4, 0};
    inst.src[2] = {Operand::Kind::VGPR, 16, 0};

    execute_instruction(*ws, inst);

    for (int lane = 0; lane < 32; lane++) {
        int col = lane % 16;
        int row_base = (lane / 16) * 8;
        for (int r = 0; r < 8; r++) {
            int row = row_base + r;
            int32_t got = (int32_t)ws->vgpr[16 + r][lane];
            int32_t expected = 5 + 16 * (row + 1) * (col + 1);
            EXPECT_EQ(got, expected);
        }
    }
}
