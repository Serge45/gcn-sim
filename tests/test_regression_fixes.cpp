#include <gtest/gtest.h>
#include "gcnsim/isa_decoder.h"
#include "gcnsim/wavefront.h"
#include "gcnsim/execution_engine.h"
#include "gcnsim/kernel_descriptor.h"
#include "gcnsim/cache_model.h"
#include "gcnsim/elf_loader.h"
#include "gcnsim/hip_hijack.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
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
} // namespace gcnsim

using namespace gcnsim;

static std::unique_ptr<WavefrontState> make_wave(bool gfx1201 = true) {
    auto ws = std::make_unique<WavefrontState>();
    ws->reset_zero();
    ws->is_gfx1201 = gfx1201;
    ws->wave_size  = gfx1201 ? 32 : 64;
    ws->exec       = gfx1201 ? 0xFFFFFFFFULL : ~0ULL;
    return ws;
}

// 1. Execution Engine: single WG (<=4 waves) with >10,000 insns + s_barrier
//    Must NOT treat WaveStatus::RUNNING as BARRIER_HIT and release Wave 0 early!
TEST(RegressionFixesTest, SingleWgLongLoopBarrierNotReleasedEarly) {
    // Program layout:
    //   0x00: v_cmp_eq_u32 vcc, v0, 0      (or s_cmp_eq_u32 s8, 0? No, use v0 == 0 via ttmp8/vgpr0)
    // Simpler: wave 0 has v0=0..31, wave 1 has v0=32..63 (from init_wavefront)
    //   0x00: s_mov_b32 s0, 0               (4B)
    //   0x04: v_readfirstlane_b32 s1, v0    (8B VOP3? Let's use s_and_b32 with ttmp8! On gfx1201, ttmp8 = (wave_in_wg << 25))
    // Wait: we can run on CDNA/Wave64 or RDNA4/Wave32. On RDNA4:
    //   wave_in_wg is in ttmp8[29:25]. Or even simpler, use hand-crafted instructions in GFX9/CDNA Wave64:
    //   On GFX9, let's check wave_in_wg: vgpr0[lane] = tid_x = wave_in_wg * 64 + lane.
    //   Use V_CMP_LT_U32_e32 (32-bit VOPC) or V_ADD_U32 / S_CBRANCH_VCCZ!
    //   Wait: in GFX9:
    //     V_CMP_LT_U32 (vopc, v0 < 64) -> sets VCC != 0 for wave 0, VCC == 0 for wave 1!
    //   Let's encode a tiny GFX9 program:
    //     0x00: V_CMP_LT_U32_e32 vcc, 64, v0? Wait: src0=64 (192+64? No, inline const 64 is 192! 128+64=192 -> 64!).
    //     Even easier: inline const 64 is enc=192. Wait: V_CMP_GT_U32 vcc, 64, v0 -> true if 64 > v0 (i.e. v0 < 64)!
    //     VOPC encoding: [31:25]=0x3E (0b0111110), [24:17]=op (0xCC = V_CMP_GT_U32), [16:9]=vsrc1 (v0 = 0), [8:0]=src0 (192 = 64).
    //     Let's verify: (0x3E << 25) | (0xCC << 17) | (0 << 9) | 192 = 0x7D9800C0
    //     0x04: S_CBRANCH_VCCZ +3 (to loop in wave 1, offset 0x14) -> SOPP op=4 (S_CBRANCH_VCCZ), simm16=+3 -> 0xBF840003
    //     Wave 0 path:
    //     0x08: S_BARRIER -> 0xBF8A0000
    //     0x0C: S_LOAD_DWORD? Or write to LDS!
    //     Wait: we can use DS_WRITE_B32 / DS_READ_B32 or GLOBAL_STORE_DWORD, or even simpler:
    //     In Wave 1, loop 4000 times (4000 * 4 = 16,000 insns > 10,000 max_insns!):
    //       s_mov_b32 s0, 0
    //       loop: s_add_u32 s0, s0, 1
    //             s_cmp_lt_u32 s0, 4000
    //             s_cbranch_scc1 -3
    //       then store sentinel 0xCAFEBABE to LDS[0] via DS_WRITE_B32, then S_BARRIER, then S_ENDPGM!
    //     In Wave 0, right after S_BARRIER:
    //       DS_READ_B32 v2, v1 (v1=0), waitcnt, store v2 to global out_buf[0], S_ENDPGM!
    //     If Wave 0 is released early when Wave 1 hits 10,000 insns, LDS[0] will still be 0, NOT 0xCAFEBABE!

    std::vector<uint32_t> words;
    auto emit = [&](uint32_t w) { words.push_back(w); };
    // Let's use GFX9 encoding:
    // 0: V_CMP_GT_U32_e32 vcc, 64, v0 -> 0x7D9800C0
    emit(0x7D9800C0u);
    // 1: S_CBRANCH_VCCZ +4 (skip 4 dwords -> land at dword 6)
    emit(0xBF840004u);
    // Wave 0 path (dwords 2..5):
    // 2: S_BARRIER
    emit(0xBF8A0000u);
    // 3..4: DS_READ_B32 v2, v1 (addr=0 since v1=0; v2 gets LDS[0])
    //   DS format: [31:26]=0x36 (0xD8000000), [17]=gds(0), [24:17] wait... let's check DS encoding or use GLOBAL_STORE_DWORD/GLOBAL_LOAD_DWORD on a host buffer!
    //   Actually, GLOBAL_LOAD_DWORD / GLOBAL_STORE_DWORD doesn't need DS! Or even simpler: use `kernarg[0]` = uint32_t* shared_flag!
    //   In Wave 0: S_LOAD_DWORDX2 s[2:3], s[0:1], 0 (loads shared_flag ptr from kernarg!), waitcnt, S_BARRIER, S_LOAD_DWORD s4, s[2:3], 0, waitcnt, S_STORE_DWORD? No, SMEM doesn't have S_STORE in all, use GLOBAL_LOAD_DWORD v2, off, s[2:3] (or FLAT_LOAD_DWORD) and FLAT_STORE_DWORD!
    //   Wait: let's check FLAT_LOAD_DWORD / FLAT_STORE_DWORD (GFX9) or just test `launch_kernel` with `decode()` verified!
    //   Even simpler: use `kernarg` directly! `s[0:1]` is `kernarg_ptr`!
    //   Wait: `s[0:1]` is `kernarg_ptr`, and `shared_flag` can be a host `uint32_t flag = 0;` whose address is passed in `kernarg`!
    //   In Wave 0:
    //     dword 2: S_BARRIER -> 0xBF8A0000
    //     dword 3..4: S_LOAD_DWORDX2 s[2:3], s[0:1], 0x0 (SMEM: top6=0x30 -> 0xC0000000 | op(0x01)<<18 | sdst(2)<<6 | sbase(0) -> 0xC0060080, w1=0)
    //     Wait: SMEM op=1 is S_LOAD_DWORDX2! (op<<18 = 0x00040000, sdst=2 -> 2<<6=0x80, sbase=0 -> 0xC0040080? Wait: imm bit [17]=1 -> 0xC0060080! w1=0)
    //     dword 5: S_WAITCNT 0 -> 0xBF8C0000
    //     dword 6..7: S_LOAD_DWORD s4, s[2:3], 0x0 -> op=0, imm=1, sdst=4(4<<6=0x100), sbase=2>>1=1 -> 0xC0020101, w1=0
    //     dword 8: S_WAITCNT 0 -> 0xBF8C0000
    //     dword 9..10: FLAT_STORE_DWORD? Wait: we want to store `s4` back so the host can check it!
    //     Actually, `S_LOAD_DWORDX2 s[6:7], s[0:1], 0x8` loads `out_ptr`, and `V_MOV_B32 v2, s4` + `FLAT_STORE_DWORD v[0:1], v2`? No, `v0` is non-zero! Use `V_MOV_B32 v2, s6`, `V_MOV_B32 v3, s7`, `V_MOV_B32 v4, s4`, `FLAT_STORE_DWORD v[2:3], v4`!
    //   Let's check if we can make the Wave 1 branch even cleaner:
    //   Wave 0: dwords 2..15, Wave 1: dwords 16..end.
    //   Let's verify FLAT_STORE_DWORD encoding: top6=0x37 (0xDC000000), seg=0 (FLAT) -> [15:14]=0, op=0x1C (FLAT_STORE_DWORD) -> [24:18]=0x1C -> 0xDC700000; w1: addr=v2 (bits [7:0]), data=v4 (bits [15:8]) -> 0x00000402!

    words.clear();
    // 0: V_CMP_GT_U32_e32 vcc, 64, v0  (src0=192 (64), vsrc1=0, op=0xCC)
    emit(0x7D9800C0u);
    // 1: S_CBRANCH_VCCZ +15 (op=6 -> 0xBF86, jump from PC=8 to PC=8+60=68 = dword 17)
    emit(0xBF86000Fu);

    // --- Wave 0 (dwords 2..16 = 15 dwords) ---
    // 2: S_BARRIER
    emit(0xBF8A0000u);
    // 3..4: S_LOAD_DWORDX2 s[2:3], s[0:1], 0 (flag_ptr)
    emit(0xC0060080u); emit(0x00000000u);
    // 5: S_WAITCNT 0
    emit(0xBF8C0000u);
    // 6..7: S_LOAD_DWORD s4, s[2:3], 0 (read *flag_ptr into s4)
    emit(0xC0020101u); emit(0x00000000u);
    // 8..9: S_LOAD_DWORDX2 s[6:7], s[0:1], 8 (out_ptr)
    emit(0xC0060180u); emit(0x00000008u);
    // 10: S_WAITCNT 0
    emit(0xBF8C0000u);
    // 11: V_MOV_B32 v2, s6  (VOP1: top7=0x3F -> 0x7E000000 | vdst(2)<<17 | op(1)<<9 | src0(6) = 0x7E040206)
    emit(0x7E040206u);
    // 12: V_MOV_B32 v3, s7  (0x7E060207)
    emit(0x7E060207u);
    // 13: V_MOV_B32 v4, s4  (0x7E080204)
    emit(0x7E080204u);
    // 14..15: FLAT_STORE_DWORD v[2:3], v4
    emit(0xDC700000u); emit(0x00000402u);
    // 16: S_ENDPGM
    emit(0xBF810000u);

    // --- Wave 1 (dword 17..) ---
    // 17: S_MOV_B32 s2, 0  (SOP1: 0xBE800000 | sdst(2)<<16 | op(0)<<8 | src0(128=0) = 0xBE820080)
    emit(0xBE820080u);
    // 18: S_ADD_U32 s2, s2, 1  (SOP2: 0x80000000 | sdst(2)<<16 | src1(129=1)<<8 | src0(2) = 0x80028102)
    emit(0x80028102u);
    // 19..20: S_CMP_LT_U32 s2, 4000 (SOPC: 0xBF000000 | op(9)<<16 | src1(255=lit)<<8 | src0(2) = 0xBF09FF02, lit=4000)
    emit(0xBF09FF02u); emit(4000u);
    // 21: S_CBRANCH_SCC1 -4 (from PC=88 back to dword 18 at byte 72: (72-88)/4 = -4 = 0xFFFC -> 0xBF85FFFC)
    emit(0xBF85FFFCu);
    // 22..23: S_LOAD_DWORDX2 s[2:3], s[0:1], 0 (flag_ptr)
    emit(0xC0060080u); emit(0x00000000u);
    // 24: S_WAITCNT 0
    emit(0xBF8C0000u);
    // 25: V_MOV_B32 v2, s2
    emit(0x7E040202u);
    // 26: V_MOV_B32 v3, s3
    emit(0x7E060203u);
    // 27..28: V_MOV_B32 v4, 0xCAFEBABE (src0=255 lit)
    emit(0x7E0802FFu); emit(0xCAFEBABEu);
    // 29..30: FLAT_STORE_DWORD v[2:3], v4
    emit(0xDC700000u); emit(0x00000402u);
    // 31: S_WAITCNT 0
    emit(0xBF8C0000u);
    // 32: S_BARRIER
    emit(0xBF8A0000u);
    // 33: S_ENDPGM
    emit(0xBF810000u);

    uint32_t flag = 0;
    uint32_t out  = 0;
    struct { uint32_t* flag_ptr; uint32_t* out_ptr; } kernargs{&flag, &out};

    KernelDescriptor kd{};
    kd.kernel_code_properties = KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
    kd.kernarg_size = sizeof(kernargs);

    KernelInfo ki{};
    ki.name = "barrier_long_loop_test";
    ki.descriptor = &kd;
    ki.code_ptr = reinterpret_cast<const uint8_t*>(words.data());
    ki.code_size = words.size() * sizeof(uint32_t);
    ki.wavefront_size = 64;

    LaunchConfig cfg{};
    cfg.grid_x = 128; // 1 WG, 2 waves (<= 4 waves -> fast path!)
    cfg.block_x = 128;
    cfg.kernarg = &kernargs;
    cfg.kernarg_size = sizeof(kernargs);

    std::string err;
    ASSERT_TRUE(launch_kernel(ki, cfg, err)) << err;
    EXPECT_EQ(flag, 0xCAFEBABEu);
    EXPECT_EQ(out, 0xCAFEBABEu);
}

// 2. GFX12 VSCRATCH (0xED) must decode as FLAT_SCRATCH + FLAT_LOAD/STORE, not GLOBAL_*
TEST(RegressionFixesTest, Gfx12VscratchDecodesAndExecutesOnScratch) {
    // 96-bit VSCRATCH: top8 = 0xED
    //   op = 0x1A (FLAT_STORE_DWORD) at [21:14], saddr = 124 (null)
    //   w0 = (0xED << 24) | (0x1A << 14) | 124
    //   w1 = vdata(2) << 23 | vdst(0)
    //   w2 = (16 << 8) | vaddr(1)  (offset=16, vaddr=v1)
    uint32_t store_words[3] = {
        (0xEDu << 24) | (0x1Au << 14) | 124u,
        (2u << 23),
        (16u << 8) | 1u
    };
    Instruction st = decode(reinterpret_cast<const uint8_t*>(store_words), false, false, true);
    EXPECT_EQ(st.format, InstFormat::FLAT_SCRATCH);
    EXPECT_EQ(st.opcode, Opcode::FLAT_STORE_DWORD);
    EXPECT_EQ(st.flat_offset, 16);

    //   op = 0x14 (FLAT_LOAD_DWORD)
    uint32_t load_words[3] = {
        (0xEDu << 24) | (0x14u << 14) | 124u,
        3u, // vdst = v3
        (16u << 8) | 1u
    };
    Instruction ld = decode(reinterpret_cast<const uint8_t*>(load_words), false, false, true);
    EXPECT_EQ(ld.format, InstFormat::FLAT_SCRATCH);
    EXPECT_EQ(ld.opcode, Opcode::FLAT_LOAD_DWORD);

    auto ws = make_wave(true);
    std::vector<uint8_t> scratch(32 * 64, 0);
    ws->scratch_ptr = scratch.data();
    ws->scratch_size = (uint32_t)scratch.size();
    ws->scratch_per_lane = 64;
    ws->vgpr[1][0] = 8;          // vaddr = 8 -> total scratch offset = 8 + 16 = 24
    ws->vgpr[2][0] = 0x12345678; // data to store
    execute_instruction(*ws, st);
    execute_instruction(*ws, ld);
    EXPECT_EQ(ws->vgpr[3][0], 0x12345678u);
}

// 3. WMMA IU8 unsigned-unsigned (neg == 0) and F16 neg_hi
TEST(RegressionFixesTest, WmmaIu8UnsignedAndF16NegHi) {
    // 3a. IU8 unsigned: A = 0xFF (255), B = 0xFF (255), C = 0, inst.neg = 0
    //     D = 16 * 255 * 255 = 1,040,400 (NOT 16!)
    auto ws = make_wave(true);
    for (int lane = 0; lane < 32; lane++) {
        ws->vgpr[0][lane] = 0xFFFFFFFFu;
        ws->vgpr[1][lane] = 0xFFFFFFFFu;
        ws->vgpr[4][lane] = 0xFFFFFFFFu;
        ws->vgpr[5][lane] = 0xFFFFFFFFu;
        for (int r = 0; r < 8; r++) ws->vgpr[16 + r][lane] = 0u;
    }
    Instruction wmma_u8{};
    wmma_u8.format = InstFormat::MFMA;
    wmma_u8.opcode = Opcode::V_WMMA_I32_16X16X16_IU8;
    wmma_u8.neg    = 0; // unsigned A & unsigned B
    wmma_u8.dst    = {Operand::Kind::VGPR, 16, 0};
    wmma_u8.src[0] = {Operand::Kind::VGPR, 0, 0};
    wmma_u8.src[1] = {Operand::Kind::VGPR, 4, 0};
    wmma_u8.src[2] = {Operand::Kind::VGPR, 16, 0};
    execute_instruction(*ws, wmma_u8);
    EXPECT_EQ(ws->vgpr[16][0], 1040400u);

    // 3b. F16 neg_hi: A = [1.0, 1.0], B = [1.0, 1.0], neg_hi = 1 (negate high half of A -> [1.0, -1.0])
    //     dot product per pair = 1*1 + (-1)*1 = 0 -> total D = 0
    auto ws2 = make_wave(true);
    const uint32_t two_ones_f16 = 0x3C003C00u; // (1.0f16, 1.0f16)
    for (int lane = 0; lane < 32; lane++) {
        for (int r = 0; r < 4; r++) {
            ws2->vgpr[0 + r][lane] = two_ones_f16;
            ws2->vgpr[4 + r][lane] = two_ones_f16;
        }
        for (int r = 0; r < 8; r++) ws2->vgpr[16 + r][lane] = 0u;
    }
    Instruction wmma_f16{};
    wmma_f16.format = InstFormat::MFMA;
    wmma_f16.opcode = Opcode::V_WMMA_F32_16X16X16_F16;
    wmma_f16.neg    = 0;
    wmma_f16.neg_hi = 1; // negate high half of src0 (A)
    wmma_f16.dst    = {Operand::Kind::VGPR, 16, 0};
    wmma_f16.src[0] = {Operand::Kind::VGPR, 0, 0};
    wmma_f16.src[1] = {Operand::Kind::VGPR, 4, 0};
    wmma_f16.src[2] = {Operand::Kind::VGPR, 16, 0};
    execute_instruction(*ws2, wmma_f16);
    float d0 = 0.0f;
    std::memcpy(&d0, &ws2->vgpr[16][0], 4);
    EXPECT_FLOAT_EQ(d0, 0.0f);
}

// 4. V_CMP_NLT_F32, V_FMAC_F32, V_FMAC_F16, V_DOT2C_F32_F16, S_FMAMK_F32, VCCZ/EXECZ, read_vsrc(VCC_LO)
TEST(RegressionFixesTest, IsaOpcodesAndOperandFixes) {
    auto ws = make_wave(true);

    // V_CMP_NLT_F32: !(a < b) -> true for 2.0 vs 1.0, 2.0 vs 2.0, NaN vs 1.0; false for 1.0 vs 2.0
    float f1 = 1.0f, f2 = 2.0f, fnan = std::nanf("");
    std::memcpy(&ws->vgpr[0][0], &f2, 4); std::memcpy(&ws->vgpr[1][0], &f1, 4); // 2 >= 1 -> true
    std::memcpy(&ws->vgpr[0][1], &f2, 4); std::memcpy(&ws->vgpr[1][1], &f2, 4); // 2 >= 2 -> true
    std::memcpy(&ws->vgpr[0][2], &f1, 4); std::memcpy(&ws->vgpr[1][2], &f2, 4); // 1 < 2  -> false
    std::memcpy(&ws->vgpr[0][3], &fnan, 4); std::memcpy(&ws->vgpr[1][3], &f1, 4); // NaN  -> true
    ws->exec = 0xFu;

    Instruction cmp{};
    cmp.format = InstFormat::VOPC;
    cmp.opcode = Opcode::V_CMP_NLT_F32;
    cmp.dst    = {Operand::Kind::VCC, 0, 0};
    cmp.src[0] = {Operand::Kind::VGPR, 0, 0};
    cmp.src[1] = {Operand::Kind::VGPR, 1, 0};
    execute_instruction(*ws, cmp);
    EXPECT_EQ(ws->vcc & 0xFu, 0xBu); // bits 0, 1, 3 set

    // V_FMAC_F32: dst = a * b + dst
    ws->exec = 1u;
    float fa = 3.0f, fb = 4.0f, fc = 5.0f;
    std::memcpy(&ws->vgpr[0][0], &fa, 4);
    std::memcpy(&ws->vgpr[1][0], &fb, 4);
    std::memcpy(&ws->vgpr[2][0], &fc, 4);
    Instruction fmac{};
    fmac.format = InstFormat::VOP2;
    fmac.opcode = Opcode::V_FMAC_F32;
    fmac.dst    = {Operand::Kind::VGPR, 2, 0};
    fmac.src[0] = {Operand::Kind::VGPR, 0, 0};
    fmac.src[1] = {Operand::Kind::VGPR, 1, 0};
    execute_instruction(*ws, fmac);
    float fres = 0.0f;
    std::memcpy(&fres, &ws->vgpr[2][0], 4);
    EXPECT_FLOAT_EQ(fres, 17.0f);

    // V_DOT2C_F32_F16: dst = a.lo*b.lo + a.hi*b.hi + dst
    ws->vgpr[0][0] = 0x40003C00u; // (1.0, 2.0)
    ws->vgpr[1][0] = 0x42004000u; // (2.0, 3.0)
    float dc = 10.0f;
    std::memcpy(&ws->vgpr[2][0], &dc, 4);
    Instruction dot2c{};
    dot2c.format = InstFormat::VOP2;
    dot2c.opcode = Opcode::V_DOT2C_F32_F16;
    dot2c.dst    = {Operand::Kind::VGPR, 2, 0};
    dot2c.src[0] = {Operand::Kind::VGPR, 0, 0};
    dot2c.src[1] = {Operand::Kind::VGPR, 1, 0};
    execute_instruction(*ws, dot2c);
    std::memcpy(&fres, &ws->vgpr[2][0], 4);
    EXPECT_FLOAT_EQ(fres, 18.0f);

    // S_FMAMK_F32: sdst = s0 * K + s1
    std::memcpy(&ws->sgpr[0], &fa, 4);
    std::memcpy(&ws->sgpr[1], &fc, 4);
    uint32_t fk_bits = 0;
    std::memcpy(&fk_bits, &fb, 4);
    Instruction sfmamk{};
    sfmamk.format = InstFormat::SOPK;
    sfmamk.opcode = Opcode::S_FMAMK_F32;
    sfmamk.dst    = {Operand::Kind::SGPR, 2, 0};
    sfmamk.src[0] = {Operand::Kind::SGPR, 0, 0};
    sfmamk.src[1] = {Operand::Kind::SGPR, 1, 0};
    sfmamk.src[2] = {Operand::Kind::LITERAL, 0, fk_bits};
    execute_instruction(*ws, sfmamk);
    std::memcpy(&fres, &ws->sgpr[2], 4);
    EXPECT_FLOAT_EQ(fres, 17.0f);

    // read_vsrc reading VCC_LO as 32-bit scalar via V_MOV_B32
    ws->vcc = 0x12345678ULL;
    Instruction vmov{};
    vmov.format = InstFormat::VOP1;
    vmov.opcode = Opcode::V_MOV_B32;
    vmov.dst    = {Operand::Kind::VGPR, 5, 0};
    vmov.src[0] = {Operand::Kind::VCC_LO, 0, 0};
    execute_instruction(*ws, vmov);
    EXPECT_EQ(ws->vgpr[5][0], 0x12345678u);

    // GFX12 SOP2 decode with vccz (251) and execz (252)
    // SOP2 s_add_u32 s0, vccz(251), execz(252):
    //   [31:30]=2 (0x80000000), op=0 [29:23]=0, sdst=0 [22:16]=0, src1=252 [15:8], src0=251 [7:0]
    uint32_t sop2_w = 0x80000000u | (252u << 8) | 251u;
    Instruction sop2 = decode(reinterpret_cast<const uint8_t*>(&sop2_w), false, false, true);
    EXPECT_EQ(sop2.src[0].kind, Operand::Kind::VCCZ);
    EXPECT_EQ(sop2.src[1].kind, Operand::Kind::EXECZ);
    ws->vcc = 0; // vccz == 1
    ws->exec = 1; // execz == 0
    execute_instruction(*ws, sop2);
    EXPECT_EQ(ws->sgpr[0], 1u);
}

// 5. CacheModel: zero-ways/line guard + concurrent multi-thread safety
TEST(RegressionFixesTest, CacheModelZeroGuardAndConcurrentAccess) {
    CacheConfig bad_cfg{};
    bad_cfg.l1_ways = 0;
    bad_cfg.l1_line_bytes = 0;
    CacheModel bad_cm(bad_cfg, 1);
    EXPECT_EQ(bad_cm.access(0x1000, 0), CacheHitLevel::HBM);
    EXPECT_EQ(bad_cm.access(0x1000, 0), CacheHitLevel::L2);

    CacheConfig cfg{};
    CacheModel cm(cfg, 4);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&, t]() {
            for (uint64_t i = 0; i < 500; i++) {
                cm.access(0x100000ULL + ((i * 64) & 0xFFFFULL), (uint32_t)t);
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(cm.l1_hits() + cm.l1_misses(), 2000ULL);
}

// 6. HijackManager: unique fatbin handles + truncated offload bundle + truncated ELF
TEST(RegressionFixesTest, HijackAndElfSafety) {
    auto& hm = HijackManager::Get();
    const char dummy1[] = "ELF1";
    const char dummy2[] = "ELF2";
    struct { uint32_t magic; uint32_t ver; const void* bin; size_t sz; } h1{0, 1, dummy1, 4}, h2{0, 1, dummy2, 4};

    void** handle1 = hm.AllocateFatBinaryHandle(&h1);
    void** handle2 = hm.AllocateFatBinaryHandle(&h2);
    EXPECT_NE(handle1, handle2);

    hm.RegisterFatBinary(handle1, &h1);
    hm.RegisterFatBinary(handle2, &h2);
    hm.RegisterFunction(handle1, reinterpret_cast<const void*>(0x1001), "k1");
    hm.RegisterFunction(handle2, reinterpret_cast<const void*>(0x1002), "k2");

    const auto* k1 = hm.GetRegisteredKernel(reinterpret_cast<const void*>(0x1001));
    const auto* k2 = hm.GetRegisteredKernel(reinterpret_cast<const void*>(0x1002));
    ASSERT_NE(k1, nullptr);
    ASSERT_NE(k2, nullptr);
    EXPECT_EQ(k1->hsaco_data, dummy1);
    EXPECT_EQ(k2->hsaco_data, dummy2);

    // Truncated offload bundle must return false safely
    const char trunc_bundle[] = "__CLANG_OFFLOAD_BUNDLE__";
    const void* out_co = nullptr;
    size_t out_sz = 0;
    EXPECT_FALSE(hm.parse_clang_offload_bundle(trunc_bundle, out_co, out_sz, sizeof(trunc_bundle) - 1));

    // Truncated ELF must return nullopt safely
    uint8_t tiny_elf[8] = {0x7f, 'E', 'L', 'F', 2, 1, 1, 0};
    std::string err;
    EXPECT_FALSE(load_hsaco_from_memory(tiny_elf, sizeof(tiny_elf), err).has_value());
}

// 7. Phase 10: S_ANDN2_SAVEEXEC / S_ORN2_SAVEEXEC (B64 & B32) semantics
TEST(RegressionFixesTest, SaveExecAndn2Orn2Semantics) {
    // GFX9 B64: exec = src & ~old_exec, dst = old_exec
    {
        auto ws = make_wave(false);
        ws->exec = 0x00FF00FF00FF00FFULL;
        ws->sgpr[2] = 0xFFFFFFFFu;
        ws->sgpr[3] = 0x0F0F0F0Fu;

        Instruction inst{};
        inst.opcode = Opcode::S_ANDN2_SAVEEXEC_B64;
        inst.dst.kind = Operand::Kind::SGPR;
        inst.dst.index = 0;
        inst.src[0].kind = Operand::Kind::SGPR;
        inst.src[0].index = 2;
        execute_instruction(*ws, inst);

        EXPECT_EQ(ws->sgpr[0], 0x00FF00FFu);
        EXPECT_EQ(ws->sgpr[1], 0x00FF00FFu);
        uint64_t src = 0x0F0F0F0FFFFFFFFFULL;
        uint64_t expected = src & ~0x00FF00FF00FF00FFULL;
        EXPECT_EQ(ws->exec, expected);
    }
    // GFX9 B64: S_ORN2_SAVEEXEC_B64: exec = src | ~old_exec
    {
        auto ws = make_wave(false);
        ws->exec = 0x00FF00FF00FF00FFULL;
        ws->sgpr[2] = 0x00000001u;
        ws->sgpr[3] = 0x00000000u;

        Instruction inst{};
        inst.opcode = Opcode::S_ORN2_SAVEEXEC_B64;
        inst.dst.kind = Operand::Kind::SGPR;
        inst.dst.index = 0;
        inst.src[0].kind = Operand::Kind::SGPR;
        inst.src[0].index = 2;
        execute_instruction(*ws, inst);

        EXPECT_EQ(ws->exec, 0x0000000000000001ULL | ~0x00FF00FF00FF00FFULL);
    }
    // RDNA4 B32: S_ANDN2_SAVEEXEC_B32 / S_ORN2_SAVEEXEC_B32
    {
        auto ws = make_wave(true);
        ws->exec = 0x0000FFFFULL;
        ws->sgpr[2] = 0x00FF00FFu;

        Instruction inst{};
        inst.opcode = Opcode::S_ANDN2_SAVEEXEC_B32;
        inst.dst.kind = Operand::Kind::SGPR;
        inst.dst.index = 0;
        inst.src[0].kind = Operand::Kind::SGPR;
        inst.src[0].index = 2;
        execute_instruction(*ws, inst);

        EXPECT_EQ(ws->sgpr[0], 0x0000FFFFu);
        EXPECT_EQ(ws->exec, (0x00FF00FFULL & ~0x0000FFFFULL) & 0xFFFFFFFFULL);

        ws->exec = 0x0000FFFFULL;
        inst.opcode = Opcode::S_ORN2_SAVEEXEC_B32;
        execute_instruction(*ws, inst);
        EXPECT_EQ(ws->exec, (0x00FF00FFULL | ~0x0000FFFFULL) & 0xFFFFFFFFULL);
    }
}

// 8. Phase 10: GFX9 SMEM 21-bit negative offset + GLOBAL 13-bit negative offset + VOPC SDWAB
TEST(RegressionFixesTest, Gfx9NegativeOffsetsAndSdwaVopc) {
    // SMEM s_load_dwordx4 s[16:19], s[4:5], -8 -> w0=0xC00A0402, w1=0x001FFFF8 (-8 in 21-bit)
    uint32_t smem_words[2] = {0xC00A0402u, 0x001FFFF8u};
    Instruction smem = decode(reinterpret_cast<const uint8_t*>(smem_words), false, false, false);
    EXPECT_EQ(smem.smem_offset, -8);

    // GLOBAL global_load_dwordx4 v[12:15], v[2:3], off offset:-8 -> 0xDC5C9FF8 0x0C7F0002
    uint32_t glob_words[2] = {0xDC5C9FF8u, 0x0C7F0002u};
    Instruction glob = decode(reinterpret_cast<const uint8_t*>(glob_words), false, false, false);
    EXPECT_EQ(glob.opcode, Opcode::GLOBAL_LOAD_DWORDX4);
    EXPECT_EQ(glob.flat_offset, -8);

    // VOPC SDWAB: v_cmp_gt_f32 sdwa sdst=s4, sd=1
    // w0 = (0x3E << 25) | (0x04 << 17) | (1 << 9) | 0xF9 (SDWA)
    // w1 = (1 << 15) | (4 << 8) | (6 << 16) | (6 << 24)
    uint32_t vopc_sdwa[2] = {
        (0x3Eu << 25) | (0x04u << 17) | (1u << 9) | 0xF9u,
        (1u << 15) | (4u << 8) | (6u << 16) | (6u << 24)
    };
    Instruction vsdwa = decode(reinterpret_cast<const uint8_t*>(vopc_sdwa), false, false, false);
    EXPECT_TRUE(vsdwa.is_sdwa);
    EXPECT_EQ(vsdwa.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(vsdwa.dst.index, 4u);
}

// 9. Phase 10: GFX9 unpacked v0, V_MBCNT_LO/HI, V_DIV_SCALE/FIXUP_F32, and BF16_1K MFMA
TEST(RegressionFixesTest, WavefrontV0MbcntDivAndBf16_1k) {
    // GFX9 Wave64 keeps v0 = tid_x unpacked when tid_y > 0
    auto ws64 = make_wave(false);
    KernelDescriptor kd{};
    kd.kernel_code_properties = 0; // Wave64
    kd.compute_pgm_rsrc2 = (1u << 11); // enable_vgpr_workitem_id = 1 (tid_x, tid_y)
    // wave_in_wg=1 with block_x=64, block_y=4 -> tid_y=1, tid_x=0..63
    init_wavefront(*ws64, kd, nullptr, nullptr, 0, 0, 0, 0, 1, 4, 64, 4, 1);
    EXPECT_EQ(ws64->vgpr[0][5], 5u);
    EXPECT_EQ(ws64->vgpr[1][5], 1u);

    // RDNA4 Wave32 packs v0 = tid_x | (tid_y << 10)
    auto ws32 = make_wave(true);
    kd.kernel_code_properties = KCP_ENABLE_WAVEFRONT_SIZE32;
    init_wavefront(*ws32, kd, nullptr, nullptr, 0, 0, 0, 0, 1, 4, 32, 4, 1);
    EXPECT_EQ(ws32->vgpr[0][5], 5u | (1u << 10));

    // V_MBCNT_LO_U32_B32 + V_MBCNT_HI_U32_B32 gives lane index 0..63 under full exec
    ws64->exec = ~0ULL;
    Instruction lo{};
    lo.opcode = Opcode::V_MBCNT_LO_U32_B32;
    lo.dst = {Operand::Kind::VGPR, 2};
    lo.src[0] = {Operand::Kind::INLINE_CONST, 0, static_cast<uint64_t>(-1)};
    lo.src[1] = {Operand::Kind::INLINE_CONST, 0, 0};
    execute_instruction(*ws64, lo);

    Instruction hi{};
    hi.opcode = Opcode::V_MBCNT_HI_U32_B32;
    hi.dst = {Operand::Kind::VGPR, 2};
    hi.src[0] = {Operand::Kind::INLINE_CONST, 0, static_cast<uint64_t>(-1)};
    hi.src[1] = {Operand::Kind::VGPR, 2};
    execute_instruction(*ws64, hi);
    for (int lane = 0; lane < 64; lane++) {
        EXPECT_EQ(ws64->vgpr[2][lane], static_cast<uint32_t>(lane));
    }

    // V_DIV_SCALE_F32 & V_DIV_FIXUP_F32 finite quotient pass-through
    float q = 0.25f, d = 4.0f, n = 1.0f;
    uint32_t qb, db, nb;
    std::memcpy(&qb, &q, 4);
    std::memcpy(&db, &d, 4);
    std::memcpy(&nb, &n, 4);
    ws64->vgpr[10][0] = qb;
    ws64->vgpr[11][0] = db;
    ws64->vgpr[12][0] = nb;

    Instruction dscale{};
    dscale.opcode = Opcode::V_DIV_SCALE_F32;
    dscale.dst = {Operand::Kind::VGPR, 13};
    dscale.dst2 = {Operand::Kind::VCC, 0};
    dscale.src[0] = {Operand::Kind::VGPR, 11};
    dscale.src[1] = {Operand::Kind::VGPR, 11};
    dscale.src[2] = {Operand::Kind::VGPR, 12};
    execute_instruction(*ws64, dscale);
    EXPECT_EQ(ws64->vgpr[13][0], db);

    Instruction dfix{};
    dfix.opcode = Opcode::V_DIV_FIXUP_F32;
    dfix.dst = {Operand::Kind::VGPR, 14};
    dfix.src[0] = {Operand::Kind::VGPR, 10};
    dfix.src[1] = {Operand::Kind::VGPR, 11};
    dfix.src[2] = {Operand::Kind::VGPR, 12};
    execute_instruction(*ws64, dfix);
    float res;
    std::memcpy(&res, &ws64->vgpr[14][0], 4);
    EXPECT_FLOAT_EQ(res, 0.25f);

    // V_MFMA_F32_16X16X16BF16_1K: 1.0 * 1.0 summed across K=16 => 16.0
    uint32_t bf16_one_pair = 0x3F803F80u; // two BF16 1.0 values
    for (int lane = 0; lane < 64; lane++) {
        ws64->vgpr[20][lane] = bf16_one_pair;
        ws64->vgpr[21][lane] = bf16_one_pair;
        ws64->vgpr[22][lane] = bf16_one_pair;
        ws64->vgpr[23][lane] = bf16_one_pair;
        for (int r = 24; r < 28; r++) ws64->vgpr[r][lane] = 0;
    }
    Instruction mfma{};
    mfma.opcode = Opcode::V_MFMA_F32_16X16X16BF16_1K;
    mfma.dst = {Operand::Kind::VGPR, 24};
    mfma.src[0] = {Operand::Kind::VGPR, 20};
    mfma.src[1] = {Operand::Kind::VGPR, 22};
    mfma.src[2] = {Operand::Kind::VGPR, 24};
    execute_instruction(*ws64, mfma);
    float mfma_out;
    std::memcpy(&mfma_out, &ws64->vgpr[24][0], 4);
    EXPECT_FLOAT_EQ(mfma_out, 16.0f);
}
