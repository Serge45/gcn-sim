#include <gtest/gtest.h>
#include "gcnsim/isa_decoder.h"
#include <cstdint>
#include <cstring>

using namespace gcnsim;

static uint8_t make_bytes(uint32_t w0) {
    // helper to get first byte — actually we need to write full arrays
    (void)w0;
    return 0;
}

static std::array<uint8_t, 8> enc2(uint32_t w0, uint32_t w1 = 0) {
    std::array<uint8_t, 8> b{};
    memcpy(b.data(),     &w0, 4);
    memcpy(b.data() + 4, &w1, 4);
    return b;
}

// --- SOPP tests ---
TEST(ISADecoder, SEndpgm) {
    // s_endpgm = 0xBF810000
    auto b = enc2(0xBF810000);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPP);
    EXPECT_EQ(inst.opcode, Opcode::S_ENDPGM);
    EXPECT_EQ(inst.size_bytes, 4);
}

TEST(ISADecoder, SNop) {
    // s_nop 0 = 0xBF800000
    auto b = enc2(0xBF800000);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPP);
    EXPECT_EQ(inst.opcode, Opcode::S_NOP);
    EXPECT_EQ(inst.size_bytes, 4);
}

TEST(ISADecoder, SWaitcntVmcnt0) {
    // s_waitcnt vmcnt(0) = 0xBF8C0F70
    auto b = enc2(0xBF8C0F70);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPP);
    EXPECT_EQ(inst.opcode, Opcode::S_WAITCNT);
    EXPECT_EQ(inst.size_bytes, 4);
}

TEST(ISADecoder, SWaitcntLgkmcnt0) {
    // s_waitcnt lgkmcnt(0) = 0xBF8CC07F
    auto b = enc2(0xBF8CC07F);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPP);
    EXPECT_EQ(inst.opcode, Opcode::S_WAITCNT);
}

// --- SOP1 tests ---
TEST(ISADecoder, SMovB32) {
    // s_mov_b32 s33, 0 = 0xBEA10080
    auto b = enc2(0xBEA10080);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP1);
    EXPECT_EQ(inst.opcode, Opcode::S_MOV_B32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 33u);
    // src0 = inline const 0 (encoding 128)
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[0].imm, 0u);
    EXPECT_EQ(inst.size_bytes, 4);
}

TEST(ISADecoder, SGetpcB64) {
    // s_getpc_b64 s[0:1] = 0xBE801C00
    auto b = enc2(0xBE801C00);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP1);
    EXPECT_EQ(inst.opcode, Opcode::S_GETPC_B64);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 0u);
}

// --- SOP2 tests ---
TEST(ISADecoder, SAddU32) {
    // s_add_u32 s0, s0, imm32 = 0x8000FF00 0x0000FFFF
    auto b = enc2(0x8000FF00, 0x0000FFFF);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_ADD_U32);
    EXPECT_EQ(inst.size_bytes, 8);  // has literal
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::LITERAL);
    EXPECT_EQ(inst.src[1].imm, 0x0000FFFFu);
}

TEST(ISADecoder, SAddI32) {
    // s_add_i32 s1, s1, s8 = 0x81010801
    auto b = enc2(0x81010801);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_ADD_I32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 1u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[1].index, 8u);
    EXPECT_EQ(inst.size_bytes, 4);
}

TEST(ISADecoder, SAndB32WithLiteral) {
    // s_and_b32 s0, s0, 0xffff = 0x8600FF00 0x0000FFFF
    auto b = enc2(0x8600FF00, 0x0000FFFF);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_AND_B32);
    EXPECT_EQ(inst.size_bytes, 8);
}

TEST(ISADecoder, SMulI32) {
    // s_mul_i32 s8, s8, s0 = 0x92080008
    auto b = enc2(0x92080008);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_MUL_I32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 8u);
}

// --- SMEM tests ---
TEST(ISADecoder, SLoadDword) {
    // s_load_dword s0, s[4:5], 0x4 = 0xC0020002 0x00000004
    auto b = enc2(0xC0020002, 0x00000004);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SMEM);
    EXPECT_EQ(inst.opcode, Opcode::S_LOAD_DWORD);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[0].index, 4u);  // sbase * 2
    EXPECT_EQ(inst.smem_offset, 4);
}

TEST(ISADecoder, SLoadDwordx2) {
    // s_load_dwordx2 s[0:1], s[6:7], 0x18 = 0xC0060003 0x00000018
    auto b = enc2(0xC0060003, 0x00000018);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SMEM);
    EXPECT_EQ(inst.opcode, Opcode::S_LOAD_DWORDX2);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.smem_offset, 0x18);
}

// --- VOP1 tests ---
TEST(ISADecoder, VMovB32Inline0) {
    // v_mov_b32_e32 v0, 0 = 0x7E000280
    auto b = enc2(0x7E000280);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP1);
    EXPECT_EQ(inst.opcode, Opcode::V_MOV_B32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[0].imm, 0u);
    EXPECT_EQ(inst.size_bytes, 4);
}

// --- VOP2 tests ---
TEST(ISADecoder, VAddU32) {
    // v_add_u32_e32 v0, s1, v0 = 0x68000001
    auto b = enc2(0x68000001);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_ADD_U32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[1].index, 0u);
}

TEST(ISADecoder, VAndB32) {
    // v_and_b32_e32 v1, 31, v0 = 0x2602009F
    // 0x9F = 159 = inline const 31
    auto b = enc2(0x2602009F);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_AND_B32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 1u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[0].imm, 31u);
}

TEST(ISADecoder, VLshlrevB32) {
    // v_lshlrev_b32_e32 v1, 2, v1 = 0x24020282
    auto b = enc2(0x24020282);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_LSHLREV_B32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 1u);
    // src0 = inline const 2 (encoding 0x82 = 130, inline = 130-128 = 2)
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[0].imm, 2u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[1].index, 1u);
}

TEST(ISADecoder, VAddCoU32) {
    // v_add_co_u32_e32 v0, vcc, s0, v0 = 0x32000000
    auto b = enc2(0x32000000);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_ADD_CO_U32);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 0u);
}

TEST(ISADecoder, VAddc) {
    // v_addc_co_u32_e32 v1, vcc, v2, v1, vcc = 0x38020302
    auto b = enc2(0x38020302);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_ADDC_CO_U32);
}

// --- VOP3 tests ---
TEST(ISADecoder, VMulLoU32) {
    // v_mul_lo_u32 v1, v0, 10 = 0xD2850001 0x00011500
    // src1 = 0x0A + 128 = 138? No: (0x00011500 >> 9) & 0x1FF = 0x8A = 138
    // 138 = SGPR 138? But inline const 10 = 128+10=138. YES!
    auto b = enc2(0xD2850001, 0x00011500);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP3A);
    EXPECT_EQ(inst.opcode, Opcode::V_MUL_LO_U32);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 1u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 0u);  // src0=256=VGPR0
    // src1 = 138 = inline const 10
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[1].imm, 10u);
}

TEST(ISADecoder, VAshrrevI64) {
    // v_ashrrev_i64 v[0:1], 30, v[0:1] = 0xD2910000 0x0002009E
    // src0 = 0x9E = 158 = inline const 30
    // src1 = (0x0002009E >> 9) & 0x1FF = 0x100 = 256 = VGPR0
    auto b = enc2(0xD2910000, 0x0002009E);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP3A);
    EXPECT_EQ(inst.opcode, Opcode::V_ASHRREV_I64);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::INLINE_CONST);
    EXPECT_EQ(inst.src[0].imm, 30u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[1].index, 0u);
}

// --- FLAT/GLOBAL tests ---
TEST(ISADecoder, GlobalLoadDword) {
    // global_load_dword v3, v1, s[0:1] = 0xDC508000 0x03000001
    auto b = enc2(0xDC508000, 0x03000001);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::FLAT_GLOBAL);
    EXPECT_EQ(inst.opcode, Opcode::GLOBAL_LOAD_DWORD);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 3u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    // saddr = 0 = s[0:1]
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[1].index, 0u);
}

TEST(ISADecoder, GlobalStoreDword) {
    // global_store_dword v[0:1], v3, off = 0xDC708000 0x007F0300
    // vaddr=v[0:1]=0, saddr=0x7F=off, vdata=v3
    auto b = enc2(0xDC708000, 0x007F0300);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::FLAT_GLOBAL);
    EXPECT_EQ(inst.opcode, Opcode::GLOBAL_STORE_DWORD);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 0u);  // vaddr
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::NONE);  // off = no saddr
    EXPECT_EQ(inst.src[2].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[2].index, 3u);  // vdata
}

// --- SOPK tests ---
TEST(ISADecoder, SMovkI32) {
    // s_movk_i32 s0, 0x1234 — op=0, SIMM16=0x1234, SDST=s0
    // Encoding: [31:28]=0xB=1011, [26:23]=op=0, [22:16]=sdst=0, [15:0]=simm16=0x1234
    // = (0xBu<<28) | (0u<<23) | (0u<<16) | 0x1234
    uint32_t w0 = (0xBu << 28) | (0u << 23) | (0u << 16) | 0x1234u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPK);
    EXPECT_EQ(inst.opcode, Opcode::S_MOVK_I32);
    EXPECT_EQ(inst.size_bytes, 4);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.simm16, (int16_t)0x1234);
}

TEST(ISADecoder, SAddk) {
    // s_addk_i32 s5, 0x0007 — op=14, sdst=5, simm16=7
    // Encoding: [31:28]=0xB, [26:23]=op=14, [22:16]=sdst=5, [15:0]=7
    uint32_t w0 = (0xBu << 28) | (14u << 23) | (5u << 16) | 7u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPK);
    EXPECT_EQ(inst.opcode, Opcode::S_ADDK_I32);
    EXPECT_EQ(inst.simm16, (int16_t)7);
    EXPECT_EQ(inst.dst.index, 5u);
}

// --- SOPC tests ---
TEST(ISADecoder, SCmpEqI32) {
    // s_cmp_eq_i32 s0, s1 — SOPC, [31:23]=0x17E, [22:16]=op=0, [15:8]=ssrc1=1, [7:0]=ssrc0=0
    uint32_t w0 = (0x17Eu << 23) | (0u << 16) | (1u << 8) | 0u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPC);
    EXPECT_EQ(inst.opcode, Opcode::S_CMP_EQ_I32);
    EXPECT_EQ(inst.size_bytes, 4);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[0].index, 0u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[1].index, 1u);
}

TEST(ISADecoder, SCmpGtU32) {
    // s_cmp_gt_u32 s2, s3 — SOPC op=8, ssrc0=2, ssrc1=3
    uint32_t w0 = (0x17Eu << 23) | (8u << 16) | (3u << 8) | 2u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOPC);
    EXPECT_EQ(inst.opcode, Opcode::S_CMP_GT_U32);
}

// --- DS tests ---
// DS encoding: [31:26]=0x36, [24:17]=OP, [16]=GDS, [15:0]=OFFSET
TEST(ISADecoder, DsWriteB32) {
    // ds_write_b32 v0, v1 — DS op=13, addr=v0 (in w1[7:0]), data0=v1 (in w1[15:8])
    // Verified: llvm-objdump encoding D81A0000 has bits[24:17]=13
    uint32_t w0 = (0x36u << 26) | (13u << 17);
    uint32_t w1 = (0u << 24) | (0u << 16) | (1u << 8) | 0u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::DS);
    EXPECT_EQ(inst.opcode, Opcode::DS_WRITE_B32);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR); // addr
    EXPECT_EQ(inst.src[0].index, 0u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR); // data0
    EXPECT_EQ(inst.src[1].index, 1u);
}

TEST(ISADecoder, DsReadB32) {
    // ds_read_b32 v2, v0 — DS op=54, vdst=v2, addr=v0
    // Verified: llvm-objdump encoding D86C0000 has bits[24:17]=54
    uint32_t w0 = (0x36u << 26) | (54u << 17);
    uint32_t w1 = (2u << 24) | 0u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::DS);
    EXPECT_EQ(inst.opcode, Opcode::DS_READ_B32);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 2u);
}

// --- MUBUF tests ---
TEST(ISADecoder, BufferLoadDword) {
    // buffer_load_dword v4, v0, s[8:11], s0 offen
    // word0: [31:26]=0x38, [24:18]=op=20, [12]=offen=1
    // word1: [31:24]=soffset=s0, [20:16]=srsrc=2(=s8), [15:8]=vdata=v4, [7:0]=vaddr=v0
    uint32_t w0 = (0x38u << 26) | (20u << 18) | (1u << 12);
    uint32_t w1 = (0u << 24) | (2u << 16) | (4u << 8) | 0u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::MUBUF);
    EXPECT_EQ(inst.opcode, Opcode::BUFFER_LOAD_DWORD);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_TRUE(inst.mubuf_offen);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 4u);
}

// --- VOP3P tests ---
TEST(ISADecoder, VPkAddF16) {
    // v_pk_add_f16 v2, v0, v1 — VOP3P format
    // [31:26]=0b110011=0x33, [25:23]=op-bits, [22:16]=vdst
    // v_pk_add_f16 has op within VOP3P
    // op for v_pk_add_f16 varies by assembler, let's use what our decoder assigns
    // From vop3p_opcode: op=0x1E corresponds to V_PK_ADD_F16 per our table
    // [31:26]=0x33, [25:23]=?, [22:16]=vdst=2
    // word0 bits: [25:16]=op_and_dst, let's construct from encoder knowledge
    // VOP3P: w[31:26]=0x33, w[25:23]=op[8:6], w[22:16]=dst, op total 9 bits
    // For simplicity: just check format detection
    uint32_t w0 = (0x33u << 26) | (2u << 16); // dst=v2, op bits in [25:23]=0
    uint32_t w1 = (0u << 9) | (1u << 0) | (0u << 9); // src1=v1=256+1, src0=256+0
    // src0/src1 encoding: VGPR0=256, VGPR1=257
    w1 = (257u << 9) | 256u; // src0=v0, src1=v1
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP3P);
    EXPECT_EQ(inst.size_bytes, 8);
}

// --- MFMA tests ---
TEST(ISADecoder, MfmaF32_4x4x1f32) {
    // v_mfma_f32_4x4x1f32 agpr[0:3], v0, v1, agpr[0:3]
    // MFMA: [31:26]=0x34 (same as VOP3), op in [25:18] = MFMA op
    // v_mfma_f32_4x4x1f32 MFMA op = 2 (per mfma_opcode table)
    // Word0: [31:26]=0x34, [25:18]=op=2, [17:16]=?, [15:8]=src1, [7:0]=src0
    // src0=v0=256, src1=v1=257, dst=agpr0=0
    // Actually the MFMA encoding has op at [25:18] in the 64-bit instruction
    // word0: [31:26]=0x34=110100, [25:18]=op=2=00000010
    // bits: 1101 0000 0000 1000 = 0xD008? Let's build it:
    // [31:26]=0b110100 → top 6 bits of w0 = 0x34 << 26
    // [25:18]=2 → bits 25:18 of w0
    uint32_t w0 = (0x34u << 26) | (2u << 18);
    // word1: [31:24]=vdst=0, [23:16]=src2, [15:8]=src1, [7:0]=src0
    uint32_t w1 = (0u << 24) | (0u << 16) | (1u << 8) | 0u; // src0=v0, src1=v1, src2=agpr0
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    // MFMA ops are in VOP3 encoding space; format may be VOP3A or MFMA
    EXPECT_TRUE(inst.format == InstFormat::MFMA || inst.format == InstFormat::VOP3A);
    EXPECT_EQ(inst.size_bytes, 8);
}

TEST(ISADecoder, CDNA3_MfmaOpcodes) {
    auto decode_mfma = [](uint32_t hw_op) {
        // [31:26]=0x34, [25:16]=hw_op, [15]=1 (is_agpr), [7:0]=vdst
        uint32_t w0 = (0x34u << 26) | ((hw_op & 0x3FFu) << 16) | (1u << 15) | 4u;
        uint32_t w1 = (256u) | (257u << 9) | (0u << 18); // src0=v0, src1=v1, src2=agpr0
        auto b = enc2(w0, w1);
        return decode(b.data());
    };

    // XF32
    Instruction inst_xf32_16 = decode_mfma(958);
    EXPECT_EQ(inst_xf32_16.format, InstFormat::MFMA);
    EXPECT_EQ(inst_xf32_16.opcode, Opcode::V_MFMA_F32_16X16X8_XF32);
    EXPECT_STREQ(opcode_name(inst_xf32_16.opcode), "v_mfma_f32_16x16x8_xf32");

    Instruction inst_xf32_32 = decode_mfma(959);
    EXPECT_EQ(inst_xf32_32.format, InstFormat::MFMA);
    EXPECT_EQ(inst_xf32_32.opcode, Opcode::V_MFMA_F32_32X32X4_XF32);
    EXPECT_STREQ(opcode_name(inst_xf32_32.opcode), "v_mfma_f32_32x32x4_xf32");

    // INT8 double-rate
    Instruction inst_i8_32 = decode_mfma(982);
    EXPECT_EQ(inst_i8_32.format, InstFormat::MFMA);
    EXPECT_EQ(inst_i8_32.opcode, Opcode::V_MFMA_I32_32X32X16I8);
    EXPECT_STREQ(opcode_name(inst_i8_32.opcode), "v_mfma_i32_32x32x16i8");

    Instruction inst_i8_16 = decode_mfma(983);
    EXPECT_EQ(inst_i8_16.format, InstFormat::MFMA);
    EXPECT_EQ(inst_i8_16.opcode, Opcode::V_MFMA_I32_16X16X32I8);
    EXPECT_STREQ(opcode_name(inst_i8_16.opcode), "v_mfma_i32_16x16x32i8");

    // FP8 / BF8 mixed and pure
    Instruction inst_bf8 = decode_mfma(1008);
    EXPECT_EQ(inst_bf8.format, InstFormat::MFMA);
    EXPECT_EQ(inst_bf8.opcode, Opcode::V_MFMA_F32_16X16X32_BF8_BF8);
    EXPECT_STREQ(opcode_name(inst_bf8.opcode), "v_mfma_f32_16x16x32_bf8_bf8");

    Instruction inst_fp8 = decode_mfma(1011);
    EXPECT_EQ(inst_fp8.format, InstFormat::MFMA);
    EXPECT_EQ(inst_fp8.opcode, Opcode::V_MFMA_F32_16X16X32_FP8_FP8);
    EXPECT_STREQ(opcode_name(inst_fp8.opcode), "v_mfma_f32_16x16x32_fp8_fp8");

    Instruction inst_fp8_bf8 = decode_mfma(1014);
    EXPECT_EQ(inst_fp8_bf8.format, InstFormat::MFMA);
    EXPECT_EQ(inst_fp8_bf8.opcode, Opcode::V_MFMA_F32_32X32X16_FP8_BF8);
    EXPECT_STREQ(opcode_name(inst_fp8_bf8.opcode), "v_mfma_f32_32x32x16_fp8_bf8");

    // CDNA4 (GFX950) New MFMA
    Instruction inst_f16_16 = decode_mfma(1016);
    EXPECT_EQ(inst_f16_16.format, InstFormat::MFMA);
    EXPECT_EQ(inst_f16_16.opcode, Opcode::V_MFMA_F32_16X16X32_F16);
    EXPECT_STREQ(opcode_name(inst_f16_16.opcode), "v_mfma_f32_16x16x32_f16");

    Instruction inst_f16_32 = decode_mfma(1017);
    EXPECT_EQ(inst_f16_32.format, InstFormat::MFMA);
    EXPECT_EQ(inst_f16_32.opcode, Opcode::V_MFMA_F32_32X32X16_F16);
    EXPECT_STREQ(opcode_name(inst_f16_32.opcode), "v_mfma_f32_32x32x16_f16");

    Instruction inst_scale_16 = decode_mfma(1020);
    EXPECT_EQ(inst_scale_16.format, InstFormat::MFMA);
    EXPECT_EQ(inst_scale_16.opcode, Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4);
    EXPECT_STREQ(opcode_name(inst_scale_16.opcode), "v_mfma_scale_f32_16x16x128_f8f6f4");

    Instruction inst_scale_32 = decode_mfma(1021);
    EXPECT_EQ(inst_scale_32.format, InstFormat::MFMA);
    EXPECT_EQ(inst_scale_32.opcode, Opcode::V_MFMA_SCALE_F32_32X32X64_F8F6F4);
    EXPECT_STREQ(opcode_name(inst_scale_32.opcode), "v_mfma_scale_f32_32x32x64_f8f6f4");
}

// --- VOP2: v_cndmask_b32 (Phase 1) ---
TEST(ISADecoder, VCndmaskB32) {
    // v_cndmask_b32_e32 v2, v0, v1, vcc
    // VOP2 op=0, vdst=2, vsrc1=1, src0=v0(=256)
    // Encoding: [31]=0, [30:25]=0, [24:17]=vdst=2, [16:9]=vsrc1=1, [8:0]=src0=256=0x100
    // But 9-bit src0 can only encode 0–511; VGPR0=256=0x100
    // w0 = (0<<25) | (2<<17) | (1<<9) | 0x100 = 0x00040300
    uint32_t w0 = (0u << 25) | (2u << 17) | (1u << 9) | 0x100u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_CNDMASK_B32);
    EXPECT_EQ(inst.size_bytes, 4);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.dst.index, 2u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 0u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[1].index, 1u);
}

// --- SOP2: s_cselect_b32 / s_cselect_b64 (Phase 1) ---
TEST(ISADecoder, SCselectB32) {
    // s_cselect_b32 s0, s1, s2 — SOP2 op=10
    // SOP2: [31:30]=10, [29:23]=op=10, [22:16]=sdst=0, [15:8]=ssrc1=2, [7:0]=ssrc0=1
    uint32_t w0 = (2u << 30) | (10u << 23) | (0u << 16) | (2u << 8) | 1u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_CSELECT_B32);
    EXPECT_EQ(inst.size_bytes, 4);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.src[1].index, 2u);
}

TEST(ISADecoder, SCselectB64) {
    // s_cselect_b64 s[4:5], s[0:1], s[2:3] — SOP2 op=11
    uint32_t w0 = (2u << 30) | (11u << 23) | (4u << 16) | (2u << 8) | 0u;
    auto b = enc2(w0);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::SOP2);
    EXPECT_EQ(inst.opcode, Opcode::S_CSELECT_B64);
    EXPECT_EQ(inst.dst.kind, Operand::Kind::SGPR);
    EXPECT_EQ(inst.dst.index, 4u);
}

// --- VOP3 modifiers (Phase 1) ---
TEST(ISADecoder, VOP3Modifiers) {
    // v_add_f32_e64 v0, -|v1|, v2 clamp mul:2
    // VOP3A: w0[31:26]=0x34, w0[25:16]=op, w0[15]=CLAMP, w0[10:8]=ABS, w0[7:0]=VDST
    // v_add_f32 is VOP2 op=1, VOP3 encoding: op = 0x100 + 1 = 0x101
    // ABS bit 0 set (abs src0) → w0[8]=1
    // CLAMP set → w0[15]=1
    // NEG bit 0 set (neg src0) → w1[29]=1
    // OMOD=1 (mul:2) → w1[27]=1
    // VDST=0, src0=v1=257, src1=v2=258
    uint32_t w0 = (0x34u << 26) | (0x101u << 16) | (1u << 15) | (1u << 8) | 0u;
    uint32_t w1 = (1u << 29) | (1u << 27) | (258u << 9) | 257u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP3A);
    EXPECT_EQ(inst.opcode, Opcode::V_ADD_F32);
    EXPECT_EQ(inst.neg, 1u);      // neg src0
    EXPECT_EQ(inst.abs_, 1u);     // abs src0
    EXPECT_EQ(inst.clamp, 1u);
    EXPECT_EQ(inst.omod, 1u);     // mul:2
    EXPECT_EQ(inst.dst.index, 0u);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.src[1].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[1].index, 2u);
}

// --- VOP2 SDWA sub-encoding (Phase 2) ---
TEST(ISADecoder, VOP2_SDWA) {
    // v_add_f32 v0, v1, v2 src0_sel:BYTE_0 src1_sel:WORD_1 dst_sel:WORD_0
    // VOP2 with src0=0xF9 triggers SDWA
    // w0: [31]=0, [30:25]=op=1(V_ADD_F32), [24:17]=vdst=0, [16:9]=vsrc1=2, [8:0]=0xF9
    uint32_t w0 = (1u << 25) | (0u << 17) | (2u << 9) | 0xF9u;
    // SDWA w1: [7:0]=real_src0=1(v1), [10:8]=src0_sel=0(BYTE_0), [18:16]=src1_sel=5(WORD_1),
    //          [26:24]=dst_sel=4(WORD_0), [28:27]=dst_unused=0(PAD)
    uint32_t w1 = 1u | (0u << 8) | (5u << 16) | (4u << 24);
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_ADD_F32);
    EXPECT_TRUE(inst.is_sdwa);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.sdwa_src0_sel, 0u);  // BYTE_0
    EXPECT_EQ(inst.sdwa_src1_sel, 5u);  // WORD_1
    EXPECT_EQ(inst.sdwa_dst_sel, 4u);   // WORD_0
}

// --- VOP2 DPP sub-encoding (Phase 2) ---
TEST(ISADecoder, VOP2_DPP) {
    // v_add_f32 v0, v1, v2 dpp:row_shr:1 bound_ctrl:0 row_mask:0xF bank_mask:0xF
    // VOP2 with src0=0xFA triggers DPP
    // w0: [31]=0, [30:25]=op=1(V_ADD_F32), [24:17]=vdst=0, [16:9]=vsrc1=2, [8:0]=0xFA
    uint32_t w0 = (1u << 25) | (0u << 17) | (2u << 9) | 0xFAu;
    // DPP w1: [7:0]=real_src0=1(v1), [16:8]=dpp_ctrl=0x111(row_shr:1),
    //         [19]=bound_ctrl=1, [27:24]=bank_mask=0xF, [31:28]=row_mask=0xF
    uint32_t w1 = 1u | (0x111u << 8) | (1u << 19) | (0xFu << 24) | (0xFu << 28);
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP2);
    EXPECT_EQ(inst.opcode, Opcode::V_ADD_F32);
    EXPECT_TRUE(inst.is_dpp);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 1u);
    EXPECT_EQ(inst.dpp_ctrl, 0x111u);   // row_shr:1
    EXPECT_TRUE(inst.dpp_bound_ctrl);
    EXPECT_EQ(inst.dpp_bank_mask, 0xFu);
    EXPECT_EQ(inst.dpp_row_mask, 0xFu);
}

// --- VOP1 SDWA sub-encoding (Phase 2) ---
TEST(ISADecoder, VOP1_SDWA) {
    // v_mov_b32 v0, v5 src0_sel:BYTE_2
    // VOP1 with src0=0xF9
    // VOP1: [31:25]=0x3F, [24:17]=vdst=0, [16:9]=op=1(v_mov_b32), [8:0]=0xF9
    uint32_t w0 = (0x3Fu << 25) | (0u << 17) | (1u << 9) | 0xF9u;
    // SDWA w1: [7:0]=real_src0=5(v5), [10:8]=src0_sel=2(BYTE_2), [26:24]=dst_sel=6(DWORD)
    uint32_t w1 = 5u | (2u << 8) | (6u << 24);
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP1);
    EXPECT_EQ(inst.opcode, Opcode::V_MOV_B32);
    EXPECT_TRUE(inst.is_sdwa);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.src[0].kind, Operand::Kind::VGPR);
    EXPECT_EQ(inst.src[0].index, 5u);
    EXPECT_EQ(inst.sdwa_src0_sel, 2u);  // BYTE_2
    EXPECT_EQ(inst.sdwa_dst_sel, 6u);   // DWORD
}

// --- MUBUF LDS bit (Phase 4) ---
TEST(ISADecoder, BufferLoadDwordLDS) {
    // buffer_load_dword v0, v0, s[0:3], 0 lds
    // word0: [31:26]=0x38, [24:18]=op=20, [16]=lds=1, [12]=offen=0
    // word1: [31:24]=soffset=0, [20:16]=srsrc=0(=s0), [15:8]=vdata=0, [7:0]=vaddr=0
    uint32_t w0 = (0x38u << 26) | (20u << 18) | (1u << 16);
    uint32_t w1 = 0u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::MUBUF);
    EXPECT_EQ(inst.opcode, Opcode::BUFFER_LOAD_DWORD);
    EXPECT_TRUE(inst.mubuf_lds);
    EXPECT_FALSE(inst.mubuf_offen);
}

// --- FLAT_SCRATCH (SEG=1) (Phase 4) ---
TEST(ISADecoder, FlatScratchLoadDword) {
    // scratch_load_dword v3, v0, off
    // FLAT: [31:26]=0b110111=0x37, [25:18]=OP=0x14(load_dword), [15:14]=SEG=01(scratch)
    uint32_t w0 = (0x37u << 26) | (0x14u << 18) | (1u << 14);
    uint32_t w1 = (3u << 24) | (0x7Fu << 16) | 0u;
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::FLAT_SCRATCH);
    EXPECT_EQ(inst.size_bytes, 8);
}

// --- VOP3P op_sel fields (Phase 3) ---
TEST(ISADecoder, VOP3P_OpSel) {
    // VOP3P: [31:26]=0x33, op with op_sel bits set
    // op_sel at bits[12:11], op_sel_hi at bits[15:13]
    // Set op_sel[0]=1 (bit 11) and op_sel_hi[0]=1 (bit 13)
    uint32_t w0 = (0x33u << 26) | (1u << 13) | (1u << 11) | (2u << 16);
    uint32_t w1 = (257u << 9) | 256u; // src0=v0, src1=v1
    auto b = enc2(w0, w1);
    Instruction inst = decode(b.data());
    EXPECT_EQ(inst.format, InstFormat::VOP3P);
    EXPECT_EQ(inst.size_bytes, 8);
    EXPECT_EQ(inst.op_sel & 1, 1u);       // op_sel bit 0 set
    EXPECT_EQ(inst.op_sel_hi & 1, 1u);    // op_sel_hi bit 0 set
}

TEST(ISADecoder, Gfx950HardwareBytecodeDecodes) {
    // 1. v_prng_b32_e32 v0, v1: 7E00B101
    {
        std::vector<uint8_t> b = {0x01, 0xb1, 0x00, 0x7e};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_PRNG_B32);
        EXPECT_EQ(inst.format, InstFormat::VOP1);
        EXPECT_EQ(inst.dst.index, 0u);
        EXPECT_EQ(inst.src[0].index, 1u);
    }
    // 2. v_mfma_f32_16x16x32_f16: D3D48000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xd4, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_F32_16X16X32_F16);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 3. v_mfma_f32_32x32x16_f16: D3D58000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xd5, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_F32_32X32X16_F16);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 4. v_mfma_f32_16x16x32_bf16: D3B58000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xb5, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_F32_16X16X32_BF16);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 5. v_mfma_f32_32x32x16_bf16: D3B78000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xb7, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_F32_32X32X16_BF16);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 6. v_mfma_i32_16x16x64_i8: D3B68000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xb6, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_I32_16X16X64_I8);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 7. v_mfma_i32_32x32x32_i8: D3B88000 04020900
    {
        std::vector<uint8_t> b = {0x00, 0x80, 0xb8, 0xd3, 0x00, 0x09, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_I32_32X32X32_I8);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 8. v_mfma_ld_scale_b32: D3AC0000 00022310
    {
        std::vector<uint8_t> b = {0x00, 0x00, 0xac, 0xd3, 0x10, 0x23, 0x02, 0x00};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_LD_SCALE_B32);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 9. v_mfma_scale_f32_16x16x128_f8f6f4: D3AD8800 04021100
    {
        std::vector<uint8_t> b = {0x00, 0x88, 0xad, 0xd3, 0x00, 0x11, 0x02, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4);
        EXPECT_EQ(inst.format, InstFormat::MFMA);
    }
    // 10. v_lshl_add_u64: D2080004 04010004
    {
        std::vector<uint8_t> b = {0x04, 0x00, 0x08, 0xd2, 0x04, 0x00, 0x01, 0x04};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_LSHL_ADD_U64);
        EXPECT_EQ(inst.format, InstFormat::VOP3A);
    }
    // 11. v_cvt_scalef32_pk_f32_fp4: D23F0000 00000102
    {
        std::vector<uint8_t> b = {0x00, 0x00, 0x3f, 0xd2, 0x02, 0x01, 0x00, 0x00};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_CVT_SCALEF32_PK_F32_FP4);
        EXPECT_EQ(inst.format, InstFormat::VOP3A);
    }
    // 12. v_cvt_scalef32_pk_f16_fp4: D2500000 00000102
    {
        std::vector<uint8_t> b = {0x00, 0x00, 0x50, 0xd2, 0x02, 0x01, 0x00, 0x00};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_CVT_SCALEF32_PK_F16_FP4);
        EXPECT_EQ(inst.format, InstFormat::VOP3A);
    }
    // 13. v_dot8_i32_i4: D3AA4000 1C020501
    {
        std::vector<uint8_t> b = {0x00, 0x40, 0xaa, 0xd3, 0x01, 0x05, 0x02, 0x1c};
        Instruction inst = decode(b.data(), true);
        EXPECT_EQ(inst.opcode, Opcode::V_DOT8_I32_I4);
        EXPECT_EQ(inst.format, InstFormat::VOP3A);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
