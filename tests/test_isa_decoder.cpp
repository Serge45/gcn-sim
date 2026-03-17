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
TEST(ISADecoder, DsWriteB32) {
    // ds_write_b32 v0, v1 — DS op=11, addr=v0 (in w1[7:0]), data0=v1 (in w1[15:8])
    // word0: [31:26]=0x36, [25:18]=op=11, offset1=0, offset0=0
    // word1: [31:24]=vdst=0, [23:16]=data1=0, [15:8]=data0=v1=1, [7:0]=addr=v0=0
    uint32_t w0 = (0x36u << 26) | (11u << 18);
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
    // ds_read_b32 v2, v0 — DS op=56, vdst=v2, addr=v0
    // word0: [31:26]=0x36, [25:18]=op=56
    // word1: [31:24]=vdst=2, [7:0]=addr=0
    uint32_t w0 = (0x36u << 26) | (56u << 18);
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
    // word0: [31:26]=0x38, [24:18]=op=4, [12]=offen=1
    // word1: [31:24]=vdata=v4, [23:16]=vaddr=v0, [12:8]=srsrc=2(=s8), [7:0]=soffset=0
    uint32_t w0 = (0x38u << 26) | (4u << 18) | (1u << 12);
    uint32_t w1 = (4u << 24) | (0u << 16) | (2u << 8) | 0u;
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
