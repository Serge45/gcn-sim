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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
