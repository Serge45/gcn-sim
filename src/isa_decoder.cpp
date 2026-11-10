#include "gcnsim/isa_decoder.h"
#include <cstring>

namespace gcnsim {

static uint32_t read_u32_le(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// GCN operand encoding: 0-105=SGPR, 106=VCC_LO, 107=VCC_HI,
// 124=M0, 126=EXEC_LO, 127=EXEC_HI, 128=inline 0, 129-192=inline consts,
// 240-247=inline floats, 251=vccz, 252=execz, 253=SCC, 255=literal,
// 256-511=VGPR0-VGPR255
static Operand decode_src(uint32_t enc, const uint8_t* pc_next4, bool is_gfx1201 = false) {
    Operand op;
    if (enc <= 105) {
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 106) {
        op.kind = Operand::Kind::VCC_LO;
    } else if (enc == 107) {
        op.kind = Operand::Kind::VCC_HI;
    } else if (enc >= 108 && enc <= 123) {
        // ttmp0..ttmp15
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 124) {
        if (is_gfx1201) {
            // GFX12: 124 (0x7C) is 'null' (reads as 0)
            op.kind = Operand::Kind::INLINE_CONST;
            op.imm = 0;
        } else {
            op.kind = Operand::Kind::M0;
        }
    } else if (enc == 125) {
        op.kind = Operand::Kind::M0; // GFX12: 125 (0x7D) is M0
    } else if (enc == 126) {
        op.kind = Operand::Kind::EXEC_LO;
    } else if (enc == 127) {
        op.kind = Operand::Kind::EXEC_HI;
    } else if (enc == 128) {
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0;
    } else if (enc >= 129 && enc <= 192) {
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = enc - 128; // 1..64
    } else if (enc >= 193 && enc <= 208) {
        // negative inline consts: -1..-16
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = (uint64_t)(int64_t)(-(int32_t)(enc - 192));
    } else if (enc == 235) {
        op.kind = Operand::Kind::SRC_SHARED_BASE;
    } else if (enc == 236) {
        op.kind = Operand::Kind::SRC_SHARED_LIMIT;
    } else if (enc == 237) {
        op.kind = Operand::Kind::SRC_PRIVATE_BASE;
    } else if (enc == 238) {
        op.kind = Operand::Kind::SRC_PRIVATE_LIMIT;
    } else if (enc == 240) { // 0.5
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0x3F000000;
    } else if (enc == 241) { // -0.5
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0xBF000000;
    } else if (enc == 242) { // 1.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0x3F800000;
    } else if (enc == 243) { // -1.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0xBF800000;
    } else if (enc == 244) { // 2.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0x40000000;
    } else if (enc == 245) { // -2.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0xC0000000;
    } else if (enc == 246) { // 4.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0x40800000;
    } else if (enc == 247) { // -4.0
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0xC0800000;
    } else if (enc == 251) {
        op.kind = Operand::Kind::SCC; // vccz
    } else if (enc == 252) {
        op.kind = Operand::Kind::SCC; // execz
    } else if (enc == 253) {
        op.kind = Operand::Kind::SCC;
    } else if (enc == 255) {
        // literal constant — next 4 bytes (only valid if pc_next4 != nullptr)
        op.kind = Operand::Kind::LITERAL;
        if (pc_next4) {
            op.imm = read_u32_le(pc_next4);
        }
    } else if (enc >= 256 && enc <= 511) {
        op.kind = Operand::Kind::VGPR;
        op.index = enc - 256;
    } else {
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0;
    }
    return op;
}

// Decode SDST (scalar dest, 7-bit or 8-bit)
static Operand decode_sdst(uint32_t enc, bool is_gfx1201 = false) {
    Operand op;
    if (enc <= 105) {
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 106) {
        op.kind = Operand::Kind::VCC_LO;
    } else if (enc == 107) {
        op.kind = Operand::Kind::VCC_HI;
    } else if (enc >= 108 && enc <= 123) {
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 124) {
        op.kind = is_gfx1201 ? Operand::Kind::NONE : Operand::Kind::M0;
    } else if (enc == 125) {
        op.kind = Operand::Kind::M0;
    } else if (enc == 126) {
        op.kind = Operand::Kind::EXEC_LO;
    } else if (enc == 127) {
        op.kind = Operand::Kind::EXEC_HI;
    } else {
        op.kind = Operand::Kind::SGPR;
        op.index = enc & 0x7F;
    }
    return op;
}

// Decode VDST (8-bit VGPR index)
static Operand decode_vdst(uint32_t enc) {
    Operand op;
    op.kind = Operand::Kind::VGPR;
    op.index = enc & 0xFF;
    return op;
}

// ---------------------------------------------------------------------------
// Opcode lookup tables
// ---------------------------------------------------------------------------

static Opcode sop2_opcode(uint32_t op) {
    // GFX9 / GCN3 SOP2 opcode table (AMD ISA spec)
    switch (op) {
        case 0:  return Opcode::S_ADD_U32;
        case 1:  return Opcode::S_SUB_U32;
        case 2:  return Opcode::S_ADD_I32;
        case 3:  return Opcode::S_SUB_U32;   // s_sub_i32 — treat as unsigned sub
        case 4:  return Opcode::S_ADDC_U32;
        case 5:  return Opcode::S_SUBB_U32;
        case 6:  return Opcode::S_MIN_I32;
        case 7:  return Opcode::S_MIN_U32;
        case 8:  return Opcode::S_MAX_I32;
        case 9:  return Opcode::S_MAX_U32;
        case 10: return Opcode::S_CSELECT_B32;
        case 11: return Opcode::S_CSELECT_B64;
        case 12: return Opcode::S_AND_B32;
        case 13: return Opcode::S_AND_B64;
        case 14: return Opcode::S_OR_B32;
        case 15: return Opcode::S_OR_B64;
        case 16: return Opcode::S_XOR_B32;
        case 17: return Opcode::S_XOR_B64;
        case 18: return Opcode::S_ANDN2_B32;
        case 19: return Opcode::S_ANDN2_B64;
        case 20: return Opcode::S_ORN2_B32;
        case 21: return Opcode::S_ORN2_B64;
        case 22: return Opcode::S_NAND_B32;
        case 23: return Opcode::S_NAND_B64;
        case 24: return Opcode::S_NOR_B32;
        case 25: return Opcode::S_NOR_B64;
        case 26: return Opcode::S_XNOR_B32;
        case 27: return Opcode::S_XNOR_B64;
        case 28: return Opcode::S_LSHL_B32;
        case 29: return Opcode::S_LSHL_B64;
        case 30: return Opcode::S_LSHR_B32;
        case 31: return Opcode::S_LSHR_B64;
        case 32: return Opcode::S_ASHR_I32;
        case 33: return Opcode::S_ASHR_I64;
        case 34: return Opcode::S_BFM_B32;
        case 35: return Opcode::S_BFM_B64;
        case 36: return Opcode::S_MUL_I32;
        case 37: return Opcode::S_BFE_U32;
        case 38: return Opcode::S_BFE_I32;
        case 39: return Opcode::S_BFE_U64;
        case 40: return Opcode::S_BFE_I64;
        case 42: return Opcode::S_ABSDIFF_I32;
        case 44: return Opcode::S_MUL_HI_U32;
        case 45: return Opcode::S_MUL_HI_I32;
        case 46: return Opcode::S_LSHL1_ADD_U32;
        case 47: return Opcode::S_LSHL2_ADD_U32;
        case 48: return Opcode::S_LSHL3_ADD_U32;
        case 49: return Opcode::S_LSHL4_ADD_U32;
        case 50: return Opcode::S_PACK_LL_B32_B16;
        case 51: return Opcode::S_PACK_LH_B32_B16;
        case 52: return Opcode::S_PACK_HH_B32_B16;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode sop1_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_MOV_B32;
        case 1:  return Opcode::S_MOV_B64;
        case 2:  return Opcode::S_CMOV_B32;
        case 3:  return Opcode::S_CMOV_B64;
        case 4:  return Opcode::S_NOT_B32;
        case 5:  return Opcode::S_NOT_B64;
        case 8:  return Opcode::S_BREV_B32;
        case 9:  return Opcode::S_BREV_B64;
        case 10: return Opcode::S_BCNT0_I32_B32;
        case 11: return Opcode::S_BCNT0_I32_B64;
        case 12: return Opcode::S_BCNT1_I32_B32;
        case 13: return Opcode::S_BCNT1_I32_B64;
        case 14: return Opcode::S_FF0_I32_B32;
        case 15: return Opcode::S_FF0_I32_B64;
        case 16: return Opcode::S_FF1_I32_B32;
        case 17: return Opcode::S_FF1_I32_B64;
        case 18: return Opcode::S_FLBIT_I32_B32;
        case 19: return Opcode::S_FLBIT_I32_B64;
        case 20: return Opcode::S_FLBIT_I32;
        case 21: return Opcode::S_FLBIT_I32_I64;
        case 22: return Opcode::S_SEXT_I32_I8;
        case 23: return Opcode::S_SEXT_I32_I16;
        case 24: return Opcode::S_BITSET0_B32;
        case 25: return Opcode::S_BITSET0_B64;
        case 26: return Opcode::S_BITSET1_B32;
        case 27: return Opcode::S_BITSET1_B64;
        case 28: return Opcode::S_GETPC_B64;      // 0x1C
        case 29: return Opcode::S_SETPC_B64;      // 0x1D
        case 30: return Opcode::S_SWAPPC_B64;     // 0x1E
        case 31: return Opcode::S_RFE_B64;        // 0x1F
        case 32: return Opcode::S_AND_SAVEEXEC_B64;  // 0x20
        case 33: return Opcode::S_OR_SAVEEXEC_B64;   // 0x21
        case 34: return Opcode::S_XOR_SAVEEXEC_B64;  // 0x22
        case 35: return Opcode::S_ANDN2_SAVEEXEC_B64; // 0x23
        case 36: return Opcode::S_ORN2_SAVEEXEC_B64;  // 0x24
        case 40: return Opcode::S_ABS_I32;
        case 44: return Opcode::S_ABS_I32;
        case 50: return Opcode::S_MOVRELS_B32;
        case 51: return Opcode::S_MOVRELS_B64;
        case 52: return Opcode::S_MOVRELD_B32;
        case 53: return Opcode::S_MOVRELD_B64;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode sopp_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_NOP;
        case 1:  return Opcode::S_ENDPGM;
        case 2:  return Opcode::S_BRANCH;
        case 3:  return Opcode::S_WAKEUP;
        case 4:  return Opcode::S_CBRANCH_SCC0;
        case 5:  return Opcode::S_CBRANCH_SCC1;
        case 6:  return Opcode::S_CBRANCH_VCCZ;
        case 7:  return Opcode::S_CBRANCH_VCCNZ;
        case 8:  return Opcode::S_CBRANCH_EXECZ;
        case 9:  return Opcode::S_CBRANCH_EXECNZ;
        case 10: return Opcode::S_BARRIER;
        case 12: return Opcode::S_WAITCNT;
        case 13: return Opcode::S_SETHALT;
        case 14: return Opcode::S_SLEEP;
        case 15: return Opcode::S_SETPRIO;
        case 16: return Opcode::S_SENDMSG;
        case 17: return Opcode::S_SENDMSGHALT;
        case 18: return Opcode::S_TRAP;
        case 19: return Opcode::S_ICACHE_INV;
        case 20: return Opcode::S_INCPERFLEVEL;
        case 21: return Opcode::S_DECPERFLEVEL;
        case 22: return Opcode::S_TTRACEDATA;
        case 23: return Opcode::S_CBRANCH_CDBGSYS;
        case 24: return Opcode::S_CBRANCH_CDBGUSER;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode sopk_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_MOVK_I32;
        case 1:  return Opcode::S_CMOVK_I32;
        case 2:  return Opcode::S_CMPK_EQ_I32;
        case 3:  return Opcode::S_CMPK_LG_I32;
        case 4:  return Opcode::S_CMPK_GT_I32;
        case 5:  return Opcode::S_CMPK_GE_I32;
        case 6:  return Opcode::S_CMPK_LT_I32;
        case 7:  return Opcode::S_CMPK_LE_I32;
        case 8:  return Opcode::S_CMPK_EQ_U32;
        case 9:  return Opcode::S_CMPK_LG_U32;
        case 10: return Opcode::S_CMPK_GT_U32;
        case 11: return Opcode::S_CMPK_GE_U32;
        case 12: return Opcode::S_CMPK_LT_U32;
        case 13: return Opcode::S_CMPK_LE_U32;
        case 14: return Opcode::S_ADDK_I32;
        case 15: return Opcode::S_MULK_I32;
        case 18: return Opcode::S_GETREG_B32;
        case 19: return Opcode::S_SETREG_B32;
        case 21: return Opcode::S_SETREG_IMM32_B32;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode sopc_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_CMP_EQ_I32;
        case 1:  return Opcode::S_CMP_LG_I32;
        case 2:  return Opcode::S_CMP_GT_I32;
        case 3:  return Opcode::S_CMP_GE_I32;
        case 4:  return Opcode::S_CMP_LT_I32;
        case 5:  return Opcode::S_CMP_LE_I32;
        case 6:  return Opcode::S_CMP_EQ_U32;
        case 7:  return Opcode::S_CMP_LG_U32;
        case 8:  return Opcode::S_CMP_GT_U32;
        case 9:  return Opcode::S_CMP_GE_U32;
        case 10: return Opcode::S_CMP_LT_U32;
        case 11: return Opcode::S_CMP_LE_U32;
        case 12: return Opcode::S_BITCMP0_B32;
        case 13: return Opcode::S_BITCMP1_B32;
        case 14: return Opcode::S_BITCMP0_B64;
        case 15: return Opcode::S_BITCMP1_B64;
        case 16: return Opcode::S_SETVSKIP;
        case 17: return Opcode::S_SET_GPR_IDX_ON;
        case 18: return Opcode::S_CMP_EQ_U64;
        case 19: return Opcode::S_CMP_LG_U64;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode smem_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_LOAD_DWORD;
        case 1:  return Opcode::S_LOAD_DWORDX2;
        case 2:  return Opcode::S_LOAD_DWORDX4;
        case 3:  return Opcode::S_LOAD_DWORDX8;
        case 4:  return Opcode::S_LOAD_DWORDX16;
        case 8:  return Opcode::S_BUFFER_LOAD_DWORD;
        case 9:  return Opcode::S_BUFFER_LOAD_DWORDX2;
        case 10: return Opcode::S_BUFFER_LOAD_DWORDX4;
        case 11: return Opcode::S_BUFFER_LOAD_DWORDX8;
        case 12: return Opcode::S_BUFFER_LOAD_DWORDX16;
        case 16: return Opcode::S_STORE_DWORD;
        case 17: return Opcode::S_STORE_DWORDX2;
        case 18: return Opcode::S_STORE_DWORDX4;
        case 19: return Opcode::S_STORE_DWORDX8;
        case 20: return Opcode::S_STORE_DWORDX16;
        case 24: return Opcode::S_BUFFER_STORE_DWORD;
        case 25: return Opcode::S_BUFFER_STORE_DWORDX2;
        case 26: return Opcode::S_BUFFER_STORE_DWORDX4;
        case 64: case 128: return Opcode::S_ATOMIC_SWAP;
        case 65: case 129: return Opcode::S_ATOMIC_CMPSWAP;
        case 66: case 130: return Opcode::S_ATOMIC_ADD;
        case 67: case 131: return Opcode::S_ATOMIC_SUB;
        case 68: case 132: return Opcode::S_ATOMIC_SMIN;
        case 69: case 133: return Opcode::S_ATOMIC_UMIN;
        case 70: case 134: return Opcode::S_ATOMIC_SMAX;
        case 71: case 135: return Opcode::S_ATOMIC_UMAX;
        case 72: case 136: return Opcode::S_ATOMIC_AND;
        case 73: case 137: return Opcode::S_ATOMIC_OR;
        case 74: case 138: return Opcode::S_ATOMIC_XOR;
        case 75: case 139: return Opcode::S_ATOMIC_INC;
        case 76: case 140: return Opcode::S_ATOMIC_DEC;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode vop2_opcode(uint32_t op) {
    // GCN3/GFX90a VOP2 opcode table (from AMD ISA spec)
    switch (op) {
        case 0:  return Opcode::V_CNDMASK_B32;
        case 1:  return Opcode::V_ADD_F32;
        case 2:  return Opcode::V_SUB_F32;
        case 3:  return Opcode::V_SUBREV_F32;
        case 4:  return Opcode::V_MUL_LEGACY_F32;
        case 5:  return Opcode::V_MUL_F32;
        case 6:  return Opcode::V_MUL_I32_I24;
        case 7:  return Opcode::V_MUL_HI_I32_I24;
        case 8:  return Opcode::V_MUL_U32_U24;
        case 9:  return Opcode::V_MUL_HI_U32_U24;
        case 10: return Opcode::V_MIN_F32;
        case 11: return Opcode::V_MAX_F32;
        case 12: return Opcode::V_MIN_I32;
        case 13: return Opcode::V_MAX_I32;
        case 14: return Opcode::V_MIN_U32;
        case 15: return Opcode::V_MAX_U32;
        case 16: return Opcode::V_LSHRREV_B32;
        case 17: return Opcode::V_ASHRREV_I32;
        case 18: return Opcode::V_LSHLREV_B32;
        case 19: return Opcode::V_AND_B32;
        case 20: return Opcode::V_OR_B32;
        case 21: return Opcode::V_XOR_B32;
        case 22: return Opcode::UNKNOWN;    // v_xnor_b32 (gfx9+)
        case 23: return Opcode::V_MAC_F32;
        case 24: return Opcode::V_MADMK_F32;
        case 25: return Opcode::V_ADD_CO_U32;
        case 26: return Opcode::V_SUB_U32;
        case 27: return Opcode::V_SUBREV_U32;
        case 28: return Opcode::V_ADDC_CO_U32;
        case 32: return Opcode::V_ADD_U32;   // v_add_nc_u32 (gfx9+, no carry)
        case 33: return Opcode::V_SUB_U32;   // v_sub_nc_u32
        case 34: return Opcode::V_SUBREV_U32; // v_subrev_nc_u32
        case 36: return Opcode::V_MAC_F32;
        case 37: return Opcode::V_MADMK_F32;
        case 38: return Opcode::V_MADAK_F32;
        case 52: return Opcode::V_ADD_U32;
        case 53: return Opcode::V_SUB_U32;
        case 54: return Opcode::V_SUBREV_U32;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode vop1_opcode(uint32_t op) {
    switch (op) {
        case 1:   return Opcode::V_MOV_B32;
        case 2:   return Opcode::V_READFIRSTLANE_B32;
        case 3:   return Opcode::V_CVT_I32_F64;
        case 4:   return Opcode::V_CVT_F64_I32;
        case 5:   return Opcode::V_CVT_F32_I32;
        case 6:   return Opcode::V_CVT_F32_U32;
        case 7:   return Opcode::V_CVT_U32_F32;
        case 8:   return Opcode::V_CVT_I32_F32;
        case 9:   return Opcode::V_CVT_F16_F32;
        case 10:  return Opcode::V_CVT_F16_F32;
        case 11:  return Opcode::V_CVT_F32_F16;
        case 12:  return Opcode::V_CVT_F64_F32;
        case 15:  return Opcode::V_CVT_F32_F64;
        case 16:  return Opcode::V_CVT_F64_F32;
        case 17:  return Opcode::V_CVT_F32_UBYTE0;
        case 18:  return Opcode::V_CVT_F32_UBYTE1;
        case 19:  return Opcode::V_CVT_F32_UBYTE2;
        case 20:  return Opcode::V_CVT_F32_UBYTE3;
        case 21:  return Opcode::V_CVT_U32_F64;
        case 22:  return Opcode::V_CVT_F64_U32;
        // GFX90a VOP1 opcodes
        case 27:  return Opcode::V_FRACT_F32;
        case 28:  return Opcode::V_TRUNC_F32;
        case 29:  return Opcode::V_CEIL_F32;
        case 30:  return Opcode::V_RNDNE_F32;
        case 31:  return Opcode::V_FLOOR_F32;
        case 32:  return Opcode::V_EXP_F32;
        case 33:  return Opcode::V_LOG_F32;
        case 34:  return Opcode::V_RCP_F32;
        case 35:  return Opcode::V_RCP_IFLAG_F32;
        case 36:  return Opcode::V_RSQ_F32;
        case 37:  return Opcode::V_RCP_F64;
        case 38:  return Opcode::V_RSQ_F64;
        case 39:  return Opcode::V_SQRT_F32;
        case 40:  return Opcode::V_SQRT_F64;
        case 41:  return Opcode::V_SIN_F32;
        case 42:  return Opcode::V_COS_F32;
        case 43:  return Opcode::V_NOT_B32;
        case 44:  return Opcode::V_BFREV_B32;
        case 45:  return Opcode::V_FFBH_U32;
        case 46:  return Opcode::V_FFBL_B32;
        case 47:  return Opcode::V_FFBH_I32;
        case 48:  return Opcode::V_FREXP_EXP_I32_F64;
        case 49:  return Opcode::V_FREXP_MANT_F64;
        case 50:  return Opcode::V_FRACT_F64;
        case 51:  return Opcode::V_FREXP_EXP_I32_F32;
        case 52:  return Opcode::V_FREXP_MANT_F32;
        case 53:  return Opcode::V_CLREXCP;
        case 56:  return Opcode::V_MOV_B64;
        case 57:  return Opcode::V_CVT_F16_U16;
        case 58:  return Opcode::V_CVT_F16_I16;
        case 59:  return Opcode::V_CVT_U16_F16;
        case 60:  return Opcode::V_CVT_I16_F16;
        case 61:  return Opcode::V_RCP_F16;
        case 62:  return Opcode::V_SQRT_F16;
        case 63:  return Opcode::V_RSQ_F16;
        case 64:  return Opcode::V_LOG_F16;
        case 65:  return Opcode::V_EXP_F16;
        case 66:  return Opcode::V_FREXP_MANT_F16;
        case 67:  return Opcode::V_FREXP_EXP_I16_F16;
        case 68:  return Opcode::V_FLOOR_F16;
        case 69:  return Opcode::V_CEIL_F16;
        case 70:  return Opcode::V_TRUNC_F16;
        case 71:  return Opcode::V_RNDNE_F16;
        case 72:  return Opcode::V_FRACT_F16;
        case 73:  return Opcode::V_SIN_F16;
        case 74:  return Opcode::V_COS_F16;
        case 79:  return Opcode::V_PRNG_B32;
        case 84:  return Opcode::V_ACCVGPR_READ_B32;
        case 86:  return Opcode::V_ACCVGPR_WRITE_B32;
        case 87:  return Opcode::V_PRNG_B32;
        case 88:  return Opcode::V_PRNG_B32;
        case 112: return Opcode::V_SWAP_B32;
        default:  return Opcode::UNKNOWN;
    }
}

static bool is_vop3_vopc_cmpx(uint32_t op) {
    return (op >= 0x30 && op <= 0x3F) || // f16 cmpx
           (op >= 0x50 && op <= 0x5F) || // f32 cmpx
           (op >= 0x70 && op <= 0x7F) || // f64 cmpx
           (op >= 0xB0 && op <= 0xBF) || // i16/u16 cmpx
           (op >= 0xD0 && op <= 0xDF) || // i32/u32 cmpx
           (op >= 0xF0 && op <= 0xFF);   // i64/u64 cmpx
}

static Opcode vop3_vopc_opcode(uint32_t op) {
    if (is_vop3_vopc_cmpx(op)) {
        op -= 0x10;
    }

    // VOP3 f32 comparisons: 0x40-0x4F
    if (op >= 0x40 && op <= 0x4F) {
        switch (op - 0x40) {
            case 0: return Opcode::V_CMP_F_F32;
            case 1: return Opcode::V_CMP_LT_F32;
            case 2: return Opcode::V_CMP_EQ_F32;
            case 3: return Opcode::V_CMP_LE_F32;
            case 4: return Opcode::V_CMP_GT_F32;
            case 5: return Opcode::V_CMP_LG_F32;
            case 6: return Opcode::V_CMP_GE_F32;
            case 7: return Opcode::V_CMP_O_F32;
            case 8: return Opcode::V_CMP_U_F32;
            case 9: return Opcode::V_CMP_NGE_F32;
            case 10: return Opcode::V_CMP_NLG_F32;
            case 11: return Opcode::V_CMP_NGT_F32;
            case 12: return Opcode::V_CMP_NLE_F32;
            case 13: return Opcode::V_CMP_NEQ_F32;
            case 14: return Opcode::V_CMP_NLT_F32;
            case 15: return Opcode::V_CMP_TRU_F32;
        }
    }
    // VOP3 f64 comparisons: 0x60-0x6F
    if (op >= 0x60 && op <= 0x6F) {
        switch (op - 0x60) {
            case 0: return Opcode::V_CMP_F_F64;
            case 1: return Opcode::V_CMP_LT_F64;
            case 2: return Opcode::V_CMP_EQ_F64;
            case 3: return Opcode::V_CMP_LE_F64;
            case 4: return Opcode::V_CMP_GT_F64;
            case 5: return Opcode::V_CMP_LG_F64;
            case 6: return Opcode::V_CMP_GE_F64;
            case 7: return Opcode::V_CMP_O_F64;
            case 8: return Opcode::V_CMP_U_F64;
            case 9: return Opcode::V_CMP_NGE_F64;
            case 10: return Opcode::V_CMP_NLG_F64;
            case 11: return Opcode::V_CMP_NGT_F64;
            case 12: return Opcode::V_CMP_NLE_F64;
            case 13: return Opcode::V_CMP_NEQ_F64;
            case 14: return Opcode::V_CMP_NLT_F64;
            case 15: return Opcode::V_CMP_TRU_F64;
        }
    }
    switch (op) {
        // i32
        case 0xC1: return Opcode::V_CMP_LT_I32;
        case 0xC2: return Opcode::V_CMP_EQ_I32;
        case 0xC3: return Opcode::V_CMP_LE_I32;
        case 0xC4: return Opcode::V_CMP_GT_I32;
        case 0xC5: return Opcode::V_CMP_NE_I32;
        case 0xC6: return Opcode::V_CMP_GE_I32;
        // u32
        case 0xC9: return Opcode::V_CMP_LT_U32;
        case 0xCA: return Opcode::V_CMP_EQ_U32;
        case 0xCB: return Opcode::V_CMP_LE_U32;
        case 0xCC: return Opcode::V_CMP_GT_U32;
        case 0xCD: return Opcode::V_CMP_NE_U32;
        case 0xCE: return Opcode::V_CMP_GE_U32;
        // i64
        case 0xE1: return Opcode::V_CMP_LT_I64;
        case 0xE2: return Opcode::V_CMP_EQ_I64;
        case 0xE3: return Opcode::V_CMP_LE_I64;
        case 0xE4: return Opcode::V_CMP_GT_I64;
        case 0xE5: return Opcode::V_CMP_NE_I64;
        case 0xE6: return Opcode::V_CMP_GE_I64;
        // u64
        case 0xE9: return Opcode::V_CMP_LT_U64;
        case 0xEA: return Opcode::V_CMP_EQ_U64;
        case 0xEB: return Opcode::V_CMP_LE_U64;
        case 0xEC: return Opcode::V_CMP_GT_U64;
        case 0xED: return Opcode::V_CMP_NE_U64;
        case 0xEE: return Opcode::V_CMP_GE_U64;
        default:   return Opcode::UNKNOWN;
    }
}

static bool is_vopc_cmpx(uint32_t op) {
    return is_vop3_vopc_cmpx(op);
}

static Opcode vopc_opcode(uint32_t op) {
    // Both 32-bit VOPC and 64-bit VOP3 use the same GCN/CDNA comparison opcode mapping:
    return vop3_vopc_opcode(op);
}

// VOP3 opcode space (10-bit field):
// 0x00-0xFF    = VOPC re-encoded (8-bit VOPC op used directly)
// 0x140-0x17F  = VOP2 re-encoded (subtract 0x140 for VOP2 op)
// 0x180-0x1BF  = VOP1 re-encoded (subtract 0x180 for VOP1 op)
// 0x200+       = true VOP3 ops
static Opcode vop3_opcode(uint32_t op) {
    switch (op) {
        // True VOP3A/B ops
        case 0x1E5: return Opcode::V_MUL_LO_U32;
        case 0x285: return Opcode::V_MUL_LO_U32;
        case 0x286: return Opcode::V_MUL_HI_U32;
        case 0x287: return Opcode::V_MUL_HI_I32;
        case 0x28F: return Opcode::V_LSHLREV_B64;
        case 0x290: return Opcode::V_LSHRREV_B64;
        case 0x291: return Opcode::V_ASHRREV_I64;
        case 0x292: return Opcode::V_LSHLREV_B64; // alternate
        case 0x280: return Opcode::V_ADD_F64;
        case 0x281: return Opcode::V_MUL_F64;
        case 0x282: return Opcode::V_MIN_F64;
        case 0x283: return Opcode::V_MAX_F64;
        case 0x284: return Opcode::V_LDEXP_F64;
        case 0x1FE: return Opcode::V_ADD_LSHL_U32;
        case 0x1FF: return Opcode::V_ADD3_U32;       // gfx90a encoding (was 0x275 on Vega)
        case 0x200: return Opcode::V_LSHL_OR_B32;   // gfx90a encoding (was 0x276 on Vega)
        case 0x201: return Opcode::V_AND_OR_B32;    // gfx90a encoding (was 0x277 on Vega)
        case 0x202: return Opcode::V_OR3_B32;       // gfx90a encoding (was 0x278 on Vega)
        case 0x208: return Opcode::V_LSHL_ADD_U64;
        case 0x23F: return Opcode::V_CVT_SCALEF32_PK_F32_FP4;
        case 0x250: return Opcode::V_CVT_SCALEF32_PK_F16_FP4;
        case 0x3AA: return Opcode::V_DOT8_I32_I4;
        case 0x204: return Opcode::UNKNOWN;         // v_cubeid_f32 (graphics)
        case 0x1C8: return Opcode::V_BFE_U32;       // gfx90a encoding
        case 0x1C9: return Opcode::V_BFE_I32;       // gfx90a encoding
        case 0x1CA: return Opcode::V_BFI_B32;       // gfx90a encoding
        case 0x1CB: return Opcode::V_FMA_F32;       // gfx90a encoding
        case 0x1CC: return Opcode::V_FMA_F64;       // gfx90a encoding
        case 0x1CE: return Opcode::V_ALIGNBIT_B32;  // gfx90a encoding
        case 0x1CF: return Opcode::V_ALIGNBYTE_B32; // gfx90a encoding
        case 0x1D0: return Opcode::V_MIN3_F32;      // gfx90a encoding
        case 0x1D1: return Opcode::V_MIN3_I32;      // gfx90a encoding
        case 0x1D2: return Opcode::V_MIN3_U32;      // gfx90a encoding
        case 0x1D3: return Opcode::V_MAX3_F32;      // gfx90a encoding
        case 0x1D4: return Opcode::V_MAX3_I32;      // gfx90a encoding
        case 0x1D5: return Opcode::V_MAX3_U32;      // gfx90a encoding
        case 0x1D6: return Opcode::V_MED3_F32;      // gfx90a encoding
        case 0x1D7: return Opcode::V_MED3_I32;      // gfx90a encoding
        case 0x1D8: return Opcode::V_MED3_U32;      // gfx90a encoding
        case 0x1D9: return Opcode::V_SAD_U8;        // gfx90a encoding
        case 0x1DA: return Opcode::V_SAD_HI_U8;     // gfx90a encoding
        case 0x1DB: return Opcode::V_SAD_U16;       // gfx90a encoding
        case 0x1DC: return Opcode::V_SAD_U32;       // gfx90a encoding
        case 0x209: return Opcode::V_BFE_U32;       // Vega encoding
        case 0x20A: return Opcode::V_BFE_I32;       // Vega encoding
        case 0x20B: return Opcode::V_BFI_B32;       // Vega encoding
        case 0x20C: return Opcode::V_FMA_F32;       // Vega encoding
        case 0x20D: return Opcode::V_FMA_F64;       // Vega encoding
        case 0x20E: return Opcode::UNKNOWN;         // v_lerp_u8
        case 0x20F: return Opcode::V_ALIGNBIT_B32;
        case 0x210: return Opcode::V_ALIGNBYTE_B32;
        case 0x211: return Opcode::V_MIN3_F32;
        case 0x212: return Opcode::V_MIN3_I32;
        case 0x213: return Opcode::V_MIN3_U32;
        case 0x214: return Opcode::V_MAX3_F32;
        case 0x215: return Opcode::V_MAX3_I32;
        case 0x216: return Opcode::V_MAX3_U32;
        case 0x217: return Opcode::V_MED3_F32;
        case 0x218: return Opcode::V_MED3_I32;
        case 0x219: return Opcode::V_MED3_U32;
        case 0x21A: return Opcode::V_SAD_U8;
        case 0x21B: return Opcode::V_SAD_HI_U8;
        case 0x21C: return Opcode::V_SAD_U16;
        case 0x21D: return Opcode::V_SAD_U32;
        case 0x21E: return Opcode::V_CVT_PKACCUM_U8_F32;
        case 0x21F: return Opcode::V_CVT_PKNORM_I16_F32;
        case 0x220: return Opcode::V_CVT_PKNORM_U16_F32;
        case 0x221: return Opcode::V_CVT_PKRTZ_F16_F32;
        case 0x222: return Opcode::V_CVT_PK_U16_U32;
        case 0x223: return Opcode::V_CVT_PK_I16_I32;
        case 0x294: return Opcode::V_CVT_PKNORM_I16_F32; // gfx90a encoding
        case 0x295: return Opcode::V_CVT_PKNORM_U16_F32; // gfx90a encoding
        case 0x296: return Opcode::V_CVT_PKRTZ_F16_F32;  // gfx90a encoding
        case 0x297: return Opcode::V_CVT_PK_U16_U32;     // gfx90a encoding
        case 0x298: return Opcode::V_CVT_PK_I16_I32;     // gfx90a encoding
        case 0x2A0: return Opcode::V_PACK_B32_F16;
        case 0x1E8: return Opcode::V_MAD_U64_U32;
        case 0x1E9: return Opcode::V_MAD_I64_I32;
        case 0x228: return Opcode::V_MAD_U64_U32;
        case 0x229: return Opcode::V_MAD_I64_I32;
        case 0x1FD: return Opcode::V_LSHL_ADD_U32;
        case 0x274: return Opcode::V_ADD_I32;    // Vega encoding
        case 0x29C: return Opcode::V_ADD_I32;    // gfx90a encoding
        case 0x275: return Opcode::V_ADD3_U32;
        case 0x276: return Opcode::V_LSHL_OR_B32;
        case 0x277: return Opcode::V_AND_OR_B32;
        case 0x278: return Opcode::V_OR3_B32;
        case 0x2F5: return Opcode::V_MAD_F32;
        case 0x25B: return Opcode::V_PERMLANE16_B32;
        case 0x25C: return Opcode::V_PERMLANEX16_B32;
        case 0x1ED:
        case 0x2EC: return Opcode::V_PERM_B32;
        case 0x2EE: return Opcode::V_XAD_U32;
        // VOP3B carry-chain
        // Alternate e64 encodings used by gfx90a compiler for carry-chain ops
        case 0x119: return Opcode::V_ADD_CO_U32;
        case 0x11A: return Opcode::V_SUB_U32;
        case 0x11B: return Opcode::V_SUBREV_U32;
        case 0x11C: return Opcode::V_ADDC_CO_U32;
        case 0x11D: return Opcode::V_SUB_CO_U32;
        case 0x11E: return Opcode::V_SUBREV_CO_U32;
        // Legacy / alternate carry-chain op range
        case 0x140: return Opcode::V_ADD_CO_U32;
        case 0x141: return Opcode::V_SUB_U32;
        case 0x142: return Opcode::V_SUBREV_U32;
        case 0x143: return Opcode::V_ADDC_CO_U32;
        case 0x144: return Opcode::V_SUB_CO_U32;
        case 0x145: return Opcode::V_SUBREV_CO_U32;
        // GFX90a AGPR ops (VOP3 op space, op >= 960 but not MFMA)
        case 0x3D8: return Opcode::V_ACCVGPR_READ_B32;
        case 0x3D9: return Opcode::V_ACCVGPR_WRITE_B32;
        // CDNA packed FP32 / B32 ops (VOP3 op space)
        case 0x3B0: return Opcode::V_PK_FMA_F32;
        case 0x3B1: return Opcode::V_PK_MUL_F32;
        case 0x3B2: return Opcode::V_PK_ADD_F32;
        case 0x3B3: return Opcode::V_PK_MOV_B32;
        default: {
            // VOP3->VOP2: GFX9/GFX90a encodes VOP2 op N as VOP3 op 0x100+N.
            // Range 0x100-0x13F covers VOP2 ops 0-63 (primary GFX9 encoding).
            // Range 0x140-0x17F covers VOP2 ops 0-63 (alternate/older encoding).
            if (op >= 0x100 && op <= 0x13F) {
                return vop2_opcode(op - 0x100);
            }
            if (op >= 0x140 && op <= 0x17F) {
                return vop2_opcode(op - 0x140);
            }
            // VOP3->VOP1
            if (op >= 0x180 && op <= 0x1BF) {
                return vop1_opcode(op - 0x180);
            }
            // VOP3->VOPC: 8-bit VOPC op used directly as the VOP3 10-bit op field
            // (e.g. V_CMP_GT_I32 = vopc op 0xC4 → VOP3 op field 0xC4)
            {
                Opcode candidate = vopc_opcode(op);
                if (candidate != Opcode::UNKNOWN)
                    return candidate;
            }
            return Opcode::UNKNOWN;
        }
    }
}

static Opcode vop3p_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::V_PK_MAD_I16;
        case 1:  return Opcode::V_PK_MUL_LO_U16;
        case 2:  return Opcode::V_PK_ADD_I16;
        case 3:  return Opcode::V_PK_SUB_I16;
        case 4:  return Opcode::V_PK_LSHLREV_B16;
        case 5:  return Opcode::V_PK_LSHRREV_B16;
        case 6:  return Opcode::V_PK_ASHRREV_I16;
        case 7:  return Opcode::V_PK_MAX_I16;
        case 8:  return Opcode::V_PK_MIN_I16;
        case 10: return Opcode::V_PK_MAD_U16;
        case 11: return Opcode::V_PK_ADD_U16;
        case 12: return Opcode::V_PK_SUB_U16;
        case 13: return Opcode::V_PK_MAX_U16;
        case 14: return Opcode::V_PK_MIN_U16;
        case 16: return Opcode::V_PK_FMA_F16;
        case 17: return Opcode::V_PK_ADD_F16;
        case 18: return Opcode::V_PK_MUL_F16;
        case 19: return Opcode::V_PK_MIN_F16;
        case 20: return Opcode::V_PK_MAX_F16;
        case 32: return Opcode::V_FMA_MIX_F32;
        case 33: return Opcode::V_FMA_MIXLO_F16;
        case 34: return Opcode::V_FMA_MIXHI_F16;
        // dot products
        case 24: return Opcode::V_DOT2_F32_F16;
        case 25: return Opcode::V_DOT2_I32_I16;
        case 26: return Opcode::V_DOT2_U32_U16;
        case 27: return Opcode::V_DOT4_I32_I8;
        case 28: return Opcode::V_DOT4_U32_U8;
        case 29: return Opcode::V_DOT8_I32_I4;
        case 30: return Opcode::V_DOT8_U32_U4;
        // MI-200 CDNA2 packed F32 ops
        case 35: return Opcode::V_PK_FMA_F32;
        case 36: return Opcode::V_PK_MUL_F32;
        case 37: return Opcode::V_PK_ADD_F32;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode flat_opcode(uint32_t op, bool is_global, bool is_scratch) {
    (void)is_scratch;
    if (is_global) {
        switch (op) {
            case 0x10: return Opcode::GLOBAL_LOAD_UBYTE;
            case 0x11: return Opcode::GLOBAL_LOAD_SBYTE;
            case 0x12: return Opcode::GLOBAL_LOAD_USHORT;
            case 0x13: return Opcode::GLOBAL_LOAD_SSHORT;
            case 0x14: return Opcode::GLOBAL_LOAD_DWORD;
            case 0x15: return Opcode::GLOBAL_LOAD_DWORDX2;
            case 0x16: return Opcode::GLOBAL_LOAD_DWORDX3;
            case 0x17: return Opcode::GLOBAL_LOAD_DWORDX4;
            case 0x18: return Opcode::GLOBAL_STORE_BYTE;
            case 0x19: return Opcode::GLOBAL_STORE_SHORT;
            case 0x1C: return Opcode::GLOBAL_STORE_DWORD;
            case 0x1D: return Opcode::GLOBAL_STORE_DWORDX2;
            case 0x1E: return Opcode::GLOBAL_STORE_DWORDX3;
            case 0x1F: return Opcode::GLOBAL_STORE_DWORDX4;
            case 0x40: return Opcode::GLOBAL_ATOMIC_SWAP;
            case 0x41: return Opcode::GLOBAL_ATOMIC_CMPSWAP;
            case 0x42: return Opcode::GLOBAL_ATOMIC_ADD;
            case 0x43: return Opcode::GLOBAL_ATOMIC_SUB;
            case 0x44: return Opcode::GLOBAL_ATOMIC_SMIN;
            case 0x45: return Opcode::GLOBAL_ATOMIC_UMIN;
            case 0x46: return Opcode::GLOBAL_ATOMIC_SMAX;
            case 0x47: return Opcode::GLOBAL_ATOMIC_UMAX;
            case 0x48: return Opcode::GLOBAL_ATOMIC_AND;
            case 0x49: return Opcode::GLOBAL_ATOMIC_OR;
            case 0x4A: return Opcode::GLOBAL_ATOMIC_XOR;
            case 0x4B: return Opcode::GLOBAL_ATOMIC_INC;
            case 0x4C: return Opcode::GLOBAL_ATOMIC_DEC;
            case 0x4D: return Opcode::GLOBAL_ATOMIC_ADD_F32;
            case 0x4F: return Opcode::GLOBAL_ATOMIC_ADD_F64;
            case 0x62: return Opcode::GLOBAL_ATOMIC_ADD_X2;
            // Legacy
            case 0x0C: return Opcode::GLOBAL_LOAD_DWORD;
            case 0x0D: return Opcode::GLOBAL_LOAD_DWORDX2;
            case 0x0E: return Opcode::GLOBAL_LOAD_DWORDX4;
            case 0x1A: return Opcode::GLOBAL_STORE_DWORD;
            case 0x1B: return Opcode::GLOBAL_STORE_DWORDX2;
            default:   return Opcode::UNKNOWN;
        }
    }
    // flat (SEG=0) and scratch (SEG=1) use the same FLAT opcode enum values
    // The executor distinguishes them by inst.format (FLAT_GLOBAL vs FLAT_SCRATCH)
    switch (op) {
        case 0x0C: return Opcode::FLAT_LOAD_DWORD;
        case 0x0D: return Opcode::FLAT_LOAD_DWORDX2;
        case 0x0E: return Opcode::FLAT_LOAD_DWORDX4;
        case 0x10: return Opcode::FLAT_LOAD_UBYTE;
        case 0x11: return Opcode::FLAT_LOAD_SBYTE;
        case 0x12: return Opcode::FLAT_LOAD_USHORT;
        case 0x13: return Opcode::FLAT_LOAD_SSHORT;
        case 0x14: return Opcode::FLAT_LOAD_DWORD;
        case 0x15: return Opcode::FLAT_LOAD_DWORDX2;
        case 0x16: return Opcode::FLAT_LOAD_DWORDX3;
        case 0x17: return Opcode::FLAT_LOAD_DWORDX4;
        case 0x18: return Opcode::FLAT_STORE_BYTE;
        case 0x19: return Opcode::FLAT_STORE_SHORT;
        case 0x1A: return Opcode::FLAT_STORE_DWORD;
        case 0x1B: return Opcode::FLAT_STORE_DWORDX2;
        case 0x1C: return Opcode::FLAT_STORE_DWORD;
        case 0x1D: return Opcode::FLAT_STORE_DWORDX2;
        case 0x1E: return Opcode::FLAT_STORE_DWORDX3;
        case 0x1F: return Opcode::FLAT_STORE_DWORDX4;
        default:   return Opcode::UNKNOWN;
    }
}

static Opcode mubuf_opcode(uint32_t op) {
    // Op values from AMD GFX9/Vega ISA Reference.
    // Ops 0-15 are BUFFER_LOAD/STORE_FORMAT (not implemented).
    // Raw load/store start at op 16.
    switch (op) {
        case 16: return Opcode::BUFFER_LOAD_UBYTE;
        case 17: return Opcode::BUFFER_LOAD_SBYTE;
        case 18: return Opcode::BUFFER_LOAD_USHORT;
        case 19: return Opcode::BUFFER_LOAD_SSHORT;
        case 20: return Opcode::BUFFER_LOAD_DWORD;
        case 21: return Opcode::BUFFER_LOAD_DWORDX2;
        case 22: return Opcode::BUFFER_LOAD_DWORDX3;
        case 23: return Opcode::BUFFER_LOAD_DWORDX4;
        case 24: return Opcode::BUFFER_STORE_BYTE;
        case 25: return Opcode::BUFFER_STORE_BYTE;
        case 26: return Opcode::BUFFER_STORE_SHORT;
        case 27: return Opcode::BUFFER_STORE_SHORT_D16_HI;
        case 28: return Opcode::BUFFER_STORE_DWORD;
        case 29: return Opcode::BUFFER_STORE_DWORDX2;
        case 30: return Opcode::BUFFER_STORE_DWORDX3;
        case 31: return Opcode::BUFFER_STORE_DWORDX4;
        case 36: return Opcode::BUFFER_LOAD_SHORT_D16;
        case 37: return Opcode::BUFFER_LOAD_SHORT_D16_HI;
        case 48: return Opcode::BUFFER_ATOMIC_SWAP;
        case 49: return Opcode::BUFFER_ATOMIC_CMPSWAP;
        case 50: return Opcode::BUFFER_ATOMIC_ADD;
        case 51: return Opcode::BUFFER_ATOMIC_SUB;
        case 52: return Opcode::BUFFER_ATOMIC_SMIN;
        case 53: return Opcode::BUFFER_ATOMIC_UMIN;
        case 54: return Opcode::BUFFER_ATOMIC_SMAX;
        case 55: return Opcode::BUFFER_ATOMIC_UMAX;
        case 56: return Opcode::BUFFER_ATOMIC_AND;
        case 57: return Opcode::BUFFER_ATOMIC_OR;
        case 58: return Opcode::BUFFER_ATOMIC_XOR;
        case 59: return Opcode::BUFFER_ATOMIC_INC;
        case 60: return Opcode::BUFFER_ATOMIC_DEC;
        case 63: return Opcode::BUFFER_ATOMIC_ADD_F32;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode mtbuf_opcode(uint32_t op) {
    switch (op) {
        case 0: return Opcode::TBUFFER_LOAD_FORMAT_X;
        case 1: return Opcode::TBUFFER_LOAD_FORMAT_XY;
        case 2: return Opcode::TBUFFER_LOAD_FORMAT_XYZ;
        case 3: return Opcode::TBUFFER_LOAD_FORMAT_XYZW;
        case 4: return Opcode::TBUFFER_STORE_FORMAT_X;
        case 5: return Opcode::TBUFFER_STORE_FORMAT_XY;
        case 6: return Opcode::TBUFFER_STORE_FORMAT_XYZ;
        case 7: return Opcode::TBUFFER_STORE_FORMAT_XYZW;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode ds_opcode(uint32_t op, bool is_gfx1201 = false) {
    // Op values from AMD GFX9/Vega ISA Reference, confirmed against llvm-objdump.
    // Op field is bits[24:17] of DS word0 (8-bit, extracted as (w0>>17)&0xFF).
    switch (op) {
        case 0:   return Opcode::DS_ADD_U32;
        case 1:   return Opcode::DS_SUB_U32;
        case 2:   return Opcode::DS_RSUB_U32;
        case 3:   return Opcode::DS_INC_U32;
        case 4:   return Opcode::DS_DEC_U32;
        case 5:   return Opcode::DS_MIN_I32;
        case 6:   return Opcode::DS_MAX_I32;
        case 7:   return Opcode::DS_AND_B32;
        case 8:   return Opcode::DS_OR_B32;
        case 9:   return Opcode::DS_XOR_B32;
        case 10:  return Opcode::DS_MSKOR_B32;
        case 13:  return Opcode::DS_WRITE_B32;
        case 14:  return Opcode::DS_WRITE2_B32;
        case 15:  return Opcode::DS_WRITE2ST64_B32;
        case 16:  return Opcode::DS_CMPST_B32;
        case 17:  return Opcode::DS_CMPST_F32;
        case 18:  return Opcode::DS_MIN_F32;
        case 19:  return Opcode::DS_MAX_F32;
        case 24:  return Opcode::DS_WRITE_ADDTID_B32;
        case 30:  return Opcode::DS_WRITE_B8;
        case 31:  return Opcode::DS_WRITE_B16;
        case 32:  return Opcode::DS_ADD_RTN_U32;
        case 33:  return Opcode::DS_SUB_RTN_U32;
        case 34:  return Opcode::DS_RSUB_RTN_U32;
        case 35:  return Opcode::DS_INC_RTN_U32;
        case 36:  return Opcode::DS_DEC_RTN_U32;
        case 37:  return Opcode::DS_MIN_RTN_I32;
        case 38:  return Opcode::DS_MAX_RTN_I32;
        case 39:  return Opcode::DS_AND_RTN_B32;
        case 40:  return Opcode::DS_OR_RTN_B32;
        case 41:  return Opcode::DS_XOR_RTN_B32;
        case 45:  return Opcode::DS_WRXCHG_RTN_B32;
        case 46:  return Opcode::DS_WRXCHG2_RTN_B32;
        case 48:  return Opcode::DS_CMPST_RTN_B32;
        case 49:  return Opcode::DS_CMPST_RTN_F32;
        case 50:  return Opcode::DS_MIN_RTN_F32;
        case 51:  return Opcode::DS_MAX_RTN_F32;
        case 52:  return Opcode::DS_WRAP_RTN_B32;
        case 53:  return Opcode::DS_ADD_F32;
        case 54:  return Opcode::DS_READ_B32;
        case 55:  return Opcode::DS_READ2_B32;
        case 56:  return Opcode::DS_READ2ST64_B32;
        case 57:  return Opcode::DS_READ_I8;
        case 58:  return Opcode::DS_READ_U8;
        case 59:  return Opcode::DS_READ_I16;
        case 60:  return Opcode::DS_READ_U16;
        case 61:  return Opcode::DS_SWIZZLE_B32;
        case 62:  return Opcode::DS_PERMUTE_B32;
        case 63:  return Opcode::DS_BPERMUTE_B32;
        case 64:  return Opcode::DS_ADD_U64;
        case 65:  return Opcode::DS_SUB_U64;
        case 66:  return Opcode::DS_RSUB_U64;
        case 67:  return Opcode::DS_INC_U64;
        case 68:  return Opcode::DS_DEC_U64;
        case 69:  return Opcode::DS_MIN_I64;
        case 70:  return Opcode::DS_MAX_I64;
        case 71:  return Opcode::DS_AND_B64;
        case 72:  return Opcode::DS_OR_B64;
        case 73:  return Opcode::DS_XOR_B64;
        case 77:  return Opcode::DS_WRITE_B64;
        case 78:  return Opcode::DS_WRITE2_B64;
        case 79:  return Opcode::DS_WRITE2ST64_B64;
        case 80:  return Opcode::DS_CMPST_B64;
        case 81:  return Opcode::DS_CMPST_F64;
        case 82:  return Opcode::DS_MIN_F64;
        case 83:  return Opcode::DS_MAX_F64;
        case 85:  return Opcode::DS_WRITE_B16_D16_HI;
        case 90:  return Opcode::DS_READ_U16_D16;
        case 91:  return Opcode::DS_READ_U16_D16_HI;
        case 96:  return Opcode::DS_ADD_RTN_U64;
        case 97:  return Opcode::DS_SUB_RTN_U64;
        case 98:  return Opcode::DS_RSUB_RTN_U64;
        case 99:  return Opcode::DS_INC_RTN_U64;
        case 100: return Opcode::DS_DEC_RTN_U64;
        case 101: return Opcode::DS_MIN_RTN_I64;
        case 102: return Opcode::DS_MAX_RTN_I64;
        case 103: return Opcode::DS_AND_RTN_B64;
        case 104: return Opcode::DS_OR_RTN_B64;
        case 105: return Opcode::DS_XOR_RTN_B64;
        case 109: return Opcode::DS_WRXCHG_RTN_B64;
        case 118: return Opcode::DS_READ_B64;
        case 119: return Opcode::DS_READ2_B64;
        case 120: return Opcode::DS_READ2ST64_B64;
        case 126: return Opcode::DS_CONDXCHG32_RTN_B64;
        case 161: return Opcode::DS_WRITE_B16_D16_HI; // 0xA1 (GFX12)
        case 179: return Opcode::DS_BPERMUTE_B32;     // 0xB3 (GFX12)
        case 221: return Opcode::DS_WRITE_B96;
        case 222: return is_gfx1201 ? Opcode::DS_WRITE_B96 : Opcode::DS_WRITE_B128;
        case 223: return Opcode::DS_WRITE_B128;
        case 253: return Opcode::DS_READ_B96;
        case 254: return is_gfx1201 ? Opcode::DS_READ_B96 : Opcode::DS_READ_B128;
        case 255: return Opcode::DS_READ_B128;
        default:  return Opcode::UNKNOWN;
    }
}

static Opcode mfma_opcode(uint32_t op, bool is_gfx950 = false) {
    switch (op) {
        // CDNA4 (GFX950) newly assigned hardware opcodes
        case 940: return Opcode::V_MFMA_LD_SCALE_B32;
        case 941: return Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4;
        case 942: return Opcode::V_MFMA_SCALE_F32_32X32X64_F8F6F4;
        case 949: return Opcode::V_MFMA_F32_16X16X32_BF16;
        case 950: return Opcode::V_MFMA_I32_16X16X64_I8;
        case 951: return Opcode::V_MFMA_F32_32X32X16_BF16;
        case 952: return Opcode::V_MFMA_I32_32X32X32_I8;

        // Direct 10-bit hardware opcodes (GFX90A & GFX942)
        case 958: return Opcode::V_MFMA_F32_16X16X8_XF32;
        case 959: return Opcode::V_MFMA_F32_32X32X4_XF32;
        case 960: return Opcode::V_MFMA_F32_32X32X1F32;
        case 961: return Opcode::V_MFMA_F32_16X16X1F32;
        case 962: return Opcode::V_MFMA_F32_4X4X1F32;
        case 964: return Opcode::V_MFMA_F32_32X32X2F32;
        case 965: return Opcode::V_MFMA_F32_16X16X4F32;
        case 968: return Opcode::V_MFMA_F32_32X32X4F16;
        case 969: return Opcode::V_MFMA_F32_16X16X4F16;
        case 970: return Opcode::V_MFMA_F32_4X4X4F16;
        case 972: return Opcode::V_MFMA_F32_32X32X8F16;
        case 973: return Opcode::V_MFMA_F32_16X16X16F16;
        case 976: return Opcode::V_MFMA_I32_32X32X4I8;
        case 977: return Opcode::V_MFMA_I32_16X16X4I8;
        case 978: return Opcode::V_MFMA_I32_4X4X4I8;
        case 980: return is_gfx950 ? Opcode::V_MFMA_F32_16X16X32_F16 : Opcode::V_MFMA_I32_32X32X8I8;
        case 981: return is_gfx950 ? Opcode::V_MFMA_F32_32X32X16_F16 : Opcode::V_MFMA_I32_16X16X16I8;
        case 982: return Opcode::V_MFMA_I32_32X32X16I8;
        case 983: return Opcode::V_MFMA_I32_16X16X32I8;
        case 989: return Opcode::V_MFMA_F32_32X32X4BF16;
        case 990: return Opcode::V_MFMA_F32_16X16X4BF16_1K;
        case 991: return Opcode::V_MFMA_F32_4X4X4BF16_1K;
        case 992: return Opcode::V_MFMA_F32_32X32X8BF16_1K;
        case 993: return Opcode::V_MFMA_F32_16X16X16BF16_1K;
        case 995: return Opcode::V_MFMA_F32_32X32X4BF16_1K;
        case 996: return Opcode::V_MFMA_F32_16X16X4BF16_1K;
        case 997: return Opcode::V_MFMA_F32_4X4X4BF16_1K;
        case 998: return Opcode::V_MFMA_F32_32X32X8BF16_1K;
        case 999: return Opcode::V_MFMA_F32_16X16X16BF16_1K;
        case 1000: return Opcode::V_MFMA_F32_32X32X2BF16;
        case 1001: return Opcode::V_MFMA_F32_16X16X2BF16;
        case 1003: return Opcode::V_MFMA_F32_4X4X2BF16;
        case 1004: return Opcode::V_MFMA_F32_32X32X4BF16;
        case 1005: return Opcode::V_MFMA_F32_16X16X8BF16;
        case 1006: return Opcode::V_MFMA_F64_16X16X4F64;
        case 1007: return Opcode::V_MFMA_F64_4X4X4F64;
        case 1008: return Opcode::V_MFMA_F32_16X16X32_BF8_BF8;
        case 1009: return Opcode::V_MFMA_F32_16X16X32_BF8_FP8;
        case 1010: return Opcode::V_MFMA_F32_16X16X32_FP8_BF8;
        case 1011: return Opcode::V_MFMA_F32_16X16X32_FP8_FP8;
        case 1012: return Opcode::V_MFMA_F32_32X32X16_BF8_BF8;
        case 1013: return Opcode::V_MFMA_F32_32X32X16_BF8_FP8;
        case 1014: return Opcode::V_MFMA_F32_32X32X16_FP8_BF8;
        case 1015: return Opcode::V_MFMA_F32_32X32X16_FP8_FP8;
        case 1016: return Opcode::V_MFMA_F32_16X16X32_F16;
        case 1017: return Opcode::V_MFMA_F32_32X32X16_F16;
        case 1018: return Opcode::V_MFMA_F32_16X16X32_BF16;
        case 1019: return Opcode::V_MFMA_F32_32X32X16_BF16;
        case 1020: return Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4;
        case 1021: return Opcode::V_MFMA_SCALE_F32_32X32X64_F8F6F4;

        // Legacy / relative (op - 960) mapping compatibility
        case 0:  return Opcode::V_MFMA_F32_32X32X1F32;
        case 1:  return Opcode::V_MFMA_F32_16X16X1F32;
        case 2:  return Opcode::V_MFMA_F32_4X4X1F32;
        case 4:  return Opcode::V_MFMA_F32_32X32X2F32;
        case 5:  return Opcode::V_MFMA_F32_16X16X4F32;
        case 8:  return Opcode::V_MFMA_F32_32X32X4F16;
        case 9:  return Opcode::V_MFMA_F32_16X16X4F16;
        case 10: return Opcode::V_MFMA_F32_4X4X4F16;
        case 12: return Opcode::V_MFMA_F32_32X32X8F16;
        case 13: return Opcode::V_MFMA_F32_16X16X16F16;
        case 16: return Opcode::V_MFMA_F32_32X32X4F16; // kept for legacy tests
        case 17: return Opcode::V_MFMA_F32_16X16X4F16;
        case 18: return Opcode::V_MFMA_F32_4X4X4F16;
        case 20: return Opcode::V_MFMA_F32_32X32X8F16;
        case 21: return Opcode::V_MFMA_F32_16X16X16F16;
        case 22: return Opcode::V_MFMA_I32_32X32X16I8;
        case 23: return Opcode::V_MFMA_I32_16X16X32I8;
        case 32: return Opcode::V_MFMA_I32_32X32X4I8;
        case 33: return Opcode::V_MFMA_I32_16X16X4I8;
        case 34: return Opcode::V_MFMA_I32_4X4X4I8;
        case 36: return Opcode::V_MFMA_I32_32X32X8I8;
        case 37: return Opcode::V_MFMA_I32_16X16X16I8;
        case 40: return Opcode::V_MFMA_F32_32X32X2BF16;
        case 41: return Opcode::V_MFMA_F32_16X16X2BF16;
        case 43: return Opcode::V_MFMA_F32_4X4X2BF16;
        case 44: return Opcode::V_MFMA_F32_32X32X4BF16;
        case 45: return Opcode::V_MFMA_F32_16X16X8BF16;
        case 46: return Opcode::V_MFMA_F64_16X16X4F64;
        case 47: return Opcode::V_MFMA_F64_4X4X4F64;
        case 48: return Opcode::V_MFMA_F32_16X16X32_BF8_BF8;
        case 49: return Opcode::V_MFMA_F32_16X16X32_BF8_FP8;
        case 50: return Opcode::V_MFMA_F32_16X16X32_FP8_BF8;
        case 51: return Opcode::V_MFMA_F32_16X16X32_FP8_FP8;
        case 52: return Opcode::V_MFMA_F32_32X32X16_BF8_BF8;
        case 53: return Opcode::V_MFMA_F32_32X32X16_BF8_FP8;
        case 54: return Opcode::V_MFMA_F32_32X32X16_FP8_BF8;
        case 55: return Opcode::V_MFMA_F32_32X32X16_FP8_FP8;
        case 56: return Opcode::V_MFMA_F32_16X16X32_F16;
        case 57: return Opcode::V_MFMA_F32_32X32X16_F16;
        case 58: return Opcode::V_MFMA_F32_16X16X32_BF16;
        case 59: return Opcode::V_MFMA_F32_32X32X16_BF16;
        case 60: return Opcode::V_MFMA_F32_32X32X4BF16_1K;
        case 61: return Opcode::V_MFMA_F32_16X16X4BF16_1K;
        case 62: return Opcode::V_MFMA_F32_4X4X4BF16_1K;
        case 63: return Opcode::V_MFMA_F32_32X32X8BF16_1K;
        case 64: return Opcode::V_MFMA_F32_16X16X16BF16_1K;
        case 65: return Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4;
        case 66: return Opcode::V_MFMA_SCALE_F32_32X32X64_F8F6F4;
        case 96: return Opcode::V_MFMA_F64_16X16X4F64;
        case 97: return Opcode::V_MFMA_F64_4X4X4F64;
        default: return Opcode::UNKNOWN;
    }
}

// Parse s_waitcnt immediate
static void parse_waitcnt(uint32_t simm16, uint8_t& vmcnt, uint8_t& lgkmcnt) {
    uint8_t vmcnt_lo = simm16 & 0xF;
    uint8_t vmcnt_hi = (simm16 >> 14) & 0x3;
    vmcnt   = vmcnt_lo | (vmcnt_hi << 4);
    lgkmcnt = (simm16 >> 8) & 0x3F;
}

// ---------------------------------------------------------------------------
// GFX12 (RDNA4 / gfx1201) instruction decoder
// ---------------------------------------------------------------------------

static Opcode decode_gfx12_vop1_op(uint32_t op) {
    switch (op) {
        case 0x00: return Opcode::V_NOP;
        case 0x01: return Opcode::V_MOV_B32;
        case 0x02: return Opcode::V_READFIRSTLANE_B32;
        case 0x03: return Opcode::V_CVT_I32_F64;
        case 0x04: return Opcode::V_CVT_F64_I32;
        case 0x05: return Opcode::V_CVT_F32_I32;
        case 0x06: return Opcode::V_CVT_F32_U32;
        case 0x07: return Opcode::V_CVT_U32_F32;
        case 0x08: return Opcode::V_CVT_I32_F32;
        case 0x0A: return Opcode::V_CVT_F16_F32;
        case 0x0B: return Opcode::V_CVT_F32_F16;
        case 0x0C: return Opcode::V_CVT_RPI_I32_F32;
        case 0x0D: return Opcode::V_CVT_FLR_I32_F32;
        case 0x0E: return Opcode::V_CVT_OFF_F32_I4;
        case 0x0F: return Opcode::V_CVT_F32_F64;
        case 0x10: return Opcode::V_CVT_F64_F32;
        case 0x11: return Opcode::V_CVT_F32_UBYTE0;
        case 0x12: return Opcode::V_CVT_F32_UBYTE1;
        case 0x13: return Opcode::V_CVT_F32_UBYTE2;
        case 0x14: return Opcode::V_CVT_F32_UBYTE3;
        case 0x15: return Opcode::V_CVT_U32_F64;
        case 0x16: return Opcode::V_CVT_F64_U32;
        case 0x17: return Opcode::V_TRUNC_F64;
        case 0x18: return Opcode::V_CEIL_F64;
        case 0x19: return Opcode::V_RNDNE_F64;
        case 0x1A: return Opcode::V_FLOOR_F64;
        case 0x20: return Opcode::V_FRACT_F32;
        case 0x21: return Opcode::V_TRUNC_F32;
        case 0x22: return Opcode::V_CEIL_F32;
        case 0x23: return Opcode::V_RNDNE_F32;
        case 0x24: return Opcode::V_FLOOR_F32;
        case 0x25: return Opcode::V_EXP_F32;
        case 0x27: return Opcode::V_LOG_F32;
        case 0x2A: return Opcode::V_RCP_F32;
        case 0x2B: return Opcode::V_RCP_IFLAG_F32;
        case 0x2E: return Opcode::V_RSQ_F32;
        case 0x2F: return Opcode::V_RCP_F64;
        case 0x31: return Opcode::V_RSQ_F64;
        case 0x33: return Opcode::V_SQRT_F32;
        case 0x34: return Opcode::V_SQRT_F64;
        case 0x35: return Opcode::V_SIN_F32;
        case 0x36: return Opcode::V_COS_F32;
        case 0x37: return Opcode::V_NOT_B32;
        case 0x38: return Opcode::V_BFREV_B32;
        case 0x39: return Opcode::V_FFBH_U32;  // v_clz_i32_u32
        case 0x3A: return Opcode::V_FFBL_B32;  // v_ctz_i32_b32
        case 0x3B: return Opcode::V_FFBH_I32;  // v_cls_i32
        case 0x3C: return Opcode::V_FREXP_EXP_I32_F64;
        case 0x3D: return Opcode::V_FREXP_MANT_F64;
        case 0x3E: return Opcode::V_FRACT_F64;
        case 0x3F: return Opcode::V_FREXP_EXP_I32_F32;
        case 0x40: return Opcode::V_FREXP_MANT_F32;
        case 0x50: return Opcode::V_CVT_F16_U16;
        case 0x51: return Opcode::V_CVT_F16_I16;
        case 0x52: return Opcode::V_CVT_U16_F16;
        case 0x53: return Opcode::V_CVT_I16_F16;
        case 0x54: return Opcode::V_RCP_F16;
        case 0x55: return Opcode::V_SQRT_F16;
        case 0x56: return Opcode::V_RSQ_F16;
        case 0x57: return Opcode::V_LOG_F16;
        case 0x58: return Opcode::V_EXP_F16;
        case 0x59: return Opcode::V_FREXP_MANT_F16;
        case 0x5A: return Opcode::V_FREXP_EXP_I16_F16;
        case 0x5B: return Opcode::V_FLOOR_F16;
        case 0x5C: return Opcode::V_CEIL_F16;
        case 0x5D: return Opcode::V_TRUNC_F16;
        case 0x5E: return Opcode::V_RNDNE_F16;
        case 0x5F: return Opcode::V_FRACT_F16;
        case 0x60: return Opcode::V_SIN_F16;
        case 0x61: return Opcode::V_COS_F16;
        case 0x6C: return Opcode::V_CVT_F32_FP8;
        case 0x6D: return Opcode::V_CVT_F32_BF8;
        default:   return Opcode::UNKNOWN;
    }
}

static Opcode decode_gfx12_vop2_op(uint32_t op) {
    switch (op) {
        case 0x01: return Opcode::V_CNDMASK_B32;
        case 0x02: return Opcode::V_ADD_F64;
        case 0x03: return Opcode::V_ADD_F32;
        case 0x04: return Opcode::V_SUB_F32;
        case 0x05: return Opcode::V_SUBREV_F32;
        case 0x06: return Opcode::V_MUL_F64;
        case 0x07: return Opcode::V_MUL_LEGACY_F32;
        case 0x08: return Opcode::V_MUL_F32;
        case 0x09: return Opcode::V_MUL_I32_I24;
        case 0x0A: return Opcode::V_MUL_HI_I32_I24;
        case 0x0B: return Opcode::V_MUL_U32_U24;
        case 0x0C: return Opcode::V_MUL_HI_U32_U24;
        case 0x0D: return Opcode::V_MIN_F64;
        case 0x0E: return Opcode::V_MAX_F64;
        case 0x11: return Opcode::V_MIN_I32;
        case 0x12: return Opcode::V_MAX_I32;
        case 0x13: return Opcode::V_MIN_U32;
        case 0x14: return Opcode::V_MAX_U32;
        case 0x15: return Opcode::V_MIN_F32;
        case 0x16: return Opcode::V_MAX_F32;
        case 0x18: return Opcode::V_LSHLREV_B32;
        case 0x19: return Opcode::V_LSHRREV_B32;
        case 0x1A: return Opcode::V_ASHRREV_I32;
        case 0x1B: return Opcode::V_AND_B32;
        case 0x1C: return Opcode::V_OR_B32;
        case 0x1D: return Opcode::V_XOR_B32;
        case 0x1E: return Opcode::V_XNOR_B32;
        case 0x1F: return Opcode::V_LSHLREV_B64;
        case 0x20: return Opcode::V_ADDC_CO_U32;    // v_add_co_ci_u32
        case 0x21: return Opcode::V_SUBB_CO_U32;    // v_sub_co_ci_u32
        case 0x22: return Opcode::V_SUBBREV_CO_U32; // v_subrev_co_ci_u32
        case 0x25: return Opcode::V_ADD_U32;        // v_add_nc_u32
        case 0x26: return Opcode::V_SUB_U32;        // v_sub_nc_u32
        case 0x27: return Opcode::V_SUBREV_U32;     // v_subrev_nc_u32
        case 0x2B: return Opcode::V_FMAC_F32;
        case 0x2C: return Opcode::V_FMAMK_F32;
        case 0x2D: return Opcode::V_FMAAK_F32;
        case 0x2F: return Opcode::V_CVT_PKRTZ_F16_F32;
        case 0x30: return Opcode::V_MIN_F16;
        case 0x31: return Opcode::V_MAX_F16;
        case 0x32: return Opcode::V_ADD_F16;
        case 0x33: return Opcode::V_SUB_F16;
        case 0x34: return Opcode::V_SUBREV_F16;
        case 0x35: return Opcode::V_MUL_F16;
        case 0x36: return Opcode::V_FMAC_F16;
        case 0x3B: return Opcode::V_LDEXP_F16;
        default:   return Opcode::UNKNOWN;
    }
}

static Opcode decode_gfx12_vopc_op(uint32_t op, bool is_cmpx) {
    uint32_t base = op & 0x7F;
    if (!is_cmpx) {
        switch (base) {
            case 0x01: return Opcode::V_CMP_LT_F16;
            case 0x02: return Opcode::V_CMP_EQ_F16;
            case 0x03: return Opcode::V_CMP_LE_F16;
            case 0x04: return Opcode::V_CMP_GT_F16;
            case 0x05: return Opcode::V_CMP_LG_F16;
            case 0x06: return Opcode::V_CMP_GE_F16;
            case 0x07: return Opcode::V_CMP_O_F16;
            case 0x08: return Opcode::V_CMP_U_F16;
            case 0x09: return Opcode::V_CMP_NGE_F16;
            case 0x0A: return Opcode::V_CMP_NLG_F16;
            case 0x0B: return Opcode::V_CMP_NGT_F16;
            case 0x0C: return Opcode::V_CMP_NLE_F16;
            case 0x0D: return Opcode::V_CMP_NEQ_F16;
            case 0x0E: return Opcode::V_CMP_NLT_F16;
            case 0x10: return Opcode::V_CMP_F_F32;
            case 0x11: return Opcode::V_CMP_LT_F32;
            case 0x12: return Opcode::V_CMP_EQ_F32;
            case 0x13: return Opcode::V_CMP_LE_F32;
            case 0x14: return Opcode::V_CMP_GT_F32;
            case 0x15: return Opcode::V_CMP_LG_F32;
            case 0x16: return Opcode::V_CMP_GE_F32;
            case 0x17: return Opcode::V_CMP_O_F32;
            case 0x18: return Opcode::V_CMP_U_F32;
            case 0x19: return Opcode::V_CMP_NGE_F32;
            case 0x1A: return Opcode::V_CMP_NLG_F32;
            case 0x1B: return Opcode::V_CMP_NGT_F32;
            case 0x1C: return Opcode::V_CMP_NLE_F32;
            case 0x1D: return Opcode::V_CMP_NEQ_F32;
            case 0x1E: return Opcode::V_CMP_NLT_F32;
            case 0x1F: return Opcode::V_CMP_TRU_F32;
            case 0x20: return Opcode::V_CMP_F_F64;
            case 0x21: return Opcode::V_CMP_LT_F64;
            case 0x22: return Opcode::V_CMP_EQ_F64;
            case 0x23: return Opcode::V_CMP_LE_F64;
            case 0x24: return Opcode::V_CMP_GT_F64;
            case 0x25: return Opcode::V_CMP_LG_F64;
            case 0x26: return Opcode::V_CMP_GE_F64;
            case 0x27: return Opcode::V_CMP_O_F64;
            case 0x28: return Opcode::V_CMP_U_F64;
            case 0x29: return Opcode::V_CMP_NGE_F64;
            case 0x2A: return Opcode::V_CMP_NLG_F64;
            case 0x2B: return Opcode::V_CMP_NGT_F64;
            case 0x2C: return Opcode::V_CMP_NLE_F64;
            case 0x2D: return Opcode::V_CMP_NEQ_F64;
            case 0x2E: return Opcode::V_CMP_NLT_F64;
            case 0x2F: return Opcode::V_CMP_TRU_F64;
            case 0x31: return Opcode::V_CMP_LT_I16;
            case 0x32: return Opcode::V_CMP_EQ_I16;
            case 0x33: return Opcode::V_CMP_LE_I16;
            case 0x34: return Opcode::V_CMP_GT_I16;
            case 0x35: return Opcode::V_CMP_NE_I16;
            case 0x36: return Opcode::V_CMP_GE_I16;
            case 0x39: return Opcode::V_CMP_LT_U16;
            case 0x3A: return Opcode::V_CMP_EQ_U16;
            case 0x3B: return Opcode::V_CMP_LE_U16;
            case 0x3C: return Opcode::V_CMP_GT_U16;
            case 0x3D: return Opcode::V_CMP_NE_U16;
            case 0x3E: return Opcode::V_CMP_GE_U16;
            case 0x41: return Opcode::V_CMP_LT_I32;
            case 0x42: return Opcode::V_CMP_EQ_I32;
            case 0x43: return Opcode::V_CMP_LE_I32;
            case 0x44: return Opcode::V_CMP_GT_I32;
            case 0x45: return Opcode::V_CMP_NE_I32;
            case 0x46: return Opcode::V_CMP_GE_I32;
            case 0x49: return Opcode::V_CMP_LT_U32;
            case 0x4A: return Opcode::V_CMP_EQ_U32;
            case 0x4B: return Opcode::V_CMP_LE_U32;
            case 0x4C: return Opcode::V_CMP_GT_U32;
            case 0x4D: return Opcode::V_CMP_NE_U32;
            case 0x4E: return Opcode::V_CMP_GE_U32;
            case 0x51: return Opcode::V_CMP_LT_I64;
            case 0x52: return Opcode::V_CMP_EQ_I64;
            case 0x53: return Opcode::V_CMP_LE_I64;
            case 0x54: return Opcode::V_CMP_GT_I64;
            case 0x55: return Opcode::V_CMP_NE_I64;
            case 0x56: return Opcode::V_CMP_GE_I64;
            case 0x59: return Opcode::V_CMP_LT_U64;
            case 0x5A: return Opcode::V_CMP_EQ_U64;
            case 0x5B: return Opcode::V_CMP_LE_U64;
            case 0x5C: return Opcode::V_CMP_GT_U64;
            case 0x5D: return Opcode::V_CMP_NE_U64;
            case 0x5E: return Opcode::V_CMP_GE_U64;
            case 0x7D: return Opcode::V_CMP_CLASS_F16;
            case 0x7E: return Opcode::V_CMP_CLASS_F32;
            case 0x7F: return Opcode::V_CMP_CLASS_F64;
            default:   return Opcode::UNKNOWN;
        }
    } else {
        switch (base) {
            case 0x01: return Opcode::V_CMPX_LT_F16;
            case 0x02: return Opcode::V_CMPX_EQ_F16;
            case 0x03: return Opcode::V_CMPX_LE_F16;
            case 0x04: return Opcode::V_CMPX_GT_F16;
            case 0x05: return Opcode::V_CMPX_LG_F16;
            case 0x06: return Opcode::V_CMPX_GE_F16;
            case 0x11: return Opcode::V_CMPX_LT_F32;
            case 0x12: return Opcode::V_CMPX_EQ_F32;
            case 0x13: return Opcode::V_CMPX_LE_F32;
            case 0x14: return Opcode::V_CMPX_GT_F32;
            case 0x15: return Opcode::V_CMPX_LG_F32;
            case 0x16: return Opcode::V_CMPX_GE_F32;
            case 0x17: return Opcode::V_CMPX_O_F32;
            case 0x18: return Opcode::V_CMPX_U_F32;
            case 0x19: return Opcode::V_CMPX_NGE_F32;
            case 0x1A: return Opcode::V_CMPX_NLG_F32;
            case 0x1B: return Opcode::V_CMPX_NGT_F32;
            case 0x1C: return Opcode::V_CMPX_NLE_F32;
            case 0x1D: return Opcode::V_CMPX_NEQ_F32;
            case 0x1E: return Opcode::V_CMPX_NLT_F32;
            case 0x21: return Opcode::V_CMPX_LT_F64;
            case 0x22: return Opcode::V_CMPX_EQ_F64;
            case 0x23: return Opcode::V_CMPX_LE_F64;
            case 0x24: return Opcode::V_CMPX_GT_F64;
            case 0x25: return Opcode::V_CMPX_LG_F64;
            case 0x26: return Opcode::V_CMPX_GE_F64;
            case 0x31: return Opcode::V_CMPX_LT_I16;
            case 0x32: return Opcode::V_CMPX_EQ_I16;
            case 0x33: return Opcode::V_CMPX_LE_I16;
            case 0x34: return Opcode::V_CMPX_GT_I16;
            case 0x35: return Opcode::V_CMPX_NE_I16;
            case 0x36: return Opcode::V_CMPX_GE_I16;
            case 0x39: return Opcode::V_CMPX_LT_U16;
            case 0x3A: return Opcode::V_CMPX_EQ_U16;
            case 0x3B: return Opcode::V_CMPX_LE_U16;
            case 0x3C: return Opcode::V_CMPX_GT_U16;
            case 0x3D: return Opcode::V_CMPX_NE_U16;
            case 0x3E: return Opcode::V_CMPX_GE_U16;
            case 0x41: return Opcode::V_CMPX_LT_I32;
            case 0x42: return Opcode::V_CMPX_EQ_I32;
            case 0x43: return Opcode::V_CMPX_LE_I32;
            case 0x44: return Opcode::V_CMPX_GT_I32;
            case 0x45: return Opcode::V_CMPX_NE_I32;
            case 0x46: return Opcode::V_CMPX_GE_I32;
            case 0x49: return Opcode::V_CMPX_LT_U32;
            case 0x4A: return Opcode::V_CMPX_EQ_U32;
            case 0x4B: return Opcode::V_CMPX_LE_U32;
            case 0x4C: return Opcode::V_CMPX_GT_U32;
            case 0x4D: return Opcode::V_CMPX_NE_U32;
            case 0x4E: return Opcode::V_CMPX_GE_U32;
            case 0x51: return Opcode::V_CMPX_LT_I64;
            case 0x52: return Opcode::V_CMPX_EQ_I64;
            case 0x53: return Opcode::V_CMPX_LE_I64;
            case 0x54: return Opcode::V_CMPX_GT_I64;
            case 0x55: return Opcode::V_CMPX_NE_I64;
            case 0x56: return Opcode::V_CMPX_GE_I64;
            case 0x59: return Opcode::V_CMPX_LT_U64;
            case 0x5A: return Opcode::V_CMPX_EQ_U64;
            case 0x5B: return Opcode::V_CMPX_LE_U64;
            case 0x5C: return Opcode::V_CMPX_GT_U64;
            case 0x5D: return Opcode::V_CMPX_NE_U64;
            case 0x5E: return Opcode::V_CMPX_GE_U64;
            default:   return Opcode::UNKNOWN;
        }
    }
}

static Opcode decode_gfx12_vopd_op(uint32_t op) {
    switch (op) {
        case 0x00: return Opcode::V_FMAC_F32;
        case 0x01: return Opcode::V_FMAAK_F32;
        case 0x02: return Opcode::V_FMAMK_F32;
        case 0x03: return Opcode::V_MUL_F32;
        case 0x04: return Opcode::V_ADD_F32;
        case 0x05: return Opcode::V_SUB_F32;
        case 0x06: return Opcode::V_SUBREV_F32;
        case 0x07: return Opcode::V_MUL_LEGACY_F32;
        case 0x08: return Opcode::V_MOV_B32;
        case 0x09: return Opcode::V_CNDMASK_B32;
        case 0x0A: return Opcode::V_MAX_F32;
        case 0x0B: return Opcode::V_MIN_F32;
        case 0x0C: return Opcode::V_DOT2C_F32_F16;
        case 0x10: return Opcode::V_ADD_U32;
        case 0x11: return Opcode::V_LSHLREV_B32;
        case 0x12: return Opcode::V_AND_B32;
        default:   return Opcode::UNKNOWN;
    }
}

static Instruction decode_gfx12(const uint8_t* pc) {
    Instruction inst;
    uint32_t w0 = read_u32_le(pc);
    inst.encoding[0] = w0;

    if (w0 == 0) {
        inst.format = InstFormat::SOPP;
        inst.opcode = Opcode::S_ENDPGM;
        inst.size_bytes = 4;
        return inst;
    }

    uint32_t top8 = w0 >> 24;
    uint32_t top6 = w0 >> 26;
    uint32_t top9 = w0 >> 23;

    // 1. SOPP: [31:23] == 0x17F (0xBF800000..0xBFFFFFFF)
    if (top9 == 0x17F) {
        inst.format = InstFormat::SOPP;
        inst.size_bytes = 4;
        uint32_t op = (w0 >> 16) & 0x7F;
        inst.simm16 = (int16_t)(w0 & 0xFFFF);
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_NOP; break;
            case 0x01: inst.opcode = Opcode::S_SETHALT; break;
            case 0x02: inst.opcode = Opcode::S_SLEEP; break;
            case 0x03: inst.opcode = Opcode::S_SETPRIO; break;
            case 0x05: inst.opcode = Opcode::S_CLAUSE; break;
            case 0x07: inst.opcode = Opcode::S_DELAY_ALU; break;
            case 0x08: inst.opcode = Opcode::S_WAIT_ALU; break;
            case 0x14: inst.opcode = Opcode::S_BARRIER_WAIT; break;
            case 0x1F: inst.opcode = Opcode::S_ENDPGM; break; // s_code_end
            case 0x20: inst.opcode = Opcode::S_BRANCH; break;
            case 0x21: inst.opcode = Opcode::S_CBRANCH_SCC0; break;
            case 0x22: inst.opcode = Opcode::S_CBRANCH_SCC1; break;
            case 0x23: inst.opcode = Opcode::S_CBRANCH_VCCZ; break;
            case 0x24: inst.opcode = Opcode::S_CBRANCH_VCCNZ; break;
            case 0x25: inst.opcode = Opcode::S_CBRANCH_EXECZ; break;
            case 0x26: inst.opcode = Opcode::S_CBRANCH_EXECNZ; break;
            case 0x30: inst.opcode = Opcode::S_ENDPGM; break;
            case 0x34: inst.opcode = Opcode::S_WAKEUP; break;
            case 0x36: inst.opcode = Opcode::S_SENDMSG; break;
            case 0x37: inst.opcode = Opcode::S_SENDMSGHALT; break;
            case 0x40: inst.opcode = Opcode::S_WAIT_LOADCNT; inst.wait_vmcnt = (uint8_t)(w0 & 0x3F); break;
            case 0x41: inst.opcode = Opcode::S_WAIT_STORECNT; inst.wait_vmcnt = (uint8_t)(w0 & 0x3F); break;
            case 0x46: inst.opcode = Opcode::S_WAIT_DSCNT; inst.wait_lgkmcnt = (uint8_t)(w0 & 0x3F); break;
            case 0x47: inst.opcode = Opcode::S_WAIT_KMCNT; inst.wait_lgkmcnt = (uint8_t)(w0 & 0x3F); break;
            case 0x48:
                inst.opcode = Opcode::S_WAIT_LOADCNT_DSCNT;
                inst.wait_vmcnt   = (uint8_t)((w0 >> 8) & 0x3F);
                inst.wait_lgkmcnt = (uint8_t)(w0 & 0x3F);
                break;
            case 0x49:
                inst.opcode = Opcode::S_WAIT_STORECNT_DSCNT;
                inst.wait_vmcnt   = (uint8_t)((w0 >> 8) & 0x3F);
                inst.wait_lgkmcnt = (uint8_t)(w0 & 0x3F);
                break;
            default:   inst.opcode = Opcode::S_NOP; break;
        }
        return inst;
    }

    // 2. SOPC: [31:23] == 0x17E (0xBF000000..0xBF7FFFFF)
    if (top9 == 0x17E) {
        inst.format = InstFormat::SOPC;
        uint32_t op   = (w0 >> 16) & 0x7F;
        uint32_t src0 = w0 & 0xFF;
        uint32_t src1 = (w0 >> 8) & 0xFF;
        bool has_literal = (src0 == 255) || (src1 == 255);
        const uint8_t* lit_ptr = has_literal ? (pc + 4) : nullptr;
        inst.size_bytes = has_literal ? 8 : 4;
        if (has_literal) inst.encoding[1] = read_u32_le(pc + 4);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        inst.src[1] = decode_src(src1, lit_ptr, true);
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_CMP_EQ_I32; break;
            case 0x01: inst.opcode = Opcode::S_CMP_LG_I32; break;
            case 0x02: inst.opcode = Opcode::S_CMP_GT_I32; break;
            case 0x03: inst.opcode = Opcode::S_CMP_GE_I32; break;
            case 0x04: inst.opcode = Opcode::S_CMP_LT_I32; break;
            case 0x05: inst.opcode = Opcode::S_CMP_LE_I32; break;
            case 0x06: inst.opcode = Opcode::S_CMP_EQ_U32; break;
            case 0x07: inst.opcode = Opcode::S_CMP_LG_U32; break;
            case 0x08: inst.opcode = Opcode::S_CMP_GT_U32; break;
            case 0x09: inst.opcode = Opcode::S_CMP_GE_U32; break;
            case 0x0A: inst.opcode = Opcode::S_CMP_LT_U32; break;
            case 0x0B: inst.opcode = Opcode::S_CMP_LE_U32; break;
            case 0x0C: inst.opcode = Opcode::S_BITCMP0_B32; break;
            case 0x0D: inst.opcode = Opcode::S_BITCMP1_B32; break;
            case 0x0E: inst.opcode = Opcode::S_BITCMP0_B64; break;
            case 0x0F: inst.opcode = Opcode::S_BITCMP1_B64; break;
            case 0x10: inst.opcode = Opcode::S_CMP_EQ_U64; break;
            case 0x11: inst.opcode = Opcode::S_CMP_LG_U64; break;
            case 0x42: inst.opcode = Opcode::S_CMP_EQ_F32; break;
            case 0x4D: inst.opcode = Opcode::S_CMP_NEQ_F32; break;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 3. SOP1: [31:23] == 0x17D (0xBE800000..0xBEFFFFFF)
    if (top9 == 0x17D) {
        inst.format = InstFormat::SOP1;
        uint32_t op   = (w0 >> 8) & 0xFF;
        uint32_t sdst = (w0 >> 16) & 0x7F;
        uint32_t src0 = w0 & 0xFF;
        bool has_literal = (src0 == 255);
        const uint8_t* lit_ptr = has_literal ? (pc + 4) : nullptr;
        inst.size_bytes = has_literal ? 8 : 4;
        if (has_literal) inst.encoding[1] = read_u32_le(pc + 4);
        inst.dst    = decode_sdst(sdst, true);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_MOV_B32; break;
            case 0x01: inst.opcode = Opcode::S_MOV_B64; break;
            case 0x02: inst.opcode = Opcode::S_CMOV_B32; break;
            case 0x03: inst.opcode = Opcode::S_CMOV_B64; break;
            case 0x04: inst.opcode = Opcode::S_BREV_B32; break;
            case 0x05: inst.opcode = Opcode::S_BREV_B64; break;
            case 0x08: inst.opcode = Opcode::S_FF1_I32_B32; break;   // s_ctz_i32_b32
            case 0x09: inst.opcode = Opcode::S_FF1_I32_B64; break;   // s_ctz_i32_b64
            case 0x0A: inst.opcode = Opcode::S_FLBIT_I32_B32; break; // s_clz_i32_u32
            case 0x0B: inst.opcode = Opcode::S_FLBIT_I32_B64; break; // s_clz_i32_u64
            case 0x0C: inst.opcode = Opcode::S_FLBIT_I32; break;     // s_cls_i32
            case 0x0D: inst.opcode = Opcode::S_FLBIT_I32_I64; break; // s_cls_i32_i64
            case 0x0E: inst.opcode = Opcode::S_SEXT_I32_I8; break;
            case 0x0F: inst.opcode = Opcode::S_SEXT_I32_I16; break;
            case 0x10: inst.opcode = Opcode::S_BITSET0_B32; break;
            case 0x11: inst.opcode = Opcode::S_BITSET0_B64; break;
            case 0x12: inst.opcode = Opcode::S_BITSET1_B32; break;
            case 0x13: inst.opcode = Opcode::S_BITSET1_B64; break;
            case 0x15: inst.opcode = Opcode::S_ABS_I32; break;
            case 0x16: inst.opcode = Opcode::S_BCNT0_I32_B32; break;
            case 0x17: inst.opcode = Opcode::S_BCNT0_I32_B64; break;
            case 0x18: inst.opcode = Opcode::S_BCNT1_I32_B32; break;
            case 0x19: inst.opcode = Opcode::S_BCNT1_I32_B64; break;
            case 0x1E: inst.opcode = Opcode::S_NOT_B32; break;
            case 0x1F: inst.opcode = Opcode::S_NOT_B64; break;
            case 0x20: inst.opcode = Opcode::S_AND_SAVEEXEC_B32; break;
            case 0x21: inst.opcode = Opcode::S_AND_SAVEEXEC_B64; break;
            case 0x22: inst.opcode = Opcode::S_OR_SAVEEXEC_B32; break;
            case 0x23: inst.opcode = Opcode::S_OR_SAVEEXEC_B64; break;
            case 0x24: inst.opcode = Opcode::S_XOR_SAVEEXEC_B32; break;
            case 0x25: inst.opcode = Opcode::S_XOR_SAVEEXEC_B64; break;
            case 0x30: inst.opcode = Opcode::S_ANDN2_SAVEEXEC_B32; break;
            case 0x31: inst.opcode = Opcode::S_ANDN2_SAVEEXEC_B64; break;
            case 0x32: inst.opcode = Opcode::S_ORN2_SAVEEXEC_B32; break;
            case 0x33: inst.opcode = Opcode::S_ORN2_SAVEEXEC_B64; break;
            case 0x40: inst.opcode = Opcode::S_MOVRELS_B32; break;
            case 0x41: inst.opcode = Opcode::S_MOVRELS_B64; break;
            case 0x42: inst.opcode = Opcode::S_MOVRELD_B32; break;
            case 0x43: inst.opcode = Opcode::S_MOVRELD_B64; break;
            case 0x47: inst.opcode = Opcode::S_GETPC_B64; break;
            case 0x48: inst.opcode = Opcode::S_SETPC_B64; break;
            case 0x49: inst.opcode = Opcode::S_SWAPPC_B64; break;
            case 0x4E: inst.opcode = Opcode::S_BARRIER_SIGNAL; break;
            case 0x62: inst.opcode = Opcode::S_TRUNC_F32; break;
            case 0x64: inst.opcode = Opcode::S_CVT_F32_I32; break;
            case 0x65: inst.opcode = Opcode::S_CVT_F32_U32; break;
            case 0x66: inst.opcode = Opcode::S_CVT_I32_F32; break;
            case 0x67: inst.opcode = Opcode::S_CVT_U32_F32; break;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 4. SOPK: [31:28] == 0xB (0xB0000000..0xBDFFFFFF)
    if ((w0 >> 28) == 0xB) {
        inst.format = InstFormat::SOPK;
        uint32_t op   = (w0 >> 23) & 0x1F;
        uint32_t sdst = (w0 >> 16) & 0x7F;
        inst.simm16   = (int16_t)(w0 & 0xFFFF);
        inst.dst      = decode_sdst(sdst, true);
        inst.size_bytes = 4;
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_MOVK_I32; break;
            case 0x01: inst.opcode = Opcode::S_CMOVK_I32; break;
            case 0x02: inst.opcode = Opcode::S_CMPK_EQ_I32; break;
            case 0x03: inst.opcode = Opcode::S_CMPK_LG_I32; break;
            case 0x04: inst.opcode = Opcode::S_CMPK_GT_I32; break;
            case 0x05: inst.opcode = Opcode::S_CMPK_GE_I32; break;
            case 0x06: inst.opcode = Opcode::S_CMPK_LT_I32; break;
            case 0x07: inst.opcode = Opcode::S_CMPK_LE_I32; break;
            case 0x08: inst.opcode = Opcode::S_CMPK_EQ_U32; break;
            case 0x09: inst.opcode = Opcode::S_CMPK_LG_U32; break;
            case 0x0A: inst.opcode = Opcode::S_CMPK_GT_U32; break;
            case 0x0B: inst.opcode = Opcode::S_CMPK_GE_U32; break;
            case 0x0C: inst.opcode = Opcode::S_CMPK_LT_U32; break;
            case 0x0D: inst.opcode = Opcode::S_CMPK_LE_U32; break;
            case 0x0E:
            case 0x0F: inst.opcode = Opcode::S_ADDK_I32; break;
            case 0x10: inst.opcode = Opcode::S_MULK_I32; break;
            case 0x11: inst.opcode = Opcode::S_GETREG_B32; break;
            case 0x12: inst.opcode = Opcode::S_SETREG_B32; break;
            case 0x13:
                inst.opcode = Opcode::S_SETREG_IMM32_B32;
                inst.size_bytes = 8;
                inst.encoding[1] = read_u32_le(pc + 4);
                inst.src[0].kind = Operand::Kind::LITERAL;
                inst.src[0].imm  = inst.encoding[1];
                break;
            default: inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 5. SOP2: [31:30] == 2 (0x80000000..0xAFFFFFFF)
    if ((w0 >> 30) == 2) {
        inst.format = InstFormat::SOP2;
        uint32_t op   = (w0 >> 23) & 0x7F;
        uint32_t sdst = (w0 >> 16) & 0x7F;
        uint32_t src0 = w0 & 0xFF;
        uint32_t src1 = (w0 >> 8) & 0xFF;
        bool is_fma_lit = (op == 0x46 || op == 0x47); // s_fmamk_f32 / s_fmaak_f32 always have 32-bit literal
        bool has_literal = (src0 == 255) || (src1 == 255) || is_fma_lit;
        const uint8_t* lit_ptr = has_literal ? (pc + 4) : nullptr;
        inst.size_bytes = has_literal ? 8 : 4;
        if (has_literal) inst.encoding[1] = read_u32_le(pc + 4);
        inst.dst    = decode_sdst(sdst, true);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        inst.src[1] = decode_src(src1, lit_ptr, true);
        if (is_fma_lit) {
            inst.src[2].kind = Operand::Kind::LITERAL;
            inst.src[2].imm  = inst.encoding[1];
        }
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_ADD_U32; break;
            case 0x01: inst.opcode = Opcode::S_SUB_U32; break;
            case 0x02: inst.opcode = Opcode::S_ADD_I32; break;
            case 0x03: inst.opcode = Opcode::S_SUB_I32; break;
            case 0x04: inst.opcode = Opcode::S_ADDC_U32; break;
            case 0x05: inst.opcode = Opcode::S_SUBB_U32; break;
            case 0x06: inst.opcode = Opcode::S_ABSDIFF_I32; break;
            case 0x08: inst.opcode = Opcode::S_LSHL_B32; break;
            case 0x09: inst.opcode = Opcode::S_LSHL_B64; break;
            case 0x0A: inst.opcode = Opcode::S_LSHR_B32; break;
            case 0x0B: inst.opcode = Opcode::S_LSHR_B64; break;
            case 0x0C: inst.opcode = Opcode::S_ASHR_I32; break;
            case 0x0D: inst.opcode = Opcode::S_ASHR_I64; break;
            case 0x0E: inst.opcode = Opcode::S_LSHL1_ADD_U32; break;
            case 0x0F: inst.opcode = Opcode::S_LSHL2_ADD_U32; break;
            case 0x10: inst.opcode = Opcode::S_LSHL3_ADD_U32; break;
            case 0x11: inst.opcode = Opcode::S_LSHL4_ADD_U32; break;
            case 0x12: inst.opcode = Opcode::S_MIN_I32; break;
            case 0x13: inst.opcode = Opcode::S_MIN_U32; break;
            case 0x14: inst.opcode = Opcode::S_MAX_I32; break;
            case 0x15: inst.opcode = Opcode::S_MAX_U32; break;
            case 0x16: inst.opcode = Opcode::S_AND_B32; break;
            case 0x17: inst.opcode = Opcode::S_AND_B64; break;
            case 0x18: inst.opcode = Opcode::S_OR_B32; break;
            case 0x19: inst.opcode = Opcode::S_OR_B64; break;
            case 0x1A: inst.opcode = Opcode::S_XOR_B32; break;
            case 0x1B: inst.opcode = Opcode::S_XOR_B64; break;
            case 0x1C: inst.opcode = Opcode::S_NAND_B32; break;
            case 0x1D: inst.opcode = Opcode::S_NAND_B64; break;
            case 0x1E: inst.opcode = Opcode::S_NOR_B32; break;
            case 0x1F: inst.opcode = Opcode::S_NOR_B64; break;
            case 0x20: inst.opcode = Opcode::S_XNOR_B32; break;
            case 0x21: inst.opcode = Opcode::S_XNOR_B64; break;
            case 0x22: inst.opcode = Opcode::S_ANDN2_B32; break; // s_and_not1_b32
            case 0x23: inst.opcode = Opcode::S_ANDN2_B64; break;
            case 0x24: inst.opcode = Opcode::S_ORN2_B32; break;  // s_or_not1_b32
            case 0x25: inst.opcode = Opcode::S_ORN2_B64; break;
            case 0x26: inst.opcode = Opcode::S_BFE_U32; break;
            case 0x27: inst.opcode = Opcode::S_BFE_I32; break;
            case 0x28: inst.opcode = Opcode::S_BFE_U64; break;
            case 0x29: inst.opcode = Opcode::S_BFE_I64; break;
            case 0x2A: inst.opcode = Opcode::S_BFM_B32; break;
            case 0x2B: inst.opcode = Opcode::S_BFM_B64; break;
            case 0x2C: inst.opcode = Opcode::S_MUL_I32; break;
            case 0x2D: inst.opcode = Opcode::S_MUL_HI_U32; break;
            case 0x2E: inst.opcode = Opcode::S_MUL_HI_I32; break;
            case 0x30: inst.opcode = Opcode::S_CSELECT_B32; break;
            case 0x31: inst.opcode = Opcode::S_CSELECT_B64; break;
            case 0x32: inst.opcode = Opcode::S_PACK_LL_B32_B16; break;
            case 0x33: inst.opcode = Opcode::S_PACK_LH_B32_B16; break;
            case 0x34: inst.opcode = Opcode::S_PACK_HH_B32_B16; break;
            case 0x44: inst.opcode = Opcode::S_MUL_F32; break;
            case 0x46: inst.opcode = Opcode::S_FMAMK_F32; break;
            case 0x47: inst.opcode = Opcode::S_FMAAK_F32; break;
            case 0x50: inst.opcode = Opcode::S_ADD_U32; break; // s_add_nc_u32
            case 0x51: inst.opcode = Opcode::S_SUB_U32; break; // s_sub_nc_u32
            case 0x53: inst.opcode = Opcode::S_ADD_NC_U64; break;
            case 0x54: inst.opcode = Opcode::S_SUB_NC_U64; break;
            case 0x55: inst.opcode = Opcode::S_MUL_U64; break;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 6. SMEM: [31:26] == 0x3D (0xF4000000..0xF7FFFFFF)
    if (top6 == 0x3D) {
        inst.format = InstFormat::SMEM;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        uint32_t sbase = (w0 & 0x3F) << 1;
        uint32_t sdst  = (w0 >> 6) & 0x7F;
        uint32_t op    = (w0 >> 13) & 0xFF;
        inst.dst.kind  = Operand::Kind::SGPR;
        inst.dst.index = sdst;
        inst.src[0].kind  = Operand::Kind::SGPR;
        inst.src[0].index = sbase;
        // Sign-extend 24-bit immediate offset in w1[23:0]
        int32_t ioff = (int32_t)(w1 & 0xFFFFFF);
        if (ioff & 0x800000) ioff |= ~0xFFFFFF;
        inst.smem_offset = ioff;
        uint32_t soff_enc = (w1 >> 25) & 0x7F;
        if (soff_enc < 106 || (soff_enc >= 108 && soff_enc <= 123)) {
            inst.src[1].kind  = Operand::Kind::SGPR;
            inst.src[1].index = soff_enc;
        } else {
            inst.src[1].kind = Operand::Kind::INLINE_CONST;
            inst.src[1].imm  = 0;
        }
        switch (op) {
            case 0x00: inst.opcode = Opcode::S_LOAD_DWORD; break;
            case 0x01: inst.opcode = Opcode::S_LOAD_DWORDX2; break;
            case 0x02: inst.opcode = Opcode::S_LOAD_DWORDX4; break;
            case 0x03: inst.opcode = Opcode::S_LOAD_DWORDX8; break;
            case 0x04: inst.opcode = Opcode::S_LOAD_DWORDX16; break;
            case 0x05: inst.opcode = Opcode::S_LOAD_DWORDX3; break; // s_load_b96
            case 0x08: inst.opcode = Opcode::S_STORE_DWORD; break;
            case 0x09: inst.opcode = Opcode::S_STORE_DWORDX2; break;
            case 0x0A: inst.opcode = Opcode::S_STORE_DWORDX4; break;
            case 0x10: inst.opcode = Opcode::S_BUFFER_LOAD_DWORD; break;
            case 0x11: inst.opcode = Opcode::S_BUFFER_LOAD_DWORDX2; break;
            case 0x12: inst.opcode = Opcode::S_BUFFER_LOAD_DWORDX4; break;
            case 0x13: inst.opcode = Opcode::S_BUFFER_LOAD_DWORDX8; break;
            case 0x14: inst.opcode = Opcode::S_BUFFER_LOAD_DWORDX16; break;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 7. 96-bit VFLAT / VSCRATCH / VGLOBAL: top8 == 0xEC, 0xED, 0xEE
    if (top8 == 0xEC || top8 == 0xED || top8 == 0xEE) {
        inst.format = (top8 == 0xED) ? InstFormat::FLAT_SCRATCH : InstFormat::FLAT_GLOBAL;
        inst.size_bytes = 12;
        uint32_t w1 = read_u32_le(pc + 4);
        uint32_t w2 = read_u32_le(pc + 8);
        inst.encoding[1] = w1;
        inst.encoding[2] = w2;

        uint32_t saddr = w0 & 0x7F;
        uint32_t op    = (w0 >> 14) & 0xFF;
        uint32_t vdst  = w1 & 0xFF;
        uint32_t vdata = (w1 >> 23) & 0xFF;
        uint32_t vaddr = w2 & 0xFF;
        int32_t  ioff  = (int32_t)w2 >> 8; // 24-bit signed offset in w2[31:8]
        inst.flat_offset = ioff;

        bool is_global = (top8 == 0xEE) || (top8 == 0xED);
        bool is_store  = false;
        switch (op) {
            case 0x10: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_UBYTE : Opcode::FLAT_LOAD_UBYTE; break;
            case 0x11: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_SBYTE : Opcode::FLAT_LOAD_SBYTE; break;
            case 0x12: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_USHORT : Opcode::FLAT_LOAD_USHORT; break;
            case 0x13: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_SSHORT : Opcode::FLAT_LOAD_SSHORT; break;
            case 0x14: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_DWORD : Opcode::FLAT_LOAD_DWORD; break;
            case 0x15: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_DWORDX2 : Opcode::FLAT_LOAD_DWORDX2; break;
            case 0x16: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_DWORDX3 : Opcode::FLAT_LOAD_DWORDX3; break;
            case 0x17: inst.opcode = is_global ? Opcode::GLOBAL_LOAD_DWORDX4 : Opcode::FLAT_LOAD_DWORDX4; break;
            case 0x18: inst.opcode = is_global ? Opcode::GLOBAL_STORE_BYTE : Opcode::FLAT_STORE_BYTE; is_store = true; break;
            case 0x19: inst.opcode = is_global ? Opcode::GLOBAL_STORE_SHORT : Opcode::FLAT_STORE_SHORT; is_store = true; break;
            case 0x1A: inst.opcode = is_global ? Opcode::GLOBAL_STORE_DWORD : Opcode::FLAT_STORE_DWORD; is_store = true; break;
            case 0x1B: inst.opcode = is_global ? Opcode::GLOBAL_STORE_DWORDX2 : Opcode::FLAT_STORE_DWORDX2; is_store = true; break;
            case 0x1C: inst.opcode = is_global ? Opcode::GLOBAL_STORE_DWORDX3 : Opcode::FLAT_STORE_DWORDX3; is_store = true; break;
            case 0x1D: inst.opcode = is_global ? Opcode::GLOBAL_STORE_DWORDX4 : Opcode::FLAT_STORE_DWORDX4; is_store = true; break;
            case 0x20: inst.opcode = Opcode::GLOBAL_LOAD_SHORT_D16; break;
            case 0x23: inst.opcode = Opcode::GLOBAL_LOAD_SHORT_D16_HI; break;
            case 0x25: inst.opcode = Opcode::GLOBAL_STORE_SHORT_D16_HI; is_store = true; break;
            case 0x2B: inst.opcode = Opcode::GLOBAL_INV; return inst;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }

        bool has_saddr = (saddr < 106) || (saddr >= 108 && saddr <= 123);
        if (!is_store) {
            inst.dst.kind     = Operand::Kind::VGPR;
            inst.dst.index    = vdst;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (has_saddr) {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr;
            } else {
                inst.src[1].kind = Operand::Kind::NONE;
            }
        } else {
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (has_saddr) {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr;
            } else {
                inst.src[1].kind = Operand::Kind::NONE;
            }
            inst.src[2].kind  = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
        }
        return inst;
    }

    // 8. 96-bit VBUFFER: top6 == 0x31 (0xC4000000..0xC7FFFFFF)
    if (top6 == 0x31) {
        inst.format = InstFormat::MUBUF;
        inst.size_bytes = 12;
        uint32_t w1 = read_u32_le(pc + 4);
        uint32_t w2 = read_u32_le(pc + 8);
        inst.encoding[1] = w1;
        inst.encoding[2] = w2;

        uint32_t soffset_enc = w0 & 0xFF;
        uint32_t op          = (w0 >> 14) & 0xFF;
        uint32_t vdata       = w1 & 0xFF;
        uint32_t srsrc       = (w1 >> 9) & 0x7C;
        inst.mubuf_offen     = (w1 >> 30) & 1;
        inst.mubuf_idxen     = (w1 >> 31) & 1;
        uint32_t vaddr       = w2 & 0xFF;
        inst.mubuf_offset    = (w2 >> 8) & 0xFFFFFF;
        inst.mubuf_soffset   = decode_src(soffset_enc, nullptr, true);

        bool is_store = false;
        switch (op) {
            case 0x10: inst.opcode = Opcode::BUFFER_LOAD_UBYTE; break;
            case 0x11: inst.opcode = Opcode::BUFFER_LOAD_SBYTE; break;
            case 0x12: inst.opcode = Opcode::BUFFER_LOAD_USHORT; break;
            case 0x13: inst.opcode = Opcode::BUFFER_LOAD_SSHORT; break;
            case 0x14: inst.opcode = Opcode::BUFFER_LOAD_DWORD; break;
            case 0x15: inst.opcode = Opcode::BUFFER_LOAD_DWORDX2; break;
            case 0x16: inst.opcode = Opcode::BUFFER_LOAD_DWORDX3; break;
            case 0x17: inst.opcode = Opcode::BUFFER_LOAD_DWORDX4; break;
            case 0x18: inst.opcode = Opcode::BUFFER_STORE_BYTE; is_store = true; break;
            case 0x19: inst.opcode = Opcode::BUFFER_STORE_SHORT; is_store = true; break;
            case 0x1A: inst.opcode = Opcode::BUFFER_STORE_DWORD; is_store = true; break;
            case 0x1B: inst.opcode = Opcode::BUFFER_STORE_DWORDX2; is_store = true; break;
            case 0x1C: inst.opcode = Opcode::BUFFER_STORE_DWORDX3; is_store = true; break;
            case 0x1D: inst.opcode = Opcode::BUFFER_STORE_DWORDX4; is_store = true; break;
            case 0x20: inst.opcode = Opcode::BUFFER_LOAD_SHORT_D16; break;
            case 0x23: inst.opcode = Opcode::BUFFER_LOAD_SHORT_D16_HI; break;
            case 0x25: inst.opcode = Opcode::BUFFER_STORE_SHORT_D16_HI; is_store = true; break;
            default:   inst.opcode = Opcode::UNKNOWN; break;
        }

        if (!is_store) {
            inst.dst.kind     = Operand::Kind::VGPR;
            inst.dst.index    = vdata;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            inst.src[1].kind  = Operand::Kind::SGPR;
            inst.src[1].index = srsrc;
        } else {
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            inst.src[1].kind  = Operand::Kind::SGPR;
            inst.src[1].index = srsrc;
            inst.src[2].kind  = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
        }
        return inst;
    }

    // 9. VOPD (Dual-Issue): top6 == 0x32 (0xC8000000..0xCBFFFFFF)
    if (top6 == 0x32) {
        inst.format = InstFormat::VOPD;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        uint32_t opx = (w0 >> 22) & 0xF;
        uint32_t opy = (w0 >> 17) & 0x1F;
        uint32_t srcx0 = w0 & 0x1FF;
        uint32_t vsrcx1 = (w0 >> 9) & 0xFF;
        uint32_t vdstx = (w1 >> 24) & 0xFF;
        uint32_t srcy0 = w1 & 0x1FF;
        uint32_t vsrcy1 = (w1 >> 9) & 0xFF;
        uint32_t vdsty_enc = (w1 >> 17) & 0x7F;
        uint32_t vdsty = (vdsty_enc << 1) | ((vdstx & 1u) ^ 1u);

        bool has_lit = (srcx0 == 255) || (srcy0 == 255) || (opx == 1) || (opx == 2) || (opy == 1) || (opy == 2);
        const uint8_t* lit_ptr = has_lit ? (pc + 8) : nullptr;
        inst.size_bytes = has_lit ? 12 : 8;
        if (has_lit) inst.encoding[2] = read_u32_le(pc + 8);

        inst.opcode     = decode_gfx12_vopd_op(opx);
        inst.dst        = decode_vdst(vdstx);
        inst.src[0]     = decode_src(srcx0, lit_ptr, true);
        inst.src[1]     = decode_vdst(vsrcx1);
        if (opx == 1 || opx == 2) {
            inst.src[2].kind = Operand::Kind::LITERAL;
            inst.src[2].imm  = inst.encoding[2];
        } else if (opx == 9) {
            inst.src[2].kind = Operand::Kind::VCC_LO;
        }

        inst.vopd_opy     = decode_gfx12_vopd_op(opy);
        inst.vopd_dsty    = decode_vdst(vdsty);
        inst.vopd_srcy[0] = decode_src(srcy0, lit_ptr, true);
        inst.vopd_srcy[1] = decode_vdst(vsrcy1);
        if (opy == 1 || opy == 2) {
            inst.vopd_srcy[2].kind = Operand::Kind::LITERAL;
            inst.vopd_srcy[2].imm  = inst.encoding[2];
        } else if (opy == 9) {
            inst.vopd_srcy[2].kind = Operand::Kind::VCC_LO;
        }
        return inst;
    }

    // 10. VOP3P / WMMA: top8 == 0xCC
    if (top8 == 0xCC) {
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        uint32_t op   = (w0 >> 16) & 0x7F;
        uint32_t vdst = w0 & 0xFF;
        inst.neg      = (w0 >> 8) & 0x7;
        inst.op_sel   = (w0 >> 11) & 0x7;
        inst.clamp    = (w0 >> 15) & 0x1;
        inst.neg_hi   = (w1 >> 29) & 0x7;
        uint32_t op_sel_hi_01 = (w1 >> 27) & 0x3;
        uint32_t op_sel_hi_2  = (w0 >> 14) & 0x1;
        inst.op_sel_hi = (uint8_t)(op_sel_hi_01 | (op_sel_hi_2 << 2));

        uint32_t s0_enc = w1 & 0x1FF;
        uint32_t s1_enc = (w1 >> 9) & 0x1FF;
        uint32_t s2_enc = (w1 >> 18) & 0x1FF;
        bool has_lit = (s0_enc == 255) || (s1_enc == 255) || (s2_enc == 255);
        const uint8_t* lit_ptr = has_lit ? (pc + 8) : nullptr;
        inst.size_bytes = has_lit ? 12 : 8;
        if (has_lit) inst.encoding[2] = read_u32_le(pc + 8);

        inst.dst    = decode_vdst(vdst);
        inst.src[0] = decode_src(s0_enc, lit_ptr, true);
        inst.src[1] = decode_src(s1_enc, lit_ptr, true);
        inst.src[2] = decode_src(s2_enc, lit_ptr, true);

        if (op >= 0x40 && op <= 0x49) {
            inst.format = InstFormat::MFMA; // RDNA4 WMMA
            switch (op) {
                case 0x40: inst.opcode = Opcode::V_WMMA_F32_16X16X16_F16; break;
                case 0x41: inst.opcode = Opcode::V_WMMA_F32_16X16X16_BF16; break;
                case 0x42: inst.opcode = Opcode::V_WMMA_F16_16X16X16_F16; break;
                case 0x43: inst.opcode = Opcode::V_WMMA_BF16_16X16X16_BF16; break;
                case 0x44: inst.opcode = Opcode::V_WMMA_I32_16X16X16_IU8; break;
                case 0x45: inst.opcode = Opcode::V_WMMA_I32_16X16X16_IU4; break;
                case 0x46: inst.opcode = Opcode::V_WMMA_F32_16X16X16_FP8_FP8; break;
                case 0x47: inst.opcode = Opcode::V_WMMA_F32_16X16X16_FP8_BF8; break;
                case 0x48: inst.opcode = Opcode::V_WMMA_F32_16X16X16_BF8_FP8; break;
                case 0x49: inst.opcode = Opcode::V_WMMA_F32_16X16X16_BF8_BF8; break;
                default:   inst.opcode = Opcode::UNKNOWN; break;
            }
        } else {
            inst.format = InstFormat::VOP3P;
            switch (op) {
                case 0x00: inst.opcode = Opcode::V_PK_FMA_F16; break;
                case 0x01: inst.opcode = Opcode::V_PK_MUL_F16; break;
                case 0x02: inst.opcode = Opcode::V_PK_ADD_F16; break;
                case 0x03: inst.opcode = Opcode::V_PK_SUB_F16; break;
                case 0x04: inst.opcode = Opcode::V_PK_MIN_F16; break;
                case 0x05: inst.opcode = Opcode::V_PK_MAX_F16; break;
                case 0x09: inst.opcode = Opcode::V_PK_ADD_U16; break;
                case 0x0A: inst.opcode = Opcode::V_PK_SUB_U16; break;
                case 0x12: inst.opcode = Opcode::V_DOT2_F32_F16; break;
                case 0x13: inst.opcode = Opcode::V_DOT2_F32_BF16; break;
                case 0x16: inst.opcode = Opcode::V_DOT4_I32_I8; break;
                case 0x17: inst.opcode = Opcode::V_DOT4_U32_U8; break;
                case 0x20: inst.opcode = Opcode::V_FMA_MIX_F32; break;
                case 0x21: inst.opcode = Opcode::V_FMA_MIXLO_F16; break;
                case 0x22: inst.opcode = Opcode::V_FMA_MIXHI_F16; break;
                default:   inst.opcode = Opcode::UNKNOWN; break;
            }
        }
        return inst;
    }

    // 11. VOP3: top6 == 0x35 (0xD4000000..0xD7FFFFFF)
    if (top6 == 0x35) {
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        uint32_t op   = (w0 >> 16) & 0x3FF;
        uint32_t vdst = w0 & 0xFF;

        uint32_t s0_enc = w1 & 0x1FF;
        uint32_t s1_enc = (w1 >> 9) & 0x1FF;
        uint32_t s2_enc = (w1 >> 18) & 0x1FF;
        bool has_lit = (s0_enc == 255) || (s1_enc == 255) || (s2_enc == 255);
        const uint8_t* lit_ptr = has_lit ? (pc + 8) : nullptr;
        inst.size_bytes = has_lit ? 12 : 8;
        if (has_lit) inst.encoding[2] = read_u32_le(pc + 8);

        inst.src[0] = decode_src(s0_enc, lit_ptr, true);
        inst.src[1] = decode_src(s1_enc, lit_ptr, true);
        inst.src[2] = decode_src(s2_enc, lit_ptr, true);

        // VOP3B opcodes (carry-out in sdst [14:8] of w0)
        bool is_vop3b = (op == 0x120 || op == 0x121 || op == 0x122 ||
                         op == 0x2FC || op == 0x2FD || op == 0x2FE || op == 0x2FF ||
                         op == 0x300 || op == 0x301 || op == 0x302);
        if (is_vop3b) {
            inst.format = InstFormat::VOP3B;
            inst.dst    = decode_vdst(vdst);
            inst.dst2   = decode_sdst((w0 >> 8) & 0x7F, true);
            inst.clamp  = (w0 >> 15) & 0x1;
            inst.omod   = (w1 >> 27) & 0x3;
            inst.neg    = (w1 >> 29) & 0x7;
            switch (op) {
                case 0x120: inst.opcode = Opcode::V_ADDC_CO_U32; break;
                case 0x121: inst.opcode = Opcode::V_SUBB_CO_U32; break;
                case 0x122: inst.opcode = Opcode::V_SUBBREV_CO_U32; break;
                case 0x2FC: inst.opcode = Opcode::V_DIV_SCALE_F32; break;
                case 0x2FD: inst.opcode = Opcode::V_DIV_SCALE_F64; break;
                case 0x2FE: inst.opcode = Opcode::V_MAD_U64_U32; break;
                case 0x2FF: inst.opcode = Opcode::V_MAD_I64_I32; break;
                case 0x300: inst.opcode = Opcode::V_ADD_CO_U32; break;
                case 0x301: inst.opcode = Opcode::V_SUB_CO_U32; break;
                case 0x302: inst.opcode = Opcode::V_SUBREV_CO_U32; break;
                default:    inst.opcode = Opcode::UNKNOWN; break;
            }
            return inst;
        }

        inst.format = InstFormat::VOP3A;
        inst.abs_   = (w0 >> 8) & 0x7;
        inst.clamp  = (w0 >> 15) & 0x1;
        inst.omod   = (w1 >> 27) & 0x3;
        inst.neg    = (w1 >> 29) & 0x7;
        // GFX12 VOP3 OPSEL is in w0[14:11]
        inst.op_sel = (w0 >> 11) & 0xF;

        if (op < 0x100) {
            // VOPC e64
            bool is_cmpx = (op & 0x80) != 0;
            inst.opcode = decode_gfx12_vopc_op(op & 0x7F, is_cmpx);
            inst.dst    = is_cmpx ? decode_sdst(126, true) : decode_sdst(vdst, true);
            return inst;
        }
        if (op >= 0x100 && op < 0x180) {
            inst.opcode = decode_gfx12_vop2_op(op & 0x3F);
            inst.dst    = decode_vdst(vdst);
            return inst;
        }
        if (op >= 0x180 && op < 0x200) {
            inst.opcode = decode_gfx12_vop1_op(op & 0x7F);
            if (inst.opcode == Opcode::V_READFIRSTLANE_B32) {
                inst.dst = decode_sdst(vdst, true);
            } else {
                inst.dst = decode_vdst(vdst);
            }
            return inst;
        }

        inst.dst = decode_vdst(vdst);
        switch (op) {
            case 0x210: inst.opcode = Opcode::V_BFE_U32; break;
            case 0x211: inst.opcode = Opcode::V_BFE_I32; break;
            case 0x212: inst.opcode = Opcode::V_BFI_B32; break;
            case 0x213: inst.opcode = Opcode::V_FMA_F32; break;
            case 0x214: inst.opcode = Opcode::V_FMA_F64; break;
            case 0x216: inst.opcode = Opcode::V_ALIGNBIT_B32; break;
            case 0x217: inst.opcode = Opcode::V_ALIGNBYTE_B32; break;
            case 0x21A: inst.opcode = Opcode::V_MIN3_I32; break;
            case 0x21B: inst.opcode = Opcode::V_MIN3_U32; break;
            case 0x21D: inst.opcode = Opcode::V_MAX3_I32; break;
            case 0x21E: inst.opcode = Opcode::V_MAX3_U32; break;
            case 0x220: inst.opcode = Opcode::V_MED3_I32; break;
            case 0x221: inst.opcode = Opcode::V_MED3_U32; break;
            case 0x225: inst.opcode = Opcode::V_SAD_U32; break;
            case 0x226: inst.opcode = Opcode::V_CVT_PK_U8_F32; break;
            case 0x227: inst.opcode = Opcode::V_DIV_FIXUP_F32; break;
            case 0x228: inst.opcode = Opcode::V_DIV_FIXUP_F64; break;
            case 0x229: inst.opcode = Opcode::V_MIN3_F32; break;
            case 0x22A: inst.opcode = Opcode::V_MAX3_F32; break;
            case 0x231: inst.opcode = Opcode::V_MED3_F32; break;
            case 0x237: inst.opcode = Opcode::V_DIV_FMAS_F32; break;
            case 0x238: inst.opcode = Opcode::V_DIV_FMAS_F64; break;
            case 0x244: inst.opcode = Opcode::V_PERM_B32; break;
            case 0x245: inst.opcode = Opcode::V_XAD_U32; break;
            case 0x246: inst.opcode = Opcode::V_LSHL_ADD_U32; break;
            case 0x247: inst.opcode = Opcode::V_ADD_LSHL_U32; break;
            case 0x255: inst.opcode = Opcode::V_ADD3_U32; break;
            case 0x256: inst.opcode = Opcode::V_LSHL_OR_B32; break;
            case 0x257: inst.opcode = Opcode::V_AND_OR_B32; break;
            case 0x258: inst.opcode = Opcode::V_OR3_B32; break;
            case 0x264: inst.opcode = Opcode::V_MAXMIN_I32; break;
            case 0x265: inst.opcode = Opcode::V_MINMAX_I32; break;
            case 0x266: inst.opcode = Opcode::V_MAXMIN_U32; break;
            case 0x267: inst.opcode = Opcode::V_MINMAX_U32; break;
            case 0x284:
                inst.opcode = Opcode::V_S_RCP_F32;
                inst.dst    = decode_sdst(vdst, true);
                break;
            case 0x311: inst.opcode = Opcode::V_PACK_B32_F16; break;
            case 0x31C: inst.opcode = Opcode::V_LDEXP_F32; break;
            case 0x31E: inst.opcode = Opcode::V_BCNT_U32_B32; break;
            case 0x31F: inst.opcode = Opcode::V_MBCNT_LO_U32_B32; break;
            case 0x320: inst.opcode = Opcode::V_MBCNT_HI_U32_B32; break;
            case 0x326: inst.opcode = Opcode::V_ADD_I32; break; // v_add_nc_i32
            case 0x327: inst.opcode = Opcode::V_SUB_I32; break; // v_sub_nc_i32
            case 0x32B: inst.opcode = Opcode::V_LDEXP_F64; break;
            case 0x32C: inst.opcode = Opcode::V_MUL_LO_U32; break;
            case 0x32D: inst.opcode = Opcode::V_MUL_HI_U32; break;
            case 0x32E: inst.opcode = Opcode::V_MUL_HI_I32; break;
            case 0x338: inst.opcode = Opcode::V_LSHLREV_B16; break;
            case 0x339: inst.opcode = Opcode::V_LSHRREV_B16; break;
            case 0x33A: inst.opcode = Opcode::V_ASHRREV_I16; break;
            case 0x33C: inst.opcode = Opcode::V_LSHLREV_B64; break;
            case 0x33D: inst.opcode = Opcode::V_LSHRREV_B64; break;
            case 0x33E: inst.opcode = Opcode::V_ASHRREV_I64; break;
            case 0x360:
                inst.opcode = Opcode::V_READLANE_B32;
                inst.dst    = decode_sdst(vdst, true);
                break;
            case 0x361: inst.opcode = Opcode::V_WRITELANE_B32; break;
            case 0x369: inst.opcode = Opcode::V_CVT_PK_FP8_F32; break;
            case 0x36A: inst.opcode = Opcode::V_CVT_PK_BF8_F32; break;
            default:    inst.opcode = Opcode::UNKNOWN; break;
        }
        return inst;
    }

    // 12. DS: top6 == 0x36 (0xD8000000..0xDBFFFFFF)
    if (top6 == 0x36) {
        inst.format = InstFormat::DS;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        uint32_t op = (w0 >> 18) & 0xFF;
        inst.ds_gds = (w0 >> 17) & 0x1;
        inst.opcode = ds_opcode(op, true);
        bool is_dual = false;
        switch (inst.opcode) {
            case Opcode::DS_WRITE2_B32:
            case Opcode::DS_WRITE2ST64_B32:
            case Opcode::DS_READ2_B32:
            case Opcode::DS_READ2ST64_B32:
            case Opcode::DS_WRITE2_B64:
            case Opcode::DS_WRITE2ST64_B64:
            case Opcode::DS_READ2_B64:
            case Opcode::DS_READ2ST64_B64:
                is_dual = true;
                break;
            default: break;
        }
        if (is_dual) {
            inst.ds_offset0 = w0 & 0xFF;
            inst.ds_offset1 = (w0 >> 8) & 0xFF;
        } else {
            inst.ds_offset0 = w0 & 0xFFFF;
            inst.ds_offset1 = 0;
        }
        inst.src[0] = decode_vdst(w1 & 0xFF);
        inst.src[1] = decode_vdst((w1 >> 8) & 0xFF);
        inst.src[2] = decode_vdst((w1 >> 16) & 0xFF);
        inst.dst    = decode_vdst((w1 >> 24) & 0xFF);
        return inst;
    }

    // 13. 32-bit VOP1: [31:25] == 0x3F (0x7E000000..0x7FFFFFFF)
    if ((w0 >> 25) == 0x3F) {
        inst.format = InstFormat::VOP1;
        uint32_t op   = (w0 >> 9) & 0xFF;
        uint32_t vdst = (w0 >> 17) & 0xFF;
        uint32_t src0 = w0 & 0x1FF;
        inst.opcode = decode_gfx12_vop1_op(op);
        if (inst.opcode == Opcode::V_READFIRSTLANE_B32) {
            inst.dst = decode_sdst(vdst, true);
        } else {
            inst.dst = decode_vdst(vdst);
        }
        bool has_lit = (src0 == 255);
        const uint8_t* lit_ptr = has_lit ? (pc + 4) : nullptr;
        inst.size_bytes = has_lit ? 8 : 4;
        if (has_lit) inst.encoding[1] = read_u32_le(pc + 4);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        return inst;
    }

    // 14. 32-bit VOPC: [31:25] == 0x3E (0x7C000000..0x7DFFFFFF)
    if ((w0 >> 25) == 0x3E) {
        inst.format = InstFormat::VOPC;
        bool is_cmpx = ((w0 >> 24) & 1) != 0;
        uint32_t op   = (w0 >> 17) & 0x7F;
        uint32_t src0 = w0 & 0x1FF;
        uint32_t vsrc1 = (w0 >> 9) & 0xFF;
        inst.opcode = decode_gfx12_vopc_op(op, is_cmpx);
        inst.dst.kind = is_cmpx ? Operand::Kind::EXEC_LO : Operand::Kind::VCC_LO;
        bool has_lit = (src0 == 255);
        const uint8_t* lit_ptr = has_lit ? (pc + 4) : nullptr;
        inst.size_bytes = has_lit ? 8 : 4;
        if (has_lit) inst.encoding[1] = read_u32_le(pc + 4);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        inst.src[1] = decode_vdst(vsrc1);
        return inst;
    }

    // 15. 32-bit VOP2: [31] == 0 (0x00000000..0x7BFFFFFF)
    if ((w0 >> 31) == 0) {
        inst.format = InstFormat::VOP2;
        uint32_t op    = (w0 >> 25) & 0x3F;
        uint32_t vdst  = (w0 >> 17) & 0xFF;
        uint32_t vsrc1 = (w0 >> 9) & 0xFF;
        uint32_t src0  = w0 & 0x1FF;
        inst.opcode = decode_gfx12_vop2_op(op);
        inst.dst    = decode_vdst(vdst);
        bool is_fma_lit = (op == 0x2C || op == 0x2D); // v_fmamk_f32 / v_fmaak_f32
        bool has_lit = (src0 == 255) || is_fma_lit;
        const uint8_t* lit_ptr = has_lit ? (pc + 4) : nullptr;
        inst.size_bytes = has_lit ? 8 : 4;
        if (has_lit) inst.encoding[1] = read_u32_le(pc + 4);
        inst.src[0] = decode_src(src0, lit_ptr, true);
        inst.src[1] = decode_vdst(vsrc1);
        if (is_fma_lit) {
            inst.src[2].kind = Operand::Kind::LITERAL;
            inst.src[2].imm  = inst.encoding[1];
        } else if (inst.opcode == Opcode::V_CNDMASK_B32 ||
                   inst.opcode == Opcode::V_ADDC_CO_U32 ||
                   inst.opcode == Opcode::V_SUBB_CO_U32 ||
                   inst.opcode == Opcode::V_SUBBREV_CO_U32) {
            inst.src[2].kind = Operand::Kind::VCC_LO;
            if (inst.opcode != Opcode::V_CNDMASK_B32) {
                inst.dst2.kind = Operand::Kind::VCC_LO;
            }
        }
        return inst;
    }

    inst.format = InstFormat::UNKNOWN;
    inst.opcode = Opcode::UNKNOWN;
    inst.size_bytes = 4;
    return inst;
}

// ---------------------------------------------------------------------------
// Main decode function
// ---------------------------------------------------------------------------

Instruction decode(const uint8_t* pc, bool is_gfx950, bool is_gfx908, bool is_gfx1201) {
    if (is_gfx1201) {
        return decode_gfx12(pc);
    }
    Instruction inst;
    uint32_t w0 = read_u32_le(pc);
    inst.encoding[0] = w0;

    uint32_t bits31_23 = (w0 >> 23) & 0x1FF;
    uint32_t bits31_26 = (w0 >> 26) & 0x3F;
    uint32_t bits31_28 = (w0 >> 28) & 0xF;
    uint32_t bits31_25 = (w0 >> 25) & 0x7F;
    uint32_t bits31_30 = (w0 >> 30) & 0x3;

    // --- SOPP: [31:23] == 0x17F ---
    if (bits31_23 == 0x17F) {
        inst.format = InstFormat::SOPP;
        inst.size_bytes = 4;
        uint32_t op = (w0 >> 16) & 0x7F;
        inst.simm16 = (int16_t)(w0 & 0xFFFF);
        inst.opcode = sopp_opcode(op);
        if (inst.opcode == Opcode::S_WAITCNT) {
            parse_waitcnt((uint16_t)(w0 & 0xFFFF), inst.wait_vmcnt, inst.wait_lgkmcnt);
        }
        return inst;
    }

    // --- SOP1: [31:23] == 0x17D ---
    // SOP1 encoding: bits[31:24]=0xBE, bit[23]=1 → bits[31:23]=0x17D
    if (bits31_23 == 0x17D) {
        inst.format = InstFormat::SOP1;
        inst.size_bytes = 4;
        uint32_t sdst  = (w0 >> 16) & 0x7F;
        uint32_t op    = (w0 >> 8) & 0xFF;
        uint32_t ssrc0 = w0 & 0xFF;
        inst.dst = decode_sdst(sdst);
        bool has_literal = (ssrc0 == 255);
        inst.src[0] = decode_src(ssrc0, has_literal ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;
        inst.opcode = sop1_opcode(op);
        return inst;
    }

    // --- SOPC: [31:23] == 0x17E ---
    // SOPC encoding: bits[31:24]=0xBF, bit[23]=0 → bits[31:23]=0x17E
    if (bits31_23 == 0x17E) {
        inst.format = InstFormat::SOPC;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 16) & 0x7F;
        uint32_t ssrc1 = (w0 >> 8) & 0xFF;
        uint32_t ssrc0 = w0 & 0xFF;
        inst.opcode = sopc_opcode(op);
        // SOPC writes SCC only; no dst operand
        inst.dst.kind = Operand::Kind::SCC;
        bool has_literal = (ssrc0 == 255 || ssrc1 == 255);
        inst.src[0] = decode_src(ssrc0, (ssrc0 == 255) ? (pc + 4) : nullptr);
        inst.src[1] = decode_src(ssrc1, (ssrc1 == 255) ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;
        return inst;
    }

    // --- SOPK: [31:28] == 0xB (1011) ---
    if (bits31_28 == 0xB) {
        inst.format = InstFormat::SOPK;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 23) & 0x1F;
        uint32_t sdst  = (w0 >> 16) & 0x7F;
        inst.simm16    = (int16_t)(w0 & 0xFFFF);
        inst.opcode    = sopk_opcode(op);
        inst.dst       = decode_sdst(sdst);
        // S_SETREG_IMM32_B32 has a literal word following
        if (inst.opcode == Opcode::S_SETREG_IMM32_B32) inst.size_bytes = 8;
        return inst;
    }

    // --- SMEM: [31:26] == 0x30 ---
    if (bits31_26 == 0x30) {
        inst.format = InstFormat::SMEM;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op       = (w0 >> 18) & 0xFF;
        uint32_t sdst     = (w0 >> 6)  & 0x7F;
        uint32_t sbase    = w0 & 0x3F;
        uint32_t imm_flag = (w0 >> 17) & 1;

        inst.opcode = smem_opcode(op);
        inst.dst.kind  = Operand::Kind::SGPR;
        inst.dst.index = sdst;
        inst.src[0].kind  = Operand::Kind::SGPR;
        inst.src[0].index = sbase * 2;

        if (imm_flag) {
            inst.smem_offset = (int32_t)w1;
        } else {
            inst.smem_offset = 0;
            inst.src[1].kind = Operand::Kind::SGPR;
            inst.src[1].index = w1 & 0x7F;
        }
        return inst;
    }

    // --- DS: [31:26] == 0x36 ---
    if (bits31_26 == 0x36) {
        inst.format = InstFormat::DS;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op      = (w0 >> 17) & 0xFF;
        inst.ds_gds      = (w0 >> 16) & 1;

        inst.opcode = ds_opcode(op);

        // Dual operations (DS_READ2/WRITE2) use two separate 8-bit offsets.
        // All other DS operations use a combined 16-bit offset in ds_offset0.
        bool is_dual = false;
        switch (inst.opcode) {
            case Opcode::DS_WRITE2_B32:
            case Opcode::DS_WRITE2ST64_B32:
            case Opcode::DS_READ2_B32:
            case Opcode::DS_READ2ST64_B32:
            case Opcode::DS_WRITE2_B64:
            case Opcode::DS_WRITE2ST64_B64:
            case Opcode::DS_READ2_B64:
            case Opcode::DS_READ2ST64_B64:
                is_dual = true;
                break;
            default: break;
        }

        if (is_dual) {
            inst.ds_offset0 = w0 & 0xFF;
            inst.ds_offset1 = (w0 >> 8) & 0xFF;
        } else {
            // Combined 16-bit offset
            inst.ds_offset0 = w0 & 0xFFFF;
            inst.ds_offset1 = 0;
        }

        uint32_t vdst  = (w1 >> 24) & 0xFF;
        uint32_t data1 = (w1 >> 16) & 0xFF;
        uint32_t data0 = (w1 >> 8)  & 0xFF;
        uint32_t addr  = w1 & 0xFF;

        // src[0] = ADDR (VGPR), src[1] = DATA0, src[2] = DATA1
        inst.src[0].kind  = Operand::Kind::VGPR;
        inst.src[0].index = addr;
        inst.src[1].kind  = Operand::Kind::VGPR;
        inst.src[1].index = data0;
        inst.src[2].kind  = Operand::Kind::VGPR;
        inst.src[2].index = data1;
        inst.dst.kind     = Operand::Kind::VGPR;
        inst.dst.index    = vdst;
        return inst;
    }

    // --- FLAT/GLOBAL/SCRATCH: [31:26] == 0x37 ---
    if (bits31_26 == 0x37) {
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        inst.size_bytes = 8;

        uint32_t op    = (w0 >> 18) & 0xFF;
        // SEG is at bits[15:14]: 00=flat, 01=scratch, 10=global.
        // (Not [17:16] — those bits are always 0 for all variants on GFX9/GFX90a.)
        uint32_t seg   = (w0 >> 14) & 0x3;

        bool is_scratch = (seg == 1);
        bool is_global  = (seg == 2);

        uint32_t vdst  = (w1 >> 24) & 0xFF;
        uint32_t saddr = (w1 >> 16) & 0xFF;
        uint32_t vdata = (w1 >> 8)  & 0xFF;
        uint32_t vaddr = (w1 >> 0)  & 0xFF;

        inst.opcode = flat_opcode(op, is_global, is_scratch);

        if (is_scratch) {
            inst.format = InstFormat::FLAT_SCRATCH;
        } else {
            inst.format = InstFormat::FLAT_GLOBAL;
        }

        bool is_store = false;
        switch (inst.opcode) {
            case Opcode::GLOBAL_STORE_BYTE:
            case Opcode::GLOBAL_STORE_SHORT:
            case Opcode::GLOBAL_STORE_DWORD:
            case Opcode::GLOBAL_STORE_DWORDX2:
            case Opcode::GLOBAL_STORE_DWORDX3:
            case Opcode::GLOBAL_STORE_DWORDX4:
            case Opcode::FLAT_STORE_BYTE:
            case Opcode::FLAT_STORE_SHORT:
            case Opcode::FLAT_STORE_DWORD:
            case Opcode::FLAT_STORE_DWORDX2:
            case Opcode::FLAT_STORE_DWORDX3:
            case Opcode::FLAT_STORE_DWORDX4:
                is_store = true;
                break;
            default: break;
        }

        bool is_atomic = false;
        switch (inst.opcode) {
            case Opcode::GLOBAL_ATOMIC_SWAP:
            case Opcode::GLOBAL_ATOMIC_CMPSWAP:
            case Opcode::GLOBAL_ATOMIC_ADD:
            case Opcode::GLOBAL_ATOMIC_SUB:
            case Opcode::GLOBAL_ATOMIC_SMIN:
            case Opcode::GLOBAL_ATOMIC_UMIN:
            case Opcode::GLOBAL_ATOMIC_SMAX:
            case Opcode::GLOBAL_ATOMIC_UMAX:
            case Opcode::GLOBAL_ATOMIC_AND:
            case Opcode::GLOBAL_ATOMIC_OR:
            case Opcode::GLOBAL_ATOMIC_XOR:
            case Opcode::GLOBAL_ATOMIC_INC:
            case Opcode::GLOBAL_ATOMIC_DEC:
            case Opcode::GLOBAL_ATOMIC_ADD_F32:
            case Opcode::GLOBAL_ATOMIC_ADD_F64:
            case Opcode::GLOBAL_ATOMIC_ADD_X2:
                is_atomic = true;
                break;
            default: break;
        }

        if (is_atomic) {
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr;
            }
            inst.src[2].kind  = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
            bool glc = (w0 >> 16) & 1;
            if (glc) {
                inst.dst.kind  = Operand::Kind::VGPR;
                inst.dst.index = vdst;
            } else {
                inst.dst.kind  = Operand::Kind::NONE;
            }
        } else if (!is_store) {
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdst;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr;
            }
        } else {
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr;
            }
            inst.src[2].kind  = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
        }

        inst.flat_offset = (int16_t)(w0 & 0x1FFF);
        return inst;
    }

    // --- MUBUF: [31:26] == 0x38 ---
    if (bits31_26 == 0x38) {
        inst.format = InstFormat::MUBUF;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op    = (w0 >> 18) & 0x7F;
        inst.mubuf_offset = w0 & 0xFFF;
        inst.mubuf_offen  = (w0 >> 12) & 1;
        inst.mubuf_idxen  = (w0 >> 13) & 1;
        inst.mubuf_lds    = (w0 >> 16) & 1;

        // GFX9 MUBUF word1: [31:24]=SOFFSET, [20:16]=SRSRC, [15:8]=VDATA, [7:0]=VADDR
        uint32_t soffset = (w1 >> 24) & 0xFF;
        uint32_t srsrc   = (w1 >> 16) & 0x1F;  // 5-bit, actual = srsrc * 4
        uint32_t vdata   = (w1 >> 8)  & 0xFF;
        uint32_t vaddr   = w1 & 0xFF;

        inst.opcode = mubuf_opcode(op);

        bool is_store = false;
        switch (inst.opcode) {
            case Opcode::BUFFER_STORE_BYTE:
            case Opcode::BUFFER_STORE_SHORT:
            case Opcode::BUFFER_STORE_SHORT_D16_HI:
            case Opcode::BUFFER_STORE_DWORD:
            case Opcode::BUFFER_STORE_DWORDX2:
            case Opcode::BUFFER_STORE_DWORDX3:
            case Opcode::BUFFER_STORE_DWORDX4:
                is_store = true;
                break;
            default: break;
        }

        // src[0] = VADDR, src[1] = SRSRC (buffer descriptor base)
        // mubuf_soffset = SOFFSET operand (SGPR scalar offset)
        inst.src[0].kind  = Operand::Kind::VGPR;
        inst.src[0].index = vaddr;
        inst.src[1].kind  = Operand::Kind::SGPR;
        inst.src[1].index = srsrc * 4;

        // Decode SOFFSET: 0-103 = SGPR, 128 = zero (inline const 0)
        if (soffset <= 103) {
            inst.mubuf_soffset.kind  = Operand::Kind::SGPR;
            inst.mubuf_soffset.index = soffset;
        } else {
            inst.mubuf_soffset.kind  = Operand::Kind::INLINE_CONST;
            inst.mubuf_soffset.imm   = 0;
        }

        if (is_store) {
            inst.dst.kind = Operand::Kind::NONE;
            // src[2] = source data VGPRs
            inst.src[2].kind  = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
        } else {
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdata;
        }
        return inst;
    }

    // --- MTBUF: [31:26] == 0x3A ---
    if (bits31_26 == 0x3A) {
        inst.format = InstFormat::MTBUF;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op = (w0 >> 15) & 0x7;
        inst.opcode = mtbuf_opcode(op);

        // GFX9 MTBUF word1: [31:24]=SOFFSET, [20:16]=SRSRC, [15:8]=VDATA, [7:0]=VADDR
        uint32_t soffset_mt = (w1 >> 24) & 0xFF;
        uint32_t srsrc   = (w1 >> 16) & 0x1F;
        uint32_t vdata   = (w1 >> 8)  & 0xFF;
        uint32_t vaddr   = w1 & 0xFF;

        inst.src[0].kind  = Operand::Kind::VGPR;
        inst.src[0].index = vaddr;
        inst.src[1].kind  = Operand::Kind::SGPR;
        inst.src[1].index = srsrc * 4;
        if (soffset_mt <= 103) {
            inst.mubuf_soffset.kind  = Operand::Kind::SGPR;
            inst.mubuf_soffset.index = soffset_mt;
        } else {
            inst.mubuf_soffset.kind  = Operand::Kind::INLINE_CONST;
            inst.mubuf_soffset.imm   = 0;
        }
        inst.dst.kind     = Operand::Kind::VGPR;
        inst.dst.index    = vdata;
        return inst;
    }

    // --- VOP3P: [31:26] == 0x33 ---
    if (bits31_26 == 0x33) {
        inst.format = InstFormat::VOP3P;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op   = (w0 >> 16) & 0x3F;
        uint32_t vdst = w0 & 0xFF;

        uint32_t src0 = w1 & 0x1FF;
        uint32_t src1 = (w1 >> 9) & 0x1FF;
        uint32_t src2 = (w1 >> 18) & 0x1FF;

        inst.opcode = vop3p_opcode(op);
        inst.dst    = decode_vdst(vdst);
        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1] = decode_src(src1, nullptr);
        inst.src[2] = decode_src(src2, nullptr);

        // VOP3P modifiers
        // op_sel: bits[14:11] of word0 — select hi(1) vs lo(0) for src0/src1/src2/dst
        inst.op_sel    = (w0 >> 11) & 0xF;
        // op_sel_hi: bits[15,14,13] but in VOP3P these are [15:13] — actually
        // VOP3P has op_sel_hi at bits[15:14] of w0 for src2/src1, bit[11] for src0
        // Simplified: extract raw bits for use by handlers
        inst.op_sel_hi = ((w0 >> 13) & 0x1) | (((w0 >> 14) & 0x1) << 1) | (((w0 >> 15) & 0x1) << 2);
        // neg_lo: bits[31:29] of w1
        inst.neg       = (w1 >> 29) & 0x7;
        // neg_hi: bits[28:26] of... actually VOP3P neg_hi is w0 bits[2:0] at a different position
        // For GFX9 VOP3P: neg_lo = w1[31:29], neg_hi = w1[28:26] — no, let me check.
        // Actually: neg fields for VOP3P use the same positions as VOP3A.
        // VOP3P w0: [15]=op_sel_hi[2], [14]=op_sel_hi[1], [13]=op_sel_hi[0],
        //           [12:11]=op_sel[1:0], bit[10]=op_sel[2], bit[9:8]=clamp(unused)+abs(unused)
        // Simplified for now: neg from w1[31:29], neg_hi from w0[2:0] — actually different.
        // The key VOP3P modifiers are op_sel. neg/neg_hi are less common.
        // Leave neg_hi at 0 for now since we already extract neg from w1.
        return inst;
    }

    // --- MFMA: [31:26] == 0x34 with specific MFMA op range ---
    // VOP3A/MFMA share [31:26]==0x34. MFMA ops are identified by op >= 0x200
    // in a separate namespace. Actual detection: check op bits for MFMA range.
    // MFMA encoding in gfx90a: [31:26]=110100, [25:18]=op(8-bit MFMA ops)
    // These don't overlap with standard VOP3 ops because MFMA ops start at 0x200
    // when treated as 10-bit. We detect MFMA by checking bit[15:14] after VOP3.

    // --- VOP3A/VOP3B: [31:26] == 0x34 ---
    if (bits31_26 == 0x34) {
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        inst.size_bytes = 8;

        uint32_t op   = (w0 >> 16) & 0x3FF;  // 10-bit opcode
        uint32_t vdst = w0 & 0xFF;
        uint32_t sdst = (w0 >> 8) & 0x7F;

        uint32_t src0 = w1 & 0x1FF;
        uint32_t src1 = (w1 >> 9) & 0x1FF;
        uint32_t src2 = (w1 >> 18) & 0x1FF;

        // Check for MFMA: op field is the 10-bit VOP3 op field [25:16].
        // MFMA instructions use op >= 960 (0x3C0) in the 10-bit field.
        // mfma_opcode() table uses 0-based MFMA op indices (op - 960).
        // v_pk_mov_b32 (VOP3P-MAI, op=0x3B3=947) is encoded with [31:26]=0x34.
        Opcode oc = vop3_opcode(op);
        if (oc == Opcode::UNKNOWN) {
            // Try MFMA: MFMA ops occupy op range 940..1023
            if (op >= 940 && op <= 1023) {
                Opcode mc = mfma_opcode(op, is_gfx950);
                if (mc == Opcode::UNKNOWN && op >= 960) {
                    mc = mfma_opcode(op - 960, is_gfx950);
                }
                if (mc != Opcode::UNKNOWN) {
                    inst.format = InstFormat::MFMA;
                    inst.opcode = mc;
                    // GFX908 always targets AGPR; GFX90a: bit 15 of word 0 = "acc" flag.
                    // 0 = accumulator is VGPR (HIP compiler path, vdst = VGPR index).
                    // 1 = accumulator is AGPR (assembly path, vdst = AGPR index).
                    bool is_agpr = is_gfx908 || ((w0 >> 15) & 1);
                    inst.dst.kind  = is_agpr ? Operand::Kind::AGPR : Operand::Kind::VGPR;
                    inst.dst.index = vdst;
                    inst.src[0] = decode_src(src0, nullptr);
                    inst.src[1] = decode_src(src1, nullptr);
                    inst.src[2] = decode_src(src2, nullptr);
                    return inst;
                }
            }
        }

        inst.opcode = oc;

        bool is_vop3b = (inst.opcode == Opcode::V_ADD_CO_U32 ||
                         inst.opcode == Opcode::V_ADDC_CO_U32 ||
                         inst.opcode == Opcode::V_SUB_CO_U32 ||
                         inst.opcode == Opcode::V_SUBREV_CO_U32 ||
                         inst.opcode == Opcode::V_MAD_U64_U32 ||
                         inst.opcode == Opcode::V_MAD_I64_I32 ||
                         inst.opcode == Opcode::V_DIV_FMAS_F32 ||
                         inst.opcode == Opcode::V_DIV_FMAS_F64);

        if (is_vop3b) {
            inst.format = InstFormat::VOP3B;
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdst;
            inst.dst2 = decode_sdst(sdst);
        } else if (vop3_vopc_opcode(op) != Opcode::UNKNOWN) {
            // VOPC ops encoded in VOP3A: 8-bit VOPC op in the 10-bit op field.
            // The destination is an SGPR pair encoded in the vdst field (or EXEC for cmpx).
            inst.format = InstFormat::VOP3A;
            inst.opcode = vop3_vopc_opcode(op);
            if (is_vop3_vopc_cmpx(op)) {
                inst.dst.kind = Operand::Kind::EXEC;
            } else {
                inst.dst = decode_sdst(vdst);
            }
        } else {
            inst.format = InstFormat::VOP3A;
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdst;
        }

        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1] = decode_src(src1, nullptr);
        inst.src[2] = decode_src(src2, nullptr);

        // Extract VOP3 source/destination modifiers
        // Note: bit[15] is CLAMP for VOP3A but "acc" flag for MFMA — skip for MFMA
        inst.clamp = (w0 >> 15) & 1;   // word0 bit 15
        inst.abs_  = (w0 >> 8) & 0x7;  // word0 bits [10:8]
        inst.neg   = (w1 >> 29) & 0x7; // word1 bits [31:29]
        inst.omod  = (w1 >> 27) & 0x3; // word1 bits [28:27]

        // CDNA packed FP32/B32 op_sel and op_sel_hi
        if (inst.opcode == Opcode::V_PK_FMA_F32 ||
            inst.opcode == Opcode::V_PK_MUL_F32 ||
            inst.opcode == Opcode::V_PK_ADD_F32 ||
            inst.opcode == Opcode::V_PK_MOV_B32 ||
            inst.opcode == Opcode::V_PACK_B32_F16) {
            inst.op_sel    = (w0 >> 11) & 0x7;
            inst.op_sel_hi = ((w1 >> 27) & 0x3) | (((w0 >> 14) & 0x1) << 2);
        }

        // GFX90a AGPR ops encoded in VOP3A:
        // v_accvgpr_write_b32: dst is AGPR, src0 is VGPR or inline const
        if (inst.opcode == Opcode::V_ACCVGPR_WRITE_B32) {
            inst.dst.kind = Operand::Kind::AGPR;
        }
        // v_accvgpr_read_b32: dst is VGPR, src0 is AGPR
        if (inst.opcode == Opcode::V_ACCVGPR_READ_B32) {
            if (inst.src[0].kind == Operand::Kind::VGPR) {
                inst.src[0].kind = Operand::Kind::AGPR;
            }
        }
        return inst;
    }

    // --- VOP1: [31:25] == 0x3F ---
    if (bits31_25 == 0x3F) {
        inst.format = InstFormat::VOP1;
        inst.size_bytes = 4;
        uint32_t vdst = (w0 >> 17) & 0xFF;
        uint32_t op   = (w0 >> 9)  & 0xFF;
        uint32_t src0 = w0 & 0x1FF;

        inst.opcode = vop1_opcode(op);
        if (inst.opcode == Opcode::V_READFIRSTLANE_B32) {
            inst.dst = decode_sdst(vdst & 0x7F);
        } else {
            inst.dst = decode_vdst(vdst);
        }

        // Check for SDWA/DPP sub-encodings
        if ((src0 & 0xFF) == 0xF9) {
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_sdwa = true;
            uint32_t real_src0 = w1 & 0xFF;
            // SDWA src0 field is always a VGPR index (0-255)
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = real_src0;
            inst.sdwa_src0_sel  = (w1 >> 8) & 0x7;
            inst.sdwa_src0_sext = (w1 >> 11) & 1;
            inst.sdwa_src0_neg  = (w1 >> 12) & 1;
            inst.sdwa_src0_abs  = (w1 >> 13) & 1;
            inst.sdwa_dst_sel   = (w1 >> 24) & 0x7;
            inst.sdwa_dst_unused = (w1 >> 27) & 0x3;
            inst.clamp          = (w1 >> 29) & 1;
        } else if ((src0 & 0xFF) == 0xFA) {
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_dpp = true;
            uint32_t real_src0 = w1 & 0xFF;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = real_src0;
            inst.dpp_ctrl       = (w1 >> 8) & 0x1FF;
            inst.dpp_bound_ctrl = (w1 >> 19) & 1;
            inst.dpp_src0_neg   = (w1 >> 20) & 1;
            inst.dpp_src0_abs   = (w1 >> 21) & 1;
            inst.dpp_src1_neg   = (w1 >> 22) & 1;
            inst.dpp_src1_abs   = (w1 >> 23) & 1;
            inst.dpp_bank_mask  = (w1 >> 24) & 0xF;
            inst.dpp_row_mask   = (w1 >> 28) & 0xF;
        } else {
            bool has_literal = (src0 == 255);
            inst.src[0] = decode_src(src0, has_literal ? (pc + 4) : nullptr);
            if (has_literal) inst.size_bytes = 8;
        }

        // accvgpr_write: dst is AGPR
        if (inst.opcode == Opcode::V_ACCVGPR_WRITE_B32) {
            inst.dst.kind = Operand::Kind::AGPR;
        }
        // accvgpr_read: src0 is AGPR
        if (inst.opcode == Opcode::V_ACCVGPR_READ_B32) {
            if (inst.src[0].kind == Operand::Kind::VGPR) {
                inst.src[0].kind = Operand::Kind::AGPR;
            }
        }
        return inst;
    }

    // --- VOPC: [31:25] == 0x3E ---
    if (bits31_25 == 0x3E) {
        inst.format = InstFormat::VOPC;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 17) & 0xFF;
        uint32_t vsrc1 = (w0 >> 9) & 0xFF;
        uint32_t src0  = w0 & 0x1FF;

        inst.opcode = vopc_opcode(op);
        inst.dst.kind = is_vopc_cmpx(op) ? Operand::Kind::EXEC : Operand::Kind::VCC;

        // Check for SDWA/DPP sub-encodings
        if ((src0 & 0xFF) == 0xF9) {
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_sdwa = true;
            uint32_t real_src0 = w1 & 0xFF;
            // SDWA src0 field is always a VGPR index (0-255)
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = real_src0;
            inst.src[1].kind  = Operand::Kind::VGPR;
            inst.src[1].index = vsrc1;
            inst.sdwa_src0_sel  = (w1 >> 8) & 0x7;
            inst.sdwa_src0_sext = (w1 >> 11) & 1;
            inst.sdwa_src0_neg  = (w1 >> 12) & 1;
            inst.sdwa_src0_abs  = (w1 >> 13) & 1;
            inst.sdwa_src1_sel  = (w1 >> 16) & 0x7;
            inst.sdwa_src1_sext = (w1 >> 19) & 1;
            inst.sdwa_src1_neg  = (w1 >> 20) & 1;
            inst.sdwa_src1_abs  = (w1 >> 21) & 1;
        } else if ((src0 & 0xFF) == 0xFA) {
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_dpp = true;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = w1 & 0xFF;
            inst.src[1].kind  = Operand::Kind::VGPR;
            inst.src[1].index = vsrc1;
            inst.dpp_ctrl       = (w1 >> 8) & 0x1FF;
            inst.dpp_bound_ctrl = (w1 >> 19) & 1;
            inst.dpp_src0_neg   = (w1 >> 20) & 1;
            inst.dpp_src0_abs   = (w1 >> 21) & 1;
            inst.dpp_src1_neg   = (w1 >> 22) & 1;
            inst.dpp_src1_abs   = (w1 >> 23) & 1;
            inst.dpp_bank_mask  = (w1 >> 24) & 0xF;
            inst.dpp_row_mask   = (w1 >> 28) & 0xF;
        } else {
            inst.src[0] = decode_src(src0, nullptr);
            inst.src[1].kind  = Operand::Kind::VGPR;
            inst.src[1].index = vsrc1;
        }
        return inst;
    }

    // --- SOP2: [31:30] == 0b10 (and not already handled) ---
    if (bits31_30 == 0x2) {
        inst.format = InstFormat::SOP2;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 23) & 0x7F;
        uint32_t sdst  = (w0 >> 16) & 0x7F;
        uint32_t ssrc1 = (w0 >> 8) & 0xFF;
        uint32_t ssrc0 = w0 & 0xFF;

        inst.opcode = sop2_opcode(op);
        inst.dst    = decode_sdst(sdst);

        bool has_literal = (ssrc0 == 255 || ssrc1 == 255);
        inst.src[0] = decode_src(ssrc0, (ssrc0 == 255) ? (pc + 4) : nullptr);
        inst.src[1] = decode_src(ssrc1, (ssrc1 == 255) ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;
        return inst;
    }

    // --- VOP2: fallback (bit 31 == 0) ---
    if ((w0 >> 31) == 0) {
        inst.format = InstFormat::VOP2;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 25) & 0x3F;
        uint32_t vdst  = (w0 >> 17) & 0xFF;
        uint32_t vsrc1 = (w0 >> 9)  & 0xFF;
        uint32_t src0  = w0 & 0x1FF;

        inst.opcode = vop2_opcode(op);
        inst.dst    = decode_vdst(vdst);

        // Check for SDWA/DPP sub-encodings
        if ((src0 & 0xFF) == 0xF9) {
            // SDWA: word1 contains SDWA fields, instruction is 8 bytes
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_sdwa = true;
            uint32_t real_src0 = w1 & 0xFF;
            // SDWA src0 field is always a VGPR index (0-255)
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = real_src0;
            inst.src[1].kind  = Operand::Kind::VGPR;
            inst.src[1].index = vsrc1;
            // SDWA word1 layout for VOP2:
            // [7:0]   = SRC0 (VGPR index)
            // [10:8]  = SRC0_SEL
            // [11]    = SRC0_SEXT
            // [12]    = SRC0_NEG
            // [13]    = SRC0_ABS
            // [18:16] = SRC1_SEL
            // [19]    = SRC1_SEXT
            // [20]    = SRC1_NEG
            // [21]    = SRC1_ABS
            // [26:24] = DST_SEL
            // [28:27] = DST_UNUSED
            // [31:29] = unused / clamp
            inst.sdwa_src0_sel  = (w1 >> 8) & 0x7;
            inst.sdwa_src0_sext = (w1 >> 11) & 1;
            inst.sdwa_src0_neg  = (w1 >> 12) & 1;
            inst.sdwa_src0_abs  = (w1 >> 13) & 1;
            inst.sdwa_src1_sel  = (w1 >> 16) & 0x7;
            inst.sdwa_src1_sext = (w1 >> 19) & 1;
            inst.sdwa_src1_neg  = (w1 >> 20) & 1;
            inst.sdwa_src1_abs  = (w1 >> 21) & 1;
            inst.sdwa_dst_sel   = (w1 >> 24) & 0x7;
            inst.sdwa_dst_unused = (w1 >> 27) & 0x3;
            inst.clamp          = (w1 >> 29) & 1;
            return inst;
        }
        if ((src0 & 0xFF) == 0xFA) {
            // DPP: word1 contains DPP fields, instruction is 8 bytes
            uint32_t w1 = read_u32_le(pc + 4);
            inst.encoding[1] = w1;
            inst.size_bytes = 8;
            inst.is_dpp = true;
            uint32_t real_src0 = w1 & 0xFF;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = real_src0;
            inst.src[1].kind  = Operand::Kind::VGPR;
            inst.src[1].index = vsrc1;
            // DPP word1 layout:
            // [7:0]   = SRC0 (VGPR)
            // [16:8]  = DPP_CTRL
            // [18:17] = reserved
            // [19]    = BOUND_CTRL
            // [20]    = SRC0_NEG
            // [21]    = SRC0_ABS
            // [22]    = SRC1_NEG
            // [23]    = SRC1_ABS
            // [27:24] = BANK_MASK
            // [31:28] = ROW_MASK
            inst.dpp_ctrl       = (w1 >> 8) & 0x1FF;
            inst.dpp_bound_ctrl = (w1 >> 19) & 1;
            inst.dpp_src0_neg   = (w1 >> 20) & 1;
            inst.dpp_src0_abs   = (w1 >> 21) & 1;
            inst.dpp_src1_neg   = (w1 >> 22) & 1;
            inst.dpp_src1_abs   = (w1 >> 23) & 1;
            inst.dpp_bank_mask  = (w1 >> 24) & 0xF;
            inst.dpp_row_mask   = (w1 >> 28) & 0xF;
            return inst;
        }

        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1].kind  = Operand::Kind::VGPR;
        inst.src[1].index = vsrc1;

        bool has_literal = (src0 == 255);
        if (has_literal) {
            inst.src[0] = decode_src(src0, pc + 4);
            inst.size_bytes = 8;
        }
        return inst;
    }

    inst.format = InstFormat::UNKNOWN;
    inst.opcode = Opcode::UNKNOWN;
    inst.size_bytes = 4;
    return inst;
}

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------

const char* opcode_name(Opcode op) {
    switch (op) {
        case Opcode::S_ADD_U32:              return "s_add_u32";
        case Opcode::S_ADD_I32:              return "s_add_i32";
        case Opcode::S_ADDC_U32:             return "s_addc_u32";
        case Opcode::S_SUB_U32:              return "s_sub_u32";
        case Opcode::S_SUBB_U32:             return "s_subb_u32";
        case Opcode::S_MIN_I32:              return "s_min_i32";
        case Opcode::S_MIN_U32:              return "s_min_u32";
        case Opcode::S_MAX_I32:              return "s_max_i32";
        case Opcode::S_MAX_U32:              return "s_max_u32";
        case Opcode::S_AND_B32:              return "s_and_b32";
        case Opcode::S_AND_B64:              return "s_and_b64";
        case Opcode::S_OR_B32:               return "s_or_b32";
        case Opcode::S_OR_B64:               return "s_or_b64";
        case Opcode::S_XOR_B32:              return "s_xor_b32";
        case Opcode::S_XOR_B64:              return "s_xor_b64";
        case Opcode::S_ANDN2_B32:            return "s_andn2_b32";
        case Opcode::S_ANDN2_B64:            return "s_andn2_b64";
        case Opcode::S_ORN2_B32:             return "s_orn2_b32";
        case Opcode::S_ORN2_B64:             return "s_orn2_b64";
        case Opcode::S_NAND_B32:             return "s_nand_b32";
        case Opcode::S_NAND_B64:             return "s_nand_b64";
        case Opcode::S_NOR_B32:              return "s_nor_b32";
        case Opcode::S_NOR_B64:              return "s_nor_b64";
        case Opcode::S_XNOR_B32:             return "s_xnor_b32";
        case Opcode::S_XNOR_B64:             return "s_xnor_b64";
        case Opcode::S_LSHL_B32:             return "s_lshl_b32";
        case Opcode::S_LSHL_B64:             return "s_lshl_b64";
        case Opcode::S_LSHR_B32:             return "s_lshr_b32";
        case Opcode::S_LSHR_B64:             return "s_lshr_b64";
        case Opcode::S_ASHR_I32:             return "s_ashr_i32";
        case Opcode::S_ASHR_I64:             return "s_ashr_i64";
        case Opcode::S_BFM_B32:              return "s_bfm_b32";
        case Opcode::S_BFM_B64:              return "s_bfm_b64";
        case Opcode::S_MUL_I32:              return "s_mul_i32";
        case Opcode::S_BFE_U32:              return "s_bfe_u32";
        case Opcode::S_BFE_I32:              return "s_bfe_i32";
        case Opcode::S_BFE_U64:              return "s_bfe_u64";
        case Opcode::S_BFE_I64:              return "s_bfe_i64";
        case Opcode::S_ABSDIFF_I32:          return "s_absdiff_i32";
        case Opcode::S_MUL_HI_U32:           return "s_mul_hi_u32";
        case Opcode::S_MUL_HI_I32:           return "s_mul_hi_i32";
        case Opcode::S_LSHL1_ADD_U32:        return "s_lshl1_add_u32";
        case Opcode::S_LSHL2_ADD_U32:        return "s_lshl2_add_u32";
        case Opcode::S_LSHL3_ADD_U32:        return "s_lshl3_add_u32";
        case Opcode::S_LSHL4_ADD_U32:        return "s_lshl4_add_u32";
        case Opcode::S_PACK_LL_B32_B16:      return "s_pack_ll_b32_b16";
        case Opcode::S_PACK_LH_B32_B16:      return "s_pack_lh_b32_b16";
        case Opcode::S_PACK_HH_B32_B16:      return "s_pack_hh_b32_b16";
        case Opcode::S_CSELECT_B32:          return "s_cselect_b32";
        case Opcode::S_CSELECT_B64:          return "s_cselect_b64";
        case Opcode::S_MOV_B32:              return "s_mov_b32";
        case Opcode::S_MOV_B64:              return "s_mov_b64";
        case Opcode::S_CMOV_B32:             return "s_cmov_b32";
        case Opcode::S_CMOV_B64:             return "s_cmov_b64";
        case Opcode::S_NOT_B32:              return "s_not_b32";
        case Opcode::S_NOT_B64:              return "s_not_b64";
        case Opcode::S_BREV_B32:             return "s_brev_b32";
        case Opcode::S_BREV_B64:             return "s_brev_b64";
        case Opcode::S_BCNT0_I32_B32:        return "s_bcnt0_i32_b32";
        case Opcode::S_BCNT1_I32_B32:        return "s_bcnt1_i32_b32";
        case Opcode::S_FF0_I32_B32:          return "s_ff0_i32_b32";
        case Opcode::S_FF1_I32_B32:          return "s_ff1_i32_b32";
        case Opcode::S_FLBIT_I32_B32:        return "s_flbit_i32_b32";
        case Opcode::S_FLBIT_I32:            return "s_flbit_i32";
        case Opcode::S_SEXT_I32_I8:          return "s_sext_i32_i8";
        case Opcode::S_SEXT_I32_I16:         return "s_sext_i32_i16";
        case Opcode::S_ABS_I32:              return "s_abs_i32";
        case Opcode::S_GETPC_B64:            return "s_getpc_b64";
        case Opcode::S_SETPC_B64:            return "s_setpc_b64";
        case Opcode::S_SWAPPC_B64:           return "s_swappc_b64";
        case Opcode::S_AND_SAVEEXEC_B64:     return "s_and_saveexec_b64";
        case Opcode::S_OR_SAVEEXEC_B64:      return "s_or_saveexec_b64";
        case Opcode::S_XOR_SAVEEXEC_B64:     return "s_xor_saveexec_b64";
        case Opcode::S_ANDN2_SAVEEXEC_B64:   return "s_andn2_saveexec_b64";
        case Opcode::S_ORN2_SAVEEXEC_B64:    return "s_orn2_saveexec_b64";
        case Opcode::S_MOVK_I32:             return "s_movk_i32";
        case Opcode::S_ADDK_I32:             return "s_addk_i32";
        case Opcode::S_MULK_I32:             return "s_mulk_i32";
        case Opcode::S_CMPK_EQ_I32:          return "s_cmpk_eq_i32";
        case Opcode::S_CMPK_LT_I32:          return "s_cmpk_lt_i32";
        case Opcode::S_CMPK_GE_I32:          return "s_cmpk_ge_i32";
        case Opcode::S_CMP_EQ_I32:           return "s_cmp_eq_i32";
        case Opcode::S_CMP_LT_I32:           return "s_cmp_lt_i32";
        case Opcode::S_CMP_GE_I32:           return "s_cmp_ge_i32";
        case Opcode::S_CMP_GT_I32:           return "s_cmp_gt_i32";
        case Opcode::S_CMP_LE_I32:           return "s_cmp_le_i32";
        case Opcode::S_CMP_LG_I32:           return "s_cmp_lg_i32";
        case Opcode::S_CMP_EQ_U32:           return "s_cmp_eq_u32";
        case Opcode::S_CMP_LT_U32:           return "s_cmp_lt_u32";
        case Opcode::S_CMP_GE_U32:           return "s_cmp_ge_u32";
        case Opcode::S_CMP_GT_U32:           return "s_cmp_gt_u32";
        case Opcode::S_CMP_LE_U32:           return "s_cmp_le_u32";
        case Opcode::S_CMP_LG_U32:           return "s_cmp_lg_u32";
        case Opcode::S_CMP_EQ_U64:           return "s_cmp_eq_u64";
        case Opcode::S_CMP_LG_U64:           return "s_cmp_lg_u64";
        case Opcode::S_BITCMP0_B32:          return "s_bitcmp0_b32";
        case Opcode::S_BITCMP1_B32:          return "s_bitcmp1_b32";
        case Opcode::S_BITCMP0_B64:          return "s_bitcmp0_b64";
        case Opcode::S_BITCMP1_B64:          return "s_bitcmp1_b64";
        case Opcode::S_SETVSKIP:             return "s_setvskip";
        case Opcode::S_SET_GPR_IDX_ON:       return "s_set_gpr_idx_on";
        case Opcode::S_NOP:                  return "s_nop";
        case Opcode::S_ENDPGM:               return "s_endpgm";
        case Opcode::S_BRANCH:               return "s_branch";
        case Opcode::S_CBRANCH_SCC0:         return "s_cbranch_scc0";
        case Opcode::S_CBRANCH_SCC1:         return "s_cbranch_scc1";
        case Opcode::S_CBRANCH_VCCZ:         return "s_cbranch_vccz";
        case Opcode::S_CBRANCH_VCCNZ:        return "s_cbranch_vccnz";
        case Opcode::S_CBRANCH_EXECZ:        return "s_cbranch_execz";
        case Opcode::S_CBRANCH_EXECNZ:       return "s_cbranch_execnz";
        case Opcode::S_BARRIER:              return "s_barrier";
        case Opcode::S_WAITCNT:              return "s_waitcnt";
        case Opcode::S_SETPRIO:              return "s_setprio";
        case Opcode::S_SENDMSG:              return "s_sendmsg";
        case Opcode::S_TRAP:                 return "s_trap";
        case Opcode::S_ICACHE_INV:           return "s_icache_inv";
        case Opcode::S_WAKEUP:               return "s_wakeup";
        case Opcode::S_LOAD_DWORD:           return "s_load_dword";
        case Opcode::S_LOAD_DWORDX2:         return "s_load_dwordx2";
        case Opcode::S_LOAD_DWORDX4:         return "s_load_dwordx4";
        case Opcode::S_LOAD_DWORDX8:         return "s_load_dwordx8";
        case Opcode::S_LOAD_DWORDX16:        return "s_load_dwordx16";
        case Opcode::S_STORE_DWORD:          return "s_store_dword";
        case Opcode::S_STORE_DWORDX2:        return "s_store_dwordx2";
        case Opcode::S_STORE_DWORDX4:        return "s_store_dwordx4";
        case Opcode::S_ATOMIC_ADD:           return "s_atomic_add";
        case Opcode::V_MOV_B32:              return "v_mov_b32";
        case Opcode::V_MOV_B64:              return "v_mov_b64";
        case Opcode::V_PK_MOV_B32:           return "v_pk_mov_b32";
        case Opcode::V_READFIRSTLANE_B32:    return "v_readfirstlane_b32";
        case Opcode::V_CVT_F32_I32:          return "v_cvt_f32_i32";
        case Opcode::V_CVT_F32_U32:          return "v_cvt_f32_u32";
        case Opcode::V_CVT_U32_F32:          return "v_cvt_u32_f32";
        case Opcode::V_CVT_I32_F32:          return "v_cvt_i32_f32";
        case Opcode::V_CVT_F32_F64:          return "v_cvt_f32_f64";
        case Opcode::V_CVT_F64_F32:          return "v_cvt_f64_f32";
        case Opcode::V_CVT_F32_F16:          return "v_cvt_f32_f16";
        case Opcode::V_CVT_F16_F32:          return "v_cvt_f16_f32";
        case Opcode::V_FRACT_F32:            return "v_fract_f32";
        case Opcode::V_TRUNC_F32:            return "v_trunc_f32";
        case Opcode::V_CEIL_F32:             return "v_ceil_f32";
        case Opcode::V_RNDNE_F32:            return "v_rndne_f32";
        case Opcode::V_FLOOR_F32:            return "v_floor_f32";
        case Opcode::V_EXP_F32:              return "v_exp_f32";
        case Opcode::V_LOG_F32:              return "v_log_f32";
        case Opcode::V_RCP_F32:              return "v_rcp_f32";
        case Opcode::V_RCP_IFLAG_F32:        return "v_rcp_iflag_f32";
        case Opcode::V_RSQ_F32:              return "v_rsq_f32";
        case Opcode::V_SQRT_F32:             return "v_sqrt_f32";
        case Opcode::V_SQRT_F64:             return "v_sqrt_f64";
        case Opcode::V_SIN_F32:              return "v_sin_f32";
        case Opcode::V_COS_F32:              return "v_cos_f32";
        case Opcode::V_NOT_B32:              return "v_not_b32";
        case Opcode::V_BFREV_B32:            return "v_bfrev_b32";
        case Opcode::V_FFBH_U32:             return "v_ffbh_u32";
        case Opcode::V_FFBL_B32:             return "v_ffbl_b32";
        case Opcode::V_ADD_U32:              return "v_add_u32";
        case Opcode::V_CNDMASK_B32:          return "v_cndmask_b32";
        case Opcode::V_ADD_F32:              return "v_add_f32";
        case Opcode::V_SUB_F32:              return "v_sub_f32";
        case Opcode::V_MUL_F32:              return "v_mul_f32";
        case Opcode::V_MIN_F32:              return "v_min_f32";
        case Opcode::V_MAX_F32:              return "v_max_f32";
        case Opcode::V_MIN_I32:              return "v_min_i32";
        case Opcode::V_MAX_I32:              return "v_max_i32";
        case Opcode::V_MIN_U32:              return "v_min_u32";
        case Opcode::V_MAX_U32:              return "v_max_u32";
        case Opcode::V_MAD_F32:              return "v_mad_f32";
        case Opcode::V_FMA_F32:              return "v_fma_f32";
        case Opcode::V_FMA_F64:              return "v_fma_f64";
        case Opcode::V_ADD_F64:              return "v_add_f64";
        case Opcode::V_MUL_F64:              return "v_mul_f64";
        case Opcode::V_MIN_F64:              return "v_min_f64";
        case Opcode::V_MAX_F64:              return "v_max_f64";
        case Opcode::V_ADD_CO_U32:           return "v_add_co_u32";
        case Opcode::V_ADDC_CO_U32:          return "v_addc_co_u32";
        case Opcode::V_SUB_U32:              return "v_sub_u32";
        case Opcode::V_SUBREV_U32:           return "v_subrev_u32";
        case Opcode::V_MUL_U32_U24:          return "v_mul_u32_u24";
        case Opcode::V_MUL_LO_U32:           return "v_mul_lo_u32";
        case Opcode::V_MUL_HI_U32:           return "v_mul_hi_u32";
        case Opcode::V_MUL_HI_I32:           return "v_mul_hi_i32";
        case Opcode::V_AND_B32:              return "v_and_b32";
        case Opcode::V_OR_B32:               return "v_or_b32";
        case Opcode::V_XOR_B32:              return "v_xor_b32";
        case Opcode::V_LSHLREV_B32:          return "v_lshlrev_b32";
        case Opcode::V_LSHRREV_B32:          return "v_lshrrev_b32";
        case Opcode::V_ASHRREV_I32:          return "v_ashrrev_i32";
        case Opcode::V_LSHLREV_B64:          return "v_lshlrev_b64";
        case Opcode::V_LSHRREV_B64:          return "v_lshrrev_b64";
        case Opcode::V_ASHRREV_I64:          return "v_ashrrev_i64";
        case Opcode::V_BFE_U32:              return "v_bfe_u32";
        case Opcode::V_BFE_I32:              return "v_bfe_i32";
        case Opcode::V_BFI_B32:              return "v_bfi_b32";
        case Opcode::V_MIN3_F32:             return "v_min3_f32";
        case Opcode::V_MIN3_I32:             return "v_min3_i32";
        case Opcode::V_MIN3_U32:             return "v_min3_u32";
        case Opcode::V_MAX3_F32:             return "v_max3_f32";
        case Opcode::V_MAX3_I32:             return "v_max3_i32";
        case Opcode::V_MAX3_U32:             return "v_max3_u32";
        case Opcode::V_MED3_F32:             return "v_med3_f32";
        case Opcode::V_MED3_I32:             return "v_med3_i32";
        case Opcode::V_MED3_U32:             return "v_med3_u32";
        case Opcode::V_CMP_LT_I32:           return "v_cmp_lt_i32";
        case Opcode::V_CMP_LE_I32:           return "v_cmp_le_i32";
        case Opcode::V_CMP_GT_I32:           return "v_cmp_gt_i32";
        case Opcode::V_CMP_GE_I32:           return "v_cmp_ge_i32";
        case Opcode::V_CMP_EQ_I32:           return "v_cmp_eq_i32";
        case Opcode::V_CMP_NE_I32:           return "v_cmp_ne_i32";
        case Opcode::V_CMP_LT_U32:           return "v_cmp_lt_u32";
        case Opcode::V_CMP_LE_U32:           return "v_cmp_le_u32";
        case Opcode::V_CMP_GT_U32:           return "v_cmp_gt_u32";
        case Opcode::V_CMP_GE_U32:           return "v_cmp_ge_u32";
        case Opcode::V_CMP_EQ_U32:           return "v_cmp_eq_u32";
        case Opcode::V_CMP_NE_U32:           return "v_cmp_ne_u32";
        case Opcode::V_CMP_LT_F32:           return "v_cmp_lt_f32";
        case Opcode::V_CMP_EQ_F32:           return "v_cmp_eq_f32";
        case Opcode::V_CMP_LE_F32:           return "v_cmp_le_f32";
        case Opcode::V_CMP_GT_F32:           return "v_cmp_gt_f32";
        case Opcode::V_CMP_GE_F32:           return "v_cmp_ge_f32";
        case Opcode::V_CMP_NEQ_F32:          return "v_cmp_neq_f32";
        case Opcode::V_CMP_NLT_F32:          return "v_cmp_nlt_f32";
        case Opcode::V_CMP_LT_F64:           return "v_cmp_lt_f64";
        case Opcode::V_CMP_EQ_F64:           return "v_cmp_eq_f64";
        case Opcode::V_CMP_GE_F64:           return "v_cmp_ge_f64";
        case Opcode::V_CMP_LT_I64:           return "v_cmp_lt_i64";
        case Opcode::V_CMP_EQ_I64:           return "v_cmp_eq_i64";
        case Opcode::V_CMP_GE_I64:           return "v_cmp_ge_i64";
        case Opcode::V_CMP_NE_I64:           return "v_cmp_ne_i64";
        case Opcode::V_CMP_LT_U64:           return "v_cmp_lt_u64";
        case Opcode::V_CMP_EQ_U64:           return "v_cmp_eq_u64";
        case Opcode::V_CMP_GE_U64:           return "v_cmp_ge_u64";
        case Opcode::V_CMP_NE_U64:           return "v_cmp_ne_u64";
        case Opcode::V_ACCVGPR_READ_B32:     return "v_accvgpr_read_b32";
        case Opcode::V_ACCVGPR_WRITE_B32:    return "v_accvgpr_write_b32";
        case Opcode::GLOBAL_LOAD_DWORD:      return "global_load_dword";
        case Opcode::GLOBAL_LOAD_DWORDX2:    return "global_load_dwordx2";
        case Opcode::GLOBAL_LOAD_DWORDX3:    return "global_load_dwordx3";
        case Opcode::GLOBAL_LOAD_DWORDX4:    return "global_load_dwordx4";
        case Opcode::GLOBAL_STORE_DWORD:     return "global_store_dword";
        case Opcode::GLOBAL_STORE_DWORDX2:   return "global_store_dwordx2";
        case Opcode::GLOBAL_STORE_DWORDX3:   return "global_store_dwordx3";
        case Opcode::GLOBAL_STORE_DWORDX4:   return "global_store_dwordx4";
        case Opcode::GLOBAL_ATOMIC_ADD:      return "global_atomic_add";
        case Opcode::GLOBAL_ATOMIC_ADD_F64:  return "global_atomic_add_f64";
        case Opcode::GLOBAL_ATOMIC_ADD_X2:   return "global_atomic_add_x2";
        case Opcode::GLOBAL_ATOMIC_SUB:      return "global_atomic_sub";
        case Opcode::FLAT_LOAD_DWORD:        return "flat_load_dword";
        case Opcode::FLAT_LOAD_DWORDX2:      return "flat_load_dwordx2";
        case Opcode::FLAT_STORE_DWORD:       return "flat_store_dword";
        case Opcode::FLAT_STORE_DWORDX2:     return "flat_store_dwordx2";
        case Opcode::DS_WRITE_B32:           return "ds_write_b32";
        case Opcode::DS_WRITE_B16:           return "ds_write_b16";
        case Opcode::DS_WRITE_B16_D16_HI:    return "ds_write_b16_d16_hi";
        case Opcode::DS_READ_B32:            return "ds_read_b32";
        case Opcode::DS_WRITE_B64:           return "ds_write_b64";
        case Opcode::DS_READ_B64:            return "ds_read_b64";
        case Opcode::DS_READ_U16:            return "ds_read_u16";
        case Opcode::DS_READ_U16_D16:        return "ds_read_u16_d16";
        case Opcode::DS_READ_U16_D16_HI:     return "ds_read_u16_d16_hi";
        case Opcode::DS_ADD_U32:             return "ds_add_u32";
        case Opcode::DS_ADD_RTN_U32:         return "ds_add_rtn_u32";
        case Opcode::DS_BPERMUTE_B32:        return "ds_bpermute_b32";
        case Opcode::DS_PERMUTE_B32:         return "ds_permute_b32";
        case Opcode::V_PERMLANE16_B32:       return "v_permlane16_b32";
        case Opcode::V_PERMLANEX16_B32:      return "v_permlanex16_b32";
        case Opcode::V_PERM_B32:             return "v_perm_b32";
        case Opcode::V_PACK_B32_F16:         return "v_pack_b32_f16";
        case Opcode::V_CVT_PKRTZ_F16_F32:    return "v_cvt_pkrtz_f16_f32";
        case Opcode::BUFFER_LOAD_DWORD:      return "buffer_load_dword";
        case Opcode::BUFFER_LOAD_DWORDX2:    return "buffer_load_dwordx2";
        case Opcode::BUFFER_LOAD_DWORDX4:    return "buffer_load_dwordx4";
        case Opcode::BUFFER_LOAD_SHORT_D16:  return "buffer_load_short_d16";
        case Opcode::BUFFER_LOAD_SHORT_D16_HI: return "buffer_load_short_d16_hi";
        case Opcode::BUFFER_STORE_BYTE:      return "buffer_store_byte";
        case Opcode::BUFFER_STORE_SHORT:     return "buffer_store_short";
        case Opcode::BUFFER_STORE_SHORT_D16_HI: return "buffer_store_short_d16_hi";
        case Opcode::BUFFER_STORE_DWORD:     return "buffer_store_dword";
        case Opcode::BUFFER_STORE_DWORDX2:   return "buffer_store_dwordx2";
        case Opcode::BUFFER_STORE_DWORDX3:   return "buffer_store_dwordx3";
        case Opcode::BUFFER_STORE_DWORDX4:   return "buffer_store_dwordx4";
        case Opcode::BUFFER_ATOMIC_ADD:      return "buffer_atomic_add";
        case Opcode::V_MFMA_F32_32X32X1F32:  return "v_mfma_f32_32x32x1f32";
        case Opcode::V_MFMA_F32_16X16X1F32:  return "v_mfma_f32_16x16x1f32";
        case Opcode::V_MFMA_F32_4X4X1F32:    return "v_mfma_f32_4x4x1f32";
        case Opcode::V_MFMA_F32_32X32X2F32:  return "v_mfma_f32_32x32x2f32";
        case Opcode::V_MFMA_F32_16X16X4F32:  return "v_mfma_f32_16x16x4f32";
        case Opcode::V_MFMA_F32_32X32X4F16:  return "v_mfma_f32_32x32x4f16";
        case Opcode::V_MFMA_F32_16X16X4F16:  return "v_mfma_f32_16x16x4f16";
        case Opcode::V_MFMA_F32_4X4X4F16:    return "v_mfma_f32_4x4x4f16";
        case Opcode::V_MFMA_I32_32X32X4I8:   return "v_mfma_i32_32x32x4i8";
        case Opcode::V_MFMA_I32_16X16X4I8:   return "v_mfma_i32_16x16x4i8";
        case Opcode::V_MFMA_F64_16X16X4F64:  return "v_mfma_f64_16x16x4f64";
        case Opcode::V_MFMA_F64_4X4X4F64:    return "v_mfma_f64_4x4x4f64";
        case Opcode::V_MFMA_F32_16X16X8_XF32: return "v_mfma_f32_16x16x8_xf32";
        case Opcode::V_MFMA_F32_32X32X4_XF32: return "v_mfma_f32_32x32x4_xf32";
        case Opcode::V_MFMA_I32_32X32X16I8:   return "v_mfma_i32_32x32x16i8";
        case Opcode::V_MFMA_I32_16X16X32I8:   return "v_mfma_i32_16x16x32i8";
        case Opcode::V_MFMA_F32_16X16X32_BF8_BF8: return "v_mfma_f32_16x16x32_bf8_bf8";
        case Opcode::V_MFMA_F32_16X16X32_BF8_FP8: return "v_mfma_f32_16x16x32_bf8_fp8";
        case Opcode::V_MFMA_F32_16X16X32_FP8_BF8: return "v_mfma_f32_16x16x32_fp8_bf8";
        case Opcode::V_MFMA_F32_16X16X32_FP8_FP8: return "v_mfma_f32_16x16x32_fp8_fp8";
        case Opcode::V_MFMA_F32_32X32X16_BF8_BF8: return "v_mfma_f32_32x32x16_bf8_bf8";
        case Opcode::V_MFMA_F32_32X32X16_BF8_FP8: return "v_mfma_f32_32x32x16_bf8_fp8";
        case Opcode::V_MFMA_F32_32X32X16_FP8_BF8: return "v_mfma_f32_32x32x16_fp8_bf8";
        case Opcode::V_MFMA_F32_32X32X16_FP8_FP8: return "v_mfma_f32_32x32x16_fp8_fp8";
        case Opcode::V_MFMA_F32_16X16X32_F16: return "v_mfma_f32_16x16x32_f16";
        case Opcode::V_MFMA_F32_32X32X16_F16: return "v_mfma_f32_32x32x16_f16";
        case Opcode::V_MFMA_F32_16X16X32_BF16: return "v_mfma_f32_16x16x32_bf16";
        case Opcode::V_MFMA_F32_32X32X16_BF16: return "v_mfma_f32_32x32x16_bf16";
        case Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4: return "v_mfma_scale_f32_16x16x128_f8f6f4";
        case Opcode::V_MFMA_SCALE_F32_32X32X64_F8F6F4: return "v_mfma_scale_f32_32x32x64_f8f6f4";
        case Opcode::V_MFMA_I32_16X16X64_I8: return "v_mfma_i32_16x16x64_i8";
        case Opcode::V_MFMA_I32_32X32X32_I8: return "v_mfma_i32_32x32x32_i8";
        case Opcode::V_MFMA_LD_SCALE_B32:    return "v_mfma_ld_scale_b32";
        case Opcode::V_CVT_SCALEF32_PK_F32_FP4: return "v_cvt_scalef32_pk_f32_fp4";
        case Opcode::V_CVT_SCALEF32_PK_F16_FP4: return "v_cvt_scalef32_pk_f16_fp4";
        case Opcode::V_PRNG_B32:             return "v_prng_b32";
        case Opcode::V_PK_ADD_F16:           return "v_pk_add_f16";
        case Opcode::V_PK_MUL_F16:           return "v_pk_mul_f16";
        case Opcode::V_PK_FMA_F32:           return "v_pk_fma_f32";
        case Opcode::V_PK_MUL_F32:           return "v_pk_mul_f32";
        case Opcode::V_PK_ADD_F32:           return "v_pk_add_f32";
        case Opcode::V_DOT2_F32_F16:         return "v_dot2_f32_f16";
        case Opcode::V_DOT4_I32_I8:          return "v_dot4_i32_i8";
        case Opcode::V_DOT8_I32_I4:          return "v_dot8_i32_i4";
        case Opcode::V_ADD_LSHL_U32:         return "v_add_lshl_u32";
        case Opcode::V_LSHL_ADD_U32:         return "v_lshl_add_u32";
        case Opcode::V_LSHL_ADD_U64:         return "v_lshl_add_u64";
        case Opcode::S_LOAD_DWORDX3:         return "s_load_dwordx3";
        case Opcode::DS_READ_B96:            return "ds_read_b96";
        case Opcode::DS_WRITE_B96:           return "ds_write_b96";
        case Opcode::S_BARRIER_SIGNAL:       return "s_barrier_signal";
        case Opcode::S_BARRIER_WAIT:         return "s_barrier_wait";
        case Opcode::S_WAIT_LOADCNT:         return "s_wait_loadcnt";
        case Opcode::S_WAIT_STORECNT:        return "s_wait_storecnt";
        case Opcode::S_WAIT_DSCNT:           return "s_wait_dscnt";
        case Opcode::S_WAIT_KMCNT:           return "s_wait_kmcnt";
        case Opcode::S_WAIT_LOADCNT_DSCNT:   return "s_wait_loadcnt_dscnt";
        case Opcode::S_WAIT_STORECNT_DSCNT:  return "s_wait_storecnt_dscnt";
        case Opcode::S_WAIT_ALU:             return "s_wait_alu";
        case Opcode::S_DELAY_ALU:            return "s_delay_alu";
        case Opcode::S_CLAUSE:               return "s_clause";
        case Opcode::GLOBAL_INV:             return "global_inv";
        case Opcode::S_AND_SAVEEXEC_B32:     return "s_and_saveexec_b32";
        case Opcode::S_OR_SAVEEXEC_B32:      return "s_or_saveexec_b32";
        case Opcode::S_XOR_SAVEEXEC_B32:     return "s_xor_saveexec_b32";
        case Opcode::S_ANDN2_SAVEEXEC_B32:   return "s_andn2_saveexec_b32";
        case Opcode::S_ORN2_SAVEEXEC_B32:    return "s_orn2_saveexec_b32";
        case Opcode::S_ADD_NC_U64:           return "s_add_nc_u64";
        case Opcode::S_SUB_NC_U64:           return "s_sub_nc_u64";
        case Opcode::S_MUL_U64:              return "s_mul_u64";
        case Opcode::S_MUL_F32:              return "s_mul_f32";
        case Opcode::S_FMAMK_F32:            return "s_fmamk_f32";
        case Opcode::S_FMAAK_F32:            return "s_fmaak_f32";
        case Opcode::S_CVT_F32_U32:          return "s_cvt_f32_u32";
        case Opcode::S_CVT_U32_F32:          return "s_cvt_u32_f32";
        case Opcode::S_CVT_F32_I32:          return "s_cvt_f32_i32";
        case Opcode::S_CVT_I32_F32:          return "s_cvt_i32_f32";
        case Opcode::S_TRUNC_F32:            return "s_trunc_f32";
        case Opcode::S_CMP_EQ_F32:           return "s_cmp_eq_f32";
        case Opcode::S_CMP_NEQ_F32:          return "s_cmp_neq_f32";
        case Opcode::V_S_RCP_F32:            return "v_s_rcp_f32";
        case Opcode::V_MAXMIN_I32:           return "v_maxmin_i32";
        case Opcode::V_MAXMIN_U32:           return "v_maxmin_u32";
        case Opcode::V_MINMAX_I32:           return "v_minmax_i32";
        case Opcode::V_MINMAX_U32:           return "v_minmax_u32";
        case Opcode::V_CVT_F32_FP8:          return "v_cvt_f32_fp8";
        case Opcode::V_CVT_F32_BF8:          return "v_cvt_f32_bf8";
        case Opcode::V_CVT_PK_FP8_F32:       return "v_cvt_pk_fp8_f32";
        case Opcode::V_CVT_PK_BF8_F32:       return "v_cvt_pk_bf8_f32";
        case Opcode::V_WMMA_F32_16X16X16_F16:     return "v_wmma_f32_16x16x16_f16";
        case Opcode::V_WMMA_F32_16X16X16_BF16:    return "v_wmma_f32_16x16x16_bf16";
        case Opcode::V_WMMA_F16_16X16X16_F16:     return "v_wmma_f16_16x16x16_f16";
        case Opcode::V_WMMA_BF16_16X16X16_BF16:   return "v_wmma_bf16_16x16x16_bf16";
        case Opcode::V_WMMA_I32_16X16X16_IU8:     return "v_wmma_i32_16x16x16_iu8";
        case Opcode::V_WMMA_I32_16X16X16_IU4:     return "v_wmma_i32_16x16x16_iu4";
        case Opcode::V_WMMA_F32_16X16X16_FP8_FP8: return "v_wmma_f32_16x16x16_fp8_fp8";
        case Opcode::V_WMMA_F32_16X16X16_FP8_BF8: return "v_wmma_f32_16x16x16_fp8_bf8";
        case Opcode::V_WMMA_F32_16X16X16_BF8_FP8: return "v_wmma_f32_16x16x16_bf8_fp8";
        case Opcode::V_WMMA_F32_16X16X16_BF8_BF8: return "v_wmma_f32_16x16x16_bf8_bf8";
        case Opcode::UNKNOWN:                return "unknown";
        default:                             return "?";
    }
}

const char* format_name(InstFormat fmt) {
    switch (fmt) {
        case InstFormat::SOP2:        return "SOP2";
        case InstFormat::SOP1:        return "SOP1";
        case InstFormat::SOPK:        return "SOPK";
        case InstFormat::SOPC:        return "SOPC";
        case InstFormat::SOPP:        return "SOPP";
        case InstFormat::VOP2:        return "VOP2";
        case InstFormat::VOP1:        return "VOP1";
        case InstFormat::VOPC:        return "VOPC";
        case InstFormat::VOP3A:       return "VOP3A";
        case InstFormat::VOP3B:       return "VOP3B";
        case InstFormat::VOP3P:       return "VOP3P";
        case InstFormat::VOPD:        return "VOPD";
        case InstFormat::SMEM:        return "SMEM";
        case InstFormat::FLAT_GLOBAL: return "FLAT_GLOBAL";
        case InstFormat::FLAT_SCRATCH: return "FLAT_SCRATCH";
        case InstFormat::DS:          return "DS";
        case InstFormat::MUBUF:       return "MUBUF";
        case InstFormat::MTBUF:       return "MTBUF";
        case InstFormat::MFMA:        return "MFMA";
        case InstFormat::VINTRP:      return "VINTRP";
        case InstFormat::EXP:         return "EXP";
        case InstFormat::UNKNOWN:     return "UNKNOWN";
        default:                      return "?";
    }
}

} // namespace gcnsim
