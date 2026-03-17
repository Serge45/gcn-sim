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
static Operand decode_src(uint32_t enc, const uint8_t* pc_next4) {
    Operand op;
    if (enc <= 103) {
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 106) {
        op.kind = Operand::Kind::VCC_LO;
    } else if (enc == 107) {
        op.kind = Operand::Kind::VCC_HI;
    } else if (enc == 124) {
        // M0 — treat as inline 0 for now
        op.kind = Operand::Kind::INLINE_CONST;
        op.imm = 0;
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
static Operand decode_sdst(uint32_t enc) {
    Operand op;
    if (enc <= 105) {
        op.kind = Operand::Kind::SGPR;
        op.index = enc;
    } else if (enc == 106) {
        op.kind = Operand::Kind::VCC_LO;
    } else if (enc == 107) {
        op.kind = Operand::Kind::VCC_HI;
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
        // 10: S_CSELECT_B32, 11: S_CSELECT_B64 (not yet implemented)
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
        default: return Opcode::UNKNOWN;
    }
}

static Opcode sop1_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_MOV_B32;
        case 1:  return Opcode::S_MOV_B64;
        case 2:  return Opcode::S_NOT_B32;
        case 3:  return Opcode::S_NOT_B64;
        case 4:  return Opcode::S_BREV_B32;
        case 5:  return Opcode::S_BREV_B64;
        case 6:  return Opcode::S_BCNT0_I32_B32;
        case 7:  return Opcode::S_BCNT0_I32_B64;
        case 8:  return Opcode::S_BCNT1_I32_B32;
        case 9:  return Opcode::S_BCNT1_I32_B64;
        case 10: return Opcode::S_FF0_I32_B32;
        case 11: return Opcode::S_FF0_I32_B64;
        case 12: return Opcode::S_FF1_I32_B32;
        case 13: return Opcode::S_FF1_I32_B64;
        case 14: return Opcode::S_FLBIT_I32_B32;
        case 15: return Opcode::S_FLBIT_I32_B64;
        case 16: return Opcode::S_FLBIT_I32;
        case 17: return Opcode::S_FLBIT_I32_I64;
        case 18: return Opcode::S_SEXT_I32_I8;
        case 19: return Opcode::S_SEXT_I32_I16;
        case 20: return Opcode::S_BITSET0_B32;
        case 21: return Opcode::S_BITSET0_B64;
        case 22: return Opcode::S_BITSET1_B32;
        case 23: return Opcode::S_BITSET1_B64;
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
        case 64: return Opcode::S_ATOMIC_SWAP;
        case 65: return Opcode::S_ATOMIC_CMPSWAP;
        case 66: return Opcode::S_ATOMIC_ADD;
        case 67: return Opcode::S_ATOMIC_SUB;
        case 68: return Opcode::S_ATOMIC_SMIN;
        case 69: return Opcode::S_ATOMIC_UMIN;
        case 70: return Opcode::S_ATOMIC_SMAX;
        case 71: return Opcode::S_ATOMIC_UMAX;
        case 72: return Opcode::S_ATOMIC_AND;
        case 73: return Opcode::S_ATOMIC_OR;
        case 74: return Opcode::S_ATOMIC_XOR;
        case 75: return Opcode::S_ATOMIC_INC;
        case 76: return Opcode::S_ATOMIC_DEC;
        default: return Opcode::UNKNOWN;
    }
}

static Opcode vop2_opcode(uint32_t op) {
    // GCN3/GFX90a VOP2 opcode table (from AMD ISA spec)
    switch (op) {
        case 0:  return Opcode::UNKNOWN;    // v_cndmask_b32
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
        case 53: return Opcode::V_SUB_CO_U32;
        case 54: return Opcode::V_SUBREV_CO_U32;
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
        case 10:  return Opcode::V_CVT_F32_F16;
        case 11:  return Opcode::V_CVT_F32_F64;   // note: some tables have this swapped
        case 12:  return Opcode::V_CVT_F64_F32;
        case 17:  return Opcode::V_CVT_F32_UBYTE0;
        case 18:  return Opcode::V_CVT_F32_UBYTE1;
        case 19:  return Opcode::V_CVT_F32_UBYTE2;
        case 20:  return Opcode::V_CVT_F32_UBYTE3;
        case 21:  return Opcode::V_CVT_U32_F64;
        case 22:  return Opcode::V_CVT_F64_U32;
        case 33:  return Opcode::V_FRACT_F32;
        case 34:  return Opcode::V_TRUNC_F32;
        case 35:  return Opcode::V_CEIL_F32;
        case 36:  return Opcode::V_RNDNE_F32;
        case 37:  return Opcode::V_FLOOR_F32;
        case 38:  return Opcode::V_EXP_F32;
        case 39:  return Opcode::V_LOG_F32;
        case 42:  return Opcode::V_RCP_F32;
        case 43:  return Opcode::V_RCP_IFLAG_F32;
        case 44:  return Opcode::V_RSQ_F32;
        case 45:  return Opcode::V_RCP_F64;
        case 47:  return Opcode::V_RSQ_F64;
        case 48:  return Opcode::V_SQRT_F32;
        case 49:  return Opcode::V_SQRT_F64;
        case 51:  return Opcode::V_SIN_F32;
        case 52:  return Opcode::V_COS_F32;
        case 53:  return Opcode::V_NOT_B32;
        case 54:  return Opcode::V_BFREV_B32;
        case 55:  return Opcode::V_FFBH_U32;
        case 56:  return Opcode::V_FFBL_B32;
        case 57:  return Opcode::V_FFBH_I32;
        case 58:  return Opcode::V_FREXP_EXP_I32_F64;
        case 59:  return Opcode::V_FREXP_MANT_F64;
        case 60:  return Opcode::V_FRACT_F64;
        case 61:  return Opcode::V_FREXP_EXP_I32_F32;
        case 62:  return Opcode::V_FREXP_MANT_F32;
        case 63:  return Opcode::V_CLREXCP;
        case 74:  return Opcode::V_TRUNC_F64;
        case 75:  return Opcode::V_CEIL_F64;
        case 76:  return Opcode::V_RNDNE_F64;
        case 77:  return Opcode::V_FLOOR_F64;
        case 84:  return Opcode::V_ACCVGPR_READ_B32;
        case 86:  return Opcode::V_ACCVGPR_WRITE_B32;
        case 112: return Opcode::V_SWAP_B32;
        default:  return Opcode::UNKNOWN;
    }
}

static Opcode vopc_opcode(uint32_t op) {
    // f32 comparisons: ops 0x10-0x1F (GFX9 canonical range 0x40-0x4F for VOPC)
    // i32 comparisons: 0xC0-0xC7, u32: 0xC8-0xCF
    // i64: 0xE0-0xE7, u64: 0xE8-0xEF
    // f64: 0x50-0x5F
    if (op >= 0x10 && op <= 0x1F) {
        // f32
        switch (op - 0x10) {
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
    if (op >= 0x20 && op <= 0x2F) {
        // f64
        switch (op - 0x20) {
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
    // Alternate encodings used in disassembly output
    if (op >= 0x40 && op <= 0x4F) {
        switch (op - 0x40) {
            case 1: return Opcode::V_CMP_LT_F32;
            case 2: return Opcode::V_CMP_EQ_F32;
            case 3: return Opcode::V_CMP_LE_F32;
            case 4: return Opcode::V_CMP_GT_F32;
            case 5: return Opcode::V_CMP_LG_F32;
            case 6: return Opcode::V_CMP_GE_F32;
            default: break;
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
        // Legacy aliases
        case 0x91: return Opcode::V_CMP_LT_U32;
        case 0x92: return Opcode::V_CMP_EQ_U32;
        case 0x93: return Opcode::V_CMP_LE_U32;
        case 0x94: return Opcode::V_CMP_GT_U32;
        case 0x95: return Opcode::V_CMP_NE_U32;
        case 0x96: return Opcode::V_CMP_GE_U32;
        case 0x81: return Opcode::V_CMP_LT_I32;
        case 0x86: return Opcode::V_CMP_GE_I32;
        default:   return Opcode::UNKNOWN;
    }
}

// VOP3 opcode space (10-bit field):
// 0x140-0x17F = VOP2 re-encoded
// 0x180-0x1BF = VOP1 re-encoded
// 0x1C0-0x1FF = VOPC re-encoded (0x1C0 + vopc_8bit_op)
// 0x200+ = true VOP3 ops
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
        case 0x200: return Opcode::V_MAD_F32;       // v_mad_legacy_f32 (treat as v_mad_f32)
        case 0x201: return Opcode::V_MAD_F32;
        case 0x202: return Opcode::V_MAD_I32_I24;
        case 0x203: return Opcode::V_MAD_U32_U24;
        case 0x204: return Opcode::UNKNOWN;         // v_cubeid_f32 (graphics)
        case 0x209: return Opcode::V_BFE_U32;
        case 0x20A: return Opcode::V_BFE_I32;
        case 0x20B: return Opcode::V_BFI_B32;
        case 0x20C: return Opcode::V_FMA_F32;
        case 0x20D: return Opcode::V_FMA_F64;
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
        case 0x1E8: return Opcode::V_MAD_U64_U32;
        case 0x1E9: return Opcode::V_MAD_I64_I32;
        case 0x228: return Opcode::V_MAD_U64_U32;
        case 0x229: return Opcode::V_MAD_I64_I32;
        case 0x1FD: return Opcode::V_LSHL_ADD_U32;
        case 0x274: return Opcode::V_ADD_I32;
        case 0x275: return Opcode::V_ADD3_U32;
        case 0x276: return Opcode::V_LSHL_OR_B32;
        case 0x277: return Opcode::V_AND_OR_B32;
        case 0x278: return Opcode::V_OR3_B32;
        case 0x2F5: return Opcode::V_MAD_F32;
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
        default: {
            // VOP3->VOPC: op = 0x1C0 | vopc_8bit
            if (op >= 0x1C0 && op <= 0x1FF) {
                return vopc_opcode(op - 0x1C0);
            }
            // VOP3->VOP2
            if (op >= 0x140 && op <= 0x17F) {
                return vop2_opcode(op - 0x140);
            }
            // VOP3->VOP1
            if (op >= 0x180 && op <= 0x1BF) {
                return vop1_opcode(op - 0x180);
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
        default: return Opcode::UNKNOWN;
    }
}

static Opcode flat_opcode(uint32_t op, bool is_global, bool is_scratch) {
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
            // Legacy
            case 0x0C: return Opcode::GLOBAL_LOAD_DWORD;
            case 0x0D: return Opcode::GLOBAL_LOAD_DWORDX2;
            case 0x0E: return Opcode::GLOBAL_LOAD_DWORDX4;
            case 0x1A: return Opcode::GLOBAL_STORE_DWORD;
            case 0x1B: return Opcode::GLOBAL_STORE_DWORDX2;
            default:   return Opcode::UNKNOWN;
        }
    }
    (void)is_scratch;
    // flat (SEG=0): treat like global with VGPR 64-bit address
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
    switch (op) {
        case 0:  return Opcode::BUFFER_LOAD_UBYTE;
        case 1:  return Opcode::BUFFER_LOAD_SBYTE;
        case 2:  return Opcode::BUFFER_LOAD_USHORT;
        case 3:  return Opcode::BUFFER_LOAD_SSHORT;
        case 4:  return Opcode::BUFFER_LOAD_DWORD;
        case 5:  return Opcode::BUFFER_LOAD_DWORDX2;
        case 6:  return Opcode::BUFFER_LOAD_DWORDX3;
        case 7:  return Opcode::BUFFER_LOAD_DWORDX4;
        case 24: return Opcode::BUFFER_STORE_BYTE;
        case 25: return Opcode::BUFFER_STORE_SHORT;
        case 28: return Opcode::BUFFER_STORE_DWORD;
        case 29: return Opcode::BUFFER_STORE_DWORDX2;
        case 30: return Opcode::BUFFER_STORE_DWORDX3;
        case 31: return Opcode::BUFFER_STORE_DWORDX4;
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

static Opcode ds_opcode(uint32_t op) {
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
        case 11:  return Opcode::DS_WRITE_B32;
        case 12:  return Opcode::DS_WRITE2_B32;
        case 13:  return Opcode::DS_WRITE2ST64_B32;
        case 14:  return Opcode::DS_CMPST_B32;
        case 15:  return Opcode::DS_CMPST_F32;
        case 16:  return Opcode::DS_MIN_F32;
        case 17:  return Opcode::DS_MAX_F32;
        case 20:  return Opcode::DS_WRITE_ADDTID_B32;
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
        case 43:  return Opcode::DS_WRXCHG_RTN_B32;
        case 44:  return Opcode::DS_WRXCHG2_RTN_B32;
        case 46:  return Opcode::DS_CMPST_RTN_B32;
        case 47:  return Opcode::DS_CMPST_RTN_F32;
        case 48:  return Opcode::DS_MIN_RTN_F32;
        case 49:  return Opcode::DS_MAX_RTN_F32;
        case 54:  return Opcode::DS_ADD_F32;
        case 55:  return Opcode::DS_NOP;
        case 56:  return Opcode::DS_READ_B32;
        case 57:  return Opcode::DS_READ2_B32;
        case 58:  return Opcode::DS_READ2ST64_B32;
        case 59:  return Opcode::DS_READ_I8;
        case 60:  return Opcode::DS_READ_U8;
        case 61:  return Opcode::DS_READ_I16;
        case 62:  return Opcode::DS_READ_U16;
        case 63:  return Opcode::DS_SWIZZLE_B32;
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
        case 75:  return Opcode::DS_WRITE_B64;
        case 76:  return Opcode::DS_WRITE2_B64;
        case 77:  return Opcode::DS_WRITE2ST64_B64;
        case 78:  return Opcode::DS_CMPST_B64;
        case 79:  return Opcode::DS_CMPST_F64;
        case 80:  return Opcode::DS_MIN_F64;
        case 81:  return Opcode::DS_MAX_F64;
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
        case 107: return Opcode::DS_WRXCHG_RTN_B64;
        case 120: return Opcode::DS_READ_B64;
        case 121: return Opcode::DS_READ2_B64;
        case 122: return Opcode::DS_READ2ST64_B64;
        case 126: return Opcode::DS_BPERMUTE_B32;
        case 191: return Opcode::DS_PERMUTE_B32;
        case 253: return Opcode::DS_WRITE_B128;
        case 255: return Opcode::DS_READ_B128;
        default:  return Opcode::UNKNOWN;
    }
}

static Opcode mfma_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::V_MFMA_F32_32X32X1F32;
        case 1:  return Opcode::V_MFMA_F32_16X16X1F32;
        case 2:  return Opcode::V_MFMA_F32_4X4X1F32;
        case 4:  return Opcode::V_MFMA_F32_32X32X2F32;
        case 5:  return Opcode::V_MFMA_F32_16X16X4F32;
        case 16: return Opcode::V_MFMA_F32_32X32X4F16;
        case 17: return Opcode::V_MFMA_F32_16X16X4F16;
        case 18: return Opcode::V_MFMA_F32_4X4X4F16;
        case 20: return Opcode::V_MFMA_F32_32X32X8F16;
        case 21: return Opcode::V_MFMA_F32_16X16X16F16;
        case 32: return Opcode::V_MFMA_I32_32X32X4I8;
        case 33: return Opcode::V_MFMA_I32_16X16X4I8;
        case 34: return Opcode::V_MFMA_I32_4X4X4I8;
        case 36: return Opcode::V_MFMA_I32_32X32X8I8;
        case 37: return Opcode::V_MFMA_I32_16X16X16I8;
        case 48: return Opcode::V_MFMA_F32_32X32X2BF16;
        case 49: return Opcode::V_MFMA_F32_16X16X2BF16;
        case 50: return Opcode::V_MFMA_F32_4X4X2BF16;
        case 52: return Opcode::V_MFMA_F32_32X32X4BF16;
        case 53: return Opcode::V_MFMA_F32_16X16X8BF16;
        case 60: return Opcode::V_MFMA_F32_32X32X4BF16_1K;
        case 61: return Opcode::V_MFMA_F32_16X16X4BF16_1K;
        case 62: return Opcode::V_MFMA_F32_4X4X4BF16_1K;
        case 63: return Opcode::V_MFMA_F32_32X32X8BF16_1K;
        case 64: return Opcode::V_MFMA_F32_16X16X16BF16_1K;
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
// Main decode function
// ---------------------------------------------------------------------------

Instruction decode(const uint8_t* pc) {
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
        }
        return inst;
    }

    // --- DS: [31:26] == 0x36 ---
    if (bits31_26 == 0x36) {
        inst.format = InstFormat::DS;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op      = (w0 >> 18) & 0xFF;
        inst.ds_gds      = (w0 >> 17) & 1;
        inst.ds_offset1  = (w0 >> 8) & 0xFF;
        inst.ds_offset0  = w0 & 0xFF;

        uint32_t vdst  = (w1 >> 24) & 0xFF;
        uint32_t data1 = (w1 >> 16) & 0xFF;
        uint32_t data0 = (w1 >> 8)  & 0xFF;
        uint32_t addr  = w1 & 0xFF;

        inst.opcode = ds_opcode(op);

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
        uint32_t seg   = (w0 >> 16) & 0x3;  // [17:16]: 00=flat, 01=scratch, 10=global

        bool is_scratch = (seg == 1);
        // Determine global vs flat by op range:
        // ops 0x10-0x4F are global-only ops; ops 0x08-0x0F are flat ops.
        // Also detect global from seg=2. Treat op >= 0x10 as global.
        bool is_global = (seg == 2) || (op >= 0x10);

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

        if (!is_store) {
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdst;
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr * 2;
            }
        } else {
            inst.src[0].kind  = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind  = Operand::Kind::SGPR;
                inst.src[1].index = saddr * 2;
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

        uint32_t vdata = (w1 >> 24) & 0xFF;
        uint32_t vaddr = (w1 >> 16) & 0xFF;
        uint32_t srsrc = (w1 >> 8)  & 0x1F;  // 5-bit, actual = srsrc * 4
        uint32_t soffset = w1 & 0xFF;

        inst.opcode = mubuf_opcode(op);

        bool is_store = false;
        switch (inst.opcode) {
            case Opcode::BUFFER_STORE_BYTE:
            case Opcode::BUFFER_STORE_SHORT:
            case Opcode::BUFFER_STORE_DWORD:
            case Opcode::BUFFER_STORE_DWORDX2:
            case Opcode::BUFFER_STORE_DWORDX3:
            case Opcode::BUFFER_STORE_DWORDX4:
                is_store = true;
                break;
            default: break;
        }

        // src[0] = VADDR, src[1] = SRSRC (buffer descriptor base), src[2] = SOFFSET
        inst.src[0].kind  = Operand::Kind::VGPR;
        inst.src[0].index = vaddr;
        inst.src[1].kind  = Operand::Kind::SGPR;
        inst.src[1].index = srsrc * 4;
        inst.src[2].kind  = Operand::Kind::SGPR;
        inst.src[2].index = soffset;

        if (is_store) {
            inst.dst.kind = Operand::Kind::NONE;
            // vdata = source data
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

        uint32_t vdata = (w1 >> 24) & 0xFF;
        uint32_t vaddr = (w1 >> 16) & 0xFF;
        uint32_t srsrc = (w1 >> 8)  & 0x1F;

        inst.src[0].kind  = Operand::Kind::VGPR;
        inst.src[0].index = vaddr;
        inst.src[1].kind  = Operand::Kind::SGPR;
        inst.src[1].index = srsrc * 4;
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
            // v_pk_mov_b32: op=947 (0x3B3), moves packed 32-bit pair
            if (op == 0x3B3) {
                inst.format    = InstFormat::VOP3A;
                inst.opcode    = Opcode::V_PK_MOV_B32;
                inst.dst.kind  = Operand::Kind::VGPR;
                inst.dst.index = vdst;
                inst.src[0] = decode_src(src0, nullptr);
                inst.src[1] = decode_src(src1, nullptr);
                return inst;
            }
            // Try MFMA: MFMA ops occupy the 10-bit op range 960..1023
            if (op >= 960) {
                uint32_t mfma_op = op - 960;
                Opcode mc = mfma_opcode(mfma_op);
                if (mc != Opcode::UNKNOWN) {
                    inst.format = InstFormat::MFMA;
                    inst.opcode = mc;
                    // On gfx90a the compiler may use either VGPR or AGPR for
                    // the accumulator (dst / src2). Keep whatever the encoding
                    // says; both VGPRs (accum VGPRs) and AGPRs are valid.
                    inst.dst.kind  = Operand::Kind::VGPR;
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
                         inst.opcode == Opcode::V_SUB_U32 ||
                         inst.opcode == Opcode::V_SUBREV_U32 ||
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
        } else {
            inst.format = InstFormat::VOP3A;
            inst.dst.kind  = Operand::Kind::VGPR;
            inst.dst.index = vdst;
        }

        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1] = decode_src(src1, nullptr);
        inst.src[2] = decode_src(src2, nullptr);
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
        inst.dst    = decode_vdst(vdst);
        bool has_literal = (src0 == 255);
        inst.src[0] = decode_src(src0, has_literal ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;

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
        inst.dst.kind = Operand::Kind::VCC;
        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1].kind  = Operand::Kind::VGPR;
        inst.src[1].index = vsrc1;
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
        case Opcode::S_MOV_B32:              return "s_mov_b32";
        case Opcode::S_MOV_B64:              return "s_mov_b64";
        case Opcode::S_NOT_B32:              return "s_not_b32";
        case Opcode::S_NOT_B64:              return "s_not_b64";
        case Opcode::S_BREV_B32:             return "s_brev_b32";
        case Opcode::S_BREV_B64:             return "s_brev_b64";
        case Opcode::S_BCNT0_I32_B32:        return "s_bcnt0_i32_b32";
        case Opcode::S_BCNT1_I32_B32:        return "s_bcnt1_i32_b32";
        case Opcode::S_FF0_I32_B32:          return "s_ff0_i32_b32";
        case Opcode::S_FF1_I32_B32:          return "s_ff1_i32_b32";
        case Opcode::S_FLBIT_I32_B32:        return "s_flbit_i32_b32";
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
        case Opcode::GLOBAL_ATOMIC_SUB:      return "global_atomic_sub";
        case Opcode::FLAT_LOAD_DWORD:        return "flat_load_dword";
        case Opcode::FLAT_LOAD_DWORDX2:      return "flat_load_dwordx2";
        case Opcode::FLAT_STORE_DWORD:       return "flat_store_dword";
        case Opcode::FLAT_STORE_DWORDX2:     return "flat_store_dwordx2";
        case Opcode::DS_WRITE_B32:           return "ds_write_b32";
        case Opcode::DS_READ_B32:            return "ds_read_b32";
        case Opcode::DS_WRITE_B64:           return "ds_write_b64";
        case Opcode::DS_READ_B64:            return "ds_read_b64";
        case Opcode::DS_ADD_U32:             return "ds_add_u32";
        case Opcode::DS_ADD_RTN_U32:         return "ds_add_rtn_u32";
        case Opcode::DS_BPERMUTE_B32:        return "ds_bpermute_b32";
        case Opcode::DS_PERMUTE_B32:         return "ds_permute_b32";
        case Opcode::BUFFER_LOAD_DWORD:      return "buffer_load_dword";
        case Opcode::BUFFER_LOAD_DWORDX2:    return "buffer_load_dwordx2";
        case Opcode::BUFFER_LOAD_DWORDX4:    return "buffer_load_dwordx4";
        case Opcode::BUFFER_STORE_DWORD:     return "buffer_store_dword";
        case Opcode::BUFFER_STORE_DWORDX2:   return "buffer_store_dwordx2";
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
        case Opcode::V_PK_ADD_F16:           return "v_pk_add_f16";
        case Opcode::V_PK_MUL_F16:           return "v_pk_mul_f16";
        case Opcode::V_DOT2_F32_F16:         return "v_dot2_f32_f16";
        case Opcode::V_DOT4_I32_I8:          return "v_dot4_i32_i8";
        case Opcode::V_DOT8_I32_I4:          return "v_dot8_i32_i4";
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
