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
        op.imm = (uint64_t)(int64_t)(int32_t)((int32_t)192 - (int32_t)enc + 1);
        // enc=193 → -1, enc=194 → -2, ...
        op.imm = (uint64_t)(int64_t)(-(int32_t)(enc - 192));
    } else if (enc == 240) { // 0.5
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0x3F000000;
        op.imm = bits;
    } else if (enc == 241) { // -0.5
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0xBF000000;
        op.imm = bits;
    } else if (enc == 242) { // 1.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0x3F800000;
        op.imm = bits;
    } else if (enc == 243) { // -1.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0xBF800000;
        op.imm = bits;
    } else if (enc == 244) { // 2.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0x40000000;
        op.imm = bits;
    } else if (enc == 245) { // -2.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0xC0000000;
        op.imm = bits;
    } else if (enc == 246) { // 4.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0x40800000;
        op.imm = bits;
    } else if (enc == 247) { // -4.0
        op.kind = Operand::Kind::INLINE_CONST;
        uint32_t bits = 0xC0800000;
        op.imm = bits;
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

// Decode SOP2 opcode
static Opcode sop2_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_ADD_U32;
        case 1:  return Opcode::S_SUB_U32;
        case 2:  return Opcode::S_ADD_I32;
        case 4:  return Opcode::S_ADDC_U32;
        case 12: return Opcode::S_AND_B32;
        case 13: return Opcode::S_OR_B32;
        case 15: return Opcode::S_AND_B64;
        case 16: return Opcode::S_LSHL_B32;
        case 18: return Opcode::S_LSHR_B32;
        case 36: return Opcode::S_MUL_I32;
        default: return Opcode::UNKNOWN;
    }
}

// Decode SOP1 opcode (field [15:8])
// GFX9 SOP1 opcode table (from AMD ISA manual):
// 0=s_mov_b32, 1=s_mov_b64, ..., 28=s_getpc_b64, 29=s_setpc_b64,
// 32=s_and_saveexec_b64, 33=s_or_saveexec_b64
static Opcode sop1_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_MOV_B32;
        case 1:  return Opcode::S_MOV_B64;
        case 28: return Opcode::S_GETPC_B64;      // 0x1C
        case 29: return Opcode::S_SETPC_B64;      // 0x1D
        case 32: return Opcode::S_AND_SAVEEXEC_B64; // 0x20
        case 33: return Opcode::S_OR_SAVEEXEC_B64;  // 0x21
        default: return Opcode::UNKNOWN;
    }
}

// Decode SOPP opcode (field [22:16])
static Opcode sopp_opcode(uint32_t op) {
    switch (op) {
        case 0:  return Opcode::S_NOP;
        case 1:  return Opcode::S_ENDPGM;
        case 2:  return Opcode::S_BRANCH;
        case 4:  return Opcode::S_CBRANCH_SCC0;
        case 5:  return Opcode::S_CBRANCH_SCC1;
        case 8:  return Opcode::S_CBRANCH_EXECZ;
        case 9:  return Opcode::S_CBRANCH_EXECNZ;
        case 12: return Opcode::S_WAITCNT;
        default: return Opcode::UNKNOWN;
    }
}

// Decode SMEM opcode (field [25:18])
static Opcode smem_opcode(uint32_t op) {
    switch (op) {
        case 0: return Opcode::S_LOAD_DWORD;
        case 1: return Opcode::S_LOAD_DWORDX2;
        case 2: return Opcode::S_LOAD_DWORDX4;
        case 3: return Opcode::S_LOAD_DWORDX8;
        case 4: return Opcode::S_LOAD_DWORDX16;
        default: return Opcode::UNKNOWN;
    }
}

// Decode VOP2 opcode (field [30:25])
// GFX9 VOP2 6-bit opcode table (verified from actual HSACO disassembly):
static Opcode vop2_opcode(uint32_t op) {
    switch (op) {
        case 1:  return Opcode::V_ADD_F32;
        case 5:  return Opcode::V_MUL_F32;
        case 8:  return Opcode::V_MUL_U32_U24;
        case 17: return Opcode::V_ASHRREV_I32;   // v_ashrrev_i32_e32 = op 17
        case 18: return Opcode::V_LSHLREV_B32;
        case 19: return Opcode::V_AND_B32;
        case 20: return Opcode::V_OR_B32;
        case 21: return Opcode::V_XOR_B32;
        case 22: return Opcode::V_LSHRREV_B32;
        case 25: return Opcode::V_ADD_CO_U32;    // v_add_co_u32_e32 (with VCC carry)
        case 26: return Opcode::V_SUB_U32;
        case 27: return Opcode::V_SUBREV_U32;
        case 28: return Opcode::V_ADDC_CO_U32;   // v_addc_co_u32_e32
        case 52: return Opcode::V_ADD_U32;        // v_add_u32_e32 (gfx9 no-carry version)
        default: return Opcode::UNKNOWN;
    }
}

// Decode VOP1 opcode (field [24:17])
static Opcode vop1_opcode(uint32_t op) {
    switch (op) {
        case 1:  return Opcode::V_MOV_B32;
        case 9:  return Opcode::V_CVT_F32_I32;
        case 11: return Opcode::V_CVT_I32_F32;
        default: return Opcode::UNKNOWN;
    }
}

// Decode VOPC opcode (field [24:17])
// GFX9 VOPC opcode space (8-bit):
//   0x40-0x7F = f32 comparisons
//   0xC0-0xC7 = i32 comparisons: 0xC0=f, 0xC1=lt, 0xC2=eq, 0xC3=le, 0xC4=gt, 0xC5=ne, 0xC6=ge, 0xC7=t
//   0xC8-0xCF = u32 comparisons: 0xC8=f, 0xC9=lt, 0xCA=eq, 0xCB=le, 0xCC=gt, 0xCD=ne, 0xCE=ge, 0xCF=t
static Opcode vopc_opcode(uint32_t op) {
    switch (op) {
        // i32 comparisons (0xC0+)
        case 0xC1: return Opcode::V_CMP_LT_I32;
        case 0xC2: return Opcode::V_CMP_EQ_I32;
        case 0xC3: return Opcode::V_CMP_LE_I32;
        case 0xC4: return Opcode::V_CMP_GT_I32;
        case 0xC5: return Opcode::V_CMP_NE_I32;
        case 0xC6: return Opcode::V_CMP_GE_I32;
        // u32 comparisons (0xC8+)
        case 0xC9: return Opcode::V_CMP_LT_U32;
        case 0xCA: return Opcode::V_CMP_EQ_U32;
        case 0xCB: return Opcode::V_CMP_LE_U32;
        case 0xCC: return Opcode::V_CMP_GT_U32;
        case 0xCD: return Opcode::V_CMP_NE_U32;
        case 0xCE: return Opcode::V_CMP_GE_U32;
        // Legacy aliases (keep old values in case older HSACO)
        case 0x91: return Opcode::V_CMP_LT_U32;
        case 0x93: return Opcode::V_CMP_LE_U32;
        case 0x94: return Opcode::V_CMP_GT_U32;
        case 0x96: return Opcode::V_CMP_GE_U32;
        case 0x92: return Opcode::V_CMP_EQ_U32;
        case 0x95: return Opcode::V_CMP_NE_U32;
        case 0x81: return Opcode::V_CMP_LT_I32;
        case 0x86: return Opcode::V_CMP_GE_I32;
        default:   return Opcode::UNKNOWN;
    }
}

// Decode VOP3 opcode (field [25:16], 10-bit)
// VOP3 opcode space: 0x140-0x17F = VOP2 re-encoded, 0x180-0x1BF = VOP1, 0x1C0-0x1FF = VOPC
// 0x200+ = true VOP3 ops
static Opcode vop3_opcode(uint32_t op) {
    // True VOP3 ops (GFX9)
    switch (op) {
        case 0x1E5: return Opcode::V_MUL_LO_U32;   // v_mul_lo_u32 = 0x285 in some tables
        case 0x285: return Opcode::V_MUL_LO_U32;
        case 0x28F: return Opcode::V_LSHLREV_B64;   // v_lshlrev_b64
        case 0x291: return Opcode::V_ASHRREV_I64;
        case 0x2F5: return Opcode::V_MAD_F32;
        // VOP3B — carry operations
        case 0x140: return Opcode::V_ADD_CO_U32;    // v_add_co_u32 VOP3B form
        case 0x141: return Opcode::V_SUB_U32;
        case 0x142: return Opcode::V_SUBREV_U32;
        case 0x143: return Opcode::V_ADDC_CO_U32;
        // VOP3A forms of VOPC (0x1C0 + VOPC_op)
        case 0x311: return Opcode::V_CMP_LT_U32;    // 0x1C0 + 0x91 = 0x251? Let me recalc
        default: {
            // VOP3A encoding of VOP2 ops: VOP3_op = 0x100 + VOP2_op (approximate)
            // VOP3A encoding of VOPC ops: base + VOPC_op
            // Try VOP3->VOPC mapping: op = 0x1C0 | vopc_op
            if (op >= 0x1C0 && op <= 0x1FF) {
                return vopc_opcode(op - 0x1C0);
            }
            // VOP3->VOP2 mapping
            if (op >= 0x140 && op <= 0x17F) {
                return vop2_opcode(op - 0x140);
            }
            // VOP3->VOP1 mapping
            if (op >= 0x180 && op <= 0x1BF) {
                return vop1_opcode(op - 0x180);
            }
            return Opcode::UNKNOWN;
        }
    }
}

// Decode FLAT opcode (field [25:18])
static Opcode flat_opcode(uint32_t op) {
    switch (op) {
        case 0x14: return Opcode::GLOBAL_LOAD_DWORD;
        case 0x15: return Opcode::GLOBAL_LOAD_DWORDX2;
        case 0x16: return Opcode::GLOBAL_LOAD_DWORDX4;
        case 0x1C: return Opcode::GLOBAL_STORE_DWORD;
        case 0x1D: return Opcode::GLOBAL_STORE_DWORDX2;
        case 0x1E: return Opcode::GLOBAL_STORE_DWORDX4;
        // Traditional flat ops (GFX9 uses same opcode space)
        case 0x0C: return Opcode::GLOBAL_LOAD_DWORD;
        case 0x0D: return Opcode::GLOBAL_LOAD_DWORDX2;
        case 0x0E: return Opcode::GLOBAL_LOAD_DWORDX4;
        case 0x1A: return Opcode::GLOBAL_STORE_DWORD;
        case 0x1B: return Opcode::GLOBAL_STORE_DWORDX2;
        default:   return Opcode::UNKNOWN;
    }
}

// Parse s_waitcnt immediate
static void parse_waitcnt(uint32_t simm16, uint8_t& vmcnt, uint8_t& lgkmcnt) {
    // GFX9 s_waitcnt encoding:
    // [15:14] = vmcnt_hi (2 bits)
    // [13:8]  = lgkmcnt (6 bits)
    // [7:4]   = expcnt (4 bits)
    // [3:0]   = vmcnt_lo (4 bits)
    uint8_t vmcnt_lo = simm16 & 0xF;
    uint8_t vmcnt_hi = (simm16 >> 14) & 0x3;
    vmcnt   = vmcnt_lo | (vmcnt_hi << 4);
    lgkmcnt = (simm16 >> 8) & 0x3F;
}

Instruction decode(const uint8_t* pc) {
    Instruction inst;
    uint32_t w0 = read_u32_le(pc);
    inst.encoding[0] = w0;

    uint32_t bits31_23 = (w0 >> 23) & 0x1FF;
    uint32_t bits31_26 = (w0 >> 26) & 0x3F;
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

    // --- SOP1: [31:23] == 0x17E ---
    // SOP1 layout: [31:23]=0x17E, [22:16]=SDST, [15:8]=OP, [7:0]=SSRC0
    if (bits31_23 == 0x17E || bits31_23 == 0x17D) {
        // 0x17E = 101111110 and 0x17D = 101111101
        // The marker for SOP1 is 0x17E per ISA docs
        // But our analysis shows 0x17D for s_mov_b32 and s_getpc_b64
        // Let's handle 0x17D as SOP1
        if (bits31_23 == 0x17D) {
            inst.format = InstFormat::SOP1;
            inst.size_bytes = 4;
            uint32_t sdst = (w0 >> 16) & 0x7F;
            uint32_t op   = (w0 >> 8) & 0xFF;
            uint32_t ssrc0 = w0 & 0xFF;
            inst.dst = decode_sdst(sdst);
            // Check if literal constant is used
            bool has_literal = (ssrc0 == 255);
            inst.src[0] = decode_src(ssrc0, has_literal ? (pc + 4) : nullptr);
            if (has_literal) inst.size_bytes = 8;
            inst.opcode = sop1_opcode(op);
            return inst;
        }
    }

    // --- SOPC: [31:23] == 0x17C ---
    if (bits31_23 == 0x17C) {
        inst.format = InstFormat::SOPC;
        inst.size_bytes = 4;
        return inst;
    }

    // --- SMEM: [31:26] == 0x30 ---
    if (bits31_26 == 0x30) {
        inst.format = InstFormat::SMEM;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op    = (w0 >> 18) & 0xFF;
        uint32_t sdst  = (w0 >> 6)  & 0x7F;   // SDATA at [12:6]
        uint32_t sbase = w0 & 0x3F;            // SBASE at [5:0], actual sgpr = sbase * 2
        uint32_t imm_flag = (w0 >> 17) & 1;   // IMM bit at bit 17

        inst.opcode = smem_opcode(op);
        inst.dst.kind = Operand::Kind::SGPR;
        inst.dst.index = sdst;
        inst.src[0].kind = Operand::Kind::SGPR;
        inst.src[0].index = sbase * 2;

        if (imm_flag) {
            inst.smem_offset = (int32_t)w1;
        } else {
            inst.smem_offset = 0;
        }
        return inst;
    }

    // --- FLAT/GLOBAL: [31:26] == 0x37 ---
    if (bits31_26 == 0x37) {
        inst.format = InstFormat::FLAT_GLOBAL;
        inst.size_bytes = 8;
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;

        uint32_t op    = (w0 >> 18) & 0xFF;
        // word1 layout: [31:24]=vdst, [23:16]=saddr, [15:8]=vdata, [7:0]=vaddr
        uint32_t vdst  = (w1 >> 24) & 0xFF;
        uint32_t saddr = (w1 >> 16) & 0xFF;
        uint32_t vdata = (w1 >> 8)  & 0xFF;
        uint32_t vaddr = (w1 >> 0)  & 0xFF;

        inst.opcode = flat_opcode(op);

        bool is_store = (inst.opcode == Opcode::GLOBAL_STORE_DWORD ||
                         inst.opcode == Opcode::GLOBAL_STORE_DWORDX2 ||
                         inst.opcode == Opcode::GLOBAL_STORE_DWORDX4);

        if (!is_store) {
            // Load: vdst = result, src[0] = vaddr (vgpr for address), src[1] = saddr (sgpr)
            inst.dst.kind = Operand::Kind::VGPR;
            inst.dst.index = vdst;
            inst.src[0].kind = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                // "off" — pure 64-bit vaddr in a pair
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind = Operand::Kind::SGPR;
                inst.src[1].index = saddr * 2;  // saddr is SGPR pair index
            }
        } else {
            // Store: data=vdata, addr=vaddr, saddr=base
            inst.src[0].kind = Operand::Kind::VGPR;
            inst.src[0].index = vaddr;
            if (saddr == 0x7F) {
                inst.src[1].kind = Operand::Kind::NONE;
            } else {
                inst.src[1].kind = Operand::Kind::SGPR;
                inst.src[1].index = saddr * 2;
            }
            // vdata is the data to store; for stores vdst field holds the source vgpr
            // Actually: in stores vdata at [23:16] is the source, vdst at [31:24] is unused (0x7F=off)
            inst.src[2].kind = Operand::Kind::VGPR;
            inst.src[2].index = vdata;
        }

        inst.flat_offset = (int16_t)(w0 & 0x1FFF); // 13-bit signed offset
        return inst;
    }

    // --- VOP3A/VOP3B: [31:26] == 0x34 ---
    if (bits31_26 == 0x34) {
        uint32_t w1 = read_u32_le(pc + 4);
        inst.encoding[1] = w1;
        inst.size_bytes = 8;

        uint32_t op   = (w0 >> 16) & 0x3FF;  // 10-bit opcode
        uint32_t vdst = w0 & 0xFF;
        uint32_t sdst = (w0 >> 8) & 0x7F;    // for VOP3B

        uint32_t src0 = w1 & 0x1FF;
        uint32_t src1 = (w1 >> 9) & 0x1FF;
        uint32_t src2 = (w1 >> 18) & 0x1FF;
        uint32_t neg  = (w1 >> 29) & 0x7;
        (void)neg;

        inst.opcode = vop3_opcode(op);

        // Determine if VOP3B (has SDST destination for carry)
        // VOP3B ops in GFX9: v_add_co_u32, v_sub_co_u32, v_subrev_co_u32, v_addc_co_u32
        bool is_vop3b = (inst.opcode == Opcode::V_ADD_CO_U32 ||
                         inst.opcode == Opcode::V_SUB_U32 ||
                         inst.opcode == Opcode::V_SUBREV_U32 ||
                         inst.opcode == Opcode::V_ADDC_CO_U32);

        if (is_vop3b) {
            inst.format = InstFormat::VOP3B;
            inst.dst.kind = Operand::Kind::VGPR;
            inst.dst.index = vdst;
            inst.dst2 = decode_sdst(sdst);
        } else {
            inst.format = InstFormat::VOP3A;
            inst.dst.kind = Operand::Kind::VGPR;
            inst.dst.index = vdst;
        }

        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1] = decode_src(src1, nullptr);
        inst.src[2] = decode_src(src2, nullptr);
        return inst;
    }

    // --- VOP1: [31:25] == 0x3F ---
    // VOP1 layout: [31:25]=0x3F marker, [24:17]=VDST(8-bit), [16:9]=OP(8-bit), [8:0]=SRC0(9-bit)
    if (bits31_25 == 0x3F) {
        inst.format = InstFormat::VOP1;
        inst.size_bytes = 4;
        uint32_t vdst = (w0 >> 17) & 0xFF;
        uint32_t op   = (w0 >> 9)  & 0xFF;
        uint32_t src0 = w0 & 0x1FF;

        inst.opcode = vop1_opcode(op);
        inst.dst = decode_vdst(vdst);
        bool has_literal = (src0 == 255);
        inst.src[0] = decode_src(src0, has_literal ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;
        return inst;
    }

    // --- VOPC: [31:25] == 0x3E ---
    if (bits31_25 == 0x3E) {
        inst.format = InstFormat::VOPC;
        inst.size_bytes = 4;
        uint32_t op   = (w0 >> 17) & 0xFF;
        uint32_t vsrc1 = (w0 >> 9) & 0xFF;
        uint32_t src0  = w0 & 0x1FF;

        inst.opcode = vopc_opcode(op);
        // Result goes implicitly to VCC
        inst.dst.kind = Operand::Kind::VCC;
        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1].kind = Operand::Kind::VGPR;
        inst.src[1].index = vsrc1;
        return inst;
    }

    // --- SOP2: [31:30] == 0b10 (and not already handled) ---
    if (bits31_30 == 0x2) {
        inst.format = InstFormat::SOP2;
        inst.size_bytes = 4;
        uint32_t op   = (w0 >> 23) & 0x7F;
        uint32_t sdst = (w0 >> 16) & 0x7F;
        uint32_t ssrc1 = (w0 >> 8) & 0xFF;
        uint32_t ssrc0 = w0 & 0xFF;

        inst.opcode = sop2_opcode(op);
        inst.dst = decode_sdst(sdst);

        bool has_literal = (ssrc0 == 255 || ssrc1 == 255);
        inst.src[0] = decode_src(ssrc0, (ssrc0 == 255) ? (pc + 4) : nullptr);
        inst.src[1] = decode_src(ssrc1, (ssrc1 == 255) ? (pc + 4) : nullptr);
        if (has_literal) inst.size_bytes = 8;
        return inst;
    }

    // --- VOP2: fallback (bit 31 == 0) ---
    // VOP2 layout: bit31=0, [30:25]=op(6-bit), [24:17]=vdst(8-bit), [16:9]=vsrc1(8-bit), [8:0]=src0(9-bit)
    if ((w0 >> 31) == 0) {
        inst.format = InstFormat::VOP2;
        inst.size_bytes = 4;
        uint32_t op    = (w0 >> 25) & 0x3F;
        uint32_t vdst  = (w0 >> 17) & 0xFF;
        uint32_t vsrc1 = (w0 >> 9)  & 0xFF;
        uint32_t src0  = w0 & 0x1FF;

        inst.opcode = vop2_opcode(op);
        inst.dst = decode_vdst(vdst);
        inst.src[0] = decode_src(src0, nullptr);
        inst.src[1].kind = Operand::Kind::VGPR;
        inst.src[1].index = vsrc1;

        bool has_literal = (src0 == 255);
        if (has_literal) {
            inst.src[0] = decode_src(src0, pc + 4);
            inst.size_bytes = 8;
        }
        return inst;
    }

    // Unknown
    inst.format = InstFormat::UNKNOWN;
    inst.opcode = Opcode::UNKNOWN;
    inst.size_bytes = 4;
    return inst;
}

const char* opcode_name(Opcode op) {
    switch (op) {
        case Opcode::S_ADD_U32:           return "s_add_u32";
        case Opcode::S_ADD_I32:           return "s_add_i32";
        case Opcode::S_ADDC_U32:          return "s_addc_u32";
        case Opcode::S_SUB_U32:           return "s_sub_u32";
        case Opcode::S_AND_B32:           return "s_and_b32";
        case Opcode::S_OR_B32:            return "s_or_b32";
        case Opcode::S_LSHL_B32:          return "s_lshl_b32";
        case Opcode::S_LSHR_B32:          return "s_lshr_b32";
        case Opcode::S_MUL_I32:           return "s_mul_i32";
        case Opcode::S_MOV_B32:           return "s_mov_b32";
        case Opcode::S_MOV_B64:           return "s_mov_b64";
        case Opcode::S_AND_B64:           return "s_and_b64";
        case Opcode::S_AND_SAVEEXEC_B64:  return "s_and_saveexec_b64";
        case Opcode::S_OR_SAVEEXEC_B64:   return "s_or_saveexec_b64";
        case Opcode::S_GETPC_B64:         return "s_getpc_b64";
        case Opcode::S_SETPC_B64:         return "s_setpc_b64";
        case Opcode::S_CBRANCH_EXECZ:     return "s_cbranch_execz";
        case Opcode::S_CBRANCH_EXECNZ:    return "s_cbranch_execnz";
        case Opcode::S_CBRANCH_SCC0:      return "s_cbranch_scc0";
        case Opcode::S_CBRANCH_SCC1:      return "s_cbranch_scc1";
        case Opcode::S_BRANCH:            return "s_branch";
        case Opcode::S_WAITCNT:           return "s_waitcnt";
        case Opcode::S_ENDPGM:            return "s_endpgm";
        case Opcode::S_NOP:               return "s_nop";
        case Opcode::S_LOAD_DWORD:        return "s_load_dword";
        case Opcode::S_LOAD_DWORDX2:      return "s_load_dwordx2";
        case Opcode::S_LOAD_DWORDX4:      return "s_load_dwordx4";
        case Opcode::S_LOAD_DWORDX8:      return "s_load_dwordx8";
        case Opcode::S_LOAD_DWORDX16:     return "s_load_dwordx16";
        case Opcode::V_MOV_B32:           return "v_mov_b32";
        case Opcode::V_ADD_U32:           return "v_add_u32";
        case Opcode::V_ADD_CO_U32:        return "v_add_co_u32";
        case Opcode::V_ADDC_CO_U32:       return "v_addc_co_u32";
        case Opcode::V_SUB_U32:           return "v_sub_u32";
        case Opcode::V_SUBREV_U32:        return "v_subrev_u32";
        case Opcode::V_MUL_U32_U24:       return "v_mul_u32_u24";
        case Opcode::V_MUL_LO_U32:        return "v_mul_lo_u32";
        case Opcode::V_AND_B32:           return "v_and_b32";
        case Opcode::V_OR_B32:            return "v_or_b32";
        case Opcode::V_XOR_B32:           return "v_xor_b32";
        case Opcode::V_LSHLREV_B32:       return "v_lshlrev_b32";
        case Opcode::V_LSHRREV_B32:       return "v_lshrrev_b32";
        case Opcode::V_ASHRREV_I32:       return "v_ashrrev_i32";
        case Opcode::V_ASHRREV_I64:       return "v_ashrrev_i64";
        case Opcode::V_LSHLREV_B64:       return "v_lshlrev_b64";
        case Opcode::V_ADD_F32:           return "v_add_f32";
        case Opcode::V_MUL_F32:           return "v_mul_f32";
        case Opcode::V_MAD_F32:           return "v_mad_f32";
        case Opcode::V_CVT_F32_I32:       return "v_cvt_f32_i32";
        case Opcode::V_CVT_I32_F32:       return "v_cvt_i32_f32";
        case Opcode::V_CMP_LT_U32:        return "v_cmp_lt_u32";
        case Opcode::V_CMP_LE_U32:        return "v_cmp_le_u32";
        case Opcode::V_CMP_GT_U32:        return "v_cmp_gt_u32";
        case Opcode::V_CMP_GE_U32:        return "v_cmp_ge_u32";
        case Opcode::V_CMP_EQ_U32:        return "v_cmp_eq_u32";
        case Opcode::V_CMP_NE_U32:        return "v_cmp_ne_u32";
        case Opcode::V_CMP_LT_I32:        return "v_cmp_lt_i32";
        case Opcode::V_CMP_LE_I32:        return "v_cmp_le_i32";
        case Opcode::V_CMP_GT_I32:        return "v_cmp_gt_i32";
        case Opcode::V_CMP_GE_I32:        return "v_cmp_ge_i32";
        case Opcode::V_CMP_EQ_I32:        return "v_cmp_eq_i32";
        case Opcode::V_CMP_NE_I32:        return "v_cmp_ne_i32";
        case Opcode::GLOBAL_LOAD_DWORD:   return "global_load_dword";
        case Opcode::GLOBAL_LOAD_DWORDX2: return "global_load_dwordx2";
        case Opcode::GLOBAL_LOAD_DWORDX4: return "global_load_dwordx4";
        case Opcode::GLOBAL_STORE_DWORD:  return "global_store_dword";
        case Opcode::GLOBAL_STORE_DWORDX2:return "global_store_dwordx2";
        case Opcode::GLOBAL_STORE_DWORDX4:return "global_store_dwordx4";
        case Opcode::UNKNOWN:             return "unknown";
        default:                          return "?";
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
        case InstFormat::SMEM:        return "SMEM";
        case InstFormat::FLAT_GLOBAL: return "FLAT_GLOBAL";
        case InstFormat::UNKNOWN:     return "UNKNOWN";
        default:                      return "?";
    }
}

} // namespace gcnsim
