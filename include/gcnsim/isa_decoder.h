#pragma once
#include <cstdint>
#include <string>

namespace gcnsim {

enum class InstFormat {
    SOP2, SOP1, SOPK, SOPC, SOPP,
    VOP2, VOP1, VOPC, VOP3A, VOP3B,
    SMEM, FLAT_GLOBAL, UNKNOWN
};

enum class Opcode {
    // Scalar ALU
    S_ADD_U32,
    S_ADD_I32,
    S_ADDC_U32,
    S_SUB_U32,
    S_AND_B32,
    S_OR_B32,
    S_LSHL_B32,
    S_LSHR_B32,
    S_MUL_I32,
    S_MOV_B32,
    S_MOV_B64,
    S_AND_B64,
    S_AND_SAVEEXEC_B64,
    S_OR_SAVEEXEC_B64,
    S_GETPC_B64,
    S_SETPC_B64,
    // Scalar control
    S_CBRANCH_EXECZ,
    S_CBRANCH_EXECNZ,
    S_CBRANCH_SCC0,
    S_CBRANCH_SCC1,
    S_BRANCH,
    S_WAITCNT,
    S_ENDPGM,
    S_NOP,
    // Scalar memory
    S_LOAD_DWORD,
    S_LOAD_DWORDX2,
    S_LOAD_DWORDX4,
    S_LOAD_DWORDX8,
    S_LOAD_DWORDX16,
    // Vector ALU
    V_MOV_B32,
    V_ADD_U32,
    V_ADD_CO_U32,
    V_ADDC_CO_U32,
    V_SUB_U32,
    V_SUBREV_U32,
    V_MUL_U32_U24,
    V_MUL_LO_U32,
    V_AND_B32,
    V_OR_B32,
    V_XOR_B32,
    V_LSHLREV_B32,
    V_LSHRREV_B32,
    V_ASHRREV_I32,
    V_ASHRREV_I64,
    V_ADD_F32,
    V_MUL_F32,
    V_MAD_F32,
    V_CVT_F32_I32,
    V_CVT_I32_F32,
    V_LSHLREV_B64,
    // Vector compare
    V_CMP_LT_U32,
    V_CMP_LE_U32,
    V_CMP_GT_U32,
    V_CMP_GE_U32,
    V_CMP_EQ_U32,
    V_CMP_NE_U32,
    V_CMP_LT_I32,
    V_CMP_LE_I32,
    V_CMP_GT_I32,
    V_CMP_GE_I32,
    V_CMP_EQ_I32,
    V_CMP_NE_I32,
    // Flat/Global memory
    GLOBAL_LOAD_DWORD,
    GLOBAL_LOAD_DWORDX2,
    GLOBAL_LOAD_DWORDX4,
    GLOBAL_STORE_DWORD,
    GLOBAL_STORE_DWORDX2,
    GLOBAL_STORE_DWORDX4,
    UNKNOWN
};

struct Operand {
    enum class Kind {
        SGPR,
        VGPR,
        LITERAL,
        INLINE_CONST,
        VCC,
        VCC_LO,
        VCC_HI,
        EXEC,
        EXEC_LO,
        EXEC_HI,
        SCC,
        NONE
    };
    Kind     kind  = Kind::NONE;
    uint32_t index = 0;    // reg index for SGPR/VGPR
    uint64_t imm   = 0;    // value for LITERAL/INLINE_CONST
};

struct Instruction {
    InstFormat format    = InstFormat::UNKNOWN;
    Opcode     opcode    = Opcode::UNKNOWN;
    uint32_t   encoding[2] = {0, 0};
    uint8_t    size_bytes  = 4;   // 4 or 8

    Operand dst;     // primary destination
    Operand dst2;    // VOP3B second destination (carry-out)
    Operand src[3];  // up to 3 sources

    int16_t  simm16 = 0;        // SOPP branch offset / SOPK immediate
    uint8_t  wait_vmcnt    = 0;
    uint8_t  wait_lgkmcnt  = 0;
    int32_t  smem_offset   = 0;
    int32_t  flat_offset   = 0;
};

// Decode one instruction at pc; returns decoded struct.
// size_bytes is 4 or 8 — caller advances PC by that amount.
Instruction decode(const uint8_t* pc);

const char* opcode_name(Opcode op);
const char* format_name(InstFormat fmt);

} // namespace gcnsim
