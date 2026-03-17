#include "gcnsim/wavefront.h"
#include "gcnsim/isa_decoder.h"
#include "gcnsim/memory_model.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cstdio>

namespace gcnsim {

// ---------------------------------------------------------------------------
// Helpers: read scalar / vector sources
// ---------------------------------------------------------------------------

static uint32_t read_src_scalar(const WavefrontState& w, const Operand& op) {
    switch (op.kind) {
        case Operand::Kind::SGPR:       return w.sgpr[op.index];
        case Operand::Kind::VCC:
        case Operand::Kind::VCC_LO:    return (uint32_t)(w.vcc & 0xFFFFFFFFULL);
        case Operand::Kind::VCC_HI:    return (uint32_t)(w.vcc >> 32);
        case Operand::Kind::EXEC:
        case Operand::Kind::EXEC_LO:   return (uint32_t)(w.exec & 0xFFFFFFFFULL);
        case Operand::Kind::EXEC_HI:   return (uint32_t)(w.exec >> 32);
        case Operand::Kind::SCC:       return w.scc;
        case Operand::Kind::INLINE_CONST:
        case Operand::Kind::LITERAL:   return (uint32_t)op.imm;
        default:                       return 0;
    }
}

static uint64_t read_src_scalar64(const WavefrontState& w, const Operand& op) {
    if (op.kind == Operand::Kind::SGPR) {
        return ((uint64_t)w.sgpr[op.index + 1] << 32) | w.sgpr[op.index];
    }
    if (op.kind == Operand::Kind::VCC || op.kind == Operand::Kind::VCC_LO) {
        return w.vcc;
    }
    if (op.kind == Operand::Kind::EXEC || op.kind == Operand::Kind::EXEC_LO) {
        return w.exec;
    }
    return (uint64_t)read_src_scalar(w, op);
}

static uint32_t read_vsrc(const WavefrontState& w, const Operand& op, int lane) {
    switch (op.kind) {
        case Operand::Kind::VGPR:
            return w.vgpr[op.index][lane];
        case Operand::Kind::SGPR:
            return w.sgpr[op.index];
        case Operand::Kind::VCC:
        case Operand::Kind::VCC_LO:
            return ((w.vcc >> lane) & 1) ? 1 : 0;
        case Operand::Kind::EXEC:
        case Operand::Kind::EXEC_LO:
            return ((w.exec >> lane) & 1) ? 1 : 0;
        case Operand::Kind::INLINE_CONST:
        case Operand::Kind::LITERAL:
            return (uint32_t)op.imm;
        default:
            return 0;
    }
}

// Write scalar dst (scalar register or VCC/EXEC pair)
static void write_sdst(WavefrontState& w, const Operand& op, uint32_t val) {
    switch (op.kind) {
        case Operand::Kind::SGPR:
            w.sgpr[op.index] = val;
            break;
        case Operand::Kind::VCC:
        case Operand::Kind::VCC_LO:
            w.vcc = (w.vcc & ~0xFFFFFFFFULL) | val;
            break;
        case Operand::Kind::VCC_HI:
            w.vcc = (w.vcc & 0xFFFFFFFFULL) | ((uint64_t)val << 32);
            break;
        case Operand::Kind::EXEC:
        case Operand::Kind::EXEC_LO:
            w.exec = (w.exec & ~0xFFFFFFFFULL) | val;
            break;
        default: break;
    }
}

static void write_sdst64(WavefrontState& w, const Operand& op, uint64_t val) {
    if (op.kind == Operand::Kind::SGPR) {
        w.sgpr[op.index]     = (uint32_t)(val & 0xFFFFFFFF);
        w.sgpr[op.index + 1] = (uint32_t)(val >> 32);
    } else if (op.kind == Operand::Kind::VCC || op.kind == Operand::Kind::VCC_LO) {
        w.vcc = val;
    } else if (op.kind == Operand::Kind::EXEC || op.kind == Operand::Kind::EXEC_LO) {
        w.exec = val;
    }
}

// ---------------------------------------------------------------------------
// Main execute dispatch
// ---------------------------------------------------------------------------

void execute_instruction(WavefrontState& wave, const Instruction& inst);

// Helper: run one wavefront until s_endpgm
void run_wavefront(WavefrontState& wave) {
    static constexpr uint64_t MAX_INSNS = 1000000ULL; // safety limit
    uint64_t count = 0;
    while (true) {
        if (wave.pc >= wave.kernel_code_size) {
            // Out of bounds — treat as endpgm
            break;
        }
        Instruction inst = decode(wave.kernel_code_ptr + wave.pc);
        wave.pc += inst.size_bytes;
        execute_instruction(wave, inst);
        if (inst.opcode == Opcode::S_ENDPGM) break;
        if (++count >= MAX_INSNS) {
            // Runaway kernel — abort
            break;
        }
    }
}

void execute_instruction(WavefrontState& wave, const Instruction& inst) {
    switch (inst.opcode) {

    // -----------------------------------------------------------------------
    // Scalar ALU
    // -----------------------------------------------------------------------
    case Opcode::S_MOV_B32: {
        uint32_t val = read_src_scalar(wave, inst.src[0]);
        write_sdst(wave, inst.dst, val);
        break;
    }
    case Opcode::S_MOV_B64: {
        uint64_t val = read_src_scalar64(wave, inst.src[0]);
        write_sdst64(wave, inst.dst, val);
        break;
    }
    case Opcode::S_ADD_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint64_t r = (uint64_t)a + b;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r > 0xFFFFFFFFULL) ? 1 : 0;
        break;
    }
    case Opcode::S_ADDC_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint64_t r = (uint64_t)a + b + wave.scc;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r > 0xFFFFFFFFULL) ? 1 : 0;
        break;
    }
    case Opcode::S_ADD_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        int64_t r = (int64_t)a + b;
        write_sdst(wave, inst.dst, (uint32_t)(int32_t)r);
        // SCC = overflow
        wave.scc = ((r > 0x7FFFFFFF) || (r < -(int64_t)0x80000000LL)) ? 1 : 0;
        break;
    }
    case Opcode::S_SUB_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        write_sdst(wave, inst.dst, a - b);
        wave.scc = (b > a) ? 1 : 0;
        break;
    }
    case Opcode::S_AND_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = a & b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_OR_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = a | b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_AND_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t b = read_src_scalar64(wave, inst.src[1]);
        uint64_t r = a & b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_LSHL_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 31;
        uint32_t r = a << b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_LSHR_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 31;
        uint32_t r = a >> b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_MUL_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        write_sdst(wave, inst.dst, (uint32_t)(a * b));
        break;
    }
    case Opcode::S_SUBB_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint64_t r = (uint64_t)a - b - wave.scc;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (b + wave.scc > a) ? 1 : 0;
        break;
    }
    case Opcode::S_MIN_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        int32_t r = (a < b) ? a : b;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (a < b) ? 1 : 0;
        break;
    }
    case Opcode::S_MIN_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = (a < b) ? a : b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (a < b) ? 1 : 0;
        break;
    }
    case Opcode::S_MAX_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        int32_t r = (a > b) ? a : b;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (a > b) ? 1 : 0;
        break;
    }
    case Opcode::S_MAX_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = (a > b) ? a : b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (a > b) ? 1 : 0;
        break;
    }
    case Opcode::S_OR_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t b = read_src_scalar64(wave, inst.src[1]);
        uint64_t r = a | b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_XOR_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = a ^ b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_XOR_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t b = read_src_scalar64(wave, inst.src[1]);
        uint64_t r = a ^ b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ANDN2_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = a & ~b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ANDN2_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t b = read_src_scalar64(wave, inst.src[1]);
        uint64_t r = a & ~b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ORN2_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = a | ~b;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ORN2_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t b = read_src_scalar64(wave, inst.src[1]);
        uint64_t r = a | ~b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_NAND_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = ~(a & b);
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_NOR_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = ~(a | b);
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_XNOR_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        uint32_t r = ~(a ^ b);
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_LSHL_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 63;
        uint64_t r = a << b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_LSHR_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 63;
        uint64_t r = a >> b;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ASHR_I32: {
        int32_t  a = (int32_t)read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 31;
        int32_t  r = a >> b;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ASHR_I64: {
        int64_t  a = (int64_t)read_src_scalar64(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]) & 63;
        int64_t  r = a >> b;
        write_sdst64(wave, inst.dst, (uint64_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_BFM_B32: {
        uint32_t width  = read_src_scalar(wave, inst.src[0]) & 31;
        uint32_t offset = read_src_scalar(wave, inst.src[1]) & 31;
        uint32_t r = ((1u << width) - 1u) << offset;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_BFE_U32: {
        uint32_t base   = read_src_scalar(wave, inst.src[0]);
        uint32_t ctrl   = read_src_scalar(wave, inst.src[1]);
        uint32_t offset = ctrl & 0x1F;
        uint32_t width  = (ctrl >> 16) & 0x7F;
        uint32_t r = (width == 0) ? 0 : ((base >> offset) & ((1u << width) - 1u));
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_BFE_I32: {
        uint32_t base   = read_src_scalar(wave, inst.src[0]);
        uint32_t ctrl   = read_src_scalar(wave, inst.src[1]);
        uint32_t offset = ctrl & 0x1F;
        uint32_t width  = (ctrl >> 16) & 0x7F;
        uint32_t raw    = (width == 0) ? 0 : ((base >> offset) & ((1u << width) - 1u));
        // Sign extend
        if (width > 0 && width < 32) {
            uint32_t sign_bit = 1u << (width - 1);
            raw = (raw ^ sign_bit) - sign_bit;
        }
        write_sdst(wave, inst.dst, raw);
        wave.scc = ((int32_t)raw != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ABSDIFF_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        int32_t r = (a >= b) ? (a - b) : (b - a);
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    // -----------------------------------------------------------------------
    // Scalar ALU — SOP1 gaps
    // -----------------------------------------------------------------------
    case Opcode::S_NOT_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t r = ~a;
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_NOT_B64: {
        uint64_t a = read_src_scalar64(wave, inst.src[0]);
        uint64_t r = ~a;
        write_sdst64(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_BREV_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        // Bit-reverse
        uint32_t r = 0;
        for (int i = 0; i < 32; i++) r |= ((a >> i) & 1) << (31 - i);
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_BCNT1_I32_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t r = 0;
        while (a) { r += a & 1; a >>= 1; }
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_FF1_I32_B32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t r = 0xFFFFFFFF;
        for (uint32_t i = 0; i < 32; i++) {
            if ((a >> i) & 1) { r = i; break; }
        }
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0xFFFFFFFF) ? 1 : 0;
        break;
    }
    case Opcode::S_FLBIT_I32_B32: {
        // Count leading zeros
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t r = (a == 0) ? 32 : 0;
        if (a != 0) {
            for (int i = 31; i >= 0; i--) {
                if ((a >> i) & 1) { r = 31 - i; break; }
            }
        }
        write_sdst(wave, inst.dst, r);
        wave.scc = (r != 0xFFFFFFFF) ? 1 : 0;
        break;
    }
    case Opcode::S_SEXT_I32_I8: {
        uint32_t a = read_src_scalar(wave, inst.src[0]) & 0xFF;
        int32_t  r = (int32_t)(int8_t)a;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_SEXT_I32_I16: {
        uint32_t a = read_src_scalar(wave, inst.src[0]) & 0xFFFF;
        int32_t  r = (int32_t)(int16_t)a;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ABS_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t r = (a < 0) ? -a : a;
        write_sdst(wave, inst.dst, (uint32_t)r);
        wave.scc = (r != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_XOR_SAVEEXEC_B64: {
        uint64_t old_exec = wave.exec;
        write_sdst64(wave, inst.dst, old_exec);
        wave.exec = old_exec ^ wave.vcc;
        wave.scc = (wave.exec != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ANDN2_SAVEEXEC_B64: {
        uint64_t old_exec = wave.exec;
        write_sdst64(wave, inst.dst, old_exec);
        wave.exec = old_exec & ~wave.vcc;
        wave.scc = (wave.exec != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_ORN2_SAVEEXEC_B64: {
        uint64_t old_exec = wave.exec;
        write_sdst64(wave, inst.dst, old_exec);
        wave.exec = old_exec | ~wave.vcc;
        wave.scc = (wave.exec != 0) ? 1 : 0;
        break;
    }
    // -----------------------------------------------------------------------
    // Scalar ALU — SOPK
    // -----------------------------------------------------------------------
    case Opcode::S_MOVK_I32: {
        write_sdst(wave, inst.dst, (uint32_t)(int32_t)inst.simm16);
        break;
    }
    case Opcode::S_ADDK_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        int32_t b = (int32_t)inst.simm16;
        int64_t r = (int64_t)a + b;
        write_sdst(wave, inst.dst, (uint32_t)(int32_t)r);
        wave.scc = ((r > 0x7FFFFFFF) || (r < -(int64_t)0x80000000LL)) ? 1 : 0;
        break;
    }
    case Opcode::S_MULK_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        int32_t b = (int32_t)inst.simm16;
        write_sdst(wave, inst.dst, (uint32_t)(a * b));
        break;
    }
    case Opcode::S_CMOVK_I32: {
        if (wave.scc) write_sdst(wave, inst.dst, (uint32_t)(int32_t)inst.simm16);
        break;
    }
    case Opcode::S_CMPK_EQ_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a == (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LG_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a != (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_GT_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a > (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_GE_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a >= (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LT_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a < (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LE_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.dst);
        wave.scc = (a <= (int32_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_EQ_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a == (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LG_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a != (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_GT_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a > (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_GE_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a >= (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LT_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a < (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_CMPK_LE_U32: {
        uint32_t a = read_src_scalar(wave, inst.dst);
        wave.scc = (a <= (uint32_t)(uint16_t)inst.simm16) ? 1 : 0;
        break;
    }
    case Opcode::S_GETREG_B32:
    case Opcode::S_SETREG_B32:
    case Opcode::S_SETREG_IMM32_B32: {
        // Hardware register access — no-op in simulation
        break;
    }
    // -----------------------------------------------------------------------
    // Scalar ALU — SOPC
    // -----------------------------------------------------------------------
    case Opcode::S_CMP_EQ_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a == b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LG_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a != b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_GT_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a > b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_GE_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a >= b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LT_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a < b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LE_I32: {
        int32_t a = (int32_t)read_src_scalar(wave, inst.src[0]);
        int32_t b = (int32_t)read_src_scalar(wave, inst.src[1]);
        wave.scc = (a <= b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_EQ_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a == b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LG_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a != b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_GT_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a > b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_GE_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a >= b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LT_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a < b) ? 1 : 0;
        break;
    }
    case Opcode::S_CMP_LE_U32: {
        uint32_t a = read_src_scalar(wave, inst.src[0]);
        uint32_t b = read_src_scalar(wave, inst.src[1]);
        wave.scc = (a <= b) ? 1 : 0;
        break;
    }
    // -----------------------------------------------------------------------
    // SOPP — new branches
    // -----------------------------------------------------------------------
    case Opcode::S_CBRANCH_VCCZ: {
        if (wave.vcc == 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_CBRANCH_VCCNZ: {
        if (wave.vcc != 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_BARRIER:
    case Opcode::S_SETHALT:
    case Opcode::S_SLEEP:
    case Opcode::S_SENDMSG:
    case Opcode::S_SENDMSGHALT:
    case Opcode::S_TRAP:
    case Opcode::S_ICACHE_INV:
    case Opcode::S_INCPERFLEVEL:
    case Opcode::S_DECPERFLEVEL:
    case Opcode::S_TTRACEDATA:
    case Opcode::S_CBRANCH_CDBGSYS:
    case Opcode::S_CBRANCH_CDBGUSER:
    case Opcode::S_WAKEUP: {
        // No-op in simulation
        break;
    }
    // -----------------------------------------------------------------------
    // SMEM — stores and atomics
    // -----------------------------------------------------------------------
    case Opcode::S_STORE_DWORD: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        mem_store_u32(addr, wave.sgpr[inst.dst.index]);
        break;
    }
    case Opcode::S_STORE_DWORDX2: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        mem_store_u32(addr,     wave.sgpr[inst.dst.index]);
        mem_store_u32(addr + 4, wave.sgpr[inst.dst.index + 1]);
        break;
    }
    case Opcode::S_STORE_DWORDX4: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        for (int i = 0; i < 4; i++)
            mem_store_u32(addr + i * 4, wave.sgpr[inst.dst.index + i]);
        break;
    }
    case Opcode::S_ATOMIC_ADD: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t val = mem_load_u32(addr);
        uint32_t addend = wave.sgpr[inst.dst.index];
        mem_store_u32(addr, val + addend);
        break;
    }
    case Opcode::S_ATOMIC_SUB: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t val = mem_load_u32(addr);
        mem_store_u32(addr, val - wave.sgpr[inst.dst.index]);
        break;
    }
    case Opcode::S_ATOMIC_AND: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t val = mem_load_u32(addr);
        mem_store_u32(addr, val & wave.sgpr[inst.dst.index]);
        break;
    }
    case Opcode::S_ATOMIC_OR: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t val = mem_load_u32(addr);
        mem_store_u32(addr, val | wave.sgpr[inst.dst.index]);
        break;
    }
    case Opcode::S_ATOMIC_XOR: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t val = mem_load_u32(addr);
        mem_store_u32(addr, val ^ wave.sgpr[inst.dst.index]);
        break;
    }
    case Opcode::S_ATOMIC_SWAP: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t old = mem_load_u32(addr);
        mem_store_u32(addr, wave.sgpr[inst.dst.index]);
        wave.sgpr[inst.dst.index] = old;
        break;
    }
    case Opcode::S_ATOMIC_CMPSWAP: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        uint32_t old = mem_load_u32(addr);
        uint32_t cmp = wave.sgpr[inst.dst.index + 1];
        uint32_t val = wave.sgpr[inst.dst.index];
        if (old == cmp) mem_store_u32(addr, val);
        wave.sgpr[inst.dst.index] = old;
        break;
    }
    case Opcode::S_BUFFER_LOAD_DWORD: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        wave.sgpr[inst.dst.index] = mem_load_u32(addr);
        break;
    }
    case Opcode::S_BUFFER_LOAD_DWORDX2: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        wave.sgpr[inst.dst.index]     = mem_load_u32(addr);
        wave.sgpr[inst.dst.index + 1] = mem_load_u32(addr + 4);
        break;
    }
    case Opcode::S_BUFFER_LOAD_DWORDX4: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        for (int i = 0; i < 4; i++)
            wave.sgpr[inst.dst.index + i] = mem_load_u32(addr + i * 4);
        break;
    }
    case Opcode::S_AND_SAVEEXEC_B64: {
        uint64_t old_exec = wave.exec;
        // Save old exec to dst
        write_sdst64(wave, inst.dst, old_exec);
        // exec = exec & VCC
        wave.exec = old_exec & wave.vcc;
        wave.scc = (wave.exec != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_OR_SAVEEXEC_B64: {
        uint64_t old_exec = wave.exec;
        write_sdst64(wave, inst.dst, old_exec);
        wave.exec = old_exec | wave.vcc;
        wave.scc = (wave.exec != 0) ? 1 : 0;
        break;
    }
    case Opcode::S_GETPC_B64: {
        // Returns current PC (address of next instruction) into SGPR pair
        // In simulation, we use wave.pc which is already advanced
        // Just set dst to 0 (PC-relative addressing not needed in pure simulation)
        write_sdst64(wave, inst.dst, 0ULL);
        break;
    }
    case Opcode::S_SETPC_B64: {
        // Set PC to value in SGPR pair — treat as no-op in simulation
        // (used for indirect calls/returns)
        break;
    }

    // -----------------------------------------------------------------------
    // Scalar memory
    // -----------------------------------------------------------------------
    case Opcode::S_LOAD_DWORD: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        wave.sgpr[inst.dst.index] = mem_load_u32(addr);
        break;
    }
    case Opcode::S_LOAD_DWORDX2: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        wave.sgpr[inst.dst.index]     = mem_load_u32(addr);
        wave.sgpr[inst.dst.index + 1] = mem_load_u32(addr + 4);
        break;
    }
    case Opcode::S_LOAD_DWORDX4: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        for (int i = 0; i < 4; i++)
            wave.sgpr[inst.dst.index + i] = mem_load_u32(addr + i * 4);
        break;
    }
    case Opcode::S_LOAD_DWORDX8: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        for (int i = 0; i < 8; i++)
            wave.sgpr[inst.dst.index + i] = mem_load_u32(addr + i * 4);
        break;
    }
    case Opcode::S_LOAD_DWORDX16: {
        uint64_t base = read_src_scalar64(wave, inst.src[0]);
        uint64_t addr = base + (uint64_t)(int64_t)inst.smem_offset;
        for (int i = 0; i < 16; i++)
            wave.sgpr[inst.dst.index + i] = mem_load_u32(addr + i * 4);
        break;
    }

    // -----------------------------------------------------------------------
    // Scalar control flow
    // -----------------------------------------------------------------------
    case Opcode::S_WAITCNT: {
        // No-op in simulation — all memory ops are synchronous
        break;
    }
    case Opcode::S_NOP: {
        break;
    }
    case Opcode::S_ENDPGM: {
        // Handled by caller
        break;
    }
    case Opcode::S_BRANCH: {
        wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_CBRANCH_SCC0: {
        if (wave.scc == 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_CBRANCH_SCC1: {
        if (wave.scc != 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_CBRANCH_EXECZ: {
        if (wave.exec == 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }
    case Opcode::S_CBRANCH_EXECNZ: {
        if (wave.exec != 0) wave.pc += (int64_t)inst.simm16 * 4;
        break;
    }

    // -----------------------------------------------------------------------
    // Vector ALU
    // -----------------------------------------------------------------------
    case Opcode::V_MOV_B32: {
        uint32_t val = read_src_scalar(wave, inst.src[0]);
        // If src is VGPR, it's per-lane; otherwise broadcast scalar
        bool src_is_vgpr = (inst.src[0].kind == Operand::Kind::VGPR);
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t s = src_is_vgpr ? wave.vgpr[inst.src[0].index][lane] : val;
            wave.vgpr[inst.dst.index][lane] = s;
        }
        break;
    }
    case Opcode::V_ADD_U32: {
        // No-carry version (gfx9 extension): VOP2 op 52
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a + b;
        }
        break;
    }
    case Opcode::V_ADD_CO_U32: {
        // With carry-out to VCC (VOP2 e32 form: dst2 implicit = VCC)
        // Also used in VOP3B form with explicit dst2
        uint64_t new_carry = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = read_vsrc(wave, inst.src[0], lane);
            uint64_t b = read_vsrc(wave, inst.src[1], lane);
            uint64_t r = a + b;
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
            if (r > 0xFFFFFFFFULL) new_carry |= (1ULL << lane);
        }
        // Write carry to dst2 (VOP3B) or VCC (VOP2)
        if (inst.format == InstFormat::VOP3B && inst.dst2.kind != Operand::Kind::NONE) {
            write_sdst64(wave, inst.dst2, new_carry);
        } else {
            wave.vcc = new_carry;
        }
        break;
    }
    case Opcode::V_ADDC_CO_U32: {
        // Add with carry-in from VCC, carry-out to VCC
        uint64_t new_carry = 0;
        uint64_t carry_in = wave.vcc;
        // If VOP3B form, carry-in from src[2]
        if (inst.format == InstFormat::VOP3B && inst.src[2].kind != Operand::Kind::NONE) {
            carry_in = read_src_scalar64(wave, inst.src[2]);
        }
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = read_vsrc(wave, inst.src[0], lane);
            uint64_t b = read_vsrc(wave, inst.src[1], lane);
            uint64_t cin = (carry_in >> lane) & 1;
            uint64_t r = a + b + cin;
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
            if (r > 0xFFFFFFFFULL) new_carry |= (1ULL << lane);
        }
        if (inst.format == InstFormat::VOP3B && inst.dst2.kind != Operand::Kind::NONE) {
            write_sdst64(wave, inst.dst2, new_carry);
        } else {
            wave.vcc = new_carry;
        }
        break;
    }
    case Opcode::V_SUB_U32: {
        uint64_t borrow_out = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a - b;
            if (b > a) borrow_out |= (1ULL << lane);
        }
        wave.vcc = borrow_out;
        break;
    }
    case Opcode::V_SUBREV_U32: {
        uint64_t borrow_out = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = b - a;
            if (a > b) borrow_out |= (1ULL << lane);
        }
        wave.vcc = borrow_out;
        break;
    }
    case Opcode::V_MUL_U32_U24: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane) & 0xFFFFFF;
            uint32_t b = read_vsrc(wave, inst.src[1], lane) & 0xFFFFFF;
            wave.vgpr[inst.dst.index][lane] = a * b;
        }
        break;
    }
    case Opcode::V_MUL_LO_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a * b;
        }
        break;
    }
    case Opcode::V_AND_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a & b;
        }
        break;
    }
    case Opcode::V_OR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a | b;
        }
        break;
    }
    case Opcode::V_XOR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = a ^ b;
        }
        break;
    }
    case Opcode::V_LSHLREV_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 31;
            uint32_t val   = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = val << shift;
        }
        break;
    }
    case Opcode::V_LSHRREV_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 31;
            uint32_t val   = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = val >> shift;
        }
        break;
    }
    case Opcode::V_ASHRREV_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 31;
            int32_t  val   = (int32_t)read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(val >> shift);
        }
        break;
    }
    case Opcode::V_ASHRREV_I64: {
        // v_ashrrev_i64 vdst_pair, src0_shift, vsrc1_pair
        // dst = [vdst, vdst+1], src1 = [src1, src1+1]
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 63;
            uint32_t lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t hi = wave.vgpr[inst.src[1].index + 1][lane];
            int64_t val = (int64_t)(((uint64_t)hi << 32) | lo);
            int64_t result = val >> shift;
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(result & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)((uint64_t)result >> 32);
        }
        break;
    }
    case Opcode::V_LSHLREV_B64: {
        // v_lshlrev_b64 vdst_pair, src0_shift, vsrc1_pair
        // dst[0:1] = src1[0:1] << src0
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 63;
            uint32_t lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t val = ((uint64_t)hi << 32) | lo;
            uint64_t result = val << shift;
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(result & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(result >> 32);
        }
        break;
    }
    case Opcode::V_ADD_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b;
            memcpy(&a, &a_bits, 4);
            memcpy(&b, &b_bits, 4);
            float r = a + b;
            uint32_t r_bits;
            memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MUL_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b;
            memcpy(&a, &a_bits, 4);
            memcpy(&b, &b_bits, 4);
            float r = a * b;
            uint32_t r_bits;
            memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MAD_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_bits = read_vsrc(wave, inst.src[2], lane);
            float a, b, c;
            memcpy(&a, &a_bits, 4);
            memcpy(&b, &b_bits, 4);
            memcpy(&c, &c_bits, 4);
            float r = a * b + c;
            uint32_t r_bits;
            memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_CVT_F32_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            float r = (float)a;
            uint32_t r_bits;
            memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_CVT_I32_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a;
            memcpy(&a, &a_bits, 4);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)a;
        }
        break;
    }

    // -----------------------------------------------------------------------
    // Vector compare (result to VCC)
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LT_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a < b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_LE_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a <= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GT_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a > b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GE_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a >= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_EQ_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a == b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_NE_U32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            if (a != b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_LT_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a < b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_LE_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a <= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GT_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a > b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GE_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a >= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_EQ_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a == b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_NE_I32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            if (a != b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }

    // -----------------------------------------------------------------------
    // Global/Flat memory
    // -----------------------------------------------------------------------
    case Opcode::GLOBAL_LOAD_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                // saddr mode: base from SGPR pair, vaddr = 32-bit offset
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                // "off" mode: vaddr is 64-bit pair
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            wave.vgpr[inst.dst.index][lane] = mem_load_u32(addr);
        }
        break;
    }
    case Opcode::GLOBAL_LOAD_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
        }
        break;
    }
    case Opcode::GLOBAL_LOAD_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            for (int k = 0; k < 4; k++)
                wave.vgpr[inst.dst.index + k][lane] = mem_load_u32(addr + k * 4);
        }
        break;
    }
    case Opcode::GLOBAL_STORE_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            uint32_t val = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, val);
        }
        break;
    }
    case Opcode::GLOBAL_STORE_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            for (int k = 0; k < 2; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::GLOBAL_STORE_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                int32_t  off  = (int32_t)wave.vgpr[inst.src[0].index][lane];
                addr = base + (int64_t)off + inst.flat_offset;
            } else {
                uint32_t lo = wave.vgpr[inst.src[0].index][lane];
                uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
                addr = ((uint64_t)hi << 32) | lo;
                addr += inst.flat_offset;
            }
            for (int k = 0; k < 4; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }

    // -----------------------------------------------------------------------
    // Vector ALU — VOP1 gaps (transcendentals, type conversions, bit ops)
    // -----------------------------------------------------------------------
    case Opcode::V_READFIRSTLANE_B32: {
        // Return value from first active lane to all lanes
        uint32_t val = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if ((wave.exec >> lane) & 1) { val = wave.vgpr[inst.src[0].index][lane]; break; }
        }
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = val;
        }
        break;
    }
    case Opcode::V_CVT_F32_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            float r = (float)a;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_CVT_U32_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)a;
        }
        break;
    }
    case Opcode::V_FRACT_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = a - floorf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_TRUNC_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = truncf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_CEIL_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = ceilf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_RNDNE_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = rintf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_FLOOR_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = floorf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_EXP_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = exp2f(a);  // GPU v_exp_f32 is base-2
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_LOG_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = log2f(a);  // GPU v_log_f32 is base-2
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_RCP_F32:
    case Opcode::V_RCP_IFLAG_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = 1.0f / a;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_RSQ_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = 1.0f / sqrtf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_SQRT_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = sqrtf(a);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_SIN_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            // GPU v_sin_f32: input is in revolutions (0..1 = 0..2*pi)
            float r = sinf(a * 6.283185307f);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_COS_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            float r = cosf(a * 6.283185307f);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_NOT_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = ~read_vsrc(wave, inst.src[0], lane);
        }
        break;
    }
    case Opcode::V_BFREV_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t r = 0;
            for (int i = 0; i < 32; i++) r |= ((a >> i) & 1) << (31 - i);
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_FFBH_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t r = 0xFFFFFFFF;
            for (int i = 31; i >= 0; i--) {
                if ((a >> i) & 1) { r = 31 - i; break; }
            }
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_FFBL_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t r = 0xFFFFFFFF;
            for (int i = 0; i < 32; i++) {
                if ((a >> i) & 1) { r = i; break; }
            }
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_CVT_F32_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane) & 0xFFFF;
            // Decode half-precision float
            uint32_t sign = (a >> 15) & 1;
            uint32_t exp  = (a >> 10) & 0x1F;
            uint32_t mant = a & 0x3FF;
            float r;
            if (exp == 0x1F) {
                r = (mant == 0) ? (sign ? -__builtin_inff() : __builtin_inff())
                                : __builtin_nanf("");
            } else if (exp == 0) {
                r = (float)mant * (1.0f / (1 << 24));
                if (sign) r = -r;
            } else {
                uint32_t f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&r, &f32bits, 4);
            }
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_CVT_F16_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            float a; memcpy(&a, &a_bits, 4);
            // Convert f32 → f16 (round to nearest)
            uint32_t sign = (a_bits >> 31) & 1;
            int32_t  exp  = ((a_bits >> 23) & 0xFF) - 127;
            uint32_t mant = a_bits & 0x7FFFFF;
            uint16_t h;
            if (exp > 15)       { h = (uint16_t)((sign << 15) | 0x7C00); } // inf
            else if (exp < -24) { h = (uint16_t)(sign << 15); }             // zero
            else if (exp < -14) { uint32_t m = (mant | 0x800000) >> (-exp - 14 + 13); h = (uint16_t)((sign << 15) | m); }
            else                { h = (uint16_t)((sign << 15) | ((exp + 15) << 10) | (mant >> 13)); }
            wave.vgpr[inst.dst.index][lane] = h;
        }
        break;
    }
    case Opcode::V_CVT_F32_UBYTE0:
    case Opcode::V_CVT_F32_UBYTE1:
    case Opcode::V_CVT_F32_UBYTE2:
    case Opcode::V_CVT_F32_UBYTE3: {
        int byte_idx = (inst.opcode == Opcode::V_CVT_F32_UBYTE0) ? 0 :
                       (inst.opcode == Opcode::V_CVT_F32_UBYTE1) ? 1 :
                       (inst.opcode == Opcode::V_CVT_F32_UBYTE2) ? 2 : 3;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            float r = (float)((a >> (byte_idx * 8)) & 0xFF);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_SWAP_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t tmp = wave.vgpr[inst.dst.index][lane];
            wave.vgpr[inst.dst.index][lane]      = wave.vgpr[inst.src[0].index][lane];
            wave.vgpr[inst.src[0].index][lane]   = tmp;
        }
        break;
    }
    // accvgpr operations
    case Opcode::V_ACCVGPR_READ_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = wave.agpr[inst.src[0].index][lane];
        }
        break;
    }
    case Opcode::V_ACCVGPR_WRITE_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.agpr[inst.dst.index][lane] = read_vsrc(wave, inst.src[0], lane);
        }
        break;
    }
    // -----------------------------------------------------------------------
    // Vector ALU — VOP2 gaps
    // -----------------------------------------------------------------------
    case Opcode::V_SUB_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            float r = a - b;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MIN_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            float r = (a < b) ? a : b;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MAX_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            float r = (a > b) ? a : b;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MIN_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((a < b) ? a : b);
        }
        break;
    }
    case Opcode::V_MAX_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((a > b) ? a : b);
        }
        break;
    }
    case Opcode::V_MIN_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (a < b) ? a : b;
        }
        break;
    }
    case Opcode::V_MAX_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (a > b) ? a : b;
        }
        break;
    }
    case Opcode::V_MAC_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_bits = wave.vgpr[inst.dst.index][lane];
            float a, b, c;
            memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4); memcpy(&c, &c_bits, 4);
            float r = a * b + c;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MUL_I32_I24: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)(read_vsrc(wave, inst.src[0], lane) << 8) >> 8;
            int32_t b = (int32_t)(read_vsrc(wave, inst.src[1], lane) << 8) >> 8;
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(a * b);
        }
        break;
    }
    // -----------------------------------------------------------------------
    // Vector compare — f32
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LT_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a < b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_EQ_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a == b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_LE_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a <= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GT_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a > b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GE_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a >= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_NEQ_F32:
    case Opcode::V_CMP_LG_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (a != b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_NLT_F32:
    case Opcode::V_CMP_NGE_F32: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4);
            if (!(a >= b)) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    // -----------------------------------------------------------------------
    // Vector compare — i64/u64
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LT_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int64_t)(((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) |
                                   wave.vgpr[inst.src[0].index][lane]);
            int64_t b = (int64_t)(((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) |
                                   wave.vgpr[inst.src[1].index][lane]);
            if (a < b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_EQ_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a == b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GE_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int64_t)(((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane]);
            int64_t b = (int64_t)(((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane]);
            if (a >= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_NE_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a != b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_LT_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a < b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_EQ_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a == b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_GE_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a >= b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    case Opcode::V_CMP_NE_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a != b) mask |= (1ULL << lane);
        }
        wave.vcc = mask;
        break;
    }
    // -----------------------------------------------------------------------
    // VOP3A — multi-source ops
    // -----------------------------------------------------------------------
    case Opcode::V_FMA_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_bits = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4); memcpy(&c, &c_bits, 4);
            float r = fmaf(a, b, c);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_ADD_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64, b64;
            memcpy(&a64, &a_lo, 4); memcpy(((uint8_t*)&a64) + 4, &a_hi, 4);
            memcpy(&b64, &b_lo, 4); memcpy(((uint8_t*)&b64) + 4, &b_hi, 4);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            double r = a + b;
            uint64_t r64; memcpy(&r64, &r, 8);
            uint32_t r_lo = (uint32_t)(r64 & 0xFFFFFFFF);
            uint32_t r_hi = (uint32_t)(r64 >> 32);
            wave.vgpr[inst.dst.index][lane]     = r_lo;
            wave.vgpr[inst.dst.index + 1][lane] = r_hi;
        }
        break;
    }
    case Opcode::V_MUL_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b64 = ((uint64_t)wave.vgpr[inst.src[1].index+1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            double r = a * b;
            uint64_t r64; memcpy(&r64, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r64 >> 32);
        }
        break;
    }
    case Opcode::V_FMA_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b64 = ((uint64_t)wave.vgpr[inst.src[1].index+1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            uint64_t c64 = ((uint64_t)wave.vgpr[inst.src[2].index+1][lane] << 32) | wave.vgpr[inst.src[2].index][lane];
            double a, b, c; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8); memcpy(&c, &c64, 8);
            double r = fma(a, b, c);
            uint64_t r64; memcpy(&r64, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r64 >> 32);
        }
        break;
    }
    case Opcode::V_MIN_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b64 = ((uint64_t)wave.vgpr[inst.src[1].index+1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            double r = (a < b) ? a : b;
            uint64_t r64; memcpy(&r64, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r64 >> 32);
        }
        break;
    }
    case Opcode::V_MAX_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b64 = ((uint64_t)wave.vgpr[inst.src[1].index+1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            double r = (a > b) ? a : b;
            uint64_t r64; memcpy(&r64, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r64 >> 32);
        }
        break;
    }
    case Opcode::V_MUL_HI_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = read_vsrc(wave, inst.src[0], lane);
            uint64_t b = read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((a * b) >> 32);
        }
        break;
    }
    case Opcode::V_MUL_HI_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int64_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((uint64_t)(a * b) >> 32);
        }
        break;
    }
    case Opcode::V_BFE_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t base   = read_vsrc(wave, inst.src[0], lane);
            uint32_t offset = read_vsrc(wave, inst.src[1], lane) & 0x1F;
            uint32_t width  = read_vsrc(wave, inst.src[2], lane) & 0x1F;
            uint32_t r = (width == 0) ? 0 : ((base >> offset) & ((1u << width) - 1u));
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_BFE_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t base   = read_vsrc(wave, inst.src[0], lane);
            uint32_t offset = read_vsrc(wave, inst.src[1], lane) & 0x1F;
            uint32_t width  = read_vsrc(wave, inst.src[2], lane) & 0x1F;
            uint32_t raw    = (width == 0) ? 0 : ((base >> offset) & ((1u << width) - 1u));
            if (width > 0 && width < 32) {
                uint32_t sign_bit = 1u << (width - 1);
                raw = (raw ^ sign_bit) - sign_bit;
            }
            wave.vgpr[inst.dst.index][lane] = raw;
        }
        break;
    }
    case Opcode::V_BFI_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t mask = read_vsrc(wave, inst.src[0], lane);
            uint32_t src1 = read_vsrc(wave, inst.src[1], lane);
            uint32_t src2 = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (src1 & mask) | (src2 & ~mask);
        }
        break;
    }
    case Opcode::V_MIN3_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_b = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4); memcpy(&c, &c_b, 4);
            float r = (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MAX3_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_b = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4); memcpy(&c, &c_b, 4);
            float r = (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MED3_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_b = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_b = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &a_b, 4); memcpy(&b, &b_b, 4); memcpy(&c, &c_b, 4);
            float r = (a < b) ? ((b < c) ? b : ((a < c) ? c : a))
                               : ((a < c) ? a : ((b < c) ? c : b));
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    case Opcode::V_MIN3_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            int32_t c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((a<b)?(a<c?a:c):(b<c?b:c));
        }
        break;
    }
    case Opcode::V_MAX3_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            int32_t c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)((a>b)?(a>c?a:c):(b>c?b:c));
        }
        break;
    }
    case Opcode::V_MED3_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            int32_t c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            int32_t mn = (a<b)?(a<c?a:c):(b<c?b:c);
            int32_t mx = (a>b)?(a>c?a:c):(b>c?b:c);
            int32_t r = a + b + c - mn - mx;
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
        }
        break;
    }
    case Opcode::V_MIN3_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (a<b)?(a<c?a:c):(b<c?b:c);
        }
        break;
    }
    case Opcode::V_MAX3_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (a>b)?(a>c?a:c):(b>c?b:c);
        }
        break;
    }
    case Opcode::V_MED3_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            uint32_t mn = (a<b)?(a<c?a:c):(b<c?b:c);
            uint32_t mx = (a>b)?(a>c?a:c):(b>c?b:c);
            wave.vgpr[inst.dst.index][lane] = a + b + c - mn - mx;
        }
        break;
    }
    case Opcode::V_LSHRREV_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t shift = read_vsrc(wave, inst.src[0], lane) & 63;
            uint64_t val = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) |
                                       wave.vgpr[inst.src[1].index][lane];
            uint64_t r = val >> shift;
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r >> 32);
        }
        break;
    }
    case Opcode::V_ADD_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int32_t a = (int32_t)read_vsrc(wave, inst.src[0], lane);
            int32_t b = (int32_t)read_vsrc(wave, inst.src[1], lane);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(a + b);
        }
        break;
    }
    case Opcode::V_ADD3_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = a + b + c;
        }
        break;
    }
    case Opcode::V_LSHL_ADD_U32: {
        // dst = (src0 << src1) + src2
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane) & 31;
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (a << b) + c;
        }
        break;
    }
    case Opcode::V_LSHL_OR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane) & 31;
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (a << b) | c;
        }
        break;
    }
    case Opcode::V_AND_OR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            wave.vgpr[inst.dst.index][lane] = (a & b) | c;
        }
        break;
    }
    case Opcode::V_OR3_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = read_vsrc(wave, inst.src[0], lane) |
                                              read_vsrc(wave, inst.src[1], lane) |
                                              read_vsrc(wave, inst.src[2], lane);
        }
        break;
    }
    case Opcode::V_PERM_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t src0 = read_vsrc(wave, inst.src[0], lane);
            uint32_t src1 = read_vsrc(wave, inst.src[1], lane);
            uint32_t sel  = read_vsrc(wave, inst.src[2], lane);
            // 8 bytes: {src0, src1}, selector picks bytes
            uint64_t combined = ((uint64_t)src0 << 32) | src1;
            uint32_t r = 0;
            for (int b = 0; b < 4; b++) {
                uint8_t s = (sel >> (b * 8)) & 0xFF;
                uint8_t byte_val;
                if (s < 8) { byte_val = (uint8_t)(combined >> (s * 8)); }
                else if (s == 0xFF) byte_val = 0xFF;
                else byte_val = 0;
                r |= ((uint32_t)byte_val) << (b * 8);
            }
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    // -----------------------------------------------------------------------
    // Global flat memory — additional widths and atomics
    // -----------------------------------------------------------------------
    case Opcode::GLOBAL_LOAD_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            for (int k = 0; k < 3; k++)
                wave.vgpr[inst.dst.index + k][lane] = mem_load_u32(addr + k * 4);
        }
        break;
    }
    case Opcode::GLOBAL_STORE_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            for (int k = 0; k < 3; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_ADD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old + wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_SUB: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old - wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_AND: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old & wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_OR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old | wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_XOR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old ^ wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_SWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_CMPSWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            uint32_t cmp = wave.vgpr[inst.src[2].index][lane];
            uint32_t val = wave.vgpr[inst.src[2].index + 1][lane];
            if (old == cmp) mem_store_u32(addr, val);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_SMIN: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            int32_t old = (int32_t)mem_load_u32(addr);
            int32_t v   = (int32_t)wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (uint32_t)(v < old ? v : old));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = (uint32_t)old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_UMIN: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            uint32_t v   = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, v < old ? v : old);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_SMAX: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            int32_t old = (int32_t)mem_load_u32(addr);
            int32_t v   = (int32_t)wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (uint32_t)(v > old ? v : old));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = (uint32_t)old;
        }
        break;
    }
    case Opcode::GLOBAL_ATOMIC_UMAX: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint32_t old = mem_load_u32(addr);
            uint32_t v   = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, v > old ? v : old);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    // Global loads — narrow widths
    case Opcode::GLOBAL_LOAD_UBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::GLOBAL_LOAD_SBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            int8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    case Opcode::GLOBAL_LOAD_USHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::GLOBAL_LOAD_SSHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            int16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    // Global stores — narrow widths
    case Opcode::GLOBAL_STORE_BYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint8_t v = (uint8_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 1);
        }
        break;
    }
    case Opcode::GLOBAL_STORE_SHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr;
            if (inst.src[1].kind == Operand::Kind::SGPR) {
                uint64_t base = read_src_scalar64(wave, inst.src[1]);
                addr = base + (int64_t)(int32_t)wave.vgpr[inst.src[0].index][lane] + inst.flat_offset;
            } else {
                addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
                addr += inst.flat_offset;
            }
            uint16_t v = (uint16_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 2);
        }
        break;
    }
    // FLAT (seg=0) instructions — treat as global (host address in VGPR pair)
    case Opcode::FLAT_LOAD_UBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::FLAT_LOAD_SBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            int8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    case Opcode::FLAT_LOAD_USHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::FLAT_LOAD_SSHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            int16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    case Opcode::FLAT_LOAD_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            wave.vgpr[inst.dst.index][lane] = mem_load_u32(addr);
        }
        break;
    }
    case Opcode::FLAT_LOAD_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
        }
        break;
    }
    case Opcode::FLAT_LOAD_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            for (int k = 0; k < 3; k++)
                wave.vgpr[inst.dst.index + k][lane] = mem_load_u32(addr + k * 4);
        }
        break;
    }
    case Opcode::FLAT_LOAD_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            for (int k = 0; k < 4; k++)
                wave.vgpr[inst.dst.index + k][lane] = mem_load_u32(addr + k * 4);
        }
        break;
    }
    case Opcode::FLAT_STORE_BYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint8_t v = (uint8_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 1);
        }
        break;
    }
    case Opcode::FLAT_STORE_SHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint16_t v = (uint16_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 2);
        }
        break;
    }
    case Opcode::FLAT_STORE_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
        }
        break;
    }
    case Opcode::FLAT_STORE_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            mem_store_u32(addr,     wave.vgpr[inst.src[2].index][lane]);
            mem_store_u32(addr + 4, wave.vgpr[inst.src[2].index + 1][lane]);
        }
        break;
    }
    case Opcode::FLAT_STORE_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            for (int k = 0; k < 3; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::FLAT_STORE_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            for (int k = 0; k < 4; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_ADD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old + wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_SUB: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old - wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_AND: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old & wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_OR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old | wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_XOR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old ^ wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_SWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_CMPSWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t cmp = wave.vgpr[inst.src[2].index][lane];
            uint32_t val = wave.vgpr[inst.src[2].index + 1][lane];
            if (old == cmp) mem_store_u32(addr, val);
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_INC: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t max_val = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (old >= max_val) ? 0 : (old + 1));
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::FLAT_ATOMIC_DEC: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t addr = ((uint64_t)wave.vgpr[inst.src[0].index+1][lane] << 32) |
                             wave.vgpr[inst.src[0].index][lane];
            addr += inst.flat_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t max_val = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, ((old == 0) || (old > max_val)) ? max_val : (old - 1));
            if (inst.dst.kind == Operand::Kind::VGPR) wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    // VINTRP — vector interpolation (graphics only, stub as no-op)
    case Opcode::V_INTERP_P1_F32:
    case Opcode::V_INTERP_P2_F32:
    case Opcode::V_INTERP_MOV_F32:
    case Opcode::V_INTERP_P1LL_F16:
    case Opcode::V_INTERP_P2_F16:
        // No-op: interpolation requires parameter memory not available in compute
        break;
    // -----------------------------------------------------------------------
    // DS (LDS) instruction execution
    // -----------------------------------------------------------------------
    case Opcode::DS_WRITE_B32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 <= wave.lds_size)
                memcpy(wave.lds_ptr + addr, &wave.vgpr[inst.src[1].index][lane], 4);
        }
        break;
    }
    case Opcode::DS_READ_B32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 <= wave.lds_size)
                memcpy(&wave.vgpr[inst.dst.index][lane], wave.lds_ptr + addr, 4);
        }
        break;
    }
    case Opcode::DS_WRITE_B64: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 <= wave.lds_size) {
                memcpy(wave.lds_ptr + addr,     &wave.vgpr[inst.src[1].index][lane], 4);
                memcpy(wave.lds_ptr + addr + 4, &wave.vgpr[inst.src[1].index + 1][lane], 4);
            }
        }
        break;
    }
    case Opcode::DS_READ_B64: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 <= wave.lds_size) {
                memcpy(&wave.vgpr[inst.dst.index][lane],     wave.lds_ptr + addr, 4);
                memcpy(&wave.vgpr[inst.dst.index + 1][lane], wave.lds_ptr + addr + 4, 4);
            }
        }
        break;
    }
    case Opcode::DS_WRITE2_B32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t base = wave.vgpr[inst.src[0].index][lane];
            uint32_t a0 = base + inst.ds_offset0 * 4;
            uint32_t a1 = base + inst.ds_offset1 * 4;
            if (a0 + 4 <= wave.lds_size)
                memcpy(wave.lds_ptr + a0, &wave.vgpr[inst.src[1].index][lane], 4);
            if (a1 + 4 <= wave.lds_size)
                memcpy(wave.lds_ptr + a1, &wave.vgpr[inst.src[2].index][lane], 4);
        }
        break;
    }
    case Opcode::DS_READ2_B32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t base = wave.vgpr[inst.src[0].index][lane];
            uint32_t a0 = base + inst.ds_offset0 * 4;
            uint32_t a1 = base + inst.ds_offset1 * 4;
            if (a0 + 4 <= wave.lds_size)
                memcpy(&wave.vgpr[inst.dst.index][lane],     wave.lds_ptr + a0, 4);
            if (a1 + 4 <= wave.lds_size)
                memcpy(&wave.vgpr[inst.dst.index + 1][lane], wave.lds_ptr + a1, 4);
        }
        break;
    }
    case Opcode::DS_ADD_U32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 <= wave.lds_size) {
                uint32_t old; memcpy(&old, wave.lds_ptr + addr, 4);
                uint32_t val = old + wave.vgpr[inst.src[1].index][lane];
                memcpy(wave.lds_ptr + addr, &val, 4);
            }
        }
        break;
    }
    case Opcode::DS_ADD_RTN_U32: {
        if (!wave.lds_ptr) break;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 <= wave.lds_size) {
                uint32_t old; memcpy(&old, wave.lds_ptr + addr, 4);
                wave.vgpr[inst.dst.index][lane] = old;
                uint32_t val = old + wave.vgpr[inst.src[1].index][lane];
                memcpy(wave.lds_ptr + addr, &val, 4);
            }
        }
        break;
    }
    case Opcode::DS_BPERMUTE_B32: {
        // Reads vgpr[addr/4] from another lane into current lane's dst
        uint32_t tmp[WAVE_SIZE];
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            uint32_t src_lane = (wave.vgpr[inst.src[0].index][lane] >> 2) & 63;
            tmp[lane] = wave.vgpr[inst.src[1].index][src_lane];
        }
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = tmp[lane];
        }
        break;
    }
    case Opcode::DS_PERMUTE_B32: {
        // Writes current lane's src1 to dst lane indexed by src0/4
        uint32_t tmp[WAVE_SIZE] = {};
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t dst_lane = (wave.vgpr[inst.src[0].index][lane] >> 2) & 63;
            tmp[dst_lane] = wave.vgpr[inst.src[1].index][lane];
        }
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = tmp[lane];
        }
        break;
    }
    case Opcode::DS_NOP:
    case Opcode::DS_SWIZZLE_B32: {
        // No-op or unsimulated swizzle
        break;
    }
    // -----------------------------------------------------------------------
    // DS — atomic integer ops (u32)
    // -----------------------------------------------------------------------
    case Opcode::DS_SUB_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            *p -= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_INC_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t v = wave.vgpr[inst.src[1].index][lane];
            *p = (*p >= v) ? 0 : (*p + 1);
        }
        break;
    }
    case Opcode::DS_DEC_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t v = wave.vgpr[inst.src[1].index][lane];
            *p = ((*p == 0) || (*p > v)) ? v : (*p - 1);
        }
        break;
    }
    case Opcode::DS_MIN_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            int32_t* p = reinterpret_cast<int32_t*>(wave.lds_ptr + addr);
            int32_t v = (int32_t)wave.vgpr[inst.src[1].index][lane];
            if (v < *p) *p = v;
        }
        break;
    }
    case Opcode::DS_MAX_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            int32_t* p = reinterpret_cast<int32_t*>(wave.lds_ptr + addr);
            int32_t v = (int32_t)wave.vgpr[inst.src[1].index][lane];
            if (v > *p) *p = v;
        }
        break;
    }
    case Opcode::DS_MIN_RTN_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            int32_t* p = reinterpret_cast<int32_t*>(wave.lds_ptr + addr);
            int32_t v = (int32_t)wave.vgpr[inst.src[1].index][lane];
            wave.vgpr[inst.dst.index][lane] = (uint32_t)*p;
            if (v < *p) *p = v;
        }
        break;
    }
    case Opcode::DS_MAX_RTN_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            int32_t* p = reinterpret_cast<int32_t*>(wave.lds_ptr + addr);
            int32_t v = (int32_t)wave.vgpr[inst.src[1].index][lane];
            wave.vgpr[inst.dst.index][lane] = (uint32_t)*p;
            if (v > *p) *p = v;
        }
        break;
    }
    case Opcode::DS_AND_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            *p &= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_OR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            *p |= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_XOR_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            *p ^= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_AND_RTN_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            wave.vgpr[inst.dst.index][lane] = *p;
            *p &= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_OR_RTN_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            wave.vgpr[inst.dst.index][lane] = *p;
            *p |= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_XOR_RTN_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            wave.vgpr[inst.dst.index][lane] = *p;
            *p ^= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_SUB_RTN_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            wave.vgpr[inst.dst.index][lane] = *p;
            *p -= wave.vgpr[inst.src[1].index][lane];
        }
        break;
    }
    case Opcode::DS_INC_RTN_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t v = wave.vgpr[inst.src[1].index][lane];
            wave.vgpr[inst.dst.index][lane] = *p;
            *p = (*p >= v) ? 0 : (*p + 1);
        }
        break;
    }
    case Opcode::DS_DEC_RTN_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t v = wave.vgpr[inst.src[1].index][lane];
            wave.vgpr[inst.dst.index][lane] = *p;
            *p = ((*p == 0) || (*p > v)) ? v : (*p - 1);
        }
        break;
    }
    case Opcode::DS_CMPST_B32: {
        // compare-and-store: if *p == data0, *p = data1 (no return)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t cmp = wave.vgpr[inst.src[1].index][lane];
            uint32_t val = wave.vgpr[inst.src[2].index][lane];
            if (*p == cmp) *p = val;
        }
        break;
    }
    case Opcode::DS_CMPST_RTN_B32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            uint32_t cmp = wave.vgpr[inst.src[1].index][lane];
            uint32_t val = wave.vgpr[inst.src[2].index][lane];
            wave.vgpr[inst.dst.index][lane] = *p;
            if (*p == cmp) *p = val;
        }
        break;
    }
    case Opcode::DS_MIN_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            float cur, v; memcpy(&cur, p, 4); memcpy(&v, &wave.vgpr[inst.src[1].index][lane], 4);
            if (v < cur) memcpy(p, &v, 4);
        }
        break;
    }
    case Opcode::DS_MAX_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            float cur, v; memcpy(&cur, p, 4); memcpy(&v, &wave.vgpr[inst.src[1].index][lane], 4);
            if (v > cur) memcpy(p, &v, 4);
        }
        break;
    }
    case Opcode::DS_MIN_RTN_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            float cur, v; memcpy(&cur, p, 4); memcpy(&v, &wave.vgpr[inst.src[1].index][lane], 4);
            wave.vgpr[inst.dst.index][lane] = *p;
            if (v < cur) memcpy(p, &v, 4);
        }
        break;
    }
    case Opcode::DS_MAX_RTN_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 4 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            float cur, v; memcpy(&cur, p, 4); memcpy(&v, &wave.vgpr[inst.src[1].index][lane], 4);
            wave.vgpr[inst.dst.index][lane] = *p;
            if (v > cur) memcpy(p, &v, 4);
        }
        break;
    }
    // -----------------------------------------------------------------------
    // DS — 64-bit atomic ops (B64)
    // -----------------------------------------------------------------------
    case Opcode::DS_ADD_U64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t v = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                         ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            *p += v;
        }
        break;
    }
    case Opcode::DS_SUB_U64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t v = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                         ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            *p -= v;
        }
        break;
    }
    case Opcode::DS_AND_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t v = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                         ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            *p &= v;
        }
        break;
    }
    case Opcode::DS_OR_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t v = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                         ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            *p |= v;
        }
        break;
    }
    case Opcode::DS_XOR_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t v = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                         ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            *p ^= v;
        }
        break;
    }
    case Opcode::DS_WRITE2_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr0 = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0 * 8;
            uint32_t addr1 = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset1 * 8;
            if (addr0 + 8 > wave.lds_size || addr1 + 8 > wave.lds_size) continue;
            uint32_t* p0 = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr0);
            uint32_t* p1 = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr1);
            p0[0] = wave.vgpr[inst.src[1].index][lane];
            p0[1] = wave.vgpr[inst.src[1].index + 1][lane];
            p1[0] = wave.vgpr[inst.src[2].index][lane];
            p1[1] = wave.vgpr[inst.src[2].index + 1][lane];
        }
        break;
    }
    case Opcode::DS_READ2_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr0 = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0 * 8;
            uint32_t addr1 = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset1 * 8;
            if (addr0 + 8 > wave.lds_size || addr1 + 8 > wave.lds_size) continue;
            const uint32_t* p0 = reinterpret_cast<const uint32_t*>(wave.lds_ptr + addr0);
            const uint32_t* p1 = reinterpret_cast<const uint32_t*>(wave.lds_ptr + addr1);
            wave.vgpr[inst.dst.index][lane]     = p0[0];
            wave.vgpr[inst.dst.index + 1][lane] = p0[1];
            wave.vgpr[inst.dst.index + 2][lane] = p1[0];
            wave.vgpr[inst.dst.index + 3][lane] = p1[1];
        }
        break;
    }
    case Opcode::DS_CMPST_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t cmp = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            uint64_t val = (uint64_t)wave.vgpr[inst.src[2].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[2].index + 1][lane] << 32);
            if (*p == cmp) *p = val;
        }
        break;
    }
    case Opcode::DS_CMPST_RTN_B64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 8 > wave.lds_size) continue;
            uint64_t* p = reinterpret_cast<uint64_t*>(wave.lds_ptr + addr);
            uint64_t cmp = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            uint64_t val = (uint64_t)wave.vgpr[inst.src[2].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[2].index + 1][lane] << 32);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(*p & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(*p >> 32);
            if (*p == cmp) *p = val;
        }
        break;
    }
    case Opcode::DS_WRITE_B128: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 16 > wave.lds_size) continue;
            uint32_t* p = reinterpret_cast<uint32_t*>(wave.lds_ptr + addr);
            p[0] = wave.vgpr[inst.src[1].index][lane];
            p[1] = wave.vgpr[inst.src[1].index + 1][lane];
            p[2] = wave.vgpr[inst.src[1].index + 2][lane];
            p[3] = wave.vgpr[inst.src[1].index + 3][lane];
        }
        break;
    }
    case Opcode::DS_READ_B128: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t addr = wave.vgpr[inst.src[0].index][lane] + inst.ds_offset0;
            if (addr + 16 > wave.lds_size) continue;
            const uint32_t* p = reinterpret_cast<const uint32_t*>(wave.lds_ptr + addr);
            wave.vgpr[inst.dst.index][lane]     = p[0];
            wave.vgpr[inst.dst.index + 1][lane] = p[1];
            wave.vgpr[inst.dst.index + 2][lane] = p[2];
            wave.vgpr[inst.dst.index + 3][lane] = p[3];
        }
        break;
    }
    // -----------------------------------------------------------------------
    // VOP3P — packed 16-bit ops
    // -----------------------------------------------------------------------
    case Opcode::V_PK_MUL_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            // Convert f16 pairs to f32, multiply, pack back
            uint16_t a_lo = a & 0xFFFF, a_hi = a >> 16;
            uint16_t b_lo = b & 0xFFFF, b_hi = b >> 16;
            // Simple f16 approximation via bit manipulation (sign|exp|mant)
            // Use host half-to-float conversion
            auto f16_to_f32 = [](uint16_t h) -> float {
                uint32_t s = (h >> 15) & 1;
                uint32_t e = (h >> 10) & 0x1F;
                uint32_t m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) { bits = (s << 31) | (m << 13); }
                else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
                else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
                float f; memcpy(&f, &bits, 4); return f;
            };
            auto f32_to_f16 = [](float f) -> uint16_t {
                uint32_t bits; memcpy(&bits, &f, 4);
                uint32_t s = bits >> 31;
                int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t m = bits & 0x7FFFFF;
                if (e <= 0) return (uint16_t)(s << 15);
                if (e >= 31) return (uint16_t)((s << 15) | (31 << 10));
                return (uint16_t)((s << 15) | (e << 10) | (m >> 13));
            };
            float ra = f16_to_f32(a_lo) * f16_to_f32(b_lo);
            float rb = f16_to_f32(a_hi) * f16_to_f32(b_hi);
            uint32_t result = f32_to_f16(ra) | ((uint32_t)f32_to_f16(rb) << 16);
            wave.vgpr[inst.dst.index][lane] = result;
        }
        break;
    }
    case Opcode::V_PK_MIN_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            auto f16_to_f32 = [](uint16_t h) -> float {
                uint32_t s = (h >> 15) & 1;
                uint32_t e = (h >> 10) & 0x1F;
                uint32_t m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) { bits = (s << 31) | (m << 13); }
                else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
                else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
                float f; memcpy(&f, &bits, 4); return f;
            };
            auto f32_to_f16 = [](float f) -> uint16_t {
                uint32_t bits; memcpy(&bits, &f, 4);
                uint32_t s = bits >> 31;
                int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t m = bits & 0x7FFFFF;
                if (e <= 0) return (uint16_t)(s << 15);
                if (e >= 31) return (uint16_t)((s << 15) | (31 << 10));
                return (uint16_t)((s << 15) | (e << 10) | (m >> 13));
            };
            float ra = (f16_to_f32(a & 0xFFFF) < f16_to_f32(b & 0xFFFF)) ? f16_to_f32(a & 0xFFFF) : f16_to_f32(b & 0xFFFF);
            float rb = (f16_to_f32(a >> 16) < f16_to_f32(b >> 16)) ? f16_to_f32(a >> 16) : f16_to_f32(b >> 16);
            wave.vgpr[inst.dst.index][lane] = f32_to_f16(ra) | ((uint32_t)f32_to_f16(rb) << 16);
        }
        break;
    }
    case Opcode::V_PK_MAX_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            auto f16_to_f32 = [](uint16_t h) -> float {
                uint32_t s = (h >> 15) & 1;
                uint32_t e = (h >> 10) & 0x1F;
                uint32_t m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) { bits = (s << 31) | (m << 13); }
                else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
                else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
                float f; memcpy(&f, &bits, 4); return f;
            };
            auto f32_to_f16 = [](float f) -> uint16_t {
                uint32_t bits; memcpy(&bits, &f, 4);
                uint32_t s = bits >> 31;
                int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t m = bits & 0x7FFFFF;
                if (e <= 0) return (uint16_t)(s << 15);
                if (e >= 31) return (uint16_t)((s << 15) | (31 << 10));
                return (uint16_t)((s << 15) | (e << 10) | (m >> 13));
            };
            float ra = (f16_to_f32(a & 0xFFFF) > f16_to_f32(b & 0xFFFF)) ? f16_to_f32(a & 0xFFFF) : f16_to_f32(b & 0xFFFF);
            float rb = (f16_to_f32(a >> 16) > f16_to_f32(b >> 16)) ? f16_to_f32(a >> 16) : f16_to_f32(b >> 16);
            wave.vgpr[inst.dst.index][lane] = f32_to_f16(ra) | ((uint32_t)f32_to_f16(rb) << 16);
        }
        break;
    }
    case Opcode::V_PK_FMA_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            auto f16_to_f32 = [](uint16_t h) -> float {
                uint32_t s = (h >> 15) & 1;
                uint32_t e = (h >> 10) & 0x1F;
                uint32_t m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) { bits = (s << 31) | (m << 13); }
                else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
                else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
                float f; memcpy(&f, &bits, 4); return f;
            };
            auto f32_to_f16 = [](float f) -> uint16_t {
                uint32_t bits; memcpy(&bits, &f, 4);
                uint32_t s = bits >> 31;
                int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t m = bits & 0x7FFFFF;
                if (e <= 0) return (uint16_t)(s << 15);
                if (e >= 31) return (uint16_t)((s << 15) | (31 << 10));
                return (uint16_t)((s << 15) | (e << 10) | (m >> 13));
            };
            float ra = fmaf(f16_to_f32(a & 0xFFFF), f16_to_f32(b & 0xFFFF), f16_to_f32(c & 0xFFFF));
            float rb = fmaf(f16_to_f32(a >> 16), f16_to_f32(b >> 16), f16_to_f32(c >> 16));
            wave.vgpr[inst.dst.index][lane] = f32_to_f16(ra) | ((uint32_t)f32_to_f16(rb) << 16);
        }
        break;
    }
    case Opcode::V_PK_ADD_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int16_t r_lo = (int16_t)(a & 0xFFFF) + (int16_t)(b & 0xFFFF);
            int16_t r_hi = (int16_t)(a >> 16) + (int16_t)(b >> 16);
            wave.vgpr[inst.dst.index][lane] = (uint16_t)r_lo | ((uint32_t)(uint16_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_SUB_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int16_t r_lo = (int16_t)(a & 0xFFFF) - (int16_t)(b & 0xFFFF);
            int16_t r_hi = (int16_t)(a >> 16) - (int16_t)(b >> 16);
            wave.vgpr[inst.dst.index][lane] = (uint16_t)r_lo | ((uint32_t)(uint16_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_LSHLREV_B16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane); // shift amount
            uint32_t b = read_vsrc(wave, inst.src[1], lane); // value
            uint16_t r_lo = (uint16_t)((b & 0xFFFF) << (a & 0xF));
            uint16_t r_hi = (uint16_t)((b >> 16) << ((a >> 16) & 0xF));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_LSHRREV_B16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((b & 0xFFFF) >> (a & 0xF));
            uint16_t r_hi = (uint16_t)((b >> 16) >> ((a >> 16) & 0xF));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_ASHRREV_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int16_t r_lo = (int16_t)(b & 0xFFFF) >> (a & 0xF);
            int16_t r_hi = (int16_t)(b >> 16) >> ((a >> 16) & 0xF);
            wave.vgpr[inst.dst.index][lane] = (uint16_t)r_lo | ((uint32_t)(uint16_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MAX_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int16_t a_lo = (int16_t)(a & 0xFFFF), b_lo = (int16_t)(b & 0xFFFF);
            int16_t a_hi = (int16_t)(a >> 16), b_hi = (int16_t)(b >> 16);
            uint16_t r_lo = (uint16_t)(a_lo > b_lo ? a_lo : b_lo);
            uint16_t r_hi = (uint16_t)(a_hi > b_hi ? a_hi : b_hi);
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MIN_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int16_t a_lo = (int16_t)(a & 0xFFFF), b_lo = (int16_t)(b & 0xFFFF);
            int16_t a_hi = (int16_t)(a >> 16), b_hi = (int16_t)(b >> 16);
            uint16_t r_lo = (uint16_t)(a_lo < b_lo ? a_lo : b_lo);
            uint16_t r_hi = (uint16_t)(a_hi < b_hi ? a_hi : b_hi);
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_ADD_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) + (b & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) + (b >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_SUB_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) - (b & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) - (b >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MAX_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) > (b & 0xFFFF) ? (a & 0xFFFF) : (b & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) > (b >> 16) ? (a >> 16) : (b >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MIN_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) < (b & 0xFFFF) ? (a & 0xFFFF) : (b & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) < (b >> 16) ? (a >> 16) : (b >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MAD_I16: {
        // dst = src0 * src1 + src2 (per 16-bit pair, signed)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            int16_t r_lo = (int16_t)(a & 0xFFFF) * (int16_t)(b & 0xFFFF) + (int16_t)(c & 0xFFFF);
            int16_t r_hi = (int16_t)(a >> 16) * (int16_t)(b >> 16) + (int16_t)(c >> 16);
            wave.vgpr[inst.dst.index][lane] = (uint16_t)r_lo | ((uint32_t)(uint16_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MAD_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) * (b & 0xFFFF) + (c & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) * (b >> 16) + (c >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_PK_MUL_LO_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint16_t r_lo = (uint16_t)((a & 0xFFFF) * (b & 0xFFFF));
            uint16_t r_hi = (uint16_t)((a >> 16) * (b >> 16));
            wave.vgpr[inst.dst.index][lane] = r_lo | ((uint32_t)r_hi << 16);
        }
        break;
    }
    case Opcode::V_FMA_MIX_F32: {
        // v_fma_mix_f32 dst, src0, src1, src2 — produces f32 result
        // In general: dst.x = src0.lo16 * src1.lo16 + src2.lo16 (after f16->f32)
        // Simplified: treat all as f32 fma
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t ab = read_vsrc(wave, inst.src[0], lane);
            uint32_t bb = read_vsrc(wave, inst.src[1], lane);
            uint32_t cb = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &ab, 4); memcpy(&b, &bb, 4); memcpy(&c, &cb, 4);
            float r = fmaf(a, b, c);
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.vgpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_FMA_MIXLO_F16:
    case Opcode::V_FMA_MIXHI_F16: {
        // Produces f16 result in lo or hi 16 bits of dst, preserving the other half
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t ab = read_vsrc(wave, inst.src[0], lane);
            uint32_t bb = read_vsrc(wave, inst.src[1], lane);
            uint32_t cb = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &ab, 4); memcpy(&b, &bb, 4); memcpy(&c, &cb, 4);
            float r = fmaf(a, b, c);
            uint32_t rb; memcpy(&rb, &r, 4);
            // Store result as f16 in lo or hi half
            auto f32_to_f16 = [](float f) -> uint16_t {
                uint32_t bits; memcpy(&bits, &f, 4);
                uint32_t s = bits >> 31;
                int32_t e = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t m = bits & 0x7FFFFF;
                if (e <= 0) return (uint16_t)(s << 15);
                if (e >= 31) return (uint16_t)((s << 15) | (31 << 10));
                return (uint16_t)((s << 15) | (e << 10) | (m >> 13));
            };
            uint16_t f16r = f32_to_f16(r);
            uint32_t prev = wave.vgpr[inst.dst.index][lane];
            if (inst.opcode == Opcode::V_FMA_MIXLO_F16) {
                wave.vgpr[inst.dst.index][lane] = (prev & 0xFFFF0000) | f16r;
            } else {
                wave.vgpr[inst.dst.index][lane] = (prev & 0x0000FFFF) | ((uint32_t)f16r << 16);
            }
        }
        break;
    }
    case Opcode::V_DOT2_I32_I16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int32_t  c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            int32_t r = (int16_t)(a & 0xFFFF) * (int16_t)(b & 0xFFFF) +
                        (int16_t)(a >> 16)    * (int16_t)(b >> 16) + c;
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
        }
        break;
    }
    case Opcode::V_DOT2_U32_U16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            uint32_t r = (a & 0xFFFF) * (b & 0xFFFF) + (a >> 16) * (b >> 16) + c;
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_DOT8_I32_I4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int32_t  c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            int32_t r = c;
            for (int k = 0; k < 8; k++) {
                int8_t av = (int8_t)(((a >> (k*4)) & 0xF) << 4) >> 4; // sign-extend 4->8
                int8_t bv = (int8_t)(((b >> (k*4)) & 0xF) << 4) >> 4;
                r += (int32_t)av * (int32_t)bv;
            }
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
        }
        break;
    }
    case Opcode::V_DOT8_U32_U4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            uint32_t r = c;
            for (int k = 0; k < 8; k++) {
                uint8_t av = (a >> (k*4)) & 0xF;
                uint8_t bv = (b >> (k*4)) & 0xF;
                r += (uint32_t)av * (uint32_t)bv;
            }
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    // -----------------------------------------------------------------------
    // MUBUF — buffer loads and stores
    // -----------------------------------------------------------------------
    case Opcode::BUFFER_LOAD_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            // Buffer descriptor: SGPR[srsrc:srsrc+1] = 64-bit base address
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint32_t ioff = inst.mubuf_idxen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            (void)ioff;
            wave.vgpr[inst.dst.index][lane] = mem_load_u32(addr);
        }
        break;
    }
    case Opcode::BUFFER_LOAD_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
        }
        break;
    }
    case Opcode::BUFFER_LOAD_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            for (int k = 0; k < 4; k++)
                wave.vgpr[inst.dst.index + k][lane] = mem_load_u32(addr + k * 4);
        }
        break;
    }
    case Opcode::BUFFER_STORE_DWORD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
        }
        break;
    }
    case Opcode::BUFFER_STORE_DWORDX2: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            for (int k = 0; k < 2; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::BUFFER_STORE_DWORDX4: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            for (int k = 0; k < 4; k++)
                mem_store_u32(addr + k * 4, wave.vgpr[inst.src[2].index + k][lane]);
        }
        break;
    }
    case Opcode::BUFFER_LOAD_UBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::BUFFER_LOAD_SBYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            int8_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 1);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    case Opcode::BUFFER_LOAD_USHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)v;
        }
        break;
    }
    case Opcode::BUFFER_LOAD_SSHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            int16_t v; memcpy(&v, reinterpret_cast<const void*>(addr), 2);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)v;
        }
        break;
    }
    case Opcode::BUFFER_LOAD_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
            wave.vgpr[inst.dst.index + 2][lane] = mem_load_u32(addr + 8);
        }
        break;
    }
    case Opcode::BUFFER_STORE_BYTE: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint8_t v = (uint8_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 1);
        }
        break;
    }
    case Opcode::BUFFER_STORE_SHORT: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint16_t v = (uint16_t)wave.vgpr[inst.src[2].index][lane];
            memcpy(reinterpret_cast<void*>(addr), &v, 2);
        }
        break;
    }
    case Opcode::BUFFER_STORE_DWORDX3: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr,     wave.vgpr[inst.src[2].index][lane]);
            mem_store_u32(addr + 4, wave.vgpr[inst.src[2].index + 1][lane]);
            mem_store_u32(addr + 8, wave.vgpr[inst.src[2].index + 2][lane]);
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_ADD: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old + wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_SUB: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old - wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_AND: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old & wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_OR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old | wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_XOR: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, old ^ wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_SMIN: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            int32_t old = (int32_t)mem_load_u32(addr);
            int32_t v   = (int32_t)wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (uint32_t)(v < old ? v : old));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = (uint32_t)old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_UMIN: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t v   = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, v < old ? v : old);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_SMAX: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            int32_t old = (int32_t)mem_load_u32(addr);
            int32_t v   = (int32_t)wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (uint32_t)(v > old ? v : old));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = (uint32_t)old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_UMAX: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t v   = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, v > old ? v : old);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_SWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_CMPSWAP: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old  = mem_load_u32(addr);
            uint32_t cmp  = wave.vgpr[inst.src[2].index][lane];
            uint32_t val  = wave.vgpr[inst.src[2].index + 1][lane];
            if (old == cmp) mem_store_u32(addr, val);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_INC: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t max_val = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, (old >= max_val) ? 0 : (old + 1));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_DEC: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t old = mem_load_u32(addr);
            uint32_t max_val = wave.vgpr[inst.src[2].index][lane];
            mem_store_u32(addr, ((old == 0) || (old > max_val)) ? max_val : (old - 1));
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = old;
        }
        break;
    }
    case Opcode::BUFFER_ATOMIC_ADD_F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            uint32_t oldb = mem_load_u32(addr);
            float old, v; memcpy(&old, &oldb, 4); memcpy(&v, &wave.vgpr[inst.src[2].index][lane], 4);
            float nv = old + v; uint32_t nvb; memcpy(&nvb, &nv, 4);
            mem_store_u32(addr, nvb);
            if (inst.dst.kind == Operand::Kind::VGPR)
                wave.vgpr[inst.dst.index][lane] = oldb;
        }
        break;
    }
    // TBUFFER ops: treat same as MUBUF for format-less simulation
    case Opcode::TBUFFER_LOAD_FORMAT_X: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane] = mem_load_u32(addr);
        }
        break;
    }
    case Opcode::TBUFFER_LOAD_FORMAT_XY: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
        }
        break;
    }
    case Opcode::TBUFFER_LOAD_FORMAT_XYZ: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
            wave.vgpr[inst.dst.index + 2][lane] = mem_load_u32(addr + 8);
        }
        break;
    }
    case Opcode::TBUFFER_LOAD_FORMAT_XYZW: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            wave.vgpr[inst.dst.index][lane]     = mem_load_u32(addr);
            wave.vgpr[inst.dst.index + 1][lane] = mem_load_u32(addr + 4);
            wave.vgpr[inst.dst.index + 2][lane] = mem_load_u32(addr + 8);
            wave.vgpr[inst.dst.index + 3][lane] = mem_load_u32(addr + 12);
        }
        break;
    }
    case Opcode::TBUFFER_STORE_FORMAT_X: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr, wave.vgpr[inst.src[2].index][lane]);
        }
        break;
    }
    case Opcode::TBUFFER_STORE_FORMAT_XY: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr,     wave.vgpr[inst.src[2].index][lane]);
            mem_store_u32(addr + 4, wave.vgpr[inst.src[2].index + 1][lane]);
        }
        break;
    }
    case Opcode::TBUFFER_STORE_FORMAT_XYZ: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr,     wave.vgpr[inst.src[2].index][lane]);
            mem_store_u32(addr + 4, wave.vgpr[inst.src[2].index + 1][lane]);
            mem_store_u32(addr + 8, wave.vgpr[inst.src[2].index + 2][lane]);
        }
        break;
    }
    case Opcode::TBUFFER_STORE_FORMAT_XYZW: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t base = read_src_scalar64(wave, inst.src[1]);
            uint32_t voff = inst.mubuf_offen ? wave.vgpr[inst.src[0].index][lane] : 0;
            uint64_t addr = base + voff + inst.mubuf_offset;
            mem_store_u32(addr,      wave.vgpr[inst.src[2].index][lane]);
            mem_store_u32(addr + 4,  wave.vgpr[inst.src[2].index + 1][lane]);
            mem_store_u32(addr + 8,  wave.vgpr[inst.src[2].index + 2][lane]);
            mem_store_u32(addr + 12, wave.vgpr[inst.src[2].index + 3][lane]);
        }
        break;
    }
    // -----------------------------------------------------------------------
    // MFMA — Matrix FMA instructions
    // -----------------------------------------------------------------------
    // Helper: BF16 to float conversion
    // BF16 is stored in upper 16 bits of a float32.
    // v_mfma_f32_4x4x1f32: simplest case, M=N=K=1, 16 lanes per block
    // Each lane has one (a,b,c) triple; result: d[lane] = a*b + c
    case Opcode::V_MFMA_F32_4X4X1F32: {
        // 4x4 output: 16 lanes hold the output, src0/src1 are scalars per lane.
        // src2 and dst may be VGPR (accumulator VGPR) or AGPR depending on
        // how the compiler allocated accumulators.
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            // Read accumulator from src2 (VGPR or AGPR)
            uint32_t c_bits;
            if (inst.src[2].kind == Operand::Kind::AGPR)
                c_bits = wave.agpr[inst.src[2].index][lane];
            else
                c_bits = read_vsrc(wave, inst.src[2], lane);
            float a, b, c;
            memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4); memcpy(&c, &c_bits, 4);
            float r = a * b + c;
            uint32_t r_bits; memcpy(&r_bits, &r, 4);
            // Write result to dst (VGPR or AGPR)
            if (inst.dst.kind == Operand::Kind::AGPR)
                wave.agpr[inst.dst.index][lane] = r_bits;
            else
                wave.vgpr[inst.dst.index][lane] = r_bits;
        }
        break;
    }
    // MFMA helper lambdas (defined as local static helpers via inline logic)
    // BF16 conversion: upper 16 bits of f32
    case Opcode::V_MFMA_F32_4X4X4F16: {
        // M=4, N=4, K=4: 4 blocks × 16 lanes per block = 64 lanes
        // Each lane produces one output element (row i, col j of a 4x4 sub-block)
        // src0[lane] = two f16 packed (a[k=0..1]), src1[lane] = two f16 (b[k=0..1])
        // For K=4: two VGPR pairs needed but simplified to single pair per lane
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
            uint32_t bits;
            if (e == 0) { bits = (s << 31) | (m << 13); }
            else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
            else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            // Each lane: a[0..1] from src0 (lo/hi 16 bits), b[0..1] from src1
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = f16_to_f32(a_pack & 0xFFFF), a1 = f16_to_f32(a_pack >> 16);
            float b0 = f16_to_f32(b_pack & 0xFFFF), b1 = f16_to_f32(b_pack >> 16);
            uint32_t cb = wave.agpr[inst.dst.index][lane];
            float c; memcpy(&c, &cb, 4);
            float r = fmaf(a0, b0, fmaf(a1, b1, c));
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_MFMA_F32_16X16X4F16:
    case Opcode::V_MFMA_F32_16X16X16F16: {
        // 16x16 output: 64 lanes, each holds one f32 output element
        // For K=4: each lane has 2 f16 from src0 and src1
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
            uint32_t bits;
            if (e == 0) { bits = (s << 31) | (m << 13); }
            else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
            else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = f16_to_f32(a_pack & 0xFFFF), a1 = f16_to_f32(a_pack >> 16);
            float b0 = f16_to_f32(b_pack & 0xFFFF), b1 = f16_to_f32(b_pack >> 16);
            uint32_t cb = wave.agpr[inst.dst.index][lane];
            float c; memcpy(&c, &cb, 4);
            float r = fmaf(a0, b0, fmaf(a1, b1, c));
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_MFMA_F32_32X32X4F16:
    case Opcode::V_MFMA_F32_32X32X8F16: {
        // 32x32 output: multiple AGPR registers hold the output
        // Simplified: process available K dimension from packed f16 in src0/src1
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
            uint32_t bits;
            if (e == 0) { bits = (s << 31) | (m << 13); }
            else if (e == 31) { bits = (s << 31) | (0xFF << 23) | (m << 13); }
            else { bits = (s << 31) | ((e + 112) << 23) | (m << 13); }
            float f; memcpy(&f, &bits, 4); return f;
        };
        // 32x32 output needs 16 AGPRs; each lane processes 1 element per AGPR
        int num_out_regs = 16;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = f16_to_f32(a_pack & 0xFFFF), a1 = f16_to_f32(a_pack >> 16);
            float b0 = f16_to_f32(b_pack & 0xFFFF), b1 = f16_to_f32(b_pack >> 16);
            for (int r = 0; r < num_out_regs; r++) {
                uint32_t cb = wave.agpr[inst.dst.index + r][lane];
                float c; memcpy(&c, &cb, 4);
                float res = fmaf(a0, b0, fmaf(a1, b1, c));
                uint32_t rb; memcpy(&rb, &res, 4);
                wave.agpr[inst.dst.index + r][lane] = rb;
            }
        }
        break;
    }
    case Opcode::V_MFMA_I32_4X4X4I8: {
        // src0/src1: 4 i8 values packed in one u32 per lane; src2: i32 accumulator
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            int32_t acc = (int32_t)wave.agpr[inst.dst.index][lane];
            for (int k = 0; k < 4; k++) {
                int8_t a = (int8_t)((a_pack >> (k*8)) & 0xFF);
                int8_t b = (int8_t)((b_pack >> (k*8)) & 0xFF);
                acc += (int32_t)a * (int32_t)b;
            }
            wave.agpr[inst.dst.index][lane] = (uint32_t)acc;
        }
        break;
    }
    case Opcode::V_MFMA_I32_16X16X4I8:
    case Opcode::V_MFMA_I32_16X16X16I8: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            int32_t acc = (int32_t)wave.agpr[inst.dst.index][lane];
            for (int k = 0; k < 4; k++) {
                int8_t a = (int8_t)((a_pack >> (k*8)) & 0xFF);
                int8_t b = (int8_t)((b_pack >> (k*8)) & 0xFF);
                acc += (int32_t)a * (int32_t)b;
            }
            wave.agpr[inst.dst.index][lane] = (uint32_t)acc;
        }
        break;
    }
    case Opcode::V_MFMA_I32_32X32X4I8:
    case Opcode::V_MFMA_I32_32X32X8I8: {
        int num_out_regs = 16;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            int32_t dot = 0;
            for (int k = 0; k < 4; k++) {
                int8_t a = (int8_t)((a_pack >> (k*8)) & 0xFF);
                int8_t b = (int8_t)((b_pack >> (k*8)) & 0xFF);
                dot += (int32_t)a * (int32_t)b;
            }
            for (int r = 0; r < num_out_regs; r++) {
                wave.agpr[inst.dst.index + r][lane] = (uint32_t)((int32_t)wave.agpr[inst.dst.index + r][lane] + dot);
            }
        }
        break;
    }
    // BF16 MFMA variants
    case Opcode::V_MFMA_F32_4X4X2BF16:
    case Opcode::V_MFMA_F32_4X4X4BF16_1K: {
        auto bf16_to_f32 = [](uint16_t b) -> float {
            uint32_t bits = (uint32_t)b << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = bf16_to_f32(a_pack & 0xFFFF), a1 = bf16_to_f32(a_pack >> 16);
            float b0 = bf16_to_f32(b_pack & 0xFFFF), b1 = bf16_to_f32(b_pack >> 16);
            uint32_t cb = wave.agpr[inst.dst.index][lane];
            float c; memcpy(&c, &cb, 4);
            float r = fmaf(a0, b0, fmaf(a1, b1, c));
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_MFMA_F32_16X16X2BF16:
    case Opcode::V_MFMA_F32_16X16X8BF16:
    case Opcode::V_MFMA_F32_16X16X4BF16_1K:
    case Opcode::V_MFMA_F32_16X16X16BF16_1K: {
        auto bf16_to_f32 = [](uint16_t b) -> float {
            uint32_t bits = (uint32_t)b << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = bf16_to_f32(a_pack & 0xFFFF), a1 = bf16_to_f32(a_pack >> 16);
            float b0 = bf16_to_f32(b_pack & 0xFFFF), b1 = bf16_to_f32(b_pack >> 16);
            uint32_t cb = wave.agpr[inst.dst.index][lane];
            float c; memcpy(&c, &cb, 4);
            float r = fmaf(a0, b0, fmaf(a1, b1, c));
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_MFMA_F32_32X32X2BF16:
    case Opcode::V_MFMA_F32_32X32X4BF16:
    case Opcode::V_MFMA_F32_32X32X4BF16_1K:
    case Opcode::V_MFMA_F32_32X32X8BF16_1K: {
        auto bf16_to_f32 = [](uint16_t b) -> float {
            uint32_t bits = (uint32_t)b << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        int num_out_regs = 16;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_pack = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_pack = read_vsrc(wave, inst.src[1], lane);
            float a0 = bf16_to_f32(a_pack & 0xFFFF), a1 = bf16_to_f32(a_pack >> 16);
            float b0 = bf16_to_f32(b_pack & 0xFFFF), b1 = bf16_to_f32(b_pack >> 16);
            float dot = a0 * b0 + a1 * b1;
            for (int r = 0; r < num_out_regs; r++) {
                uint32_t cb = wave.agpr[inst.dst.index + r][lane];
                float c; memcpy(&c, &cb, 4);
                float res = c + dot;
                uint32_t rb; memcpy(&rb, &res, 4);
                wave.agpr[inst.dst.index + r][lane] = rb;
            }
        }
        break;
    }
    // F32 MFMA larger sizes
    case Opcode::V_MFMA_F32_16X16X1F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_bits = wave.agpr[inst.dst.index][lane];
            float a, b, c; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4); memcpy(&c, &c_bits, 4);
            float r = fmaf(a, b, c);
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_MFMA_F32_32X32X1F32:
    case Opcode::V_MFMA_F32_32X32X2F32: {
        int num_out_regs = 16;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            float prod = a * b;
            for (int r = 0; r < num_out_regs; r++) {
                uint32_t cb = wave.agpr[inst.dst.index + r][lane];
                float c; memcpy(&c, &cb, 4);
                float res = c + prod;
                uint32_t rb; memcpy(&rb, &res, 4);
                wave.agpr[inst.dst.index + r][lane] = rb;
            }
        }
        break;
    }
    case Opcode::V_MFMA_F32_16X16X4F32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_bits = read_vsrc(wave, inst.src[0], lane);
            uint32_t b_bits = read_vsrc(wave, inst.src[1], lane);
            float a, b; memcpy(&a, &a_bits, 4); memcpy(&b, &b_bits, 4);
            uint32_t cb = wave.agpr[inst.dst.index][lane];
            float c; memcpy(&c, &cb, 4);
            float r = fmaf(a, b, c);
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.agpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    // F64 MFMA
    case Opcode::V_MFMA_F64_4X4X4F64: {
        // 4x4 F64 MFMA: src0/src1 are f64 VGPR pairs, accumulator is AGPR pair
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = (uint64_t)wave.vgpr[inst.src[0].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32);
            uint64_t b64 = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            uint64_t c64 = (uint64_t)wave.agpr[inst.dst.index][lane] |
                           ((uint64_t)wave.agpr[inst.dst.index + 1][lane] << 32);
            double a, b, c; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8); memcpy(&c, &c64, 8);
            double r = fma(a, b, c);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.agpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.agpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_MFMA_F64_16X16X4F64: {
        // 16x16 F64: each lane holds one output f64 in an AGPR pair
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a64 = (uint64_t)wave.vgpr[inst.src[0].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32);
            uint64_t b64 = (uint64_t)wave.vgpr[inst.src[1].index][lane] |
                           ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32);
            uint64_t c64 = (uint64_t)wave.agpr[inst.dst.index][lane] |
                           ((uint64_t)wave.agpr[inst.dst.index + 1][lane] << 32);
            double a, b, c; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8); memcpy(&c, &c64, 8);
            double r = fma(a, b, c);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.agpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.agpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    // -----------------------------------------------------------------------
    // VOP3P (packed 16-bit) — stubs
    // -----------------------------------------------------------------------
    case Opcode::V_PK_MOV_B32: {
        // Move VGPR pair: dst[0] = src0, dst[1] = src1 (ignores op_sel)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane]     = read_vsrc(wave, inst.src[0], lane);
            wave.vgpr[inst.dst.index + 1][lane] = read_vsrc(wave, inst.src[1], lane);
        }
        break;
    }
    case Opcode::V_PK_ADD_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            // Process lo/hi halves separately (simplified: treat as u16 add for now)
            uint16_t a_lo = a & 0xFFFF, a_hi = a >> 16;
            uint16_t b_lo = b & 0xFFFF, b_hi = b >> 16;
            // Convert f16->f32, add, convert back (simplified bit manipulation)
            wave.vgpr[inst.dst.index][lane] = ((uint32_t)(a_hi + b_hi) << 16) | (a_lo + b_lo);
        }
        break;
    }
    case Opcode::V_DOT4_I32_I8: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            int32_t  c = (int32_t)read_vsrc(wave, inst.src[2], lane);
            int32_t r = c;
            for (int k = 0; k < 4; k++) {
                int8_t  ai = (int8_t)((a >> (k * 8)) & 0xFF);
                int8_t  bi = (int8_t)((b >> (k * 8)) & 0xFF);
                r += (int32_t)ai * bi;
            }
            wave.vgpr[inst.dst.index][lane] = (uint32_t)r;
        }
        break;
    }
    case Opcode::V_DOT4_U32_U8: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c = read_vsrc(wave, inst.src[2], lane);
            uint32_t r = c;
            for (int k = 0; k < 4; k++) {
                r += ((a >> (k * 8)) & 0xFF) * ((b >> (k * 8)) & 0xFF);
            }
            wave.vgpr[inst.dst.index][lane] = r;
        }
        break;
    }
    case Opcode::V_DOT2_F32_F16: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            // Simplified: treat f16 as raw bits, accumulate into f32
            uint32_t a = read_vsrc(wave, inst.src[0], lane);
            uint32_t b = read_vsrc(wave, inst.src[1], lane);
            uint32_t c_bits = read_vsrc(wave, inst.src[2], lane);
            float c; memcpy(&c, &c_bits, 4);
            // Just use c as output (proper f16 mul would need f16->f32 conversion)
            (void)a; (void)b;
            wave.vgpr[inst.dst.index][lane] = c_bits;
        }
        break;
    }

    // -----------------------------------------------------------------------
    // VOP1 — F64 transcendentals and conversions
    // -----------------------------------------------------------------------
    case Opcode::V_CVT_F32_F64: {
        // Convert f64 src (VGPR pair) to f32 dst
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            float f = (float)d;
            uint32_t fb; memcpy(&fb, &f, 4);
            wave.vgpr[inst.dst.index][lane] = fb;
        }
        break;
    }
    case Opcode::V_CVT_F64_F32: {
        // Convert f32 src to f64 dst (VGPR pair)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t sb = read_vsrc(wave, inst.src[0], lane);
            float f; memcpy(&f, &sb, 4);
            double d = (double)f;
            uint64_t bits64; memcpy(&bits64, &d, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(bits64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(bits64 >> 32);
        }
        break;
    }
    case Opcode::V_CVT_F64_I32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t sb = read_vsrc(wave, inst.src[0], lane);
            double d = (double)(int32_t)sb;
            uint64_t bits64; memcpy(&bits64, &d, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(bits64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(bits64 >> 32);
        }
        break;
    }
    case Opcode::V_CVT_F64_U32: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t sb = read_vsrc(wave, inst.src[0], lane);
            double d = (double)sb;
            uint64_t bits64; memcpy(&bits64, &d, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(bits64 & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(bits64 >> 32);
        }
        break;
    }
    case Opcode::V_CVT_I32_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)d;
        }
        break;
    }
    case Opcode::V_CVT_U32_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(uint64_t)d;
        }
        break;
    }
    case Opcode::V_RCP_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = 1.0 / d;
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_RSQ_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = 1.0 / sqrt(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_SQRT_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = sqrt(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_FREXP_MANT_F32: {
        // Returns mantissa portion: src = m * 2^e, result = m in [-1,-0.5) U [0.5,1)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t sb = read_vsrc(wave, inst.src[0], lane);
            float f; memcpy(&f, &sb, 4);
            int exp;
            float m = frexpf(f, &exp);
            uint32_t rb; memcpy(&rb, &m, 4);
            wave.vgpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_FREXP_EXP_I32_F32: {
        // Returns exponent: src = m * 2^e, result = e as int32
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t sb = read_vsrc(wave, inst.src[0], lane);
            float f; memcpy(&f, &sb, 4);
            int exp;
            frexpf(f, &exp);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)exp;
        }
        break;
    }
    case Opcode::V_FREXP_MANT_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            int exp;
            double m = frexp(d, &exp);
            uint64_t rb; memcpy(&rb, &m, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_FREXP_EXP_I32_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            int exp;
            frexp(d, &exp);
            wave.vgpr[inst.dst.index][lane] = (uint32_t)(int32_t)exp;
        }
        break;
    }
    case Opcode::V_FRACT_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = d - floor(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_TRUNC_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = trunc(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_CEIL_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = ceil(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_FLOOR_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = floor(d);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_RNDNE_F64: {
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            double r = rint(d); // rounds to nearest even
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_MOVRELD_B32: {
        // VGPR[dst + M0] = VGPR[src]; only for active lanes
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t val = read_vsrc(wave, inst.src[0], lane);
            uint32_t idx = inst.dst.index + wave.m0;
            if (idx < NUM_VGPR) wave.vgpr[idx][lane] = val;
        }
        break;
    }
    case Opcode::V_MOVRELS_B32: {
        // VGPR[dst] = VGPR[src + M0]; only for active lanes
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t idx = inst.src[0].index + wave.m0;
            uint32_t val = (idx < NUM_VGPR) ? wave.vgpr[idx][lane] : 0;
            wave.vgpr[inst.dst.index][lane] = val;
        }
        break;
    }
    // -----------------------------------------------------------------------
    // VOPC — f64 comparisons
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LT_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a < b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_EQ_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a == b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_LE_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a <= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GT_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a > b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_LG_F64: {
        // unordered not-equal (a != b, no NaN check)
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a != b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GE_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (a >= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_NEQ_F64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t a_lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t a_hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint32_t b_lo = wave.vgpr[inst.src[1].index][lane];
            uint32_t b_hi = wave.vgpr[inst.src[1].index + 1][lane];
            uint64_t a64 = (uint64_t)a_lo | ((uint64_t)a_hi << 32);
            uint64_t b64 = (uint64_t)b_lo | ((uint64_t)b_hi << 32);
            double a, b; memcpy(&a, &a64, 8); memcpy(&b, &b64, 8);
            if (!(a == b)) mask |= (1ULL << lane); // ordered not-equal
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    // -----------------------------------------------------------------------
    // VOPC — i64 comparisons (LE, GT missing)
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LE_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int64_t)(((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane]);
            int64_t b = (int64_t)(((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane]);
            if (a <= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GT_I64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int64_t)(((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane]);
            int64_t b = (int64_t)(((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane]);
            if (a > b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    // -----------------------------------------------------------------------
    // VOPC — u64 comparisons (LE, GT missing)
    // -----------------------------------------------------------------------
    case Opcode::V_CMP_LE_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a <= b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    case Opcode::V_CMP_GT_U64: {
        uint64_t mask = 0;
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = ((uint64_t)wave.vgpr[inst.src[0].index + 1][lane] << 32) | wave.vgpr[inst.src[0].index][lane];
            uint64_t b = ((uint64_t)wave.vgpr[inst.src[1].index + 1][lane] << 32) | wave.vgpr[inst.src[1].index][lane];
            if (a > b) mask |= (1ULL << lane);
        }
        if (inst.dst.kind == Operand::Kind::VCC || inst.dst.kind == Operand::Kind::NONE)
            wave.vcc = mask;
        else write_sdst64(wave, inst.dst, mask);
        break;
    }
    // -----------------------------------------------------------------------
    // VOP3A — additional f64 ops
    // -----------------------------------------------------------------------
    case Opcode::V_DIV_FIXUP_F64: {
        // Fixup result of double division: computes correct sign/NaN
        // Simplified: just pass through src2 (the approximate quotient)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane]     = wave.vgpr[inst.src[2].index][lane];
            wave.vgpr[inst.dst.index + 1][lane] = wave.vgpr[inst.src[2].index + 1][lane];
        }
        break;
    }
    case Opcode::V_LDEXP_F64: {
        // dst = src0 * 2^src1 (src1 is int32)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t lo = wave.vgpr[inst.src[0].index][lane];
            uint32_t hi = wave.vgpr[inst.src[0].index + 1][lane];
            uint64_t bits64 = (uint64_t)lo | ((uint64_t)hi << 32);
            double d; memcpy(&d, &bits64, 8);
            int32_t exp = (int32_t)read_vsrc(wave, inst.src[1], lane);
            double r = ldexp(d, exp);
            uint64_t rb; memcpy(&rb, &r, 8);
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(rb & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(rb >> 32);
        }
        break;
    }
    case Opcode::V_MAD_U64_U32: {
        // 64-bit multiply-add: dst[63:0] = src0 * src1 + src2[63:0]
        // dst  = VGPR pair [dst, dst+1]
        // dst2 = SGPR pair (carry/overflow, not precisely defined — set to 0)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint64_t a = (uint64_t)read_vsrc(wave, inst.src[0], lane);
            uint64_t b = (uint64_t)read_vsrc(wave, inst.src[1], lane);
            // src2 is a 64-bit VGPR pair
            uint64_t c = (uint64_t)read_vsrc(wave, inst.src[2], lane) |
                         ((uint64_t)wave.vgpr[inst.src[2].index + 1][lane] << 32);
            uint64_t r = a * b + c;
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)(r & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)(r >> 32);
        }
        // dst2 (carry) = 0 for the general case
        if (inst.dst2.kind == Operand::Kind::SGPR) {
            wave.sgpr[inst.dst2.index]     = 0;
            wave.sgpr[inst.dst2.index + 1] = 0;
        }
        break;
    }
    case Opcode::V_MAD_I64_I32: {
        // Signed 64-bit multiply-add
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            int64_t a = (int64_t)(int32_t)read_vsrc(wave, inst.src[0], lane);
            int64_t b = (int64_t)(int32_t)read_vsrc(wave, inst.src[1], lane);
            uint64_t c = (uint64_t)read_vsrc(wave, inst.src[2], lane) |
                         ((uint64_t)wave.vgpr[inst.src[2].index + 1][lane] << 32);
            int64_t r = a * b + (int64_t)c;
            wave.vgpr[inst.dst.index][lane]     = (uint32_t)((uint64_t)r & 0xFFFFFFFF);
            wave.vgpr[inst.dst.index + 1][lane] = (uint32_t)((uint64_t)r >> 32);
        }
        if (inst.dst2.kind == Operand::Kind::SGPR) {
            wave.sgpr[inst.dst2.index]     = 0;
            wave.sgpr[inst.dst2.index + 1] = 0;
        }
        break;
    }
    case Opcode::V_DIV_FMAS_F32: {
        // Simplified: same as fma(src0,src1,src2)
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            uint32_t ab = read_vsrc(wave, inst.src[0], lane);
            uint32_t bb = read_vsrc(wave, inst.src[1], lane);
            uint32_t cb = read_vsrc(wave, inst.src[2], lane);
            float a, b, c; memcpy(&a, &ab, 4); memcpy(&b, &bb, 4); memcpy(&c, &cb, 4);
            float r = fmaf(a, b, c);
            uint32_t rb; memcpy(&rb, &r, 4);
            wave.vgpr[inst.dst.index][lane] = rb;
        }
        break;
    }
    case Opcode::V_DIV_FIXUP_F32: {
        // Simplified: pass through src2
        for (int lane = 0; lane < WAVE_SIZE; lane++) {
            if (!((wave.exec >> lane) & 1)) continue;
            wave.vgpr[inst.dst.index][lane] = read_vsrc(wave, inst.src[2], lane);
        }
        break;
    }

    default:
        // Unknown or unimplemented — skip
        break;
    }
}

} // namespace gcnsim
