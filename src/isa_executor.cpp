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

    default:
        // Unknown or unimplemented — skip
        break;
    }
}

} // namespace gcnsim
