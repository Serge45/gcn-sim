#include <gtest/gtest.h>
#include "gcnsim/wavefront.h"
#include "gcnsim/isa_decoder.h"
#include "gcnsim/kernel_descriptor.h"
#include <cstring>
#include <cstddef>
#include <cmath>
#include <memory>

namespace gcnsim {
// Forward declarations from isa_executor.cpp / wavefront.cpp
void execute_instruction(WavefrontState& wave, const Instruction& inst);
WaveStatus run_wavefront(WavefrontState& wave);
void init_wavefront(WavefrontState& wave,
                    const KernelDescriptor& kd,
                    const uint8_t* kernarg_ptr,
                    const uint8_t* code_ptr,
                    size_t code_size,
                    uint32_t wg_id_x, uint32_t wg_id_y, uint32_t wg_id_z,
                    uint32_t wave_in_wg, uint32_t waves_per_wg,
                    uint32_t block_x, uint32_t block_y, uint32_t block_z);
}

using namespace gcnsim;

// Helper: create a zeroed WavefrontState on the heap (too large for stack)
static std::unique_ptr<WavefrontState> make_wave() {
    auto ws = std::make_unique<WavefrontState>();
    // Zero POD fields only; std::vector members are default-constructed by make_unique
    memset((void*)ws.get(), 0, offsetof(WavefrontState, lgkm_pending));
    ws->exec = ~0ULL;  // all 64 lanes active
    return ws;
}

// Helper: build a minimal Instruction with the given opcode/format
static Instruction make_inst(Opcode op, InstFormat fmt) {
    Instruction inst{};
    inst.opcode = op;
    inst.format = fmt;
    inst.size_bytes = 4;
    return inst;
}

// Helper: float-to-uint32 bit cast
static uint32_t f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

// Helper: uint32-to-float bit cast
static float u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// ===== v_cndmask_b32 tests =====
TEST(ISAExecutor, VCndmaskB32_VCC) {
    auto ws = make_wave();
    // Set up: src0 = 10, src1 = 20 for all lanes
    // VCC: odd lanes set (lane & 1)
    // Expected: even lanes get src0 (10), odd lanes get src1 (20)
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = 10;
        ws->vgpr[2][lane] = 20;
    }
    ws->vcc = 0xAAAAAAAAAAAAAAAAULL; // odd lanes

    Instruction inst = make_inst(Opcode::V_CNDMASK_B32, InstFormat::VOP2);
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        if (lane & 1) {
            EXPECT_EQ(ws->vgpr[0][lane], 20u) << "lane " << lane << " (VCC=1) should get src1";
        } else {
            EXPECT_EQ(ws->vgpr[0][lane], 10u) << "lane " << lane << " (VCC=0) should get src0";
        }
    }
}

TEST(ISAExecutor, VCndmaskB32_ExecMask) {
    auto ws = make_wave();
    ws->exec = 0x0000000000000001ULL; // only lane 0 active
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[0][lane] = 0xDEAD; // sentinel
        ws->vgpr[1][lane] = 100;
        ws->vgpr[2][lane] = 200;
    }
    ws->vcc = ~0ULL; // all VCC bits set

    Instruction inst = make_inst(Opcode::V_CNDMASK_B32, InstFormat::VOP2);
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    EXPECT_EQ(ws->vgpr[0][0], 200u) << "lane 0 active, VCC=1 → src1";
    EXPECT_EQ(ws->vgpr[0][1], 0xDEADu) << "lane 1 inactive → unchanged";
}

// ===== s_cselect_b32 tests =====
TEST(ISAExecutor, SCselectB32_SccSet) {
    auto ws = make_wave();
    ws->sgpr[1] = 42;
    ws->sgpr[2] = 99;
    ws->scc = 1;

    Instruction inst = make_inst(Opcode::S_CSELECT_B32, InstFormat::SOP2);
    inst.dst.kind = Operand::Kind::SGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::SGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::SGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    EXPECT_EQ(ws->sgpr[0], 42u) << "SCC=1 → dst = src0";
}

TEST(ISAExecutor, SCselectB32_SccClear) {
    auto ws = make_wave();
    ws->sgpr[1] = 42;
    ws->sgpr[2] = 99;
    ws->scc = 0;

    Instruction inst = make_inst(Opcode::S_CSELECT_B32, InstFormat::SOP2);
    inst.dst.kind = Operand::Kind::SGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::SGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::SGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    EXPECT_EQ(ws->sgpr[0], 99u) << "SCC=0 → dst = src1";
}

// ===== VOP3 source modifiers (NEG, ABS) =====
TEST(ISAExecutor, VOP3_NegModifier) {
    auto ws = make_wave();
    // v_add_f32 v0, -v1, v2  (neg bit 0 set)
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(3.0f);
        ws->vgpr[2][lane] = f2u(10.0f);
    }

    Instruction inst = make_inst(Opcode::V_ADD_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.neg = 0x1; // negate src0

    execute_instruction(*ws, inst);

    // Expected: -3.0 + 10.0 = 7.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 7.0f) << "lane " << lane;
    }
}

TEST(ISAExecutor, VOP3_AbsModifier) {
    auto ws = make_wave();
    // v_add_f32 v0, |v1|, v2  (abs bit 0 set)
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(-5.0f);
        ws->vgpr[2][lane] = f2u(1.0f);
    }

    Instruction inst = make_inst(Opcode::V_ADD_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.abs_ = 0x1; // abs src0

    execute_instruction(*ws, inst);

    // Expected: |-5.0| + 1.0 = 6.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 6.0f) << "lane " << lane;
    }
}

TEST(ISAExecutor, VOP3_NegAbsCombined) {
    auto ws = make_wave();
    // v_add_f32 v0, -|v1|, v2  (neg bit 0 + abs bit 0)
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(-5.0f);
        ws->vgpr[2][lane] = f2u(10.0f);
    }

    Instruction inst = make_inst(Opcode::V_ADD_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.neg = 0x1;
    inst.abs_ = 0x1;

    execute_instruction(*ws, inst);

    // Expected: -|-5.0| + 10.0 = -5.0 + 10.0 = 5.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 5.0f) << "lane " << lane;
    }
}

// ===== VOP3 destination modifiers (CLAMP, OMOD) =====
TEST(ISAExecutor, VOP3_ClampModifier) {
    auto ws = make_wave();
    // v_add_f32 v0, v1, v2 clamp — result should be clamped to [0,1]
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(0.8f);
        ws->vgpr[2][lane] = f2u(0.5f);
    }

    Instruction inst = make_inst(Opcode::V_ADD_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.clamp = 1;

    execute_instruction(*ws, inst);

    // 0.8 + 0.5 = 1.3 → clamp to 1.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 1.0f) << "lane " << lane;
    }
}

TEST(ISAExecutor, VOP3_OmodMul2) {
    auto ws = make_wave();
    // v_add_f32 v0, v1, v2 mul:2
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(3.0f);
        ws->vgpr[2][lane] = f2u(2.0f);
    }

    Instruction inst = make_inst(Opcode::V_ADD_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.omod = 1; // *2

    execute_instruction(*ws, inst);

    // (3.0 + 2.0) * 2 = 10.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 10.0f) << "lane " << lane;
    }
}

TEST(ISAExecutor, VOP3_OmodDiv2) {
    auto ws = make_wave();
    // v_mul_f32 v0, v1, v2 div:2
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(4.0f);
        ws->vgpr[2][lane] = f2u(3.0f);
    }

    Instruction inst = make_inst(Opcode::V_MUL_F32, InstFormat::VOP3A);
    inst.size_bytes = 8;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;
    inst.omod = 3; // /2

    execute_instruction(*ws, inst);

    // (4.0 * 3.0) / 2 = 6.0
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 6.0f) << "lane " << lane;
    }
}

// ===== s_getpc_b64 returns actual PC =====
TEST(ISAExecutor, SGetpcB64_ReturnsPC) {
    auto ws = make_wave();
    ws->pc = 0x100; // Current PC (already advanced past s_getpc)

    Instruction inst = make_inst(Opcode::S_GETPC_B64, InstFormat::SOP1);
    inst.dst.kind = Operand::Kind::SGPR;
    inst.dst.index = 0;

    execute_instruction(*ws, inst);

    uint64_t result = ((uint64_t)ws->sgpr[1] << 32) | ws->sgpr[0];
    EXPECT_EQ(result, 0x100ULL) << "s_getpc should return wave.pc";
}

// ===== s_barrier returns BARRIER_HIT =====
TEST(ISAExecutor, SBarrier_ReturnsBarrierHit) {
    auto ws = make_wave();
    // Build a tiny code buffer: s_barrier (0xBF8A0000) followed by s_endpgm (0xBF810000)
    uint32_t code[] = { 0xBF8A0000, 0xBF810000 };
    ws->kernel_code_ptr = reinterpret_cast<const uint8_t*>(code);
    ws->kernel_code_size = sizeof(code);
    ws->pc = 0;

    WaveStatus status = run_wavefront(*ws);
    EXPECT_EQ(status, WaveStatus::BARRIER_HIT) << "s_barrier should return BARRIER_HIT";
    EXPECT_EQ(ws->pc, 4u) << "PC should advance past s_barrier";
}

TEST(ISAExecutor, SBarrier_ThenEndpgm) {
    auto ws = make_wave();
    // s_barrier, s_endpgm
    uint32_t code[] = { 0xBF8A0000, 0xBF810000 };
    ws->kernel_code_ptr = reinterpret_cast<const uint8_t*>(code);
    ws->kernel_code_size = sizeof(code);
    ws->pc = 0;

    // First call: barrier
    WaveStatus s1 = run_wavefront(*ws);
    EXPECT_EQ(s1, WaveStatus::BARRIER_HIT);

    // Second call: endpgm
    WaveStatus s2 = run_wavefront(*ws);
    EXPECT_EQ(s2, WaveStatus::ENDED);
}

// ===== Scratch memory via FLAT_SCRATCH =====
TEST(ISAExecutor, ScratchStoreLoadDword) {
    auto ws = make_wave();
    ws->exec = 1ULL; // only lane 0

    // Allocate scratch: 256 bytes per lane
    uint32_t scratch_per_lane = 256;
    std::vector<uint8_t> scratch(scratch_per_lane * WAVE_SIZE, 0);
    ws->scratch_ptr = scratch.data();
    ws->scratch_size = (uint32_t)scratch.size();
    ws->scratch_per_lane = scratch_per_lane;

    // Store 0xCAFEBABE to scratch at offset 0 for lane 0
    ws->vgpr[0][0] = 0; // vaddr offset = 0
    ws->vgpr[1][0] = 0xCAFEBABEu; // data

    Instruction store_inst{};
    store_inst.opcode = Opcode::FLAT_STORE_DWORD;
    store_inst.format = InstFormat::FLAT_SCRATCH;
    store_inst.size_bytes = 8;
    store_inst.src[0].kind = Operand::Kind::VGPR;
    store_inst.src[0].index = 0; // vaddr
    store_inst.src[2].kind = Operand::Kind::VGPR;
    store_inst.src[2].index = 1; // vdata
    store_inst.flat_offset = 0;

    execute_instruction(*ws, store_inst);

    // Now load it back
    ws->vgpr[3][0] = 0; // clear dst

    Instruction load_inst{};
    load_inst.opcode = Opcode::FLAT_LOAD_DWORD;
    load_inst.format = InstFormat::FLAT_SCRATCH;
    load_inst.size_bytes = 8;
    load_inst.dst.kind = Operand::Kind::VGPR;
    load_inst.dst.index = 3;
    load_inst.src[0].kind = Operand::Kind::VGPR;
    load_inst.src[0].index = 0; // vaddr
    load_inst.flat_offset = 0;

    execute_instruction(*ws, load_inst);

    EXPECT_EQ(ws->vgpr[3][0], 0xCAFEBABEu) << "Should load back stored value from scratch";
}

TEST(ISAExecutor, ScratchPerLaneIsolation) {
    auto ws = make_wave();
    ws->exec = 3ULL; // lanes 0 and 1

    uint32_t scratch_per_lane = 64;
    std::vector<uint8_t> scratch(scratch_per_lane * WAVE_SIZE, 0);
    ws->scratch_ptr = scratch.data();
    ws->scratch_size = (uint32_t)scratch.size();
    ws->scratch_per_lane = scratch_per_lane;

    // Lane 0 stores 111, lane 1 stores 222, both at offset 0
    ws->vgpr[0][0] = 0;
    ws->vgpr[0][1] = 0;
    ws->vgpr[1][0] = 111;
    ws->vgpr[1][1] = 222;

    Instruction store_inst{};
    store_inst.opcode = Opcode::FLAT_STORE_DWORD;
    store_inst.format = InstFormat::FLAT_SCRATCH;
    store_inst.size_bytes = 8;
    store_inst.src[0].kind = Operand::Kind::VGPR;
    store_inst.src[0].index = 0;
    store_inst.src[2].kind = Operand::Kind::VGPR;
    store_inst.src[2].index = 1;
    store_inst.flat_offset = 0;

    execute_instruction(*ws, store_inst);

    // Load back: each lane should see its own value
    Instruction load_inst{};
    load_inst.opcode = Opcode::FLAT_LOAD_DWORD;
    load_inst.format = InstFormat::FLAT_SCRATCH;
    load_inst.size_bytes = 8;
    load_inst.dst.kind = Operand::Kind::VGPR;
    load_inst.dst.index = 3;
    load_inst.src[0].kind = Operand::Kind::VGPR;
    load_inst.src[0].index = 0;
    load_inst.flat_offset = 0;

    execute_instruction(*ws, load_inst);

    EXPECT_EQ(ws->vgpr[3][0], 111u) << "Lane 0 scratch isolation";
    EXPECT_EQ(ws->vgpr[3][1], 222u) << "Lane 1 scratch isolation";
}

// ===== V_MUL_F32 basic (sanity check for modifier baseline) =====
TEST(ISAExecutor, VMulF32_NoMod) {
    auto ws = make_wave();
    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        ws->vgpr[1][lane] = f2u(3.0f);
        ws->vgpr[2][lane] = f2u(4.0f);
    }

    Instruction inst = make_inst(Opcode::V_MUL_F32, InstFormat::VOP2);
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    for (int lane = 0; lane < WAVE_SIZE; lane++) {
        EXPECT_FLOAT_EQ(u2f(ws->vgpr[0][lane]), 12.0f);
    }
}

// ===== M0 Register Read/Write =====
TEST(ISAExecutor, M0_ReadWrite) {
    auto ws = make_wave();
    ws->sgpr[0] = 0x12345678;
    ws->m0 = 0;

    // s_mov_b32 m0, s0
    Instruction mov_to_m0 = make_inst(Opcode::S_MOV_B32, InstFormat::SOP1);
    mov_to_m0.dst.kind = Operand::Kind::M0;
    mov_to_m0.src[0].kind = Operand::Kind::SGPR;
    mov_to_m0.src[0].index = 0;
    execute_instruction(*ws, mov_to_m0);
    EXPECT_EQ(ws->m0, 0x12345678u);

    // s_mov_b32 s1, m0
    Instruction mov_from_m0 = make_inst(Opcode::S_MOV_B32, InstFormat::SOP1);
    mov_from_m0.dst.kind = Operand::Kind::SGPR;
    mov_from_m0.dst.index = 1;
    mov_from_m0.src[0].kind = Operand::Kind::M0;
    execute_instruction(*ws, mov_from_m0);
    EXPECT_EQ(ws->sgpr[1], 0x12345678u);

    // v_mov_b32 v0, m0
    Instruction vmov = make_inst(Opcode::V_MOV_B32, InstFormat::VOP1);
    vmov.dst.kind = Operand::Kind::VGPR;
    vmov.dst.index = 0;
    vmov.src[0].kind = Operand::Kind::M0;
    execute_instruction(*ws, vmov);
    EXPECT_EQ(ws->vgpr[0][0], 0x12345678u);
    EXPECT_EQ(ws->vgpr[0][63], 0x12345678u);
}

// ===== S_SWAPPC_B64 Call / Return =====
TEST(ISAExecutor, S_SWAPPC_B64_CallReturn) {
    auto ws = make_wave();
    ws->pc = 0x1000;
    ws->sgpr[0] = 0x5000; // Target func PC (low)
    ws->sgpr[1] = 0x0000; // Target func PC (high)

    // s_swappc_b64 s[2:3], s[0:1]
    Instruction swappc = make_inst(Opcode::S_SWAPPC_B64, InstFormat::SOP1);
    swappc.dst.kind = Operand::Kind::SGPR;
    swappc.dst.index = 2; // Return PC destination
    swappc.src[0].kind = Operand::Kind::SGPR;
    swappc.src[0].index = 0; // Target PC source

    execute_instruction(*ws, swappc);

    EXPECT_EQ(ws->pc, 0x5000u) << "PC should jump to target";
    EXPECT_EQ(ws->sgpr[2], 0x1000u) << "Return address saved into s[2]";
    EXPECT_EQ(ws->sgpr[3], 0x0000u) << "Return address high saved into s[3]";

    // Return: s_setpc_b64 s[2:3]
    Instruction setpc = make_inst(Opcode::S_SETPC_B64, InstFormat::SOP1);
    setpc.src[0].kind = Operand::Kind::SGPR;
    setpc.src[0].index = 2;

    execute_instruction(*ws, setpc);
    EXPECT_EQ(ws->pc, 0x1000u) << "PC should return to saved address";
}

// ===== GLOBAL_ATOMIC_ADD_F64 =====
TEST(ISAExecutor, GLOBAL_ATOMIC_ADD_F64) {
    auto ws = make_wave();
    double mem_val = 100.5;
    uint64_t mem_addr = reinterpret_cast<uint64_t>(&mem_val);

    // Setup lane 0 to add 25.25 with return value (glc)
    ws->exec = 1ULL; // Lane 0 active only
    ws->vgpr[0][0] = (uint32_t)(mem_addr & 0xFFFFFFFF);
    ws->vgpr[1][0] = (uint32_t)(mem_addr >> 32);

    double add_val = 25.25;
    uint64_t add_bits;
    std::memcpy(&add_bits, &add_val, sizeof(add_bits));
    ws->vgpr[2][0] = (uint32_t)(add_bits & 0xFFFFFFFF);
    ws->vgpr[3][0] = (uint32_t)(add_bits >> 32);

    Instruction atomic_add{};
    atomic_add.opcode = Opcode::GLOBAL_ATOMIC_ADD_F64;
    atomic_add.format = InstFormat::FLAT_GLOBAL;
    atomic_add.size_bytes = 8;
    atomic_add.src[0].kind = Operand::Kind::VGPR;
    atomic_add.src[0].index = 0; // v[0:1] = addr
    atomic_add.src[1].kind = Operand::Kind::NONE; // saddr = off
    atomic_add.src[2].kind = Operand::Kind::VGPR;
    atomic_add.src[2].index = 2; // v[2:3] = data
    atomic_add.dst.kind = Operand::Kind::VGPR;
    atomic_add.dst.index = 4; // v[4:5] = return old val

    execute_instruction(*ws, atomic_add);

    EXPECT_DOUBLE_EQ(mem_val, 125.75);

    // Verify old value returned in v[4:5]
    uint64_t ret_bits = ((uint64_t)ws->vgpr[5][0] << 32) | ws->vgpr[4][0];
    double ret_val;
    std::memcpy(&ret_val, &ret_bits, sizeof(ret_val));
    EXPECT_DOUBLE_EQ(ret_val, 100.5);
}

// ===== GLOBAL_ATOMIC_ADD_X2 =====
TEST(ISAExecutor, GLOBAL_ATOMIC_ADD_X2) {
    auto ws = make_wave();
    uint64_t mem_val = 0x1000000000000000ULL;
    uint64_t mem_addr = reinterpret_cast<uint64_t>(&mem_val);

    ws->exec = 1ULL; // Lane 0 active
    ws->vgpr[0][0] = (uint32_t)(mem_addr & 0xFFFFFFFF);
    ws->vgpr[1][0] = (uint32_t)(mem_addr >> 32);

    uint64_t add_val = 0x000000000000002AULL; // 42
    ws->vgpr[2][0] = (uint32_t)(add_val & 0xFFFFFFFF);
    ws->vgpr[3][0] = (uint32_t)(add_val >> 32);

    Instruction atomic_add{};
    atomic_add.opcode = Opcode::GLOBAL_ATOMIC_ADD_X2;
    atomic_add.format = InstFormat::FLAT_GLOBAL;
    atomic_add.size_bytes = 8;
    atomic_add.src[0].kind = Operand::Kind::VGPR;
    atomic_add.src[0].index = 0;
    atomic_add.src[1].kind = Operand::Kind::NONE;
    atomic_add.src[2].kind = Operand::Kind::VGPR;
    atomic_add.src[2].index = 2;
    atomic_add.dst.kind = Operand::Kind::VGPR;
    atomic_add.dst.index = 4;

    execute_instruction(*ws, atomic_add);

    EXPECT_EQ(mem_val, 0x100000000000002AULL);
    uint64_t ret_val = ((uint64_t)ws->vgpr[5][0] << 32) | ws->vgpr[4][0];
    EXPECT_EQ(ret_val, 0x1000000000000000ULL);
}

// ===== V_PERMLANE16_B32 & V_PERMLANEX16_B32 =====
TEST(ISAExecutor, V_PERMLANE16_B32) {
    auto ws = make_wave();
    for (int i = 0; i < 64; i++) {
        ws->vgpr[0][i] = 100 + i;
    }

    // Selector to reverse order within each 16-lane row:
    // Lane 0 selects 15, lane 1 selects 14 ... lane 7 selects 8 -> s0 = 0x89ABCDEF
    // Lane 8 selects 7 ... lane 15 selects 0 -> s1 = 0x01234567
    ws->sgpr[0] = 0x89ABCDEFu;
    ws->sgpr[1] = 0x01234567u;

    Instruction inst{};
    inst.opcode = Opcode::V_PERMLANE16_B32;
    inst.format = InstFormat::VOP3A;
    inst.size_bytes = 8;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::SGPR;
    inst.src[1].index = 0;
    inst.src[2].kind = Operand::Kind::SGPR;
    inst.src[2].index = 1;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 1;

    execute_instruction(*ws, inst);

    // Row 0 (0..15)
    EXPECT_EQ(ws->vgpr[1][0], 115u);
    EXPECT_EQ(ws->vgpr[1][1], 114u);
    EXPECT_EQ(ws->vgpr[1][15], 100u);

    // Row 1 (16..31)
    EXPECT_EQ(ws->vgpr[1][16], 131u);
    EXPECT_EQ(ws->vgpr[1][31], 116u);
}

TEST(ISAExecutor, V_PERMLANEX16_B32) {
    auto ws = make_wave();
    for (int i = 0; i < 64; i++) {
        ws->vgpr[0][i] = 100 + i;
    }

    // Invert within row:
    ws->sgpr[0] = 0x89ABCDEFu;
    ws->sgpr[1] = 0x01234567u;

    Instruction inst{};
    inst.opcode = Opcode::V_PERMLANEX16_B32;
    inst.format = InstFormat::VOP3A;
    inst.size_bytes = 8;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::SGPR;
    inst.src[1].index = 0;
    inst.src[2].kind = Operand::Kind::SGPR;
    inst.src[2].index = 1;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 1;

    execute_instruction(*ws, inst);

    // Row 0 reads from Row 1 (lanes 16..31)
    EXPECT_EQ(ws->vgpr[1][0], 131u); // 100 + 16 + 15
    EXPECT_EQ(ws->vgpr[1][15], 116u); // 100 + 16 + 0

    // Row 1 reads from Row 0 (lanes 0..15)
    EXPECT_EQ(ws->vgpr[1][16], 115u); // 100 + 0 + 15
    EXPECT_EQ(ws->vgpr[1][31], 100u); // 100 + 0 + 0
}

TEST(ISAExecutor, V_CVT_PKRTZ_F16_F32) {
    auto ws = make_wave();
    float f0 = 1.5f;
    float f1 = 2.5f;
    for (int i = 0; i < 64; i++) {
        std::memcpy(&ws->vgpr[0][i], &f0, 4);
        std::memcpy(&ws->vgpr[1][i], &f1, 4);
    }

    Instruction inst{};
    inst.opcode = Opcode::V_CVT_PKRTZ_F16_F32;
    inst.format = InstFormat::VOP3A;
    inst.size_bytes = 8;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 1;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 2;

    execute_instruction(*ws, inst);

    uint32_t packed = ws->vgpr[2][0];
    uint16_t h0 = packed & 0xFFFF;
    uint16_t h1 = packed >> 16;
    EXPECT_EQ(h0, 0x3E00u); // 1.5 in IEEE fp16
    EXPECT_EQ(h1, 0x4100u); // 2.5 in IEEE fp16
}

TEST(ISAExecutor, DS_SWIZZLE_B32) {
    auto ws = make_wave();
    for (int i = 0; i < 64; i++) {
        ws->vgpr[0][i] = 100 + i;
    }

    // Quad-Permute mode: swap adjacent pairs (1, 0, 3, 2)
    // pattern = 0x8000 | (1 << 0) | (0 << 2) | (3 << 4) | (2 << 6) = 0x8000 | 1 | 0 | 48 | 128 = 0x80B1
    Instruction inst{};
    inst.opcode = Opcode::DS_SWIZZLE_B32;
    inst.format = InstFormat::DS;
    inst.size_bytes = 8;
    inst.ds_offset0 = 0x80B1u;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 1;

    execute_instruction(*ws, inst);

    EXPECT_EQ(ws->vgpr[1][0], 101u);
    EXPECT_EQ(ws->vgpr[1][1], 100u);
    EXPECT_EQ(ws->vgpr[1][2], 103u);
    EXPECT_EQ(ws->vgpr[1][3], 102u);
}

// ===== CDNA3 (GFX942) MFMA & Unified Register Tests =====

TEST(ISAExecutor, CDNA3_UnifiedRegisters_AccvgprReadWrite) {
    auto ws = make_wave();
    ws->is_gfx942 = true;
    ws->accum_offset = 128; // Remap AGPRs to VGPR[128..]

    // Set VGPR0[lane] = 0x12340000 + lane
    for (int lane = 0; lane < 64; lane++) {
        ws->vgpr[0][lane] = 0x12340000u + lane;
    }

    // Write to AGPR 4 using v_accvgpr_write_b32
    Instruction inst_w{};
    inst_w.opcode = Opcode::V_ACCVGPR_WRITE_B32;
    inst_w.format = InstFormat::VOP1;
    inst_w.dst.kind = Operand::Kind::AGPR;
    inst_w.dst.index = 4;
    inst_w.src[0].kind = Operand::Kind::VGPR;
    inst_w.src[0].index = 0;
    execute_instruction(*ws, inst_w);

    // Verify VGPR[128 + 4] holds written value, while legacy AGPR[4] is 0
    for (int lane = 0; lane < 64; lane++) {
        EXPECT_EQ(ws->vgpr[128 + 4][lane], 0x12340000u + lane);
        EXPECT_EQ(ws->agpr[4][lane], 0u);
    }

    // Read back to VGPR 5 using v_accvgpr_read_b32
    Instruction inst_r{};
    inst_r.opcode = Opcode::V_ACCVGPR_READ_B32;
    inst_r.format = InstFormat::VOP1;
    inst_r.dst.kind = Operand::Kind::VGPR;
    inst_r.dst.index = 5;
    inst_r.src[0].kind = Operand::Kind::AGPR;
    inst_r.src[0].index = 4;
    execute_instruction(*ws, inst_r);

    for (int lane = 0; lane < 64; lane++) {
        EXPECT_EQ(ws->vgpr[5][lane], 0x12340000u + lane);
    }
}

TEST(ISAExecutor, CDNA3_MfmaF32_16x16x8_XF32) {
    auto ws = make_wave();
    ws->is_gfx942 = true;
    ws->accum_offset = 64;

    // Fill VGPR 0..1 (src0) with 1.0f, VGPR 2..3 (src1) with 2.0f
    for (int lane = 0; lane < 64; lane++) {
        ws->vgpr[0][lane] = f2u(1.0f);
        ws->vgpr[1][lane] = f2u(1.0f);
        ws->vgpr[2][lane] = f2u(2.0f);
        ws->vgpr[3][lane] = f2u(2.0f);
    }
    // Zero accumulator VGPR 64..67
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            ws->vgpr[ws->accum_offset + gpr][lane] = 0;
        }
    }

    Instruction inst{};
    inst.opcode = Opcode::V_MFMA_F32_16X16X8_XF32;
    inst.format = InstFormat::MFMA;
    inst.dst.kind = Operand::Kind::AGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    // K=8 with A=1.0, B=2.0 -> each output element = 8 * 1.0 * 2.0 = 16.0f
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            float val = u2f(ws->vgpr[ws->accum_offset + gpr][lane]);
            EXPECT_FLOAT_EQ(val, 16.0f);
        }
    }
}

TEST(ISAExecutor, CDNA3_MfmaI32_16x16x32I8) {
    auto ws = make_wave();
    ws->is_gfx942 = true;
    ws->accum_offset = 64;

    // Fill 64-bit pair with int8 1's (0x01010101)
    for (int lane = 0; lane < 64; lane++) {
        ws->vgpr[0][lane] = 0x01010101u;
        ws->vgpr[1][lane] = 0x01010101u;
        ws->vgpr[2][lane] = 0x02020202u; // int8 2's
        ws->vgpr[3][lane] = 0x02020202u;
    }
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            ws->vgpr[ws->accum_offset + gpr][lane] = 0;
        }
    }

    Instruction inst{};
    inst.opcode = Opcode::V_MFMA_I32_16X16X32I8;
    inst.format = InstFormat::MFMA;
    inst.dst.kind = Operand::Kind::AGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    // K=32 with A=1, B=2 -> dot product = 32 * 1 * 2 = 64
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            int32_t val = (int32_t)ws->vgpr[ws->accum_offset + gpr][lane];
            EXPECT_EQ(val, 64);
        }
    }
}

TEST(ISAExecutor, CDNA3_MfmaF32_16x16x32_FP8_FP8) {
    auto ws = make_wave();
    ws->is_gfx942 = true;
    ws->accum_offset = 64;

    // 0x38 in E4M3 is 1.0f
    // 0x40 in E4M3: sign=0, exp=8, m=0 -> 2^(8-7) = 2.0f
    uint32_t packed_1 = 0x38383838u;
    uint32_t packed_2 = 0x40404040u;
    for (int lane = 0; lane < 64; lane++) {
        ws->vgpr[0][lane] = packed_1;
        ws->vgpr[1][lane] = packed_1;
        ws->vgpr[2][lane] = packed_2;
        ws->vgpr[3][lane] = packed_2;
    }
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            ws->vgpr[ws->accum_offset + gpr][lane] = 0;
        }
    }

    Instruction inst{};
    inst.opcode = Opcode::V_MFMA_F32_16X16X32_FP8_FP8;
    inst.format = InstFormat::MFMA;
    inst.dst.kind = Operand::Kind::AGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 2;

    execute_instruction(*ws, inst);

    // K=32 with A=1.0, B=2.0 -> dot product = 32 * 1.0 * 2.0 = 64.0f
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            float val = u2f(ws->vgpr[ws->accum_offset + gpr][lane]);
            EXPECT_FLOAT_EQ(val, 64.0f);
        }
    }
}

TEST(ISAExecutor, V_PRNG_B32) {
    auto ws = make_wave();
    ws->exec = ~0ULL;
    uint32_t seed = 0x12345678;
    for (int lane = 0; lane < 64; lane++) {
        ws->vgpr[1][lane] = seed + lane;
    }

    Instruction inst{};
    inst.opcode = Opcode::V_PRNG_B32;
    inst.format = InstFormat::VOP1;
    inst.dst.kind = Operand::Kind::VGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 1;

    execute_instruction(*ws, inst);

    for (int lane = 0; lane < 64; lane++) {
        uint32_t x = seed + lane;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        EXPECT_EQ(ws->vgpr[0][lane], x);
    }
}

TEST(ISAExecutor, CDNA4_MFMA_F16_16x16x32) {
    auto ws = make_wave();
    ws->exec = ~0ULL;
    ws->is_gfx950 = true;
    ws->accum_offset = 64;

    // 1.0 in f16 is 0x3C00 -> packed pair is 0x3C003C00
    // 2.0 in f16 is 0x4000 -> packed pair is 0x40004000
    uint32_t packed_1 = 0x3C003C00;
    uint32_t packed_2 = 0x40004000;
    for (int lane = 0; lane < 64; lane++) {
        // A has 4 DWORDs (vgpr 0..3)
        ws->vgpr[0][lane] = packed_1;
        ws->vgpr[1][lane] = packed_1;
        ws->vgpr[2][lane] = packed_1;
        ws->vgpr[3][lane] = packed_1;
        // B has 4 DWORDs (vgpr 4..7)
        ws->vgpr[4][lane] = packed_2;
        ws->vgpr[5][lane] = packed_2;
        ws->vgpr[6][lane] = packed_2;
        ws->vgpr[7][lane] = packed_2;
    }
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            ws->vgpr[ws->accum_offset + gpr][lane] = 0;
        }
    }

    Instruction inst{};
    inst.opcode = Opcode::V_MFMA_F32_16X16X32_F16;
    inst.format = InstFormat::MFMA;
    inst.dst.kind = Operand::Kind::AGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 4;

    execute_instruction(*ws, inst);

    // K=32 with A=1.0, B=2.0 -> dot product = 32 * 1.0 * 2.0 = 64.0f
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            float val = u2f(ws->vgpr[ws->accum_offset + gpr][lane]);
            EXPECT_FLOAT_EQ(val, 64.0f);
        }
    }
}

TEST(ISAExecutor, CDNA4_MFMA_SCALE_FP4_16x16x128) {
    auto ws = make_wave();
    ws->exec = ~0ULL;
    ws->is_gfx950 = true;
    ws->accum_offset = 64;

    // FP4 E2M1: 1.0 is 0x2 (e=1, m=0), 2.0 is 0x4 (e=2, m=0)
    uint32_t packed_1 = 0x22222222;
    uint32_t packed_2 = 0x44444444;
    for (int lane = 0; lane < 64; lane++) {
        for (int d = 0; d < 4; d++) {
            ws->vgpr[0 + d][lane] = packed_1;
            ws->vgpr[4 + d][lane] = packed_2;
        }
    }
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            ws->vgpr[ws->accum_offset + gpr][lane] = 0;
        }
    }

    Instruction inst{};
    inst.opcode = Opcode::V_MFMA_SCALE_F32_16X16X128_F8F6F4;
    inst.format = InstFormat::MFMA;
    inst.dst.kind = Operand::Kind::AGPR;
    inst.dst.index = 0;
    inst.src[0].kind = Operand::Kind::VGPR;
    inst.src[0].index = 0;
    inst.src[1].kind = Operand::Kind::VGPR;
    inst.src[1].index = 4;

    execute_instruction(*ws, inst);

    // K=128 with A=1.0, B=2.0 -> dot product = 128 * 1.0 * 2.0 = 256.0f
    for (int gpr = 0; gpr < 4; gpr++) {
        for (int lane = 0; lane < 64; lane++) {
            float val = u2f(ws->vgpr[ws->accum_offset + gpr][lane]);
            EXPECT_FLOAT_EQ(val, 256.0f);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
