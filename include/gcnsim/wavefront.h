#pragma once
#include <cstdint>
#include <cstddef>

namespace gcnsim {

static constexpr int WAVE_SIZE     = 64;
static constexpr int NUM_SGPR      = 108;
static constexpr int NUM_VGPR      = 256;
static constexpr int VCC_LO_IDX    = 106;   // sgpr[106] = VCC_LO
static constexpr int VCC_HI_IDX    = 107;   // sgpr[107] = VCC_HI

struct WavefrontState {
    uint32_t sgpr[NUM_SGPR];                // SGPRs; VCC at [106:107]
    uint32_t vgpr[NUM_VGPR][WAVE_SIZE];     // [reg_index][lane]

    uint64_t exec;                          // active lane mask
    uint64_t vcc;                           // vector condition code
    uint32_t scc;                           // scalar condition code
    uint32_t m0;

    uint64_t pc;                            // byte offset into kernel_code_ptr
    const uint8_t* kernarg_ptr;
    const uint8_t* kernel_code_ptr;
    size_t   kernel_code_size;

    uint32_t workgroup_id[3];
    uint32_t workitem_id[WAVE_SIZE][3];     // [lane][dim]
};

} // namespace gcnsim
