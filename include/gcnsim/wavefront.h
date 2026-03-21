#pragma once
#include <cstdint>
#include <cstddef>

namespace gcnsim {

static constexpr int WAVE_SIZE     = 64;
static constexpr int NUM_SGPR      = 108;
static constexpr int NUM_VGPR      = 256;
static constexpr int NUM_AGPR      = 256;
static constexpr int VCC_LO_IDX    = 106;   // sgpr[106] = VCC_LO
static constexpr int VCC_HI_IDX    = 107;   // sgpr[107] = VCC_HI

enum class WaveStatus {
    ENDED,       // s_endpgm or out-of-bounds
    BARRIER_HIT  // s_barrier — wavefront paused until all waves sync
};

struct WavefrontState {
    uint32_t sgpr[NUM_SGPR];                // SGPRs; VCC at [106:107]
    uint32_t vgpr[NUM_VGPR][WAVE_SIZE];     // [reg_index][lane]
    uint32_t agpr[NUM_AGPR][WAVE_SIZE];     // Accumulator VGPRs (gfx90a MFMA)

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

    // LDS pointer — shared across waves in a workgroup
    // Set by execution engine; nullptr if no LDS allocated
    uint8_t* lds_ptr;
    uint32_t lds_size;                      // bytes, from group_segment_fixed_size

    // Scratch (private) memory — per-wavefront, one segment per lane
    // Set by execution engine; nullptr if no scratch allocated
    uint8_t* scratch_ptr;
    uint32_t scratch_size;                  // total bytes (private_segment_fixed_size * 64)
    uint32_t scratch_per_lane;              // bytes per lane (private_segment_fixed_size)
};

} // namespace gcnsim
