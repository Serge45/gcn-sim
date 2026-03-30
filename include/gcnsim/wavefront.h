#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace gcnsim { class CacheModel; }

namespace gcnsim {

static constexpr int WAVE_SIZE     = 64;
static constexpr int NUM_SGPR      = 108;
static constexpr int NUM_VGPR      = 256;
static constexpr int NUM_AGPR      = 256;
static constexpr int VCC_LO_IDX    = 106;   // sgpr[106] = VCC_LO
static constexpr int VCC_HI_IDX    = 107;   // sgpr[107] = VCC_HI

enum class WaveStatus {
    ENDED,       // s_endpgm or out-of-bounds
    BARRIER_HIT, // s_barrier — wavefront paused until all waves sync
    RUNNING      // one instruction executed (used by step_wavefront for CU interleaving)
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

    // Cycle counting — optional, enabled per LaunchConfig::enable_cycle_counting
    bool     count_cycles = false;
    uint64_t cycle_count  = 0;
    uint32_t global_mem_latency = 300;   // configurable VMEM/FLAT/GLOBAL latency
    uint32_t smem_latency = 20;          // configurable SMEM latency
    // MFMA co-execution: miSIMD pipeline state
    uint64_t mfma_shadow_remaining = 0;  // shadow cycles left from in-flight MFMA
    uint64_t mfma_busy_until = 0;        // cycle when miSIMD becomes free
    // Multi-pipe issue: per-pipe ready timestamps (instruction starts at max(cycle, pipe_ready))
    uint64_t pipe_salu_ready = 0;
    uint64_t pipe_valu_ready = 0;
    uint64_t pipe_vmem_ready = 0;
    uint64_t pipe_lds_ready  = 0;
    uint64_t pipe_smem_ready = 0;
    uint64_t mfma_total_exec_cycles = 0; // accumulated MFMA execution cycles (for utilization)
    uint64_t valu_total_issue_cycles = 0; // accumulated VALU+MFMA issue cycles (for utilization)
    // Cache model (shared per-workgroup for L1, global for L2); nullptr = no cache sim
    CacheModel* cache = nullptr;
    // Trace output (Chrome Trace Event Format); nullptr = no tracing
    FILE*    trace_file = nullptr;
    uint32_t trace_wave_id = 0;          // wave index within workgroup
    uint32_t trace_wg_flat_id = 0;       // flat workgroup index
    // Step-mode instruction counter (safety limit for CU interleaving)
    uint64_t insn_count = 0;
    // In-flight memory op queues (completion timestamps, drained by s_waitcnt)
    std::vector<uint64_t> lgkm_pending;  // DS reads/writes and SMEM loads
    std::vector<uint64_t> vm_pending;    // MUBUF/MTBUF/FLAT/GLOBAL loads/stores
};

} // namespace gcnsim
