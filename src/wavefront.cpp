#include "gcnsim/wavefront.h"
#include "gcnsim/kernel_descriptor.h"
#include <cstring>
#include <cstddef>
#include <algorithm>
#include <cstdint>

namespace gcnsim {

// Forward declaration from isa_executor.cpp
WaveStatus run_wavefront(WavefrontState& wave);

void init_wavefront(WavefrontState& wave,
                    const KernelDescriptor& kd,
                    const uint8_t* kernarg_ptr,
                    const uint8_t* code_ptr,
                    size_t code_size,
                    uint32_t wg_id_x, uint32_t wg_id_y, uint32_t wg_id_z,
                    uint32_t wave_in_wg, uint32_t /*waves_per_wg*/,
                    uint32_t block_x, uint32_t block_y, uint32_t block_z)
{
    // Zero POD fields manually to avoid memset on a type with std::vector members
    memset((void*)&wave, 0, offsetof(WavefrontState, lgkm_pending));
    wave.lgkm_pending.clear();
    wave.vm_pending.clear();
    wave.kernarg_ptr     = kernarg_ptr;
    wave.kernel_code_ptr = code_ptr;
    wave.kernel_code_size = code_size;
    wave.workgroup_id[0] = wg_id_x;
    wave.workgroup_id[1] = wg_id_y;
    wave.workgroup_id[2] = wg_id_z;
    wave.pc = 0;
    // lds_ptr and lds_size are set by the execution engine per workgroup
    wave.lds_ptr = nullptr;
    wave.lds_size = 0;

    // -----------------------------------------------------------------------
    // EXEC mask — active lanes
    // -----------------------------------------------------------------------
    uint32_t first_lane    = wave_in_wg * WAVE_SIZE;
    uint32_t total_threads = block_x * block_y * block_z;
    uint32_t active_lanes;
    if (first_lane >= total_threads) {
        active_lanes = 0;
    } else {
        active_lanes = std::min((uint32_t)WAVE_SIZE, total_threads - first_lane);
    }
    wave.exec = (active_lanes == 64) ? ~0ULL : ((1ULL << active_lanes) - 1);

    // -----------------------------------------------------------------------
    // Per-lane workitem IDs
    // -----------------------------------------------------------------------
    for (uint32_t lane = 0; lane < (uint32_t)WAVE_SIZE; lane++) {
        uint32_t flat_id = first_lane + lane;
        wave.workitem_id[lane][0] = flat_id % block_x;
        wave.workitem_id[lane][1] = (flat_id / block_x) % block_y;
        wave.workitem_id[lane][2] = flat_id / (block_x * block_y);
    }

    // VGPR0 = workitem_id.x (always provided by hardware)
    for (int lane = 0; lane < WAVE_SIZE; lane++)
        wave.vgpr[0][lane] = wave.workitem_id[lane][0];

    // VGPR1 = workitem_id.y, VGPR2 = workitem_id.z (if enabled by kernel descriptor)
    // enable_vgpr_workitem_id is bits [1:0] of compute_pgm_rsrc2[11:12]
    // Actually: RSRC2 bits [12:11] = enable_vgpr_workitem_id
    // 0 = only X, 1 = X and Y, 2 = X, Y, and Z
    uint32_t enable_vgpr_workitem_id = (kd.compute_pgm_rsrc2 >> 11) & 0x3;
    if (enable_vgpr_workitem_id >= 1) {
        for (int lane = 0; lane < WAVE_SIZE; lane++)
            wave.vgpr[1][lane] = wave.workitem_id[lane][1];
    }
    if (enable_vgpr_workitem_id >= 2) {
        for (int lane = 0; lane < WAVE_SIZE; lane++)
            wave.vgpr[2][lane] = wave.workitem_id[lane][2];
    }

    // -----------------------------------------------------------------------
    // SGPR initialization based on kernel_code_properties and compute_pgm_rsrc2
    //
    // Order of SGPRs (from AMD ISA ABI):
    //   [Private Segment Buffer] 4 SGPRs if KCP bit 0
    //   [Dispatch Ptr]           2 SGPRs if KCP bit 1
    //   [Queue Ptr]              2 SGPRs if KCP bit 2
    //   [Kernarg Segment Ptr]    2 SGPRs if KCP bit 3
    //   [Dispatch ID]            2 SGPRs if KCP bit 4
    //   [Flat Scratch Init]      2 SGPRs if KCP bit 5
    //   [Private Segment Size]   1 SGPR  if KCP bit 6
    //   [Work-Group ID X]        1 SGPR  if RSRC2 bit 7
    //   [Work-Group ID Y]        1 SGPR  if RSRC2 bit 8
    //   [Work-Group ID Z]        1 SGPR  if RSRC2 bit 9
    // -----------------------------------------------------------------------
    int sgpr_idx = 0;
    uint16_t kcp = kd.kernel_code_properties;
    uint32_t rsrc2 = kd.compute_pgm_rsrc2;

    if (kcp & KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) {
        // 4 SGPRs for private segment buffer descriptor (scratch ring)
        // Fill with zeros in simulation (no scratch)
        sgpr_idx += 4;
    }
    if (kcp & KCP_ENABLE_SGPR_DISPATCH_PTR) {
        // 2 SGPRs for AQL dispatch packet pointer — set to 0
        sgpr_idx += 2;
    }
    if (kcp & KCP_ENABLE_SGPR_QUEUE_PTR) {
        // 2 SGPRs for queue pointer
        sgpr_idx += 2;
    }
    if (kcp & KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR) {
        // 2 SGPRs for kernarg segment pointer
        uint64_t ka = reinterpret_cast<uint64_t>(kernarg_ptr);
        wave.sgpr[sgpr_idx]     = (uint32_t)(ka & 0xFFFFFFFF);
        wave.sgpr[sgpr_idx + 1] = (uint32_t)(ka >> 32);
        sgpr_idx += 2;
    }
    if (kcp & KCP_ENABLE_SGPR_DISPATCH_ID) {
        sgpr_idx += 2;
    }
    if (kcp & KCP_ENABLE_SGPR_FLAT_SCRATCH_INIT) {
        sgpr_idx += 2;
    }
    if (kcp & KCP_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) {
        sgpr_idx += 1;
    }

    // Work-group IDs
    if (rsrc2 & RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) {
        wave.sgpr[sgpr_idx++] = wg_id_x;
    }
    if (rsrc2 & RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) {
        wave.sgpr[sgpr_idx++] = wg_id_y;
    }
    if (rsrc2 & RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) {
        wave.sgpr[sgpr_idx++] = wg_id_z;
    }

    // Also mirror VCC to sgpr[106:107] for instructions that read it there
    wave.sgpr[VCC_LO_IDX] = 0;
    wave.sgpr[VCC_HI_IDX] = 0;
}

} // namespace gcnsim
