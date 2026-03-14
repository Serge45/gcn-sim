#pragma once
#include <cstdint>
#include <cassert>

namespace gcnsim {

// Code Object V3/V4 kernel descriptor — 64 bytes at <kernel_name>.kd symbol
struct KernelDescriptor {
    uint32_t group_segment_fixed_size;       // bytes of LDS
    uint32_t private_segment_fixed_size;     // bytes of scratch per workitem
    uint32_t kernarg_size;                   // bytes of kernel arguments
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset;  // offset from this struct to first instruction
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;              // gfx90a: ACCVGPR count
    uint32_t compute_pgm_rsrc1;              // granulated VGPR/SGPR counts, float mode
    uint32_t compute_pgm_rsrc2;              // user SGPR enables, workgroup ID enables
    uint16_t kernel_code_properties;         // ENABLE_SGPR_* bits
    uint8_t  reserved2[6];
};
static_assert(sizeof(KernelDescriptor) == 64, "KernelDescriptor must be 64 bytes");

// kernel_code_properties bits (from AMD_KERNEL_CODE_PROPERTIES_* in amd_hsa_kernel_code.h)
static constexpr uint16_t KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER    = (1 << 0);  // 4 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_DISPATCH_PTR              = (1 << 1);  // 2 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_QUEUE_PTR                 = (1 << 2);  // 2 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR       = (1 << 3);  // 2 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_DISPATCH_ID               = (1 << 4);  // 2 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_FLAT_SCRATCH_INIT         = (1 << 5);  // 2 SGPRs
static constexpr uint16_t KCP_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE      = (1 << 6);  // 1 SGPR

// compute_pgm_rsrc2 bits
static constexpr uint32_t RSRC2_ENABLE_SGPR_WORKGROUP_ID_X          = (1 << 7);
static constexpr uint32_t RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y          = (1 << 8);
static constexpr uint32_t RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z          = (1 << 9);
// bits [5:1] = user_sgpr_count
static constexpr uint32_t RSRC2_USER_SGPR_COUNT_SHIFT               = 1;
static constexpr uint32_t RSRC2_USER_SGPR_COUNT_MASK                = 0x1f;

} // namespace gcnsim
