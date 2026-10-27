// cdna4_gemm.s — Hand-written CDNA4 (GFX950 / MI-350) kernel
// Demonstrates native CDNA4 instructions:
//   - v_prng_b32
//   - v_lshl_add_u64
//   - v_mfma_f32_16x16x32_f16 (double-rate FP16 Matrix Core)

    .text
    .amdgcn_target "amdgcn-amd-amdhsa--gfx950"

    .amdhsa_kernel cdna4_gemm
        .amdhsa_user_sgpr_kernarg_segment_ptr   1
        .amdhsa_system_sgpr_workgroup_id_x      1
        .amdhsa_system_sgpr_workgroup_id_y      0
        .amdhsa_system_sgpr_workgroup_id_z      0
        .amdhsa_next_free_vgpr                  32
        .amdhsa_next_free_sgpr                  16
        .amdhsa_accum_offset                    16
        .amdhsa_group_segment_fixed_size        0
        .amdhsa_private_segment_fixed_size      0
        .amdhsa_float_round_mode_32             0
        .amdhsa_float_round_mode_16_64          0
        .amdhsa_float_denorm_mode_32            3
        .amdhsa_float_denorm_mode_16_64         3
        .amdhsa_dx10_clamp                      1
        .amdhsa_ieee_mode                       1
    .end_amdhsa_kernel

    .section .text.cdna4_gemm,"ax"
    .globl cdna4_gemm
    .p2align 8
cdna4_gemm:
    // s[0:1] = kernarg pointer
    // s[2]   = workgroup_id.x
    // v0     = workitem_id.x

    // Load args:
    // offset 0:  A ptr (64-bit)
    // offset 8:  B ptr (64-bit)
    // offset 16: C ptr (64-bit)
    s_load_dwordx4 s[4:7], s[0:1], 0x00
    s_load_dwordx2 s[8:9], s[0:1], 0x10
    s_waitcnt lgkmcnt(0)

    // Save workitem_id.x
    v_mov_b32_e32 v10, v0

    // Clear accumulator a[0:3]
    v_accvgpr_write_b32 a0, 0
    v_accvgpr_write_b32 a1, 0
    v_accvgpr_write_b32 a2, 0
    v_accvgpr_write_b32 a3, 0

    // Set input registers v[0:3] = 1.0f (fp16: 0x3c003c00), v[4:7] = 2.0f (fp16: 0x40004000)
    v_mov_b32_e32 v0, 0x3c003c00
    v_mov_b32_e32 v1, 0x3c003c00
    v_mov_b32_e32 v2, 0x3c003c00
    v_mov_b32_e32 v3, 0x3c003c00

    v_mov_b32_e32 v4, 0x40004000
    v_mov_b32_e32 v5, 0x40004000
    v_mov_b32_e32 v6, 0x40004000
    v_mov_b32_e32 v7, 0x40004000

    // Test CDNA4 PRNG instruction
    v_prng_b32_e32 v8, v0

    // Execute CDNA4 double-rate FP16 MFMA: 16x16x32
    v_mfma_f32_16x16x32_f16 a[0:3], v[0:3], v[4:7], a[0:3]

    // Read back a0 to v9
    v_accvgpr_read_b32 v9, a0

    // Compute byte offset: lane_id * 4
    v_lshlrev_b32_e32 v10, 2, v10

    // Calculate 64-bit destination address using CDNA4 v_lshl_add_u64:
    // v[12:13] = (s[8:9] << 0) + v[10:11]
    v_mov_b32_e32 v11, 0
    v_lshl_add_u64 v[12:13], s[8:9], 0, v[10:11]

    // Store computed MFMA result to C[lane]
    global_store_dword v[12:13], v9, off
    s_waitcnt vmcnt(0)

    s_endpgm
