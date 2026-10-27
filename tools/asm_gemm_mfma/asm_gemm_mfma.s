// asm_gemm_mfma.s — hand-written GFX90A MFMA GEMM kernel
//
// C = A * B  (row-major, float32, C = A*B, C initialized to 0)
// Uses v_mfma_f32_4x4x1f32 with 4-thread workgroup per 4x4 tile.
//
// SGPR layout (from kernel descriptor flags):
//   s[0:1]  = kernarg_segment_ptr
//   s[2]    = workgroup_id_x  (wg_col: which 4-col tile)
//   s[3]    = workgroup_id_y  (wg_row: which 4-row tile)
//
// VGPR layout:
//   v0      = workitem_id.x (lane: 0..3)
//
// Kernel argument layout:
//   offset  0, size 8: A ptr (global_buffer)
//   offset  8, size 8: B ptr (global_buffer)
//   offset 16, size 8: C ptr (global_buffer)
//   offset 24, size 4: M    (by_value)
//   offset 28, size 4: N    (by_value)
//   offset 32, size 4: K    (by_value)
//   total kernarg_segment_size = 40

    .text
    .amdgcn_target "amdgcn-amd-amdhsa--gfx90a:xnack-"

    .amdhsa_kernel asm_gemm_mfma
        .amdhsa_user_sgpr_kernarg_segment_ptr   1
        .amdhsa_system_sgpr_workgroup_id_x      1
        .amdhsa_system_sgpr_workgroup_id_y      1
        .amdhsa_system_sgpr_workgroup_id_z      0
        .amdhsa_next_free_vgpr                  16
        .amdhsa_next_free_sgpr                  24
        .amdhsa_accum_offset                    8
        .amdhsa_group_segment_fixed_size        0
        .amdhsa_private_segment_fixed_size      0
        .amdhsa_float_round_mode_32             0
        .amdhsa_float_round_mode_16_64          0
        .amdhsa_float_denorm_mode_32            3
        .amdhsa_float_denorm_mode_16_64         3
        .amdhsa_dx10_clamp                      1
        .amdhsa_ieee_mode                       1
    .end_amdhsa_kernel

    .section .text.asm_gemm_mfma,"ax"
    .globl asm_gemm_mfma
    .p2align 8
asm_gemm_mfma:

// -------------------------------------------------------------------
// Step 1: Load kernel arguments
//   s[4:7]  = { A_ptr(64), B_ptr(64) }
//   s[8:11] = { C_ptr(64), M(32), N(32) }
//   s[12]   = K
// -------------------------------------------------------------------
    s_load_dwordx4 s[4:7],  s[0:1], 0x00
    s_load_dwordx4 s[8:11], s[0:1], 0x10
    s_load_dword   s[12],   s[0:1], 0x20
    s_waitcnt lgkmcnt(0)

// -------------------------------------------------------------------
// Step 2: Compute indices
//   lane = v0 (0..3)
//   c_row = wg_row * 4 + lane   → v1
//   c_col_base in s[13]
//   Copy SGPRs used in VALU ops into VGPRs to avoid constant bus issues:
//   v12 = s[5]  (A_ptr high)
//   v13 = s[7]  (B_ptr high)
//   v14 = s[9]  (C_ptr high)
//   v15 = s[11] (N)
// -------------------------------------------------------------------
    // c_row = s[3] * 4 + v0
    v_mov_b32      v1, s[3]
    v_lshlrev_b32  v1, 2, v1       // v1 = wg_row * 4
    v_add_u32      v1, v0, v1      // v1 = wg_row * 4 + lane

    // c_col_base = wg_col * 4
    s_lshl_b32     s[13], s[2], 2  // s[13] = wg_col * 4

    // Copy pointer high halves and N into VGPRs (constant bus restriction workaround)
    v_mov_b32      v12, s[5]       // A_ptr[63:32]
    v_mov_b32      v13, s[7]       // B_ptr[63:32]
    v_mov_b32      v14, s[9]       // C_ptr[63:32]
    v_mov_b32      v15, s[11]      // N

// -------------------------------------------------------------------
// Step 3: Zero the accumulators a[0:3]
// -------------------------------------------------------------------
    v_accvgpr_write_b32  a0, 0
    v_accvgpr_write_b32  a1, 0
    v_accvgpr_write_b32  a2, 0
    v_accvgpr_write_b32  a3, 0

// -------------------------------------------------------------------
// Step 4: K loop
//   s[14] = k (loop counter, 0..K-1)
//
// Per iteration:
//   Load A[c_row][k]:           addr = A_ptr + (c_row * K + k) * 4
//   Load B[k][c_col_base+lane]: addr = B_ptr + (k * N + c_col_base + lane) * 4
//   v_mfma_f32_4x4x1f32 a[0:3], v_a, v_b, a[0:3]
//
// To avoid constant bus violations: use VGPRs for both operands when
// multiplying two scalars via VALU (move one SGPR to VGPR first).
// -------------------------------------------------------------------
    s_mov_b32 s[14], 0              // k = 0

.Lk_loop:
    // ---- Compute A address: A_ptr + (c_row * K + k) * 4 ----
    // v2 = c_row * K + k
    v_mul_lo_u32   v2, v1, s[12]   // v2 = c_row * K  (v1=VGPR, s[12]=SGPR: OK)
    v_add_u32      v2, v2, s[14]   // v2 = c_row * K + k
    v_lshlrev_b32  v2, 2, v2       // byte offset (low 32)
    v_mov_b32      v3, 0
    // addr = A_ptr + v2  →  v[2:3]
    v_add_co_u32   v2, vcc, s[4], v2    // low: s[4]=SGPR, v2=VGPR: OK
    v_addc_co_u32  v3, vcc, v12, v3, vcc // high: use v12 (copy of s[5])
    // Load A element
    flat_load_dword v4, v[2:3]

    // ---- Compute B address: B_ptr + (k * N + c_col_base + lane) * 4 ----
    // v6 = k * N + c_col_base + lane
    // k is in s[14]; N is in v15 (copy of s[11])
    v_mov_b32      v6, s[14]            // v6 = k (VGPR copy)
    v_mul_lo_u32   v6, v6, v15          // v6 = k * N  (two VGPRs: OK)
    v_add_u32      v6, v6, s[13]        // v6 = k*N + c_col_base
    v_add_u32      v6, v6, v0           // v6 = k*N + c_col_base + lane
    v_lshlrev_b32  v6, 2, v6            // byte offset (low 32)
    v_mov_b32      v7, 0
    // addr = B_ptr + v6  →  v[6:7]
    v_add_co_u32   v6, vcc, s[6], v6    // low: s[6]=SGPR, v6=VGPR: OK
    v_addc_co_u32  v7, vcc, v13, v7, vcc // high: use v13 (copy of s[7])
    // Load B element
    flat_load_dword v5, v[6:7]

    s_waitcnt vmcnt(0)

    // ---- MFMA: a[0:3] += a_val * b_val (outer product across 4 lanes) ----
    v_mfma_f32_4x4x1f32  a[0:3], v4, v5, a[0:3]

    // ---- Loop increment and branch ----
    s_add_u32      s[14], s[14], 1
    s_cmp_lt_u32   s[14], s[12]
    s_cbranch_scc1 .Lk_loop

// -------------------------------------------------------------------
// Step 5: Write results
//   Each lane writes 4 elements:
//     C[c_row][c_col_base + g] = a[g]   for g = 0..3
//   addr = C_ptr + (c_row * N + c_col_base + g) * 4
//   Use v8 = c_row * N as base row byte offset.
// -------------------------------------------------------------------
    // c_row * N
    v_mul_lo_u32   v8, v1, v15     // v8 = c_row * N  (both VGPRs: OK)

    // g = 0
    v_accvgpr_read_b32  v9, a0
    v_add_u32      v10, v8, s[13]      // c_row*N + c_col_base
    // +0 already
    v_lshlrev_b32  v10, 2, v10
    v_mov_b32      v11, 0
    v_add_co_u32   v10, vcc, s[8], v10
    v_addc_co_u32  v11, vcc, v14, v11, vcc
    flat_store_dword v[10:11], v9

    // g = 1
    v_accvgpr_read_b32  v9, a1
    v_add_u32      v10, v8, s[13]
    v_add_u32      v10, v10, 1
    v_lshlrev_b32  v10, 2, v10
    v_mov_b32      v11, 0
    v_add_co_u32   v10, vcc, s[8], v10
    v_addc_co_u32  v11, vcc, v14, v11, vcc
    flat_store_dword v[10:11], v9

    // g = 2
    v_accvgpr_read_b32  v9, a2
    v_add_u32      v10, v8, s[13]
    v_add_u32      v10, v10, 2
    v_lshlrev_b32  v10, 2, v10
    v_mov_b32      v11, 0
    v_add_co_u32   v10, vcc, s[8], v10
    v_addc_co_u32  v11, vcc, v14, v11, vcc
    flat_store_dword v[10:11], v9

    // g = 3
    v_accvgpr_read_b32  v9, a3
    v_add_u32      v10, v8, s[13]
    v_add_u32      v10, v10, 3
    v_lshlrev_b32  v10, 2, v10
    v_mov_b32      v11, 0
    v_add_co_u32   v10, vcc, s[8], v10
    v_addc_co_u32  v11, vcc, v14, v11, vcc
    flat_store_dword v[10:11], v9

    s_waitcnt vmcnt(0)
    s_endpgm

// -------------------------------------------------------------------
// Kernel metadata (COV4 YAML → MsgPack by assembler)
// -------------------------------------------------------------------
    .amdgpu_metadata
---
amdhsa.kernels:
  - .name:            asm_gemm_mfma
    .symbol:          asm_gemm_mfma.kd
    .kernarg_segment_size: 40
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 16
    .agpr_count: 4
    .max_flat_workgroup_size: 4
    .args:
      - .name:       A
        .offset:     0
        .size:       8
        .value_kind: global_buffer
        .address_space: global
      - .name:       B
        .offset:     8
        .size:       8
        .value_kind: global_buffer
        .address_space: global
      - .name:       C
        .offset:     16
        .size:       8
        .value_kind: global_buffer
        .address_space: global
      - .name:       M
        .offset:     24
        .size:       4
        .value_kind: by_value
      - .name:       size_n
        .offset:     28
        .size:       4
        .value_kind: by_value
      - .name:       K
        .offset:     32
        .size:       4
        .value_kind: by_value
amdhsa.target: amdgcn-amd-amdhsa--gfx90a:xnack-
amdhsa.version:
  - 1
  - 1
...
    .end_amdgpu_metadata
