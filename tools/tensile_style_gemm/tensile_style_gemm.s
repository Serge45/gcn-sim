
.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
.text
.globl generated_gemm
.p2align 8
.type generated_gemm,@function

generated_gemm:
label_load_args:
//Load all arguments
s_load_dwordx4 s[36:39], s[0:1] 0
s_load_dwordx4 s[40:43], s[0:1] 16
s_load_dwordx4 s[44:47], s[0:1] 32
s_load_dwordx4 s[48:51], s[0:1] 48
s_load_dwordx2 s[52:53], s[0:1] 64
s_load_dword s[54], s[0:1] 72
s_waitcnt lgkmcnt(0)
//Setup Srd{A, B}
s_mov_b32 s[7], 131072
s_mov_b32 s[11], 131072
s_mov_b64 s[4:5], s[36:37]
s_mov_b64 s[8:9], s[38:39]
//Setup sizes, m, n and k
s_mov_b32 s[12], s[44]
s_mov_b32 s[13], s[45]
s_mov_b32 s[14], s[46]
s_mov_b32 s[20], 1
s_mov_b32 s[21], s[47]
s_mov_b32 s[22], 1
s_mov_b32 s[23], s[48]
s_mov_b32 s[24], 1
s_mov_b32 s[25], s[49]
s_mov_b32 s[26], 1
s_mov_b32 s[27], s[50]
label_setup_gl_offsets:
//Setup global read offsets
s_lshl_b32 s[16], s[2], 8
s_lshl_b32 s[17], s[3], 7
s_mul_i32 s[55], s[21], s[14]
s_lshl_b32 s[6], s[55], 2
s_mul_i32 s[55], s[13], s[23]
s_lshl_b32 s[10], s[55], 2
s_mul_i32 s[18], s[16], s[20]
s_lshl_b32 s[18], s[18], 2
s_mul_i32 s[19], s[17], s[23]
s_lshl_b32 s[19], s[19], 2
s_mov_b32 s[15], 0
v_accvgpr_write_b32 acc[0], 0
v_accvgpr_write_b32 acc[1], 0
v_accvgpr_write_b32 acc[2], 0
v_accvgpr_write_b32 acc[3], 0
v_accvgpr_write_b32 acc[4], 0
v_accvgpr_write_b32 acc[5], 0
v_accvgpr_write_b32 acc[6], 0
v_accvgpr_write_b32 acc[7], 0
v_accvgpr_write_b32 acc[8], 0
v_accvgpr_write_b32 acc[9], 0
v_accvgpr_write_b32 acc[10], 0
v_accvgpr_write_b32 acc[11], 0
v_accvgpr_write_b32 acc[12], 0
v_accvgpr_write_b32 acc[13], 0
v_accvgpr_write_b32 acc[14], 0
v_accvgpr_write_b32 acc[15], 0
v_accvgpr_write_b32 acc[16], 0
v_accvgpr_write_b32 acc[17], 0
v_accvgpr_write_b32 acc[18], 0
v_accvgpr_write_b32 acc[19], 0
v_accvgpr_write_b32 acc[20], 0
v_accvgpr_write_b32 acc[21], 0
v_accvgpr_write_b32 acc[22], 0
v_accvgpr_write_b32 acc[23], 0
v_accvgpr_write_b32 acc[24], 0
v_accvgpr_write_b32 acc[25], 0
v_accvgpr_write_b32 acc[26], 0
v_accvgpr_write_b32 acc[27], 0
v_accvgpr_write_b32 acc[28], 0
v_accvgpr_write_b32 acc[29], 0
v_accvgpr_write_b32 acc[30], 0
v_accvgpr_write_b32 acc[31], 0
v_accvgpr_write_b32 acc[32], 0
v_accvgpr_write_b32 acc[33], 0
v_accvgpr_write_b32 acc[34], 0
v_accvgpr_write_b32 acc[35], 0
v_accvgpr_write_b32 acc[36], 0
v_accvgpr_write_b32 acc[37], 0
v_accvgpr_write_b32 acc[38], 0
v_accvgpr_write_b32 acc[39], 0
v_accvgpr_write_b32 acc[40], 0
v_accvgpr_write_b32 acc[41], 0
v_accvgpr_write_b32 acc[42], 0
v_accvgpr_write_b32 acc[43], 0
v_accvgpr_write_b32 acc[44], 0
v_accvgpr_write_b32 acc[45], 0
v_accvgpr_write_b32 acc[46], 0
v_accvgpr_write_b32 acc[47], 0
v_accvgpr_write_b32 acc[48], 0
v_accvgpr_write_b32 acc[49], 0
v_accvgpr_write_b32 acc[50], 0
v_accvgpr_write_b32 acc[51], 0
v_accvgpr_write_b32 acc[52], 0
v_accvgpr_write_b32 acc[53], 0
v_accvgpr_write_b32 acc[54], 0
v_accvgpr_write_b32 acc[55], 0
v_accvgpr_write_b32 acc[56], 0
v_accvgpr_write_b32 acc[57], 0
v_accvgpr_write_b32 acc[58], 0
v_accvgpr_write_b32 acc[59], 0
v_accvgpr_write_b32 acc[60], 0
v_accvgpr_write_b32 acc[61], 0
v_accvgpr_write_b32 acc[62], 0
v_accvgpr_write_b32 acc[63], 0
v_accvgpr_write_b32 acc[64], 0
v_accvgpr_write_b32 acc[65], 0
v_accvgpr_write_b32 acc[66], 0
v_accvgpr_write_b32 acc[67], 0
v_accvgpr_write_b32 acc[68], 0
v_accvgpr_write_b32 acc[69], 0
v_accvgpr_write_b32 acc[70], 0
v_accvgpr_write_b32 acc[71], 0
v_accvgpr_write_b32 acc[72], 0
v_accvgpr_write_b32 acc[73], 0
v_accvgpr_write_b32 acc[74], 0
v_accvgpr_write_b32 acc[75], 0
v_accvgpr_write_b32 acc[76], 0
v_accvgpr_write_b32 acc[77], 0
v_accvgpr_write_b32 acc[78], 0
v_accvgpr_write_b32 acc[79], 0
v_accvgpr_write_b32 acc[80], 0
v_accvgpr_write_b32 acc[81], 0
v_accvgpr_write_b32 acc[82], 0
v_accvgpr_write_b32 acc[83], 0
v_accvgpr_write_b32 acc[84], 0
v_accvgpr_write_b32 acc[85], 0
v_accvgpr_write_b32 acc[86], 0
v_accvgpr_write_b32 acc[87], 0
v_accvgpr_write_b32 acc[88], 0
v_accvgpr_write_b32 acc[89], 0
v_accvgpr_write_b32 acc[90], 0
v_accvgpr_write_b32 acc[91], 0
v_accvgpr_write_b32 acc[92], 0
v_accvgpr_write_b32 acc[93], 0
v_accvgpr_write_b32 acc[94], 0
v_accvgpr_write_b32 acc[95], 0
v_accvgpr_write_b32 acc[96], 0
v_accvgpr_write_b32 acc[97], 0
v_accvgpr_write_b32 acc[98], 0
v_accvgpr_write_b32 acc[99], 0
v_accvgpr_write_b32 acc[100], 0
v_accvgpr_write_b32 acc[101], 0
v_accvgpr_write_b32 acc[102], 0
v_accvgpr_write_b32 acc[103], 0
v_accvgpr_write_b32 acc[104], 0
v_accvgpr_write_b32 acc[105], 0
v_accvgpr_write_b32 acc[106], 0
v_accvgpr_write_b32 acc[107], 0
v_accvgpr_write_b32 acc[108], 0
v_accvgpr_write_b32 acc[109], 0
v_accvgpr_write_b32 acc[110], 0
v_accvgpr_write_b32 acc[111], 0
v_accvgpr_write_b32 acc[112], 0
v_accvgpr_write_b32 acc[113], 0
v_accvgpr_write_b32 acc[114], 0
v_accvgpr_write_b32 acc[115], 0
v_accvgpr_write_b32 acc[116], 0
v_accvgpr_write_b32 acc[117], 0
v_accvgpr_write_b32 acc[118], 0
v_accvgpr_write_b32 acc[119], 0
v_accvgpr_write_b32 acc[120], 0
v_accvgpr_write_b32 acc[121], 0
v_accvgpr_write_b32 acc[122], 0
v_accvgpr_write_b32 acc[123], 0
v_accvgpr_write_b32 acc[124], 0
v_accvgpr_write_b32 acc[125], 0
v_accvgpr_write_b32 acc[126], 0
v_accvgpr_write_b32 acc[127], 0
label_addr_calculations:
v_and_b32 v[4], v[0], 63
v_mul_lo_u32 v[4], v[4], 4
v_lshrrev_b32 v[5], 6, v[0]
//gl_addr_a_0_0
s_mov_b32 s[55], 0
v_add_u32 v[32], v[5], s[55]
v_mul_lo_u32 v[32], v[32], s[21]
v_add_u32 v[32], v[32], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[32], v[32], s[55]
v_mul_lo_u32 v[32], 4, v[32]
//gl_addr_a_0_1
s_mov_b32 s[55], 4
v_add_u32 v[33], v[5], s[55]
v_mul_lo_u32 v[33], v[33], s[21]
v_add_u32 v[33], v[33], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[33], v[33], s[55]
v_mul_lo_u32 v[33], 4, v[33]
//gl_addr_a_0_2
s_mov_b32 s[55], 8
v_add_u32 v[34], v[5], s[55]
v_mul_lo_u32 v[34], v[34], s[21]
v_add_u32 v[34], v[34], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[34], v[34], s[55]
v_mul_lo_u32 v[34], 4, v[34]
//gl_addr_a_0_3
s_mov_b32 s[55], 12
v_add_u32 v[35], v[5], s[55]
v_mul_lo_u32 v[35], v[35], s[21]
v_add_u32 v[35], v[35], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[35], v[35], s[55]
v_mul_lo_u32 v[35], 4, v[35]
v_and_b32 v[4], v[0], 3
v_mul_lo_u32 v[4], v[4], 4
v_lshrrev_b32 v[5], 2, v[0]
//gl_addr_b_0_0
s_mov_b32 s[55], 0
v_add_u32 v[36], v[5], s[55]
v_mul_lo_u32 v[36], v[36], s[23]
v_add_u32 v[36], v[36], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[36], v[36], s[55]
v_mul_lo_u32 v[36], v[36], 4
//gl_addr_b_0_1
s_mov_b32 s[55], 64
v_add_u32 v[37], v[5], s[55]
v_mul_lo_u32 v[37], v[37], s[23]
v_add_u32 v[37], v[37], v[4]
s_mov_b32 s[55], 0
v_add_u32 v[37], v[37], s[55]
v_mul_lo_u32 v[37], v[37], 4
//gl_a
buffer_load_dwordx4 v[8:11], v[32], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[12:15], v[33], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[16:19], v[34], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[20:23], v[35], s[4:7], s[18] offen offset:0
//gl_b
buffer_load_dwordx4 v[24:27], v[36], s[8:11], s[19] offen offset:0
buffer_load_dwordx4 v[28:31], v[37], s[8:11], s[19] offen offset:0
//lw_a
v_and_b32 v[4], v[0], 63
v_mul_lo_u32 v[4], v[4], 4
v_lshrrev_b32 v[5], 6, v[0]
//lw_addr_a_0_0
s_mov_b32 s[55], 0
v_add_u32 v[38], v[5], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[38], v[38], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[38], v[38], s[55]
v_add_u32 v[38], v[38], v[4]
v_mul_lo_u32 v[38], v[38], 4
//lw_addr_a_0_1
s_mov_b32 s[55], 4
v_add_u32 v[39], v[5], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[39], v[39], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[39], v[39], s[55]
v_add_u32 v[39], v[39], v[4]
v_mul_lo_u32 v[39], v[39], 4
//lw_addr_a_0_2
s_mov_b32 s[55], 8
v_add_u32 v[40], v[5], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[40], v[40], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[40], v[40], s[55]
v_add_u32 v[40], v[40], v[4]
v_mul_lo_u32 v[40], v[40], 4
//lw_addr_a_0_3
s_mov_b32 s[55], 12
v_add_u32 v[41], v[5], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[41], v[41], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[41], v[41], s[55]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//lw_b
v_and_b32 v[4], v[0], 3
v_mul_lo_u32 v[4], v[4], 4
v_lshrrev_b32 v[5], 2, v[0]
//lw_addr_b_0_0
s_mov_b32 s[55], 0
v_add_u32 v[42], v[5], s[55]
s_mov_b32 s[55], 17
v_mul_lo_u32 v[42], v[42], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[42], v[42], s[55]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//lw_addr_b_0_1
s_mov_b32 s[55], 64
v_add_u32 v[43], v[5], s[55]
s_mov_b32 s[55], 17
v_mul_lo_u32 v[43], v[43], s[55]
s_mov_b32 s[55], 0
v_add_u32 v[43], v[43], s[55]
v_add_u32 v[43], v[43], v[4]
v_mul_lo_u32 v[43], v[43], 4
s_waitcnt vmcnt(0)
ds_write_b128 v[38], v[8:11], offset:0
ds_write_b128 v[39], v[12:15], offset:0
ds_write_b128 v[40], v[16:19], offset:0
ds_write_b128 v[41], v[20:23], offset:0
ds_write_b128 v[42], v[24:27], offset:16384
ds_write_b128 v[43], v[28:31], offset:16384
//gl_offset increments for unrolled loop
s_mul_i32 s[55], s[21], 64
s_add_i32 s[18], s[18], s[55]
s_add_i32 s[19], s[19], 64
//swap ds write address
s_mov_b32 s[30], 26624
v_add_u32 v[38], v[38], s[30]
v_add_u32 v[39], v[39], s[30]
v_add_u32 v[40], v[40], s[30]
v_add_u32 v[41], v[41], s[30]
v_add_u32 v[42], v[42], s[30]
v_add_u32 v[43], v[43], s[30]
label_lds_wave_offsets:
//lds read addresses: wave offsets
v_lshrrev_b32 v[1], 6, v[0]
v_and_b32 v[6], 63, v[0]
v_lshrrev_b32 v[3], 1, v[1]
v_mul_lo_u32 v[3], v[3], 32
v_and_b32 v[2], 1, v[1]
v_mul_lo_u32 v[2], v[2], 32
//lds read addresses: thread offsets a
v_and_b32 v[4], 31, v[6]
v_lshrrev_b32 v[5], 5, v[6]
v_add_u32 v[4], v[4], v[2]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[44], v[5], s[55]
v_add_u32 v[44], v[44], v[4]
v_mul_lo_u32 v[44], v[44], 4
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[45], v[5], s[55]
v_add_u32 v[45], v[45], v[4]
v_mul_lo_u32 v[45], v[45], 4
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[46], v[5], s[55]
v_add_u32 v[46], v[46], v[4]
v_mul_lo_u32 v[46], v[46], 4
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
s_mov_b32 s[55], 256
v_mul_lo_u32 v[47], v[5], s[55]
v_add_u32 v[47], v[47], v[4]
v_mul_lo_u32 v[47], v[47], 4
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
//lds read addresses: thread offsets b
v_and_b32 v[5], 31, v[6]
v_lshrrev_b32 v[4], 5, v[6]
v_add_u32 v[5], v[5], v[3]
s_mov_b32 s[55], 17
v_mul_lo_u32 v[48], v[5], s[55]
v_add_u32 v[48], v[48], v[4]
v_mul_lo_u32 v[48], v[48], 4
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
s_mov_b32 s[55], 17
v_mul_lo_u32 v[49], v[5], s[55]
v_add_u32 v[49], v[49], v[4]
v_mul_lo_u32 v[49], v[49], 4
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
//sync prefetch
s_waitcnt lgkmcnt(0)
s_barrier
label_outer_loop:
ds_read_b32 v[50], v[44], offset:0
ds_read_b32 v[51], v[45], offset:0
ds_read_b32 v[52], v[46], offset:0
ds_read_b32 v[53], v[47], offset:0
ds_read_b32 v[58], v[48], offset:16384
ds_read_b32 v[59], v[49], offset:16384
//gl_a
buffer_load_dwordx4 v[8:11], v[32], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[12:15], v[33], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[16:19], v[34], s[4:7], s[18] offen offset:0
buffer_load_dwordx4 v[20:23], v[35], s[4:7], s[18] offen offset:0
//gl_b
buffer_load_dwordx4 v[24:27], v[36], s[8:11], s[19] offen offset:0
buffer_load_dwordx4 v[28:31], v[37], s[8:11], s[19] offen offset:0
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:2048
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16392
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:2048
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16392
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:2048
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:2048
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:4096
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16400
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:4096
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16400
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:4096
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:4096
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:6144
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16408
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:6144
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16408
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:6144
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:6144
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:8192
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16416
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:8192
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16416
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:8192
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:8192
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:10240
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16424
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:10240
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16424
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:10240
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:10240
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:12288
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16432
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:12288
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16432
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:12288
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:12288
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:14336
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16440
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:14336
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16440
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:14336
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:14336
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
s_waitcnt vmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_write_b128 v[38], v[8:11], offset:0
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_write_b128 v[42], v[24:27], offset:16384
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_write_b128 v[39], v[12:15], offset:0
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_write_b128 v[43], v[28:31], offset:16384
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_write_b128 v[40], v[16:19], offset:0
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
ds_write_b128 v[41], v[20:23], offset:0
v_add_i32 v[44], v[44], s[30]
v_add_i32 v[45], v[45], s[30]
v_add_i32 v[46], v[46], s[30]
v_add_i32 v[47], v[47], s[30]
v_add_i32 v[48], v[48], s[30]
v_add_i32 v[49], v[49], s[30]
s_mul_i32 s[30], s[30], -1
v_add_i32 v[38], v[38], s[30]
v_add_i32 v[39], v[39], s[30]
v_add_i32 v[40], v[40], s[30]
v_add_i32 v[41], v[41], s[30]
v_add_i32 v[42], v[42], s[30]
v_add_i32 v[43], v[43], s[30]
s_waitcnt lgkmcnt(0)
s_barrier
s_add_i32 s[15], s[15], 16
//gl_offset increments for unrolled loop
s_mul_i32 s[56], s[21], 64
s_add_i32 s[18], s[18], s[56]
s_add_i32 s[19], s[19], 64
s_add_i32 s[55], s[15], 16
s_cmp_lt_u32 s[55], s[14]
s_cbranch_scc1 label_outer_loop
label_prefetch_last_loop:
//prefetch last loop
ds_read_b32 v[50], v[44], offset:0
ds_read_b32 v[51], v[45], offset:0
ds_read_b32 v[52], v[46], offset:0
ds_read_b32 v[53], v[47], offset:0
ds_read_b32 v[58], v[48], offset:16384
ds_read_b32 v[59], v[49], offset:16384
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:2048
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16392
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:2048
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16392
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:2048
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:2048
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:4096
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16400
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:4096
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16400
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:4096
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:4096
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:6144
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16408
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:6144
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16408
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:6144
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:6144
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:8192
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16416
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:8192
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16416
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:8192
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:8192
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:10240
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16424
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:10240
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16424
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:10240
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:10240
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
ds_read_b32 v[50], v[44], offset:12288
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
ds_read_b32 v[58], v[48], offset:16432
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
ds_read_b32 v[51], v[45], offset:12288
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
ds_read_b32 v[59], v[49], offset:16432
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
ds_read_b32 v[52], v[46], offset:12288
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
ds_read_b32 v[53], v[47], offset:12288
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[50], v[58], acc[0:15]
ds_read_b32 v[54], v[44], offset:14336
v_mfma_f32_32x32x2f32 acc[16:31], v[51], v[58], acc[16:31]
ds_read_b32 v[60], v[48], offset:16440
v_mfma_f32_32x32x2f32 acc[32:47], v[52], v[58], acc[32:47]
ds_read_b32 v[55], v[45], offset:14336
v_mfma_f32_32x32x2f32 acc[48:63], v[53], v[58], acc[48:63]
ds_read_b32 v[61], v[49], offset:16440
v_mfma_f32_32x32x2f32 acc[64:79], v[50], v[59], acc[64:79]
ds_read_b32 v[56], v[46], offset:14336
v_mfma_f32_32x32x2f32 acc[80:95], v[51], v[59], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[52], v[59], acc[96:111]
ds_read_b32 v[57], v[47], offset:14336
v_mfma_f32_32x32x2f32 acc[112:127], v[53], v[59], acc[112:127]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 acc[0:15], v[54], v[60], acc[0:15]
v_mfma_f32_32x32x2f32 acc[16:31], v[55], v[60], acc[16:31]
v_mfma_f32_32x32x2f32 acc[32:47], v[56], v[60], acc[32:47]
v_mfma_f32_32x32x2f32 acc[48:63], v[57], v[60], acc[48:63]
v_mfma_f32_32x32x2f32 acc[64:79], v[54], v[61], acc[64:79]
v_mfma_f32_32x32x2f32 acc[80:95], v[55], v[61], acc[80:95]
v_mfma_f32_32x32x2f32 acc[96:111], v[56], v[61], acc[96:111]
v_mfma_f32_32x32x2f32 acc[112:127], v[57], v[61], acc[112:127]
s_mov_b32 s[28], s[51]
s_mov_b32 s[29], s[52]
//setup srd{c, d}
s_mov_b64 s[4:5], s[40:41]
s_mov_b64 s[8:9], s[42:43]
s_mov_b32 s[7], 131072
s_mov_b32 s[11], 131072
s_mov_b32 s[6], s[12]
s_mul_i32 s[6], s[6], s[25]
s_mul_i32 s[6], s[6], 4
s_mov_b32 s[10], s[12]
s_mul_i32 s[10], s[10], s[27]
s_mul_i32 s[10], s[10], 4
s_mul_i32 s[18], s[17], s[25]
s_add_i32 s[18], s[18], s[16]
s_mul_i32 s[18], s[18], 4
s_mul_i32 s[19], s[17], s[27]
s_add_i32 s[19], s[19], s[16]
s_mul_i32 s[19], s[19], 4
//gw_addr_0_0
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 0
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 0
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_0_0
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_0_0
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_0_0
v_accvgpr_read_b32 v[58], acc[0]
v_accvgpr_read_b32 v[59], acc[1]
v_accvgpr_read_b32 v[60], acc[2]
v_accvgpr_read_b32 v[61], acc[3]
v_accvgpr_read_b32 v[62], acc[4]
v_accvgpr_read_b32 v[63], acc[5]
v_accvgpr_read_b32 v[64], acc[6]
v_accvgpr_read_b32 v[65], acc[7]
v_accvgpr_read_b32 v[66], acc[8]
v_accvgpr_read_b32 v[67], acc[9]
v_accvgpr_read_b32 v[68], acc[10]
v_accvgpr_read_b32 v[69], acc[11]
v_accvgpr_read_b32 v[70], acc[12]
v_accvgpr_read_b32 v[71], acc[13]
v_accvgpr_read_b32 v[72], acc[14]
v_accvgpr_read_b32 v[73], acc[15]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_1_0
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 0
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_1_0
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_1_0
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_1_0
v_accvgpr_read_b32 v[58], acc[16]
v_accvgpr_read_b32 v[59], acc[17]
v_accvgpr_read_b32 v[60], acc[18]
v_accvgpr_read_b32 v[61], acc[19]
v_accvgpr_read_b32 v[62], acc[20]
v_accvgpr_read_b32 v[63], acc[21]
v_accvgpr_read_b32 v[64], acc[22]
v_accvgpr_read_b32 v[65], acc[23]
v_accvgpr_read_b32 v[66], acc[24]
v_accvgpr_read_b32 v[67], acc[25]
v_accvgpr_read_b32 v[68], acc[26]
v_accvgpr_read_b32 v[69], acc[27]
v_accvgpr_read_b32 v[70], acc[28]
v_accvgpr_read_b32 v[71], acc[29]
v_accvgpr_read_b32 v[72], acc[30]
v_accvgpr_read_b32 v[73], acc[31]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_2_0
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 0
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 128
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_2_0
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_2_0
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_2_0
v_accvgpr_read_b32 v[58], acc[32]
v_accvgpr_read_b32 v[59], acc[33]
v_accvgpr_read_b32 v[60], acc[34]
v_accvgpr_read_b32 v[61], acc[35]
v_accvgpr_read_b32 v[62], acc[36]
v_accvgpr_read_b32 v[63], acc[37]
v_accvgpr_read_b32 v[64], acc[38]
v_accvgpr_read_b32 v[65], acc[39]
v_accvgpr_read_b32 v[66], acc[40]
v_accvgpr_read_b32 v[67], acc[41]
v_accvgpr_read_b32 v[68], acc[42]
v_accvgpr_read_b32 v[69], acc[43]
v_accvgpr_read_b32 v[70], acc[44]
v_accvgpr_read_b32 v[71], acc[45]
v_accvgpr_read_b32 v[72], acc[46]
v_accvgpr_read_b32 v[73], acc[47]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_3_0
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 0
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 192
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_3_0
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_3_0
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_3_0
v_accvgpr_read_b32 v[58], acc[48]
v_accvgpr_read_b32 v[59], acc[49]
v_accvgpr_read_b32 v[60], acc[50]
v_accvgpr_read_b32 v[61], acc[51]
v_accvgpr_read_b32 v[62], acc[52]
v_accvgpr_read_b32 v[63], acc[53]
v_accvgpr_read_b32 v[64], acc[54]
v_accvgpr_read_b32 v[65], acc[55]
v_accvgpr_read_b32 v[66], acc[56]
v_accvgpr_read_b32 v[67], acc[57]
v_accvgpr_read_b32 v[68], acc[58]
v_accvgpr_read_b32 v[69], acc[59]
v_accvgpr_read_b32 v[70], acc[60]
v_accvgpr_read_b32 v[71], acc[61]
v_accvgpr_read_b32 v[72], acc[62]
v_accvgpr_read_b32 v[73], acc[63]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_0_1
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 0
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_0_1
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_0_1
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_0_1
v_accvgpr_read_b32 v[58], acc[64]
v_accvgpr_read_b32 v[59], acc[65]
v_accvgpr_read_b32 v[60], acc[66]
v_accvgpr_read_b32 v[61], acc[67]
v_accvgpr_read_b32 v[62], acc[68]
v_accvgpr_read_b32 v[63], acc[69]
v_accvgpr_read_b32 v[64], acc[70]
v_accvgpr_read_b32 v[65], acc[71]
v_accvgpr_read_b32 v[66], acc[72]
v_accvgpr_read_b32 v[67], acc[73]
v_accvgpr_read_b32 v[68], acc[74]
v_accvgpr_read_b32 v[69], acc[75]
v_accvgpr_read_b32 v[70], acc[76]
v_accvgpr_read_b32 v[71], acc[77]
v_accvgpr_read_b32 v[72], acc[78]
v_accvgpr_read_b32 v[73], acc[79]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_1_1
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 64
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_1_1
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_1_1
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_1_1
v_accvgpr_read_b32 v[58], acc[80]
v_accvgpr_read_b32 v[59], acc[81]
v_accvgpr_read_b32 v[60], acc[82]
v_accvgpr_read_b32 v[61], acc[83]
v_accvgpr_read_b32 v[62], acc[84]
v_accvgpr_read_b32 v[63], acc[85]
v_accvgpr_read_b32 v[64], acc[86]
v_accvgpr_read_b32 v[65], acc[87]
v_accvgpr_read_b32 v[66], acc[88]
v_accvgpr_read_b32 v[67], acc[89]
v_accvgpr_read_b32 v[68], acc[90]
v_accvgpr_read_b32 v[69], acc[91]
v_accvgpr_read_b32 v[70], acc[92]
v_accvgpr_read_b32 v[71], acc[93]
v_accvgpr_read_b32 v[72], acc[94]
v_accvgpr_read_b32 v[73], acc[95]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_2_1
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 128
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_2_1
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_2_1
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_2_1
v_accvgpr_read_b32 v[58], acc[96]
v_accvgpr_read_b32 v[59], acc[97]
v_accvgpr_read_b32 v[60], acc[98]
v_accvgpr_read_b32 v[61], acc[99]
v_accvgpr_read_b32 v[62], acc[100]
v_accvgpr_read_b32 v[63], acc[101]
v_accvgpr_read_b32 v[64], acc[102]
v_accvgpr_read_b32 v[65], acc[103]
v_accvgpr_read_b32 v[66], acc[104]
v_accvgpr_read_b32 v[67], acc[105]
v_accvgpr_read_b32 v[68], acc[106]
v_accvgpr_read_b32 v[69], acc[107]
v_accvgpr_read_b32 v[70], acc[108]
v_accvgpr_read_b32 v[71], acc[109]
v_accvgpr_read_b32 v[72], acc[110]
v_accvgpr_read_b32 v[73], acc[111]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
//gw_addr_3_1
v_and_b32 v[5], 31, v[6]
s_mov_b32 s[55], 64
v_add_u32 v[5], v[5], s[55]
v_lshrrev_b32 v[4], 5, v[6]
v_mul_lo_u32 v[4], 4, v[4]
s_mov_b32 s[55], 192
v_add_u32 v[4], v[4], s[55]
v_add_i32 v[5], v[5], v[3]
v_add_i32 v[4], v[4], v[2]
//setup voffset_c_3_1
v_mul_lo_u32 v[41], v[5], s[25]
v_add_u32 v[41], v[41], v[4]
v_mul_lo_u32 v[41], v[41], 4
//setup voffset_d_3_1
v_mul_lo_u32 v[42], v[5], s[27]
v_add_u32 v[42], v[42], v[4]
v_mul_lo_u32 v[42], v[42], 4
//gw_3_1
v_accvgpr_read_b32 v[58], acc[112]
v_accvgpr_read_b32 v[59], acc[113]
v_accvgpr_read_b32 v[60], acc[114]
v_accvgpr_read_b32 v[61], acc[115]
v_accvgpr_read_b32 v[62], acc[116]
v_accvgpr_read_b32 v[63], acc[117]
v_accvgpr_read_b32 v[64], acc[118]
v_accvgpr_read_b32 v[65], acc[119]
v_accvgpr_read_b32 v[66], acc[120]
v_accvgpr_read_b32 v[67], acc[121]
v_accvgpr_read_b32 v[68], acc[122]
v_accvgpr_read_b32 v[69], acc[123]
v_accvgpr_read_b32 v[70], acc[124]
v_accvgpr_read_b32 v[71], acc[125]
v_accvgpr_read_b32 v[72], acc[126]
v_accvgpr_read_b32 v[73], acc[127]
buffer_load_dwordx4 v[24:27], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[28:31], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[32:35], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
buffer_load_dwordx4 v[36:39], v[41], s[4:7], s[18] offen offset:0
s_mul_i32 s[55], s[24], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[41], v[41], s[55]
v_mul_f32 v[58], v[58], s[28]
v_mul_f32 v[59], v[59], s[28]
v_mul_f32 v[60], v[60], s[28]
v_mul_f32 v[61], v[61], s[28]
v_mul_f32 v[62], v[62], s[28]
v_mul_f32 v[63], v[63], s[28]
v_mul_f32 v[64], v[64], s[28]
v_mul_f32 v[65], v[65], s[28]
v_mul_f32 v[66], v[66], s[28]
v_mul_f32 v[67], v[67], s[28]
v_mul_f32 v[68], v[68], s[28]
v_mul_f32 v[69], v[69], s[28]
v_mul_f32 v[70], v[70], s[28]
v_mul_f32 v[71], v[71], s[28]
v_mul_f32 v[72], v[72], s[28]
v_mul_f32 v[73], v[73], s[28]
s_waitcnt vmcnt(0)
v_fma_f32 v[58], s[29], v[24], v[58]
v_fma_f32 v[59], s[29], v[25], v[59]
v_fma_f32 v[60], s[29], v[26], v[60]
v_fma_f32 v[61], s[29], v[27], v[61]
v_fma_f32 v[62], s[29], v[28], v[62]
v_fma_f32 v[63], s[29], v[29], v[63]
v_fma_f32 v[64], s[29], v[30], v[64]
v_fma_f32 v[65], s[29], v[31], v[65]
v_fma_f32 v[66], s[29], v[32], v[66]
v_fma_f32 v[67], s[29], v[33], v[67]
v_fma_f32 v[68], s[29], v[34], v[68]
v_fma_f32 v[69], s[29], v[35], v[69]
v_fma_f32 v[70], s[29], v[36], v[70]
v_fma_f32 v[71], s[29], v[37], v[71]
v_fma_f32 v[72], s[29], v[38], v[72]
v_fma_f32 v[73], s[29], v[39], v[73]
buffer_store_dwordx4 v[58:61], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[62:65], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[66:69], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
buffer_store_dwordx4 v[70:73], v[42], s[8:11], s[19] offen offset:0
s_mul_i32 s[55], s[26], 8
s_mul_i32 s[55], s[55], 4
v_add_u32 v[42], v[42], s[55]
s_endpgm
.Lgenerated_gemm_end:
    .size generated_gemm, .Lgenerated_gemm_end - generated_gemm

.rodata
.p2align 6
.amdhsa_kernel generated_gemm
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_accum_offset 76
  .amdhsa_group_segment_fixed_size 53248
  .amdhsa_next_free_vgpr 202
  .amdhsa_next_free_sgpr 55
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
- .agpr_count: 128
  .args:
  - .address_space: global
    .name: a
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .name: b
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .name: c
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .name: d
    .offset: 24
    .size: 8
    .value_kind: global_buffer
  - .name: m
    .offset: 32
    .size: 4
    .value_kind: by_value
  - .name: size_n
    .offset: 36
    .size: 4
    .value_kind: by_value
  - .name: k
    .offset: 40
    .size: 4
    .value_kind: by_value
  - .name: lda
    .offset: 44
    .size: 4
    .value_kind: by_value
  - .name: ldb
    .offset: 48
    .size: 4
    .value_kind: by_value
  - .name: ldc
    .offset: 52
    .size: 4
    .value_kind: by_value
  - .name: ldd
    .offset: 56
    .size: 4
    .value_kind: by_value
  - .name: alpha
    .offset: 60
    .size: 4
    .value_kind: by_value
  - .name: beta
    .offset: 64
    .size: 4
    .value_kind: by_value
  - .name: numWorkgroupX
    .offset: 68
    .size: 4
    .value_kind: by_value
  - .name: numWorkgroupY
    .offset: 72
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 53248
  .kernarg_segment_align: 8
  .kernarg_segment_size: 76
  .max_flat_workgroup_size: 256
  .name: generated_gemm
  .private_segment_fixed_size: 0
  .sgpr_count: 55
  .symbol: generated_gemm.kd
  .vgpr_count: 74
  .wavefront_size: 64
amdhsa.target: amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
- 1
- 1

.end_amdgpu_metadata
