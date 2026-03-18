# CDNA2 (GFX90A) MFMA Lane Layout Reference

This document explains the lane-to-register mapping for every MFMA instruction
variant supported by gcn-sim.  It is intended for developers writing or
debugging kernel code that uses MFMA, and for simulator engineers maintaining
`isa_executor.cpp`.

---

## Background: What is MFMA?

MFMA (Matrix-Fused-Multiply-Add) is a wavefront-wide instruction family on
AMD CDNA2 (MI-200 / GFX90A) GPUs.  A single MFMA instruction causes all 64
lanes of a wavefront to cooperate in computing a small dense matrix product:

```
D = A * B + C
```

where A, B, C, D are sub-tiles of the full matrices, and the dimensions of
those tiles are encoded in the instruction name (e.g. `16x16x1` means
M=16, N=16, K=1 per call).

Key properties:

- **MFMA ignores EXEC.**  All 64 lanes participate unconditionally.
- **Cross-lane reads.**  The hardware reads src0/src1 values from other lanes
  within the same "block" (see below).  A compiler cannot treat MFMA like an
  ordinary per-lane arithmetic instruction.
- **Accumulator registers (AGPRs).**  Output is accumulated into AGPRs
  (`a0`, `a1`, …).  On GFX90A the compiler may also use VGPRs for the
  accumulator when the register pressure allows.
- **One call, one K-slice.**  For the `K>1` variants (e.g. `16x16x4f32`) a
  single instruction covers `K` consecutive rows/columns of A and B.  The
  kernel loops over the full K dimension by calling the instruction multiple
  times, re-using the same accumulator.

---

## Terminology

| Term | Meaning |
|---|---|
| **lane** | One of the 64 threads in the wavefront (0–63) |
| **block** | A contiguous group of lanes that form one MFMA sub-unit |
| **block_base** | First lane index of the block that contains `lane` |
| **t** | Lane's position within its block (`lane % block_size`) |
| **K-group (kg)** | One of the sub-groups that each provide one K-slice of A and B |
| **gpr** | Register index offset from the base register (0, 1, 2, …) |
| **AGPR** | Accumulator register (`a[N]` in GCN assembly) |
| **src0, src1, src2** | MFMA instruction source operands |
| **dst** | MFMA instruction destination (also used as accumulator input) |

In the equations below:

- `A[row][k]` is read from `src0` of lane `block_base + row`
- `B[k][col]` is read from `src1` of lane `block_base + col`
- `D[row][col]` is written to `(a/v)[dst.index + gpr][lane]`

---

## F32 Input Variants (Exact Cross-Lane Semantics)

These five variants have correct CDNA2 cross-lane outer-product behavior in
the simulator.

---

### `v_mfma_f32_4x4x1f32`

**Matrix tile:** 4 × 4, one K-slice per call
**Wavefront partitioning:** 16 independent blocks of 4 lanes
**Output GPRs per lane:** 4
**Accumulator type:** AGPR or VGPR (compiler decides)

#### Lane-to-tile mapping

```
block_id   = lane / 4
t          = lane % 4          (row index within the 4×4 tile)
block_base = block_id * 4

src0[lane] = A[t][k]           (this lane's own A row element)
src1[lane] = B[k][t]           (this lane's own B column element)

Output register dst[gpr][lane] = D[row=t][col=gpr]
  where gpr ∈ {0,1,2,3}

B[k][col=gpr] is src1 of lane (block_base + gpr)   ← cross-lane read
```

#### What the hardware computes (per block, per K call):

```
for col in 0..3:
    D[row][col] += A[row][k] * B[k][col]
```

Both `A[row][k]` and `B[k][col]` are sourced from lanes within the same block.
The hardware broadcast is: lane `block_base+t` provides its `src1` value as
`B[k][t]` to all other lanes in the block.

#### Visual for one 4×4 block (lanes 0–3):

```
Lane 0: src0=A[0][k], src1=B[k][0] → writes D[0][0..3]
Lane 1: src0=A[1][k], src1=B[k][1] → writes D[1][0..3]
Lane 2: src0=A[2][k], src1=B[k][2] → writes D[2][0..3]
Lane 3: src0=A[3][k], src1=B[k][3] → writes D[3][0..3]

dst[0][lane 0] = D[0][0],  dst[1][lane 0] = D[0][1],
dst[2][lane 0] = D[0][2],  dst[3][lane 0] = D[0][3]

B[k][col=1] used by lane 0 comes from src1 of lane 1  ← cross-lane
```

---

### `v_mfma_f32_16x16x1f32`

**Matrix tile:** 16 × 16, one K-slice per call
**Wavefront partitioning:** 4 independent blocks of 16 lanes
**Output GPRs per lane:** 16 (AGPRs: `a[base+0]`…`a[base+15]`)
**Accumulator type:** AGPR or VGPR

#### Lane-to-tile mapping

```
block_id   = lane / 16
t          = lane % 16
block_base = block_id * 16

src0[lane] = A[t][k]
src1[lane] = B[k][t]

dst[gpr][lane] = D[row=t][col=gpr],  gpr ∈ {0..15}

B[k][col=gpr] comes from src1 of lane (block_base + gpr)  ← cross-lane
```

Each of the 4 blocks independently computes a 16×16 tile.  In a typical GEMM
kernel a single workgroup spans exactly one block (16 threads), and the four
blocks of the wavefront happen to process different tiles (depending on how
the compiler assigns work).

---

### `v_mfma_f32_16x16x4f32`

**Matrix tile:** 16 × 16, four K-slices per call
**Wavefront partitioning:** 1 block of 64 lanes split into 4 K-groups of 16
**Output GPRs per lane:** 4 (AGPRs)
**Accumulator type:** AGPR or VGPR

#### Lane-to-tile mapping

```
kg_out     = lane / 16          (K-group index: 0..3)
col        = lane % 16          (output column index)

For gpr ∈ {0..3}:
    row = kg_out * 4 + gpr      (output row index)
    dst[gpr][lane] = D[row][col]

For each K-group kg ∈ {0..3}:
    A[kg*4+gpr][k_base+kg] comes from src0 of lane (kg*16 + gpr)
    B[k_base+kg][col]      comes from src1 of lane (kg*16 + col)
```

All four K-groups cooperate to produce the full 16×16 output tile in a single
instruction call.  Each lane accumulates across all four K-groups into its
4 GPRs.

#### Visual (one instruction call covers K = k_base..k_base+3):

```
Lanes 0..15  (kg=0): provide A[0..3][k0] in src0, B[k0][0..15] in src1
Lanes 16..31 (kg=1): provide A[4..7][k1] in src0, B[k1][0..15] in src1
Lanes 32..47 (kg=2): provide A[8..11][k2] in src0, B[k2][0..15] in src1
Lanes 48..63 (kg=3): provide A[12..15][k3] in src0, B[k3][0..15] in src1

Lane 0  (kg=0, col=0): dst[0]=D[0][0], dst[1]=D[1][0], dst[2]=D[2][0], dst[3]=D[3][0]
Lane 16 (kg=1, col=0): dst[0]=D[4][0], dst[1]=D[5][0], dst[2]=D[6][0], dst[3]=D[7][0]
```

---

### `v_mfma_f32_32x32x1f32`

**Matrix tile:** 32 × 32, one K-slice per call
**Wavefront partitioning:** 2 independent blocks of 32 lanes
**Output GPRs per lane:** 16 (AGPRs `a[base+0]`…`a[base+15]`)
**Accumulator type:** AGPR or VGPR

#### Lane-to-tile mapping

```
block_id   = lane / 32
t          = lane % 32
block_base = block_id * 32

src0[lane] = A[block_id*16 + (t % 16)][k]  (only t < 16 contribute new A rows)
src1[lane] = B[k][t]                         (column = t)

For gpr ∈ {0..15}:
    dst[gpr][lane] = D[block_id*16 + gpr][col=t]

A[block_id*16+gpr][k] comes from src0 of lane (block_base + gpr)  ← cross-lane
```

The 32-lane block produces a 16×32 sub-tile.  Lanes 0..15 and 16..31 within
a block redundantly hold A rows 0..15 (i.e. `t%16` gives the A row index
within the block), but only GPRs 0..15 are meaningful output — not 16..31.

#### Visual for block 0 (lanes 0–31):

```
Lane  0: src0=A[0][k],  src1=B[k][0]  → writes D[0..15][0]   via dst[0..15][lane 0]
Lane  1: src0=A[1][k],  src1=B[k][1]  → writes D[0..15][1]   via dst[0..15][lane 1]
…
Lane 15: src0=A[15][k], src1=B[k][15] → writes D[0..15][15]  via dst[0..15][lane 15]
Lane 16: src0=A[0][k],  src1=B[k][16] → writes D[0..15][16]  via dst[0..15][lane 16]
…
Lane 31: src0=A[15][k], src1=B[k][31] → writes D[0..15][31]  via dst[0..15][lane 31]
```

Cross-lane: `A[gpr][k]` for each gpr is read from `src0` of `lane = block_base+gpr`.

---

### `v_mfma_f32_32x32x2f32`

**Matrix tile:** 32 × 32, two K-slices per call
**Wavefront partitioning:** 1 block of 64 lanes split into 2 K-groups of 32
**Output GPRs per lane:** 16 (AGPRs)
**Accumulator type:** AGPR or VGPR

#### Lane-to-tile mapping

```
kg_out     = lane / 32        (K-group index: 0 or 1)
col        = lane % 32        (output column)

For gpr ∈ {0..15}:
    row = kg_out * 16 + gpr
    dst[gpr][lane] = D[row][col]

For each K-group kg ∈ {0,1}:
    A[kg*16+gpr][k_base+kg] comes from src0 of lane (kg*32 + gpr)
    B[k_base+kg][col]       comes from src1 of lane (kg*32 + col)
```

This is the multi-K analogue of 32×32×1.  Each instruction call contributes
K = 2 columns of A and rows of B.

---

## F16 Input Variants (Simplified Per-Lane Model)

> **Simulator note:** These variants use a simplified per-lane model in
> `isa_executor.cpp`, not exact CDNA2 cross-lane semantics.  They accumulate
> the dot product of the two packed F16 values in each lane's src0/src1 into
> the accumulator, without cross-lane outer-product.  Results are not
> bitwise-identical to hardware for real GEMM kernels.

### `v_mfma_f32_4x4x4f16`

**Matrix tile:** 4 × 4, K=4 per call (two f16 in src0, two f16 in src1)
**Output GPRs per lane:** 1 (AGPR)
**Simulator model:** `acc[lane] += dot(a[0:1], b[0:1])` (per lane)

### `v_mfma_f32_16x16x4f16` / `v_mfma_f32_16x16x16f16`

**Matrix tile:** 16 × 16
**Output GPRs per lane:** 1 (AGPR)
**Simulator model:** per-lane dot product of packed f16 operands

### `v_mfma_f32_32x32x4f16` / `v_mfma_f32_32x32x8f16`

**Matrix tile:** 32 × 32
**Output GPRs per lane:** 16 (AGPRs), but all 16 receive the same per-lane dot
**Simulator model:** simplified — not a true 32×32 cross-lane outer product

---

## BF16 Input Variants (Simplified Per-Lane Model)

Same simplified model as F16.  BF16 values are stored as the upper 16 bits of
a float32 word (i.e. bfloat16 = f32 with mantissa zeroed to 7 bits).

### `v_mfma_f32_4x4x2bf16` / `v_mfma_f32_4x4x4bf16_1k`

**Output GPRs per lane:** 1 (AGPR)

### `v_mfma_f32_16x16x2bf16` / `_16x16x8bf16` / `_16x16x4bf16_1k` / `_16x16x16bf16_1k`

**Output GPRs per lane:** 1 (AGPR)

### `v_mfma_f32_32x32x2bf16` / `_32x32x4bf16` / `_32x32x4bf16_1k` / `_32x32x8bf16_1k`

**Output GPRs per lane:** 16 (AGPRs)

---

## I32 Output / I8 Input Variants (Simplified Per-Lane Model)

### `v_mfma_i32_4x4x4i8`

**Matrix tile:** 4 × 4, K=4 (four i8 packed in one u32)
**Output GPRs per lane:** 1 (AGPR, i32)
**Simulator model:** `acc[lane] += dot4(a_i8[0:3], b_i8[0:3])` per lane

### `v_mfma_i32_16x16x4i8` / `v_mfma_i32_16x16x16i8`

**Output GPRs per lane:** 1 (AGPR, i32)
**Simulator model:** per-lane i8 dot product

### `v_mfma_i32_32x32x4i8` / `v_mfma_i32_32x32x8i8`

**Output GPRs per lane:** 16 (AGPRs, i32), all 16 receive the same per-lane dot
**Simulator model:** simplified

---

## F64 Variants (Simplified Per-Lane Model)

F64 MFMA uses VGPR pairs for src0/src1 and AGPR pairs for the accumulator.

### `v_mfma_f64_4x4x4f64`

**Matrix tile:** 4 × 4, K=4
**src0:** VGPR pair (f64) per lane
**src1:** VGPR pair (f64) per lane
**dst:** AGPR pair (f64) per lane
**Output GPRs per lane:** 2 AGPRs (one f64)
**Simulator model:** `acc_f64[lane] = fma(a_f64[lane], b_f64[lane], acc_f64[lane])` per lane

### `v_mfma_f64_16x16x4f64`

**Matrix tile:** 16 × 16, K=4
**Output GPRs per lane:** 2 AGPRs (one f64)
**Simulator model:** same per-lane fma

---

## Summary Table

| Instruction | Tile M×N | K/call | Blocks | Lanes/block | Out GPRs/lane | Acc type | Exact cross-lane? |
|---|---|---|---|---|---|---|---|
| `f32_4x4x1f32`     | 4×4   | 1 | 16 | 4  | 4  | AGPR/VGPR | **Yes** |
| `f32_16x16x1f32`   | 16×16 | 1 | 4  | 16 | 16 | AGPR/VGPR | **Yes** |
| `f32_32x32x1f32`   | 32×32 | 1 | 2  | 32 | 16 | AGPR/VGPR | **Yes** |
| `f32_16x16x4f32`   | 16×16 | 4 | 1 (4 K-groups×16) | 16 | 4 | AGPR/VGPR | **Yes** |
| `f32_32x32x2f32`   | 32×32 | 2 | 1 (2 K-groups×32) | 32 | 16 | AGPR/VGPR | **Yes** |
| `f32_4x4x4f16`     | 4×4   | 4 | — | — | 1  | AGPR | No (simplified) |
| `f32_16x16x4f16`   | 16×16 | 4 | — | — | 1  | AGPR | No |
| `f32_16x16x16f16`  | 16×16 | 16| — | — | 1  | AGPR | No |
| `f32_32x32x4f16`   | 32×32 | 4 | — | — | 16 | AGPR | No |
| `f32_32x32x8f16`   | 32×32 | 8 | — | — | 16 | AGPR | No |
| `f32_4x4x2bf16`    | 4×4   | 2 | — | — | 1  | AGPR | No |
| `f32_4x4x4bf16_1k` | 4×4   | 4 | — | — | 1  | AGPR | No |
| `f32_16x16x*bf16*` | 16×16 | varies | — | — | 1  | AGPR | No |
| `f32_32x32x*bf16*` | 32×32 | varies | — | — | 16 | AGPR | No |
| `i32_4x4x4i8`      | 4×4   | 4 | — | — | 1  | AGPR | No |
| `i32_16x16x*i8`    | 16×16 | varies | — | — | 1  | AGPR | No |
| `i32_32x32x*i8`    | 32×32 | varies | — | — | 16 | AGPR | No |
| `f64_4x4x4f64`     | 4×4   | 4 | — | — | 2 (f64 pair) | AGPR | No |
| `f64_16x16x4f64`   | 16×16 | 4 | — | — | 2 (f64 pair) | AGPR | No |

---

## Writing a GEMM Kernel: Worked Example (4×4×1 F32)

This shows the minimum-correct pattern for a kernel using `v_mfma_f32_4x4x1f32`
with a 4-thread workgroup (one MFMA block per workgroup):

```hip
// block=(4,1,1), grid=((N/4)*4, M/4, 1)
extern "C" __global__
void gemm_4x4(const float* A, const float* B, float* C, int M, int N, int K)
{
    using float32x4_t = __attribute__((__vector_size__(4*sizeof(float)))) float;

    int wg_row = blockIdx.y;
    int wg_col = blockIdx.x;
    int lane   = threadIdx.x;  // 0..3

    // lane = t = row index within this block's 4×4 output tile
    int c_row = wg_row * 4 + lane;

    float32x4_t acc = {0, 0, 0, 0};  // acc[gpr] = D[lane][gpr]

    for (int k = 0; k < K; k++) {
        // Each lane provides its own A row element:
        float a = (c_row < M) ? A[c_row * K + k] : 0.f;

        // Each lane provides its own B column element:
        // lane t provides B[k][wg_col*4 + t], which the hardware broadcasts
        // to produce B[k][col=gpr] in acc[gpr] for all other lanes.
        float b = (wg_col*4 + lane < N) ? B[k*N + wg_col*4 + lane] : 0.f;

        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(a, b, acc, 0, 0, 0);
    }

    // After the loop: acc[gpr] = D[row=lane][col=gpr]
    if (c_row < M) {
        for (int g = 0; g < 4; g++) {
            int c_col = wg_col * 4 + g;
            if (c_col < N) C[c_row * N + c_col] = acc[g];
        }
    }
}
```

Key points:
- Each lane loads **its own** A row element (`A[c_row][k]`).
- Each lane loads **its own** B column element (`B[k][wg_col*4+lane]`).
- The MFMA hardware does the cross-lane broadcast: `acc[gpr]` on lane `t`
  accumulates `A[t][k] * B[k][gpr]` where `B[k][gpr]` comes from lane `gpr`.
- After the K loop, each lane writes **4 output elements** (one full row of
  the 4×4 tile).

---

## Common Mistakes

1. **Treating MFMA as per-lane multiply-add.**  `acc[0] += a * b` per lane
   produces garbage: lane 0 would only accumulate `A[0][k] * B[k][0]` and miss
   `A[0][k] * B[k][1..3]`.

2. **Checking EXEC before writing accumulator.**  MFMA always operates on all
   64 lanes; gating on EXEC corrupts blocks where EXEC happens to be 0.

3. **Using 16-thread workgroups for 4×4×1.**  16 threads = 4 MFMA blocks.
   Lanes 4–15 form blocks 1–3 and compute different 4×4 tiles.  The kernel
   must map `c_col = wg_col*16 + (lane/4)*4 + gpr` and
   `c_row = wg_row*4 + lane%4` if all 16 lanes cover a 4×16 sub-tile.

4. **Wrong accumulator GPR count.**  Using only 1 AGPR when the instruction
   produces 4/16 means 3/15 output elements are silently discarded.
