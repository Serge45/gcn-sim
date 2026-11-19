# AMD Matrix Core (MFMA & WMMA) Lane & Register Layout Guide

The **AMD CDNA (`gfx908`, `gfx90a`, `gfx942`, `gfx950`)** and **RDNA4 (`gfx1201`)** architectures accelerate dense linear algebra, GEMMs, and deep learning primitives via Matrix Core units (**MFMA** — Matrix Fused Multiply-Add on CDNA; **WMMA** — Wave Matrix Multiply-Accumulate on RDNA4 Wave32).

Unlike conventional Vector ALUs (VALU) where each lane operates independently in SIMD fashion, **MFMA / WMMA instructions operate cooperatively across all lanes of a Wavefront** (64 lanes on CDNA, 32 lanes on RDNA4). Lanes collectively hold fragments of input matrices $A$ and $B$, broadcast them across an internal systolic interconnect, and accumulate outer products into **Accumulator VGPRs (AGPRs)** or **unified VGPRs** (`v_mfma_*` / `v_wmma_*`).

`gcn-sim` implements bit-accurate functional emulation of CDNA1–CDNA4 MFMA and RDNA4 WMMA v2 matrix instructions, cross-lane fragment broadcasting, and cycle-approximate pipeline timing. The detailed lane walkthrough below uses **CDNA2 (`gfx90a`)** as the baseline reference.

---

## 1. Architectural Overview & Wavefront Mapping

<div align="center">

![MFMA Lane Layout Architecture](images/mfma_lane_layout.png)

</div>

### Key Architectural Concepts
1. **Wavefront Size = 64 Lanes**:
   All 64 threads (`lane = 0 .. 63`) in the wavefront collaborate to compute a matrix tile.
2. **EXEC Mask Bypass**:
   Per AMD CDNA ISA specifications, **MFMA instructions bypass the 64-bit `EXEC` mask**. Even if a lane is disabled by conditional branching, it participates in matrix outer products to guarantee correct accumulation.
3. **Dedicated Accumulator Storage (AGPRs)**:
   GFX90a provides 256 standard Vector General-Purpose Registers (`v[0:255]`) and 256 Accumulator Registers (`a[0:255]`).
   - GEMM kernels accumulate directly into `a[...]` without occupying standard vector registers.
   - This enables perfect double-buffering: global/LDS memory prefetches load into `v[...]` while the Matrix Cores compute on `a[...]` in parallel.

---

## 2. Instruction Matrix Formats & Lane Layouts

### A. `v_mfma_f32_32x32x1f32` (Single-Precision Outer Product, $K=1$)

Computes a $32 \times 32$ matrix multiply with $K=1$:
$$\mathbf{D}_{32 \times 32} = \mathbf{A}_{32 \times 1} \times \mathbf{B}_{1 \times 32} + \mathbf{C}_{32 \times 32}$$

- **Input Fragments**:
  - `src0` (Matrix $A$): 1 float per lane (32 unique rows across block).
  - `src1` (Matrix $B$): 1 float per lane (32 unique columns across block).
- **Output Storage**:
  - 1024 output elements total ($32 \times 32$).
  - Distributed across **64 lanes $\times$ 16 registers = 1024 elements** (`a[0:15]` or `v[0:15]`).

#### Block & Lane Partitioning
The 64-lane wavefront is split into **2 blocks of 32 lanes**:
- **Block 0 (Lanes 0–31)**: Computes matrix rows $0 \dots 15$.
- **Block 1 (Lanes 32–63)**: Computes matrix rows $16 \dots 31$.

```mermaid
flowchart TD
    subgraph Wavefront["64-Lane Wavefront"]
        subgraph B0["Block 0 (Lanes 0..31)"]
            L0["Lane 0: col 0"]
            L1["Lane 1: col 1"]
            L31["Lane 31: col 31"]
        end
        subgraph B1["Block 1 (Lanes 32..63)"]
            L32["Lane 32: col 0"]
            L33["Lane 33: col 1"]
            L63["Lane 63: col 31"]
        end
    end

    subgraph OutputTile["32x32 Output Tile D (1024 Elements)"]
        R0["Rows 0..15: 16 GPRs x 32 Lanes (Block 0)"]
        R1["Rows 16..31: 16 GPRs x 32 Lanes (Block 1)"]
    end

    B0 --> R0
    B1 --> R1
```

#### Mapping Equations:
For lane $L \in [0, 63]$ and register index $g \in [0, 15]$:
$$\text{block} = \lfloor L / 32 \rfloor \in \{0, 1\}$$
$$\text{col} = L \pmod{32}$$
$$\text{row} = \text{block} \times 16 + g$$
$$\mathbf{D}[\text{row}][\text{col}] \leftarrow \mathbf{a}[g][L]$$

- **Matrix $A$ Broadcast**: $A[\text{row}][0]$ resides in `src0` of Lane $(\text{block} \times 32 + g)$. The Matrix Core broadcasts this scalar to all 32 lanes in the block.
- **Matrix $B$ Vector**: $B[0][\text{col}]$ resides in `src1` of Lane $L$.

---

### B. `v_mfma_f32_32x32x2f32` (Unrolled Step, $K=2$)

Computes a $32 \times 32$ matrix multiply accumulating over $K=2$:
$$\mathbf{D}_{32 \times 32} = \sum_{k=0}^{1} \mathbf{A}_{32 \times 1}^{(k)} \times \mathbf{B}_{1 \times 32}^{(k)} + \mathbf{C}_{32 \times 32}$$

This instruction is the workhorse of AMD Tensile and rocBLAS FP32 GEMMs.

#### K-Group Partitioning
The 64 lanes are partitioned into **two K-groups**:
- **K-group 0 (Lanes 0–31)**: Holds $A[:, 0]$ and $B[0, :]$.
- **K-group 1 (Lanes 32–63)**: Holds $A[:, 1]$ and $B[1, :]$.

```
Lane Index (0..63):
┌───────────────────────────────┬───────────────────────────────┐
│     K-group 0 (Lanes 0..31)   │    K-group 1 (Lanes 32..63)   │
│       Processes K = 0         │        Processes K = 1        │
└───────────────────────────────┴───────────────────────────────┘
```

#### Register & Matrix Coordinate Mapping:
For lane $L$ ($lane\_hi = \lfloor L / 32 \rfloor$, $col = L \pmod{32}$) and GPR $g \in [0, 15]$:
$$\text{row} = 8 \times \lfloor g / 4 \rfloor + 4 \times lane\_hi + (g \pmod 4)$$

```
GPR Mapping Matrix (Row Index vs GPR g and Lane Group):
┌───────────┬───────────────┬───────────────┐
│ GPR Index │ Lane 0..31    │ Lane 32..63   │
│           │ (lane_hi = 0) │ (lane_hi = 1) │
├───────────┼───────────────┼───────────────┤
│ g = 0..3  │ Rows 0..3     │ Rows 4..7     │
│ g = 4..7  │ Rows 8..11    │ Rows 12..15   │
│ g = 8..11 │ Rows 16..19   │ Rows 20..23   │
│ g = 12..15│ Rows 24..27   │ Rows 28..31   │
└───────────┴───────────────┴───────────────┘
```

- Each lane computes all 16 rows assigned to its column across both $K=0$ and $K=1$ in a single hardware cycle burst.

---

### C. `v_mfma_f32_16x16x4f32` (Compact Tile, $K=4$)

Computes a $16 \times 16$ matrix multiply accumulating over $K=4$:
$$\mathbf{D}_{16 \times 16} = \sum_{k=0}^{3} \mathbf{A}_{16 \times 1}^{(k)} \times \mathbf{B}_{1 \times 16}^{(k)} + \mathbf{C}_{16 \times 16}$$

- **Tile Size**: $16 \times 16 = 256$ elements.
- **Output Storage**: 64 lanes $\times$ 4 registers (`a[0:3]`) = 256 elements.
- **Four K-groups of 16 lanes**:
  - $kg = \lfloor L / 16 \rfloor \in \{0, 1, 2, 3\}$.
  - Lane $L$ contributes to:
    $$\text{col} = L \pmod{16}$$
    $$\text{row} = \lfloor L / 16 \rfloor \times 4 + g \quad (g \in [0, 3])$$

---

## 3. Supported MFMA & WMMA Instruction Summary

### CDNA1 / CDNA2 (`gfx908` / `gfx90a`, Wave64)

| Opcode | Input Types | Output Type | Tile Size ($M \times N \times K$) | Output GPRs / Lane | Total Elements |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `v_mfma_f32_32x32x1f32` | FP32 $\times$ FP32 | FP32 | $32 \times 32 \times 1$ | 16 | 1024 |
| `v_mfma_f32_32x32x2f32` | FP32 $\times$ FP32 | FP32 | $32 \times 32 \times 2$ | 16 | 1024 |
| `v_mfma_f32_16x16x1f32` | FP32 $\times$ FP32 | FP32 | $16 \times 16 \times 1$ | 16 | 256 (×4 blocks) |
| `v_mfma_f32_16x16x4f32` | FP32 $\times$ FP32 | FP32 | $16 \times 16 \times 4$ | 4 | 256 |
| `v_mfma_f32_4x4x1f32` | FP32 $\times$ FP32 | FP32 | $4 \times 4 \times 1$ | 4 | 16 (×16 blocks) |
| `v_mfma_f32_32x32x4f16` / `x8f16` | FP16 $\times$ FP16 | FP32 | $32 \times 32 \times \{4, 8\}$ | 16 | 1024 |
| `v_mfma_f32_16x16x4f16` / `x16f16` | FP16 $\times$ FP16 | FP32 | $16 \times 16 \times \{4, 16\}$ | 4 | 256 |
| `v_mfma_f32_32x32x4bf16` / `x8bf16_1k` | BF16 $\times$ BF16 | FP32 | $32 \times 32 \times \{4, 8\}$ | 16 | 1024 |
| `v_mfma_f32_16x16x8bf16` / `x16bf16_1k` | BF16 $\times$ BF16 | FP32 | $16 \times 16 \times \{8, 16\}$ | 4 | 256 |
| `v_mfma_i32_32x32x4i8` / `x8i8` | INT8 $\times$ INT8 | INT32 | $32 \times 32 \times \{4, 8\}$ | 16 | 1024 |
| `v_mfma_i32_16x16x4i8` / `x16i8` | INT8 $\times$ INT8 | INT32 | $16 \times 16 \times \{4, 16\}$ | 4 | 256 |
| `v_mfma_f64_16x16x4f64` | FP64 $\times$ FP64 | FP64 | $16 \times 16 \times 4$ | 8 (pairs) | 256 |

### CDNA3 (`gfx942`) & CDNA4 (`gfx950`, Wave64) Additions

| Arch | Opcodes | Data Formats & Feature Notes |
| :--- | :--- | :--- |
| **CDNA3 (`gfx942`)** | `v_mfma_f32_16x16x8_xf32`, `32x32x4_xf32` | TF32 / XF32 19-bit matrix multiply |
| **CDNA3 (`gfx942`)** | `v_mfma_i32_16x16x32i8`, `32x32x16i8` | Double-rate INT8 |
| **CDNA3 (`gfx942`)** | `v_mfma_f32_{16x16x32,32x32x16}_{fp8,bf8}_{fp8,bf8}` | OCP FP8 (E4M3) & BF8 (E5M2) mixed/pure |
| **CDNA4 (`gfx950`)** | `v_mfma_f32_{16x16x32,32x32x16}_{f16,bf16}`, `v_mfma_i32_{16x16x64,32x32x32}_i8` | Double-rate FP16 / BF16 / INT8 |
| **CDNA4 (`gfx950`)** | `v_mfma_ld_scale_b32`, `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` | OCP Microscaling (MXFP4 / MXFP6 / MXFP8) with block scale factors |

### RDNA4 (`gfx1201`, Wave32) WMMA v2 Summary

| Opcode | Input Types | Output Type | Tile Size | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `v_wmma_f32_16x16x16_f16`, `bf16` | FP16 / BF16 | FP32 | $16 \times 16 \times 16$ | Wave32 dual-WGP matrix core |
| `v_wmma_f16_16x16x16_f16`, `bf16_16x16x16_bf16` | FP16 / BF16 | FP16 / BF16 | $16 \times 16 \times 16$ | Native 16-bit accumulator output |
| `v_wmma_i32_16x16x16_iu8`, `iu4` | INT8 / INT4 | INT32 | $16 \times 16 \times 16$ | Quantized integer WMMA |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` | FP8 / BF8 | FP32 | $16 \times 16 \times 16$ | OCP E4M3 / E5M2 mixed/pure |

---

## 4. How `gcn-sim` Simulates Matrix Cores

In `src/isa_executor.cpp` and `src/execution_engine.cpp`:
1. **Cross-Lane Collection**:
   Before updating any lane state, `read_vsrc()` (and `read_acc()` where appropriate) extracts all source lane values into host arrays (`a_all`, `b_all`). This mirrors the hardware's internal register read stage.
2. **Hardware Outer-Product Emulation**:
   For each lane, the exact row and column formulas map incoming matrix elements into standard IEEE-754 `fmaf(a, b, c)`, double-precision `fma()`, or integer/FP8/MXFP4 dot products.
3. **Execution Shadow & Pipeline Interaction**:
   `src/isa_executor.cpp` and `src/execution_engine.cpp` track the MFMA execution duration (e.g., 64 cycles on `32x32x2f32`) and enable overlapping independent VALU instructions in the shadow window without extra latency overhead.
