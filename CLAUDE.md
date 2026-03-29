# CLAUDE.md — Agent Guide for gcn-sim

This file is for AI agents working on the gcn-sim codebase.
Read it fully before making any changes.

---

## What this project is

`gcn-sim` is a CPU-based instruction-level simulator for AMD gfx90a (MI-200 / CDNA2) GPU kernels.
It loads compiled HSACO ELF binaries, decodes gfx90a machine code, and executes it on the host CPU
with no GPU or GPU driver dependency.

Key entry point: `include/gcnsim/gcnsim.h` — three public functions:
`module_load`, `module_get_function`, `launch_kernel`.

---

## Build commands

```bash
# Configure (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGCNSIM_BUILD_TESTS=ON

# Build everything (library + tests + kernel .hsaco files)
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build -V

# Run a single test binary directly
./build/tests/test_isa_decoder
./build/tests/test_isa_executor
./build/tests/test_vector_add
./build/tests/test_gemm_mfma --hsaco build/tools/gemm_mfma/gemm_mfma.hsaco
./build/tests/test_mfma_kernels --hsaco build/tools/mfma_kernels/mfma_kernels.hsaco
./build/tests/test_asm_gemm --hsaco build/tools/asm_gemm_mfma/asm_gemm_mfma.co
./build/tests/test_tensile_gemm --hsaco build/tools/tensile_style_gemm/tensile_style_gemm.co
```

**Expected:** 99 tests pass across 11 suites.
Always run the full test suite after any change to `src/` or `include/`.

---

## Project layout

```
include/gcnsim/    Public headers — data structures and API declarations
src/               Implementation — one .cpp per concern
tests/             GTest unit and integration tests
tools/vector_add/       Vector-add HIP kernel (.hsaco) for integration tests
tools/gemm_mfma/        MFMA GEMM HIP kernel (.hsaco) for MFMA integration tests
tools/mfma_kernels/     Multi-shape MFMA kernels (.hsaco) for integration tests
tools/asm_gemm_mfma/    Hand-written GFX90A assembly MFMA GEMM (.co)
tools/tensile_style_gemm/ Tensile-style SGEMM assembly kernel (.co) for end-to-end GEMM test
```

| File | Responsibility |
|---|---|
| `elf_loader.cpp` | Parse HSACO ELF: sections, symbol table, kernel descriptors |
| `metadata_parser.cpp` | Decode MsgPack-encoded kernel metadata from ELF .note section |
| `isa_decoder.cpp` | Decode raw bytes → `Instruction` struct |
| `isa_executor.cpp` | Execute one `Instruction` against a `WavefrontState` |
| `wavefront.cpp` | `init_wavefront()` (SGPR/VGPR setup from KD) + `run_wavefront()` (returns `WaveStatus`) |
| `execution_engine.cpp` | Grid dispatch, wavefront splitting, LDS/scratch allocation, barrier sync, hidden arg injection |
| `gcnsim_api.cpp` | Public API wrappers: thin glue over the above |

---

## Coding conventions

- **C++20**, no exceptions thrown across public API boundaries (use `std::optional` + `std::string& err`).
- **`namespace gcnsim`** wraps everything; no `using namespace` in headers.
- **No raw `reinterpret_cast` for memory loads** — use `memcpy`-based helpers in `memory_model.h`
  (`mem_load_u32`, `mem_store_f32`, etc.) to avoid strict aliasing UB.
- **Bit-field extraction** uses plain shifts and masks on `uint32_t` words; no C++ bitfield structs.
- All float bit-casting goes through `memcpy`, not unions or pointer casts.
- Comments are placed above the code they describe, not on the same line unless very short.
- Keep functions in `isa_executor.cpp` and `isa_decoder.cpp` as plain `static` helpers within
  the translation unit; avoid adding new headers for internal helpers.

---

## Key data structures

### `WavefrontState` (`include/gcnsim/wavefront.h`)
The complete execution state of one 64-lane wavefront:
- `sgpr[108]` — scalar registers; VCC is `sgpr[106:107]`
- `vgpr[256][64]` — indexed `[reg_index][lane]`
- `agpr[256][64]` — accumulator registers (gfx90a MFMA destinations)
- `exec` (uint64_t) — active lane bitmask
- `pc` — **byte offset** into `kernel_code_ptr` (not a pointer)
- `lds_ptr` / `lds_size` — per-workgroup LDS buffer; set by execution engine
- `scratch_ptr` / `scratch_size` / `scratch_per_lane` — per-wavefront scratch memory;
  set by execution engine (`scratch_size = private_segment_fixed_size * 64`)

`sizeof(WavefrontState)` ≈ 132 KB — **never allocate on the stack**. The execution engine uses `std::make_unique`.

### `WaveStatus` (`include/gcnsim/wavefront.h`)
Returned by `run_wavefront()`:
- `ENDED` — wavefront hit `s_endpgm` or ran out of code
- `BARRIER_HIT` — wavefront hit `s_barrier` and is paused until all waves in the
  workgroup reach the barrier

### `Instruction` (`include/gcnsim/isa_decoder.h`)
Decoded instruction. `size_bytes` is 4 or 8. The caller advances `wave.pc += inst.size_bytes`
**before** calling `execute_instruction`, so branch handlers add their offset to the already-advanced PC.

Key fields beyond `dst`/`src`/`opcode`:
- **VOP3 modifiers**: `neg` (3 bits), `abs_` (3 bits), `clamp` (1 bit), `omod` (2 bits)
- **SDWA fields**: `is_sdwa`, `sdwa_dst_sel`, `sdwa_src0_sel`, `sdwa_src1_sel`,
  `sdwa_src0_sext/neg/abs`, `sdwa_src1_sext/neg/abs`
- **DPP fields**: `is_dpp`, `dpp_ctrl`, `dpp_bound_ctrl`, `dpp_bank_mask`, `dpp_row_mask`,
  `dpp_src0_neg/abs`, `dpp_src1_neg/abs`
- **VOP3P fields**: `op_sel` (4 bits), `op_sel_hi` (3 bits), `neg_hi` (3 bits)
- **MUBUF fields**: `mubuf_offset`, `mubuf_offen`, `mubuf_idxen`, `mubuf_lds`

### `KernelDescriptor` (`include/gcnsim/kernel_descriptor.h`)
Exactly 64 bytes. Lives at the `.kd`-suffixed symbol in the ELF. The first instruction is at:
```
code_ptr = kd_ptr + kd.kernel_code_entry_byte_offset   // always 256 for gfx9
```

---

## ISA decoder rules

Detection order in `decode()` (most specific first):

| Check | Format |
|---|---|
| `w[31:23] == 0x17F` | SOPP |
| `w[31:23] == 0x17D` | SOP1 |
| `w[31:23] == 0x17E` | SOPC |
| `w[31:28] == 0xB` | SOPK |
| `w[31:26] == 0b110000` | SMEM (8 bytes) |
| `w[31:26] == 0b110110` | DS (8 bytes) |
| `w[31:26] == 0b110111` | FLAT/GLOBAL/SCRATCH (8 bytes) |
| `w[31:26] == 0b111000` | MUBUF (8 bytes) |
| `w[31:26] == 0b111010` | MTBUF (8 bytes) |
| `w[31:26] == 0b110011` | VOP3P (8 bytes) |
| `w[31:26] == 0b110100` | VOP3A/VOP3B/MFMA (8 bytes) |
| `w[31:25] == 0b0111111` | VOP1 |
| `w[31:25] == 0b0111110` | VOPC |
| `w[31:30] == 0b10` | SOP2 |
| `w[31] == 0` | VOP2 |

**Critical encoding values** (verified against `llvm-objdump` disassembly):
- SOP1: `bits[31:23] = 0x17D` (top byte `0xBE`, bit 23 set)
- SOPC: `bits[31:23] = 0x17E` (top byte `0xBF`, bit 23 clear)
- SOPP: `bits[31:23] = 0x17F` (top byte `0xBF`, bit 23 set)

**FLAT/GLOBAL/SCRATCH word 0 layout** (bits[31:26] = `110111`):
- bits[25:18]: OP (8-bit) — same value for flat and global variants of the same operation
- bits[15:14]: SEG — `00`=flat, `01`=scratch, `10`=global (**not** bits[17:16])
- Example: `flat_load_dword` = `0xDC50_0000` (SEG=00), `global_load_dword` = `0xDC50_8000` (SEG=10)

**MFMA word 0 bit 15** — "acc" flag distinguishing accumulator register file:
- `0`: accumulator is VGPR (HIP compiler path; `vdst` = VGPR index, e.g. `v[2:5]`)
- `1`: accumulator is AGPR (assembly path; `vdst` = AGPR index, e.g. `a[0:3]`)
- Example: `D3C2_0002` = VGPR acc (bit15=0), `D3C2_8000` = AGPR acc (bit15=1)

**VOP3A opcode ranges for VOP2 re-encoding** (both must be handled):
- `0x100`–`0x13F`: VOP2 op N encoded as VOP3 op `0x100+N` (primary GFX9 encoding)
- `0x140`–`0x17F`: VOP2 op N encoded as VOP3 op `0x140+N` (alternate encoding)
- `0x180`–`0x1BF`: VOP1 re-encoding
- `0x3D8`: V_ACCVGPR_READ_B32 (AGPR → VGPR)
- `0x3D9`: V_ACCVGPR_WRITE_B32 (VGPR/const → AGPR)

**Always verify opcode numbers against actual disassembly**, not only the ISA PDF.
Use `llvm-objdump -d <kernel>.hsaco` and cross-check the hex encoding. The PDF
table ordering does not always match the binary op field value.

When adding a new instruction:
1. Identify its format from the table above.
2. Add the opcode to the `Opcode` enum in `isa_decoder.h`.
3. Add a case to the relevant format switch in `isa_decoder.cpp`.
4. Add an execution handler in `isa_executor.cpp` under `execute_instruction()`.
5. Add a test case in `tests/test_isa_decoder.cpp` using a **real encoding byte
   sequence taken from `llvm-objdump` output**, not a hand-constructed value.

**VOP3 modifier bit layout** (GFX9 VOP3A, verified against AMD Vega ISA Reference):
```
Word 0: [31:26]=110100  [25:16]=OP  [15]=CLAMP  [10:8]=ABS[2:0]  [7:0]=VDST
Word 1: [31:29]=NEG[2:0]  [28:27]=OMOD[1:0]  [26:18]=SRC2  [17:9]=SRC1  [8:0]=SRC0
```
- CLAMP bit 15 conflicts with MFMA "acc" flag — decoder only extracts CLAMP for non-MFMA VOP3
- Source modifier application order: `ABS` first, then `NEG`
- Destination modifier application order: `OMOD` first, then `CLAMP`
- `apply_src_mod(float val, bool neg, bool abs_)` and `apply_dst_mod(float val, bool clamp, uint8_t omod)`
  are static helpers in `isa_executor.cpp`

**SDWA sub-encoding** (triggered by VOP1/VOP2/VOPC with src0 raw field == 0xF9):
```
Word 1: [7:0]=SRC0_VGPR  [10:8]=SRC0_SEL  [11]=SRC0_SEXT  [12]=SRC0_NEG
         [13]=SRC0_ABS  [18:16]=SRC1_SEL  [19]=SRC1_SEXT  [20]=SRC1_NEG
         [21]=SRC1_ABS  [26:24]=DST_SEL  [28:27]=DST_UNUSED  [29]=CLAMP
```
- Sel values: 0=BYTE_0, 1=BYTE_1, 2=BYTE_2, 3=BYTE_3, 4=WORD_0, 5=WORD_1, 6=DWORD
- `read_vsrc_sdwa()` / `write_vdst_sdwa()` / `apply_sdwa_sel()` in `isa_executor.cpp`

**DPP sub-encoding** (triggered by VOP1/VOP2/VOPC with src0 raw field == 0xFA):
```
Word 1: [8:0]=DPP_CTRL  [19]=BOUND_CTRL  [20]=SRC0_NEG  [21]=SRC0_ABS
         [22]=SRC1_NEG  [23]=SRC1_ABS  [27:24]=BANK_MASK  [31:28]=ROW_MASK
         [7:0]=SRC0_VGPR
```
- Key DPP_CTRL patterns: `row_shl:N` (0x01-0x0F), `row_shr:N` (0x11-0x1F),
  `row_ror:N` (0x21-0x2F), quad permute (0x00-0xFF in quad range),
  `row_bcast:15` (0x142), `row_bcast:31` (0x143)
- `dpp_src_lane()` / `read_vsrc_dpp()` in `isa_executor.cpp`

GCN source operand encoding (9-bit field):
- 0–103 → SGPR
- 106 → VCC_LO, 107 → VCC_HI
- 126 → EXEC_LO, 127 → EXEC_HI
- 128–192 → inline constants 0–64
- 193–208 → inline constants −1 to −16
- 240–247 → float constants (0.5, −0.5, 1.0, −1.0, 2.0, −2.0, 4.0, −4.0)
- 255 → literal constant (next 4 bytes; instruction becomes 8 bytes)
- 256–511 → VGPR 0–255

---

## Verified GFX9 opcode tables

These were confirmed against `llvm-objdump` disassembly of real kernels. Use them
as ground truth when adding or debugging instructions.

### SOP2 (bits[31:30] = `10`, op = bits[29:23])

| op | Mnemonic |
|---|---|
| 0 | S_ADD_U32 |
| 1 | S_SUB_U32 |
| 2 | S_ADD_I32 |
| 3 | S_SUB_I32 |
| 4 | S_ADDC_U32 |
| 5 | S_SUBB_U32 |
| 6 | S_MIN_I32 |
| 7 | S_MIN_U32 |
| 8 | S_MAX_I32 |
| 9 | S_MAX_U32 |
| 10 | S_CSELECT_B32 |
| 11 | S_CSELECT_B64 |
| 12 | S_AND_B32 |
| 13 | S_AND_B64 |
| 14 | S_OR_B32 |
| 15 | S_OR_B64 |
| 16 | S_XOR_B32 |
| 17 | S_XOR_B64 |
| 18 | S_ANDN2_B32 |
| 19 | S_ANDN2_B64 |
| 20 | S_ORN2_B32 |
| 21 | S_ORN2_B64 |
| 22 | S_NAND_B32 |
| 23 | S_NAND_B64 |
| 24 | S_NOR_B32 |
| 25 | S_NOR_B64 |
| 26 | S_XNOR_B32 |
| 27 | S_XNOR_B64 |
| 28 | S_LSHL_B32 ← **confirmed: enc `8E0B8206`** |
| 29 | S_LSHL_B64 |
| 30 | S_LSHR_B32 |
| 31 | S_LSHR_B64 |
| 32 | S_ASHR_I32 |
| 33 | S_ASHR_I64 |
| 34 | S_BFM_B32 |
| 35 | S_BFM_B64 |
| 36 | S_MUL_I32 ← **confirmed: enc `92060006`** |
| 37 | S_BFE_U32 |
| 38 | S_BFE_I32 |
| 39 | S_BFE_U64 |
| 40 | S_BFE_I64 |
| 42 | S_ABSDIFF_I32 |

### VOP2 (bits[31] = 0, op = bits[30:25])

| op | Mnemonic |
|---|---|
| 0 | V_CNDMASK_B32 |
| 1 | V_ADD_F32 |
| 5 | V_MUL_F32 |
| 16 | V_LSHRREV_B32 ← **confirmed: enc `20020082`** |
| 17 | V_ASHRREV_I32 |
| 18 | V_LSHLREV_B32 |
| 19 | V_AND_B32 |
| 20 | V_OR_B32 |
| 21 | V_XOR_B32 |
| 23 | V_MAC_F32 |
| 25 | V_ADD_CO_U32 ← **confirmed: enc `32040002`** |
| 26 | V_SUB_U32 |
| 27 | V_SUBREV_U32 |
| 28 | V_ADDC_CO_U32 ← **confirmed: enc `38060303`** |
| 32 | V_ADD_U32 (no carry, gfx9+) |
| 52 | V_ADD_U32 (alt, enc `68000006`) |

---

## Execution model

- **All vector instructions** must iterate `lane` in `[0, 64)` and skip when `(wave.exec >> lane) & 1 == 0`.
- **EXEC mask changes** (`s_and_saveexec_b64`, `s_or_saveexec_b64`) must also update `wave.scc`
  based on whether the new EXEC is non-zero.
- **`s_waitcnt`** is a no-op for functional execution (simulation memory is synchronous),
  but when cycle counting is enabled, it models stalls by draining the LGKM/VM pending
  queues and advancing `cycle_count`.
- **Branch offset** (SOPP `simm16`): value is in DWORD units. Since `wave.pc` is already advanced
  past the branch instruction before `execute_instruction` runs, apply as:
  `wave.pc += (int64_t)inst.simm16 * 4;`
- **SMEM SBASE**: the 7-bit field encodes `actual_sgpr / 2`. Always multiply by 2 to get the base SGPR index.
- **VOP3B** vs **VOP3A**: distinguished by bit 7 of word0. VOP3B has a second SGPR destination (`dst2`) for carry-out.
- **`LaunchConfig.grid_x/y/z`** are **total thread counts**, not workgroup counts.
  The execution engine divides by `block_x/y/z` to get the workgroup grid dimensions.
- **Barrier synchronization**: `run_wavefront()` returns `WaveStatus::BARRIER_HIT` when
  hitting `s_barrier`, and `WaveStatus::ENDED` on `s_endpgm` or out-of-bounds. The
  execution engine runs all wavefronts in a workgroup round-robin: each wave runs
  until barrier or end, and when all active waves have hit the barrier, they all resume.
- **Scratch memory**: per-wavefront scratch buffer is allocated by the execution engine
  (size = `private_segment_fixed_size * 64`). FLAT_SCRATCH instructions use
  `scratch_ptr + lane * scratch_per_lane + offset` for per-lane private memory.
- **VOP3 source modifiers** (NEG, ABS) and **destination modifiers** (CLAMP, OMOD) are
  decoded and applied to all VOP3 float instructions. Applied order: ABS then NEG for
  sources; OMOD then CLAMP for destinations.
- **SDWA/DPP**: VOP1/VOP2/VOPC instructions with `src0 == 0xF9` (SDWA) or `src0 == 0xFA`
  (DPP) are decoded as 8-byte instructions with the sub-encoding fields extracted from
  word1. SDWA applies byte/word selection; DPP applies cross-lane data movement.
- **MFMA ignores EXEC mask** — all 64 lanes always participate in matrix operations.
- **MUBUF LDS bit** (word0 bit 16): when set, `buffer_load_dword` writes loaded data
  directly to LDS at `wave.m0 + lane * element_size` instead of to VGPR.

---

## Memory model

All pointers in simulation are host `malloc` addresses. Use the helpers in `memory_model.h`:

```cpp
uint32_t v = mem_load_u32(addr);
mem_store_f32(addr, result);
```

Do **not** use raw pointer casts (`*(uint32_t*)addr`) — this violates strict aliasing.

---

## Cycle simulation

The simulator has an optional per-wavefront cycle-counting mode, enabled at runtime
via `LaunchConfig::enable_cycle_counting = true`. When disabled, zero overhead.

### Enabling

```cpp
LaunchConfig cfg;
cfg.enable_cycle_counting = true;      // turn on cycle model
cfg.global_mem_latency = 300;          // HBM miss latency (default 300); L1/L2 hits use lower values
cfg.smem_latency = 20;                // SMEM (scalar load) latency (default 20)
cfg.trace_path = "trace.json";        // Chrome Trace output (optional, nullptr = off)

uint64_t total_cycles = 0;
launch_kernel(kernel, cfg, err, &total_cycles);
```

Public API: `launch_kernel(fn, grid, block, params, err, enable_cycle_counting,
total_cycles_ptr, global_mem_latency, trace_path)`.

### Latency model (gfx90a / CDNA2)

| Instruction class | Issue cycles | Completion latency | Counter |
|---|---|---|---|
| VALU (VOP1/2/3/3P/VOPC) | 4 | immediate | — |
| SALU (SOP1/2/K/C/P/SOPP) | 4 | immediate | — |
| s_nop N | 4*(N+1) | immediate | — |
| MFMA | 4 (issue) | 8–64 (variant-dependent, on miSIMD) | — |
| SMEM (s_load_*) | 4 | +`smem_latency` (default 20, configurable) | LGKM |
| DS (LDS read/write) | 4 + 4*(k-1) per transaction | +20 (read-to-use) | LGKM |
| MUBUF/MTBUF | 4 | L1 hit: 100, L2 hit: 200, HBM miss: `global_mem_latency` | VM |
| FLAT/GLOBAL/SCRATCH | 4 | L1 hit: 100, L2 hit: 200, HBM miss: `global_mem_latency` | VM |

**s_nop N** inserts N+1 issue slots of pipeline delay (e.g., for MFMA result hazards).
Each issue slot = 4 wavefront cycles, so `s_nop 3` = 16 cycles.

### DS bank-conflict model

gfx90a LDS has **32 banks**, each 4 bytes wide. A 64-lane wavefront is split
into transactions based on data width:

| DS variant | Dwords/lane | Transactions | Lanes/txn |
|---|---|---|---|
| `ds_*_b32` | 1 | 2 | 32 |
| `ds_*_b64` | 2 | 4 | 16 |
| `ds_*_b128` | 4 | 8 | 8 |

Bank conflicts are checked **per transaction group** (not whole wavefront).
Each transaction costs `4 + 4*(k-1)` cycles where k = max lanes hitting the
same bank within that group.

Source: [AMD CK LDS bank conflict docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html)

### L1/L2 cache simulation

When cycle counting is enabled, VMEM instructions (MUBUF, FLAT, GLOBAL) probe a
two-level set-associative cache model before determining completion latency:

| Cache level | Size | Associativity | Line size | Hit latency |
|---|---|---|---|---|
| vL1D | 16 KB | 4-way | 64 B | 100 cycles |
| L2 | 8 MB | 16-way | 128 B | 200 cycles |
| HBM (miss) | — | — | — | `global_mem_latency` (default 300) |

- LRU eviction policy at both levels
- Cache is created once per kernel launch and shared across all wavefronts
- Stats tracked: `l1_hits`, `l1_misses`, `l2_hits`, `l2_misses`
- Implementation: `include/gcnsim/cache_model.h` + `src/cache_model.cpp`

### VMEM address coalescing

Before probing the cache, 64 lanes' addresses are coalesced into unique L1 cache
lines (64B aligned). A fully coalesced 256B wavefront access generates only 4 cache
line probes instead of 64. The worst-case latency across all unique lines determines
the VMEM completion timestamp. This accurately models bandwidth-bound vs
latency-bound access patterns.

### Multi-pipe issue overlap

Real GFX9 CUs have separate issue pipes that can operate in parallel. The cycle
model tracks per-pipe ready timestamps:

| Pipe | Instruction types |
|---|---|
| SALU | SOP1/2/K/C/P, s_waitcnt, s_nop |
| VALU | VOP1/2/3/3P/VOPC, MFMA |
| VMEM | MUBUF/MTBUF/FLAT/GLOBAL/SCRATCH |
| LDS | DS |
| SMEM | s_load/s_buffer_load |

Each instruction starts at `max(cycle_count, pipe_ready)` and blocks only its own
pipe for `issue_cycles`. Other pipes can issue in the same cycle window. For example,
an SALU instruction issued while the VALU pipe is busy does not stall.

### s_waitcnt stall model

In-flight memory ops push completion timestamps onto per-wavefront queues:
- `lgkm_pending` — DS and SMEM operations
- `vm_pending` — MUBUF, MTBUF, FLAT, GLOBAL, SCRATCH operations

When `s_waitcnt lgkmcnt(N)` / `vmcnt(N)` executes, the queue is sorted and
drained until `size() <= N`, advancing `cycle_count` to `max(cycle_count,
completion_time)` for each drained entry. This models the stall.

### MFMA / VALU co-execution

MFMA executes on the **miSIMD** (matrix unit), separate from the SIMD used by VALU.
They co-execute but do **not** co-issue (each takes 4 cycles to issue).

After an MFMA issues, a **shadow window** of `exec_cycles - 4` opens. Non-MFMA
instructions issued during this window consume shadow instead of advancing
`cycle_count`. When shadow is exhausted, normal costing resumes. A second MFMA
stalls until `mfma_busy_until`.

**MFMA execution latencies** (from [AMD matrix_calculator.py](https://github.com/ROCm/amd_matrix_instruction_calculator)):

| Matrix size | Exec cycles | Opcodes |
|---|---|---|
| 4×4 (F32/F16/I8/BF16) | 8 | `*_4x4x1f32`, `*_4x4x4f16`, `*_4x4x4i8`, `*_4x4x2bf16`, `*_4x4x4bf16_1k` |
| 4×4 (F64) | 16 | `*_f64_4x4x4f64` |
| 16×16 (short K) | 32 | `*_16x16x1f32`, `*_16x16x4f32`, `*_16x16x4f16`, `*_16x16x4i8`, `*_16x16x2bf16`, `*_16x16x4bf16_1k`, `*_f64_16x16x4f64` |
| 16×16 (long K) | 16 | `*_16x16x16f16`, `*_16x16x16i8`, `*_16x16x8bf16`, `*_16x16x16bf16_1k` |
| 32×32 (short K) | 64 | `*_32x32x1f32`, `*_32x32x2f32`, `*_32x32x4f16`, `*_32x32x4i8`, `*_32x32x2bf16`, `*_32x32x4bf16_1k` |
| 32×32 (long K) | 32 | `*_32x32x8f16`, `*_32x32x8i8`, `*_32x32x4bf16`, `*_32x32x8bf16_1k` |

### WavefrontState cycle fields

```
count_cycles            bool      — enable flag (set from LaunchConfig)
cycle_count             uint64_t  — accumulated simulated cycles
global_mem_latency      uint32_t  — configurable HBM miss latency (default 300)
smem_latency            uint32_t  — configurable SMEM latency (default 20)
mfma_shadow_remaining   uint64_t  — remaining shadow from in-flight MFMA
mfma_busy_until         uint64_t  — cycle when miSIMD becomes free
mfma_total_exec_cycles  uint64_t  — sum of MFMA exec cycles (for utilization)
valu_total_issue_cycles uint64_t  — sum of VALU issue cycles (for utilization)
pipe_salu_ready         uint64_t  — SALU pipe ready timestamp
pipe_valu_ready         uint64_t  — VALU pipe ready timestamp
pipe_vmem_ready         uint64_t  — VMEM pipe ready timestamp
pipe_lds_ready          uint64_t  — LDS pipe ready timestamp
pipe_smem_ready         uint64_t  — SMEM pipe ready timestamp
cache                   CacheModel* — L1/L2 cache model (nullptr = flat latency)
trace_file              FILE*     — Chrome Trace output file (nullptr = off)
trace_wave_id           uint32_t  — wave index within workgroup
trace_wg_flat_id        uint32_t  — flat workgroup index for trace pid
lgkm_pending            vector    — DS/SMEM completion timestamps
vm_pending              vector    — MUBUF/FLAT/GLOBAL completion timestamps
```

All cycle fields are placed before `lgkm_pending` in the struct so the existing
`memset(..., offsetof(WavefrontState, lgkm_pending))` in `init_wavefront()` and
`make_wave()` zeroes them. **Do not place non-POD fields before `lgkm_pending`.**

### Chrome Trace timeline output

When `trace_path` is set and cycle counting is enabled, the simulator writes a
[Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
JSON file. Open in [Perfetto UI](https://ui.perfetto.dev) or `chrome://tracing`.

Each wavefront gets its own row (pid=workgroup, tid=wave). The row label shows
utilization: `wave0 (VALU 21.9% MFMA 55.8%)`.

**Event categories and colors:**

| Category | Instructions | Color in Perfetto |
|---|---|---|
| MFMA | `v_mfma_*` | purple |
| VALU | VOP1/2/3/3P/VOPC | blue |
| SALU | SOP1/2/K/C/P | green |
| DS | `ds_*` | orange |
| SMEM | `s_load_*` | cyan |
| VMEM | MUBUF/FLAT/GLOBAL/SCRATCH | red |
| STALL | waitcnt stalls, MFMA busy stalls | gray |

**Stall events** are emitted explicitly for:
- `stall:waitcnt` — `s_waitcnt` draining pending queues (dur = cycles stalled)
- `stall:mfma_busy` — MFMA issued while miSIMD still busy (dur = cycles waited)

**Utilization metrics** (per-wavefront):
- **MFMA utilization** = `mfma_total_exec_cycles / cycle_count` — fraction of time
  miSIMD is doing useful work
- **VALU utilization** = `valu_total_issue_cycles / cycle_count` — fraction of time
  SIMD issue port is occupied by pure VALU (excludes MFMA issue cycles)

### Cycle simulation implementation files

All cycle-counting logic lives in these files:

| File | What |
|---|---|
| `src/isa_executor.cpp` | `ds_bank_conflict_k()`, `ds_issue_cycles()`, `issue_cycles()`, `pipe_for()`, `push_pending()`, `push_vmem_with_cache()`, `handle_waitcnt_cycles()`, `mfma_exec_cycles()`, `mubuf_addr()`, trace helpers, cycle-accounting block in `run_wavefront()` |
| `src/cache_model.cpp` | `CacheSet`, `CacheLevel`, `CacheModel` — set-associative L1/L2 cache with LRU eviction |
| `src/execution_engine.cpp` | Creates `CacheModel`, opens/closes trace file, sets cycle fields on wavefronts, accumulates `total_cycles`, writes utilization metadata |
| `src/gcnsim_api.cpp` | Forwards `enable_cycle_counting`, `total_cycles`, `global_mem_latency`, `smem_latency`, `trace_path` |

---

## Adding a new test kernel

### HIP C++ kernel

1. Write the kernel in `tools/<name>/` directory.
2. Add a `CMakeLists.txt` that builds the `.hsaco` using `AMDGPU_CLANG` (discovered
   in the root CMakeLists.txt):
   ```cmake
   if(NOT AMDGPU_CLANG)
       message(WARNING "clang with amdgcn support not found; skipping <name>")
       return()
   endif()
   set(HSACO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/<name>.hsaco)
   add_custom_command(
       OUTPUT  ${HSACO_OUTPUT}
       COMMAND ${AMDGPU_CLANG} -x hip --cuda-device-only --offload-arch=gfx90a
               --no-gpu-bundle-output -O2 -c
               -o ${HSACO_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/<name>.hip
       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/<name>.hip
       COMMENT "Compiling <name>.hip → <name>.hsaco"
   )
   add_custom_target(<name>_hsaco ALL DEPENDS ${HSACO_OUTPUT})
   ```
3. Add `add_subdirectory(tools/<name>)` to the root `CMakeLists.txt`.
4. Write an integration test in `tests/` that uses `module_load` → `module_get_function` → `launch_kernel`.
5. Pass the `.hsaco` path as a command-line argument (`--hsaco <path>`) and register it
   in `CMakeLists.txt` via `add_test(... --hsaco ...)`.

### Hand-written assembly kernel

A clang with amdgcn target support (ROCm clang at `/opt/rocm/llvm/bin/clang`, or
upstream LLVM built with AMDGPU) can assemble GFX90A `.s` files. The CMake variable
`AMDGPU_CLANG` is discovered in the root CMakeLists.txt. The build requires two
steps (assemble then link):

```bash
clang -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 \
      -mcpu=gfx90a:xnack- -mwavefrontsize64 -c -o kernel.o kernel.s
clang -target amdgcn-amd-amdhsa -mcpu=gfx90a:xnack- kernel.o -o kernel.co
```

See `tools/asm_gemm_mfma/` for a complete example using `.amdhsa_kernel`,
`.amdgpu_metadata` (COV4 YAML), and the corresponding `CMakeLists.txt`.

Key assembly constraints (GFX9 constant bus rule):
- A VALU instruction may read at most **one** SGPR operand.
- When you need two SGPR values in a VALU op, copy one to a VGPR first with `v_mov_b32`.
- `v_addc_co_u32` carry-in/out and src1 cannot both be SGPRs; use a VGPR copy for src1.
- GFX9 does **not** support `.amdhsa_wavefront_size32` (that flag is GFX10+ only).

---

## Test suites

| Suite | Tests | What it covers |
|---|---|---|
| `test_elf_loader` | 6 | ELF parsing, symbol table, section extraction |
| `test_isa_decoder` | 41 | Instruction decoding for all formats (SOPP, SOP1/2, SOPK, SOPC, SMEM, VOP1/2/3, VOPC, DS, MUBUF, VOP3P, MFMA, FLAT/GLOBAL, SDWA, DPP) |
| `test_isa_executor` | 16 | Instruction execution: `v_cndmask_b32`, `s_cselect_b32`, VOP3 modifiers (NEG/ABS/CLAMP/OMOD), `s_getpc_b64`, `s_barrier` (WaveStatus), scratch memory store/load/isolation, `v_mul_f32` baseline |
| `test_wavefront` | 9 | Wavefront initialization: EXEC masks, workitem IDs, SGPR setup, VGPR0-2 for workitem_id.x/y/z |
| `test_vector_add` | 2 | End-to-end vector-add kernel (HSACO integration) |
| `test_gemm_mfma` | 3 | MFMA 4x4x1 GEMM integration |
| `test_mfma_kernels` | 4 | Multi-shape MFMA integration (16x16x1, 16x16x4, 32x32x1, 32x32x2) |
| `test_asm_gemm` | 2 | Hand-written assembly MFMA GEMM integration |
| `test_tensile_gemm` | 3 | Tensile-style 128x256 SGEMM with LDS, barriers, multi-wavefront |
| `test_cycle_model` | 8 | MFMA shadow absorption, back-to-back MFMA stalls, s_waitcnt stalls, DS bank conflict setup, VALU/MFMA utilization tracking, s_nop delay |
| `test_cache_model` | 5 | L1 hit on reuse, same-line hit, L2 hit after L1 eviction, sequential reuse, stats accumulation |

### Writing new tests

**Decoder tests** (`test_isa_decoder.cpp`): use `enc2(w0, w1)` to build byte sequences,
call `decode()`, and verify format/opcode/operands. Use real encodings from
`llvm-objdump` where possible; hand-constructed bit patterns are acceptable for new
sub-encodings (SDWA, DPP).

**Executor tests** (`test_isa_executor.cpp`): forward-declare `execute_instruction`
and `run_wavefront` from the `gcnsim` namespace, build an `Instruction` struct manually
(set opcode, format, operands, modifiers), allocate `WavefrontState` on the heap via
`make_wave()`, and verify VGPR/SGPR outputs. `WavefrontState` is ~132 KB — never put
it on the stack. See `make_inst()` and `f2u()`/`u2f()` helpers in the test file.

**Wavefront tests** (`test_wavefront.cpp`): call `init_wavefront()` with a
`KernelDescriptor` built via `make_kd(kcp, rsrc2)` and verify initial register state.

**Integration tests**: load a real HSACO via `module_load`, find the kernel via
`module_get_function`, execute via `launch_kernel`, and compare output against a CPU
reference.

---

## Debugging decoder bugs

When a kernel produces wrong results or skips instructions unexpectedly:

1. **Disassemble the kernel** with `llvm-objdump -d <kernel>.hsaco` and note the hex encoding
   of the failing instruction.
2. **Extract the op field** in Python:
   ```python
   w = 0xXXXXXXXX
   print((w >> 23) & 0x7F)   # SOP1/SOPC/SOP2 op
   print((w >> 25) & 0x3F)   # VOP2 op
   ```
3. **Cross-check** the extracted op against the verified opcode tables above.
4. **Add a temporary trace** in `run_wavefront()` only while debugging — remove it before committing:
   ```cpp
   fprintf(stderr, "PC=%#llx enc=%08X op=%d fmt=%d\n",
           pre_pc, inst.encoding[0], (int)inst.opcode, (int)inst.format);
   ```
5. Check whether a branch instruction is taken unexpectedly (wrong SCC/EXEC state due to a
   misidentified comparison instruction).

Common past mistakes to watch for:
- SOP2 AND/OR/XOR group: B32 and B64 variants interleave (B32=even, B64=odd op numbers).
- SOP2 shift/logic group starts at op 28, not op 18.
- SOPC encoding is `0x17E`, not `0x17C` or `0x17D`.
- Test byte sequences must come from real `llvm-objdump` output, not hand-crafted values.
- **FLAT/GLOBAL SEG field is at bits[15:14]**, not bits[17:16]. Bits[17:16] are always
  zero on GFX9/GFX90a and cannot distinguish flat from global. Correct extraction:
  `seg = (w0 >> 14) & 0x3`  (0=flat, 1=scratch, 2=global).
  Both flat and global share the same 8-bit OP value (e.g. 0x14 = load_dword); only
  the SEG field disambiguates them.
- **MFMA accumulator kind**: bit 15 of MFMA word 0 is the "acc" flag.
  0 = VGPR accumulator (HIP compiler path — vdst encodes a physical VGPR index).
  1 = AGPR accumulator (hand-written assembly path — vdst encodes an AGPR index).
  Getting this wrong silently overwrites general-purpose VGPRs (e.g. v0 = lane id,
  v1 = row index) with MFMA results, corrupting subsequent address computations.
- **VOP3→VOP2 opcode range**: GFX9 encodes VOP2 op N as VOP3 op `0x100+N` *and*
  as `0x140+N`. Both ranges must be handled. Missing the 0x100–0x13F range causes
  `v_add_u32_e64` (op 0x134) to decode as UNKNOWN.
- **SDWA src0 field** (word1 bits[7:0]) is always a **VGPR index** (0-255), NOT a
  standard 9-bit source encoding. Do NOT pass it to `decode_src()` — that maps
  indices 0-103 to SGPRs. Set `inst.src[0].kind = Operand::Kind::VGPR` directly,
  same as the DPP path does.

---

## Known limitations (do not work around silently — fix or document)

- **`s_waitcnt`** is a no-op for functional execution (all simulation memory is
  synchronous). When cycle counting is enabled, it models stalls via the LGKM/VM
  pending queues.
- **Divergent loops** with mid-loop EXEC changes across iterations are not guaranteed correct
  (re-convergence stack is not implemented).
- The instruction safety limit in `run_wavefront()` is 1,000,000 iterations; infinite-loop kernels
  will be silently cut off.
- **MIMG** (image sampling) instructions are not decoded — they require texture hardware state.
  All MIMG instructions silently skip.
- **F16/BF16 precision**: F16↔F32 and BF16→F32 conversions use software helpers in
  `isa_executor.cpp` (`f16_to_f32`, `f32_to_f16`, `bf16_to_f32`). These may differ from
  hardware rounding behavior in edge cases (denormals, NaN payloads).
- **MFMA lane mapping**: all MFMA shapes (F32, F16, BF16, I8, F64) now use CDNA2
  cross-lane outer-product semantics. F16/BF16/I8 results are functionally correct
  but may have minor rounding differences vs. hardware due to FMA accumulation order.
- **Cycle model is approximate**: the cycle simulation models L1/L2 cache hit/miss
  with set-associative caches and VMEM coalescing, but does not model instruction
  fetch latency or multi-wavefront CU occupancy effects. The DS bank-conflict model
  uses per-transaction grouping (confirmed by AMD docs) with a fixed
  4-cycle-per-transaction base cost.
- **MUBUF OOB protection**: out-of-bounds MUBUF accesses (e.g., kernel prefetch past
  array end) are detected via `msync()` probing and return 0, matching hardware
  behavior. This replaces the previous segfault on large-K Tensile GEMM kernels.

---

## Future work

### ISA / functional simulation

- **Re-convergence stack**: properly handle divergent loops where EXEC changes mid-loop,
  allowing accurate simulation of branchy kernels.
- **s_sub_i32 signed semantics** (SOP2 op 3): currently aliased to `S_SUB_U32`.
- **MIMG instructions**: require texture descriptor and sampler state; currently silently skipped.
- **DPP16/DPP8** (GFX10+ only): not relevant for gfx90a but would be needed for newer targets.
- **Multi-tile Tensile GEMM**: the single-tile (128x256x32) Tensile SGEMM passes end-to-end;
  extending to multi-tile problems would exercise inter-workgroup correctness.
- **DS read2/write2 bank conflicts**: the bank-conflict model currently only checks
  `ds_offset0` for single-address DS ops. `ds_read2_b32`/`ds_write2_b32` access two
  addresses per lane (via both `ds_offset0` and `ds_offset1`) and need both checked.

### Cycle model — remaining improvements

- **Multi-wavefront CU occupancy**: the cycle model runs each wavefront in isolation —
  all stall cycles are "paid" fully. Real hardware interleaves 4-8 wavefronts per SIMD;
  when wave 0 stalls on memory, wave 1 runs. Simulate CU-level round-robin scheduling
  in `execution_engine.cpp`: advance each wave by one instruction, track a shared CU
  cycle counter, interleave waves on stalls. Shows how occupancy hides latency.
- **Instruction fetch latency**: model the instruction cache (sL1I). Rare misses for
  hot loops but could matter for large kernels with poor i-cache locality.

---

## Commit style

- No co-author lines.
- Subject line: imperative mood, ≤72 chars.
- Body: bullet list of what changed and why, wrapped at 72 chars.
- Do not commit build artifacts (`build/`, `*.hsaco`, `*.o`, `*.so`).
