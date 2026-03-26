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

**Expected:** 86 tests pass across 9 suites.
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
- **`s_waitcnt`** is a no-op — simulation memory is always synchronous.
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

- **`s_waitcnt`** is a no-op; there is no memory ordering model.
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

---

## Future work

- **Re-convergence stack**: properly handle divergent loops where EXEC changes mid-loop,
  allowing accurate simulation of branchy kernels.
- **s_sub_i32 signed semantics** (SOP2 op 3): currently aliased to `S_SUB_U32`.
- **Memory ordering model**: `s_waitcnt` is currently a no-op; a simple counter-based
  model would allow detection of missing barriers in test kernels.
- **MIMG instructions**: require texture descriptor and sampler state; currently silently skipped.
- **DPP16/DPP8** (GFX10+ only): not relevant for gfx90a but would be needed for newer targets.
- **Multi-tile Tensile GEMM**: the single-tile (128x256x32) Tensile SGEMM passes end-to-end;
  extending to multi-tile problems would exercise inter-workgroup correctness.

---

## Commit style

- No co-author lines.
- Subject line: imperative mood, ≤72 chars.
- Body: bullet list of what changed and why, wrapped at 72 chars.
- Do not commit build artifacts (`build/`, `*.hsaco`, `*.o`, `*.so`).
