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
./build/tests/test_vector_add
./build/tests/test_gemm_mfma --hsaco build/tools/gemm_mfma/gemm_mfma.hsaco
./build/tests/test_mfma_kernels --hsaco build/tools/mfma_kernels/mfma_kernels.hsaco
```

**Expected:** 51 tests pass across 6 suites.
Always run the full test suite after any change to `src/` or `include/`.

---

## Project layout

```
include/gcnsim/    Public headers — data structures and API declarations
src/               Implementation — one .cpp per concern
tests/             GTest unit and integration tests
tools/vector_add/  Vector-add HIP kernel (.hsaco) for integration tests
tools/gemm_mfma/   MFMA GEMM HIP kernel (.hsaco) for MFMA integration tests
```

| File | Responsibility |
|---|---|
| `elf_loader.cpp` | Parse HSACO ELF: sections, symbol table, kernel descriptors |
| `metadata_parser.cpp` | Decode MsgPack-encoded kernel metadata from ELF .note section |
| `isa_decoder.cpp` | Decode raw bytes → `Instruction` struct |
| `isa_executor.cpp` | Execute one `Instruction` against a `WavefrontState` |
| `wavefront.cpp` | `init_wavefront()` + `run_wavefront()` execution loop |
| `execution_engine.cpp` | Grid dispatch, wavefront splitting, LDS allocation, hidden arg injection |
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

`sizeof(WavefrontState)` ≈ 132 KB — **never allocate on the stack**. The execution engine uses `std::make_unique`.

### `Instruction` (`include/gcnsim/isa_decoder.h`)
Decoded instruction. `size_bytes` is 4 or 8. The caller advances `wave.pc += inst.size_bytes`
**before** calling `execute_instruction`, so branch handlers add their offset to the already-advanced PC.

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

1. Write the kernel in `tools/<name>/` directory.
2. Add a `CMakeLists.txt` that builds the `.hsaco` with `hipcc --offload-arch=gfx90a -c --genco`.
3. Add `add_subdirectory(tools/<name>)` to the root `CMakeLists.txt`.
4. Write an integration test in `tests/` that uses `module_load` → `module_get_function` → `launch_kernel`.
5. Pass the `.hsaco` path as a command-line argument (`--hsaco <path>`) and register it
   in `CMakeLists.txt` via `add_test(... --hsaco ...)`.

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

---

## Known limitations (do not work around silently — fix or document)

- **`s_waitcnt`** is a no-op; there is no memory ordering model.
- **Divergent loops** with mid-loop EXEC changes across iterations are not guaranteed correct
  (re-convergence stack is not implemented).
- The instruction safety limit in `run_wavefront()` is 1,000,000 iterations; infinite-loop kernels
  will be silently cut off.
- **MFMA F32 data layout**: all five F32 MFMA shapes (4×4×1, 16×16×1, 16×16×4,
  32×32×1, 32×32×2) implement the correct CDNA2 cross-lane outer-product semantics.
  Non-F32 shapes (F16, BF16, I8) still use a simplified per-lane approximation and
  are not bitwise-identical to hardware.
- **MIMG** (image sampling) instructions are not decoded — they require texture hardware state.
  All MIMG instructions silently skip.
- **SDWA / DPP** sub-encoding variants are not decoded; instructions using `src0 == 0xF9/0xFA`
  are treated as UNKNOWN and skipped.
- **Scratch segment** FLAT instructions (SEG=1) are treated as flat (VGPR-pair address) rather
  than using the scratch base + offset addressing.

---

## Future work

- **SDWA decode path**: if VOP1/VOP2/VOPC has `src0 == 0xF9`, word1 contains SDWA fields
  (sub-dword byte/word extraction). Required for kernels that manipulate packed data types.
- **DPP decode path**: if `src0 == 0xFA`, word1 contains DPP control for cross-lane data
  movement (`row_shl`, `row_shr`, `wave_shl/shr`, broadcast patterns). Required for
  warp-level reductions without LDS.
- **Exact MFMA lane layout for non-F32**: F16, BF16, I8 shapes still use per-lane
  approximation; implement full CDNA2 matrix-block lane tables for those shapes.
- **Re-convergence stack**: properly handle divergent loops where EXEC changes mid-loop,
  allowing accurate simulation of branchy kernels.
- **Scratch segment addressing**: route FLAT SEG=1 instructions through
  `flat_scratch_base + lane_offset` rather than treating them as VGPR-pair addresses.
- **S_CSELECT_B32/B64** (SOP2 op 10/11): not yet implemented.
- **s_sub_i32 signed semantics** (SOP2 op 3): currently aliased to `S_SUB_U32`.
- **Memory ordering model**: `s_waitcnt` is currently a no-op; a simple counter-based
  model would allow detection of missing barriers in test kernels.

---

## Commit style

- No co-author lines.
- Subject line: imperative mood, ≤72 chars.
- Body: bullet list of what changed and why, wrapped at 72 chars.
- Do not commit build artifacts (`build/`, `*.hsaco`, `*.o`, `*.so`).
