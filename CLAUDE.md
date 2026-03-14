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

# Build everything (library + tests + vector_add.hsaco kernel)
cmake --build build -j$(nproc)

# Run all tests
ctest --test-dir build -V

# Run a single test binary directly
./build/tests/test_isa_decoder
./build/tests/test_vector_add
```

**Expected:** 34 tests pass across 4 suites.
Always run the full test suite after any change to `src/` or `include/`.

---

## Project layout

```
include/gcnsim/   Public headers — data structures and API declarations
src/              Implementation — one .cpp per concern
tests/            GTest unit and integration tests
tools/vector_add/ Reference HIP kernel compiled to .hsaco for integration tests
```

| File | Responsibility |
|---|---|
| `elf_loader.cpp` | Parse HSACO ELF: sections, symbol table, kernel descriptors |
| `metadata_parser.cpp` | Decode MsgPack-encoded kernel metadata from ELF .note section |
| `isa_decoder.cpp` | Decode raw bytes → `Instruction` struct |
| `isa_executor.cpp` | Execute one `Instruction` against a `WavefrontState` |
| `wavefront.cpp` | `init_wavefront()` + `run_wavefront()` execution loop |
| `execution_engine.cpp` | Grid dispatch, wavefront splitting, hidden arg injection |
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
- `exec` (uint64_t) — active lane bitmask
- `pc` — **byte offset** into `kernel_code_ptr` (not a pointer)

`sizeof(WavefrontState)` ≈ 66 KB — **never allocate on the stack**. The execution engine uses `new`/`delete`.

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
| `w[31:23] == 0x17E` | SOP1 |
| `w[31:23] == 0x17D` | SOPC |
| `w[31:26] == 0b110000` | SMEM (8 bytes) |
| `w[31:26] == 0b110111` | FLAT_GLOBAL (8 bytes) |
| `w[31:26] == 0b110100` | VOP3A/VOP3B (8 bytes) |
| `w[31:25] == 0b0111111` | VOP1 |
| `w[31:25] == 0b0111110` | VOPC |
| `w[31:30] == 0b10` | SOP2 |
| `w[31] == 0` | VOP2 |

When adding a new instruction:
1. Identify its format from the table above.
2. Add the opcode to the `Opcode` enum in `isa_decoder.h`.
3. Add a case to the relevant format switch in `isa_decoder.cpp`.
4. Add an execution handler in `isa_executor.cpp` under `execute_instruction()`.
5. Add a test case in `tests/test_isa_decoder.cpp` using a known encoding byte sequence.

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

1. Write the kernel in `tools/vector_add/` or create a new `tools/<name>/` directory.
2. Add a `CMakeLists.txt` that builds the `.hsaco` with `hipcc --offload-arch=gfx90a -c --genco`.
3. Add `add_subdirectory(tools/<name>)` to the root `CMakeLists.txt`.
4. Write an integration test in `tests/` that uses `module_load` → `module_get_function` → `launch_kernel`.

---

## Known limitations (do not work around silently — fix or document)

- **SOPK** instructions are not decoded or executed.
- **Flat memory** (SEG=0) is not implemented; only SEG=2 (GLOBAL) is supported.
- **LDS / scratch memory** (`ds_read_b32`, `ds_write_b32`) is not implemented.
- **`s_waitcnt`** is a no-op; there is no memory ordering model.
- **Divergent loops** with mid-loop EXEC changes across iterations are not guaranteed correct
  (re-convergence stack is not implemented).
- The instruction safety limit in `run_wavefront()` is 1,000,000 iterations; infinite-loop kernels
  will be silently cut off.

---

## Commit style

- No co-author lines.
- Subject line: imperative mood, ≤72 chars.
- Body: bullet list of what changed and why, wrapped at 72 chars.
- Do not commit build artifacts (`build/`, `*.hsaco`, `*.o`, `*.so`).
