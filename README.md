# gcn-sim — MI-200 (gfx90a) CPU-Based Instruction-Level Kernel Simulator

A C++ library that loads HSACO (HIP/HSA Code Object) ELF files, decodes gfx90a ISA instructions, and simulates AMD GPU kernel execution entirely on the CPU — no GPU or GPU driver required.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [What Was Implemented](#what-was-implemented)
  - [Phase 1: ELF Loader](#phase-1-elf-loader)
  - [Phase 2: ISA Decoder](#phase-2-isa-decoder)
  - [Phase 3: Wavefront Executor](#phase-3-wavefront-executor)
  - [Phase 4: Public API & Execution Engine](#phase-4-public-api--execution-engine)
  - [Phase 5: Integration Test](#phase-5-integration-test)
- [Key Design Decisions](#key-design-decisions)
- [Building](#building)
- [Running Tests](#running-tests)
- [Usage Example](#usage-example)
- [Supported Instructions](#supported-instructions)
- [GCN ISA Reference Notes](#gcn-isa-reference-notes)
- [Potential Improvements](#potential-improvements)
- [Future Work](#future-work)

---

## Overview

`gcn-sim` is an instruction-level simulator for the AMD gfx90a (MI-200 / CDNA2) GPU ISA. It operates by:

1. Parsing a compiled HSACO ELF binary to locate kernel symbols and their 64-byte kernel descriptors
2. Decoding raw machine code instructions into structured `Instruction` objects
3. Maintaining a `WavefrontState` (SGPRs, VGPRs, EXEC mask, PC, etc.) per wave
4. Executing instructions one at a time against that state, using host memory for all data

The result is a GPU kernel running correctly on a CPU with zero GPU dependency. All memory operations (`global_load_dword`, `global_store_dword`, `s_load_dwordx4`, etc.) directly dereference host `malloc`-allocated pointers, making simulation transparent and debuggable.

**Target environment:** ROCm 7.0, gfx90a (MI-200 / CDNA2), wavefront size = 64.

---

## Architecture

```
                  ┌─────────────────────────────────────────────────────┐
                  │                   Public API (gcnsim.h)             │
                  │  module_load()  module_get_function()  launch_kernel()│
                  └────────────────────────┬────────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
     ┌────────▼────────┐        ┌──────────▼──────────┐      ┌─────────▼────────┐
     │   ELF Loader    │        │  Execution Engine   │      │  Metadata Parser │
     │  (elf_loader.h) │        │(execution_engine.h) │      │  (msgpack notes) │
     │                 │        │                     │      │                  │
     │ • ELF validation│        │ • Grid → workgroups │      │ • MsgPack reader │
     │ • Section parse │        │ • Wavefront dispatch│      │ • Arg offsets    │
     │ • Symbol lookup │        │ • Kernarg build     │      │ • value_kind     │
     │ • KD extraction │        │ • Hidden args fill  │      └──────────────────┘
     └─────────────────┘        └──────────┬──────────┘
                                           │
                          ┌────────────────▼───────────────┐
                          │           Wavefront             │
                          │        (wavefront.h/.cpp)       │
                          │                                 │
                          │ • SGPR/VGPR initialization      │
                          │ • EXEC mask setup               │
                          │ • Work-item ID layout           │
                          └────────────────┬───────────────┘
                                           │
                     ┌─────────────────────▼──────────────────────┐
                     │              ISA Decoder + Executor         │
                     │         (isa_decoder.h / isa_executor.cpp)  │
                     │                                             │
                     │  decode(pc) → Instruction                   │
                     │  execute_instruction(wave, inst)            │
                     │                                             │
                     │  Formats: SOP1/2, SOPK, SOPP, VOP1/2/3A/B, │
                     │           VOPC, SMEM, FLAT_GLOBAL           │
                     └─────────────────────────────────────────────┘
```

---

## Directory Structure

```
gcn-sim/
├── CMakeLists.txt                   # Root build file
├── PLAN.md                          # Original implementation plan
├── README.md                        # This file
│
├── include/gcnsim/
│   ├── gcnsim.h                     # Public C++ API
│   ├── elf_loader.h                 # HSACO ELF parser types
│   ├── kernel_descriptor.h          # 64-byte Code Object V3/V4 descriptor
│   ├── isa_decoder.h                # Instruction/Opcode/Operand types + decode()
│   ├── wavefront.h                  # WavefrontState struct + constants
│   ├── execution_engine.h           # LaunchConfig + internal launch_kernel()
│   └── memory_model.h               # Inline host-memory load/store helpers
│
├── src/
│   ├── elf_loader.cpp               # HSACO ELF parsing (~280 lines)
│   ├── metadata_parser.cpp          # MsgPack .note section parser (~315 lines)
│   ├── isa_decoder.cpp              # Instruction decoder (~680 lines)
│   ├── isa_executor.cpp             # Per-instruction handlers (~840 lines)
│   ├── wavefront.cpp                # Wavefront init + run_wavefront() (~120 lines)
│   ├── execution_engine.cpp         # Grid dispatch + kernarg build (~100 lines)
│   └── gcnsim_api.cpp               # Public API wrappers (~95 lines)
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_elf_loader.cpp          # 6 tests: ELF validation, kernel discovery
│   ├── test_isa_decoder.cpp         # 22 tests: instruction format decoding
│   ├── test_wavefront.cpp           # 6 tests: SGPR/VGPR/EXEC initialization
│   └── test_vector_add.cpp          # 2 tests: end-to-end kernel execution
│
└── tools/vector_add/
    ├── vector_add.hip               # Test kernel (float A+B→C)
    └── CMakeLists.txt               # Builds vector_add.hsaco via hipcc
```

---

## What Was Implemented

### Phase 1: ELF Loader

**Files:** `src/elf_loader.cpp`, `src/metadata_parser.cpp`, `include/gcnsim/elf_loader.h`

The ELF loader reads a binary `.hsaco` file into memory and extracts all kernel metadata needed to simulate it.

**Validation:**
- Checks ELF magic bytes (`\x7FELF`)
- Verifies `e_ident[EI_CLASS] == ELFCLASS64` (64-bit ELF)
- Verifies `e_machine == EM_AMDGPU (224)`
- Verifies `e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA (64)`
- Detects Code Object Version from `e_ident[EI_ABIVERSION]`:
  - 0 → COV2, 1 → COV3, 2 → COV4, 3 → COV5

**Section parsing:** Iterates `Elf64_Shdr` entries to locate `.text`, `.rodata`, `.dynsym`/`.symtab`, `.note`, and their associated string tables.

**Kernel discovery:** Scans the symbol table for entries whose `ELF64_ST_TYPE(st_info) == STT_AMDGPU_HSA_KERNEL (10)` and whose name ends in `.kd`. Each such symbol points to a 64-byte `KernelDescriptor` struct. The kernel's first instruction is at:
```
code_ptr = kd_ptr + kd->kernel_code_entry_byte_offset
```
For gfx9 targets, `kernel_code_entry_byte_offset` is always 256 (64-byte KD + 192 bytes padding).

**Metadata parsing (`metadata_parser.cpp`):** Scans the `.note` section for an entry with `n_type == 32` (`NT_AMDGPU_METADATA`) and `n_name == "AMDGPU"`. The descriptor bytes contain a MsgPack-encoded blob with the full kernel metadata. A hand-written minimal MsgPack reader (`MsgPackReader` class) walks the nested map/array structure to extract per-argument information:
- `.name` — argument name
- `.type_name` — C++ type string
- `.size` — byte size
- `.offset` — byte offset within the kernarg buffer
- `.value_kind` — `"global_buffer"`, `"by_value"`, or `"hidden_*"` for compiler-injected args

**Test (`test_elf_loader.cpp`, 6 tests):** Loads the reference HSACO at `/opt/rocm/bin/gfx90a/vector_add_memory_fault_kernels.hsaco` and asserts that: the file opens, at least one kernel is found, the descriptor and code pointers are non-null, kernel names are non-empty, COV version is in range, and the code entry offset is positive.

---

### Phase 2: ISA Decoder

**Files:** `src/isa_decoder.cpp`, `include/gcnsim/isa_decoder.h`

The decoder takes a pointer to raw bytes at the current PC and returns a fully-decoded `Instruction` struct in a single pass. Instructions are either 4 or 8 bytes; the `size_bytes` field tells the caller how much to advance the PC.

**Format detection** (priority order, most-specific first):

| Bit pattern | Format | Instruction size |
|---|---|---|
| `w[31:23] == 0x17F` | SOPP | 4 bytes |
| `w[31:23] == 0x17E` | SOP1 | 4 bytes (8 if SRC0 is literal) |
| `w[31:23] == 0x17D` | SOPC | 4 bytes |
| `w[31:30] == 0b10` (remainder) | SOP2 | 4 bytes |
| `w[31:26] == 0b110000` | SMEM | 8 bytes |
| `w[31:26] == 0b110111` | FLAT/GLOBAL | 8 bytes |
| `w[31:26] == 0b110100` | VOP3A/VOP3B | 8 bytes |
| `w[31:25] == 0b0111111` | VOP1 | 4 bytes |
| `w[31:25] == 0b0111110` | VOPC | 4 bytes |
| `w[31] == 0` (remainder) | VOP2 | 4 bytes |

**Operand encoding:** GCN encodes source operands in 9 bits:

| Value range | Meaning |
|---|---|
| 0–103 | SGPR 0–103 |
| 106–107 | VCC_LO / VCC_HI |
| 124 | M0 |
| 126–127 | EXEC_LO / EXEC_HI |
| 128 | Inline constant: 0 |
| 129–192 | Inline constants: 1–64 |
| 193–208 | Inline constants: −1 to −16 |
| 240–247 | Float constants: 0.5, −0.5, 1.0, −1.0, 2.0, −2.0, 4.0, −4.0 |
| 253 | SCC |
| 255 | Literal (next 4 bytes — makes instruction 8 bytes) |
| 256–511 | VGPR 0–255 |

**`s_waitcnt`** — parses `vmcnt` and `lgkmcnt` fields from the 16-bit immediate but executes as a no-op (all simulation memory is synchronous).

**Test (`test_isa_decoder.cpp`, 22 tests):** Encodes known instruction words in byte arrays and decodes them, asserting format, opcode, operand kinds, register indices, immediate values, and size. Covers SOPP, SOP1, SOP2, SMEM, VOP1, VOP2, VOPC, VOP3A, VOP3B, and FLAT_GLOBAL instructions.

---

### Phase 3: Wavefront Executor

**Files:** `src/wavefront.cpp`, `src/isa_executor.cpp`, `include/gcnsim/wavefront.h`

**Wavefront state** (`WavefrontState`):
- `sgpr[108]` — scalar registers; VCC lives at `sgpr[106:107]`
- `vgpr[256][64]` — vector registers indexed as `[reg][lane]`
- `exec` (64-bit) — active lane bitmask
- `vcc`, `scc`, `m0` — condition and special registers
- `pc` — byte offset into `kernel_code_ptr`

**Wavefront initialization (`init_wavefront()`):**

SGPR slots are assigned left-to-right in order dictated by `kernel_code_properties` and `compute_pgm_rsrc2` bits:

| SGPRs | Content | Enable condition |
|---|---|---|
| [+0:+3] | Private segment buffer | `KCP_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER` |
| [+0:+1] | Dispatch pointer | `KCP_ENABLE_SGPR_DISPATCH_PTR` |
| [+0:+1] | Queue pointer | `KCP_ENABLE_SGPR_QUEUE_PTR` |
| [+0:+1] | **Kernarg segment pointer** | `KCP_ENABLE_SGPR_KERNARG_SEGMENT_PTR` |
| [+0:+1] | Dispatch ID | `KCP_ENABLE_SGPR_DISPATCH_ID` |
| [+0:+1] | Flat scratch init | `KCP_ENABLE_SGPR_FLAT_SCRATCH_INIT` |
| [+0] | Private segment size | `KCP_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE` |
| [+0] | Work-group ID X | `RSRC2_ENABLE_SGPR_WORKGROUP_ID_X` |
| [+0] | Work-group ID Y | `RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y` |
| [+0] | Work-group ID Z | `RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z` |

EXEC mask for a partial final wavefront: `exec = (1ULL << active_lanes) - 1`.

VGPR[0][lane] = work-item X ID within the wavefront (ABI requirement).

**Execution loop (`run_wavefront()`):**
```
while true:
    inst = decode(wave.kernel_code_ptr + wave.pc)
    wave.pc += inst.size_bytes
    execute_instruction(wave, inst)
    if inst.opcode == S_ENDPGM: break
```
A safety limit of 1,000,000 instructions prevents infinite loops.

**Key instruction semantics:**

- **All vector instructions** iterate all 64 lanes and skip where `(exec >> lane) & 1 == 0`, implementing the SIMT execution model.

- **`s_and_saveexec_b64`** — saves current EXEC to a SGPR pair, then `exec = exec & vcc`. Essential for `if (i < N)` style masking.

- **`v_cmp_lt_u32` (VOPC)** — computes per-lane comparison, accumulates results into `vcc` as a 64-bit bitmask (one bit per lane).

- **`v_add_co_u32` (VOP3B)** — vector add with carry-out. Writes result to VDST and carry to a second destination (DST2, typically VCC or a SGPR pair).

- **`v_addc_co_u32` (VOP3B)** — vector add with carry-in from DST2_SRC and carry-out to DST2. Used for 64-bit pointer arithmetic.

- **`global_load_dword` / `global_store_dword`** — two addressing modes:
  - *SGPR base* mode: `addr = sgpr_pair_base + vgpr_offset + flat_offset`
  - *`off`* mode: `addr = vgpr_pair (64-bit) + flat_offset`

- **`s_load_dwordx4`** — loads 4 consecutive DWORDs from address `sgpr_pair + offset` into 4 consecutive SGPRs. Used to load pointer-sized kernel arguments.

- **Float ops** (`v_add_f32`, `v_mul_f32`, `v_mad_f32`) — bit-cast via `memcpy` to avoid strict aliasing UB.

**Test (`test_wavefront.cpp`, 6 tests):** Verifies EXEC mask for full and partial waves, VGPR[0] lane initialization, kernarg pointer placement in SGPRs, work-group ID SGPR placement, and multi-dimensional work-item ID layout.

---

### Phase 4: Public API & Execution Engine

**Files:** `src/execution_engine.cpp`, `src/gcnsim_api.cpp`, `include/gcnsim/gcnsim.h`, `include/gcnsim/execution_engine.h`

**Execution engine** iterates the 3D workgroup grid and dispatches one wavefront per 64 threads:

```
wg_count_x = ceil(grid_x / block_x)
wg_count_y = ceil(grid_y / block_y)
wg_count_z = ceil(grid_z / block_z)
waves_per_wg = ceil(block_x * block_y * block_z / 64)

for each (wg_z, wg_y, wg_x):
  for wave in [0, waves_per_wg):
    ws = new WavefrontState   // heap-allocated (~66 KB)
    init_wavefront(ws, ...)
    run_wavefront(ws)
```

`WavefrontState` is heap-allocated because its ~66 KB size exceeds safe stack space.

**Hidden kernel arguments** are automatically filled by scanning the metadata for args whose `value_kind` starts with `"hidden_"`. Recognized names and what they are set to:

| Hidden arg name | Value |
|---|---|
| `hidden_block_count_x/y/z` | Workgroup grid dimension |
| `hidden_group_size_x/y/z` | Block (work-group) size |
| `hidden_remainder_x/y/z` | `grid_dim % block_size` (partial last block) |
| `hidden_global_offset_x/y/z` | 0 (not supported) |

**Public API** (`gcnsim.h`):
```cpp
// Load an HSACO file
std::optional<SimModule>   module_load(const std::string& path, std::string& err);

// Find a kernel by name
std::optional<SimFunction> module_get_function(SimModule& mod,
                                                const std::string& name,
                                                std::string& err);

// Execute: kernel_params is an array of pointers, one per argument
bool launch_kernel(SimFunction& fn,
                   uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                   uint32_t block_x, uint32_t block_y, uint32_t block_z,
                   void** kernel_params, std::string& err);
```

Parameter marshalling: `launch_kernel` iterates the kernel's metadata arg list, skipping hidden args, and copies each user-supplied `void*` param into the kernarg buffer at its metadata-specified offset. Falls back to natural alignment packing if metadata is unavailable.

---

### Phase 5: Integration Test

**Files:** `tools/vector_add/vector_add.hip`, `tests/test_vector_add.cpp`

The reference kernel:
```cpp
extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```

Compiled with:
```bash
hipcc --offload-arch=gfx90a -c --genco vector_add.hip -o vector_add.hsaco
```

The integration test (`test_vector_add.cpp`) runs with N=1024, block=256, grid=4:
- Allocates three host `float` arrays (A, B, C)
- Fills A and B with deterministic values
- Builds the 28-byte kernarg buffer `{A(8), B(8), C(8), N(4)}`
- Calls `launch_kernel` which dispatches 16 total wavefronts (4 workgroups × 4 waves each)
- Verifies `|C[i] - (A[i] + B[i])| < 1e-5` for all 1024 elements

---

## Key Design Decisions

### 1. Explicit API over LD_PRELOAD
The public API is a plain C++ library rather than a HIP runtime interceptor. This makes the simulator testable with standard `ctest`, avoids any dependency on the GPU stack, and keeps scope well-defined.

### 2. Custom ISA decoder over text disassembly
Decoding raw bytes into structured `Instruction` objects (rather than parsing `llvm-objdump` output) is required for execution — you need field values as integers, not text. It also runs orders of magnitude faster.

### 3. Pure host memory
All addresses are host `malloc` pointers. `global_load_dword` dereferences them directly via `memcpy`-based helpers in `memory_model.h`. This avoids any GPU memory management and makes the simulator trivially debuggable with a host debugger.

### 4. Metadata-aware kernarg packing
Rather than guessing argument layout, the simulator reads the MsgPack-encoded kernel metadata from the ELF `.note` section, giving it exact byte offsets for each argument. Hidden HIP arguments (`hidden_block_count_x`, etc.) are detected by `value_kind` prefix and automatically computed from the launch configuration.

### 5. EXEC mask as 64-bit integer
The EXEC mask is a plain `uint64_t` where bit `n` represents lane `n`. All vector instructions iterate `lane` in `[0, 64)` and `continue` if the bit is clear. This is simple, correct, and easy to verify in a debugger.

### 6. Heap-allocated WavefrontState
`sizeof(WavefrontState)` is approximately 66 KB (`108*4 + 256*64*4 + ...`). Allocating it on the stack for multi-wave kernels would overflow. The execution engine `new`s each wavefront, runs it, then `delete`s it.

---

## Building

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| CMake | ≥ 3.20 | Build system |
| C++ compiler | GCC ≥ 11 or Clang ≥ 14 | C++20 required |
| yaml-cpp | any | MsgPack/metadata parser dependency |
| GTest | any | Unit tests |
| ROCm | ≥ 6.0 at `/opt/rocm` | Headers + `hipcc` for kernel compilation |

### Build steps

```bash
# Configure (tests enabled by default)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGCNSIM_BUILD_TESTS=ON

# Build library, tests, and the vector_add.hsaco kernel
cmake --build build -j$(nproc)
```

The `vector_add.hsaco` kernel is built automatically as part of the `tools/vector_add` target if `hipcc` is found. If `hipcc` is not present, that target is silently skipped (the integration test will then be skipped or fail with a clear error).

---

## Running Tests

```bash
# Run all tests with verbose output
ctest --test-dir build -V

# Or run individual test binaries
./build/tests/test_elf_loader
./build/tests/test_isa_decoder
./build/tests/test_wavefront
./build/tests/test_vector_add
```

Expected output: **34 tests pass**.

| Test suite | Count | What it covers |
|---|---|---|
| `test_elf_loader` | 6 | ELF validation, kernel discovery, descriptor parsing |
| `test_isa_decoder` | 22 | All instruction formats and operand encoding |
| `test_wavefront` | 6 | SGPR/VGPR/EXEC initialization, 2D work-item layout |
| `test_vector_add` | 2 | End-to-end gfx90a kernel simulation |

---

## Usage Example

```cpp
#include <gcnsim/gcnsim.h>
#include <vector>
#include <cassert>

int main() {
    std::string err;

    // Load the compiled kernel
    auto mod = gcnsim::module_load("vector_add.hsaco", err);
    assert(mod && "load failed: " + err);

    auto fn = gcnsim::module_get_function(*mod, "vector_add", err);
    assert(fn && "kernel not found: " + err);

    // Prepare data on the host
    const int N = 1024;
    std::vector<float> A(N), B(N), C(N, 0.f);
    for (int i = 0; i < N; i++) { A[i] = float(i); B[i] = float(N - i); }

    int n_val = N;
    float* pA = A.data();
    float* pB = B.data();
    float* pC = C.data();

    // void** kernel_params: one pointer per argument
    void* params[] = { &pA, &pB, &pC, &n_val };

    bool ok = gcnsim::launch_kernel(*fn,
        /*grid*/  N, 1, 1,
        /*block*/ 256, 1, 1,
        params, err);
    assert(ok);

    for (int i = 0; i < N; i++)
        assert(std::abs(C[i] - (A[i] + B[i])) < 1e-5f);
}
```

---

## Supported Instructions

### Scalar (SOPP / SOP1 / SOP2)

| Instruction | Format | Notes |
|---|---|---|
| `s_endpgm` | SOPP | Terminates execution loop |
| `s_nop` | SOPP | No-op |
| `s_branch` | SOPP | Unconditional PC-relative branch |
| `s_cbranch_scc0/1` | SOPP | Branch on SCC |
| `s_cbranch_execz/execnz` | SOPP | Branch on EXEC == 0 or ≠ 0 |
| `s_waitcnt` | SOPP | No-op (simulation is synchronous) |
| `s_mov_b32/b64` | SOP1 | Scalar move |
| `s_and_saveexec_b64` | SOP1 | Save EXEC, mask with VCC |
| `s_or_saveexec_b64` | SOP1 | Save EXEC, OR with VCC |
| `s_add_u32/i32` | SOP2 | Add with SCC carry |
| `s_sub_u32` | SOP2 | Subtract with SCC borrow |
| `s_and_b32/b64` | SOP2 | Bitwise AND |
| `s_or_b32` | SOP2 | Bitwise OR |
| `s_lshl_b32` | SOP2 | Logical shift left |
| `s_lshr_b32` | SOP2 | Logical shift right |
| `s_mul_i32` | SOP2 | Scalar multiply |

### Scalar Memory (SMEM)

| Instruction | Notes |
|---|---|
| `s_load_dword` | Load 1 DWORD from SGPR-pair base + offset |
| `s_load_dwordx2` | Load 2 DWORDs |
| `s_load_dwordx4` | Load 4 DWORDs — used for pointer args |
| `s_load_dwordx8` | Load 8 DWORDs |
| `s_load_dwordx16` | Load 16 DWORDs |

### Vector ALU (VOP1 / VOP2 / VOP3A / VOP3B / VOPC)

| Instruction | Format | Notes |
|---|---|---|
| `v_mov_b32` | VOP1 | Copy SGPR/literal → VGPR |
| `v_add_f32` | VOP2 | Float add |
| `v_mul_f32` | VOP2 | Float multiply |
| `v_add_u32` | VOP2 | Integer add (no carry) |
| `v_sub_u32` | VOP2 | Integer subtract |
| `v_lshlrev_b32` | VOP2 | Left shift |
| `v_lshrrev_b32` | VOP2 | Logical right shift |
| `v_ashrrev_i32` | VOP2 | Arithmetic right shift |
| `v_and_b32` | VOP2 | Bitwise AND |
| `v_or_b32` | VOP2 | Bitwise OR |
| `v_add_co_u32` | VOP3B | Add with carry-out |
| `v_addc_co_u32` | VOP3B | Add with carry-in and carry-out |
| `v_ashrrev_i64` | VOP3A | 64-bit arithmetic right shift |
| `v_lshlrev_b64` | VOP3A | 64-bit left shift |
| `v_mad_f32` | VOP3A | Fused multiply-add |
| `v_cvt_f32_i32` | VOP1 | Int → float |
| `v_cvt_i32_f32` | VOP1 | Float → int |
| `v_cmp_lt/le/gt/ge/eq/ne_u32` | VOPC | Unsigned compare → VCC |
| `v_cmp_lt/ge_i32` | VOPC | Signed compare → VCC |

### Global Memory (FLAT_GLOBAL)

| Instruction | Notes |
|---|---|
| `global_load_dword` | Per-lane 32-bit load |
| `global_load_dwordx2` | Per-lane 64-bit load |
| `global_load_dwordx4` | Per-lane 128-bit load |
| `global_store_dword` | Per-lane 32-bit store |
| `global_store_dwordx2` | Per-lane 64-bit store |
| `global_store_dwordx4` | Per-lane 128-bit store |

Both addressing modes are supported:
- **SGPR base + VGPR offset**: `addr = sgpr_pair + vgpr + flat_offset`
- **`off` (pure VGPR 64-bit pair)**: `addr = vgpr_pair + flat_offset`

---

## GCN ISA Reference Notes

- `kernel_code_entry_byte_offset` is always **256** for gfx9 targets: 64-byte KD + 192 bytes of alignment padding before the first instruction.
- `STT_AMDGPU_HSA_KERNEL = STT_LOOS + 0 = 10` — kernel symbols in the ELF symbol table.
- VCC in 108-SGPR allocation: `sgpr[106]` (VCC_LO), `sgpr[107]` (VCC_HI).
- SMEM SBASE encoding: the 7-bit field stores `actual_sgpr / 2` (always a SGPR pair).
- VOP3B: bit `[7]` of word0 distinguishes VOP3B from VOP3A (DST2 = SGPR destination for carry-out).
- Branch offsets in SOPP are in **DWORD units** (×4 bytes), relative to the instruction following the branch.
- `s_waitcnt` field layout: `vmcnt = bits[3:0]`, `lgkmcnt = bits[11:8]` (in the 16-bit SIMM16).
- EXEC zeroing via `s_and_saveexec_b64` is the standard mechanism for `if (i < N)`-style bounds checks in GPU kernels.

---

## Potential Improvements

### Correctness

- **Carry/borrow propagation in SOP2** — `s_add_u32`/`s_sub_u32` currently set SCC but don't propagate a multi-word carry chain. Add `s_addc_u32` and `s_subb_u32`.
- **`s_getpc_b64`** — reads the current PC into a SGPR pair. Needed by kernels that use position-independent code or call device functions.
- **`v_readfirstlane_b32`** — copies lane 0's VGPR value to a SGPR. Required by some idioms for extracting a uniform value from a vector register.
- **`v_readlane_b32` / `v_writelane_b32`** — cross-lane data movement. Needed for reductions and shuffles.
- **Inline float constant accuracy** — the decoder handles the standard set (±0.5, ±1.0, ±2.0, ±4.0) but does not handle the extended set present in some encodings (e.g., 1/π, 2/π).
- **SOPK instructions** — `s_movk_i32`, `s_cmpk_eq_i32`, etc. are not yet decoded or executed.
- **`s_load_*` with SGPR offset** — the IMM=0 path (using a SGPR as the offset register rather than a literal) is not fully implemented.
- **Flat memory segment** — SEG=0 (plain `flat_load/store`) is not handled; only SEG=2 (`global_*`) is implemented.

### Performance

- **Parallel wavefront execution** — each wavefront is currently sequential. An `std::async` or thread pool over the wavefront loop in `execution_engine.cpp` would exploit multi-core CPUs and correctly model the GPU's wave-parallel execution.
- **SIMD vectorization of lane loops** — the inner `for (int lane = 0; lane < 64; lane++)` loop in the executor is a natural target for AVX-512 auto-vectorization with a 64-element predicate mask.
- **Decode cache** — the same instruction bytes are decoded every time the kernel loops. A decoded instruction cache keyed on PC offset would eliminate repeated decoding for loops.

### Usability

- **Instruction trace / logging** — an optional `--trace` mode that prints each decoded instruction and the registers it touches before/after execution would dramatically accelerate debugging new kernels.
- **Register dump on S_ENDPGM** — print or return the final SGPR/VGPR state, useful for unit-testing small compute shaders.
- **Error on unknown instruction** — currently `UNKNOWN` opcode is silently skipped. It should trigger a configurable error or assertion to catch unsupported instructions early.
- **Bounds-checked memory model** — the current `memory_model.h` dereferences any address blindly. Adding a registered-region table and checking each access would turn silent segfaults into meaningful simulator errors.

### Build / Integration

- **`pkg-config` / CMake `find_package` export** — install a `gcnsim-config.cmake` so downstream projects can consume the library with a single `find_package(gcnsim)`.
- **Python bindings** — a thin pybind11 wrapper over the public API would make it easy to drive simulation from Python test scripts.

---

## Future Work

### Scratch / LDS Memory
Local Data Share (LDS / shared memory in CUDA terminology) is not simulated. Supporting `ds_read_b32` / `ds_write_b32` and the `group_segment_fixed_size` field of the kernel descriptor would enable simulation of kernels that use shared memory for reductions, tiling, etc.

Scratch (private) memory for stack spills is similarly absent. Adding an allocator that backs `FLAT_SCRATCH` addresses with host memory would support more complex kernels generated by optimizing compilers.

### Divergent Control Flow
The current EXEC mask handling is correct for `if/else` constructs that use `s_and_saveexec_b64` + `s_branch`. True divergent loops (where some lanes exit the loop body early) require a re-convergence stack — tracking pushed EXEC states and restoring them at merge points. Without this, loops with `s_cbranch_execz` that span multiple iterations are not correctly simulated.

### Multi-Wavefront Synchronization
GPU kernels use `s_barrier` to synchronize wavefronts within a workgroup. Since the current engine runs wavefronts sequentially, no synchronization is needed — but this means kernels relying on inter-wave communication via LDS are not correctly simulated. Supporting barriers would require running all wavefronts of a workgroup together (interleaved at barrier points), or a split-phase approach.

### F16 / BF16 / INT8 Support
The MI-200 (gfx90a) is designed for mixed-precision AI workloads. Supporting `v_pk_add_f16`, `v_dot2_f32_f16`, and the matrix (MFMA) instruction family (`v_mfma_f32_32x32x8bf16`, etc.) would make the simulator useful for validating AI kernels without hardware.

### Additional ISA Coverage

| Category | Missing instructions |
|---|---|
| Scalar | SOPK family, `s_cmp_*`, `s_setpc_b64`, `s_swappc_b64` |
| Vector | `v_min/max_*`, `v_med3_*`, `v_bfe_u32`, `v_bfi_b32` |
| Atomic | `global_atomic_add`, `global_atomic_cmpswap`, etc. |
| Texture | `image_load`, `image_store`, `image_sample` |
| MFMA | All `v_mfma_*` matrix multiply-accumulate instructions |

### Symbolic Execution / Taint Analysis
Instead of simulating one concrete input, run the simulator with symbolic values (e.g., using the Z3 SMT solver or a custom interval domain). This would enable automatic verification of kernel correctness across all inputs, or detection of out-of-bounds access for any possible index.

### Code Object V5 / V6
The current implementation supports COV3 and COV4 fully. COV5 (ROCm 5.x+) and COV6 (ROCm 6.x+) introduced changes to the kernel descriptor and metadata format. Full support would extend compatibility to kernels compiled with newer toolchains.

### Disassembler Output
Add a mode that takes an HSACO and prints a human-readable disassembly (similar to `llvm-objdump -d`) using the existing decoder. This would serve as both a useful standalone tool and a regression test — outputs can be diffed against `llvm-objdump` to validate the decoder.
