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
                     │  Formats: SOP1/2, SOPK, SOPC, SOPP,        │
                     │    VOP1/2/3A/3B/3P, VOPC, SMEM, DS, MUBUF, │
                     │    MTBUF, FLAT/GLOBAL/SCRATCH, MFMA         │
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
│   ├── isa_decoder.cpp              # Instruction decoder
│   ├── isa_executor.cpp             # Per-instruction handlers
│   ├── wavefront.cpp                # Wavefront init + run_wavefront()
│   ├── execution_engine.cpp         # Grid dispatch, LDS/scratch alloc, barrier sync
│   └── gcnsim_api.cpp               # Public API wrappers (~95 lines)
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_elf_loader.cpp          # 6 tests: ELF validation, kernel discovery
│   ├── test_isa_decoder.cpp         # 41 tests: instruction format decoding
│   ├── test_isa_executor.cpp        # 16 tests: instruction execution
│   ├── test_wavefront.cpp           # 9 tests: SGPR/VGPR/EXEC initialization
│   ├── test_vector_add.cpp          # 2 tests: end-to-end vector-add
│   ├── test_gemm_mfma.cpp           # 3 tests: MFMA 4x4x1 GEMM
│   ├── test_mfma_kernels.cpp        # 4 tests: multi-shape MFMA
│   ├── test_asm_gemm.cpp            # 2 tests: hand-written assembly GEMM
│   └── test_tensile_gemm.cpp        # 3 tests: Tensile-style SGEMM
│
├── tools/vector_add/                # Vector-add HIP kernel (.hsaco)
├── tools/gemm_mfma/                 # MFMA GEMM HIP kernel (.hsaco)
├── tools/mfma_kernels/              # Multi-shape MFMA kernels (.hsaco)
├── tools/asm_gemm_mfma/             # Hand-written GFX90A assembly GEMM (.co)
└── tools/tensile_style_gemm/        # Tensile-style SGEMM assembly kernel (.co)
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
| `w[31:23] == 0x17D` | SOP1 | 4 bytes (8 if SRC0 is literal) |
| `w[31:23] == 0x17E` | SOPC | 4 bytes |
| `w[31:28] == 0xB` | SOPK | 4 bytes |
| `w[31:26] == 0b110000` | SMEM | 8 bytes |
| `w[31:26] == 0b110110` | DS | 8 bytes |
| `w[31:26] == 0b110111` | FLAT/GLOBAL/SCRATCH | 8 bytes |
| `w[31:26] == 0b111000` | MUBUF | 8 bytes |
| `w[31:26] == 0b111010` | MTBUF | 8 bytes |
| `w[31:26] == 0b110011` | VOP3P | 8 bytes |
| `w[31:26] == 0b110100` | VOP3A/VOP3B/MFMA | 8 bytes |
| `w[31:25] == 0b0111111` | VOP1 | 4 bytes (8 if SDWA/DPP) |
| `w[31:25] == 0b0111110` | VOPC | 4 bytes (8 if SDWA/DPP) |
| `w[31:30] == 0b10` (remainder) | SOP2 | 4 bytes |
| `w[31] == 0` (remainder) | VOP2 | 4 bytes (8 if SDWA/DPP) |

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

**Wavefront state** (`WavefrontState`, ~132 KB — heap-allocated):
- `sgpr[108]` — scalar registers; VCC lives at `sgpr[106:107]`
- `vgpr[256][64]` — vector registers indexed as `[reg][lane]`
- `agpr[256][64]` — accumulator registers (MFMA destinations)
- `exec` (64-bit) — active lane bitmask
- `vcc`, `scc`, `m0` — condition and special registers
- `pc` — byte offset into `kernel_code_ptr`
- `lds_ptr` / `lds_size` — per-workgroup LDS buffer
- `scratch_ptr` / `scratch_size` / `scratch_per_lane` — per-wavefront scratch memory

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

**Execution engine** iterates the 3D workgroup grid and dispatches wavefronts per workgroup:

```
wg_count_x = ceil(grid_x / block_x)
wg_count_y = ceil(grid_y / block_y)
wg_count_z = ceil(grid_z / block_z)
waves_per_wg = ceil(block_x * block_y * block_z / 64)

for each (wg_z, wg_y, wg_x):
  allocate LDS buffer (group_segment_fixed_size bytes)
  allocate scratch buffers (per wavefront)
  for wave in [0, waves_per_wg):
    init_wavefront(ws, ...)
  run wavefronts round-robin until all end (with barrier sync)
```

`WavefrontState` is ~132 KB (with AGPRs) and always heap-allocated.

Wavefronts within a workgroup share LDS and are synchronized at `s_barrier` instructions using round-robin scheduling: each wavefront runs until it hits a barrier or ends, and all wavefronts resume together when every active wave has reached the barrier.

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
clang -x hip --cuda-device-only --offload-arch=gfx90a --no-gpu-bundle-output -O2 -c \
  -o vector_add.hsaco vector_add.hip
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
`sizeof(WavefrontState)` is approximately 132 KB (SGPRs + VGPRs + AGPRs). Allocating it on the stack for multi-wave kernels would overflow. The execution engine uses `std::make_unique` per wavefront.

---

## Building

### Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| CMake | ≥ 3.20 | Build system |
| C++ compiler | GCC ≥ 11 or Clang ≥ 14 | C++20 required |
| yaml-cpp | any | MsgPack/metadata parser dependency |
| GTest | any | Unit tests |
| clang (amdgcn) | ROCm clang at `/opt/rocm/llvm/bin/clang`, or upstream LLVM built with `-DLLVM_TARGETS_TO_BUILD=AMDGPU` | Kernel compilation (HIP `.hip` → `.hsaco`, assembly `.s` → `.co`) |

### Build steps

```bash
# Configure (tests enabled by default)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGCNSIM_BUILD_TESTS=ON

# Build library, tests, and the vector_add.hsaco kernel
cmake --build build -j$(nproc)
```

The `vector_add.hsaco` kernel is built automatically as part of the `tools/vector_add` target if a clang with amdgcn target support is found. If clang is not present, that target is silently skipped (the integration test will then be skipped or fail with a clear error).

---

## Running Tests

```bash
# Run all tests with verbose output
ctest --test-dir build -V

# Or run individual test binaries
./build/tests/test_elf_loader
./build/tests/test_isa_decoder
./build/tests/test_isa_executor
./build/tests/test_wavefront
./build/tests/test_vector_add
./build/tests/test_gemm_mfma --hsaco build/tools/gemm_mfma/gemm_mfma.hsaco
./build/tests/test_mfma_kernels --hsaco build/tools/mfma_kernels/mfma_kernels.hsaco
./build/tests/test_asm_gemm --hsaco build/tools/asm_gemm_mfma/asm_gemm_mfma.co
./build/tests/test_tensile_gemm --hsaco build/tools/tensile_style_gemm/tensile_style_gemm.co
```

Expected output: **104 tests pass across 11 suites**.

| Test suite | Count | What it covers |
|---|---|---|
| `test_elf_loader` | 6 | ELF validation, kernel discovery, descriptor parsing |
| `test_isa_decoder` | 41 | All instruction formats, SDWA, DPP, VOP3 modifiers |
| `test_isa_executor` | 16 | Instruction execution, modifiers, scratch memory |
| `test_wavefront` | 9 | SGPR/VGPR/EXEC init, workitem_id.x/y/z |
| `test_vector_add` | 2 | End-to-end vector-add kernel |
| `test_gemm_mfma` | 3 | MFMA 4x4x1 GEMM integration |
| `test_mfma_kernels` | 4 | Multi-shape MFMA (16x16x1, 16x16x4, 32x32x1, 32x32x2) |
| `test_asm_gemm` | 2 | Hand-written assembly MFMA GEMM |
| `test_tensile_gemm` | 3 | Tensile-style 128x256 SGEMM with LDS, barriers, multi-wavefront |
| `test_cycle_model` | 13 | MFMA shadow, s_waitcnt stalls, bank conflicts, utilization, per-SIMD interleaving |
| `test_cache_model` | 5 | L1/L2 cache hit/miss, eviction, stats |

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

The simulator covers ~340 instruction opcodes across all major GFX9/GFX90a formats. Key categories:

### Scalar (SOPP / SOP1 / SOP2 / SOPK / SOPC)

- **SOPP**: `s_endpgm`, `s_nop`, `s_branch`, `s_cbranch_scc0/1`, `s_cbranch_execz/execnz`, `s_cbranch_vccz/vccnz`, `s_waitcnt` (no-op), `s_barrier`
- **SOP1**: `s_mov_b32/b64`, `s_not_b32/b64`, `s_wqm_b64`, `s_and/or/xor/andn2/orn2_saveexec_b64`, `s_getpc_b64`, `s_setpc_b64`, `s_swappc_b64`, `s_ff1_i32_b32/b64`, `s_bitcnt1_b32_b64`, `s_abs_i32`, `s_movrels_b32`, `s_movreld_b32`
- **SOP2**: full set including `s_add/sub_u32/i32`, `s_addc/subb_u32`, `s_min/max_i32/u32`, `s_cselect_b32/b64`, `s_and/or/xor/andn2/orn2/nand/nor/xnor_b32/b64`, `s_lshl/lshr_b32/b64`, `s_ashr_i32/i64`, `s_bfm/bfe_b32/b64`, `s_mul_i32`, `s_absdiff_i32`
- **SOPK**: `s_movk_i32`, `s_cmovk_i32`, `s_cmpk_eq/lg/gt/ge/lt/le_i32/u32`, `s_addk_i32`, `s_mulk_i32`
- **SOPC**: `s_cmp_eq/lg/gt/ge/lt/le_i32/u32`, `s_bitcmp0/1_b32`

### Scalar Memory (SMEM)

`s_load_dword/x2/x4/x8/x16`, `s_buffer_load_dword/x2/x4/x8/x16`, `s_store_dword/x2/x4`

### Vector ALU (VOP1 / VOP2 / VOP3A / VOP3B / VOPC)

- **VOP1**: `v_mov_b32`, `v_cvt_*` (f32↔i32, f32↔u32, f32↔f16, f32↔f64, f16↔u16), `v_rcp_f32/f64`, `v_rsq_f32/f64`, `v_sqrt_f32/f64`, `v_floor/ceil/trunc/fract/rndne_f32/f64`, `v_exp_f32`, `v_log_f32`, `v_not_b32`, `v_bfrev_b32`, `v_readfirstlane_b32`
- **VOP2**: `v_cndmask_b32`, `v_add/sub/mul_f32`, `v_mac_f32`, `v_add/sub_u32`, `v_add_co_u32`, `v_addc_co_u32`, `v_lshlrev/lshrrev/ashrrev_b32`, `v_and/or/xor_b32`, `v_min/max_f32/i32/u32`, `v_madmk/madak_f32`, `v_mul_hi_u32`, `v_subrev_u32`
- **VOP3A**: `v_mad/fma_f32/f64`, `v_mul_lo/hi_u32/i32`, `v_lshlrev_b64`, `v_ashrrev_i64`, `v_lshl_add_u32`, `v_add3_u32`, `v_lshl_or_b32`, `v_and_or_b32`, `v_or3_b32`, `v_med3_f32/i32/u32`, `v_bfe_u32/i32`, `v_bfi_b32`, `v_alignbit_b32`, `v_perm_b32`, `v_mad_u64_u32`, `v_readlane_b32`, `v_writelane_b32`, `v_accvgpr_read/write_b32`
- **VOP3B**: `v_add_co_u32`, `v_addc_co_u32`, `v_sub_co_u32`, `v_subb_co_u32`, `v_div_scale_f32/f64`
- **VOPC**: full `v_cmp_*` set for F32, F64, I32, U32, I64, U64 (all conditions)
- **VOP3 modifiers**: NEG, ABS (source), CLAMP, OMOD (destination) — applied to all float instructions
- **SDWA**: byte/word selection sub-encoding (src0 == 0xF9)
- **DPP**: cross-lane data movement sub-encoding (src0 == 0xFA)

### VOP3P (Packed Operations)

`v_pk_add/mul/fma_f16`, `v_pk_add/mul/fma_f32` (CDNA2), `v_dot2_f32_f16`, `v_dot2c_f32_f16`, `v_fma_mix_f32`, `v_fma_mixlo/mixhi_f16` — with op_sel/op_sel_hi/neg_hi support

### DS (Local Data Share)

`ds_read_b32/b64/b128/u8/u16`, `ds_write_b32/b64/b128/b8/b16`, `ds_read2_b32/b64`, `ds_write2_b32/b64`, `ds_add/sub/min/max/and/or/xor_u32/u64/rtn`, `ds_cmpst_b32/b64/rtn`

### MUBUF / MTBUF (Buffer Memory)

`buffer_load_dword/x2/x3/x4`, `buffer_load_ubyte/ushort/sbyte/sshort`, `buffer_store_dword/x2/x3/x4`, `buffer_store_byte/short`, `buffer_atomic_add/sub/smin/smax/umin/umax/and/or/xor/swap/cmpswap` — with LDS-direct flag support

### FLAT / GLOBAL / SCRATCH

`flat/global/scratch_load_dword/x2/x3/x4/ubyte/ushort/sbyte/sshort`, `flat/global/scratch_store_dword/x2/x3/x4/byte/short`, `flat/global_atomic_add/sub/smin/smax/umin/umax/and/or/xor/swap/cmpswap/add_x2`

### MFMA (Matrix Fused Multiply-Accumulate)

All CDNA2 MFMA shapes with correct cross-lane outer-product semantics:
- **F32**: `v_mfma_f32_4x4x1f32`, `v_mfma_f32_16x16x1f32`, `v_mfma_f32_16x16x4f32`, `v_mfma_f32_32x32x1f32`, `v_mfma_f32_32x32x2f32`
- **F16/BF16**: `v_mfma_f32_4x4x4f16`, `v_mfma_f32_16x16x4f16`, `v_mfma_f32_16x16x16f16`, `v_mfma_f32_32x32x4f16`, `v_mfma_f32_32x32x8f16` (and bf16_1k equivalents)
- **I8**: `v_mfma_i32_4x4x4i8`, `v_mfma_i32_16x16x4i8`, `v_mfma_i32_16x16x16i8`, `v_mfma_i32_32x32x4i8`, `v_mfma_i32_32x32x8i8`
- **F64**: `v_mfma_f64_4x4x4f64`, `v_mfma_f64_16x16x4f64`

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

### Performance

- **Parallel wavefront execution** — wavefronts across independent workgroups could run in parallel. An `std::async` or thread pool would exploit multi-core CPUs.
- **SIMD vectorization of lane loops** — the inner `for (int lane = 0; lane < 64; lane++)` loop in the executor is a natural target for AVX-512 auto-vectorization.
- **Decode cache** — the same instruction bytes are decoded every time the kernel loops. A decoded instruction cache keyed on PC offset would eliminate repeated decoding.

### Usability

- **Instruction trace / logging** — an optional `--trace` mode that prints each decoded instruction and register state would accelerate debugging new kernels.
- **Bounds-checked memory model** — the current `memory_model.h` dereferences any address blindly. Adding a registered-region table would turn silent segfaults into meaningful errors.

### Build / Integration

- **`pkg-config` / CMake `find_package` export** — install a `gcnsim-config.cmake` so downstream projects can consume the library with `find_package(gcnsim)`.
- **Python bindings** — a thin pybind11 wrapper over the public API for driving simulation from Python test scripts.

---

## Future Work

### Re-convergence Stack
The current EXEC mask handling is correct for `if/else` constructs that use `s_and_saveexec_b64` + `s_branch`. True divergent loops (where some lanes exit the loop body early) require a re-convergence stack — tracking pushed EXEC states and restoring them at merge points.

### Multi-tile Tensile GEMM
The single-tile (128x256x32) Tensile SGEMM passes end-to-end. Extending to multi-tile problems would exercise inter-workgroup correctness and validate the full production GEMM path.

### Memory Ordering Model
`s_waitcnt` is currently a no-op since all simulation memory is synchronous. A simple counter-based model would allow detection of missing barriers in test kernels.

### MIMG (Texture) Instructions
Image sampling instructions require texture descriptor and sampler hardware state. Currently silently skipped.

### Code Object V5 / V6
The current implementation supports COV3 and COV4 fully. COV5 (ROCm 5.x+) and COV6 (ROCm 6.x+) introduced changes to the kernel descriptor and metadata format. Full support would extend compatibility to kernels compiled with newer toolchains.

### Symbolic Execution / Taint Analysis
Run the simulator with symbolic values (e.g., using Z3) to enable automatic verification of kernel correctness or detection of out-of-bounds access for any possible index.

### Disassembler Output
Add a mode that prints human-readable disassembly (similar to `llvm-objdump -d`) using the existing decoder. Useful as a standalone tool and a regression test against `llvm-objdump`.
