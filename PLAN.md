# GCN-SIM: MI-200 (gfx90a) CPU-Based Instruction-Level Kernel Simulator

## Context

Build a C++ library that loads HSACO (HIP/HSA Code Object) ELF files, decodes gfx90a ISA instructions, and simulates GPU kernel execution on the CPU at the instruction level — no real GPU required. The first test kernel is vector add (A[i] + B[i] → C[i]).

**Environment:** ROCm 7.3.0 at `/opt/rocm-7.3.0`, wavefront size = 64 (gfx90a).

**Key design decisions:**
- **Explicit simulator API** (not LD_PRELOAD) — easier to test, zero GPU dependency
- **Custom ISA decoder** (not COMGR text output) — needed for structured execution
- **Pure host memory** — kernarg pointers are host `malloc` ptrs; `global_load/store` dereferences them directly

---

## Directory Structure

```
gcn-sim/
├── CMakeLists.txt
├── cmake/
├── include/gcnsim/
│   ├── gcnsim.h            # Public API (module_load, module_get_function, launch_kernel)
│   ├── elf_loader.h        # HSACO ELF parser
│   ├── kernel_descriptor.h # 64-byte Code Object V3/V4 kernel descriptor struct
│   ├── isa_decoder.h       # Instruction/Opcode/Operand structs + decode()
│   ├── wavefront.h         # WavefrontState (SGPRs, VGPRs, EXEC, VCC, PC)
│   ├── execution_engine.h  # Grid→wavefront dispatch
│   └── memory_model.h      # Bounds-checked host pointer accessor
├── src/
│   ├── elf_loader.cpp
│   ├── metadata_parser.cpp # Parse YAML .note section (yaml-cpp)
│   ├── isa_decoder.cpp
│   ├── isa_executor.cpp    # Per-instruction execute handlers
│   ├── wavefront.cpp
│   ├── execution_engine.cpp
│   └── gcnsim_api.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_elf_loader.cpp
│   ├── test_isa_decoder.cpp
│   ├── test_wavefront.cpp
│   └── test_vector_add.cpp # Integration test
└── tools/vector_add/
    ├── vector_add.hip       # Test kernel source
    └── CMakeLists.txt       # Builds .hsaco via hipcc --genco --offload-arch=gfx90a
```

---

## Key Data Structures

### `kernel_descriptor.h` — Code Object V3/V4 64-byte descriptor

```cpp
struct KernelDescriptor {            // Located at <kernel>.kd symbol in .rodata or .text
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint8_t  reserved0[4];
    int64_t  kernel_code_entry_byte_offset; // offset from .kd to first instruction (256 for gfx9)
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;      // ACCVGPR count (gfx90a)
    uint32_t compute_pgm_rsrc1;      // granulated VGPR/SGPR counts, float modes
    uint32_t compute_pgm_rsrc2;      // user SGPR count (bits[5:1]), workgroup ID enables
    uint16_t kernel_code_properties; // ENABLE_SGPR_KERNARG_SEGMENT_PTR etc.
    uint8_t  reserved2[6];
};
static_assert(sizeof(KernelDescriptor) == 64);
```

### `isa_decoder.h` — Instruction representation

```cpp
enum class InstFormat { SOP2, SOP1, SOPK, SOPC, SOPP, VOP2, VOP1, VOPC,
                        VOP3A, VOP3B, SMEM, FLAT_GLOBAL, UNKNOWN };
enum class Opcode {
    S_ADD_U32, S_AND_B32, S_AND_SAVEEXEC_B64, S_CBRANCH_EXECZ, S_BRANCH,
    S_LOAD_DWORD, S_LOAD_DWORDX2, S_LOAD_DWORDX4, S_WAITCNT, S_ENDPGM,
    V_MOV_B32, V_ADD_U32, V_ADD_CO_U32, V_ADDC_CO_U32,
    V_ASHRREV_I32, V_ASHRREV_I64, V_ADD_F32, V_CMP_LT_U32,
    GLOBAL_LOAD_DWORD, GLOBAL_STORE_DWORD, UNKNOWN
};
struct Operand { enum class Kind { SGPR, VGPR, LITERAL, INLINE_CONST, VCC, EXEC, SCC } kind;
                 uint32_t index; uint64_t imm; };
struct Instruction {
    InstFormat format; Opcode opcode;
    uint32_t   encoding[2]; uint8_t size_bytes;  // 4 or 8
    Operand    dst, dst2, src[3];
    int16_t    simm16;          // SOPP branch offset or SOPK imm
    uint8_t    wait_vmcnt, wait_lgkmcnt;
    int32_t    smem_offset, flat_offset;
};
Instruction decode(const uint8_t* pc);
```

### `wavefront.h` — Execution state

```cpp
struct WavefrontState {
    uint32_t sgpr[108];              // SGPRs (108 logical; VCC = sgpr[106:107])
    uint32_t vgpr[256][64];          // VGPRs: [reg_index][lane]
    uint64_t exec, vcc;
    uint32_t scc, m0;
    uint64_t pc;                     // byte offset into kernel_code_ptr
    const uint8_t* kernarg_ptr;
    const uint8_t* kernel_code_ptr;
    size_t   kernel_code_size;
    uint32_t workgroup_id[3];
    uint32_t workitem_id[64][3];     // per-lane thread ID
};
```

---

## Implementation Phases

### Phase 1 — ELF Loader (`src/elf_loader.cpp`)

- `mmap`/read HSACO into `vector<uint8_t>`
- Validate: `e_ident[EI_OSABI] == 64` (ELFOSABI_AMDGPU_HSA), `e_flags & 0xfff == 0x03f` (gfx90a)
- Code Object version from `e_ident[EI_ABIVERSION]`: 2=COV3, 3=COV4, 4=COV5
- Walk `Elf64_Shdr` entries: locate `.text`, `.rodata`, `.dynsym`/`.symtab`, `.note`, strtab
- Find kernel symbols: type `STT_AMDGPU_HSA_KERNEL` (= 10); symbol `<name>.kd` → 64-byte `KernelDescriptor`
- Kernel entry = `kd_ptr + kd.kernel_code_entry_byte_offset` (always 256 for gfx9)
- Parse `.note` YAML (yaml-cpp) for kernel arg types, sizes, offsets

Reference headers:
- `/opt/rocm-7.3.0/include/hsa/amd_hsa_elf.h`
- `/usr/include/elf.h`

**Test:** Load `vector_add_memory_fault_kernels.hsaco`, assert kernel found, descriptor parsed correctly.

### Phase 2 — ISA Decoder (`src/isa_decoder.cpp`)

Bit-pattern priority decode (top bits → format):

| Top bits           | Format       | Example instructions                              |
|--------------------|--------------|---------------------------------------------------|
| `[31:16]=0xBF8*`   | SOPP         | `s_waitcnt`, `s_endpgm`, `s_branch`, `s_cbranch_execz` |
| `[31:23]=0x5F`     | SOP1         | `s_and_saveexec_b64`                              |
| `[31:30]=0b10`     | SOP2         | `s_add_u32`, `s_and_b32`                          |
| `[31:26]=0b110000` | SMEM         | `s_load_dword/x2/x4` — 64-bit                    |
| `[31:26]=0b110100` | VOP3A/B      | `v_add_co_u32`, `v_addc_co_u32`, `v_ashrrev_i64` — 64-bit |
| `[31:25]=0b0111110`| VOPC         | `v_cmp_lt_u32` → writes VCC                       |
| `[31:25]=0b0111111`| VOP1         | `v_mov_b32`                                       |
| `[31:25]=0b0`      | VOP2         | `v_add_u32`, `v_add_f32`, `v_ashrrev_i32`         |
| `[31:26]=0b110111` | FLAT/GLOBAL  | `global_load/store_dword` — 64-bit                |

**Test:** Decode known byte sequences from disassembly, assert all fields.

### Phase 3 — Wavefront Executor (`src/isa_executor.cpp`, `src/wavefront.cpp`)

**SGPR initialization** (order fixed by ABI, read `kernel_code_properties` bits):
```
sgpr[0:1] = kernarg_segment_ptr   (if bit 1 of kernel_code_properties set)
sgpr[N]   = workgroup_id_x        (if bit 7 of compute_pgm_rsrc2 set)
```
**VGPR init:** `vgpr[0][lane] = lane` (workitem ID X within wave).

**EXEC mask:** All VALU/memory ops iterate all 64 lanes, skip where `(exec >> lane) & 1 == 0`.

**Memory ops** (direct host dereference):
```cpp
uint64_t addr = ((uint64_t)wave.vgpr[addr_hi][lane] << 32) | wave.vgpr[addr_lo][lane];
wave.vgpr[dst][lane] = *reinterpret_cast<const uint32_t*>(addr);
```

**`s_waitcnt`:** No-op (all memory is synchronous in simulation).

**Execution loop:**
```cpp
while (true) {
    Instruction inst = decode(wave.kernel_code_ptr + wave.pc);
    wave.pc += inst.size_bytes;
    execute_instruction(wave, inst);
    if (inst.opcode == Opcode::S_ENDPGM) break;
}
```

**Tricky cases:**
- `s_and_saveexec_b64`: save exec → dst SGPR pair, then `exec = exec & VCC`
- `v_cmp_lt_u32` (VOPC): writes per-lane comparison bitmask to `wave.vcc`
- `v_add_co_u32` (VOP3B): 64-bit encoding, writes carry-out to `dst2` (VCC or SGPR pair)

**Test:** Hand-assembled mini-kernels verifying each instruction family.

### Phase 4 — Public API & Execution Engine

**`gcnsim_api.cpp`** — `SimModule` holds parsed ELF; `SimFunction` is a named kernel handle.

**`execution_engine.cpp`** — Grid dispatch:
```
for (wg_x, wg_y, wg_z) in grid:
  wave_count = ceil(block_x * block_y * block_z / 64)
  for wave_in_wg in [0, wave_count):
    init_wavefront(...)   // set SGPRs, VGPRs, exec mask for partial last wave
    run_wavefront(...)
```

**`void*` kernel_params → kernarg buffer:** Lay out args at offsets from YAML metadata.

### Phase 5 — Vector Add Integration Test

**`tools/vector_add/vector_add.hip`:**
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
```
Compiled: `hipcc --offload-arch=gfx90a -c --genco vector_add.hip -o vector_add.hsaco`

**`tests/test_vector_add.cpp`:** N=1024 floats, launch with block=256, verify `|C[i] - (A[i]+B[i])| < 1e-5`.

---

## CMake Build

```cmake
cmake_minimum_required(VERSION 3.20)
project(gcn-sim VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(yaml-cpp REQUIRED)

add_library(gcnsim SHARED
    src/elf_loader.cpp src/metadata_parser.cpp
    src/isa_decoder.cpp src/isa_executor.cpp
    src/wavefront.cpp src/execution_engine.cpp src/gcnsim_api.cpp)

target_include_directories(gcnsim
    PUBLIC  include
    PRIVATE /opt/rocm-7.3.0/include)   # amd_hsa_elf.h, amd_hsa_kernel_code.h

target_link_libraries(gcnsim PRIVATE yaml-cpp)

if(GCNSIM_BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()
add_subdirectory(tools/vector_add)
```

---

## Critical Reference Files

| File | Purpose |
|------|---------|
| `/opt/rocm-7.3.0/include/hsa/amd_hsa_elf.h` | `EF_AMDGPU_MACH_AMDGCN_GFX90A`, `ELFOSABI_AMDGPU_HSA`, `STT_AMDGPU_HSA_KERNEL` |
| `/opt/rocm-7.3.0/include/hsa/amd_hsa_kernel_code.h` | `amd_kernel_code_t`, `AMD_COMPUTE_PGM_RSRC_*` bit enums |
| `/opt/rocm-7.3.0/include/hsa/hsa_ext_amd.h` | COV4 kernel descriptor layout reference |
| `/usr/include/elf.h` | `Elf64_Ehdr`, `Elf64_Shdr`, `Elf64_Sym` |
| `/opt/rocm-7.3.0/bin/gfx90a/vector_add_memory_fault_kernels.hsaco` | Reference binary to test ELF loader against |

---

## Verification Steps

1. `cmake -B build && cmake --build build`
2. `ctest --test-dir build -V` — all unit tests pass
3. `build/tests/test_vector_add` — integration test passes with correct float results
4. Cross-check with `llvm-objdump -d vector_add.hsaco` to verify decoded instructions match

---

## Notes on gfx90a ABI

- `EI_OSABI` = 64 (`ELFOSABI_AMDGPU_HSA`)
- `EI_ABIVERSION` = 2 → COV4, 3 → COV5
- `e_flags & 0xff` = `0x3f` → `EF_AMDGPU_MACH_AMDGCN_GFX90A`
- `STT_AMDGPU_HSA_KERNEL` = `STT_LOOS + 0` = 10
- Kernel descriptor symbol: `<kernel_name>.kd` (suffix `.kd`)
- `kernel_code_entry_byte_offset` = 256 for all gfx9 targets (KD is 64 bytes, followed by 192 bytes padding, then instructions)
- Wavefront size = 64 lanes
- VCC lives in `sgpr[106:107]` (for 108-SGPR allocation)
- `kernel_code_properties` bit 1: `ENABLE_SGPR_KERNARG_SEGMENT_PTR`
- `compute_pgm_rsrc2` bit 7: `ENABLE_SGPR_WORKGROUP_ID_X`
