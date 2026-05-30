# gcndb Debugger Usage Guide

This guide describes how to use `gcndb` (GCN GPU ISA debugger) to debug HSACO kernel binaries and hijacked host applications.

---

## 1. CLI/Standalone Debugging Mode

You can run `gcndb` directly from the command line in interactive REPL mode:

```bash
# Load an HSACO file directly and start the debugger REPL
./build/gcndb build/tools/vector_add/vector_add.hsaco
```

### Essential CLI Commands
Once inside the `(gdb)` prompt, you can use standard GDB commands:

* **Breakpoints**:
  * `break *0x10` (or `b *16`): Set a breakpoint at instruction byte offset 16.
  * `break vector_add.hip:24` (or `b vector_add.hip:24`): Set a breakpoint at a source file and line number.
  * `info breakpoints` (or `i b`): List all set breakpoints.
  * `delete <id>` (or `d <id>`): Delete a breakpoint by its number.
* **Stepping & Execution**:
  * `run` (or `r`): Start executing the kernel.
  * `stepi` (or `si`): Step exactly one ISA instruction.
  * `step` (or `s`): Step to the next source code line.
  * `continue` (or `c`): Resume execution until the next breakpoint or completion.
* **Inspecting Registers & Memory**:
  * `info registers` (or `i r`): Print values of all non-zero SGPRs, VGPRs (at the active lane), AGPRs (at the active lane), and special registers (PC, EXEC, VCC, SCC, M0).
  * `print $s0` (or `p $s0`): Print the value of scalar register `s0`.
  * `print $v0` (or `p $v0`): Print the value of vector register `v0` at the **active lane**.
  * `print $v0[15]`: Print the value of vector register `v0` specifically for lane 15.
  * `print $v0[all]`: Print all 64 lanes of vector register `v0`.
  * `set lane 5` (or `$lane = 5`): Set the active lane index for vector register queries to 5.
  * `x/4xw 0x7fffffffd000`: Examine memory at the specified address (4 hexadecimal words).
  * `disassemble` (or `disas`): Disassemble 32 bytes of instructions around the current PC.
* **Inspecting Threads (Wavefronts)**:
  * `info threads` (or `i thread`): List all active wavefront threads, showing their Wavefront ID, Workgroup, and current PC.
  * `thread <id>` (or `t <id>`): Switch active focus to the specified wavefront thread.

---

## 2. Using gcndb with cgdb (Curses Frontend)

`cgdb` provides a split-screen curses interface showing your source code alongside the debugger console. You can launch it using `gcndb` as the debugger backend:

```bash
# Launch cgdb wrapping gcndb to debug a host application
cgdb -d ./build/gcndb --args ./build/tests/real_rocm_app
```

1. Press `Esc` to enter the source view, where you can navigate the code and set breakpoints using `Space`.
2. Press `i` to return to the interactive console prompt.
3. Type `run` (or `r`) to start the application. The hijacked library will intercept the kernel launch and pause at the beginning of kernel execution, displaying the source code position in the upper pane.

---

## 3. Using gcndb with VS Code

You can configure VS Code's C/C++ debugger extension to use `gcndb` as its GDB/MI debugger path. This enables graphical breakpoints, variables inspection, registers pane, call stacks, and tooltips.

### VS Code `launch.json` Configuration
Add the following configuration to your `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "gcndb - Hijacked ROCm App",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/tests/real_rocm_app",
            "args": [],
            "stopAtEntry": true,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "${workspaceFolder}/build/gcndb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

### VS Code Usage Tips
* **Breakpoints**: Click to the left of the line numbers in your `.hip`, `.cpp`, or `.s` files to set breakpoints graphically before running.
* **Threads View**: Each wavefront is mapped to a thread in the Call Stack pane. Select a thread to view its PC and registers.
* **Variables Pane**: Displays the arguments passed to the kernel (e.g., `A`, `B`, `C`, `N`) along with their actual values (pointers or scalars).
* **Registers Pane**: Displays all scalar, vector (for the active lane), accumulator, and special registers.
* **Debug Console**: You can execute CLI commands directly by prefixing them with `-exec`, for example:
  * `-exec set lane 12` (switches registers display to lane 12).
  * `-exec print $v0[all]` (prints all 64 lanes of `v0`).

---

## 4. Debugging Dynamically Loaded Code Objects (Tensile Assembly Kernels)

You can debug dynamically loaded AMDGCN assembly code objects (like `tensile_style_gemm.co`) either standalone or wrapped in client applications.

### A. Assembly Debug Symbols Compilation
To enable line-level breakpoints and stepping within assembly files (e.g. `tensile_style_gemm.s`), compile them with the **`-g`** flag:
```bash
# Assemble with debug symbols
clang -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx90a -g -c -o tensile_style_gemm.o tensile_style_gemm.s
# Link
clang -target amdgcn-amd-amdhsa -mcpu=gfx90a tensile_style_gemm.o -o tensile_style_gemm.co
```

### B. Debugging Client Applications
If your host application (client app) dynamically loads code objects at runtime using HIP Module APIs:
```cpp
hipModuleLoad(&module, "path/to/tensile_style_gemm.co");
hipModuleGetFunction(&function, module, "generated_gemm");
hipModuleLaunchKernel(function, ...);
```
You can run and debug the client application under `gcndb`:
```bash
# Debug the client application
./build/gcndb --args ./build/tests/tensile_gemm_client --co build/tools/tensile_style_gemm/tensile_style_gemm.co
```

When `gcndb` stops at the launch of `generated_gemm`, you can step through the assembly instructions, set breakpoints on assembly source lines, and inspect ACC/VGPR/SGPR registers.
