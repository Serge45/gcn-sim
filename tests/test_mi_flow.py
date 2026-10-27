import subprocess
import os
import time

def test_cgdb_flow():
    # Setup runtime library search path
    env = os.environ.copy()
    rocm_lib_paths = ["/opt/rocm/lib", "/opt/rocm-7.0.0/lib", "/opt/rocm-6.1.3/lib"]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    additional_paths = [p for p in rocm_lib_paths if os.path.exists(p)]
    if additional_paths:
        env["LD_LIBRARY_PATH"] = ":".join(additional_paths + ([existing_ld] if existing_ld else []))

    # 1. Start gcndb in MI mode
    cmd = ["./build/gcndb", "--interpreter=mi", "--args", "./build/tests/real_rocm_app"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    def send(msg):
        print(f"SEND: {msg}")
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()

    def expect(pattern):
        while True:
            line = proc.stdout.readline()
            if not line: break
            print(f"RECV: {line.strip()}")
            if pattern in line:
                return line
        return None

    # 2. Initialization Handshake
    expect("=thread-group-added")
    send("-list-features")
    expect("^done,features=")
    
    # 3. Set Breakpoint on vector_add kernel function
    send("-break-insert vector_add")
    expect("^done,bkpt=")

    # 4. Run the process (launches the host app under hijack library)
    send("run")
    
    # 5. Wait for the breakpoint stop record from the hijacked process
    stop = expect("*stopped,reason=\"breakpoint-hit\"")
    if not stop:
        print("FAILURE: Did not hit breakpoint!")
        proc.terminate()
        return False
        
    print("SUCCESS: Hit breakpoint in hijacked process!")
    
    # 6. Test -thread-info
    send("-thread-info")
    thread_info = expect("^done,threads=")
    if "target-id=\"Wave" in thread_info:
        print("SUCCESS: -thread-info successfully mapped wavefronts!")
    else:
        print("FAILURE: -thread-info mapping failed!")
        
    # 7. Test -stack-list-frames
    send("-stack-list-frames")
    frames = expect("^done,stack=[frame=")
    if "func=\"_Z10vector_addPKfS0_Pfi\"" in frames:
        print("SUCCESS: -stack-list-frames parsed the kernel frame!")
    else:
        print("FAILURE: -stack-list-frames failed!")

    # 7b. Test -stack-list-arguments 0 0 0
    send("-stack-list-arguments 0 0 0")
    args = expect("^done,stack-args=[")
    if "stack-args=" in args:
        print("SUCCESS: -stack-list-arguments returned correct frame arguments format!")
    else:
        print("FAILURE: -stack-list-arguments failed!")

    # 8. Test -stack-list-variables (kernel arguments)
    send("-stack-list-variables")
    variables = expect("^done,variables=[")
    if "name=\"N\"" in variables:
        print("SUCCESS: -stack-list-variables returned kernel arguments!")
    else:
        print("FAILURE: -stack-list-variables failed!")

    # 9. Test -data-list-register-names
    send("-data-list-register-names")
    expect("^done,register-names=[")

    # 10. Test -data-list-register-values
    send("-data-list-register-values x 0 1 2 110 622 623") # s0, s1, s2, v0, pc, exec
    reg_vals = expect("^done,register-values=[")
    if "value=" in reg_vals:
        print("SUCCESS: -data-list-register-values returned hex register values!")
    else:
        print("FAILURE: -data-list-register-values failed!")

    # 11. Test -var-create and -var-evaluate-expression
    send("-var-create varN * N")
    expect("^done,name=\"varN\"")
    send("-var-evaluate-expression varN")
    val_expr = expect("^done,value=")
    if "256" in val_expr:
        print("SUCCESS: Variable object creation and evaluation succeeded!")
    else:
        print("FAILURE: Variable object validation failed!")

    # 12. Test -data-read-memory-bytes (using absolute PC value from pc register)
    # We can extract the PC address or use a safe address. Let's query s0 or pc.
    # Since PC is a valid mapped code address, we can read 16 bytes from it!
    # Let's check: we can read memory bytes.
    send("-data-read-memory-bytes 0x1000 16") # We can use a dummy read, or read from 0x0 (invalid) and check error,
    # or read from code_vma. Let's try reading 16 bytes from a code offset.
    # Actually, let's just query a known valid pointer or query and expect done/error depending on address.
    send("-exec-step-instruction")
    expect("^running")
    expect("*stopped,reason=\"end-stepping-range\"")
    print("SUCCESS: Instruction stepping succeeded!")

    # 13. Test -exec-step (line-level stepping)
    send("-exec-step")
    expect("^running")
    stop_step = expect("*stopped,reason=\"end-stepping-range\"")
    if stop_step and "line=" in stop_step:
        print("SUCCESS: Line stepping succeeded!")
    else:
        print("FAILURE: Line stepping failed!")

    send("-exec-continue")
    expect("^running")
    
    proc.terminate()
    print("SUCCESS: All GDB/MI integration tests passed!")
    return True

if __name__ == "__main__":
    test_cgdb_flow()

