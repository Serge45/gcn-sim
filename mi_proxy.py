#!/usr/bin/env python3
import sys
import subprocess
import threading
import os

# Log file paths
GDB_IN = "gdb_in.log"
GDB_OUT = "gdb_out.log"

# Clear old logs
if os.path.exists(GDB_IN): os.remove(GDB_IN)
if os.path.exists(GDB_OUT): os.remove(GDB_OUT)

cmd = ["./build/gcndb"] + sys.argv[1:]
# Use os.pipe to get unbuffered access
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0)

def handle_in():
    with open(GDB_IN, "wb") as f:
        while True:
            char = sys.stdin.buffer.read(1)
            if not char: break
            proc.stdin.write(char)
            proc.stdin.flush()
            f.write(char)
            f.flush()

def handle_out():
    with open(GDB_OUT, "wb") as f:
        while True:
            char = proc.stdout.read(1)
            if not char: break
            sys.stdout.buffer.write(char)
            sys.stdout.buffer.flush()
            f.write(char)
            f.flush()

t1 = threading.Thread(target=handle_in, daemon=True)
t2 = threading.Thread(target=handle_out, daemon=True)
t1.start()
t2.start()

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
