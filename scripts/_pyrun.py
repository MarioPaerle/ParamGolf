#!/usr/bin/env python3
"""Helper to run shell commands when interactive bash is restricted."""
import subprocess
import sys

cmd = sys.argv[1:]
res = subprocess.run(cmd, capture_output=True, text=True)
sys.stdout.write(res.stdout)
sys.stderr.write(res.stderr)
sys.exit(res.returncode)
