#!/usr/bin/env python3
"""Extract snitch.riscv_assembly string attributes from an MLIR module and
write each to its own .S file. Mirrors QuidditchTarget.cpp::assembleXDSLOutput
without requiring a custom target backend.
"""

import re
import sys

mlir_path, out_prefix = sys.argv[1], sys.argv[2]
text = open(mlir_path).read()

count = 0
for m in re.finditer(r'snitch\.riscv_assembly = "((?:[^"\\]|\\.)*)"', text):
    raw = m.group(1)
    # MLIR string attrs escape newlines as \0A and quotes/backslashes with \\.
    unescaped = re.sub(r'\\0A', '\n', raw)
    unescaped = re.sub(r'\\22', '"', unescaped)
    unescaped = unescaped.replace('\\\\', '\\')
    out_path = f"{out_prefix}{count}.S"
    with open(out_path, "w") as f:
        f.write(unescaped)
    print(f"wrote {out_path}")
    count += 1

if count == 0:
    print("no snitch.riscv_assembly attributes found", file=sys.stderr)
    sys.exit(1)
