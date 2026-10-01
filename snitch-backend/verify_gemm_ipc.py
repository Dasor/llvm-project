#!/usr/bin/env python3
"""Run a gemm_main.c ELF on the Verilator model and verify C on the host.

    verify_gemm_ipc.py <sim_bin> <elf> <M> <N> <K> [--mismatch-csv PATH]

Same approach as snitch_cluster's Verifier (util/sim/verif_utils.py), used by
myrtle-experiments/many_gemms.sh: the simulator is started with the
testbench's IPC interface (two FIFOs, see target/sim/tb/ipc.cc); once the
program has written `tohost` we read `gemm_c` out of simulated memory and
compare it against numpy. Verification therefore costs no simulated cycles,
unlike gemm_main.c's -DVERIFY reference loop on the DM core.

Only SnitchSim.py is reused from snitch_cluster: verif_utils/Elf.py import
data_utils, which pulls in torch and pyflexfloat. Needs numpy and pyelftools
(both in build/.venv). Called by run_gemm.py; the simulator's stdout is
inherited, so it ends up in run_gemm.py's run.log.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from elftools.elf.elffile import ELFFile

ELEM = np.dtype("<f8")


def load_snitch_sim():
    path = Path(os.environ["SNITCH_CLUSTER_ROOT"], "util", "sim", "SnitchSim.py")
    spec = importlib.util.spec_from_file_location("SnitchSim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SnitchSim


def symbols(elf, names):
    """{name: (address, size)} for each requested symbol in elf's .symtab."""
    with open(elf, "rb") as f:
        symtab = ELFFile(f).get_section_by_name(".symtab")
        if symtab is None:
            sys.exit(f"[!] {elf} has no symbol table")
        out = {}
        for name in names:
            syms = symtab.get_symbol_by_name(name)
            if not syms or len(syms) != 1:
                sys.exit(f"[!] expected exactly one '{name}' symbol in {elf}, "
                         f"found {len(syms or [])}")
            out[name] = (syms[0].entry["st_value"], syms[0].entry["st_size"])
        return out


def expected_result(M, N, K):
    # Mirrors the DM core's input initialization in kernels/gemm_main.c.
    a = (np.arange(M * K) % 7 + 1).astype(ELEM).reshape(M, K)
    b = (np.arange(K * N) % 5 + 1).astype(ELEM).reshape(K, N)
    return a @ b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sim_bin")
    p.add_argument("elf")
    p.add_argument("M", type=int)
    p.add_argument("N", type=int)
    p.add_argument("K", type=int)
    p.add_argument("--mismatch-csv", type=Path,
                   help="write every mismatching element here on failure")
    args = p.parse_args()
    M, N, K = args.M, args.N, args.K

    syms = symbols(args.elf, ["tohost", "gemm_c"])
    c_addr, c_size = syms["gemm_c"]
    if c_size != M * N * ELEM.itemsize:
        sys.exit(f"[!] gemm_c is {c_size} bytes, expected {M * N * ELEM.itemsize} "
                 f"for M={M} N={N}")

    # The simulator can only exit once we close the FIFOs in finish(), so the
    # memory read after poll() sees the final state. If it dies early,
    # SnitchSim's monitor thread interrupts us and exits with status 1.
    sim = load_snitch_sim()(args.sim_bin, args.elf)
    sim.start()
    sim.poll(syms["tohost"][0], 1, 0)
    raw = sim.read(c_addr, c_size)
    proc = sim.sim
    sim.finish(wait_for_sim=True)
    sys.stdout.flush()
    if proc.returncode != 0:
        print(f"[!] simulation exited with status {proc.returncode}", flush=True)
        return proc.returncode

    got = np.frombuffer(raw, dtype=ELEM).reshape(M, N)
    expected = expected_result(M, N, K)
    # Integer-valued inputs: every partial sum is exact in f64, so compare
    # exactly, like gemm_main.c's reference check.
    bad = np.argwhere(got != expected)
    if len(bad) == 0:
        print(f"all {M * N} values matched", flush=True)
        return 0

    print(f"{len(bad)} of {M * N} values mismatched", flush=True)
    for i, j in bad[:10]:
        print(f"mismatch at ({i},{j}): got {got[i, j]} expected {expected[i, j]}")
    if args.mismatch_csv:
        with open(args.mismatch_csv, "w") as f:
            f.write("i,j,got,expected\n")
            for i, j in bad:
                f.write(f"{i},{j},{float(got[i, j])!r},{float(expected[i, j])!r}\n")
        print(f"all mismatches written to {args.mismatch_csv}")
    sys.stdout.flush()
    return 1


if __name__ == "__main__":
    sys.exit(main())
