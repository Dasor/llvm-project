#!/usr/bin/env python3
"""Break down a Snitch hart trace from a run_gemm.py Verilator run.

    trace_breakdown.py <elf> <trace_hart_NNNNN.dasm> [--steps] [--M M --N N --K K]

Reports, for the traced hart:
  * cycles per ELF function (where the time goes: runtime startup, harness,
    the kernel, barriers, ...);
  * the kernel window (first to last instruction of 'matmul$dma' on the DMA
    hart, or 'matmul' on a compute hart);
  * on the DMA hart, the barrier structure of the kernel (software barriers,
    snrt_partial_barrier, or hardware barriers, csrr 0x7c2): the long barrier
    waits are the compute steps (the DMA hart waits for the compute cores),
    the short ones the synchronization/fill/drain gaps between them.
    With --M/--N/--K, also the ideal FMA time (M*N*K / 8 compute cores).

Reads the raw .dasm (no gen_trace.py needed), so it works on the
TRACE_DMA_ONLY Verilator build's trace_hart_00008.dasm. Run with
build/.venv/bin/python3 (needs pyelftools).
"""
import argparse
import bisect
from collections import defaultdict

from elftools.elf.elffile import ELFFile

COMPUTE_CORES = 8
# A barrier wait longer than this is a compute step (the DMA hart waiting for
# the compute cores), shorter ones are synchronization gaps.
COMPUTE_STEP_THRESHOLD = 2000


def functions(elf):
    syms = []
    with open(elf, "rb") as f:
        for s in ELFFile(f).get_section_by_name(".symtab").iter_symbols():
            if s["st_info"]["type"] == "STT_FUNC" and s["st_size"] > 0:
                syms.append((s["st_value"], s["st_value"] + s["st_size"], s.name))
    syms.sort()
    return syms


# csrr x0, 0x7c2: the cluster hardware barrier (snrt_cluster_hw_barrier, and
# snitch.barrier lowered with hw-barrier=true).
HW_BARRIER = "DASM(7c202073)"


def trace(dasm):
    """Yields (cycle, pc, is_hw_barrier) of every retired integer-core
    instruction."""
    with open(dasm) as f:
        for line in f:
            fields = line.split(None, 5)
            if "z" in fields[3]:  # FPSS-only lines carry no PC
                continue
            yield int(fields[1]), int(fields[3], 16), fields[4] == HW_BARRIER


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("elf")
    p.add_argument("dasm")
    p.add_argument("--steps", action="store_true",
                   help="print every barrier interval")
    p.add_argument("--top", type=int, default=10, help="functions to list")
    for dim in "MNK":
        p.add_argument(f"--{dim}", type=int)
    args = p.parse_args()

    syms = functions(args.elf)
    starts = [s[0] for s in syms]

    def function_of(pc):
        i = bisect.bisect_right(starts, pc) - 1
        return syms[i][2] if i >= 0 and pc < syms[i][1] else "?"

    per_function = defaultdict(int)
    kernel = None  # [first, last] cycle inside the kernel function
    barrier_enters, barrier_exits = [], []
    in_barrier = in_hw_barrier = False
    prev_cycle = prev_function = None
    for cycle, pc, hw_barrier in trace(args.dasm):
        name = function_of(pc)
        # The trace has the barrier csrr when it issues: the wait is the time
        # until the next instruction.
        if in_hw_barrier:
            barrier_exits.append(cycle)
        in_hw_barrier = hw_barrier and kernel is not None
        if in_hw_barrier:
            barrier_enters.append(cycle)
        if prev_function is not None:
            per_function[prev_function] += cycle - prev_cycle
        if name in ("matmul$dma", "matmul"):
            kernel = kernel or [cycle, cycle]
            kernel[1] = cycle
        is_barrier = name == "snrt_partial_barrier"
        if is_barrier and not in_barrier:
            barrier_enters.append(cycle)
        if in_barrier and not is_barrier:
            barrier_exits.append(cycle)
        in_barrier = is_barrier
        prev_cycle, prev_function = cycle, name

    total = sum(per_function.values())
    print(f"traced span: {total} cycles")
    for name, cycles in sorted(per_function.items(), key=lambda x: -x[1])[:args.top]:
        print(f"  {cycles:>10} ({100 * cycles / total:5.1f}%)  {name}")
    if not kernel:
        print("no 'matmul$dma'/'matmul' instructions in this trace")
        return
    window = kernel[1] - kernel[0]
    print(f"kernel window: {kernel[0]}..{kernel[1]} = {window} cycles")

    steps, gaps = [], []
    prev = kernel[0]
    for i, (enter, exit_) in enumerate(zip(barrier_enters, barrier_exits)):
        if not kernel[0] <= enter <= kernel[1]:
            continue
        interval = exit_ - prev
        (steps if exit_ - enter > COMPUTE_STEP_THRESHOLD else gaps).append(interval)
        if args.steps:
            print(f"  barrier {i:4d}: wait {exit_ - enter:>8}, since previous {interval:>8}")
        prev = exit_
    if steps:
        print(f"compute steps: {len(steps)}, {sum(steps)} cycles "
              f"(avg {sum(steps) / len(steps):.0f})")
        print(f"sync/fill/drain gaps: {len(gaps)}, {sum(gaps)} cycles "
              f"({100 * sum(gaps) / window:.1f}% of the kernel)")
    if args.M and args.N and args.K:
        ideal = args.M * args.N * args.K // COMPUTE_CORES
        print(f"ideal FMA time: {ideal} cycles ({100 * ideal / window:.1f}% of the kernel)")
        if steps:
            print(f"compute steps over ideal (microkernel overhead + padding): "
                  f"{sum(steps) - ideal} cycles")


if __name__ == "__main__":
    main()
