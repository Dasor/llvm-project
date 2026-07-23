#!/usr/bin/env python3
"""Text-templating autotuning driver.

This is the reusable scaffold answering "how are parameters derived":

  1. READ an arbitrary .mlir schedule file and EXTRACT every unresolved
     `transform.tune.knob<"name"> options = [...]` in it via a regex --
     there is no MLIR-IR-level API for this (this build has
     MLIR_ENABLE_BINDINGS_PYTHON=0, and even with bindings there is no
     built-in "walk and resolve" helper; see README.md). Knobs that
     already carry a `selected` value are left untouched -- they're
     treated as fixed, not searched over.
  2. ENUMERATE candidates over the extracted knobs (exhaustive grid search
     by default, optionally filtered by a --constraint expression).
  3. For each candidate, RENDER the schedule with those knobs resolved to
     `= <value> from options = [...]` -- the only mechanism the dialect
     actually supports (see ../schedules/knobs_unresolved.mlir and
     ../schedules/knobs_invalid_selection.mlir for the two ways an
     unresolved/invalid schedule is rejected).
  4. COMPILE it ahead-of-time to a standalone native executable via
     compile_schedule.sh (mlir-opt --transform-interpreter + full CPU
     lowering pipeline + mlir-translate + llc + a system C compiler) --
     NOT timed.
  5. RUN that executable (no mlir-opt, no JIT) and time only that, taking
     the min across a few repeats; see README.md's timing note for why the
     compile step above is deliberately excluded from this measurement.
  6. MEASURE correctness (every printed number equals --expect-uniform-value,
     if given) and kernel time.
  7. RECORD every candidate to a CSV log, and WRITE the fastest correct
     candidate's fully-resolved schedule out to a standalone .mlir file.

To plug in a different/custom search algorithm: keep steps 3-7 (render /
compile / run / measure / record) and replace step 2's exhaustive
itertools.product() with your own candidate-proposal loop. Everything
downstream only depends on `knobs` (extracted from the file) and a
candidate being a dict of {knob_name: chosen_option_token}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

DRIVER_DIR = Path(__file__).resolve().parent
POC_DIR = DRIVER_DIR.parent

sys.path.insert(0, str(POC_DIR))
from mlir_schedule import (  # noqa: E402
    detect_matmul_dims, extract_knobs, render_schedule, run_candidate, token_to_python,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "schedule",
        nargs="?",
        type=Path,
        default=POC_DIR / "schedules" / "tunable_matmul.mlir",
        help="Path to a .mlir file containing a payload + transform schedule "
        "with one or more unresolved transform.tune.knob ops. "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--constraint",
        default=None,
        help="Python boolean expression over knob names plus M/N/K (auto-detected "
        "from the schedule's own linalg.matmul op, if present), "
        'evaluated per candidate to filter the search space, e.g. '
        '"tile_m <= M and tile_n <= N and tile_k <= K". No filtering by default.',
    )
    p.add_argument(
        "--expect-uniform-value",
        default=None,
        help="If set, a candidate is only 'correct' when every printed "
        "numeric value in its output equals this. Otherwise correctness is "
        "just 'the pipeline exited 0'.",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=POC_DIR / "results" / "grid_search.csv",
        help="Where to write the per-candidate CSV log.",
    )
    p.add_argument(
        "--best-schedule",
        type=Path,
        default=POC_DIR / "results" / "best_schedule.mlir",
        help="Where to write the fully-resolved schedule for the best "
        "candidate found.",
    )
    p.add_argument(
        "--compile-script",
        type=Path,
        default=DRIVER_DIR / "compile_schedule.sh",
        help="Script that ahead-of-time compiles one fully-resolved schedule "
        "to a standalone native executable (see README.md's timing note).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    schedule_text = args.schedule.read_text()
    all_knobs = extract_knobs(schedule_text)
    tunable = [k for k in all_knobs if not k.already_resolved]
    fixed = [k for k in all_knobs if k.already_resolved]

    if not tunable:
        sys.exit(
            f"No unresolved transform.tune.knob ops found in {args.schedule}. "
            "Nothing to tune."
        )

    print(f"Found {len(tunable)} tunable knob(s) in {args.schedule}:")
    for k in tunable:
        print(f"  {k.name}: options = {k.options}")
    if fixed:
        print(f"({len(fixed)} already-resolved knob(s) left untouched: "
              f"{', '.join(k.name for k in fixed)})")

    names = [k.name for k in tunable]
    option_lists = [k.options for k in tunable]
    total = 1
    for opts in option_lists:
        total *= len(opts)

    gemm_dims = {}
    if args.constraint is not None:
        try:
            gemm_dims = detect_matmul_dims(schedule_text)
        except ValueError:
            pass  # let eval() below raise its own NameError if the constraint actually needs M/N/K

    candidates = []
    for combo in itertools.product(*option_lists):
        candidate = dict(zip(names, combo))
        if args.constraint is not None:
            values = {n: token_to_python(t) for n, t in candidate.items()}
            if not eval(args.constraint, {"__builtins__": {}}, {**gemm_dims, **values}):
                continue
        candidates.append(candidate)

    print(
        f"\n{len(candidates)} candidate(s) to evaluate "
        f"(out of {total} total combinations)"
        + (f", filtered by: {args.constraint}" if args.constraint else "")
    )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.results.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names + ["kernel_time_s", "correct"])
        for i, candidate in enumerate(candidates, 1):
            result = run_candidate(
                schedule_text, candidate, args.compile_script, args.expect_uniform_value
            )
            kernel_str = f"{result.kernel_time_s:.6f}" if result.kernel_time_s is not None else ""
            writer.writerow([candidate[n] for n in names] + [kernel_str, result.correct])
            f.flush()
            status = "ok" if result.correct else "FAILED"
            desc = " ".join(f"{n}={candidate[n]}" for n in names)
            time_desc = (
                f"{result.kernel_time_s * 1e3:8.4f}ms" if result.kernel_time_s is not None else "     n/a"
            )
            print(f"[{i}/{len(candidates)}] {desc}  kernel={time_desc}  {status}")
            if not result.correct and result.stderr:
                sys.stderr.write(result.stderr)
            results.append((candidate, result.kernel_time_s, result.correct))

    correct_results = [r for r in results if r[2] and r[1] is not None]
    if not correct_results:
        sys.exit("\nNo candidate produced a correct result with a measurable kernel time.")

    best_candidate, best_time, _ = min(correct_results, key=lambda r: r[1])
    best_desc = " ".join(f"{n}={best_candidate[n]}" for n in names)
    print(f"\nBest candidate: {best_desc} ({best_time * 1e3:.4f} ms)")

    args.best_schedule.parent.mkdir(parents=True, exist_ok=True)
    args.best_schedule.write_text(render_schedule(schedule_text, best_candidate))
    print(f"Best schedule written to: {args.best_schedule}")
    print(f"Full log: {args.results}")


if __name__ == "__main__":
    main()
