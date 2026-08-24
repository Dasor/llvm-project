#!/usr/bin/env python3
"""Tune tile_m/tile_n/tile_k transform.tune.knob values for the Snitch
matmul pipeline with the GBDT+GA search from ../tuner_framework, the same
search strategy ../mlir_tuner/run_gbdt_tuner.py uses for the CPU matmul PoC
-- only the backend differs (see backend.py's module docstring for exactly
what and why).

Two modes, via --mode:
  pipelined (default) -- Uses double buffering which restrict the search space to tile sizes 
  that evenly divide M/N/K
  padded -- Uses transform.structured.pad to handle any remainder

Usage:
    python3 snitch_tuner/run_gbdt_tuner.py \\
        --iterations 4 --init-samples 4 --ga-candidates-per-iter 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_POC_DIR = _HERE.parent
sys.path.insert(0, str(_POC_DIR))

import pandas as pd  # noqa: E402

from mlir_schedule import detect_matmul_dims, extract_knobs, render_schedule  # noqa: E402
from mlir_tuner.backend import KnobSearchSpace, detect_dtype_bytes  # noqa: E402
from snitch_tuner.backend import (  # noqa: E402
    SnitchGemmKnobFeatureExtractor, SnitchScheduleEvaluator, max_l1_tile_dim,
    snitch_divisor_sample_value, snitch_padded_tile_constraint,
    snitch_padded_tile_sample_value, snitch_padded_search_space_size,
    snitch_search_space_size, snitch_tile_constraint,
)
from tuner_framework import Tuner, TunerConfig  # noqa: E402

# Per-mode defaults: (schedule path, run-script path). Both relative to _HERE.
_MODE_DEFAULTS = {
    "pipelined": ("schedules/tile_l1_pipelined_tunable.mlir", "driver/tune_candidate.sh"),
    "padded": ("schedules/tile_l1_multicore_padded_tunable.mlir", "driver/tune_candidate_padded.sh"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode", choices=sorted(_MODE_DEFAULTS), default="padded",
        help= "pipelined or padded"
    )
    p.add_argument(
        "schedule", nargs="?", type=Path, default=None,
        help="Path to the .mlir transform schedule with unresolved "
        "transform.tune.knob ops. (default: depends on --mode)",
    )
    p.add_argument(
        "--payload", type=Path, default=_HERE / "payload" / "matmul_128.mlir",
        help="Path to the matmul payload module -- M/N/K are detected from "
        "here, not from the schedule file. (default: %(default)s)",
    )
    p.add_argument("--results", type=Path, default=_HERE / "results" / "gbdt_search.csv")
    p.add_argument("--best-schedule", type=Path, default=_HERE / "results" / "best_schedule_gbdt.mlir")
    p.add_argument(
        "--run-script", type=Path, default=None,
        help="Script that compiles one fully-resolved schedule and runs it "
        "on gvsoc. (default: depends on --mode)",
    )
    p.add_argument(
        "--timeout-s", type=float, default=300.0,
        help="Hard wall-clock timeout (with process-group kill) per candidate "
        "-- see SnitchScheduleEvaluator's docstring for why this isn't "
        "optional. (default: %(default)s)",
    )

    p.add_argument("--iterations", type=int, default=10, help="outer iterations (default: %(default)s)")
    p.add_argument("--init-samples", type=int, default=12, help="random samples in iteration 1 (default: %(default)s)")
    p.add_argument("--ga-population", type=int, default=100, help="GA population size (default: %(default)s)")
    p.add_argument("--ga-generations", type=int, default=100, help="GA generations (default: %(default)s)")
    p.add_argument("--ga-candidates-per-iter", type=int, default=12,
                    help="how many of the GA's top candidates get really evaluated per iteration "
                    "(default: %(default)s)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--verbose", action="store_true",
        help="Print each candidate as it starts and finishes running, in addition "
        "to the per-iteration/per-GA-generation summaries.",
    )
    p.add_argument(
        "--keep-workdirs", action="store_true",
        help="Don't delete each candidate's compile/simulate working directory "
        "afterward -- useful for debugging a failing or timed-out candidate by "
        "hand. Off by default since each one holds a full compiled ELF plus "
        "gvsoc logs.",
    )
    args = p.parse_args()

    default_schedule, default_run_script = _MODE_DEFAULTS[args.mode]
    if args.schedule is None:
        args.schedule = _HERE / default_schedule
    if args.run_script is None:
        args.run_script = _HERE / default_run_script
    return args


def main() -> None:
    args = parse_args()

    schedule_text = args.schedule.read_text()
    payload_text = args.payload.read_text()
    all_knobs = extract_knobs(schedule_text)
    tunable = [k for k in all_knobs if not k.already_resolved]
    fixed = [k for k in all_knobs if k.already_resolved]

    if not tunable:
        sys.exit(f"No unresolved transform.tune.knob ops found in {args.schedule}. Nothing to tune.")

    print(f"Mode: {args.mode}")
    print(f"Found {len(tunable)} tunable knob(s) in {args.schedule} (their "
          "file-declared options below are informational only -- NOT used "
          "to constrain the search; only the sample_value function is):")
    for k in tunable:
        print(f"  {k.name}: options = {k.options}")
    if fixed:
        print(f"({len(fixed)} already-resolved knob(s) left untouched: "
              f"{', '.join(k.name for k in fixed)})")

    dims = detect_matmul_dims(payload_text)
    elem_bytes = detect_dtype_bytes(payload_text, default="f64")
    max_tile = max_l1_tile_dim(elem_bytes)
    print(f"Detected GEMM dimensions from {args.payload}'s own linalg.matmul op: "
          f"M={dims['M']}, N={dims['N']}, K={dims['K']}, element size {elem_bytes} bytes.")

    if args.mode == "pipelined":
        print("tile_m/tile_n/tile_k must each evenly divide M/N/K (dual-buffered "
              "pipelining hard-crashes on padded/remainder tiles today"
              "tile_m is additionally"
              "restricted to multiples of 8")
        sample_value = lambda name, rng: snitch_divisor_sample_value(name, rng, dims, elem_bytes)  # noqa: E731
        constraint = snitch_tile_constraint(dims)
        max_candidates = snitch_search_space_size(dims, elem_bytes)
    else:
        print("tile_m/tile_n/tile_k do NOT need to divide M/N/K evenly -- "
              "transform.structured.pad handles any remainder "
              ". tile_m is still "
              "restricted to multiples of 8 (barrier-participant safety -- NOT "
              "relaxed by padding).")
        sample_value = lambda name, rng: snitch_padded_tile_sample_value(name, rng, dims, elem_bytes)  # noqa: E731
        constraint = snitch_padded_tile_constraint(dims)
        max_candidates = snitch_padded_search_space_size(dims, elem_bytes)

    search_space = KnobSearchSpace(tunable, sample_value, constraint=constraint)
    features = SnitchGemmKnobFeatureExtractor(payload_text)
    evaluator = SnitchScheduleEvaluator(
        schedule_text, args.run_script, args.payload, timeout_s=args.timeout_s,
        verbose=args.verbose, cleanup=not args.keep_workdirs,
    )

    print(f"Total distinct valid (tile_m, tile_n, tile_k) combinations for "
          f"this payload: {max_candidates}.")

    def clamp(flag: str, value: int) -> int:
        if value > max_candidates:
            print(f"  --{flag}={value} exceeds that ({max_candidates} available) -- "
                  f"clamping to {max_candidates} to avoid Tuner's GA population-filling "
                  "loop spinning forever looking for a candidate that doesn't exist.")
            return max_candidates
        return value

    init_samples = clamp("init-samples", args.init_samples)
    ga_population = clamp("ga-population", args.ga_population)
    ga_candidates_per_iter = clamp("ga-candidates-per-iter", args.ga_candidates_per_iter)
    if args.iterations * ga_candidates_per_iter > max_candidates + init_samples:
        print(f"  note: --iterations={args.iterations} will exhaust all {max_candidates} "
              "candidates well before the last iteration -- later iterations will just "
              "report '(no new candidates to evaluate this iteration)'. Consider a "
              "smaller --iterations for this search space.")

    config = TunerConfig(
        iterations=args.iterations,
        init_samples=init_samples,
        ga_population=ga_population,
        ga_generations=args.ga_generations,
        ga_candidates_per_iter=ga_candidates_per_iter,
        seed=args.seed,
    )
    tuner = Tuner(search_space, features, evaluator, config)

    print(
        f"\nRunning {config.iterations} iteration(s): iteration 1 evaluates "
        f"{config.init_samples} random candidate(s); iterations 2+ propose "
        f"{config.ga_candidates_per_iter} candidate(s) per round via a GA "
        f"guided by the GBDT surrogate (population={config.ga_population}, "
        f"generations={config.ga_generations}). Each real candidate is a full "
        f"compile + gvsoc run, timeout {args.timeout_s:.0f}s -- expect this to "
        "take a while."
    )
    best = tuner.run()

    n_valid = sum(r.result.valid for r in tuner.history)
    print(f"\n{len(tuner.history)} candidate(s) evaluated for real, {n_valid} valid")
    best_desc = " ".join(f"{n}={best[n]}" for n in sorted(best))
    print(f"Best candidate: {best_desc}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuner.to_records()).to_csv(args.results, index=False)
    print(f"Full log: {args.results}")

    args.best_schedule.parent.mkdir(parents=True, exist_ok=True)
    args.best_schedule.write_text(render_schedule(schedule_text, best))
    print(f"Best schedule written to: {args.best_schedule}")


if __name__ == "__main__":
    main()
