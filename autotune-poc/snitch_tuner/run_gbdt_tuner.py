#!/usr/bin/env python3
"""Tune schedules/tile_l1_pipelined_tunable.mlir's tile_m/tile_n/tile_k
transform.tune.knob values (the Snitch Milestone 6 pipeline: dual-buffered
pipelining + all 8 compute cores) with the GBDT+GA search from
../tuner_framework, the same search strategy ../mlir_tuner/run_gbdt_tuner.py
uses for the CPU matmul PoC -- only the backend differs (see backend.py's
module docstring for exactly what and why).

Usage:
    python3 snitch_tuner/run_gbdt_tuner.py \\
        --iterations 4 --init-samples 4 --ga-candidates-per-iter 3

Every REAL candidate evaluated here is a full xDSL+LLVM compile followed by
a gvsoc simulation run -- much more expensive than the CPU PoC's native
compile+microbenchmark -- so the defaults below are deliberately much
smaller than mlir_tuner/run_gbdt_tuner.py's (iterations=15, init_samples=16).
Scale up once one round is confirmed to work end-to-end (see the plan's
Verification section). ga-population/ga-generations are cheap (pure
surrogate-model prediction, no real evaluation) and can stay larger without
affecting wall-clock cost.

M/N/K are auto-detected from --payload's own linalg.matmul op (see
mlir_schedule.detect_matmul_dims), not passed on the CLI. The schedule
file's own `options = [...]` is NEVER consulted for validity -- see
mlir_tuner.backend.KnobSearchSpace's docstring; validity here comes
entirely from backend.snitch_tile_constraint.
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
from mlir_tuner.backend import KnobSearchSpace  # noqa: E402
from snitch_tuner.backend import (  # noqa: E402
    SnitchGemmKnobFeatureExtractor, SnitchScheduleEvaluator,
    snitch_divisor_sample_value, snitch_search_space_size, snitch_tile_constraint,
)
from tuner_framework import Tuner, TunerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "schedule", nargs="?", type=Path,
        default=_HERE / "schedules" / "tile_l1_pipelined_tunable.mlir",
        help="Path to the .mlir transform schedule with unresolved "
        "transform.tune.knob ops. (default: %(default)s)",
    )
    p.add_argument(
        "--payload", type=Path, default=_HERE / "payload" / "matmul_32.mlir",
        help="Path to the matmul payload module -- M/N/K are detected from "
        "here, not from the schedule file (Snitch keeps them in separate "
        "files; see CLAUDE.md's tile_l1.mlir note). (default: %(default)s)",
    )
    p.add_argument("--results", type=Path, default=_HERE / "results" / "gbdt_search.csv")
    p.add_argument("--best-schedule", type=Path, default=_HERE / "results" / "best_schedule_gbdt.mlir")
    p.add_argument(
        "--run-script", type=Path, default=_HERE / "driver" / "tune_candidate.sh",
        help="Script that compiles one fully-resolved schedule through the "
        "full Snitch M6 pipeline and runs it on gvsoc. (default: %(default)s)",
    )
    p.add_argument(
        "--timeout-s", type=float, default=300.0,
        help="Hard wall-clock timeout (with process-group kill) per candidate "
        "-- see SnitchScheduleEvaluator's docstring for why this isn't "
        "optional. (default: %(default)s)",
    )

    p.add_argument("--iterations", type=int, default=10, help="outer iterations (default: %(default)s)")
    p.add_argument("--init-samples", type=int, default=64, help="random samples in iteration 1 (default: %(default)s)")
    p.add_argument("--ga-population", type=int, default=100, help="GA population size (default: %(default)s)")
    p.add_argument("--ga-generations", type=int, default=100, help="GA generations (default: %(default)s)")
    p.add_argument("--ga-candidates-per-iter", type=int, default=64,
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    schedule_text = args.schedule.read_text()
    payload_text = args.payload.read_text()
    all_knobs = extract_knobs(schedule_text)
    tunable = [k for k in all_knobs if not k.already_resolved]
    fixed = [k for k in all_knobs if k.already_resolved]

    if not tunable:
        sys.exit(f"No unresolved transform.tune.knob ops found in {args.schedule}. Nothing to tune.")

    print(f"Found {len(tunable)} tunable knob(s) in {args.schedule} (their "
          "file-declared options below are informational only -- NOT used "
          "to constrain the search; only snitch_divisor_sample_value is):")
    for k in tunable:
        print(f"  {k.name}: options = {k.options}")
    if fixed:
        print(f"({len(fixed)} already-resolved knob(s) left untouched: "
              f"{', '.join(k.name for k in fixed)})")

    dims = detect_matmul_dims(payload_text)
    print(f"Detected GEMM dimensions from {args.payload}'s own linalg.matmul op: "
          f"M={dims['M']}, N={dims['N']}, K={dims['K']}. tile_m is restricted to "
          "multiples of 8 (required for M6's barrier-participant symmetry across "
          "all 8 compute cores); tile_k excludes 1 and 2 (too many tile "
          f"iterations); tile_n is fixed at N//2={dims['N'] // 2} (LowerPipelineOp's "
          "on-ramp/steady-state/off-ramp expansion only generalizes correctly to a "
          "pipeline trip count of exactly 2 -- found empirically, see "
          "backend.snitch_tile_constraint's docstring for the two isolation tests "
          "that pinned this down). tile_m and tile_k remain free.")

    sample_value = lambda name, rng: snitch_divisor_sample_value(name, rng, dims)  # noqa: E731
    constraint = snitch_tile_constraint(dims)

    search_space = KnobSearchSpace(tunable, sample_value, constraint=constraint)
    features = SnitchGemmKnobFeatureExtractor(payload_text)
    evaluator = SnitchScheduleEvaluator(
        schedule_text, args.run_script, timeout_s=args.timeout_s,
        verbose=args.verbose, cleanup=not args.keep_workdirs,
    )

    # The search space here is deliberately tiny (tile_n is pinned to a
    # singleton -- see snitch_tile_constraint's docstring), so it's easy to
    # request more samples/population than distinct valid candidates exist.
    # That's not just wasteful, it HANGS: tuner_framework.Tuner's
    # _propose_via_ga() fills its GA population with
    # `while len(pop) < ga_population: ... sample_random() ...`, which
    # never terminates once every valid candidate is already in the
    # population and ga_population was requested higher than that -- see
    # snitch_search_space_size's docstring. Clamp here rather than in
    # tuner_framework/tuner.py itself, since that file is shared with the
    # CPU backend and this ceiling is a Snitch-specific fact, not a general
    # one.
    max_candidates = snitch_search_space_size(dims)
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
