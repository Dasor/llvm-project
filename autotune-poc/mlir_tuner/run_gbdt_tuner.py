#!/usr/bin/env python3
"""Tune an MLIR schedule's transform.tune.knob values with the GBDT+GA
search from tuner_framework, instead of driver/autotune.py's exhaustive
grid search.

Same knob extraction/rendering/running as driver/autotune.py
(mlir_schedule.py) -- only the search strategy differs. For the ~27
candidates in schedules/tunable_matmul.mlir, exhaustive grid search (the
driver/ script) is the right tool and this will do no better; this exists
for when a schedule has enough knobs/options that exhaustive search stops
being practical.

Usage:
    python3 mlir_tuner/run_gbdt_tuner.py schedules/tunable_matmul.mlir \\
        --expect-uniform-value 512 \\
        --iterations 4 --init-samples 6 --ga-population 12 --ga-generations 10

The search space is defined entirely by `divisor_sample_value`, used for
both brand-new candidates and GA mutation alike: candidates are always
drawn from the exact divisors of each knob's own dimension (tile_m -> M,
tile_n -> N, tile_k -> K). M/N/K are auto-detected from the schedule
file's own `linalg.matmul ins(... : tensor<MxKx...>, tensor<KxNx...>)` op
(see mlir_schedule.detect_matmul_dims) -- not passed on the CLI -- so they
always match whatever payload the schedule actually embeds. The schedule file's own
`options = [...]` is NEVER consulted for validity -- see
mlir_tuner/backend.py's KnobSearchSpace docstring. Divisibility is a hard
requirement, not just a heuristic: tunable_matmul.mlir's
transform.structured.vectorize has no explicit vector_sizes, so it only
accepts tile sizes that evenly divide their dimension (a non-divisor tile
leaves a dynamic-shaped remainder, which fails vectorization's
static-shape requirement -- masked vectorization via an explicit
vector_sizes was tried and reverted, since this build's
-buffer-deallocation-pipeline pass mis-lowers the resulting non-trivial
vector.mask ops).

Divisors are additionally capped at hw_detect.max_square_tile_dim(): a
loose, host-cache-derived upper bound (see that function's docstring for
why it's deliberately not tight) that keeps the search from wasting
attempts on tiles that can never be cache-resident on this machine, with
zero CLI flags -- no --constraint expression to write by hand, no
tile-size bound to keep in sync with the host by hand. If a schedule's
tuning goals ever need something other than "a divisor of the dimension,
no bigger than roughly fits in cache", replace divisor_sample_value with
something that better reflects that different search space by
constructing KnobSearchSpace directly instead of going through this
generic CLI.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_POC_DIR = _HERE.parent
sys.path.insert(0, str(_POC_DIR))

import pandas as pd  # noqa: E402

from mlir_schedule import detect_matmul_dims, extract_knobs, render_schedule  # noqa: E402
from mlir_tuner import hw_detect  # noqa: E402
from mlir_tuner.backend import (  # noqa: E402
    GemmKnobFeatureExtractor, KnobSearchSpace, ScheduleEvaluator, detect_dtype_bytes,
)
from tuner_framework import Tuner, TunerConfig  # noqa: E402


def divisors(n: int) -> List[int]:
    """All positive divisors of n, ascending."""
    return [d for d in range(1, n + 1) if n % d == 0]


_KNOB_DIM_NAME = {"tile_m": "M", "tile_n": "N", "tile_k": "K"}


def divisor_sample_value(name: str, rng: random.Random, dims: Dict[str, int], max_tile: int) -> str:
    """Uniform choice among the ACTUAL divisors of the dimension `name`
    tiles (tile_m -> M, tile_n -> N, tile_k -> K, via `dims`), further
    capped at `max_tile` -- NOT an arbitrary integer up to some bound. See
    this module's docstring for why divisibility is a hard requirement
    (vectorization's static-shape check) while max_tile is only a loose
    cache-fit cap (hw_detect.max_square_tile_dim). Used for both brand-new
    candidates and GA mutation alike: unlike a power-of-two-vs-uniform
    split, there's no "loosen it for mutation" option here, since a
    non-divisor tile always fails compilation regardless of where it came
    from, and 1 -- always a divisor -- is always <= max_tile."""
    dim = dims[_KNOB_DIM_NAME[name]]
    return str(rng.choice([d for d in divisors(dim) if d <= max_tile]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "schedule",
        nargs="?",
        type=Path,
        default=_POC_DIR / "schedules" / "tunable_matmul.mlir",
        help="Path to a .mlir file with unresolved transform.tune.knob ops. "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--expect-uniform-value",
        default=None,
        help="If set, a candidate is only 'correct' when every printed numeric "
        "value in its output equals this.",
    )
    p.add_argument("--results", type=Path, default=_POC_DIR / "results" / "gbdt_search.csv")
    p.add_argument("--best-schedule", type=Path, default=_POC_DIR / "results" / "best_schedule_gbdt.mlir")
    p.add_argument(
        "--compile-script",
        type=Path,
        default=_POC_DIR / "driver" / "compile_schedule.sh",
        help="Script that ahead-of-time compiles one fully-resolved schedule "
        "to a standalone native executable (see README.md's timing note).",
    )

    p.add_argument("--iterations", type=int, default=15, help="outer iterations (default: 10)")
    p.add_argument("--init-samples", type=int, default=16, help="random samples in iteration 1 (default: 8)")
    p.add_argument("--ga-population", type=int, default=500, help="GA population size (default: 100)")
    p.add_argument("--ga-generations", type=int, default=100, help="GA generations (default: 100)")
    p.add_argument("--ga-candidates-per-iter", type=int, default=8,
                    help="how many of the GA's top candidates get really evaluated per iteration")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--verbose", action="store_true",
        help="Print each candidate as it starts and finishes running, in addition "
        "to the per-iteration/per-GA-generation summaries.",
    )
    p.add_argument(
        "--timeout-factor", type=float, default=1.5,
        help="Kill a candidate's run, and mark it invalid, once it's running "
        "this many times longer than the best valid candidate's time seen so "
        "far (0 disables). (default: %(default)s)",
    )
    p.add_argument(
        "--compile-timeout-factor", type=float, default=4,
        help="Kill a candidate's compile, and mark it invalid, once it's "
        "taking this many times longer than the best valid candidate's "
        "compile time seen so far (0 disables). (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    schedule_text = args.schedule.read_text()
    all_knobs = extract_knobs(schedule_text)
    tunable = [k for k in all_knobs if not k.already_resolved]
    fixed = [k for k in all_knobs if k.already_resolved]

    if not tunable:
        sys.exit(f"No unresolved transform.tune.knob ops found in {args.schedule}. Nothing to tune.")

    print(f"Found {len(tunable)} tunable knob(s) in {args.schedule} (their "
          "file-declared options below are informational only -- NOT used "
          "to constrain the search; only divisor_sample_value does):")
    for k in tunable:
        print(f"  {k.name}: options = {k.options}")
    if fixed:
        print(f"({len(fixed)} already-resolved knob(s) left untouched: "
              f"{', '.join(k.name for k in fixed)})")

    elem_bytes = detect_dtype_bytes(schedule_text)
    max_tile = hw_detect.max_square_tile_dim(elem_bytes)
    dims = detect_matmul_dims(schedule_text)
    sample_value = lambda name, rng: divisor_sample_value(name, rng, dims, max_tile)  # noqa: E731
    print(f"Detected GEMM dimensions from the schedule's own linalg.matmul op: "
          f"M={dims['M']}, N={dims['N']}, K={dims['K']}. Sampling every candidate "
          f"(brand-new and GA-mutated alike) from exact divisors of its own "
          f"dimension (tile_m|{dims['M']}, tile_n|{dims['N']}, tile_k|{dims['K']}) "
          f"-- required for vectorization's static-shape check -- capped at "
          f"{max_tile} per dimension, a loose bound so a cubic tile's A+B+C "
          f"footprint fits this host's L2 (see hw_detect.max_square_tile_dim); "
          "see divisor_sample_value's docstring")

    search_space = KnobSearchSpace(tunable, sample_value)
    features = GemmKnobFeatureExtractor(schedule_text)
    evaluator = ScheduleEvaluator(
        schedule_text, args.compile_script, args.expect_uniform_value, verbose=args.verbose,
        timeout_factor=args.timeout_factor, compile_timeout_factor=args.compile_timeout_factor,
    )

    config = TunerConfig(
        iterations=args.iterations,
        init_samples=args.init_samples,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        ga_candidates_per_iter=args.ga_candidates_per_iter,
        seed=args.seed,
    )
    tuner = Tuner(search_space, features, evaluator, config)

    print(
        f"\nRunning {config.iterations} iteration(s): iteration 1 evaluates "
        f"{config.init_samples} random candidate(s); iterations 2+ propose "
        f"{config.ga_candidates_per_iter} candidate(s) per round via a GA "
        f"guided by the GBDT surrogate (population={config.ga_population}, "
        f"generations={config.ga_generations})."
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
