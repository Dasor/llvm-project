#!/usr/bin/env python3
"""
Genetic algorithm tile-size search using a trained XGBoost surrogate model.

Replaces random sampling in iteration N+1: evolves the best configurations
found so far (seeded from the previous iteration's CSV) toward higher predicted
throughput, then outputs the top-N candidates for actual simulation.

Usage:
    python ga.py --M 128 --N 128 --K 128 \\
                 --model iter1_model.json \\
                 --seed-csv ../iter1_configs.csv \\
                 --out ../iter2_ga_candidates.csv \\
                 [--pop 200 --generations 100 --top-n 30 \\
                  --cxpb 0.5 --mutpb 0.3 --sigma 16 \\
                  --allow-boundary --seed 42]

Output: CSV in the same format as generate_configs.py output (all feature
columns), ready to feed directly into the compile + simulate step.
"""

import argparse
import os
import re
import random
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from deap import base, creator, tools

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "../../myrtle/myrtle"))

from generate_configs import (
    _is_valid,
    _remainder_tag,
    _valid_k_values,
    _valid_m_values,
    _valid_n_values,
    BYTES_PER_ELEM,
    L1_BYTES,
)
from tile_static_analysis.TSA_C_Remainder import TSA_C_Remainder
from tile_static_analysis.remainder_utils import TilingScheme
from boost import _feature_matrix
from ga_repair import repair

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# ---------------------------------------------------------------------------
# Feature row construction (mirrors generate_configs.generate())
# ---------------------------------------------------------------------------

def _build_row(
    M: int, N: int, K: int,
    m: int, n: int, k: int,
    tsa: TSA_C_Remainder,
) -> dict:
    remainder = _remainder_tag(M, N, K, m, n, k)
    ts = TilingScheme(M, N, K, m, n, k, u=8, p=8, remainderTiles=remainder)
    metrics = tsa.tilingSchemeMetrics(ts)
    m_prime = m // 8
    row = {
        "FakeNN JSON Name": f"{M}x{N}x{K}w{m}-{n}-{k}",
        "M": M, "N": N, "K": K,
        "m": m, "n": n, "k": k,
        "JSON Name": f"{m}-{n}-{k}",
        "remainderTiles": remainder,
        "Space Needed in L1": 2 * (m * k + n * k + m * n) * BYTES_PER_ELEM,
        "Row Dim": n,
        "tileB_cc": n * k * BYTES_PER_ELEM,
        "Reduction Dim": k,
        "tileA": m * k * BYTES_PER_ELEM,
        "tileB": n * k * BYTES_PER_ELEM,
        "tileC_cc": m_prime * n * BYTES_PER_ELEM,
        "tileA_cc": m_prime * k * BYTES_PER_ELEM,
        "Space Remaining": L1_BYTES - 2 * (m * k + n * k + m * n) * BYTES_PER_ELEM,
        "tileC": m * n * BYTES_PER_ELEM,
        "m Dim": m,
        "Weight Matrix Tile Size": 2 * n * k * BYTES_PER_ELEM,
    }
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# Batch fitness evaluation
# ---------------------------------------------------------------------------

def _score_population(
    individuals: list,
    M: int, N: int, K: int,
    tsa: TSA_C_Remainder,
    model: xgb.Booster,
    row_cache: dict,
    score_cache: dict,
) -> None:
    """Assign fitness to every individual that lacks a valid fitness value.

    Uses score_cache keyed by (m, n, k) to avoid re-scoring configs seen in
    earlier generations.
    """
    to_eval = []
    for ind in individuals:
        key = (ind[0], ind[1], ind[2])
        if key in score_cache:
            ind.fitness.values = (score_cache[key],)
        else:
            to_eval.append(ind)

    if not to_eval:
        return

    rows = []
    for ind in to_eval:
        key = (ind[0], ind[1], ind[2])
        if key not in row_cache:
            row_cache[key] = _build_row(M, N, K, ind[0], ind[1], ind[2], tsa)
        rows.append(row_cache[key])

    df = pd.DataFrame(rows)
    X = _feature_matrix(df)
    scores = model.predict(xgb.DMatrix(X, feature_names=X.columns.tolist()))

    for ind, score in zip(to_eval, scores):
        key = (ind[0], ind[1], ind[2])
        score_cache[key] = float(score)
        ind.fitness.values = (float(score),)


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _random_individual(
    M: int, N: int, K: int, boundary: bool, rng: random.Random,
) -> "creator.Individual | None":
    vm = _valid_m_values(M, boundary)
    vn = _valid_n_values(N, boundary)
    vk = _valid_k_values(K, boundary)
    for _ in range(10_000):
        m = rng.choice(vm)
        n = rng.choice(vn)
        k = rng.choice(vk)
        if _is_valid(m, n, k):
            return creator.Individual([m, n, k])
    return None


def _mutate(
    ind, M: int, N: int, K: int, sigma: float, boundary: bool, rng: random.Random,
) -> tuple:
    m = int(ind[0] + rng.gauss(0, sigma))
    n = int(ind[1] + rng.gauss(0, sigma))
    k = int(ind[2] + rng.gauss(0, sigma))
    result = repair(m, n, k, M, N, K, boundary)
    if result is not None:
        ind[0], ind[1], ind[2] = result
        del ind.fitness.values
    return (ind,)


def _crossover(
    ind1, ind2, M: int, N: int, K: int, boundary: bool, rng: random.Random,
) -> tuple:
    """Uniform crossover on the 3-integer chromosome, followed by repair."""
    new1 = [rng.choice([ind1[i], ind2[i]]) for i in range(3)]
    new2 = [rng.choice([ind1[i], ind2[i]]) for i in range(3)]
    r1 = repair(new1[0], new1[1], new1[2], M, N, K, boundary)
    r2 = repair(new2[0], new2[1], new2[2], M, N, K, boundary)
    if r1 is not None:
        ind1[0], ind1[1], ind1[2] = r1
        del ind1.fitness.values
    if r2 is not None:
        ind2[0], ind2[1], ind2[2] = r2
        del ind2.fitness.values
    return ind1, ind2


# ---------------------------------------------------------------------------
# Main GA driver
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r'\d+x\d+x\d+w(\d+)-(\d+)-(\d+)')


def _load_blocklist(path: str | None) -> set:
    """Return a set of (m, n, k) tuples parsed from a blocklist file."""
    blocked: set = set()
    if not path or not os.path.exists(path):
        return blocked
    with open(path) as f:
        for line in f:
            m = _NAME_RE.match(line.strip())
            if m:
                blocked.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    print(f"Blocklist: {len(blocked)} timed-out config(s) will be excluded from candidates")
    return blocked


def run(
    M: int, N: int, K: int,
    model_path: str,
    seed_csv: str | None,
    out_path: str,
    pop_size: int,
    n_generations: int,
    top_n: int,
    cxpb: float,
    mutpb: float,
    sigma: float,
    boundary: bool,
    rng_seed: int | None,
    blocklist_path: str | None = None,
) -> None:
    rng = random.Random(rng_seed)
    blocked = _load_blocklist(blocklist_path)

    model = xgb.Booster()
    model.load_model(model_path)
    tsa = TSA_C_Remainder(unrollAndJamFactor=8, degreeOfParallelism=8)

    row_cache: dict = {}    # (m,n,k) -> feature row dict
    score_cache: dict = {}  # (m,n,k) -> predicted score (persists across all gens)

    # --- Build initial population ---
    pop: list = []

    if seed_csv:
        df_seed = pd.read_csv(seed_csv)
        for _, row in df_seed.iterrows():
            key = (int(row["m"]), int(row["n"]), int(row["k"]))
            if key not in blocked:
                pop.append(creator.Individual(list(key)))
        print(f"Seeded {len(pop)} individuals from {seed_csv}")

    # Top up with random valid configs if population is smaller than pop_size
    missing = pop_size - len(pop)
    if missing > 0:
        added = 0
        for _ in range(missing * 10):
            ind = _random_individual(M, N, K, boundary, rng)
            if ind is not None:
                pop.append(ind)
                added += 1
                if added >= missing:
                    break
        if added < missing:
            print(
                f"Warning: only found {added} additional random configs "
                f"(needed {missing}); search space may be small.",
                file=sys.stderr,
            )

    pop = pop[:pop_size]

    # Evaluate initial population
    _score_population(pop, M, N, K, tsa, model, row_cache, score_cache)

    best0 = max(ind.fitness.values[0] for ind in pop)
    avg0 = sum(ind.fitness.values[0] for ind in pop) / len(pop)
    print(f"Gen   0 / {n_generations}: best={best0:.4f}  avg={avg0:.4f}  "
          f"pop={len(pop)}")

    toolbox = base.Toolbox()
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Main evolution loop ---
    for gen in range(1, n_generations + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, pop_size)))

        for i in range(1, len(offspring), 2):
            if rng.random() < cxpb:
                _crossover(offspring[i - 1], offspring[i], M, N, K, boundary, rng)

        for ind in offspring:
            if rng.random() < mutpb:
                _mutate(ind, M, N, K, sigma, boundary, rng)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        _score_population(invalid, M, N, K, tsa, model, row_cache, score_cache)

        pop[:] = offspring

        if gen % 10 == 0 or gen == n_generations:
            best = max(ind.fitness.values[0] for ind in pop)
            avg = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(f"Gen {gen:>3} / {n_generations}: best={best:.4f}  avg={avg:.4f}  "
                  f"evaluated={len(score_cache)}")

    # --- Select top-N unique configs across all evaluated (not just final pop) ---
    # Exclude timed-out configs from the blocklist.
    top = sorted(
        [(k, v) for k, v in score_cache.items() if k not in blocked],
        key=lambda x: -x[1],
    )[:top_n]

    print(f"\nTop {top_n} candidates (predicted normalized throughput):")
    for i, ((m, n, k), score) in enumerate(top, 1):
        print(f"  {i:>3}.  {M}x{N}x{K}w{m}-{n}-{k}  score={score:.4f}")

    # Build feature rows for the top-N (using cache where available)
    out_rows = []
    for (m, n, k), _ in top:
        key = (m, n, k)
        if key not in row_cache:
            row_cache[key] = _build_row(M, N, K, m, n, k, tsa)
        out_rows.append(row_cache[key])

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_rows)} candidates → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Genetic algorithm GEMM tile-size search (XGBoost surrogate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--M", type=int, required=True, help="GEMM M dimension")
    p.add_argument("--N", type=int, required=True, help="GEMM N dimension")
    p.add_argument("--K", type=int, required=True, help="GEMM K dimension")
    p.add_argument("--model", required=True, metavar="PATH",
                   help="trained XGBoost model (.json)")
    p.add_argument("--seed-csv", default=None, metavar="PATH",
                   help="CSV to seed the initial population (must have m, n, k columns)")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="output CSV of top-N candidates (same schema as generate_configs.py)")
    p.add_argument("--pop", type=int, default=200, metavar="N",
                   help="population size (default: 200)")
    p.add_argument("--generations", type=int, default=100,
                   help="number of generations (default: 100)")
    p.add_argument("--top-n", type=int, default=30,
                   help="number of top candidates to output (default: 30)")
    p.add_argument("--cxpb", type=float, default=0.5,
                   help="crossover probability per pair (default: 0.5)")
    p.add_argument("--mutpb", type=float, default=0.3,
                   help="mutation probability per individual (default: 0.3)")
    p.add_argument("--sigma", type=float, default=16.0,
                   help="mutation Gaussian std-dev in tile-size units (default: 16)")
    p.add_argument("--allow-boundary", action="store_true",
                   help="allow boundary/remainder-tile configurations (requires gemm_boundary)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducibility")
    p.add_argument("--blocklist", default=None, metavar="PATH",
                   help="text file of timed-out config names (one per line) to exclude "
                        "from candidates; accumulated across iterations by run_loop.sh")
    args = p.parse_args()

    run(
        M=args.M, N=args.N, K=args.K,
        model_path=args.model,
        seed_csv=args.seed_csv,
        out_path=args.out,
        pop_size=args.pop,
        n_generations=args.generations,
        top_n=args.top_n,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        sigma=args.sigma,
        boundary=args.allow_boundary,
        rng_seed=args.seed,
        blocklist_path=args.blocklist,
    )


if __name__ == "__main__":
    main()
