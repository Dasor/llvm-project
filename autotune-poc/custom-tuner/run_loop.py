#!/usr/bin/env python3
"""
run_loop.py — Generic Ansor-style GBDT loop for GEMM tile-size search

Iteration 1: random sampling → compile → simulate → train initial model
Iterations 2+: GA (seeded from all previous configs, guided by latest model)
             → compile → simulate → retrain on ALL accumulated data

Usage (from /repo inside Docker):
  python testrun/run_loop.py [OPTIONS]
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print()
    print("════════════════════════════════════════════════════")
    print(f"  {msg}")
    print("════════════════════════════════════════════════════")


def seed_args(seed: int | None, iter_n: int) -> list[str]:
    if seed is not None:
        return ["--seed", str(seed + iter_n - 1)]
    return []


def run(cmd: list[str], env: dict | None = None) -> None:
    merged_env = {**os.environ, **(env or {})}
    subprocess.run(cmd, check=True, env=merged_env)


# ── Inline pandas operations (replacing heredocs) ─────────────────────────────

def build_ga_seed_csv(workdir: Path, iter_n: int) -> Path:
    """Concatenate configs from iterations 1..iter_n-1 for GA seeding."""
    dfs = [pd.read_csv(workdir / f"iter{i}_configs.csv") for i in range(1, iter_n)]
    out = workdir / f"iter{iter_n}_ga_seed.csv"
    pd.concat(dfs).drop_duplicates("FakeNN JSON Name").to_csv(out, index=False)
    return out


def append_blocklist(configs_csv: Path, blocklist: Path) -> None:
    configs = pd.read_csv(configs_csv)["FakeNN JSON Name"]
    with open(blocklist, "a") as f:
        for name in configs:
            f.write(name + "\n")
    print(f"Exclusion list: added {len(configs)} config(s) → {blocklist}")


def build_combined_datasets(workdir: Path, iter_n: int) -> tuple[Path, Path]:
    """Concatenate all feature and result CSVs up to iter_n."""
    feat_dfs = [pd.read_csv(workdir / f"iter{i}_configs.csv") for i in range(1, iter_n + 1)]
    result_dfs = [pd.read_csv(workdir / f"iter{i}_configs-results.csv") for i in range(1, iter_n + 1)]

    all_features = workdir / f"all_configs_iter1_to_{iter_n}.csv"
    all_results = workdir / f"all_results_iter1_to_{iter_n}.csv"

    pd.concat(feat_dfs).drop_duplicates("FakeNN JSON Name").to_csv(all_features, index=False)
    pd.concat(result_dfs).drop_duplicates("FakeNN JSON Name").to_csv(all_results, index=False)

    total = len(pd.read_csv(all_features))
    print(f"Combined dataset: {total} unique configs across {iter_n} iteration(s)")
    return all_features, all_results


def validate_checkpoint(workdir: Path, resume_from: int) -> None:
    missing = []
    for i in range(1, resume_from):
        for name in (f"iter{i}_configs.csv", f"iter{i}_configs-results.csv"):
            if not (workdir / name).exists():
                missing.append(name)
    prev_model = f"iter{resume_from - 1}_model.json"
    if not (workdir / prev_model).exists():
        missing.append(prev_model)
    if missing:
        print("Error: cannot resume — the following checkpoint files are missing:")
        for f in missing:
            print(f"  {workdir / f}")
        sys.exit(1)


def print_summary(workdir: Path) -> None:
    files = sorted(glob.glob(str(workdir / "iter*_configs-results.csv")))
    if not files:
        print("No results found.")
        return

    df = pd.concat([pd.read_csv(f) for f in files]).reset_index(drop=True)
    df = df[df["Global Sim E2E_dma"] > 0]
    if df.empty:
        print("No valid (non-timeout) simulation results.")
        return

    best = df.loc[df["Global Sim E2E_dma"].idxmin()]
    worst = df.loc[df["Global Sim E2E_dma"].idxmax()]
    improvement = (worst["Global Sim E2E_dma"] - best["Global Sim E2E_dma"]) / worst["Global Sim E2E_dma"] * 100

    print(f"Configs evaluated : {len(df)}")
    print(f"Best  → {best['FakeNN JSON Name']:40s}  {best['Global Sim E2E_dma']:.0f} cycles")
    print(f"Worst → {worst['FakeNN JSON Name']:40s}  {worst['Global Sim E2E_dma']:.0f} cycles")
    print(f"Range : {improvement:.1f}% spread between best and worst evaluated")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ansor-style GBDT loop for GEMM tile-size search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--M", type=int, default=384, help="GEMM M dimension")
    p.add_argument("--N", type=int, default=384, help="GEMM N dimension")
    p.add_argument("--K", type=int, default=384, help="GEMM K dimension")
    p.add_argument("--iterations", type=int, default=15, dest="n_iters", help="Total loop iterations")
    p.add_argument("--init-pool", type=int, default=3000, help="Random configs generated in iter 1")
    p.add_argument("--init-configs", type=int, default=32, help="Top-N of the pool to actually simulate")
    p.add_argument("--ga-candidates", type=int, default=32, help="GA top-N per iter 2+")
    p.add_argument("--ga-pop", type=int, default=1000, help="GA population size")
    p.add_argument("--ga-generations", type=int, default=100, help="GA generations per run")
    p.add_argument("--allow-boundary", action="store_true", default=True,
                   help="Enable boundary/remainder tile mode")
    p.add_argument("--workdir", default="testrun", help="Artifact directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--timeout", type=int, default=7200, dest="timeout_secs",
                   help="Wall-clock timeout per simulation in seconds (0 = no limit)")
    p.add_argument("--resume-from", type=int, default=1, dest="resume_from",
                   help="Start from this iteration number (1 = fresh run); all earlier iterations must already exist in workdir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path("myrtle-experiments/many_gemms.sh").exists():
        print("Error: run this script from /repo (the repo root), not from testrun/")
        sys.exit(1)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    experiment_dir = str(workdir.resolve())

    subprocess_env = {
        "gemmDir": "/repo/sw/kernels/blas/gemm_boundary",
        "experimentDir": experiment_dir,
        "beta": "0",
        "spm_opt": "0",
        "TIMEOUT": str(args.timeout_secs),
    }

    boundary_flag = ["--allow-boundary"] if args.allow_boundary else []

    if args.resume_from > 1:
        validate_checkpoint(workdir, args.resume_from)
        banner(f"RESUMING from iteration {args.resume_from} (skipping 1–{args.resume_from - 1})")

    for iter_n in range(args.resume_from, args.n_iters + 1):
        configs_csv = workdir / f"iter{iter_n}_configs.csv"
        results_csv = workdir / f"iter{iter_n}_configs-results.csv"
        model_json = workdir / f"iter{iter_n}_model.json"

        banner(f"ITERATION {iter_n} / {args.n_iters}  ({args.M}×{args.N}×{args.K})")

        # ── Step 1: Generate candidate configs ────────────────────────────────
        print()
        print("── Step 1: Generate configs ──")

        if iter_n == 1:
            pool_csv = workdir / "iter1_pool.csv"
            run(
                ["python", "testrun/generate_configs.py",
                 str(args.M), str(args.N), str(args.K), str(args.init_pool),
                 *seed_args(args.seed, iter_n),
                 *boundary_flag,
                 "--out", str(pool_csv)],
            )
            df_pool = pd.read_csv(pool_csv)
            df_pool.head(args.init_configs).to_csv(configs_csv, index=False)
            print(f"Selected top {args.init_configs} configs (by tile size) from pool of {args.init_pool} → {configs_csv}")
        else:
            prev_model = workdir / f"iter{iter_n - 1}_model.json"
            ga_seed_csv = build_ga_seed_csv(workdir, iter_n)

            blocklist_path = workdir / "tested_configs.txt"
            blocklist_arg = ["--blocklist", str(blocklist_path)] if blocklist_path.exists() else []

            run(
                ["python", "testrun/train/ga.py",
                 "--M", str(args.M), "--N", str(args.N), "--K", str(args.K),
                 "--model", str(prev_model),
                 "--seed-csv", str(ga_seed_csv),
                 "--out", str(configs_csv),
                 "--pop", str(args.ga_pop),
                 "--generations", str(args.ga_generations),
                 "--top-n", str(args.ga_candidates),
                 *seed_args(args.seed, iter_n),
                 *boundary_flag,
                 *blocklist_arg],
            )

        # ── Step 2a: Compile ──────────────────────────────────────────────────
        print()
        print("── Step 2a: Compile ──")
        run(
            ["bash", "myrtle-experiments/many_gemms.sh", f"./{configs_csv}", "compile", "no", "no", "no"],
            env=subprocess_env,
        )

        # ── Step 2b: Simulate ─────────────────────────────────────────────────
        print()
        print("── Step 2b: Simulate ──")
        run(
            ["bash", "myrtle-experiments/many_gemms.sh", f"./{configs_csv}", "check", "run", "no", "no"],
            env=subprocess_env,
        )

        # ── Step 2c: Merge results ────────────────────────────────────────────
        print()
        print("── Step 2c: Merge results ──")
        run(
            ["python", "myrtle-experiments/combineTilingSchemeDataIntoSingleCSV.py",
             f"./{configs_csv}", experiment_dir],
        )

        if not results_csv.exists():
            print(f"Error: {results_csv} not found — simulation may have failed")
            sys.exit(1)

        # ── Step 2d: Update blocklist ─────────────────────────────────────────
        append_blocklist(configs_csv, workdir / "tested_configs.txt")

        # ── Step 3: Train on all data so far ──────────────────────────────────
        print()
        print(f"── Step 3: Train model on iterations 1–{iter_n} ──")

        all_features, all_results = build_combined_datasets(workdir, iter_n)

        run(
            ["python", "testrun/train/boost.py", "train",
             str(all_features), str(all_results),
             "--save", str(model_json)],
        )

        print()
        print(f"Iteration {iter_n} complete → model: {model_json}")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner(f"LOOP COMPLETE — {args.n_iters} iteration(s) for {args.M}×{args.N}×{args.K}")
    print_summary(workdir)


if __name__ == "__main__":
    main()
