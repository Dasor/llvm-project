# snitch-backend

Self-contained container setup for the Snitch RISC-V MLIR backend matmul prototype.

## One-time setup

```shell
cp .env.example .env
```

Edit `.env` and set `LLVM_PROJECT_DIR` to the absolute path of your
`llvm-project` checkout, and `UID`/`GID` to your own (`id -u` / `id -g`).
`.env` is gitignored. 

The RISC-V toolchain (clang/lld/pulp-as/etc.) is baked into the container
image itself (`QUIDDITCH_TOOLCHAIN=/opt/quidditch-toolchain`), so there's no
local `toolchain/` directory to set up here. If you ever run `build.sh` or
the driver scripts natively, outside Docker, export `QUIDDITCH_TOOLCHAIN` to
point at an equivalent toolchain checkout yourself.

## Quickstart

```shell
docker compose build
docker compose run --rm snitch ./build.sh --all
docker compose run --rm snitch ./run_matmul32_multicore_pipelined_gvsoc.sh
docker compose run --rm snitch ./run_matmul32_multicore_pipelined.sh
```

Expect `all 1024 values matched` at the end of each run.

For the (potentially long) `--reconfigure-llvm`, first-time gvsoc build, or
the Verilator RTL build below, prefer a persistent container over `run --rm`
so a dropped terminal doesn't kill the build:

```shell
docker compose up -d
docker compose exec snitch ./build.sh --all
```

## gvosc

For this case we use the gvsoc simulator which is much faster but not cycle accurate. Since our goal is just to check for correctness that is fine.

## Verilator (cycle-accurate)

`run_matmul32_multicore_pipelined.sh` (no `_gvsoc` suffix) runs the same
build on a real, cycle-accurate Verilator RTL model instead of gvsoc, via the
`snitch-backend/snitch_cluster` submodule (a
[fork](https://github.com/CAPS-UMU/snitch_cluster) of
`pulp-platform/snitch_cluster`, pinned to a specific commit). `build.sh`
(default, or `--all`, or `--verilator` on its own) verilates the RTL model
(`cfg/default.json`, the same 8-compute+1-DMA-core/128 KB TCDM config gvsoc
already targets) with `TRACE_DMA_ONLY=1`, so only the DMA hart's instruction
trace is emitted, the compute-hart traces are otherwise multi-hundred-MB
per run and aren't needed for correctness checking.

Linking still uses the already-vendored `runtime/snruntime/`. 

Use `./build.sh --gvsoc --xdsl --llvm` (omitting `--verilator`)
if you only need gvsoc for quick correctness iteration.

## GBDT autotuner

`autotune-poc/snitch_tuner/run_gbdt_tuner.py` runs the same build under a GBDT+GA-based autotuning search. 

```shell
docker compose run --rm snitch ./build/.venv/bin/python3 \
  ../autotune-poc/snitch_tuner/run_gbdt_tuner.py --verbose --mode padded --iterations 10
```