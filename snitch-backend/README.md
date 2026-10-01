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
docker compose run --rm snitch ./run_gemm.py 32 32 32 16 16 16 true gvsoc xdsl true
docker compose run --rm snitch ./run_gemm.py 32 32 32 16 16 16 true verilator xdsl true
```

Expect `all 1024 values matched` at the end of each run.

## run_gemm.py

A single driver compiles and runs every GEMM configuration:

```shell
./run_gemm.py <N> <M> <K> <n> <m> <k> <pipelined> <simulator> <microkernel> <verify>
```

| Argument | Values | Meaning |
|---|---|---|
| `N M K` | positive ints | C (MxN) = A (MxK) @ B (KxN) |
| `n m k` | positive ints | L1 tile size along N, M, K |
| `pipelined` | `true`, `false` | dual-buffered DMA/compute pipelining |
| `simulator` | `gvsoc`, `verilator` | fast ISS or cycle-accurate RTL model |
| `microkernel` | `xdsl`, `library` | xDSL-generated asm, or snitch_cluster's `gemm_fp64.h` via `kernels/gemm_library_kernel.cpp` |
| `verify` | `true`, `false` | check C against a reference: on gvsoc by the DM core (`gemm_main.c -DVERIFY`); on Verilator on the host by `verify_gemm_ipc.py`, which reads C out of simulated memory over the testbench IPC interface (no extra simulated cycles) |

Constraints (checked up front):
- `m` must be a multiple of the 8 compute cores.
- Dimensions need not be divisible by their tiles: remainder tiles are zero-padded up to the full tile size by the DMA copy into L1, in both modes.
- At least one of M, N, K must be split into more than one tile (at least two when pipelined), due to current bufferization limitations.
- `library` needs `n % 8 == 0`.
- `library` also needs `snitch_cluster/sw/runtime/impl/snitch_cluster_cfg.h`, which `./build.sh --verilator` generates.
- A tile that doesn't fit in L1 fails in `snitch-lower-l1-allocations`.

All intermediates for a configuration go to
`$OUT/gemm_M<M>_N<N>_K<K>_t<m>x<n>x<k>_<pipe|nopipe>_<microkernel>/`.

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

`run_gemm.py ... verilator ...` runs the same
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