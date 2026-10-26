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
docker compose run --rm snitch ./run_gemm.py 32x32x32w16-16-16 --microkernel xdsl --simulator gvsoc
docker compose run --rm snitch ./run_gemm.py 32x32x32w16-16-16 --microkernel xdsl --simulator verilator
```

Expect `all 1024 values matched` at the end of each run.

## run_gemm.py

A single driver compiles and runs every GEMM configuration:

```shell
./run_gemm.py <MxNxKwm-n-k> --microkernel xdsl|library --simulator gvsoc|verilator \
              [--tiling remainder|pad] [--pipelined true|false] [--verify true|false] [--keep-trace]
```

| Argument | Values | Meaning |
|---|---|---|
| `MxNxKwm-n-k` | e.g. `128x128x128w64-24-64` | C (MxN) = A (MxK) @ B (KxN), L1 tile m x n x k (the `many_gemms.sh` notation) |
| `--microkernel` | `xdsl`, `library` | xDSL-generated asm, or snitch_cluster's `gemm_fp64.h` via `kernels/gemm_library_kernel.cpp` |
| `--simulator` | `gvsoc`, `verilator` | fast ISS (correctness only) or cycle-accurate RTL model (performance) |
| `--tiling` | `remainder` (default), `pad` | tiles that don't divide the problem are computed at their real size, or zero-padded to the full tile size |
| `--pipelined` | `true` (default), `false` | dual-buffered DMA/compute pipelining |
| `--verify` | `true` (default), `false` | check C against a reference: on gvsoc by the DM core (`gemm_main.c -DVERIFY`); on Verilator on the host by `verify_gemm_ipc.py`, which reads C out of simulated memory over the testbench IPC interface (no extra simulated cycles) |
| `--keep-trace` | | Verilator: keep the DMA hart's trace after extracting the CSV (e.g. for `tools/trace_breakdown.py`) |

On Verilator, the DMA hart's trace is turned into `verilator/logs/<config>.csv`
with the same scripts as `many_gemms.sh` (`gen_trace.py`,
`extractDataFromJsons2.py`); its `dma` column, printed at the end of the run,
is the performance metric. This needs a snitch-aware `llvm-mc` at
`out/snitch-llvm-mc`, which `snitch_cluster/myrtle-experiments/extract_dma_csv.sh`
copies there once.

Constraints (checked up front):
- `m` must be a multiple of the 8 compute cores.
- Dimensions need not be divisible by their tiles. With `--tiling remainder`, M/N remainder tiles get their own peeled loop nest at their real size (M padded only to a multiple of 8, and N to a multiple of 8 for `library`), and the K remainder tile is computed at its real size in the last step of the (uniform, dual-buffered) K loop. With `--tiling pad`, remainder tiles are zero-padded to the full tile size by the DMA copy into L1.
- At least one of M, N, K must be split into more than one tile, due to a current bufferization limitation.
- Pipelined mode dual-buffers the K loop, so it needs `k < K`.
- `library` needs `n % 8 == 0`, `k >= 2` and a K remainder of at least 2.
- `library` also needs `snitch_cluster/sw/runtime/impl/snitch_cluster_cfg.h`, which `./build.sh --verilator` generates.
- A tile that doesn't fit in L1 fails in `snitch-lower-l1-allocations`. The C tile stays in L1 across the K loop (`snitch-hoist-reduction-accumulator`), so L1 holds A + B + C non-pipelined and 2·(A + B) + C pipelined (f64, 8 bytes per element; budget 112,640 bytes).
- Pipelined mode also issues the next output tile's first A and B copies during the current tile's last K step, when the K loop has an even number of steps (see `PERFORMANCE.md`, Change 6). `GEMM_PREFETCH=0` turns this off.
- `snitch.barrier` is lowered to the cluster's hardware barrier, since every barrier involves all 9 harts (see `PERFORMANCE.md`, Change 7). `GEMM_HW_BARRIER=0` selects the runtime's software barrier instead.
- `GEMM_K_REMAINDER=0` pads the K remainder tile even with `--tiling remainder` (see `PERFORMANCE.md`, Change 8).

All intermediates for a configuration go to
`$OUT/gemm_<MxNxKwm-n-k>_<rem|pad>_<pipe|nopipe>_<microkernel>_<simulator>/`.

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