# snitch-backend

Self-contained container setup for the Snitch RISC-V MLIR backend matmul prototype.

## One-time setup

```shell
cp .env.example .env
```

Edit `.env` and set `LLVM_PROJECT_DIR` to the absolute path of your
`llvm-project` checkout, and `UID`/`GID` to your own (`id -u` / `id -g`).
`.env` is gitignored. 

## Quickstart

```shell
docker compose build
docker compose run --rm snitch ./build.sh --all
docker compose run --rm snitch ./run_matmul32_multicore_pipelined_gvsoc.sh
```

Expect `all 1024 values matched` at the end of the run.

For the (potentially long) `--reconfigure-llvm` or first-time gvsoc build,
prefer a persistent container over `run --rm` so a dropped terminal doesn't
kill the build:

```shell
docker compose up -d
docker compose exec snitch ./build.sh --all
```

## gvosc

For this case we use the gvsoc simulator which is much faster but not cycle accurate. Since our goal is just to check for correctness that is fine.