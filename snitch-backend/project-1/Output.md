# xDSL → C-library microkernel migration: findings

## Context

Today `run_matmul128_multicore.sh` (and the other `run_matmul*` scripts) lower
a tiled `linalg.matmul` down to the per-hart microkernel level with the
MLIR/transform pipeline, then hand that microkernel off to **xDSL**
(`convert-to-riscv{xdsl-opt-path=...}`), which compiles it straight to Snitch
assembly. That `.S` is extracted (`extract_asm.py`) and assembled with
`pulp-as`, then merged into the final object with `ld.lld -r`. This step was
reported to be slow, and the `snitch_cluster` repo (vendored as a submodule
here) is said to have a faster, more mature C library doing the same job for
BLAS kernels — this investigates whether it can replace xDSL.

## Step 1 — Finding and understanding the C library

**Location:** `snitch_cluster/sw/kernels/blas/gemm/`, specifically
`src/gemm_fp64.h` (and its `fp32`/`fp16`/`fp8` siblings), included from
`src/gemm.h`. This is a small, header-only BLAS library that ships with the
`snitch_cluster` project itself — it is not a separate code-generator tool.
`sw/kernels/blas/gemm/` is exactly the directory `many_gemms.sh` refers to as
`$gemmDir`; its `compile()` step literally runs `make DEBUG=ON sw -j` inside
it.

**How it "generates" assembly:** it doesn't, in the sense of a text-emitting
tool. `gemm_fp64_opt()` is ordinary C with hand-written **inline assembly**
for the innermost FMA sequence, plus calls into `sw/runtime/src/ssr.h`'s
`static inline` helpers (`snrt_ssr_loop_3d/4d`, `snrt_ssr_repeat`,
`snrt_ssr_read/write`, `snrt_ssr_enable/disable`), which themselves are thin
inline-asm wrappers around the Snitch **SSR** (streaming register) and
**FREP** (hardware repeat-loop) custom instructions (`scfgwi`, `frep.o`).
The actual instruction stream is produced by the project's own Clang/LLVM
fork (`-mcpu=snitch`) at `-O3`, the same way any C function is compiled — the
"generator" is just the C compiler, guided by these inline-asm templates.

**Commands to generate code with this library** (as `many_gemms.sh` already
does, driven by `sw/kernels/blas/gemm/data/params.json` /
`scripts/datagen.py`):

```bash
# 1. Set the GEMM shape/params (many_gemms.sh does this via prepareParams.py
#    from a search-space CSV; can also be edited directly):
#    sw/kernels/blas/gemm/data/params.json
#    { m, n, k, transa, transb, beta, gemm_fp: "gemm_fp64_opt", ... }

# 2. Build, from the snitch_cluster root, using its own toolchain
#    (ghcr.io/pulp-platform/snitch_cluster-sw:main, which provides
#    SN_LLVM_BINROOT=/tools/riscv-llvm with -mcpu=snitch support):
make DEBUG=ON gemm -j$(nproc)
#    -> sw/kernels/blas/gemm/build/gemm.elf
#    -> sw/kernels/blas/gemm/build/gemm.dump   (disassembly, since DEBUG=ON)
```

**A build wrinkle worth flagging:** building the *full* app (step 2 above)
also compiles `sw/runtime/impl/snrt.cc`, which uses a `csrr x0, barrier`
pseudo-instruction. The public `ghcr.io/pulp-platform/snitch_cluster-sw:main`
image's Clang 15 does not recognize `barrier` as a named CSR and fails to
assemble it — this looks like a version-skew issue between the vendored
submodule (pinned to a custom `myrtle` fork commit) and the `main`-tagged
public image, unrelated to GEMM itself. It didn't block this investigation
(see Step 2 methodology below) but **will need to be resolved** (matching
toolchain version, or patching `sync.h`) before `make DEBUG=ON gemm` can be
run end-to-end in this environment.

## Step 2 — Comparing the generated assembly

See [`asm/Comparison.md`](asm/Comparison.md) for the full instruction-level
comparison; `asm/xDSL.asm` and `asm/library.asm` are the two raw outputs.

Because of the toolchain wrinkle above, `library.asm` was not produced via
the full `make DEBUG=ON gemm` app build. Instead, `gemm_fp64_opt()` and the
exact `ssr.h` helper functions it calls were copied verbatim into a small
standalone harness, specialized (with compile-time-constant arguments) to
the **same tile shape** xDSL compiled for this pipeline — `M=4, N=40, K=73,
beta=1`, taken directly from `m128mc_xdsl_kernel0.S`'s own SSR configuration
(a 32×40×73 L1 tile, thread-tiled 8-way on `M`, giving `M=4` per hart) — then
compiled with the project's real toolchain (`clang -mcpu=snitch
-menable-experimental-extensions`, from `ghcr.io/pulp-platform/snitch_cluster-sw:main`).
This sidesteps the unrelated runtime-build issue while still producing
codegen that is faithful to what `make DEBUG=ON gemm` would emit for this
exact shape, since it's the same compiler, same flags, same source, same
inline-asm helpers — just without the surrounding application/runtime
scaffolding.

**Bottom line:** both kernels already use the same underlying hardware
mechanisms (SSR streaming + FREP hardware loop for `K`). The library's
compiler-generated code is structurally leaner where it matters: it fully
unrolls the `N` tile and uses simple strength-reduced pointer arithmetic (1
`mul` total, executed once), while the xDSL-generated kernel keeps both
tile loops dynamic and **recomputes every address via multiplication on
every iteration** (24 static `mul`s, ~960 dynamic multiplies for this tile).
The library also explicitly pads its FREP body to hide FMA latency, which
xDSL's kernel does not do. Full details and code snippets in `Comparison.md`.

## Step 3 — Is a swap from xDSL to the library feasible?

**Short answer: yes, but it's a bridge to build, not a flag to flip.**

The two live at different levels of abstraction:

- **xDSL** consumes arbitrary MLIR (post `snitch-form-microkernels`) and must
  handle whatever shapes/strides/layouts the tiling pipeline produces. It's
  a general (if currently unoptimized) compiler backend, invoked generically
  via `convert-to-riscv{xdsl-opt-path=...}`.
- **The C library** is a hand-written, parameter-driven template
  (`gemm_fp64_opt(setup_ssr, partition_banks, transa, transb, M, N, K, A,
  lda, B, ldb, beta, C, ldc)`). It only "generates" the right assembly once
  it's given concrete values — someone has to specialize it per microkernel
  shape, exactly as this investigation's harness did.

So replacing xDSL means adding a small lowering step, not swapping one
`xdsl-opt-path` binary for another. Two ways to do it, in increasing order of
integration effort:

1. **Shim generation (matches what this investigation already prototyped):**
   after `snitch-form-microkernels` isolates the microkernel's shape
   (`M`, `N`, `K`, strides, transpose flags, `beta`), emit a tiny C
   translation unit that calls `gemm_fp64_opt`/`sc_st_gemm` with those values
   as compile-time constants, compile it with the `snitch_cluster` toolchain
   (`-mcpu=snitch`), and link the resulting object in place of the
   xDSL-produced one — i.e. replace the
   "`convert-to-riscv` → `extract_asm.py` → `pulp-as`" leg of
   `run_matmul128_multicore.sh` with
   "`emit shim .c` → `clang -mcpu=snitch -c` → done" (still merged via the
   same `ld.lld -r` step).
2. **Deeper integration:** teach the MLIR pipeline to recognize a
   `linalg.matmul`-shaped microkernel and pattern-match it directly to a
   `gemm_fp64_opt`/`sc_st_gemm` call instead of generalizing it to
   `linalg.generic` for xDSL at all — more invasive, but avoids the
   shim-generation step entirely for the common matmul case.

**What makes this tractable:** the calling convention is a small, mechanical
translation. By the time `snitch-lower-l1-allocations` has run, the operands
are already raw base pointers + strides in TCDM — exactly what
`gemm_fp64_opt` expects. No representation mismatch to bridge.

**What needs care:**
- Toolchain coexistence: the library needs the `snitch_cluster` project's own
  LLVM fork (`-mcpu=snitch`, custom CSRs/mnemonics like `scfgwi`/`frep.o`),
  confirmed here to be a *different* toolchain from the "quidditch" one this
  repo currently uses to compile `main32_multicore.c` (which only supports
  `-mcpu=generic-rv32` and rejects `scfgwi` outright). Both can coexist —
  today's pipeline already assembles xDSL's `.S` with a third tool
  (`pulp-as`) — but a build-system decision is needed on which compiler
  produces the microkernel object going forward.
- The library only covers **matmul** (and a handful of other BLAS/DNN
  kernels under `sw/kernels/`); anything the MLIR pipeline currently hands to
  xDSL that isn't one of those shapes would still need xDSL (or another
  path), so this is realistically a fast-path for matmul-shaped microkernels
  specifically, not a full xDSL replacement.
- The toolchain version-skew issue found in Step 1 (`barrier` CSR) should be
  resolved first so the library can actually be built end-to-end in this
  environment for validation against the golden model, before wiring it into
  the pipeline.

**Recommendation:** prototype option 1 for the matmul path first (it's a
small, self-contained change confined to the tail end of
`run_matmul128_multicore.sh`), validate cycle counts against the existing
Verilator flow, and only invest in option 2 if the shim-generation overhead
itself turns out to matter.
