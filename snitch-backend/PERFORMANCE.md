# GEMM performance log

This log covers each change that makes the MLIR-compiled f64 GEMM (`run_gemm.py`) faster on the Snitch cluster. For each change it gives what changed, why it should help, and how much it helped when measured.

## How performance is measured

- **Simulator.** Every number is a Verilator run (`simulator=verilator`). gvsoc is not cycle-accurate, so it is used only for correctness checks.
- **Metric.** Cycles are the `dma` column that `snitch_cluster/myrtle-experiments/extract_dma_csv.sh` computes from the DMA hart's trace (`verilator/logs/trace_hart_00008.dasm`). This is the same metric `many_gemms.sh` uses for the hand-written `snitch_cluster` GEMM.
  - `tools/perf_compare.sh N M K n m k` runs one configuration and prints this number.
  - `tools/trace_breakdown.py <elf> <dasm> --M .. --N .. --K ..` splits the kernel into time per function, compute steps and synchronisation gaps.
- **Setup.** Unless stated otherwise: C = A·B at 128×128×128, 8 compute cores, pipelined (double-buffered), `library` microkernel (`gemm_fp64_opt` from `snitch_cluster`).
- **Tile notation.** Tiles are written M×N×K (m×n×k). `run_gemm.py` takes them in the order `n m k`.
- **Ideal.** The lower bound is 128³ / 8 cores = **262,144** cycles of FMAs per core.
- **Reference.** `many_gemms.sh` runs the hand-written kernel (`sw/kernels/blas/gemm_boundary`). On the same 64×24×64 tile it takes **286,776** cycles.

## Results so far

All runs: 128×128×128, pipelined, library microkernel, Verilator, verified.

| Tile (M×N×K) | Baseline (K outermost) | Changes 1–5 | Now (Changes 1–7) | Change 6+7 | Total |
|---|---|---|---|---|---|
| 64×32×32 | 366,576 | 325,476 | 305,177 | −6.2 % | −16.8 % |
| 32×32×32 | 421,476 | 353,677 | 313,876 | −11.3 % | −25.5 % |
| 32×32×64 | 362,176 | 321,176 | 300,277 | −6.5 % | −17.1 % |
| 16×64×32 | 421,777 | 350,676 | 311,177 | −11.3 % | −26.2 % |
| 40×32×64 | 469,577 | 323,976 | 299,977 | −7.4 % | **−36.1 %** |
| 64×64×32 | doesn't fit in L1 | 304,776 | **296,176** | −2.8 % | best tile |
| 24×64×64 | doesn't fit in L1 | 312,176 | 296,677 | −5.0 % | |
| 64×24×64 | doesn't fit in L1 | 318,277 | 302,677 | −4.9 % | |

- **Best tile overall:** 362,176 cycles at the baseline (32×32×64) → 304,776 after Change 5 → **296,176 now (64×64×32), −18.2 % overall**. That's within **3.3 %** of the hand-written kernel's 286,776.
- **Same tile as the reference (64×24×64):** 302,677 vs 286,776, **+5.5 %** (was +11 % after Change 5).

Each change's own before/after numbers are in its section below.

---

## Measurement fix: inputs initialised at compile time (not a speedup)

**What changed.**
- Before: `gemm_main.c` filled A and B with loops on the DM core, and all three arrays were in `.bss`, which the runtime clears on the simulated cores.
- Now: `run_gemm.py` writes `gemm_data.h`, with A, B and a zeroed C as initialised arrays forced into `.data`. They are loaded together with the ELF.

**Why it matters.** The DMA hart's trace covers the whole program, and about **1.68 M** cycles of it were harness setup:

| Part of the old trace (40×32×64) | Cycles |
|---|---|
| A/B init loops | ~1.47 M |
| `.bss` clearing | ~0.2 M |
| The kernel itself | ~0.44 M |

That made our numbers look about 7× worse than `many_gemms.sh`, whose inputs are pre-initialised by its data generator. Runtime startup is now about 2,200 cycles, so the `dma` column is directly comparable with `many_gemms.sh`.

**Effect.** For example, 40×32×64 went from 2,149,577 to 469,577 cycles: the same kernel, measured honestly. All the numbers in this document use the new harness.

---

## Change 1: keep the C tile in L1 across the K loop (double buffering without the K-outermost interchange)

**Before.**
- Pipelined mode tiled with `interchange = [2, 0, 1]`, which puts K outermost.
- The reason: `snitch-pipeline-copy-compute` mis-pipelines a K-innermost loop. With K innermost, the C tile is copied L3→L1 and back on every K step. The pass moved that copy-in into the copy stage, and the copy stage of iteration k+1 runs at the same time as the compute stage of iteration k, so it read a stale partial sum.
- With K outermost, each pipelined iteration works on a different C tile, so it was correct. But C made an L3 round trip on every K tile, and C was double-buffered too.

**What changed.**
- New pass `snitch-hoist-reduction-accumulator` (`mlir/lib/Dialect/Snitch/Transforms/HoistReductionAccumulator.cpp`):
  - runs upstream loop-invariant code motion and `hoistLoopInvariantSubsets` on the K loop;
  - makes the loop carry the padded C tile itself: the pad moves before the loop, the unpadding slice after it, and the per-iteration copy-back is dropped.
- `snitch-promote-operands-to-l1` copies a loop-carried operand into L1 once, at the loop entry.
- `snitch-pipeline-copy-compute` gains a legality check. It refuses to pipeline when a copy depends on a loop-carried value, except when the copy reads a slice that the same iteration writes back (the old K-outermost structure). Previously this case was silently miscompiled.
- `run_gemm.py`: the interchange is removed, K is innermost in both modes, and the hoisting pass runs after L1 tiling.

**Why it is faster.**
1. **Less DMA traffic.** C is loaded and stored once per output tile instead of once per K step, which cuts C traffic by a factor of K/k.
2. **Less L1.** Only A and B are double-buffered: L1 holds 2·(A+B) + C instead of 2·(A+B+C). Larger and better tiles now fit in the 112,640-byte budget, e.g. 64×64×32 (98,304 B) and 64×24×64 (102,400 B). Both needed 114–131 KB before.
3. **Enables later work.** Keeping C in L1 across K gives a well-defined "first K step", which `beta(k)` needs (planned).

**Measured.**
- On tiles that fit in both versions: **−2.5 % to −9.0 %** (table above).
- The best tile overall, 64×64×32, only fits with this change: **318,977** cycles. That's **12 % faster** than the best tile measured before (32×32×64, 362,176).

Correctness: 24/24 configurations verified on gvsoc (pipelined and non-pipelined, xdsl and library, divisible and padded sizes), plus every Verilator run.

---

## Change 2: wait for the current tile before issuing the next one, one barrier per wait group

**Before.** In every pipelined step the DMA core did:
1. start A_{k+1}, start B_{k+1};
2. wait A_k → barrier;
3. wait B_k → barrier;
4. barrier after compute.

The compute cores were held in the first two barriers while the DMA core *issued* the next transfers in software (`snrt_dma_start_2d`). That's about 670 idle cycles per step, and three barriers per step.

**What changed.**
- `snitch-lower-pipeline-op` (`LowerPipelineOp.cpp`): when emitting the stages that run in the same pipeline step (on-ramp, steady state, off-ramp), it puts the later stage's leading `dma.wait_for_transfer` ops (waits for transfers an earlier stage issued in a previous step) **before** the earlier stage's new transfers. Those waits don't depend on the next iteration, so this is always legal.
- `snitch-specialize-dma-code` (`SpecializeDMACode.cpp`): consecutive synchronising ops share one barrier.

Steady state is now: wait A_k, wait B_k → one barrier → issue A_{k+1}, B_{k+1} while the cores compute → barrier after compute.

**Why it is faster.** The compute cores now only wait for data to *arrive*, not for the DMA core to *issue* the next transfers, and there is one barrier fewer per step.

**Measured.**

| Tile | Before | After | Sync gaps before → after | Avg compute step before → after |
|---|---|---|---|---|
| 64×64×32 | 318,977 | 317,077 (**−0.6 %**) | ~20.0k → 13.5k | 18,420 → 18,704 |
| 64×24×64 | 371,676 | 369,777 (**−0.5 %**) | 51.2k → 46.7k | 13,173 → 13,279 |
| 24×64×64 | 394,377 | 387,377 (**−1.8 %**) | | |

The sync gaps shrank as intended, but each compute step got ~100–280 cycles slower. The next transfers now run during compute instead of during the barrier wait. The likely cause is TCDM bank contention between those transfers and the cores' SSR streams; this will be checked with the compute-core profile. Net gain is 0.5–1.8 %, less than the 3–4 % estimated.

Correctness: lit tests (`mlir/test/Dialect/Snitch/lower-pipeline-op.mlir`, `specialize-dma-code.mlir`) and 16/16 gvsoc configurations.

---

## Change 3: zero-fill only remainder tiles, without a barrier

**Before.**
- When a tiled dimension doesn't divide the problem (e.g. N=128 with n=24), every copy of a tile that *may* be a remainder tile (B and C here) was lowered by `StartTensorCopyOp::bufferize` (`mlir/lib/Dialect/DMA/IR/DMAOps.cpp`) to: zero-fill the **whole** L1 buffer → wait → copy the valid part → wait.
- This happened for every tile, including the 5 of 6 N tiles that are full size.
- Worse, `dma.wait_for_transfer` always counted as needing synchronisation, so the zero-fill's wait got a **cluster barrier**. Inside the pipelined K loop, that barrier came before the compute of the current step. The DMA engine works in order, so the zero-fill only completed after the next A tile, issued just before it, had arrived. Compute k therefore waited for the transfer of A_{k+1}, which defeats double buffering on padded tiles.

**What changed.**
- `StartTensorCopyOp::bufferize`: the zero-fill and its wait sit in an `scf.if` that is true only when a padding amount is nonzero at runtime. They are unconditional only when the padding is statically nonzero.
- `WaitForTransferOp`'s `needsSynchronization` (`mlir/lib/Dialect/DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.cpp`) returns false when waiting for a zero-fill. That wait only orders the DMA core's own two transfers into the same buffer; the compute cores still synchronise on the wait for the real copy.

**Why it is faster.**
- Full tiles no longer pay for an extra L1 write of the whole buffer, plus a DMA round trip and a barrier.
- On remainder tiles, the compute cores are no longer stopped by the zero-fill, so double buffering works again.

**Measured.**
- 64×24×64: 369,777 → **353,276 (−4.5 %)**. Sync/fill/drain gaps fell from 46.7k to 28.6k cycles.
- 64×64×32: unchanged at 317,077, as expected, since nothing is padded.

Correctness: lit tests, plus gvsoc on all padded configurations (72×100×90, 100×72×90, 56×40×36, 128³ with 24×64×64 and 64×24×64), pipelined and not, xdsl and library.

---

## Profile: where a microkernel call spends its time (no change, analysis only)

**How it was measured.**
- The Verilator model normally traces only the DMA hart (`TRACE_DMA_ONLY`). A second model that traces every hart is built into `snitch_cluster/target/sim/build-fulltrace`, and `run_gemm.py` uses it when `SNITCH_FULL_TRACE=1`.
- 64×64×32 was run on it, and compute core 0's trace was processed with `gen_trace.py --dump-hart-perf`. Upstream `gemm_fp64_opt` reads `mcycle` at the start and end of its compute, so every call is its own trace region.
- Stalls were attributed by looking at the cycles between consecutive FP instructions.

**Findings for one K step of 64×64×32** (DMA-side step time 18,704; ideal 8 rows × 64 × 32 = 16,384 FMAs):

| | Cycles | Share |
|---|---|---|
| FPU issues FMAs | 16,384 | 87.6 % |
| SSR stream stalls: no A/B operand ready, spread over all 8 FMA slots | ~1,160 | 6.2 % |
| `fld` preload of C (8 per block of 8 outputs, because `beta=1`) | 512 | 2.7 % |
| Call, SSR setup (~35), FPU fence, barrier release | ~620 | 3.3 % |

**Conclusions.**
- The stalls are the same with or without the DMA writing the next tiles during the step: steps with no concurrent DMA stall ~1,120–1,200 cycles, steps with concurrent DMA ~950–1,400. So they come from the 8 cores contending with each other for L1 banks. Every core streams the same B tile, and a core's B stream stays on one group of 8 banks for a whole column block. The hand-written kernel uses the same `gemm_fp64_opt` with the same B sharing, so it very likely pays the same.
- Configuring SSRs only once would save ~35 cycles per call (0.2 %). Not worth it.
- Cyclic row distribution (as in `sc_st_gemm`) does not change bank mapping for these tiles: A rows are a multiple of 256 B apart either way. Not worth it here.
- `beta(k)` would remove the 512 `fld` cycles on the first K step of every output tile only: ≈0.7 % overall.

So the microkernel itself is close to what the hand-written kernel achieves. The remaining gap is padding and per-tile fill/drain.

---

## Change 4: compute remainder tiles at their real size (no padding)

**Before.**
- When a tile size doesn't divide the problem (N=128 with n=24 leaves an 8-column remainder; M=128 with m=40 leaves 8 rows), the remainder tile was zero-padded to the full tile size and computed in full.
- 64×24×64 computes 6 × 24 = 144 columns for 128 real ones (+12.5 % FMAs); 40×32×64 computes 160 rows for 128 (+25 %).

**What changed.**
- `run_gemm.py` (`gen_tile_transform`) peels the last iteration of the M and N tile loops (`transform.loop.peel`) and canonicalises, so the remainder tiles get their own loop nests with **static, smaller** shapes.
- Only K is still padded (to k: the double-buffered K loop needs uniform tiles). M and N are padded only to a multiple of 8, which thread tiling and the library kernel's unroll require: e.g. an N remainder of 8 needs none.
- Each nest gets its own pipeline and microkernel. `run_gemm.py` reads every microkernel's shape and row strides from its call in `step5_riscv.mlir` and compiles the library wrapper per symbol (`-DTILE_M/N/K`, `-DLDA/LDB/LDC`). For 64×24×64 that's 8×24×64 per hart for full tiles and 8×8×64 for the remainder.
- `snitch-lower-l1-allocations` (`LowerL1Allocations.cpp`) now reuses L1 between loop nests that run one after another. A buffer defined at a point in a block is live from there to the end of the block, and the allocations nested in an op are placed above only the buffers already live at that op. Without this, the main and remainder nests would need L1 for both and no longer fit. If a buffer could outlive its block, the pass falls back to the old linear layout and emits a remark.

**Why it is faster.** The FMAs on padding zeros are gone. A remainder tile now costs time in proportion to its real size.

**Measured.**

| Tile | Padded | Peeled | Change |
|---|---|---|---|
| 64×24×64 | 353,276 | **318,277** | **−9.9 %** |
| 40×32×64 | 424,177 | **346,376** | **−18.3 %** |

Dividing tiles (e.g. 64×64×32) are unaffected. On 64×24×64 the gap to `many_gemms.sh` (286,776) shrinks from 23 % to 11 %.

Correctness: lit test `mlir/test/Dialect/Snitch/lower-l1-allocations.mlir`. gvsoc 28/28: 72×100×90, 100×72×90, 56×40×36, and 128³ with 24×64×64, 64×24×64, 40×32×64, 64×64×32; pipelined and not; xdsl and library; staggering on by default where it applies. Verilator verify on every measured run.

---

## Change 5: stagger the column-block order per core

**Before.**
- All 8 cores stream the same B tile.
- Each core's `gemm_fp64_opt` walks B's 8-column blocks in the same order, 0, 1, 2, …, and a core's B stream stays on one group of 8 L1 banks for a whole block.
- So the cores contend for the same banks, and the profile above showed ~1,160 stall cycles per K step on 64×64×32 (6.2 %).

**What changed.**
- `kernels/gemm_library_kernel.cpp`: core c starts at column block `c mod (TILE_N/8)` and wraps around, which takes two `gemm_fp64_opt` calls (columns [n0, N) then [0, n0)) instead of one.
- `run_gemm.py` turns this on per microkernel when it has at least 4 column blocks (`STAGGER_MIN_BLOCKS`). `GEMM_STAGGER_N=0` turns it off, `=1` forces it on.

This staggers the iteration order (loop skewing); it does not swizzle the data layout. With 32 banks and 8-column blocks, 64-wide B tiles still put two cores on each bank group. Pairing it with a layout swizzle or padding of B's rows in L1 is a possible follow-up.

**Why it is faster.** At any moment the cores read different column blocks of B, which mostly live in different L1 banks, so the SSR streams stall less. The second call costs ~35 cycles of SSR setup, which is negligible.

**Measured.**

| Tile | Column blocks per core | Without | With | Change |
|---|---|---|---|---|
| 64×64×32 | 8 | 317,077 | **304,776** | **−3.9 %** (compute step 18,704 → 17,955) |
| 32×32×32 | 4 | 366,977 | **353,677** | **−3.6 %** |
| 64×24×64 | 3 | 353,276 | 355,476 | +0.6 % (so off by default below 4 blocks) |

Correctness: gvsoc and Verilator verify on the measured tiles.

---

## Correctness fix: DMA transfer IDs when requests queue up (not a speedup)

**Symptom.** The first version of Change 6 passed every gvsoc run but failed Verilator verify on 3 of 4 tiles. A few dozen C values were about twice their expected value, for example 3,063 instead of 1,525. They were always the first rows of the later harts in the first tile of an M row. That is the end of the C tile's L1 buffer, the last bytes the C load writes. The cores had started computing before the C load had finished, on top of the previous tile's C.

**Cause (Snitch's iDMA frontend).**
- `dmcpy` returns the frontend's next-ID counter as soon as the request is accepted into the request FIFO (`DMAReqFifoDepth = 3`).
- The counter is incremented only when a request leaves the FIFO (`issue_id = nd_req_valid & nd_req_ready` in `idma_inst64_top.sv`).
- So with the backend busy and one request already waiting in the FIFO, the next `dmcpy` returns the *same* ID as the waiting request. Waiting for it (`completed_id >= id`, which is also how `snrt_dma_wait` waits) returns as soon as the waiting request completes, before the transfer itself has.
- Change 6 issues three transfers back to back (A₀, B₀ of the first tile, then its C load), so the C load got B₀'s ID. Before, no schedule ever had two requests waiting in the FIFO, so the bug never showed. gvsoc's DMA model doesn't have the FIFO, so gvsoc can't catch this.

**What changed.** `convert-dma-to-llvm` (`mlir/lib/Conversion/DMAToLLVM/DMAToLLVM.cpp`, `getTransferId`):
- The hardware issues and retires transfers in order, so a transfer's real ID is one more than the previous one's.
- Every function that starts transfers gets a stack slot (TCDM) holding the last ID. The token of each transfer is `max(returned ID, last ID + 1)`.
- Cost: a load, an add, a `max` and a store on the DMA core per transfer. This wasn't measured separately: it's part of every number from here on.

Correctness: lit test `mlir/test/Conversion/DMAToLLVM/transfer-id.mlir`, and Verilator verify on every run below.

---

## Change 6: issue the next output tile's first A/B copies during the current tile's last K step (step "1b")

**Before.** Each output tile ran its own isolated pipeline. On the DMA core:

```
C load            start(C) ; wait ; barrier            <- cores idle
on-ramp           start(A0) ; start(B0)
steady state      wait(Ak) ; wait(Bk) ; barrier ; start(Ak+1) ; start(Bk+1) ; barrier(compute k done)
off-ramp          wait(Alast) ; wait(Blast) ; barrier ; barrier(compute last done)
C store           start(C) ; wait ; barrier            <- cores idle
```

The cores were idle during the C load, then again until A₀ and B₀ had arrived (nothing was in flight when a tile started), then during the C store.

**What changed.**
- `snitch-lower-pipeline-op` (`LowerPipelineOp.cpp`), new option `prefetch-across-iterations` (default on). When a two-stage pipeline is the body of an `scf.for` (the loop over output tiles), the copy stage of the next tile's first K step moves into the current tile's off-ramp. It goes after the waits for the last A/B (and their barrier) and before the last compute stage:
  ```
  off-ramp          wait(Alast) ; wait(Blast) ; barrier ;
                    if (next tile exists) { start(A0') ; start(B0') } ; barrier(compute last done)
  ```
  The tile loop carries the tokens (and buffers) to the next iteration. Its first K step waits for them instead of issuing them, and the first tile's A₀/B₀ are issued before the loop. This is the "rotate the tile loop around the K pipeline" variant of the route the handover suggested.
- The A/B buffers are now used across tile-loop iterations, so they're hoisted out of the loop into a `memref.alloca_scope` around it. `snitch-lower-l1-allocations` treats the scope like any other region: a peeled remainder nest after the loop still reuses that memory. Once the allocations are lowered it inlines the scope, since later lowerings don't support it.
- It's only applied when it's legal, checked by the pass:
  - The first and last K step use different A/B buffers, i.e. an even number of K steps (pitfall 4 of the handover). The buffers rotate by K-step index; a global slot counter would also allow odd counts, but isn't implemented.
  - The copy stage only starts transfers and computes pure values, and inside the loop it depends only on pure ops and the induction variable.
  - Nothing else in the loop writes memory that the copy stage reads.
  - The pipeline is the only one in the loop. Otherwise the copy wouldn't overlap with the compute that precedes it: e.g. the main N loop and the peeled N remainder of 64×24×64 share the M loop.
  - There are no barriers inside the `scf.if` (pitfall 5): the condition only depends on the loop index, so all cores take the same branch anyway.
- `run_gemm.py`: when the M and N tile loops are perfectly nested (N not peeled), the tiling transform marks the M loop. A new stage after accumulator hoisting coalesces M and N into one loop over output tiles (`transform.loop.coalesce`, `step1a3_coalesced.mlir`). The prefetch then also crosses M rows. This must run after hoisting, because before it the K loop is perfectly nested too and would be coalesced as well. `GEMM_PREFETCH=0` turns both off.

**Why it is faster.** The next tile's A₀/B₀ (16–32 KB) now arrive during the last K step's compute instead of while the cores wait. At a tile boundary the cores now only wait for the C store, the C load and barriers.

**Measured.** Two measurements, because Change 7 changed what limits a tile boundary:

| Tile | Software barrier: without → with | Gaps | Hardware barrier (Change 7): without → with | Gaps |
|---|---|---|---|---|
| 64×64×32 | 304,776 → 305,876 (+0.4 %) | 13.2k → 14.4k | 296,677 → **296,176** (−0.2 %) | 9.4k → 8.5k |
| 64×24×64 | 318,277 → 313,477 (−1.5 %) | 27.8k → 22.8k | 307,677 → **302,677** (−1.6 %) | 24.9k → 18.0k |
| 32×32×32 | 353,677 → 352,276 (−0.4 %) | 34.5k → 33.9k | 317,877 → **313,876** (−1.3 %) | 16.4k → 11.2k |
| 40×32×64 | 323,976 → 317,176 (−2.1 %) | 26.5k → 18.4k | 306,876 → **299,977** (−2.2 %) | 19.5k → 10.2k |

- The "software barrier" columns compare the compiler before this work against Change 6 plus the transfer-ID fix, because Change 6 isn't correct without the fix. The "hardware barrier" columns differ only in `GEMM_PREFETCH`.
- **With the software barrier, the boundary is barrier-bound.** On 64×64×32 the A₀/B₀ wait (~450 cycles) disappeared. But the DMA core then waited ~390 cycles for the compute cores in the next barrier, with no work in between: back-to-back software barriers cost that much (see Change 7). Issuing A₀/B₀ before the first tile's C load also delays that load once (+~600 cycles). Net +0.4 %.
- **With the hardware barrier** the prefetch saves 0.2–2.2 %, most where tiles have few K steps (K/k = 2 on 64×24×64 and 40×32×64). There, the C-load wait and the first step's (already satisfied) A/B waits become adjacent after the one-iteration steady loop is unrolled, and share one barrier: one barrier fewer per tile.
- **Compute steps get 30–80 cycles slower on average** (e.g. 64×24×64: 11,607 → 11,690), because the prefetch now runs during the last K step. Same effect as in Change 2: TCDM contention with the SSR streams.

Correctness:
- Lit tests: `lower-pipeline-op.mlir` covers prefetch, odd trip count, loop writes the source, and option off. `lower-l1-allocations.mlir` covers the alloca scope.
- gvsoc matrix **32/32**: the 28 configurations of Change 4, plus two pipelined ones that exercise the prefetch with padding (all old padded ones have an odd number of K steps). These are 72×100×90 with k=24 (zero-fill in the prefetched copy stage) and 100×72×128 (M and N peeled), each with xdsl and library.
- Verilator verify on every measured run, plus those two extra configurations.

---

## Change 7: hardware cluster barrier instead of the software barrier

**Before.** `convert-snitch-to-llvm` lowered every `snitch.barrier` to `snrt_partial_barrier(&_snrt_barrier, 9)`. Each of the 9 harts does an atomic add on a counter, and the last one bumps an iteration word that the others poll. `_snrt_barrier` is at `0x80063cd8`, in L3. A barrier therefore costs atomics on, and polling of, DRAM. The trace shows ~300 cycles in every K step between the "compute done" barrier and the next one, with nothing to do in between. The "barrier release" in the microkernel profile above was mostly this.

**What changed.**
- `convert-snitch-to-llvm` has a new option `hw-barrier`. It lowers `snitch.barrier` to `csrr x0, 0x7c2`, the cluster hardware barrier: what `snrt_cluster_hw_barrier()` and the hand-written kernel use. It's only correct if every hart of the cluster executes every barrier.
- That is the case here: `run_gemm.py` uses `barrier-participants = 9` (8 compute cores + DMA core = the whole cluster), and `snitch-specialize-dma-code` gives both clones the same barriers. `run_gemm.py` turns the option on, and `GEMM_HW_BARRIER=0` turns it off.
- `tools/trace_breakdown.py` recognises hardware barriers in the trace (the `csrr` stalls until release).

**Why it is faster.** A hardware barrier releases all harts within a few cycles of the last arrival. It replaces ~300–400 cycles per barrier, and there are 2 per K step plus 3–4 per output tile.

**Measured** (without Change 6, `GEMM_PREFETCH=0`):

| Tile | Before | After | Change | Avg compute step | Gaps |
|---|---|---|---|---|---|
| 64×64×32 | 304,776 | 296,677 | **−2.7 %** | 17,955 → 17,693 | 13.2k → 9.4k |
| 64×24×64 | 318,277 | 307,677 | **−3.3 %** | 11,923 → 11,607 | 27.8k → 24.9k |
| 32×32×32 | 353,677 | 317,877 | **−10.1 %** | 4,919 → 4,646 | 34.5k → 16.4k |
| 40×32×64 | 323,976 | 306,876 | **−5.3 %** | 9,165 → 8,849 | 26.5k → 19.5k |

Tiles with more, shorter K steps gain most: 32×32×32 has 160 barriers. Both the compute steps (the "compute done" barrier is part of each step) and the gaps shrink.

Correctness: lit test `mlir/test/Conversion/SnitchToLLVM/barrier.mlir`, gvsoc matrix 32/32, Verilator verify on every measured run.

---

## Where the remaining time goes (after Change 7)

64×64×32 (kernel window 292,017 cycles):

| | Cycles | Share |
|---|---|---|
| Ideal FMAs | 262,144 | 89.8 % |
| Microkernel overhead (SSR stalls left after staggering, `fld` preload, call/fence, barrier) | ~21.4k | 7.3 % |
| Per-output-tile fill/drain + sync | ~8.5k | 2.9 % |

(After Change 5 these were ~87 % / ~25k / ~13k of ~300k.)

What a tile boundary costs now, from `trace_breakdown.py --steps`:

| Tile | C store | C load | First A/B | Per boundary |
|---|---|---|---|---|
| 64×64×32 (32 KB C tile) | ~1,440 | ~640 | overlapped (Change 6) | ~2.1k × 4 tiles |
| 64×24×64 (12 KB C tile) | ~480 | ~320 | overlapped (Change 6) | ~0.8k × 12 tiles |

The C store runs at ~23 B/cycle (writes to L3), the C load at ~50 B/cycle. On 64×24×64, the step from the main N loop into the peeled remainder nest costs another ~1.4k per M row, because the prefetch doesn't cross nests.

Possible next steps, largest first:
1. **No C load, `beta(k)`** (C = A·B, step 1 of the 1b route below): removes the C load (~640 cycles per tile on 64×64×32, ~320 on 64×24×64) and the `fld` preload of the first K step (512 cycles per tile on 64×64×32). Roughly 4–5k on both best tiles, ~1.5 %.
2. **L1 layout swizzle/padding for B**, so the staggered cores land on distinct bank groups. Targets part of the remaining SSR stalls.
3. **A second C buffer where L1 allows** (step 4 of the route) overlaps the C store too. Neither 64×64×32 nor 64×24×64 has room for it.
4. **Prefetch across loop nests** (main N loop → peeled remainder), ~3k on 64×24×64.

---

# Handover: methodology, current state, and step 1b

This section is for whoever continues the work. It covers:
- how the work has been done so far (the methodology to keep following);
- what the code looks like now;
- step "1b: overlap across output tiles": its specification, what Changes 6 and 7 implemented of it, and what's left.

## Methodology

Every change followed the same loop:
1. **Measure and attribute first.** Find the largest measured cost before changing anything, using the tools below. Estimates were often wrong: Change 2 was estimated at 3–4 % and gave 0.5–1.8 %. The "remaining time" table above is the current attribution.
2. **Make one change at a time**, behind the existing pipeline. Keep a way to reproduce the "before" for comparison (see *before/after comparisons*).
3. **Check correctness:**
   - lit tests for every compiler change (`mlir/test/Dialect/Snitch/`, `mlir/test/Conversion/{DMAToLLVM,SnitchToLLVM}/`);
   - the **gvsoc correctness matrix**: pipelined and non-pipelined × `xdsl` and `library` × divisible and non-divisible sizes, `verify=true`;
   - **Verilator `verify=true`** as well: gvsoc doesn't model DMA timing, so it can't catch DMA ordering bugs (see the transfer-ID fix).
4. **Measure performance on Verilator only**, with verify=true, on the reference tiles. Compare the `dma` column and the trace breakdown (useful FMAs / padding / microkernel overhead / sync gaps) against the previous numbers.
5. **Document** each change here: what changed, why it's faster, measured before/after. Keep changes that are correct and faster. Keep changes that are correct and neutral only if they simplify later work (Change 2). Turn changes that help only some shapes into shape-dependent defaults (Change 5).

### Rules learned the hard way

- **Never use gvsoc cycle counts.** gvsoc is not cycle-accurate; use it only for correctness. All performance numbers come from Verilator.
- **The performance metric is the `dma` column** of `snitch_cluster/myrtle-experiments/extract_dma_csv.sh`, run on the DMA hart's trace (`out/<config>/verilator/logs/trace_hart_00008.dasm`). It's the same metric `many_gemms.sh` uses for the hand-written kernel.
- **Don't add `mcycle` reads to `kernels/gemm_main.c`.** `gen_trace.py` starts a new trace region at every `mcycle` read, and `extractDataFromJsons2.py` requires the DMA trace to have exactly one region.
- **`run_gemm.py` argument order is `N M K n m k`**, but tiles are written M×N×K. For the reference tile 64×24×64 (m=64, n=24, k=64) the call is `./run_gemm.py 128 128 128 24 64 64 ...`. Getting this wrong cost one measurement round.
- **gvsoc passing proves little about DMA synchronisation.** The transfer-ID bug passed 32/32 gvsoc runs and failed 3/4 Verilator runs. Any change to the order or overlap of transfers needs Verilator verify runs.
- **Barriers are not free.** The software barrier cost ~300–400 cycles each (Change 7). Count barriers when estimating a schedule change: with the hardware barrier they're a few cycles.
- **The output directory name doesn't include the simulator.** Two runs of the same config at once (e.g. a gvsoc correctness run and a Verilator measurement) overwrite each other's files. For concurrent runs, copy `run_gemm.py` with a suffix on `Config.name` (e.g. `_perf`) and pass it to `tools/perf_compare.sh --driver <copy> --suffix _perf`. Delete the copies afterwards.

### Commands

The commands below assume `/home/daniel/llvm-project` (the repo root) or `snitch-backend/`.

| What | How |
|---|---|
| Build `mlir-opt` (host) | `~/.local/bin/ninja -C build mlir-opt`. `/usr/bin/ninja` on the host is too old. After **adding a source file** to a CMakeLists, first `/usr/local/bin/cmake --regenerate-during-build -S llvm -B build` on the host; the container has no CMake at that path. |
| Lit tests | `build/bin/llvm-lit mlir/test/Dialect/Snitch mlir/test/Conversion/DMAToLLVM mlir/test/Conversion/SnitchToLLVM` |
| One run | `cd snitch-backend && docker compose exec -T snitch ./run_gemm.py N M K n m k <pipelined> <gvsoc\|verilator> <xdsl\|library> <verify>` |
| One Verilator measurement | `tools/perf_compare.sh N M K n m k` (host) prints `dma=<cycles>`. Options: `--driver`/`--suffix`; it forwards `GEMM_STAGGER_N`/`GEMM_PREFETCH`/`GEMM_HW_BARRIER`/`SNITCH_FULL_TRACE`. |
| Where the kernel time goes | `build/.venv/bin/python3 tools/trace_breakdown.py out/<cfg>/gemm.elf out/<cfg>/verilator/logs/trace_hart_00008.dasm --M 128 --N 128 --K 128 [--steps]`. It prints the kernel window, the compute steps (long barrier waits of the DMA hart) and the sync/fill/drain gaps between them. `--steps` lists every barrier interval. It handles both software (`snrt_partial_barrier`) and hardware (`csrr 0x7c2`) barriers. |
| Compute-core profile | `SNITCH_FULL_TRACE=1` selects the Verilator model that traces every hart (`snitch_cluster/target/sim/build-fulltrace`, already built). Then `snitch_cluster/.venv/bin/python3 util/trace/gen_trace.py <trace_hart_0000N.dasm> --mc-exec $PWD/out/snitch-llvm-mc --mc-flags '-disassemble -mcpu=snitch' --dump-hart-perf <json> -o <txt>` inside the container. `gemm_fp64_opt` reads `mcycle` around its compute, so each call is a region. Traces are ~220 MB per hart. |
| gvsoc correctness matrix | Loop over the sizes in the Change 4 section × `true/false` × `xdsl/library` with `gvsoc ... true`, checking for `[+] verified`: 28 runs. Add the two pipelined configurations of Change 6 (`./run_gemm.py 100 72 90 32 32 24 ...` and `72 100 128 32 32 32 ...`, xdsl and library), which exercise the prefetch with padding: 32 runs. About 1 h sequentially, ~10 min with 8 in parallel. For parallel runs, set `OUT` to a separate directory (`docker compose exec -e OUT=$PWD/out_dev ...`) so they don't overwrite Verilator runs in `out/`. |

Run times: a 128³ Verilator run plus extraction takes ~20–30 min. They can run in parallel (the machine has 80 cores, and simulated cycles don't depend on host load) as long as their output directories differ.

**Before/after comparisons.** `mlir-opt` is shared by every driver, so a compiler change can't be switched off per run unless it has its own switch. Keep a driver snapshot (copy of `run_gemm.py`) from before a driver-level change, or put compiler changes behind pass options and environment variables, as `GEMM_STAGGER_N`, `GEMM_PREFETCH` and `GEMM_HW_BARRIER` do. To compare against an earlier compiler, copy `build/bin/mlir-opt` before changing it and point a driver copy's `MLIR_OPT` at the copy. That's how the "before" numbers of Changes 6 and 7 were re-measured, and they reproduced the earlier numbers exactly.

## Current state of the code

None of this is committed yet; it's all uncommitted on branch `autotune-snitch`.

**Compiler** (`mlir/lib/Dialect/...`):
- `Snitch/Transforms/HoistReductionAccumulator.cpp` (new pass `snitch-hoist-reduction-accumulator`): keeps C in L1 across the K loop (Change 1).
- `Snitch/Transforms/PromoteToL1.cpp`: promotes loop-carried operands at the loop entry.
- `Snitch/Transforms/PipelineCopyCompute.cpp`: legality check against loop-carried copies.
- `Snitch/Transforms/LowerPipelineOp.cpp`: hoists waits ahead of the next copies (`insertConcurrentStages`). Prefetches the next tile's first copy stage across iterations of the enclosing loop (`prepareCrossIterationPrefetch`, `insertNextCopyStage`, option `prefetch-across-iterations`, Change 6).
- `Snitch/Transforms/SpecializeDMACode.cpp`: one barrier per group of synchronising ops.
- `Snitch/Transforms/LowerL1Allocations.cpp`: L1 reuse between sequential nests (`assignOffsets`, `mayEscapeBlock`). Inlines `memref.alloca_scope`s once their allocations are lowered.
- `DMA/IR/DMAOps.cpp` (`StartTensorCopyOp::bufferize`): conditional zero-fill.
- `DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.cpp`: no barrier for zero-fill waits.
- `mlir/lib/Conversion/DMAToLLVM/DMAToLLVM.cpp`: corrected transfer IDs (`getTransferId`, the transfer-ID fix).
- `mlir/lib/Conversion/SnitchToLLVM/SnitchToLLVM.cpp`: hardware barrier (option `hw-barrier`, Change 7).
- Tests: `mlir/test/Dialect/Snitch/*.mlir` (6 files), `mlir/test/Conversion/DMAToLLVM/transfer-id.mlir`, `mlir/test/Conversion/SnitchToLLVM/barrier.mlir`.

**Driver** (`snitch-backend/`):
- `run_gemm.py`:
  - no interchange;
  - accumulator hoisting stage;
  - M/N remainder peeling;
  - per-microkernel shapes;
  - staggering default;
  - coalescing of the M/N tile loops and `GEMM_PREFETCH` (Change 6);
  - `hw-barrier` and `GEMM_HW_BARRIER` (Change 7);
  - `SNITCH_FULL_TRACE`.
- `kernels/gemm_main.c` + generated `gemm_data.h`: inputs in `.data`.
- `kernels/gemm_library_kernel.cpp`: `LDA/LDB/LDC`, `STAGGER_N`.
- `tools/`.

**Pass order** (`run_gemm.py`):
1. L1 tiling transform (tile M, N, K with K innermost; peel M/N remainders; pad K, plus M/N to multiples of 8);
2. `snitch-hoist-reduction-accumulator`;
   - coalescing of the marked M/N tile loops (`transform.loop.coalesce`, pipelined only);
3. `snitch-promote-pads-to-l1`, `snitch-promote-operands-to-l1`;
4. `snitch-pipeline-copy-compute` (pipelined only);
5. thread tiling (`tile_using_forall`, 8 cores along M);
6. `snitch-form-microkernels`;
7. `snitch-bufferize`, `snitch-lower-pipeline-op` (with `prefetch-across-iterations`), `snitch-lower-forall-op`;
8. `snitch-lower-l1-allocations`, `snitch-specialize-dma-code`, DMA legalisation;
9. xDSL `convert-to-riscv`;
10. LLVM (`convert-snitch-to-llvm` with `hw-barrier=true`).

## Step 1b: overlap across output tiles

### Status

Implemented as Change 6, with two findings along the way that became the transfer-ID fix and Change 7. In terms of the suggested route below:

| Route step | Status |
|---|---|
| 2. Prefetch the next tile's first A/B during the current tile's last K step | **Done** (Change 6). Implemented in `LowerPipelineOp` as suggested, plus coalescing of the M/N tile loops so that it also crosses M rows. No global slot counter: it only applies with an even number of K steps (pitfall 4). |
| 1. C = A·B, no C load, `beta(k)` | Not done. Now the largest remaining boundary cost (see "Where the remaining time goes"). The reference kernel loads C too, but double-buffers it and uses `beta=0` on the first K step. |
| 3. Asynchronous C store | Not done. With one C buffer the next tile's C load (or, with step 1, its first compute) must wait for the store anyway. What's left to gain is the DMA core's wait before issuing the load, a few dozen cycles. |
| 4. Second C buffer where L1 allows | Not done. Neither of the two best tiles has room (pitfall 1). |

Acceptance criteria (below), as met:
- **Lit tests:** `mlir/test/Dialect/Snitch` and the two new Conversion tests pass, 8 files.
- **gvsoc matrix:** 32/32, the 28 configurations plus 4 that exercise the prefetch with padding. Verilator verify on every measured run, the two extra configurations, and a non-pipelined run.
- **Verilator `dma`:** no regression. With the hardware barrier, the prefetch improves all four reference tiles (−0.2 % to −2.2 %) and reduces their gaps. With the software barrier it was +0.4 % on 64×64×32, which is what led to Change 7.
- **Documentation:** the Change 6 section above, plus the transfer-ID fix and Change 7.

The rest of this section is the original specification, kept for the remaining route steps.

### What we want to achieve

Today each output tile (one (i, j) iteration of the M/N tile loops) runs as its own isolated pipeline. Here is the DMA-core code for one output tile of 64×64×32 (`step3_l1_dma.mlir`, `@matmul$dma`, K/k = 4):

```
C load            start(C) ; wait ; barrier            <- exposed: cores idle
on-ramp           start(A0) ; start(B0)
steady state x3   wait(Ak) ; wait(Bk) ; barrier ; start(Ak+1) ; start(Bk+1) ; barrier(compute k done)
off-ramp          wait(A3) ; wait(B3) ; barrier ; barrier(compute 3 done)
C store           start(C) ; wait ; barrier            <- exposed: cores idle
```

The compute cores sit idle during:
- the C load;
- the first A/B load of the tile (nothing is in flight when a tile starts);
- the C store.

On 64×24×64 the gaps per tile were measured at roughly: C load ~430 cycles, first A/B ~720, between the 2 K steps ~320, C store ~580. That's ~2k per tile × 12 tiles. On 64×64×32 the sync/fill/drain total is ~13k cycles (4.4 % of the kernel).

**Goal:** issue the next tile's loads (and the current tile's store) while the cores are computing, like the hand-written kernel (`snitch_cluster/sw/kernels/blas/gemm_boundary/src/gemm_boundary.h`). That kernel runs **one flattened loop over (output tile, k)**, with one `snrt_cluster_hw_barrier()` per iteration:
- A/B are double-buffered by the global iteration index;
- C is loaded only at k == 0 and stored only at the last k;
- C is double-buffered by output-tile parity (`c_buff_idx = mn % 2`).

So while tile t computes its last K step, the DMA writes back tile t-1's C and loads tile t+1's C and first A/B.

**Expected gain:** ~10k cycles on 64×64×32 (~3 %); ~15–20k on 64×24×64 (~5–6 %); more for tiles with few K steps (K/k = 2).

**Numbers to beat** (Verilator `dma`): when this was written, 64×64×32 = 304,776 and 64×24×64 = 318,277. After Changes 6 and 7: **296,176** and **302,677**. The hand-written kernel on 64×24×64 = 286,776.

### Constraints and pitfalls

1. **L1 budget (the main constraint).** A second C buffer costs one more C tile of L1. With double-buffered A/B, L1 holds 2·(A+B)+2·C instead of 2·(A+B)+C, and the budget is 112,640 B (`l1-memory-bytes` in `Passes.td`). Both current best tiles would no longer fit:
   - 64×64×32: 131,072 B;
   - 64×24×64: 114,688 B.

   So 1b must not force a second C buffer. It has to work with **one** C buffer and use two only when they fit (the driver or a pass option decides).
2. **With one C buffer**, the next tile's C load can only start after the current tile's C store has read the buffer. The DMA engine executes transfers in order, so issuing "store C(t); load C(t+1)" back to back is ordered. But the cores must not start tile t+1 until the load is done.
   - What *can* still overlap is the next tile's first A/B load (A/B have their own double buffers), plus the store/load DMA time overlapping with the last K step's compute if they're issued early enough. The store can't be issued before the last compute finishes, though.
   - So with one C buffer the realistic gain is the first-A/B part (~720 cycles per tile on 64×24×64) plus moving C store/load issue off the cores' critical path where possible.
3. **Removing the C load entirely** (worth considering first). The kernel computes C += A·B because `gen_matmul_mlir` emits a plain `linalg.matmul` into `%out`, and the harness zero-initialises C. If the input becomes `linalg.fill(0)` + `linalg.matmul` (C = A·B, which is what `many_gemms.sh` measures with `beta=0`), the C tile never has to be read from L3:
   - either zero-fill in L1 (`dma.start_zero_mem_transfer`, no L3 traffic), or
   - better, `beta=0` on the first K step of the library kernel (`gemm_fp64_opt`'s nested-FREP path), which also removes the `fld` preload on that step.

   That removes ~430 cycles per tile with no extra L1, and makes the single-C-buffer variant of 1b much simpler. Expect fill fusion work in the tiling transform. The hoisting pass must keep working: the C tile then comes from a fill, not an `extract_slice`.
4. **Two A/B buffer slots rotate by iteration index.** `makeResourceUsageExplicit` in `LowerPipelineOp.cpp` picks slot `(iv / step) % stages`. If the next tile's first A/B is prefetched during the current tile's last K step, its slot must differ from the slot that step is reading:
   - the last step of a tile uses slot (K/k − 1) % 2 and the next tile's first step uses slot 0;
   - with **odd K/k** they collide.

   So either rotate slots by a **global** step counter that carries across tiles (as `gemm_boundary.h` does), or only prefetch when K/k is even.
5. **Barriers in control flow.** `SpecializeDMACode.cpp` inserts a cluster barrier after every synchronising `dma.wait_for_transfer` and states that barriers inside divergent control flow are unsupported. Conditional copies (`scf.if k == 0` / `k == last`) therefore must not contain synchronising waits. Make the waits unconditional, e.g. wait on a `dma.completed_token` when no copy was issued, or keep the conditional part DMA-only. Change 3 shows the pattern for non-synchronising waits (`needsSynchronization` returns false for zero-fill waits).
6. **`snitch-pipeline-copy-compute` legality.** It moves only pure ops into the copy stage, and it refuses copies that depend on loop-carried values, except per-iteration slices written back by the same iteration (`readsPerIterationSubset`). Conditional C copies and a carried accumulator in a flattened loop will hit both checks, so they need deliberate extensions, not removals: the check prevents a real stale-read miscompile, see Change 1.
7. **Bufferization.** An `scf.if` that yields either a fresh L1 C tile or the carried accumulator must bufferize in place. One-Shot Bufferize silently inserts copies when it can't prove that. Inspect `step2a_bufferized.mlir` for unexpected `dma.start_transfer`s between L1 buffers.
8. **Peeled remainder nests.** With M/N remainders (Change 4) there are several tile loop nests, each with its own pipeline. Start by overlapping within each nest. Overlapping from the last tile of one nest into the next nest is a later refinement.
9. **Pipelined mode requires k < K** (`validate()` in `run_gemm.py`). K/k = 1 means no K pipeline; 1b could still pipeline across tiles there, but that's out of scope at first.

### Suggested route (incremental, measure after each)

1. **C = A·B semantics + no C load** (pitfall 3): `fill + matmul` input; zero C in L1 instead of loading it; `beta(k)` in the library kernel as the next refinement. Small and independent, and it removes ~430 cycles per tile.
2. **Prefetch the next tile's first A/B during the current tile's last K step** (one C buffer). This is about rotating the (i, j) loop around the K pipeline at memref level: after `snitch-lower-pipeline-op`, move the on-ramp copy stage of tile t+1 into the off-ramp of tile t, with a global A/B slot counter (pitfall 4). It can be done as an extension of `LowerPipelineOp` when the pipeline is the last stage of an enclosing `scf.for` body, or as a separate pass on the lowered IR.
3. **Asynchronous C store**: issue the store and wait for it only before the C buffer is next written (with one buffer, that's right before the next tile's first compute, or immediately if C isn't loaded any more; see step 1).
4. **Optional, only where L1 allows:** a second C buffer rotated by output-tile parity, for full overlap of store and next tile.

The general alternative is to coalesce the tile loops into one loop (`mlir::coalesceLoops` / `transform.loop.coalesce`) and pipeline that loop with conditional copies. It matches `gemm_boundary.h` exactly, but it hits pitfalls 4–7 all at once. Prefer the incremental route unless it stalls.

### Acceptance criteria

- **Lit tests** for every changed pass, and all of `mlir/test/Dialect/Snitch` passing.
- **gvsoc correctness matrix 28/28** (sizes as in Change 4; pipelined and not; xdsl and library), plus Verilator `verify=true` on every measured run.
- **Verilator `dma`** on 64×64×32, 64×24×64, 32×32×32, 40×32×64: no regression on any of them, and a reduction in the sync/fill/drain gaps reported by `tools/trace_breakdown.py`.
- **A new "Change 6" section** in this file, in the same format as the others.
