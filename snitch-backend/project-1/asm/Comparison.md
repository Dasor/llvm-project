# xDSL vs. C-library microkernel: assembly comparison

Both files implement the **same microkernel shape**: a single-hart tile of
`matmul_128.mlir` after the full `run_matmul128_multicore.sh` tiling pipeline —
`M=4, N=40, K=73`, `f64`, no transpose, `beta=1` (accumulate into `C`). This is
the exact per-hart tile handed to xDSL in step 4 of that script (see
`Output.md` step 1 for how `library.asm` was produced for the identical
shape).

## Headline numbers

| | `xDSL.asm` | `library.asm` |
|---|---|---|
| Total lines | 141 | 347 |
| Real instructions (excl. directives/labels) | 131 | 219 |
| Static `mul` instructions | 24 | 1 |
| Static `fmadd.d` instructions | 4 | 120 |
| Real loops kept dynamic | 2 (nested: M, N/4) | 1 (M only; N/8 fully unrolled) |
| Uses SSR (`scfgwi`) | yes | yes |
| Uses FREP hardware loop (`frep.o`) for K | yes | yes |
| FMA-latency "warm-up" padding | no | yes (dummy `fmadd.d ft2, ...`) |
| Accumulator unroll factor | 4 | 8 |

Both kernels already rely on the **same two hardware features** — SSR
(streaming register file, configured via `scfgwi`) to auto-generate the
address sequences for `A`/`B`/`C`, and FREP (`frep.o`) to turn the `K=73`
reduction into a single hardware-repeated instruction sequence instead of a
real branch-per-iteration loop. So the two are not fundamentally different
strategies — the difference is in the code **surrounding** the hot loop.

## Difference 1: address computation (the big one)

xDSL keeps both the `M` loop and the `N/4` loop as real, dynamic loops and
recomputes every element's byte address from scratch on every iteration:

```asm
scf_body_0_for:
    li t6, 4
    mul t6, t5, t6
    li a3, 40
    mul a3, t3, a3
    add a3, a3, t6
    li t6, 8
    mul a3, a3, t6                               # multiply by element size
    add a3, t2, a3
    fld ft6, 0(a3)                               # load double from memref of shape (4, 40)
```

This 8-instruction dance (2 `li`, 2 `mul`, 2 `add`) repeats for **every one**
of the 4 loads and 4 stores in the loop body, and the whole body runs
`M × N/4 = 4 × 10 = 40` times. That is where the 24 static `mul`s come from,
and dynamically it means **≈960 integer multiplies** just to re-derive
addresses the compiler already knew were a fixed stride apart.

clang, compiling the C library at `-O3`, performs classical strength
reduction: it fully unrolls the `N/8` loop (5 iterations) and turns every
inner address into a loop-invariant base plus a compile-time-constant offset:

```asm
	addi	a2, a4, 64
	#APP
	fld	ft3, 0(a2)
	...
	addi	a2, a4, 128
	...
	addi	a2, a4, 192
	...
	addi	a2, a4, 256
```

Only the outer `M` loop remains dynamic, and its only per-iteration address
update is a single pointer increment (`add a4, a4, a3`) — the entire kernel
has **one** `mul` instruction in it, executed once at setup (`ldc * 8`), not
once per element per iteration.

This is the single most consequential difference: xDSL's `convert-to-riscv`
lowering does not perform loop-invariant-code-motion / induction-variable
strength reduction on the affine index arithmetic it emits, so every
`memref` access is indexed "from scratch". This is a well-understood, fixable
missed optimization, not an inherent limitation of the SSR/FREP hardware.

## Difference 2: FMA-latency hiding

The library's FREP body issues 16 `fmadd.d` per hardware repeat — 8 "real"
accumulations plus 8 throw-away ones into a scratch register `ft2`:

```asm
frep.o a0, 8, 0, 0
fmadd.d ft3, ft0, ft1, ft3
...
fmadd.d fa2, ft0, ft1, fa2
fmadd.d ft2, ft0, ft1, ft3
...
fmadd.d ft2, ft0, ft1, fa2
```

This is the standard Snitch GEMM idiom (documented in the source comment:
*"Unrolling factor of most inner loop. Should be at least as high as the FMA
delay for maximum utilization"*): the extra `ft2` accumulations keep enough
independent multiply-accumulates in flight to hide the FPU's multi-cycle
latency and avoid the sequencer stalling.

xDSL's FREP body has no such padding — just the 4 "real" `fmadd.d`s per
repeat:

```asm
frep.o t6, 4, 0, 0
fmadd.d ft6, ft0, ft1, ft6
fmadd.d ft5, ft0, ft1, ft5
fmadd.d ft4, ft0, ft1, ft4
fmadd.d ft3, ft0, ft1, ft3
```

With only 4 independent accumulator chains and no latency-hiding padding,
this body is more likely to stall waiting for the previous `fmadd.d` on the
same register to retire, depending on the FPU's actual pipeline depth.

## Difference 3: accumulator unroll factor

The library defaults to unrolling 8 elements of `N` per iteration (one
`fmadd.d` chain per accumulator register); xDSL unrolled only 4. Both choices
are valid in principle (N=40 is divisible by both), but the library's factor
is deliberately tied to the hardware's FMA latency, while 4 looks like a
fixed/generic choice from the MLIR-side lowering, independent of the target
FPU's actual latency.

## Net take-away

For the *identical* tile shape and using the *identical* underlying hardware
primitives (SSR + FREP), the C-library version is structurally leaner in the
part that actually costs cycles: it pays for its larger static code size (219
vs. 131 instructions, because it fully unrolls `N/8`) by eliminating almost
all address-recomputation overhead (1 vs. 24 `mul`s) and by explicitly
hiding FMA latency. xDSL's kernel is smaller in static code but pays a real,
avoidable per-iteration integer-arithmetic tax that has nothing to do with
the GEMM math itself.
