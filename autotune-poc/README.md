# Autotuning a linalg.matmul on CPU with upstream MLIR's transform.tune dialect

A proof-of-concept, built entirely from primitives already in this LLVM
checkout (no external "Lighthouse" project needed), reproducing the
knob/constraint/search workflow described in the EuroLLVM 2026 talk
*"Auto-tuning MLIR schedules for Intel GPUs"* (Karna, Morel) -- but for
**CPU** execution via `mlir-runner`, and using `mlir-opt`'s CLI directly
rather than the MLIR Python bindings (this build has
`MLIR_ENABLE_BINDINGS_PYTHON=0`).

Requires `build/bin/mlir-opt` and `build/bin/mlir-runner` to already be
built (they are, at the time this was written). Set `LLVM_BUILD_DIR` if
your build directory isn't `../build` relative to this directory.

## Layout

```
payload/matmul_256.mlir              Step 1: plain linalg.matmul payload, 256x256x256 f32
schedules/baseline_fixed_tiles.mlir  Step 2: payload + transform schedule, literal tile_sizes
schedules/knobs_unresolved.mlir      Step 3: transform.tune.knob left unresolved -> interpreter failure
schedules/knobs_invalid_selection.mlir  Step 3: selected value not in options -> verifier failure
schedules/knobs_resolved_valid.mlir  Step 3: knobs resolved to valid values -> runs, correct output
schedules/smt_constraint_declaration.mlir  Step 3: transform.smt.constrain_params (declarative only, see below)
schedules/tunable_matmul.mlir        Step 4: default input to both drivers -- payload + UNresolved knobs
mlir_schedule.py                     shared knob parse/render/compile/run code, used by both drivers below
driver/run_one.sh                    JIT reference path: runs one resolved schedule through mlir-opt + mlir-runner (Step 2/3 demos only)
driver/compile_schedule.sh           ahead-of-time compiles one resolved schedule to a standalone native executable (used for all timing)
driver/autotune.py                   Step 4: exhaustive grid search over a schedule's knobs
results/grid_search.csv              driver/autotune.py's per-candidate log (generated)
results/best_schedule.mlir           driver/autotune.py's fully-resolved best schedule (generated)

tuner_framework/                     Step 5: hardware-agnostic GBDT+GA search library (no MLIR knowledge)
  interfaces.py                        SearchSpace / FeatureExtractor / Evaluator / EvalResult
  tuner.py                             Tuner: the surrogate-guided search loop itself
mlir_tuner/                          Step 5: MLIR backend for tuner_framework (implements the 3 interfaces)
  backend.py                           KnobSearchSpace / KnobFeatureExtractor / ScheduleEvaluator
  run_gbdt_tuner.py                    CLI entry point, same knobs/flags spirit as driver/autotune.py
results/gbdt_search.csv              run_gbdt_tuner.py's per-candidate log (generated)
results/best_schedule_gbdt.mlir      run_gbdt_tuner.py's fully-resolved best schedule (generated)

custom-tuner/                        Reference only: the original GEMM+simulator GBDT+GA project
                                      tuner_framework/ and mlir_tuner/ are a reusable extraction of its
                                      search algorithm (see interfaces.py's docstrings for the mapping)
```

## Step 1-2: payload + baseline tiling

```sh
./driver/run_one.sh schedules/baseline_fixed_tiles.mlir
```

`A` is filled with `1.0`, `B` with `2.0`, so every output entry is
deterministically `256 * 1.0 * 2.0 = 512`, making correctness trivial to
check regardless of how the matmul got tiled.

## Step 3: how `transform.tune.knob` resolution actually works

A knob is declared with a `name` and a set of `options`:

```mlir
%tile_m = transform.tune.knob<"tile_m"> options = [16, 32, 64] -> !transform.param<i64>
```

There is **no separate resolver pass or C++/Python API** to inject a value
-- this was confirmed by reading the op's implementation
(`mlir/lib/Dialect/Transform/TuneExtension/TuneExtensionOps.cpp`) and its
Python bindings (`mlir/python/mlir/dialects/transform/tune.py`, which only
*construct* new ops, they don't walk/resolve existing ones). The only
mechanism is textual/IR-attribute: rewrite the op to also carry a
`selected` attribute:

```mlir
%tile_m = transform.tune.knob<"tile_m"> = 32 from options = [16, 32, 64] -> !transform.param<i64>
```

Three demonstrations of this, runnable directly:

```sh
# 1. Unresolved knob -> interpreter raises a definite failure
build/bin/mlir-opt schedules/knobs_unresolved.mlir --transform-interpreter
#   error: non-deterministic choice "tile_m" is only resolved through
#   providing a `selected` attr

# 2. selected value not a member of options -> verifier rejects at parse time
build/bin/mlir-opt schedules/knobs_invalid_selection.mlir --transform-interpreter
#   error: 'transform.tune.knob' op provided `selected` attribute is not an
#   element of `options` array of attributes

# 3. Valid resolved knobs -> runs like transform.param.constant
./driver/run_one.sh schedules/knobs_resolved_valid.mlir
```

### Joint constraints (`transform.smt.constrain_params`) -- declarative only

The talk's SMT-based joint-constraint mechanism (e.g. `tile_m * tile_n <=
bound`) exists in this tree (`mlir/include/mlir/Dialect/Transform/SMTExtension/`)
and parses/verifies fine (`schedules/smt_constraint_declaration.mlir`), but
**as of this LLVM revision it has no interpreted semantics**:

```sh
build/bin/mlir-opt schedules/smt_constraint_declaration.mlir --transform-interpreter
#   error: op does not have interpreted semantics yet
```

(Upstream's own test, `mlir/test/Dialect/Transform/test-smt-extension.mlir`,
only ever parses this op too -- it's never run through
`--transform-interpreter` there either.) Practically: `transform.smt.constrain_params`
is a declarative constraint language for an external tool/solver to consume
when *building* the search space; it cannot be left inside a schedule that
actually gets executed. Both drivers therefore enforce constraints
(e.g. tile sizes must not exceed the GEMM dimensions) in plain Python via
`--constraint` -- see Step 4.

`options` is typed `AnyAttr`, so it can technically hold something other
than an explicit array -- one upstream test uses an `affine_set` instead.
But `KnobOp::verify()` only checks membership for `ArrayAttr`; for
anything else it silently skips verification (a debug-only trace, no
error), and `KnobOp::apply()` doesn't inspect `options` at all. That test
is literally named `..._is_unverified`, and it's the only non-array usage
anywhere in the tree -- there's no working symbolic/range option space in
this dialect today (an `options = range<lo,hi>` syntax that seemed to
appear in the talk's slides doesn't exist anywhere in this checkout;
either aspirational or from a downstream fork).

## Step 4: the driver (`driver/autotune.py`)

The driver takes an arbitrary `.mlir` file and does not know in advance
what knobs it contains, how many there are, or what they're named -- it
extracts them from the file's text via a regex over
`transform.tune.knob<"name"> options = [...]`, exactly matching what the
dialect itself accepts (there's no MLIR-IR-level "enumerate the knobs"
API to call instead; see below).

```sh
python3 driver/autotune.py schedules/tunable_matmul.mlir \
  --constraint "tile_m <= M and tile_n <= N and tile_k <= K" \
  --expect-uniform-value 512
```

(All flags are optional -- with no `schedule` argument it defaults to
`schedules/tunable_matmul.mlir`; with no `--constraint` the full cross
product is searched; with no `--expect-uniform-value` a candidate is
"correct" whenever the pipeline just exits 0. `M`/`N`/`K` are just extra
names available to `--constraint` -- they're auto-detected from the
schedule's own `linalg.matmul ins(... : tensor<MxKx...>, tensor<KxNx...>)`
op (`mlir_schedule.detect_matmul_dims`), not passed on the CLI, so they
always match whatever payload the schedule actually embeds. Run `--help`
for the rest.)

What it does, matching the talk's "walker extracts -> oracle picks ->
driver rewrites" picture (slide p.10), done via text templating since
Python IR-walking isn't available here:

1. **Read** the schedule file, **extract** every unresolved
   `transform.tune.knob` (name + its `options` list) via `extract_knobs()`.
   Knobs that already carry a `selected` value are left fixed and untouched
   -- useful for exploring only a subset of a larger schedule's knobs (see
   the mixed-knob behavior verified during development).
2. **Enumerate** candidates over the extracted knobs (exhaustive grid
   search by default), optionally filtered by `--constraint`, a Python
   boolean expression evaluated per candidate against the knob names plus
   the schedule's auto-detected `M`/`N`/`K` -- this is the Python-side equivalent of the
   (currently non-interpretable) SMT constraint above, and the mechanism
   that actually decides validity, independent of whatever's written in
   the schedule's `options = [...]` list.
3. For each candidate, **render** the schedule with those knobs resolved to
   `= <value> from options = [...]` via `render_schedule()`, **compile** it
   ahead-of-time to a standalone native executable via `compile_schedule.sh`
   (not timed), then **run** that executable and time only that, checking
   correctness against its stdout.
4. **Log** every candidate to `--results` (default
   `results/grid_search.csv`); **write** the fastest correct candidate's
   fully-resolved schedule to `--best-schedule` (default
   `results/best_schedule.mlir`) -- a standalone, directly runnable `.mlir`
   file, e.g. `./driver/compile_schedule.sh results/best_schedule.mlir
   /tmp/best_exe && /tmp/best_exe`.

**To plug in a different search algorithm** (the whole point of this
PoC): keep steps 1, 3, and 4 (extract / render+compile+run+measure /
record) exactly as they are, and replace step 2's `itertools.product` grid
enumeration with your own candidate-proposal loop (e.g. genetic algorithm,
Bayesian optimization). The interface it needs is just: given the `knobs`
list `extract_knobs()` returned, produce a `dict[name] -> chosen_option`;
`render_schedule()` and everything after it stays the same.

**Timing**: the tuning objective is `mlir_schedule.run_candidate()`'s own
measurement, not the wall time of a `mlir-opt | mlir-runner` pipeline.
`compile_candidate()` first ahead-of-time compiles the rendered schedule to
a standalone native executable -- `mlir-opt` (same lowering pipeline as
`run_one.sh`) piped through `mlir-translate --mlir-to-llvmir`, `llc
-filetype=obj`, and a system C compiler linking against the same two
runtime libraries `run_one.sh` already uses
(`libmlir_c_runner_utils.so`, `libmlir_runner_utils.so`) -- and this step is
**not timed**. `run_candidate()` then runs that executable
`NUM_TIMING_REPEATS` (5) times, each timed tightly with
`time.perf_counter()`, and keeps the minimum as `kernel_time_s` (the
standard microbenchmark trick: interference can only slow a run down,
never speed one up). Because compilation happens in a wholly separate,
earlier subprocess call, it is excluded from the measured interval by
construction -- not by any clever in-process instrumentation. This fixes an
earlier version of this PoC, which timed the whole `mlir-opt | mlir-runner`
subprocess per candidate: that measurement was dominated by `mlir-opt`'s
lowering passes and `mlir-runner`'s one-time LLVM JIT compilation (both
roughly fixed cost, independent of tile size) and showed no discernible
trend across wildly different tile sizes -- see git history for
`results/grid_search.csv`/`results/gbdt_search.csv`.

One wrinkle: the payload's `func.func @main()` takes no arguments and
returns `void`, which is an ABI mismatch with the standard C entry-point
convention (`int main(int, char**)`) that a real compiled executable's
runtime startup expects. The call itself is harmless (unused incoming
registers are simply ignored on x86-64), but the executable's exit code
becomes effectively undefined rather than reliably 0 -- so
`run_candidate()` does **not** use the exit code as a correctness signal;
only a negative return code (death by signal, e.g. an actual crash) counts
as failure, and correctness is otherwise judged purely from stdout via
`check_uniform_value()`, same as before.

`driver/run_one.sh` (the JIT-based `mlir-opt | mlir-runner` pipeline) is
unchanged and still used for the Step 2/3 manual demonstrations above,
which aren't timing-sensitive.

A real performance shoot-out would still want vectorization
(`transform.structured.vectorize`) and larger problem sizes -- this fix
only removes compile/JIT noise from the measurement, it doesn't change
what's being measured (a naive, unvectorized scalar loop nest at 256^3).

## Step 5: a second search strategy -- GBDT + genetic algorithm (`tuner_framework/`, `mlir_tuner/`)

`driver/autotune.py` always exhaustively enumerates the knob space, which
is fine for 27 candidates and would not be fine for thousands. `tuner_framework/`
is a hardware-agnostic search library extracted from a separate reference
project (`custom-tuner/`, not part of this PoC, kept for comparison) that
tunes GEMM tile sizes on custom hardware via an XGBoost surrogate model +
genetic algorithm (Ansor-style: iteration 1 evaluates random candidates for
real and trains an initial model; iterations 2+ run a GA *entirely against
the surrogate's predictions*, and only the GA's top-N candidates per
iteration get evaluated for real, after which the model is retrained on
everything accumulated so far).

`tuner_framework/interfaces.py` defines three seams (`SearchSpace`,
`FeatureExtractor`, `Evaluator`) that `tuner_framework/tuner.py`'s `Tuner`
class drives without knowing anything about GEMMs, MLIR, or hardware
simulators. `mlir_tuner/backend.py` implements those three interfaces for
this PoC's knob-based schedules, reusing the exact same `mlir_schedule.py`
parsing/rendering/running code `driver/autotune.py` uses -- only the search
*strategy* differs between the two drivers, not how a candidate becomes a
schedule or how a schedule gets run.

`KnobSearchSpace` never reads a knob's file-declared `options` for
validity at all. The search space is defined entirely in Python by two
things the caller supplies: a `sample_value(name, rng) -> str` that draws
one raw candidate value (from whatever domain the caller wants -- a range,
a formula, anything), and a `constraint(values) -> bool` that is the SOLE
authority on which candidates are valid -- exactly like a constraint in an
SMT/ILP formulation, expressing the search space mathematically instead of
as an enumerated list. There's no constraint solver wired in, so this is
rejection sampling: draw raw values, keep the candidate only if
`constraint` accepts it, retry otherwise.

This works end to end because `render_schedule()` never reuses the file's
original `options` text either -- it always writes back a fresh singleton
`options = [value]` for whatever was chosen, and `KnobOp`'s verifier only
ever checks that `selected` is a member of `options`, nothing more
(`apply()` doesn't read `options` at all). `extract_knobs()` is only ever
used to discover which knob *names* are unresolved in a file (so the
tuner knows what to search over and rewrite); it's never the authority on
what values are legal.

`run_gbdt_tuner.py` uses a deliberately naive default `sample_value`
(uniform random integer over `[1, max(M, N, K)]`, identical for every
knob) and relies entirely on `--constraint` to shape the space -- a real
backend should replace that sampler with something structured (powers of
two, divisors of a dimension, etc.) by constructing `KnobSearchSpace`
directly instead of going through this generic CLI. Verified directly: run
with just `--constraint "tile_m <= M and tile_n <= N and tile_k <= K"`
(no other guidance) against `tunable_matmul.mlir` (whose own `options`
only ever declare `[16, 32, 64]`), the search evaluated real candidates
like `tile_m=53, tile_n=128, tile_k=7` -- arbitrary integers, none of them
divisors of 256, none of them anywhere in the file -- and that schedule
rendered, verified, and ran correctly through the real pipeline (MLIR's
`tile_using_for` generates the necessary boundary/remainder tiles
automatically for non-divisor tile sizes).

```sh
python3 mlir_tuner/run_gbdt_tuner.py schedules/tunable_matmul.mlir \
  --constraint "tile_m <= M and tile_n <= N and tile_k <= K" \
  --expect-uniform-value 512 \
  --iterations 4 --init-samples 6 --ga-population 10 --ga-generations 8
```

Same output shape as `driver/autotune.py`: a per-candidate CSV
(`results/gbdt_search.csv`) and a standalone resolved best schedule
(`results/best_schedule_gbdt.mlir`).

See `tuner_framework/interfaces.py`'s docstrings for the precise mapping
from each interface method to the `custom-tuner/` function it replaces
(e.g. `SearchSpace.repair()` <-> `ga_repair.py`'s `repair()`), and
`tuner_framework/tuner.py`'s module docstring for the two deliberate
departures from `custom-tuner/`'s `ga.py` (no DEAP dependency; no
pool-then-`head(N)` indirection for the initial random sample).

## Does this work on CPU? (yes)

The `transform.tune` and `transform.smt` extensions are entirely
dialect-agnostic -- they just resolve param values fed into ordinary
transform ops (here, `transform.structured.tile_using_for`, which lowers
through the fully generic Linalg -> SCF -> LLVM CPU path). The GPU-specific
parts of the talk (XeGPU dialect, DPAS layout annotations, Xe prefetch
ops) are a separate, unrelated lowering path relevant only to Intel GPU
codegen -- they have no bearing on the knob/constraint mechanism itself,
which is exactly what's exercised here.
