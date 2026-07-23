"""Concrete tuner_framework backend for MLIR transform.tune.knob schedules.

Wires SearchSpace / FeatureExtractor / Evaluator (see
../tuner_framework/interfaces.py) to the same knob-parsing/rendering/running
code driver/autotune.py's exhaustive grid search already uses
(../mlir_schedule.py) -- this backend only swaps the *search strategy*,
from "enumerate every combination" to the GBDT+GA loop in
tuner_framework/tuner.py. How a candidate becomes a schedule, and how a
schedule gets run, is unchanged.

Candidate representation: Dict[str, str] mapping knob name -> chosen value
(as text, ready to splice into the rendered schedule). Values are never
read from the schedule's own `options = [...]` list -- see
KnobSearchSpace's docstring for how the search space is instead defined
entirely in Python, as a sampler plus a constraint.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_POC_DIR = _HERE.parent
sys.path.insert(0, str(_POC_DIR))

from mlir_schedule import Knob, run_candidate, token_to_python  # noqa: E402
from mlir_tuner import hw_detect  # noqa: E402
from tuner_framework.interfaces import EvalResult, Evaluator, FeatureExtractor, SearchSpace  # noqa: E402

MlirCandidate = Dict[str, str]


class KnobSearchSpace(SearchSpace):
    """Search over a fixed set of knob names discovered in a schedule file
    via mlir_schedule.extract_knobs() -- but NOT over their file-declared
    `options`. `knobs` is only ever used to find out which knob *names*
    are unresolved (so we know which ops need resolving); their `options`
    field is never read.

    The search space itself is defined entirely in Python, by things the
    caller supplies:

      - `sample_value(name, rng) -> str`: draws one raw candidate value for
        a given knob name, used by `mutate()` to perturb one knob of an
        existing candidate. Whatever domain this draws from -- a range, a
        formula, a fixed list, anything -- is entirely the caller's
        choice; this class has no opinion on it.
      - `init_sample_value(name, rng) -> str`, optional: like
        `sample_value`, but used by `sample_random()` instead -- i.e. for
        drawing a brand-new candidate from scratch (the initial random
        iteration, and topping up the GA's population), as opposed to
        perturbing one that already exists. Defaults to `sample_value`
        itself if not given. Splitting these lets a caller seed the
        from-scratch case with a stronger heuristic (e.g. powers of two
        for GEMM tile sizes) while keeping `mutate()`'s fine-grained
        perturbation unrestricted, so the GA can still refine away from
        that heuristic once real data guides it.
      - `constraint(values) -> bool`: the SOLE authority on whether a
        candidate (raw values for every knob) is valid. No value is ever
        rejected or accepted because of what the schedule file says --
        only `constraint` decides. This is the Python-side equivalent of
        transform.smt.constrain_params (see ../README.md's note on why
        that op can't be embedded in an executable schedule today), except
        here it's not just a filter on an existing list -- it's the actual
        mathematical definition of the space, exactly like a constraint in
        an SMT/ILP formulation.

    No constraint solver is wired in, so validity is enforced via
    rejection sampling: draw raw values with `sample_value`, keep the
    candidate only if `constraint` accepts it, retry (up to
    `max_attempts`) otherwise. If `constraint` is very restrictive relative
    to what `sample_value` tends to produce, this can waste attempts or
    fail outright -- pick `sample_value` to already lean toward what
    `constraint` wants.

    Whatever value ends up chosen, mlir_schedule.render_schedule() always
    writes it back as a fresh singleton `options = [value]`, so the
    rendered schedule is valid regardless of what the source file's
    `options` list ever said (see that function's docstring for why
    that's safe) -- rendering never needs the file's original list to stay
    legal.
    """

    def __init__(
        self,
        knobs: Sequence[Knob],
        sample_value: Callable[[str, object], str],
        constraint: Optional[Callable[[Dict[str, object]], bool]] = None,
        max_attempts: int = 1000,
        init_sample_value: Optional[Callable[[str, object], str]] = None,
    ):
        self.names = [k.name for k in knobs if not k.already_resolved]
        if not self.names:
            raise ValueError("no unresolved transform.tune.knob ops to tune")
        self.sample_value = sample_value
        self.init_sample_value = init_sample_value or sample_value
        self.constraint = constraint
        self.max_attempts = max_attempts

    def _satisfies_constraint(self, candidate: MlirCandidate) -> bool:
        if self.constraint is None:
            return True
        values = {name: token_to_python(tok) for name, tok in candidate.items()}
        return self.constraint(values)

    def sample_random(self, rng) -> Optional[MlirCandidate]:
        for _ in range(self.max_attempts):
            candidate = {name: self.init_sample_value(name, rng) for name in self.names}
            if self._satisfies_constraint(candidate):
                return candidate
        return None

    def mutate(self, candidate: MlirCandidate, rng) -> MlirCandidate:
        name = rng.choice(self.names)
        mutated = dict(candidate)
        mutated[name] = self.sample_value(name, rng)
        return mutated

    def crossover(self, a: MlirCandidate, b: MlirCandidate, rng) -> Tuple[MlirCandidate, MlirCandidate]:
        child1 = {name: rng.choice([a[name], b[name]]) for name in self.names}
        child2 = {name: rng.choice([a[name], b[name]]) for name in self.names}
        return child1, child2

    def repair(self, candidate: MlirCandidate) -> Optional[MlirCandidate]:
        return dict(candidate) if self._satisfies_constraint(candidate) else None

    def key(self, candidate: MlirCandidate):
        return tuple(sorted(candidate.items()))


class KnobFeatureExtractor(FeatureExtractor):
    """One numeric feature per knob (its chosen value, parsed to float where
    possible; one-hot'd if it's not numeric, e.g. a string/bool knob).

    This is a generic default, not a domain-tuned one -- custom-tuner/'s
    ga.py additionally engineers features like L1 footprint and per-tile
    byte counts from (m, n, k) via TSA_C_Remainder. A schedule-specific
    subclass can do the same here by overriding extract() to call
    super().extract() and add derived columns (e.g. tile_m * tile_n).
    """

    def extract(self, candidate: MlirCandidate) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for name, token in candidate.items():
            value = token_to_python(token)
            try:
                features[name] = float(value)
            except (TypeError, ValueError):
                features[f"{name}={value}"] = 1.0
        return features


# Matches the element type of a tensor operand in the schedule text, e.g.
# 'tensor<256x256xf32>' -> 'f32'.
_DTYPE_RE = re.compile(r"x(f16|bf16|f32|f64|i8|i16|i32|i64)>")
_DTYPE_BYTES = {
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
}


def detect_dtype_bytes(schedule_text: str, default: str = "f32") -> int:
    """Best-effort element size (bytes) for GEMM byte-footprint features,
    via regex over the schedule's own text (e.g. 'xf32' -> 4) -- there's
    no MLIR-IR-level type query available here (see mlir_schedule.py's
    module docstring on why this whole file is regex-over-text). Falls
    back to `default` (f32, this repo's only payload dtype today -- see
    payload/matmul_256.mlir) if no tensor element type is found.

    Public (no leading underscore) because run_gbdt_tuner.py also needs
    it, to size hw_detect.max_square_tile_dim's elem_bytes argument the
    same way GemmKnobFeatureExtractor sizes its own byte-footprint
    features -- both should agree on what one tile element costs."""
    m = _DTYPE_RE.search(schedule_text)
    dtype = m.group(1) if m else default
    return _DTYPE_BYTES[dtype]


class GemmKnobFeatureExtractor(KnobFeatureExtractor):
    """Adds BLIS-style derived features (Low et al., "Analytical Modeling
    Is Enough for High-Performance BLIS") on top of KnobFeatureExtractor's
    generic per-knob columns, for schedules that tile a linalg.matmul by
    tile_m/tile_n/tile_k (as schedules/tunable_matmul.mlir does): per-tile
    byte footprints for the A/B/C blocks touched by one tile iteration,
    their ratio to this host's actual L1/L2 capacity, arithmetic
    intensity, and vector-width alignment -- the quantities that paper
    identifies as actually driving GEMM tiling performance, as opposed to
    the raw tile sizes KnobFeatureExtractor emits on their own. (This
    schedule doesn't use an explicit linalg.pack step -- it's plain
    transform.structured.tile_using_for -- so "tile footprint" here means
    the m*k/n*k/m*n working set touched per tile iteration, which is the
    same quantity the BLIS cache-residency analysis cares about
    regardless of whether it's been repacked into contiguous buffers.)

    Element size (bytes) is detected once at construction time by
    regexing `schedule_text` itself (see _detect_dtype_bytes) -- not a
    constructor flag. Cache sizes and vector width come from
    hw_detect.detect(), which auto-detects this host's hardware -- also
    not a constructor flag, deliberately: this PoC always compiles+runs
    candidates natively on whatever host the script runs on (see
    ScheduleEvaluator's docstring), so "the target" is always just this
    machine, with no separate config to keep in sync.

    extract() only adds the derived columns when tile_m, tile_n, and
    tile_k are ALL present in the candidate; otherwise it's identical to
    KnobFeatureExtractor.extract(). This makes it a safe drop-in
    replacement for the generic extractor even against schedules whose
    knobs aren't a GEMM tiling -- they still get the base per-knob
    columns, just none of the derived ones, instead of a KeyError.
    """

    _REQUIRED_KNOBS = ("tile_m", "tile_n", "tile_k")

    def __init__(self, schedule_text: str):
        self._elem_bytes = detect_dtype_bytes(schedule_text)

    def extract(self, candidate: MlirCandidate) -> Dict[str, float]:
        features = super().extract(candidate)
        if not all(name in candidate for name in self._REQUIRED_KNOBS):
            return features

        hw = hw_detect.detect()
        elem_bytes = self._elem_bytes
        m = int(token_to_python(candidate["tile_m"]))
        n = int(token_to_python(candidate["tile_n"]))
        k = int(token_to_python(candidate["tile_k"]))

        tile_a_bytes = float(m * k * elem_bytes)
        tile_b_bytes = float(n * k * elem_bytes)
        tile_c_bytes = float(m * n * elem_bytes)
        total_tile_bytes = tile_a_bytes + tile_b_bytes + tile_c_bytes

        flops = 2.0 * m * n * k  # one fused multiply-add per (m, n, k) point
        arithmetic_intensity = flops / total_tile_bytes if total_tile_bytes > 0 else 0.0

        vector_width_elems = max(hw.vector_width_bytes // elem_bytes, 1)

        features.update({
            "tile_a_bytes": tile_a_bytes,
            "tile_b_bytes": tile_b_bytes,
            "tile_c_bytes": tile_c_bytes,
            "total_tile_bytes": total_tile_bytes,
            "tile_bytes_over_l1": total_tile_bytes / hw.l1_bytes,
            "tile_bytes_over_l2": total_tile_bytes / hw.l2_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "tile_m_vector_aligned": 1.0 if m % vector_width_elems == 0 else 0.0,
            "tile_n_vector_aligned": 1.0 if n % vector_width_elems == 0 else 0.0,
        })
        return features


class ScheduleEvaluator(Evaluator):
    """Runs each candidate via mlir_schedule.run_candidate(): ahead-of-time
    compiles it to a standalone native executable (compile_schedule.sh),
    then times only that executable's execution -- `objective` is the min
    wall-clock time (seconds, lower is better) across a few repeats, which
    excludes mlir-opt/JIT compile overhead by construction since
    compilation is a separate, untimed step. See README.md's timing note.

    evaluate_batch() is a plain sequential loop: there's nothing to batch
    here (each candidate's schedule differs), unlike custom-tuner/'s
    many_gemms.sh which compiles+simulates a whole CSV of configs together.

    `timeout_factor` bounds how long a bad candidate is allowed to run, and
    `compile_timeout_factor` bounds how long it's allowed to *compile*: this
    evaluator remembers the fastest valid candidate's `kernel_time_s` and
    `compile_time_s` seen so far (across every evaluate_batch() call over
    this instance's lifetime, not just the current batch), and any later
    candidate is killed -- via run_candidate()'s `run_timeout_s` /
    `compile_timeout_s` -- the moment it runs, respectively compiles,
    longer than the matching factor times that best time, then reported as
    an invalid candidate (same as a correctness failure) rather than run
    to completion. Both are independent: a candidate can be killed for
    compiling too slowly before it's even run. Neither kicks in until a
    first fully-valid candidate has been seen; the very first candidate(s)
    always run to completion, which is also what seeds both baselines.
    Set either factor to None (or <= 0) to disable that one check and
    always let that phase run to completion, as before.
    """

    def __init__(
        self,
        schedule_text: str,
        compile_script: Path,
        expect_uniform_value: Optional[str] = None,
        verbose: bool = False,
        timeout_factor: Optional[float] = 1.5,
        compile_timeout_factor: Optional[float] = 1.5,
    ):
        self.schedule_text = schedule_text
        self.compile_script = compile_script
        self.expect_uniform_value = expect_uniform_value
        self.verbose = verbose
        self.timeout_factor = timeout_factor
        self.compile_timeout_factor = compile_timeout_factor
        self._best_time_s: Optional[float] = None
        self._best_compile_time_s: Optional[float] = None

    def evaluate_batch(self, candidates: Sequence[MlirCandidate]) -> List[EvalResult]:
        return [self._evaluate_one(c, i, len(candidates)) for i, c in enumerate(candidates, 1)]

    @staticmethod
    def _describe(candidate: MlirCandidate) -> str:
        return " ".join(f"{name}={value}" for name, value in sorted(candidate.items()))

    def _evaluate_one(self, candidate: MlirCandidate, index: int, total: int) -> EvalResult:
        if self.verbose:
            print(f"    [{index}/{total}] running: {self._describe(candidate)}", flush=True)
        run_timeout_s = None
        if self.timeout_factor is not None and self.timeout_factor > 0 and self._best_time_s is not None:
            run_timeout_s = self._best_time_s * self.timeout_factor
        compile_timeout_s = None
        if (self.compile_timeout_factor is not None and self.compile_timeout_factor > 0
                and self._best_compile_time_s is not None):
            compile_timeout_s = self._best_compile_time_s * self.compile_timeout_factor
        start = time.monotonic()
        result = run_candidate(
            self.schedule_text, candidate, self.compile_script, self.expect_uniform_value,
            run_timeout_s=run_timeout_s, compile_timeout_s=compile_timeout_s,
        )
        elapsed = time.monotonic() - start
        if not result.correct or result.kernel_time_s is None:
            if self.verbose:
                if "compile timed out" in result.stderr:
                    reason = "compile timed out"
                elif "timed out" in result.stderr:
                    reason = "timed out"
                else:
                    reason = "invalid"
                print(f"    [{index}/{total}] done in {elapsed:.2f}s: {reason} "
                      f"({self._describe(candidate)})", flush=True)
            if result.stderr:
                sys.stderr.write(result.stderr if result.stderr.endswith("\n") else result.stderr + "\n")
            return EvalResult(valid=False, objective=None)
        if self._best_time_s is None or result.kernel_time_s < self._best_time_s:
            self._best_time_s = result.kernel_time_s
        if result.compile_time_s is not None and (
                self._best_compile_time_s is None or result.compile_time_s < self._best_compile_time_s):
            self._best_compile_time_s = result.compile_time_s
        if self.verbose:
            print(f"    [{index}/{total}] done in {elapsed:.2f}s: "
                  f"objective={result.kernel_time_s:.6g}s ({self._describe(candidate)})", flush=True)
        return EvalResult(valid=True, objective=result.kernel_time_s)
