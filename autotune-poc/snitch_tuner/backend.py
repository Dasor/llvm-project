"""Concrete tuner_framework backend for the Snitch Milestone 6 pipeline
(dual-buffered pipelining + all 8 compute cores): tunes
schedules/tile_l1_pipelined_tunable.mlir's tile_m/tile_n/tile_k
transform.tune.knob values by actually compiling the full xDSL/LLVM
pipeline and running the result on gvsoc via driver/tune_candidate.sh.

Mirrors ../mlir_tuner/backend.py's shape (same three interfaces, same
Dict[str, str] candidate representation, same render_schedule() knob
protocol from ../mlir_schedule.py) but the CPU-specific pieces are swapped
for Snitch-specific ones:

  - KnobSearchSpace / KnobFeatureExtractor are reused UNMODIFIED from
    mlir_tuner.backend -- neither has anything CPU-specific in it (see
    their docstrings there).
  - GemmKnobFeatureExtractor's host-cache-autodetected L1/L2 sizes
    (mlir_tuner/hw_detect.py) don't make sense here: the "hardware" is
    always the same simulated Snitch cluster, not whatever machine happens
    to run this script. SnitchGemmKnobFeatureExtractor uses a fixed L1_BYTES
    constant (the cluster's TCDM size, from CLAUDE.md) instead.
  - ScheduleEvaluator's approach (AOT-compile, then execute the native host
    binary NUM_TIMING_REPEATS=5 times, min wall-clock as the objective)
    doesn't apply: the compiled artifact is a RISC-V ELF that can't run on
    the host at all, and gvsoc's simulated cycle count is a deterministic
    function of the program, not subject to OS scheduling noise -- so
    SnitchScheduleEvaluator runs each candidate exactly once and reads its
    objective from the harness's own instrumented `total_cycles: N` print
    (see runtime/main32_multicore_tune.c), instead of timing anything on
    the host.

The correctness/safety constraints below (snitch_tile_constraint,
snitch_divisor_sample_value) are not arbitrary search-space narrowing --
see their docstrings. Getting them wrong doesn't just waste a candidate, it
can hang gvsoc forever (the exact barrier-deadlock class of bug CLAUDE.md's
Milestone 5/6 sections document at length), which is why
SnitchScheduleEvaluator also owns a hard timeout with process-group kill
independent of these constraints -- the constraints are the first line of
defense, the timeout is the one that makes it actually safe to run an
unattended search.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_POC_DIR = _HERE.parent
sys.path.insert(0, str(_POC_DIR))

from mlir_schedule import Knob, render_schedule, token_to_python  # noqa: E402
from mlir_tuner.backend import KnobFeatureExtractor, KnobSearchSpace, detect_dtype_bytes  # noqa: E402
from tuner_framework.interfaces import EvalResult, Evaluator  # noqa: E402

SnitchCandidate = Dict[str, str]

# The Snitch cluster's TCDM (L1 scratchpad) size -- see CLAUDE.md's gvsoc
# note ("128 KB TCDM... matches the real snitch_cluster hardware config").
# Fixed because the target is always this one simulated cluster, unlike
# GemmKnobFeatureExtractor's host-autodetected cache sizes.
L1_BYTES = 128 * 1024

_KNOB_DIM_NAME = {"tile_m": "M", "tile_n": "N", "tile_k": "K"}


class SnitchGemmKnobFeatureExtractor(KnobFeatureExtractor):
    """Same derived BLIS-style features as mlir_tuner.backend.GemmKnobFeatureExtractor
    (per-tile A/B/C byte footprints, arithmetic intensity, footprint-over-L1
    ratio), computed against the fixed L1_BYTES budget above instead of
    hw_detect-autodetected host cache sizes. No vector-width-alignment
    features here -- those were x86 SIMD-specific and have no Snitch
    equivalent in this schedule (there's no explicit vectorize step; the
    xDSL microkernel handles its own register blocking, see CLAUDE.md's
    xDSL quirks section)."""

    _REQUIRED_KNOBS = ("tile_m", "tile_n", "tile_k")

    def __init__(self, payload_text: str):
        self._elem_bytes = detect_dtype_bytes(payload_text, default="f64")

    def extract(self, candidate: SnitchCandidate) -> Dict[str, float]:
        features = super().extract(candidate)
        if not all(name in candidate for name in self._REQUIRED_KNOBS):
            return features

        elem_bytes = self._elem_bytes
        m = int(token_to_python(candidate["tile_m"]))
        n = int(token_to_python(candidate["tile_n"]))
        k = int(token_to_python(candidate["tile_k"]))

        tile_a_bytes = float(m * k * elem_bytes)
        tile_b_bytes = float(n * k * elem_bytes)
        tile_c_bytes = float(m * n * elem_bytes)
        total_tile_bytes = tile_a_bytes + tile_b_bytes + tile_c_bytes
        dual_buffered_bytes = 2.0 * total_tile_bytes

        flops = 2.0 * m * n * k
        arithmetic_intensity = flops / total_tile_bytes if total_tile_bytes > 0 else 0.0

        features.update({
            "tile_a_bytes": tile_a_bytes,
            "tile_b_bytes": tile_b_bytes,
            "tile_c_bytes": tile_c_bytes,
            "total_tile_bytes": total_tile_bytes,
            "tile_bytes_over_l1": total_tile_bytes / L1_BYTES,
            "dual_buffered_bytes_over_l1": dual_buffered_bytes / L1_BYTES,
            "arithmetic_intensity": arithmetic_intensity,
        })
        return features


def snitch_tile_constraint(dims: Dict[str, int]):
    """Build the constraint function KnobSearchSpace uses as the sole
    authority on candidate validity (see its docstring) for the Snitch M6
    schedule. Two requirements, both load-bearing, not stylistic.

    1. tile_m, tile_n, tile_k must each evenly divide dims['M']/['N']/['K'].
       A non-divisor tile size exercises PromotePadsToL1's padding path which is untested

    2. tile_m must additionally be a multiple of 8 since we have 8 cores
       and the schedule's outermost M-loop is parallelized across all of them.
       Else it will create uneven work that can result into a deadlock.
    """

    def constraint(values: Dict[str, object]) -> bool:
        m, n, k = int(values["tile_m"]), int(values["tile_n"]), int(values["tile_k"])
        if dims["M"] % m or dims["N"] % n or dims["K"] % k:
            return False
        return m % 8 == 0

    return constraint


def _divisors(n: int) -> List[int]:
    """All positive divisors of n, ascending -- mirrors
    ../mlir_tuner/run_gbdt_tuner.py's divisors() for the CPU backend."""
    return [d for d in range(1, n + 1) if n % d == 0]


def max_l1_tile_dim(elem_bytes: int) -> int:
    budget = L1_BYTES // 2  # /2 for headroom under the dual-buffered 2x
    # dual-buffered footprint ~= 2 * (t*t + t*t + t*t) * elem_bytes = 6*t^2*elem_bytes
    return max(1, int((budget / (3 * elem_bytes)) ** 0.5))


def _knob_option_pool(name: str, dims: Dict[str, int], elem_bytes: int) -> List[int]:
    """
    Candidate pool for the pipelined mode's knobs
    """
    dim = dims[_KNOB_DIM_NAME[name]]
    divs = _divisors(dim)
    max_tile = max_l1_tile_dim(elem_bytes)

    if name == "tile_m":
        pool = [d for d in divs if d % 8 == 0 and d <= max_tile]
        return pool or [d for d in divs if d % 8 == 0]

    min_tile = max(1, -(-dim // 8))  # ceil(dim / 8)
    pool = [d for d in divs if min_tile <= d <= max_tile]
    if pool:
        return pool
    pool = [d for d in divs if d <= max_tile]
    return pool or divs


def snitch_divisor_sample_value(name: str, rng, dims: Dict[str, int], elem_bytes: int) -> str:
    """ Provide a random (from a pool of good options)
    valid divisor of the corresponding dimension for the given knob name."""
    valid = _knob_option_pool(name, dims, elem_bytes)
    if not valid:
        raise ValueError(f"no valid options for knob {name!r} against dimension {dims[_KNOB_DIM_NAME[name]]}")
    return str(rng.choice(valid))


def snitch_search_space_size(dims: Dict[str, int], elem_bytes: int) -> int:
    """Total number of distinct valid (tile_m, tile_n, tile_k) combinations
    that exist for this payload's dims/element size. 
    """
    size = 1
    for name in _KNOB_DIM_NAME:
        size *= len(_knob_option_pool(name, dims, elem_bytes))
    return size


def snitch_padded_tile_constraint(dims: Dict[str, int]):
    """Constraint function for the non-pipelined, padded mode"""

    def constraint(values: Dict[str, object]) -> bool:
        m = int(values["tile_m"])
        return m % 8 == 0

    return constraint


def _padded_knob_option_pool(name: str, dims: Dict[str, int], elem_bytes: int) -> List[int]:
    """Candidate pool for the padded mode's knobs """
    dim = dims[_KNOB_DIM_NAME[name]]
    upper = max(1, min(dim, max_l1_tile_dim(elem_bytes)))

    if name == "tile_m":
        pool = list(range(8, upper + 1, 8))
        return pool or [8]

    lower = max(4, min(upper, -(-dim // 8)))  # same ceil(dim/8) floor heuristic as the divisor mode
    if lower > upper:
        lower = upper
    return list(range(lower, upper + 1))


def snitch_padded_tile_sample_value(name: str, rng, dims: Dict[str, int], elem_bytes: int) -> str:
    """Sample a tile size for the padded mode -- any integer in range, not
    restricted to divisors."""
    valid = _padded_knob_option_pool(name, dims, elem_bytes)
    if not valid:
        raise ValueError(f"no valid options for knob {name!r} against dimension {dims[_KNOB_DIM_NAME[name]]}")
    return str(rng.choice(valid))


def snitch_padded_search_space_size(dims: Dict[str, int], elem_bytes: int) -> int:
    """Total number of distinct valid (tile_m, tile_n, tile_k) combinations
    for the padded mode -- same purpose as snitch_search_space_size."""
    size = 1
    for name in _KNOB_DIM_NAME:
        size *= len(_padded_knob_option_pool(name, dims, elem_bytes))
    return size


_CORRECT_RE = re.compile(r"all (\d+) values matched")
_CYCLES_RE = re.compile(r"total_cycles:\s*(\d+)")


class SnitchScheduleEvaluator(Evaluator):
    """
    Evaluator for the Snitch schedule. It compiles the schedule with the given knobs, runs it in gvsoc,
    and extracts the cycle count from the output.
    """

    def __init__(
        self,
        schedule_text: str,
        run_script: Path,
        payload_path: Path,
        timeout_s: float = 300.0,
        verbose: bool = False,
        cleanup: bool = True,
    ):
        self.schedule_text = schedule_text
        self.run_script = run_script
        self.payload_path = Path(payload_path).resolve()
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.cleanup = cleanup

    def evaluate_batch(self, candidates: Sequence[SnitchCandidate]) -> List[EvalResult]:
        return [self._evaluate_one(c, i, len(candidates)) for i, c in enumerate(candidates, 1)]

    @staticmethod
    def _describe(candidate: SnitchCandidate) -> str:
        return " ".join(f"{name}={value}" for name, value in sorted(candidate.items()))

    def _evaluate_one(self, candidate: SnitchCandidate, index: int, total: int) -> EvalResult:
        if self.verbose:
            print(f"    [{index}/{total}] running: {self._describe(candidate)}", flush=True)

        rendered = render_schedule(self.schedule_text, candidate)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(rendered)
            schedule_path = tmp.name
        workdir = tempfile.mkdtemp(prefix="snitch_tune_")

        try:
            result = self._run_once(schedule_path, workdir)
        finally:
            try:
                Path(schedule_path).unlink()
            except FileNotFoundError:
                pass
            if self.cleanup:
                import shutil
                shutil.rmtree(workdir, ignore_errors=True)

        if self.verbose:
            if result.valid:
                print(f"    [{index}/{total}] done: objective={result.objective:.6g} cycles "
                      f"({self._describe(candidate)})", flush=True)
            else:
                print(f"    [{index}/{total}] done: invalid ({self._describe(candidate)})", flush=True)
        return result

    def _run_once(self, schedule_path: str, workdir: str) -> EvalResult:
        proc = subprocess.Popen(
            [str(self.run_script), schedule_path, str(self.payload_path), workdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if self.verbose:
                sys.stderr.write(
                    f"    candidate timed out after {self.timeout_s:.3g}s -- killed "
                    "(possible barrier deadlock; see snitch_tile_constraint's docstring)\n"
                )
            return EvalResult(valid=False, objective=None)

        if proc.returncode != 0:
            if self.verbose:
                sys.stderr.write(stdout)
            return EvalResult(valid=False, objective=None)

        correct = _CORRECT_RE.search(stdout) is not None
        cycles_match = _CYCLES_RE.search(stdout)
        if not correct or cycles_match is None:
            if self.verbose:
                sys.stderr.write(stdout)
            return EvalResult(valid=False, objective=None)

        return EvalResult(valid=True, objective=float(cycles_match.group(1)))
