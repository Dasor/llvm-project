"""Shared utilities for the `transform.tune.knob` text protocol: parsing
knobs out of an MLIR schedule file, rendering a resolved schedule for a
chosen candidate, and ahead-of-time compiling + timing one resolved
schedule (compile_schedule.sh).

There is no MLIR-IR-level API for any of this (this build has
MLIR_ENABLE_BINDINGS_PYTHON=0, and even with bindings there is no built-in
"walk and resolve" helper for transform.tune.knob; see README.md) -- these
are plain regexes over the schedule text, matching exactly what
`mlir-opt --transform-interpreter` itself accepts.

Used by both driver/autotune.py (exhaustive grid search) and
mlir_tuner/backend.py (GBDT+GA search via tuner_framework.Tuner), so this
logic exists in exactly one place regardless of which search strategy is
driving it.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# Matches both the unresolved form:
#   transform.tune.knob<"name"> options = [...]
# and the already-resolved form:
#   transform.tune.knob<"name"> = <value> from options = [...]
# `selected` is None for the former, capturing the raw value text for the
# latter. Only the array-of-scalars flavor of `options` is supported (as
# opposed to e.g. an affine_set) -- that covers every knob this PoC deals
# with and is the common case for tuning discrete parameters like tile sizes.
KNOB_RE = re.compile(
    r'transform\.tune\.knob<"(?P<name>[^"]+)">\s*'
    r"(?:=\s*(?P<selected>.+?)\s+from\s+)?"
    r"options\s*=\s*(?P<options>\[[^\]]*\])"
)


class Knob:
    def __init__(self, name: str, options: List[str], already_resolved: bool):
        self.name = name
        self.options = options
        self.already_resolved = already_resolved


def extract_knobs(schedule_text: str) -> List[Knob]:
    knobs = []
    seen = set()
    for m in KNOB_RE.finditer(schedule_text):
        name = m.group("name")
        if name in seen:
            continue  # a knob's result may be referenced multiple times, but it's declared once
        seen.add(name)
        options_text = m.group("options")[1:-1]  # strip surrounding [ ]
        options = [tok.strip() for tok in options_text.split(",") if tok.strip()]
        knobs.append(Knob(name, options, m.group("selected") is not None))
    return knobs


def render_schedule(schedule_text: str, candidate: Dict[str, str]) -> str:
    """Rewrite every unresolved knob in `schedule_text` named in `candidate`
    to carry `= <chosen value> from options = [<chosen value>]`.
    Already-resolved knobs are left as-is.

    The emitted `options` is always a fresh singleton array holding just the
    chosen value -- NOT the original file's `options` list. This is
    deliberate, not cosmetic: KnobOp::verify() only checks that `selected`
    is a member of `options` (mlir/lib/Dialect/Transform/TuneExtension/
    TuneExtensionOps.cpp), and KnobOp::apply() never reads `options` at all
    -- so a singleton `[value]` is exactly as legal as the full original
    list, for any value. That's what decouples a search's chosen candidate
    from what happened to be enumerated in the source schedule: a
    SearchSpace can propose any value it wants, from any domain it wants,
    entirely independent of the file's `options = [...]` text, and it will
    still always render back into a valid, verifier-passing schedule --
    the file's declared options are only ever a hint for extract_knobs()
    callers that choose to use them (e.g. KnobSearchSpace's default), never
    a hard constraint on what render_schedule can produce.
    """

    def repl(m: re.Match) -> str:
        name = m.group("name")
        if m.group("selected") is not None:
            return m.group(0)
        if name not in candidate:
            raise KeyError(f"no chosen value for knob {name!r}")
        value = candidate[name]
        return f'transform.tune.knob<"{name}"> = {value} from options = [{value}]'

    return KNOB_RE.sub(repl, schedule_text)


_MATMUL_DIMS_RE = re.compile(
    r"linalg\.matmul\s+ins\(\s*%\S+\s*,\s*%\S+\s*:\s*"
    r"tensor<(?P<m>\d+)x(?P<k1>\d+)x\w+>\s*,\s*tensor<(?P<k2>\d+)x(?P<n>\d+)x\w+>\)"
)


def detect_matmul_dims(schedule_text: str) -> Dict[str, int]:
    """Infer the GEMM M/N/K dimensions from the first `linalg.matmul
    ins(%a, %b : tensor<MxKxDTYPE>, tensor<KxNxDTYPE>)` op in `schedule_text`
    -- the payload's own shapes, not a value the caller has to supply and
    keep in sync by hand. Raises ValueError if no such op is found, or if
    the two operands' inner (K) dimensions disagree."""
    m = _MATMUL_DIMS_RE.search(schedule_text)
    if m is None:
        raise ValueError("no `linalg.matmul ins(...)` found in schedule text to infer M/N/K from")
    dims = {"M": int(m.group("m")), "N": int(m.group("n")), "K": int(m.group("k1"))}
    k2 = int(m.group("k2"))
    if k2 != dims["K"]:
        raise ValueError(f"inconsistent K: ins operand shapes disagree ({dims['K']} vs {k2})")
    return dims


def token_to_python(token: str):
    """Best-effort conversion of a raw option token (e.g. '32', '2.5 : f32',
    'true', '"dog"') into a Python value, for evaluating a --constraint
    expression or building numeric GBDT features."""
    stripped = re.sub(r"\s*:\s*\w+\s*$", "", token.strip())  # drop ': f32' etc.
    if stripped in ("true", "false"):
        return stripped == "true"
    for conv in (int, float):
        try:
            return conv(stripped)
        except ValueError:
            pass
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
    return stripped


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def check_uniform_value(stdout: str, expect_value: str) -> bool:
    data = stdout[stdout.index("data =") :] if "data =" in stdout else stdout
    values = NUMBER_RE.findall(data)
    return bool(values) and all(v == expect_value for v in values)


class CandidateResult(NamedTuple):
    kernel_time_s: Optional[float]  # min wall time across NUM_TIMING_REPEATS runs of
                                     # the already-compiled executable; None if
                                     # compilation or every run failed
    correct: bool
    stderr: str
    compile_time_s: Optional[float] = None  # wall time spent in compile_candidate(),
                                              # regardless of whether compilation
                                              # succeeded, failed, or timed out


NUM_TIMING_REPEATS = 5


def compile_candidate(
    schedule_text: str,
    candidate: Dict[str, str],
    compile_script: Path,
    timeout_s: Optional[float] = None,
) -> Tuple[Optional[Path], str, float]:
    """Render `candidate` into `schedule_text` and ahead-of-time compile it
    to a standalone native executable via `compile_script`
    (compile_schedule.sh) -- deliberately NOT part of run_candidate()'s
    execution-timing interval below, so mlir-opt's lowering passes + LLVM
    codegen never leak into kernel_time_s.

    If `timeout_s` is given, compilation is killed the moment it exceeds
    that many seconds -- see ScheduleEvaluator (mlir_tuner/backend.py) for
    how the threshold is derived from the best candidate compile time seen
    so far. A candidate that's slow to *compile* (e.g. tile sizes that
    blow up unrolling/vectorization) is often also slow to *run*, so this
    is worth bounding even independently of the execution timeout.

    Returns (path_to_executable, stderr, elapsed_s). Path is None if
    compilation failed or timed out, in which case the caller should
    surface stderr and treat the candidate as invalid. Caller owns
    deleting the returned executable."""
    rendered = render_schedule(schedule_text, candidate)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
        tmp.write(rendered)
        schedule_path = tmp.name

    exe_fd, exe_path = tempfile.mkstemp(suffix=".exe")
    os.close(exe_fd)

    start = time.perf_counter()
    try:
        try:
            proc = subprocess.run(
                [str(compile_script), schedule_path, exe_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            try:
                Path(exe_path).unlink()
            except FileNotFoundError:
                pass
            return None, f"compile timed out (exceeded {timeout_s:.3g}s threshold)", elapsed
    finally:
        try:
            Path(schedule_path).unlink()
        except FileNotFoundError:
            pass
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        try:
            Path(exe_path).unlink()
        except FileNotFoundError:
            pass
        return None, proc.stderr, elapsed
    return Path(exe_path), proc.stderr, elapsed


def run_candidate(
    schedule_text: str,
    candidate: Dict[str, str],
    compile_script: Path,
    expect_uniform_value: Optional[str],
    run_timeout_s: Optional[float] = None,
    compile_timeout_s: Optional[float] = None,
) -> CandidateResult:
    """Ahead-of-time compile `candidate` (via compile_candidate()), then run
    the resulting native executable NUM_TIMING_REPEATS times, timing each
    tightly with time.perf_counter() and taking the minimum as
    kernel_time_s -- interference can only slow a run down, never speed one
    up, so the min across a handful of repeats is a good estimate of actual
    execution time. Compilation (mlir-opt's lowering passes + LLVM codegen)
    is never part of this measured interval; its own wall time is reported
    separately as `compile_time_s`. See README.md's timing note.

    If `run_timeout_s` is given, each repeat run is killed the moment it
    exceeds that many seconds, and the candidate is immediately treated as
    invalid without waiting for the remaining repeats. `compile_timeout_s`
    does the same for the compile step itself (see compile_candidate()).
    See ScheduleEvaluator (mlir_tuner/backend.py) for how both thresholds
    are derived from the best candidate seen so far.

    The compiled executable's exit code is NOT used as a correctness
    signal: its `@main` takes no arguments and returns void, an ABI
    mismatch with the standard C entry point convention that leaves the
    real exit code effectively undefined (harmless at the call-ABI level,
    but not a reliable 0/nonzero signal). Only a negative return code
    (subprocess.run reports these for death-by-signal, e.g. an actual
    crash) is treated as a hard failure; otherwise correctness is judged
    purely from stdout via check_uniform_value(), same as before.
    """
    exe_path, stderr, compile_time_s = compile_candidate(
        schedule_text, candidate, compile_script, timeout_s=compile_timeout_s
    )
    if exe_path is None:
        return CandidateResult(None, False, stderr, compile_time_s)

    try:
        best_elapsed: Optional[float] = None
        last_stdout = ""
        last_stderr = stderr
        for _ in range(NUM_TIMING_REPEATS):
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    [str(exe_path)], capture_output=True, text=True, timeout=run_timeout_s
                )
            except subprocess.TimeoutExpired:
                return CandidateResult(
                    None, False,
                    f"timed out (exceeded {run_timeout_s:.3g}s threshold)",
                    compile_time_s,
                )
            elapsed = time.perf_counter() - start

            if proc.returncode < 0:
                return CandidateResult(None, False, proc.stderr, compile_time_s)

            last_stdout, last_stderr = proc.stdout, proc.stderr
            if best_elapsed is None or elapsed < best_elapsed:
                best_elapsed = elapsed
    finally:
        try:
            exe_path.unlink()
        except FileNotFoundError:
            pass

    if expect_uniform_value is not None:
        correct = check_uniform_value(last_stdout, expect_uniform_value)
    else:
        correct = True  # exit code isn't a reliable signal here -- see docstring
    return CandidateResult(best_elapsed, correct, last_stderr, compile_time_s)
