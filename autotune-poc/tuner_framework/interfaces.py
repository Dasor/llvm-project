"""Hardware-agnostic interfaces for the GBDT-surrogate + genetic-algorithm
autotuning loop.

This module defines the seams that let one Tuner implementation (not
written yet -- see below) drive completely different backends: the
custom-tuner/ reference project (GEMM tile sizes, a hardware simulator,
XGBoost + DEAP) and this repo's MLIR CPU PoC (transform.tune.knob values,
mlir-opt/mlir-runner) alike.

A `Candidate` is deliberately opaque to the framework -- it's whatever a
concrete SearchSpace says it is (a `(m, n, k)` tuple in custom-tuner/, a
`dict[str, str]` of knob name -> chosen option in the MLIR driver). Nothing
in these interfaces inspects a candidate's shape; they only ever produce,
transform, or key on one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, List, Optional, Sequence, Tuple, TypeVar

Candidate = TypeVar("Candidate")


class SearchSpace(ABC, Generic[Candidate]):
    """What a valid candidate looks like, and how to move around in the space.

    Replaces, from custom-tuner/: generate_configs.py's
    _valid_m_values/_valid_n_values/_valid_k_values/_is_valid (referenced
    but not present in this dir), ga.py's _random_individual/_mutate/
    _crossover, and ga_repair.py's repair().

    Design note vs. the original ga.py: there, _mutate/_crossover each call
    repair() internally and silently no-op if repair fails (see ga.py's
    `if result is not None: ...` -- otherwise the individual is returned
    unchanged). Here that policy is factored out: mutate()/crossover() may
    return invalid candidates, and the Tuner is the one that calls repair()
    and falls back to the pre-mutation candidate on failure. Same observable
    behavior, but the "what happens when repair fails" policy lives in one
    place instead of being duplicated in every mutate/crossover
    implementation.
    """

    @abstractmethod
    def sample_random(self, rng) -> Optional[Candidate]:
        """Return a uniformly-random *valid* candidate, or None if none could
        be found within a reasonable number of attempts (mirrors ga.py's
        _random_individual, which gives up after 10_000 tries)."""

    @abstractmethod
    def mutate(self, candidate: Candidate, rng) -> Candidate:
        """Return a mutated candidate. Need not be valid -- the Tuner repairs
        it. (mirrors ga.py's _mutate, minus the inline repair() call)"""

    @abstractmethod
    def crossover(self, a: Candidate, b: Candidate, rng) -> Tuple[Candidate, Candidate]:
        """Combine two candidates into two children. Need not be valid.
        (mirrors ga.py's _crossover, minus the inline repair() call)"""

    @abstractmethod
    def repair(self, candidate: Candidate) -> Optional[Candidate]:
        """Snap a possibly-invalid candidate back into the valid region, or
        return None if it cannot be repaired at all. (mirrors
        ga_repair.py's repair(), generalized past (m, n, k) triples)"""

    @abstractmethod
    def key(self, candidate: Candidate) -> Hashable:
        """A hashable identity for caching. ga.py caches row_cache/
        score_cache on the raw (m, n, k) tuple, which is already hashable;
        for a dict-shaped candidate (e.g. the MLIR knob driver) this would
        be something like `tuple(sorted(candidate.items()))`."""


class FeatureExtractor(ABC, Generic[Candidate]):
    """Turns a candidate into the numeric feature row the GBDT trains/
    predicts on.

    Replaces, from custom-tuner/: ga.py's _build_row (which calls
    TSA_C_Remainder for static-analysis metrics) and boost.py's
    _feature_matrix/_encode_remainder_tiles/_DROP column bookkeeping.

    In the custom-tuner/ project, features came from an external static-
    analysis process (TSA_C_Remainder) fed cache sizes and the GEMM m/n/k;
    for the MLIR PoC, an equivalent extractor would emit things like the
    chosen tile sizes themselves, the matched op's iteration-space bounds,
    and target cache/register sizes -- whatever's cheap to compute without
    actually compiling+running.
    """

    @abstractmethod
    def extract(self, candidate: Candidate) -> Dict[str, float]:
        """One row of named numeric features for this candidate. The
        Tuner is responsible for aligning/reindexing columns across rows
        before handing them to XGBoost (replacing boost.py's
        `df.reindex(sorted(df.columns), axis=1)`), so extract() doesn't
        need to worry about column order or omitted columns being
        consistent across calls."""


@dataclass
class EvalResult:
    """One candidate's outcome from actually running it.

    `valid=False` covers everything from "failed to compile" to "simulator
    timed out" to (for the MLIR case) "produced incorrect output" --
    anything that should be excluded from GBDT training and reported as a
    failure, without necessarily being fatal to the whole batch. When
    `valid` is False, `objective` should be None.
    """

    valid: bool
    objective: Optional[float]  # lower is better (cycles, wall-clock seconds, ...)


class Evaluator(ABC, Generic[Candidate]):
    """Actually runs candidates and reports how good they are.

    Replaces, from custom-tuner/: run_loop.py's subprocess calls into
    myrtle-experiments/many_gemms.sh (compile, then simulate) and
    combineTilingSchemeDataIntoSingleCSV.py (merge results back out).
    For the MLIR PoC, a concrete implementation would wrap
    driver/run_one.sh: render each candidate's knobs into a schedule,
    run it, time it, and check correctness.

    Batched rather than one-at-a-time because the reference implementation
    is: run_loop.py hands many_gemms.sh a whole CSV of configs per
    iteration and compiles/simulates them together. An evaluator that's
    naturally one-at-a-time (like the MLIR case) just loops inside
    evaluate_batch(); the interface doesn't force per-candidate subprocess
    calls on backends that can do better.
    """

    @abstractmethod
    def evaluate_batch(self, candidates: Sequence[Candidate]) -> List[EvalResult]:
        """Evaluate `candidates` and return one EvalResult per candidate, in
        the same order -- including entries for candidates that failed or
        timed out, so the Tuner can still log/blocklist them."""
