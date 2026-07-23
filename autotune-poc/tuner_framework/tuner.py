"""The reusable GBDT-surrogate + genetic-algorithm search loop.

Mirrors run_loop.py's structure from custom-tuner/ (iteration 1: random
sample -> evaluate for real -> train; iteration 2+: GA search guided by the
surrogate model -> evaluate the GA's top-N for real -> retrain on the full
accumulated dataset), but driven entirely through the three interfaces in
interfaces.py instead of GEMM/simulator specifics.

Two deliberate departures from custom-tuner/'s ga.py, both there to make
this reusable across processes/backends without extra dependencies or
hidden global state:

  - No DEAP. ga.py uses `deap.creator.create(...)` to define its Individual/
    Fitness classes at *module import time* -- global mutable state that
    doesn't play well with a library meant to be instantiated more than
    once (e.g. tuning several different ops in one process). The GA here
    is ~40 lines of tournament-selection/crossover/mutation using plain
    Python, built directly on SearchSpace's operators.
  - No big "pool then take head(N)" step for the initial random sample
    (that indirection lived inside custom-tuner/'s generate_configs.py,
    which isn't part of this framework). Iteration 1 just calls
    SearchSpace.sample_random() until it has enough unique valid
    candidates.

Orchestration concerns from run_loop.py that stay OUT of this class on
purpose -- they're per-project, not per-hardware, and belong in a thin
driver script like run_loop.py itself: CLI parsing, --workdir/--resume-from
checkpointing, and blocklist persistence to disk. `Tuner.history` and
`Tuner.to_records()` expose everything a driver script needs to add that
back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .interfaces import Candidate, EvalResult, Evaluator, FeatureExtractor, SearchSpace

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
}


@dataclass
class TunerConfig:
    iterations: int = 15
    init_samples: int = 32            # candidates evaluated for real in iteration 1
    ga_candidates_per_iter: int = 32  # top-N GA proposes for real evaluation, iterations 2+
    ga_population: int = 200
    ga_generations: int = 100
    cxpb: float = 0.5                 # crossover probability per pair
    mutpb: float = 0.3                # mutation probability per individual
    tournament_size: int = 3
    min_training_points: int = 2      # don't train a model on fewer valid results than this
    seed: Optional[int] = None
    xgb_params: dict = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    xgb_boost_rounds: int = 400
    verbose: bool = True


@dataclass
class EvalRecord(Generic[Candidate]):
    candidate: Candidate
    features: Dict[str, float]
    result: EvalResult


class Tuner(Generic[Candidate]):
    def __init__(
        self,
        search_space: SearchSpace[Candidate],
        features: FeatureExtractor[Candidate],
        evaluator: Evaluator[Candidate],
        config: Optional[TunerConfig] = None,
    ):
        self.search_space = search_space
        self.features = features
        self.evaluator = evaluator
        self.config = config or TunerConfig()

        self.rng = random.Random(self.config.seed)
        self.history: List[EvalRecord[Candidate]] = []
        self._tested_keys: "set[Hashable]" = set()
        self._model: Optional[xgb.Booster] = None
        self._feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Candidate:
        """Run all configured iterations; return the best candidate found."""
        for it in range(1, self.config.iterations + 1):
            self._log(f"=== iteration {it}/{self.config.iterations} ===")
            if it == 1 or self._model is None:
                candidates = self._propose_initial()
            else:
                candidates = self._propose_via_ga()
            self._evaluate_and_record(candidates)
            self._retrain_model()
        return self.best_candidate()

    def best_candidate(self) -> Candidate:
        valid = [r for r in self.history if r.result.valid]
        if not valid:
            raise RuntimeError("no candidate was ever evaluated successfully")
        best = min(valid, key=lambda r: r.result.objective)
        return best.candidate

    def to_records(self) -> List[dict]:
        """Flatten `history` into plain dicts, ready for
        `pd.DataFrame(tuner.to_records()).to_csv(...)`."""
        return [
            {
                **r.features,
                "objective": r.result.objective,
                "valid": r.result.valid,
                "candidate": repr(r.candidate),
            }
            for r in self.history
        ]

    # ------------------------------------------------------------------
    # Iteration 1: real evaluation of random candidates
    # ------------------------------------------------------------------

    def _propose_initial(self) -> List[Candidate]:
        out: List[Candidate] = []
        seen = set()
        max_attempts = self.config.init_samples * 50
        for _ in range(max_attempts):
            if len(out) >= self.config.init_samples:
                break
            c = self.search_space.sample_random(self.rng)
            if c is None:
                continue
            k = self.search_space.key(c)
            if k in seen or k in self._tested_keys:
                continue
            seen.add(k)
            out.append(c)
        if len(out) < self.config.init_samples:
            self._log(
                f"warning: only found {len(out)}/{self.config.init_samples} "
                "unique random valid candidates; search space may be small"
            )
        return out

    # ------------------------------------------------------------------
    # Iteration 2+: GA search over the surrogate model
    # ------------------------------------------------------------------

    def _propose_via_ga(self) -> List[Candidate]:
        cfg = self.config
        candidate_cache: Dict[Hashable, Candidate] = {}
        score_cache: Dict[Hashable, float] = {}

        def predicted_score(candidates: Sequence[Candidate]) -> None:
            to_score = []
            for c in candidates:
                k = self.search_space.key(c)
                candidate_cache[k] = c
                if k not in score_cache:
                    to_score.append(c)
            if not to_score:
                return
            rows = [self.features.extract(c) for c in to_score]
            for c, p in zip(to_score, self._predict(rows)):
                score_cache[self.search_space.key(c)] = p

        # Seed the population from every candidate evaluated so far
        # (mirrors ga.py's --seed-csv, built from all prior iterations'
        # attempted configs), topped up with fresh random valid candidates.
        pop: List[Candidate] = []
        seen: "set[Hashable]" = set()
        for r in self.history:
            k = self.search_space.key(r.candidate)
            if k not in seen:
                seen.add(k)
                pop.append(r.candidate)
        while len(pop) < cfg.ga_population:
            c = self.search_space.sample_random(self.rng)
            if c is None:
                continue
            k = self.search_space.key(c)
            if k in seen:
                continue
            seen.add(k)
            pop.append(c)
        pop = pop[: cfg.ga_population]

        predicted_score(pop)

        for gen in range(1, cfg.ga_generations + 1):
            offspring = [self._tournament_select(pop, score_cache) for _ in pop]

            for i in range(1, len(offspring), 2):
                if self.rng.random() < cfg.cxpb:
                    c1, c2 = self.search_space.crossover(
                        offspring[i - 1], offspring[i], self.rng
                    )
                    offspring[i - 1] = self._repair_or_keep(offspring[i - 1], c1)
                    offspring[i] = self._repair_or_keep(offspring[i], c2)

            for i, c in enumerate(offspring):
                if self.rng.random() < cfg.mutpb:
                    mutated = self.search_space.mutate(c, self.rng)
                    offspring[i] = self._repair_or_keep(c, mutated)

            predicted_score(offspring)
            pop = offspring

            if cfg.verbose and (gen % 10 == 0 or gen == cfg.ga_generations):
                gen_scores = [score_cache[self.search_space.key(c)] for c in pop]
                self._log(
                    f"  GA gen {gen:>3}/{cfg.ga_generations}: "
                    f"best={max(gen_scores):.4f}  avg={sum(gen_scores)/len(gen_scores):.4f}  "
                    f"evaluated={len(score_cache)}"
                )

        # Top-N unique candidates across every generation's evaluations
        # (not just the final population), excluding anything already
        # sent to the real Evaluator in a previous iteration.
        ranked = sorted(score_cache.items(), key=lambda kv: -kv[1])
        chosen: List[Candidate] = []
        for k, _ in ranked:
            if k in self._tested_keys:
                continue
            chosen.append(candidate_cache[k])
            if len(chosen) >= cfg.ga_candidates_per_iter:
                break
        return chosen

    def _tournament_select(
        self, pop: Sequence[Candidate], score_cache: Dict[Hashable, float]
    ) -> Candidate:
        competitors = self.rng.sample(list(pop), min(self.config.tournament_size, len(pop)))
        return max(competitors, key=lambda c: score_cache[self.search_space.key(c)])

    def _repair_or_keep(self, original: Candidate, mutated: Candidate) -> Candidate:
        repaired = self.search_space.repair(mutated)
        return repaired if repaired is not None else original

    # ------------------------------------------------------------------
    # Real evaluation + surrogate model training
    # ------------------------------------------------------------------

    def _evaluate_and_record(self, candidates: Sequence[Candidate]) -> None:
        unique: List[Candidate] = []
        for c in candidates:
            k = self.search_space.key(c)
            if k in self._tested_keys:
                continue
            self._tested_keys.add(k)
            unique.append(c)
        if not unique:
            self._log("  (no new candidates to evaluate this iteration)")
            return

        results = self.evaluator.evaluate_batch(unique)
        n_valid = sum(r.valid for r in results)
        self._log(f"  evaluated {len(unique)} candidate(s), {n_valid} valid")

        for c, r in zip(unique, results):
            self.history.append(EvalRecord(c, self.features.extract(c), r))

    def _retrain_model(self) -> None:
        valid_records = [r for r in self.history if r.result.valid]
        if len(valid_records) < self.config.min_training_points:
            self._log(
                f"  skipping training: only {len(valid_records)} valid result(s) so far "
                f"(need {self.config.min_training_points})"
            )
            return

        rows = [r.features for r in valid_records]
        objectives = np.array([r.result.objective for r in valid_records], dtype=float)

        self._feature_columns = sorted({col for row in rows for col in row})
        X = pd.DataFrame(rows).reindex(columns=self._feature_columns, fill_value=0.0)

        # Normalized "goodness": best-so-far objective -> 1.0, worse -> smaller.
        # Used as both the regression target AND the sample weight, exactly
        # mirroring boost.py's train_model (so the model focuses on fast
        # configs, matching the Ansor weighting strategy the reference
        # project follows).
        goodness = objectives.min() / objectives

        dtrain = xgb.DMatrix(
            X, label=goodness, weight=goodness, feature_names=self._feature_columns
        )
        self._model = xgb.train(
            {**DEFAULT_XGB_PARAMS, **self.config.xgb_params},
            dtrain,
            num_boost_round=self.config.xgb_boost_rounds,
            verbose_eval=False,
        )
        self._log(f"  retrained model on {len(valid_records)} valid result(s)")

    def _predict(self, rows: List[Dict[str, float]]) -> List[float]:
        assert self._model is not None
        X = pd.DataFrame(rows).reindex(columns=self._feature_columns, fill_value=0.0)
        dmat = xgb.DMatrix(X, feature_names=self._feature_columns)
        return self._model.predict(dmat).tolist()

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)
