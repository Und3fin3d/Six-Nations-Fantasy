"""C7: a row-varying blend weight driven by player career history depth.

Unlike a per-tournament or per-hemisphere weight -- which would break the
one-model mandate -- this reads only the player's own prior match count, so the
blend stays competition-independent. Hypothesis: the empirical prior should earn
weight where a player has little history for the GBDT to exploit.

Row-varying weights cannot use the scalar-weight loss table, so this evaluates
against the fold cache directly, with the same leave-one-fold-out cross-fitting
and temporal split as every other candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import allrugby as A
from .allrugby_cache import SCORED, build_cache, fold_stable_score
from .config import SEED

K_GRID = (3.0, 10.0, 30.0)
A_GRID = np.round(np.arange(0.0, 0.81, 0.1), 3)
B_GRID = np.round(np.arange(-0.4, 0.81, 0.1), 3)
FROZEN_BASELINE = 0.8864704364786675


@dataclass(frozen=True)
class HistoryRule:
    a: float
    b: float
    k: float

    def weights_for(self, cache) -> np.ndarray:
        career = np.asarray(cache.career, dtype=float)
        return np.clip(self.a + self.b * career / (career + self.k), 0.0, 1.0)


def fit(train) -> HistoryRule:
    best, best_score = None, np.inf
    for k in K_GRID:
        for a in A_GRID:
            for b in B_GRID:
                rule = HistoryRule(float(a), float(b), float(k))
                score = float(np.nanmean([
                    fold_stable_score(cache, {}, rule.weights_for(cache))
                    for cache in train
                ]))
                if score < best_score:
                    best, best_score = rule, score
    return best


def _scores(cache, rule, cohort="all") -> float:
    return fold_stable_score(cache, {}, rule.weights_for(cache), cohort=cohort)


def run() -> dict:
    caches = build_cache()
    base = np.array([fold_stable_score(cache, {}, 0.5) for cache in caches])
    cross, fits = [], []
    for index, cache in enumerate(caches):
        train = [other for position, other in enumerate(caches) if position != index]
        rule = fit(train)
        fits.append({"held_out_fold": cache.label,
                     "rule": {"a": rule.a, "b": rule.b, "k": rule.k}})
        cross.append(_scores(cache, rule))
        print(f"[{cache.label}] a={rule.a} b={rule.b} k={rule.k} -> {cross[-1]:.6f}", flush=True)
    cross = np.array(cross)
    differences = cross - base
    boot = A.bootstrap_difference(differences, seed=SEED)

    north_b = np.array([fold_stable_score(c, {}, 0.5, cohort="north") for c in caches])
    south_b = np.array([fold_stable_score(c, {}, 0.5, cohort="south") for c in caches])
    rules = [fit([o for p, o in enumerate(caches) if p != i]) for i in range(len(caches))]
    north_c = np.array([_scores(c, rules[i], "north") for i, c in enumerate(caches)])
    south_c = np.array([_scores(c, rules[i], "south") for i, c in enumerate(caches)])

    cut = len(caches) // 2
    temporal_rule = fit(caches[:cut])
    held = caches[cut:]
    t_cand = np.array([_scores(c, temporal_rule) for c in held])
    t_base = np.array([fold_stable_score(c, {}, 0.5) for c in held])

    frame = pd.DataFrame({
        "fold": [c.label for c in caches],
        "tournament": [c.tournament for c in caches],
        "stable_base": base, "stable_cand": cross,
    })
    tournaments = frame.groupby("tournament")[["stable_base", "stable_cand"]].mean()
    reasons = []
    score = float(np.mean(cross))
    if not score < FROZEN_BASELINE:
        reasons.append(f"(a) stable_score {score:.6f} did not improve on frozen {FROZEN_BASELINE:.6f}")
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(f"(b) paired-by-fold bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0")
    for name, b_, c_ in (("north", north_b, north_c), ("south", south_b, south_c)):
        if np.nanmean(c_) > np.nanmean(b_) * 1.02:
            reasons.append(f"(c) {name} cohort regressed by more than 2%")
    regressed = tournaments.index[
        tournaments["stable_cand"] > tournaments["stable_base"] * 1.02
    ].tolist()
    if regressed:
        reasons.append("(d) tournament-family stable regression >2%: " + ", ".join(regressed))

    record = {
        "name": "C7_history_depth_weight", "n_fitted_parameters": 3,
        "hypothesis": __doc__.strip().splitlines()[2],
        "fitting": "leave-one-fold-out cross-fitted (out-of-sample)",
        "stable_score": score, "baseline_stable_score": float(np.mean(base)),
        "delta": float(np.mean(differences)),
        "improvement_vs_frozen": 1.0 - score / FROZEN_BASELINE,
        "bootstrap_paired_by_fold": boot,
        "north": float(np.nanmean(north_c)), "south": float(np.nanmean(south_c)),
        "north_base": float(np.nanmean(north_b)), "south_base": float(np.nanmean(south_b)),
        "per_tournament": {
            k: {"baseline": float(r["stable_base"]), "candidate": float(r["stable_cand"])}
            for k, r in tournaments.iterrows()
        },
        "temporal_holdout": {
            "n_fit_folds": cut, "n_held_folds": len(held),
            "held_out_stable": float(np.mean(t_cand)),
            "held_out_baseline": float(np.mean(t_base)),
            "held_out_delta": float(np.mean(t_cand - t_base)),
            "held_out_bootstrap": A.bootstrap_difference(t_cand - t_base, seed=SEED),
            "fitted_rule": {"a": temporal_rule.a, "b": temporal_rule.b, "k": temporal_rule.k},
        },
        "in_sample_reference": float(np.mean([
            _scores(c, fit(caches)) for c in caches
        ])),
        "cross_fits": fits,
        "accepted": not reasons, "reasons": reasons,
    }
    (A.WORK / "c7_history.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    return record


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: result[k] for k in (
        "stable_score", "delta", "in_sample_reference", "accepted", "reasons",
    )}, indent=2, default=float))
