"""Driver for the all-rugby greedy hill-climb on the P3 unified event model.

Every trial is cross-fitted leave-one-fold-out: the weight applied to a fold is
never fitted on that fold. Each trial is additionally checked on a strict
temporal split (earliest folds fit, latest folds held out). Results land in
data/unified/raw_benchmark/allrugby_p3/; the frozen v1 ledger is read-only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import allrugby as A
from . import allrugby_search as S
from .allrugby_cache import (
    FoldCache, SCORED, build_cache, fold_stable_score, relative_loss,
)
from .config import SEED
from ..schema import distribution_family

WORK = A.WORK
GRID = S.GRID


# --------------------------------------------------------------------------
# extra fitters
# --------------------------------------------------------------------------

def target_density(caches: list[FoldCache]) -> dict[str, float]:
    density = {}
    for target in SCORED:
        rates = [
            float(np.mean(cache.actual[target][cache.valid[target]] > 0))
            for cache in caches if cache.valid[target].any()
        ]
        density[target] = float(np.mean(rates)) if rates else 0.0
    return density


def fit_density_rule(train: list[FoldCache]) -> dict[str, float]:
    """Two-parameter rule: w(target) = clip(a + b * positive_rate, 0, 1).

    Encodes the structural hypothesis directly -- the GBDT earns weight on
    dense volume events, the empirical prior on sparse ones -- with two fitted
    numbers instead of one per target.
    """
    density = target_density(train)
    best, best_score = None, np.inf
    for a in np.round(np.arange(0.0, 0.81, 0.05), 3):
        for b in np.round(np.arange(-0.4, 0.81, 0.05), 3):
            weights = {
                target: float(np.clip(a + b * density[target], 0.0, 1.0))
                for target in SCORED
            }
            score = float(np.nanmean([
                fold_stable_score(cache, weights, 0.5) for cache in train
            ]))
            if score < best_score:
                best, best_score = weights, score
    return best


def fit_shrunk_per_target(train: list[FoldCache]) -> dict[str, float]:
    """Per-target weights shrunk toward the global weight.

    The shrinkage strength is itself chosen by an inner leave-one-fold-out
    pass over the training folds only, so no held-out fold informs either the
    weights or the shrinkage.
    """
    lambdas = np.round(np.arange(0.0, 1.001, 0.1), 3)
    inner = []
    for value in lambdas:
        scores = []
        for index, cache in enumerate(train):
            inner_train = [
                other for position, other in enumerate(train) if position != index
            ]
            weights = S.fit_per_target_weights(inner_train, shrink=float(value))
            scores.append(fold_stable_score(cache, weights, 0.5))
        inner.append(float(np.nanmean(scores)))
    chosen = float(lambdas[int(np.argmin(inner))])
    return S.fit_per_target_weights(train, shrink=chosen)


LOG_FAMILIES = {
    "lognormal_only": frozenset({"metres"}),
    "lognormal_and_minutes": frozenset({"metres", "minutes"}),
    "counts": frozenset(
        target for target in SCORED
        if target not in {"minutes"} and distribution_family(target) == "negative_binomial"
    ),
    "all": frozenset(SCORED),
}


# --------------------------------------------------------------------------
# trials
# --------------------------------------------------------------------------

def fixed_rule_scores(
    caches: list[FoldCache], weights: dict[str, float], default: float,
    log_targets: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": cache.label, "tournament": cache.tournament,
        "hemisphere": cache.hemisphere,
        "stable": fold_stable_score(
            cache, weights, default, log_targets=log_targets,
        ),
        "north": fold_stable_score(
            cache, weights, default, cohort="north", log_targets=log_targets,
        ),
        "south": fold_stable_score(
            cache, weights, default, cohort="south", log_targets=log_targets,
        ),
    } for cache in caches])


def temporal_check(caches: list[FoldCache], fitter) -> dict:
    """Fit on the earliest half of folds, evaluate on the latest half."""
    train, held = S.temporal_split(caches)
    fitted = fitter(train)
    weights, default = (fitted, 0.5) if isinstance(fitted, dict) else ({}, float(fitted))
    candidate = fixed_rule_scores(held, weights, default)
    baseline = fixed_rule_scores(held, {}, 0.5)
    differences = candidate["stable"].to_numpy(float) - baseline["stable"].to_numpy(float)
    return {
        "n_fit_folds": len(train), "n_held_folds": len(held),
        "held_out_stable": float(candidate["stable"].mean()),
        "held_out_baseline": float(baseline["stable"].mean()),
        "held_out_delta": float(np.mean(differences)),
        "held_out_bootstrap": A.bootstrap_difference(differences, seed=SEED),
        "fitted_weights": weights or {"__global__": default},
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

CONTROL_TOLERANCE = 1e-9


def run_control(caches: list[FoldCache]) -> dict:
    """Reproduce the frozen p3_event_50 stable score before trusting anything."""
    per_fold = S.baseline_frame(caches)
    score = float(per_fold["stable"].mean())
    delta = abs(score - S.FROZEN_BASELINE)
    ledger = pd.read_csv(A.OUT / "event_metrics.csv")
    ledger = ledger[
        ledger["engine"].eq("p3_event_50") & ledger["cohort"].eq("all")
        & ledger["tier"].isin(["stable", "minutes"])
    ]
    ledger_per_fold = ledger.groupby("fold")["relative_loss"].mean()
    merged = per_fold.set_index("fold")["stable"]
    worst = float((merged - ledger_per_fold.reindex(merged.index)).abs().max())
    return {
        "reconstructed_stable_score": score,
        "frozen_stable_score": S.FROZEN_BASELINE,
        "abs_difference": delta,
        "max_abs_per_fold_difference_vs_ledger": worst,
        "n_folds": len(per_fold),
        "reproduced": bool(delta < CONTROL_TOLERANCE and worst < CONTROL_TOLERANCE),
    }


def diagnostics(caches: list[FoldCache]) -> dict[str, pd.DataFrame]:
    per_target = S.per_target_table(caches)
    per_fold = S.baseline_frame(caches)
    density = target_density(caches)
    per_target["density"] = per_target["target"].map(density)
    global_curve = pd.DataFrame({
        "weight_v4": GRID,
        "stable_score": [
            float(np.nanmean([fold_stable_score(cache, {}, float(w)) for cache in caches]))
            for w in GRID
        ],
    })
    tournament = per_fold.groupby("tournament", as_index=False)["stable"].mean()
    hemisphere = pd.DataFrame([{
        "cohort": name,
        "stable": float(np.nanmean([
            fold_stable_score(cache, {}, 0.5, cohort=name) for cache in caches
        ])),
    } for name in ("all", "north", "south", "starter", "bench", "low_history")])
    return {
        "per_target": per_target, "per_fold": per_fold,
        "global_weight_curve": global_curve, "per_tournament": tournament,
        "per_cohort": hemisphere,
    }


def trial(
    caches: list[FoldCache], baseline: pd.DataFrame, name: str, hypothesis: str,
    *, fitter=None, weights=None, default=0.5,
    log_targets: frozenset[str] = frozenset(),
) -> dict:
    """Score one candidate under the precommitted acceptance rule."""
    if fitter is not None:
        scores, fits = S.cross_fitted_scores(caches, fitter)
        fitted_note = "leave-one-fold-out cross-fitted"
        extra = {"temporal_holdout": temporal_check(caches, fitter),
                 "cross_fits": fits}
    else:
        scores = fixed_rule_scores(caches, weights or {}, default, log_targets)
        fitted_note = "no fitted parameter (fixed rule)"
        extra = {"weights": weights or {"__global__": default},
                 "log_targets": sorted(log_targets)}
    record = S.evaluate_candidate(caches, scores, baseline, name=name)
    record.update({
        "hypothesis": hypothesis, "fitting": fitted_note,
        "per_fold": scores.to_dict(orient="records"), **extra,
    })
    return record


QUEUE = [
    dict(
        name="C1_global_weight",
        hypothesis=(
            "The global 0.5 was chosen on 6N-2025 LORO, never on this target. "
            "Refitting one global v4 weight on the historical folds should move it "
            "toward the all-rugby optimum."
        ),
        fitter=S.fit_global_weight,
    ),
    dict(
        name="C2_per_target_weight",
        hypothesis=(
            "Empirical priors win on sparse counts and the GBDT wins on dense "
            "volume events, so one weight per target should beat a single global "
            "weight. 24 free parameters -- the highest overfitting risk in the queue."
        ),
        fitter=S.fit_per_target_weights,
    ),
    dict(
        name="C3_per_loss_family_weight",
        hypothesis=(
            "The metric mixes Poisson deviance, log-squared, log-loss and MAE. "
            "One weight per loss family (3 parameters) captures most of the "
            "per-target structure at a fraction of the variance."
        ),
        fitter=S.fit_family_weights,
    ),
    dict(
        name="C4_density_parametric_weight",
        hypothesis=(
            "Encode the density hypothesis directly: w(target) = a + b * "
            "positive_rate, two fitted parameters, monotone in event density."
        ),
        fitter=fit_density_rule,
    ),
    dict(
        name="C5_shrunk_per_target_weight",
        hypothesis=(
            "Per-target weights shrunk toward the fitted global weight, with the "
            "shrinkage strength chosen by an inner leave-one-fold-out pass over "
            "the training folds only. Keeps C2's signal, discards its noise."
        ),
        fitter=fit_shrunk_per_target,
    ),
]

FIXED_QUEUE = [
    dict(
        name=f"C6_log_space_{key}",
        hypothesis=(
            "The blend is on event MEANS, but the metric is Poisson deviance / "
            "log-squared, whose natural link is log. Blending on the log1p scale "
            f"for the '{key}' target set adds no fitted parameter."
        ),
        weights={}, default=0.5, log_targets=targets,
    )
    for key, targets in LOG_FAMILIES.items()
]


def main() -> None:
    caches = build_cache()
    print(f"loaded {len(caches)} fold caches", flush=True)

    control = run_control(caches)
    print(json.dumps(control, indent=2), flush=True)
    if not control["reproduced"]:
        raise SystemExit(
            "CONTROL FAILED: could not reproduce the frozen p3_event_50 stable "
            f"score ({control['reconstructed_stable_score']:.10f} vs "
            f"{control['frozen_stable_score']:.10f}). Stopping."
        )

    diag = diagnostics(caches)
    for name, frame in diag.items():
        frame.to_csv(WORK / f"diagnostic_{name}.csv", index=False)

    baseline = S.baseline_frame(caches)
    trials = []
    for spec in QUEUE:
        print(f"--- {spec['name']} ---", flush=True)
        trials.append(trial(caches, baseline, spec["name"], spec["hypothesis"],
                            fitter=spec["fitter"]))
        print(json.dumps({k: trials[-1][k] for k in
                          ("stable_score", "delta", "accepted", "reasons")},
                         indent=2), flush=True)
    for spec in FIXED_QUEUE:
        print(f"--- {spec['name']} ---", flush=True)
        trials.append(trial(caches, baseline, spec["name"], spec["hypothesis"],
                            weights=spec["weights"], default=spec["default"],
                            log_targets=spec["log_targets"]))
        print(json.dumps({k: trials[-1][k] for k in
                          ("stable_score", "delta", "accepted", "reasons")},
                         indent=2), flush=True)

    payload = {
        "schema_version": 1, "seed": SEED,
        "frozen_baseline": S.FROZEN_BASELINE, "control": control,
        "n_folds": len(caches),
        "acceptance_rule": {
            "a": "competition-balanced stable_score improves on 0.886470",
            "b": "paired-by-fold bootstrap difference excludes 0",
            "c": "neither north nor south cohort regresses by >2%",
            "d": "no tournament-family stable regression >2%",
            "e": ("no extended-event loss regression >5% -- satisfied by "
                  "construction: every candidate leaves extended events at the "
                  "frozen 0.5 weight"),
            "f": ("any newly fitted parameter is cross-fitted leave-one-fold-out "
                  "and additionally checked on a strict temporal split"),
        },
        "trials": trials,
    }
    (WORK / "ledger.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {WORK / 'ledger.json'}", flush=True)


if __name__ == "__main__":
    main()
