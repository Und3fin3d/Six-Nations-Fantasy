"""Driver for the all-rugby greedy hill-climb on the P3 unified event model.

Every fitted candidate is cross-fitted leave-one-fold-out -- the weights applied
to a fold are never fitted on that fold -- and additionally checked on a strict
temporal split (earliest folds fit, latest folds held out). Results land in
data/unified/raw_benchmark/allrugby_p3/; the frozen v1 ledger is read-only.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd

from . import allrugby as A
from . import allrugby_fit as F
from .allrugby_cache import SCORED, build_cache
from .allrugby_table import GRID, LossTable, build_table
from .config import SEED

WORK = A.WORK
FROZEN_BASELINE = 0.8864704364786675
CONTROL_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# control and diagnostics
# --------------------------------------------------------------------------

def baseline_indices(table: LossTable) -> np.ndarray:
    return np.full(len(table.targets), table.index_of(0.5), dtype=int)


def run_control(table: LossTable) -> dict:
    per_fold = table.stable("all", baseline_indices(table))
    score = float(np.mean(per_fold))
    ledger = pd.read_csv(A.OUT / "event_metrics.csv")
    ledger = ledger[
        ledger["engine"].eq("p3_event_50") & ledger["cohort"].eq("all")
        & ledger["tier"].isin(["stable", "minutes"])
    ].groupby("fold")["relative_loss"].mean()
    worst = float(max(
        abs(per_fold[index] - ledger[label])
        for index, label in enumerate(table.folds)
    ))
    delta = abs(score - FROZEN_BASELINE)
    return {
        "reconstructed_stable_score": score, "frozen_stable_score": FROZEN_BASELINE,
        "abs_difference": delta, "max_abs_per_fold_difference_vs_ledger": worst,
        "n_folds": len(table.folds),
        "reproduced": bool(delta < CONTROL_TOLERANCE and worst < 1e-12),
    }


def density_vector(caches) -> np.ndarray:
    values = []
    for target in SCORED:
        rates = [
            float(np.mean(cache.actual[target][cache.valid[target]] > 0))
            for cache in caches if cache.valid[target].any()
        ]
        values.append(float(np.mean(rates)) if rates else 0.0)
    return np.asarray(values)


# --------------------------------------------------------------------------
# evaluation against the precommitted acceptance rule
# --------------------------------------------------------------------------

def score_frame(table: LossTable, per_fold_indices, *, log: bool = False) -> pd.DataFrame:
    return pd.DataFrame({
        "fold": table.folds, "tournament": table.tournaments,
        "hemisphere": table.hemispheres,
        "stable": F.all_fold_scores(table, per_fold_indices, log=log),
        "north": F.all_fold_scores(table, per_fold_indices, cohort="north", log=log),
        "south": F.all_fold_scores(table, per_fold_indices, cohort="south", log=log),
    })


def evaluate(table: LossTable, candidate: pd.DataFrame, baseline: pd.DataFrame, name: str) -> dict:
    merged = baseline.merge(
        candidate, on=["fold", "tournament", "hemisphere"], suffixes=("_base", "_cand"),
    )
    if len(merged) != len(baseline):
        raise ValueError("candidate/baseline fold cohorts differ")
    score = float(merged["stable_cand"].mean())
    differences = merged["stable_cand"].to_numpy(float) - merged["stable_base"].to_numpy(float)
    boot = A.bootstrap_difference(differences, seed=SEED)
    reasons = []
    if not score < FROZEN_BASELINE:
        reasons.append(
            f"(a) stable_score {score:.6f} did not improve on frozen {FROZEN_BASELINE:.6f}"
        )
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(
            f"(b) paired-by-fold bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0"
        )
    for cohort in ("north", "south"):
        base = float(merged[f"{cohort}_base"].mean())
        cand = float(merged[f"{cohort}_cand"].mean())
        if cand > base * 1.02:
            reasons.append(f"(c) {cohort} cohort regressed by more than 2%")
    tournaments = merged.groupby("tournament")[["stable_base", "stable_cand"]].mean()
    regressed = tournaments.index[
        tournaments["stable_cand"] > tournaments["stable_base"] * 1.02
    ].tolist()
    if regressed:
        reasons.append("(d) tournament-family stable regression >2%: " + ", ".join(regressed))
    return {
        "name": name, "stable_score": score,
        "baseline_stable_score": float(merged["stable_base"].mean()),
        "delta": score - float(merged["stable_base"].mean()),
        "improvement_vs_frozen": 1.0 - score / FROZEN_BASELINE,
        "bootstrap_paired_by_fold": boot,
        "north": float(merged["north_cand"].mean()), "south": float(merged["south_cand"].mean()),
        "north_base": float(merged["north_base"].mean()),
        "south_base": float(merged["south_base"].mean()),
        "per_tournament": {
            key: {"baseline": float(row["stable_base"]), "candidate": float(row["stable_cand"])}
            for key, row in tournaments.iterrows()
        },
        "per_fold": merged[[
            "fold", "tournament", "hemisphere", "stable_base", "stable_cand",
        ]].to_dict(orient="records"),
        "accepted": not reasons, "reasons": reasons,
    }


def temporal_check(table: LossTable, fitter, *, log: bool = False) -> dict:
    """Fit on the earliest half of folds, evaluate on the latest half."""
    n = len(table.folds)
    cut = n // 2
    train, held = np.arange(cut), np.arange(cut, n)
    fitted = fitter(table, train)
    indices = fitted[0] if isinstance(fitted, tuple) else fitted
    base = baseline_indices(table)
    candidate = np.array([F.fold_score(table, i, indices, log=log) for i in held])
    baseline = np.array([F.fold_score(table, i, base) for i in held])
    differences = candidate - baseline
    return {
        "n_fit_folds": int(cut), "n_held_folds": int(n - cut),
        "fit_folds": [table.folds[i] for i in train],
        "held_out_folds": [table.folds[i] for i in held],
        "held_out_stable": float(np.mean(candidate)),
        "held_out_baseline": float(np.mean(baseline)),
        "held_out_delta": float(np.mean(differences)),
        "held_out_bootstrap": A.bootstrap_difference(differences, seed=SEED),
        "held_out_improves": bool(np.mean(differences) < 0),
        "fitted_weights": {
            target: float(GRID[indices[position]])
            for position, target in enumerate(table.targets)
        },
    }


def trial_fitted(table, baseline, name, hypothesis, fitter, *, log=False) -> dict:
    per_fold, fits = F.cross_fit(table, fitter, log=log)
    record = evaluate(table, score_frame(table, per_fold, log=log), baseline, name)
    record.update({
        "hypothesis": hypothesis,
        "fitting": "leave-one-fold-out cross-fitted (out-of-sample)",
        "n_fitted_parameters": name,
        "cross_fits": fits,
        "temporal_holdout": temporal_check(table, fitter, log=log),
        "in_sample_reference": float(np.mean([
            F.fold_score(table, i, (fitter(table, np.arange(len(table.folds)))[0]
                                    if isinstance(fitter(table, np.arange(len(table.folds))), tuple)
                                    else fitter(table, np.arange(len(table.folds)))), log=log)
            for i in range(len(table.folds))
        ])),
    })
    return record


def trial_fixed(table, baseline, name, hypothesis, indices, *, log=False) -> dict:
    per_fold = [indices] * len(table.folds)
    record = evaluate(table, score_frame(table, per_fold, log=log), baseline, name)
    record.update({
        "hypothesis": hypothesis,
        "fitting": "no fitted parameter (fixed rule)",
        "weights": {
            target: float(GRID[indices[position]])
            for position, target in enumerate(table.targets)
        },
        "log_space": log,
    })
    return record


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    caches = build_cache()
    table = build_table(caches)
    control = run_control(table)
    print(json.dumps(control, indent=2), flush=True)
    if not control["reproduced"]:
        raise SystemExit("CONTROL FAILED -- refusing to evaluate candidates.")

    density = density_vector(caches)
    base_frame = score_frame(table, [baseline_indices(table)] * len(table.folds))
    trials = []

    fitted_queue = [
        ("C1_global_weight", 1,
         "The global 0.5 was chosen on 6N-2025 LORO, never on this target. "
         "Refitting one global v4 weight on the historical folds should move it "
         "toward the all-rugby optimum.",
         F.fit_global),
        ("C2_per_target_weight", 24,
         "Empirical priors win on sparse/attacking events and the GBDT wins on "
         "dense volume events, so one weight per target should beat a single "
         "global weight. Highest overfitting risk in the queue.",
         F.fit_per_target),
        ("C3_per_loss_family_weight", 4,
         "The metric mixes Poisson deviance, log-squared, log-loss and MAE. One "
         "weight per loss family captures the structural split at a fraction of "
         "the variance of 24 free weights.",
         F.fit_family),
        ("C4_density_parametric_weight", 2,
         "Encode the density hypothesis directly: w(target) = clip(a + b * "
         "positive_rate, 0, 1), monotone in event density.",
         partial(F.fit_density, density=density)),
        ("C5_shrunk_per_target_weight", 25,
         "Per-target weights shrunk toward the fitted global weight, with the "
         "shrinkage strength chosen by an inner leave-one-fold-out pass over the "
         "training folds only. Keeps C2's signal, discards its noise.",
         F.fit_shrunk_per_target),
    ]
    for name, n_params, hypothesis, fitter in fitted_queue:
        print(f"--- {name} ---", flush=True)
        record = trial_fitted(table, base_frame, name, hypothesis, fitter)
        record["n_fitted_parameters"] = n_params
        trials.append(record)
        print(json.dumps({k: record[k] for k in (
            "stable_score", "delta", "in_sample_reference", "accepted", "reasons",
        )}, indent=2, default=float), flush=True)

    half = baseline_indices(table)
    fixed_queue = [
        ("C6_log_space_all_targets",
         "The blend is on event MEANS, but the metric is Poisson deviance / "
         "log-squared, whose natural link is log. Blending every target on the "
         "log1p scale at the frozen 0.5 weight adds no fitted parameter.",
         half, True),
    ]
    for name, hypothesis, indices, log in fixed_queue:
        print(f"--- {name} ---", flush=True)
        record = trial_fixed(table, base_frame, name, hypothesis, indices, log=log)
        trials.append(record)
        print(json.dumps({k: record[k] for k in (
            "stable_score", "delta", "accepted", "reasons",
        )}, indent=2, default=float), flush=True)

    # C8: the best structural rule re-fitted in log space, cross-fitted.
    print("--- C8_log_space_per_target_weight ---", flush=True)
    record = trial_fitted(
        table, base_frame, "C8_log_space_per_target_weight",
        "If the log link is the right space for these losses, the per-target "
        "weight should be fitted there too rather than in mean space.",
        F.fit_per_target, log=True,
    )
    record["n_fitted_parameters"] = 24
    trials.append(record)
    print(json.dumps({k: record[k] for k in (
        "stable_score", "delta", "accepted", "reasons",
    )}, indent=2, default=float), flush=True)

    payload = {
        "schema_version": 1, "seed": SEED, "frozen_baseline": FROZEN_BASELINE,
        "control": control, "n_folds": len(table.folds),
        "weight_grid_step": float(GRID[1] - GRID[0]),
        "acceptance_rule": {
            "a": "competition-balanced stable_score improves on 0.886470",
            "b": "paired-by-fold bootstrap difference excludes 0",
            "c": "neither north nor south cohort regresses by >2%",
            "d": "no tournament-family stable regression >2%",
            "e": ("no extended-event loss regression >5% -- satisfied by "
                  "construction: every candidate leaves the 8 extended events at "
                  "the frozen 0.5 weight, so their predictions are unchanged"),
            "f": ("any newly fitted parameter is cross-fitted leave-one-fold-out "
                  "and additionally checked on a strict temporal split"),
        },
        "trials": trials,
    }
    (WORK / "ledger.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"wrote {WORK / 'ledger.json'}", flush=True)


if __name__ == "__main__":
    main()
