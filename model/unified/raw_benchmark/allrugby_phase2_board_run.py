"""Build the per-competition scoreboard for the phase-2 winner.

Every engine is expressed as a weight vector over one union component tuple, so
the frozen P3 blend, the phase-1 C5 incumbent, the phase-2 winner and each bare
component are all scored on identical folds and slates under both rubrics.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import allrugby_phase2 as P
from . import allrugby_phase2_board as B
from .allrugby_phase2_run import BASE_PAIR, T3, UNION


def _lift(components: tuple[str, ...], vector) -> list[float]:
    """Express a weight vector over ``components`` in UNION coordinates."""
    out = [0.0] * len(UNION)
    for value, component in zip(np.asarray(vector, dtype=float), components):
        out[UNION.index(component)] = float(value)
    return out


def _cross_fitted_rule(result: dict):
    """Per-fold weights from a candidate's leave-one-fold-out fits."""
    components = tuple(result["components"])
    by_fold = {
        entry["held_out_fold"]: {
            target: _lift(components, vector)
            for target, vector in entry["weights"].items()
        }
        for entry in result["fits"]
    }
    return B.per_fold_rule(by_fold, len(UNION))


def _calibrated_rule(result: dict, base: dict):
    """A calibration candidate reuses its base candidate's per-fold weights.

    ``run_calibration_candidate`` fits the same blend rule as its base with the
    same fitter on the same folds, so the weights are exactly the base's
    cross-fitted fits; only the per-target scales are new.
    """
    components = tuple(result["components"])
    weights_by_fold = {
        entry["held_out_fold"]: {
            target: _lift(tuple(base["components"]), vector)
            for target, vector in entry["weights"].items()
        }
        for entry in base["fits"]
    }
    scales_by_fold = dict(zip(
        result["per_fold"]["folds"], result["per_fold"]["scales"],
    ))
    targets = list(result["deployed_scales"])

    def factory(fold_label: str):
        table = weights_by_fold[fold_label]
        scales = dict(zip(targets, scales_by_fold[fold_label]))

        def weight_for(target: str):
            return np.asarray(table.get(target, [1.0 / len(UNION)] * len(UNION)))

        def scale_for(target: str):
            return float(scales.get(target, 1.0))
        return weight_for, scale_for
    return factory


def main() -> None:
    results = json.loads((P.WORK / "results.json").read_text())
    by_name = {entry["candidate"]: entry for entry in results}
    # Row-group candidates carry one weight vector per (target, group); the board
    # blends per target, so only per-target rules can be scored here.
    blendable = [
        entry for entry in results
        if entry["accepted"] and entry.get("fits")
        and all("weights" in fit for fit in entry["fits"])
    ]
    calibrated = [
        entry for entry in results
        if entry["accepted"] and entry.get("deployed_scales")
    ]
    accepted = blendable + calibrated
    winner = min(accepted, key=lambda entry: entry["stable_score"]) if accepted else None

    control = json.loads((P.WORK / "control.json").read_text())
    rules = {
        "v4_only": B.constant_rule(_lift(("v4",), [1.0]), len(UNION)),
        "empirical_only": B.constant_rule(_lift(("empirical",), [1.0]), len(UNION)),
        f"empirical_{T3}_only": B.constant_rule(_lift((T3,), [1.0]), len(UNION)),
        "naive_comparator": B.constant_rule(_lift((P.NAIVE,), [1.0]), len(UNION)),
        "p3_event_50_frozen": B.constant_rule(_lift(BASE_PAIR, [0.5, 0.5]), len(UNION)),
    }
    if "c5_fits" in control:
        rules["C5_phase1"] = B.per_fold_rule({
            entry["held_out_fold"]: {
                target: _lift(BASE_PAIR, vector)
                for target, vector in entry["weights"].items()
            }
            for entry in control["c5_fits"]
        }, len(UNION))
    best_blend = min(blendable, key=lambda entry: entry["stable_score"]) if blendable else None
    if best_blend is not None:
        rules[f"{best_blend['candidate']}_phase2"] = _cross_fitted_rule(best_blend)
    for entry in calibrated:
        # The base is the blend candidate it was fitted on top of: same
        # components, same point set, same fitter.
        base = next(
            (other for other in blendable
             if other["components"] == entry["components"]
             and other["n_points"] == entry["n_points"]),
            None,
        )
        if base is None:
            print(f"skipping {entry['candidate']}: no matching base candidate")
            continue
        rules[f"{entry['candidate']}_phase2"] = _calibrated_rule(entry, base)

    overall, by_tournament = B.build(UNION, rules, "board")
    print(overall.to_string(index=False))
    print()
    print(by_tournament.to_string(index=False))
    winners = B.winner_table(by_tournament, "mae", higher_is_better=False)
    winners.to_csv(P.WORK / "board_winners_mae.csv", index=False)
    print()
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
