"""Alternative component predictions for the all-rugby P3 blend search.

Phase 1 (see ``allrugby_p3/LEDGER.md``) held the two components fixed and
searched blend rules; the family was exhausted at C5. Phase 2 changes the
components themselves. The empirical engine has a config surface
(``empirical.VARIANTS``) whose ``t3_eb_off_ebk_head`` variant scores 0.904538
standalone against the frozen component's 0.939031, so re-blending against it
is the natural next move.

``predict_frame`` on the empirical engine reads only base store columns, so a
variant component needs no v4 feature build -- it is fitted and predicted
directly off the fold's strict training frame and masked candidates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import allrugby as A
from .empirical import VARIANTS, EmpiricalEventModel
from .folds import build_folds, evaluation_frame, masked_candidates, strict_training_frame


def component_path(fold_label: str, component: str) -> Path:
    return A.WORK / "components" / component / f"{fold_label}.jsonl"


def build_empirical_variant(
    variant: str, fold_labels: tuple[str, ...] | None = None,
) -> str:
    """Fit and predict ``empirical.VARIANTS[variant]`` on every historical fold."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown empirical variant {variant!r}")
    component = f"empirical_{variant}"
    store = A._store()
    folds = build_folds(store)
    if fold_labels:
        folds = [fold for fold in folds if fold.label in set(fold_labels)]
    for fold in folds:
        target = component_path(fold.label, component)
        if target.exists():
            print(f"[{fold.label}] {component} cached", flush=True)
            continue
        train = strict_training_frame(store, fold)
        candidates = masked_candidates(evaluation_frame(store, fold).reset_index(drop=True))
        model = EmpiricalEventModel(asof=fold.cutoff, config=VARIANTS[variant]).fit(train)
        A._write_predictions(target, model.predict_frame(candidates))
        print(f"[{fold.label}] {component} written ({len(candidates):,} rows)", flush=True)
    return component


if __name__ == "__main__":
    import sys

    for name in sys.argv[1:] or ["t3_eb_off_ebk_head"]:
        build_empirical_variant(name)
