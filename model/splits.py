#!/usr/bin/env python3
"""model/splits.py  —  splits + fold infrastructure (Phase 2).

Locked design:
  Holdout      : 2026 (sealed; touched once in the final phase).
  Backtest     : train components on 2023+2024, predict 2025.
  Deployment   : train components on 2023+2024+2025, predict 2026.
  HP tuning    : GroupKFold(5) on `fixture_id` over the component-training years
                 (all ~46 players in a match stay together -> no match leakage).
  Latent calib : SEPARATE and forward-chained.  When predicting round R of a
                 modern season, the latent is fit on all modern-labelled matches
                 with date < that round's date.  2025 R1 has no prior modern ->
                 cold-start fallback.

Within-season recency lives in the features (build_features decays FORM/CLASS on
date < fixture_date), so components are trained ONCE on the train years and
applied to every test-season row; no per-round component retraining.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def component_train(df: pd.DataFrame, test_season: int) -> np.ndarray:
    """Boolean mask of rows used to TRAIN components: all 6N rows with
    season < test_season.  Never includes the test season."""
    mask = (df["season"] < test_season).to_numpy()
    if df.loc[mask, "season"].ge(test_season).any():
        raise AssertionError("component_train leaked test-season rows")
    return mask


def group_kfold_indices(
    df_train: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (tr_idx, va_idx) positional indices via GroupKFold on `fixture_id`.

    Groups = matches, so no fixture appears in both train and validation of a
    fold (asserted).  Indices are positional into `df_train` (0..len-1).
    """
    groups = df_train["fixture_id"].to_numpy()
    X = np.zeros((len(df_train), 1))
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(X, groups=groups, y=None):
        tr_g = set(groups[tr_idx])
        va_g = set(groups[va_idx])
        if tr_g & va_g:
            raise AssertionError("GroupKFold produced overlapping fixture groups")
        yield tr_idx, va_idx


def latent_calib_rows(df: pd.DataFrame, asof_date) -> np.ndarray:
    """Boolean mask of modern-labelled rows strictly before `asof_date`
    (forward-chain).  `is_modern` excludes 2023's old-scale labels."""
    asof = pd.Timestamp(asof_date)
    mask = (
        df["is_modern"].to_numpy()
        & df["has_label"].to_numpy()
        & (df["date"] < asof).to_numpy()
    )
    # forward-chain guard: no row dated >= asof may enter calibration
    if (df.loc[mask, "date"] >= asof).any():
        raise AssertionError("latent_calib_rows included a row dated >= asof")
    # old-scale guard: 2023 official labels must never be calibration inputs
    if df.loc[mask, "season"].eq(2023).any():
        raise AssertionError("2023 old-scale rows leaked into latent calibration")
    return mask


def round_iter(
    df: pd.DataFrame, test_season: int
) -> Iterator[tuple[int, pd.Timestamp, np.ndarray]]:
    """Yield (round, asof_date, row_idx) for the test season in date order.

    `asof_date` is the earliest fixture date of that round, so latent
    calibration for round R uses only strictly-earlier matches (no intra-round
    leakage).  `row_idx` is the positional index of the round's rows in `df`.
    """
    pos = np.arange(len(df))
    season_mask = (df["season"] == test_season).to_numpy()
    rounds = sorted(df.loc[season_mask, "round"].unique())
    # order rounds by their earliest date
    order = sorted(
        rounds,
        key=lambda r: df.loc[season_mask & (df["round"] == r), "date"].min(),
    )
    for r in order:
        rmask = season_mask & (df["round"] == r).to_numpy()
        asof = df.loc[rmask, "date"].min()
        yield int(r), pd.Timestamp(asof), pos[rmask]


def _selfcheck() -> None:
    from model.data import load

    df = load()
    # backtest training rows
    m = component_train(df, 2025)
    print(f"component_train(2025): {m.sum()} rows, seasons={sorted(df.loc[m,'season'].unique())}")
    assert m.sum() == 1380 and set(df.loc[m, "season"]) == {2023, 2024}

    m26 = component_train(df, 2026)
    print(f"component_train(2026): {m26.sum()} rows, seasons={sorted(df.loc[m26,'season'].unique())}")
    assert m26.sum() == 2070 and set(df.loc[m26, "season"]) == {2023, 2024, 2025}

    # GroupKFold disjoint
    dtr = df.loc[m].reset_index(drop=True)
    nf = 0
    for tr, va in group_kfold_indices(dtr, 5):
        assert not (set(dtr.fixture_id.iloc[tr]) & set(dtr.fixture_id.iloc[va]))
        nf += 1
    print(f"GroupKFold folds={nf}, groups disjoint OK")

    # forward-chained latent
    rounds = list(round_iter(df, 2025))
    r1, asof1, idx1 = rounds[0]
    n_cold = latent_calib_rows(df, asof1).sum()
    print(f"2025 R{r1} asof={asof1.date()} -> latent_calib rows={n_cold} (cold-start expected 0)")
    assert n_cold == 0
    r_last, asof_last, _ = rounds[-1]
    n_last = latent_calib_rows(df, asof_last).sum()
    print(f"2025 R{r_last} asof={asof_last.date()} -> latent_calib rows={n_last} (>0 expected)")
    assert n_last > 0
    print("Phase 2 splits OK")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    _selfcheck()
