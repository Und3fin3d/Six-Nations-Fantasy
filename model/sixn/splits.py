from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def component_train(df: pd.DataFrame, test_season: int) -> np.ndarray:
    return (df["season"] < test_season).to_numpy()


def chronological_indices(df_train: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(df_train["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("chronological folds require complete match dates")
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 2:
        raise ValueError("chronological folds require at least two match dates")
    blocks = np.array_split(unique_dates, min(n_splits + 1, len(unique_dates)))
    for block in blocks[1:]:
        yield np.flatnonzero(dates.lt(block[0])), np.flatnonzero(dates.isin(block))


def latent_calib_rows(df: pd.DataFrame, asof_date) -> np.ndarray:
    asof = pd.Timestamp(asof_date)
    mask = df["is_modern"].to_numpy() & df["has_label"].to_numpy() & (df["date"] < asof).to_numpy()
    return mask


def round_iter(df: pd.DataFrame, test_season: int) -> Iterator[tuple[int, pd.Timestamp, np.ndarray]]:
    pos = np.arange(len(df))
    season_mask = (df["season"] == test_season).to_numpy()
    rounds = sorted(df.loc[season_mask, "round"].unique())
    order = sorted(rounds, key=lambda r: df.loc[season_mask & (df["round"] == r), "date"].min())
    for r in order:
        rmask = season_mask & (df["round"] == r).to_numpy()
        asof = df.loc[rmask, "date"].min()
        yield int(r), pd.Timestamp(asof), pos[rmask]
