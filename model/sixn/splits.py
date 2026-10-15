from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def component_train(df: pd.DataFrame, test_season: int) -> np.ndarray:
    return (df["season"] < test_season).to_numpy()


def group_kfold_indices(df_train: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    yield from GroupKFold(n_splits=n_splits).split(np.zeros((len(df_train), 1)), groups=df_train["fixture_id"].to_numpy(), y=None)


def latent_calib_rows(df: pd.DataFrame, asof_date) -> np.ndarray:
    return df["is_modern"].to_numpy() & df["has_label"].to_numpy() & (df["date"] < pd.Timestamp(asof_date)).to_numpy()


def round_iter(df: pd.DataFrame, test_season: int) -> Iterator[tuple[int, pd.Timestamp, np.ndarray]]:
    pos = np.arange(len(df))
    season_mask = (df["season"] == test_season).to_numpy()
    rounds = sorted(df.loc[season_mask, "round"].unique())
    order = sorted(rounds, key=lambda r: df.loc[season_mask & (df["round"] == r), "date"].min())
    for r in order:
        rmask = season_mask & (df["round"] == r).to_numpy()
        asof = df.loc[rmask, "date"].min()
        yield int(r), pd.Timestamp(asof), pos[rmask]
