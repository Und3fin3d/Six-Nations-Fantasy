from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge, RidgeCV, TweedieRegressor
from sklearn.preprocessing import StandardScaler

from pipeline.sixn.build_targets import score_components
from model.sixn.data import CATEGORICAL_COLS, feature_view
from model.sixn.splits import group_kfold_indices

SCORED = [
    "tries", "tackles", "metres", "try_assists", "conversion_goals", "penalty_goals", "drop_goals_converted", "defenders_beaten", "offload", "tackle_turnover",
    "penalties_conceded", "yellow_cards", "red_cards",
]
FORM_COL = {c: None if c in ("yellow_cards", "red_cards") else f"form_per80_{c}" for c in SCORED}

COMP_KIND = {
    "tackles": "rate", "metres": "continuous", "tries": "count", "try_assists": "count", "defenders_beaten": "count", "offload": "count",
    "tackle_turnover": "count", "penalties_conceded": "count", "conversion_goals": "kick", "penalty_goals": "kick", "drop_goals_converted": "rare",
    "red_cards": "rare", "yellow_cards": "card",
}

MIN_MINUTES = 10


class PositionMeanImputer:
    def fit(self, X: pd.DataFrame, pos: pd.Series) -> "PositionMeanImputer":
        self.global_ = X.mean(numeric_only=True)
        self.pos_table_ = X.groupby(pos.to_numpy()).mean(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
        posframe = self.pos_table_.reindex(pos.to_numpy())
        posframe.index = X.index
        return X.fillna(posframe).fillna(self.global_).fillna(0.0)


def numeric_cols(df: pd.DataFrame, mode: str) -> list[str]:
    return [c for c in feature_view(df, mode) if c not in CATEGORICAL_COLS]


def build_matrix(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = numeric_cols(df, mode)
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    imp = PositionMeanImputer().fit(tr[cols].astype(float), tr["canonical_pos"])
    Xtr, Xte = imp.transform(tr[cols].astype(float), tr["canonical_pos"]), imp.transform(te[cols].astype(float), te["canonical_pos"])
    sc = StandardScaler().fit(Xtr.to_numpy())
    return sc.transform(Xtr.to_numpy()), sc.transform(Xte.to_numpy()), cols


def expected_minutes(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> np.ndarray:
    tr = df.iloc[train_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows: pd.DataFrame) -> np.ndarray:
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    if mode == "pre_team_sheet":
        return np.clip(prior(df.iloc[test_idx]), 0, 80)

    design = lambda rows: np.column_stack([rows[["started", "jersey", "is_forward"]].astype(float).to_numpy(), prior(rows)])
    return np.clip(Ridge(alpha=10.0).fit(design(tr), tr["minutes"].to_numpy(float)).predict(design(df.iloc[test_idx])), 0, 80)


def _rate_target(df: pd.DataFrame, comp: str) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return df[f"y_{comp}"].to_numpy(float) / (df["minutes"].to_numpy(float) / 80.0)


def rates_to_components(rates: pd.DataFrame, minutes_hat: np.ndarray) -> pd.DataFrame:
    return rates.clip(lower=0.0).multiply(minutes_hat / 80.0, axis=0)


def score_recon(comp: pd.DataFrame, is_forward: np.ndarray) -> np.ndarray:
    return np.array([score_components(c, bool(f)) for c, f in zip(comp[SCORED].to_dict("records"), is_forward)], dtype=float)


def predict_rates(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, engine: str) -> pd.DataFrame:
    if engine == "naive":
        return _rates_naive(df, train_idx, test_idx)
    return _rates_linear(df, train_idx, test_idx, mode, engine)


def _position_rate_prior(df: pd.DataFrame, train_idx: np.ndarray, comp: str) -> tuple[pd.Series, float]:
    tr = df.iloc[train_idx]
    tr = tr[tr["minutes"] >= MIN_MINUTES]
    rate = _rate_target(tr, comp)
    return pd.Series(rate, index=tr.index).groupby(tr["canonical_pos"].to_numpy()).mean(), float(np.nanmean(rate))


def _rates_naive(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    for comp in SCORED:
        by_pos, glob = _position_rate_prior(df, train_idx, comp)
        prior = te["canonical_pos"].map(by_pos).fillna(glob).to_numpy(float)
        out[comp] = prior if FORM_COL[comp] is None else np.where(np.isnan(te[FORM_COL[comp]].to_numpy(float)), prior, te[FORM_COL[comp]].to_numpy(float))
    return out.clip(lower=0.0)


def _rates_linear(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, engine: str) -> pd.DataFrame:
    use_train_idx = train_idx[(df.iloc[train_idx]["minutes"] >= MIN_MINUTES).to_numpy()]
    Xtr, Xte, _ = build_matrix(df, use_train_idx, test_idx, mode)
    out = pd.DataFrame(index=df.iloc[test_idx].index, columns=SCORED, dtype=float)
    for comp in SCORED:
        y = np.nan_to_num(_rate_target(df.iloc[use_train_idx], comp), nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(y, 0.0):
            out[comp] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if engine == "ridge":
                pred = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, y).predict(Xte)
            else:
                pred = (
                    (TweedieRegressor(power=1.3, alpha=1.0, max_iter=400) if comp == "metres" else PoissonRegressor(alpha=1.0, max_iter=400))
                    .fit(Xtr, np.clip(y, 0, None))
                    .predict(Xte)
                )
        out[comp] = np.clip(pred, 0.0, None)
    return out


def direct_points_oof(df: pd.DataFrame, season: int, mode: str) -> pd.Series:
    sub = df[(df["season"] == season) & df["is_modern"] & df["has_label"]].copy()
    pred = pd.Series(np.nan, index=sub.index, dtype=float)
    sub_reset = sub.reset_index(drop=True)
    for tr, va in group_kfold_indices(sub_reset, 5):
        Xtr, Xva, _ = build_matrix(sub_reset, tr, va, mode)
        pred.iloc[va] = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, sub_reset.iloc[tr]["official_pts"].to_numpy(float)).predict(Xva)
    return pred
