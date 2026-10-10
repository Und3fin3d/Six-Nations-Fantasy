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
    "tries", "tackles", "metres", "try_assists", "conversion_goals",
    "penalty_goals", "drop_goals_converted", "defenders_beaten", "offload",
    "tackle_turnover", "penalties_conceded", "yellow_cards", "red_cards",
]
FORM_COL = {c: f"form_per80_{c}" for c in SCORED}
FORM_COL["yellow_cards"] = None
FORM_COL["red_cards"] = None

COMP_KIND = {
    "tackles": "rate", "metres": "continuous",
    "tries": "count", "try_assists": "count", "defenders_beaten": "count",
    "offload": "count", "tackle_turnover": "count", "penalties_conceded": "count",
    "conversion_goals": "kick", "penalty_goals": "kick",
    "drop_goals_converted": "rare", "red_cards": "rare", "yellow_cards": "card",
}

MIN_MINUTES = 10


class PositionMeanImputer:
    def fit(self, X: pd.DataFrame, pos: pd.Series) -> "PositionMeanImputer":
        self.cols_ = list(X.columns)
        self.global_ = X.mean(numeric_only=True)
        self.pos_table_ = X.groupby(pos.to_numpy()).mean(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
        posframe = self.pos_table_.reindex(pos.to_numpy())
        posframe.index = X.index
        return X.fillna(posframe).fillna(self.global_).fillna(0.0)


def numeric_cols(df: pd.DataFrame, mode: str) -> list[str]:
    return [c for c in feature_view(df, mode) if c not in CATEGORICAL_COLS]


def build_matrix(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = numeric_cols(df, mode)
    Xtr = df.iloc[train_idx][cols].astype(float)
    Xte = df.iloc[test_idx][cols].astype(float)
    imp = PositionMeanImputer().fit(Xtr, df.iloc[train_idx]["canonical_pos"])
    Xtr = imp.transform(Xtr, df.iloc[train_idx]["canonical_pos"])
    Xte = imp.transform(Xte, df.iloc[test_idx]["canonical_pos"])
    sc = StandardScaler().fit(Xtr.to_numpy())
    return sc.transform(Xtr.to_numpy()), sc.transform(Xte.to_numpy()), cols


def expected_minutes(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> np.ndarray:
    tr = df.iloc[train_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows: pd.DataFrame) -> np.ndarray:
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    if mode == "pre_team_sheet":
        return np.clip(prior(df.iloc[test_idx]), 0, 80)

    feats = ["started", "jersey", "is_forward"]
    Xtr = np.column_stack([tr[feats].astype(float).to_numpy(), prior(tr)])
    ytr = tr["minutes"].to_numpy(float)
    reg = Ridge(alpha=10.0).fit(Xtr, ytr)
    te = df.iloc[test_idx]
    Xte = np.column_stack([te[feats].astype(float).to_numpy(), prior(te)])
    return np.clip(reg.predict(Xte), 0, 80)


def _rate_target(df: pd.DataFrame, comp: str) -> np.ndarray:
    ms = (df["minutes"].to_numpy(float) / 80.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = df[f"y_{comp}"].to_numpy(float) / ms
    return r


def rates_to_components(
    rates: pd.DataFrame, minutes_hat: np.ndarray
) -> pd.DataFrame:
    factor = (minutes_hat / 80.0)
    comp = rates.clip(lower=0.0).multiply(factor, axis=0)
    return comp


def score_recon(comp: pd.DataFrame, is_forward: np.ndarray) -> np.ndarray:
    recs = comp[SCORED].to_dict("records")
    return np.array(
        [score_components(c, bool(f)) for c, f in zip(recs, is_forward)],
        dtype=float,
    )


def predict_rates(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    mode: str, engine: str,
) -> pd.DataFrame:
    if engine == "naive":
        return _rates_naive(df, train_idx, test_idx)
    return _rates_linear(df, train_idx, test_idx, mode, engine)


def _position_rate_prior(df: pd.DataFrame, train_idx: np.ndarray, comp: str) -> tuple[pd.Series, float]:
    tr = df.iloc[train_idx]
    tr = tr[tr["minutes"] >= MIN_MINUTES]
    rate = _rate_target(tr, comp)
    s = pd.Series(rate, index=tr.index)
    by_pos = s.groupby(tr["canonical_pos"].to_numpy()).mean()
    return by_pos, float(np.nanmean(rate))


def _rates_naive(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    for comp in SCORED:
        fc = FORM_COL[comp]
        by_pos, glob = _position_rate_prior(df, train_idx, comp)
        prior = te["canonical_pos"].map(by_pos).fillna(glob).to_numpy(float)
        if fc is None:
            out[comp] = prior
        else:
            base = te[fc].to_numpy(float)
            out[comp] = np.where(np.isnan(base), prior, base)
    return out.clip(lower=0.0)


def _rates_linear(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, engine: str
) -> pd.DataFrame:
    tr_full = df.iloc[train_idx]
    keep = (tr_full["minutes"] >= MIN_MINUTES).to_numpy()
    use_train_idx = train_idx[keep]
    Xtr, Xte, _ = build_matrix(df, use_train_idx, test_idx, mode)
    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    for comp in SCORED:
        y = _rate_target(df.iloc[use_train_idx], comp)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(y, 0.0):
            out[comp] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if engine == "ridge":
                mdl = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, y)
                pred = mdl.predict(Xte)
            elif engine == "bayesian":
                pred = _bayesian_component_predict(
                    df, use_train_idx, test_idx, Xtr, Xte, y, comp)
            else:
                if comp == "metres":
                    mdl = TweedieRegressor(power=1.3, alpha=1.0, max_iter=400)
                else:
                    mdl = PoissonRegressor(alpha=1.0, max_iter=400)
                mdl.fit(Xtr, np.clip(y, 0, None))
                pred = mdl.predict(Xte)
        out[comp] = np.clip(pred, 0.0, None)
    return out


def _bayesian_component_predict(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Xtr: np.ndarray,
    Xte: np.ndarray,
    y: np.ndarray,
    comp: str,
) -> np.ndarray:
    te = df.iloc[test_idx]
    by_pos, glob = _position_rate_prior(df, train_idx, comp)
    prior_rate = te["canonical_pos"].map(by_pos).fillna(glob).to_numpy(float)
    prior_rate = np.clip(np.nan_to_num(prior_rate, nan=0.0), 0.0, None)

    if COMP_KIND[comp] == "rare":
        return np.zeros(len(test_idx), dtype=float)

    use_log_rate = COMP_KIND[comp] in {"count", "kick", "card"}
    y_fit = np.clip(y, 0.0, None)
    prior_target = prior_rate
    if use_log_rate:
        y_fit = np.log1p(y_fit)
        prior_target = np.log1p(prior_rate)

    Xtr_aug = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte_aug = np.column_stack([np.ones(len(Xte)), Xte])
    prior_precision = np.concatenate([[1e-6], np.full(Xtr.shape[1], 10.0)])
    a = Xtr_aug.T @ Xtr_aug + np.diag(prior_precision)
    b = Xtr_aug.T @ y_fit
    coef = np.linalg.solve(a, b)
    a_inv_xte = np.linalg.solve(a, Xte_aug.T)
    mean = Xte_aug @ coef
    resid = y_fit - Xtr_aug @ coef
    sigma2 = float(np.mean(resid ** 2))
    if not np.isfinite(sigma2) or sigma2 < 1e-6:
        sigma2 = float(np.var(y_fit) + 1e-6)
    leverage = np.sum(Xte_aug * a_inv_xte.T, axis=1)
    std = np.sqrt(np.clip(sigma2 * (1.0 + leverage), 0.0, None))
    scale = float(np.nanstd(y_fit))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    trust = scale / (scale + np.clip(std, 0.0, None))
    pred_target = trust * mean + (1.0 - trust) * prior_target
    pred = np.expm1(pred_target) if use_log_rate else pred_target

    if comp in {"conversion_goals", "penalty_goals"}:
        gate = (te["role_goal_kicker_rate"].fillna(0.0).to_numpy(float) > 0.0).astype(float)
        pred = pred * gate
    return np.clip(pred, 0.0, None)


def direct_points_oof(df: pd.DataFrame, season: int, mode: str) -> pd.Series:
    sub = df[(df["season"] == season) & df["is_modern"] & df["has_label"]].copy()
    pos_local = np.arange(len(sub))
    pred = pd.Series(np.nan, index=sub.index, dtype=float)
    sub_reset = sub.reset_index(drop=True)
    for tr, va in group_kfold_indices(sub_reset, 5):
        Xtr, Xva, _ = build_matrix(sub_reset, tr, va, mode)
        y = sub_reset.iloc[tr]["official_pts"].to_numpy(float)
        mdl = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, y)
        pred.iloc[va] = mdl.predict(Xva)
    return pred
