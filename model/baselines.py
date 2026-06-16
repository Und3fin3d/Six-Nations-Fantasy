#!/usr/bin/env python3
"""model/baselines.py  —  the must-beat bar + shared model utilities (Phase 3).

Baselines (predict the 13 SCORED per-80 component rates -> x minutes_hat/80 ->
score with the frozen `score_components`):
  B0 naive  — predicted rate = `form_per80_<comp>` (cards -> position prior).
  B1 ridge  — RidgeCV per component on scaled, position-mean-imputed features.
  B2 glm    — Poisson (Tweedie for metres) GLM per component, strong L2.
  B3 direct — GroupKFold OOF Ridge on `official_pts` (cross-check ONLY, never deployed).

This module also hosts the shared pieces the later phases reuse (kept here, not
in a new file, to respect the planned output set and avoid circular imports):
  SCORED / FORM_COL / COMP_KIND     component metadata
  PositionMeanImputer               cold-start imputation by canonical_pos
  numeric_cols / build_matrix       linear-model design matrix
  expected_minutes                  minutes_hat per mode (Phase-4 also uses this)
  rates_to_components / score_recon component dict assembly + deterministic score
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge, RidgeCV, TweedieRegressor
from sklearn.preprocessing import StandardScaler

from build_targets import score_components
from model.data import CATEGORICAL_COLS, feature_view
from model.splits import group_kfold_indices

# --- the 13 SCORED components (everything else is non-scoring) --------------
SCORED = [
    "tries", "tackles", "metres", "try_assists", "conversion_goals",
    "penalty_goals", "drop_goals_converted", "defenders_beaten", "offload",
    "tackle_turnover", "penalties_conceded", "yellow_cards", "red_cards",
]
# component -> its naive-form per-80 column (None where no form rate exists)
FORM_COL = {c: f"form_per80_{c}" for c in SCORED}
FORM_COL["yellow_cards"] = None
FORM_COL["red_cards"] = None

# distribution family per component (drives the per-component objective map)
COMP_KIND = {
    "tackles": "rate", "metres": "continuous",
    "tries": "count", "try_assists": "count", "defenders_beaten": "count",
    "offload": "count", "tackle_turnover": "count", "penalties_conceded": "count",
    "conversion_goals": "kick", "penalty_goals": "kick",
    "drop_goals_converted": "rare", "red_cards": "rare", "yellow_cards": "card",
}

MIN_MINUTES = 10  # rows used to learn per-80 rates (avoid tiny-minute blowups)


# ---------------------------------------------------------------------------
# imputation + design matrix
# ---------------------------------------------------------------------------
class PositionMeanImputer:
    """Fill NaN with the TRAIN canonical_pos group mean, then global mean.

    Pairs with the explicit `has_*` cold-start flags already in the matrix, so
    the model can tell imputed cold-start rows from observed ones.
    """

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
    """Feature-view columns minus the categorical text column(s)."""
    return [c for c in feature_view(df, mode) if c not in CATEGORICAL_COLS]


def build_matrix(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Imputed + standardized design matrices for linear/GLM baselines.

    Imputer and scaler are fit on TRAIN only; bools cast to float.
    """
    cols = numeric_cols(df, mode)
    Xtr = df.iloc[train_idx][cols].astype(float)
    Xte = df.iloc[test_idx][cols].astype(float)
    imp = PositionMeanImputer().fit(Xtr, df.iloc[train_idx]["canonical_pos"])
    Xtr = imp.transform(Xtr, df.iloc[train_idx]["canonical_pos"])
    Xte = imp.transform(Xte, df.iloc[test_idx]["canonical_pos"])
    sc = StandardScaler().fit(Xtr.to_numpy())
    return sc.transform(Xtr.to_numpy()), sc.transform(Xte.to_numpy()), cols


# ---------------------------------------------------------------------------
# expected minutes (points scale with time)
# ---------------------------------------------------------------------------
def expected_minutes(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> np.ndarray:
    """minutes_hat for the test rows.

    pre_team_sheet : decayed `form_minutes_recent`, cold-start -> position mean.
    post_team_sheet: small Ridge on PIT lineup fields (started/jersey) + form.
    Always clipped to [0, 80]; realised `minutes` is never an input.
    """
    tr = df.iloc[train_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows: pd.DataFrame) -> np.ndarray:
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    if mode == "pre_team_sheet":
        return np.clip(prior(df.iloc[test_idx]), 0, 80)

    # post_team_sheet: started/jersey known -> a stronger minutes model
    feats = ["started", "jersey", "is_forward"]
    Xtr = np.column_stack([tr[feats].astype(float).to_numpy(), prior(tr)])
    ytr = tr["minutes"].to_numpy(float)
    reg = Ridge(alpha=10.0).fit(Xtr, ytr)
    te = df.iloc[test_idx]
    Xte = np.column_stack([te[feats].astype(float).to_numpy(), prior(te)])
    return np.clip(reg.predict(Xte), 0, 80)


# ---------------------------------------------------------------------------
# rate targets + scoring
# ---------------------------------------------------------------------------
def _rate_target(df: pd.DataFrame, comp: str) -> np.ndarray:
    """Realised per-80 rate of a component (y / min_share)."""
    ms = (df["minutes"].to_numpy(float) / 80.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = df[f"y_{comp}"].to_numpy(float) / ms
    return r


def rates_to_components(
    rates: pd.DataFrame, minutes_hat: np.ndarray
) -> pd.DataFrame:
    """Convert per-80 rates to non-negative per-match component counts."""
    factor = (minutes_hat / 80.0)
    comp = rates.clip(lower=0.0).multiply(factor, axis=0)
    return comp


def score_recon(comp: pd.DataFrame, is_forward: np.ndarray) -> np.ndarray:
    """Deterministic modern-formula score of a component frame (reuses the
    frozen `score_components`; no coefficients re-declared here)."""
    recs = comp[SCORED].to_dict("records")
    return np.array(
        [score_components(c, bool(f)) for c, f in zip(recs, is_forward)],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# baseline component-rate predictors  -> returns per-80 rate DataFrame
# ---------------------------------------------------------------------------
def predict_rates(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    mode: str, engine: str,
) -> pd.DataFrame:
    """Per-80 rate predictions (index=test_idx, cols=SCORED) for a baseline engine."""
    if engine == "naive":
        return _rates_naive(df, train_idx, test_idx)
    if engine in ("ridge", "glm"):
        return _rates_linear(df, train_idx, test_idx, mode, engine)
    raise ValueError(f"unknown baseline engine {engine!r}")


def _position_rate_prior(df: pd.DataFrame, train_idx: np.ndarray, comp: str) -> tuple[pd.Series, float]:
    """Position-group mean realised per-80 rate (train, minutes>=10) + global."""
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
        if fc is None:                       # cards: no form rate -> position prior
            out[comp] = prior
        else:                                 # naive form, cold-start -> position prior
            base = te[fc].to_numpy(float)
            out[comp] = np.where(np.isnan(base), prior, base)
    return out.clip(lower=0.0)


def _rates_linear(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, engine: str
) -> pd.DataFrame:
    # train only on rows with enough minutes for a stable per-80 rate
    tr_full = df.iloc[train_idx]
    keep = (tr_full["minutes"] >= MIN_MINUTES).to_numpy()
    use_train_idx = train_idx[keep]
    Xtr, Xte, _ = build_matrix(df, use_train_idx, test_idx, mode)
    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    for comp in SCORED:
        y = _rate_target(df.iloc[use_train_idx], comp)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if np.allclose(y, 0.0):               # degenerate (e.g. red cards) -> 0
            out[comp] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if engine == "ridge":
                mdl = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, y)
                pred = mdl.predict(Xte)
            else:  # glm: Poisson for counts, Tweedie for metres
                if comp == "metres":
                    mdl = TweedieRegressor(power=1.3, alpha=1.0, max_iter=400)
                else:
                    mdl = PoissonRegressor(alpha=1.0, max_iter=400)
                mdl.fit(Xtr, np.clip(y, 0, None))
                pred = mdl.predict(Xte)
        out[comp] = np.clip(pred, 0.0, None)
    return out


# ---------------------------------------------------------------------------
# B3 direct-points cross-check (NOT deployable)
# ---------------------------------------------------------------------------
def direct_points_oof(df: pd.DataFrame, season: int, mode: str) -> pd.Series:
    """GroupKFold OOF Ridge predicting modern `official_pts` within `season`.

    There is no prior modern-labelled season for the 2025 backtest, so this
    honest in-season cross-check uses fixture-grouped OOF.  Cross-check only.
    """
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


# ---------------------------------------------------------------------------
# selfcheck (Phase 3 verification — recon-only; latent added in Phase 5)
# ---------------------------------------------------------------------------
def _selfcheck() -> None:
    from scipy.stats import spearmanr

    from model.data import load
    from model.splits import component_train

    df = load()
    test_season, mode = 2025, "post_team_sheet"
    train_mask = component_train(df, test_season)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where((df["season"] == test_season).to_numpy())[0]
    mh = expected_minutes(df, train_idx, test_idx, mode)
    print(f"minutes_hat: mean={mh.mean():.1f} range=[{mh.min():.0f},{mh.max():.0f}]")

    lab = df.iloc[test_idx]
    lab_mask = (lab["is_modern"] & lab["has_label"]).to_numpy()
    off = lab["official_pts"].to_numpy(float)
    isf = lab["is_forward"].to_numpy()

    print(f"\n{'engine':8s} {'pts_MAE(recon)':>14s} {'metres_MAE/80':>14s} {'spearman':>9s}")
    for engine in ("naive", "ridge", "glm"):
        rates = predict_rates(df, train_idx, test_idx, mode, engine)
        comp = rates_to_components(rates, mh)
        recon = score_recon(comp, isf)
        mae = float(np.abs(off[lab_mask] - recon[lab_mask]).mean())
        # component sanity: metres per-80 MAE vs realised on labelled, minutes>=10
        m10 = lab_mask & (lab["minutes"].to_numpy() >= MIN_MINUTES)
        rm = _rate_target(lab, "metres")
        met_mae = float(np.abs(rm[m10] - rates["metres"].to_numpy()[m10]).mean())
        rho = spearmanr(recon[lab_mask], off[lab_mask]).correlation
        print(f"{engine:8s} {mae:14.3f} {met_mae:14.2f} {rho:9.3f}")
        assert (rates.to_numpy() >= 0).all(), f"{engine}: negative rate"
        assert 0 <= rates["metres"].mean() <= 150

    b3 = direct_points_oof(df, test_season, mode)
    b3_mae = float(np.abs(df.loc[b3.index, "official_pts"] - b3).mean())
    print(f"{'B3direct':8s} {b3_mae:14.3f} {'(OOF, cross-check only)':>24s}")

    # kicking concentrates on kickers (naive engine): non-kickers ~0
    rates = predict_rates(df, train_idx, test_idx, "post_team_sheet", "naive")
    nonkick = lab["role_goal_kicker_rate"].fillna(0).to_numpy() < 0.05
    print(f"\nnaive kicking on non-kickers (pen+conv per80) mean="
          f"{(rates['penalty_goals']+rates['conversion_goals']).to_numpy()[nonkick].mean():.3f}")
    print("Phase 3 baselines OK")


if __name__ == "__main__":
    _selfcheck()
