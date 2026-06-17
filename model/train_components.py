#!/usr/bin/env python3
"""model/train_components.py  —  component model + minutes (Phase 4).

Per-component objective map, baseline-gated XGBoost, and a component registry.

Selection rule (no test peeking): for each SCORED component we measure
GroupKFold(5) OOF per-80 MAE **on the component-training years only** (2023+2024
for the 2025 backtest).  XGBoost / blend are promoted only when they strictly
beat the best of {naive, ridge, glm}; ties prefer the simpler engine.  The same
selected ensemble is then used for both the 2025 backtest and the 2026
deployment — 2025 is never used to choose components.

Per-component families (COMP_KIND):
  rare  (drop_goals_converted, red_cards) -> zero_prior
  card  (yellow_cards)                     -> position/discipline base rate (naive)
  kick  (conversion_goals, penalty_goals)  -> naive form, gated to likely kickers
  count (tries, try_assists, defenders_beaten, offload, tackle_turnover,
         penalties_conceded)               -> Poisson family + gated XGBoost
  continuous (metres)                       -> Tweedie + gated XGBoost
  rate  (tackles)                           -> squared-error + gated XGBoost
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from model import baselines as B
from model.baselines import (
    MIN_MINUTES,
    SCORED,
    _rate_target,
    predict_rates,
)
from model.data import CATEGORICAL_COLS, DATA, feature_view
from model.splits import group_kfold_indices

RNG = 0
# XGBoost/blend must beat the best baseline by at least this RELATIVE margin to
# be promoted. Small data (30 train fixtures): sub-margin OOF wins are noise and
# the plan's whole stance is "simplest model that beats naive" -> demote them.
XGB_MARGIN = 0.02
# convex blend weight on the XGBoost leg of the "blend" candidate (1-w on glm).
# Default 0.5 reproduces the frozen behaviour; tuned by model.research.
BLEND_WEIGHT = 0.5
# components for which an XGBoost candidate is even considered
XGB_COMPS = {
    "tackles", "metres", "tries", "try_assists", "defenders_beaten",
    "offload", "tackle_turnover", "penalties_conceded",
}
# always-zero / always-prior components (never fit a flexible model)
ZERO_PRIOR = {"drop_goals_converted", "red_cards"}
KICK_COMPS = {"conversion_goals", "penalty_goals"}

# conservative, heavily-regularized fixed config (small-data: 30 train fixtures)
_XGB_BASE = dict(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_child_weight=10.0, reg_lambda=5.0, colsample_bytree=0.8,
    subsample=0.9, random_state=RNG, n_jobs=1, verbosity=0,
    tree_method="hist", max_bin=64,
)


class _ConstantRateModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _xgb_objective(comp: str) -> dict:
    if comp == "metres":
        return dict(objective="reg:tweedie", tweedie_variance_power=1.3)
    if comp == "tackles":
        return dict(objective="reg:squarederror")
    return dict(objective="count:poisson", max_delta_step=1.0)  # counts


def _xgb_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """XGBoost design frame: raw feature view with categorical codes.

    Missing category values remain NaN. No imputation, no scaling, no fillna(0).
    """
    cols = feature_view(df, mode)
    X = df[cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            cat = X[c] if isinstance(X[c].dtype, pd.CategoricalDtype) else X[c].astype("category")
            X[c] = cat.cat.codes.replace(-1, np.nan).astype(float)
    return X


def _fit_xgb_component(
    df: pd.DataFrame, tr_idx: np.ndarray, va_idx: np.ndarray | None,
    comp: str, mode: str,
):
    """Fit one XGBoost rate model; early-stops on va_idx if provided."""
    keep = (df.iloc[tr_idx]["minutes"] >= MIN_MINUTES).to_numpy()
    use = tr_idx[keep]
    X = _xgb_frame(df, mode)
    if len(use) == 0:
        return _ConstantRateModel(0.0)
    y = np.clip(np.nan_to_num(_rate_target(df.iloc[use], comp), nan=0.0,
                              posinf=0.0, neginf=0.0), 0.0, None)
    if np.allclose(y, 0.0):
        return _ConstantRateModel(0.0)
    params = {**_XGB_BASE, **_xgb_objective(comp)}
    fit_kw = {}
    if va_idx is not None:
        vkeep = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[vkeep]
        if len(vu):
            yv = np.clip(np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0,
                                       posinf=0.0, neginf=0.0), 0.0, None)
            params["early_stopping_rounds"] = 50
            fit_kw = dict(eval_set=[(X.iloc[vu], yv)], verbose=False)
    model = xgb.XGBRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.iloc[use], y, **fit_kw)
    return model


def _xgb_predict(model, df: pd.DataFrame, idx: np.ndarray, mode: str) -> np.ndarray:
    X = _xgb_frame(df, mode)
    return np.clip(model.predict(X.iloc[idx]), 0.0, None)


def _kick_gate(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """1.0 for plausible goal-kickers, else 0.0 (concentrates kicking mass)."""
    r = df.iloc[idx]["role_goal_kicker_rate"].fillna(0.0).to_numpy(float)
    return (r > 0.0).astype(float)


# ---------------------------------------------------------------------------
# OOF candidate scoring on the train years (no test peeking)
# ---------------------------------------------------------------------------
def _oof_candidate_mae(
    df: pd.DataFrame, train_mask: np.ndarray, mode: str
) -> dict[str, dict[str, float]]:
    """Pooled GroupKFold OOF per-80 MAE for each (component, engine) candidate."""
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)

    # accumulate abs errors per (comp, engine)
    err: dict[str, dict[str, list]] = {c: {} for c in SCORED}

    def add(comp, engine, y_true, y_pred):
        err[comp].setdefault(engine, []).append(np.abs(y_true - y_pred))

    for tr_local, va_local in group_kfold_indices(dft, 5):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        vmask = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[vmask]
        if len(vu) == 0:
            continue
        # baseline engines (full 13-col frames, sliced per comp)
        frames = {
            e: predict_rates(df, tr_idx, vu, mode, e)
            for e in ("naive", "ridge", "glm")
        }
        # xgb per eligible component
        for comp in SCORED:
            ytrue = _rate_target(df.iloc[vu], comp)
            ytrue = np.nan_to_num(ytrue, nan=0.0, posinf=0.0, neginf=0.0)
            for e in ("naive", "ridge", "glm"):
                add(comp, e, ytrue, frames[e][comp].to_numpy(float))
            if comp in XGB_COMPS:
                m = _fit_xgb_component(df, tr_idx, va_idx, comp, mode)
                yx = _xgb_predict(m, df, vu, mode)
                add(comp, "xgb", ytrue, yx)
                # blend candidate: w*xgb + (1-w)*glm
                yb = BLEND_WEIGHT * yx + (1.0 - BLEND_WEIGHT) * frames["glm"][comp].to_numpy(float)
                add(comp, "blend", ytrue, yb)

    out: dict[str, dict[str, float]] = {}
    for comp in SCORED:
        out[comp] = {e: float(np.concatenate(v).mean()) for e, v in err[comp].items()}
    return out


def select_components(
    df: pd.DataFrame, train_mask: np.ndarray, mode: str
) -> pd.DataFrame:
    """Build the per-component registry via OOF MAE with baseline-gated XGBoost."""
    maes = _oof_candidate_mae(df, train_mask, mode)
    rows = []
    for comp in SCORED:
        kind = B.COMP_KIND[comp]
        m = maes[comp]
        if comp in ZERO_PRIOR:
            choice, reason = "zero_prior", "100%/99% zero -> constant 0"
        elif comp in KICK_COMPS:
            choice, reason = "naive", "kicker-gated form (assignment problem)"
        elif comp == "yellow_cards":
            choice, reason = "naive", "position/discipline base rate"
        else:
            simple = {e: m[e] for e in ("naive", "ridge", "glm") if e in m}
            best_simple = min(simple, key=simple.get)
            best_simple_mae = simple[best_simple]
            choice, reason = best_simple, f"best baseline (OOF {best_simple_mae:.3f})"
            # gate: promote xgb/blend only if it beats the best baseline by a
            # meaningful relative margin (epsilon wins on 30 fixtures are noise).
            threshold = best_simple_mae * (1.0 - XGB_MARGIN)
            best_adv, best_adv_mae = None, threshold
            for adv in ("xgb", "blend"):
                if adv in m and m[adv] < best_adv_mae:
                    best_adv, best_adv_mae = adv, m[adv]
            if best_adv is not None:
                gain = 100.0 * (1.0 - m[best_adv] / best_simple_mae)
                choice = best_adv
                reason = (f"{best_adv} beats {best_simple} OOF by {gain:.1f}% "
                          f"({m[best_adv]:.3f} < {best_simple_mae:.3f})")
        rows.append(dict(
            component=comp, kind=kind, engine=choice,
            oof_naive=m.get("naive", np.nan), oof_ridge=m.get("ridge", np.nan),
            oof_glm=m.get("glm", np.nan), oof_xgb=m.get("xgb", np.nan),
            oof_blend=m.get("blend", np.nan), reason=reason,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# prediction with selected engines (retrained on full train years)
# ---------------------------------------------------------------------------
def predict_rates_registry(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    mode: str, registry: pd.DataFrame,
) -> pd.DataFrame:
    """Per-80 rate predictions for the chosen-engine ensemble."""
    eng = dict(zip(registry["component"], registry["engine"]))
    need_base = {e for e in eng.values() if e in ("naive", "ridge", "glm")}
    need_base |= {"glm"} if "blend" in eng.values() else set()
    need_base |= {"naive"}  # kicking fallback always available
    base = {e: predict_rates(df, train_idx, test_idx, mode, e) for e in need_base}

    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    gate = _kick_gate(df, test_idx)
    for comp in SCORED:
        e = eng[comp]
        if e == "zero_prior":
            out[comp] = 0.0
        elif e in base:
            out[comp] = base[e][comp].to_numpy(float)
        elif e == "xgb":
            m = _fit_xgb_component(df, train_idx, None, comp, mode)
            out[comp] = _xgb_predict(m, df, test_idx, mode)
        elif e == "blend":
            m = _fit_xgb_component(df, train_idx, None, comp, mode)
            yx = _xgb_predict(m, df, test_idx, mode)
            out[comp] = BLEND_WEIGHT * yx + (1.0 - BLEND_WEIGHT) * base["glm"][comp].to_numpy(float)
        # gate kicking to plausible kickers
        if comp in KICK_COMPS:
            out[comp] = out[comp].to_numpy(float) * gate
    return out.clip(lower=0.0)


def predict_rates_xgb_only(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> pd.DataFrame:
    """Every trainable component via XGBoost (rare/degenerate -> zero/naive).
    Used for the XGBoost-only comparison column."""
    naive = predict_rates(df, train_idx, test_idx, mode, "naive")
    te = df.iloc[test_idx]
    out = pd.DataFrame(index=te.index, columns=SCORED, dtype=float)
    gate = _kick_gate(df, test_idx)
    for comp in SCORED:
        if comp in ZERO_PRIOR:
            out[comp] = 0.0
        elif comp in XGB_COMPS:
            m = _fit_xgb_component(df, train_idx, None, comp, mode)
            out[comp] = _xgb_predict(m, df, test_idx, mode)
        else:                                  # cards/kick: naive prior
            out[comp] = naive[comp].to_numpy(float)
        if comp in KICK_COMPS:
            out[comp] = out[comp].to_numpy(float) * gate
    return out.clip(lower=0.0)


def save_registry(registry: pd.DataFrame, path=None) -> None:
    path = path or (DATA / "model_component_registry.csv")
    registry.to_csv(path, index=False)


def _selfcheck() -> None:
    from model.baselines import expected_minutes, rates_to_components, score_recon
    from model.data import load
    from model.splits import component_train

    df = load()
    mode = "post_team_sheet"
    train_mask = component_train(df, 2025)
    reg = select_components(df, train_mask, mode)
    pd.set_option("display.width", 160)
    print(reg[["component", "kind", "engine", "oof_naive", "oof_ridge",
               "oof_glm", "oof_xgb", "reason"]].to_string(index=False))
    save_registry(reg)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where((df["season"] == 2025).to_numpy())[0]
    mh = expected_minutes(df, train_idx, test_idx, mode)
    rates = predict_rates_registry(df, train_idx, test_idx, mode, reg)
    comp = rates_to_components(rates, mh)
    lab = df.iloc[test_idx]
    lab_mask = (lab["is_modern"] & lab["has_label"]).to_numpy()
    recon = score_recon(comp, lab["is_forward"].to_numpy())
    off = lab["official_pts"].to_numpy(float)
    print(f"\nensemble recon-only pts MAE (2025) = "
          f"{np.abs(off[lab_mask]-recon[lab_mask]).mean():.3f}")
    # kicking concentrates on kickers: rows gated to 0 (rate==0/NaN) must be exactly 0
    gate0 = _kick_gate(df, test_idx) == 0.0
    kmass_gated = (rates['penalty_goals'] + rates['conversion_goals']).to_numpy()[gate0]
    occ = lab["role_goal_kicker_rate"].fillna(0).to_numpy() < 0.05
    kmass_occ = (rates['penalty_goals'] + rates['conversion_goals']).to_numpy()[occ].mean()
    print(f"kicking mass on hard non-kickers = {kmass_gated.max():.6f} (must be 0); "
          f"on rate<0.05 = {kmass_occ:.4f}")
    assert kmass_gated.max() == 0.0
    assert (rates.to_numpy() >= 0).all()
    assert 0 <= rates["metres"].mean() <= 150
    print("Phase 4 components OK")


if __name__ == "__main__":
    _selfcheck()
