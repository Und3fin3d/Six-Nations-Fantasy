from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.sixn import baselines as B
from model.sixn.baselines import MIN_MINUTES, SCORED, _rate_target, predict_rates
from model.sixn.data import feature_view
from model.sixn.splits import chronological_indices

LGBM_MARGIN = 0.02
BLEND_WEIGHT = 0.5
LGBM_COMPS = {"tackles", "metres", "tries", "try_assists", "defenders_beaten", "offload", "tackle_turnover", "penalties_conceded"}
ZERO_PRIOR = {"drop_goals_converted", "red_cards"}
KICK_COMPS = {"conversion_goals", "penalty_goals"}
LGBM_ONLY = {c: "zero_prior" if c in ZERO_PRIOR else "lgbm" if c in LGBM_COMPS else "naive" for c in SCORED}

_LGBM_BASE = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=50, reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
    random_state=0, n_jobs=1, verbosity=-1,
)


def _lgbm_objective(comp: str) -> dict:
    if comp == "metres":
        return dict(objective="tweedie", tweedie_variance_power=1.3)
    if comp == "tackles":
        return dict(objective="regression_l1")
    return dict(objective="poisson")


def _lgbm_frame(df: pd.DataFrame, mode: str, teamplay: str = "off") -> pd.DataFrame:
    return df[feature_view(df, mode, teamplay)].copy()


def _fit_lgbm_component(df: pd.DataFrame, X: pd.DataFrame, tr_idx: np.ndarray, comp: str):
    use = tr_idx[(df.iloc[tr_idx]["minutes"] >= MIN_MINUTES).to_numpy()]
    y = np.clip(np.nan_to_num(_rate_target(df.iloc[use], comp), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    model = lgb.LGBMRegressor(**_LGBM_BASE, **_lgbm_objective(comp))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.iloc[use], y)
    return model


def _lgbm_predict(model, X: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(X.iloc[idx]), 0.0, None)


def _kick_gate(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    return (df.iloc[idx]["role_goal_kicker_rate"].fillna(0.0).to_numpy(float) > 0.0).astype(float)


def _oof_candidate_mae(df: pd.DataFrame, train_mask: np.ndarray, mode: str) -> dict[str, dict[str, float]]:
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)

    err: dict[str, dict[str, list]] = {c: {} for c in SCORED}
    X = _lgbm_frame(df, mode)

    def add(comp, engine, y_true, y_pred):
        err[comp].setdefault(engine, []).append(np.abs(y_true - y_pred))

    for tr_local, va_local in chronological_indices(dft, 5):
        tr_idx, va_idx = train_pos[tr_local], train_pos[va_local]
        vu = va_idx[(df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()]
        frames = {e: predict_rates(df, tr_idx, vu, mode, e) for e in ("naive", "ridge", "glm")}
        for comp in SCORED:
            ytrue = np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0, posinf=0.0, neginf=0.0)
            for e in ("naive", "ridge", "glm"):
                add(comp, e, ytrue, frames[e][comp].to_numpy(float))
            if comp in LGBM_COMPS:
                yl = _lgbm_predict(_fit_lgbm_component(df, X, tr_idx, comp), X, vu)
                add(comp, "lgbm", ytrue, yl)
                add(comp, "blend", ytrue, BLEND_WEIGHT * yl + (1.0 - BLEND_WEIGHT) * frames["glm"][comp].to_numpy(float))
    return {comp: {e: float(np.concatenate(v).mean()) for e, v in err[comp].items()} for comp in SCORED}


def select_components(df: pd.DataFrame, train_mask: np.ndarray, mode: str) -> pd.DataFrame:
    maes = _oof_candidate_mae(df, train_mask, mode)
    rows = []
    for comp in SCORED:
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
            threshold = best_simple_mae * (1.0 - LGBM_MARGIN)
            best_adv = min((adv for adv in ("lgbm", "blend") if adv in m and m[adv] < threshold), key=m.get, default=None)
            if best_adv is not None:
                gain = 100.0 * (1.0 - m[best_adv] / best_simple_mae)
                choice = best_adv
                reason = f"{best_adv} beats {best_simple} OOF by {gain:.1f}% ({m[best_adv]:.3f} < {best_simple_mae:.3f})"
        rows.append(
            dict(
                component=comp, kind=B.COMP_KIND[comp], engine=choice, **{f"oof_{e}": m.get(e, np.nan) for e in ("naive", "ridge", "glm", "lgbm", "blend")},
                reason=reason,
            )
        )
    return pd.DataFrame(rows)


def lgbm_model(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, teamplay: str = "off"):
    X = _lgbm_frame(df, mode, teamplay)
    return lambda comp: _lgbm_predict(_fit_lgbm_component(df, X, train_idx, comp), X, test_idx)


def predict_rates_registry(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, eng: dict, model=None) -> pd.DataFrame:
    need_base = {e for e in eng.values() if e in ("naive", "ridge", "glm")} | ({"glm"} if "blend" in eng.values() else set()) | {"naive"}
    base = {e: predict_rates(df, train_idx, test_idx, mode, e) for e in need_base}
    model = model or lgbm_model(df, train_idx, test_idx, mode)
    out = pd.DataFrame(index=df.iloc[test_idx].index, columns=SCORED, dtype=float)
    gate = _kick_gate(df, test_idx)
    for comp in SCORED:
        e = eng[comp]
        if e == "zero_prior":
            out[comp] = 0.0
        elif e in base:
            out[comp] = base[e][comp].to_numpy(float)
        else:
            yl = model(comp)
            out[comp] = yl if e == "lgbm" else BLEND_WEIGHT * yl + (1.0 - BLEND_WEIGHT) * base["glm"][comp].to_numpy(float)
        if comp in KICK_COMPS:
            out[comp] = out[comp].to_numpy(float) * gate
    return out.clip(lower=0.0)


def predict_rates_lgbm_only(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str, teamplay: str = "off") -> pd.DataFrame:
    return predict_rates_registry(df, train_idx, test_idx, mode, LGBM_ONLY, lgbm_model(df, train_idx, test_idx, mode, teamplay))
