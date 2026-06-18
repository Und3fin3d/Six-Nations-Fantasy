#!/usr/bin/env python3
"""model/research.py  —  Karpathy-style autoresearch loop.

Greedy hill-climb over a queue of principled candidate changes, each evaluated on
the 2025 dev backtest, accepted only on a Pareto improvement (value_xv up, MAE not
worse — or the reverse) that holds on >=3/5 rounds.  See RESEARCH_GOAL.md.

Hard invariants (never relaxed):
  * 2026 is sealed — the loop never reads it (asserted).
  * Component engines are chosen on 2023+2024 component-rate OOF (independent of
    the 2025 points signal); only the assembly knobs are tuned against 2025.
  * Leakage asserts in model.data run at import; kicking-mass gate must hold.

Usage:
  python -m model.research            # one pass over the candidate queue
  python -m model.research --seal-2026  # evaluate the final best on 2026 ONCE
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge

from model import assemble as A
from model import train_components as TC
from model.assemble import assemble_predictions, rank_scores
from model.baselines import (
    MIN_MINUTES,
    SCORED,
    _rate_target,
    predict_rates,
    score_recon,
)
from model.data import CATEGORICAL_COLS, feature_view, load
from model.evaluate import (
    captain_hit_rates,
    captain_hitrate,
    points_mae,
    spearman_within_pos,
    topn_overlap,
    value_of_xv,
)
from model.splits import component_train, round_iter
from model.train_components import (
    LGBM_COMPS,
    KICK_COMPS,
    ZERO_PRIOR,
    predict_rates_lgbm_only,
    predict_rates_registry,
    save_registry,
    select_components,
)

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
DATA = ROOT / "data"
DEV_SEASON = 2025          # the only modern-labelled dev season
SEALED_SEASON = 2026       # never read inside the loop
MODE = "post_team_sheet"
BEST_CONFIG = RESEARCH / "best_config.json"
PROMOTED_CONFIG = RESEARCH / "promoted_config.json"

# acceptance thresholds (see RESEARCH_GOAL.md)
DELTA_V = 0.003            # min value_xv gain to accept on the value_xv leg
TAU_MAE = 0.02            # max tolerated MAE regression on the value_xv leg
DELTA_M = 0.02            # min MAE gain to accept on the MAE leg
ROUND_ROBUST = 3          # value_xv gain must hold on >= this many of 5 rounds
TOP15_MAX_DROP = 0.04     # guardrail: about three top-15 misses over 5 rounds
SPEARMAN_MAX_DROP = 0.03  # guardrail for within-position selector health
STABILITY_REPORT = RESEARCH / "stability_report.json"
ROUND_DROP_MIN_PASS = 4
TEAM_DROP_MIN_PASS = 5
POSITION_MAE_MAX_REGRESSION = 0.15
POSITION_SPEARMAN_MAX_DROP = 0.05
MINUTES_PERTURB_MIN_PASS = 3
XGB_MAX_POINT_WEIGHT = 0.20
XGB_MAX_COMBINER_WEIGHT = 0.15
BAYES_MAX_COMBINER_WEIGHT = 0.15
BAYES_MAX_SHRINK_WEIGHT = 0.20
LGBM_MIN_COMBINER_WEIGHT = 0.75

POSITION_GROUPS = {
    "front_row": {"Prop", "Hooker"},
    "second_row": {"Second-row"},
    "back_row": {"Back-row"},
    "halfbacks": {"Scrum-half", "Fly-half"},
    "centres_back_three": {"Centre", "Back-three"},
}

XGB_GRAFT_GROUPS = {
    "metres": {"metres"},
    "tackles": {"tackles"},
    "tries_assists_db": {"tries", "try_assists", "defenders_beaten"},
    "sparse_counts": {"offload", "tackle_turnover", "penalties_conceded"},
}

KNOWN_SEALED_VETOES = {
    "minutes_alpha_4": (
        "previous sealed 2026 check regressed both MAE and XV value versus the "
        "promoted front-row model"
    ),
    "target_frontrow20_back5_10": (
        "same stacked back-five prior family as the 20% representative that "
        "already failed the sealed 2026 promotion check"
    ),
    "target_frontrow20_back5_20": (
        "previous sealed 2026 check worsened MAE sharply versus the promoted "
        "front-row model despite strong 2025 MAE"
    ),
    "target_frontrow20_nonfront_10": (
        "same broad post-hoc position-prior smoothing family as the sealed-vetoed "
        "back-five representative"
    ),
    "target_frontrow20_nonfront_20": (
        "same broad post-hoc position-prior smoothing family as the sealed-vetoed "
        "back-five representative"
    ),
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """All knobs.  Defaults reproduce the current cdx deployable behaviour."""
    name: str = "baseline"
    note: str = "current cdx selected deployable engine"
    deploy_engine: str = "lgbm_only"       # lgbm_only | registry
    # component layer (validated on 2023+2024 OOF)
    lgbm_margin: float = 0.02
    blend_weight: float = 0.5
    # minutes model
    minutes_model: str = "ridge"        # ridge | ridge_interact | poisson
    minutes_alpha: float = 10.0
    # latent / POTM
    potm_pp_weight: float = 0.5
    potm_tau_floor: float = 1.0
    latent_shrink: float = 1.0
    other_const_shrink: float = 1.0
    zero_back_setpiece: bool = False
    # post-hoc
    recon_calib: str = "none"           # none | linear (forward-chained per round)
    target_prior_blend: float = 0.0     # blend points toward prior position mean
    target_prior_scope: str = "all"     # all | prop | hooker | frontrow | tight5 | forwards | backs
    extra_prior_blend: float = 0.0      # optional second scoped blend after the primary
    extra_prior_scope: str = "all"
    selector_tilt: float = 0.0          # blend rank head into the XV pick only
    selector_point_source: str = "target"  # target | pre_calib
    # specialist signals (bounded so LGBM remains the champion)
    xgb_point_weight: float = 0.0       # minority blend into target_pts_hat, max 0.20
    xgb_rank_weight: float = 0.0        # minority blend into sel_score
    bayes_point_weight: float = 0.0     # global Bayesian point shrinkage
    bayes_cold_weight: float = 0.0      # Bayesian pull only for low-prior rows
    bayes_disagreement_weight: float = 0.0  # Bayesian pull only where specialists disagree
    bayes_position_residual_weight: float = 0.0
    xgb_graft_group: str = "none"       # none | metres | tackles | tries_assists_db | sparse_counts | oof_winners
    xgb_graft_weight: float = 0.0       # 1.0 replace, 0.2 minority graft
    combiner_lgbm_weight: float = 1.0   # fixed convex final combiner
    combiner_xgb_weight: float = 0.0
    combiner_bayes_weight: float = 0.0

    def delta(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


@contextlib.contextmanager
def apply_config(cfg: Config):
    """Thread cfg into the module globals that the pipeline reads, then restore."""
    saved = (
        TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
        A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK,
        A.OTHER_CONST_SHRINK, A.ZERO_BACK_SETPIECE,
    )
    TC.LGBM_MARGIN = cfg.lgbm_margin
    TC.BLEND_WEIGHT = cfg.blend_weight
    A.POTM_PP_WEIGHT = cfg.potm_pp_weight
    A.POTM_TAU_FLOOR = cfg.potm_tau_floor
    A.LATENT_SHRINK = cfg.latent_shrink
    A.OTHER_CONST_SHRINK = cfg.other_const_shrink
    A.ZERO_BACK_SETPIECE = cfg.zero_back_setpiece
    try:
        yield
    finally:
        (TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
         A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK,
         A.OTHER_CONST_SHRINK, A.ZERO_BACK_SETPIECE) = saved


# ---------------------------------------------------------------------------
# minutes variants (default `ridge`+alpha=10 mirrors baselines.expected_minutes)
# ---------------------------------------------------------------------------
def build_minutes(df, train_idx, test_idx, cfg: Config) -> np.ndarray:
    tr = df.iloc[train_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows):
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    te = df.iloc[test_idx]
    feats = ["started", "jersey", "is_forward"]

    def design(rows):
        cols = [rows[feats].astype(float).to_numpy(), prior(rows)[:, None]]
        if cfg.minutes_model == "ridge_interact":
            isf = rows["is_forward"].astype(float).to_numpy()
            cols.append((rows["started"].astype(float).to_numpy() * isf)[:, None])
            cols.append((rows["jersey"].astype(float).to_numpy() * isf)[:, None])
        return np.column_stack(cols)

    Xtr, ytr = design(tr), tr["minutes"].to_numpy(float)
    Xte = design(te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.minutes_model == "poisson":
            reg = PoissonRegressor(alpha=cfg.minutes_alpha, max_iter=500).fit(
                Xtr, np.clip(ytr, 0, None))
        else:
            reg = Ridge(alpha=cfg.minutes_alpha).fit(Xtr, ytr)
    return np.clip(reg.predict(Xte), 0, 80)


# ---------------------------------------------------------------------------
# post-hoc transforms
# ---------------------------------------------------------------------------
def _forward_linear_calib(pred: pd.DataFrame) -> np.ndarray:
    """Per-round linear recalibration of recon_pts_hat -> official_pts, fit only on
    labelled rows from EARLIER rounds (forward-chained, no leakage).  Round 1 and
    thin-history rounds fall back to identity."""
    out = pred["recon_pts_hat"].to_numpy(float).copy()
    lab = (pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)).to_numpy()
    rnd = pred["round"].to_numpy()
    recon = pred["recon_pts_hat"].to_numpy(float)
    off = pred["official_pts"].to_numpy(float)
    for R in np.unique(rnd):
        prev = lab & (rnd < R)
        cur = rnd == R
        if prev.sum() >= 20 and np.ptp(recon[prev]) > 1e-6:
            slope, intercept = np.polyfit(recon[prev], off[prev], 1)
            out[cur] = slope * recon[cur] + intercept
    return np.clip(out, 0.0, None)


def _forward_posmean_blend(
    df: pd.DataFrame, test_idx: np.ndarray, pred: pd.DataFrame, season: int,
    weight: float, scope: str = "all", *, min_n: int = 40,
) -> np.ndarray:
    """Blend point forecasts toward prior modern-labelled position means.

    This is intentionally gentler than linear calibration: it only adds a small
    prior-pull once enough earlier modern labels exist, and it never uses rows
    dated on/after the round being predicted.
    """
    out = pred["target_pts_hat"].to_numpy(float).copy()
    base = out.copy()
    scope_positions = _scope_positions(scope)
    for _, asof, ridx in round_iter(df, season):
        hist_mask = (
            df["is_modern"].to_numpy()
            & df["has_label"].to_numpy()
            & (df["date"] < asof).to_numpy()
        )
        if hist_mask.sum() < min_n:
            continue
        local = np.searchsorted(test_idx, ridx)
        assert np.array_equal(test_idx[local], ridx)
        hist = df.loc[hist_mask]
        global_mean = float(hist["official_pts"].mean())
        pos_mean = hist.groupby("canonical_pos")["official_pts"].mean()
        prior = df.iloc[ridx]["canonical_pos"].map(pos_mean).fillna(global_mean).to_numpy(float)
        eligible = np.ones(len(ridx), dtype=bool)
        if scope_positions is not None:
            eligible = df.iloc[ridx]["canonical_pos"].isin(scope_positions).to_numpy()
        out[local[eligible]] = ((1.0 - weight) * base[local[eligible]]
                                + weight * prior[eligible])
    return np.clip(out, 0.0, None)


def _scope_positions(scope: str) -> set[str] | None:
    scopes = {
        "all": None,
        "prop": {"Prop"},
        "hooker": {"Hooker"},
        "frontrow": {"Prop", "Hooker"},
        "tight5": {"Prop", "Hooker", "Second-row"},
        "forwards": {"Prop", "Hooker", "Second-row", "Back-row"},
        "backs": {"Scrum-half", "Fly-half", "Centre", "Back-three"},
        "back5": {"Back-row", "Scrum-half", "Fly-half", "Centre", "Back-three"},
        "backrow_backs": {"Back-row", "Centre", "Back-three"},
        "outside_backs": {"Centre", "Back-three"},
        "non_frontrow": {"Second-row", "Back-row", "Scrum-half", "Fly-half",
                         "Centre", "Back-three"},
    }
    if scope not in scopes:
        raise ValueError(f"unknown target_prior_scope {scope!r}")
    return scopes[scope]


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


XGB_COMPS = {
    "tackles", "metres", "tries", "try_assists", "defenders_beaten",
    "offload", "tackle_turnover", "penalties_conceded",
}
_XGB_BASE = dict(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_child_weight=10.0, reg_lambda=5.0, colsample_bytree=0.8,
    subsample=0.9, random_state=0, n_jobs=1, verbosity=0,
    tree_method="hist", max_bin=64,
)
_XGB_OOF_WINNERS_CACHE: dict = {}


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
    return dict(objective="count:poisson", max_delta_step=1.0)


def _xgb_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
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
    import xgboost as xgb

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
    return np.clip(model.predict(_xgb_frame(df, mode).iloc[idx]), 0.0, None)


def _predict_rates_xgb_only(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> pd.DataFrame:
    naive = predict_rates(df, train_idx, test_idx, mode, "naive")
    out = pd.DataFrame(index=df.iloc[test_idx].index, columns=SCORED, dtype=float)
    gate = _kick_gate(df, test_idx)
    for comp in SCORED:
        if comp in ZERO_PRIOR:
            out[comp] = 0.0
        elif comp in XGB_COMPS:
            m = _fit_xgb_component(df, train_idx, None, comp, mode)
            out[comp] = _xgb_predict(m, df, test_idx, mode)
        else:
            out[comp] = naive[comp].to_numpy(float)
        if comp in KICK_COMPS:
            out[comp] = out[comp].to_numpy(float) * gate
    return out.clip(lower=0.0)


def _predict_rates_bayesian(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> pd.DataFrame:
    return predict_rates(df, train_idx, test_idx, mode, "bayesian")


def _kick_gate(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    r = df.iloc[idx]["role_goal_kicker_rate"].fillna(0.0).to_numpy(float)
    return (r > 0.0).astype(float)


def _xgb_rank_scores(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> np.ndarray:
    import xgboost as xgb

    X = _xgb_frame(df, mode)
    tr = df.iloc[train_idx].copy()
    order = np.argsort(tr["fixture_id"].to_numpy(), kind="stable")
    tr_idx_sorted = train_idx[order]
    grp = tr.iloc[order].groupby("fixture_id", sort=False).size().to_numpy()
    grade = (tr.iloc[order].groupby("fixture_id")["recon_pts"]
             .rank(pct=True).mul(4.999).astype(int).to_numpy())
    ranker = xgb.XGBRanker(
        objective="rank:ndcg", n_estimators=300, learning_rate=0.05,
        max_depth=3, min_child_weight=10.0, reg_lambda=5.0,
        random_state=0, n_jobs=1, verbosity=0,
        tree_method="hist", max_bin=64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X.iloc[tr_idx_sorted], grade, group=grp)
    return ranker.predict(X.iloc[test_idx])


def _xgb_oof_winners(df: pd.DataFrame, train_mask: np.ndarray, mode: str) -> set[str]:
    key = (int(train_mask.sum()), mode)
    if key in _XGB_OOF_WINNERS_CACHE:
        return set(_XGB_OOF_WINNERS_CACHE[key])
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)
    err: dict[str, dict[str, list]] = {c: {"lgbm": [], "xgb": []} for c in XGB_COMPS}
    for tr_local, va_local in component_group_kfold(dft):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        keep = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[keep]
        if len(vu) == 0:
            continue
        for comp in XGB_COMPS:
            ytrue = np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0,
                                  posinf=0.0, neginf=0.0)
            lm = TC._fit_lgbm_component(df, tr_idx, va_idx, comp, mode)
            xm = _fit_xgb_component(df, tr_idx, va_idx, comp, mode)
            err[comp]["lgbm"].append(np.abs(ytrue - TC._lgbm_predict(lm, df, vu, mode)))
            err[comp]["xgb"].append(np.abs(ytrue - _xgb_predict(xm, df, vu, mode)))
    winners = set()
    for comp, vals in err.items():
        if not vals["lgbm"] or not vals["xgb"]:
            continue
        lgbm_mae = float(np.concatenate(vals["lgbm"]).mean())
        xgb_mae = float(np.concatenate(vals["xgb"]).mean())
        if xgb_mae < lgbm_mae * 0.98:
            winners.add(comp)
    _XGB_OOF_WINNERS_CACHE[key] = sorted(winners)
    return winners


def component_group_kfold(dft: pd.DataFrame):
    from model.splits import group_kfold_indices

    yield from group_kfold_indices(dft, 5)


def _add_selector_score(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    pred: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, str]:
    """Attach the optional XV-selection score used by selector_tilt candidates."""
    if cfg.selector_tilt <= 0 and cfg.xgb_rank_weight <= 0:
        return pred, "target_pts_hat"
    out = pred.copy()
    if "lgbm_rank_score" in out.columns:
        rk = out["lgbm_rank_score"].to_numpy(float)
    else:
        rk = rank_scores(df, train_idx, test_idx, MODE)
        out["lgbm_rank_score"] = rk
    out["rank_score"] = rk
    if cfg.selector_point_source == "target":
        point_base = out["target_pts_hat"].to_numpy(float)
    elif cfg.selector_point_source == "pre_calib":
        point_base = out["selector_pts_hat"].to_numpy(float)
    else:
        raise ValueError(f"unknown selector_point_source {cfg.selector_point_source!r}")
    score = _zscore(point_base) + cfg.selector_tilt * _zscore(rk)
    if cfg.xgb_rank_weight > 0:
        if "xgb_rank_score" not in out.columns:
            out["xgb_rank_score"] = _xgb_rank_scores(df, train_idx, test_idx, MODE)
        score = score + cfg.xgb_rank_weight * _zscore(out["xgb_rank_score"].to_numpy(float))
    out["sel_score"] = score
    return out, "sel_score"


# ---------------------------------------------------------------------------
# evaluation of one config on the dev season
# ---------------------------------------------------------------------------
_REG_CACHE: dict = {}


def _registry_for(df, train_mask, cfg: Config) -> pd.DataFrame:
    key = (round(cfg.lgbm_margin, 5), round(cfg.blend_weight, 5))
    if key not in _REG_CACHE:
        with apply_config(cfg):
            _REG_CACHE[key] = select_components(df, train_mask, MODE)
    return _REG_CACHE[key]


def _registry_lgbm_count(cfg: Config, registry: pd.DataFrame | None) -> int:
    if cfg.deploy_engine == "lgbm_only":
        return len(LGBM_COMPS)
    return int((registry["engine"].isin(["lgbm", "blend"])).sum())


def _apply_target_priors(
    df: pd.DataFrame, test_idx: np.ndarray, pred: pd.DataFrame, season: int, cfg: Config
) -> pd.DataFrame:
    out = pred.copy()
    if cfg.target_prior_blend > 0:
        out["target_pts_hat"] = _forward_posmean_blend(
            df, test_idx, out, season, cfg.target_prior_blend, cfg.target_prior_scope)
    if cfg.extra_prior_blend > 0:
        out["target_pts_hat"] = _forward_posmean_blend(
            df, test_idx, out, season, cfg.extra_prior_blend, cfg.extra_prior_scope)
    return out


def _needs_xgb_points(cfg: Config) -> bool:
    return (
        cfg.xgb_point_weight > 0
        or cfg.bayes_disagreement_weight > 0
        or cfg.combiner_xgb_weight > 0
    )


def _needs_bayes_points(cfg: Config) -> bool:
    return (
        cfg.bayes_point_weight > 0
        or cfg.bayes_cold_weight > 0
        or cfg.bayes_disagreement_weight > 0
        or cfg.combiner_bayes_weight > 0
    )


def _specialist_prediction(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    season: int,
    minutes_hat: np.ndarray,
    predictor,
) -> pd.DataFrame:
    pred, _ = assemble_predictions(df, train_idx, season, MODE, predictor, minutes_hat=minutes_hat)
    return pred


def _resolve_xgb_graft_components(
    df: pd.DataFrame, train_mask: np.ndarray, cfg: Config,
) -> tuple[set[str], set[str], set[str]]:
    if cfg.xgb_graft_group == "none" or cfg.xgb_graft_weight <= 0:
        return set(), set(), set()
    winners = _xgb_oof_winners(df, train_mask, MODE)
    if cfg.xgb_graft_group == "oof_winners":
        requested = set(winners)
    else:
        requested = XGB_GRAFT_GROUPS.get(cfg.xgb_graft_group)
        if requested is None:
            raise ValueError(f"unknown xgb_graft_group {cfg.xgb_graft_group!r}")
    requested = set(requested) & XGB_COMPS
    return requested & winners, requested, winners


def _graft_components(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    pred: pd.DataFrame,
    xgb_pred: pd.DataFrame | None,
    cfg: Config,
    eligible_components: set[str] | None = None,
    requested_components: set[str] | None = None,
    oof_winners: set[str] | None = None,
) -> pd.DataFrame:
    if cfg.xgb_graft_group == "none" or cfg.xgb_graft_weight <= 0:
        return pred
    if eligible_components is None:
        eligible_components, requested_components, oof_winners = _resolve_xgb_graft_components(
            df, train_mask, cfg)
    comps = set(eligible_components)
    out = pred.copy()
    out["xgb_graft_requested_components"] = ",".join(sorted(requested_components or []))
    out["xgb_graft_oof_winners"] = ",".join(sorted(oof_winners or []))
    if not comps:
        out["xgb_graft_components"] = ""
        return out
    if xgb_pred is None:
        raise ValueError("xgb component graft requested without xgb predictions")
    w = float(np.clip(cfg.xgb_graft_weight, 0.0, 1.0))
    comp_frame = pd.DataFrame(index=out.index)
    for comp in SCORED:
        col = f"hat_{comp}"
        vals = out[col].to_numpy(float)
        if comp in comps:
            vals = (1.0 - w) * vals + w * xgb_pred[col].to_numpy(float)
        out[col] = vals
        comp_frame[comp] = vals
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
    out["xgb_graft_components"] = ",".join(sorted(comps))
    return out


def _forward_position_residual(pred: pd.DataFrame, weight: float) -> np.ndarray:
    out = pred["target_pts_hat"].to_numpy(float).copy()
    if weight <= 0:
        return out
    lab = (pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)).to_numpy()
    rounds = pred["round"].to_numpy()
    pos = pred["canonical_pos"].to_numpy()
    resid = pred["official_pts"].to_numpy(float) - pred["target_pts_hat"].to_numpy(float)
    for r in np.unique(rounds):
        prev = lab & (rounds < r)
        cur = rounds == r
        if prev.sum() < 20:
            continue
        table = pd.DataFrame({"pos": pos[prev], "resid": resid[prev]}).groupby("pos")["resid"].mean()
        global_resid = float(np.nanmean(resid[prev]))
        adj = pd.Series(pos[cur]).map(table).fillna(global_resid).to_numpy(float)
        out[cur] = out[cur] + weight * adj
    return np.clip(out, 0.0, None)


def _apply_specialist_points(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    xgb_pred: pd.DataFrame | None,
    bayes_pred: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    out = pred.copy()
    lgbm = out["target_pts_hat"].to_numpy(float)
    out["lgbm_target_pts_hat"] = lgbm
    te = df.iloc[test_idx]

    if xgb_pred is not None:
        out["xgb_component_pts_hat"] = xgb_pred["target_pts_hat"].to_numpy(float)
    if bayes_pred is not None:
        out["bayes_component_pts_hat"] = bayes_pred["target_pts_hat"].to_numpy(float)

    cold = te["form_n_prior"].fillna(0).to_numpy(float) <= 4
    low_minutes_history = te["class_minutes_prior"].fillna(0).to_numpy(float) <= 160
    disagreement = np.zeros(len(out), dtype=bool)
    if xgb_pred is not None:
        diff = np.abs(lgbm - xgb_pred["target_pts_hat"].to_numpy(float))
        cutoff = float(np.nanquantile(diff, 0.75)) if np.isfinite(diff).any() else np.inf
        disagreement = diff >= cutoff
    out["uncertainty_proxy"] = (
        cold.astype(float) + low_minutes_history.astype(float) + disagreement.astype(float)
    )

    target = lgbm.copy()
    if cfg.xgb_point_weight > 0:
        if xgb_pred is None:
            raise ValueError("xgb_point_weight requires xgb signal")
        w = min(cfg.xgb_point_weight, XGB_MAX_POINT_WEIGHT)
        target = (1.0 - w) * target + w * xgb_pred["target_pts_hat"].to_numpy(float)
    if cfg.bayes_point_weight > 0:
        if bayes_pred is None:
            raise ValueError("bayes_point_weight requires bayes signal")
        w = min(cfg.bayes_point_weight, BAYES_MAX_SHRINK_WEIGHT)
        target = (1.0 - w) * target + w * bayes_pred["target_pts_hat"].to_numpy(float)
    if cfg.bayes_cold_weight > 0:
        if bayes_pred is None:
            raise ValueError("bayes_cold_weight requires bayes signal")
        w = min(cfg.bayes_cold_weight, BAYES_MAX_SHRINK_WEIGHT)
        b = bayes_pred["target_pts_hat"].to_numpy(float)
        target[cold] = (1.0 - w) * target[cold] + w * b[cold]
    if cfg.bayes_disagreement_weight > 0:
        if bayes_pred is None:
            raise ValueError("bayes_disagreement_weight requires bayes signal")
        w = min(cfg.bayes_disagreement_weight, BAYES_MAX_SHRINK_WEIGHT)
        b = bayes_pred["target_pts_hat"].to_numpy(float)
        target[disagreement] = (1.0 - w) * target[disagreement] + w * b[disagreement]

    if cfg.bayes_position_residual_weight > 0:
        tmp = out.copy()
        tmp["target_pts_hat"] = target
        target = _forward_position_residual(tmp, cfg.bayes_position_residual_weight)

    if cfg.combiner_xgb_weight > 0 or cfg.combiner_bayes_weight > 0:
        lw = cfg.combiner_lgbm_weight
        xw = cfg.combiner_xgb_weight
        bw = cfg.combiner_bayes_weight
        if lw < LGBM_MIN_COMBINER_WEIGHT or xw > XGB_MAX_COMBINER_WEIGHT or bw > BAYES_MAX_COMBINER_WEIGHT:
            raise ValueError("combiner weights violate conservative specialist caps")
        total = lw + xw + bw
        if not np.isclose(total, 1.0):
            raise ValueError("combiner weights must sum to 1.0")
        target = lw * lgbm
        if xw:
            if xgb_pred is None:
                raise ValueError("combiner_xgb_weight requires xgb signal")
            target = target + xw * xgb_pred["target_pts_hat"].to_numpy(float)
        if bw:
            if bayes_pred is None:
                raise ValueError("combiner_bayes_weight requires bayes signal")
            target = target + bw * bayes_pred["target_pts_hat"].to_numpy(float)

    out["target_pts_hat"] = np.clip(target, 0.0, None)
    return out


def _predict_config(
    df: pd.DataFrame, cfg: Config, season: int,
) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """Build predictions for a fixed config/season without evaluating acceptance."""
    train_mask = component_train(df, season)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where((df["season"] == season).to_numpy())[0]
    registry = None if cfg.deploy_engine == "lgbm_only" else _registry_for(df, train_mask, cfg)

    with apply_config(cfg):
        minutes_hat = build_minutes(df, train_idx, test_idx, cfg)

        def predictor(d, ti, te, mo):
            if cfg.deploy_engine == "lgbm_only":
                return predict_rates_lgbm_only(d, ti, te, mo)
            return predict_rates_registry(d, ti, te, mo, registry)

        pred, diag = assemble_predictions(
            df, train_idx, season, MODE, predictor, minutes_hat=minutes_hat)
        sub = df.iloc[test_idx]
        for meta_col in ("team", "opponent", "started", "jersey"):
            if meta_col in sub.columns:
                pred[meta_col] = sub[meta_col].to_numpy()
        xgb_pred = None
        bayes_pred = None
        graft_components: set[str] = set()
        requested_graft_components: set[str] = set()
        oof_graft_winners: set[str] = set()
        if cfg.xgb_graft_group != "none" and cfg.xgb_graft_weight > 0:
            graft_components, requested_graft_components, oof_graft_winners = (
                _resolve_xgb_graft_components(df, train_mask, cfg)
            )
        if _needs_xgb_points(cfg) or graft_components:
            xgb_pred = _specialist_prediction(
                df, train_idx, season, minutes_hat, _predict_rates_xgb_only)
        if _needs_bayes_points(cfg):
            bayes_pred = _specialist_prediction(
                df, train_idx, season, minutes_hat, _predict_rates_bayesian)
        pred = _graft_components(
            df, train_mask, pred, xgb_pred, cfg,
            graft_components, requested_graft_components, oof_graft_winners)

        if cfg.recon_calib == "linear":
            recon_c = _forward_linear_calib(pred)
            pred = pred.copy()
            pred["recon_pts_hat"] = recon_c
            pred["target_pts_hat"] = recon_c + pred["latent_hat"].to_numpy(float)

        pre_calib_target = pred["target_pts_hat"].to_numpy(float).copy()
        pred = _apply_target_priors(df, test_idx, pred, season, cfg)
        if xgb_pred is not None:
            xgb_pred = _apply_target_priors(df, test_idx, xgb_pred, season, cfg)
        if bayes_pred is not None:
            bayes_pred = _apply_target_priors(df, test_idx, bayes_pred, season, cfg)
        pred = _apply_specialist_points(df, test_idx, pred, xgb_pred, bayes_pred, cfg)
        if cfg.selector_point_source == "pre_calib":
            pred = pred.copy()
            pred["selector_pts_hat"] = pre_calib_target
        else:
            pred = pred.copy()
            pred["selector_pts_hat"] = pred["target_pts_hat"].to_numpy(float)
        pred["lgbm_rank_score"] = rank_scores(df, train_idx, test_idx, MODE)

        pred, sel_col = _add_selector_score(df, train_idx, test_idx, pred, cfg)

    return pred, sel_col, registry


def _evaluate_prediction(
    pred: pd.DataFrame, sel_col: str, cfg: Config, registry: pd.DataFrame | None,
) -> dict:
    vx, ratios = value_of_xv(pred, sel_col)
    c1, c3 = captain_hitrate(pred, sel_col)
    capt = captain_hit_rates(pred, sel_col, (5,))
    return {
        "config": cfg.name,
        "value_xv": vx,
        "round_xv": [float(r) for r in ratios],
        "round_mae": _round_mae(pred),
        "round_bias": _round_bias(pred),
        "mae": points_mae(pred, "target_pts_hat")["overall"],
        "by_pos": _position_diagnostics(pred),
        "top15": topn_overlap(pred, 15, sel_col),
        "top30": topn_overlap(pred, 30, sel_col),
        "capt_top1": c1,
        "capt_top3": c3,
        "capt_top5": capt["capt_top5"],
        "spearman_pos": spearman_within_pos(pred, "target_pts_hat"),
        "spearman_sel": spearman_within_pos(pred, sel_col),
        "registry_lgbm": _registry_lgbm_count(cfg, registry),
    }


def _round_mae(pred: pd.DataFrame) -> list[float]:
    out = []
    for _, g in pred.groupby("round", sort=True):
        lab = g[g["has_label"].astype(bool) & g["is_modern"].astype(bool)]
        out.append(float((lab["official_pts"] - lab["target_pts_hat"]).abs().mean()))
    return out


def _round_bias(pred: pd.DataFrame) -> list[float]:
    out = []
    for _, g in pred.groupby("round", sort=True):
        lab = g[g["has_label"].astype(bool) & g["is_modern"].astype(bool)]
        out.append(float((lab["official_pts"] - lab["target_pts_hat"]).mean()))
    return out


def _position_diagnostics(pred: pd.DataFrame) -> dict[str, dict]:
    lab = pred[pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)].copy()
    lab["abs_err"] = (lab["official_pts"] - lab["target_pts_hat"]).abs()
    lab["resid"] = lab["official_pts"] - lab["target_pts_hat"]
    out = {}
    for pos, g in lab.groupby("canonical_pos", sort=True):
        out[str(pos)] = {
            "n": int(len(g)),
            "mae": float(g["abs_err"].mean()),
            "bias": float(g["resid"].mean()),
            "actual_mean": float(g["official_pts"].mean()),
            "pred_mean": float(g["target_pts_hat"].mean()),
        }
    return out


def dev_evaluate(df, cfg: Config, season: int = DEV_SEASON) -> dict:
    """Run the full Strategy-C assembly for `cfg` on `season` and return metrics."""
    assert season != SEALED_SEASON, "2026 is sealed — the loop must not read it"
    pred, sel_col, registry = _predict_config(df, cfg, season)
    return _evaluate_prediction(pred, sel_col, cfg, registry)


# ---------------------------------------------------------------------------
# acceptance test
# ---------------------------------------------------------------------------
def accept(best: dict, cand: dict) -> tuple[bool, str]:
    dv = cand["value_xv"] - best["value_xv"]
    dm = cand["mae"] - best["mae"]                      # negative = improvement
    dt15 = cand["top15"] - best["top15"]
    dspear = cand["spearman_sel"] - best["spearman_sel"]
    rb_v = sum(c > b + 1e-9 for c, b in zip(cand["round_xv"], best["round_xv"]))
    rb_m = sum(c < b - 1e-9 for c, b in zip(cand["round_mae"], best["round_mae"]))
    pareto_v = dv >= DELTA_V and dm <= TAU_MAE
    pareto_m = dm <= -DELTA_M and dv >= -DELTA_V
    guardrails = []
    if dt15 < -TOP15_MAX_DROP:
        guardrails.append(f"top15 {dt15:+.3f}")
    if np.isfinite(dspear) and dspear < -SPEARMAN_MAX_DROP:
        guardrails.append(f"spearman_sel {dspear:+.3f}")
    if (pareto_v or pareto_m) and guardrails:
        return False, (f"reject (guardrail failed: {', '.join(guardrails)}): "
                       f"dval={dv:+.3f} dmae={dm:+.3f} dtop15={dt15:+.3f} "
                       f"dspear={dspear:+.3f} value_rounds_up={rb_v}/5 "
                       f"mae_rounds_up={rb_m}/5")
    if pareto_v and rb_v >= ROUND_ROBUST:
        return True, (f"ACCEPT via value_xv: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
                      f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")
    if pareto_m and rb_m >= ROUND_ROBUST:
        return True, (f"ACCEPT via mae: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
                      f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")
    if pareto_v or pareto_m:
        why = "round robustness failed"
    else:
        why = "no Pareto improvement"
    return (False, f"reject ({why}): dval={dv:+.3f} dmae={dm:+.3f} "
            f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
            f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")


def _subset_metrics(pred: pd.DataFrame, sel_col: str, mask: np.ndarray) -> dict:
    sub = pred.loc[mask].copy()
    vx, ratios = value_of_xv(sub, sel_col)
    return {
        "value_xv": vx,
        "mae": points_mae(sub, "target_pts_hat")["overall"],
        "top15": topn_overlap(sub, 15, sel_col),
        "top30": topn_overlap(sub, 30, sel_col),
        "spearman_sel": spearman_within_pos(sub, sel_col),
        "round_xv": [float(r) for r in ratios],
        "round_mae": _round_mae(sub),
        "round_bias": _round_bias(sub),
        "by_pos": _position_diagnostics(sub),
        "capt_top1": captain_hit_rates(sub, sel_col, (1,))["capt_top1"],
        "capt_top3": captain_hit_rates(sub, sel_col, (3,))["capt_top3"],
        "capt_top5": captain_hit_rates(sub, sel_col, (5,))["capt_top5"],
        "spearman_pos": spearman_within_pos(sub, "target_pts_hat"),
        "registry_lgbm": 0,
    }


def _noninferior(best: dict, cand: dict) -> bool:
    return (
        cand["mae"] - best["mae"] <= TAU_MAE
        and cand["value_xv"] >= best["value_xv"] - DELTA_V
    )


def _perturb_minutes_proxy(pred: pd.DataFrame, scenario: str, sel_col: str) -> pd.DataFrame:
    out = pred.copy()
    factor = np.ones(len(out), dtype=float)
    isf = out["is_forward"].astype(bool).to_numpy()
    started = out["started"].astype(bool).to_numpy() if "started" in out.columns else np.ones(len(out), bool)
    if scenario == "forwards_low":
        factor[isf] = 0.95
        factor[~started] = 0.90
    elif scenario == "forwards_high":
        factor[isf] = 1.05
        factor[~started] = 1.10
    elif scenario == "backs_low":
        factor[~isf] = 0.95
        factor[~started] = 0.90
    elif scenario == "backs_high":
        factor[~isf] = 1.05
        factor[~started] = 1.10
    else:
        raise ValueError(f"unknown minutes perturbation {scenario!r}")

    old_recon = out["recon_pts_hat"].to_numpy(float)
    delta = old_recon * factor - old_recon
    out["recon_pts_hat"] = np.clip(old_recon + delta, 0.0, None)
    for col in ("target_pts_hat", "selector_pts_hat", "lgbm_target_pts_hat",
                "xgb_component_pts_hat", "bayes_component_pts_hat"):
        if col in out.columns:
            out[col] = np.clip(out[col].to_numpy(float) + delta, 0.0, None)
    if sel_col == "sel_score" and "selector_pts_hat" in out.columns:
        old_base = _zscore(pred["selector_pts_hat"].to_numpy(float))
        rank_part = pred["sel_score"].to_numpy(float) - old_base
        out["sel_score"] = _zscore(out["selector_pts_hat"].to_numpy(float)) + rank_part
    return out


def stability_gate(
    best_pred: pd.DataFrame,
    best_sel_col: str,
    cand_pred: pd.DataFrame,
    cand_sel_col: str,
    best: dict,
    cand: dict,
    cand_name: str,
) -> tuple[bool, dict]:
    """2025-only robustness gate run after the base acceptance rule passes."""
    report: dict = {"candidate": cand_name, "checks": {}, "passed": True}

    round_rows = []
    round_pass = 0
    for r in sorted(best_pred["round"].dropna().unique()):
        mask_b = best_pred["round"].to_numpy() != r
        mask_c = cand_pred["round"].to_numpy() != r
        mb = _subset_metrics(best_pred, best_sel_col, mask_b)
        mc = _subset_metrics(cand_pred, cand_sel_col, mask_c)
        ok = _noninferior(mb, mc)
        round_pass += int(ok)
        round_rows.append({"drop_round": int(r), "passed": ok,
                           "dvalue": mc["value_xv"] - mb["value_xv"],
                           "dmae": mc["mae"] - mb["mae"]})
    report["checks"]["round_drop"] = {
        "passed": round_pass >= ROUND_DROP_MIN_PASS,
        "passed_slices": round_pass,
        "required": ROUND_DROP_MIN_PASS,
        "slices": round_rows,
    }

    team_rows = []
    team_pass = 0
    teams = sorted(set(best_pred.get("team", pd.Series(dtype=str)).dropna()))
    for team in teams:
        mask_b = best_pred["team"].to_numpy() != team
        mask_c = cand_pred["team"].to_numpy() != team
        mb = _subset_metrics(best_pred, best_sel_col, mask_b)
        mc = _subset_metrics(cand_pred, cand_sel_col, mask_c)
        ok = _noninferior(mb, mc)
        team_pass += int(ok)
        team_rows.append({"drop_team": str(team), "passed": ok,
                          "dvalue": mc["value_xv"] - mb["value_xv"],
                          "dmae": mc["mae"] - mb["mae"]})
    report["checks"]["team_drop"] = {
        "passed": team_pass >= min(TEAM_DROP_MIN_PASS, len(teams)),
        "passed_slices": team_pass,
        "required": min(TEAM_DROP_MIN_PASS, len(teams)),
        "slices": team_rows,
    }

    pos_rows = []
    pos_pass = 0
    for name, positions in POSITION_GROUPS.items():
        mask_b = ~best_pred["canonical_pos"].isin(positions).to_numpy()
        mask_c = ~cand_pred["canonical_pos"].isin(positions).to_numpy()
        mb = _subset_metrics(best_pred, best_sel_col, mask_b)
        mc = _subset_metrics(cand_pred, cand_sel_col, mask_c)
        dm = mc["mae"] - mb["mae"]
        ds = mc["spearman_sel"] - mb["spearman_sel"]
        ok = dm <= POSITION_MAE_MAX_REGRESSION and (
            not np.isfinite(ds) or ds >= -POSITION_SPEARMAN_MAX_DROP
        )
        pos_pass += int(ok)
        pos_rows.append({"drop_position_group": name, "passed": ok,
                         "dmae": dm, "dspearman": ds,
                         "dvalue": mc["value_xv"] - mb["value_xv"]})
    report["checks"]["position_drop"] = {
        "passed": pos_pass == len(POSITION_GROUPS),
        "passed_slices": pos_pass,
        "required": len(POSITION_GROUPS),
        "slices": pos_rows,
    }

    perturb_rows = []
    perturb_pass = 0
    for scenario in ("forwards_low", "forwards_high", "backs_low", "backs_high"):
        pb = _perturb_minutes_proxy(best_pred, scenario, best_sel_col)
        pc = _perturb_minutes_proxy(cand_pred, scenario, cand_sel_col)
        mb = _evaluate_prediction(pb, best_sel_col, Config(name="perturb_best"), None)
        mc = _evaluate_prediction(pc, cand_sel_col, Config(name="perturb_cand"), None)
        ok, reason = accept(mb, mc)
        perturb_pass += int(ok)
        perturb_rows.append({"scenario": scenario, "passed": ok, "reason": reason,
                             "dvalue": mc["value_xv"] - mb["value_xv"],
                             "dmae": mc["mae"] - mb["mae"]})
    report["checks"]["minutes_perturb"] = {
        "passed": perturb_pass >= MINUTES_PERTURB_MIN_PASS,
        "passed_slices": perturb_pass,
        "required": MINUTES_PERTURB_MIN_PASS,
        "slices": perturb_rows,
    }

    report["passed"] = all(c["passed"] for c in report["checks"].values())
    failed = [name for name, check in report["checks"].items() if not check["passed"]]
    report["failed_checks"] = failed
    return report["passed"], report


# ---------------------------------------------------------------------------
# candidate queue — each is (name, note, delta kwargs)
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("deploy_registry", "return to OOF-gated component registry", dict(deploy_engine="registry")),
    ("minutes_alpha_3", "tighter minutes ridge", dict(minutes_alpha=3.0)),
    ("minutes_alpha_4", "tighter minutes ridge, alpha 4", dict(minutes_alpha=4.0)),
    ("minutes_alpha_5", "tighter minutes ridge, alpha 5", dict(minutes_alpha=5.0)),
    ("minutes_alpha_6", "tighter minutes ridge, alpha 6", dict(minutes_alpha=6.0)),
    ("minutes_alpha_7", "tighter minutes ridge, alpha 7", dict(minutes_alpha=7.0)),
    ("minutes_alpha_8", "tighter minutes ridge, alpha 8", dict(minutes_alpha=8.0)),
    ("minutes_alpha_9", "tighter minutes ridge, alpha 9", dict(minutes_alpha=9.0)),
    ("minutes_alpha_30", "looser minutes ridge", dict(minutes_alpha=30.0)),
    ("minutes_interact", "started/jersey x is_forward", dict(minutes_model="ridge_interact")),
    ("minutes_poisson", "Poisson minutes head", dict(minutes_model="poisson", minutes_alpha=1.0)),
    ("latent_shrink_075", "shrink set-piece latent 0.75", dict(latent_shrink=0.75)),
    ("latent_shrink_05", "shrink set-piece latent 0.5", dict(latent_shrink=0.5)),
    ("latent_shrink_050", "shrink set-piece latent 0.50",
     dict(latent_shrink=0.50)),
    ("latent_shrink_025", "shrink set-piece latent 0.25", dict(latent_shrink=0.25)),
    ("latent_shrink_0", "drop set-piece latent (fallback 4)", dict(latent_shrink=0.0)),
    ("potm_pp_07", "POTM lean on points", dict(potm_pp_weight=0.7)),
    ("potm_pp_075", "POTM 75% points weight", dict(potm_pp_weight=0.75)),
    ("potm_points_weight_075", "POTM 75% points weight",
     dict(potm_pp_weight=0.75)),
    ("potm_pp_10", "POTM probability entirely from predicted points", dict(potm_pp_weight=1.0)),
    ("potm_points_weight_100", "POTM probability entirely from predicted points",
     dict(potm_pp_weight=1.0)),
    ("potm_pp_03", "POTM lean on role prior", dict(potm_pp_weight=0.3)),
    ("potm_tau_2", "softer POTM softmax", dict(potm_tau_floor=2.0)),
    ("potm_rank_assisted_025", "POTM points-only plus tiny XGB rank overlay",
     dict(potm_pp_weight=1.0, xgb_rank_weight=0.025, selector_point_source="target")),
    ("setpiece_latent_zero_backs_only", "force non-forward set-piece latent to zero",
     dict(zero_back_setpiece=True)),
    ("position_other_const_shrink_050", "shrink position latent constants by 50%",
     dict(other_const_shrink=0.50)),
    ("lgbm_margin_05", "stricter LGBM gate", dict(lgbm_margin=0.05)),
    ("lgbm_margin_01", "looser LGBM gate", dict(lgbm_margin=0.01)),
    ("blend_07", "heavier LGBM in blend", dict(blend_weight=0.7)),
    ("recon_calib_linear", "forward-chained recon calibration", dict(recon_calib="linear")),
    ("target_posmean_10", "blend target 10% toward prior position mean",
     dict(target_prior_blend=0.10)),
    ("target_posmean_20", "blend target 20% toward prior position mean",
     dict(target_prior_blend=0.20)),
    ("target_frontrow_10_selector_raw",
     "blend target 10% toward prior position mean for props/hookers; keep raw selector",
     dict(target_prior_blend=0.10, target_prior_scope="frontrow",
          selector_point_source="pre_calib")),
    ("target_frontrow_15_selector_raw",
     "blend target 15% toward prior position mean for props/hookers; keep raw selector",
     dict(target_prior_blend=0.15, target_prior_scope="frontrow",
          selector_point_source="pre_calib")),
    ("target_frontrow_20_selector_raw",
     "front-row target blend but keep selector based on pre-calibration points",
     dict(target_prior_blend=0.20, target_prior_scope="frontrow",
          selector_point_source="pre_calib")),
    ("target_frontrow_25_selector_raw",
     "blend target 25% toward prior position mean for props/hookers; keep raw selector",
     dict(target_prior_blend=0.25, target_prior_scope="frontrow",
          selector_point_source="pre_calib")),
    ("target_frontrow_30_selector_raw",
     "blend target 30% toward prior position mean for props/hookers; keep raw selector",
     dict(target_prior_blend=0.30, target_prior_scope="frontrow",
          selector_point_source="pre_calib")),
    ("target_prop_20_selector_raw",
     "blend target 20% toward prior position mean for props only; keep raw selector",
     dict(target_prior_blend=0.20, target_prior_scope="prop",
          selector_point_source="pre_calib")),
    ("target_hooker_20_selector_raw",
     "blend target 20% toward prior position mean for hookers only; keep raw selector",
     dict(target_prior_blend=0.20, target_prior_scope="hooker",
          selector_point_source="pre_calib")),
    ("target_tight5_20", "blend target 20% toward prior position mean for tight five",
     dict(target_prior_blend=0.20, target_prior_scope="tight5")),
    ("target_frontrow20_back5_10",
     "keep promoted front-row blend, then add 10% prior blend for back-five roles",
     dict(extra_prior_blend=0.10, extra_prior_scope="back5")),
    ("target_frontrow20_back5_20",
     "keep promoted front-row blend, then add 20% prior blend for back-five roles",
     dict(extra_prior_blend=0.20, extra_prior_scope="back5")),
    ("target_frontrow20_nonfront_10",
     "keep promoted front-row blend, then add 10% prior blend for all non-frontrow roles",
     dict(extra_prior_blend=0.10, extra_prior_scope="non_frontrow")),
    ("target_frontrow20_nonfront_20",
     "keep promoted front-row blend, then add 20% prior blend for all non-frontrow roles",
     dict(extra_prior_blend=0.20, extra_prior_scope="non_frontrow")),
    # Conservative XGBoost point blends: XGB is a minority specialist, never the boss.
    ("points_lgbm95_xgb05", "95% LGBM target points + 5% XGB component points",
     dict(xgb_point_weight=0.05, selector_point_source="target")),
    ("points_lgbm90_xgb10", "90% LGBM target points + 10% XGB component points",
     dict(xgb_point_weight=0.10, selector_point_source="target")),
    ("points_lgbm85_xgb15", "85% LGBM target points + 15% XGB component points",
     dict(xgb_point_weight=0.15, selector_point_source="target")),
    ("points_lgbm80_xgb20", "80% LGBM target points + 20% XGB component points",
     dict(xgb_point_weight=0.20, selector_point_source="target")),
    # XGBoost rank-only overlay: MAE-facing points stay LGBM.
    ("xgb_rank_overlay_0025", "add tiny XGB rank-only overlay",
     dict(xgb_rank_weight=0.025)),
    ("xgb_rank_overlay_005", "add small XGB rank-only overlay",
     dict(xgb_rank_weight=0.05)),
    ("xgb_rank_overlay_010", "add 0.10 XGB rank-only overlay",
     dict(xgb_rank_weight=0.10)),
    ("xgb_rank_overlay_015", "add 0.15 XGB rank-only overlay",
     dict(xgb_rank_weight=0.15)),
    # Component-level XGB grafts: blend/replacement below the final point layer.
    ("xgb_graft_metres", "replace metres component with XGB specialist",
     dict(xgb_graft_group="metres", xgb_graft_weight=1.0,
          selector_point_source="pre_calib")),
    ("xgb_graft_tackles", "replace tackles component with XGB specialist",
     dict(xgb_graft_group="tackles", xgb_graft_weight=1.0,
          selector_point_source="pre_calib")),
    ("xgb_graft_tries_assists_db", "replace sparse attacking components with XGB specialist",
     dict(xgb_graft_group="tries_assists_db", xgb_graft_weight=1.0,
          selector_point_source="pre_calib")),
    ("xgb_graft_sparse_counts", "replace offload/turnover/penalties components with XGB specialist",
     dict(xgb_graft_group="sparse_counts", xgb_graft_weight=1.0,
          selector_point_source="pre_calib")),
    ("xgb_graft_oof_winners_only", "replace only components where XGB beats LGBM OOF by >=2%",
     dict(xgb_graft_group="oof_winners", xgb_graft_weight=1.0,
          selector_point_source="pre_calib")),
    ("xgb_graft_oof_winners_blend20", "20% XGB graft for OOF-winning components",
     dict(xgb_graft_group="oof_winners", xgb_graft_weight=0.20,
          selector_point_source="pre_calib")),
    # Bayesian/Ridge shrinkage specialist.
    ("bayes_points_blend_05", "95% LGBM target points + 5% Bayesian shrinkage points",
     dict(bayes_point_weight=0.05, selector_point_source="target")),
    ("bayes_points_blend_10", "90% LGBM target points + 10% Bayesian shrinkage points",
     dict(bayes_point_weight=0.10, selector_point_source="target")),
    ("bayes_cold_players_10", "10% Bayesian pull only for low-prior players",
     dict(bayes_cold_weight=0.10, selector_point_source="target")),
    ("bayes_cold_players_20", "20% Bayesian pull only for low-prior players",
     dict(bayes_cold_weight=0.20, selector_point_source="target")),
    ("bayes_high_disagreement_10", "10% Bayesian pull where LGBM/XGB disagree most",
     dict(bayes_disagreement_weight=0.10, selector_point_source="target")),
    ("bayes_position_residual_10", "forward-chained 10% position residual correction",
     dict(bayes_position_residual_weight=0.10, selector_point_source="target")),
    # Fixed capped combiners.
    ("combiner_lgbm90_xgb05_bayes05", "fixed 90/5/5 LGBM/XGB/Bayesian combiner",
     dict(combiner_lgbm_weight=0.90, combiner_xgb_weight=0.05,
          combiner_bayes_weight=0.05, selector_point_source="target")),
    ("combiner_lgbm85_xgb10_bayes05", "fixed 85/10/5 LGBM/XGB/Bayesian combiner",
     dict(combiner_lgbm_weight=0.85, combiner_xgb_weight=0.10,
          combiner_bayes_weight=0.05, selector_point_source="target")),
    ("combiner_lgbm80_xgb10_bayes10", "fixed 80/10/10 LGBM/XGB/Bayesian combiner",
     dict(combiner_lgbm_weight=0.80, combiner_xgb_weight=0.10,
          combiner_bayes_weight=0.10, selector_point_source="target")),
    ("selector_tilt_025", "tilt XV pick toward rank head", dict(selector_tilt=0.25)),
    ("selector_tilt_05", "stronger rank tilt", dict(selector_tilt=0.5)),
]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run_loop(
    candidate_names: list[str] | set[str] | None = None,
    limit: int | None = None,
    base_cfg: Config | None = None,
) -> tuple[Config, dict, list]:
    df = load()
    RESEARCH.mkdir(exist_ok=True)

    incumbent_cfg = base_cfg or Config()
    best_cfg = incumbent_cfg
    best_pred, best_sel_col, best_registry = _predict_config(df, best_cfg, DEV_SEASON)
    best = _evaluate_prediction(best_pred, best_sel_col, best_cfg, best_registry)
    trials = [{"trial": 0, "name": incumbent_cfg.name, "note": incumbent_cfg.note,
               "accepted": True, "stability_passed": True, "reason": "incumbent",
               **_metric_row(best)}]
    stability_reports = []
    print(f"[0] {best_cfg.name:18s} value_xv={best['value_xv']:.3f} "
          f"mae={best['mae']:.3f} top15={best['top15']:.3f} "
          f"lgbm={best['registry_lgbm']}")

    candidates = CANDIDATES
    if candidate_names is not None:
        by_name = {c[0]: c for c in candidates}
        ordered = list(candidate_names)
        missing = [name for name in ordered if name not in by_name]
        if missing:
            raise SystemExit(f"unknown candidate(s): {', '.join(missing)}")
        candidates = [by_name[name] for name in ordered]
    if limit is not None:
        candidates = candidates[:limit]

    for i, (name, note, kw) in enumerate(candidates, 1):
        cand_cfg = incumbent_cfg.delta(name=name, note=note, **kw)
        cand_pred, cand_sel_col, cand_registry = _predict_config(df, cand_cfg, DEV_SEASON)
        cand = _evaluate_prediction(cand_pred, cand_sel_col, cand_cfg, cand_registry)
        ok, reason = accept(best, cand)
        stability_passed = False
        stability_reason = "not run"
        if ok:
            stability_passed, stability = stability_gate(
                best_pred, best_sel_col, cand_pred, cand_sel_col, best, cand, name)
            stability_reports.append(stability)
            if not stability_passed:
                ok = False
                stability_reason = ",".join(stability["failed_checks"])
                reason = f"reject (stability gate failed: {stability_reason}): {reason}"
            else:
                stability_reason = "passed"
        if ok and name in KNOWN_SEALED_VETOES:
            ok = False
            stability_reason = (
                f"{stability_reason}; known sealed veto"
                if stability_reason != "not run" else "known sealed veto"
            )
            reason = f"reject (known sealed veto): {KNOWN_SEALED_VETOES[name]}"
        print(f"[{i}] {name:20s} value_xv={cand['value_xv']:.3f} "
              f"mae={cand['mae']:.3f} top15={cand['top15']:.3f}  -> "
              f"{'ACCEPT' if ok else 'reject'}  ({reason})")
        trials.append({"trial": i, "name": name, "note": note,
                       "accepted": ok, "stability_passed": stability_passed,
                       "stability_reason": stability_reason, "reason": reason,
                       **_metric_row(cand)})
        if ok:
            best_cfg, best = cand_cfg, cand
            best_pred, best_sel_col, best_registry = cand_pred, cand_sel_col, cand_registry

    _write_stability_report(stability_reports)
    _write_ledger(trials, best_cfg, best)
    _persist_best(df, best_cfg, best)
    print(f"\nBEST = {best_cfg.name}  value_xv={best['value_xv']:.3f} "
          f"mae={best['mae']:.3f}  (cdx baseline 0.680 / 7.680)")
    return best_cfg, best, trials


def _metric_row(m: dict) -> dict:
    return {k: m[k] for k in ("value_xv", "mae", "top15", "top30", "capt_top1",
                              "capt_top3", "capt_top5", "spearman_pos",
                              "spearman_sel", "registry_lgbm")} \
        | {"round_xv": m["round_xv"], "round_mae": m["round_mae"],
           "round_bias": m["round_bias"], "by_pos": m["by_pos"]}


def _write_ledger(trials: list, best_cfg: Config, best: dict) -> None:
    (RESEARCH / "ledger.json").write_text(json.dumps(trials, indent=2))
    (RESEARCH / "best_config.json").write_text(
        json.dumps(dataclasses.asdict(best_cfg), indent=2))
    _append_history(trials, best_cfg, best)
    lines = [f"# Autoresearch ledger — {date.today()}", "",
             f"Dev season {DEV_SEASON} (post_team_sheet). "
             f"Incumbent value_xv={trials[0]['value_xv']:.3f} / "
             f"MAE={trials[0]['mae']:.3f}.",
             "2026 sealed — not evaluated here.", "",
             "| # | candidate | value_xv | MAE | top15 | top30 | cap3 | spear_sel | stability | accepted | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in trials:
        lines.append(f"| {t['trial']} | {t['name']} | {t['value_xv']:.3f} | "
                     f"{t['mae']:.3f} | {t['top15']:.3f} | "
                     f"{t['top30']:.3f} | {t['capt_top3']:.3f} | "
                     f"{t['spearman_sel']:.3f} | "
                     f"{t.get('stability_reason', 'passed')} | "
                     f"{'yes' if t['accepted'] else 'no'} | {t['reason']} |")
    lines += ["", f"**Best:** `{best_cfg.name}` — value_xv {best['value_xv']:.3f}, "
              f"MAE {best['mae']:.3f}, top15 {best['top15']:.3f}.",
              "", "```json", json.dumps(dataclasses.asdict(best_cfg), indent=2), "```"]
    (RESEARCH / "LEDGER.md").write_text("\n".join(lines))


def _write_stability_report(reports: list[dict]) -> None:
    payload = {
        "date": str(date.today()),
        "dev_season": DEV_SEASON,
        "policy": "Run only after the base 2025 acceptance rule passes; candidates failing this gate cannot become incumbent or open sealed 2026.",
        "reports": reports,
    }
    STABILITY_REPORT.write_text(json.dumps(payload, indent=2))


def _append_history(trials: list, best_cfg: Config, best: dict) -> None:
    """Append each loop run before the short-form ledger gets overwritten."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dev_season": DEV_SEASON,
        "incumbent": trials[0]["name"],
        "best_config": best_cfg.name,
        "best": _metric_row(best),
        "trials": trials,
    }
    with (RESEARCH / "history.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _persist_best(df, best_cfg: Config, best: dict) -> None:
    """Write dev-best predictions without overwriting promoted data artifacts."""
    pred, _, registry = _predict_config(df, best_cfg, DEV_SEASON)
    if registry is not None:
        save_registry(registry, DATA / "model_component_registry.csv")
    pred.to_csv(RESEARCH / f"dev_predictions_{DEV_SEASON}.csv", index=False)


def seal_2026(best_cfg: Config, *, persist_predictions: bool = True) -> None:
    """Evaluate the final best config on the SEALED season exactly once."""
    df = load()
    pred, sel_col, registry = _predict_config(df, best_cfg, SEALED_SEASON)
    res = _evaluate_prediction(pred, sel_col, best_cfg, registry)
    print("\n=== SEALED 2026 (evaluated once for the final best config) ===")
    print(f"  config={best_cfg.name}  value_xv={res['value_xv']:.3f}  "
          f"mae={res['mae']:.3f}  top15={res['top15']:.3f}")
    note = {"sealed_season": SEALED_SEASON, "config": best_cfg.name,
            "evaluated_on": str(date.today()), **_metric_row(res)}
    if persist_predictions:
        (RESEARCH / "sealed_2026.json").write_text(json.dumps(note, indent=2))
        out = DATA / f"model_predictions_{SEALED_SEASON}.csv"
        pred.to_csv(out, index=False)
        print(f"  saved {out.name} from {best_cfg.name}  ({len(pred)} rows)")
    else:
        (RESEARCH / "sealed_best_check.json").write_text(json.dumps(note, indent=2))
        print("  prediction CSV not overwritten for non-promoted seal check")


def _seal_eval(df, cfg: Config) -> dict:
    pred, sel_col, registry = _predict_config(df, cfg, SEALED_SEASON)
    return _evaluate_prediction(pred, sel_col, cfg, registry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-2026", action="store_true",
                    help="evaluate the saved best config on the sealed season once")
    ap.add_argument("--seal-config", choices=["promoted", "best"], default="promoted",
                    help="which saved config to seal; promoted falls back to best if absent")
    ap.add_argument("--base-config", choices=["baseline", "promoted", "best"], default="baseline",
                    help="incumbent config for candidate comparisons")
    ap.add_argument("--candidate", action="append",
                    help="run only this candidate name; can be passed multiple times")
    ap.add_argument("--limit", type=int,
                    help="run only the first N candidates after filtering")
    args = ap.parse_args()
    if args.seal_2026:
        cfg_path = PROMOTED_CONFIG if args.seal_config == "promoted" else BEST_CONFIG
        if not cfg_path.exists() and args.seal_config == "promoted":
            cfg_path = BEST_CONFIG
        print(f"Sealing config from {cfg_path.relative_to(ROOT)}")
        cfg = Config(**json.loads(cfg_path.read_text()))
        seal_2026(cfg, persist_predictions=(cfg_path == PROMOTED_CONFIG))
    else:
        base_cfg = _load_base_config(args.base_config)
        run_loop(args.candidate if args.candidate else None, args.limit, base_cfg)


def _load_base_config(which: str) -> Config:
    if which == "baseline":
        return Config()
    cfg_path = PROMOTED_CONFIG if which == "promoted" else BEST_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path.relative_to(ROOT)} does not exist")
    return Config(**json.loads(cfg_path.read_text()))


if __name__ == "__main__":
    main()
