#!/usr/bin/env python3
"""Six Nations picker pipeline for one saved assembly config (`model/sixn/promoted_config.json`)."""
from __future__ import annotations

import contextlib
import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge

from model.sixn import assemble as A
from model.sixn import data as MD
from model.sixn import train_components as TC
from model.sixn.assemble import assemble_predictions, rank_scores
from model.sixn.baselines import (
    MIN_MINUTES,
    SCORED,
    _rate_target,
    predict_rates,
    score_recon,
)
from model.sixn.data import CATEGORICAL_COLS, feature_view
from model.sixn.splits import component_train, round_iter
from model.sixn.train_components import (
    LGBM_COMPS,
    KICK_COMPS,
    ZERO_PRIOR,
    predict_rates_lgbm_only,
    predict_rates_registry,
    select_components,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODE = "post_team_sheet"
PROMOTED_CONFIG = Path(__file__).resolve().parent / "promoted_config.json"
XGB_MAX_POINT_WEIGHT = 0.20
# The 0.20 XGB cap above guards DECISION scores (selection-XGB > 0.20 hurts value).
# The post-selector forecast path is MAE-only with frozen decisions, so it carries no
# value risk and gets its own higher cap: starters-forecast XGB improves MAE
# monotonically on both 2025 and 2026 well past 0.20.
POST_TARGET_XGB_MAX_WEIGHT = 1.0
XGB_MAX_COMBINER_WEIGHT = 0.15
BAYES_MAX_COMBINER_WEIGHT = 0.15
BAYES_MAX_SHRINK_WEIGHT = 0.20
LGBM_MIN_COMBINER_WEIGHT = 0.75

XGB_GRAFT_GROUPS = {
    "metres": {"metres"},
    "tackles": {"tackles"},
    "tries_assists_db": {"tries", "try_assists", "defenders_beaten"},
    "sparse_counts": {"offload", "tackle_turnover", "penalties_conceded"},
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
    teamplay_graft_mode: str = "off"    # off | raw | aspects
    teamplay_graft_group: str = "none"  # none | all | attack | defense | metres | tackles | oof_winners
    teamplay_graft_weight: float = 0.0
    teamplay_graft_min_gain: float = 0.02
    combiner_lgbm_weight: float = 1.0   # fixed convex final combiner
    combiner_xgb_weight: float = 0.0
    combiner_bayes_weight: float = 0.0
    # MAE-only target adjustments applied after selector/captain/supersub scores
    # are built.  These test whether a better calibrated point forecast can coexist
    # with the promoted decision stack without letting the new target perturb XV,
    # captain, or supersub choices.
    post_target_xgb_weight: float = 0.0
    post_target_bayes_weight: float = 0.0
    post_target_teamplay_weight: float = 0.0
    post_target_teamplay_mode: str = "raw"  # raw | aspects
    post_target_teamplay_component_features: str = "off"
    post_target_matchup_features: bool = False
    post_target_teamplay_graft_mode: str = "off"
    post_target_teamplay_graft_group: str = "none"
    post_target_teamplay_graft_weight: float = 0.0
    post_target_latent_shrink: float = -1.0
    post_target_scope: str = "all"      # all | starters | bench | forwards | backs | starters_forwards | starters_backs | backrow | starters_backrow | backs_backrow | starters_backs_backrow
    post_target_filter: str = "all"     # all | xgb_agree50 | xgb_agree75 | xgb_agree90
    post_target_residual_weight: float = 0.0
    post_target_residual_group: str = "none"  # none | global | forward_back | position | position_started | team | team_position
    post_target_residual_scope: str = "inherit"
    post_target_residual_min_n: int = 20
    post_target_residual_group_min_n: int = 4
    post_target_residual_clip: float = 6.0
    post_target_selector_delta_weight: float = 0.0
    post_target_selector_xgb_weight: float = -1.0  # <0 inherits MAE target; >=0 builds a separate selector signal
    post_target_selector_matchup_features: bool | None = None
    post_target_selector_delta_min_prior_n: int = 0
    post_target_selector_delta_scope: str = "all"
    post_target_selector_delta_clip: float = 0.0
    post_target_refresh_captain: bool = False
    post_target_refresh_selector: bool = False
    teamplay_features: str = "off"      # off | on/raw | aspects (global, including rank)
    teamplay_component_features: str = "off"  # off | raw | aspects (rate models only)
    teamplay_aspect_adjust: str = "none"      # none | attack | defense | kicking | pressure | multi
    teamplay_aspect_weight: float = 0.0
    teamplay_aspect_clip: float = 0.20
    weather_features: bool = False
    rolecert_features: bool = False
    style_features: str | bool = "off"   # off | raw/True | aspects
    matchup_features: bool = False
    weather_aspect_adjust: str = "none"       # none | attack_suppress | kicking_suppress | tackle_boost | multi
    weather_aspect_weight: float = 0.0
    weather_aspect_clip: float = 0.15
    rolecert_kick_realloc_weight: float = 0.0
    rolecert_kick_realloc_scope: str = "all"  # all | starters | bench
    rolecert_bench_kick_shrink: float = 0.0
    rolecert_supersub_uncertainty_weight: float = 0.0
    # learned bench/supersub layer
    bench_model: str = "none"           # none | ridge | lgbm | two_stage_ridge | two_stage_lgbm
    bench_context_features: str | bool = "off"  # off | slot | team | all
    bench_supersub_context_features: str | bool = "off"  # supersub-only context
    bench_replacement_features: bool = False  # named-starter coverage context
    bench_supersub_replacement_features: bool = False
    bench_history_features: bool = False  # player's PIT replacement history
    bench_supersub_history_features: bool = False
    bench_team_history_features: bool = False  # PIT team-by-jersey usage
    bench_supersub_team_history_features: bool = False
    bench_points_weight: float = 0.0    # blend bench target points toward learned bench head
    bench_supersub_weight: float = 0.0  # blend supersub selector toward learned bench head
    bench_alpha: float = 20.0
    bench_uncertainty_weight: float = 0.0
    bench_high_minutes_threshold: float = 30.0
    bench_high_minutes_weight: float = 0.0
    bench_low_minutes_penalty_weight: float = 0.0
    bench_kick_model: str = "none"      # none | shrink | ridge | lgbm
    bench_kick_shrink: float = 0.0
    # decoupled set-piece latent for the supersub head only.  -1.0 disables
    # (supersub uses the same latent_shrink as the rest of the model); any value
    # in [0, 1] gives the bench/supersub head its own set-piece latent treatment
    # so a regularised point/XV/captain path can coexist with a supersub head that
    # keeps the full set-piece signal impact bench forwards rely on.
    supersub_latent_shrink: float = -1.0
    # Ceiling tilt for the 3x supersub pick (mirror of the captain ceiling bet).
    # Tested 0.10-1.50 on the captain_upside_full base: it hurts 2025 team value
    # (>=0.50 makes 2025 rounds worse) for no robust 2026 gain. Unlike the captain (a
    # ~80' starter with real ceiling), the supersub is a ~20' impact sub whose ceiling
    # is minutes-capped, so its two-stage expected-minutes mean is the right signal.
    # Kept as infrastructure but left at 0.0; do not re-test without a new rationale.
    supersub_upside_weight: float = 0.0
    # selector/captain specialist heads
    upside_scope: str = "all"
    selector_upside_weight: float = 0.0
    captain_head: str = "none"          # none | mean
    captain_upside_weight: float = 0.0
    captain_rank_weight: float = 0.0
    captain_kicker_weight: float = 0.0  # positive = floor boost, negative = floor penalty
    # Role-certainty / fixture-shape frontier experiments.
    kicking_realloc_weight: float = 0.0
    kicking_realloc_scope: str = "all"  # all | starters | bench
    kicking_realloc_min_share: float = 0.0
    post_target_forward_combiner: str = "none"  # none | conservative_grid
    post_target_combiner_scope: str = "all"

    def delta(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


@contextlib.contextmanager
def apply_config(cfg: Config):
    """Thread cfg into the module globals that the pipeline reads, then restore."""
    saved = (
        TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
        A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK,
        A.OTHER_CONST_SHRINK, A.ZERO_BACK_SETPIECE,
        MD.TEAMPLAY_ENABLED, MD.TEAMPLAY_MODE,
        MD.ROLECERT_FEATURES_ENABLED, MD.STYLE_FEATURES_ENABLED,
        MD.STYLE_FEATURES_MODE, MD.MATCHUP_FEATURES_ENABLED,
    )
    TC.LGBM_MARGIN = cfg.lgbm_margin
    TC.BLEND_WEIGHT = cfg.blend_weight
    A.POTM_PP_WEIGHT = cfg.potm_pp_weight
    A.POTM_TAU_FLOOR = cfg.potm_tau_floor
    A.LATENT_SHRINK = cfg.latent_shrink
    A.OTHER_CONST_SHRINK = cfg.other_const_shrink
    A.ZERO_BACK_SETPIECE = cfg.zero_back_setpiece
    MD.TEAMPLAY_MODE = _normalise_teamplay_mode(cfg.teamplay_features)
    MD.TEAMPLAY_ENABLED = MD.TEAMPLAY_MODE != "off"
    MD.WEATHER_FEATURES_ENABLED = bool(cfg.weather_features)
    MD.ROLECERT_FEATURES_ENABLED = bool(cfg.rolecert_features)
    MD.MATCHUP_FEATURES_ENABLED = bool(cfg.matchup_features)
    if isinstance(cfg.style_features, bool):
        style_mode = "raw" if cfg.style_features else "off"
    else:
        style_mode = "raw" if cfg.style_features == "on" else str(cfg.style_features)
    if style_mode not in {"off", "raw", "aspects"}:
        raise ValueError(f"unknown style_features mode {cfg.style_features!r}")
    MD.STYLE_FEATURES_MODE = style_mode
    MD.STYLE_FEATURES_ENABLED = style_mode != "off"
    try:
        yield
    finally:
        (TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
         A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK,
         A.OTHER_CONST_SHRINK, A.ZERO_BACK_SETPIECE,
         MD.TEAMPLAY_ENABLED, MD.TEAMPLAY_MODE,
         MD.ROLECERT_FEATURES_ENABLED, MD.STYLE_FEATURES_ENABLED,
         MD.STYLE_FEATURES_MODE, MD.MATCHUP_FEATURES_ENABLED) = saved


def _normalise_teamplay_mode(mode: str) -> str:
    if mode == "on":
        return "raw"
    if mode in {"off", "raw", "aspects"}:
        return mode
    raise ValueError(f"unknown teamplay feature mode {mode!r}")


@contextlib.contextmanager
def _teamplay_mode(mode: str):
    saved = (MD.TEAMPLAY_ENABLED, MD.TEAMPLAY_MODE)
    MD.TEAMPLAY_MODE = _normalise_teamplay_mode(mode)
    MD.TEAMPLAY_ENABLED = MD.TEAMPLAY_MODE != "off"
    try:
        yield
    finally:
        MD.TEAMPLAY_ENABLED, MD.TEAMPLAY_MODE = saved


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
        "backthree": {"Back-three"},
        "backrow_backs": {"Back-row", "Centre", "Back-three"},
        "halfbacks_backs": {"Scrum-half", "Fly-half", "Centre", "Back-three"},
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
    from model.sixn.splits import group_kfold_indices

    yield from group_kfold_indices(dft, 5)


def _add_selector_score(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    pred: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, str]:
    """Attach the optional XV-selection score used by selector_tilt candidates."""
    out = pred.copy()
    if cfg.selector_tilt <= 0 and cfg.xgb_rank_weight <= 0 and cfg.selector_upside_weight <= 0:
        sel_col = "target_pts_hat"
    else:
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
        if cfg.selector_upside_weight > 0:
            if "upside_score" not in out.columns:
                raise ValueError("selector_upside_weight requires upside_score")
            score = score + cfg.selector_upside_weight * out["upside_score"].to_numpy(float)
        if cfg.xgb_rank_weight > 0:
            if "xgb_rank_score" not in out.columns:
                out["xgb_rank_score"] = _xgb_rank_scores(df, train_idx, test_idx, MODE)
            score = score + cfg.xgb_rank_weight * _zscore(out["xgb_rank_score"].to_numpy(float))
        out["sel_score"] = score
        sel_col = "sel_score"

    if (
        cfg.bench_supersub_weight > 0
        or cfg.bench_uncertainty_weight > 0
        or cfg.bench_high_minutes_weight > 0
        or cfg.bench_low_minutes_penalty_weight > 0
        or cfg.rolecert_supersub_uncertainty_weight > 0
    ):
        base = _zscore(out[sel_col].to_numpy(float))
        if cfg.bench_supersub_weight > 0:
            if "bench_pts_hat" not in out.columns:
                raise ValueError("bench_supersub_weight requires bench_pts_hat")
            bench_col = (
                "supersub_bench_pts_hat"
                if "supersub_bench_pts_hat" in out.columns
                else "bench_pts_hat"
            )
            base = base + cfg.bench_supersub_weight * _zscore(
                out[bench_col].to_numpy(float))
        if cfg.bench_high_minutes_weight > 0:
            if "bench_high_minutes_prob" not in out.columns:
                raise ValueError("bench_high_minutes_weight requires bench_high_minutes_prob")
            base = base + cfg.bench_high_minutes_weight * _zscore(
                out["bench_high_minutes_prob"].to_numpy(float))
        if cfg.bench_low_minutes_penalty_weight > 0:
            if "bench_low_minutes_prob" not in out.columns:
                raise ValueError("bench_low_minutes_penalty_weight requires bench_low_minutes_prob")
            base = base - cfg.bench_low_minutes_penalty_weight * _zscore(
                out["bench_low_minutes_prob"].to_numpy(float))
        if cfg.bench_uncertainty_weight > 0:
            if "bench_minutes_uncertainty" not in out.columns:
                raise ValueError("bench_uncertainty_weight requires bench_minutes_uncertainty")
            base = base - cfg.bench_uncertainty_weight * _zscore(
                out["bench_minutes_uncertainty"].to_numpy(float))
        if cfg.rolecert_supersub_uncertainty_weight > 0:
            if "rolecert_replacement_role_uncertainty" not in out.columns:
                raise ValueError(
                    "rolecert_supersub_uncertainty_weight requires rolecert_replacement_role_uncertainty"
                )
            base = base - cfg.rolecert_supersub_uncertainty_weight * _zscore(
                out["rolecert_replacement_role_uncertainty"].to_numpy(float))
        if cfg.supersub_upside_weight > 0:
            if "upside_score" not in out.columns:
                raise ValueError("supersub_upside_weight requires upside_score")
            base = base + cfg.supersub_upside_weight * out["upside_score"].to_numpy(float)
        out["supersub_score"] = base

    if cfg.captain_head != "none":
        if cfg.captain_head != "mean":
            raise ValueError(f"unknown captain_head {cfg.captain_head!r}")
        cap = _zscore(out["target_pts_hat"].to_numpy(float))
        if cfg.captain_upside_weight > 0:
            if "upside_score" not in out.columns:
                raise ValueError("captain_upside_weight requires upside_score")
            cap = cap + cfg.captain_upside_weight * out["upside_score"].to_numpy(float)
        if cfg.captain_rank_weight > 0:
            if "lgbm_rank_score" not in out.columns:
                out["lgbm_rank_score"] = rank_scores(df, train_idx, test_idx, MODE)
            cap = cap + cfg.captain_rank_weight * _zscore(
                out["lgbm_rank_score"].to_numpy(float))
        if not np.isclose(cfg.captain_kicker_weight, 0.0):
            kick = (
                2.0 * out["hat_conversion_goals"].to_numpy(float)
                + 3.0 * out["hat_penalty_goals"].to_numpy(float)
            )
            cap = cap + cfg.captain_kicker_weight * _zscore(kick)
        out["captain_score"] = cap
    return out, sel_col


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
        or cfg.post_target_xgb_weight > 0
        or cfg.post_target_selector_xgb_weight > 0
        or cfg.post_target_forward_combiner != "none"
    )


def _needs_bayes_points(cfg: Config) -> bool:
    return (
        cfg.bayes_point_weight > 0
        or cfg.bayes_cold_weight > 0
        or cfg.bayes_disagreement_weight > 0
        or cfg.combiner_bayes_weight > 0
        or cfg.post_target_bayes_weight > 0
        or cfg.post_target_forward_combiner != "none"
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


TEAMPLAY_GRAFT_GROUPS = {
    "all": set(LGBM_COMPS),
    "attack": {"tries", "try_assists", "defenders_beaten", "offload", "metres"} & set(LGBM_COMPS),
    "defense": {"tackles", "tackle_turnover"} & set(LGBM_COMPS),
    "metres": {"metres"},
    "tackles": {"tackles"},
    "sparse_attack": {"tries", "try_assists", "defenders_beaten"} & set(LGBM_COMPS),
}
_TEAMPLAY_OOF_CACHE: dict = {}


def _predict_rates_lgbm_teamplay(
    feature_mode: str,
):
    def predict(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> pd.DataFrame:
        with _teamplay_mode(feature_mode):
            return predict_rates_lgbm_only(df, train_idx, test_idx, mode)

    return predict


def _teamplay_oof_winners(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    feature_mode: str,
    min_gain: float,
) -> tuple[set[str], dict[str, dict[str, float]]]:
    """Component OOF gate for the team-play specialist layer.

    Compare base LGBM component rates with the same LGBM component rates fitted
    with raw/aspect team-play features.  Only components clearing the relative
    gain threshold are eligible for grafting.
    """
    feature_mode = _normalise_teamplay_mode(feature_mode)
    key = (int(train_mask.sum()), MODE, feature_mode, round(float(min_gain), 5))
    if key in _TEAMPLAY_OOF_CACHE:
        return _TEAMPLAY_OOF_CACHE[key]
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)
    err: dict[str, dict[str, list[np.ndarray]]] = {
        c: {"base": [], "teamplay": []} for c in LGBM_COMPS
    }
    for tr_local, va_local in component_group_kfold(dft):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        vmask = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[vmask]
        if len(vu) == 0:
            continue
        for comp in LGBM_COMPS:
            ytrue = np.nan_to_num(
                _rate_target(df.iloc[vu], comp),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            with _teamplay_mode("off"):
                base_model = TC._fit_lgbm_component(df, tr_idx, va_idx, comp, MODE)
                base_pred = TC._lgbm_predict(base_model, df, vu, MODE)
            with _teamplay_mode(feature_mode):
                tp_model = TC._fit_lgbm_component(df, tr_idx, va_idx, comp, MODE)
                tp_pred = TC._lgbm_predict(tp_model, df, vu, MODE)
            err[comp]["base"].append(np.abs(ytrue - base_pred))
            err[comp]["teamplay"].append(np.abs(ytrue - tp_pred))

    diagnostics: dict[str, dict[str, float]] = {}
    winners: set[str] = set()
    for comp, vals in err.items():
        if not vals["base"] or not vals["teamplay"]:
            continue
        base_mae = float(np.concatenate(vals["base"]).mean())
        tp_mae = float(np.concatenate(vals["teamplay"]).mean())
        gain = 1.0 - tp_mae / base_mae if base_mae > 0 else 0.0
        diagnostics[comp] = {
            "base_oof_mae": base_mae,
            "teamplay_oof_mae": tp_mae,
            "relative_gain": gain,
            "accepted": bool(gain >= min_gain),
        }
        if gain >= min_gain:
            winners.add(comp)
    _TEAMPLAY_OOF_CACHE[key] = (winners, diagnostics)
    return winners, diagnostics


def _resolve_teamplay_graft_components(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    cfg: Config,
) -> tuple[set[str], set[str], set[str], dict[str, dict[str, float]]]:
    if (
        cfg.teamplay_graft_group == "none"
        or cfg.teamplay_graft_weight <= 0
        or cfg.teamplay_graft_mode == "off"
    ):
        return set(), set(), set(), {}
    winners, diagnostics = _teamplay_oof_winners(
        df, train_mask, cfg.teamplay_graft_mode, cfg.teamplay_graft_min_gain)
    if cfg.teamplay_graft_group == "oof_winners":
        requested = set(winners)
    else:
        requested = TEAMPLAY_GRAFT_GROUPS.get(cfg.teamplay_graft_group)
        if requested is None:
            raise ValueError(f"unknown teamplay_graft_group {cfg.teamplay_graft_group!r}")
    requested = set(requested) & set(LGBM_COMPS)
    return requested & winners, requested, winners, diagnostics


def _graft_teamplay_components(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    train_idx: np.ndarray,
    season: int,
    minutes_hat: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    comps, requested, winners, diagnostics = _resolve_teamplay_graft_components(df, train_mask, cfg)
    if cfg.teamplay_graft_group == "none" or cfg.teamplay_graft_weight <= 0:
        return pred
    out = pred.copy()
    out["teamplay_graft_requested_components"] = ",".join(sorted(requested))
    out["teamplay_graft_oof_winners"] = ",".join(sorted(winners))
    out["teamplay_graft_components"] = ",".join(sorted(comps))
    out["teamplay_graft_oof_summary"] = ";".join(
        f"{comp}:{vals['relative_gain']:.3f}" for comp, vals in sorted(diagnostics.items())
    )
    if not comps:
        return out
    tp_pred = _specialist_prediction(
        df,
        train_idx,
        season,
        minutes_hat,
        _predict_rates_lgbm_teamplay(cfg.teamplay_graft_mode),
    )
    weight = float(np.clip(cfg.teamplay_graft_weight, 0.0, 1.0))
    comp_frame = pd.DataFrame(index=out.index)
    for comp in SCORED:
        vals = out[f"hat_{comp}"].to_numpy(float).copy()
        if comp in comps:
            vals = (1.0 - weight) * vals + weight * tp_pred[f"hat_{comp}"].to_numpy(float)
        out[f"hat_{comp}"] = np.clip(vals, 0.0, None)
        comp_frame[comp] = out[f"hat_{comp}"].to_numpy(float)
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
    return out


TEAMPLAY_ATTACK_COMPS = {"tries", "try_assists", "defenders_beaten", "offload", "metres"}
TEAMPLAY_DEFENSE_COMPS = {"tackles", "tackle_turnover"}
TEAMPLAY_KICK_COMPS = {"conversion_goals", "penalty_goals"}
TEAMPLAY_PRESSURE_COMPS = {"penalties_conceded", "yellow_cards"}
TEAMPLAY_TEMPO_COMPS = {"metres", "defenders_beaten", "offload", "tries", "try_assists"}


def _mean_zscore(te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    vals = []
    for col in cols:
        if col not in te.columns:
            continue
        x = pd.to_numeric(te[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).any():
            continue
        fill = float(np.nanmean(x))
        vals.append(_zscore(np.where(np.isfinite(x), x, fill)))
    if not vals:
        return np.zeros(len(te), dtype=float)
    return np.mean(vals, axis=0)


def _teamplay_factor(signal: np.ndarray, weight: float, clip: float) -> np.ndarray:
    raw = 1.0 + float(weight) * signal
    return np.clip(raw, 1.0 - float(clip), 1.0 + float(clip))


def _apply_teamplay_aspect_adjustments(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Component-specific fixture-shape nudges from the team-play aspect layer.

    This is intentionally separate from `teamplay_features_on`: the adjustment
    treats match shape as multiple channels and only applies each channel to the
    components where the rugby logic is coherent.
    """
    if cfg.teamplay_aspect_adjust == "none" or cfg.teamplay_aspect_weight <= 0:
        return pred
    te = df.iloc[test_idx]
    out = pred.copy()
    w = float(cfg.teamplay_aspect_weight)
    clip = float(cfg.teamplay_aspect_clip)
    attack = _mean_zscore(te, [
        "teamplay_aspect_attack_balance_hat",
        "teamplay_aspect_attack_volume_hat",
        "teamplay_aspect_points_edge_hat",
        "teamplay_aspect_possession_edge_hat",
    ])
    defense = _mean_zscore(te, [
        "teamplay_aspect_defensive_load_hat",
        "teamplay_aspect_tackle_load_edge_hat",
        "teamplay_aspect_pressure_hat",
    ])
    kicking = _mean_zscore(te, [
        "teamplay_aspect_kicking_opportunity_hat",
        "teamplay_points_for_hat",
        "teamplay_aspect_points_edge_hat",
    ])
    strength = _mean_zscore(te, [
        "teamplay_points_for_hat",
        "teamplay_margin_hat",
        "teamplay_win_prob_hat",
        "teamplay_dominance_prob_hat",
        "teamplay_aspect_points_edge_hat",
        "teamplay_aspect_dominance_edge_hat",
    ])
    tempo = _mean_zscore(te, [
        "teamplay_total_points_hat",
        "teamplay_aspect_open_game_hat",
        "teamplay_aspect_attack_volume_hat",
        "teamplay_runs_hat",
        "teamplay_metres_hat",
    ])
    pressure = _mean_zscore(te, [
        "teamplay_aspect_pressure_hat",
        "teamplay_aspect_defensive_load_hat",
    ])
    groups: dict[str, tuple[set[str], np.ndarray, float]] = {}
    kind = cfg.teamplay_aspect_adjust
    if kind in {"attack", "multi"}:
        groups["attack"] = (TEAMPLAY_ATTACK_COMPS, attack, w)
    if kind in {"defense", "multi"}:
        groups["defense"] = (TEAMPLAY_DEFENSE_COMPS, defense, w)
    if kind in {"kicking", "multi"}:
        groups["kicking"] = (TEAMPLAY_KICK_COMPS, kicking, w)
    if kind in {"pressure", "multi"}:
        groups["pressure"] = (TEAMPLAY_PRESSURE_COMPS, pressure, w * 0.5)
    if kind in {"strength", "fixture_strength"}:
        groups["strength_attack"] = (TEAMPLAY_ATTACK_COMPS, strength, w)
        groups["strength_kicking"] = (TEAMPLAY_KICK_COMPS, strength, w * 0.75)
    if kind == "tempo":
        groups["tempo"] = (TEAMPLAY_TEMPO_COMPS, tempo, w)
    if kind == "strength_multi":
        groups["strength_attack"] = (TEAMPLAY_ATTACK_COMPS, strength, w)
        groups["strength_kicking"] = (TEAMPLAY_KICK_COMPS, kicking, w * 0.75)
        groups["tempo"] = (TEAMPLAY_TEMPO_COMPS, tempo, w * 0.50)
        groups["defense"] = (TEAMPLAY_DEFENSE_COMPS, defense, w * 0.50)
    if not groups:
        raise ValueError(f"unknown teamplay_aspect_adjust {kind!r}")

    comp_frame = pd.DataFrame(index=out.index)
    adjusted: set[str] = set()
    for comp in SCORED:
        vals = out[f"hat_{comp}"].to_numpy(float).copy()
        for comps, signal, group_w in groups.values():
            if comp in comps:
                vals = vals * _teamplay_factor(signal, group_w, clip)
                adjusted.add(comp)
        out[f"hat_{comp}"] = np.clip(vals, 0.0, None)
        comp_frame[comp] = out[f"hat_{comp}"].to_numpy(float)
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
    out["teamplay_aspect_adjusted_components"] = ",".join(sorted(adjusted))
    return out


WEATHER_ATTACK_COMPS = {"tries", "try_assists", "defenders_beaten", "offload", "metres"}
WEATHER_KICK_COMPS = {"conversion_goals", "penalty_goals"}
WEATHER_TACKLE_COMPS = {"tackles", "tackle_turnover"}
WEATHER_PRESSURE_COMPS = {"penalties_conceded", "yellow_cards"}


def _apply_weather_aspect_adjustments(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Component-specific weather nudges.

    Weather failed as a blanket feature family, so this keeps it small and rugby-
    shaped: wet/windy/cold conditions suppress open attacking and kicking outputs,
    while optionally lifting tackle/pressure components.
    """
    if cfg.weather_aspect_adjust == "none" or cfg.weather_aspect_weight <= 0:
        return pred
    te = df.iloc[test_idx]
    if not any(c.startswith("weather_") for c in te.columns):
        return pred

    out = pred.copy()
    w = float(cfg.weather_aspect_weight)
    clip = float(cfg.weather_aspect_clip)
    wet = _mean_zscore(te, ["weather_wet_index", "weather_rain_mm", "weather_precip_mm"])
    wind = _mean_zscore(te, ["weather_wind_index", "weather_wind_kph", "weather_wind_gust_kph"])
    cold = _mean_zscore(te, ["weather_cold_index"])
    bad = np.mean([wet, wind, cold], axis=0)
    slippery = np.mean([wet, wind], axis=0)

    groups: dict[str, tuple[set[str], np.ndarray, float]] = {}
    kind = cfg.weather_aspect_adjust
    if kind in {"attack_suppress", "multi"}:
        groups["attack_suppress"] = (WEATHER_ATTACK_COMPS, -bad, w)
    if kind in {"kicking_suppress", "multi"}:
        groups["kicking_suppress"] = (WEATHER_KICK_COMPS, -slippery, w * 0.75)
    if kind in {"tackle_boost", "multi"}:
        groups["tackle_boost"] = (WEATHER_TACKLE_COMPS, bad, w * 0.50)
        groups["pressure_boost"] = (WEATHER_PRESSURE_COMPS, slippery, w * 0.25)
    if not groups:
        raise ValueError(f"unknown weather_aspect_adjust {kind!r}")

    comp_frame = pd.DataFrame(index=out.index)
    adjusted: set[str] = set()
    for comp in SCORED:
        vals = out[f"hat_{comp}"].to_numpy(float).copy()
        for comps, signal, group_w in groups.values():
            if comp in comps:
                vals = vals * _teamplay_factor(signal, group_w, clip)
                adjusted.add(comp)
        out[f"hat_{comp}"] = np.clip(vals, 0.0, None)
        comp_frame[comp] = out[f"hat_{comp}"].to_numpy(float)
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
    out["weather_aspect_adjusted_components"] = ",".join(sorted(adjusted))
    return out


def _apply_kicking_reallocation(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Concentrate team kicking components onto the learned likely kicker.

    This is a role-certainty experiment, not player hardcoding.  The share comes
    from PIT goal-kicker rate and kick-attempt history within each fixture-team.
    """
    if cfg.kicking_realloc_weight <= 0:
        return pred
    out = pred.copy()
    te = df.iloc[test_idx].reset_index(drop=True)
    weight = float(np.clip(cfg.kicking_realloc_weight, 0.0, 1.0))
    min_share = float(np.clip(cfg.kicking_realloc_min_share, 0.0, 1.0))
    role_rate = te["role_goal_kicker_rate"].fillna(0.0).to_numpy(float)
    attempts = te["role_kick_attempts"].fillna(0.0).to_numpy(float)
    signal = np.clip(role_rate, 0.0, None) * (1.0 + np.log1p(np.clip(attempts, 0.0, None)))
    started = te["started"].astype(bool).to_numpy() if "started" in te.columns else np.ones(len(te), bool)
    if cfg.kicking_realloc_scope == "all":
        scope = np.ones(len(te), dtype=bool)
    elif cfg.kicking_realloc_scope == "starters":
        scope = started
    elif cfg.kicking_realloc_scope == "bench":
        scope = ~started
    else:
        raise ValueError(f"unknown kicking_realloc_scope {cfg.kicking_realloc_scope!r}")

    applied = np.zeros(len(te), dtype=float)
    max_share = np.zeros(len(te), dtype=float)
    keys = pd.DataFrame({
        "fixture_id": te["fixture_id"].to_numpy(),
        "team_id": te["team_id"].to_numpy(),
    })
    for _, ix in keys.groupby(["fixture_id", "team_id"], sort=False).groups.items():
        loc = np.asarray(list(ix), dtype=int)
        elig = loc[scope[loc]]
        if len(elig) == 0:
            continue
        sig = signal[elig]
        if sig.sum() <= 0:
            continue
        share = sig / sig.sum()
        if share.max() < min_share:
            continue
        max_share[elig] = share.max()
        for comp in KICK_COMPS:
            col = f"hat_{comp}"
            old = out[col].to_numpy(float)
            total = float(old[elig].sum())
            if total <= 0:
                continue
            repl = total * share
            new_vals = (1.0 - weight) * old[elig] + weight * repl
            applied[elig] += np.abs(new_vals - old[elig])
            old[elig] = new_vals
            out[col] = np.clip(old, 0.0, None)

    if applied.sum() <= 0:
        return out
    comp_frame = pd.DataFrame(index=out.index)
    for comp in SCORED:
        comp_frame[comp] = out[f"hat_{comp}"].to_numpy(float)
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
    out["kicking_realloc_applied"] = applied
    out["kicking_realloc_max_share"] = max_share
    return out


def _apply_rolecert_kicking_adjustments(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Use team-sheet-relative role certainty only in the kicking component path."""
    if (
        cfg.rolecert_kick_realloc_weight <= 0
        and cfg.rolecert_bench_kick_shrink <= 0
    ):
        return pred
    te = df.iloc[test_idx].reset_index(drop=True)
    required = {"rolecert_goal_kicker_prob", "rolecert_team_kicker_confidence", "started"}
    if not required.issubset(te.columns):
        return pred

    out = pred.copy()
    started = te["started"].astype(bool).to_numpy()
    prob = te["rolecert_goal_kicker_prob"].fillna(0.0).to_numpy(float)
    conf = te["rolecert_team_kicker_confidence"].fillna(0.0).to_numpy(float)

    for comp in KICK_COMPS:
        col = f"hat_{comp}"
        vals = out[col].to_numpy(float).copy()

        if cfg.rolecert_kick_realloc_weight > 0:
            weight = float(np.clip(cfg.rolecert_kick_realloc_weight, 0.0, 1.0))
            scope_name = cfg.rolecert_kick_realloc_scope
            if scope_name == "all":
                scope = np.ones(len(te), dtype=bool)
            elif scope_name == "starters":
                scope = started
            elif scope_name == "bench":
                scope = ~started
            else:
                raise ValueError(f"unknown rolecert_kick_realloc_scope {scope_name!r}")
            for _, ix in te.groupby(["fixture_id", "team_id"], sort=False).groups.items():
                loc = np.asarray(list(ix), dtype=int)
                elig = loc[scope[loc]]
                if len(elig) == 0:
                    continue
                p = np.clip(prob[elig], 0.0, None)
                psum = float(p.sum())
                total = float(vals[elig].sum())
                if psum <= 0 or total <= 0:
                    continue
                target = total * p / psum
                vals[elig] = (1.0 - weight) * vals[elig] + weight * target

        if cfg.rolecert_bench_kick_shrink > 0:
            weight = float(np.clip(cfg.rolecert_bench_kick_shrink, 0.0, 1.0))
            relative = np.divide(
                prob,
                np.maximum(conf, 1e-6),
                out=np.zeros_like(prob),
                where=np.isfinite(conf),
            )
            uncertainty = np.clip(1.0 - relative, 0.0, 1.0)
            shrink = 1.0 - weight * (~started).astype(float) * uncertainty
            vals = vals * np.clip(shrink, 0.0, 1.0)

        out[col] = np.clip(vals, 0.0, None)

    comp_frame = pd.DataFrame(index=out.index)
    for comp in SCORED:
        comp_frame[comp] = out[f"hat_{comp}"].to_numpy(float)
    recon = score_recon(comp_frame, out["is_forward"].to_numpy())
    out["recon_pts_hat"] = recon
    out["target_pts_hat"] = recon + out["latent_hat"].to_numpy(float)
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


def _apply_post_selection_target(
    pred: pd.DataFrame,
    xgb_pred: pd.DataFrame | None,
    bayes_pred: pd.DataFrame | None,
    teamplay_pred: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    """Apply MAE-only target tweaks after selector role scores are frozen."""
    if (
        cfg.post_target_xgb_weight <= 0
        and cfg.post_target_bayes_weight <= 0
        and cfg.post_target_teamplay_weight <= 0
        and cfg.post_target_latent_shrink < 0
    ):
        return pred
    out = pred.copy()
    target = out["target_pts_hat"].to_numpy(float).copy()
    mask = (
        _post_target_mask(out, cfg.post_target_scope)
        & _post_target_filter_mask(out, cfg.post_target_filter)
    )
    if not mask.any():
        return out
    if cfg.post_target_latent_shrink >= 0:
        shrink = float(np.clip(cfg.post_target_latent_shrink, 0.0, 1.0))
        target[mask] = target[mask] - (1.0 - shrink) * out["latent_hat"].to_numpy(float)[mask]
    if cfg.post_target_xgb_weight > 0:
        if xgb_pred is None:
            raise ValueError("post_target_xgb_weight requires xgb signal")
        w = min(cfg.post_target_xgb_weight, POST_TARGET_XGB_MAX_WEIGHT)
        xgb_target = xgb_pred["target_pts_hat"].to_numpy(float)
        target[mask] = (1.0 - w) * target[mask] + w * xgb_target[mask]
    if cfg.post_target_bayes_weight > 0:
        if bayes_pred is None:
            raise ValueError("post_target_bayes_weight requires bayes signal")
        w = min(cfg.post_target_bayes_weight, BAYES_MAX_SHRINK_WEIGHT)
        bayes_target = bayes_pred["target_pts_hat"].to_numpy(float)
        target[mask] = (1.0 - w) * target[mask] + w * bayes_target[mask]
    if cfg.post_target_teamplay_weight > 0:
        if teamplay_pred is None:
            raise ValueError("post_target_teamplay_weight requires teamplay prediction")
        w = float(np.clip(cfg.post_target_teamplay_weight, 0.0, 1.0))
        teamplay_target = teamplay_pred["target_pts_hat"].to_numpy(float)
        target[mask] = (1.0 - w) * target[mask] + w * teamplay_target[mask]
    out["target_pts_hat"] = np.clip(target, 0.0, None)
    return out


def _apply_post_target_forward_combiner(
    pred: pd.DataFrame,
    xgb_pred: pd.DataFrame | None,
    bayes_pred: pd.DataFrame | None,
    teamplay_pred: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    """Forward-chained conservative stacker over specialist point forecasts.

    The grid is chosen using only earlier labelled rounds from the same season.
    It is deliberately post-selection: captain/XV/supersub decisions stay frozen
    unless a separate candidate explicitly refreshes them.
    """
    if cfg.post_target_forward_combiner == "none":
        return pred
    if cfg.post_target_forward_combiner != "conservative_grid":
        raise ValueError(f"unknown post_target_forward_combiner {cfg.post_target_forward_combiner!r}")
    if xgb_pred is None or bayes_pred is None or teamplay_pred is None:
        raise ValueError("conservative_grid combiner requires xgb, bayes, and teamplay signals")

    out = pred.copy()
    base = out["target_pts_hat"].to_numpy(float)
    sources = {
        "lgbm": base,
        "xgb": xgb_pred["target_pts_hat"].to_numpy(float),
        "bayes": bayes_pred["target_pts_hat"].to_numpy(float),
        "teamplay": teamplay_pred["target_pts_hat"].to_numpy(float),
    }
    grid = [
        ("lgbm100", (1.00, 0.00, 0.00, 0.00)),
        ("lgbm90_xgb05_bayes05", (0.90, 0.05, 0.05, 0.00)),
        ("lgbm85_xgb10_bayes05", (0.85, 0.10, 0.05, 0.00)),
        ("lgbm85_xgb05_bayes05_tp05", (0.85, 0.05, 0.05, 0.05)),
        ("lgbm80_xgb10_bayes05_tp05", (0.80, 0.10, 0.05, 0.05)),
        ("lgbm80_xgb05_bayes05_tp10", (0.80, 0.05, 0.05, 0.10)),
        ("lgbm75_xgb10_bayes05_tp10", (0.75, 0.10, 0.05, 0.10)),
        ("lgbm75_xgb05_bayes10_tp10", (0.75, 0.05, 0.10, 0.10)),
    ]
    rounds = out["round"].to_numpy()
    official = out["official_pts"].to_numpy(float)
    lab = out["has_label"].astype(bool).to_numpy() & out["is_modern"].astype(bool).to_numpy()
    scope = _post_target_mask(out, cfg.post_target_combiner_scope)
    target = base.copy()
    chosen = np.repeat("lgbm100", len(out)).astype(object)

    def blend(weights: tuple[float, float, float, float]) -> np.ndarray:
        lw, xw, bw, tw = weights
        return (
            lw * sources["lgbm"]
            + xw * sources["xgb"]
            + bw * sources["bayes"]
            + tw * sources["teamplay"]
        )

    blended = {name: blend(weights) for name, weights in grid}
    for r in np.unique(rounds):
        prev = lab & scope & (rounds < r)
        cur = scope & (rounds == r)
        if prev.sum() < 100 or not cur.any():
            continue
        scores = []
        for name, pred_vals in blended.items():
            mae = float(np.mean(np.abs(official[prev] - pred_vals[prev])))
            scores.append((mae, name))
        _, best_name = min(scores, key=lambda x: x[0])
        target[cur] = blended[best_name][cur]
        chosen[cur] = best_name

    out["target_pts_hat"] = np.clip(target, 0.0, None)
    out["post_target_combiner_choice"] = chosen
    return out


def _post_target_residual_keys(pred: pd.DataFrame, group: str) -> np.ndarray:
    if group == "global":
        return np.repeat("all", len(pred))
    if group == "forward_back":
        return np.where(pred["is_forward"].astype(bool).to_numpy(), "forward", "back")
    if group == "position":
        return pred["canonical_pos"].astype(str).to_numpy()
    if group == "position_started":
        started = pred["started"].astype(bool).map({True: "start", False: "bench"}).to_numpy()
        return pred["canonical_pos"].astype(str).to_numpy() + "|" + started
    if group == "team":
        if "team" not in pred.columns:
            raise ValueError("post_target_residual_group='team' requires team column")
        return pred["team"].astype(str).to_numpy()
    if group == "team_position":
        if "team" not in pred.columns:
            raise ValueError("post_target_residual_group='team_position' requires team column")
        return pred["team"].astype(str).to_numpy() + "|" + pred["canonical_pos"].astype(str).to_numpy()
    raise ValueError(f"unknown post_target_residual_group {group!r}")


def _apply_post_target_residual(pred: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Forward-chain residual correction after specialist target forecasts land."""
    if cfg.post_target_residual_weight <= 0 or cfg.post_target_residual_group == "none":
        return pred
    out = pred.copy()
    target = out["target_pts_hat"].to_numpy(float).copy()
    base_target = target.copy()
    lab = (out["has_label"].astype(bool) & out["is_modern"].astype(bool)).to_numpy()
    rounds = out["round"].to_numpy()
    residual = out["official_pts"].to_numpy(float) - base_target
    group = cfg.post_target_residual_group
    keys = _post_target_residual_keys(out, group)
    scope = cfg.post_target_scope if cfg.post_target_residual_scope == "inherit" else cfg.post_target_residual_scope
    scope_mask = _post_target_mask(out, scope)
    min_n = max(1, int(cfg.post_target_residual_min_n))
    group_min_n = max(1, int(cfg.post_target_residual_group_min_n))
    weight = float(np.clip(cfg.post_target_residual_weight, 0.0, 1.0))
    clip = float(max(0.0, cfg.post_target_residual_clip))
    applied = np.zeros(len(out), dtype=float)

    for r in np.unique(rounds):
        prev = lab & (rounds < r)
        cur = (rounds == r) & scope_mask
        if prev.sum() < min_n or not cur.any():
            continue
        global_resid = float(np.nanmean(residual[prev]))
        if group == "global":
            adj = np.repeat(global_resid, cur.sum())
        else:
            stats = (
                pd.DataFrame({"key": keys[prev], "resid": residual[prev]})
                .groupby("key")["resid"]
                .agg(["mean", "count"])
            )
            valid = stats[stats["count"] >= group_min_n]["mean"]
            adj = pd.Series(keys[cur]).map(valid).fillna(global_resid).to_numpy(float)
        if clip > 0:
            adj = np.clip(adj, -clip, clip)
        applied[cur] = weight * adj
        target[cur] = target[cur] + applied[cur]

    out["post_target_residual_adj"] = applied
    out["target_pts_hat"] = np.clip(target, 0.0, None)
    return out


def _apply_post_target_pipeline(
    pred: pd.DataFrame,
    xgb_pred: pd.DataFrame | None,
    bayes_pred: pd.DataFrame | None,
    teamplay_pred: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    """Apply the complete forecast-only stack to one target signal."""
    out = _apply_post_selection_target(pred, xgb_pred, bayes_pred, teamplay_pred, cfg)
    out = _apply_post_target_forward_combiner(
        out, xgb_pred, bayes_pred, teamplay_pred, cfg
    )
    return _apply_post_target_residual(out, cfg)


def _post_target_mask(pred: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(pred), dtype=bool)
    started = pred["started"].astype(bool).to_numpy() if "started" in pred.columns else np.ones(len(pred), bool)
    is_forward = pred["is_forward"].astype(bool).to_numpy()
    pos = pred["canonical_pos"].astype(str).to_numpy()
    is_backrow = pos == "Back-row"
    is_back = ~is_forward
    if scope == "starters":
        return started
    if scope == "bench":
        return ~started
    if scope == "forwards":
        return is_forward
    if scope == "backs":
        return is_back
    if scope == "starters_forwards":
        return started & is_forward
    if scope == "starters_backs":
        return started & is_back
    if scope == "backrow":
        return is_backrow
    if scope == "starters_backrow":
        return started & is_backrow
    if scope == "backs_backrow":
        return is_back | is_backrow
    if scope == "starters_backs_backrow":
        return started & (is_back | is_backrow)
    raise ValueError(f"unknown post_target_scope {scope!r}")


def _post_target_filter_mask(pred: pd.DataFrame, filt: str) -> np.ndarray:
    if filt == "all":
        return np.ones(len(pred), dtype=bool)
    prefix = "xgb_agree"
    if not filt.startswith(prefix):
        raise ValueError(f"unknown post_target_filter {filt!r}")
    if "lgbm_target_pts_hat" not in pred.columns or "xgb_component_pts_hat" not in pred.columns:
        raise ValueError(f"{filt} requires lgbm/xgb target columns")
    try:
        pct = float(filt.removeprefix(prefix)) / 100.0
    except ValueError as exc:
        raise ValueError(f"unknown post_target_filter {filt!r}") from exc
    if pct <= 0 or pct > 1:
        raise ValueError(f"unknown post_target_filter {filt!r}")
    diff = np.abs(
        pred["lgbm_target_pts_hat"].to_numpy(float)
        - pred["xgb_component_pts_hat"].to_numpy(float)
    )
    cutoff = float(np.nanquantile(diff, pct))
    return diff <= cutoff


def _selector_delta_prior_mask(pred: pd.DataFrame, min_prior_n: int) -> np.ndarray:
    if min_prior_n <= 0:
        return np.ones(len(pred), dtype=bool)
    lab = (pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)).to_numpy()
    rounds = pred["round"].to_numpy()
    out = np.zeros(len(pred), dtype=bool)
    for r in np.unique(rounds):
        if int((lab & (rounds < r)).sum()) >= min_prior_n:
            out[rounds == r] = True
    return out


def _refresh_post_target_role_scores(pred: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, str]:
    """Optionally let role heads see post-selector forecast improvements.

    The default promoted path freezes all decisions before post-target forecast
    calibration.  These opt-in experiments test whether the better target should
    affect captain or XV selection, while keeping supersub_score untouched.
    """
    out = pred.copy()
    sel_col = "sel_score" if "sel_score" in out.columns else "target_pts_hat"
    if cfg.post_target_refresh_selector:
        point_base = out["target_pts_hat"].to_numpy(float)
        if cfg.selector_tilt <= 0 and cfg.xgb_rank_weight <= 0 and cfg.selector_upside_weight <= 0:
            sel_col = "target_pts_hat"
        else:
            if "lgbm_rank_score" not in out.columns:
                raise ValueError("post_target_refresh_selector requires lgbm_rank_score")
            score = _zscore(point_base) + cfg.selector_tilt * _zscore(
                out["lgbm_rank_score"].to_numpy(float)
            )
            if cfg.selector_upside_weight > 0:
                if "upside_score" not in out.columns:
                    raise ValueError("post_target_refresh_selector requires upside_score")
                score = score + cfg.selector_upside_weight * out["upside_score"].to_numpy(float)
            if cfg.xgb_rank_weight > 0:
                if "xgb_rank_score" not in out.columns:
                    raise ValueError("post_target_refresh_selector requires xgb_rank_score")
                score = score + cfg.xgb_rank_weight * _zscore(
                    out["xgb_rank_score"].to_numpy(float)
                )
            out["sel_score"] = score
            sel_col = "sel_score"

    if cfg.post_target_selector_delta_weight > 0:
        if "selector_pts_hat" not in out.columns:
            raise ValueError("post_target_selector_delta_weight requires selector_pts_hat")
        base = out[sel_col].to_numpy(float) if sel_col in out.columns else out["target_pts_hat"].to_numpy(float)
        signal_col = (
            "post_target_selector_signal_hat"
            if "post_target_selector_signal_hat" in out.columns
            else "target_pts_hat"
        )
        delta = out[signal_col].to_numpy(float) - out["selector_pts_hat"].to_numpy(float)
        mask = (
            _post_target_mask(out, cfg.post_target_selector_delta_scope)
            & _selector_delta_prior_mask(out, cfg.post_target_selector_delta_min_prior_n)
        )
        score = base.copy()
        if mask.any():
            adj = np.zeros(len(out), dtype=float)
            z = _zscore(delta[mask])
            if cfg.post_target_selector_delta_clip > 0:
                clip = float(cfg.post_target_selector_delta_clip)
                z = np.clip(z, -clip, clip)
            adj[mask] = z
            score = score + cfg.post_target_selector_delta_weight * adj
        out["sel_score"] = score
        sel_col = "sel_score"

    if cfg.post_target_refresh_captain:
        cap = _zscore(out["target_pts_hat"].to_numpy(float))
        if cfg.captain_upside_weight > 0:
            if "upside_score" not in out.columns:
                raise ValueError("post_target_refresh_captain requires upside_score")
            cap = cap + cfg.captain_upside_weight * out["upside_score"].to_numpy(float)
        if cfg.captain_rank_weight > 0:
            if "lgbm_rank_score" not in out.columns:
                raise ValueError("post_target_refresh_captain requires lgbm_rank_score")
            cap = cap + cfg.captain_rank_weight * _zscore(
                out["lgbm_rank_score"].to_numpy(float)
            )
        if not np.isclose(cfg.captain_kicker_weight, 0.0):
            kick = (
                2.0 * out["hat_conversion_goals"].to_numpy(float)
                + 3.0 * out["hat_penalty_goals"].to_numpy(float)
            )
            cap = cap + cfg.captain_kicker_weight * _zscore(kick)
        out["captain_score"] = cap
    return out, sel_col


def _needs_bench_head(cfg: Config) -> bool:
    return (
        (cfg.bench_model != "none"
         and (cfg.bench_points_weight > 0
              or cfg.bench_supersub_weight > 0
              or cfg.bench_supersub_context_features != "off"
              or cfg.bench_supersub_replacement_features
              or cfg.bench_supersub_history_features
              or cfg.bench_supersub_team_history_features
              or cfg.bench_uncertainty_weight > 0
              or cfg.bench_high_minutes_weight > 0
              or cfg.bench_low_minutes_penalty_weight > 0))
        or cfg.bench_kick_model != "none"
    )


BENCH_COVER_FEATURES = [
    "bench_cover_starter_count",
    "bench_cover_form_minutes_mean",
    "bench_cover_form_minutes_max",
    "bench_cover_start_rate_mean",
    "bench_cover_form_n_mean",
    "bench_cover_same_role_count",
    "bench_cover_minutes_gap",
]

BENCH_HISTORY_FEATURES = [
    "benchhist_n_prior",
    "benchhist_minutes_recent",
    "benchhist_play10_rate",
    "benchhist_high30_rate",
    "benchhist_days_since_last",
]

BENCH_TEAM_HISTORY_FEATURES = [
    "benchteam_slot_n_prior",
    "benchteam_slot_minutes_recent",
    "benchteam_slot_play10_rate",
    "benchteam_slot_high30_rate",
    "benchteam_slot_days_since_last",
]


def _bench_context_frame(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.bench_replacement_features:
        return df
    out = df.copy()
    started = out["started"].astype(bool)
    position_keys = ["fixture_id", "team_id", "canonical_pos"]
    role_keys = ["fixture_id", "team_id", "is_forward"]

    def starter_stat(column: str, stat: str) -> pd.Series:
        values = pd.to_numeric(out[column], errors="coerce").where(started)
        exact = values.groupby([out[key] for key in position_keys]).transform(stat)
        fallback = values.groupby([out[key] for key in role_keys]).transform(stat)
        return exact.fillna(fallback)

    starter_flag = started.astype(float)
    exact_count = starter_flag.groupby(
        [out[key] for key in position_keys]
    ).transform("sum")
    role_count = starter_flag.groupby([out[key] for key in role_keys]).transform("sum")
    bench_flag = (~started).astype(float)
    same_role_bench = bench_flag.groupby(
        [out[key] for key in position_keys]
    ).transform("sum")

    out["bench_cover_starter_count"] = exact_count.where(exact_count > 0, role_count)
    out["bench_cover_form_minutes_mean"] = starter_stat("form_minutes_recent", "mean")
    out["bench_cover_form_minutes_max"] = starter_stat("form_minutes_recent", "max")
    out["bench_cover_start_rate_mean"] = starter_stat("form_start_rate", "mean")
    out["bench_cover_form_n_mean"] = starter_stat("form_n_prior", "mean")
    out["bench_cover_same_role_count"] = same_role_bench
    out["bench_cover_minutes_gap"] = (
        pd.to_numeric(out["form_minutes_recent"], errors="coerce")
        - out["bench_cover_form_minutes_mean"]
    )
    return out


def _bench_design(rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = [c for c in feature_view(rows, MODE) if c not in CATEGORICAL_COLS]
    X = rows[cols].astype(float).copy()
    pos = pd.get_dummies(rows["canonical_pos"], prefix="pos", dtype=float)
    blocks = [X.reset_index(drop=True), pos.reset_index(drop=True)]
    cover_cols = [col for col in BENCH_COVER_FEATURES if col in rows.columns]
    if cover_cols:
        blocks.append(
            rows[cover_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        )
    if cfg.bench_history_features:
        history_cols = [col for col in BENCH_HISTORY_FEATURES if col in rows.columns]
        if len(history_cols) != len(BENCH_HISTORY_FEATURES):
            missing = sorted(set(BENCH_HISTORY_FEATURES) - set(history_cols))
            raise ValueError(
                "bench history features require a rebuilt feature store; missing "
                + ", ".join(missing)
            )
        blocks.append(
            rows[history_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        )
    if cfg.bench_team_history_features:
        team_history_cols = [
            col for col in BENCH_TEAM_HISTORY_FEATURES if col in rows.columns
        ]
        if len(team_history_cols) != len(BENCH_TEAM_HISTORY_FEATURES):
            missing = sorted(
                set(BENCH_TEAM_HISTORY_FEATURES) - set(team_history_cols)
            )
            raise ValueError(
                "bench team history features require a rebuilt feature store; missing "
                + ", ".join(missing)
            )
        blocks.append(
            rows[team_history_cols]
            .apply(pd.to_numeric, errors="coerce")
            .reset_index(drop=True)
        )
    context_mode = "all" if cfg.bench_context_features is True else str(cfg.bench_context_features)
    if context_mode != "off":
        team = rows["team_id"].astype(str)
        slot = (
            pd.to_numeric(rows["jersey"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .astype(str)
        )
        position = rows["canonical_pos"].astype(str)
        context_values: dict[str, pd.Series] = {}
        if context_mode in {"slot", "all"}:
            context_values["bench_slot"] = slot
            context_values["bench_position_slot"] = position + "|" + slot
        if context_mode in {"team", "all"}:
            context_values["bench_team"] = team
            context_values["bench_team_position"] = team + "|" + position
        if context_mode == "all":
            context_values["bench_team_slot"] = team + "|" + slot
        if not context_values:
            raise ValueError(f"unknown bench_context_features {context_mode!r}")
        context = pd.DataFrame(context_values)
        blocks.append(
            pd.get_dummies(
                context,
                dtype=float,
            ).reset_index(drop=True)
        )
    return pd.concat(blocks, axis=1)


def _align_design(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = _bench_design(train_rows, cfg)
    Xte = _bench_design(test_rows, cfg).reindex(columns=Xtr.columns, fill_value=0.0)
    means = Xtr.mean(numeric_only=True)
    return Xtr.fillna(means).fillna(0.0), Xte.fillna(means).fillna(0.0)


def _ridge_predict(Xtr: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame, alpha: float) -> np.ndarray:
    scale = Xtr.std(axis=0, ddof=0).replace(0.0, 1.0)
    center = Xtr.mean(axis=0)
    Xtr_s = (Xtr - center) / scale
    Xte_s = (Xte - center) / scale
    return Ridge(alpha=alpha).fit(Xtr_s.to_numpy(), y).predict(Xte_s.to_numpy())


def _lgbm_reg_predict(Xtr: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=30,
        reg_lambda=10.0,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, y)
    return model.predict(Xte)


def _bench_model_predict(
    cfg: Config, Xtr: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame
) -> np.ndarray:
    if len(y) == 0 or np.allclose(y, y[0]):
        return np.full(len(Xte), float(y[0]) if len(y) else 0.0)
    family = cfg.bench_model.removeprefix("two_stage_")
    if family == "ridge":
        return _ridge_predict(Xtr, y, Xte, cfg.bench_alpha)
    if family == "lgbm":
        return _lgbm_reg_predict(Xtr, y, Xte)
    raise ValueError(f"unknown bench_model {cfg.bench_model!r}")


def _bench_target_points(rows: pd.DataFrame) -> np.ndarray:
    fallback = rows["target_pts"].fillna(rows["recon_pts"]).to_numpy(float)
    modern_label = rows["has_label"].astype(bool) & rows["is_modern"].astype(bool)
    official = rows["official_pts"].to_numpy(float)
    return np.where(modern_label.to_numpy() & np.isfinite(official), official, fallback)


def _bench_train_rows(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    train = _bench_context_frame(df, cfg).iloc[train_idx].copy()
    train = train[~train["started"].astype(bool)].copy()
    return train


def _bench_minutes_signals(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    context_df = _bench_context_frame(df, cfg)
    train = context_df.iloc[train_idx].copy()
    train = train[~train["started"].astype(bool)].copy()
    test = context_df.iloc[test_idx]
    zeros = np.zeros(len(test_idx), dtype=float)
    if train.empty or cfg.bench_model == "none":
        return {
            "bench_play_prob": zeros,
            "bench_high_minutes_prob": zeros,
            "bench_low_minutes_prob": np.ones(len(test_idx), dtype=float),
            "bench_minutes_if_on": zeros,
            "bench_expected_minutes": zeros,
            "bench_minutes_uncertainty": np.ones(len(test_idx), dtype=float),
        }
    Xtr, Xte = _align_design(train, test, cfg)
    play_y = (train["minutes"].to_numpy(float) >= 10.0).astype(float)
    play_prob = np.clip(_bench_model_predict(cfg, Xtr, play_y, Xte), 0.0, 1.0)
    high_threshold = float(max(10.0, cfg.bench_high_minutes_threshold))
    high_y = (train["minutes"].to_numpy(float) >= high_threshold).astype(float)
    high_prob = np.clip(_bench_model_predict(cfg, Xtr, high_y, Xte), 0.0, 1.0)
    low_prob = np.clip(1.0 - play_prob, 0.0, 1.0)

    on = train["minutes"].to_numpy(float) >= 10.0
    if on.sum() >= 20:
        Xon, Xte_on = _align_design(train.loc[on], test, cfg)
        minutes_if_on_y = train.loc[on, "minutes"].to_numpy(float)
        minutes_if_on = np.clip(
            _bench_model_predict(cfg, Xon, minutes_if_on_y, Xte_on), 0.0, 80.0)
    else:
        minutes_if_on = np.full(len(test_idx), float(train["minutes"].mean()))
    expected_minutes = np.clip(play_prob * minutes_if_on, 0.0, 80.0)
    return {
        "bench_play_prob": play_prob,
        "bench_high_minutes_prob": high_prob,
        "bench_low_minutes_prob": low_prob,
        "bench_minutes_if_on": minutes_if_on,
        "bench_expected_minutes": expected_minutes,
        "bench_minutes_uncertainty": (
            play_prob * (1.0 - play_prob)
            + np.abs(expected_minutes - df.iloc[test_idx]["form_minutes_recent"].fillna(0).to_numpy(float)) / 80.0
        ),
    }


def _learned_bench_points(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
    minutes_signals: dict[str, np.ndarray],
) -> np.ndarray:
    if cfg.bench_model == "none":
        return pred["target_pts_hat"].to_numpy(float).copy()
    context_df = _bench_context_frame(df, cfg)
    train = _bench_train_rows(df, train_idx, cfg)
    if train.empty:
        return pred["target_pts_hat"].to_numpy(float).copy()
    if cfg.bench_model.startswith("two_stage_"):
        minutes_hat = np.maximum(pred["minutes_hat"].to_numpy(float), 1.0)
        return np.clip(
            pred["target_pts_hat"].to_numpy(float)
            * minutes_signals["bench_expected_minutes"] / minutes_hat,
            0.0,
            None,
        )

    y = _bench_target_points(train)
    keep = np.isfinite(y)
    train = train.loc[keep]
    y = y[keep]
    if len(train) < 20:
        return pred["target_pts_hat"].to_numpy(float).copy()
    Xtr, Xte = _align_design(train, context_df.iloc[test_idx], cfg)
    return np.clip(_bench_model_predict(cfg, Xtr, y, Xte), 0.0, None)


def _learned_bench_kick_points(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, cfg: Config
) -> np.ndarray:
    context_df = _bench_context_frame(df, cfg)
    train = _bench_train_rows(df, train_idx, cfg)
    if train.empty:
        return np.zeros(len(test_idx), dtype=float)
    y = (
        2.0 * train["y_conversion_goals"].fillna(0).to_numpy(float)
        + 3.0 * train["y_penalty_goals"].fillna(0).to_numpy(float)
    )
    if np.allclose(y, 0.0):
        return np.zeros(len(test_idx), dtype=float)
    Xtr, Xte = _align_design(train, context_df.iloc[test_idx], cfg)
    if cfg.bench_kick_model == "ridge":
        pred = _ridge_predict(Xtr, y, Xte, cfg.bench_alpha)
    elif cfg.bench_kick_model == "lgbm":
        pred = _lgbm_reg_predict(Xtr, y, Xte)
    else:
        raise ValueError(f"unknown bench_kick_model {cfg.bench_kick_model!r}")
    return np.clip(pred, 0.0, None)


def _apply_bench_head(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if not _needs_bench_head(cfg):
        return pred
    out = pred.copy()
    minutes_signals = _bench_minutes_signals(df, train_idx, test_idx, cfg)
    for col, vals in minutes_signals.items():
        out[col] = vals
    bench_hat = _learned_bench_points(df, train_idx, test_idx, out, cfg, minutes_signals)
    current_kick = (
        2.0 * out.get("hat_conversion_goals", 0.0)
        + 3.0 * out.get("hat_penalty_goals", 0.0)
    )
    current_kick = np.asarray(current_kick, dtype=float)
    if cfg.bench_kick_model == "shrink":
        out["bench_kick_pts_hat"] = np.clip(
            current_kick * (1.0 - np.clip(cfg.bench_kick_shrink, 0.0, 1.0)),
            0.0,
            None,
        )
    elif cfg.bench_kick_model in {"ridge", "lgbm"}:
        out["bench_kick_pts_hat"] = _learned_bench_kick_points(
            df, train_idx, test_idx, cfg)
    else:
        out["bench_kick_pts_hat"] = current_kick
    if cfg.bench_kick_model != "none":
        bench_hat = np.clip(bench_hat + out["bench_kick_pts_hat"].to_numpy(float) - current_kick, 0.0, None)
    out["bench_pts_hat"] = bench_hat
    if (
        cfg.bench_supersub_context_features != "off"
        or cfg.bench_supersub_replacement_features
        or cfg.bench_supersub_history_features
        or cfg.bench_supersub_team_history_features
    ):
        supersub_cfg = cfg.delta(
            bench_context_features=cfg.bench_supersub_context_features,
            bench_supersub_context_features="off",
            bench_replacement_features=cfg.bench_supersub_replacement_features,
            bench_supersub_replacement_features=False,
            bench_history_features=cfg.bench_supersub_history_features,
            bench_supersub_history_features=False,
            bench_team_history_features=cfg.bench_supersub_team_history_features,
            bench_supersub_team_history_features=False,
        )
        supersub_minutes = _bench_minutes_signals(
            df, train_idx, test_idx, supersub_cfg
        )
        out["supersub_bench_pts_hat"] = _learned_bench_points(
            df,
            train_idx,
            test_idx,
            out,
            supersub_cfg,
            supersub_minutes,
        )
    if cfg.bench_points_weight > 0:
        w = float(np.clip(cfg.bench_points_weight, 0.0, 1.0))
        bench_mask = ~df.iloc[test_idx]["started"].astype(bool).to_numpy()
        target = out["target_pts_hat"].to_numpy(float).copy()
        target[bench_mask] = (
            (1.0 - w) * target[bench_mask] + w * bench_hat[bench_mask]
        )
        out["target_pts_hat"] = np.clip(target, 0.0, None)
    return out


def _needs_upside(cfg: Config) -> bool:
    return (cfg.selector_upside_weight > 0 or cfg.captain_upside_weight > 0
            or cfg.supersub_upside_weight > 0)


def _prior_player_variance(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = df.iloc[train_idx].copy()
    target = _bench_target_points(train)
    train = train.assign(_target_for_var=target)
    player_std = train.groupby("player_id")["_target_for_var"].std()
    player_n = train.groupby("player_id")["_target_for_var"].size()
    player_std = player_std.where(player_n >= 3)
    pos_std = train.groupby("canonical_pos")["_target_for_var"].std()
    pos_ceiling = (
        train.groupby("canonical_pos")["_target_for_var"].quantile(0.80)
        - train.groupby("canonical_pos")["_target_for_var"].mean()
    )
    te = df.iloc[test_idx]
    std = te["player_id"].map(player_std)
    std = std.fillna(te["canonical_pos"].map(pos_std)).fillna(float(np.nanstd(target)))
    ceiling = te["canonical_pos"].map(pos_ceiling).fillna(float(np.nanstd(target)))
    return std.to_numpy(float), ceiling.to_numpy(float)


def _apply_upside_head(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pred: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if not _needs_upside(cfg):
        return pred
    out = pred.copy()
    try_pts = np.where(out["is_forward"].to_numpy(bool), 15.0, 10.0) * out["hat_tries"].to_numpy(float)
    attack = (
        try_pts
        + 4.0 * out["hat_try_assists"].to_numpy(float)
        + 2.0 * out["hat_defenders_beaten"].to_numpy(float)
        + 2.0 * out["hat_offload"].to_numpy(float)
        + np.floor(out["hat_metres"].to_numpy(float) / 10.0)
    )
    disagreement = np.zeros(len(out), dtype=float)
    if "lgbm_target_pts_hat" in out.columns and "xgb_component_pts_hat" in out.columns:
        disagreement = np.abs(
            out["lgbm_target_pts_hat"].to_numpy(float)
            - out["xgb_component_pts_hat"].to_numpy(float)
        )
    player_std, pos_ceiling = _prior_player_variance(df, train_idx, test_idx)
    raw = (
        _zscore(attack)
        + 0.50 * _zscore(disagreement)
        + 0.50 * _zscore(player_std)
        + 0.25 * _zscore(pos_ceiling)
    )
    eligible = np.ones(len(out), dtype=bool)
    positions = _scope_positions(cfg.upside_scope)
    if positions is not None:
        eligible = df.iloc[test_idx]["canonical_pos"].isin(positions).to_numpy()
    scoped = np.zeros(len(out), dtype=float)
    if eligible.any():
        scoped[eligible] = _zscore(raw[eligible])
    out["upside_score"] = scoped
    return out


def config_from_dict(d: dict) -> "Config":
    """Build a Config, dropping keys it no longer knows about.

    Archived promoted_config.prev_*.json files still carry the removed betting-market
    knobs (market_features, post_target_market_features, captain_market_weight). They
    are historical records; loading one must not explode.
    """
    known = {f.name for f in dataclasses.fields(Config)}
    dropped = sorted(set(d) - known)
    if dropped:
        print(f"  note: ignoring retired config keys: {', '.join(dropped)}")
    return Config(**{k: v for k, v in d.items() if k in known})


def _predict_config(
    df: pd.DataFrame, cfg: Config, season: int,
    train_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """Build predictions for a fixed config/season without evaluating acceptance.

    `train_mask` overrides the default `season < test_season` component-training
    set.  Used by the enhanced validation harness (validate.py) for the
    2026-OOF / cross-year views; production paths pass None and keep the
    forward-chained split.  The override must never include the test season."""
    if train_mask is None:
        train_mask = component_train(df, season)
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
        if df.loc[train_mask, "season"].eq(season).any():
            raise AssertionError("custom train_mask leaked the test season")
    train_idx = np.where(train_mask)[0]
    test_idx = np.where((df["season"] == season).to_numpy())[0]
    registry = None if cfg.deploy_engine == "lgbm_only" else _registry_for(df, train_mask, cfg)

    with apply_config(cfg):
        minutes_hat = build_minutes(df, train_idx, test_idx, cfg)

        def predictor(d, ti, te, mo):
            with _teamplay_mode(cfg.teamplay_component_features):
                if cfg.deploy_engine == "lgbm_only":
                    return predict_rates_lgbm_only(d, ti, te, mo)
                return predict_rates_registry(d, ti, te, mo, registry)

        pred, diag = assemble_predictions(
            df, train_idx, season, MODE, predictor, minutes_hat=minutes_hat)
        sub = df.iloc[test_idx]
        for meta_col in (
            "team", "opponent", "started", "jersey",
            "rolecert_goal_kicker_prob",
            "rolecert_team_kicker_confidence",
            "rolecert_bench_fh_kick_share",
            "rolecert_bench_kick_takeover_risk",
            "rolecert_replacement_role_uncertainty",
        ):
            if meta_col in sub.columns:
                pred[meta_col] = sub[meta_col].to_numpy()
        pred = _apply_teamplay_aspect_adjustments(df, test_idx, pred, cfg)
        pred = _apply_weather_aspect_adjustments(df, test_idx, pred, cfg)
        pred = _graft_teamplay_components(
            df, train_mask, train_idx, season, minutes_hat, pred, cfg)
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
        pred = _apply_kicking_reallocation(df, test_idx, pred, cfg)
        pred = _apply_rolecert_kicking_adjustments(df, test_idx, pred, cfg)

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
        pred = _apply_bench_head(df, train_idx, test_idx, pred, cfg)
        if cfg.selector_point_source == "pre_calib":
            pred = pred.copy()
            pred["selector_pts_hat"] = pre_calib_target
        else:
            pred = pred.copy()
            pred["selector_pts_hat"] = pred["target_pts_hat"].to_numpy(float)
        pred = _apply_upside_head(df, train_idx, test_idx, pred, cfg)
        pred["lgbm_rank_score"] = rank_scores(df, train_idx, test_idx, MODE)

        pred, sel_col = _add_selector_score(df, train_idx, test_idx, pred, cfg)
        teamplay_pred = None
        if cfg.post_target_teamplay_weight > 0 or cfg.post_target_forward_combiner != "none":
            inner_cfg = cfg.delta(
                teamplay_features=cfg.post_target_teamplay_mode,
                teamplay_component_features=cfg.post_target_teamplay_component_features,
                matchup_features=(
                    cfg.post_target_matchup_features or cfg.matchup_features
                ),
                teamplay_graft_mode=cfg.post_target_teamplay_graft_mode,
                teamplay_graft_group=cfg.post_target_teamplay_graft_group,
                teamplay_graft_weight=cfg.post_target_teamplay_graft_weight,
                post_target_teamplay_weight=0.0,
                post_target_residual_weight=0.0,
                post_target_residual_group="none",
                post_target_selector_delta_weight=0.0,
                post_target_selector_xgb_weight=-1.0,
                post_target_selector_delta_min_prior_n=0,
                post_target_forward_combiner="none",
            )
            teamplay_pred, _, _ = _predict_config(df, inner_cfg, season, train_mask=train_mask)
        selector_signal = None
        if cfg.post_target_selector_xgb_weight >= 0:
            selector_cfg = cfg.delta(
                post_target_xgb_weight=cfg.post_target_selector_xgb_weight,
                post_target_selector_xgb_weight=-1.0,
                post_target_selector_delta_weight=0.0,
            )
            selector_pred, _, _ = _predict_config(df, selector_cfg, season, train_mask=train_mask)
            selector_signal = selector_pred["target_pts_hat"].to_numpy(float)
        if cfg.post_target_selector_matchup_features is not None:
            if selector_signal is not None:
                raise ValueError(
                    "only one alternate post-target selector signal may be configured"
                )
            selector_cfg = cfg.delta(
                post_target_matchup_features=bool(
                    cfg.post_target_selector_matchup_features
                ),
                post_target_selector_matchup_features=None,
            )
            selector_pred, _, _ = _predict_config(df, selector_cfg, season, train_mask=train_mask)
            selector_signal = selector_pred["target_pts_hat"].to_numpy(float)
        post_target_input = pred
        pred = _apply_post_target_pipeline(
            post_target_input, xgb_pred, bayes_pred, teamplay_pred, cfg
        )
        if selector_signal is not None:
            pred["post_target_selector_signal_hat"] = selector_signal
        pred, sel_col = _refresh_post_target_role_scores(pred, cfg)

    if (
        cfg.supersub_latent_shrink >= 0.0
        and not np.isclose(cfg.supersub_latent_shrink, cfg.latent_shrink)
        and "supersub_score" in pred.columns
    ):
        # Rebuild the supersub head from a separately-shrunk set-piece latent.
        # The inner config disables further decoupling to avoid recursion.
        ss_cfg = cfg.delta(
            latent_shrink=cfg.supersub_latent_shrink, supersub_latent_shrink=-1.0)
        ss_pred, _, _ = _predict_config(df, ss_cfg, season, train_mask=train_mask)
        # Same season -> identical test_idx ordering, so positional assignment aligns.
        pred = pred.copy()
        pred["supersub_score"] = ss_pred["supersub_score"].to_numpy(float)

    return pred, sel_col, registry
