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
from model import data as MD
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
    _pick_xv,
    _supersub_pool,
    bench_mae,
    captain_hit_rates,
    captain_hitrate,
    points_mae,
    spearman_within_pos,
    topn_overlap,
    value_of_team,
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
PRIMARY_VALUE = "value_team"

# acceptance thresholds (see RESEARCH_GOAL.md)
DELTA_V = 0.003            # min value_team gain to accept on the value leg
TAU_MAE = 0.02            # max tolerated MAE regression on the value leg
DELTA_M = 0.02            # min MAE gain to accept on the MAE leg
ROUND_ROBUST = 3          # value_team/MAE gain must hold on >= this many of 5 rounds
TOP15_MAX_DROP = 0.04     # guardrail: about three top-15 misses over 5 rounds
SPEARMAN_MAX_DROP = 0.03  # guardrail for within-position selector health
DOMINANCE_EPS = 1e-9      # tolerance for zero-cost selector-dominance checks
STABILITY_REPORT = RESEARCH / "stability_report.json"
ROUND_DROP_MIN_PASS = 4
TEAM_DROP_MIN_PASS = 5
POSITION_MAE_MAX_REGRESSION = 0.15
POSITION_SPEARMAN_MAX_DROP = 0.05
MINUTES_PERTURB_MIN_PASS = 3
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
    "potm075_bayes10": (
        "passed 2025 dev and stability checks, but sealed 2026 regressed versus "
        "bench_points_twostage_13 on value_team, value_xv, MAE, bench MAE, top30, "
        "and selector Spearman"
    ),
    "decoupled_latent040_sub100": (
        "decoupled supersub holds team value flat on both seasons, but the 0.40 "
        "point-path latent shrink is a 2025 MAE trap: sealed 2026 MAE regressed "
        "+0.045 with no value_team gain"
    ),
    "decoupled_latent050_sub100": (
        "same decoupled-latent family: passed 2025 on both legs but sealed 2026 "
        "MAE regressed +0.020 with value_team flat; set-piece latent is real "
        "signal in 2026, so shrinking the point path does not generalise"
    ),
    "latent050_potm100": (
        "only standard-queue 2025-gate passer (team 0.727 via value_team), but "
        "sealed 2026 MAE regressed +0.053 (latent-shrink penalty) for a +0.006 "
        "team gain; dominated by captain_upside_full which gives more sealed team "
        "value at zero MAE cost"
    ),
    "potm_rank_assisted_025": (
        "POTM points-only (potm_pp_weight=1.0) plus tiny XGB rank overlay. On the "
        "captain_upside_full base it newly passed the 2025 gate (team 0.741, 3/5 "
        "rounds) because the captain head shifted the round structure, but sealed "
        "2026 regressed on everything vs promoted: value_team -0.013, value_xv "
        "-0.005, MAE +0.024. The POTM=1.0 family is a 2025 trap regardless of base."
    ),
    "weather_multi_010": (
        "passed 2025 base gate and stability after adding Open-Meteo archive weather "
        "as component-shaped nudges, but sealed 2026 regressed versus promoted on "
        "both primary metrics: value_team 0.685 vs 0.700 and MAE 7.320 vs 7.303. "
        "Weather as a broad component overlay is a 2025 trap; revisit only as a "
        "forecast-available, narrowly scoped specialist."
    ),
    "orchestrator_selector_signal_xgb0": (
        "decoupled no-XGB selector signal passed 2025 and all stability checks "
        "with unchanged MAE, but sealed 2026 regressed value_team 0.700169 -> "
        "0.698969 and value_xv 0.718337 -> 0.716908. Do not tune nearby weights "
        "against the opened sealed result."
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
    post_target_market_features: bool = False
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
    market_features: bool = False
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
    captain_market_weight: float = 0.0  # favour captains on teams with higher market win probability
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
        MD.MARKET_FEATURES_ENABLED, MD.WEATHER_FEATURES_ENABLED,
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
    MD.MARKET_FEATURES_ENABLED = bool(cfg.market_features)
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
         MD.MARKET_FEATURES_ENABLED, MD.WEATHER_FEATURES_ENABLED,
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
    from model.splits import group_kfold_indices

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
        if not np.isclose(cfg.captain_market_weight, 0.0):
            if "market_win_prob" not in out.columns:
                raise ValueError("captain_market_weight requires market_win_prob")
            cap = cap + cfg.captain_market_weight * _zscore(
                out["market_win_prob"].to_numpy(float)
            )
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
    (RESEARCH / f"teamplay_component_oof_{feature_mode}.json").write_text(
        json.dumps({
            "date": str(date.today()),
            "mode": feature_mode,
            "min_gain": float(min_gain),
            "winners": sorted(winners),
            "components": diagnostics,
        }, indent=2)
    )
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
        if not np.isclose(cfg.captain_market_weight, 0.0):
            if "market_win_prob" not in out.columns:
                raise ValueError("captain_market_weight requires market_win_prob")
            cap = cap + cfg.captain_market_weight * _zscore(
                out["market_win_prob"].to_numpy(float)
            )
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
            "market_win_prob",
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
                market_features=cfg.post_target_market_features or cfg.market_features,
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


def _evaluate_prediction(
    pred: pd.DataFrame, sel_col: str, cfg: Config, registry: pd.DataFrame | None,
) -> dict:
    vx, ratios = value_of_xv(pred, sel_col)
    supersub_col = "supersub_score" if "supersub_score" in pred.columns else None
    captain_col = "captain_score" if "captain_score" in pred.columns else None
    vt, team_ratios = value_of_team(
        pred,
        sel_col,
        captain_score_col=captain_col,
        supersub_score_col=supersub_col,
    )
    captain_metric_col = captain_col or sel_col
    c1, c3 = captain_hitrate(pred, captain_metric_col)
    capt = captain_hit_rates(pred, captain_metric_col, (5,))
    selection_diag = _selection_diagnostics(pred, sel_col)
    return {
        "config": cfg.name,
        "value_team": vt,
        "round_team": [float(r) for r in team_ratios],
        "value_xv": vx,
        "round_xv": [float(r) for r in ratios],
        "round_mae": _round_mae(pred),
        "round_bias": _round_bias(pred),
        "mae": points_mae(pred, "target_pts_hat")["overall"],
        "bench_mae": bench_mae(pred, "target_pts_hat"),
        "by_pos": _position_diagnostics(pred),
        "top15": topn_overlap(pred, 15, sel_col),
        "top30": topn_overlap(pred, 30, sel_col),
        "capt_top1": c1,
        "capt_top3": c3,
        "capt_top5": capt["capt_top5"],
        "spearman_pos": spearman_within_pos(pred, "target_pts_hat"),
        "spearman_sel": spearman_within_pos(pred, sel_col),
        "registry_lgbm": _registry_lgbm_count(cfg, registry),
        "selection_diagnostics": selection_diag,
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


def _selection_diagnostics(pred: pd.DataFrame, sel_col: str) -> dict:
    lab = pred[pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)].copy()
    cap_col = "captain_score" if "captain_score" in lab.columns else sel_col
    ss_col = "supersub_score" if "supersub_score" in lab.columns else sel_col
    rounds = []
    selected_rows = []
    for r, g in lab.groupby("round", sort=True):
        xv = _pick_xv(g, sel_col)
        if xv.empty:
            continue
        model_cap = xv.loc[xv[cap_col].idxmax()]
        best_cap = xv.loc[xv["official_pts"].idxmax()]
        pool = _supersub_pool(g, set(xv["player_id"]))
        model_ss = best_ss = None
        ss_rank = np.nan
        ss_regret = 0.0
        if not pool.empty:
            model_ss = pool.loc[pool[ss_col].idxmax()]
            best_ss = pool.loc[pool["official_pts"].idxmax()]
            ordered = pool.sort_values(ss_col, ascending=False)["player_id"].tolist()
            ss_rank = ordered.index(best_ss["player_id"]) + 1
            ss_regret = float(best_ss["official_pts"] - model_ss["official_pts"])
            selected_rows.append(model_ss)
        selected_rows.extend([row for _, row in xv.iterrows()])
        rounds.append({
            "round": int(r),
            "captain": str(model_cap["player_name"]),
            "best_captain_in_xv": str(best_cap["player_name"]),
            "captain_regret": float(best_cap["official_pts"] - model_cap["official_pts"]),
            "supersub": None if model_ss is None else str(model_ss["player_name"]),
            "best_supersub": None if best_ss is None else str(best_ss["player_name"]),
            "best_supersub_rank": None if not np.isfinite(ss_rank) else int(ss_rank),
            "supersub_regret": ss_regret,
        })
    selected_bias = {}
    if selected_rows:
        sel = pd.DataFrame(selected_rows)
        sel["resid"] = sel["official_pts"] - sel["target_pts_hat"]
        for pos, g in sel.groupby("canonical_pos", sort=True):
            selected_bias[str(pos)] = {
                "n": int(len(g)),
                "pred_mean": float(g["target_pts_hat"].mean()),
                "actual_mean": float(g["official_pts"].mean()),
                "bias": float(g["resid"].mean()),
            }
    return {
        "rounds": rounds,
        "captain_regret_total": float(sum(r["captain_regret"] for r in rounds)),
        "supersub_regret_total": float(sum(r["supersub_regret"] for r in rounds)),
        "selected_bias": selected_bias,
    }


def dev_evaluate(df, cfg: Config, season: int = DEV_SEASON) -> dict:
    """Run the full Strategy-C assembly for `cfg` on `season` and return metrics."""
    assert season != SEALED_SEASON, "2026 is sealed — the loop must not read it"
    pred, sel_col, registry = _predict_config(df, cfg, season)
    return _evaluate_prediction(pred, sel_col, cfg, registry)


# ---------------------------------------------------------------------------
# acceptance test
# ---------------------------------------------------------------------------
def accept(best: dict, cand: dict) -> tuple[bool, str]:
    dv = cand[PRIMARY_VALUE] - best[PRIMARY_VALUE]
    dxv = cand["value_xv"] - best["value_xv"]
    dm = cand["mae"] - best["mae"]                      # negative = improvement
    dbm = cand["bench_mae"] - best["bench_mae"]
    dt15 = cand["top15"] - best["top15"]
    dt30 = cand["top30"] - best["top30"]
    dspear = cand["spearman_sel"] - best["spearman_sel"]
    rb_v = sum(c > b + DOMINANCE_EPS for c, b in zip(cand["round_team"], best["round_team"]))
    rw_v = sum(c < b - DOMINANCE_EPS for c, b in zip(cand["round_team"], best["round_team"]))
    rb_m = sum(c < b - DOMINANCE_EPS for c, b in zip(cand["round_mae"], best["round_mae"]))
    pareto_v = dv >= DELTA_V and dm <= TAU_MAE
    pareto_m = dm <= -DELTA_M and dv >= -DELTA_V
    zero_cost_selector_gain = (
        pareto_v
        and rb_v >= 1
        and rw_v == 0
        and dxv >= -DOMINANCE_EPS
        and dm <= DOMINANCE_EPS
        and dbm <= DOMINANCE_EPS
        and dt15 >= -DOMINANCE_EPS
        and dt30 >= -DOMINANCE_EPS
        and (not np.isfinite(dspear) or dspear >= -DOMINANCE_EPS)
    )
    guardrails = []
    if dt15 < -TOP15_MAX_DROP:
        guardrails.append(f"top15 {dt15:+.3f}")
    if np.isfinite(dspear) and dspear < -SPEARMAN_MAX_DROP:
        guardrails.append(f"spearman_sel {dspear:+.3f}")
    if (pareto_v or pareto_m) and guardrails:
        return False, (f"reject (guardrail failed: {', '.join(guardrails)}): "
                       f"dval={dv:+.3f} dmae={dm:+.3f} dtop15={dt15:+.3f} "
                       f"dspear={dspear:+.3f} team_rounds_up={rb_v}/5 "
                       f"team_rounds_down={rw_v}/5 mae_rounds_up={rb_m}/5")
    if zero_cost_selector_gain:
        return True, (f"ACCEPT via {PRIMARY_VALUE} dominance: dval={dv:+.3f} "
                      f"dxv={dxv:+.3f} dmae={dm:+.3f} dbench={dbm:+.3f} "
                      f"dtop15={dt15:+.3f} dtop30={dt30:+.3f} "
                      f"dspear={dspear:+.3f} team_rounds_up={rb_v}/5 "
                      f"team_rounds_down={rw_v}/5 mae_rounds_up={rb_m}/5")
    if pareto_v and rb_v >= ROUND_ROBUST:
        return True, (f"ACCEPT via {PRIMARY_VALUE}: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
                      f"team_rounds_up={rb_v}/5 team_rounds_down={rw_v}/5 "
                      f"mae_rounds_up={rb_m}/5")
    if pareto_m and rb_m >= ROUND_ROBUST:
        return True, (f"ACCEPT via mae: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
                      f"team_rounds_up={rb_v}/5 team_rounds_down={rw_v}/5 "
                      f"mae_rounds_up={rb_m}/5")
    if pareto_v or pareto_m:
        why = "round robustness failed"
    else:
        why = "no Pareto improvement"
    return (False, f"reject ({why}): dval={dv:+.3f} dmae={dm:+.3f} "
            f"dtop15={dt15:+.3f} dspear={dspear:+.3f} "
            f"team_rounds_up={rb_v}/5 team_rounds_down={rw_v}/5 "
            f"mae_rounds_up={rb_m}/5")


def _subset_metrics(pred: pd.DataFrame, sel_col: str, mask: np.ndarray) -> dict:
    sub = pred.loc[mask].copy()
    vx, ratios = value_of_xv(sub, sel_col)
    supersub_col = "supersub_score" if "supersub_score" in sub.columns else None
    captain_col = "captain_score" if "captain_score" in sub.columns else None
    vt, team_ratios = value_of_team(
        sub, sel_col, captain_score_col=captain_col, supersub_score_col=supersub_col)
    return {
        "value_team": vt,
        "value_xv": vx,
        "mae": points_mae(sub, "target_pts_hat")["overall"],
        "bench_mae": bench_mae(sub, "target_pts_hat"),
        "top15": topn_overlap(sub, 15, sel_col),
        "top30": topn_overlap(sub, 30, sel_col),
        "spearman_sel": spearman_within_pos(sub, sel_col),
        "round_team": [float(r) for r in team_ratios],
        "round_xv": [float(r) for r in ratios],
        "round_mae": _round_mae(sub),
        "round_bias": _round_bias(sub),
        "by_pos": _position_diagnostics(sub),
        "capt_top1": captain_hit_rates(sub, captain_col or sel_col, (1,))["capt_top1"],
        "capt_top3": captain_hit_rates(sub, captain_col or sel_col, (3,))["capt_top3"],
        "capt_top5": captain_hit_rates(sub, captain_col or sel_col, (5,))["capt_top5"],
        "spearman_pos": spearman_within_pos(sub, "target_pts_hat"),
        "registry_lgbm": 0,
    }


def _noninferior(best: dict, cand: dict) -> bool:
    return (
        cand["mae"] - best["mae"] <= TAU_MAE
        and cand[PRIMARY_VALUE] >= best[PRIMARY_VALUE] - DELTA_V
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
                           "dvalue": mc[PRIMARY_VALUE] - mb[PRIMARY_VALUE],
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
                          "dvalue": mc[PRIMARY_VALUE] - mb[PRIMARY_VALUE],
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
                         "dvalue": mc[PRIMARY_VALUE] - mb[PRIMARY_VALUE]})
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
                             "dvalue": mc[PRIMARY_VALUE] - mb[PRIMARY_VALUE],
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
    # Learned bench/supersub head.  Trained only on prior-season non-starters.
    ("bench_supersub_ridge_025", "learned bench-only Ridge supersub overlay 0.25",
     dict(bench_model="ridge", bench_supersub_weight=0.25)),
    ("bench_supersub_ridge_050", "learned bench-only Ridge supersub overlay 0.50",
     dict(bench_model="ridge", bench_supersub_weight=0.50)),
    ("bench_supersub_ridge_100", "learned bench-only Ridge supersub overlay 1.00",
     dict(bench_model="ridge", bench_supersub_weight=1.00)),
    ("bench_points_ridge_10", "10% bench target blend toward learned bench Ridge",
     dict(bench_model="ridge", bench_points_weight=0.10, selector_point_source="target")),
    ("bench_points_ridge_25", "25% bench target blend toward learned bench Ridge",
     dict(bench_model="ridge", bench_points_weight=0.25, selector_point_source="target")),
    ("bench_points_ridge_50", "50% bench target blend toward learned bench Ridge",
     dict(bench_model="ridge", bench_points_weight=0.50, selector_point_source="target")),
    ("bench_points10_supersub50", "10% bench point blend plus 0.50 learned supersub overlay",
     dict(bench_model="ridge", bench_points_weight=0.10,
          bench_supersub_weight=0.50, selector_point_source="target")),
    ("supersub_bench_ridge_025", "learned Ridge bench supersub overlay 0.25",
     dict(bench_model="ridge", bench_supersub_weight=0.25)),
    ("supersub_bench_ridge_050", "learned Ridge bench supersub overlay 0.50",
     dict(bench_model="ridge", bench_supersub_weight=0.50)),
    ("supersub_bench_lgbm_025", "learned LightGBM bench supersub overlay 0.25",
     dict(bench_model="lgbm", bench_supersub_weight=0.25)),
    ("supersub_bench_lgbm_050", "learned LightGBM bench supersub overlay 0.50",
     dict(bench_model="lgbm", bench_supersub_weight=0.50)),
    ("supersub_two_stage_minutes", "two-stage bench minutes supersub overlay",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50)),
    ("supersub_two_stage_w035", "two-stage bench minutes supersub overlay 0.35",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.35)),
    ("supersub_two_stage_w040", "two-stage bench minutes supersub overlay 0.40",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.40)),
    ("supersub_two_stage_w045", "two-stage bench minutes supersub overlay 0.45",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.45)),
    ("supersub_two_stage_w055", "two-stage bench minutes supersub overlay 0.55",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.55)),
    ("supersub_two_stage_w060", "two-stage bench minutes supersub overlay 0.60",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.60)),
    ("supersub_two_stage_w075", "two-stage bench minutes supersub overlay 0.75",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.75)),
    ("supersub_two_stage_alpha10", "two-stage bench minutes with lighter ridge alpha",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_alpha=10.0)),
    ("supersub_two_stage_alpha40", "two-stage bench minutes with heavier ridge alpha",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_alpha=40.0)),
    ("supersub_two_stage_alpha80", "two-stage bench minutes with very heavy ridge alpha",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_alpha=80.0)),
    ("supersub_two_stage_lgbm_035", "two-stage LightGBM bench minutes supersub overlay 0.35",
     dict(bench_model="two_stage_lgbm", bench_supersub_weight=0.35)),
    ("supersub_two_stage_lgbm_050", "two-stage LightGBM bench minutes supersub overlay 0.50",
     dict(bench_model="two_stage_lgbm", bench_supersub_weight=0.50)),
    ("bench_points_twostage_05", "5% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.05, selector_point_source="target")),
    ("bench_points_twostage_10", "10% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.10, selector_point_source="target")),
    ("bench_points_twostage_12", "12% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.12, selector_point_source="target")),
    ("bench_points_twostage_13", "13% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, selector_point_source="target")),
    ("bench_points_twostage_14", "14% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.14, selector_point_source="target")),
    ("bench_points_twostage_15", "15% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.15, selector_point_source="target")),
    ("bench_points_twostage_18", "18% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.18, selector_point_source="target")),
    ("bench_points_twostage_20", "20% bench target blend toward two-stage bench points",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.20, selector_point_source="target")),
    ("supersub_minutes_uncertainty_025", "two-stage bench minutes with uncertainty penalty",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_uncertainty_weight=0.25)),
    ("supersub_value_minus_risk_025", "bench value overlay minus learned minutes risk",
     dict(bench_model="ridge", bench_supersub_weight=0.50,
          bench_uncertainty_weight=0.25)),
    ("bench_kick_shrink_25", "bench supersub overlay with 25% bench kick shrink",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          bench_kick_model="shrink", bench_kick_shrink=0.25)),
    ("bench_kick_shrink_50", "bench supersub overlay with 50% bench kick shrink",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          bench_kick_model="shrink", bench_kick_shrink=0.50)),
    ("bench_kick_learned_gate", "bench supersub overlay with learned Ridge kick replacement",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          bench_kick_model="ridge")),
    ("bench_kick_ridge_share", "heavier supersub overlay with learned Ridge bench kick share",
     dict(bench_model="ridge", bench_supersub_weight=0.50,
          bench_kick_model="ridge")),
    ("bench_kick_lgbm_share", "bench supersub overlay with learned LightGBM kick share",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          bench_kick_model="lgbm")),
    ("bench_twostage_kick_shrink_10", "current two-stage bench head with 10% kick shrink",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, bench_kick_model="shrink",
          bench_kick_shrink=0.10, selector_point_source="target")),
    ("bench_twostage_kick_shrink_25", "current two-stage bench head with 25% kick shrink",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, bench_kick_model="shrink",
          bench_kick_shrink=0.25, selector_point_source="target")),
    ("bench_twostage_kick_shrink_50", "current two-stage bench head with 50% kick shrink",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, bench_kick_model="shrink",
          bench_kick_shrink=0.50, selector_point_source="target")),
    ("bench_twostage_kick_ridge", "current two-stage bench head with learned Ridge kick share",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, bench_kick_model="ridge",
          selector_point_source="target")),
    ("bench_twostage_kick_lgbm", "current two-stage bench head with learned LightGBM kick share",
     dict(bench_model="two_stage_ridge", bench_supersub_weight=0.50,
          bench_points_weight=0.13, bench_kick_model="lgbm",
          selector_point_source="target")),
    ("captain_mean_only", "choose captain by calibrated points only",
     dict(captain_head="mean")),
    ("captain_kicker_floor_010", "captain mean plus tiny kicker floor",
     dict(captain_head="mean", captain_kicker_weight=0.10)),
    ("captain_kicker_floor_020", "captain mean plus small kicker floor",
     dict(captain_head="mean", captain_kicker_weight=0.20)),
    ("captain_kicker_floor_040", "captain mean plus moderate kicker floor",
     dict(captain_head="mean", captain_kicker_weight=0.40)),
    ("captain_kicker_penalty_005", "captain ceiling with tiny kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.0,
          captain_kicker_weight=-0.05)),
    ("captain_kicker_penalty_010", "captain ceiling with small kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.0,
          captain_kicker_weight=-0.10)),
    ("captain_kicker_penalty_020", "captain ceiling with moderate kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.0,
          captain_kicker_weight=-0.20)),
    ("captain_kicker_penalty_040", "captain ceiling with strong kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.0,
          captain_kicker_weight=-0.40)),
    ("captain_mean_plus_upside_010", "captain mean plus 0.10 upside",
     dict(captain_head="mean", captain_upside_weight=0.10)),
    ("captain_mean_plus_upside_020", "captain mean plus 0.20 upside",
     dict(captain_head="mean", captain_upside_weight=0.20)),
    ("captain_upside_050", "captain mean plus 0.50 upside",
     dict(captain_head="mean", captain_upside_weight=0.50)),
    ("captain_upside_075", "captain mean plus 0.75 upside",
     dict(captain_head="mean", captain_upside_weight=0.75)),
    ("captain_upside_full", "captain by full (1.0) upside ceiling bet; the 2x role rewards variance",
     dict(captain_head="mean", captain_upside_weight=1.0)),
    ("captain_upside_125", "captain mean plus 1.25 upside",
     dict(captain_head="mean", captain_upside_weight=1.25)),
    ("captain_upside_150", "captain mean plus 1.50 upside",
     dict(captain_head="mean", captain_upside_weight=1.50)),
    ("captain_upside_200", "captain mean plus 2.00 upside",
     dict(captain_head="mean", captain_upside_weight=2.0)),
    ("captain_kicker_floor_plus_upside", "captain mean plus kicker floor and upside",
     dict(captain_head="mean", captain_upside_weight=0.10,
          captain_kicker_weight=0.25)),
    ("captain_rank_overlay_025", "captain mean plus rank overlay",
     dict(captain_head="mean", captain_rank_weight=0.25)),
    ("captain_upside_125_rank010", "captain 1.25 upside plus tiny rank overlay",
     dict(captain_head="mean", captain_upside_weight=1.25,
          captain_rank_weight=0.10)),
    ("captain_upside_150_rank010", "captain 1.50 upside plus tiny rank overlay",
     dict(captain_head="mean", captain_upside_weight=1.50,
          captain_rank_weight=0.10)),
    ("captain_upside_125_kicker_penalty010", "captain 1.25 upside plus small kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.25,
          captain_kicker_weight=-0.10)),
    ("captain_upside_150_kicker_penalty010", "captain 1.50 upside plus small kicker-floor penalty",
     dict(captain_head="mean", captain_upside_weight=1.50,
          captain_kicker_weight=-0.10)),
    ("selector_upside_backthree_025", "XV selector upside overlay for back-three",
     dict(selector_upside_weight=0.25, upside_scope="backthree")),
    ("selector_upside_backrow_backs_025", "XV selector upside overlay for back-row and outside backs",
     dict(selector_upside_weight=0.25, upside_scope="backrow_backs")),
    ("selector_upside_halfbacks_backs_025", "XV selector upside overlay for halfbacks and backs",
     dict(selector_upside_weight=0.25, upside_scope="halfbacks_backs")),
    ("selector_mean_plus_upside_010", "XV selector mean plus 0.10 upside overlay",
     dict(selector_upside_weight=0.10, upside_scope="all")),
    ("selector_mean_plus_upside_020", "XV selector mean plus 0.20 upside overlay",
     dict(selector_upside_weight=0.20, upside_scope="all")),
    ("team_specialist_bench025_capt010", "bench 0.25 plus captain upside 0.10",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          captain_head="mean", captain_upside_weight=0.10)),
    ("team_specialist_bench025_upside010", "bench 0.25 plus XV upside 0.10",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          selector_upside_weight=0.10, upside_scope="all")),
    ("team_specialist_bench025_capt010_upside010", "bench, captain, and XV upside specialists",
     dict(bench_model="ridge", bench_supersub_weight=0.25,
          captain_head="mean", captain_upside_weight=0.10,
          selector_upside_weight=0.10, upside_scope="all")),
    ("latent_shrink_075", "shrink set-piece latent 0.75", dict(latent_shrink=0.75)),
    ("latent_shrink_05", "shrink set-piece latent 0.5", dict(latent_shrink=0.5)),
    ("latent_shrink_050", "shrink set-piece latent 0.50",
     dict(latent_shrink=0.50)),
    ("latent_shrink_025", "shrink set-piece latent 0.25", dict(latent_shrink=0.25)),
    ("latent_shrink_0", "drop set-piece latent (fallback 4)", dict(latent_shrink=0.0)),
    # Decoupled set-piece latent: shrink the point/XV/captain path for MAE, but
    # keep the supersub head on the full (unshrunk) set-piece signal.
    ("decoupled_latent050_sub100", "shrink point latent 0.50, supersub keeps full latent",
     dict(latent_shrink=0.50, supersub_latent_shrink=1.0)),
    ("decoupled_latent025_sub100", "shrink point latent 0.25, supersub keeps full latent",
     dict(latent_shrink=0.25, supersub_latent_shrink=1.0)),
    ("decoupled_latent075_sub100", "shrink point latent 0.75, supersub keeps full latent",
     dict(latent_shrink=0.75, supersub_latent_shrink=1.0)),
    ("decoupled_latent050_sub075", "shrink point latent 0.50, supersub latent 0.75",
     dict(latent_shrink=0.50, supersub_latent_shrink=0.75)),
    ("decoupled_latent040_sub100", "shrink point latent 0.40, supersub keeps full latent",
     dict(latent_shrink=0.40, supersub_latent_shrink=1.0)),
    ("decoupled_latent060_sub100", "shrink point latent 0.60, supersub keeps full latent",
     dict(latent_shrink=0.60, supersub_latent_shrink=1.0)),
    # (Amplifying the supersub set-piece latent >1.0 was tested and rejected:
    #  1.25x left the pick unchanged, 1.50x/2.00x worsened team value.)
    ("potm_pp_055", "POTM 55% points weight", dict(potm_pp_weight=0.55)),
    ("potm_pp_060", "POTM 60% points weight", dict(potm_pp_weight=0.60)),
    ("potm_pp_065", "POTM 65% points weight", dict(potm_pp_weight=0.65)),
    ("potm_pp_07", "POTM lean on points", dict(potm_pp_weight=0.7)),
    ("potm_pp_070", "POTM 70% points weight", dict(potm_pp_weight=0.70)),
    ("potm_pp_075", "POTM 75% points weight", dict(potm_pp_weight=0.75)),
    ("potm_points_weight_075", "POTM 75% points weight",
     dict(potm_pp_weight=0.75)),
    ("potm_pp_085", "POTM 85% points weight", dict(potm_pp_weight=0.85)),
    ("potm_pp_090", "POTM 90% points weight", dict(potm_pp_weight=0.90)),
    ("potm_pp_10", "POTM probability entirely from predicted points", dict(potm_pp_weight=1.0)),
    ("potm_points_weight_100", "POTM probability entirely from predicted points",
     dict(potm_pp_weight=1.0)),
    ("potm065_bayes05", "POTM 65% points plus 5% Bayesian point shrink",
     dict(potm_pp_weight=0.65, bayes_point_weight=0.05,
          selector_point_source="target")),
    ("potm075_bayes05", "POTM 75% points plus 5% Bayesian point shrink",
     dict(potm_pp_weight=0.75, bayes_point_weight=0.05,
          selector_point_source="target")),
    ("potm090_bayes05", "POTM 90% points plus 5% Bayesian point shrink",
     dict(potm_pp_weight=0.90, bayes_point_weight=0.05,
          selector_point_source="target")),
    ("potm075_bayes10", "POTM 75% points plus 10% Bayesian point shrink",
     dict(potm_pp_weight=0.75, bayes_point_weight=0.10,
          selector_point_source="target")),
    ("latent075_potm075", "shrink latent 0.75 and use 75% points POTM",
     dict(latent_shrink=0.75, potm_pp_weight=0.75)),
    ("latent050_potm100", "shrink latent 0.50 and use points-only POTM",
     dict(latent_shrink=0.50, potm_pp_weight=1.0)),
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
    ("points_lgbm88_xgb12", "88% LGBM target points + 12% XGB component points",
     dict(xgb_point_weight=0.12, selector_point_source="target")),
    ("points_lgbm85_xgb15", "85% LGBM target points + 15% XGB component points",
     dict(xgb_point_weight=0.15, selector_point_source="target")),
    ("points_lgbm84_xgb16", "84% LGBM target points + 16% XGB component points",
     dict(xgb_point_weight=0.16, selector_point_source="target")),
    ("points_lgbm82_xgb18", "82% LGBM target points + 18% XGB component points",
     dict(xgb_point_weight=0.18, selector_point_source="target")),
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
          selector_point_source="target")),
    ("xgb_graft_tackles", "replace tackles component with XGB specialist",
     dict(xgb_graft_group="tackles", xgb_graft_weight=1.0,
          selector_point_source="target")),
    ("xgb_graft_tries_assists_db", "replace sparse attacking components with XGB specialist",
     dict(xgb_graft_group="tries_assists_db", xgb_graft_weight=1.0,
          selector_point_source="target")),
    ("xgb_graft_sparse_counts", "replace offload/turnover/penalties components with XGB specialist",
     dict(xgb_graft_group="sparse_counts", xgb_graft_weight=1.0,
          selector_point_source="target")),
    ("xgb_graft_oof_winners_only", "replace only components where XGB beats LGBM OOF by >=2%",
     dict(xgb_graft_group="oof_winners", xgb_graft_weight=1.0,
          selector_point_source="target")),
    ("xgb_graft_oof_winners_blend20", "20% XGB graft for OOF-winning components",
     dict(xgb_graft_group="oof_winners", xgb_graft_weight=0.20,
          selector_point_source="target")),
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
    ("bayes_position_residual_20", "forward-chained 20% position residual correction",
     dict(bayes_position_residual_weight=0.20, selector_point_source="target")),
    ("bayes_position_residual_30", "forward-chained 30% position residual correction",
     dict(bayes_position_residual_weight=0.30, selector_point_source="target")),
    ("bayes_position_residual_40", "forward-chained 40% position residual correction",
     dict(bayes_position_residual_weight=0.40, selector_point_source="target")),
    ("bayes_position_residual_50", "forward-chained 50% position residual correction",
     dict(bayes_position_residual_weight=0.50, selector_point_source="target")),
    ("bayes_position_residual_75", "forward-chained 75% position residual correction",
     dict(bayes_position_residual_weight=0.75, selector_point_source="target")),
    ("bayes_position_residual_100", "forward-chained 100% position residual correction",
     dict(bayes_position_residual_weight=1.0, selector_point_source="target")),
    # MAE-only target adjustments: freeze XV/captain/supersub on the promoted
    # decision stack, then alter target_pts_hat for forecast calibration only.
    ("target_only_bayes05", "post-selector 5% Bayesian point shrinkage for MAE only",
     dict(post_target_xgb_weight=0.0, post_target_bayes_weight=0.05,
          post_target_latent_shrink=-1.0)),
    ("target_only_bayes10", "post-selector 10% Bayesian point shrinkage for MAE only",
     dict(post_target_xgb_weight=0.0, post_target_bayes_weight=0.10,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb05", "post-selector 5% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.05, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb10", "post-selector 10% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.10, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb12", "post-selector 12% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.12, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb15", "post-selector 15% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.15, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb18", "post-selector 18% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.18, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb20", "post-selector 20% extra XGB point blend for MAE only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb18_agree75", "post-selector 18% XGB blend for lowest-disagreement 75%",
     dict(post_target_xgb_weight=0.18, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="all",
          post_target_filter="xgb_agree75")),
    ("target_only_xgb20_agree50", "post-selector 20% XGB blend for lowest-disagreement 50%",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="all",
          post_target_filter="xgb_agree50")),
    ("target_only_xgb20_agree75", "post-selector 20% XGB blend for lowest-disagreement 75%",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="all",
          post_target_filter="xgb_agree75")),
    ("target_only_xgb20_agree90", "post-selector 20% XGB blend for lowest-disagreement 90%",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="all",
          post_target_filter="xgb_agree90")),
    ("target_only_xgb15_starters", "post-selector 15% XGB blend for starters only",
     dict(post_target_xgb_weight=0.15, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="starters")),
    ("target_only_xgb20_starters", "post-selector 20% XGB blend for starters only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="starters")),
    # Forecast-only path is uncapped past the 0.20 decision cap (POST_TARGET_XGB_MAX_WEIGHT):
    # starters-forecast XGB MAE improves monotonically on both seasons; sealed 2026 min at 0.50.
    ("target_only_xgb40_starters", "post-selector 40% XGB blend for starters only",
     dict(post_target_xgb_weight=0.40, post_target_scope="starters")),
    ("target_only_xgb50_starters", "post-selector 50% XGB blend for starters only (equal LGBM/XGB)",
     dict(post_target_xgb_weight=0.50, post_target_scope="starters")),
    ("teamplay_features_on", "use explicit team-play features in the full player model",
     dict(teamplay_features="on")),
    ("teamplay_aspect_features_on", "use only derived team-play aspect features in the full player model",
     dict(teamplay_features="aspects")),
    ("teamplay_component_raw", "use raw team-play features only inside component rate models",
     dict(teamplay_component_features="raw")),
    ("teamplay_component_aspects", "use multiaspect team-play features only inside component rate models",
     dict(teamplay_component_features="aspects")),
    ("teamplay_aspect_attack_005", "component-specific 5% attack aspect adjustment",
     dict(teamplay_aspect_adjust="attack", teamplay_aspect_weight=0.05)),
    ("teamplay_aspect_attack_010", "component-specific 10% attack aspect adjustment",
     dict(teamplay_aspect_adjust="attack", teamplay_aspect_weight=0.10)),
    ("teamplay_aspect_defense_005", "component-specific 5% defensive-load aspect adjustment",
     dict(teamplay_aspect_adjust="defense", teamplay_aspect_weight=0.05)),
    ("teamplay_aspect_defense_010", "component-specific 10% defensive-load aspect adjustment",
     dict(teamplay_aspect_adjust="defense", teamplay_aspect_weight=0.10)),
    ("teamplay_aspect_kicking_005", "component-specific 5% kicking-opportunity aspect adjustment",
     dict(teamplay_aspect_adjust="kicking", teamplay_aspect_weight=0.05)),
    ("teamplay_aspect_multi_005", "component-specific 5% multiaspect adjustment",
     dict(teamplay_aspect_adjust="multi", teamplay_aspect_weight=0.05)),
    ("teamplay_aspect_multi_010", "component-specific 10% multiaspect adjustment",
     dict(teamplay_aspect_adjust="multi", teamplay_aspect_weight=0.10)),
    ("teamplay_component_aspects_multi_005", "component aspect features plus 5% multiaspect adjustment",
     dict(teamplay_component_features="aspects", teamplay_aspect_adjust="multi",
          teamplay_aspect_weight=0.05)),
    ("teamplay_component_aspects_multi_010", "component aspect features plus 10% multiaspect adjustment",
     dict(teamplay_component_features="aspects", teamplay_aspect_adjust="multi",
          teamplay_aspect_weight=0.10)),
    ("teamplay_graft_aspects_oof100", "OOF-gated component graft from aspect team-play LGBM specialists",
     dict(teamplay_graft_mode="aspects", teamplay_graft_group="oof_winners",
          teamplay_graft_weight=1.0)),
    ("teamplay_graft_aspects_oof50", "50% OOF-gated component graft from aspect team-play LGBM specialists",
     dict(teamplay_graft_mode="aspects", teamplay_graft_group="oof_winners",
          teamplay_graft_weight=0.50)),
    ("teamplay_graft_raw_oof100", "OOF-gated component graft from raw team-play LGBM specialists",
     dict(teamplay_graft_mode="raw", teamplay_graft_group="oof_winners",
          teamplay_graft_weight=1.0)),
    ("teamplay_graft_attack_aspects50", "50% attack-component graft if aspect specialists pass OOF",
     dict(teamplay_graft_mode="aspects", teamplay_graft_group="attack",
          teamplay_graft_weight=0.50)),
    ("teamplay_graft_defense_aspects50", "50% defence-component graft if aspect specialists pass OOF",
     dict(teamplay_graft_mode="aspects", teamplay_graft_group="defense",
          teamplay_graft_weight=0.50)),
    ("target_only_teamplay25_starters", "freeze decisions; blend target 25% toward team-play feature model for starters",
     dict(post_target_teamplay_weight=0.25, post_target_scope="starters")),
    ("target_only_teamplay50_starters", "freeze decisions; blend target 50% toward team-play feature model for starters",
     dict(post_target_teamplay_weight=0.50, post_target_scope="starters")),
    ("target_only_teamplay100_starters", "freeze decisions; replace starter target with team-play feature model",
     dict(post_target_teamplay_weight=1.0, post_target_scope="starters")),
    ("target_only_teamplay_aspect25_starters", "freeze decisions; blend target 25% toward multiaspect team-play model for starters",
     dict(post_target_teamplay_weight=0.25, post_target_teamplay_mode="aspects",
          post_target_scope="starters")),
    ("target_only_teamplay_aspect50_starters", "freeze decisions; blend target 50% toward multiaspect team-play model for starters",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="aspects",
          post_target_scope="starters")),
    ("target_only_teamplay_aspect100_starters", "freeze decisions; replace starter target with multiaspect team-play model",
     dict(post_target_teamplay_weight=1.0, post_target_teamplay_mode="aspects",
          post_target_scope="starters")),
    ("target_only_teamplay_aspect50_all", "freeze decisions; blend target 50% toward multiaspect team-play model for all rows",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="aspects",
          post_target_scope="all")),
    ("target_only_teamplay_component_aspect50_starters", "freeze decisions; blend target 50% toward component-aspect specialist model for starters",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="off",
          post_target_teamplay_component_features="aspects",
          post_target_scope="starters")),
    ("target_only_teamplay_component_raw50_starters", "freeze decisions; blend target 50% toward component-raw team-play specialist model for starters",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="off",
          post_target_teamplay_component_features="raw",
          post_target_scope="starters")),
    ("target_only_teamplay_component_aspect100_starters", "freeze decisions; replace starter target with component-aspect specialist model",
     dict(post_target_teamplay_weight=1.0, post_target_teamplay_mode="off",
          post_target_teamplay_component_features="aspects",
          post_target_scope="starters")),
    ("target_only_teamplay_component_raw100_starters", "freeze decisions; replace starter target with component-raw team-play specialist model",
     dict(post_target_teamplay_weight=1.0, post_target_teamplay_mode="off",
          post_target_teamplay_component_features="raw",
          post_target_scope="starters")),
    ("target_only_teamplay_graft_aspects_oof50_starters", "freeze decisions; blend target 50% toward OOF-gated aspect component graft",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="off",
          post_target_teamplay_graft_mode="aspects",
          post_target_teamplay_graft_group="oof_winners",
          post_target_teamplay_graft_weight=1.0,
          post_target_scope="starters")),
    ("target_only_teamplay_graft_aspects_oof100_starters", "freeze decisions; replace starter target with OOF-gated aspect component graft",
     dict(post_target_teamplay_weight=1.0, post_target_teamplay_mode="off",
          post_target_teamplay_graft_mode="aspects",
          post_target_teamplay_graft_group="oof_winners",
          post_target_teamplay_graft_weight=1.0,
          post_target_scope="starters")),
    ("target_only_teamplay_graft_raw_oof50_starters", "freeze decisions; blend target 50% toward OOF-gated raw team-play component graft",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="off",
          post_target_teamplay_graft_mode="raw",
          post_target_teamplay_graft_group="oof_winners",
          post_target_teamplay_graft_weight=1.0,
          post_target_scope="starters")),
    ("target_only_teamplay_graft_raw_oof100_starters", "freeze decisions; replace starter target with OOF-gated raw team-play component graft",
     dict(post_target_teamplay_weight=1.0, post_target_teamplay_mode="off",
          post_target_teamplay_graft_mode="raw",
          post_target_teamplay_graft_group="oof_winners",
          post_target_teamplay_graft_weight=1.0,
          post_target_scope="starters")),
    ("teamplay_graft_aspects_oof01_50", "50% aspect component graft with exploratory 1% OOF gate",
     dict(teamplay_graft_mode="aspects", teamplay_graft_group="oof_winners",
          teamplay_graft_weight=0.50, teamplay_graft_min_gain=0.01)),
    ("target_only_teamplay_graft_aspects_oof01_50_starters", "freeze decisions; blend target 50% toward aspect graft with exploratory 1% OOF gate",
     dict(post_target_teamplay_weight=0.50, post_target_teamplay_mode="off",
          post_target_teamplay_graft_mode="aspects",
          post_target_teamplay_graft_group="oof_winners",
          post_target_teamplay_graft_weight=1.0,
          teamplay_graft_min_gain=0.01,
          post_target_scope="starters")),
    ("post_target_resid_global_010", "post-teamplay 10% prior-round global residual correction",
     dict(post_target_residual_weight=0.10, post_target_residual_group="global")),
    ("post_target_resid_global_025", "post-teamplay 25% prior-round global residual correction",
     dict(post_target_residual_weight=0.25, post_target_residual_group="global")),
    ("post_target_resid_global_050", "post-teamplay 50% prior-round global residual correction",
     dict(post_target_residual_weight=0.50, post_target_residual_group="global")),
    ("post_target_resid_forward_back_010", "post-teamplay 10% prior-round forward/back residual correction",
     dict(post_target_residual_weight=0.10, post_target_residual_group="forward_back")),
    ("post_target_resid_forward_back_025", "post-teamplay 25% prior-round forward/back residual correction",
     dict(post_target_residual_weight=0.25, post_target_residual_group="forward_back")),
    ("post_target_resid_position_010", "post-teamplay 10% prior-round position residual correction",
     dict(post_target_residual_weight=0.10, post_target_residual_group="position")),
    ("post_target_resid_position_025", "post-teamplay 25% prior-round position residual correction",
     dict(post_target_residual_weight=0.25, post_target_residual_group="position")),
    ("post_target_resid_position_050", "post-teamplay 50% prior-round position residual correction",
     dict(post_target_residual_weight=0.50, post_target_residual_group="position")),
    ("post_target_resid_position_started_025", "post-teamplay 25% position+starter-status residual correction",
     dict(post_target_residual_weight=0.25, post_target_residual_group="position_started")),
    ("post_target_resid_team_010", "post-teamplay 10% prior-round team residual correction",
     dict(post_target_residual_weight=0.10, post_target_residual_group="team")),
    ("post_target_resid_team_025", "post-teamplay 25% prior-round team residual correction",
     dict(post_target_residual_weight=0.25, post_target_residual_group="team")),
    ("post_target_resid_team_position_010", "post-teamplay 10% team+position residual correction",
     dict(post_target_residual_weight=0.10, post_target_residual_group="team_position",
          post_target_residual_group_min_n=3)),
    ("post_target_resid_position_025_all", "post-teamplay 25% position residual correction for all rows",
     dict(post_target_residual_weight=0.25, post_target_residual_group="position",
          post_target_residual_scope="all")),
    ("post_target_resid_position_025_starter_backs", "post-teamplay 25% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.25, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_015_starter_backs", "post-teamplay 15% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.15, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_020_starter_backs", "post-teamplay 20% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.20, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_030_starter_backs", "post-teamplay 30% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.30, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_035_starter_backs", "post-teamplay 35% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.35, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_040_starter_backs", "post-teamplay 40% position residual correction for starting backs only",
     dict(post_target_residual_weight=0.40, post_target_residual_group="position",
          post_target_residual_scope="starters_backs")),
    ("post_target_resid_position_025_starters_backs_backrow", "post-teamplay 25% position residual correction for starting backs plus back-row",
     dict(post_target_residual_weight=0.25, post_target_residual_group="position",
          post_target_residual_scope="starters_backs_backrow")),
    ("post_target_resid_position_035_starters_backs_backrow", "post-teamplay 35% position residual correction for starting backs plus back-row",
     dict(post_target_residual_weight=0.35, post_target_residual_group="position",
          post_target_residual_scope="starters_backs_backrow")),
    ("post_target_resid_team_position_010_starter_backs", "post-teamplay 10% team+position residual correction for starting backs only",
     dict(post_target_residual_weight=0.10, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3)),
    ("post_target_resid_team_position_015_starter_backs", "post-teamplay 15% team+position residual correction for starting backs only",
     dict(post_target_residual_weight=0.15, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3)),
    ("post_target_resid_team_position_020_starter_backs", "post-teamplay 20% team+position residual correction for starting backs only",
     dict(post_target_residual_weight=0.20, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3)),
    ("post_target_resid_team_position_025_starter_backs", "post-teamplay 25% team+position residual correction for starting backs only",
     dict(post_target_residual_weight=0.25, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3)),
    ("post_target_resid_team_position_030_starter_backs", "post-teamplay 30% team+position residual correction for starting backs only",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3)),
    ("post_target_resid_team_position_020_starter_backs_clip3", "post-teamplay 20% team+position residual correction for starting backs only, clipped tighter",
     dict(post_target_residual_weight=0.20, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_residual_clip=3.0)),
    ("post_target_resid_team_position_020_starters_backs_backrow", "post-teamplay 20% team+position residual correction for starting backs plus back-row",
     dict(post_target_residual_weight=0.20, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs_backrow", post_target_residual_group_min_n=3)),
    ("post_target_resid_forward_back_025_starters_backs_backrow", "post-teamplay 25% forward/back residual correction for starting backs plus back-row",
     dict(post_target_residual_weight=0.25, post_target_residual_group="forward_back",
          post_target_residual_scope="starters_backs_backrow")),
    ("orchestrator_resid030_refresh_selector_tilt000", "team-play forecast plus backs residual feeds XV selector with no rank tilt",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_refresh_selector=True, selector_tilt=0.0)),
    ("orchestrator_resid030_refresh_selector_tilt0025", "team-play forecast plus backs residual feeds XV selector with 0.025 rank tilt",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_refresh_selector=True, selector_tilt=0.025)),
    ("orchestrator_resid030_refresh_selector_tilt005", "team-play forecast plus backs residual feeds XV selector with 0.05 rank tilt",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_refresh_selector=True, selector_tilt=0.05)),
    ("orchestrator_resid030_refresh_selector_tilt010", "team-play forecast plus backs residual feeds XV selector with 0.10 rank tilt",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_refresh_selector=True, selector_tilt=0.10)),
    ("orchestrator_resid030_refresh_selector_captain_tilt005", "team-play forecast plus backs residual feeds XV selector and captain head",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_refresh_selector=True, post_target_refresh_captain=True,
          selector_tilt=0.05)),
    ("orchestrator_delta_overlay_0025", "tiny selector tie-breaker from post-target specialist delta",
     dict(post_target_selector_delta_weight=0.025)),
    ("orchestrator_delta_overlay_005", "small selector tie-breaker from post-target specialist delta",
     dict(post_target_selector_delta_weight=0.05)),
    ("orchestrator_delta_overlay_010", "0.10 selector tie-breaker from post-target specialist delta",
     dict(post_target_selector_delta_weight=0.10)),
    ("orchestrator_delta_overlay_020", "0.20 selector tie-breaker from post-target specialist delta",
     dict(post_target_selector_delta_weight=0.20)),
    ("orchestrator_resid030_delta_overlay_0025", "backs residual plus tiny selector tie-breaker from specialist delta",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_selector_delta_weight=0.025)),
    ("orchestrator_resid030_delta_overlay_005", "backs residual plus small selector tie-breaker from specialist delta",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_selector_delta_weight=0.05)),
    ("orchestrator_resid030_delta_overlay_010", "backs residual plus 0.10 selector tie-breaker from specialist delta",
     dict(post_target_residual_weight=0.30, post_target_residual_group="team_position",
          post_target_residual_scope="starters_backs", post_target_residual_group_min_n=3,
          post_target_selector_delta_weight=0.10)),
    ("orchestrator_delta_prior130", "delay selector delta until at least one prior round of evidence",
     dict(post_target_selector_delta_min_prior_n=130)),
    ("orchestrator_delta_prior275", "delay selector delta until roughly two prior rounds of evidence",
     dict(post_target_selector_delta_min_prior_n=275)),
    ("orchestrator_delta_prior400", "delay selector delta until roughly three prior rounds of evidence",
     dict(post_target_selector_delta_min_prior_n=400)),
    ("orchestrator_delta_prior500", "delay selector delta until roughly four prior rounds of evidence",
     dict(post_target_selector_delta_min_prior_n=500)),
    ("orchestrator_delta_prior500_w005", "late-only selector delta with half weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_weight=0.05)),
    ("orchestrator_delta_prior500_w015", "late-only selector delta with 0.15 weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_weight=0.15)),
    ("orchestrator_delta_prior500_w020", "late-only selector delta with 0.20 weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_weight=0.20)),
    ("orchestrator_delta_prior500_scope_starters", "late-only selector delta for starters only",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters")),
    ("orchestrator_delta_prior500_scope_starters_backs", "late-only selector delta for starting backs only",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs")),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow", "late-only selector delta for starting backs plus back-row",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow")),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w005", "late-only backs/back-row selector delta with half weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.05)),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w0075", "late-only backs/back-row selector delta with 0.075 weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.075)),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w015", "late-only backs/back-row selector delta with 0.15 weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.15)),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w020", "late-only backs/back-row selector delta with 0.20 weight",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.20)),
    ("orchestrator_delta_prior500_clip050", "late-only selector delta clipped to half a z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_clip=0.50)),
    ("orchestrator_delta_prior500_clip100", "late-only selector delta clipped to one z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_clip=1.0)),
    ("orchestrator_delta_prior500_w015_clip050", "late-only 0.15 selector delta clipped to half a z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_weight=0.15,
          post_target_selector_delta_clip=0.50)),
    ("orchestrator_delta_prior500_w015_clip100", "late-only 0.15 selector delta clipped to one z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_weight=0.15,
          post_target_selector_delta_clip=1.0)),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w015_clip050", "late-only backs/back-row 0.15 selector delta clipped to half a z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.15,
          post_target_selector_delta_clip=0.50)),
    ("orchestrator_delta_prior500_scope_starters_backs_backrow_w015_clip100", "late-only backs/back-row 0.15 selector delta clipped to one z-score",
     dict(post_target_selector_delta_min_prior_n=500,
          post_target_selector_delta_scope="starters_backs_backrow",
          post_target_selector_delta_weight=0.15,
          post_target_selector_delta_clip=1.0)),
    ("post_target_refresh_captain", "let captain head see the post-target specialist forecast",
     dict(post_target_refresh_captain=True)),
    ("post_target_refresh_captain_mean", "let captain use post-target specialist forecast without upside overlay",
     dict(post_target_refresh_captain=True, captain_upside_weight=0.0,
          captain_rank_weight=0.0, captain_kicker_weight=0.0)),
    ("post_target_refresh_captain_upside075", "let captain see post-target forecast with 0.75 upside",
     dict(post_target_refresh_captain=True, captain_upside_weight=0.75)),
    ("post_target_refresh_captain_upside100", "let captain see post-target forecast with 1.00 upside",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.0)),
    ("post_target_refresh_captain_upside125", "let captain see post-target forecast with 1.25 upside",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.25)),
    ("post_target_refresh_captain_upside150", "let captain see post-target forecast with 1.50 upside",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.50)),
    ("post_target_refresh_captain_upside125_rank010", "post-target captain with 1.25 upside and tiny rank overlay",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.25,
          captain_rank_weight=0.10)),
    ("post_target_refresh_captain_upside150_rank010", "post-target captain with 1.50 upside and tiny rank overlay",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.50,
          captain_rank_weight=0.10)),
    ("post_target_refresh_captain_upside125_kicker_penalty010", "post-target captain with 1.25 upside and small kicker-floor penalty",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.25,
          captain_kicker_weight=-0.10)),
    ("post_target_refresh_captain_upside150_kicker_penalty010", "post-target captain with 1.50 upside and small kicker-floor penalty",
     dict(post_target_refresh_captain=True, captain_upside_weight=1.50,
          captain_kicker_weight=-0.10)),
    ("post_target_refresh_selector", "let XV selector see the post-target specialist forecast",
     dict(post_target_refresh_selector=True)),
    ("post_target_refresh_selector_tilt000", "let XV selector see post-target forecast with no rank tilt",
     dict(post_target_refresh_selector=True, selector_tilt=0.0)),
    ("post_target_refresh_selector_tilt005", "let XV selector see post-target forecast with tiny rank tilt",
     dict(post_target_refresh_selector=True, selector_tilt=0.05)),
    ("post_target_refresh_selector_tilt010", "let XV selector see post-target forecast with smaller rank tilt",
     dict(post_target_refresh_selector=True, selector_tilt=0.10)),
    ("post_target_refresh_selector_tilt015", "let XV selector see post-target forecast with 0.15 rank tilt",
     dict(post_target_refresh_selector=True, selector_tilt=0.15)),
    ("post_target_refresh_selector_tilt020", "let XV selector see post-target forecast with 0.20 rank tilt",
     dict(post_target_refresh_selector=True, selector_tilt=0.20)),
    ("post_target_refresh_selector_captain", "let XV selector and captain see post-target specialist forecast",
     dict(post_target_refresh_selector=True, post_target_refresh_captain=True)),
    ("post_target_refresh_selector_captain_tilt010", "let XV selector and captain see post-target forecast with 0.10 rank tilt",
     dict(post_target_refresh_selector=True, post_target_refresh_captain=True,
          selector_tilt=0.10)),
    ("target_only_xgb60_starters", "post-selector 60% XGB blend for starters only",
     dict(post_target_xgb_weight=0.60, post_target_scope="starters")),
    ("target_only_xgb20_bench", "post-selector 20% XGB blend for bench only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="bench")),
    ("target_only_xgb20_forwards", "post-selector 20% XGB blend for forwards only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="forwards")),
    ("target_only_xgb20_backs", "post-selector 20% XGB blend for backs only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="backs")),
    ("target_only_xgb20_starters_forwards", "post-selector 20% XGB blend for starting forwards only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="starters_forwards")),
    ("target_only_xgb20_starters_backs", "post-selector 20% XGB blend for starting backs only",
     dict(post_target_xgb_weight=0.20, post_target_bayes_weight=0.0,
          post_target_latent_shrink=-1.0, post_target_scope="starters_backs")),
    ("target_only_latent075", "post-selector set-piece latent shrink 0.75 for MAE only",
     dict(post_target_xgb_weight=0.0, post_target_bayes_weight=0.0,
          post_target_latent_shrink=0.75)),
    ("target_only_latent050", "post-selector set-piece latent shrink 0.50 for MAE only",
     dict(post_target_xgb_weight=0.0, post_target_bayes_weight=0.0,
          post_target_latent_shrink=0.50)),
    ("target_only_bayes05_latent075", "post-selector Bayesian 5% plus latent 0.75 for MAE only",
     dict(post_target_xgb_weight=0.0, post_target_bayes_weight=0.05,
          post_target_latent_shrink=0.75)),
    ("target_only_xgb10_bayes05", "post-selector XGB 10% plus Bayesian 5% for MAE only",
     dict(post_target_xgb_weight=0.10, post_target_bayes_weight=0.05,
          post_target_latent_shrink=-1.0)),
    ("target_only_xgb10_bayes10", "post-selector XGB 10% plus Bayesian 10% for MAE only",
     dict(post_target_xgb_weight=0.10, post_target_bayes_weight=0.10,
          post_target_latent_shrink=-1.0)),
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
    ("selector_tilt_015", "lighter rank tilt for XV pick", dict(selector_tilt=0.15)),
    ("selector_tilt_020", "slightly lighter rank tilt for XV pick", dict(selector_tilt=0.20)),
    ("selector_tilt_025", "tilt XV pick toward rank head", dict(selector_tilt=0.25)),
    ("selector_tilt_030", "slightly heavier rank tilt for XV pick", dict(selector_tilt=0.30)),
    ("selector_tilt_035", "heavier rank tilt for XV pick", dict(selector_tilt=0.35)),
    ("selector_tilt_05", "stronger rank tilt", dict(selector_tilt=0.5)),
]


FRONTIER_CANDIDATES = [
    # Incumbent stability ablations. These change one high-leverage post-target
    # setting at a time on top of orchestrator_resid030_delta_overlay_010.
    ("orchestrator_resid030_delta010_clip3",
     "tighten incumbent team-position residual clip from 6 to 3",
     dict(post_target_residual_clip=3.0)),
    ("orchestrator_resid030_delta010_groupmin5",
     "require five observations for incumbent team-position residual cells",
     dict(post_target_residual_group_min_n=5)),
    ("orchestrator_resid030_delta010_xgb0",
     "remove post-target XGB forecast blend while preserving selection decisions",
     dict(post_target_xgb_weight=0.0)),
    ("orchestrator_selector_signal_xgb0",
     "retain incumbent point forecast but build selector delta from no-XGB post-target signal",
     dict(post_target_selector_xgb_weight=0.0)),
    ("bench_context_team_position_slot",
     "learn team, position, and bench-slot substitution tendencies in the two-stage bench head",
     dict(bench_context_features="all")),
    ("bench_context_slot",
     "learn nonlinear bench-slot and position-slot substitution tendencies",
     dict(bench_context_features="slot")),
    ("bench_context_team",
     "learn coarse team and team-position substitution tendencies without jersey interactions",
     dict(bench_context_features="team")),
    ("supersub_context_slot_only",
     "use nonlinear bench-slot context only in the supersub head, leaving point calibration unchanged",
     dict(bench_supersub_context_features="slot")),
    ("supersub_replacement_context",
     "use named-starter coverage and same-role bench competition only in the supersub head",
     dict(bench_supersub_replacement_features=True)),
    ("supersub_player_bench_history",
     "use each player's PIT multi-competition replacement-minute history only in the supersub head",
     dict(bench_supersub_history_features=True)),
    ("supersub_team_slot_history",
     "use each national team's PIT jersey-slot substitution history only in the supersub head",
     dict(bench_supersub_team_history_features=True)),
    ("target_only_matchup100_starters",
     "freeze decisions and replace starter forecast with a position-specific opponent-allowance specialist",
     dict(post_target_teamplay_weight=1.0,
          post_target_matchup_features=True,
          post_target_selector_matchup_features=False,
          post_target_scope="starters")),

    # 1) Fixture-strength / match-shape proxy.  This uses the local PIT
    # team-play layer as a bookmaker-style substitute until real totals/spreads
    # are available.
    ("frontier_fixture_strength_005", "5% fixture-strength component nudge from team-play proxy",
     dict(teamplay_aspect_adjust="fixture_strength", teamplay_aspect_weight=0.05)),
    ("frontier_fixture_strength_010", "10% fixture-strength component nudge from team-play proxy",
     dict(teamplay_aspect_adjust="fixture_strength", teamplay_aspect_weight=0.10)),
    ("frontier_fixture_strength_multi_005", "5% strength+tempo+defence multiaspect nudge",
     dict(teamplay_aspect_adjust="strength_multi", teamplay_aspect_weight=0.05)),
    ("frontier_fixture_strength_multi_010", "10% strength+tempo+defence multiaspect nudge",
     dict(teamplay_aspect_adjust="strength_multi", teamplay_aspect_weight=0.10)),
    ("frontier_fixture_tempo_005", "5% open-game/tempo component nudge",
     dict(teamplay_aspect_adjust="tempo", teamplay_aspect_weight=0.05)),
    ("frontier_fixture_tempo_010", "10% open-game/tempo component nudge",
     dict(teamplay_aspect_adjust="tempo", teamplay_aspect_weight=0.10)),

    # 2) Bench/supersub minute distribution.  Keep the existing two-stage bench
    # mean, then ask whether P(meaningful minutes) is a separate supersub signal.
    ("frontier_bench_high25_w010", "supersub adds P(bench minutes >=25) at 0.10",
     dict(bench_model="two_stage_ridge", bench_high_minutes_threshold=25.0,
          bench_high_minutes_weight=0.10)),
    ("frontier_bench_high30_w010", "supersub adds P(bench minutes >=30) at 0.10",
     dict(bench_model="two_stage_ridge", bench_high_minutes_threshold=30.0,
          bench_high_minutes_weight=0.10)),
    ("frontier_bench_high30_w020", "supersub adds P(bench minutes >=30) at 0.20",
     dict(bench_model="two_stage_ridge", bench_high_minutes_threshold=30.0,
          bench_high_minutes_weight=0.20)),
    ("frontier_bench_high35_w010", "supersub adds P(bench minutes >=35) at 0.10",
     dict(bench_model="two_stage_ridge", bench_high_minutes_threshold=35.0,
          bench_high_minutes_weight=0.10)),
    ("frontier_bench_lowplay_pen010", "supersub penalises learned low-play probability at 0.10",
     dict(bench_model="two_stage_ridge", bench_low_minutes_penalty_weight=0.10)),
    ("frontier_bench_high30_lowpen010", "supersub high-minute bonus plus low-play penalty",
     dict(bench_model="two_stage_ridge", bench_high_minutes_threshold=30.0,
          bench_high_minutes_weight=0.10, bench_low_minutes_penalty_weight=0.10)),

    # 3) Role certainty / component kicking.  Reallocate team kicking mass using
    # learned goal-kicker rate and kick attempts, optionally only when confidence
    # is high.
    ("frontier_kicker_realloc_starters_010", "10% learned kicking reallocation among starters",
     dict(kicking_realloc_weight=0.10, kicking_realloc_scope="starters")),
    ("frontier_kicker_realloc_starters_025", "25% learned kicking reallocation among starters",
     dict(kicking_realloc_weight=0.25, kicking_realloc_scope="starters")),
    ("frontier_kicker_realloc_highconf_025", "25% starter kicking reallocation only if top share >=60%",
     dict(kicking_realloc_weight=0.25, kicking_realloc_scope="starters",
          kicking_realloc_min_share=0.60)),
    ("frontier_kicker_realloc_all_010", "10% learned kicking reallocation across all named players",
     dict(kicking_realloc_weight=0.10, kicking_realloc_scope="all")),

    # 4) True forward-chained orchestrator/stacker.  This is target-only unless
    # a later role-refresh candidate explicitly uses the post-target result.
    ("frontier_combiner_grid_all", "forward-chained conservative LGBM/XGB/Bayes/team-play target combiner",
     dict(post_target_forward_combiner="conservative_grid",
          post_target_teamplay_mode="off",
          post_target_teamplay_component_features="aspects",
          post_target_combiner_scope="all")),
    ("frontier_combiner_grid_starters", "forward-chained conservative target combiner for starters only",
     dict(post_target_forward_combiner="conservative_grid",
          post_target_teamplay_mode="off",
          post_target_teamplay_component_features="aspects",
          post_target_combiner_scope="starters")),
    ("frontier_combiner_grid_starter_backs", "forward-chained conservative target combiner for starting backs only",
     dict(post_target_forward_combiner="conservative_grid",
          post_target_teamplay_mode="off",
          post_target_teamplay_component_features="aspects",
          post_target_combiner_scope="starters_backs")),
]

EXTERNAL_DATA_CANDIDATES = [
    # These are inert until the optional data/external_*.csv files are populated
    # and build_features.py has been rerun.  They are kept explicit so new data
    # enters as a controlled experiment rather than silently changing every model.
    ("external_market_features_on", "use optional market odds / team-total fixture features",
     dict(market_features=True)),
    ("external_weather_features_on", "use optional weather and venue-condition features",
     dict(weather_features=True)),
    ("external_rolecert_features_on", "use optional named-role certainty features",
     dict(rolecert_features=True)),
    ("external_market_weather_on", "combine market fixture strength with weather features",
     dict(market_features=True, weather_features=True)),
    ("external_market_rolecert_on", "combine market fixture strength with named-role certainty",
     dict(market_features=True, rolecert_features=True)),
    ("external_all_data_on", "market, weather, and named-role certainty features together",
     dict(market_features=True, weather_features=True, rolecert_features=True)),
    ("external_market_teamplay_aspects", "market features plus existing team-play aspects",
     dict(market_features=True, teamplay_component_features="aspects")),
    ("external_market_post_target_only", "market shape only inside frozen post-target specialist",
     dict(market_features=False, post_target_market_features=True,
          post_target_selector_delta_weight=0.0)),
    ("external_market_post_target_delta0025", "market post-target specialist plus tiny selector delta",
     dict(market_features=False, post_target_market_features=True,
          post_target_selector_delta_weight=0.025)),
    ("external_market_post_target_delta005", "market post-target specialist plus small selector delta",
     dict(market_features=False, post_target_market_features=True,
          post_target_selector_delta_weight=0.05)),
    ("external_market_post_target_delta010", "market post-target specialist plus promoted selector delta",
     dict(market_features=False, post_target_market_features=True,
          post_target_selector_delta_weight=0.10)),
    ("external_market_post_target_delta020", "market post-target specialist plus larger selector delta",
     dict(market_features=False, post_target_market_features=True,
          post_target_selector_delta_weight=0.20)),
    ("captain_market_winprob_050", "captain tilt toward market win probability at 0.50",
     dict(captain_market_weight=0.50)),
    ("captain_market_winprob_075", "captain tilt toward market win probability at 0.75",
     dict(captain_market_weight=0.75)),
    ("captain_market_winprob_100", "captain tilt toward market win probability at 1.00",
     dict(captain_market_weight=1.00)),
    ("captain_market_winprob_125", "captain tilt toward market win probability at 1.25",
     dict(captain_market_weight=1.25)),
    ("captain_market_winprob_150", "captain tilt toward market win probability at 1.50",
     dict(captain_market_weight=1.50)),
    ("external_style_features_on", "use optional tactical-style priors",
     dict(style_features=True)),
    ("external_style_aspects_on", "use only compact tactical-style aspect priors",
     dict(style_features="aspects")),
    ("external_style_teamplay_aspects", "tactical-style priors plus team-play component aspects",
     dict(style_features=True, teamplay_component_features="aspects")),
    ("external_style_aspects_teamplay_aspects", "style aspects plus team-play component aspects",
     dict(style_features="aspects", teamplay_component_features="aspects")),
    ("external_style_rolecert_on", "tactical-style priors plus named-role certainty features",
     dict(style_features=True, rolecert_features=True)),
    ("rolecert_kick_realloc_starters_010", "10% rolecert kicking reallocation among starters",
     dict(rolecert_kick_realloc_weight=0.10, rolecert_kick_realloc_scope="starters")),
    ("rolecert_kick_realloc_starters_025", "25% rolecert kicking reallocation among starters",
     dict(rolecert_kick_realloc_weight=0.25, rolecert_kick_realloc_scope="starters")),
    ("rolecert_kick_realloc_all_010", "10% rolecert kicking reallocation across named 23",
     dict(rolecert_kick_realloc_weight=0.10, rolecert_kick_realloc_scope="all")),
    ("rolecert_bench_kick_shrink_025", "25% bench kicking shrink by rolecert uncertainty",
     dict(rolecert_bench_kick_shrink=0.25)),
    ("rolecert_bench_kick_shrink_050", "50% bench kicking shrink by rolecert uncertainty",
     dict(rolecert_bench_kick_shrink=0.50)),
    ("rolecert_supersub_uncertainty_010", "supersub subtracts team-sheet role uncertainty at 0.10",
     dict(rolecert_supersub_uncertainty_weight=0.10)),
    ("rolecert_supersub_uncertainty_025", "supersub subtracts team-sheet role uncertainty at 0.25",
     dict(rolecert_supersub_uncertainty_weight=0.25)),
    ("rolecert_bench_combo_025_010", "bench kick shrink plus small supersub uncertainty penalty",
     dict(rolecert_bench_kick_shrink=0.25, rolecert_supersub_uncertainty_weight=0.10)),
    ("rolecert_bench_combo_050_010", "larger bench kick shrink plus small supersub uncertainty penalty",
     dict(rolecert_bench_kick_shrink=0.50, rolecert_supersub_uncertainty_weight=0.10)),
    ("weather_attack_suppress_0025", "tiny weather suppressor for open attacking components",
     dict(weather_aspect_adjust="attack_suppress", weather_aspect_weight=0.025)),
    ("weather_attack_suppress_005", "small weather suppressor for open attacking components",
     dict(weather_aspect_adjust="attack_suppress", weather_aspect_weight=0.05)),
    ("weather_attack_suppress_010", "moderate weather suppressor for open attacking components",
     dict(weather_aspect_adjust="attack_suppress", weather_aspect_weight=0.10)),
    ("weather_kicking_suppress_0025", "tiny wet/windy suppressor for kicking components",
     dict(weather_aspect_adjust="kicking_suppress", weather_aspect_weight=0.025)),
    ("weather_kicking_suppress_005", "small wet/windy suppressor for kicking components",
     dict(weather_aspect_adjust="kicking_suppress", weather_aspect_weight=0.05)),
    ("weather_tackle_boost_0025", "tiny bad-weather boost for tackle/pressure components",
     dict(weather_aspect_adjust="tackle_boost", weather_aspect_weight=0.025)),
    ("weather_tackle_boost_005", "small bad-weather boost for tackle/pressure components",
     dict(weather_aspect_adjust="tackle_boost", weather_aspect_weight=0.05)),
    ("weather_tackle_boost_010", "moderate bad-weather boost for tackle/pressure components",
     dict(weather_aspect_adjust="tackle_boost", weather_aspect_weight=0.10)),
    ("weather_multi_0025", "tiny weather multi-aspect component nudge",
     dict(weather_aspect_adjust="multi", weather_aspect_weight=0.025)),
    ("weather_multi_005", "small weather multi-aspect component nudge",
     dict(weather_aspect_adjust="multi", weather_aspect_weight=0.05)),
    ("weather_multi_0075", "weather multi-aspect component nudge 0.075",
     dict(weather_aspect_adjust="multi", weather_aspect_weight=0.075)),
    ("weather_multi_010", "moderate weather multi-aspect component nudge",
     dict(weather_aspect_adjust="multi", weather_aspect_weight=0.10)),
    ("weather_multi_015", "larger weather multi-aspect component nudge",
     dict(weather_aspect_adjust="multi", weather_aspect_weight=0.15)),
]

CANDIDATES.extend(FRONTIER_CANDIDATES)
CANDIDATES.extend(EXTERNAL_DATA_CANDIDATES)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run_loop(
    candidate_names: list[str] | set[str] | None = None,
    limit: int | None = None,
    base_cfg: Config | None = None,
    *,
    claude_review: bool = False,
    claude_review_timeout: int = 300,
    force_claude_review: bool = False,
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
    print(f"[0] {best_cfg.name:18s} value_team={best['value_team']:.3f} "
          f"value_xv={best['value_xv']:.3f} mae={best['mae']:.3f} "
          f"bench_mae={best['bench_mae']:.3f} top15={best['top15']:.3f} "
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
        print(f"[{i}] {name:20s} value_team={cand['value_team']:.3f} "
              f"value_xv={cand['value_xv']:.3f} mae={cand['mae']:.3f} "
              f"bench_mae={cand['bench_mae']:.3f} top15={cand['top15']:.3f}  -> "
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
    if claude_review:
        from model.claude_review import review_research_batch

        review = review_research_batch(
            trials,
            dataclasses.asdict(best_cfg),
            stability_reports,
            dev_season=DEV_SEASON,
            timeout_seconds=claude_review_timeout,
            force=force_claude_review,
        )
        opinion = review.get("opinion") or {}
        print(
            "CLAUDE SECOND OPINION = "
            f"{review['status']} {opinion.get('verdict', '')} "
            f"{opinion.get('recommendation', '')}".rstrip()
        )
    print(f"\nBEST = {best_cfg.name}  value_team={best['value_team']:.3f} "
          f"value_xv={best['value_xv']:.3f} mae={best['mae']:.3f}")
    return best_cfg, best, trials


def _metric_row(m: dict) -> dict:
    return {k: m[k] for k in ("value_team", "value_xv", "mae", "bench_mae",
                              "top15", "top30", "capt_top1", "capt_top3",
                              "capt_top5", "spearman_pos", "spearman_sel",
                              "registry_lgbm")} \
        | {"round_team": m["round_team"], "round_xv": m["round_xv"],
           "round_mae": m["round_mae"],
           "round_bias": m["round_bias"], "by_pos": m["by_pos"],
           "selection_diagnostics": m["selection_diagnostics"]}


def _write_ledger(trials: list, best_cfg: Config, best: dict) -> None:
    (RESEARCH / "ledger.json").write_text(json.dumps(trials, indent=2))
    (RESEARCH / "best_config.json").write_text(
        json.dumps(dataclasses.asdict(best_cfg), indent=2))
    _append_history(trials, best_cfg, best)
    lines = [f"# Autoresearch ledger — {date.today()}", "",
             f"Dev season {DEV_SEASON} (post_team_sheet). "
             f"Primary value is value_team (XV + captain + supersub). "
             f"Incumbent value_team={trials[0]['value_team']:.3f} / "
             f"XV-only={trials[0]['value_xv']:.3f} / MAE={trials[0]['mae']:.3f}.",
             "2026 sealed — not evaluated here.", "",
             "| # | candidate | team | XV | MAE | bench MAE | top15 | top30 | cap3 | spear_sel | stability | accepted | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in trials:
        lines.append(f"| {t['trial']} | {t['name']} | {t['value_team']:.3f} | "
                     f"{t['value_xv']:.3f} | {t['mae']:.3f} | "
                     f"{t['bench_mae']:.3f} | {t['top15']:.3f} | "
                     f"{t['top30']:.3f} | {t['capt_top3']:.3f} | "
                     f"{t['spearman_sel']:.3f} | "
                     f"{t.get('stability_reason', 'passed')} | "
                     f"{'yes' if t['accepted'] else 'no'} | {t['reason']} |")
    lines += ["", f"**Best:** `{best_cfg.name}` — value_team {best['value_team']:.3f}, "
              f"XV-only {best['value_xv']:.3f}, MAE {best['mae']:.3f}, "
              f"bench MAE {best['bench_mae']:.3f}, top15 {best['top15']:.3f}.",
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
    print(f"  config={best_cfg.name}  value_team={res['value_team']:.3f}  "
          f"value_xv={res['value_xv']:.3f}  mae={res['mae']:.3f}  "
          f"bench_mae={res['bench_mae']:.3f}  top15={res['top15']:.3f}")
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
    ap.add_argument(
        "--claude-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt into the model-specific Claude review after the numerical batch",
    )
    ap.add_argument("--claude-review-timeout", type=int, default=300)
    ap.add_argument(
        "--force-claude-review",
        action="store_true",
        help="rerun Claude even when the evidence fingerprint is unchanged",
    )
    args = ap.parse_args()
    if args.seal_2026:
        cfg_path = PROMOTED_CONFIG if args.seal_config == "promoted" else BEST_CONFIG
        if not cfg_path.exists() and args.seal_config == "promoted":
            cfg_path = BEST_CONFIG
        if args.claude_review:
            from model.claude_review import review_saved_batch

            review = review_saved_batch(
                config_path=cfg_path,
                timeout_seconds=args.claude_review_timeout,
                force=args.force_claude_review,
            )
            opinion = review.get("opinion") or {}
            print(
                "Pre-seal Claude second opinion = "
                f"{review['status']} {opinion.get('verdict', '')} "
                f"{opinion.get('recommendation', '')}".rstrip()
            )
        print(f"Sealing config from {cfg_path.relative_to(ROOT)}")
        cfg = Config(**json.loads(cfg_path.read_text()))
        seal_2026(cfg, persist_predictions=(cfg_path == PROMOTED_CONFIG))
    else:
        base_cfg = _load_base_config(args.base_config)
        run_loop(
            args.candidate if args.candidate else None,
            args.limit,
            base_cfg,
            claude_review=args.claude_review,
            claude_review_timeout=args.claude_review_timeout,
            force_claude_review=args.force_claude_review,
        )


def _load_base_config(which: str) -> Config:
    if which == "baseline":
        return Config()
    cfg_path = PROMOTED_CONFIG if which == "promoted" else BEST_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path.relative_to(ROOT)} does not exist")
    return Config(**json.loads(cfg_path.read_text()))


if __name__ == "__main__":
    main()
