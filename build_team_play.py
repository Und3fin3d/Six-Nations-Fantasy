#!/usr/bin/env python3
"""
build_team_play.py — PIT team edge and fixture-shape predictions
================================================================

Builds one row per Six Nations match-team with predicted game-script features:
win probability, expected margin, expected points, possession, attacking volume,
defensive tackle demand, and kicking opportunity.

This is intentionally an intermediate layer.  The player component models should
condition on "what kind of game do we expect?" rather than trying to infer match
shape only from scattered raw opponent/own-team features.

Leakage rule: for each fixture-team row, the model is trained only on rows with
date < fixture date.  Prior feature summaries are also computed only from rows
before that fixture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA = BASE / "data"
SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]

CONTEXT_COLS = [
    "team_score", "opp_score", "margin", "tries_for", "tries_against",
    "conversion_goals", "penalty_goals", "kicks_from_hand", "possession",
    "runs", "carries_metres", "clean_breaks", "defenders_beaten", "offload",
    "passes", "tackles", "missed_tackles", "turnovers_conceded",
    "turnovers_won", "penalties_conceded", "scrums_won", "scrums_success",
    "lineout_won_steal", "lineout_success",
]

PREDICTION_TARGETS = {
    "teamplay_points_for_hat": ("team_score", "continuous"),
    "teamplay_points_against_hat": ("opp_score", "continuous"),
    "teamplay_margin_hat": ("margin", "continuous"),
    "teamplay_total_points_hat": ("total_points", "continuous"),
    "teamplay_win_prob_hat": ("win", "probability"),
    "teamplay_dominance_prob_hat": ("dominant_win", "probability"),
    "teamplay_close_game_prob_hat": ("close_game", "probability"),
    "teamplay_possession_hat": ("possession", "bounded"),
    "teamplay_runs_hat": ("runs", "continuous"),
    "teamplay_metres_hat": ("carries_metres", "continuous"),
    "teamplay_tackles_required_hat": ("tackles", "continuous"),
    "teamplay_kicks_from_hand_hat": ("kicks_from_hand", "continuous"),
    "teamplay_penalty_goal_opportunity_hat": ("penalty_goals", "continuous"),
}


def _decayed_mean(values: pd.Series, dates: pd.Series, asof: pd.Timestamp, half_life: float) -> float:
    if values.empty:
        return np.nan
    days = (asof - dates).dt.days.to_numpy(dtype=float)
    w = np.power(0.5, days / half_life)
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    w = w[ok]
    v = v[ok]
    sw = w.sum()
    return float((w * v).sum() / sw) if sw > 0 else np.nan


def _context_rows(six: pd.DataFrame, half_life: float) -> pd.DataFrame:
    rows = []
    for r in six.itertuples(index=False):
        asof = pd.Timestamp(r.date)
        own = six[(six.team_id == r.team_id) & (six.date < asof)]
        opp = six[(six.team_id == r.opponent_id) & (six.date < asof)]
        against_opp = six[(six.opponent_id == r.opponent_id) & (six.date < asof)]
        h2h = six[
            (six.team_id == r.team_id)
            & (six.opponent_id == r.opponent_id)
            & (six.date < asof)
        ].sort_values("date").tail(3)

        rec = {
            "season": int(r.season),
            "round": int(r.round),
            "fixture_id": int(r.fixture_id),
            "team_id": int(r.team_id),
            "team": r.team,
            "opponent_id": int(r.opponent_id),
            "opponent": r.opponent,
            "date": asof,
            "teamplay_team_n_prior": int(len(own)),
            "teamplay_opp_n_prior": int(len(opp)),
            "teamplay_h2h_n_prior": int(len(h2h)),
            "home_away_num": 1.0 if str(r.home_away).lower() == "home" else 0.0,
        }
        for col in CONTEXT_COLS:
            rec[f"ctx_own_{col}"] = _decayed_mean(own[col], own.date, asof, half_life)
            rec[f"ctx_opp_{col}"] = _decayed_mean(opp[col], opp.date, asof, half_life)
            rec[f"ctx_vsopp_{col}"] = _decayed_mean(
                against_opp[col], against_opp.date, asof, half_life
            )
        rec["ctx_h2h_margin"] = float(h2h.margin.mean()) if len(h2h) else np.nan
        rec["ctx_h2h_winrate"] = float((h2h.margin > 0).mean()) if len(h2h) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _merge_wr(ctx: pd.DataFrame) -> pd.DataFrame:
    wr_path = DATA / "wr_rankings.csv"
    if not wr_path.exists():
        return ctx
    wr = pd.read_csv(wr_path)[["snapshot_date", "team", "wr_pos", "wr_pts"]].copy()
    wr["snapshot_date"] = pd.to_datetime(wr["snapshot_date"], errors="coerce")
    ctx = ctx.copy()
    ctx["_date_key"] = pd.to_datetime(ctx["date"]).dt.date.astype(str)
    wr["date"] = wr["snapshot_date"].dt.date.astype(str)
    own = wr.rename(columns={"team": "team", "wr_pos": "team_wr_pos", "wr_pts": "team_wr_pts"})
    opp = wr.rename(columns={"team": "opponent", "wr_pos": "opp_wr_pos", "wr_pts": "opp_wr_pts"})
    ctx = ctx.merge(own[["date", "team", "team_wr_pos", "team_wr_pts"]],
                    left_on=["_date_key", "team"], right_on=["date", "team"], how="left")
    ctx = ctx.drop(columns=["date_y"]).rename(columns={"date_x": "date"})
    ctx = ctx.merge(opp[["date", "opponent", "opp_wr_pos", "opp_wr_pts"]],
                    left_on=["_date_key", "opponent"], right_on=["date", "opponent"], how="left")
    ctx = ctx.drop(columns=["date_y", "_date_key"]).rename(columns={"date_x": "date"})
    ctx["ctx_wr_pts_gap"] = ctx["team_wr_pts"] - ctx["opp_wr_pts"]
    ctx["ctx_wr_rank_gap"] = ctx["opp_wr_pos"] - ctx["team_wr_pos"]
    return ctx


def _add_targets(six: pd.DataFrame) -> pd.DataFrame:
    out = six.copy()
    out["total_points"] = out["team_score"] + out["opp_score"]
    out["win"] = (out["margin"] > 0).astype(float)
    out["dominant_win"] = (out["margin"] >= 8).astype(float)
    out["close_game"] = (out["margin"].abs() <= 7).astype(float)
    return out


def _predict_by_date(frame: pd.DataFrame, feature_cols: list[str], target: str, kind: str) -> np.ndarray:
    yhat = np.full(len(frame), np.nan, dtype=float)
    for asof in sorted(frame["date"].dropna().unique()):
        tr = (frame["date"] < asof).to_numpy()
        te = (frame["date"] == asof).to_numpy()
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        ok = tr & np.isfinite(y)
        if ok.sum() < 24 or np.nanstd(y[ok]) < 1e-9:
            fill = float(np.nanmean(y[ok])) if ok.any() else float(np.nanmean(y))
            yhat[te] = fill
            continue
        cols = [
            c for c in feature_cols
            if frame.loc[ok, c].notna().any()
        ]
        if not cols:
            yhat[te] = float(np.nanmean(y[ok]))
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=50.0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.loc[ok, cols], y[ok])
            yhat[te] = model.predict(frame.loc[te, cols])

    if kind == "probability":
        yhat = np.clip(yhat, 0.0, 1.0)
    elif kind == "bounded":
        yhat = np.clip(yhat, 0.0, 1.0)
    elif target != "margin":
        yhat = np.clip(yhat, 0.0, None)
    return yhat


def _safe_div(num: pd.Series, den: float) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / float(den)


def _add_aspects(out: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable match-shape aspect features from own-vs-opponent predictions.

    The raw predictions answer "what do we expect this team to do?".  These
    aspects answer the next question the player model cares about: is this side
    expected to own possession/territory, play on the front foot, defend a lot,
    or create goal-kicking chances?
    """
    pred_cols = [c for c in out.columns if c.startswith("teamplay_") and c.endswith("_hat")]
    opp = out[["fixture_id", "team_id", *pred_cols]].copy()
    opp = opp.rename(columns={"team_id": "_opp_team_id", **{c: f"_opp_{c}" for c in pred_cols}})
    merged = out.merge(
        opp,
        left_on=["fixture_id", "opponent_id"],
        right_on=["fixture_id", "_opp_team_id"],
        how="left",
        validate="many_to_one",
    )

    def own(col: str) -> pd.Series:
        return pd.to_numeric(merged[col], errors="coerce")

    def edge(col: str) -> pd.Series:
        return own(col) - pd.to_numeric(merged[f"_opp_{col}"], errors="coerce")

    points_edge = edge("teamplay_points_for_hat")
    possession_edge = edge("teamplay_possession_hat")
    runs_edge = edge("teamplay_runs_hat")
    metres_edge = edge("teamplay_metres_hat")
    tackles_edge = edge("teamplay_tackles_required_hat")

    out = out.copy()
    out["teamplay_aspect_points_edge_hat"] = points_edge
    out["teamplay_aspect_win_edge_hat"] = edge("teamplay_win_prob_hat")
    out["teamplay_aspect_dominance_edge_hat"] = edge("teamplay_dominance_prob_hat")
    out["teamplay_aspect_possession_edge_hat"] = possession_edge
    out["teamplay_aspect_runs_edge_hat"] = runs_edge
    out["teamplay_aspect_metres_edge_hat"] = metres_edge
    out["teamplay_aspect_tackle_load_edge_hat"] = tackles_edge

    out["teamplay_aspect_attack_volume_hat"] = (
        _safe_div(own("teamplay_points_for_hat"), 30.0)
        + _safe_div(own("teamplay_runs_hat"), 110.0)
        + _safe_div(own("teamplay_metres_hat"), 500.0)
        + _safe_div(own("teamplay_possession_hat"), 0.50)
    ) / 4.0
    out["teamplay_aspect_attack_balance_hat"] = (
        _safe_div(points_edge, 20.0)
        + _safe_div(possession_edge, 0.10)
        + _safe_div(runs_edge, 30.0)
        + _safe_div(metres_edge, 150.0)
    ) / 4.0
    out["teamplay_aspect_defensive_load_hat"] = (
        _safe_div(own("teamplay_tackles_required_hat"), 150.0)
        - _safe_div(possession_edge, 0.10)
        + _safe_div(pd.to_numeric(merged["_opp_teamplay_runs_hat"], errors="coerce"), 110.0)
    ) / 3.0
    out["teamplay_aspect_open_game_hat"] = (
        _safe_div(own("teamplay_total_points_hat"), 50.0)
        + _safe_div(own("teamplay_runs_hat"), 110.0)
        + _safe_div(own("teamplay_metres_hat"), 500.0)
    ) / 3.0
    out["teamplay_aspect_kicking_opportunity_hat"] = (
        _safe_div(own("teamplay_penalty_goal_opportunity_hat"), 2.0)
        + _safe_div(own("teamplay_points_for_hat"), 30.0)
        + own("teamplay_close_game_prob_hat")
    ) / 3.0
    out["teamplay_aspect_pressure_hat"] = (
        -_safe_div(points_edge, 20.0)
        - _safe_div(possession_edge, 0.10)
        + _safe_div(own("teamplay_tackles_required_hat"), 150.0)
    ) / 3.0
    return out


def build(team_csv: Path, half_life: float) -> pd.DataFrame:
    raw = pd.read_csv(team_csv)
    raw = raw[raw.comp_id == SIX_NATIONS].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "team_id", "opponent_id", "fixture_id"])
    raw = raw.sort_values(["date", "fixture_id", "team_id"]).reset_index(drop=True)
    raw = _add_targets(raw)

    ctx = _context_rows(raw, half_life)
    ctx = _merge_wr(ctx)
    frame = ctx.merge(
        raw[[
            "fixture_id", "team_id", "team_score", "opp_score", "margin",
            "total_points", "win", "dominant_win", "close_game", "possession",
            "runs", "carries_metres", "tackles", "kicks_from_hand", "penalty_goals",
        ]],
        on=["fixture_id", "team_id"],
        how="left",
        validate="one_to_one",
    )
    feature_cols = [
        c for c in frame.columns
        if c.startswith("ctx_") or c.startswith("teamplay_") or c == "home_away_num"
    ]

    out_cols = [
        "season", "round", "fixture_id", "team_id", "team", "opponent_id",
        "opponent", "date", "teamplay_team_n_prior", "teamplay_opp_n_prior",
        "teamplay_h2h_n_prior",
    ]
    out = frame[out_cols].copy()
    for name, (target, kind) in PREDICTION_TARGETS.items():
        out[name] = _predict_by_date(frame, feature_cols, target, kind)

    out = _add_aspects(out)
    out = out[out["season"].isin(TARGET_SEASONS)].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-csv", default=str(DATA / "api_team_match.csv"))
    ap.add_argument("--half-life", type=float, default=365.0,
                    help="decay half-life in days for team context (default 365)")
    args = ap.parse_args()

    out = build(Path(args.team_csv), args.half_life)
    path = DATA / "team_play_predictions.csv"
    out.to_csv(path, index=False)

    pred_cols = [c for c in out.columns if c.startswith("teamplay_") and c.endswith("_hat")]
    print(f"✅  {len(out)} (fixture, team) rows; {out.fixture_id.nunique()} fixtures")
    print(f"    {len(pred_cols)} team-play predictions: {pred_cols}")
    print(f"    prior coverage: team n mean={out.teamplay_team_n_prior.mean():.1f}, "
          f"opp n mean={out.teamplay_opp_n_prior.mean():.1f}")
    print(f"💾  {path}")


if __name__ == "__main__":
    main()
