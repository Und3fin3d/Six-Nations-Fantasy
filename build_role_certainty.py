#!/usr/bin/env python3
"""Build team-sheet-relative role-certainty features.

Input:  data/model_player_match.csv
Output: data/external_player_roles.csv

This is not player hardcoding.  It converts PIT role history plus the named
team-sheet context into probabilities such as "likely primary kicker within this
23" and "lineout role certainty within this pack".

Run order:
1. build_features.py          # creates base PIT role/form features
2. build_role_certainty.py    # derives rolecert_* from the base feature store
3. build_features.py          # merges rolecert_* back into model_player_match
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"

BACK_POSITIONS = {"Scrum-half", "Fly-half", "Centre", "Back-three"}
BENCH_CREATOR_JERSEYS = {21, 22, 23}


def _safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _normalise_positive(x: pd.Series) -> pd.Series:
    x = _safe_num(x, 0.0).clip(lower=0.0)
    total = float(x.sum())
    if total <= 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return x / total


def _position_kick_prior(pos: pd.Series, jersey: pd.Series) -> pd.Series:
    # Very small role prior used only when historical kick evidence is thin.
    p = pd.Series(0.02, index=pos.index, dtype=float)
    p = p.mask(pos.eq("Fly-half"), 0.40)
    p = p.mask(pos.eq("Back-three"), 0.12)
    p = p.mask(pos.eq("Centre"), 0.06)
    p = p.mask(pos.eq("Scrum-half"), 0.04)
    j = _safe_num(jersey, 0)
    p = p.mask(j.eq(10), p + 0.35)
    p = p.mask(j.eq(15), p + 0.08)
    p = p.mask(j.isin([22, 23]), p + 0.04)
    return p


def _position_lineout_prior(pos: pd.Series) -> pd.Series:
    p = pd.Series(0.01, index=pos.index, dtype=float)
    p = p.mask(pos.eq("Second-row"), 0.55)
    p = p.mask(pos.eq("Back-row"), 0.28)
    p = p.mask(pos.eq("Hooker"), 0.03)
    return p


def _within_team_roles(g: pd.DataFrame) -> pd.DataFrame:
    out = g[[
        "season", "round", "fixture_id", "team_id", "team",
        "player_id", "player_name",
    ]].copy()
    started = g["started"].astype(bool)
    jersey = _safe_num(g["jersey"], 0)
    pos = g["canonical_pos"].astype(str)
    bench = ~started

    kick_history = (
        _safe_num(g["role_goal_kicker_rate"], 0.0)
        * (1.0 + np.log1p(_safe_num(g["role_kick_attempts"], 0.0)))
    )
    kick_signal = kick_history + 0.15 * _position_kick_prior(pos, jersey)
    # Being named to start is evidence the historic kicker role can be used for
    # most of the match.  Bench history is still signal, but lower certainty.
    kick_signal = kick_signal * np.where(started, 1.0, 0.45)
    kick_prob = _normalise_positive(kick_signal)
    top_kick = float(kick_prob.max()) if len(kick_prob) else 0.0
    out["rolecert_goal_kicker_prob"] = kick_prob.to_numpy(float)
    out["rolecert_primary_goal_kicker"] = (kick_prob >= top_kick - 1e-12).astype(float)
    out["rolecert_team_kicker_confidence"] = top_kick
    out["rolecert_kicking_ambiguity"] = 1.0 - top_kick

    bench_creator = bench & (pos.isin(BACK_POSITIONS) | jersey.isin(BENCH_CREATOR_JERSEYS))
    out["rolecert_bench_fh_kick_share"] = np.where(
        bench_creator,
        kick_prob.to_numpy(float),
        0.0,
    )
    out["rolecert_bench_kick_takeover_risk"] = np.where(
        bench_creator,
        kick_prob.to_numpy(float) * top_kick,
        0.0,
    )

    lineout_signal = (
        _safe_num(g["role_lineout_per80"], 0.0)
        + _position_lineout_prior(pos)
    ) * np.where(started, 1.0, 0.35)
    lineout_prob = _normalise_positive(lineout_signal)
    top_lineout = float(lineout_prob.max()) if len(lineout_prob) else 0.0
    out["rolecert_lineout_role_prob"] = lineout_prob.to_numpy(float)
    out["rolecert_primary_lineout_role"] = (lineout_prob >= top_lineout - 1e-12).astype(float)
    out["rolecert_team_lineout_confidence"] = top_lineout

    potm_signal = (
        _safe_num(g["role_potm_rate"], 0.0)
        + 0.02 * _safe_num(g["role_potm_n_label"], 0.0)
        + 0.01 * started.astype(float)
    )
    out["rolecert_potm_role_prob"] = _normalise_positive(potm_signal).to_numpy(float)

    form_start = _safe_num(g["form_start_rate"], 0.0).clip(0.0, 1.0)
    recent_minutes = _safe_num(g["form_minutes_recent"], 0.0).clip(0.0, 80.0) / 80.0
    bench_back = bench & pos.isin(BACK_POSITIONS)
    out["rolecert_replacement_role_uncertainty"] = np.where(
        bench,
        (0.55 + 0.25 * bench_back.astype(float)) * (1.0 - 0.5 * form_start) * (1.0 - 0.25 * recent_minutes),
        0.0,
    )
    return out


def build(feature_csv: Path) -> pd.DataFrame:
    feat = pd.read_csv(feature_csv)
    feat = feat[feat["season"].isin([2023, 2024, 2025, 2026])].copy()
    required = {
        "season", "round", "fixture_id", "team_id", "team", "player_id",
        "player_name", "started", "jersey", "canonical_pos", "role_goal_kicker_rate",
        "role_kick_attempts", "role_lineout_per80", "role_potm_rate",
        "role_potm_n_label", "form_start_rate", "form_minutes_recent",
    }
    missing = sorted(required - set(feat.columns))
    if missing:
        raise ValueError(f"missing required columns in {feature_csv}: {missing}")
    out = (
        feat.groupby(["fixture_id", "team_id"], sort=False, group_keys=False)
        .apply(_within_team_roles)
        .reset_index(drop=True)
    )
    if out.duplicated(["fixture_id", "player_id"]).any():
        raise ValueError("duplicate role-certainty rows after build")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-csv", default=str(DATA / "model_player_match.csv"))
    ap.add_argument("--output", default=str(DATA / "external_player_roles.csv"))
    args = ap.parse_args()

    out = build(Path(args.feature_csv))
    path = Path(args.output)
    out.to_csv(path, index=False)
    print(f"OK {len(out)} player-role rows -> {path}")
    cols = [
        "season", "round", "team", "player_name", "rolecert_goal_kicker_prob",
        "rolecert_bench_fh_kick_share", "rolecert_lineout_role_prob",
        "rolecert_replacement_role_uncertainty",
    ]
    print(out[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
