#!/usr/bin/env python3
"""Build optional PIT tactical-style features for Six Nations fixtures.

Output: data/external_team_style.csv

This layer summarizes how each named national side has recently played before a
fixture: tempo, kicking volume, attacking edge, tackle demand, discipline risk,
and set-piece strength.  It is intentionally feature-only and remains gated off
unless a research candidate enables `style_` features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]

STYLE_COLS = [
    "tries_for", "tries_against", "team_score", "opp_score", "runs", "passes",
    "carries_metres", "clean_breaks", "defenders_beaten", "offload",
    "kicks_from_hand", "tackles", "missed_tackles", "turnovers_conceded",
    "turnovers_won", "penalties_conceded", "possession", "scrums_success",
    "lineout_success",
]


def _decayed_mean(hist: pd.DataFrame, col: str, asof: pd.Timestamp, half_life: float) -> float:
    if hist.empty or col not in hist.columns:
        return np.nan
    v = pd.to_numeric(hist[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    days = (asof - hist["date"]).dt.days.to_numpy(float)[ok]
    w = np.power(0.5, days / half_life)
    sw = float(w.sum())
    return float((w * v[ok]).sum() / sw) if sw > 0 else np.nan


def _history_summary(hist: pd.DataFrame, asof: pd.Timestamp, half_life: float, prefix: str) -> dict:
    out = {f"{prefix}_{c}": _decayed_mean(hist, c, asof, half_life) for c in STYLE_COLS}
    out[f"{prefix}_n_prior"] = int(len(hist))
    attack = np.nanmean([
        out[f"{prefix}_tries_for"] / 4.0,
        out[f"{prefix}_carries_metres"] / 450.0,
        out[f"{prefix}_clean_breaks"] / 7.0,
        out[f"{prefix}_defenders_beaten"] / 24.0,
        out[f"{prefix}_offload"] / 8.0,
    ])
    tempo = np.nanmean([
        out[f"{prefix}_runs"] / 110.0,
        out[f"{prefix}_passes"] / 165.0,
        out[f"{prefix}_possession"] / 0.50,
    ])
    kick = out[f"{prefix}_kicks_from_hand"] / 28.0
    tackle_load = out[f"{prefix}_tackles"] / 145.0
    discipline = np.nanmean([
        out[f"{prefix}_penalties_conceded"] / 11.0,
        out[f"{prefix}_turnovers_conceded"] / 13.0,
    ])
    setpiece = np.nanmean([
        out[f"{prefix}_scrums_success"],
        out[f"{prefix}_lineout_success"],
    ])
    defensive_leak = np.nanmean([
        out[f"{prefix}_tries_against"] / 4.0,
        out[f"{prefix}_opp_score"] / 25.0,
        out[f"{prefix}_missed_tackles"] / 18.0,
    ])
    out[f"{prefix}_attack_index"] = float(attack)
    out[f"{prefix}_tempo_index"] = float(tempo)
    out[f"{prefix}_kick_index"] = float(kick) if np.isfinite(kick) else np.nan
    out[f"{prefix}_tackle_load_index"] = float(tackle_load) if np.isfinite(tackle_load) else np.nan
    out[f"{prefix}_discipline_risk_index"] = float(discipline)
    out[f"{prefix}_setpiece_index"] = float(setpiece)
    out[f"{prefix}_defensive_leak_index"] = float(defensive_leak)
    return out


def build(team_csv: Path, half_life: float) -> pd.DataFrame:
    raw = pd.read_csv(team_csv)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "team_id", "opponent_id", "fixture_id"])
    target = raw[(raw["comp_id"] == SIX_NATIONS) & raw["season"].isin(TARGET_SEASONS)].copy()
    target = target.sort_values(["date", "fixture_id", "team_id"])
    histories = {tid: g.sort_values("date") for tid, g in raw.groupby("team_id")}

    rows = []
    for r in target.itertuples(index=False):
        asof = pd.Timestamp(r.date)
        own_hist = histories.get(int(r.team_id), raw.iloc[0:0])
        opp_hist = histories.get(int(r.opponent_id), raw.iloc[0:0])
        own_hist = own_hist[own_hist["date"] < asof]
        opp_hist = opp_hist[opp_hist["date"] < asof]
        rec = {
            "season": int(r.season),
            "round": int(r.round),
            "fixture_id": int(r.fixture_id),
            "team_id": int(r.team_id),
            "team": r.team,
            "opponent_id": int(r.opponent_id),
            "opponent": r.opponent,
        }
        rec.update(_history_summary(own_hist, asof, half_life, "style_own"))
        rec.update(_history_summary(opp_hist, asof, half_life, "style_opp"))
        rec["style_attack_edge"] = (
            rec["style_own_attack_index"] + rec["style_opp_defensive_leak_index"] - 1.0
        )
        rec["style_tempo_edge"] = rec["style_own_tempo_index"] - rec["style_opp_tempo_index"]
        rec["style_kick_edge"] = rec["style_own_kick_index"] - rec["style_opp_kick_index"]
        rec["style_tackle_demand"] = np.nanmean([
            rec["style_own_tackle_load_index"],
            rec["style_opp_tempo_index"],
            rec["style_opp_attack_index"],
        ])
        rec["style_setpiece_edge"] = rec["style_own_setpiece_index"] - rec["style_opp_setpiece_index"]
        rec["style_discipline_edge"] = rec["style_own_discipline_risk_index"] - rec["style_opp_discipline_risk_index"]
        rec["style_aspect_attack_edge"] = rec["style_attack_edge"]
        rec["style_aspect_tempo_edge"] = rec["style_tempo_edge"]
        rec["style_aspect_kick_edge"] = rec["style_kick_edge"]
        rec["style_aspect_tackle_demand"] = rec["style_tackle_demand"]
        rec["style_aspect_setpiece_edge"] = rec["style_setpiece_edge"]
        rec["style_aspect_discipline_edge"] = rec["style_discipline_edge"]
        rec["style_aspect_own_attack"] = rec["style_own_attack_index"]
        rec["style_aspect_opp_attack"] = rec["style_opp_attack_index"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-csv", default=str(DATA / "api_team_match.csv"))
    ap.add_argument("--half-life", type=float, default=365.0)
    ap.add_argument("--output", default=str(DATA / "external_team_style.csv"))
    args = ap.parse_args()

    out = build(Path(args.team_csv), args.half_life)
    path = Path(args.output)
    out.to_csv(path, index=False)
    print(f"OK {len(out)} fixture-team rows -> {path}")
    print(out[[
        "season", "round", "team", "opponent", "style_own_n_prior",
        "style_own_attack_index", "style_own_tempo_index",
        "style_attack_edge", "style_tackle_demand",
    ]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
