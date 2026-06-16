#!/usr/bin/env python3
"""
Fixture difficulty  —  point-in-time opponent permissiveness
============================================================
Turns data/api_team_match.csv (one row per match-team) into a per-fixture
"how favourable is this opponent" feature vector, for the player model.

A row in the team table describes what `team` did *to* `opponent`. So an
opponent Y's difficulty profile is read two ways, strictly from matches that
finished BEFORE the fixture in question (no leakage), exponentially decayed
toward the fixture date (≈90-day half-life):

  • CONCEDED (what attacks do to Y)  — rows where opponent_id == Y:
      tries_for, carries_metres, defenders_beaten, clean_breaks, offload,
      passes  → how much attacking output Y leaks.
  • OWN     (what Y itself does)     — rows where team_id == Y:
      turnovers_conceded (ball Y coughs up → breakdown-steal chances for us),
      missed_tackles (Y's tackle frailty), runs/carries (Y's carry volume →
      tackle chances for us), tackles, possession, penalties_conceded.

National-team profiles are built from **internationals only** (per the design
decision); pass --comps to widen the international pool (6N is the default and
is fully cached → runs free). Club-form difficulty is a separate later build
off the same table with the club competitions.

Cold start (opponent unseen before the fixture) → NaN feature + n_prior=0, so
the modelling layer can shrink to a position×competition baseline.

Output: data/fixture_difficulty.csv, keyed (fixture_id, team_id) — left-join it
onto data/6n_player_match.csv on those two columns.

Usage:
    python build_team_form.py                     # default 90-day half-life, 6N
    python build_team_form.py --half-life 120 --comps 1266
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent

# opponent CONCEDED signals — read from rows where opponent_id == Y
#   team-table column            -> output feature suffix
CONCEDE_COLS = {
    "tries_for":         "tries",
    "carries_metres":    "metres",
    "defenders_beaten":  "db",
    "clean_breaks":      "breaks",
    "offload":           "offloads",
    "passes":            "passes",
    # set-piece MISMATCH signals — how much the opponent leaks at their own
    # set piece (steals made *against* Y, scrums won *against* Y). Team-stable
    # (Italy ~1.2 steals conceded/match vs Ireland ~0.5) → an expected-value
    # tilt for our jumpers/pack, though any single match is low-count noise.
    "lineout_won_steal": "lineout_steal",
    "scrums_won":        "scrums",
}
# opponent OWN signals — read from rows where team_id == Y
OWN_COLS = {
    "turnovers_conceded": "turnovers_conceded",   # → breakdown-steal chances for us
    "missed_tackles":     "missed_tackles",        # tackle frailty
    "runs":               "runs",                  # carry volume → tackle chances
    "tackles":            "tackles",
    "possession":         "possession",
    "penalties_conceded": "pens_conceded",
    # set-piece permissiveness — feeds the SW / LS latent prediction:
    "scrums_success":     "scrum_success",         # Y's scrum strength (↓ → SW for us)
    "scrums_won":         "scrums_won",
    "lineout_success":    "lineout_success",       # Y's lineout security (↓ → LS for us)
    "lineout_won_steal":  "lineout_steal",         # Y's own steal threat
}


def _load_h2h_meetings(team_df: pd.DataFrame) -> pd.DataFrame:
    """Prior-meetings pool for head-to-head, UNIONing the rich
    api_team_match rows (already in `team_df`) with the free Pulselive
    international results (data/intl_results.csv).

    This only ever deepens H2H (autumn/summer meetings now count); it does NOT
    feed the permissiveness vector, which stays sourced from `team_df`.

    Columns returned: team, opponent, date (datetime64), margin, _src.
    Dedup is by (date, team, opponent) preferring the api_team_match row (_src
    'api') over the Pulselive row ('intl'). No-op if intl_results.csv absent.
    Join is by team/opponent NAME — Pulselive ids are a different id-space."""
    base = team_df[["team", "opponent", "date", "margin"]].copy()
    base["_src"] = "api"

    intl_path = BASE / "data" / "intl_results.csv"
    if not intl_path.exists():
        return base

    intl = pd.read_csv(intl_path)
    intl = intl[["team", "opponent", "date", "margin"]].copy()
    intl["date"] = pd.to_datetime(intl["date"], errors="coerce")
    intl = intl.dropna(subset=["team", "opponent", "date"])
    intl["_src"] = "intl"

    pool = pd.concat([base, intl], ignore_index=True)
    # prefer 'api' over 'intl' on the same (date, team, opponent)
    pool["_pref"] = (pool["_src"] == "api").astype(int)
    pool = (pool.sort_values("_pref", ascending=False)
                .drop_duplicates(subset=["date", "team", "opponent"], keep="first")
                .drop(columns="_pref"))
    return pool


def _decayed_mean(values: pd.Series, days_ago: pd.Series, half_life: float) -> float:
    """Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life)."""
    if len(values) == 0:
        return np.nan
    w = np.power(0.5, days_ago.to_numpy(dtype=float) / half_life)
    v = values.to_numpy(dtype=float)
    sw = w.sum()
    return float((w * v).sum() / sw) if sw > 0 else np.nan


def build(team_csv: Path, comps: list[int], half_life: float) -> pd.DataFrame:
    df = pd.read_csv(team_csv)
    df = df[df.comp_id.isin(comps)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team_id", "opponent_id"])

    # H2H prior-meetings pool: api_team_match ∪ intl_results (by name+date).
    # Permissiveness vector below still reads only from `df` (rich rugby-live-data).
    h2h_pool = _load_h2h_meetings(df)

    out_rows = []
    for r in df.itertuples(index=False):
        asof = r.date
        y = r.opponent_id                       # the opponent we want a read on

        against_y = df[(df.opponent_id == y) & (df.date < asof)]   # attacks vs Y
        y_own = df[(df.team_id == y) & (df.date < asof)]           # Y's own lines

        rec = {
            "fixture_id": r.fixture_id,
            "team_id": r.team_id,
            "team": r.team,
            "opponent_id": y,
            "opponent": r.opponent,
            "date": asof.date().isoformat(),
            "opp_n_prior": int(len(against_y)),
        }
        # ---- head-to-head: last 3 meetings of THIS team vs THIS opponent ----
        # Match by NAME against the widened pool (api ∪ intl); Pulselive ids
        # are a different id-space so name is the only safe join key here.
        h2h = (h2h_pool[(h2h_pool.team == r.team)
                        & (h2h_pool.opponent == r.opponent)
                        & (h2h_pool.date < asof)]
               .sort_values("date").tail(3))
        rec["h2h_n_prior"] = int(len(h2h))
        rec["h2h_last3_winrate"] = float((h2h.margin > 0).mean()) if len(h2h) else np.nan
        rec["h2h_last3_margin"] = float(h2h.margin.mean()) if len(h2h) else np.nan
        ad = (asof - against_y.date).dt.days if len(against_y) else None
        for col, suf in CONCEDE_COLS.items():
            rec[f"opp_concede_{suf}"] = (
                _decayed_mean(against_y[col], ad, half_life) if len(against_y) else np.nan
            )
        od = (asof - y_own.date).dt.days if len(y_own) else None
        for col, suf in OWN_COLS.items():
            rec[f"opp_{suf}"] = (
                _decayed_mean(y_own[col], od, half_life) if len(y_own) else np.nan
            )
        out_rows.append(rec)

    fd = pd.DataFrame(out_rows)
    return _merge_wr(fd)


def _merge_wr(fd: pd.DataFrame) -> pd.DataFrame:
    """PIT-join World Rugby strength (data/wr_rankings.csv) by (date, team) and
    (date, opponent); add own/opp pts+pos and the gaps. No-op if file absent."""
    wr_path = BASE / "data" / "wr_rankings.csv"
    if not wr_path.exists() or fd.empty:
        return fd
    wr = pd.read_csv(wr_path)[["snapshot_date", "team", "wr_pos", "wr_pts"]]
    wr = wr.rename(columns={"snapshot_date": "date"})
    own = wr.rename(columns={"wr_pos": "team_wr_pos", "wr_pts": "team_wr_pts"})
    opp = wr.rename(columns={"team": "opponent",
                             "wr_pos": "opp_wr_pos", "wr_pts": "opp_wr_pts"})
    fd = fd.merge(own, on=["date", "team"], how="left")
    fd = fd.merge(opp, on=["date", "opponent"], how="left")
    fd["wr_pts_gap"] = fd.team_wr_pts - fd.opp_wr_pts          # +ve = we're stronger
    fd["wr_rank_gap"] = fd.opp_wr_pos - fd.team_wr_pos         # +ve = we're ranked higher
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-csv", default=str(BASE / "data" / "api_team_match.csv"))
    ap.add_argument("--comps", nargs="+", type=int, default=[1266],
                    help="competition ids forming the international pool (default 6N)")
    ap.add_argument("--half-life", type=float, default=90.0,
                    help="decay half-life in days (default 90)")
    args = ap.parse_args()

    fd = build(Path(args.team_csv), args.comps, args.half_life)
    out = BASE / "data" / "fixture_difficulty.csv"
    fd.to_csv(out, index=False)

    cold = (fd.opp_n_prior == 0).sum()
    idcols = {"fixture_id", "team_id", "team", "opponent_id", "opponent", "date"}
    feats = [c for c in fd.columns if c not in idcols]
    wr_cov = (100 * fd["wr_pts_gap"].notna().mean()) if "wr_pts_gap" in fd else 0
    print(f"✅  {len(fd)} (fixture, team) rows; "
          f"{fd.fixture_id.nunique()} matches; comps {args.comps}; "
          f"half-life {args.half_life:g}d")
    print(f"    cold-start (no prior on opponent): {cold} rows "
          f"({100*cold/len(fd):.0f}%) → NaN, shrink later")
    print(f"    WR coverage: {wr_cov:.0f}% of rows have wr_pts_gap")
    print(f"    {len(feats)} features: {feats}")
    print(f"💾  {out}")


if __name__ == "__main__":
    main()
