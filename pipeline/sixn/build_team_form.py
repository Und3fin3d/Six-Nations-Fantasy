from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

CONCEDE_COLS = {
    "tries_for":         "tries",
    "carries_metres":    "metres",
    "defenders_beaten":  "db",
    "clean_breaks":      "breaks",
    "offload":           "offloads",
    "passes":            "passes",
    "lineout_won_steal": "lineout_steal",
    "scrums_won":        "scrums",
}
OWN_COLS = {
    "turnovers_conceded": "turnovers_conceded",
    "missed_tackles":     "missed_tackles",
    "runs":               "runs",
    "tackles":            "tackles",
    "possession":         "possession",
    "penalties_conceded": "pens_conceded",
    "scrums_success":     "scrum_success",
    "scrums_won":         "scrums_won",
    "lineout_success":    "lineout_success",
    "lineout_won_steal":  "lineout_steal",
}


def _load_h2h_meetings(team_df: pd.DataFrame) -> pd.DataFrame:
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
    pool["_pref"] = (pool["_src"] == "api").astype(int)
    pool = (pool.sort_values("_pref", ascending=False)
                .drop_duplicates(subset=["date", "team", "opponent"], keep="first")
                .drop(columns="_pref"))
    return pool


def _decayed_mean(values: pd.Series, days_ago: pd.Series, half_life: float) -> float:
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

    h2h_pool = _load_h2h_meetings(df)

    out_rows = []
    for r in df.itertuples(index=False):
        asof = r.date
        y = r.opponent_id

        against_y = df[(df.opponent_id == y) & (df.date < asof)]
        y_own = df[(df.team_id == y) & (df.date < asof)]

        rec = {
            "fixture_id": r.fixture_id,
            "team_id": r.team_id,
            "team": r.team,
            "opponent_id": y,
            "opponent": r.opponent,
            "date": asof.date().isoformat(),
            "opp_n_prior": int(len(against_y)),
        }
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
    fd["wr_pts_gap"] = fd.team_wr_pts - fd.opp_wr_pts
    fd["wr_rank_gap"] = fd.opp_wr_pos - fd.team_wr_pos
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
