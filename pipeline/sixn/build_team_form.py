from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 90.0
CONCEDE_COLS = {
    "tries_for": "tries", "carries_metres": "metres", "defenders_beaten": "db", "clean_breaks": "breaks", "offload": "offloads", "passes": "passes",
    "lineout_won_steal": "lineout_steal", "scrums_won": "scrums",
}
OWN_COLS = {
    "turnovers_conceded": "turnovers_conceded", "missed_tackles": "missed_tackles", "runs": "runs", "tackles": "tackles", "possession": "possession",
    "penalties_conceded": "pens_conceded", "scrums_success": "scrum_success", "scrums_won": "scrums_won", "lineout_success": "lineout_success",
    "lineout_won_steal": "lineout_steal",
}


def _load_h2h_meetings(team_df: pd.DataFrame) -> pd.DataFrame:
    intl = pd.read_csv("data/intl_results.csv")[["team", "opponent", "date", "margin"]].copy()
    intl["date"] = pd.to_datetime(intl["date"], errors="coerce")
    pool = pd.concat(
        [team_df[["team", "opponent", "date", "margin"]].assign(_src="api"), intl.dropna(subset=["team", "opponent", "date"]).assign(_src="intl")],
        ignore_index=True,
    )
    pool["_pref"] = (pool["_src"] == "api").astype(int)
    return pool.sort_values("_pref", ascending=False).drop_duplicates(subset=["date", "team", "opponent"], keep="first").drop(columns="_pref")


def _decayed_mean(frame: pd.DataFrame, col: str, asof: pd.Timestamp) -> float:
    if frame.empty:
        return np.nan
    w = np.power(0.5, (asof - frame.date).dt.days.to_numpy(dtype=float) / HALF_LIFE)
    return float((w * frame[col].to_numpy(dtype=float)).sum() / w.sum()) if w.sum() > 0 else np.nan


def build() -> pd.DataFrame:
    df = pd.read_csv("data/api_team_match.csv")
    df = df[df.comp_id.isin([1266])].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team_id", "opponent_id"])
    h2h_pool = _load_h2h_meetings(df)
    rows = []
    for r in df.itertuples(index=False):
        asof, y = r.date, r.opponent_id
        against_y, y_own = df[(df.opponent_id == y) & (df.date < asof)], df[(df.team_id == y) & (df.date < asof)]
        h2h = h2h_pool[(h2h_pool.team == r.team) & (h2h_pool.opponent == r.opponent) & (h2h_pool.date < asof)].sort_values("date").tail(3)
        rows.append(
            {
                "fixture_id": r.fixture_id,
                "team_id": r.team_id,
                "team": r.team,
                "opponent_id": y,
                "opponent": r.opponent,
                "date": asof.date().isoformat(),
                "opp_n_prior": int(len(against_y)),
                "h2h_n_prior": int(len(h2h)),
                "h2h_last3_winrate": float((h2h.margin > 0).mean()) if len(h2h) else np.nan,
                "h2h_last3_margin": float(h2h.margin.mean()) if len(h2h) else np.nan,
                **{f"opp_concede_{suf}": _decayed_mean(against_y, col, asof) for col, suf in CONCEDE_COLS.items()},
                **{f"opp_{suf}": _decayed_mean(y_own, col, asof) for col, suf in OWN_COLS.items()},
            }
        )
    wr = pd.read_csv("data/wr_rankings.csv")[["snapshot_date", "team", "wr_pos", "wr_pts"]].rename(columns={"snapshot_date": "date"})
    fd = pd.DataFrame(rows).merge(wr.rename(columns={"wr_pos": "team_wr_pos", "wr_pts": "team_wr_pts"}), on=["date", "team"], how="left")
    fd = fd.merge(wr.rename(columns={"team": "opponent", "wr_pos": "opp_wr_pos", "wr_pts": "opp_wr_pts"}), on=["date", "opponent"], how="left")
    fd["wr_pts_gap"] = fd.team_wr_pts - fd.opp_wr_pts
    fd["wr_rank_gap"] = fd.opp_wr_pos - fd.team_wr_pos
    return fd


if __name__ == "__main__":
    fd = build()
    fd.to_csv("data/fixture_difficulty.csv", index=False)
    cold = (fd.opp_n_prior == 0).sum()
    feats = [c for c in fd.columns if c not in {"fixture_id", "team_id", "team", "opponent_id", "opponent", "date"}]
    print(f"✅  {len(fd)} (fixture, team) rows; {fd.fixture_id.nunique()} matches; comps [1266]; half-life {HALF_LIFE:g}d")
    print(f"    cold-start (no prior on opponent): {cold} rows ({100 * cold / len(fd):.0f}%) → NaN, shrink later")
    print(f"    WR coverage: {100 * fd['wr_pts_gap'].notna().mean():.0f}% of rows have wr_pts_gap")
    print(f"    {len(feats)} features: {feats}")
    print("💾  data/fixture_difficulty.csv")
