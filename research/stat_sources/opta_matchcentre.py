"""Opta public match-centre widgets (englandrugby.com match centre) vs the
official Six Nations fantasy workbook and the repo store.

The England Rugby match centre renders Stats Perform/Opta widgets
(`<opta-widget sport="rugby" competition="209" season="2026" match="947118">`).
Rendering requires a browser (Playwright) and the site's widget key, so we do not
re-implement the scrape here; instead we test the published scrape of those
widgets (github.com/bpcsaund/rugby-analytics data/raw/rfu_player_stats.csv:
165 England tests 2013-2026, both teams, 7.4k player rows) against our labels.

Question answered: do Opta "Turnovers won" / "Metres" (public widget definitions)
reproduce the official Six Nations fantasy categories BS (breakdown steals) and
MC (metres carried) better than our RapidAPI proxies?

    python research/stat_sources/opta_matchcentre.py --rfu PATH/rfu_player_stats.csv [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, agreement, load_store, surname_key  # noqa: E402

SIX_NATIONS_OPTA_COMP = 209


def build(rfu_path: str, store_path: str | None = None) -> pd.DataFrame:
    rfu = pd.read_csv(rfu_path)
    rfu = rfu[rfu.competition == SIX_NATIONS_OPTA_COMP].copy()
    store = load_store(store_path, competitions=["Six Nations"])
    off = pd.read_csv(REPO / "data" / "official_player_match.csv")
    off["sur"] = off.name.map(surname_key)

    # match -> (season, frozenset(teams)) -> store fixture / round
    teams = rfu.groupby("match_id").agg(season=("season", "first"), teams=("team", lambda s: frozenset(s)))
    fx = store.groupby("fixture_id").agg(season=("season", "first"), round=("round", "first"),
                                         teams=("team", lambda s: frozenset(s)))
    fx["season"] = fx.season.astype(int)
    key = {(r.season, r.teams): (fid, r["round"]) for fid, r in fx.iterrows()}
    teams["fixture_id"] = [key.get((s, t), (None, None))[0] for s, t in zip(teams.season, teams.teams)]
    teams["round"] = [key.get((s, t), (None, None))[1] for s, t in zip(teams.season, teams.teams)]
    rfu = rfu.merge(teams[["fixture_id", "round"]], left_on="match_id", right_index=True, how="left")
    rfu["sur"] = rfu.player.map(surname_key)

    # unique surname within team-fixture on every side
    def uniq(df, keys):
        return df.groupby(keys).filter(lambda g: len(g) == 1)

    r_u = uniq(rfu[rfu.fixture_id.notna()], ["fixture_id", "team", "sur"])
    s_u = uniq(store, ["fixture_id", "team", "sur"])
    j = r_u.merge(s_u, on=["fixture_id", "team", "sur"], how="left", suffixes=("", "_store"))
    o_u = uniq(off, ["season", "round", "team", "sur"])
    j = j.merge(o_u[["season", "round", "team", "sur", "Min", "MC", "Ta", "BS", "LS", "KR", "POTM"]],
                on=["season", "round", "team", "sur"], how="left")
    return j


def summarise(j: pd.DataFrame) -> dict:
    lab = j[j.MC.notna()]
    res = {
        "rfu_6n_rows": int(len(j)), "rfu_6n_matches": int(j.match_id.nunique()),
        "joined_store": int(j.player_id.notna().sum()),
        "joined_official": int(j.MC.notna().sum()),
        "official_matches": int(lab.match_id.nunique()),
        "vs_official": {
            "opta_metres~MC": agreement(lab.Metres_Gained, lab.MC),
            "api_metres~MC": agreement(lab.metres, lab.MC),
            "opta_tackles~Ta": agreement(lab.Tackles_Attempted, lab.Ta),
            "api_tackles~Ta": agreement(lab.tackles, lab.Ta),
            "opta_turnovers_won~BS": agreement(lab.Turnovers_Won, lab.BS),
            "api_tackle_turnover~BS": agreement(lab.tackle_turnover, lab.BS),
            "opta_lineout_steals~LS": agreement(lab.Lineout_Steals, lab.LS),
        },
        "vs_store_all_6n": {
            "opta_metres~api_metres": agreement(j.Metres_Gained, j.metres),
            "opta_tackles~api_tackles": agreement(j.Tackles_Attempted, j.tackles),
            "opta_turnovers_won~api_tackle_turnover": agreement(j.Turnovers_Won, j.tackle_turnover),
        },
    }
    # BS hit-rate: when official BS>0, how often does each proxy say >0?
    pos = lab[lab.BS > 0]
    if len(pos):
        res["BS_positive_rows"] = int(len(pos))
        res["opta_TW>0_given_BS>0"] = round(float((pos.Turnovers_Won > 0).mean()), 3)
        res["api_TT>0_given_BS>0"] = round(float((pos.tackle_turnover > 0).mean()), 3)
        neg = lab[lab.BS == 0]
        res["opta_TW>0_given_BS=0"] = round(float((neg.Turnovers_Won > 0).mean()), 3)
        res["api_TT>0_given_BS=0"] = round(float((neg.tackle_turnover > 0).mean()), 3)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfu", required=True)
    ap.add_argument("--store")
    ap.add_argument("--json")
    ap.add_argument("--csv", help="write joined rows")
    a = ap.parse_args(argv)
    j = build(a.rfu, a.store)
    res = summarise(j)
    print(json.dumps(res, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str))
    if a.csv:
        j.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
