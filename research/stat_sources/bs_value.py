"""Value test: does pre-tournament club Opta "turnovers won" (InCrowd feed) predict
official Six Nations breakdown steals (BS) better than our RapidAPI tackle_turnover?

Point-in-time: club rates use only club matches dated before the season's first
Six Nations fixture. Labels: data/official_player_match.csv (2025, 2026; 2023 has
no 2024-25 club mirror coverage).

    python research/stat_sources/bs_value.py --mirror "DIR/[ptucj]*_player_stats.csv"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, initial_key, join_players, load_store, match_fixtures  # noqa: E402
from incrowd import load_mirror  # noqa: E402


CLUB_COMPS = {"Investec Champions Cup", "United Rugby Championship", "TOP 14", "Challenge Cup",
              "PREM Rugby", "Japan Rugby League One", "Super Rugby Pacific"}


def official_with_ids(store: pd.DataFrame) -> pd.DataFrame:
    off = pd.read_csv(REPO / "data" / "official_player_match.csv")
    six = store[store.competition == "Six Nations"].copy()
    six["season"] = six.season.astype(int)
    six["round"] = six["round"].astype(int)
    six = six.groupby(["season", "round", "team", "ikey"]).filter(lambda g: len(g) == 1)
    off["ikey"] = off.key.astype(str)
    j = off.merge(six[["season", "round", "team", "ikey", "player_id", "date", "tackle_turnover",
                       "minutes", "position"]],
                  on=["season", "round", "team", "ikey"], how="inner")
    return j


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--store")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    store = load_store(a.store)
    ext = load_mirror(a.mirror)
    ext = ext[~ext.source_file.str.startswith("premcup")]
    st = store[store.date >= pd.to_datetime(ext.match_date).min().date() - pd.Timedelta(days=2)]
    fmap = match_fixtures(ext, st)
    club = join_players(ext, st, fmap)
    club["date"] = pd.to_datetime(club["date"])
    off = official_with_ids(store)
    off["date"] = pd.to_datetime(off["date"])
    out = {}
    rows = []
    for season in (2025, 2026):
        start = off[off.season == season].date.min()
        lo = start - pd.Timedelta(days=200)
        c = club[(club.date < start) & (club.date >= lo)]
        g = c.groupby("player_id").agg(club_min=("minutesPlayedTotal", "sum"),
                                       tw=("turnoverWon", "sum"), tt=("tackle_turnover", "sum"),
                                       tk=("tackles", "sum"), n=("fixture_id", "nunique"))
        g = g[g.club_min >= 160]
        g["tw80"] = g.tw / g.club_min * 80
        g["tt80"] = g.tt / g.club_min * 80
        g["tk80"] = g.tk / g.club_min * 80
        o = off[(off.season == season) & (off.Min > 0)]
        po = o.groupby("player_id").agg(bs=("BS", "sum"), mins=("Min", "sum"), pos=("position", "first"))
        po = po[po.mins >= 80]
        po["bs80"] = po.bs / po.mins * 80
        m = po.join(g, how="inner")
        m["season"] = season
        # baseline already in the repo: RugbyPass completed prior club season (CLASS)
        rp = pd.read_csv(REPO / "data" / "rp_compstats.csv")
        rp = rp[(rp.season == f"{season - 2}/{season - 1}") & rp.competition.isin(CLUB_COMPS)]
        rpg = rp.groupby("key").agg(rp_tw=("turnovers_won", "sum"), rp_min=("minutes", "sum"))
        rpg = rpg[rpg.rp_min >= 160]
        rpg["rp_tw80"] = rpg.rp_tw / rpg.rp_min * 80
        keymap = o.groupby("player_id").key.first()
        m["key"] = keymap.reindex(m.index).values
        m = m.merge(rpg[["rp_tw80"]], left_on="key", right_index=True, how="left")
        rows.append(m)
    m = pd.concat(rows)
    res = {"players": int(len(m)), "per_season": m.groupby("season").size().to_dict()}
    both = m[m.rp_tw80.notna()]
    res["with_rp_prior_season"] = int(len(both))
    for col in ("tw80", "rp_tw80", "tt80"):
        res[f"common_pearson_{col}~official_bs80"] = round(float(both[col].corr(both.bs80)), 3)
    res["common_pearson_mean(tw80,rp_tw80)~official_bs80"] = round(float(
        ((both.tw80 + both.rp_tw80) / 2).corr(both.bs80)), 3)
    for col in ("tw80", "tt80", "tk80"):
        res[f"pearson_{col}~official_bs80"] = round(float(m[col].corr(m.bs80)), 3)
        res[f"spearman_{col}~official_bs80"] = round(float(m[col].corr(m.bs80, method="spearman")), 3)
    fw = m[m.pos.astype(str).str.contains("Flank|Number|Lock|Back Row|Hooker|Prop", case=False, regex=True)]
    res["forwards_n"] = int(len(fw))
    for col in ("tw80", "tt80"):
        res[f"forwards_pearson_{col}~official_bs80"] = round(float(fw[col].corr(fw.bs80)), 3)
    # simple out-of-sample check: fit bs80 ~ rate on 2025, score 2026
    for col in ("tw80", "tt80"):
        tr, te = m[m.season == 2025], m[m.season == 2026]
        b = np.polyfit(tr[col], tr.bs80, 1)
        pred = np.polyval(b, te[col])
        base = np.full(len(te), tr.bs80.mean())
        res[f"oos2026_mae_{col}"] = round(float(np.abs(pred - te.bs80).mean()), 4)
        res["oos2026_mae_const"] = round(float(np.abs(base - te.bs80).mean()), 4)
        res[f"oos2026_r_{col}"] = round(float(np.corrcoef(pred, te.bs80)[0, 1]), 3)
    print(json.dumps(res, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
