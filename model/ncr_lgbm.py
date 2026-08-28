#!/usr/bin/env python3
"""
model/ncr_lgbm.py — in-domain LightGBM with the NCR model's ideas
=================================================================
Same feature philosophy as the empirical-Bayes NCR model (decayed international
FORM, calibrated club FORM, RugbyPass CLASS prior, WR matchup, position, sample
size) — but instead of a fixed shrinkage blend, a LightGBM learns the mapping to
per-80 NCR-point components (att / dfn / dsc). Trained ON NCR match history
(every international player-match is a labelled row: PIT features → realised
per-80 that match), so it is in-domain, unlike the 6n-champion transplant.

Predicts the three components → feeds ncr_project.build_projection(override_rates)
→ same matchup / minutes / MILP / scoring. Evaluate via model.ncr_eval.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

import model.ncr_project as NP

ASOF = NP.ASOF
HL = NP.HALFLIFE_DAYS
ATT, DEF, DISC = NP.ATT, NP.DEF, NP.DISC
CANON = {"PROP": "Prop", "HOOKER": "Hooker", "LOCK": "Second-row", "LOOSE FORWARD": "Back-row",
         "SCRUM-HALF": "Scrum-half", "FLY-HALF": "Fly-half", "CENTRE": "Centre", "BACK THREE": "Back-three"}
POS_CODE = {p: i for i, p in enumerate(sorted(set(CANON.values())))}
LGBM = dict(n_estimators=350, learning_rate=0.04, num_leaves=15, min_child_samples=40,
            reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9, subsample_freq=1,
            random_state=17, n_jobs=1, verbosity=-1, objective="regression_l1")


def _components(g, w):
    wm = (w * g["minutes"]).sum()
    if wm <= 0:
        return None
    r = {e: (w * g[e]).sum() / wm * 80 for e in {**ATT, **DEF, **DISC}}
    return (sum(r[e] * v for e, v in ATT.items()),
            sum(r[e] * v for e, v in DEF.items()),
            sum(r[e] * v for e, v in DISC.items()), wm)


def _feature_row(ph, ch, asof, pos, wr_own, wr_opp):
    """PIT features from a player's prior intl history `ph` and club history `ch`."""
    f = {"pos": POS_CODE.get(pos, -1), "wr_own": wr_own, "wr_opp": wr_opp,
         "wr_gap": wr_own - wr_opp, "n_intl": len(ph), "n_club": len(ch)}
    if len(ph):
        w = 0.5 ** ((asof - ph["date"]).dt.days / HL)
        c = _components(ph, w)
        if c:
            f["intl_att"], f["intl_dfn"], f["intl_dsc"], f["intl_wm"] = c
        f["min_recent"] = (w * ph["minutes"]).sum() / w.sum()
        f["start_rate"] = (w * ph["started"].astype(float)).sum() / w.sum()
        f["days_since"] = (asof - ph["date"].max()).days
    if len(ch):
        w = 0.5 ** ((asof - ch["date"]).dt.days / HL)
        c = _components(ch, w)
        if c:
            f["club_att"] = c[0] * NP.CLUB_ATT_CAL
            f["club_dfn"] = c[1] * NP.CLUB_DEF_CAL
            f["club_dsc"] = c[2]
            f["club_wm"] = c[3]
    return f


FEATURES = ["pos", "wr_own", "wr_opp", "wr_gap", "n_intl", "n_club",
            "intl_att", "intl_dfn", "intl_dsc", "intl_wm", "min_recent",
            "start_rate", "days_since", "club_att", "club_dfn", "club_dsc", "club_wm"]


def build_training(min_date="2022-06-01"):
    """One labelled row per (player, intl match, minutes≥20): PIT features → the
    realised per-80 att/dfn/dsc in that match."""
    intl = pd.read_csv(NP.NCR / "ncr_player_match.csv")
    intl["date"] = pd.to_datetime(intl["date"])
    club = pd.read_csv(NP.NCR / "club_player_match.csv")
    club["date"] = pd.to_datetime(club["date"])
    wr = pd.read_csv(NP.DATA / "wr_rankings.csv")
    wr["snapshot_date"] = pd.to_datetime(wr["snapshot_date"])
    wr_by_date = {d: g.set_index("team")["wr_pts"].to_dict() for d, g in wr.groupby("snapshot_date")}
    wr_dates = np.array(sorted(wr_by_date))

    def wr_at(team, date):
        i = np.searchsorted(wr_dates, date)          # nearest snapshot on/before date
        for j in (i - 1, i, i - 2):
            if 0 <= j < len(wr_dates) and team in wr_by_date[wr_dates[j]]:
                return wr_by_date[wr_dates[j]][team]
        return 80.0

    intl_by = {pid: g.sort_values("date") for pid, g in intl.groupby("player_id")}
    club_by = {pid: g.sort_values("date") for pid, g in club.groupby("player_id")}
    rows, Y = [], []
    tgt = intl[(intl["date"] >= min_date) & (intl["minutes"] >= 20)]
    for r in tgt.itertuples():
        w1 = np.array([1.0])
        realised = _components(pd.DataFrame([{**{e: getattr(r, e) for e in {**ATT, **DEF, **DISC}},
                                              "minutes": r.minutes}]), w1)
        if realised is None:
            continue
        ph = intl_by.get(r.player_id); ph = ph[ph["date"] < r.date] if ph is not None else intl.iloc[:0]
        ch = club_by.get(r.player_id); ch = ch[ch["date"] < r.date] if ch is not None else club.iloc[:0]
        pos = None
        row = _feature_row(ph, ch, r.date, _canon_from_history(ph), wr_at(r.team, r.date), wr_at(r.opponent, r.date))
        rows.append(row)
        Y.append(realised[:3])
    X = pd.DataFrame(rows).reindex(columns=FEATURES)
    return X, np.array(Y)


def _canon_from_history(ph):
    if not len(ph):
        return None
    j = pd.to_numeric(ph["jersey"], errors="coerce").dropna()
    if not len(j):
        return None
    from ingest_6n import JERSEY_GROUP
    grp = {"Prop": "Prop", "Hooker": "Hooker", "Second-row": "Second-row", "Back-row": "Back-row",
           "Scrum-half": "Scrum-half", "Fly-half": "Fly-half", "Centre": "Centre", "Back-three": "Back-three"}
    m = j.map(JERSEY_GROUP).mode()
    return grp.get(m.iloc[0]) if len(m) else None


def train_and_predict():
    X, Y = build_training()
    models = []
    for k in range(3):
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(X, Y[:, k])
        models.append(m)

    # GW1 prediction rows: reuse the same PIT feature build as-of ASOF
    pool = pd.read_csv(NP.NCR / "ncr_players.csv")
    pool = pool[pool.player_status.isin(["P", "B"])]
    intl = pd.read_csv(NP.NCR / "ncr_player_match.csv"); intl["date"] = pd.to_datetime(intl["date"])
    club = pd.read_csv(NP.NCR / "club_player_match.csv"); club["date"] = pd.to_datetime(club["date"])
    intl = intl[intl["date"] < ASOF]; club = club[club["date"] < ASOF]
    pid_map = NP.match_players(pool, intl)
    intl_by = {pid: g.sort_values("date") for pid, g in intl.groupby("player_id")}
    club_by = {pid: g.sort_values("date") for pid, g in club.groupby("player_id")}
    fx = pd.read_csv(NP.NCR / "ncr_fixtures.csv"); gw1 = fx[fx.gameday == 1]
    opp = {}
    for _, r in gw1.iterrows():
        opp[r.home] = r.away; opp[r.away] = r.home
    wr = pd.read_csv(NP.DATA / "wr_rankings.csv"); wr["snapshot_date"] = pd.to_datetime(wr["snapshot_date"])
    wr = wr[wr.snapshot_date == wr.snapshot_date.max()].set_index("team")["wr_pts"].to_dict()

    rows, ids = [], []
    for r in pool.itertuples():
        pid = pid_map.get(r.id)
        ph = intl_by.get(pid, intl.iloc[:0]); ch = club_by.get(pid, club.iloc[:0])
        rows.append(_feature_row(ph, ch, ASOF, CANON[r.skill_desc],
                                 wr.get(r.team_name, 80), wr.get(opp.get(r.team_name), 80)))
        ids.append(r.id)
    Xp = pd.DataFrame(rows).reindex(columns=FEATURES)
    att = np.clip(models[0].predict(Xp), 0, None)
    dfn = np.clip(models[1].predict(Xp), 0, None)
    dsc = models[2].predict(Xp)
    return pd.DataFrame({"fantasy_id": ids, "att": att, "dfn": dfn, "dsc": dsc})


if __name__ == "__main__":
    rates = train_and_predict()
    df = NP.build_projection(override_rates=rates)
    squad, _, _ = NP.optimise(df)
    NP.render(df, squad, "_lgbm", [], title="in-domain LGBM (NCR ideas)")
