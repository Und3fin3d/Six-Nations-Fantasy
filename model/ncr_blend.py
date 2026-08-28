#!/usr/bin/env python3
"""
model/ncr_blend.py — in-domain LGBM+XGBoost rate heads, NCR-native.
===================================================================
Same ideas as the empirical NCR model (ncr_project): recency-weighted per-80 form,
calibrated club-form blend, then the SAME downstream matchup / minutes / NCR scoring /
MILP. The only swap: instead of hand-tuned shrinkage toward a prior, a LightGBM +
XGBoost blend LEARNS the (prior-form → realised-rate) mapping from NCR match history —
trained in-domain (internationals of the 12 nations, 2019-2026), never on 6n.

Each training row is one player-match (minutes>=10). Features are PIT (strictly before
that match): the player's decayed prior per-80 for each scored component, split into
international and (calibrated) club priors, plus position / experience. Targets are that
match's realised per-80 rates. 15 components × {LGBM Poisson, XGB Poisson}, blended 50/50.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

import model.ncr_project as NP
from model.ncr_predict_model import NCR_PTS, ATT_C, DEF_C, POS2CANON, FORWARD

ASOF = NP.ASOF                    # derived from the current gameday
HL = 300.0                       # prior half-life (days)
CLUB_ATT_CAL = 0.84              # club→test attack calibration (as in ncr_project)
# components predicted (superset of SCORED that the NCR rubric needs)
COMPS = ["tries", "try_assists", "conversion_goals", "penalty_goals", "drop_goals_converted",
         "defenders_beaten", "offload", "clean_breaks", "tackles", "missed_tackles",
         "tackle_turnover", "penalties_conceded", "turnovers_conceded", "yellow_cards", "red_cards"]
ATT_EXTRA = {"clean_breaks": 3}   # NCR line-break, not in NCR_PTS
DEF_EXTRA = {"missed_tackles": -1}
DSC_EXTRA = {"turnovers_conceded": -1}
POS_ORD = {"Prop": 0, "Hooker": 1, "Second-row": 2, "Back-row": 3,
           "Scrum-half": 4, "Fly-half": 5, "Centre": 6, "Back-three": 7}
LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40,
             reg_lambda=5.0, subsample=0.9, colsample_bytree=0.8, objective="poisson",
             random_state=0, n_jobs=1, verbosity=-1)
XGB_P = dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,
             reg_lambda=5.0, subsample=0.9, colsample_bytree=0.8, objective="count:poisson",
             random_state=0, n_jobs=1, verbosity=0)


def _canon(jersey):
    j = int(jersey) if pd.notna(jersey) else 0
    return {1: "Prop", 2: "Hooker", 3: "Prop", 4: "Second-row", 5: "Second-row",
            6: "Back-row", 7: "Back-row", 8: "Back-row", 9: "Scrum-half", 10: "Fly-half",
            11: "Back-three", 12: "Centre", 13: "Centre", 14: "Back-three", 15: "Back-three"}.get(j)


def _priors(prior_rows, asof, calibrate):
    """decayed per-80 rate per component over prior_rows (a player's earlier matches)."""
    if prior_rows is None or not len(prior_rows):
        return {c: np.nan for c in COMPS}, 0, np.nan
    w = 0.5 ** ((asof - prior_rows["date"]).dt.days.to_numpy() / HL)
    mins = prior_rows["minutes"].to_numpy(float)
    wm = (w * mins).sum()
    out = {}
    for c in COMPS:
        if wm <= 0:
            out[c] = np.nan
        else:
            r = (w * prior_rows[c].to_numpy(float)).sum() / wm * 80
            out[c] = r * (CLUB_ATT_CAL if (calibrate and c in ATT_C | {"clean_breaks"}) else 1.0)
    return out, len(prior_rows), float((w * mins).sum() / w.sum()) if w.sum() > 0 else np.nan


def _feature_row(intl_prior, club_prior, n_intl, n_club, avg_min, canon):
    row = {}
    for c in COMPS:
        row[f"i_{c}"] = intl_prior[c]
        row[f"c_{c}"] = club_prior[c]
    row["n_intl"] = n_intl
    row["n_club"] = n_club
    row["avg_min"] = avg_min
    row["pos"] = POS_ORD.get(canon, -1)
    row["is_fwd"] = float(canon in FORWARD)
    return row


def _load_history(leak_cutoff=None):
    intl = pd.read_csv(NP.NCR / "ncr_player_match.csv")
    club = pd.read_csv(NP.NCR / "club_player_match.csv")
    for df in (intl, club):
        df["date"] = pd.to_datetime(df["date"])
        for c in COMPS:
            if c not in df.columns:
                df[c] = 0
    if leak_cutoff is not None:
        intl = intl[intl["date"] < leak_cutoff]
        club = club[club["date"] < leak_cutoff]
    return intl, club


def build_training(leak_cutoff=None):
    intl, club = _load_history(leak_cutoff=leak_cutoff)
    intl = intl.sort_values("date")
    club_by_pid = {pid: g.sort_values("date") for pid, g in club.groupby("player_id")}
    canon_by_pid = (intl[intl["jersey"].between(1, 15)].assign(cp=lambda d: d["jersey"].map(_canon))
                    .dropna(subset=["cp"]).groupby("player_id")["cp"]
                    .agg(lambda s: s.mode().iloc[0]).to_dict())
    rows, targets = [], {c: [] for c in COMPS}
    for pid, g in intl.groupby("player_id"):
        g = g.sort_values("date")
        cg = club_by_pid.get(pid)
        canon = canon_by_pid.get(pid) or _canon(g["jersey"].iloc[0])
        for i, (_, m) in enumerate(g.iterrows()):
            if m["minutes"] < 10:
                continue
            d = m["date"]
            ip, ni, am = _priors(g.iloc[:i], d, calibrate=False)
            cp, nc, _ = _priors(cg[cg["date"] < d] if cg is not None else None, d, calibrate=True)
            if ni == 0 and nc == 0:
                continue                                   # no PIT signal at all → skip
            rows.append(_feature_row(ip, cp, ni, nc, am, canon))
            f = 80.0 / m["minutes"]
            for c in COMPS:
                targets[c].append(m[c] * f)
    X = pd.DataFrame(rows)
    Y = {c: np.clip(np.array(v, float), 0, None) for c, v in targets.items()}
    return X, Y


def train_heads(X, Y):
    heads = {}
    for c in COMPS:
        y = Y[c]
        if y.sum() == 0:
            heads[c] = ("zero", None, None)
            continue
        lm = lgb.LGBMRegressor(**LGB_P).fit(X, y)
        xm = xgb.XGBRegressor(**XGB_P).fit(X, y)
        heads[c] = ("blend", lm, xm)
    return heads


def predict(heads, Xte):
    out = {}
    for c in COMPS:
        kind, lm, xm = heads[c]
        if kind == "zero":
            out[c] = np.zeros(len(Xte))
        else:
            out[c] = np.clip(0.5 * lm.predict(Xte) + 0.5 * xm.predict(Xte), 0, None)
    return pd.DataFrame(out)


def gw1_features(leak_cutoff=ASOF):
    """PIT feature rows for the GW1 pool (priors strictly before leak_cutoff)."""
    pool = pd.read_csv(NP.NCR / "ncr_players.csv")
    pool = pool[pool.player_status.isin(["P", "B"])]
    intl, club = _load_history(leak_cutoff=leak_cutoff)
    pid_map = NP.match_players(pool, intl)
    intl_by_pid = {pid: g.sort_values("date") for pid, g in intl.groupby("player_id")}
    club_by_pid = {pid: g.sort_values("date") for pid, g in club.groupby("player_id")}
    rows, fids = [], []
    for r in pool.itertuples():
        pid = pid_map.get(r.id)
        canon = POS2CANON[NP.SKILL2POS[r.skill_desc]]
        ip, ni, am = _priors(intl_by_pid.get(pid), leak_cutoff, calibrate=False)
        cp, nc, _ = _priors(club_by_pid.get(pid), leak_cutoff, calibrate=True)
        rows.append(_feature_row(ip, cp, ni, nc, am, canon))
        fids.append(r.id)
    return pd.DataFrame(rows), fids


def rates_to_override(rates: pd.DataFrame, fids) -> pd.DataFrame:
    r = rates.fillna(0.0)
    att = sum(NCR_PTS[c] * r[c] for c in ATT_C if c in r) + sum(v * r[c] for c, v in ATT_EXTRA.items())
    dfn = sum(NCR_PTS[c] * r[c] for c in DEF_C if c in r) + sum(v * r[c] for c, v in DEF_EXTRA.items())
    dsc = (NCR_PTS["penalties_conceded"] * r["penalties_conceded"]
           + NCR_PTS["yellow_cards"] * r["yellow_cards"] + NCR_PTS["red_cards"] * r["red_cards"]
           + sum(v * r[c] for c, v in DSC_EXTRA.items()))
    return pd.DataFrame({"fantasy_id": fids, "att": att, "dfn": dfn, "dsc": dsc})


def build_override(leak_cutoff=ASOF, train_cutoff=None):
    X, Y = build_training(leak_cutoff=train_cutoff)
    heads = train_heads(X, Y)
    Xte, fids = gw1_features(leak_cutoff)
    Xte = Xte.reindex(columns=X.columns)
    rates = predict(heads, Xte)
    return rates_to_override(rates, fids), (X, heads)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    ov, _ = build_override()
    df = NP.build_projection(override_rates=ov)
    df.to_csv(NP.NCR / f"ncr_gw{NP.CUR_GD}_projections_blend.csv", index=False)
    squad, _, _ = NP.optimise(df)
    NP.render(df, squad, "_blend", (), title="in-domain LGBM+XGB blend")
