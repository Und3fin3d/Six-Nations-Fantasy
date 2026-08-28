"""The empirical machinery as a competition-independent model.

Generalizes model/ncr_on_6n.py: one shared engine — 420-day recency-weighted
per-80 rates, calibrated club blend, season-dated RugbyPass player priors,
K=220 empirical-Bayes shrinkage toward position baselines, WR-margin matchup
multipliers, status->minutes maps, POTM heuristic — driven by a per-competition
rubric config (the deterministic-scorer seam, only expressed as weight dicts).
No learned or hand-set parameter is competition-specific; only the scoring
weights differ, exactly as the unified mandate requires.

Everything is point-in-time per round: history/club/WR strictly before the
round's first match; RugbyPass seasons only if concluded (June 30 of the
season's end year) before the round. NCR GW1-2 numbers here are therefore the
HONEST version of the deployed incumbent's slightly in-window 8.90.

Outputs data/unified/v4/empirical_unified_{folds,predictions}.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.ncr_on_6n import (CLUB_ATT_CAL, CLUB_CONF, CLUB_DEF_CAL,
                             CLUB_DSC_CAL, HALFLIFE_DAYS, K,
                             RP_HALFLIFE_YEARS, _norm, _season_end_year)
from model.unified.benchmark_v2 import _group_metrics
from model.unified.data import ROOT
from model.unified.labels import build_fantasy_labels
from model.unified.v3.cohorts import match_labels_to_store
from model.unified.v3.harness import attach_match_timestamps

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"

CONFIGS = {
    "six_nations": dict(
        try_weight=lambda fwd: 15.0 if bool(fwd) else 10.0,
        att={"try_assists": 4, "conversion_goals": 2, "penalty_goals": 3,
             "drop_goals_converted": 4, "defenders_beaten": 2, "offload": 2,
             "metres": 0.1, "fifty_22": 7, "kicks_retained": 2},
        dfn={"tackles": 1, "tackle_turnover": 5, "lineout_steals": 7},
        dsc={"penalties_conceded": -1, "yellow_cards": -5, "red_cards": -8},
        rp_att={"try_assists": 4, "defenders_beaten": 2, "offloads": 2, "metres": 0.1},
        rp_dfn={"tackles": 1, "turnovers_won": 5},
        front_row_bonus=1.0,
    ),
    "ncr": dict(
        # ncr_project.py's exact groups (lineouts_won deliberately dropped there
        # as a hooker-inflation artifact; kept dropped for fidelity).
        try_weight=lambda fwd: 12.0,
        att={"try_assists": 5, "conversion_goals": 2, "missed_conversion_goals": -1,
             "penalty_goals": 3, "missed_penalty_goals": -1, "drop_goals_converted": 5,
             "defenders_beaten": 2, "offload": 2, "clean_breaks": 3},
        dfn={"tackles": 1, "missed_tackles": -1, "tackle_turnover": 4},
        dsc={"turnovers_conceded": -1, "penalties_conceded": -1,
             "yellow_cards": -5, "red_cards": -10},
        rp_att={"try_assists": 5, "defenders_beaten": 2, "offloads": 2, "clean_breaks": 3},
        rp_dfn={"tackles": 1, "missed_tackles": -1, "turnovers_won": 4},
        front_row_bonus=2.0,
    ),
}
RP_DSC = {"penalties_conceded": -1, "yellow_cards": -5, "red_cards": -10}


def event_rates(g: pd.DataFrame, events: list[str]):
    wm = float((g.w * g.minutes).sum())
    if wm <= 0:
        return None, 0.0
    rates = {e: float((g.w * pd.to_numeric(g.get(e), errors="coerce").fillna(0)).sum())
             / wm * 80 for e in events}
    return rates, wm


def to_points(rates: dict, tryw: float, cfg: dict) -> tuple[float, float, float]:
    att = rates["tries"] * tryw + sum(rates[e] * v for e, v in cfg["att"].items())
    dfn = sum(rates[e] * v for e, v in cfg["dfn"].items())
    dsc = sum(rates[e] * v for e, v in cfg["dsc"].items())
    return att, dfn, dsc


def profiles(hist, club, tryw_by_pid, cfg) -> dict:
    events = ["tries", *cfg["att"], *cfg["dfn"], *cfg["dsc"]]
    club_by_pid = {pid: g for pid, g in club.groupby("player_id")} if len(club) else {}
    out = {}
    for pid, g in hist.groupby("player_id"):
        rates, wm = event_rates(g, events)
        if rates is None:
            continue
        tryw = tryw_by_pid.get(pid, 12.0)
        att, dfn, dsc = to_points(rates, tryw, cfg)
        cg = club_by_pid.get(pid)
        if cg is not None:
            crates, cwm = event_rates(cg, events)
            if crates is not None:
                catt, cdfn, cdsc = to_points(crates, tryw, cfg)
                cw = CLUB_CONF * cwm
                att = (wm * att + cw * catt * CLUB_ATT_CAL) / (wm + cw)
                dfn = (wm * dfn + cw * cdfn * CLUB_DEF_CAL) / (wm + cw)
                dsc = (wm * dsc + cw * cdsc * CLUB_DSC_CAL) / (wm + cw)
                wm = wm + cw
        starts, bench = g[g.started.astype(bool)], g[~g.started.astype(bool)]
        start_min = (float((starts.w * starts.minutes).sum() / starts.w.sum())
                     if len(starts) and starts.w.sum() > 0 else 72.0)
        bench_min = (float((bench.w * bench.minutes).sum() / bench.w.sum())
                     if len(bench) and bench.w.sum() > 0 else np.nan)
        out[pid] = dict(att=att, dfn=dfn, dsc=dsc, wmin=wm,
                        start_min=min(80.0, start_min),
                        bench_min=min(35.0, max(8.0, bench_min)) if bench_min == bench_min else np.nan)
    return out


def rp_prior(asof: pd.Timestamp, cfg: dict) -> dict:
    cs = pd.read_csv(DATA / "rp_compstats.csv")
    end_year = cs["season"].map(_season_end_year)
    concluded = pd.to_datetime(end_year.astype(str) + "-06-30")
    cs = cs[concluded < asof.tz_localize(None) if asof.tzinfo else concluded < asof]
    cs["w"] = 0.5 ** ((asof.year - cs["season"].map(_season_end_year)).clip(lower=0)
                      / RP_HALFLIFE_YEARS)
    cs = cs[cs["w"] > 0.15]
    events = ["tries", *cfg["rp_att"], *cfg["rp_dfn"], *RP_DSC]
    table = {}
    for slug, g in cs.groupby("slug"):
        wm = float((g.w * g.minutes).sum())
        if wm < 120:
            continue
        r = {e: float((g.w * pd.to_numeric(g.get(e), errors="coerce").fillna(0)).sum())
             / wm * 80 for e in events}
        table[_norm(str(slug).replace("-", " "))] = dict(
            try_rate=r["tries"],
            att_rest=sum(r[e] * v for e, v in cfg["rp_att"].items()),
            dfn=sum(r[e] * v for e, v in cfg["rp_dfn"].items()),
            dsc=sum(r[e] * v for e, v in RP_DSC.items()),
            rp_min=wm,
        )
    return table


def calibrate_rp(table, prof, names, tryw_by_pid) -> float:
    ratios = []
    for pid, p in prof.items():
        if p["wmin"] < 300:
            continue
        rp = table.get(_norm(names.get(pid, "")))
        if not rp:
            continue
        rp_att = rp["try_rate"] * tryw_by_pid.get(pid, 12.0) + rp["att_rest"]
        if rp_att > 1:
            ratios.append(p["att"] / rp_att)
    return float(np.clip(np.median(ratios), 0.5, 1.2)) if len(ratios) >= 20 else 0.84


def project_round(block: pd.DataFrame, store: pd.DataFrame, competition: str,
                  wr: pd.DataFrame) -> pd.DataFrame:
    cfg = CONFIGS[competition]
    matched = match_labels_to_store(block, store)
    matched = matched.rename(columns={"_label_row_id": "label_row_id"})
    played = matched[matched["store_matched"]].copy()
    asof = pd.to_datetime(played["date"]).min()

    hist = store[store.competition_level.eq("international") & (store.date < asof)].copy()
    hist["w"] = 0.5 ** ((asof - hist.date).dt.days / HALFLIFE_DAYS)
    club = store[store.competition_level.eq("club") & (store.date < asof)].copy()
    club["w"] = 0.5 ** ((asof - club.date).dt.days / HALFLIFE_DAYS)
    club = club[club.w > 0.05]

    tryw_by_pid = {str(pid): cfg["try_weight"](fwd) for pid, fwd in
                   hist.groupby("player_id")["is_forward"].last().items()}
    for r in played.itertuples():
        tryw_by_pid[str(r.player_id)] = cfg["try_weight"](r.is_forward)
    hist["player_id"] = hist["player_id"].astype(str)
    club["player_id"] = club["player_id"].astype(str)
    prof = profiles(hist, club, tryw_by_pid, cfg)
    names = hist.groupby("player_id")["player_name"].first().to_dict()
    rp_table = rp_prior(asof, cfg)
    fac = calibrate_rp(rp_table, prof, names, tryw_by_pid)

    snap = wr[wr.snapshot_date.le(asof)]
    snap = snap[snap.snapshot_date.eq(snap.snapshot_date.max())]
    wr_pts = snap.set_index("team")["wr_pts"].to_dict()
    bench_hist = hist[~hist.started.astype(bool) & hist.minutes.gt(0)]
    bench_pos_prior = bench_hist.groupby("position")["minutes"].mean().to_dict()

    base = {}
    for pos, sub in played.groupby("position"):
        ps = [prof[str(r.player_id)] for r in sub.itertuples() if str(r.player_id) in prof]
        if ps:
            w = np.array([p["wmin"] for p in ps])
            base[pos] = {k: float(np.average([p[k] for p in ps], weights=w))
                         for k in ("att", "dfn", "dsc")}
        else:
            base[pos] = {"att": 8, "dfn": 10, "dsc": -1}

    predictions = []
    for r in played.itertuples():
        pid, pos = str(r.player_id), r.position
        tryw = cfg["try_weight"](r.is_forward)
        b = base.get(pos, {"att": 8, "dfn": 10, "dsc": -1})
        rp = rp_table.get(_norm(str(r.player_name)))
        if rp:
            rc = min(rp["rp_min"], 900) / 900 * 200
            rp_att = (rp["try_rate"] * tryw + rp["att_rest"]) * fac
            prior = {"att": (rc * rp_att + 80 * b["att"]) / (rc + 80),
                     "dfn": (rc * rp["dfn"] * CLUB_DEF_CAL + 80 * b["dfn"]) / (rc + 80),
                     "dsc": (rc * rp["dsc"] + 80 * b["dsc"]) / (rc + 80)}
        else:
            prior = b
        p = prof.get(pid)
        if p:
            W = p["wmin"]
            att = (W * p["att"] + K * prior["att"]) / (W + K)
            dfn = (W * p["dfn"] + K * prior["dfn"]) / (W + K)
            dsc = (W * p["dsc"] + K * prior["dsc"]) / (W + K)
            start_min = p["start_min"] if p["start_min"] > 20 else 72.0
            bench_min = (p["bench_min"] if p["bench_min"] == p["bench_min"]
                         else bench_pos_prior.get(pos, 24.0))
        else:
            att, dfn, dsc = prior["att"], prior["dfn"], prior["dsc"]
            start_min, bench_min = 70.0, bench_pos_prior.get(pos, 24.0)
        if str(pos).lower() in ("prop", "hooker"):
            dfn += cfg["front_row_bonus"]

        margin = 2.0 * (wr_pts.get(r.team, 80) - wr_pts.get(r.opponent, 80)
                        + (3 if str(r.home_away) == "home" else -3))
        att_mult = float(np.clip(1 + margin / 60.0, 0.72, 1.35))
        def_mult = float(np.clip(1 - margin / 90.0, 0.80, 1.28))
        per80 = att * att_mult + dfn * def_mult + dsc

        started = bool(r.started)
        exp_min = start_min if started else bench_min
        base_pts = per80 * (exp_min / 80.0)
        potm = float(np.clip(base_pts / 145.0, 0, 0.13 if started else 0.05) * 15)
        predictions.append({"label_row_id": r.label_row_id, "position": pos,
                            "predicted_points": base_pts + potm,
                            "official_pts": r.official_pts})

    pred = pd.DataFrame(predictions)
    missing = matched[~matched["store_matched"]]
    if len(missing):
        med = pred.groupby("position")["predicted_points"].median()
        pos_col = "position_label" if "position_label" in missing.columns else "position"
        fills = [{"label_row_id": m.label_row_id, "position": getattr(m, pos_col),
                  "predicted_points": float(med.get(getattr(m, pos_col), med.median())),
                  "official_pts": m.official_pts} for m in missing.itertuples()]
        pred = pd.concat([pred, pd.DataFrame(fills)], ignore_index=True)
    return pred


def main() -> None:
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                      parse_dates=["date"])
    store = attach_match_timestamps(raw)
    labels = build_fantasy_labels()
    wr = pd.read_csv(DATA / "wr_rankings.csv", parse_dates=["snapshot_date"])

    rows, fold_rows = [], []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        competition = str(block["competition"].iloc[0])
        pred = project_round(block, store, competition, wr)
        metrics = _group_metrics(pred.rename(columns={"official_pts": "actual"}),
                                 "predicted_points", actual_col="actual")
        fold_rows.append({"group_id": group_id, "competition": competition, **metrics})
        rows.append(pred.assign(group_id=group_id, competition=competition))
        print(f"[{group_id}] n={metrics['n']} mae={metrics['mae']:.2f} "
              f"spearman={metrics['spearman']:.3f} "
              f"top10={metrics['top_10_capture']:.1%}", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "empirical_unified_folds.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(
        OUT / "empirical_unified_predictions.csv", index=False)
    cols = ["mae", "spearman", "top_10_capture", "top_25_capture",
            "top_50_capture", "top_100_capture"]
    summary = folds.assign(season=folds.group_id.str[:7]).groupby(
        ["competition", "season"])[cols].mean(numeric_only=True)
    print("\n== Empirical unified (strict PIT, per evaluation) ==")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
