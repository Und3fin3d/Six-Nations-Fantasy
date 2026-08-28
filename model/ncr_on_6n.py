"""Replay the NCR empirical projection machinery on the Six Nations rounds.

Answers "how would the NCR incumbent do on 6N?" by porting `ncr_project.py`'s
exact mechanisms — 420-day recency-weighted per-80 rates, calibrated club blend,
RugbyPass player-specific priors, K=220 empirical-Bayes shrinkage toward
position baselines, WR-margin matchup multipliers, status->minutes maps, and
the POTM heuristic — onto 6N fixtures under the 6N scoring rubric (tries 15/10
by role, metres/10, tackle-turnover 5, lineout steal 7, fifty-22 7, POTM 15).

Deviations from the NCR deployment, all noted in the report:
- Point-in-time history per round (STRICTER than its own NCR GW1-2 numbers,
  which were produced in-window).
- RugbyPass aggregates are current-day snapshots (hindsight in its favour,
  exactly as in its NCR evaluation).
- No weather feed for 6N (factor 1.0, the model's own missing-data default).

Evaluated on the same official-points parity cohort as every other model in
data/unified/v3/benchmark (position-median fill for crosswalk-unmatched rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.rp_rates import _norm, _season_end_year
from model.unified.benchmark_v2 import _group_metrics
from model.unified.data import ROOT
from model.unified.labels import build_fantasy_labels
from model.unified.v3.cohorts import match_labels_to_store

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"

HALFLIFE_DAYS = 420.0
K = 220.0
CLUB_ATT_CAL, CLUB_DEF_CAL, CLUB_DSC_CAL = 0.84, 0.97, 1.0
CLUB_CONF = 0.55
RP_HALFLIFE_YEARS = 1.6

# 6N rubric split into the same att/def/disc groups the NCR model uses for its
# matchup multipliers. Tries are weighted per player (15 forward / 10 back).
ATT_EVENTS = {"try_assists": 4, "conversion_goals": 2, "penalty_goals": 3,
              "drop_goals_converted": 4, "defenders_beaten": 2, "offload": 2,
              "metres": 0.1, "fifty_22": 7, "kicks_retained": 2}
DEF_EVENTS = {"tackles": 1, "tackle_turnover": 5, "lineout_steals": 7}
DSC_EVENTS = {"penalties_conceded": -1, "yellow_cards": -5, "red_cards": -8}
ALL_EVENTS = ["tries", *ATT_EVENTS, *DEF_EVENTS, *DSC_EVENTS]
FRONT_ROW = ("prop", "hooker")


def try_weight(is_forward) -> float:
    return 15.0 if bool(is_forward) else 10.0


def event_rates(g: pd.DataFrame):
    wm = float((g.w * g.minutes).sum())
    if wm <= 0:
        return None, 0.0
    rates = {e: float((g.w * pd.to_numeric(g.get(e), errors="coerce").fillna(0)).sum())
             / wm * 80 for e in ALL_EVENTS}
    return rates, wm


def to_points(rates: dict, tryw: float) -> tuple[float, float, float]:
    att = rates["tries"] * tryw + sum(rates[e] * v for e, v in ATT_EVENTS.items())
    dfn = sum(rates[e] * v for e, v in DEF_EVENTS.items())
    dsc = sum(rates[e] * v for e, v in DSC_EVENTS.items())
    return att, dfn, dsc


def profiles(hist: pd.DataFrame, club: pd.DataFrame, tryw_by_pid: dict) -> dict:
    club_by_pid = {pid: g for pid, g in club.groupby("player_id")} if len(club) else {}
    out = {}
    for pid, g in hist.groupby("player_id"):
        rates, wm = event_rates(g)
        if rates is None:
            continue
        tryw = tryw_by_pid.get(pid, 12.5)
        att, dfn, dsc = to_points(rates, tryw)
        cg = club_by_pid.get(pid)
        if cg is not None:
            crates, cwm = event_rates(cg)
            if crates is not None:
                catt, cdfn, cdsc = to_points(crates, tryw)
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


def rp_prior_6n(asof_year: int = 2026) -> dict:
    """RugbyPass per-80 6N-point components, keyed by normalised name.

    Try points are kept as a separate rate so each lookup can apply the
    player's own forward/back try weight. Strict point-in-time: only seasons
    that CONCLUDED before the 6N window are used (northern club seasons end in
    June, so a Feb-Mar round of year Y admits seasons ending <= Y-1). This is
    conservative — a live scrape would also have the in-progress season's
    partial stats — so it brackets the honest number from below.
    """
    cs = pd.read_csv(DATA / "rp_compstats.csv")
    cs = cs[cs["season"].map(_season_end_year) < asof_year]
    cs["w"] = 0.5 ** ((asof_year - cs["season"].map(_season_end_year)).clip(lower=0)
                      / RP_HALFLIFE_YEARS)
    cs = cs[cs["w"] > 0.15]
    table = {}
    for slug, g in cs.groupby("slug"):
        wm = float((g.w * g.minutes).sum())
        if wm < 120:
            continue
        r = {e: float((g.w * pd.to_numeric(g.get(e), errors="coerce").fillna(0)).sum())
             / wm * 80
             for e in ("tries", "try_assists", "defenders_beaten", "offloads",
                       "metres", "tackles", "turnovers_won", "penalties_conceded",
                       "yellow_cards", "red_cards")}
        table[_norm(str(slug).replace("-", " "))] = dict(
            try_rate=r["tries"],
            att_rest=r["try_assists"] * 4 + r["defenders_beaten"] * 2
                     + r["offloads"] * 2 + r["metres"] * 0.1,
            dfn=r["tackles"] * 1 + r["turnovers_won"] * 5,
            dsc=r["penalties_conceded"] * -1 + r["yellow_cards"] * -5
                + r["red_cards"] * -8,
            rp_min=wm,
        )
    return table


def calibrate_rp(table: dict, prof: dict, names: dict, tryw_by_pid: dict) -> float:
    """Club->test attack calibration on players present in both sources."""
    ratios = []
    for pid, p in prof.items():
        if p["wmin"] < 300:
            continue
        rp = table.get(_norm(names.get(pid, "")))
        if not rp:
            continue
        rp_att = rp["try_rate"] * tryw_by_pid.get(pid, 12.5) + rp["att_rest"]
        if rp_att > 1:
            ratios.append(p["att"] / rp_att)
    return float(np.clip(np.median(ratios), 0.5, 1.2)) if len(ratios) >= 20 else 0.84


def main() -> None:
    store = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                        parse_dates=["date"])
    labels = build_fantasy_labels()
    labels = labels[labels["competition"].eq("six_nations")].copy()
    wr = pd.read_csv(DATA / "wr_rankings.csv", parse_dates=["snapshot_date"])
    rp_cache = {}

    rows, fold_rows = [], []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        matched = match_labels_to_store(block, store)
        matched = matched.rename(columns={"_label_row_id": "label_row_id"})
        played = matched[matched["store_matched"]].copy()
        asof = pd.to_datetime(played["date"]).min()

        hist = store[(store.competition_level.eq("international"))
                     & (store.date < asof)].copy()
        hist["w"] = 0.5 ** ((asof - hist.date).dt.days / HALFLIFE_DAYS)
        club = store[(store.competition_level.eq("club")) & (store.date < asof)].copy()
        club["w"] = 0.5 ** ((asof - club.date).dt.days / HALFLIFE_DAYS)
        club = club[club.w > 0.05]
        year = int(asof.year)
        if year not in rp_cache:
            rp_cache[year] = rp_prior_6n(year)
        rp_table = rp_cache[year]

        tryw_by_pid = {str(pid): try_weight(fwd) for pid, fwd in
                       hist.groupby("player_id")["is_forward"].last().items()}
        for r in played.itertuples():
            tryw_by_pid[str(r.player_id)] = try_weight(r.is_forward)
        hist["player_id"] = hist["player_id"].astype(str)
        club["player_id"] = club["player_id"].astype(str)
        prof = profiles(hist, club, tryw_by_pid)
        names = hist.groupby("player_id")["player_name"].first().to_dict()
        fac = calibrate_rp(rp_table, prof, names, tryw_by_pid)

        snap = wr[wr.snapshot_date.le(asof)]
        snap = snap[snap.snapshot_date.eq(snap.snapshot_date.max())]
        wr_pts = snap.set_index("team")["wr_pts"].to_dict()

        bench_hist = hist[~hist.started.astype(bool) & hist.minutes.gt(0)]
        bench_pos_prior = bench_hist.groupby("position")["minutes"].mean().to_dict()

        cohort = [p for p in (prof.get(str(r.player_id)) for r in played.itertuples()) if p]
        base = {}
        for pos, sub in played.groupby("position"):
            ps = [prof[str(r.player_id)] for r in sub.itertuples()
                  if str(r.player_id) in prof]
            if ps:
                w = np.array([p["wmin"] for p in ps])
                base[pos] = {k: float(np.average([p[k] for p in ps], weights=w))
                             for k in ("att", "dfn", "dsc")}
            else:
                base[pos] = {"att": 8, "dfn": 10, "dsc": -1}

        predictions = []
        for r in played.itertuples():
            pid = str(r.player_id)
            pos, tryw = r.position, try_weight(r.is_forward)
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
            if str(pos).lower() in FRONT_ROW:
                dfn += 1.0  # ~1 scrum/80 at the 6N scrum weight

            margin = 2.0 * (wr_pts.get(r.team, 80) - wr_pts.get(r.opponent, 80)
                            + (3 if str(r.home_away) == "home" else -3))
            att_mult = float(np.clip(1 + margin / 60.0, 0.72, 1.35))
            def_mult = float(np.clip(1 - margin / 90.0, 0.80, 1.28))
            per80 = att * att_mult + dfn * def_mult + dsc

            started = bool(r.started)
            exp_min = start_min if started else bench_min
            base_pts = per80 * (exp_min / 80.0)
            potm = float(np.clip(base_pts / 145.0, 0, 0.13 if started else 0.05) * 15)
            predictions.append({"label_row_id": r.label_row_id,
                                "position": pos,
                                "predicted_points": base_pts + potm,
                                "official_pts": r.official_pts})

        pred = pd.DataFrame(predictions)
        missing = matched[~matched["store_matched"]]
        if len(missing):
            med = pred.groupby("position")["predicted_points"].median()
            fills = [{"label_row_id": m.label_row_id, "position": m.position_label
                      if hasattr(m, "position_label") else m.position,
                      "predicted_points": float(med.get(getattr(m, "position_label",
                                                                getattr(m, "position", None)),
                                                        med.median())),
                      "official_pts": m.official_pts} for m in missing.itertuples()]
            pred = pd.concat([pred, pd.DataFrame(fills)], ignore_index=True)

        metrics = _group_metrics(pred.rename(columns={"official_pts": "actual"}),
                                 "predicted_points", actual_col="actual")
        season = int(block["season"].iloc[0])
        fold_rows.append({"group_id": group_id, "season": season, **metrics})
        rows.append(pred.assign(group_id=group_id))
        print(f"[{group_id}] n={metrics['n']} mae={metrics['mae']:.2f} "
              f"spearman={metrics['spearman']:.3f} "
              f"top10={metrics['top_10_capture']:.1%} (rp_fac={fac:.2f})", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "ncr_on_6n_folds.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "ncr_on_6n_predictions.csv", index=False)
    cols = ["mae", "spearman", "top_10_capture", "top_25_capture",
            "top_50_capture", "top_100_capture"]
    print("\n== NCR empirical machinery replayed on 6N (per season) ==")
    print(folds.groupby("season")[cols].mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
