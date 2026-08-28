#!/usr/bin/env python3
"""
model/ncr_champion.py — the 6n champion, transplanted onto the current NCR gameweek.
===================================================================================
Runs the promoted research config (`orchestrator_resid030_delta_overlay_010`) end to
end — LGBM rate heads, ridge minutes, latent set-piece/POTM block, XGB point specialist,
residual + selector overlays — on the current gameweek's NCR pool, then feeds its points
into the same MILP squad optimiser the empirical model uses.

The champion is FROZEN 6n: it never retrains on NCR labels. It is a static, high-quality
benchmark, not a model that improves as the tournament goes on.

Staging gotchas this encodes (each one cost an evening):
  * NCR rows enter the 2,760-row 6n store as a synthetic season 2027, round 1.
  * ONE fixture-id space per match. Mixed API/fantasy ids fragment the POTM softmax into
    singleton groups → pp.std()=NaN → max(NaN, floor)=NaN → latent → target all NaN.
  * `is_jumper` is a TARGET-side column: assemble._expected_drives reads it, not
    `role_is_jumper`. If it's missing the latent block silently poisons everything.
  * `date` must be datetime64[ns] on both frames or pd.concat raises on the datetime unit.

Its `target_pts_hat` is on the 6n points scale. Squad choice is invariant to a global
scale factor, so the XV is unaffected; for comparison against the empirical model we
rescale by the ratio of prediction means (PIT — never uses actuals).

Usage:
    /tmp/6n-model-pinned/bin/python -m model.ncr_champion
    /tmp/6n-model-pinned/bin/python -m model.ncr_champion --exclude "New Zealand" "Italy"
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

import model.ncr_project as NP
import model.ncr_predict_model as NPM
from model.data import load
from model.research import PROMOTED_CONFIG, _predict_config, config_from_dict

TYPICAL_J = {"Prop": 1, "Hooker": 2, "Second-row": 4, "Back-row": 6,
             "Scrum-half": 9, "Fly-half": 10, "Centre": 12, "Back-three": 14}
BENCH_J = {"Prop": 17, "Hooker": 16, "Second-row": 19, "Back-row": 20,
           "Scrum-half": 21, "Fly-half": 22, "Centre": 23, "Back-three": 23}


def stage_rows() -> pd.DataFrame:
    """NCR pool as season-2027 rows in the 6n feature-store schema."""
    ncr = NPM.build_ncr_feature_rows()                     # already ASOF/gameday aware
    pool = pd.read_csv(NP.NCR / "ncr_players.csv")
    pool = pool[pool.player_status.isin(["P", "B"])]
    hist = pd.read_csv(NP.NCR / "ncr_player_match.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    pid_map = NP.match_players(pool, hist[hist["date"] < NP.ASOF])
    ncr["api_pid"] = [pid_map.get(i) for i in ncr.fantasy_id]

    # jerseys: last-known real jersey, else a position-typical one
    prior = hist[hist["date"] < NP.ASOF].sort_values("date")
    last_j = prior.drop_duplicates("player_id", keep="last").set_index("player_id")["jersey"]
    status = pool.set_index("id").player_status
    ncr["started"] = ncr.fantasy_id.map(status).eq("P")
    ncr["jersey"] = ncr.api_pid.map(last_j)
    fallback = ncr.apply(lambda r: TYPICAL_J[r.canonical_pos] if r.started
                         else BENCH_J[r.canonical_pos], axis=1)
    ncr["jersey"] = ncr["jersey"].fillna(fallback)

    # ONE fixture-id space: the current gameday's fantasy match_id, per team
    fx = pd.read_csv(NP.NCR / "ncr_fixtures.csv")
    gw = fx[fx.gameday == NP.CUR_GD]
    tf = {}
    for _, r in gw.iterrows():
        tf[r.home] = r.match_id; tf[r.away] = r.match_id
    ncr["fixture_id"] = ncr.team.map(tf)
    if ncr["fixture_id"].isna().any():
        raise AssertionError("some players have no fixture this gameday")
    return ncr


def predict() -> pd.DataFrame:
    """Champion prediction frame for the current gameweek, keyed by fantasy_id."""
    train = load()
    ncr = stage_rows()

    staged = ncr.reindex(columns=list(train.columns) + ["fantasy_id", "api_pid"])
    for c in ncr.columns:
        if c in train.columns:
            staged[c] = ncr[c].values
    staged["season"] = 2027
    staged["round"] = 1
    staged["player_id"] = np.where(ncr.api_pid.notna(), ncr.api_pid, 10_000_000 + ncr.fantasy_id)
    staged["has_label"] = False
    staged["is_modern"] = True
    staged["official_pts"] = np.nan
    staged["minutes"] = np.nan
    staged["started"] = staged["started"].astype(bool)
    staged["is_jumper"] = staged["canonical_pos"].isin(["Second-row", "Back-row"])
    staged["date"] = pd.Series([NP.ASOF] * len(staged)).astype("datetime64[ns]")
    train["date"] = train["date"].astype("datetime64[ns]")

    comb = pd.concat([train, staged], ignore_index=True)
    comb["bio_position"] = comb["bio_position"].astype("category")

    cfg = config_from_dict(json.loads(PROMOTED_CONFIG.read_text()))
    print(f"champion config: {cfg.name}")
    pred, _, _ = _predict_config(comb, cfg, 2027)
    pred["fantasy_id"] = pred["player_id"].map(dict(zip(staged.player_id, staged.fantasy_id)))
    pred = pred.dropna(subset=["fantasy_id"])
    pred["fantasy_id"] = pred["fantasy_id"].astype(int)
    if pred["target_pts_hat"].isna().all():
        raise AssertionError("target_pts_hat all NaN — check fixture_id grouping / is_jumper")
    return pred


def build_projection(exclude_teams=()) -> pd.DataFrame:
    """Empirical frame for pool metadata, with starter_exp replaced by the champion's
    RAW predicted points. No rescaling: these are the model's own numbers.

    They sit on the 6n points scale (its training target), which runs ~24% higher than the
    NCR scale. That does NOT affect this model's own squad — the MILP objective is linear
    in points and every constraint is on price, so a global scale factor leaves the optimum
    unchanged. It DOES matter when averaging against another model's points (see
    ncr_ensemble, which reports the effective weighting).
    """
    base = NP.build_projection(exclude_teams=exclude_teams)
    pred = predict().set_index("fantasy_id")
    common = base.set_index("id").index.intersection(pred.index)
    df = base.set_index("id").loc[common].copy()

    champ = pred.loc[common, "target_pts_hat"].astype(float)
    ratio = champ.mean() / df["starter_exp"].mean()
    df["starter_exp"] = champ                              # raw, unrescaled
    df["supersub_exp"] = np.where(df.status.eq("B"), 3 * df.starter_exp, 0.5 * df.starter_exp)
    if "captain_score" in pred.columns:
        df["captain_score"] = pred.loc[common, "captain_score"]
    print(f"champion raw points on {len(df)} players; mean is {ratio:.3f}x the empirical model's "
          f"(6n scale, not rescaled)")
    return df.reset_index().rename(columns={"index": "id"})


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    df = build_projection(exclude_teams=args.exclude)
    tag = "_champion" + ("_ex_" + "_".join(t.replace(" ", "") for t in args.exclude)
                         if args.exclude else "")
    df.to_csv(NP.NCR / f"ncr_gw{NP.CUR_GD}_projections{tag}.csv", index=False)
    squad, _, _ = NP.optimise(df)

    # The MILP designates the captain by raw points. The champion has a DEDICATED
    # captain head (captain_score), which measurably disagrees — honour it, within the
    # XV the MILP chose. Squad is unchanged; only the armband moves.
    if "captain_score" in squad.columns and squad["captain_score"].notna().any():
        xv = squad[~squad.is_sub & squad.status.eq("P")]
        if len(xv):
            pick = xv.loc[xv["captain_score"].idxmax(), "name"]
            old = squad.loc[squad.is_capt, "name"].iloc[0] if squad.is_capt.any() else "—"
            if pick != old:
                print(f"captain: points-argmax {old} → champion's captain head picks {pick}")
            squad["is_capt"] = squad["name"].eq(pick)

    NP.render(df, squad, tag, args.exclude, title="6n champion (frozen, transplanted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
