#!/usr/bin/env python3
"""
NCR GW1 predictions from the *real* research.py rate model
==========================================================
Runs the trained 6n component-rate heads (model/baselines.predict_rates) on the
Nations Championship GW1 players, then scores the predicted per-80 rates under the
NCR rubric instead of the 6n formula.

Why this works: the model predicts 13 competition-agnostic per-80 SCORED rates
(tries, tackles, try_assists, conversion/penalty/drop goals, defenders_beaten,
offload, tackle_turnover, penalties_conceded, cards, metres). Those are exactly the
raw ingredients of the NCR scoring rubric, so we reuse the trained rate heads and
swap only the final rate→points mapping.

Pipeline:
  1. Build NCR GW1 feature rows in the model's schema (FORM/CLASS/ROLE/BIO/MATCHUP
     from ncr_player_match + rp_* ; unbuilt families → NaN, PositionMeanImputer fills
     them with the 6n train position-mean).
  2. Train on the 2,760-row 6n store, predict the 13 rates for the NCR rows (GLM
     engine — LightGBM isn't installed; GLM is a research.py baseline head).
  3. rate→NCR points, × confirmed minutes, × market/weather matchup (from ncr_project),
     then the same MILP squad optimiser.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import build_features as BF
from model.data import load, FEATURE_COLS
from model.baselines import predict_rates, SCORED
import model.ncr_project as NP

ASOF = NP.ASOF                    # derived from the current gameday
HALF_LIFE = 90.0

# NCR points per predicted rate component (metres is NOT scored in NCR → 0)
NCR_PTS = {"tries": 12, "try_assists": 5, "conversion_goals": 2, "penalty_goals": 3,
           "drop_goals_converted": 5, "defenders_beaten": 2, "offload": 2,
           "tackles": 1, "tackle_turnover": 4, "penalties_conceded": -1,
           "yellow_cards": -5, "red_cards": -10, "metres": 0}
ATT_C = {"tries", "try_assists", "conversion_goals", "penalty_goals",
         "drop_goals_converted", "defenders_beaten", "offload"}
DEF_C = {"tackles", "tackle_turnover"}
# NCR position (fantasy skill_desc) → model canonical_pos vocabulary
POS2CANON = {"Prop": "Prop", "Hooker": "Hooker", "Lock": "Second-row",
             "Loose Forward": "Back-row", "Scrum Half": "Scrum-half",
             "Fly Half": "Fly-half", "Centre": "Centre", "Back Three": "Back-three"}
FORWARD = {"Prop", "Hooker", "Second-row", "Back-row"}


def build_ncr_feature_rows():
    pool = pd.read_csv(NP.NCR / "ncr_players.csv")
    pool = pool[pool.player_status.isin(["P", "B"])].copy()
    hist = pd.read_csv(NP.NCR / "ncr_player_match.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    fx = pd.read_csv(NP.NCR / "ncr_fixtures.csv")
    cur_gd = int(fx.loc[fx.get("iscurrent", 0) == 1, "gameday"].min()) \
        if (fx.get("iscurrent", 0) == 1).any() else 1
    gwc = fx[fx.gameday == cur_gd]                    # current gameday's opponents
    opp = {}
    for _, r in gwc.iterrows():
        opp[r.home] = r.away; opp[r.away] = r.home

    # fantasy player → historical api player_id (reuse ncr_project matcher)
    pid_map = NP.match_players(pool.rename(columns={"id": "id"}), hist)
    canon_of = lambda skill: POS2CANON[NP.SKILL2POS[skill]]
    # canonical pos per historical player (for matchup fallback) from fantasy pool
    pool_pos = {pid_map[r.id]: canon_of(r.skill_desc)
                for r in pool.itertuples() if r.id in pid_map}

    # rp compstats / bio, keyed by slug (all 12 nations now scraped)
    cs = pd.read_csv(NP.DATA.parent / "data" / "rp_compstats.csv")
    cs["_end_year"] = cs["season"].map(BF._season_end_year)
    for s in BF.CLASS_STATS:
        cs[s] = pd.to_numeric(cs.get(s), errors="coerce").fillna(0.0)
    cs["minutes"] = pd.to_numeric(cs["minutes"], errors="coerce").fillna(0.0)
    cs["games"] = pd.to_numeric(cs["games"], errors="coerce").fillna(0)
    cs_by_slug = {slug: g for slug, g in cs.groupby("slug")}
    bio = pd.read_csv(NP.DATA.parent / "data" / "rp_bio.csv").drop_duplicates("slug").set_index("slug")

    # fantasy player → rp slug: resolve from the pool's own full_name against the
    # scraped slug set (the scrape was keyed off these very names), trying the same
    # candidate variants the scraper used; fall back to the historical api name.
    def slug_candidates(name):                            # mirrors rugbypass_backfill's
        base = NP.norm(name)                              # variants without importing it
        hyphen = base.replace(" ", "-")                   # (that module needs `requests`)
        cands = [hyphen, f"{hyphen}-1", f"{hyphen}-2"]
        toks = base.split(" ")
        if len(toks) >= 2:
            cands.append("".join(toks))
        return cands
    slug_set = set(bio.index)
    name_by_pid = hist.groupby("player_id")["player_name"].first().to_dict()

    def resolve_slug(full_name, pid=None):
        names = [full_name] + ([name_by_pid[pid]] if pid in name_by_pid else [])
        for nm in names:
            for cand in slug_candidates(nm):
                if cand in slug_set:
                    return cand
        return None

    # matchup history feed: per-opponent, with the model's position tags
    hist["_matchup_pos"] = hist["player_id"].map(pool_pos)
    hist["_matchup_is_forward"] = hist["_matchup_pos"].isin(FORWARD)
    hist_by_pid = {pid: g.sort_values("date") for pid, g in hist.groupby("player_id")}
    hist_by_opp = {t: g.sort_values("date") for t, g in hist.groupby("opponent")}

    rows = []
    for r in pool.itertuples():
        pid = pid_map.get(r.id)
        canon = canon_of(r.skill_desc)
        rec = {"fantasy_id": r.id, "player_name": r.full_name, "team": r.team_name,
               "canonical_pos": canon, "is_forward": canon in FORWARD,
               "role_is_jumper": canon in {"Second-row", "Back-row"},
               "role_is_front_row": canon in {"Prop", "Hooker"},
               "minutes": 80.0, "started": True, "jersey": 1, "bio_position": None}
        slug = resolve_slug(r.full_name, pid)
        # PIT cutoff 2027: for a July-2026 fixture the 2025/26 club season and Super
        # Rugby 2026 are COMPLETE (end-year 2026), unlike a February Six Nations.
        rec.update(BF.class_features(slug, 2027, cs_by_slug))
        ph = hist_by_pid.get(pid)
        ph = ph[ph["date"] < ASOF] if ph is not None else hist.iloc[0:0]
        rec.update(BF.form_role_features(ph, ASOF, HALF_LIFE))
        oh = hist_by_opp.get(opp.get(r.team_name))
        oh = oh[oh["date"] < ASOF] if oh is not None else hist.iloc[0:0]
        rec.update(BF.matchup_features(oh, ASOF, canon, canon in FORWARD, HALF_LIFE))
        if slug in bio.index:
            b = bio.loc[slug]
            age = pd.to_numeric(b.get("age"), errors="coerce")
            rec["bio_age_at_fixture"] = float(age) if pd.notna(age) and age > 0 else np.nan
            rec["bio_height_cm"] = pd.to_numeric(b.get("height_cm"), errors="coerce")
            rec["bio_weight_kg"] = pd.to_numeric(b.get("weight_kg"), errors="coerce")
            rec["bio_position"] = b.get("position")
        rec["has_class"] = rec.get("class_minutes_prior", 0) > 0
        rec["has_form"] = rec.get("form_n_prior", 0) > 0
        rec["has_bio"] = bool(slug in bio.index)
        rows.append(rec)
    return pd.DataFrame(rows)


def predict_ncr_rates(engine="glm"):
    train = load()
    ncr = build_ncr_feature_rows()
    # align NCR rows to the training columns (unbuilt feature families → NaN → imputed)
    ncr = ncr.reindex(columns=train.columns.tolist() + ["fantasy_id"])
    ncr["bio_position"] = ncr["bio_position"].astype("category")
    combined = pd.concat([train.assign(fantasy_id=np.nan), ncr], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").fillna(ASOF)
    combined["bio_position"] = combined["bio_position"].astype("category")
    for c in ("role_is_jumper", "role_is_front_row"):    # object after concat → bool
        combined[c] = combined[c].fillna(False).astype(bool)
    tr_idx = np.arange(len(train))
    te_idx = np.arange(len(train), len(combined))
    if engine == "lgbm":
        from model.train_components import predict_rates_lgbm_only
        rates = predict_rates_lgbm_only(combined, tr_idx, te_idx, "pre_team_sheet")
    else:
        rates = predict_rates(combined, tr_idx, te_idx, "pre_team_sheet", engine)
    rates = rates.reset_index(drop=True)
    rates["fantasy_id"] = combined.iloc[te_idx]["fantasy_id"].to_numpy()
    # supplement rates the model doesn't predict, from the player's own FORM (data)
    for extra in ("clean_breaks", "missed_tackles"):
        col = f"form_per80_{extra}"
        rates[extra] = combined.iloc[te_idx][col].to_numpy() if col in combined else 0.0
    return rates


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()

    rates = predict_ncr_rates()
    # per-player NCR per-80 components from the MODEL-predicted rates
    r = rates.fillna(0.0)
    att = sum(NCR_PTS[c] * r[c] for c in ATT_C) + 3.0 * r["clean_breaks"].clip(lower=0)
    dfn = sum(NCR_PTS[c] * r[c] for c in DEF_C) - 1.0 * r["missed_tackles"].clip(lower=0)
    dsc = (NCR_PTS["penalties_conceded"] * r["penalties_conceded"]
           + NCR_PTS["yellow_cards"] * r["yellow_cards"] + NCR_PTS["red_cards"] * r["red_cards"])
    model_per80 = pd.DataFrame({"fantasy_id": r["fantasy_id"], "att": att, "dfn": dfn, "dsc": dsc})

    # hand the model rates to ncr_project's matchup + minutes + scoring + optimiser
    df = NP.build_projection(exclude_teams=args.exclude, override_rates=model_per80)
    tag = "_ex_" + "_".join(t.replace(" ", "") for t in args.exclude) if args.exclude else ""
    df.to_csv(NP.NCR / f"ncr_gw{NP.CUR_GD}_projections{tag}.csv", index=False)
    squad, _, _ = NP.optimise(df)
    NP.render(df, squad, tag, args.exclude, title="research.py rate model")


if __name__ == "__main__":
    main()
