#!/usr/bin/env python3
"""
build_features.py  —  the model feature store (Phase 5)
=======================================================
Assembles one row per (player, Six Nations match, season) for 2023–2026 with the
six feature families from the plan, every value **point-in-time (PIT)**:

  1. CLASS    — per-80 component rates from RugbyPass season aggregates
                (rp_compstats.csv), **prior completed seasons only**.
  2. FORM     — exp-decayed (90d half-life) recent per-80 rates from the rich
                per-match feed (api_player_match.csv: 6N + club backfill),
                strictly date < fixture_date; plus minutes/start trend and a
                hot/cold delta vs CLASS.
  3. FIXTURE  — the 22 opponent-context features (fixture_difficulty.csv),
                left-joined on (fixture_id, team_id).
  4. BIO      — age-at-fixture / height / weight / position (rp_bio.csv).
  5. ROLE     — goal-kicker rate, starter prob, set-piece (lineout) involvement,
                all decayed & PIT.
  6. OWN-TEAM — own side's decayed attacking strength (api_team_match.csv) plus
                own WR points / expected margin (already in FIXTURE).

All cross-source joins go through data/player_crosswalk.csv (api_player_id ↔ key
↔ rp_slug). Canonical position is taken from the crosswalk (a fixed, resolved
value) rather than re-derived per run — closes the run-dependent-position flag.

Cold start (no prior history for a family) → NaN feature + an explicit n_prior /
has_* flag, so the modelling layer can shrink to a position×competition baseline.

No API calls — pure local joins. Output: data/model_player_match.csv.

Usage:
    python build_features.py                 # 90-day FORM half-life
    python build_features.py --half-life 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_api_official import norm_key

BASE = Path(__file__).parent
DATA = BASE / "data"

SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]
REFERENCE_DATE = pd.Timestamp("2026-06-15")   # ~rp_bio scrape date, for age-at-fixture
FORWARD_GROUPS = {"Prop", "Hooker", "Second-row", "Back-row"}

# rich per-match stats decayed for FORM (must exist in api_player_match.csv)
FORM_STATS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals",
    "drop_goals_converted", "metres", "defenders_beaten", "clean_breaks",
    "offload", "runs", "tackles", "missed_tackles", "tackle_turnover",
    "passes", "turnovers_conceded", "penalties_conceded", "lineouts_won",
]
# RugbyPass season-aggregate columns → CLASS per-80 (rp_compstats.csv names)
CLASS_STATS = [
    "tries", "try_assists", "metres", "post_contact_metres", "defenders_beaten",
    "clean_breaks", "offloads", "passes", "tackles", "missed_tackles",
    "turnovers_won", "turnovers_conceded", "penalties_conceded",
]
# own-team attacking strength, decayed per match (api_team_match.csv names)
OWNTEAM_STATS = {
    "carries_metres": "ownteam_metres", "tries_for": "ownteam_tries",
    "clean_breaks": "ownteam_breaks", "defenders_beaten": "ownteam_db",
    "offload": "ownteam_offloads", "possession": "ownteam_possession",
    # set-piece strength — drives the SW / LS latent block in build_targets.py:
    # how many scrums / lineout-steals this side tends to win (→ pack's SW/LS pts)
    "scrums_won": "ownteam_scrums_won", "scrums_success": "ownteam_scrum_success",
    "lineout_won_steal": "ownteam_lineout_steal", "lineout_success": "ownteam_lineout_success",
}

JUMPER_GROUPS = {"Second-row", "Back-row"}     # primary lineout-steal targets
FRONT_ROW_GROUPS = {"Prop", "Hooker"}          # scrum core, no lineout jump

ID_COLS = [
    "season", "comp_id", "round", "fixture_id", "date", "team", "team_id",
    "opponent", "opponent_id", "home_away", "player_id", "player_name",
    "jersey", "started", "minutes", "canonical_pos", "is_forward",
]


def _decay_weights(asof: pd.Timestamp, dates: pd.Series, hl: float) -> np.ndarray:
    days = (asof - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, days / hl)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


# ─── CLASS ──────────────────────────────────────────────────────────────────
def _season_end_year(season) -> int:
    """RugbyPass labels are mixed: '2025/2026' (club) and '2026' (intl). The
    PIT-comparable key is the season's END year."""
    s = str(season)
    return int(s.split("/")[-1]) if s and s.split("/")[-1].isdigit() else -1


def class_features(slug: str, fixture_season: int, cs_by_slug: dict) -> dict:
    """Minutes-weighted per-80 rates over the player's **completed prior**
    RugbyPass seasons (end_year < fixture_season). 6N-only subset added too."""
    out = {f"class_per80_{s}": np.nan for s in CLASS_STATS}
    out.update({"class_games_prior": 0, "class_minutes_prior": 0.0,
                "class6n_per80_metres": np.nan, "class6n_per80_tries": np.nan,
                "class6n_minutes_prior": 0.0})
    rows = cs_by_slug.get(slug)
    if rows is None:
        return out
    prior = rows[rows["_end_year"] < fixture_season]
    if not prior.empty:
        mins = prior["minutes"].to_numpy(dtype=float)
        tot_min = mins.sum()
        out["class_games_prior"] = int(prior["games"].sum())
        out["class_minutes_prior"] = float(tot_min)
        if tot_min > 0:
            for s in CLASS_STATS:
                out[f"class_per80_{s}"] = float(prior[s].to_numpy(float).sum() / tot_min * 80)
    six = prior[prior["competition"].astype(str).str.startswith("Six Nations")]
    if not six.empty:
        m6 = six["minutes"].to_numpy(float).sum()
        out["class6n_minutes_prior"] = float(m6)
        if m6 > 0:
            out["class6n_per80_metres"] = float(six["metres"].sum() / m6 * 80)
            out["class6n_per80_tries"] = float(six["tries"].sum() / m6 * 80)
    return out


# ─── FORM + ROLE ──────────────────────────────────────────────────────────────
def form_role_features(hist: pd.DataFrame, asof: pd.Timestamp, hl: float) -> dict:
    """Decayed per-80 FORM rates + minutes/start trend + ROLE signals, over the
    player's matches strictly before the fixture (any competition)."""
    out = {f"form_per80_{s}": np.nan for s in FORM_STATS}
    out.update({
        "form_n_prior": 0, "form_minutes_recent": np.nan,
        "form_start_rate": np.nan, "form_days_since_last": np.nan,
        "form_hot_metres": np.nan, "form_hot_tries": np.nan,
        "role_goal_kicker_rate": np.nan, "role_kick_attempts": np.nan,
        "role_lineout_per80": np.nan,
        # POTM propensity: decayed rate of past 6N Player-of-the-Match awards
        # (official label, strictly prior). Sparse — a weak prior; the main POTM
        # signal at model time is the player's predicted performance rank.
        "role_potm_rate": np.nan, "role_potm_n_label": 0,
    })
    if hist.empty:
        return out
    out["form_n_prior"] = int(len(hist))
    w = _decay_weights(asof, hist["date"], hl)
    sw = w.sum()
    # POTM propensity over prior matches that carry an official label
    if "potm" in hist.columns and "has_potm_label" in hist.columns:
        lblm = hist["has_potm_label"].to_numpy(bool)
        if lblm.any():
            wl = w[lblm]
            out["role_potm_n_label"] = int(lblm.sum())
            if wl.sum() > 0:
                out["role_potm_rate"] = float(
                    (wl * hist["potm"].to_numpy(float)[lblm]).sum() / wl.sum())
    mins = hist["minutes"].to_numpy(dtype=float)
    wm = w * mins
    swm = wm.sum()
    if sw > 0:
        out["form_minutes_recent"] = float((w * mins).sum() / sw)
        out["form_start_rate"] = float((w * _bool(hist["started"]).to_numpy(float)).sum() / sw)
    out["form_days_since_last"] = float((asof - hist["date"].max()).days)
    if swm > 0:
        for s in FORM_STATS:
            out[f"form_per80_{s}"] = float((w * hist[s].to_numpy(float)).sum() / swm * 80)
        # ROLE: goal-kicking and set-piece involvement
        attempts = (hist["conversion_goals"] + hist["penalty_goals"]
                    + hist["missed_conversion_goals"] + hist["missed_penalty_goals"]).to_numpy(float)
        out["role_goal_kicker_rate"] = float((w * (attempts > 0)).sum() / sw) if sw > 0 else np.nan
        out["role_kick_attempts"] = float((w * attempts).sum() / sw) if sw > 0 else np.nan
        out["role_lineout_per80"] = out["form_per80_lineouts_won"]
    return out


# ─── OWN-TEAM ──────────────────────────────────────────────────────────────────
def ownteam_features(hist: pd.DataFrame, asof: pd.Timestamp, hl: float) -> dict:
    """Own side's decayed attacking output over its matches before the fixture."""
    out = {v: np.nan for v in OWNTEAM_STATS.values()}
    out["ownteam_n_prior"] = 0
    if hist.empty:
        return out
    out["ownteam_n_prior"] = int(len(hist))
    w = _decay_weights(asof, hist["date"], hl)
    sw = w.sum()
    if sw > 0:
        for col, name in OWNTEAM_STATS.items():
            out[name] = float((w * hist[col].to_numpy(float)).sum() / sw)
    return out


def build(half_life: float) -> pd.DataFrame:
    ap = pd.read_csv(DATA / "api_player_match.csv")
    ap["date"] = pd.to_datetime(ap["date"], errors="coerce")
    ap = ap.dropna(subset=["date", "player_id"])

    # attach the official POTM label to each api_player_match row it can be
    # joined to (initial-key on season/round/team), for the PIT POTM-propensity
    # FORM signal. has_potm_label marks rows where a label exists at all, so the
    # rate is taken only over labelled prior matches (not diluted by 2024/club).
    off = pd.read_csv(DATA / "official_player_match.csv")
    off["nk"] = off["name"].map(norm_key)
    off["potm"] = pd.to_numeric(off["POTM"], errors="coerce").fillna(0)
    off_potm = (off.dropna(subset=["Pts"])
                  .drop_duplicates(["season", "round", "team", "nk"])
                  [["season", "round", "team", "nk", "potm"]])
    ap["nk"] = ap["player_name"].map(norm_key)
    ap = ap.merge(off_potm, on=["season", "round", "team", "nk"], how="left")
    ap["has_potm_label"] = ap["potm"].notna()
    ap["potm"] = ap["potm"].fillna(0)

    tm = pd.read_csv(DATA / "api_team_match.csv")
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.dropna(subset=["date", "team_id"])

    xw = pd.read_csv(DATA / "player_crosswalk.csv")
    # Drop NA on the FRAME (not just the id column) so api_player_id and rp_slug
    # stay row-aligned — zipping a .dropna()'d id Series against the full rp_slug
    # Series shifts them apart at every RugbyPass/official-only row, mismapping
    # every player to the wrong slug.
    _xw = xw.dropna(subset=["api_player_id"])
    pid2slug = dict(zip(_xw.api_player_id.astype(int), _xw.rp_slug))

    # Canonical position from the FIXED 6N-start window (domain-correct, keyed by a
    # unique player_id), with the player's all-comp modal-start jersey as fallback
    # for bench-only players. Deliberately NOT the crosswalk's canonical_pos column —
    # that one is unreliable (mislabels e.g. hookers/centres as Fly-half).
    pl = pd.read_csv(DATA / "6n_players.csv")
    pos_6n = dict(zip(pl.player_id, pl.canonical_pos))
    pos_api = (ap.dropna(subset=["canonical_pos"])
                 .drop_duplicates("player_id")
                 .set_index("player_id")["canonical_pos"].to_dict())

    def canon_pos(pid: int):
        p = pos_6n.get(pid)
        return p if isinstance(p, str) and p else pos_api.get(pid)

    cs = pd.read_csv(DATA / "rp_compstats.csv")
    cs["_end_year"] = cs["season"].map(_season_end_year)
    for s in CLASS_STATS:
        cs[s] = pd.to_numeric(cs.get(s), errors="coerce").fillna(0.0)
    cs["minutes"] = pd.to_numeric(cs["minutes"], errors="coerce").fillna(0.0)
    cs["games"] = pd.to_numeric(cs["games"], errors="coerce").fillna(0)
    cs_by_slug = {slug: g for slug, g in cs.groupby("slug")}

    bio = pd.read_csv(DATA / "rp_bio.csv").set_index("slug")

    fd = pd.read_csv(DATA / "fixture_difficulty.csv")
    fd_feats = [c for c in fd.columns
                if c not in {"team", "opponent", "opponent_id", "date"}]

    # pre-group history feeds by entity for fast PIT slicing
    hist_by_pid = {pid: g.sort_values("date") for pid, g in ap.groupby("player_id")}
    hist_by_team = {tid: g.sort_values("date") for tid, g in tm.groupby("team_id")}

    # base grid = 6N target player-matches 2023–2026
    base = ap[(ap.comp_id == SIX_NATIONS) & (ap.season.isin(TARGET_SEASONS))].copy()
    base = base.sort_values(["season", "date", "fixture_id", "team_id", "jersey"])

    recs = []
    for r in base.itertuples(index=False):
        pid = int(r.player_id)
        asof = r.date
        slug = pid2slug.get(pid)

        rec = {c: getattr(r, c, None) for c in ID_COLS if hasattr(r, c)}
        # fixed canonical position from the 6N-start window (see canon_pos)
        rec["canonical_pos"] = canon_pos(pid)
        rec["is_forward"] = rec["canonical_pos"] in FORWARD_GROUPS
        # set-piece role flags — who the SW / LS latent block accrues to
        rec["role_is_jumper"] = rec["canonical_pos"] in JUMPER_GROUPS
        rec["role_is_front_row"] = rec["canonical_pos"] in FRONT_ROW_GROUPS

        # CLASS (prior completed seasons)
        rec.update(class_features(slug, int(r.season), cs_by_slug))

        # FORM + ROLE (strictly date < fixture)
        ph = hist_by_pid.get(pid)
        ph = ph[ph["date"] < asof] if ph is not None else ap.iloc[0:0]
        rec.update(form_role_features(ph, asof, half_life))

        # hot/cold = recent FORM vs lifetime CLASS
        if pd.notna(rec.get("form_per80_metres")) and pd.notna(rec.get("class_per80_metres")):
            rec["form_hot_metres"] = rec["form_per80_metres"] - rec["class_per80_metres"]
        if pd.notna(rec.get("form_per80_tries")) and pd.notna(rec.get("class_per80_tries")):
            rec["form_hot_tries"] = rec["form_per80_tries"] - rec["class_per80_tries"]

        # OWN-TEAM (own side, strictly date < fixture)
        th = hist_by_team.get(int(r.team_id))
        th = th[th["date"] < asof] if th is not None else tm.iloc[0:0]
        rec.update(ownteam_features(th, asof, half_life))

        # BIO
        if slug in bio.index:
            b = bio.loc[slug]
            age = pd.to_numeric(b.get("age"), errors="coerce")
            if pd.notna(age) and age <= 0:   # 0 == RugbyPass scrape gap, not a real age
                age = np.nan
            rec["bio_age_at_fixture"] = (
                float(age - (REFERENCE_DATE - asof).days / 365.25)
                if pd.notna(age) else np.nan)
            rec["bio_height_cm"] = pd.to_numeric(b.get("height_cm"), errors="coerce")
            rec["bio_weight_kg"] = pd.to_numeric(b.get("weight_kg"), errors="coerce")
            rec["bio_position"] = b.get("position")
        else:
            rec.update({"bio_age_at_fixture": np.nan, "bio_height_cm": np.nan,
                        "bio_weight_kg": np.nan, "bio_position": None})

        # coverage flags
        rec["has_class"] = rec["class_minutes_prior"] > 0
        rec["has_form"] = rec["form_n_prior"] > 0
        rec["has_bio"] = bool(slug in bio.index)
        recs.append(rec)

    feat = pd.DataFrame(recs)

    # FIXTURE (the 22 opponent-context features), left-join on (fixture_id, team_id)
    feat = feat.merge(fd[fd_feats], on=["fixture_id", "team_id"], how="left")
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-life", type=float, default=90.0)
    args = ap.parse_args()

    feat = build(args.half_life)
    out = DATA / "model_player_match.csv"
    feat.to_csv(out, index=False)

    fam = {
        "CLASS": [c for c in feat if c.startswith("class")],
        "FORM": [c for c in feat if c.startswith("form")],
        "ROLE": [c for c in feat if c.startswith("role")],
        "OWN-TEAM": [c for c in feat if c.startswith("ownteam")],
        "BIO": [c for c in feat if c.startswith("bio")],
        "FIXTURE": [c for c in feat if c.startswith(("opp_", "h2h_", "team_wr", "wr_"))],
    }
    print(f"✅  {len(feat)} rows, {feat.fixture_id.nunique()} matches, "
          f"seasons {sorted(feat.season.unique())}")
    for name, cols in fam.items():
        if not cols:
            continue
        cov = 100 * feat[cols].notna().any(axis=1).mean()
        print(f"    {name:9s} {len(cols):2d} cols, {cov:5.1f}% rows non-null")
    print(f"    cold-start: CLASS {100*(~feat.has_class).mean():.0f}%, "
          f"FORM {100*(~feat.has_form).mean():.0f}%, BIO {100*(~feat.has_bio).mean():.0f}%")
    print(f"💾  {out}")


if __name__ == "__main__":
    main()
