#!/usr/bin/env python3
"""
build_targets.py  —  the model target store (Phase 6, Strategy C)
=================================================================
For every (player, 6N match, season) row in 2023–2026 this emits:

  • y_*        — the **measurable components** the model learns to predict
                 (tries, assists, conversions, penalties, drop goals, metres,
                 defenders-beaten, clean-breaks, offloads, runs, tackles, missed
                 tackles, tackle-turnovers, passes, penalties-conceded, cards),
                 taken straight from the realised api_player_match line.
  • recon_pts  — the **deterministic fantasy score** of those components under
                 the *modern* (2025/2026) official scoring formula, recovered by
                 least-squares from the official category breakdown — with the
                 measurable categories it matches official Pts at MAE≈0.006
                 (exact; see SCORING below). recon is in modern points for
                 every season, so it is directly comparable across 2023–2026.
  • official_pts — the official fantasy `Pts` **label** where it exists (2023
                 R1–4 in the *old* formula, 2025, 2026 modern). 2024 has no
                 official labels (game not archived). Reference only — the 2023
                 label is on the old scale and is NOT used to fit the latent block.

  • the **latent block** — the five fantasy-only categories the API can't see,
    decomposed (instead of a single flat per-position residual) into real,
    match-specific, attributable pieces:
       lat_sw    Scrum-Won      ≈ a · team scrums_won · pack-membership · min-share
                 (team scrums_won validated corr 0.84 vs official SW totals)
       lat_ls    Lineout-Steal  ≈ b · team lineout_won_steal · jumper · min-share
                 (team lineout_won_steal validated corr 0.96, ratio 1.00)
       lat_potm  POTM           = 15 · potm_winner   (realised label; a per-position
                 baseline 15·P(POTM) for the unlabelled 2024/2023 rows)
       lat_other per-position const absorbing the genuinely unrecoverable
                 50-22 (+7, 0.6% freq) and Kick-Retained (+2, 5.6% freq) plus noise
    a, b, the position constants and the POTM baseline are FIT on the modern
    (2025+2026) labelled rows. SW/LS/POTM now vary with the team's actual
    set-piece dominance and the player's role — the real fix for forward
    undervaluation (a back-row on a pack that wins 8 scrums is lifted; one on a
    beaten pack is not), rather than a flat "+5 to every back-row".
  • target_pts = recon_pts + lat_sw + lat_ls + lat_potm + lat_other — the
                 calibrated modern-points target the picker ranks on, consistent
                 across all four seasons.

SCORING (modern 2025/2026 official, recovered by OLS, R²=0.9994, MAE≈0.006):
    try            +15 forward / +10 back     (position-split confirmed)
    tackle (Ta)    +1     metres (MC)  floor(m/10)   assist (As)  +4
    conversion     +2     penalty (Pen)+3      drop goal    +4
    def-beaten     +2     offload      +2      breakdown-steal (BS=tackle_turnover) +5
    pen-conceded   -1     yellow card  -5      red card     -8
    (no appearance/minutes bonus — intercept ≈ 0)
  Latent block (folded into lat_*, never scored from components):
    50-22 +7   lineout-steal +7   POTM +15   scrum-won +1   kick-retained +2

No API calls. Output: data/model_targets.csv.

Usage:  python build_targets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compare_api_official import norm_key

BASE = Path(__file__).parent
DATA = BASE / "data"
SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]
MODERN_SEASONS = [2025, 2026]          # seasons on the current scoring formula
FORWARD_GROUPS = {"Prop", "Hooker", "Second-row", "Back-row"}
JUMPER_GROUPS = {"Second-row", "Back-row"}   # primary lineout-steal targets

# realised component columns carried as targets (api_player_match.csv names)
COMPONENTS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals",
    "drop_goals_converted", "metres", "defenders_beaten", "clean_breaks",
    "offload", "runs", "tackles", "missed_tackles", "tackle_turnover",
    "passes", "penalties_conceded", "yellow_cards", "red_cards",
]


def score_components(c: dict, is_forward: bool) -> float:
    """Deterministic fantasy score under the **modern** (2025/2026) official
    formula, recovered by OLS from the official category breakdown (MAE≈0.006).
    Scores only API-measurable categories; the five API-blind categories
    (50-22, LS, POTM, SW, KR) are restored by the latent block."""
    try_val = 15 if is_forward else 10
    return float(
        try_val * c["tries"]
        + 1.0 * c["tackles"]
        + c["metres"] // 10                 # 1 pt per *completed* 10m (floor, exact)
        + 4 * c["try_assists"]
        + 2 * c["conversion_goals"]
        + 3 * c["penalty_goals"]
        + 4 * c["drop_goals_converted"]
        + 2 * c["defenders_beaten"]
        + 2 * c["offload"]
        + 5 * c["tackle_turnover"]          # BS — breakdown steal
        - 1 * c["penalties_conceded"]
        - 5 * c["yellow_cards"]
        - 8 * c["red_cards"]
    )


def build() -> tuple[pd.DataFrame, dict]:
    ap = pd.read_csv(DATA / "api_player_match.csv")
    six = ap[(ap.comp_id == SIX_NATIONS) & (ap.season.isin(TARGET_SEASONS))].copy()

    # corrected canonical position / is_forward come from the feature store
    feat = pd.read_csv(DATA / "model_player_match.csv")[
        ["fixture_id", "player_id", "canonical_pos", "is_forward"]]
    six = six.merge(feat, on=["fixture_id", "player_id"], how="left",
                    suffixes=("", "_feat"))
    six["canonical_pos"] = six["canonical_pos_feat"].fillna(six["canonical_pos"])
    six["is_forward"] = six["is_forward_feat"].fillna(six["is_forward"]).astype(bool)
    six["is_jumper"] = six["canonical_pos"].isin(JUMPER_GROUPS)

    for col in COMPONENTS:
        six[col] = pd.to_numeric(six[col], errors="coerce").fillna(0)
    six["minutes"] = pd.to_numeric(six["minutes"], errors="coerce").fillna(0)
    six["min_share"] = (six["minutes"] / 80.0).clip(0, 1)

    # recon = modern-formula deterministic score of realised components
    six["recon_pts"] = [
        score_components({c: getattr(r, c) for c in COMPONENTS}, bool(r.is_forward))
        for r in six.itertuples(index=False)
    ]

    # ---- realised team set-piece counts (SW/LS attribution drivers) ----
    tm = pd.read_csv(DATA / "api_team_match.csv")[
        ["fixture_id", "team", "scrums_won", "lineout_won_steal"]]
    for c in ["scrums_won", "lineout_won_steal"]:
        tm[c] = pd.to_numeric(tm[c], errors="coerce").fillna(0)
    six = six.merge(tm.rename(columns={"scrums_won": "team_scrums_won",
                                       "lineout_won_steal": "team_lineout_steal"}),
                    on=["fixture_id", "team"], how="left")
    six[["team_scrums_won", "team_lineout_steal"]] = \
        six[["team_scrums_won", "team_lineout_steal"]].fillna(0)

    # ---- official label join: initial-key on (season, round, team) ----
    off = pd.read_csv(DATA / "official_player_match.csv")
    off["nk"] = off["name"].map(norm_key)
    off["POTM"] = pd.to_numeric(off["POTM"], errors="coerce").fillna(0)
    off_lbl = (off[["season", "round", "team", "nk", "Pts", "POTM"]]
               .dropna(subset=["Pts"])
               .drop_duplicates(["season", "round", "team", "nk"]))
    six["nk"] = six["player_name"].map(norm_key)
    six = six.merge(off_lbl.rename(columns={"Pts": "official_pts",
                                            "POTM": "potm_winner"}),
                    on=["season", "round", "team", "nk"], how="left")
    six["has_label"] = six["official_pts"].notna()
    six["is_modern"] = six["season"].isin(MODERN_SEASONS)

    # attribution drivers (per-player share of the team set-piece count)
    six["sw_drive"] = six["is_forward"].astype(float) * six["team_scrums_won"] * six["min_share"]
    six["ls_drive"] = six["is_jumper"].astype(float) * six["team_lineout_steal"] * six["min_share"]

    fit = _fit_latent(six)

    # ---- apply the fitted decomposition to every row ----
    six["lat_sw"] = fit["a"] * six["sw_drive"]
    six["lat_ls"] = fit["b"] * six["ls_drive"]
    # POTM: realised 15·winner where labelled, else per-position 15·P(POTM) baseline
    pos_potm = six["canonical_pos"].map(fit["potm_base"]).fillna(fit["potm_base_global"])
    six["lat_potm"] = np.where(six["has_label"],
                               15.0 * six["potm_winner"].fillna(0),
                               pos_potm)
    six["lat_other"] = six["canonical_pos"].map(fit["other_const"]).fillna(fit["other_global"])
    six["latent_total"] = six[["lat_sw", "lat_ls", "lat_potm", "lat_other"]].sum(axis=1)
    six["target_pts"] = six["recon_pts"] + six["latent_total"]

    out = six.rename(columns={c: f"y_{c}" for c in COMPONENTS})
    keep = (["season", "comp_id", "round", "fixture_id", "date", "team",
             "player_id", "player_name", "canonical_pos", "is_forward", "is_jumper",
             "is_modern", "minutes", "started", "min_share",
             "team_scrums_won", "team_lineout_steal"]
            + [f"y_{c}" for c in COMPONENTS]
            + ["recon_pts", "official_pts", "potm_winner", "has_label",
               "lat_sw", "lat_ls", "lat_potm", "lat_other",
               "latent_total", "target_pts"])
    return out[keep], fit


def _fit_latent(six: pd.DataFrame) -> dict:
    """Fit the latent decomposition on MODERN labelled rows (2025+2026) — the
    only modern-scoring labels. 2023 is on the old scale and is excluded.

    Steps:
      1. lat_potm is realised (15·winner) → subtract it first.
      2. Regress (official − recon − 15·potm) on sw_drive + ls_drive +
         per-position dummies (no global intercept). The slopes a, b are the
         per-unit SW / LS attribution; the position constants absorb the
         unrecoverable 50-22 + KR baseline (+ noise) per position.
      3. Position POTM baseline = mean(15·winner) by position, for unlabelled rows.
    """
    lab = six[six.has_label & six.is_modern].copy()
    y = (lab["official_pts"] - lab["recon_pts"] - 15.0 * lab["potm_winner"].fillna(0)).to_numpy(float)

    positions = sorted(lab["canonical_pos"].dropna().unique())
    pos_idx = {p: i for i, p in enumerate(positions)}
    n, k = len(lab), len(positions)
    X = np.zeros((n, k + 2))
    for i, p in enumerate(lab["canonical_pos"]):
        if p in pos_idx:
            X[i, pos_idx[p]] = 1.0           # per-position constant
    X[:, k] = lab["sw_drive"].to_numpy(float)
    X[:, k + 1] = lab["ls_drive"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    other_const = {p: float(coef[pos_idx[p]]) for p in positions}
    a, b = float(coef[k]), float(coef[k + 1])

    potm_base = (lab.groupby("canonical_pos")
                 .apply(lambda g: 15.0 * g["potm_winner"].fillna(0).mean()))
    return {
        "a": a, "b": b,
        "other_const": other_const,
        "other_global": float(np.mean(list(other_const.values()))),
        "potm_base": potm_base.to_dict(),
        "potm_base_global": float(15.0 * lab["potm_winner"].fillna(0).mean()),
        "positions": positions,
    }


def main():
    tgt, fit = build()
    out = DATA / "model_targets.csv"
    tgt.to_csv(out, index=False)

    lab = tgt[tgt.has_label & tgt.is_modern]          # MAE measured on modern only
    mae_recon = float((lab.official_pts - lab.recon_pts).abs().mean())
    mae_flat = float((lab.official_pts - (lab.recon_pts
                      + lab.canonical_pos.map(
                          lab.assign(r=lab.official_pts-lab.recon_pts)
                             .groupby("canonical_pos")["r"].mean()))).abs().mean())
    mae_decomp = float((lab.official_pts - lab.target_pts).abs().mean())
    print(f"✅  {len(tgt)} rows, {tgt.fixture_id.nunique()} matches, "
          f"seasons {sorted(tgt.season.unique())}")
    print(f"    modern (2025+26) labelled rows used for fit/MAE: {len(lab)}")
    print(f"    MAE vs official:  recon={mae_recon:.2f}  "
          f"+flat-residual={mae_flat:.2f}  +decomposed-latent={mae_decomp:.2f}")
    print(f"    fit: SW a={fit['a']:.2f} pt/(scrum·share)   "
          f"LS b={fit['b']:.2f} pt/(steal·share)   "
          f"POTM coeff=15 (realised)")
    print("    mean latent piece by position (fwd/bck):")
    g = (lab.groupby("canonical_pos")[["lat_sw", "lat_ls", "lat_potm", "lat_other", "latent_total"]]
         .mean().sort_values("latent_total", ascending=False))
    for pos, row in g.iterrows():
        f = "fwd" if pos in FORWARD_GROUPS else "bck"
        print(f"        {str(pos):12s} ({f})  sw={row.lat_sw:4.1f} ls={row.lat_ls:4.1f} "
              f"potm={row.lat_potm:4.1f} other={row.lat_other:5.1f}  total={row.latent_total:5.1f}")
    fwd = lab.is_forward
    print(f"    forward mean latent {lab[fwd].latent_total.mean():+.1f} "
          f"vs back {lab[~fwd].latent_total.mean():+.1f}")
    print(f"💾  {out}")


if __name__ == "__main__":
    main()
