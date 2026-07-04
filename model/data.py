#!/usr/bin/env python3
"""model/data.py  —  the model-layer data loader (Phase 1).

Joins the two frozen, PIT-clean extraction stores 1:1 on (fixture_id, player_id)
and exposes the leakage-audited feature column families.

Stores (read-only):
  data/model_player_match.csv  — 2,760 rows with PIT feature families + id/meta/flags
  data/model_targets.csv       — 2,760 rows of y_* components, recon/official/target pts,
                                 latent block, set-piece/POTM labels.

Feature families (91+ cols, by prefix):
  CLASS 18 (class*)   FORM 23 (form*)   ROLE 7 (role*)   OWNTEAM 11 (ownteam*)
  BIO 4 (bio*)        FIXTURE 28 (opp_* / h2h_* / team_wr* / wr_*)
  TEAMPLAY (teamplay_*) explicit team edge / game-script predictions
  MARKET (market_*) optional external betting/market fixture-strength features
  WEATHER (weather_*) optional venue/weather features
  ROLECERT (rolecert_*) optional named-role certainty features
  STYLE (style_*) optional tactical-style priors

Everything that is a TARGET, LABEL, or realised-outcome (y_*, recon_pts,
official_pts, lat_*, target_pts, latent_total, team_scrums_won,
team_lineout_steal, potm_winner, min_share) is kept out of FEATURE_COLS and
guarded by an explicit denylist assertion.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
# Legacy bool plus explicit mode.  Mode is preferred; the bool keeps older
# exploratory snippets working.
TEAMPLAY_ENABLED = False
TEAMPLAY_MODE = "off"  # off | raw | aspects
MARKET_FEATURES_ENABLED = False
WEATHER_FEATURES_ENABLED = False
ROLECERT_FEATURES_ENABLED = False
STYLE_FEATURES_ENABLED = False
STYLE_FEATURES_MODE = "off"  # off | raw | aspects
MATCHUP_FEATURES_ENABLED = False

# --- column taxonomy -------------------------------------------------------
# id / meta columns that are PIT-known but are not model inputs.
META_COLS = [
    "season", "comp_id", "round", "fixture_id", "date", "team", "team_id",
    "opponent", "opponent_id", "home_away", "player_id", "player_name",
    "canonical_pos", "is_forward",
]
# PIT-known lineup fields: only usable in the post-team-sheet feature view.
LINEUP_COLS = ["started", "jersey"]
# realised minutes is a TARGET-side quantity (drives minutes_hat), never a feature.
REALISED_LINEUP = ["minutes"]
# cold-start flags — legitimate features, but not part of the 91 "named family"
# set, so they are exposed separately and added to the model matrix on demand.
FLAG_COLS = ["has_class", "has_form", "has_bio"]

# bio_position is categorical text -> category for LightGBM, dropped for linear.
CATEGORICAL_COLS = ["bio_position"]

# Prefixes that define the 91 feature columns. "opp_" (with the underscore)
# deliberately excludes the meta cols "opponent"/"opponent_id".
_FEATURE_PREFIXES = (
    "class", "form", "role", "ownteam", "bio",
    "opp_", "h2h_", "team_wr", "wr_", "teamplay_",
    "market_", "weather_", "rolecert_", "style_",
    "matchup_",
)

# Realised-outcome / target / label columns that must NEVER be a feature.
LEAKAGE_DENYLIST = [
    "recon_pts", "official_pts", "target_pts", "latent_total",
    "lat_sw", "lat_ls", "lat_potm", "lat_other",
    "team_scrums_won", "team_lineout_steal", "potm_winner", "min_share",
    "has_label", "is_modern", "is_jumper",
    # realised lineup (minutes) is a target-side input, not a feature
    "minutes",
]


def _feature_cols(columns: list[str]) -> list[str]:
    """The 91 family-prefix feature columns, preserving store order."""
    cols = [
        c for c in columns
        if c.startswith(_FEATURE_PREFIXES)
        and c not in FLAG_COLS              # has_* start with 'has', already excluded
        and not c.startswith("y_")
    ]
    return cols


def load() -> pd.DataFrame:
    """Read both stores, inner-join 1:1 on (fixture_id, player_id), parse date.

    Target-only columns are taken from the targets store; everything shared
    (season/round/date/team/...) comes from the feature store to avoid suffix
    collisions.  Asserts 2,760 rows with no orphans.
    """
    feat = pd.read_csv(DATA / "model_player_match.csv")
    tgt = pd.read_csv(DATA / "model_targets.csv")

    keys = ["fixture_id", "player_id"]
    tgt_only = keys + [c for c in tgt.columns if c not in feat.columns]
    df = feat.merge(tgt[tgt_only], on=keys, how="inner", validate="one_to_one")

    if len(df) != len(feat) or len(df) != len(tgt):
        raise AssertionError(
            f"join not 1:1: feat={len(feat)} tgt={len(tgt)} joined={len(df)}")
    if len(df) != 2760:
        raise AssertionError(f"expected 2760 rows, got {len(df)}")

    df["date"] = pd.to_datetime(df["date"])
    df["bio_position"] = df["bio_position"].astype("category")
    return df


# Resolved once from the store header so callers can `from data import FEATURE_COLS`.
FEATURE_COLS: list[str] = _feature_cols(
    list(pd.read_csv(DATA / "model_player_match.csv", nrows=0).columns)
)


def _assert_no_leakage(cols: list[str]) -> None:
    bad = [c for c in cols if c in LEAKAGE_DENYLIST or c.startswith("y_")]
    if bad:
        raise AssertionError(f"leakage columns in feature set: {bad}")


def feature_view(df: pd.DataFrame, mode: str) -> list[str]:
    """Return the column list to use as model inputs for a given mode.

    mode == "pre_team_sheet":  roster priors / form only — NO realised lineup
                               (`started`, `jersey`, realised `minutes`).
    mode == "post_team_sheet": adds the PIT-known team-sheet fields
                               (`started`, `jersey`).  Realised `minutes` is
                               still excluded (it is a target).
    Both views append the cold-start FLAG_COLS.
    """
    if mode not in ("pre_team_sheet", "post_team_sheet"):
        raise ValueError(f"unknown mode {mode!r}")
    cols = list(FEATURE_COLS)
    teamplay_mode = TEAMPLAY_MODE
    if teamplay_mode == "off" and TEAMPLAY_ENABLED:
        teamplay_mode = "raw"
    if teamplay_mode != "off":
        teamplay_cols = [c for c in cols if c.startswith("teamplay_")]
        if not teamplay_cols:
            raise AssertionError(
                "TEAMPLAY_ENABLED but no teamplay_* columns found; run build_team_play.py "
                "then build_features.py before testing team-play candidates"
            )
        if teamplay_mode == "aspects":
            cols = [
                c for c in cols
                if not c.startswith("teamplay_") or c.startswith("teamplay_aspect_")
            ]
        elif teamplay_mode != "raw":
            raise ValueError(f"unknown TEAMPLAY_MODE {TEAMPLAY_MODE!r}")
    else:
        cols = [c for c in cols if not c.startswith("teamplay_")]
    optional_families = [
        ("MARKET_FEATURES_ENABLED", MARKET_FEATURES_ENABLED, "market_"),
        ("WEATHER_FEATURES_ENABLED", WEATHER_FEATURES_ENABLED, "weather_"),
        ("ROLECERT_FEATURES_ENABLED", ROLECERT_FEATURES_ENABLED, "rolecert_"),
        ("MATCHUP_FEATURES_ENABLED", MATCHUP_FEATURES_ENABLED, "matchup_"),
    ]
    for label, enabled, prefix in optional_families:
        fam_cols = [c for c in cols if c.startswith(prefix)]
        if not enabled or not fam_cols:
            cols = [c for c in cols if not c.startswith(prefix)]
    style_mode = STYLE_FEATURES_MODE
    if style_mode == "off" and STYLE_FEATURES_ENABLED:
        style_mode = "raw"
    style_cols = [c for c in cols if c.startswith("style_")]
    if style_mode == "off" or not style_cols:
        cols = [c for c in cols if not c.startswith("style_")]
    elif style_mode == "aspects":
        cols = [
            c for c in cols
            if not c.startswith("style_") or c.startswith("style_aspect_")
        ]
    elif style_mode != "raw":
        raise ValueError(f"unknown STYLE_FEATURES_MODE {STYLE_FEATURES_MODE!r}")
    cols = cols + list(FLAG_COLS)
    if mode == "post_team_sheet":
        cols = cols + [c for c in LINEUP_COLS if c in df.columns]
    else:  # pre_team_sheet leakage guard: realised lineup must be absent
        for c in LINEUP_COLS + REALISED_LINEUP:
            if c in cols:
                raise AssertionError(f"{c} leaked into pre_team_sheet view")
    _assert_no_leakage(cols)
    return cols


# Leakage guard runs at import time so any regression fails fast.
_assert_no_leakage(FEATURE_COLS)
assert len(FEATURE_COLS) >= 91, f"expected at least 91 feature cols, got {len(FEATURE_COLS)}"


def _selfcheck() -> None:
    df = load()
    print(f"load() rows={len(df)}  cols={df.shape[1]}")
    print(f"FEATURE_COLS={len(FEATURE_COLS)}  FLAG_COLS={FLAG_COLS}")
    fams = {}
    for c in FEATURE_COLS:
        fam = ("TEAMPLAY" if c.startswith("teamplay_")
               else "MARKET" if c.startswith("market_")
               else "WEATHER" if c.startswith("weather_")
               else "ROLECERT" if c.startswith("rolecert_")
               else "STYLE" if c.startswith("style_")
               else "FIXTURE" if c.startswith(("opp_", "h2h_", "team_wr", "wr_"))
               else c.split("_")[0].upper().replace("CLASS6N", "CLASS"))
        fam = "CLASS" if fam.startswith("CLASS") else fam
        fams[fam] = fams.get(fam, 0) + 1
    print("family counts:", fams)
    _assert_no_leakage(FEATURE_COLS)
    pre = feature_view(df, "pre_team_sheet")
    post = feature_view(df, "post_team_sheet")
    assert "started" not in pre and "jersey" not in pre and "minutes" not in pre
    assert "started" in post and "jersey" in post
    print(f"pre_team_sheet={len(pre)}  post_team_sheet={len(post)}  OK")


if __name__ == "__main__":
    _selfcheck()
