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
  3. FIXTURE  — the opponent-context features (fixture_difficulty.csv),
                left-joined on (fixture_id, team_id).
  4. TEAMPLAY — explicit PIT team edge / game-script predictions
                (team_play_predictions.csv), left-joined on (fixture_id, team_id).
  5. EXTERNAL — optional weather / role-certainty files. These are
                joined only when present and are off by default in model/data.py.
  6. BIO      — age-at-fixture / height / weight / position (rp_bio.csv).
  7. ROLE     — goal-kicker rate, starter prob, set-piece (lineout) involvement,
                all decayed & PIT.
  8. OWN-TEAM — own side's decayed attacking strength (api_team_match.csv) plus
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
MATCHUP_STATS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals",
    "metres", "defenders_beaten", "offload", "tackles",
    "tackle_turnover", "penalties_conceded",
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

OPTIONAL_EXTERNAL_FILES = {
    "weather": DATA / "external_fixture_weather.csv",
    "rolecert": DATA / "external_player_roles.csv",
    "style": DATA / "external_team_style.csv",
}

FIXTURE_JOIN_KEYS = [
    ["fixture_id", "team_id"],
    ["fixture_id", "team"],
    ["season", "round", "team_id"],
    ["season", "round", "team"],
]

WEATHER_JOIN_KEYS = [
    ["fixture_id"],
    ["season", "round"],
]

ROLECERT_JOIN_KEYS = [
    ["fixture_id", "player_id"],
    ["fixture_id", "team_id", "_player_key"],
    ["fixture_id", "team", "_player_key"],
    ["season", "round", "player_id"],
    ["season", "round", "team_id", "_player_key"],
    ["season", "round", "team", "_player_key"],
]


def _decay_weights(asof: pd.Timestamp, dates: pd.Series, hl: float) -> np.ndarray:
    days = (asof - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, days / hl)


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def _numeric_prefixed(raw: pd.DataFrame, prefix: str) -> list[str]:
    cols: list[str] = []
    for col in raw.columns:
        if not col.startswith(prefix):
            continue
        val = pd.to_numeric(raw[col], errors="coerce")
        if val.notna().any():
            raw[col] = val
            cols.append(col)
    return cols


def _join_keys(raw: pd.DataFrame, feat: pd.DataFrame, candidates: list[list[str]]) -> list[str]:
    raw_cols = set(raw.columns)
    feat_cols = set(feat.columns)
    for keys in candidates:
        if set(keys).issubset(raw_cols) and set(keys).issubset(feat_cols):
            return keys
    raise ValueError(
        "external file has no supported join key; expected one of "
        + ", ".join("[" + ", ".join(k) + "]" for k in candidates)
    )


def _dedupe_external(raw: pd.DataFrame, keys: list[str], source: str) -> pd.DataFrame:
    dupes = raw.duplicated(keys, keep=False)
    if dupes.any():
        examples = raw.loc[dupes, keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{source} has duplicate rows for keys {keys}: {examples}")
    return raw


def _merge_external(
    feat: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    source: str,
    prefix: str,
    join_candidates: list[list[str]],
    timestamp_col: str | None = None,
) -> pd.DataFrame:
    if raw.empty:
        return feat
    raw = raw.copy()
    feature_cols = _numeric_prefixed(raw, prefix)
    if not feature_cols:
        return feat

    ts_col = None
    if timestamp_col and timestamp_col in raw.columns:
        ts_col = f"_{prefix.rstrip('_')}_timestamp"
        raw[ts_col] = pd.to_datetime(raw[timestamp_col], errors="coerce", utc=True)

    keys = _join_keys(raw, feat, join_candidates)
    keep = keys + feature_cols + ([ts_col] if ts_col else [])
    raw = _dedupe_external(raw[keep], keys, source)
    out = feat.merge(raw, on=keys, how="left", validate="many_to_one")

    has_col = f"{prefix}has_data"
    out[has_col] = out[feature_cols].notna().any(axis=1).astype(float)

    if ts_col:
        fixture_date = pd.to_datetime(out["date"], errors="coerce", utc=True)
        days = (fixture_date - out[ts_col]).dt.total_seconds() / 86400.0
        fresh_col = f"{prefix}days_before_fixture"
        out[fresh_col] = days.where(days >= 0)
        bad = days < 0
        if bad.any():
            out.loc[bad, feature_cols] = np.nan
            out.loc[bad, has_col] = 0.0
        out = out.drop(columns=[ts_col])
    return out



def _merge_external_weather(feat: pd.DataFrame) -> pd.DataFrame:
    path = OPTIONAL_EXTERNAL_FILES["weather"]
    if not path.exists():
        return feat
    raw = pd.read_csv(path)
    if {"weather_rain_mm", "weather_precip_probability"}.issubset(raw.columns):
        raw["weather_wet_index"] = (
            np.log1p(pd.to_numeric(raw["weather_rain_mm"], errors="coerce"))
            + pd.to_numeric(raw["weather_precip_probability"], errors="coerce")
        )
    elif "weather_rain_mm" in raw.columns:
        raw["weather_wet_index"] = np.log1p(
            pd.to_numeric(raw["weather_rain_mm"], errors="coerce")
        )
    if {"weather_wind_kph", "weather_wind_gust_kph"}.issubset(raw.columns):
        raw["weather_wind_index"] = (
            pd.to_numeric(raw["weather_wind_kph"], errors="coerce")
            + 0.5 * pd.to_numeric(raw["weather_wind_gust_kph"], errors="coerce")
        ) / 40.0
    elif "weather_wind_kph" in raw.columns:
        raw["weather_wind_index"] = (
            pd.to_numeric(raw["weather_wind_kph"], errors="coerce") / 30.0
        )
    if "weather_temp_c" in raw.columns:
        raw["weather_cold_index"] = np.clip(
            (10.0 - pd.to_numeric(raw["weather_temp_c"], errors="coerce")) / 10.0,
            0.0,
            None,
        )
    return _merge_external(
        feat, raw, source=str(path), prefix="weather_", join_candidates=WEATHER_JOIN_KEYS,
        timestamp_col=None,
    )


def _merge_external_roles(feat: pd.DataFrame) -> pd.DataFrame:
    path = OPTIONAL_EXTERNAL_FILES["rolecert"]
    if not path.exists():
        return feat
    raw = pd.read_csv(path)
    feat_in = feat.copy()
    raw = raw.copy()
    if "player_key" in raw.columns:
        raw["_player_key"] = raw["player_key"].map(norm_key)
        feat_in["_player_key"] = feat_in["player_name"].map(norm_key)
    elif "player_name" in raw.columns:
        raw["_player_key"] = raw["player_name"].map(norm_key)
        feat_in["_player_key"] = feat_in["player_name"].map(norm_key)
    out = _merge_external(
        feat_in, raw, source=str(path), prefix="rolecert_",
        join_candidates=ROLECERT_JOIN_KEYS, timestamp_col="rolecert_source_timestamp",
    )
    if "_player_key" in out.columns:
        out = out.drop(columns=["_player_key"])
    return out


def _merge_external_style(feat: pd.DataFrame) -> pd.DataFrame:
    path = OPTIONAL_EXTERNAL_FILES["style"]
    if not path.exists():
        return feat
    raw = pd.read_csv(path)
    return _merge_external(
        feat, raw, source=str(path), prefix="style_", join_candidates=FIXTURE_JOIN_KEYS,
        timestamp_col=None,
    )


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
        # Kept under a separate prefix so these replacement-history signals
        # are available only to the opt-in bench specialist.
        "benchhist_n_prior": 0,
        "benchhist_minutes_recent": np.nan,
        "benchhist_play10_rate": np.nan,
        "benchhist_high30_rate": np.nan,
        "benchhist_days_since_last": np.nan,
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

    bench = hist.loc[~_bool(hist["started"])].copy()
    if not bench.empty:
        bench_w = _decay_weights(asof, bench["date"], hl)
        bench_sw = bench_w.sum()
        bench_minutes = bench["minutes"].to_numpy(dtype=float)
        out["benchhist_n_prior"] = int(len(bench))
        out["benchhist_days_since_last"] = float((asof - bench["date"].max()).days)
        if bench_sw > 0:
            out["benchhist_minutes_recent"] = float(
                (bench_w * bench_minutes).sum() / bench_sw
            )
            out["benchhist_play10_rate"] = float(
                (bench_w * (bench_minutes >= 10.0)).sum() / bench_sw
            )
            out["benchhist_high30_rate"] = float(
                (bench_w * (bench_minutes >= 30.0)).sum() / bench_sw
            )

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


def matchup_features(
    hist: pd.DataFrame,
    asof: pd.Timestamp,
    canonical_pos: str | None,
    is_forward: bool,
    hl: float,
) -> dict:
    """PIT opponent allowance for the current player's role.

    Exact-position history is preferred. When fewer than eight prior player
    appearances exist, the larger forward/back role supplies a stable fallback.
    A one-year minimum half-life preserves signal across annual tournaments.
    """
    out = {f"matchup_per80_{s}": np.nan for s in MATCHUP_STATS}
    out.update({
        "matchup_n_prior": 0,
        "matchup_fixture_n_prior": 0,
        "matchup_minutes_prior": 0.0,
        "matchup_exact_position": 0.0,
    })
    if hist.empty:
        return out
    exact = hist[hist["_matchup_pos"] == canonical_pos]
    if len(exact) >= 8:
        use = exact
        out["matchup_exact_position"] = 1.0
    else:
        use = hist[hist["_matchup_is_forward"] == bool(is_forward)]
    if use.empty:
        return out
    w = _decay_weights(asof, use["date"], max(float(hl), 365.0))
    mins = pd.to_numeric(use["minutes"], errors="coerce").fillna(0.0).to_numpy(float)
    weighted_minutes = w * mins
    denom = weighted_minutes.sum()
    out["matchup_n_prior"] = int(len(use))
    out["matchup_fixture_n_prior"] = int(use["fixture_id"].nunique())
    out["matchup_minutes_prior"] = float(denom)
    if denom > 0:
        for stat in MATCHUP_STATS:
            values = pd.to_numeric(use[stat], errors="coerce").fillna(0.0).to_numpy(float)
            out[f"matchup_per80_{stat}"] = float((w * values).sum() / denom * 80.0)
    return out


def team_bench_slot_features(
    hist: pd.DataFrame,
    asof: pd.Timestamp,
    jersey: float | int | None,
    hl: float,
) -> dict:
    """PIT national-team substitution tendency for the named bench slot."""
    out = {
        "benchteam_slot_n_prior": 0,
        "benchteam_slot_minutes_recent": np.nan,
        "benchteam_slot_play10_rate": np.nan,
        "benchteam_slot_high30_rate": np.nan,
        "benchteam_slot_days_since_last": np.nan,
    }
    slot = pd.to_numeric(pd.Series([jersey]), errors="coerce").iloc[0]
    if hist.empty or pd.isna(slot):
        return out
    bench = hist.loc[
        (~_bool(hist["started"]))
        & pd.to_numeric(hist["jersey"], errors="coerce").eq(float(slot))
    ]
    if bench.empty:
        return out
    weights = _decay_weights(asof, bench["date"], max(float(hl), 365.0))
    sw = weights.sum()
    minutes = pd.to_numeric(
        bench["minutes"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    out["benchteam_slot_n_prior"] = int(len(bench))
    out["benchteam_slot_days_since_last"] = float(
        (asof - bench["date"].max()).days
    )
    if sw > 0:
        out["benchteam_slot_minutes_recent"] = float(
            (weights * minutes).sum() / sw
        )
        out["benchteam_slot_play10_rate"] = float(
            (weights * (minutes >= 10.0)).sum() / sw
        )
        out["benchteam_slot_high30_rate"] = float(
            (weights * (minutes >= 30.0)).sum() / sw
        )
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

    ap["_matchup_pos"] = ap["player_id"].map(canon_pos)
    ap["_matchup_is_forward"] = ap["_matchup_pos"].isin(FORWARD_GROUPS)

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
    player_hist_by_team = {
        tid: g.sort_values("date") for tid, g in ap.groupby("team_id")
    }
    hist_by_opponent = {
        tid: g.sort_values("date") for tid, g in ap.groupby("opponent_id")
    }

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
        pth = player_hist_by_team.get(int(r.team_id))
        pth = pth[pth["date"] < asof] if pth is not None else ap.iloc[0:0]
        rec.update(team_bench_slot_features(
            pth, asof, getattr(r, "jersey", None), half_life
        ))

        # hot/cold = recent FORM vs lifetime CLASS
        if pd.notna(rec.get("form_per80_metres")) and pd.notna(rec.get("class_per80_metres")):
            rec["form_hot_metres"] = rec["form_per80_metres"] - rec["class_per80_metres"]
        if pd.notna(rec.get("form_per80_tries")) and pd.notna(rec.get("class_per80_tries")):
            rec["form_hot_tries"] = rec["form_per80_tries"] - rec["class_per80_tries"]

        # OWN-TEAM (own side, strictly date < fixture)
        th = hist_by_team.get(int(r.team_id))
        th = th[th["date"] < asof] if th is not None else tm.iloc[0:0]
        rec.update(ownteam_features(th, asof, half_life))

        # POSITION-SPECIFIC MATCHUP (what this opponent allowed to the current
        # role in prior matches, never including the fixture being predicted).
        oh = hist_by_opponent.get(int(r.opponent_id))
        oh = oh[oh["date"] < asof] if oh is not None else ap.iloc[0:0]
        rec.update(matchup_features(
            oh,
            asof,
            rec["canonical_pos"],
            bool(rec["is_forward"]),
            half_life,
        ))

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

    # FIXTURE (opponent-context features), left-join on (fixture_id, team_id)
    feat = feat.merge(fd[fd_feats], on=["fixture_id", "team_id"], how="left")

    # TEAMPLAY (explicit predicted match-shape layer), left-join on (fixture_id, team_id)
    tp_path = DATA / "team_play_predictions.csv"
    if tp_path.exists():
        tp = pd.read_csv(tp_path)
        tp_feats = [
            c for c in tp.columns
            if c not in {"season", "round", "team", "opponent", "opponent_id", "date"}
        ]
        feat = feat.merge(tp[tp_feats], on=["fixture_id", "team_id"], how="left",
                          validate="many_to_one")
    feat = _merge_external_weather(feat)
    feat = _merge_external_roles(feat)
    feat = _merge_external_style(feat)
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
        "ROLE": [c for c in feat if c.startswith("role") and not c.startswith("rolecert_")],
        "OWN-TEAM": [c for c in feat if c.startswith("ownteam")],
        "BIO": [c for c in feat if c.startswith("bio")],
        "FIXTURE": [c for c in feat if c.startswith(("opp_", "h2h_", "team_wr", "wr_"))],
        "TEAMPLAY": [c for c in feat if c.startswith("teamplay_")],
        "WEATHER": [c for c in feat if c.startswith("weather_")],
        "ROLECERT": [c for c in feat if c.startswith("rolecert_")],
        "STYLE": [c for c in feat if c.startswith("style_")],
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
