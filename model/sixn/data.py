from __future__ import annotations

import pandas as pd

TEAMPLAY_MODE = "off"
MATCHUP_FEATURES_ENABLED = False
FLAG_COLS = ["has_class", "has_form", "has_bio"]
CATEGORICAL_COLS = ["bio_position"]
FEATURE_PREFIXES = ("class", "form", "role", "ownteam", "bio", "opp_", "h2h_", "team_wr", "wr_", "teamplay_", "weather_", "rolecert_", "style_", "matchup_")
FEATURE_COLS = [c for c in pd.read_csv("data/model_player_match.csv", nrows=0).columns if c.startswith(FEATURE_PREFIXES) and c not in FLAG_COLS]


def load() -> pd.DataFrame:
    feat, tgt = pd.read_csv("data/model_player_match.csv"), pd.read_csv("data/model_targets.csv")
    keys = ["fixture_id", "player_id"]
    df = feat.merge(tgt[keys + [c for c in tgt.columns if c not in feat.columns]], on=keys, how="inner")
    if len(df) != 2760:
        raise AssertionError(f"expected 2760 rows, got {len(df)}")
    df["date"] = pd.to_datetime(df["date"])
    df["bio_position"] = df["bio_position"].astype("category")
    return df


def _keep(c: str) -> bool:
    if c.startswith("teamplay_"):
        return TEAMPLAY_MODE == "aspects" and c.startswith("teamplay_aspect_")
    if c.startswith("matchup_"):
        return MATCHUP_FEATURES_ENABLED
    return not c.startswith(("weather_", "rolecert_", "style_"))


def feature_view(df: pd.DataFrame, mode: str) -> list[str]:
    cols = [c for c in FEATURE_COLS if _keep(c)] + FLAG_COLS
    return cols + [c for c in ("started", "jersey") if c in df.columns] if mode == "post_team_sheet" else cols
