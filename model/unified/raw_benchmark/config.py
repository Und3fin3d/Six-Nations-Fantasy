"""Frozen contracts for historical raw benchmark v1."""

from __future__ import annotations

from pathlib import Path

from ..data import ROOT

VERSION = "v1"
OUT = ROOT / "data" / "unified" / "raw_benchmark" / VERSION
CACHE = ROOT / "data" / "cache"
LEGACY_STORE = ROOT / "data" / "unified" / "player_match.csv"
OFFICIAL = ROOT / "data" / "official_player_match.csv"

STABLE_EVENTS = (
    "tries", "try_assists", "conversion_goals", "missed_conversion_goals",
    "penalty_goals", "missed_penalty_goals", "drop_goals_converted",
    "drop_goal_missed", "defenders_beaten", "clean_breaks", "offload",
    "runs", "tackles", "missed_tackles", "rucks_won", "rucks_lost",
    "passes", "bad_passes", "turnovers_conceded", "penalties_conceded",
    "yellow_cards", "red_cards", "metres",
)

EXTENDED_EVENTS = (
    "dominant_tackles", "tackle_turnover", "tackle_try_saver", "fifty_22",
    "lineout_steals", "scrums_won", "kicks_retained", "potm",
)

DIAGNOSTIC_ONLY_EVENTS = ("lineouts_won",)
UNOBSERVED_RUBRIC_EVENTS = ("interceptions", "handling_errors", "lineout_errors")
DIRECT_EVENTS = tuple(
    event for event in (*STABLE_EVENTS, *EXTENDED_EVENTS, *DIAGNOSTIC_ONLY_EVENTS)
    if event not in {"yellow_cards", "red_cards", "fifty_22", "lineout_steals",
                     "scrums_won", "kicks_retained", "potm"}
)
DERIVED_EVENTS = ("minutes", "yellow_cards", "red_cards")
OFFICIAL_COLUMNS = {
    "50-22": "fifty_22", "LS": "lineout_steals", "SW": "scrums_won",
    "KR": "kicks_retained", "POTM": "potm",
}

MIN_FIXTURES = 5
EVALUATION_START = "2022-01-01T00:00:00Z"
DENSE_THRESHOLD = 0.90
SEED = 17
TOP_NS = (10, 25, 50, 100)

NORTH = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}
SOUTH = {"Argentina", "Australia", "Fiji", "Japan", "New Zealand", "South Africa"}

CORE_ENGINE_ORDER = (
    "v1", "gbdt_v3", "v4", "v5_t", "empirical_event", "p3_event_50",
)
CHALLENGER_ENGINE_ORDER = ("v1_neural",)
ENGINE_ORDER = (*CORE_ENGINE_ORDER, *CHALLENGER_ENGINE_ORDER)


def output_path(*parts: str) -> Path:
    return OUT.joinpath(*parts)
