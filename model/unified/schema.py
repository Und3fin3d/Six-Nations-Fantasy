"""Canonical player-match schema and target taxonomy."""

from __future__ import annotations

COUNT_EVENTS = (
    "tries", "try_assists", "conversion_goals", "missed_conversion_goals",
    "penalty_goals", "missed_penalty_goals", "drop_goals_converted",
    "drop_goal_missed", "defenders_beaten", "clean_breaks", "offload",
    "runs", "tackles", "missed_tackles", "dominant_tackles",
    "tackle_turnover", "tackle_try_saver", "rucks_won", "rucks_lost",
    "passes", "bad_passes", "lineouts_won", "turnovers_conceded",
    "penalties_conceded", "yellow_cards", "red_cards",
    "fifty_22", "lineout_steals", "scrums_won", "kicks_retained", "potm",
)
CONTINUOUS_EVENTS = ("metres",)
EVENTS = COUNT_EVENTS + CONTINUOUS_EVENTS

SCORING_EVENTS = (
    "tries", "try_assists", "conversion_goals", "missed_conversion_goals",
    "penalty_goals", "missed_penalty_goals", "drop_goals_converted",
    "defenders_beaten", "clean_breaks", "offload", "tackles",
    "missed_tackles", "tackle_turnover", "turnovers_conceded",
    "penalties_conceded", "yellow_cards", "red_cards", "metres",
    "fifty_22", "lineout_steals", "scrums_won", "kicks_retained", "potm",
)

KEY_COLUMNS = ("fixture_id", "player_id", "team")
META_COLUMNS = (
    "date", "competition", "competition_id", "competition_level", "season",
    "round", "fixture_id", "player_id", "player_name", "team", "team_id",
    "opponent", "opponent_id", "position", "is_forward", "jersey",
    "started", "minutes", "source", "source_priority",
)

POSITION_BY_JERSEY = {
    1: "Prop", 2: "Hooker", 3: "Prop", 4: "Second-row", 5: "Second-row",
    6: "Back-row", 7: "Back-row", 8: "Back-row", 9: "Scrum-half",
    10: "Fly-half", 11: "Back-three", 12: "Centre", 13: "Centre",
    14: "Back-three", 15: "Back-three", 16: "Hooker", 17: "Prop", 18: "Prop",
    19: "Second-row", 20: "Back-row", 21: "Scrum-half", 22: "Fly-half",
    23: "Back-three",
}
FORWARD_POSITIONS = {"Prop", "Hooker", "Second-row", "Back-row"}


def distribution_family(event: str) -> str:
    if event in {"yellow_cards", "red_cards", "potm"}:
        return "bernoulli"
    if event in CONTINUOUS_EVENTS or event == "minutes":
        return "lognormal"
    return "negative_binomial"
