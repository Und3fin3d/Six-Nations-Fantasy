"""Complete friendly teamsheets, including non-fantasy-country opponents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .raw_benchmark.coverage import parse_cache_fixture
from .schema import EVENTS, POSITION_BY_JERSEY, FORWARD_POSITIONS

KEY = ["fixture_id", "player_id", "team"]


def complete_fixture_cohort(store: pd.DataFrame, fixture: str, cache: Path) -> pd.DataFrame:
    """Use every cached teamsheet player, not only the canonical tracked side.

    Training history is unchanged. Previously untracked opponents are genuine
    cold-start evaluation cases, with any available club history still usable.
    Added positions use pre-match jersey-slot inference, never event outcomes.
    """
    fixture = str(fixture)
    known = store[store["fixture_id"].astype(str).eq(fixture)].copy()
    if known.empty or known.duplicated(KEY).any():
        raise ValueError("friendly fixture needs a unique canonical anchor")
    path = cache / f"match_{fixture}.json"
    match = json.loads(path.read_text())["results"]["match"]
    _, players, _ = parse_cache_fixture(path)
    expected_teams = {str(match["home_team"]), str(match["away_team"])}
    by_key = {tuple(str(row[k]) for k in KEY): row for row in known.to_dict("records")}
    output = []
    for player in players:
        key = tuple(str(player[k]) for k in KEY)
        row = dict(by_key.get(key, known.iloc[0].to_dict()))
        team = str(player["team"])
        jersey = int(player["jersey_cache"])
        position = row["position"] if key in by_key else POSITION_BY_JERSEY.get(jersey)
        if position is None:
            raise ValueError(f"cannot infer pre-match position for {key}")
        row.update({
            "fixture_id": fixture, "player_id": str(player["player_id"]),
            "player_name": str(player["player_name_cache"]), "team": team,
            "opponent": str(player["opponent_cache"]), "jersey": jersey,
            "started": bool(player["started_cache"]), "position": position,
            "is_forward": position in FORWARD_POSITIONS,
            "home_away": "home" if team == str(match["home_team"]) else "away",
            "competition_level": "international", "competition_id_cache": 30,
            "match_at": pd.Timestamp(player["match_at"]),
            "date": pd.Timestamp(player["match_at"]).tz_convert(None).normalize(),
            "canonical_player": key in by_key,
        })
        # Cache semantics preserve missing labels; never inherit another
        # player's events from the metadata-only anchor row.
        for target in ("minutes", *EVENTS):
            available = bool(player.get(f"available__{target}_cache", False))
            row[target] = player.get(f"{target}_cache", np.nan) if available else np.nan
            row[f"available__{target}"] = available
        output.append(row)
    result = pd.DataFrame(output)
    if result.empty or result.duplicated(KEY).any() or set(result["team"]) != expected_teams:
        raise ValueError("friendly cache does not contain both complete keyed teamsheets")
    if not set(by_key).issubset(set(result[KEY].itertuples(index=False, name=None))):
        raise ValueError("canonical players are missing from the cached teamsheet")
    return result.sort_values(KEY).reset_index(drop=True)
