"""Offline tests for the fantasy-source probes (research/fantasy_sources)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research" / "fantasy_sources"
SAMPLES = SRC / "samples"
sys.path.insert(0, str(SRC))

import fantasy_common as common  # noqa: E402
import pfr  # noqa: E402
import sixn_platform  # noqa: E402


def test_name_keys():
    assert common.name_key("Thomás  O'Ramos-Smith") == "thomas oramos smith"
    assert common.initial_key("Thomas Ramos") == common.initial_key("T. Ramos") == "t ramos"
    assert common.surname_key("Antoine Dupont") == "dupont"


def test_pfr_tidy_real_rwc2023_excerpt():
    players = json.loads((SAMPLES / "pfr_rwc2023_players_FRA_NZL.json").read_text())
    squads = json.loads((SAMPLES / "pfr_rwc2023_squads.json").read_text())
    df = pfr.tidy_players(players, squads)
    assert set(df["team"]) == {"France", "New Zealand"}
    assert df["round"].between(1, 8).all()
    scored = df[df["points"].notna()]
    assert len(scored) > 150
    assert scored.groupby(["pfr_player_id", "round"]).size().max() == 1
    # ownership is recorded for every round the player was listed
    assert df["selected_pct"].notna().mean() > 0.95


def test_pfr_join_store_by_round_windows():
    tidy = pd.DataFrame({
        "pfr_player_id": [1, 1, 2], "name": ["Antoine Dupont", "Antoine Dupont", "Will Jordan"],
        "team": ["France", "France", "New Zealand"], "round": [1, 2, 1], "points": [40, 12, 55],
    })
    store = pd.DataFrame({
        "date": ["2023-09-08", "2023-09-08", "2023-09-21"], "team": ["France", "New Zealand", "France"],
        "player_id": [10, 20, 10], "player_name": ["A. Dupont", "Will Jordan", "A. Dupont"],
        "fixture_id": [100, 100, 101], "minutes": [80, 80, 70], "tries": [0, 1, 0], "tackles": [5, 2, 6],
    })
    merged, stats = pfr.join_store(tidy, store, pfr.WINDOWS["rwc2023"])
    assert stats["players_matched"] == 2
    # Dupont round 2 window (14-18 Sep) has no store row; the other two join
    assert stats["scored_rounds_joined_to_store_row"] == 2
    assert stats["store_row_coverage"] == round(2 / 3, 4)


def test_sixn_platform_tidy_detail_real_2026_excerpt():
    players = json.loads((SAMPLES / "sixn2026_statsjoueur_detail_4players.json").read_text())
    df = sixn_platform.tidy_detail(players)
    assert len(df) == 4 * 5
    ramos = df[(df["short_name"] == "T. Ramos") & (df["round"] == 5)].iloc[0]
    assert ramos["points"] == 30 and ramos["minutes"] == 80 and bool(ramos["starter"])
    assert ramos["price_before"] == 19.1 and ramos["price_after"] == 19.2
    for col in ("breakdown_steals", "potm", "metres_carried", "lineout_steals", "kick_50_22"):
        assert col in df.columns


def test_sixn_platform_compare_official_identity():
    players = json.loads((SAMPLES / "sixn2026_statsjoueur_detail_4players.json").read_text())
    tidy = sixn_platform.tidy_detail(players)
    played = tidy[tidy["played"]]
    official = pd.DataFrame({
        "season": 2026, "round": played["round"], "team": played["team"], "name": played["short_name"],
        "Pts": played["points"], "Min": played["minutes"], "Ta": played["tackles"],
        "MC": played["metres_carried"], "BS": played["breakdown_steals"], "POTM": played["potm"],
        "T": played["tries"],
    })
    res = sixn_platform.compare_official(tidy, official, 2026)
    assert res["join_rate_platform"] == 1.0
    assert res["exact_points"] == 1.0 and res["exact_breakdown_steals"] == 1.0
