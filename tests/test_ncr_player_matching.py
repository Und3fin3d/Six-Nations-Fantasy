import pandas as pd

from model.ncr_project import match_players


def _history(*names):
    return pd.DataFrame(
        [
            {
                "player_id": player_id,
                "player_name": name,
                "team": team,
                "fixture_id": player_id,
            }
            for player_id, name, team in names
        ]
    )


def test_matches_feed_aliases_and_apostrophe_variants():
    pool = pd.DataFrame(
        [
            {"id": 1, "full_name": "Tom OToole", "team_name": "Ireland"},
            {"id": 2, "full_name": "Samisoni Taukeiaho", "team_name": "New Zealand"},
            {"id": 3, "full_name": "Nacho Brex", "team_name": "Italy"},
            {"id": 4, "full_name": "Peni Ravai Kovekalou", "team_name": "Fiji"},
        ]
    )
    hist = _history(
        (10, "Tom O'Toole", "Ireland"),
        (20, "Samisoni Taukei'aho", "New Zealand"),
        (30, "Juan Ignacio Brex", "Italy"),
        (40, "Peni Ravai", "Fiji"),
    )

    assert match_players(pool, hist) == {1: 10, 2: 20, 3: 30, 4: 40}


def test_does_not_match_unique_surname_with_different_initial():
    pool = pd.DataFrame([{"id": 1, "full_name": "Kane James", "team_name": "Wales"}])
    hist = _history((10, "Eddie James", "Wales"))

    assert match_players(pool, hist) == {}
