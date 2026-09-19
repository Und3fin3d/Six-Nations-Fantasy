import numpy as np
import pandas as pd

from model.ncr_project import match_players
from model.unified.v3 import shadow


def test_different_first_names_do_not_share_a_surname_prior():
    history = pd.DataFrame({"player_id": [226080], "player_name": ["Joaquin Oviedo"],
                            "team": ["Argentina"], "fixture_id": ["old"]})
    pool = pd.DataFrame({"id": [183329], "full_name": ["Leonel Oviedo"],
                         "team_name": ["Argentina"]})
    assert match_players(pool, history) == {}


def test_verified_projection_identity_overrides_bad_crosswalk(tmp_path, monkeypatch):
    ncr = tmp_path / "ncr"
    ncr.mkdir()
    pd.DataFrame({"fantasy_id": [207343, 252481], "api_player_id": [178674, 178674],
                  "full_name": ["Grant Williams", "Jaco Williams"]}).to_csv(ncr / "ncr_player_crosswalk.csv", index=False)
    pd.DataFrame({"gameday": [1], "home": ["South Africa"], "away": ["Italy"],
                  "match_id": ["match"], "game_date": ["2026-07-04"]}).to_csv(ncr / "ncr_fixtures.csv", index=False)
    monkeypatch.setattr(shadow, "DATA", tmp_path)
    projection = pd.DataFrame({"id": [207343, 252481], "name": ["Grant Williams", "Jaco Williams"],
                               "team": ["South Africa"] * 2, "pos": ["Scrum Half", "Back Three"],
                               "status": ["P", "B"], "history_player_id": [178674, np.nan]})
    candidates = shadow.ncr_candidates(1, projection)
    assert candidates["player_id"].tolist() == ["178674", "fantasy_252481"]
    assert candidates["player_name"].tolist() == ["Grant Williams", "Jaco Williams"]
    assert not candidates.duplicated(["fixture_id", "player_id", "team"]).any()
