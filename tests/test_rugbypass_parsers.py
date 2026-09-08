from bs4 import BeautifulSoup

from compare_three_way import slug_key
from rugbypass_batch import extract_comp_stats, extract_match_log
import rugbypass_backfill as rb


def test_hame_faiva_uses_current_rugbypass_slug():
    assert rb.SLUG_OVERRIDES["Hame Faiva"] == "epalahame-faiva"
    assert slug_key("epalahame-faiva") == "h|faiva"


def test_oli_kebble_uses_current_rugbypass_slug():
    assert rb.SLUG_OVERRIDES["Oli Kebble"] == "oliver-kebble"
    assert slug_key("oliver-kebble") == "o|kebble"


def test_noah_nene_uses_current_rugbypass_slug():
    assert rb.SLUG_OVERRIDES["Noah Nene"] == "noah-tisie-nene"
    assert "Noah Nene" not in rb.KNOWN_ABSENT
    assert slug_key("noah-tisie-nene") == "n|nene"


def test_extract_comp_stats_accepts_null_payload():
    soup = BeautifulSoup('<div id="app-comp-stats">null</div>', "html.parser")

    assert extract_comp_stats(soup) == []


def test_extract_match_log_accepts_null_payload():
    soup = BeautifulSoup('<div id="app-competitions">null</div>', "html.parser")

    assert extract_match_log(soup) == []


def test_full_refresh_retains_nonempty_sections_when_page_returns_null(tmp_path, monkeypatch):
    players = tmp_path / "players"
    players.mkdir()
    old = {
        "player": "example-player",
        "csv_name": "Example Player",
        "bio": {"Nationality": "Example"},
        "competition_stats": [{"competition": "Example League"}],
        "match_log": [{"match": "Example A v B"}],
    }
    path = players / "rugbypass_example_player.json"
    path.write_text(__import__("json").dumps(old))
    html = """<div class='player-details'></div>
    <div id='app-comp-stats'>null</div><div id='app-competitions'>null</div>"""

    monkeypatch.setattr(rb, "OUT_DIR", players)
    monkeypatch.setattr(rb, "BASE", tmp_path)
    monkeypatch.setattr(rb, "fetch", lambda session, slug: (200, html))
    monkeypatch.setattr(rb.time, "sleep", lambda delay: None)

    manifest = rb.refresh_existing(delay=0)

    assert __import__("json").loads(path.read_text()) == old
    assert manifest["unchanged_profiles"] == 1
    assert manifest["retained_nonempty_sections"] == {
        "bio": 1,
        "competition_stats": 1,
        "match_log": 1,
    }
