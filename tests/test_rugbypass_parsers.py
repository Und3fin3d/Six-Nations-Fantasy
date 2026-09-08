from bs4 import BeautifulSoup

from rugbypass_batch import extract_comp_stats, extract_match_log


def test_extract_comp_stats_accepts_null_payload():
    soup = BeautifulSoup('<div id="app-comp-stats">null</div>', "html.parser")

    assert extract_comp_stats(soup) == []


def test_extract_match_log_accepts_null_payload():
    soup = BeautifulSoup('<div id="app-competitions">null</div>', "html.parser")

    assert extract_match_log(soup) == []
