from model import ncr_score_gw as scorer


def test_gw2_official_feed_scores_all_three_saved_teams():
    actuals = scorer.load_official_actuals(2)
    totals = {}
    details = {}
    for label, template in scorer.TEAM_FILES.items():
        details[label], totals[label] = scorer.score_squad_official(
            scorer.DATA / template.format(gw=2), actuals
        )

    assert totals == {
        "Bayes / NCR empirical": 560.0,
        "6N champion": 624.0,
        "Blend": 589.0,
    }
    pollock = details["6N champion"].set_index("Player").loc["Henry Pollock"]
    assert pollock["Base"] == 65
    assert pollock["Multiplier"] == 3
    assert pollock["Contribution"] == 195
