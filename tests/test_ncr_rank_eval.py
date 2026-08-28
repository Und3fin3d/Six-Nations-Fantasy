import pandas as pd

from model.ncr_rank_eval import evaluate_round, tie_aware_hits, ROUNDS


def test_tie_aware_hits_fractionally_weights_boundary_tie():
    actual = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "actual": [10, 8, 8, 8],
    })

    hits, cutoff = tie_aware_hits({1, 2}, actual, n=2)

    assert cutoff == 8
    assert hits == 1 + 1 / 3


def test_saved_gw2_rank_comparison_uses_one_common_cohort():
    results, quality, _ = evaluate_round(2, ROUNDS[2])
    top25 = results[results["n"].eq(25)].set_index("model")

    assert quality["common_players"] == 262
    assert top25.loc["NCR", "tie_adjusted_hits"] == 9
    assert top25.loc["Champion", "tie_adjusted_hits"] == 6
    assert top25.loc["NCR", "points_captured"] > top25.loc["Champion", "points_captured"]
