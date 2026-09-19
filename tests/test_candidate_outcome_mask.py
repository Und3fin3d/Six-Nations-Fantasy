import pandas as pd

from model.unified.raw_benchmark.folds import masked_candidates


def test_auxiliary_outcomes_are_masked_without_mutating_input():
    source = pd.DataFrame({"player_id": ["p"], "fixture_id": ["f"], "team": ["A"],
                           "started": [True], "date": ["2025-01-01"],
                           "auxiliary_event": [7], "available__auxiliary_event": [True],
                           "team_score": [40], "opp_score": [3]})
    result = masked_candidates(source)
    assert not result.filter(like="available__").any().any()
    assert result[["auxiliary_event", "team_score", "opp_score"]].isna().all().all()
    assert result["started"].tolist() == [True]
    assert source["auxiliary_event"].tolist() == [7]
