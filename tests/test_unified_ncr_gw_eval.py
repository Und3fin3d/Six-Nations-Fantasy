from __future__ import annotations

import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.ncr_gw_eval import _team_contributions, expected_ncr_points


def test_expected_ncr_points_uses_linear_adapter_and_omits_missing_events() -> None:
    prediction = RawPrediction(
        fixture_id="1", player_id="2", player_name="Player", team="A",
        opponent="B", position="Prop", is_forward=True,
        events={
            "tries": EventDistribution("negative_binomial", 0.5, 1.0),
            "tackles": EventDistribution("negative_binomial", 8.0, 1.0),
            "scrums_won": EventDistribution("negative_binomial", 2.0, 1.0),
            "penalties_conceded": EventDistribution("negative_binomial", 1.0, 1.0),
        },
        minutes=EventDistribution("lognormal", 60.0, 10.0),
    )
    assert expected_ncr_points(prediction) == pytest.approx(17.0)


def test_team_contributions_apply_captain_and_bench_super_sub() -> None:
    import pandas as pd

    squad = pd.DataFrame([
        {"id": 1, "name": "Captain", "team": "A", "is_capt": True, "is_sub": False},
        {"id": 2, "name": "Sub", "team": "B", "is_capt": False, "is_sub": True},
    ])
    actuals = {"1": (10.0, 1.0, "P"), "2": (7.0, 1.0, "B")}
    result = _team_contributions(squad, actuals)
    assert result["contribution"].tolist() == [20.0, 21.0]
