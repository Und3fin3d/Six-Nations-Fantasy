from pathlib import Path

import numpy as np
import pandas as pd

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.data import build_canonical_store
from model.unified.evaluation import promotion_gate
from model.unified.features import build_pit_features
from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer


def _prediction(events):
    return RawPrediction(
        fixture_id="f1", player_id="p1", player_name="Player", team="A", opponent="B",
        position="Back-three", is_forward=False,
        events={k: EventDistribution("poisson", float(v)) for k, v in events.items()},
        minutes=EventDistribution("lognormal", 70, 20),
    )


def test_same_raw_prediction_scores_under_both_rule_adapters():
    events = {"tries": np.array([1.]), "tackles": np.array([5.]), "metres": np.array([29.]),
              "tackle_turnover": np.array([1.]), "red_cards": np.array([1.])}
    ncr = NationsChampionshipScorer().score_samples(events, is_forward=False)
    six = SixNationsScorer().score_samples(events, is_forward=False)
    assert ncr[0] == 12 + 5 + 4 - 10
    assert six[0] == 10 + 5 + 2 + 5 - 8


def test_ncr_converter_includes_api_blind_scoring_categories():
    events = {
        "lineouts_won": np.array([2.]), "lineout_steals": np.array([1.]),
        "scrums_won": np.array([1.]), "interceptions": np.array([1.]),
        "potm": np.array([1.]),
    }
    score = NationsChampionshipScorer().score_samples(events, is_forward=True)
    assert score[0] == 2 + 5 + 2 + 5 + 15


def test_distribution_sampling_and_serialization_are_reproducible():
    prediction = _prediction({"tries": 1.2})
    assert np.array_equal(prediction.sample(25, seed=4)["tries"], prediction.sample(25, seed=4)["tries"])
    restored = RawPrediction.from_dict(prediction.to_dict())
    assert restored.events["tries"].mean == 1.2


def test_canonical_store_deduplicates_by_priority_and_preserves_missing_mask(tmp_path: Path):
    common = {
        "date": ["2025-01-01"], "comp_id": [1], "comp_name": ["Test"],
        "fixture_id": [10], "team": ["A"], "team_id": [1], "opponent": ["B"],
        "opponent_id": [2], "player_id": [99], "player_name": ["P"], "jersey": [7],
        "started": [True], "minutes": [70], "tries": [1],
    }
    low = pd.DataFrame(common)
    low["tries"] = 9
    low["offload"] = 2
    high = pd.DataFrame(common)
    low_path, high_path = tmp_path / "low.csv", tmp_path / "high.csv"
    low.to_csv(low_path, index=False)
    high.to_csv(high_path, index=False)
    result = build_canonical_store(((low_path, "low", 1, "club"), (high_path, "high", 2, "international")))
    assert len(result) == 1
    assert result.loc[0, "tries"] == 1
    assert result.loc[0, "offload"] == 2
    assert bool(result.loc[0, "available__offload"])
    assert pd.isna(result.loc[0, "potm"])
    assert not bool(result.loc[0, "available__potm"])


def test_pit_features_never_include_current_match_target():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-08"]), "fixture_id": [1, 2],
        "player_id": [9, 9], "team": ["A", "A"], "opponent": ["B", "C"],
        "minutes": [80, 80], "started": [True, True], "is_forward": [True, True],
        "team_score": [10, 20], "opp_score": [0, 0],
    })
    from model.unified.schema import EVENTS
    for event in EVENTS:
        frame[event] = np.nan
        frame[f"available__{event}"] = False
    frame["tries"] = [2, 99]
    frame["available__tries"] = True
    pit = build_pit_features(frame)
    assert pd.isna(pit.iloc[0]["form_per80__tries"])
    assert pit.iloc[1]["form_per80__tries"] == 2


def test_promotion_gate_requires_both_competitions_and_aggregate_gain():
    incumbent = {c: {"points_mae": 10, "top_10_capture": .50} for c in ("ncr", "six")}
    good = {c: {"points_mae": 10.1, "top_10_capture": .54} for c in ("ncr", "six")}
    bad = {"ncr": {"points_mae": 9, "top_10_capture": .60}}
    assert promotion_gate(good, incumbent).passed
    decision = promotion_gate(bad, incumbent)
    assert not decision.passed
    assert any("missing" in reason for reason in decision.reasons)
