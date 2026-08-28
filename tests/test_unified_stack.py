"""Tests for the v2 stage-2 ranking stack: labels, rubric, features, model."""

import numpy as np
import pandas as pd

from model.unified.labels import build_fantasy_labels
from model.unified.rubric import RUBRIC_EVENTS, rubric_columns, rubric_vector
from model.unified.rank_stack import (
    RankStack, _apply_priors, _position_code, _relevance, feature_columns,
)
from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer


def test_pooled_labels_have_within_group_percentiles_and_ncr_cohort_parity():
    labels = build_fantasy_labels()
    # NCR cohort equals the incumbent projection cohort (parity by construction).
    ncr = labels[labels.competition == "ncr"]
    assert dict(ncr.groupby("round").size()) == {1: 270, 2: 262, 3: 270}
    # Percentiles live in (0, 1] and rank monotonically with official points
    # within each group (ties share the mean rank, so the max need not be 1.0).
    assert labels["label_percentile"].between(0, 1).all()
    for _, block in labels.groupby("group_id"):
        ordered = block.sort_values("official_pts")
        assert (ordered["label_percentile"].diff().dropna() >= -1e-9).all()
        assert ordered["label_percentile"].iloc[-1] >= 0.9
    # 2024 (no fantasy points) is excluded; 2026 is retained as the sealed holdout.
    assert set(labels.query("competition=='six_nations'").season) == {2025, 2026}


def test_rubric_vector_matches_scorer_weights_exactly():
    ncr = dict(zip(RUBRIC_EVENTS, rubric_vector("ncr")))
    six = dict(zip(RUBRIC_EVENTS, rubric_vector("six_nations")))
    # Position-dependent Six Nations try encoded as forward/back components.
    assert six["try_forward"] == 15 and six["try_back"] == 10
    assert six["metres_per_metre"] == 0.1
    # NCR flat try value fills both try components; metres are unscored.
    assert ncr["try_forward"] == ncr["try_back"] == NationsChampionshipScorer.weights["tries"]
    assert ncr["metres_per_metre"] == 0.0
    # Shared events read straight from the weight tables.
    assert six["tackle_turnover"] == SixNationsScorer.weights["tackle_turnover"]
    assert ncr["tackle_turnover"] == NationsChampionshipScorer.weights["tackle_turnover"]


def test_position_code_maps_both_naming_conventions():
    assert _position_code("Prop") == _position_code("LOOSE-HEAD PROP") == 1
    assert _position_code("Back Three") == _position_code("BACK THREE") == 8
    assert _position_code("Loose Forward") == 4  # NCR skill_desc naming
    assert _position_code("Fly-half") == 6


def test_priors_fill_unmatched_rows_from_matched_position_medians():
    frame = pd.DataFrame({
        "position_code": [8, 8, 8],
        "s1_exp_points": [10.0, 20.0, np.nan],
        "s1_minutes": [70.0, 60.0, np.nan],
        "s1_start_rate": [1.0, 1.0, np.nan],
        "s1_matched": [1.0, 1.0, np.nan],
        **{f"s1_ev__{e}": [1.0, 3.0, np.nan] for e in
           ("tries", "try_assists", "tackles", "metres", "defenders_beaten",
            "tackle_turnover", "clean_breaks", "offload", "penalties_conceded")},
    })
    out = _apply_priors(frame)
    assert out.loc[2, "s1_matched"] == 0.0            # flagged as a prior fallback
    assert out.loc[2, "s1_exp_points"] == 15.0        # median of matched back-three rows
    assert out["s1_exp_points"].notna().all()


def test_relevance_is_bounded_decile_and_monotone():
    pct = pd.Series([0.001, 0.25, 0.5, 0.75, 1.0])
    rel = _relevance(pct)
    assert rel.min() >= 0 and rel.max() <= 9
    assert list(rel) == sorted(rel)  # monotone in percentile


def test_feature_columns_exclude_competition_identity():
    """The pooled model conditions on the rubric, never on which competition a
    row belongs to — otherwise it collapses back into two specialists."""
    cols = feature_columns()
    assert not any("competition" in c or c in {"season", "group_id"} for c in cols)
    assert any(c.startswith("rubric__") for c in cols)
    assert "s1_exp_points" in cols


def test_stack_predicts_points_in_range_for_an_unseen_competition():
    """A stack trained on one competition must still emit sane points for the
    other (the NCR GW1 pure-transfer case) — no unit-less blow-up."""
    cols = feature_columns()
    rng = np.random.default_rng(0)
    train = pd.DataFrame({c: rng.normal(size=200) for c in cols})
    train["s1_exp_points"] = rng.uniform(0, 30, size=200)
    train["group_id"] = np.repeat([f"6n-r{i}" for i in range(5)], 40)
    train["official_pts"] = train["s1_exp_points"] + rng.normal(0, 3, size=200)
    train["label_percentile"] = train.groupby("group_id")["official_pts"].rank(pct=True)
    stack = RankStack().fit(train)
    test = pd.DataFrame({c: rng.normal(size=50) for c in cols})
    test["s1_exp_points"] = rng.uniform(0, 30, size=50)
    test["group_id"] = "ncr-gw1"
    pred = stack.predict(test)
    # Anchored by s1_exp_points, so expected points stay in a rugby-fantasy range.
    assert pred["expected_points"].between(-10, 90).all()
    assert pred["stack_score"].notna().all()
