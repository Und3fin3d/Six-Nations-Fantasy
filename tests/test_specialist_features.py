from __future__ import annotations

import pandas as pd

from build_features import matchup_features, team_bench_slot_features
from model import data as model_data


def test_specialist_feature_families_are_opt_in() -> None:
    df = model_data.load()
    old = model_data.MATCHUP_FEATURES_ENABLED
    try:
        model_data.MATCHUP_FEATURES_ENABLED = False
        base = model_data.feature_view(df, "post_team_sheet")
        assert not any(col.startswith("matchup_") for col in base)
        assert not any(col.startswith("benchhist_") for col in base)
        assert not any(col.startswith("benchteam_") for col in base)

        model_data.MATCHUP_FEATURES_ENABLED = True
        enabled = model_data.feature_view(df, "post_team_sheet")
        assert sum(col.startswith("matchup_") for col in enabled) == 14
        assert not any(col.startswith("benchhist_") for col in enabled)
        assert not any(col.startswith("benchteam_") for col in enabled)
    finally:
        model_data.MATCHUP_FEATURES_ENABLED = old


def test_matchup_features_prefer_supported_exact_position() -> None:
    asof = pd.Timestamp("2025-02-01")
    rows = []
    for i in range(8):
        rows.append({
            "date": asof - pd.Timedelta(days=30 + i),
            "fixture_id": i,
            "_matchup_pos": "Back-row",
            "_matchup_is_forward": True,
            "minutes": 80,
            "tries": 1,
            "try_assists": 0,
            "conversion_goals": 0,
            "penalty_goals": 0,
            "metres": 40,
            "defenders_beaten": 2,
            "offload": 1,
            "tackles": 10,
            "tackle_turnover": 1,
            "penalties_conceded": 1,
        })
    rows.append({**rows[0], "fixture_id": 99, "_matchup_pos": "Prop", "metres": 0})
    result = matchup_features(
        pd.DataFrame(rows), asof, "Back-row", True, 90.0
    )

    assert result["matchup_exact_position"] == 1.0
    assert result["matchup_n_prior"] == 8
    assert result["matchup_fixture_n_prior"] == 8
    assert result["matchup_per80_metres"] == 40.0


def test_team_bench_slot_features_use_only_named_slot() -> None:
    asof = pd.Timestamp("2025-02-01")
    hist = pd.DataFrame([
        {
            "date": asof - pd.Timedelta(days=30),
            "jersey": 20,
            "started": False,
            "minutes": 40,
        },
        {
            "date": asof - pd.Timedelta(days=20),
            "jersey": 20,
            "started": False,
            "minutes": 20,
        },
        {
            "date": asof - pd.Timedelta(days=10),
            "jersey": 21,
            "started": False,
            "minutes": 80,
        },
    ])
    result = team_bench_slot_features(hist, asof, 20, 90.0)

    assert result["benchteam_slot_n_prior"] == 2
    assert 20.0 < result["benchteam_slot_minutes_recent"] < 40.0
    assert result["benchteam_slot_play10_rate"] == 1.0
