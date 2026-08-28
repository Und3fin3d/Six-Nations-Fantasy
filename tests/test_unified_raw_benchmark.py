from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.raw_benchmark import coverage
from model.unified.raw_benchmark.blend import EventBlend50
from model.unified.raw_benchmark.config import (
    CHALLENGER_ENGINE_ORDER, CORE_ENGINE_ORDER, DIAGNOSTIC_ONLY_EVENTS,
    EXTENDED_EVENTS, STABLE_EVENTS,
)
from model.unified.raw_benchmark.empirical import EmpiricalEventModel
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import (
    build_folds, masked_candidates, strict_training_frame,
)
from model.unified.raw_benchmark.report import _extended_fixture_support
from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer
from model.unified.schema import EVENTS


def _cache_payload(players, *, fixture=1, date="2022-02-05T14:15:00+00:00", comp=1266):
    return {
        "results": {
            "match": {
                "id": fixture, "date": date, "comp_id": comp,
                "comp_name": "Six Nations", "season": 2022,
                "game_week": 1, "round_id": 1,
                "home_team": "France", "away_team": "Wales",
                "home_id": 1, "away_id": 2,
            },
            "events": [],
            "home": {"teamsheet": players},
            "away": {"teamsheet": []},
        }
    }


def _player(pid, stats):
    return {
        "player_id": pid, "name": f"Player {pid}", "position": pid,
        "substitute": False, "match_stats": stats,
    }


def test_missing_source_key_is_unavailable_but_explicit_zero_is_observed(tmp_path):
    path = tmp_path / "match_1.json"
    path.write_text(json.dumps(_cache_payload([
        _player(1, {"tries": "0"}), _player(2, {}),
    ])))
    _, players, ledger = coverage.parse_cache_fixture(path)
    first, second = players
    assert first["tries_cache"] == 0
    assert first["available__tries_cache"] is True
    assert np.isnan(second["tries_cache"])
    assert second["available__tries_cache"] is False
    tries = next(row for row in ledger if row["event"] == "tries")
    assert tries["status"] == "ambiguous"
    assert tries["coverage_fraction"] == 0.5


def test_dense_fixture_threshold_does_not_turn_missing_player_key_into_zero(tmp_path):
    players = [_player(i, {"tries": "0"}) for i in range(1, 10)] + [_player(10, {})]
    path = tmp_path / "match_1.json"
    path.write_text(json.dumps(_cache_payload(players)))
    _, rows, ledger = coverage.parse_cache_fixture(path)
    assert next(item for item in ledger if item["event"] == "tries")["status"] == "recorded"
    assert rows[-1]["available__tries_cache"] is False
    assert np.isnan(rows[-1]["tries_cache"])


def test_event_tiers_are_frozen_and_corrupt_lineout_is_diagnostic_only():
    assert len(STABLE_EVENTS) == 23
    assert STABLE_EVENTS[0] == "tries" and STABLE_EVENTS[-1] == "metres"
    assert EXTENDED_EVENTS == (
        "dominant_tackles", "tackle_turnover", "tackle_try_saver", "fifty_22",
        "lineout_steals", "scrums_won", "kicks_retained", "potm",
    )
    assert "lineouts_won" not in STABLE_EVENTS + EXTENDED_EVENTS
    assert DIAGNOSTIC_ONLY_EVENTS == ("lineouts_won",)


def test_official_extensions_cannot_attach_to_ncr(monkeypatch, tmp_path):
    official = tmp_path / "official.csv"
    official.write_text(
        "season,round,team,name,50-22,LS,SW,KR,POTM\n"
        "2026,1,France,A. Example,1,1,1,1,1\n"
    )
    monkeypatch.setattr(coverage, "OFFICIAL", official)
    frame = pd.DataFrame([
        {"competition_id": 1266, "season": 2026, "round": 1, "team": "France",
         "fixture_id": "six", "player_name": "A. Example"},
        {"competition_id": 696, "season": 2026, "round": 1, "team": "France",
         "fixture_id": "ncr", "player_name": "A. Example"},
    ])
    attached, report = coverage._attach_official_extensions(frame)
    assert report["matched_rows"] == 1
    assert attached.loc[0, "available__potm"]
    assert not attached.loc[1, "available__potm"]
    assert np.isnan(attached.loc[1, "potm"])


def _fold_store():
    rows = []
    for fixture, date, comp, competition, year in (
        ("old", "2021-11-01T12:00:00Z", 30, "International", 2021),
        ("fw", "2025-01-31T20:15:00Z", 1266, "Six Nations", 2025),
        ("x2", "2025-02-01T14:00:00Z", 1266, "Six Nations", 2025),
        ("x3", "2025-02-01T16:30:00Z", 1266, "Six Nations", 2025),
        ("x4", "2025-02-08T14:00:00Z", 1266, "Six Nations", 2025),
        ("x5", "2025-02-08T16:30:00Z", 1266, "Six Nations", 2025),
    ):
        for team in ("France", "Wales"):
            row = {
                "fixture_id": fixture, "match_at": date, "date": date[:10],
                "competition_id_cache": comp, "competition_cache": competition,
                "competition_level": "international", "calendar_year": year,
                "source_game_week": 1, "source_round": 1,
                "team": team, "hemisphere": "north", "player_id": team,
                "player_name": team, "opponent": "Wales" if team == "France" else "France",
                "position": "Back-row", "started": True, "is_forward": True,
                "minutes": 80, "source": "test", "team_score": 10, "opp_score": 8,
            }
            for event in EVENTS:
                row[event] = 0
                row[f"available__{event}"] = True
            row["available__minutes"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def test_exact_cutoff_excludes_france_wales_from_2025_fold_training():
    store = _fold_store()
    fold = next(fold for fold in build_folds(store) if fold.label == "six_nations_2025")
    assert fold.cutoff == "2025-01-31T20:15:00+00:00"
    train = strict_training_frame(store, fold)
    assert set(train["fixture_id"]) == {"old"}
    assert "fw" not in set(train["fixture_id"])


def test_masked_candidates_and_frozen_features_cannot_consume_eval_targets():
    train = _fold_store().query("fixture_id == 'old'").copy()
    evaluation = _fold_store().query("fixture_id in ['fw', 'x2']").copy()
    evaluation.loc[evaluation["fixture_id"].eq("fw"), "tries"] = 9
    candidates = masked_candidates(evaluation)
    assert candidates["tries"].isna().all()
    assert not candidates["available__tries"].any()
    _, features = build_frozen_feature_frames(train, candidates)
    for player, block in features.groupby("player_id"):
        assert block["form_per80__tries"].nunique(dropna=False) == 1


class _StaticModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_frame(self, frame):
        return self.predictions


def _raw(mean):
    return RawPrediction(
        fixture_id="1", player_id="1", player_name="P", team="France",
        opponent="Wales", position="Back-row", is_forward=True,
        events={"tries": EventDistribution("negative_binomial", mean, 1.0)},
        minutes=EventDistribution("lognormal", 70, 20),
    )


def test_event_blend_preserves_raw_contract_and_both_scorers_accept_it():
    blend = EventBlend50(_StaticModel([_raw(0.2)]), _StaticModel([_raw(0.6)]))
    prediction = blend.predict_frame(pd.DataFrame([{"unused": 1}]))[0]
    assert prediction.events["tries"].mean == pytest.approx(0.4)
    six = SixNationsScorer().score_prediction(prediction, n=20, seed=1)
    ncr = NationsChampionshipScorer().score_prediction(prediction, n=20, seed=1)
    assert np.isfinite(six.mean) and np.isfinite(ncr.mean)


def test_empirical_serialization_is_deterministic_and_covers_lineup_roles(
    monkeypatch, tmp_path,
):
    # Keep the unit test independent of mutable RugbyPass and rankings files.
    monkeypatch.setattr("model.unified.raw_benchmark.empirical.DATA", tmp_path)
    train = _fold_store().query("fixture_id == 'old'").copy()
    train.loc[train.index[1], "started"] = False
    train.loc[train.index[1], "minutes"] = 0
    candidates = _fold_store().query("fixture_id == 'fw'").copy()
    candidates.loc[candidates.index[1], "started"] = False
    unseen = candidates.iloc[[0]].copy()
    unseen["player_id"] = "never-seen"
    unseen["player_name"] = "Never Seen"
    unseen["started"] = False
    candidates = pd.concat([candidates, unseen], ignore_index=True)

    model = EmpiricalEventModel(asof="2025-01-31T20:15:00Z").fit(train)
    before = model.predict_frame(candidates)
    path = tmp_path / "empirical.pkl"
    model.save(path)
    after = EmpiricalEventModel.load(path).predict_frame(candidates)

    assert [prediction.to_dict() for prediction in before] == [
        prediction.to_dict() for prediction in after
    ]
    assert len(after) == 3
    assert all(np.isfinite(prediction.minutes.mean) for prediction in after)
    assert after[-1].player_id == "never-seen"


def test_raw_prediction_round_trip_is_exact():
    prediction = _raw(0.4)
    assert RawPrediction.from_dict(prediction.to_dict()) == prediction


def test_extended_fixture_support_is_not_multiplied_by_engine_count():
    metrics = pd.DataFrame([
        {"engine": "v1", "target": "fifty_22", "fixtures": 42},
        {"engine": "v4", "target": "fifty_22", "fixtures": 42},
        {"engine": "p3_event_50", "target": "fifty_22", "fixtures": 42},
    ])
    assert _extended_fixture_support(metrics)["fifty_22"] == 42


def test_challengers_do_not_change_the_frozen_core_engine_set():
    assert CORE_ENGINE_ORDER == (
        "v1", "gbdt_v3", "v4", "v5_t", "empirical_event", "p3_event_50",
    )
    assert CHALLENGER_ENGINE_ORDER == ("v1_neural",)
