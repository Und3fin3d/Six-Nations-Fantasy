from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.raw_benchmark import coverage
from model.unified.raw_benchmark.blend import EventBlend50, EventWeightedBlend
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
from model.unified.raw_benchmark.p3_hillclimb import (
    RESULT_FILES, TargetEvidence, accepted_coordinate_table, add_output_hashes,
    infer_left_mean, reweight_saved_predictions, select_coordinate,
    validate_existing_manifest,
)
from model.unified.raw_benchmark.p3_checkpoint import (
    COMPLETE_TARGETS, DEVELOPMENT_RAW_LIMIT, MAX_FOLD_REGRESSION,
    RAW_WEIGHT_GRID, SELECTION_RAW_LIMIT, RawGrid, _load_complete_weights,
    run_checkpoint,
)
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


def test_event_weighted_blend_changes_only_named_raw_event():
    empirical = _StaticModel([_raw(0.2)])
    v4 = _StaticModel([_raw(0.8)])
    blend = EventWeightedBlend(
        empirical, v4, weight_v4=0.5, event_weights_v4={"tries": 0.25},
    )
    prediction = blend.predict_frame(pd.DataFrame([{"unused": 1}]))[0]
    assert prediction.events["tries"].mean == pytest.approx(0.35)
    assert prediction.events["tries"].dispersion != pytest.approx(1.0)
    assert prediction.minutes.mean == pytest.approx(70.0)
    assert prediction.metadata["event_weights_v4"] == {"tries": 0.25}
    assert np.isfinite(SixNationsScorer().score_prediction(
        prediction, n=20, seed=1,
    ).mean)
    assert np.isfinite(NationsChampionshipScorer().score_prediction(
        prediction, n=20, seed=1,
    ).mean)
    with pytest.raises(ValueError, match="blend weights"):
        EventWeightedBlend(empirical, v4, event_weights_v4={"tries": 1.01})
    with pytest.raises(ValueError, match="unknown event"):
        EventWeightedBlend(empirical, v4, event_weights_v4={"competition": 0.4})


def test_hillclimb_recovers_component_mean_and_does_not_consult_confirmation():
    empirical = np.asarray([0.2, 0.4])
    v4 = np.asarray([0.8, 0.6])
    blended = 0.5 * empirical + 0.5 * v4
    np.testing.assert_allclose(infer_left_mean(blended, empirical), v4)
    trials = pd.DataFrame([
        {"weight_v4": 0.4, "development_score": 0.90,
         "selection_score": 1.001, "confirmation_score": 9.0},
        {"weight_v4": 0.5, "development_score": 0.91,
         "selection_score": 1.000, "confirmation_score": 8.0},
        {"weight_v4": 0.6, "development_score": 0.92,
         "selection_score": 0.990, "confirmation_score": 0.0},
    ])
    decision = select_coordinate(trials)
    assert decision["accepted"]
    assert decision["selected_weight_v4"] == 0.4
    assert decision["confirmation_consulted"] is False


def test_hillclimb_selection_guard_rejects_development_only_gain():
    trials = pd.DataFrame([
        {"weight_v4": 0.4, "development_score": 0.90, "selection_score": 1.003},
        {"weight_v4": 0.5, "development_score": 0.91, "selection_score": 1.000},
    ])
    decision = select_coordinate(trials)
    assert not decision["accepted"]
    assert decision["selected_weight_v4"] == 0.5
    assert decision["reason"] == "selection raw loss regresses beyond the guard"


def test_hillclimb_report_table_supports_all_rejected_coordinates():
    table = accepted_coordinate_table([{"target": "tries", "accepted": False}])
    assert table.empty
    assert list(table) == [
        "target", "selected_weight_v4", "development_gain", "selection_delta",
    ]


def test_hillclimb_reweight_rejects_truncated_and_misaligned_components():
    empirical = (_raw(0.2),)
    baseline = (_raw(0.4),)
    with pytest.raises(ValueError, match="lengths differ"):
        reweight_saved_predictions(empirical, (), {"tries": 0.3})
    with pytest.raises(ValueError, match="rows are misaligned"):
        reweight_saved_predictions(
            empirical, (replace(baseline[0], player_id="different"),),
            {"tries": 0.3},
        )


def test_hillclimb_manifest_rejects_incompatible_overwrite(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    manifest = {"schema_version": 1, "input_sha256": {"source": "abc"}}
    for name in RESULT_FILES:
        (output / name).write_text(f"{name}\n")
    completed = add_output_hashes(manifest, output)
    (output / "manifest.json").write_text(json.dumps(completed))
    validate_existing_manifest(output, manifest)
    with pytest.raises(FileExistsError, match="different sources"):
        validate_existing_manifest(
            output, {"schema_version": 1, "input_sha256": {"source": "def"}},
        )
    extra = output / "unexpected.txt"
    extra.write_text("unexpected\n")
    with pytest.raises(FileExistsError, match="unexpected or missing"):
        validate_existing_manifest(output, manifest)
    extra.unlink()
    (output / RESULT_FILES[0]).write_text("tampered\n")
    with pytest.raises(FileExistsError, match="does not match"):
        validate_existing_manifest(output, manifest)


def test_hillclimb_manifest_requires_every_result_hash(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    manifest = {"schema_version": 1}
    for name in RESULT_FILES:
        (output / name).write_text(f"{name}\n")
    completed = add_output_hashes(manifest, output)
    completed["output_sha256"].pop(RESULT_FILES[0])
    (output / "manifest.json").write_text(json.dumps(completed))
    with pytest.raises(FileExistsError, match="incomplete manifested results"):
        validate_existing_manifest(output, manifest)


def test_hillclimb_manifest_rejects_nonempty_unmanifested_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "old-result.csv").write_text("stale\n")
    with pytest.raises(FileExistsError, match="unmanifested results"):
        validate_existing_manifest(output, {"schema_version": 1})


def test_checkpoint_loads_complete_global_vector_and_rejects_unknown_target(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_model": "p3_event_50",
        "default_weight_v4": 0.5,
        "event_weights_v4": {"tries": 0.1},
    }))
    _, weights = _load_complete_weights(path)
    assert set(weights) == set(COMPLETE_TARGETS)
    assert weights["tries"] == 0.1
    assert weights["tackles"] == 0.5

    path.write_text(json.dumps({
        "base_model": "p3_event_50",
        "event_weights_v4": {"competition": 0.1},
    }))
    with pytest.raises(ValueError, match="unknown targets"):
        _load_complete_weights(path)


def test_checkpoint_raw_grid_enforces_all_global_guards():
    evidence = (
        TargetEvidence(
            fold="development_fold", tournament="International",
            calendar_year=2024, hemisphere="north", target="tries",
            actual=np.asarray([0.0]), empirical=np.asarray([0.0]),
            v4=np.asarray([0.0]), naive_loss=1.0,
        ),
        TargetEvidence(
            fold="selection_fold", tournament="International",
            calendar_year=2025, hemisphere="south", target="tries",
            actual=np.asarray([0.0]), empirical=np.asarray([0.0]),
            v4=np.asarray([0.0]), naive_loss=1.0,
        ),
        TargetEvidence(
            fold="retrospective_fold", tournament="Six Nations",
            calendar_year=2026, hemisphere="north", target="scrums_won",
            actual=np.asarray([0.0]), empirical=np.asarray([0.0]),
            v4=np.asarray([0.0]), naive_loss=1.0,
        ),
    )
    values = np.full((3, len(RAW_WEIGHT_GRID)), 0.85)
    values[:, 50] = 0.85
    values[0, 60] = max(DEVELOPMENT_RAW_LIMIT, 0.85 + MAX_FOLD_REGRESSION) + 0.001
    values[1, 60] = SELECTION_RAW_LIMIT + 0.001
    values[2, 60] = 0.95
    weights = {target: 0.5 for target in COMPLETE_TARGETS}
    weights["tries"] = 0.6
    weights["scrums_won"] = 0.6
    summary = RawGrid(evidence=evidence, values=values).summary(weights)
    assert not summary["passes"]
    assert set(summary["failures"]) == {
        "development_raw_limit", "selection_raw_limit", "maximum_fold_regression",
    }
    assert summary["uses_2026_raw_feedback"]
    assert summary["max_fold_regression_fold"] == "retrospective_fold"
    assert summary["optimized_extended_targets"] == ["scrums_won"]


def test_checkpoint_refuses_existing_output_before_search(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must be a new directory"):
        run_checkpoint(benchmark, config, output)


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
