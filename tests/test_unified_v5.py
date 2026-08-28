from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import RawPrediction
from model.unified.data import ROOT
from model.unified.schema import EVENTS
from model.unified.scoring import NationsChampionshipScorer, scorer_for
from model.unified.v5 import (
    ShrunkFormGBDT, V5Config, ablated_expected_points, add_shrunk_features,
    attach_ncr_potm, first_cap_club_rich_mask, fit_tables, low_history_mask,
    mask_events_globally, paired_bootstrap_ci, position_bias, subgroup_mae,
)

STORE = ROOT / "data" / "unified" / "player_match.csv"


def _row(date, fixture, player, team, opp, position, jersey, started, minutes,
         level="international", tries=0.0):
    return {
        "date": pd.Timestamp(date), "competition": "Test", "competition_id": 1,
        "competition_level": level, "season": 2023, "round": 1,
        "fixture_id": fixture, "player_id": player, "player_name": player,
        "team": team, "team_id": team, "opponent": opp, "opponent_id": opp,
        "position": position,
        "is_forward": position in {"Prop", "Hooker", "Second-row", "Back-row"},
        "jersey": jersey, "started": started, "minutes": minutes,
        "team_score": 20, "opp_score": 10, "source": "test", "source_priority": 1,
        "tries": tries,
    }


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    for event in EVENTS:
        if event not in frame:
            frame[event] = 0.0
        frame[f"available__{event}"] = True
    frame["available__minutes"] = True
    return frame


def _frame(rows: int = 96, seed: int = 17, level: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = np.where(np.arange(rows) % 6 == 0, 0.0,
                       rng.integers(20, 81, rows).astype(float))
    started = np.arange(rows) % 6 != 0
    levels = (
        np.where(np.arange(rows) % 3 == 0, "club", "international")
        if level is None else np.repeat(level, rows)
    )
    frame = pd.DataFrame({
        "date": pd.date_range("2023-01-07", periods=rows, freq="7D"),
        "competition": "Test", "competition_id": 1, "competition_level": levels,
        "season": 2023, "round": np.arange(rows) + 1,
        "fixture_id": [f"f{i // 2}" for i in range(rows)],
        "player_id": [f"p{i % 16}" for i in range(rows)],
        "player_name": [f"Player {i % 16}" for i in range(rows)],
        "team": np.where(np.arange(rows) % 2, "B", "A"),
        "team_id": np.where(np.arange(rows) % 2, "2", "1"),
        "opponent": np.where(np.arange(rows) % 2, "A", "B"),
        "opponent_id": np.where(np.arange(rows) % 2, "1", "2"),
        "position": np.where(np.arange(rows) % 2, "Centre", "Back-row"),
        "is_forward": np.arange(rows) % 2 == 0,
        "jersey": np.where(started, 7, 20), "started": started,
        "minutes": minutes,
        "team_score": rng.integers(10, 40, rows),
        "opp_score": rng.integers(10, 40, rows),
        "source": "test", "source_priority": 1, "available__minutes": True,
    })
    for event in EVENTS:
        if event == "metres":
            values = rng.gamma(2.0, 15.0, rows)
        elif event == "potm":
            values = (np.arange(rows) % 29 == 0).astype(float)
        else:
            values = rng.poisson(0.35, rows).astype(float)
        values[minutes == 0] = 0.0
        frame[event] = values
        frame[f"available__{event}"] = True
    return frame


def _history_frame() -> pd.DataFrame:
    rows = []
    seq: dict[tuple[str, str], int] = {}

    def add(player, n, level, tries, minutes=80.0, position="Back-row",
            start="2022-01-01", step_days=14):
        # Fixture ids continue per (player, level) across add() calls so the
        # (fixture_id, player_id, team) key stays unique: a repeated call for
        # the same player/level models *further* matches, not the same one.
        base = pd.Timestamp(start)
        j0 = seq.get((player, level), 0)
        for j in range(n):
            rows.append(_row(base + pd.Timedelta(days=step_days * j),
                             f"{player}-f{level}-{j0 + j}", player, "T", "O",
                             position, 7, True, minutes, level=level,
                             tries=tries))
        seq[(player, level)] = j0 + n

    add("vet", 12, "international", 1.0)
    add("clubform", 12, "club", 2.0)
    add("dual", 8, "club", 2.0)
    add("dual", 4, "international", 1.0, start="2023-01-01")
    add("bg1", 10, "international", 0.0, position="Centre")
    add("bg2", 10, "international", 0.0, position="Prop")
    add("new", 1, "international", 0.0, position="Centre", start="2024-06-01")
    # vet's 13th international -> "vet-finternational-12" (not a re-used id).
    add("vet", 1, "international", 0.0, start="2024-06-01")
    return _finish(pd.DataFrame(rows))


def _tiny_config(**overrides) -> V5Config:
    base = dict(n_estimators=8, num_leaves=7, min_child_samples=5,
                min_positives=2, seed=19)
    base.update(overrides)
    return V5Config(**base)


def _eb_config(**overrides) -> V5Config:
    """Opt-in config with the killed EB + slot layers re-enabled (ablation)."""
    base = dict(use_eb_features=True, use_slot_minutes=True)
    base.update(overrides)
    return _tiny_config(**base)


def test_config_round_trip_and_exclusion():
    config = V5Config(mask_attribution_corrupted=False, context_blocks=("wr",))
    restored = V5Config.from_dict(config.to_dict())
    assert restored == config
    assert restored.excluded_events() == ()
    # Terminal-rung default (post-K1 corrective cycle): masking ON.
    assert V5Config().excluded_events() == ("lineouts_won",)


def test_default_config_is_terminal_rung_t():
    """Default V5Config = v1 + guards + masking (terminal rung T, plan §4)."""
    config = V5Config()
    assert not config.use_eb_features and not config.use_slot_minutes
    frame = _history_frame()
    # With both layers off, add_shrunk_features is a pass-through.
    out = add_shrunk_features(frame, fit_tables(frame, config), config)
    assert list(out.columns) == list(frame.columns)
    # Even with lineouts_won requested as a target, default masking excludes
    # it from fitting and prediction; no shrink/slot columns are consumed.
    model = ShrunkFormGBDT(
        config=_tiny_config(), events=("tries", "metres", "lineouts_won"),
    ).fit(_frame(96))
    numeric = model.encoder.numeric_columns
    assert not any(c.startswith(("shrink_", "slot_")) for c in numeric)
    assert "lineouts_won" not in model.active_events
    prediction = model.predict_frame(_frame(8).tail(2))[0]
    assert "lineouts_won" not in prediction.events
    assert prediction.metadata["shrink_weight_tackles"] is None


def test_shrunk_features_are_point_in_time():
    frame = _history_frame()
    key = ["fixture_id", "player_id", "team"]
    # Fixture contract: the synthetic key is unique. A duplicated key would
    # fan out the merge below for reasons unrelated to point-in-time safety.
    assert not frame.duplicated(subset=key).any()
    config = _eb_config()
    prefix = frame.iloc[:30].copy()
    tables = fit_tables(prefix, config)
    cols = [c for c in add_shrunk_features(prefix, tables, config)
            if c.startswith(("shrink_", "slot_", "hist_"))]
    feats_prefix = add_shrunk_features(prefix, tables, config)
    feats_full = add_shrunk_features(frame, tables, config)
    merged = feats_prefix[key + cols].merge(
        feats_full[key + cols], on=key, suffixes=("_a", "_b"),
    )
    assert len(merged) == len(prefix)
    for col in cols:
        pd.testing.assert_series_equal(
            merged[f"{col}_a"], merged[f"{col}_b"], check_names=False,
        )


def test_shrinkage_direction_prior_vs_observed():
    frame = _history_frame()
    config = _eb_config()
    tables = fit_tables(frame, config)
    feats = add_shrunk_features(frame, tables, config)
    prior_centre = tables.prior_rate["tries"]["Centre"]
    prior_backrow = tables.prior_rate["tries"]["Back-row"]

    new_row = feats[feats["player_id"].eq("new")].iloc[0]
    assert new_row["shrink_per80__tries"] == pytest.approx(prior_centre)
    assert new_row["shrink_weight__tries"] == pytest.approx(0.0)

    vet_last = feats[feats["player_id"].eq("vet")].sort_values("date").iloc[-1]
    observed = 1.0
    assert prior_backrow < vet_last["shrink_per80__tries"] < observed
    # 12 prior 80-minute matches: shrinkage sits far closer to observed than prior.
    assert vet_last["shrink_per80__tries"] > (prior_backrow + observed) / 2
    assert 0.5 < vet_last["shrink_weight__tries"] < 1.0


def test_club_calibration_direction_and_bounds():
    frame = _history_frame()
    config = _eb_config()
    tables = fit_tables(frame, config)
    # The only dual-history player scores at 2.0/80 in club vs 1.0/80 in tests;
    # 640 dual club minutes against a 4000-minute scale keeps 16% of the effect.
    assert tables.cal["tries"] == pytest.approx(0.92, abs=1e-6)
    assert config.cal_clip_lo <= tables.cal["tries"] <= 1.0
    feats = add_shrunk_features(frame, tables, config)
    club_row = feats[feats["player_id"].eq("clubform")].sort_values("date").iloc[-1]
    assert 0.0 < club_row["shrink_per80__tries"] < 2.0

    intl_only = _finish(pd.DataFrame([
        _row("2022-01-01", "f0", "p0", "T", "O", "Centre", 12, True, 80.0),
        _row("2022-01-15", "f1", "p0", "T", "O", "Centre", 12, True, 80.0),
    ]))
    tables_intl = fit_tables(intl_only, config)
    assert all(value == 1.0 for value in tables_intl.cal.values())


def test_rare_event_guard_uses_level_fallback():
    frame = _frame(120)
    frame["tries"] = 0.0
    frame.loc[frame.index[[3, 17, 42]], "tries"] = 1.0  # 3 positives < 5
    config = _tiny_config(min_positives=5)
    model = ShrunkFormGBDT(config=config, events=("tries", "metres", "potm")).fit(frame)
    assert "tries" not in model.models
    assert "metres" in model.models
    predictions = model.predict_frame(frame.tail(4))
    levels = frame.tail(4)["competition_level"].astype(str)
    for prediction, lvl in zip(predictions, levels):
        expected = model.fallback_means["tries"].get(
            lvl, model.fallback_means["tries"]["__all__"])
        assert prediction.events["tries"].mean == pytest.approx(expected)


def test_masking_excludes_event_and_scorer_reads_absent_as_zero():
    frame = _frame(96)
    config = _tiny_config(mask_attribution_corrupted=True)
    model = ShrunkFormGBDT(
        config=config, events=("tries", "metres", "lineouts_won"),
    ).fit(frame)
    prediction = model.predict_frame(frame.tail(2))[0]
    assert "lineouts_won" not in prediction.events
    assert "tries" in prediction.events

    scorer = NationsChampionshipScorer()
    base = {"tries": np.array([1.0, 0.0])}
    with_zeros = {"tries": np.array([1.0, 0.0]),
                  "lineouts_won": np.array([0.0, 0.0])}
    np.testing.assert_array_equal(
        scorer.score_samples(base, is_forward=False),
        scorer.score_samples(with_zeros, is_forward=False),
    )


def test_mask_events_globally_marks_unavailable():
    frame = _frame(24)
    masked = mask_events_globally(frame, ("lineouts_won",))
    assert not masked["available__lineouts_won"].any()
    assert masked["lineouts_won"].isna().all()
    assert frame["available__lineouts_won"].all()  # input untouched


def test_attach_ncr_potm_labels(tmp_path: Path):
    frame = _finish(pd.DataFrame([
        _row("2026-07-11", "8546000", "p1", "T", "O", "Centre", 12, True, 80.0),
        _row("2026-07-11", "8546000", "p2", "T", "O", "Back-row", 7, True, 80.0),
        _row("2026-07-11", "8546000", "p3", "O", "T", "Prop", 3, True, 80.0),
        _row("2026-07-11", "8546001", "p4", "X", "Y", "Centre", 12, True, 80.0),
    ]))
    frame["available__potm"] = False
    path = tmp_path / "potm.csv"
    path.write_text(
        "date,fixture_id,team,player_id,player_name,source\n"
        "2026-07-11,8546000,T,p2,Player Two,test\n"
    )
    updated, report = attach_ncr_potm(frame, path)
    by_player = updated.set_index("player_id")
    assert by_player.loc["p2", "potm"] == 1.0
    assert by_player.loc["p1", "potm"] == 0.0
    assert by_player.loc["p3", "potm"] == 0.0
    assert bool(by_player.loc["p1", "available__potm"])
    assert not bool(by_player.loc["p4", "available__potm"])
    assert report["winners"] == 1 and report["covered_fixtures"] == 1
    assert report["rows_labelled"] == 3


def test_ablated_expected_points_linearity():
    frame = _frame(96)
    model = ShrunkFormGBDT(
        config=_tiny_config(mask_attribution_corrupted=False),
        events=("tries", "metres", "lineouts_won"),
    ).fit(frame)
    predictions = model.predict_frame(frame.tail(3))
    base = ablated_expected_points(predictions, "ncr")
    dropped = ablated_expected_points(predictions, "ncr", drop_events=("lineouts_won",))
    # NCR scoring is linear and lineouts_won carries weight 1.
    for i, prediction in enumerate(predictions):
        assert base[i] - dropped[i] == pytest.approx(
            prediction.events["lineouts_won"].mean)


def test_determinism_serialization_and_contract(tmp_path: Path):
    frame = _frame()
    events = ("tries", "metres", "potm")
    first = ShrunkFormGBDT(config=_tiny_config(), events=events).fit(frame)
    second = ShrunkFormGBDT(config=_tiny_config(), events=events).fit(frame)
    candidate = frame.tail(4).copy()
    candidate.loc[candidate.index[-1], "player_id"] = "never-seen"
    pred_first = first.predict_frame(candidate)
    pred_second = second.predict_frame(candidate)
    assert [p.minutes.mean for p in pred_first] == pytest.approx(
        [p.minutes.mean for p in pred_second])
    artifact = tmp_path / "v5.pkl"
    first.save(artifact)
    restored = ShrunkFormGBDT.load(artifact).predict_frame(candidate)
    assert [p.events["tries"].mean for p in restored] == pytest.approx(
        [p.events["tries"].mean for p in pred_first])
    assert all(0 <= p.minutes.mean <= 80 for p in restored)
    round_trip = RawPrediction.from_dict(restored[0].to_dict())
    assert round_trip.fixture_id == restored[0].fixture_id
    for competition in ("six_nations", "ncr"):
        summary = scorer_for(competition).score_prediction(restored[-1], n=64, seed=2)
        assert np.isfinite(summary.mean)


def test_zero_minute_bench_pattern_predicts_fewer_minutes():
    rows = []
    for j in range(12):
        rows.append(_row(pd.Timestamp("2022-01-01") + pd.Timedelta(days=14 * j),
                         f"s{j}", "starter", "T", "O", "Back-row", 7, True,
                         70.0 + (j % 3) * 5, tries=1.0))
        rows.append(_row(pd.Timestamp("2022-01-01") + pd.Timedelta(days=14 * j),
                         f"b{j}", "bench", "T", "O", "Back-row", 20, False, 0.0))
    rows.append(_row("2023-01-01", "eval", "starter", "T", "O", "Back-row", 7, True, 0.0))
    rows.append(_row("2023-01-01", "eval", "bench", "T", "O", "Back-row", 20, False, 0.0))
    frame = _finish(pd.DataFrame(rows))
    # Predict on the full frame so candidate rows carry their own history,
    # matching the repo's concat-then-select shadow pattern.
    model = ShrunkFormGBDT(
        config=_tiny_config(n_estimators=60), events=("tries",),
    ).fit(frame)
    predictions = {p.player_id: p for p in model.predict_frame(frame)}
    assert predictions["bench"].minutes.mean < predictions["starter"].minutes.mean


def test_metrics_helpers():
    frame = pd.DataFrame({
        "position": ["Prop", "Prop", "Centre"],
        "official_pts": [10.0, 20.0, 30.0],
        "pred": [12.0, 14.0, 25.0],
        "career_matches": [1.0, 9.0, 2.0],
        "hist_intl_matches": [0.0, 5.0, 3.0],
        "hist_club_matches": [8.0, 1.0, 4.0],
    })
    bias = position_bias(frame, "pred")
    assert set(bias["position"]) == {"Prop", "Centre"}
    assert float(bias[bias["position"].eq("Prop")]["bias"].iloc[0]) == pytest.approx(-2.0)
    mask = low_history_mask(frame)
    assert mask.tolist() == [True, False, True]
    assert first_cap_club_rich_mask(frame).tolist() == [True, False, False]
    assert subgroup_mae(frame, "pred", "official_pts", mask) == pytest.approx(3.5)
    interval = paired_bootstrap_ci(
        frame["official_pts"].to_numpy(), frame["pred"].to_numpy(),
        frame["official_pts"].to_numpy(), n_boot=200)
    assert np.isfinite(interval["mean"])
    assert interval["p05"] <= interval["p95"]


@pytest.mark.skipif(not STORE.exists(), reason="canonical store not built")
def test_store_smoke_fit_predict_contract():
    raw = pd.read_csv(STORE, low_memory=False, parse_dates=["date"])
    # The store's dense coverage starts in 2022; 2024-01-01 leaves ~56k
    # training rows while keeping the tiny-config fit fast.
    train = raw[raw["date"] < "2024-01-01"]
    candidates = raw[raw["date"] >= "2024-01-01"].head(50)
    if len(train) < 1000 or candidates.empty:
        pytest.skip("store slice too small for a smoke fit")
    model = ShrunkFormGBDT(config=_tiny_config()).fit(train)
    # Candidates are scored with their history present (concat-then-select).
    combined = pd.concat([train, candidates], ignore_index=True, sort=False)
    predictions = model.predict_frame(combined)[-len(candidates):]
    assert len(predictions) == len(candidates)
    for competition in ("six_nations", "ncr"):
        summary = scorer_for(competition).score_prediction(predictions[0], n=32, seed=5)
        assert np.isfinite(summary.mean)
