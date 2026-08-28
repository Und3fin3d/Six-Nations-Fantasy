from __future__ import annotations

import numpy as np
import pandas as pd

from model.unified.contracts import EventDistribution
from model.unified.features import build_pit_features
from model.unified.gbdt import UniversalGBDT
from model.unified.schema import EVENTS
from model.unified.v4.features import (
    EB_EVENTS, add_v4_base_stats, apply_eb_features, fit_shrinkage_k,
)
from model.unified.v4.gbdt import V4GBDT


def _raw(rows: int = 120, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = np.where(np.arange(rows) % 5 == 0, 0.0, rng.integers(18, 81, rows))
    data = {
        "date": pd.date_range("2023-01-01", periods=rows, freq="4D"),
        "competition": "Test", "competition_id": 1,
        "competition_level": np.where(np.arange(rows) % 3 == 0, "club", "international"),
        "season": 2023, "round": np.arange(rows) + 1,
        "fixture_id": [f"f{i // 2}" for i in range(rows)],
        "player_id": [f"p{i % 12}" for i in range(rows)],
        "player_name": [f"Player {i % 12}" for i in range(rows)],
        "team": np.where(np.arange(rows) % 2, "B", "A"),
        "team_id": np.where(np.arange(rows) % 2, "2", "1"),
        "opponent": np.where(np.arange(rows) % 2, "A", "B"),
        "opponent_id": np.where(np.arange(rows) % 2, "1", "2"),
        "position": np.where(np.arange(rows) % 2, "Centre", "Back-row"),
        "is_forward": np.arange(rows) % 2 == 0,
        "jersey": np.where(np.arange(rows) % 5 == 0, 20, 7),
        "started": np.arange(rows) % 5 != 0,
        "minutes": minutes,
        "team_score": rng.integers(10, 40, rows),
        "opp_score": rng.integers(10, 40, rows),
        "source": "test", "source_priority": 1,
        "available__minutes": True,
    }
    frame = pd.DataFrame(data)
    for event in EVENTS:
        if event == "metres":
            values = rng.gamma(2.0, 15.0, rows)
        elif event == "potm":
            values = (np.arange(rows) % 29 == 0).astype(float)
        else:
            values = rng.poisson(0.6, rows).astype(float)
        values[minutes == 0] = 0
        frame[event] = values
        frame[f"available__{event}"] = True
    return frame


def _store(rows: int = 120) -> pd.DataFrame:
    return add_v4_base_stats(build_pit_features(_raw(rows)))


def test_flags_off_is_bit_identical_to_v1_baseline():
    frame = build_pit_features(_raw())
    v1 = UniversalGBDT(weighting="natural").fit(frame)
    v4 = V4GBDT(weighting="natural").fit(frame)
    p1 = v1.predict_frame(frame.tail(10))
    p4 = v4.predict_frame(frame.tail(10))
    for a, b in zip(p1, p4):
        assert a.minutes.mean == b.minutes.mean
        for event in a.events:
            assert a.events[event].mean == b.events[event].mean
            assert a.events[event].dispersion == b.events[event].dispersion


def test_eb_rate_collapses_to_prior_and_to_player_rate():
    store = _store()
    k = {event: 200.0 for event in EB_EVENTS}
    out = apply_eb_features(store, k)
    zero_history = out[out["cum_min__tries"].eq(0.0)]
    assert len(zero_history) > 0
    assert np.allclose(zero_history["eb_rate__tries"],
                       zero_history["prior_per80__tries"])
    tiny = apply_eb_features(store, {event: 1e-9 for event in EB_EVENTS})
    rich = tiny[tiny["cum_min__tries"].gt(0)]
    player_rate = 80.0 * rich["cum_count__tries"] / rich["cum_min__tries"]
    assert np.allclose(rich["eb_rate__tries"], player_rate)


def test_v4_features_are_point_in_time():
    raw = _raw()
    base = add_v4_base_stats(build_pit_features(raw))
    future = raw.iloc[[-1]].assign(
        date=raw["date"].max() + pd.Timedelta(days=30),
        fixture_id="future", tries=9.0)
    extended = add_v4_base_stats(build_pit_features(
        pd.concat([raw, future], ignore_index=True)))
    cols = [c for c in base.columns if c.startswith((
        "eb_", "cum_", "prior_per80__", "form_per80_intl__", "form_per80_club__",
        "intl_prior_matches", "club_prior_matches"))]
    key = ["fixture_id", "player_id"]
    merged = base[key + cols].merge(
        extended[key + cols], on=key, suffixes=("_a", "_b"), validate="one_to_one")
    for col in cols:
        pd.testing.assert_series_equal(
            merged[f"{col}_a"], merged[f"{col}_b"],
            check_names=False, atol=1e-12)


def test_shrinkage_k_is_fitted_within_bounds_and_per_frame():
    store = _store(240)
    cutoff = store["date"].quantile(0.6)
    k_early = fit_shrinkage_k(store[store["date"] < cutoff])
    k_full = fit_shrinkage_k(store)
    for k in (k_early, k_full):
        for event, value in k.items():
            assert 40.0 <= value <= 4000.0, (event, value)


def test_hurdle_moment_match_reproduces_mean():
    store = _store(240)
    model = V4GBDT(weighting="natural", hurdle_events=("tries",)).fit(store)
    assert "tries" in model.hurdle_models
    predictions = model.predict_frame(store.tail(20))
    for prediction in predictions:
        dist = prediction.events["tries"]
        assert dist.family == "negative_binomial"
        samples = dist.sample(np.random.default_rng(3), 20000)
        assert abs(samples.mean() - dist.mean) < max(0.05, 0.12 * dist.mean)


def test_player_effects_shrink_to_one_for_unknown_players():
    store = _store(240)
    model = V4GBDT(weighting="natural", pool_player_id=True,
                   player_effects=True, effect_min_rows=50).fit(store)
    assert model.effects, "expected fitted player effects"
    for value in model.effects.values():
        assert 0.25 <= value <= 4.0
    rows = store.tail(4).copy()
    rows["player_id"] = "never-seen-player"
    known = model.predict_frame(store.tail(4))
    unknown = model.predict_frame(rows)
    assert len(known) == len(unknown)  # unknown players fall back to effect 1.0


def test_no_competition_identity_in_features():
    store = _store()
    model = V4GBDT(weighting="natural").fit(store)
    assert "competition" not in model.encoder.categories
    assert "competition_id" not in model.encoder.categories
    assert all("competition_id" not in c for c in model.encoder.numeric_columns)


def test_jersey_train_serve_skew_is_bounded():
    """Prediction-time jerseys are synthesized from position (v3/shadow.py);
    verify swapping a real jersey for the positional default does not move the
    prediction materially (guards the train/serve skew the red team flagged)."""
    store = _store(240)
    model = V4GBDT(weighting="natural").fit(store)
    rows = store.tail(8).copy()
    swapped = rows.copy()
    swapped["jersey"] = np.where(swapped["is_forward"], 7, 13)
    base = model.predict_frame(rows)
    alt = model.predict_frame(swapped)
    for a, b in zip(base, alt):
        assert abs(a.minutes.mean - b.minutes.mean) < 25.0


def test_event_distribution_contract_unchanged():
    dist = EventDistribution("negative_binomial", 0.4, 0.5)
    samples = dist.sample(np.random.default_rng(0), 5000)
    assert samples.min() >= 0
