import numpy as np
import pandas as pd
import pytest

from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.raw_benchmark.status_rates import (
    POOLED, StatusAwareEmpiricalEventModel, estimate_status_factors, starter_equivalent, status_factor)

EVENTS = (*STABLE_EVENTS, *EXTENDED_EVENTS)


def _history(bench_boost=1.5, n_players=40, n_games=30, seed=0):
    """Each player starts and comes off the bench; bench minutes are bench_boost x as productive."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2023-01-07", periods=n_games, freq="7D", tz="UTC")
    for p in range(n_players):
        position = "Back-row" if p % 2 else "Prop"
        talent = rng.uniform(0.5, 2.0)
        for g, date in enumerate(dates):
            started = (g + p) % 3 != 0
            minutes = 70.0 if started else 20.0
            rate = talent * (1.0 if started else bench_boost)
            row = {"fixture_id": f"{g}", "player_id": f"P{p}", "player_name": f"Player {p}", "team": "A" if p < 20 else "B",
                   "opponent": "B" if p < 20 else "A", "date": date.tz_convert(None), "match_at": date,
                   "competition_level": "international" if g % 2 else "club", "position": position,
                   "is_forward": True, "started": started, "minutes": minutes, "available__minutes": True,
                   "home_away": "home", "season": "2023"}
            for event in EVENTS:
                row[event] = 0.0
                row[f"available__{event}"] = True
            row["tackles"] = float(rng.poisson(rate * 10 * minutes / 80))
            row["tries"] = float(rng.poisson(rate * 0.3 * minutes / 80))
            row["penalty_goals"] = float(rng.poisson(0.5 * minutes / 80))
            rows.append(row)
    return pd.DataFrame(rows)


def test_within_player_bench_factor_is_recovered():
    history = _history(bench_boost=1.5)
    factors = estimate_status_factors(history, ("tackles", "penalty_goals"), "2024-01-01", prior_events=0.0)
    assert factors[("Back-row", False, "tackles")] == pytest.approx(1.5, rel=0.05)
    assert factors[("Prop", False, "tackles")] == pytest.approx(1.5, rel=0.05)
    assert factors[(POOLED, False, "tackles")] == pytest.approx(1.5, rel=0.05)
    # Goal kicking is never status-adjusted.
    assert ("Prop", False, "penalty_goals") not in factors
    assert status_factor(factors, "Prop", True, "tackles") == 1.0
    assert status_factor(factors, "Unknown", False, "tackles") == factors[(POOLED, False, "tackles")]


def test_no_status_effect_gives_unit_factors_and_shrinkage_pulls_to_pooled():
    factors = estimate_status_factors(_history(bench_boost=1.0), ("tackles",), "2024-01-01")
    assert factors[("Prop", False, "tackles")] == pytest.approx(1.0, abs=0.05)
    loose = estimate_status_factors(_history(bench_boost=1.5), ("tries",), "2024-01-01", prior_events=0.0)
    tight = estimate_status_factors(_history(bench_boost=1.5), ("tries",), "2024-01-01", prior_events=1e9)
    assert tight[("Prop", False, "tries")] == pytest.approx(tight[(POOLED, False, "tries")])
    assert loose[("Prop", False, "tries")] != pytest.approx(loose[(POOLED, False, "tries")])


def test_starter_equivalent_only_rescales_bench_rows_and_adjusted_events():
    history = _history()
    factors = {("Prop", False, "tackles"): 2.0, (POOLED, False, "tackles"): 1.5}
    out = starter_equivalent(history, factors, ("tackles", "penalty_goals"))
    bench, prop = ~history.started, history.position.eq("Prop")
    assert np.allclose(out.loc[bench & prop, "tackles"], history.loc[bench & prop, "tackles"] / 2.0)
    assert np.allclose(out.loc[bench & ~prop, "tackles"], history.loc[bench & ~prop, "tackles"] / 1.5)
    assert out.loc[history.started, "tackles"].equals(history.loc[history.started, "tackles"])
    assert out["penalty_goals"].equals(history["penalty_goals"])


def test_model_raises_bench_and_keeps_starters_and_kicking_close():
    history = _history(bench_boost=1.5)
    cutoff = "2024-01-01"
    candidates = history[history.fixture_id.eq("0")].drop(columns="started").merge(
        pd.DataFrame({"player_id": [f"P{p}" for p in range(40)], "started": [p % 2 == 0 for p in range(40)]}),
        on="player_id")
    plain = RobustEmpiricalEventModel(asof=cutoff).fit(history).predict_frame(candidates)
    status_model = StatusAwareEmpiricalEventModel(asof=cutoff).fit(history)
    status = status_model.predict_frame(candidates)
    for a, b, started in zip(plain, status, candidates.started):
        assert a.minutes.mean == b.minutes.mean
        assert b.events["penalty_goals"].mean == pytest.approx(a.events["penalty_goals"].mean)
        ratio = b.events["tackles"].mean / a.events["tackles"].mean
        if started:
            assert ratio < 1.0  # starter rates no longer include bench intensity
        else:
            assert ratio > 1.15
    assert status_model.status_factors[("Prop", False, "tackles")] > 1.3
