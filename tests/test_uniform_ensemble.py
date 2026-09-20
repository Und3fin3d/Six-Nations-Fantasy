"""Regressions for a fixed uniform ensemble; no fantasy outcomes in fitting."""
from dataclasses import replace
import pickle

import numpy as np
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.domain_experiment import Candidate, CANDIDATES, CONTROL, STUDIES, run_job
from model.unified.uniform_ensemble import (
    UniformRawEnsemble, mean_distribution, mean_predictions,
)


def prediction(mean=2.0, player="p"):
    return RawPrediction(
        fixture_id="f", player_id=player, player_name="Player", team="A",
        opponent="B", position="Prop", is_forward=True,
        events={"tackles": EventDistribution("negative_binomial", mean, mean)},
        minutes=EventDistribution("lognormal", 60.0, 25.0),
        metadata={"model": "control"},
    )


class Stub:
    def __init__(self, raw):
        self.raw = raw

    def predict_frame(self, frame):
        return self.raw


def test_negative_binomial_mixture_moments():
    result = mean_distribution([
        EventDistribution("negative_binomial", 2, 2),
        EventDistribution("negative_binomial", 4, 4),
    ])
    assert result.mean == 3
    assert result.dispersion == pytest.approx(2.25)
    assert result.mean + result.mean ** 2 / result.dispersion == pytest.approx(7)


def test_lognormal_mixture_preserves_between_fit_variance():
    result = mean_distribution([
        EventDistribution("lognormal", 4, 9), EventDistribution("lognormal", 8, 25),
    ])
    assert result.mean == 6
    assert result.dispersion == 21


def test_bernoulli_mixture():
    result = mean_distribution([
        EventDistribution("bernoulli", .1), EventDistribution("bernoulli", .7),
    ])
    assert result.mean == pytest.approx(.4)
    assert result.dispersion == 1


def test_poisson_mixture_preserves_overdispersion():
    result = mean_distribution([EventDistribution("poisson", 2), EventDistribution("poisson", 4)])
    assert result.family == "negative_binomial"
    assert result.mean == 3
    assert result.mean + result.mean ** 2 / result.dispersion == pytest.approx(4)


def test_identical_poisson_stays_poisson():
    result = mean_distribution([EventDistribution("poisson", 2)] * 3)
    assert result.family == "poisson"
    assert result.mean == 2


def test_three_members_are_equally_weighted():
    result = mean_predictions([[prediction(1)], [prediction(4)], [prediction(10)]])[0]
    assert result.events["tackles"].mean == pytest.approx(5)
    assert result.minutes.mean == 60
    assert result.metadata["members"] == 3


def test_single_member_is_exact_control_including_metadata():
    raw = [prediction()]
    result = UniformRawEnsemble((Stub(raw),)).predict_frame(None)
    assert result[0] is raw[0]
    assert result[0].to_dict() == raw[0].to_dict()
    assert mean_distribution([raw[0].minutes]) is raw[0].minutes


@pytest.mark.parametrize("change", [
    {"team": "C"}, {"fixture_id": "other"}, {"player_id": "other"},
    {"position": "Wing"}, {"opponent": "C"}, {"is_forward": False},
    {"player_name": "Other"}, {"events": {}},
])
def test_different_key_context_or_support_fails(change):
    with pytest.raises(ValueError):
        mean_predictions([[prediction()], [replace(prediction(), **change)]])


def test_missing_row_fails():
    with pytest.raises(ValueError):
        mean_predictions([[prediction()], []])


def test_duplicate_keys_fail():
    with pytest.raises(ValueError, match="duplicate"):
        mean_predictions([[prediction(), prediction()]])


def test_order_is_checked():
    with pytest.raises(ValueError, match="keys"):
        mean_predictions([[prediction(player="a"), prediction(player="b")],
                          [prediction(player="b"), prediction(player="a")]])


def test_empty_inputs_and_mixed_families_fail():
    with pytest.raises(ValueError):
        mean_predictions([])
    with pytest.raises(ValueError):
        mean_distribution([])
    with pytest.raises(ValueError):
        mean_distribution([EventDistribution("poisson", 1), EventDistribution("bernoulli", .5)])
    with pytest.raises(ValueError):
        UniformRawEnsemble(())
    with pytest.raises(ValueError):
        UniformRawEnsemble((object(),))
    assert mean_predictions([[], []]) == []


def test_saved_artifact_round_trip(tmp_path):
    model = UniformRawEnsemble((Stub([prediction(2)]), Stub([prediction(4)])))
    path = tmp_path / "nested" / "model.pkl"
    model.save(path)
    restored = UniformRawEnsemble.load(path)
    assert [p.to_dict() for p in restored.predict_frame(None)] == [p.to_dict() for p in model.predict_frame(None)]
    path.write_bytes(pickle.dumps("not a model"))
    with pytest.raises(TypeError):
        UniformRawEnsemble.load(path)


@pytest.mark.parametrize("seeds", [(), (17,17), (-1,), (True,), (1.5,), [17]])
def test_invalid_seeds_fail(seeds):
    with pytest.raises(ValueError):
        Candidate("natural", seeds=seeds)


def test_existing_control_and_candidate_settings():
    assert CANDIDATES[CONTROL] == Candidate("natural", seeds=(17,))
    assert STUDIES["seedbag"][CONTROL] == CANDIDATES[CONTROL]
    assert STUDIES["seedbag"]["p3_seedbag_native"] == Candidate("natural", seeds=(17,29,43))
    assert "p3_seedbag_native" not in CANDIDATES


def test_unknown_study_and_duplicate_engines_fail_before_data_read(tmp_path):
    with pytest.raises(ValueError, match="study"):
        run_job(tmp_path, "test:invalid", (CONTROL,), study="unknown")
    with pytest.raises(ValueError, match="duplicate"):
        run_job(tmp_path, "test:invalid", (CONTROL,CONTROL), study="seedbag")
    with pytest.raises(ValueError, match="candidate"):
        run_job(tmp_path, "test:invalid", ("p3_balanced_native",), study="seedbag")
