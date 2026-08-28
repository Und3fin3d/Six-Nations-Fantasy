"""Universal per-event GBDT control using the shared canonical feature seam."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import EventDistribution, RawPrediction
from .features import FeatureEncoder
from .schema import EVENTS, distribution_family


class _ConstantProbability:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.probability, 0, 1), dtype=float)
        return np.column_stack([1 - p, p])


class UniversalGBDT:
    """One control model over every competition and every raw event.

    Heads vary by event because their labels and distributions vary; there is no
    competition-specific head or fantasy-points target.
    """

    def __init__(
        self, events: tuple[str, ...] = EVENTS, random_state: int = 17,
        weighting: str = "level_balanced",
        time_half_life_days: float | None = None,
        n_estimators: int = 180, num_leaves: int = 23,
    ):
        self.events = tuple(events)
        self.random_state = random_state
        self.weighting = weighting
        self.time_half_life_days = time_half_life_days
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.encoder = FeatureEncoder()
        self.models: dict[str, object] = {}
        self.dispersion: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "UniversalGBDT":
        import lightgbm as lgb

        self.encoder.fit(frame)
        cat, numeric = self.encoder.transform(frame)
        X = np.column_stack([cat, numeric])
        level = frame["competition_level"].fillna("unknown").astype(str)
        sample_weight = np.ones(len(frame), dtype=float)
        if self.weighting == "level_balanced":
            domain_count = level.map(level.value_counts()).to_numpy(float)
            sample_weight *= len(frame) / np.maximum(domain_count, 1)
        elif self.weighting == "international_x2":
            sample_weight *= np.where(level.eq("international"), 2.0, 1.0)
        elif self.weighting == "international_x4":
            sample_weight *= np.where(level.eq("international"), 4.0, 1.0)
        elif self.weighting != "natural":
            raise ValueError(f"unknown weighting scheme {self.weighting!r}")
        if self.time_half_life_days:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            age = (dates.max() + pd.Timedelta(days=1) - dates).dt.days
            age = age.fillna(0).clip(lower=0).to_numpy(float)
            sample_weight *= np.power(0.5, age / float(self.time_half_life_days))
        sample_weight /= sample_weight.mean()
        targets = ("minutes",) + self.events
        for target in targets:
            mask_col = f"available__{target}"
            valid = frame[mask_col].fillna(False).to_numpy(bool) & frame[target].notna().to_numpy()
            if valid.sum() < 20:
                continue
            y = np.clip(frame.loc[valid, target].to_numpy(float), 0, None)
            family = distribution_family(target)
            kwargs = dict(
                n_estimators=self.n_estimators, learning_rate=.045,
                num_leaves=self.num_leaves,
                min_child_samples=35, reg_lambda=4.0, subsample=.85,
                subsample_freq=1, colsample_bytree=.8, random_state=self.random_state,
                n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
            )
            if family == "bernoulli":
                binary = (y > 0).astype(int)
                if binary.min() == binary.max():
                    model = _ConstantProbability(float(binary.mean()))
                else:
                    model = lgb.LGBMClassifier(objective="binary", **kwargs)
                    model.fit(X[valid], binary, sample_weight=sample_weight[valid])
                fitted = model.predict_proba(X[valid])[:, 1]
            else:
                objective = "tweedie" if family == "lognormal" else "poisson"
                model = lgb.LGBMRegressor(
                    objective=objective, tweedie_variance_power=1.35, **kwargs,
                )
                model.fit(X[valid], y, sample_weight=sample_weight[valid])
                fitted = np.clip(model.predict(X[valid]), 0, None)
            resid_var = float(np.mean((y - fitted) ** 2))
            mean = float(np.mean(fitted))
            self.dispersion[target] = max(resid_var, mean, 1e-4)
            self.models[target] = model
        if "minutes" not in self.models:
            raise ValueError("not enough observed minutes to fit universal model")
        return self

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        cat, numeric = self.encoder.transform(frame)
        X = np.column_stack([cat, numeric])
        means = {}
        for name, model in self.models.items():
            if distribution_family(name) == "bernoulli" and hasattr(model, "predict_proba"):
                means[name] = np.clip(model.predict_proba(X)[:, 1], 0, 1)
            else:
                means[name] = np.clip(model.predict(X), 0, None)
        output = []
        for i, row in enumerate(frame.itertuples(index=False)):
            events = {}
            for event in self.events:
                if event not in means:
                    continue
                mean = float(means[event][i])
                family = distribution_family(event)
                if family == "bernoulli":
                    mean = min(mean, 1.0)
                    dispersion = 1.0
                elif family == "negative_binomial":
                    var = max(self.dispersion[event], mean + 1e-6)
                    dispersion = max(mean * mean / max(var - mean, 1e-6), .05)
                else:
                    dispersion = self.dispersion[event]
                events[event] = EventDistribution(family, mean, dispersion)
            mmean = min(float(means["minutes"][i]), 80.0)
            return_minutes = EventDistribution("lognormal", mmean, self.dispersion["minutes"])
            output.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id),
                player_name=str(row.player_name), team=str(row.team), opponent=str(row.opponent),
                position=str(row.position), is_forward=bool(row.is_forward), events=events,
                minutes=return_minutes,
                metadata={"model": "universal_gbdt", "competition_level": row.competition_level},
            ))
        return output

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "UniversalGBDT":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not a UniversalGBDT artifact")
        return model
