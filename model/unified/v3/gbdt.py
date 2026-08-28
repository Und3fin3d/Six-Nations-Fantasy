"""Exposure-aware universal LightGBM model for raw rugby events."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction
from ..schema import EVENTS, distribution_family
from .config import GBDTV3Config
from .context import V3FeatureEncoder


@dataclass
class _ConstantProbability:
    probability: float

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.probability, 0, 1), dtype=float)
        return np.column_stack([1 - p, p])


def training_weights(frame: pd.DataFrame, weighting: str,
                     half_life_days: float | None) -> np.ndarray:
    weights = np.ones(len(frame), dtype=float)
    level = frame["competition_level"].fillna("unknown").astype(str)
    if weighting == "level_balanced":
        counts = level.map(level.value_counts()).to_numpy(float)
        weights *= len(frame) / np.maximum(counts, 1)
    elif weighting == "international_x2":
        weights *= np.where(level.eq("international"), 2.0, 1.0)
    elif weighting == "international_x4":
        weights *= np.where(level.eq("international"), 4.0, 1.0)
    elif weighting != "natural":
        raise ValueError(f"unknown weighting scheme {weighting!r}")
    if half_life_days:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        asof = dates.max() + pd.Timedelta(days=1)
        age = (asof - dates).dt.days.fillna(0).clip(lower=0).to_numpy(float)
        weights *= np.power(0.5, age / float(half_life_days))
    weights /= max(float(weights.mean()), 1e-12)
    return weights


@dataclass
class ExposureRateGBDT:
    """Predict play/minutes exposure, then per-80 rates and rare binaries."""

    config: GBDTV3Config = field(default_factory=GBDTV3Config)
    events: tuple[str, ...] = EVENTS
    encoder: V3FeatureEncoder = field(default_factory=V3FeatureEncoder)
    appearance_model: object | None = None
    minutes_model: object | None = None
    event_models: dict[str, object] = field(default_factory=dict)
    event_caps: dict[str, float] = field(default_factory=dict)
    dispersion: dict[str, float] = field(default_factory=dict)
    active_events: set[str] = field(default_factory=set)

    def _regressor(self, objective: str):
        import lightgbm as lgb
        kwargs = dict(
            objective=objective,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            reg_lambda=self.config.reg_lambda,
            subsample=self.config.subsample,
            subsample_freq=1,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.seed,
            n_jobs=-1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        if objective == "tweedie":
            kwargs["tweedie_variance_power"] = 1.35
        return lgb.LGBMRegressor(**kwargs)

    def fit(self, frame: pd.DataFrame) -> "ExposureRateGBDT":
        import lightgbm as lgb

        self.encoder.fit(frame)
        cat, numeric = self.encoder.transform(frame)
        X = np.column_stack([cat, numeric])
        base_weight = training_weights(
            frame, self.config.weighting, self.config.time_half_life_days,
        )
        minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0).to_numpy(float)
        minute_available = frame["available__minutes"].fillna(False).astype(bool).to_numpy()
        played = minutes > 0

        if self.config.use_appearance and minute_available.sum() >= 20 and played.any() and (~played).any():
            self.appearance_model = lgb.LGBMClassifier(
                objective="binary", n_estimators=max(120, self.config.n_estimators // 2),
                learning_rate=self.config.learning_rate, num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                reg_lambda=self.config.reg_lambda, subsample=self.config.subsample,
                subsample_freq=1,
                colsample_bytree=self.config.colsample_bytree, random_state=self.config.seed,
                n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
            )
            self.appearance_model.fit(
                X[minute_available], played[minute_available].astype(int),
                sample_weight=base_weight[minute_available],
            )
        else:
            self.appearance_model = _ConstantProbability(float(played[minute_available].mean()))

        positive = minute_available & played
        if positive.sum() < 20:
            raise ValueError("not enough positive-minute rows for v3 minutes model")
        self.minutes_model = self._regressor("tweedie")
        self.minutes_model.fit(
            X[positive], minutes[positive], sample_weight=base_weight[positive],
        )

        self.event_models = {}
        self.event_caps = {}
        self.dispersion = {}
        self.active_events = set()
        for event in self.events:
            available = frame[f"available__{event}"].fillna(False).astype(bool).to_numpy()
            valid = available & positive
            if valid.sum() < 20:
                continue
            truth = pd.to_numeric(frame.loc[valid, event], errors="coerce").to_numpy(float)
            family = distribution_family(event)
            if family == "bernoulli":
                positives = int((truth > 0).sum())
                if positives < 2 or positives == len(truth):
                    model = _ConstantProbability(float(np.mean(truth > 0)))
                else:
                    model = lgb.LGBMClassifier(
                        objective="binary", n_estimators=max(120, self.config.n_estimators // 2),
                        learning_rate=self.config.learning_rate, num_leaves=self.config.num_leaves,
                        min_child_samples=self.config.min_child_samples,
                        reg_lambda=self.config.reg_lambda, subsample=self.config.subsample,
                        subsample_freq=1,
                        colsample_bytree=self.config.colsample_bytree,
                        random_state=self.config.seed, n_jobs=-1, verbosity=-1,
                        deterministic=True, force_col_wise=True,
                    )
                    model.fit(
                        X[valid], (truth > 0).astype(int), sample_weight=base_weight[valid],
                    )
                fitted = model.predict_proba(X[valid])[:, 1]
                cap = 1.0
                event_mean = fitted
            else:
                rate = truth / np.maximum(minutes[valid], 1e-6) * 80.0
                cap = float(np.quantile(rate, self.config.rate_cap_quantile))
                rate = np.clip(rate, 0, max(cap, 1e-6))
                objective = "tweedie" if family == "lognormal" else "poisson"
                model = self._regressor(objective)
                exposure_weight = base_weight[valid] * np.clip(minutes[valid] / 80.0, 0.05, 1.0)
                model.fit(X[valid], rate, sample_weight=exposure_weight)
                fitted_rate = np.clip(model.predict(X[valid]), 0, cap)
                event_mean = fitted_rate * minutes[valid] / 80.0
            residual_var = float(np.mean((truth - event_mean) ** 2))
            mean = float(np.mean(event_mean))
            self.event_models[event] = model
            self.event_caps[event] = cap
            self.dispersion[event] = max(residual_var, mean, 1e-4)
            self.active_events.add(event)
        return self

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        if self.minutes_model is None:
            raise RuntimeError("ExposureRateGBDT is not fit")
        cat, numeric = self.encoder.transform(frame)
        X = np.column_stack([cat, numeric])
        p_play = self.appearance_model.predict_proba(X)[:, 1]
        conditional_minutes = np.clip(self.minutes_model.predict(X), 1, 80)
        expected_minutes = np.clip(p_play * conditional_minutes, 0, 80)
        means = {}
        for event, model in self.event_models.items():
            family = distribution_family(event)
            if family == "bernoulli":
                means[event] = np.clip(model.predict_proba(X)[:, 1] * p_play, 0, 1)
            else:
                rate = np.clip(model.predict(X), 0, self.event_caps[event])
                means[event] = np.clip(rate * expected_minutes / 80.0, 0, None)

        output = []
        for i, row in enumerate(frame.itertuples(index=False)):
            events = {}
            for event in self.active_events:
                mean = float(means[event][i])
                family = distribution_family(event)
                if family == "bernoulli":
                    dispersion = 1.0
                elif family == "negative_binomial":
                    var = max(self.dispersion[event], mean + 1e-6)
                    dispersion = max(mean * mean / max(var - mean, 1e-6), 0.05)
                else:
                    dispersion = max(self.dispersion[event], 1e-6)
                events[event] = EventDistribution(family, mean, dispersion)
            mmean = float(expected_minutes[i])
            output.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id),
                player_name=str(row.player_name), team=str(row.team), opponent=str(row.opponent),
                position=str(row.position), is_forward=bool(row.is_forward),
                events=events,
                minutes=EventDistribution("lognormal", mmean, max(25.0, mmean * 0.5)),
                metadata={
                    "model": "exposure_rate_gbdt_v3",
                    "p_play": float(p_play[i]),
                    "conditional_minutes": float(conditional_minutes[i]),
                    "competition_level": str(row.competition_level),
                },
            ))
        return output

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "ExposureRateGBDT":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not an ExposureRateGBDT artifact")
        return model
