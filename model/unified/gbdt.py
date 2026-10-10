from __future__ import annotations

import os
import pickle

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
    model_name = "universal_gbdt"

    def __init__(self, events: tuple[str, ...] = EVENTS, weighting: str = "natural", n_estimators: int = 180, native_categories: bool = False):
        self.events, self.weighting, self.n_estimators, self.native_categories = tuple(events), weighting, n_estimators, bool(native_categories)
        self.encoder = FeatureEncoder()
        self.models: dict[str, object] = {}
        self.dispersion: dict[str, float] = {}

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def _matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, int]:
        cat, numeric = self.encoder.transform(self._frame(frame))
        return np.column_stack([cat, numeric]), cat.shape[1]

    def _design(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
        X, n_cat = self._matrix(frame)
        return X, {"categorical_feature": list(range(n_cat))} if self.native_categories else {}

    def _lgbm_kwargs(self) -> dict:
        return dict(
            n_estimators=self.n_estimators, learning_rate=0.045, num_leaves=23, min_child_samples=35, reg_lambda=4.0, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, random_state=17, n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
        )

    def fit(self, frame: pd.DataFrame) -> "UniversalGBDT":
        import lightgbm as lgb

        self.encoder.fit(self._frame(frame))
        X, fit_kwargs = self._design(frame)
        sample_weight = np.ones(len(frame), dtype=float)
        for target in ("minutes",) + self.events:
            valid = frame[f"available__{target}"].fillna(False).to_numpy(bool) & frame[target].notna().to_numpy()
            if valid.sum() < 20:
                continue
            y = np.clip(frame.loc[valid, target].to_numpy(float), 0, None)
            family = distribution_family(target)
            if family == "bernoulli":
                binary = (y > 0).astype(int)
                if binary.min() == binary.max():
                    model = _ConstantProbability(float(binary.mean()))
                else:
                    model = lgb.LGBMClassifier(objective="binary", **self._lgbm_kwargs())
                    model.fit(X[valid], binary, sample_weight=sample_weight[valid], **fit_kwargs)
                fitted = model.predict_proba(X[valid])[:, 1]
            else:
                model = lgb.LGBMRegressor(objective="tweedie" if family == "lognormal" else "poisson", tweedie_variance_power=1.35, **self._lgbm_kwargs())
                model.fit(X[valid], y, sample_weight=sample_weight[valid], **fit_kwargs)
                fitted = np.clip(model.predict(X[valid]), 0, None)
            self.dispersion[target] = max(float(np.mean((y - fitted) ** 2)), float(np.mean(fitted)), 1e-4)
            self.models[target] = model
        return self

    def _means(self, X: np.ndarray, frame: pd.DataFrame) -> tuple[dict, dict]:
        means = {
            name: np.clip(model.predict_proba(X)[:, 1], 0, 1)
            if distribution_family(name) == "bernoulli" and hasattr(model, "predict_proba")
            else np.clip(model.predict(X), 0, None)
            for name, model in self.models.items()
        }
        return means, {}

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        means, variances = self._means(self._matrix(frame)[0], frame)
        output = []
        for i, row in enumerate(frame.itertuples(index=False)):
            events = {}
            for event in (e for e in self.events if e in means):
                mean, family = float(means[event][i]), distribution_family(event)
                if family == "bernoulli":
                    mean, dispersion = min(mean, 1.0), 1.0
                elif family == "negative_binomial":
                    var = max(float(variances[event][i]) if event in variances else self.dispersion[event], mean + 1e-6)
                    dispersion = max(mean * mean / max(var - mean, 1e-6), 0.05)
                else:
                    dispersion = self.dispersion[event]
                events[event] = EventDistribution(family, mean, dispersion)
            output.append(
                RawPrediction(
                    fixture_id=str(row.fixture_id), player_id=str(row.player_id), player_name=str(row.player_name), team=str(row.team),
                    opponent=str(row.opponent), position=str(row.position), is_forward=bool(row.is_forward), events=events,
                    minutes=EventDistribution("lognormal", min(float(means["minutes"][i]), 80.0), self.dispersion["minutes"]),
                    metadata={"model": self.model_name, "competition_level": row.competition_level},
                )
            )
        return output

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> "UniversalGBDT":
        with open(path, "rb") as handle:
            return pickle.load(handle)
