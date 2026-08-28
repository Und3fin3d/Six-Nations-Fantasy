"""Shrunk-form direct-totals GBDT — the v5 candidate model (Family A).

Same target structure as the v1 universal control
(`model/unified/gbdt.py::UniversalGBDT`): direct per-event match totals fit on
all rows with available labels (zero-minute rows included), Poisson / tweedie
/ bernoulli heads, minutes as a direct target, deterministic scorers
downstream. Differences, per `data/unified/v5/RESEARCH_PLAN.md` §4:

1. The feature layer *can* add the empirical-Bayes shrunk-rate and
   slot-minute features of :mod:`model.unified.v5.shrinkage` (tables fit per
   training window, row features shifted/ewm — both point-in-time safe).
   **Post-benchmark default is terminal rung T: both layers off** — the
   frozen-protocol benchmark fired kill-switch K1 (no low-history gain) and
   showed the layers compress the NCR top-10 boundary against try-scoring
   backs. See `data/unified/v5/corrective_cycle.md`.
2. Rare-event guards: a count head is fit only with >= ``min_valid_rows``
   rows and >= ``min_positives`` positives; otherwise a level-stratified
   global mean is emitted (kills the v3 degenerate rare-event regime).
3. Attribution-corrupted events (default on, e.g. ``lineouts_won``) are
   excluded from both fitting and prediction; the unchanged deterministic
   scorers read absent events as zero.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction
from ..features import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_pit_features
from ..schema import EVENTS, distribution_family
from .config import V5Config
from .shrinkage import ShrunkTables, add_shrunk_features, fit_tables, shrunk_feature_columns

LEGACY_PREFIXES = ("form_per80__", "history_count__")
CONTEXT_PREFIXES = ("rolecert_", "style_", "weather_", "wr_")


def training_weights(frame: pd.DataFrame, weighting: str,
                     half_life_days: float | None) -> np.ndarray:
    """Row weights identical in spirit to the v3/v1 schemes (self-contained)."""
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


def numeric_feature_columns(frame: pd.DataFrame, keep_legacy: bool) -> list[str]:
    base = [c for c in BASE_NUMERIC_FEATURES if c in frame]
    legacy = [
        c for c in frame
        if keep_legacy and c.startswith(LEGACY_PREFIXES)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    context = [
        c for c in frame
        if c.startswith(CONTEXT_PREFIXES) and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return list(dict.fromkeys(
        base + legacy + shrunk_feature_columns(frame) + context
    ))


@dataclass
class V5FeatureEncoder:
    """Train-only category vocabularies and numeric normalisation."""

    categories: dict[str, dict[str, int]] | None = None
    numeric_columns: list[str] | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    keep_legacy: bool = True

    def fit(self, frame: pd.DataFrame) -> "V5FeatureEncoder":
        self.categories = {}
        for col in CATEGORICAL_FEATURES:
            values = frame.get(col, pd.Series("unknown", index=frame.index))
            values = values.fillna("unknown").astype(str)
            self.categories[col] = {
                value: i + 1 for i, value in enumerate(sorted(values.unique()))
            }
        self.numeric_columns = numeric_feature_columns(frame, self.keep_legacy)
        numeric = frame.reindex(columns=self.numeric_columns).apply(
            pd.to_numeric, errors="coerce")
        matrix = numeric.to_numpy(float)
        self.means = numeric.mean(axis=0, skipna=True).fillna(0.0).to_numpy()
        filled = np.where(np.isnan(matrix), self.means, matrix)
        self.scales = np.std(filled, axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.categories is None or self.numeric_columns is None:
            raise RuntimeError("V5FeatureEncoder must be fit before transform")
        cat = np.column_stack([
            frame.get(col, pd.Series("unknown", index=frame.index))
            .fillna("unknown").astype(str).map(mapping).fillna(0).astype(int).to_numpy()
            for col, mapping in self.categories.items()
        ])
        numeric = frame.reindex(columns=self.numeric_columns).apply(
            pd.to_numeric, errors="coerce")
        matrix = numeric.to_numpy(float)
        filled = np.where(np.isnan(matrix), self.means, matrix)
        return cat, (filled - self.means) / self.scales


class _ConstantProbability:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        p = np.full(len(X), np.clip(self.probability, 0, 1), dtype=float)
        return np.column_stack([1 - p, p])


@dataclass
class ShrunkFormGBDT:
    """Direct-totals per-event GBDT with empirical-Bayes shrunk features."""

    config: V5Config = field(default_factory=V5Config)
    events: tuple[str, ...] = EVENTS
    encoder: V5FeatureEncoder = field(default_factory=V5FeatureEncoder)
    tables: ShrunkTables = field(default_factory=ShrunkTables)
    models: dict[str, object] = field(default_factory=dict)
    dispersion: dict[str, float] = field(default_factory=dict)
    fallback_means: dict[str, dict[str, float]] = field(default_factory=dict)
    active_events: set[str] = field(default_factory=set)

    def _feature_layers_on(self) -> bool:
        return bool(self.config.use_eb_features or self.config.use_slot_minutes)

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
            n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
        )
        if objective == "tweedie":
            kwargs["tweedie_variance_power"] = 1.35
        return lgb.LGBMRegressor(**kwargs)

    def _classifier(self, **overrides):
        import lightgbm as lgb
        kwargs = dict(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            reg_lambda=self.config.reg_lambda,
            subsample=self.config.subsample,
            subsample_freq=1,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.seed,
            n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
        )
        kwargs.update(overrides)
        return lgb.LGBMClassifier(**kwargs)

    def _build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        pit = build_pit_features(frame)
        if not self._feature_layers_on():
            return pit
        return add_shrunk_features(pit, self.tables, self.config)

    def fit(self, frame: pd.DataFrame) -> "ShrunkFormGBDT":
        # Tables are fit on exactly the rows the model trains on; callers must
        # pass a frame already restricted to the fold's training window. With
        # both feature layers off (terminal rung T) no tables are needed.
        if self._feature_layers_on():
            self.tables = fit_tables(frame, self.config)
        feats = self._build_features(frame)
        self.encoder.keep_legacy = self.config.keep_legacy_form_features
        self.encoder.fit(feats)
        cat, numeric = self.encoder.transform(feats)
        X = np.column_stack([cat, numeric])
        sample_weight = training_weights(
            feats, self.config.weighting, self.config.time_half_life_days,
        )
        excluded = set(self.config.excluded_events())
        targets = ("minutes",) + tuple(e for e in self.events if e not in excluded)
        level = feats["competition_level"].fillna("unknown").astype(str)

        self.models, self.dispersion, self.fallback_means = {}, {}, {}
        self.active_events = set()
        for target in targets:
            mask_col = f"available__{target}"
            available = feats.get(mask_col, pd.Series(False, index=feats.index))
            valid = (
                available.fillna(False).astype(bool).to_numpy()
                & feats[target].notna().to_numpy()
            )
            family = distribution_family(target)
            y_all = pd.to_numeric(feats[target], errors="coerce")
            # Level-stratified marginal means for guarded fallbacks.
            means: dict[str, float] = {}
            if valid.any():
                for lvl in sorted(level[valid].unique()):
                    block = valid & level.eq(lvl).to_numpy()
                    if block.any():
                        means[str(lvl)] = float(y_all[block].mean())
                means["__all__"] = float(y_all[valid].mean())
            self.fallback_means[target] = means

            positives = int((y_all[valid] > 0).sum()) if valid.any() else 0
            fit_head = valid.sum() >= self.config.min_valid_rows and (
                family in {"lognormal", "bernoulli"}
                or positives >= self.config.min_positives
            )
            if not fit_head:
                var = float(y_all[valid].var()) if valid.sum() > 1 else 1.0
                self.dispersion[target] = (
                    var if np.isfinite(var) and var > 1e-4 else 1.0
                )
                if target == "minutes":
                    raise ValueError("not enough observed minutes to fit v5 model")
                self.active_events.add(target)
                continue
            y = np.clip(y_all[valid].to_numpy(float), 0, None)
            if family == "bernoulli":
                binary = (y > 0).astype(int)
                if binary.min() == binary.max():
                    model = _ConstantProbability(float(binary.mean()))
                else:
                    model = self._classifier()
                    model.fit(X[valid], binary, sample_weight=sample_weight[valid])
                fitted = model.predict_proba(X[valid])[:, 1]
            else:
                objective = "tweedie" if family == "lognormal" else "poisson"
                model = self._regressor(objective)
                model.fit(X[valid], y, sample_weight=sample_weight[valid])
                fitted = np.clip(model.predict(X[valid]), 0, None)
            resid_var = float(np.mean((y - fitted) ** 2))
            mean = float(np.mean(fitted))
            self.dispersion[target] = max(resid_var, mean, 1e-4)
            self.models[target] = model
            self.active_events.add(target)
        return self

    def _predict_mean(self, target: str, X: np.ndarray,
                      level: pd.Series) -> np.ndarray:
        model = self.models.get(target)
        if model is None:
            means = self.fallback_means.get(target, {"__all__": 0.0})
            return level.map(means).fillna(means.get("__all__", 0.0)).to_numpy(float)
        if distribution_family(target) == "bernoulli" and hasattr(model, "predict_proba"):
            return np.clip(model.predict_proba(X)[:, 1], 0, 1)
        return np.clip(model.predict(X), 0, None)

    def predict_features(self, feats: pd.DataFrame) -> list[RawPrediction]:
        """Predict from already point-in-time-safe features.

        This is used by historical fixed-tournament holdouts, where candidate
        rows must not update later candidate rows. Feature-layer configurations
        still require their shrunk columns to be supplied by the caller.
        """
        if "minutes" not in self.models:
            raise RuntimeError("ShrunkFormGBDT is not fit")
        cat, numeric = self.encoder.transform(feats)
        X = np.column_stack([cat, numeric])
        level = feats["competition_level"].fillna("unknown").astype(str)
        targets = set(self.active_events) | {"minutes"}
        means = {target: self._predict_mean(target, X, level) for target in targets}
        output = []
        for i, row in enumerate(feats.itertuples(index=False)):
            events = {}
            for event in sorted(self.active_events):
                if event == "minutes":
                    continue
                mean = float(means[event][i])
                family = distribution_family(event)
                if family == "bernoulli":
                    mean = min(mean, 1.0)
                    dispersion = 1.0
                elif family == "negative_binomial":
                    var = max(self.dispersion[event], mean + 1e-6)
                    dispersion = max(mean * mean / max(var - mean, 1e-6), 0.05)
                else:
                    dispersion = max(self.dispersion[event], 1e-6)
                events[event] = EventDistribution(family, mean, dispersion)
            mmean = min(float(means["minutes"][i]), 80.0)
            shrink_w = getattr(row, "shrink_weight__tackles", None)
            output.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id),
                player_name=str(row.player_name), team=str(row.team),
                opponent=str(row.opponent), position=str(row.position),
                is_forward=bool(row.is_forward), events=events,
                minutes=EventDistribution(
                    "lognormal", mmean, self.dispersion.get("minutes", 25.0)),
                metadata={
                    "model": "shrunk_form_gbdt_v5",
                    "shrink_weight_tackles": (
                        float(shrink_w)
                        if shrink_w is not None and np.isfinite(shrink_w) else None
                    ),
                    "competition_level": str(row.competition_level),
                },
            ))
        return output

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        return self.predict_features(self._build_features(frame))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "ShrunkFormGBDT":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not a ShrunkFormGBDT artifact")
        return model
