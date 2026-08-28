"""v4 GBDT: the v1 direct-totals architecture plus flag-gated structural fixes.

Flags off => behaviour is bit-identical to ``UniversalGBDT`` (regression-
guarded in tests). The three additive components:

- extended encoder that admits the v4 PIT feature columns (EB shrunk rates,
  level-split form, intl/club history counts);
- ``pool_player_id`` + ``player_effects``: fit without a player identity
  feature, then apply empirical-Bayes multiplicative per-player effects
  estimated from the training frame only (variant b);
- ``hurdle_events``: P(>=1) classifier x positive-count Poisson regressor,
  moment-matched into the existing negative_binomial ``EventDistribution`` so
  ``contracts.py``/``scoring.py`` are untouched (P2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction
from ..features import FeatureEncoder, numeric_feature_columns
from ..gbdt import UniversalGBDT
from ..schema import distribution_family
from .features import EB_EVENTS, V4_BASE_NUMERIC, V4_FEATURE_PREFIXES

EFFECT_BOUNDS = (0.25, 4.0)


class V4FeatureEncoder(FeatureEncoder):
    """Parent encoder plus the v4 numeric columns present in the frame."""

    def fit(self, frame: pd.DataFrame) -> "V4FeatureEncoder":
        super().fit(frame)
        extra = [c for c in V4_BASE_NUMERIC if c in frame] + [
            c for c in frame.columns if c.startswith(V4_FEATURE_PREFIXES)
        ]
        base = numeric_feature_columns(frame)
        self.numeric_columns = base + [c for c in extra if c not in base]
        numeric_frame = frame[self.numeric_columns].astype(float)
        matrix = numeric_frame.to_numpy()
        self.means = numeric_frame.mean(axis=0, skipna=True).fillna(0.0).to_numpy()
        filled = np.where(np.isnan(matrix), self.means, matrix)
        self.scales = np.std(filled, axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self


class V4GBDT(UniversalGBDT):
    def __init__(self, *, pool_player_id: bool = False, player_effects: bool = False,
                 hurdle_events: tuple[str, ...] = (), effect_min_rows: int = 500,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = V4FeatureEncoder()
        self.pool_player_id = pool_player_id
        self.player_effects = player_effects
        self.effect_min_rows = int(effect_min_rows)
        self.hurdle_events = tuple(hurdle_events)
        self.hurdle_models: dict[str, tuple] = {}
        self.effects: dict[tuple[str, str], float] = {}
        if player_effects and not pool_player_id:
            raise ValueError("player_effects requires pool_player_id "
                             "(the identity feature must not be double-counted)")

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pool_player_id:
            return frame.assign(player_id="pooled")
        return frame

    def fit(self, frame: pd.DataFrame) -> "V4GBDT":
        import lightgbm as lgb

        super().fit(self._frame(frame))
        if self.hurdle_events:
            cat, numeric = self.encoder.transform(self._frame(frame))
            X = np.column_stack([cat, numeric])
            kwargs = dict(
                n_estimators=self.n_estimators, learning_rate=.045,
                num_leaves=self.num_leaves, min_child_samples=35, reg_lambda=4.0,
                subsample=.85, subsample_freq=1, colsample_bytree=.8,
                random_state=self.random_state, n_jobs=-1, verbosity=-1,
                deterministic=True, force_col_wise=True,
            )
            for event in self.hurdle_events:
                if distribution_family(event) != "negative_binomial":
                    raise ValueError(f"hurdle head only supports count events, got {event}")
                valid = (frame[f"available__{event}"].fillna(False).to_numpy(bool)
                         & frame[event].notna().to_numpy())
                if valid.sum() < 200:
                    continue
                y = np.clip(frame.loc[valid, event].to_numpy(float), 0, None)
                positive = y > 0
                if positive.sum() < 50 or positive.all():
                    continue
                clf = lgb.LGBMClassifier(objective="binary", **kwargs)
                clf.fit(X[valid], positive.astype(int))
                reg = lgb.LGBMRegressor(objective="poisson", **kwargs)
                reg.fit(X[valid][positive], y[positive])
                fitted_pos = np.clip(reg.predict(X[valid][positive]), 1e-6, None)
                resid_var_pos = float(np.mean((y[positive] - fitted_pos) ** 2))
                self.hurdle_models[event] = (clf, reg, resid_var_pos)
        if self.player_effects:
            self._fit_effects(frame)
        return self

    def _fit_effects(self, frame: pd.DataFrame) -> None:
        """EB multiplicative per-player effects from the training frame only."""
        from .features import fit_shrinkage_k

        pooled = self._frame(frame)
        cat, numeric = self.encoder.transform(pooled)
        X = np.column_stack([cat, numeric])
        minutes = pd.to_numeric(frame["minutes"], errors="coerce")
        k_by_event = fit_shrinkage_k(frame)
        for event in EB_EVENTS:
            model = self.models.get(event)
            if model is None or distribution_family(event) == "bernoulli":
                continue
            valid = (frame[f"available__{event}"].fillna(False).to_numpy(bool)
                     & frame[event].notna().to_numpy() & minutes.gt(0).to_numpy())
            if valid.sum() < self.effect_min_rows:
                continue
            actual = np.clip(frame.loc[valid, event].to_numpy(float), 0, None)
            if event in self.hurdle_models:
                predicted = self._hurdle_means(event, X[valid])[0]
            else:
                predicted = np.clip(model.predict(X[valid]), 1e-6, None)
            players = frame.loc[valid, "player_id"].astype(str).to_numpy()
            table = pd.DataFrame({"player": players, "actual": actual, "pred": predicted})
            agg = table.groupby("player").sum()
            # Prior mass = expected events over K moment-matched minutes at the
            # pooled rate (same K machinery as the feature variant; not pinned).
            pooled_rate = float(actual.sum() / max(minutes[valid].sum(), 1.0)) * 80.0
            alpha = max(k_by_event.get(event, 220.0) * pooled_rate / 80.0, 0.5)
            effect = (alpha + agg["actual"]) / (alpha + agg["pred"])
            effect = effect.clip(*EFFECT_BOUNDS)
            for player, value in effect.items():
                self.effects[(event, player)] = float(value)

    def _hurdle_means(self, event: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clf, reg, resid_var_pos = self.hurdle_models[event]
        p = np.clip(clf.predict_proba(X)[:, 1], 0.0, 1.0)
        m_pos = np.clip(reg.predict(X), 1e-6, None)
        mean = p * m_pos
        var_pos = m_pos + resid_var_pos
        var = np.clip(p * (var_pos + m_pos ** 2) - mean ** 2, 1e-6, None)
        return mean, var

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        pooled = self._frame(frame)
        cat, numeric = self.encoder.transform(pooled)
        X = np.column_stack([cat, numeric])
        means: dict[str, np.ndarray] = {}
        variances: dict[str, np.ndarray] = {}
        for name, model in self.models.items():
            if name in self.hurdle_models:
                means[name], variances[name] = self._hurdle_means(name, X)
            elif distribution_family(name) == "bernoulli" and hasattr(model, "predict_proba"):
                means[name] = np.clip(model.predict_proba(X)[:, 1], 0, 1)
            else:
                means[name] = np.clip(model.predict(X), 0, None)
        if self.player_effects:
            players = frame["player_id"].astype(str).to_numpy()
            for event in set(e for e, _ in self.effects):
                if event not in means:
                    continue
                factor = np.array([
                    self.effects.get((event, player), 1.0) for player in players
                ])
                means[event] = means[event] * factor
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
                    if event in variances:
                        var = max(float(variances[event][i]), mean + 1e-6)
                    else:
                        var = max(self.dispersion[event], mean + 1e-6)
                    dispersion = max(mean * mean / max(var - mean, 1e-6), .05)
                else:
                    dispersion = self.dispersion[event]
                events[event] = EventDistribution(family, mean, dispersion)
            mmean = min(float(means["minutes"][i]), 80.0)
            return_minutes = EventDistribution("lognormal", mmean, self.dispersion["minutes"])
            output.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id),
                player_name=str(row.player_name), team=str(row.team),
                opponent=str(row.opponent), position=str(row.position),
                is_forward=bool(row.is_forward), events=events, minutes=return_minutes,
                metadata={"model": "v4_gbdt", "competition_level": row.competition_level},
            ))
        return output
