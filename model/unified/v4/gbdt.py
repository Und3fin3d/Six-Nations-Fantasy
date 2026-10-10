from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FeatureEncoder, numeric_feature_columns
from ..gbdt import UniversalGBDT
from ..schema import distribution_family
from .features import EB_EVENTS, V4_BASE_NUMERIC, V4_FEATURE_PREFIXES, fit_shrinkage_k

EFFECT_BOUNDS = (0.25, 4.0)


class V4FeatureEncoder(FeatureEncoder):
    def columns(self, frame: pd.DataFrame) -> list[str]:
        base = numeric_feature_columns(frame)
        return base + [c for c in [c for c in V4_BASE_NUMERIC if c in frame] + [c for c in frame.columns if c.startswith(V4_FEATURE_PREFIXES)] if c not in base]


class V4GBDT(UniversalGBDT):
    model_name = "v4_gbdt"

    def __init__(
        self, *, pool_player_id: bool = False, player_effects: bool = False, hurdle_events: tuple[str, ...] = (), effect_min_rows: int = 500, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = V4FeatureEncoder()
        self.pool_player_id, self.player_effects, self.effect_min_rows, self.hurdle_events = (
            pool_player_id, player_effects, int(effect_min_rows), tuple(hurdle_events),
        )
        self.hurdle_models: dict[str, tuple] = {}
        self.effects: dict[tuple[str, str], float] = {}

    def _frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(player_id="pooled") if self.pool_player_id else frame

    def fit(self, frame: pd.DataFrame) -> "V4GBDT":
        import lightgbm as lgb

        super().fit(frame)
        X, fit_kwargs = self._design(frame)
        for event in self.hurdle_events:
            valid = frame[f"available__{event}"].fillna(False).to_numpy(bool) & frame[event].notna().to_numpy()
            if valid.sum() < 200:
                continue
            y = np.clip(frame.loc[valid, event].to_numpy(float), 0, None)
            positive = y > 0
            if positive.sum() < 50 or positive.all():
                continue
            clf = lgb.LGBMClassifier(objective="binary", **self._lgbm_kwargs())
            clf.fit(X[valid], positive.astype(int), **fit_kwargs)
            reg = lgb.LGBMRegressor(objective="poisson", **self._lgbm_kwargs())
            reg.fit(X[valid][positive], y[positive], **fit_kwargs)
            self.hurdle_models[event] = (clf, reg, float(np.mean((y[positive] - np.clip(reg.predict(X[valid][positive]), 1e-6, None)) ** 2)))
        if self.player_effects:
            minutes = pd.to_numeric(frame["minutes"], errors="coerce")
            k_by_event = fit_shrinkage_k(frame)
            for event in EB_EVENTS:
                model = self.models.get(event)
                valid = frame[f"available__{event}"].fillna(False).to_numpy(bool) & frame[event].notna().to_numpy() & minutes.gt(0).to_numpy()
                if model is None or distribution_family(event) == "bernoulli" or valid.sum() < self.effect_min_rows:
                    continue
                actual = np.clip(frame.loc[valid, event].to_numpy(float), 0, None)
                predicted = self._hurdle_means(event, X[valid])[0] if event in self.hurdle_models else np.clip(model.predict(X[valid]), 1e-6, None)
                agg = (
                    pd.DataFrame({"player": frame.loc[valid, "player_id"].astype(str).to_numpy(), "actual": actual, "pred": predicted}).groupby("player").sum()
                )
                alpha = max(k_by_event.get(event, 220.0) * (float(actual.sum() / max(minutes[valid].sum(), 1.0)) * 80.0) / 80.0, 0.5)
                for player, value in ((alpha + agg["actual"]) / (alpha + agg["pred"])).clip(*EFFECT_BOUNDS).items():
                    self.effects[(event, player)] = float(value)
        return self

    def _hurdle_means(self, event: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clf, reg, resid_var_pos = self.hurdle_models[event]
        p = np.clip(clf.predict_proba(X)[:, 1], 0.0, 1.0)
        m_pos = np.clip(reg.predict(X), 1e-6, None)
        mean = p * m_pos
        return mean, np.clip(p * (m_pos + resid_var_pos + m_pos**2) - mean**2, 1e-6, None)

    def _means(self, X: np.ndarray, frame: pd.DataFrame) -> tuple[dict, dict]:
        means, variances = super()._means(X, frame)
        for name in self.hurdle_models:
            means[name], variances[name] = self._hurdle_means(name, X)
        if self.player_effects:
            players = frame["player_id"].astype(str).to_numpy()
            for event in {e for e, _ in self.effects} & set(means):
                means[event] = means[event] * np.array([self.effects.get((event, player), 1.0) for player in players])
        return means, variances
