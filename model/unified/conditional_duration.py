import numpy as np
import pandas as pd

from .contracts import EventDistribution, RawPrediction
from .schema import distribution_family
from .v4.features import EB_EVENTS, fit_shrinkage_k
from .v4.gbdt import EFFECT_BOUNDS, V4GBDT


class ConstantMean:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, matrix):
        return np.full(len(matrix), self.value)

    def predict_proba(self, matrix):
        value = self.predict(matrix)
        return np.column_stack([1 - value, value])


class ConditionalDurationGBDT(V4GBDT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hurdle_events or self.weighting != 'natural' or self.time_half_life_days:
            raise ValueError('This registered model supports only natural unweighted conditional heads')
        self.duration_model = None
        self.duration_classes = np.array([], dtype=int)
        self.duration_centres = np.zeros(10)
        self.duration_variances = np.zeros(10)
        self.duration_counts = np.zeros(10, dtype=int)
        self.training_support = []

    def _parameters(self):
        return dict(n_estimators=self.n_estimators, learning_rate=.045, num_leaves=self.num_leaves,
                    min_child_samples=35, reg_lambda=4.0, subsample=.85, subsample_freq=1,
                    colsample_bytree=.8, random_state=self.random_state, n_jobs=2, verbosity=-1,
                    deterministic=True, force_col_wise=True)

    def _encoded(self, frame):
        categories, numeric = self.encoder.transform(self._frame(frame))
        matrix = np.column_stack([categories, numeric])
        arguments = {'categorical_feature': list(range(categories.shape[1]))} if self.native_categories else {}
        return matrix, arguments

    @staticmethod
    def _duration_labels(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 80)):
            raise ValueError('Observed duration outside the registered [0,80] contract')
        return np.where(values == 0, 0, np.minimum(9, np.floor(values / 10).astype(int) + 1))

    def _fit_duration(self, matrix, minutes, valid, arguments):
        import lightgbm as lgb

        if valid.sum() < 20:
            raise ValueError('Insufficient observed duration for the registered model')
        y = minutes[valid]
        labels = self._duration_labels(y)
        self.duration_classes = np.unique(labels)
        for label in self.duration_classes:
            subset = y[labels == label]
            self.duration_centres[label] = subset.mean()
            self.duration_variances[label] = subset.var()
            self.duration_counts[label] = len(subset)
        if len(self.duration_classes) > 1:
            self.duration_model = lgb.LGBMClassifier(objective='multiclass', **self._parameters())
            self.duration_model.fit(matrix[valid], labels, **arguments)

    def _fit_event(self, frame, matrix, event, duration_valid, minutes, arguments):
        import lightgbm as lgb

        observed = frame[f'available__{event}'].fillna(False).to_numpy(bool) & frame[event].notna().to_numpy()
        valid = observed & duration_valid
        support = frame[['competition_level', 'started']].copy()
        support['event_observed'] = observed
        support['joint_observed'] = valid
        for (level, started), group in support.groupby(['competition_level', 'started'], dropna=False):
            self.training_support.append(dict(event=event, level=str(level), started=bool(started),
                                              rows=len(group), event_observed=int(group.event_observed.sum()),
                                              joint_observed=int(group.joint_observed.sum()),
                                              excluded_unknown_duration=int((group.event_observed & ~group.joint_observed).sum())))
        if valid.sum() < 20:
            return
        y = np.maximum(frame.loc[valid, event].to_numpy(float), 0)
        conditional = np.column_stack([matrix[valid], minutes[valid] / 80])
        family = distribution_family(event)
        if family == 'bernoulli':
            y = (y > 0).astype(int)
        if np.all(y == y[0]):
            model = ConstantMean(y[0])
        elif family == 'bernoulli':
            model = lgb.LGBMClassifier(objective='binary', **self._parameters())
            model.fit(conditional, y, **arguments)
        else:
            objective = 'tweedie' if family == 'lognormal' else 'poisson'
            model = lgb.LGBMRegressor(objective=objective, tweedie_variance_power=1.35, **self._parameters())
            model.fit(conditional, y, **arguments)
        fitted = self._mean_prediction(model, conditional, family)
        self.models[event] = model
        self.dispersion[event] = max(float(np.mean(np.square(y - fitted))), float(fitted.mean()), 1e-4)

    def fit(self, frame):
        self.models, self.dispersion, self.effects = {}, {}, {}
        self.training_support = []
        self.encoder.fit(self._frame(frame))
        matrix, arguments = self._encoded(frame)
        minutes = pd.to_numeric(frame.minutes, errors='coerce').to_numpy(float)
        valid = frame.available__minutes.fillna(False).to_numpy(bool) & np.isfinite(minutes)
        self._fit_duration(matrix, minutes, valid, arguments)
        for event in self.events:
            self._fit_event(frame, matrix, event, valid, minutes, arguments)
        if self.player_effects:
            self._fit_conditional_effects(frame, matrix, minutes, valid)
        return self

    @staticmethod
    def _mean_prediction(model, matrix, family):
        if family == 'bernoulli':
            return np.clip(model.predict_proba(matrix)[:, 1], 0, 1)
        return np.maximum(model.predict(matrix), 0)

    def _fit_conditional_effects(self, frame, matrix, minutes, duration_valid):
        shrinkage = fit_shrinkage_k(frame)
        for event in EB_EVENTS:
            if event not in self.models or distribution_family(event) == 'bernoulli':
                continue
            valid = (duration_valid & (minutes > 0) & frame[f'available__{event}'].fillna(False).to_numpy(bool)
                     & frame[event].notna().to_numpy())
            if valid.sum() < self.effect_min_rows:
                continue
            actual = np.maximum(frame.loc[valid, event].to_numpy(float), 0)
            conditional = np.column_stack([matrix[valid], minutes[valid] / 80])
            predicted = np.maximum(self.models[event].predict(conditional), 1e-6)
            grouped = pd.DataFrame(dict(player=frame.loc[valid, 'player_id'].astype(str).to_numpy(),
                                        actual=actual, predicted=predicted)).groupby('player').sum()
            rate = float(actual.sum() / max(minutes[valid].sum(), 1.0))
            alpha = max(shrinkage.get(event, 220.0) * rate, 0.5)
            effects = ((alpha + grouped.actual) / (alpha + grouped.predicted)).clip(*EFFECT_BOUNDS)
            self.effects.update({(event, player): float(value) for player, value in effects.items()})

    def duration_probabilities(self, frame):
        matrix, _ = self._encoded(frame)
        probabilities = np.zeros((len(frame), 10))
        if self.duration_model is None:
            if len(self.duration_classes) != 1:
                raise RuntimeError('ConditionalDurationGBDT has not been fitted')
            probabilities[:, self.duration_classes[0]] = 1
        else:
            probabilities[:, self.duration_model.classes_.astype(int)] = self.duration_model.predict_proba(matrix)
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError('Invalid pre-match duration probability distribution')
        return probabilities

    def predict_scenarios(self, frame):
        matrix, _ = self._encoded(frame)
        probabilities = self.duration_probabilities(frame)
        players = frame.player_id.astype(str).to_numpy()
        means, variances = {}, {}
        for event, model in self.models.items():
            family = distribution_family(event)
            values = np.zeros((len(frame), 10))
            for label in self.duration_classes:
                scenario = np.column_stack([matrix, np.full(len(frame), self.duration_centres[label] / 80)])
                values[:, label] = self._mean_prediction(model, scenario, family)
            if self.player_effects:
                values *= np.array([self.effects.get((event, player), 1.0) for player in players])[:, None]
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'Invalid conditional means: {event}')
            means[event] = values
            if family == 'bernoulli':
                variances[event] = values * (1 - values)
            elif family == 'negative_binomial':
                variances[event] = np.maximum(self.dispersion[event], values + 1e-6)
            else:
                variances[event] = np.full_like(values, self.dispersion[event])
        return probabilities, means, variances

    @staticmethod
    def _moments(probabilities, means, variances):
        mean = (probabilities * means).sum(axis=1)
        variance = np.maximum((probabilities * (variances + means * means)).sum(axis=1) - mean * mean, 1e-8)
        return mean, variance

    @staticmethod
    def _distribution(event, mean, variance):
        family = distribution_family(event)
        if family == 'bernoulli':
            return EventDistribution(family, min(float(mean), 1.0), 1.0)
        if family == 'negative_binomial':
            dispersion = max(float(mean * mean / max(variance - mean, 1e-6)), .05)
        else:
            dispersion = max(float(variance), 1e-8)
        return EventDistribution(family, float(mean), dispersion)

    def predict_frame(self, frame):
        probabilities, conditional_means, conditional_variances = self.predict_scenarios(frame)
        moments = {event: self._moments(probabilities, values, conditional_variances[event])
                   for event, values in conditional_means.items()}
        minute_mean, minute_variance = self._moments(probabilities, self.duration_centres[None, :], self.duration_variances[None, :])
        output = []
        for index, row in enumerate(frame.itertuples(index=False)):
            events = {event: self._distribution(event, values[0][index], values[1][index]) for event, values in moments.items()}
            output.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=str(row.player_id), player_name=str(row.player_name),
                team=str(row.team), opponent=str(row.opponent), position=str(row.position), is_forward=bool(row.is_forward),
                events=events, minutes=EventDistribution('lognormal', float(minute_mean[index]), float(minute_variance[index])),
                metadata=dict(model='conditional_duration_v1', competition_level=row.competition_level,
                              duration_probabilities=probabilities[index].tolist(), duration_centres=self.duration_centres.tolist(),
                              duration_variances=self.duration_variances.tolist(),
                              conditional_event_means={event: values[index].tolist() for event, values in conditional_means.items()},
                              conditional_event_variances={event: values[index].tolist() for event, values in conditional_variances.items()})))
        return output
