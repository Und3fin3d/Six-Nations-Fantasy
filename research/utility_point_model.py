import copy
import dataclasses

import lightgbm as lgb
import numpy as np

from model.unified.rolling_eval import expected_points
from model.unified.scoring import NationsChampionshipScorer
from model.unified.v4.gbdt import V4GBDT
from research.conditional_duration_metrics import only_metres_points
from research.duration_evidence_validation import close
from research.ncr_rubric_completion import expected_ncr_points
from research.utility_supervision_audit import core_targets, key_hash


TREE_PARAMETERS = dict(n_estimators=180, learning_rate=0.045, num_leaves=23,
                       min_child_samples=35, reg_lambda=4.0, subsample=0.85,
                       subsample_freq=1, colsample_bytree=0.8, random_state=17,
                       n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True)


def encoded_matrix(encoder, frame):
    categorical, numeric = encoder.transform(frame.assign(player_id='pooled'))
    return np.column_stack([categorical, numeric]), list(range(categorical.shape[1]))


class UtilityCoreMean:
    def __init__(self, *, encoder, competition, season, rubric):
        self.encoder = copy.deepcopy(encoder)
        self.competition = competition
        self.season = int(season)
        self.rubric = rubric
        self.model = None
        self.core_events = ()
        self.training_receipt = None

    def fit(self, frame):
        target, valid, events = core_targets(frame, self.competition, self.season, self.rubric)
        if valid.sum() < 20:
            raise ValueError('Insufficient complete signed utility supervision')
        matrix, categorical = encoded_matrix(self.encoder, frame)
        model = lgb.LGBMRegressor(objective='regression_l2', **TREE_PARAMETERS)
        model.fit(matrix[valid], target.loc[valid].to_numpy(float),
                  sample_weight=np.ones(int(valid.sum())), categorical_feature=categorical)
        fitted = model.predict(matrix[valid])
        if not np.isfinite(fitted).all():
            raise ValueError('Utility fit produced nonfinite predictions')
        self.model = model
        self.core_events = events
        self.training_receipt = dict(rows=len(frame), supervised_rows=int(valid.sum()),
            excluded_unknown_rows=int((~valid).sum()), training_keys_sha256=key_hash(frame),
            supervised_keys_sha256=key_hash(frame.loc[valid]), negative_targets=int(target.loc[valid].lt(0).sum()),
            training_mse=float(np.square(fitted-target.loc[valid].to_numpy(float)).mean()),
            feature_columns=int(matrix.shape[1]), categorical_columns=categorical,
            objective='regression_l2', target='signed_observed_core_points',
            inference_clipping=False, raw_statistics_changed=False)
        return self

    def predict_core(self, frame):
        if self.model is None:
            raise RuntimeError('Utility model has not been fitted')
        matrix, _ = encoded_matrix(self.encoder, frame)
        values = np.asarray(self.model.predict(matrix), dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError('Incomplete utility forecast pool')
        return values


def matched_core_control(original, frame, competition, season, rubric):
    if not isinstance(original, V4GBDT) or not original.pool_player_id or not original.native_categories:
        raise ValueError('The fixed V4 native-category pooled-identity control is required')
    if original.weighting != 'natural' or original.n_estimators != 180 or original.num_leaves != 23 or original.random_state != 17:
        raise ValueError('The original tree recipe differs from the registered utility comparison')
    _, valid, events = core_targets(frame, competition, season, rubric)
    control = copy.deepcopy(original)
    control.events = events
    control.models = {event: model for event, model in control.models.items() if event in ('minutes', *events)}
    control.dispersion = {event: value for event, value in control.dispersion.items() if event in control.models}
    control.effects = {}
    control.hurdle_models = {}
    changed, receipt = [], []
    working = frame.copy()
    for event in events:
        observed = frame[f'available__{event}'].fillna(False).to_numpy(bool) & frame[event].notna().to_numpy()
        identical = bool(np.array_equal(observed, valid))
        if event not in original.models:
            raise ValueError(f'Original core head is missing: {event}')
        if not identical:
            changed.append(event)
        working[f'available__{event}'] = valid
        receipt.append(dict(event=event, original_rows=int(observed.sum()), matched_rows=int(valid.sum()),
                            reused_identical_head=identical, matched_keys_sha256=key_hash(frame.loc[valid])))
    if changed:
        replacement = V4GBDT(events=tuple(changed), weighting='natural', pool_player_id=True,
                             player_effects=False, native_categories=True)
        replacement.fit(working)
        before, before_categories = encoded_matrix(original.encoder, frame)
        after, after_categories = encoded_matrix(replacement.encoder, frame)
        if before_categories != after_categories or original.encoder.numeric_columns != replacement.encoder.numeric_columns:
            raise ValueError('Matched support refit changed the feature encoder contract')
        close(before, after, 'matched support encoder', tolerance=0)
        for event in changed:
            control.models[event] = replacement.models[event]
            control.dispersion[event] = replacement.dispersion[event]
    control.player_effects = True
    control._fit_player_effects(working)
    return control, receipt


def total_points(predictions, competition, season, rubric):
    if competition == 'ncr' and rubric == 'missing_attack_v3':
        return expected_ncr_points(predictions)
    return expected_points(predictions, competition, season=season)


def core_points(predictions, events, competition, season, rubric):
    for prediction in predictions:
        if not set(events).issubset(prediction.events):
            raise ValueError('Core scoring cannot replace a missing event forecast with zero')
    core = [dataclasses.replace(prediction, events={event: prediction.events[event] for event in events})
            for prediction in predictions]
    if competition == 'six_nations':
        return expected_points(core, competition, season=season)
    scorer = NationsChampionshipScorer(version='ncr_front_row_v2')
    values = np.array([scorer.expected_points(prediction) for prediction in core])
    if rubric == 'missing_attack_v3':
        values += only_metres_points(core)
    return values


def utility_points(model, candidates, raw_predictions):
    if len(candidates) != len(raw_predictions):
        raise ValueError('The utility head and raw forecast must cover the identical pool')
    expected_keys = [(str(row.fixture_id), str(row.player_id), str(row.team)) for row in candidates.itertuples(index=False)]
    if expected_keys != [(row.fixture_id, row.player_id, row.team) for row in raw_predictions]:
        raise ValueError('Utility and raw candidate identities differ')
    total = total_points(raw_predictions, model.competition, model.season, model.rubric)
    old_core = core_points(raw_predictions, model.core_events, model.competition, model.season, model.rubric)
    fitted_core = model.predict_core(candidates)
    complement = total-old_core
    points = fitted_core+complement
    if not np.isfinite(points).all():
        raise ValueError('Completed utility forecast is nonfinite')
    return points, dict(original_points=total, original_core=old_core, learned_core=fitted_core,
                        unchanged_complement=complement)
