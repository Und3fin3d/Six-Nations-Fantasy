import argparse
import json
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.features import build_pit_features
from model.unified.raw_benchmark.config import STABLE_EVENTS, EXTENDED_EVENTS
from model.unified.raw_benchmark.coverage import sha256
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import build_folds, evaluation_frame, masked_candidates
from model.unified.raw_benchmark.metrics import NaiveComparator, event_metrics, prediction_means
from model.unified.rolling_eval import ROOT, prepare_store, fit_comparison_models
from model.unified.v4.features import add_v4_base_stats


def save_manifest(output, fold, train, evaluation):
    sources = [*sorted((ROOT/'model').rglob('*.py')), ROOT/'official_labels.py',
               ROOT/'compare_api_official.py', ROOT/'research'/'raw_comparison.py']
    manifest = {
        'fold': fold.label, 'cutoff': fold.cutoff, 'training_rows': len(train),
        'training_fixtures': int(train.fixture_id.nunique()),
        'latest_training_match': str(train.match_at.max()),
        'evaluation_rows': len(evaluation), 'evaluation_fixtures': list(fold.fixture_ids),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'store_sha256': sha256(output/'inputs'/'player_match.csv'),
        'weight_config_sha256': sha256(ROOT/'data/unified/p3_hillclimb/config.json'),
        'python': platform.python_version(),
        'packages': {n: version(n) for n in ('numpy','pandas','scipy','scikit-learn','lightgbm','torch')},
        'history': 'Frozen tournament start; shared three-hour result-availability convention',
        'interpretation': 'Retrospective raw-stat comparison; existing weights; no official fantasy or production claim',
    }
    path = output/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Comparison inputs changed; use a new output directory')
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


def fixture_errors(evaluation, predictions, engine):
    tables = []
    for target in ('minutes', *STABLE_EVENTS, *EXTENDED_EVENTS):
        predicted = prediction_means(predictions, target)
        actual = pd.to_numeric(evaluation[target], errors='coerce').to_numpy(float)
        valid = evaluation[f'available__{target}'].to_numpy(bool) & np.isfinite(actual)
        if not np.isfinite(predicted[valid]).all():
            raise ValueError(f'{engine}: missing predictions for observed {target}')
        rows = evaluation.loc[valid, ['fixture_id']].copy()
        rows['absolute_error'] = abs(actual[valid]-predicted[valid])
        rows = rows.groupby('fixture_id').agg(error_sum=('absolute_error','sum'), n=('absolute_error','size'))
        tables.append(rows.reset_index().assign(engine=engine, target=target))
    return pd.concat(tables, ignore_index=True)


def common_supported_rows(train, evaluation, forecasts):
    scored = evaluation.copy()
    support = []
    international = train[train.competition_level.eq('international')]
    for target in ('minutes', *STABLE_EVENTS, *EXTENDED_EVENTS):
        observed = evaluation[f'available__{target}'].fillna(False).to_numpy(bool)
        observed &= np.isfinite(pd.to_numeric(evaluation[target], errors='coerce').to_numpy(float))
        common = observed.copy()
        training_support = international[f'available__{target}'].fillna(False).any()
        for engine, predictions in forecasts.items():
            finite = np.isfinite(prediction_means(predictions, target))
            if training_support and not finite[observed].all():
                raise ValueError(f'{engine}: missing predictions for supported {target}')
            support.append(dict(engine=engine, target=target, observed=int(observed.sum()),
                                predicted_observed=int((observed & finite).sum()),
                                training_supported=bool(training_support)))
            common &= finite
        scored[f'available__{target}'] = common
        for row in support[-len(forecasts):]:
            row['common_observed'] = int(common.sum())
    return scored, pd.DataFrame(support)


def run(output, fold_name):
    from model_env_preflight import check
    environment_errors = check(ROOT/'requirements-model.txt')
    if environment_errors:
        raise RuntimeError('; '.join(environment_errors))
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    store = prepare_store(output)
    folds = {f.label: f for f in build_folds(store)}
    fold = folds[fold_name]
    evaluation = evaluation_frame(store, fold).reset_index(drop=True)
    train = past_matches(store, fold.cutoff_timestamp).sort_values(['date','fixture_id','team','player_id'])
    if set(train.fixture_id) & set(evaluation.fixture_id):
        raise ValueError('Evaluation fixture entered training')
    save_manifest(output, fold, train, evaluation)
    prepared = add_v4_base_stats(build_pit_features(store))
    features, candidates = build_frozen_feature_frames(
        train, masked_candidates(evaluation), v4=True, prepared_train=prepared.loc[train.index])
    config = json.loads((ROOT/'data/unified/p3_hillclimb/config.json').read_text())
    directory = output/'models'
    directory.mkdir(exist_ok=True)
    print(f'Fitting {fold_name} on {len(train):,} prior rows', flush=True)
    blend, models = fit_comparison_models(train, features, fold.cutoff_timestamp, directory, config, True)
    models = {'empirical_event': blend.empirical, **{n+'_native': m for n,m in models.items()}}
    naive = NaiveComparator.fit(train, ('minutes', *STABLE_EVENTS, *EXTENDED_EVENTS))
    history = train.groupby('player_id').fixture_id.nunique()
    evaluation['career_matches'] = evaluation.player_id.map(history).fillna(0)
    forecasts = {}
    for engine, model in models.items():
        predictions = model.predict_frame(candidates)
        actual_keys = list(zip(evaluation.fixture_id.astype(str), evaluation.player_id.astype(str), evaluation.team))
        predicted_keys = [(p.fixture_id,p.player_id,p.team) for p in predictions]
        if actual_keys != predicted_keys:
            raise ValueError(f'{engine}: prediction keys do not match the evaluation')
        (output/f'{engine}.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in predictions))
        forecasts[engine] = predictions
    scored, support = common_supported_rows(train, evaluation, forecasts)
    support.assign(fold=fold_name).to_csv(output/'target_support.csv', index=False)
    event_tables, fixture_error_tables = [], []
    for engine, predictions in forecasts.items():
        event_tables.append(event_metrics(scored, predictions, naive, engine=engine, fold=fold_name))
        fixture_error_tables.append(fixture_errors(scored, predictions, engine).assign(fold=fold_name))
    pd.concat(event_tables, ignore_index=True).to_csv(output/'events.csv', index=False)
    pd.concat(fixture_error_tables, ignore_index=True).to_csv(output/'fixture_errors.csv', index=False)
    print(f'Completed {fold_name}: {len(evaluation):,} rows, {len(fold.fixture_ids)} fixtures', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(ROOT/args.output, args.fold)


if __name__ == '__main__':
    main()
