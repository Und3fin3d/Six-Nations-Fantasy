"""Cache-only matched-history backtest; never promotes models or opens PRs.

NCR has archived prices. Six Nations squad scores are price-free diagnostics,
not evidence of budget-feasible superiority. All outcomes are official points.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import lognorm

from model.history import past_matches
from model.empirical_unified import project_candidates
from model.ncr_project import build_projection, optimise
from model.ncr_eval import load_actuals as load_team_actuals, team_points
from model.ncr_rank_eval import load_actuals
from .data import ROOT, DEFAULT_SOURCES, build_canonical_store
from .features import build_pit_features
from .contracts import RawPrediction
from .scoring import SixNationsScorer
from .ncr_gw_eval import expected_ncr_points
from .raw_benchmark.coverage import build_corrected_store, sha256
from .raw_benchmark.empirical import EmpiricalEventModel
from .raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from .raw_benchmark.config import STABLE_EVENTS, EXTENDED_EVENTS
from .raw_benchmark.features import build_frozen_feature_frames
from .raw_benchmark.folds import masked_candidates
from .raw_benchmark.blend import EventBlend50, EventWeightedBlend
from .v4.gbdt import V4GBDT
from .v4.features import add_v4_base_stats
from .v3.shadow import ncr_candidates

DATA = ROOT / 'data'
DEFAULT_OUTPUT = DATA / 'unified' / 'rolling_eval'
EVALUATION_INPUTS = (
    'model_targets.csv', 'ncr/ncr_fixtures.csv', 'ncr/ncr_teams.csv',
    'ncr/ncr_player_crosswalk.csv', 'ncr/feeds/players_gw1.json',
    'ncr/feeds/players_gw2.json', 'ncr/feeds/players_gw3.json',
    'ncr/ncr_gw1_projections.csv', 'ncr/ncr_gw2_projections.csv',
)
KEY = ['fixture_id', 'player_id', 'team']
POS = {'Prop': 'Prop', 'Hooker': 'Hooker', 'Second-row': 'Lock',
       'Back-row': 'Loose Forward', 'Scrum-half': 'Scrum Half',
       'Fly-half': 'Fly Half', 'Centre': 'Centre', 'Back-three': 'Back Three'}

@dataclass
class Slate:
    name: str
    competition: str
    season: int
    round: int
    cutoff: pd.Timestamp
    candidates: pd.DataFrame
    pool: pd.DataFrame
    actual: np.ndarray
    team_actuals: dict
    budget_verified: bool
    lineup_basis: str
    baseline: np.ndarray | None = None


def prepare_store(output: Path) -> pd.DataFrame:
    """Current cached source tables, not the stale July v1 canonical snapshot."""
    directory = output / 'inputs'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'player_match.csv'
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p, *_ in DEFAULT_SOURCES}
    for name in ('official_labels.py', 'compare_api_official.py'):
        inputs[name] = sha256(ROOT/name)
    for p in (DATA/'official_player_match.csv', DATA/'rp_compstats.csv', DATA/'wr_rankings.csv'):
        inputs[str(p.relative_to(ROOT))] = sha256(p)
    manifest = directory/'sources.json'
    if path.exists():
        previous = json.loads(manifest.read_text())
        if previous['inputs'] != inputs or previous['store_sha256'] != sha256(path):
            raise ValueError('research inputs changed; choose a new output directory')
        for fixture, digest in previous['cache_sha256'].items():
            if sha256(DATA/'cache'/f'match_{fixture}.json') != digest:
                raise ValueError('cached match changed; choose a new output directory')
        return pd.read_csv(path, low_memory=False, dtype={key: str for key in KEY}, parse_dates=['date','match_at'])
    legacy = directory/'canonical.csv'
    build_canonical_store(asof='2026-09-19').to_csv(legacy, index=False)
    store, _, report = build_corrected_store(legacy)
    venue, cache_hashes = {}, {}
    for fixture in store.fixture_id.astype(str).unique():
        source = DATA/'cache'/f'match_{fixture}.json'
        match = json.loads(source.read_text())['results']['match']
        venue[(fixture, str(match['home_team']))] = 'home'
        venue[(fixture, str(match['away_team']))] = 'away'
        cache_hashes[fixture] = sha256(source)
    store['home_away'] = [venue.get((str(r.fixture_id), str(r.team)), '') for r in store.itertuples()]
    if store.home_away.eq('').any() or store.duplicated(KEY).any():
        raise ValueError('missing venue or duplicate player-match keys')
    store = store.sort_values(['date','fixture_id','team','player_id']).reset_index(drop=True)
    store.to_csv(path, index=False)
    manifest.write_text(json.dumps({'inputs': inputs, 'cache_sha256': cache_hashes,
        'store_sha256': sha256(path), 'audit': report}, indent=2, sort_keys=True))
    legacy.unlink()
    print(f'Prepared {len(store):,} rows / {store.fixture_id.nunique():,} fixtures', flush=True)
    return store


def official_slates(store: pd.DataFrame, competitions: tuple[str, ...]) -> list[Slate]:
    slates = []
    if 'six_nations' in competitions:
        labels = pd.read_csv(DATA/'model_targets.csv', dtype={'fixture_id': str, 'player_id': str})
        labels = labels[labels.season.isin([2025,2026])][KEY+['official_pts']]
        six = store[store.competition_id_cache.eq(1266) & store.calendar_year.isin([2025,2026])]
        joined = six.merge(labels, on=KEY, how='left', validate='one_to_one')
        if len(six.merge(labels[KEY], on=KEY, how='inner')) != len(labels):
            raise ValueError('Six Nations official label coverage is incomplete')
        for (year, round_no), rows in joined.groupby(['season','round']):
            rows = rows.sort_values(KEY).reset_index(drop=True)
            if rows.team.nunique() != 6 or not rows.groupby('team').size().eq(23).all():
                raise ValueError(f'{year}/round {round_no}: incomplete Six Nations teamsheets')
            cutoff = pd.to_datetime(rows.match_at, utc=True).min()
            ids = np.arange(1, len(rows)+1)
            pool = pd.DataFrame({'id': ids, 'name': rows.player_name, 'team': rows.team,
                'pos': rows.position.map(POS), 'hemi': 1, 'value': 0.0,
                'status': np.where(rows.started, 'P', 'B')})
            if pool.pos.isna().any():
                raise ValueError('unknown Six Nations position')
            actual = rows.official_pts.to_numpy(float)
            team_actuals = {str(i): (float(p),float(m),str(s)) for i,p,m,s in zip(ids,actual,rows.minutes,pool.status)}
            candidates = masked_candidates(rows.drop(columns=['official_pts','team_score','opp_score'], errors='ignore'))
            candidates['label_row_id'] = ids
            slates.append(Slate(f'six_nations_{int(year)}_r{int(round_no)}','six_nations',int(year),int(round_no),
                cutoff,candidates,pool,actual,team_actuals,False,'complete teamsheet pool; kickoff lock proxy; NO archived prices; missing scores remain unknown'))
    if 'ncr' in competitions:
        fixtures = pd.read_csv(DATA/'ncr'/'ncr_fixtures.csv')
        for gw in (1,2,3):
            fx = fixtures[pd.to_numeric(fixtures.gameday).eq(gw)]
            cutoff = pd.to_datetime(fx.lock_date, utc=True).min()
            history = past_matches(store, cutoff)
            feed = DATA/'ncr'/'feeds'/f'players_gw{gw}.json'
            source_pool = pd.DataFrame(json.loads(feed.read_text())['Data']['Value']['Players'])
            for column in ('id', 'hemisphere', 'value', 'sel_percentage'):
                source_pool[column] = pd.to_numeric(source_pool[column], errors='raise')
            source_pool['id'] = source_pool['id'].astype(int)
            if gw in (1,2):
                saved = pd.read_csv(DATA/'ncr'/f'ncr_gw{gw}_projections.csv')
                ids = pd.to_numeric(source_pool.id).astype(int)
                source_pool['player_status'] = ids.map(saved.set_index('id').status).fillna('OUT')
                source_pool['value'] = ids.map(saved.set_index('id').value).fillna(source_pool.value)
            projection = build_projection(pool_override=source_pool, asof=cutoff, gameday=gw,
                history_override=history[history.competition_level.eq('international')],
                club_override=history[history.competition_level.eq('club')], use_weather=False)
            if gw == 1:
                projection = projection[~projection.team.isin(['New Zealand','France'])]
            projection = projection.sort_values('id').reset_index(drop=True)
            candidates = ncr_candidates(gw, projection=projection)
            candidates['date'] = pd.to_datetime(candidates.date, utc=True).dt.tz_convert(None)
            actual_frame = load_actuals(feed).set_index('id')
            actual = projection.id.map(actual_frame.actual).to_numpy(float)
            if not np.isfinite(actual).all():
                raise ValueError(f'GW{gw}: incomplete official outcomes')
            slates.append(Slate(f'ncr_2026_r{gw}','ncr',2026,gw,cutoff,candidates,projection,actual,
                load_team_actuals(feed),True,'saved pre-lock prices/status' if gw<3 else 'final corrected lineup; retrospective prices',
                projection.starter_exp.to_numpy(float)))
    return sorted(slates, key=lambda s: (s.cutoff,s.name))


def expected_points(predictions: list[RawPrediction], competition: str) -> np.ndarray:
    if competition not in ('ncr','six_nations'):
        raise ValueError(f'unsupported scoring rules: {competition}')
    if competition == 'ncr':
        return np.array([expected_ncr_points(p) for p in predictions])
    scorer, result = SixNationsScorer(), []
    for p in predictions:
        value = sum(w*p.events[e].mean for e,w in scorer.weights.items() if e in p.events)
        value += (15 if p.is_forward else 10)*(p.events['tries'].mean if 'tries' in p.events else 0)
        metres = p.events.get('metres')
        if metres is not None and metres.mean > 0:
            # E[floor(X/10)] = sum P(X >= 10k), not floor(E[X]/10).
            sigma2 = np.log1p(metres.dispersion/metres.mean**2)
            sigma, scale = np.sqrt(sigma2), np.exp(np.log(metres.mean)-sigma2/2)
            end = int(np.ceil(lognorm.isf(1e-10,sigma,scale=scale)/10))
            if end > 1_000_000:
                raise ValueError('metres tail exceeds the bounded expectation calculation')
            value += float(lognorm.sf(10*np.arange(1,end+1),sigma,scale=scale).sum())
        result.append(value)
    return np.array(result)


def evaluate(slate: Slate, engine: str, points: np.ndarray, output: Path) -> dict:
    points = np.asarray(points,dtype=float)
    if points.shape != slate.actual.shape or not np.isfinite(points).all():
        raise ValueError(f'{slate.name}/{engine}: incomplete predictions')
    pool = slate.pool.copy()
    pool['starter_exp'] = points
    pool['supersub_exp'] = points*np.where(pool.status.eq('B'),3.0,0.5)
    kwargs = {} if slate.competition=='ncr' else dict(budget=np.inf,max_nation=4,max_hemi=None)
    squad, _, _ = optimise(pool,**kwargs)
    directory = output/slate.name
    directory.mkdir(parents=True,exist_ok=True)
    squad.to_csv(directory/f'{engine}_squad.csv',index=False)
    pd.DataFrame({'id': pool.id,'predicted': points,'actual': slate.actual}).to_csv(directory/f'{engine}_predictions.csv',index=False)
    labelled = np.isfinite(slate.actual)
    if not labelled.any():
        raise ValueError(f'{slate.name}: no observed outcomes')
    actual_by_id = dict(zip(pool.id, slate.actual))
    missing_selected = sum(not np.isfinite(actual_by_id[i]) for i in squad.id)
    return dict(slate=slate.name,competition=slate.competition,season=slate.season,round=slate.round,
        engine=engine,n=len(points),n_labelled=int(labelled.sum()),
        n_unlabelled=int((~labelled).sum()),team_unlabelled=missing_selected,
        mae=float(np.abs(points[labelled]-slate.actual[labelled]).mean()),
        team_points=team_points(squad,slate.team_actuals),budget_verified=slate.budget_verified,lineup_basis=slate.lineup_basis)


def fit_comparison_models(train, features, cutoff, model_dir, config, native_categories):
    path = model_dir/'p3.pkl'
    if path.exists():
        blend = EventBlend50.load(path)
    else:
        empirical = EmpiricalEventModel(asof=cutoff).fit(train)
        v4 = V4GBDT(events=(*STABLE_EVENTS,*EXTENDED_EVENTS),weighting='natural',pool_player_id=True,
            player_effects=True,native_categories=native_categories).fit(features)
        blend = EventBlend50(empirical,v4)
        blend.save(path)
    models = {'p3_rolling': blend,'p3_weighted': EventWeightedBlend(blend.empirical,blend.v4,
        weight_v4=config['default_weight_v4'],event_weights_v4=config['event_weights_v4'])}
    robust = RobustEmpiricalEventModel(asof=cutoff).fit(train)
    models['p3_robust'] = EventWeightedBlend(robust, blend.v4,
        weight_v4=config['default_weight_v4'], event_weights_v4=config['event_weights_v4'])
    return blend, models


def run(output: Path, competitions: tuple[str,...], *, prepare_only: bool=False, round_job: str | None=None, native_categories: bool=False) -> pd.DataFrame:
    from model_env_preflight import check
    errors = check(ROOT/'requirements-model.txt')
    if errors and not prepare_only:
        raise RuntimeError('incompatible model environment: ' + '; '.join(errors))
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning)
    store = prepare_store(output)
    if prepare_only:
        return pd.DataFrame()
    manifest_path = output/'run_manifest.json'
    manifest = {
        'native_categories': bool(native_categories),
        'source_base_commit': '086b474334a50d4e3d34880a20fdd4e6d573cd85',
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy','pandas','scipy','scikit-learn','lightgbm','torch')},
        'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in
            [*sorted((ROOT/'model').rglob('*.py')), ROOT/'official_labels.py', ROOT/'compare_api_official.py']},
        'store_sha256': sha256(output/'inputs'/'player_match.csv'),
        'evaluation_inputs_sha256': {name: sha256(DATA/name) for name in EVALUATION_INPUTS},
        'weight_config_sha256': sha256(DATA/'unified'/'p3_hillclimb'/'config.json'),
        'result_availability': 'kickoff plus conservative three-hour lag; unversioned historical feeds',
        'selection': 'fixed existing event weights; robust challenger has fixed shared four-appearance prior',
        'validation': 'retrospective; 2025 already used to select existing event weights; Six Nations has no archived prices',
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('source/config/environment changed; choose a new output directory')
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    all_slates = official_slates(store,competitions)
    slates = [slate for slate in all_slates if slate.name == round_job] if round_job else all_slates
    if not slates:
        raise ValueError(f'no matching evaluation slates: {round_job}')
    wr = pd.read_csv(DATA/'wr_rankings.csv',parse_dates=['snapshot_date'])
    print('Building shared prior-only training features',flush=True)
    feature_path = output/'inputs'/'training_features.pkl'
    if feature_path.exists():
        prepared = pd.read_pickle(feature_path)
    else:
        prepared = add_v4_base_stats(build_pit_features(store))
        prepared.to_pickle(feature_path)
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    results, manifests, frozen = [], [], {}
    for slate in slates:
        start = time.monotonic()
        print(f'Running {slate.name} at {slate.cutoff.isoformat()}',flush=True)
        train = past_matches(store,slate.cutoff).sort_values(['date','fixture_id','team','player_id'])
        if set(train.fixture_id.astype(str)) & set(slate.candidates.fixture_id.astype(str)):
            raise ValueError(f'{slate.name}: evaluation fixture entered training')
        features,candidates = build_frozen_feature_frames(train,slate.candidates,v4=True,prepared_train=prepared.loc[train.index])
        if slate.baseline is None:
            baseline = project_candidates(slate.candidates,train,slate.competition,wr,asof=slate.cutoff)
            slate.baseline = baseline.set_index('label_row_id').reindex(slate.candidates.label_row_id).predicted_points.to_numpy(float)
        results.append(evaluate(slate,'empirical_baseline',slate.baseline,output))
        model_dir = output/'models'/slate.name
        model_dir.mkdir(parents=True,exist_ok=True)
        blend, models = fit_comparison_models(train, features, slate.cutoff, model_dir, config, native_categories)
        for name,model in models.items():
            if native_categories:
                name += "_native"
            model.save(model_dir/f'{name}.pkl')
            raw = model.predict_frame(candidates)
            (model_dir/f'{name}.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))
            results.append(evaluate(slate,name,expected_points(raw,slate.competition),output))
        key = (slate.competition,slate.season)
        if key not in frozen:
            frozen.clear()  # release the preceding tournament's large frames
            frozen[key] = (blend,train,features)
        old_model,old_train,old_features = frozen[key]
        if round_job:
            # Independent round jobs fit exactly one model. The first job in
            # each season emits every frozen control; later jobs cannot relabel
            # their later-cutoff model as a tournament-frozen control.
            season_slates = [s for s in all_slates if (s.competition,s.season) == key]
            controls = season_slates if slate.name == season_slates[0].name else []
        else:
            controls = [slate]
        for control in controls:
            _,old_candidates = build_frozen_feature_frames(old_train,control.candidates,v4=True,prepared_train=old_features)
            results.append(evaluate(control,'p3_tournament_frozen_native' if native_categories else 'p3_tournament_frozen',expected_points(old_model.predict_frame(old_candidates),control.competition),output))
        manifests.append({'slate': slate.name,'cutoff': slate.cutoff.isoformat(),'training_rows': len(train),
            'training_fixtures': int(train.fixture_id.nunique()),'training_match_at_max': train.match_at.max().isoformat(),
            'same_tournament_prior_rows': int((train.competition_id_cache.eq(696 if slate.competition=='ncr' else 1266)&train.calendar_year.eq(slate.season)).sum()),
            'lineup_basis': slate.lineup_basis,'budget_verified': slate.budget_verified})
        metrics = pd.DataFrame(results)
        metrics.to_csv(output/'metrics.csv',index=False)
        (output/'round_manifests.json').write_text(json.dumps(manifests,indent=2))
        print(metrics[metrics.slate.eq(slate.name)][['engine','mae','team_points']].to_string(index=False),flush=True)
        print(f'Finished in {time.monotonic()-start:.1f}s',flush=True)
    summary = pd.DataFrame(results).groupby(['competition','season','engine']).agg(
        mae=('mae','mean'),team_points=('team_points',lambda values: values.sum(min_count=len(values))),
        labelled_players=('n_labelled','sum'),unlabelled_players=('n_unlabelled','sum'),
        scored_teams=('team_points','count'),rounds=('round','nunique')).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--competitions',nargs='+',choices=['ncr','six_nations'],default=['six_nations','ncr'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--native-categories',action='store_true',help='Research-only native categorical tree splits; no promotion')
    parser.add_argument('--round-job',help='Fit one round; season-first jobs also emit every frozen control')
    args = parser.parse_args()
    run(args.output,tuple(args.competitions),prepare_only=args.prepare_only,round_job=args.round_job,native_categories=args.native_categories)

if __name__ == '__main__':
    main()
