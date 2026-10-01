"""Score raw-forecast post-processing candidates on development and evaluation slates.

Each candidate starts from saved robust P3 raw forecasts (no refit) and applies,
in order, optional steps from its JSON spec:

``matchup``  path to matchup coefficients (``model.unified.matchup``);
``beta``     path to edge-only team-strength coefficients (``v4.context``);
``gamma``    goal-kicking concentration exponent (``model.unified.kicking``);
``weight``   fantasy-points blend weight on the model (rest: empirical baseline).

Fantasy MAE, realised squad points and smoothed squad points (mean over 40
identical 1% multiplicative jitters) use the unchanged rolling_eval
optimiser and team scorer. Friendly-25 count-stat metrics use the unchanged
friendly evaluator. Matchup features are cached per slate.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.ncr_eval import team_points
from model.ncr_project import optimise
from model.unified.friendly_eval import score_stats, summarise_stats
from model.unified.kicking import concentrate_kicking
from model.unified.matchup import matchup_features, matchup_scale
from model.unified.rolling_eval import ROOT, expected_points
from model.unified.v4.context import add_candidate_context, pre_match_context, team_strength_scale
from research.context_experiment import load_store
from research.friendly_experiment import load_raw

JITTER, DRAWS, SEED = 0.01, 40, 12345
EVENTS = ('bad_passes', 'clean_breaks', 'conversion_goals', 'defenders_beaten', 'drop_goal_missed',
          'drop_goals_converted', 'metres', 'missed_conversion_goals', 'missed_penalty_goals', 'missed_tackles',
          'offload', 'passes', 'penalties_conceded', 'penalty_goals', 'red_cards', 'rucks_lost', 'rucks_won',
          'runs', 'tackles', 'tries', 'try_assists', 'turnovers_conceded', 'yellow_cards')


def squad(slate, points: np.ndarray):
    pool = slate.pool.copy()
    pool['starter_exp'] = points
    pool['supersub_exp'] = points*np.where(pool.status.eq('B'), 3.0, 0.5)
    kwargs = {} if slate.competition == 'ncr' else dict(budget=np.inf, max_nation=4, max_hemi=None)
    return optimise(pool, **kwargs)[0]


def squad_points(slate, points: np.ndarray) -> tuple[float, float]:
    realised = team_points(squad(slate, points), slate.team_actuals)
    rng = np.random.default_rng(SEED)
    smoothed = [team_points(squad(slate, points*np.exp(rng.normal(0, JITTER, len(points)))), slate.team_actuals)
                for _ in range(DRAWS)]
    return float(realised), float(np.mean(smoothed))


class Context:
    """Per-slate pre-lock features shared by every candidate."""

    def __init__(self, store: pd.DataFrame, cache: Path):
        self.store, self.cache = store, cache
        cache.mkdir(parents=True, exist_ok=True)

    def features(self, name: str, frame: pd.DataFrame, cutoff) -> pd.DataFrame:
        path = self.cache/f'{name}.pkl'
        if path.exists():
            return pd.read_pickle(path)
        history = past_matches(self.store, cutoff)
        features = matchup_features(frame.reset_index(drop=True), history, cutoff, EVENTS)
        features['elo_edge'] = add_candidate_context(frame.reset_index(drop=True), pre_match_context(history)[1])[
            'ctx__elo_edge'].to_numpy(float)
        features.to_pickle(path)
        return features


def transform(raw, features: pd.DataFrame, spec: dict):
    if spec.get('matchup'):
        coef = json.loads(Path(spec['matchup']).read_text())
        raw = [matchup_scale(p, row, coef) for p, row in zip(raw, features.to_dict('records'))]
    if spec.get('beta'):
        beta = json.loads(Path(spec['beta']).read_text())
        raw = [team_strength_scale(p, float(e), beta) for p, e in zip(raw, features['elo_edge'])]
    if spec.get('gamma'):
        raw = concentrate_kicking(raw, float(spec['gamma']))
    return raw


def official(slates, raw_dir: Path, context: Context, specs: dict, smooth: bool,
             players: list | None = None) -> pd.DataFrame:
    rows = []
    for slate in slates:
        reference = load_raw(raw_dir/slate.name/'p3_robust_native.jsonl')
        keys = list(zip(slate.candidates.fixture_id.astype(str), slate.candidates.player_id.astype(str),
                        slate.candidates.team.astype(str)))
        if keys != [(p.fixture_id, p.player_id, p.team) for p in reference]:
            raise ValueError(f'{slate.name}: forecasts misaligned')
        features = context.features(slate.name, slate.candidates, slate.cutoff)
        labelled = np.isfinite(slate.actual)
        values = {'empirical_baseline': np.asarray(slate.baseline, float)}
        for name, spec in specs.items():
            points = expected_points(transform(reference, features, spec), slate.competition)
            weight = float(spec.get('weight', 1.0))
            values[name] = weight*points + (1 - weight)*values['empirical_baseline']
        for name, points in values.items():
            if players is not None:
                players.append(pd.DataFrame({'slate': slate.name, 'engine': name,
                                             'fixture_id': slate.candidates.fixture_id.astype(str).to_numpy(),
                                             'predicted': points, 'actual': slate.actual}))
            realised, smoothed = squad_points(slate, points) if smooth else (
                team_points(squad(slate, points), slate.team_actuals), np.nan)
            rows.append({'slate': slate.name, 'competition': slate.competition, 'season': slate.season,
                         'round': slate.round, 'engine': name, 'n_labelled': int(labelled.sum()),
                         'mae': float(np.abs(points[labelled] - slate.actual[labelled]).mean()),
                         'team_points': realised, 'smoothed_points': smoothed})
        print(f'{slate.name} done', flush=True)
    return pd.DataFrame(rows)


def friendly(runs: Path, context: Context, specs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((ROOT/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    stats = []
    for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
        directory = runs/'f25_ctx'/'days'/str(day)
        truth = pd.read_pickle(directory/'truth.pkl')
        reference = load_raw(directory/'p3_robust_native.jsonl')
        by_key = {(p.fixture_id, p.player_id, p.team): p for p in reference}
        frame = truth.reset_index(drop=True)
        aligned = [by_key[k] for k in zip(frame.fixture_id.astype(str), frame.player_id.astype(str),
                                          frame.team.astype(str))]
        features = context.features(f'friendly_{day}', frame, group.kickoff.min())
        stats.append(score_stats(truth, aligned, engine='p3_robust_native'))
        for name, spec in specs.items():
            if spec.get('matchup') or spec.get('beta') or spec.get('gamma'):
                stats.append(score_stats(truth, transform(aligned, features, spec), engine=name))
    stats = pd.concat(stats, ignore_index=True)
    return stats, summarise_stats(stats, fixtures.fixture_id.astype(str))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='scratch runs directory (base/, devcr/, f25_ctx/)')
    parser.add_argument('--specs', type=Path, required=True)
    parser.add_argument('--set', choices=['dev', 'official', 'friendly'], required=True)
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.specs.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    store, _ = load_store(args.runs/'base')
    context = Context(store, args.runs/'feature_cache')
    if args.set == 'friendly':
        stats, summary = friendly(args.runs, context, specs)
        stats.to_csv(args.output/'friendly_stats.csv', index=False)
        summary.to_csv(args.output/'friendly_summary.csv', index=False)
        print(summary.to_string(index=False))
        return
    if args.set == 'dev':
        slates, raw_dir = pickle.loads((args.runs/'devcr'/'slates.pkl').read_bytes()), args.runs/'devcr'/'models'
    else:
        from research.reweight_eval import cached_slates
        slates, raw_dir = cached_slates(args.runs/'base'), args.runs/'base'/'models'
    players = []
    rounds = official(slates, raw_dir, context, specs, not args.no_smooth, players)
    rounds.to_csv(args.output/'rounds.csv', index=False)
    pd.concat(players, ignore_index=True).to_pickle(args.output/'players.pkl')
    seasons = rounds.groupby(['competition', 'season', 'engine']).agg(
        mae=('mae', 'mean'), team_points=('team_points', 'sum'), smoothed_points=('smoothed_points', 'sum'),
        rounds=('round', 'nunique')).reset_index()
    seasons.to_csv(args.output/'seasons.csv', index=False)
    print(seasons.to_string(index=False))


if __name__ == '__main__':
    main()
