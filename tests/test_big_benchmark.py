"""Focused tests for the large computed-rubric benchmark (research.big_benchmark)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.ncr_project import REQUIRED, optimise
from model.unified.contracts import EventDistribution, RawPrediction
from research.big_benchmark import rubrics, slates
from research.big_benchmark.decision import candidate_roles, decide_slate, realised_points, solve
from research.big_benchmark.stats import (paired_bootstrap, power_slates, slate_correlations)

LAYOUT = ['Prop', 'Hooker', 'Prop', 'Lock', 'Lock', 'Loose Forward', 'Loose Forward', 'Loose Forward',
          'Scrum Half', 'Fly Half', 'Back Three', 'Centre', 'Centre', 'Back Three', 'Back Three',
          'Hooker', 'Prop', 'Prop', 'Lock', 'Loose Forward', 'Scrum Half', 'Fly Half', 'Back Three']


def random_pool(fixtures: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [dict(team=f'T{f}{side}', pos=p, started=j < 15)
            for f in range(fixtures) for side in range(2) for j, p in enumerate(LAYOUT)]
    pool = pd.DataFrame(rows)
    pool['value'] = rng.gamma(2.5, 5, len(pool))*np.where(pool.started, 1.0, 0.45)
    return pool


@pytest.mark.parametrize('fixtures,seed', [(3, 0), (3, 1), (4, 2), (6, 3), (8, 4), (12, 5)])
def test_solver_matches_reference_optimiser(fixtures, seed):
    pool = random_pool(fixtures, seed)
    bench = (~pool.started).to_numpy(float)
    ours = solve(pool.pos, pool.team, pool.started, pool.value, 3*pool.value*bench)
    reference = pool.assign(id=np.arange(1, len(pool)+1), name='x', hemi=1, value_m=0.0)
    reference = reference.rename(columns={'value': 'starter_exp'}).assign(
        value=0.0, status=np.where(pool.started, 'P', 'B'))
    reference['supersub_exp'] = reference.starter_exp*np.where(reference.status.eq('B'), 3.0, 0.5)
    squad, _, result = optimise(reference, budget=np.inf, max_nation=4, max_hemi=None)
    assert ours.objective == pytest.approx(-result.fun, abs=1e-6)
    assert set(ours.starters) | {ours.sub} == set(squad.id - 1)
    assert ours.captain == int(squad.loc[squad.is_capt, 'id'].iloc[0]) - 1
    assert ours.sub == int(squad.loc[squad.is_sub, 'id'].iloc[0]) - 1


def test_squad_invariants():
    pool = random_pool(5, 9)
    bench = (~pool.started).to_numpy(float)
    squad = solve(pool.pos, pool.team, pool.started, pool.value, 3*pool.value*bench)
    members = np.r_[squad.starters, squad.sub]
    assert len(squad.starters) == 15 and len(set(members)) == 16
    assert pool.pos.iloc[squad.starters].value_counts().to_dict() == REQUIRED
    assert pool.team.iloc[members].value_counts().max() <= 4
    assert bool(pool.started.iloc[squad.captain]) and squad.captain in squad.starters
    assert not bool(pool.started.iloc[squad.sub])


def test_pruning_never_drops_a_needed_role():
    pool = random_pool(10, 11)
    keep_x, keep_c = candidate_roles(pool.pos.to_numpy(), pool.team.to_numpy(), pool.started.to_numpy(),
                                     pool.value.to_numpy())
    assert keep_c.sum() <= keep_x.sum() + pool.started.sum()
    for p, quota in REQUIRED.items():
        assert keep_x[pool.pos.eq(p).to_numpy()].sum() >= quota + 5


def test_infeasible_pool_returns_none():
    pool = random_pool(1, 0)  # two teams cannot supply 16 players under a cap of four
    bench = (~pool.started).to_numpy(float)
    assert solve(pool.pos, pool.team, pool.started, pool.value, 3*pool.value*bench) is None


def test_realised_points_and_oracle_regret_non_negative():
    pool = random_pool(4, 21)
    rng = np.random.default_rng(3)
    actual = np.round(pool.value.to_numpy()*rng.uniform(0, 2, len(pool)))
    pool['minutes'] = np.where(rng.uniform(size=len(pool)) < .1, 0, 60)
    actual[pool.minutes.eq(0).to_numpy()] = 0
    preds = {'a': pool.value.to_numpy(), 'b': pool.value.to_numpy()[::-1].copy()}
    rows = decide_slate(pool, preds, actual, draws=3, seed_key=1)
    assert {r['engine'] for r in rows} == {'a', 'b'}
    for row in rows:
        assert row['regret'] >= -1e-9 and row['smoothed_regret'] >= -1e-9
        assert row['oracle_flat'] >= row['flat'] - 1e-9
    # A non-playing super-sub scores nothing; a captain counts twice.
    bench = (~pool.started).to_numpy(float)
    squad = solve(pool.pos, pool.team, pool.started, actual, 3*actual*bench)
    played = pool.minutes.to_numpy() > 0
    expected = actual[squad.starters].sum() + actual[squad.captain] + 3*actual[squad.sub]*played[squad.sub]
    assert realised_points(squad, actual, played) == pytest.approx(expected)


def _prediction(events: dict, forward: bool = True) -> RawPrediction:
    dists = {k: EventDistribution('lognormal' if k == 'metres' else 'poisson', v, 400.0 if k == 'metres' else 1.0)
             for k, v in events.items()}
    return RawPrediction('f', 'p', 'n', 't', 'o', 'Prop', forward, dists, EventDistribution('lognormal', 60, 100))


def test_actual_points_per_rubric():
    row = pd.DataFrame([{'is_forward': True, 'tries': 1, 'try_assists': 1, 'conversion_goals': 0,
                         'missed_conversion_goals': 0, 'penalty_goals': 0, 'missed_penalty_goals': 0,
                         'drop_goals_converted': 0, 'defenders_beaten': 2, 'offload': 1, 'clean_breaks': 1,
                         'tackles': 10, 'missed_tackles': 2, 'tackle_turnover': 1, 'turnovers_conceded': 1,
                         'penalties_conceded': 1, 'yellow_cards': 0, 'red_cards': 0, 'metres': 37}])
    assert rubrics.actual_points(row, rubrics.SIX_NATIONS)[0] == 15 + 4 + 4 + 2 + 10 + 5 - 1 + 3
    assert rubrics.actual_points(row, rubrics.NCR)[0] == 12 + 5 + 4 + 2 + 3 + 10 - 2 + 4 - 1 - 1
    assert rubrics.actual_points(row, rubrics.SUPER_RUGBY)[0] == 15 + 9 + 7 + 10 - 2 + 4 + 4 + 2 - 1 + 3
    back = row.assign(is_forward=False)
    assert rubrics.actual_points(back, rubrics.SIX_NATIONS)[0] == 10 + 4 + 4 + 2 + 10 + 5 - 1 + 3
    missing = row.assign(available__tackle_turnover=False)
    assert np.isnan(rubrics.actual_points(missing, rubrics.NCR)[0])


def test_expected_points_linear_and_metres_floor():
    p = _prediction({'tries': 0.2, 'tackles': 8.0, 'metres': 30.0})
    rng = np.random.default_rng(0)
    sigma2 = np.log1p(400/30**2)
    draws = rng.lognormal(np.log(30) - sigma2/2, np.sqrt(sigma2), 400_000)
    expected = 15*0.2 + 8 + np.floor(draws/10).mean()
    assert rubrics.expected_points([p], rubrics.SIX_NATIONS)[0] == pytest.approx(expected, abs=0.01)
    assert rubrics.expected_points([p], rubrics.NCR)[0] == pytest.approx(12*0.2 + 8)


def test_strip_potm_inverts_baseline_term():
    base = np.array([-3.0, 0.0, 5.0, 18.0, 19.0, 40.0, 3.0, 7.5, 20.0])
    started = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    cap = np.where(started, 0.13, 0.05)
    with_potm = base + np.clip(base/145.0, 0, cap)*15
    assert rubrics.strip_potm(with_potm, started) == pytest.approx(base)


def test_slates_respect_rounds_late_games_and_locks():
    kick = pd.to_datetime(['2024-03-01 19:00', '2024-03-02 15:00', '2024-03-02 17:00', '2024-03-20 19:00',
                           '2024-02-29 20:00', '2024-07-06 09:00', '2024-07-06 15:00'], utc=True)
    fixtures = pd.DataFrame({
        'fixture_id': list('abcdefg'), 'match_at': kick,
        'competition_id_cache': [1230, 1230, 1230, 1230, 1236, 30, 1296],
        'source_season': [2024]*7, 'source_round': [20, 20, 20, 20, 15, 1, 2],
        'competition_level': ['club']*5 + ['international']*2})
    fixtures['comp'] = fixtures.competition_id_cache.map(slates.COMPETITIONS)
    out = slates.assign_slates(fixtures).set_index('fixture_id')
    assert out.loc['a', 'slate'] == out.loc['c', 'slate'] == 'top14_2024_r20'
    assert out.loc['d', 'slate'] == 'top14_2024_r20_late'
    assert out.loc['e', 'lock'] == pd.Timestamp('2024-02-01', tz='UTC')
    assert out.loc['a', 'lock'] == pd.Timestamp('2024-03-01', tz='UTC')
    # Tests from different competitions in one ISO week share an international slate.
    assert out.loc['f', 'slate'] == out.loc['g', 'slate'] == 'intl_2024_w27'
    assert out.loc['f', 'family'] == 'international'
    assert (out['slate_first_kickoff'] >= out['lock']).all()


def test_stats_helpers():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({'slate': np.repeat(np.arange(40), 2), 'engine': ['a', 'b']*40,
                          'value': rng.normal(0, 1, 80)})
    frame.loc[frame.engine.eq('b'), 'value'] += 0.5
    result = paired_bootstrap(frame, 'b', 'a', 'value', cluster='slate', n_boot=500)
    assert result['p05'] > 0 and result['n'] == 40
    # n = ((z_a + z_b) sd / delta)^2 with z = 1.96 + 0.84.
    assert power_slates(sd=10, delta=5) == pytest.approx(((1.959964 + 0.841621)*2)**2, rel=1e-4)
    players = pd.DataFrame({'slate': [1]*5 + [2]*5, 'predicted': [1, 2, 3, 4, 5]*2,
                            'actual': [2, 4, 6, 8, 10, 5, 4, 3, 2, 1]})
    corr = slate_correlations(players, 'predicted', 'actual')
    assert corr.pearson.tolist() == pytest.approx([1.0, -1.0])


class _Fixed:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict_frame(self, frame):
        return self.predictions


def test_sk_reblend_recovers_the_tree_exactly():
    from model.unified.raw_benchmark.blend import EventWeightedBlend
    from research.big_benchmark.status import reblend
    minutes = EventDistribution('lognormal', 60.0, 90.0)

    def player(events, mins=minutes):
        return RawPrediction('f', 'p', 'n', 't', 'o', 'Prop', True, events, mins)

    tree = [player({'tries': EventDistribution('negative_binomial', 0.3, 2.0),
                    'metres': EventDistribution('lognormal', 40.0, 600.0),
                    'potm': EventDistribution('bernoulli', 0.05, 1.0)})]
    plain = [player({'tries': EventDistribution('negative_binomial', 0.2, 1.5),
                     'metres': EventDistribution('lognormal', 30.0, 500.0),
                     'drop_goal_missed': EventDistribution('poisson', 0.01, 1.0)})]
    new = [player({'tries': EventDistribution('negative_binomial', 0.25, 1.2),
                   'metres': EventDistribution('lognormal', 33.0, 450.0),
                   'drop_goal_missed': EventDistribution('poisson', 0.02, 1.0)}, EventDistribution('lognormal', 61.0, 95.0))]
    weights = {'tries': 0.1, 'metres': 0.1}
    blended = EventWeightedBlend(_Fixed(plain), _Fixed(tree), 0.5, weights).predict_frame(None)
    direct = EventWeightedBlend(_Fixed(new), _Fixed(tree), 0.5, weights).predict_frame(None)[0]
    rebuilt = reblend(blended, plain, new, 0.5, weights)[0]
    for event in ('tries', 'metres', 'potm', 'drop_goal_missed'):
        assert rebuilt.events[event].mean == pytest.approx(direct.events[event].mean, rel=1e-9)
        assert rebuilt.events[event].dispersion == pytest.approx(direct.events[event].dispersion, rel=1e-6)
    assert rebuilt.minutes.mean == pytest.approx(direct.minutes.mean)


def test_null_factor_is_reproducible_and_three_percent():
    from research.big_benchmark.score import null_factor
    keys = pd.DataFrame({'fixture_id': np.arange(20000).astype(str), 'player_id': '7'})
    a, b = null_factor(keys, 'null3_a'), null_factor(keys, 'null3_a')
    assert np.array_equal(a, b)
    assert np.std(np.log(a)) == pytest.approx(0.03, rel=0.05)
    assert abs(np.corrcoef(np.log(a), np.log(null_factor(keys, 'null3_b')))[0, 1]) < 0.05


def test_block_bootstrap_weights_slates_and_flags_samples():
    from research.big_benchmark.report import add_sample_flags, interval
    frame = pd.DataFrame({'slate': ['s1', 's1', 's2', 's2', 's3', 's3'], 'block': ['b1']*4 + ['b2']*2,
                          'engine': ['a', 'b']*3, 'value': [0, 1, 0, 1, 0, 4], 'family': 'club'})
    result = paired_bootstrap(frame, 'b', 'a', 'value', cluster='block', unit='slate', n_boot=200)
    assert result['mean'] == pytest.approx(2.0) and result['clusters'] == 2 and result['n'] == 3
    manifest = {'slates': [{'slate': 's1', 'coefficients_in_sample': True, 'official_selection_overlap': False},
                           {'slate': 's2', 'coefficients_in_sample': False, 'official_selection_overlap': True},
                           {'slate': 's3', 'coefficients_in_sample': False, 'official_selection_overlap': False}]}
    flagged = add_sample_flags(frame, manifest)
    assert flagged.drop_duplicates('slate')['sample'].tolist() == ['in_sample', 'in_sample', 'out_of_sample']
    row = pd.Series({'slate_p05': -1.0, 'slate_p95': 1.0, 'blocks': 2, 'block_p05': 0.5, 'block_p95': 0.6,
                     'locks': 9, 'lock_p05': -2.0, 'lock_p95': 0.5})
    assert interval(row) == (-2.0, 1.0)  # two blocks are too few to count
