"""POTM: who wins it, and how well can pre-lock information forecast it?"""
import json, sys
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from compare_api_official import norm_key
from model.unified.contracts import RawPrediction
from model.unified.rolling_eval import expected_points
from model.history import past_matches
from model.unified.v4.context import pre_match_context, add_candidate_context
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
store = pd.read_csv(S+'/runs/base/inputs/player_match.csv', low_memory=False, dtype={'fixture_id': str, 'player_id': str, 'team': str}, parse_dates=['date', 'match_at'])
J = pd.read_pickle(S+'/agentA/join6n.pkl')
J['match_at'] = pd.to_datetime(J.match_at, utc=True)

def load_raw(slate):
    for base in ('base', 'devcr'):
        try:
            return [RawPrediction.from_dict(json.loads(l)) for l in open(f'{S}/runs/{base}/models/{slate}/p3_robust_native.jsonl')]
        except FileNotFoundError:
            pass
rows = []
for (yr, rnd), cand in J[J.calendar_year.isin([2023, 2025, 2026])].groupby(['calendar_year', 'round']):
    raw = load_raw(f'six_nations_{yr}_r{int(rnd)}')
    if raw is None: continue
    cutoff = cand.match_at.min()
    nop = [replace(p, events={k: v for k, v in p.events.items() if k != 'potm'}) for p in raw]
    xp = expected_points(nop, 'six_nations')
    fc = pd.DataFrame({'fixture_id': [p.fixture_id for p in raw], 'player_id': [p.player_id for p in raw], 'team': [p.team for p in raw],
                       'xp': xp, 'p_model': [p.events['potm'].mean if 'potm' in p.events else 0 for p in raw]})
    hist = past_matches(store, cutoff)
    state = pre_match_context(hist)[1]
    c = cand.merge(fc, on=['fixture_id', 'player_id', 'team'], how='left')
    c['edge'] = add_candidate_context(c, state)['ctx__elo_edge'].to_numpy()
    c['pwin'] = 1/(1+10**(-c.edge/400))
    rows.append(c)
D = pd.concat(rows, ignore_index=True)
D = D[D.fixture_id.isin(D[D.off_POTM > 0].fixture_id)]  # fixtures with a labelled POTM
D['y'] = (D.off_POTM > 0).astype(int)
print('missing xp', D.xp.isna().sum(), 'missing p_model', D.p_model.isna().sum(), D[D.xp.isna()].calendar_year.value_counts().to_dict())
D['p_model'] = D.p_model.fillna(0); D['xp'] = D.xp.fillna(D.xp.median())
D['pts_nopotm'] = (D.off_Pts - 15*D.off_POTM.fillna(0)).fillna(0)
D['won'] = (D.result == 'W').astype(int)
print('fixtures', D.fixture_id.nunique(), D.groupby('calendar_year').fixture_id.nunique().to_dict())
print('POTM starters', D[D.y == 1].started.mean(), ' on winning team', D[D.y == 1].won.mean())
print('POTM xp rank within team starters:')
D['xp_rank_team'] = D[D.started].groupby(['fixture_id', 'team']).xp.rank(ascending=False)
D['xp_rank_match'] = D.groupby('fixture_id').xp.rank(ascending=False)
print(D[D.y == 1].xp_rank_team.describe().round(2).to_dict()); print('match rank', D[D.y == 1].xp_rank_match.describe().round(2).to_dict())
print('pwin of POTM team', D[D.y == 1].pwin.describe().round(3).to_dict(), 'favourite won share', D.groupby('fixture_id').apply(lambda g: g.loc[g.pwin.idxmax(), 'won']).mean())

def share(g, score, beta, bench_w):
    w = np.exp(beta*(score - score.max()))*np.where(g.started, 1.0, bench_w)
    return w/w.sum()
def potm_probs(D, beta, bench_w=0.05, use_pwin=True, score='xp'):
    p = pd.Series(0.0, index=D.index)
    for (f, t), g in D.groupby(['fixture_id', 'team']):
        tw = g.pwin.iloc[0] if use_pwin else 0.5
        p.loc[g.index] = tw*share(g, g[score], beta, bench_w)
    # renormalise per fixture to 1
    return p/p.groupby(D.fixture_id).transform('sum')
def metrics(name, p, mask):
    y = D.y[mask]; p = p[mask].clip(1e-6, 1)
    ll = -np.log(p[y == 1]).mean()  # per-match log-loss of the actual winner (multinomial)
    return dict(model=name, n_matches=int(y.sum()), mll=round(ll, 3), auc=round(roc_auc_score(y, p), 3), brier=round(np.mean((y-p)**2)*1000, 3),
                corr=round(np.corrcoef(y, p)[0, 1], 3), mean_p=round(p.mean(), 4))
out = []
for yr in (2023, 2025, 2026, 'test'):
    m = D.calendar_year.isin([2025, 2026]) if yr == 'test' else D.calendar_year.eq(yr)
    n = D[m].groupby('fixture_id').size()
    unif = 1/D[m].groupby('fixture_id').y.transform('size')
    out.append({**metrics('uniform', pd.Series(1/46, index=D.index), m), 'yr': yr})
    pm = (D.p_model/D.p_model.groupby(D.fixture_id).transform('sum')).fillna(1/46)
    out.append({**metrics('model_potm(renorm)', pm, m), 'yr': yr})
    out.append({**metrics('model_potm(raw)', D.p_model, m), 'yr': yr})
    for beta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
        out.append({**metrics(f'pwin*softmax b={beta}', potm_probs(D, beta), m), 'yr': yr})
    out.append({**metrics('no-pwin b=0.15', potm_probs(D, 0.15, use_pwin=False), m), 'yr': yr})
    # oracles (ex post, not achievable)
    Do = D.assign(pwin=D.won.astype(float))
    p_or = pd.Series(0.0, index=D.index)
    for (f, t), g in Do.groupby(['fixture_id', 'team']):
        p_or.loc[g.index] = g.pwin.iloc[0]*share(g, g.xp, 0.15, 0.05)
    out.append({**metrics('ORACLE winner known, xp b=.15', p_or/p_or.groupby(D.fixture_id).transform('sum'), m), 'yr': yr})
    p_or2 = pd.Series(0.0, index=D.index)
    for (f, t), g in Do.groupby(['fixture_id', 'team']):
        p_or2.loc[g.index] = g.pwin.iloc[0]*share(g, g.pts_nopotm, 0.15, 0.05)
    out.append({**metrics('ORACLE realised pts b=.15', p_or2/p_or2.groupby(D.fixture_id).transform('sum'), m), 'yr': yr})
print(pd.DataFrame(out).to_string(index=False))
D.to_pickle(S+'/agentA/potm_study.pkl')
