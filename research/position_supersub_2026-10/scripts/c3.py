"""Super-sub decision on archived blocks (observable points): one 'round' = one block weekend.

Pick = bench player with the highest forecast in the weekend pool; value = 3 x observable points
(0 if he did not play). Compares engines and simple rules, with weekend-bootstrap intervals.
"""
from common import *
import numpy as np, pandas as pd
P = pd.read_pickle(B+'/runs/eval_blocks2/players.pkl')
st = pd.read_pickle(B+'/store.pkl')[['fixture_id', 'date']].drop_duplicates('fixture_id')
P = P.merge(st, on='fixture_id', how='left')
P['date'] = pd.to_datetime(P.date)
P['week'] = P.date.dt.to_period('W-MON').astype(str)
P['era'] = np.where(P.unit.str.contains('2025|2026'), 'post', 'pre')
P['real'] = np.where(P.minutes > 0, P.actual, 0.0)
B_ = P[~P.started]
wide = B_.pivot_table(index=['era', 'unit', 'week', 'fixture_id', 'player_id', 'position', 'real'], columns='engine', values='predicted').reset_index()
rows = []
for (era, unit, week), g in wide.groupby(['era', 'unit', 'week']):
    if g.fixture_id.nunique() < 2:
        continue
    r = dict(era=era, unit=unit, week=week, n=len(g), random=3*g.real.mean(), hindsight=3*g.real.max())
    for eng in ('p3_robust_native', 'status_rates', 'status_shrunk'):
        r[eng] = 3*g.real.iloc[np.argmax(g[eng].to_numpy())]
        # top-k mean: expected value of a pick drawn among near-ties (within 0.5 pts)
        top = g[eng].max()
        r[eng+'_tie_n'] = int((g[eng] > top-0.5).sum())
    br = g[g.position.eq('Back-row')]
    r['top_backrow'] = 3*br.real.iloc[np.argmax(br.p3_robust_native.to_numpy())] if len(br) else np.nan
    r['same_pick'] = g.p3_robust_native.idxmax() == g.status_rates.idxmax()
    rows.append(r)
R = pd.DataFrame(rows)
rng = np.random.default_rng(3)
cols = ['random', 'p3_robust_native', 'status_rates', 'top_backrow', 'hindsight', 'status_shrunk']
for era, g in R.groupby('era'):
    print('=====', era, 'weekends', len(g), 'mean pool', g.n.mean().round(1), 'same pick share', g.same_pick.mean().round(2),
          'median near-ties (within 0.5)', g.p3_robust_native_tie_n.median())
    m = g[cols].mean()
    boots = np.array([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))[cols].mean().to_numpy() for _ in range(2000)])
    d = boots[:, 2]-boots[:, 1]
    print((m*10).round(1).to_string(), '\n(per 10 weekends)')
    print('status - ref per 10 weekends: %.1f  90%% [%.1f, %.1f]' % (10*(m.status_rates-m.p3_robust_native), *np.percentile(10*d, [5, 95])))
    d2 = boots[:, 3]-boots[:, 1]
    print('top back-row - ref per 10: %.1f [%.1f, %.1f]' % (10*(m.top_backrow-m.p3_robust_native), *np.percentile(10*d2, [5, 95])))
    d3 = boots[:, 5]-boots[:, 1]
    print('shrunk - ref per 10: %.1f [%.1f, %.1f]' % (10*(m.status_shrunk-m.p3_robust_native), *np.percentile(10*d3, [5, 95])))
R.to_csv(B+'/supersub_blocks.csv', index=False)
