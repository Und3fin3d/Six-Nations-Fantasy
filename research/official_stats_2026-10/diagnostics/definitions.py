import numpy as np, pandas as pd
from scipy.stats import spearmanr
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
J = pd.read_pickle(S+'/agentA/join6n.pkl')
J = J[J.off_matched].copy()
J['yr'] = J.calendar_year
print('=== BS vs API tackle_turnover')
for yr, g in J.groupby('yr'):
    tt = g.tackle_turnover.fillna(0); bs = g.off_BS.fillna(0)
    print(yr, 'n', len(g), 'API tt sum', tt.sum(), 'off BS sum', bs.sum(), 'corr', round(np.corrcoef(tt, bs)[0, 1], 3),
          'exact', round((tt == bs).mean(), 3), 'tt avail', g.available__tackle_turnover.mean().round(3))
print(J.groupby(['yr', 'position']).apply(lambda g: pd.Series({'n': len(g), 'min': g.minutes.sum(),
      'tt80': 80*g.tackle_turnover.sum()/g.minutes.sum(), 'bs80': 80*g.off_BS.sum()/g.minutes.sum(),
      'corr': np.corrcoef(g.tackle_turnover.fillna(0), g.off_BS.fillna(0))[0, 1]})).round(3).unstack(0).to_string())
print('=== metres')
for yr, g in J.groupby('yr'):
    m = g.metres.fillna(0); mc = g.off_MC.fillna(0)
    print(yr, 'ratio', round(mc.sum()/m.sum(), 3), 'corr', round(np.corrcoef(m, mc)[0, 1], 3), 'spearman', round(spearmanr(m, mc)[0], 3),
          'corr floor pts', round(np.corrcoef(np.floor(m/10), np.floor(mc/10))[0, 1], 3))
print(J.groupby(['yr', 'position']).apply(lambda g: g.off_MC.sum()/g.metres.sum()).round(3).unstack(0).to_string())
# per-player ratio stability: player-season ratio for players with >=150 API metres in both seasons
ps = J.groupby(['player_id', 'yr']).agg(m=('metres', 'sum'), mc=('off_MC', 'sum'), n=('minutes', 'size'), pos=('position', 'first')).reset_index()
ps['r'] = ps.mc/ps.m
w = ps[ps.m >= 100].pivot(index='player_id', columns='yr', values='r')
for a, b in [(2023, 2025), (2025, 2026), (2023, 2026)]:
    x = w[[a, b]].dropna()
    print('ratio stability', a, b, 'n', len(x), 'corr', round(np.corrcoef(x[a], x[b])[0, 1], 3))
# residual of MC on API metres: is the residual (MC - k*m) a stable player trait?
J['mres'] = J.off_MC - J.groupby('yr').apply(lambda g: g.off_MC.sum()/g.metres.sum()).reindex(J.yr).to_numpy()*J.metres
pr = J.groupby(['player_id', 'yr']).agg(res=('mres', 'mean'), n=('mres', 'size')).reset_index()
w = pr[pr.n >= 3].pivot(index='player_id', columns='yr', values='res')
for a, b in [(2023, 2025), (2025, 2026), (2023, 2026)]:
    x = w[[a, b]].dropna()
    print('metres residual per-match stability', a, b, 'n', len(x), 'corr', round(np.corrcoef(x[a], x[b])[0, 1], 3))
print('=== POTM')
J['pts_off'] = J.off_Pts
for yr, g in J.groupby('yr'):
    print(yr, 'POTM count', g.off_POTM.sum(), 'starters', g.loc[g.off_POTM > 0, 'started'].mean(),
          'win team share', (g.loc[g.off_POTM > 0, 'result'] == 'W').mean() if 'result' in g else None)
J['pts_nopotm'] = J.off_Pts - 15*J.off_POTM.fillna(0)
J['rank_in_match'] = J.groupby('fixture_id').pts_nopotm.rank(ascending=False, method='min')
J['rank_in_team'] = J.groupby(['fixture_id', 'team']).pts_nopotm.rank(ascending=False, method='min')
w = J[J.off_POTM > 0]
print('POTM rank in match (pts excl potm):'); print(w.rank_in_match.describe()); print(w.rank_in_match.value_counts().sort_index().head(15))
print('result of POTM team', w.result.value_counts().to_dict())
print('positions', w.position.value_counts().to_dict())
print(J.result.value_counts().to_dict())
