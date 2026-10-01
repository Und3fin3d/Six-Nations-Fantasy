"""Block/friendly metrics for all engines vs reference; optional era filter (pre/post/friendly)."""
from common import *
import sys
import numpy as np, pandas as pd
from model.unified.friendly_eval import COUNT_TARGETS
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
d = sys.argv[1]; kind = sys.argv[2]; want = sys.argv[3]
E = pd.read_csv(d+'/event_metrics.csv'); P = pd.read_pickle(d+'/players.pkl'); C = pd.read_csv(d+'/count_stats.csv')
era = (lambda u: np.where(pd.Series(u).str.contains('2025|2026'), 'post', 'pre')) if kind == 'blocks' else (lambda u: np.array(['friendly']*len(u)))
E['era'] = era(E.unit); P['era'] = era(P.unit); C['era'] = era(C.unit)
E, P, C = E[E.era.eq(want)], P[P.era.eq(want)], C[C.era.eq(want)]
a = E.groupby(['event', 'status', 'engine'])[['n', 'pred_sum', 'actual_sum', 'sse', 'deviance', 'abs']].sum()
a['ratio'] = a.pred_sum/a.actual_sum; a['mse'] = a.sse/a.n; a['dev'] = a.deviance/a.n
print(a[['ratio', 'mse', 'dev']].unstack('engine').round(4).to_string())
al = a.xs('all', level='status')
for eng in al.index.get_level_values('engine').unique():
    if eng == 'p3_robust_native':
        continue
    dv = al['dev'].unstack('engine'); ms = al['mse'].unstack('engine')
    print(eng, 'lower deviance', int((dv[eng] < dv.p3_robust_native).sum()), '/', len(dv), ' lower MSE', int((ms[eng] < ms.p3_robust_native).sum()), '/', len(ms))
print('count-stat MAE', C[C.target.isin(COUNT_TARGETS)].groupby('engine').mae.mean().round(5).to_dict())
P['err'] = P.predicted-P.actual
t = P.groupby(['started', 'position', 'engine']).err.mean().unstack('engine')
t.insert(0, 'n', P[P.engine.eq('p3_robust_native')].groupby(['started', 'position']).size())
print('observable points bias'); print(t.round(2).to_string())
s = P.groupby(['started', 'engine']).agg(bias=('err', 'mean'), mae=('err', lambda x: x.abs().mean()), mse=('err', lambda x: (x**2).mean())).unstack('engine')
print(s.round(3).to_string())
pr = P.groupby(['engine', 'unit']).apply(lambda h: np.corrcoef(h.predicted, h.actual)[0, 1]).groupby('engine').mean()
pb = P[~P.started].groupby(['engine', 'unit']).apply(lambda h: np.corrcoef(h.predicted, h.actual)[0, 1]).groupby('engine').mean()
ps = P[P.started].groupby(['engine', 'unit']).apply(lambda h: np.corrcoef(h.predicted, h.actual)[0, 1]).groupby('engine').mean()
print('pearson', pr.round(4).to_dict()); print('bench pearson', pb.round(4).to_dict()); print('starter pearson', ps.round(4).to_dict())
