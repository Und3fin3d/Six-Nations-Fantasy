"""Positional bias / MAE / MSE by engine from position_supersub_eval players.pkl."""
from common import *
import sys
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
path = sys.argv[1]; excl = sys.argv[2:] if len(sys.argv) > 2 else []
P = pd.read_pickle(path)
P = P[np.isfinite(P.actual) & ~P.slate.isin(excl)]
eng = [e for e in ['p3_robust_native', 'status_rates', 'status_shrunk', 'MK', 'status_MK', 'status_shrunk_MK', 'empirical_baseline'] if e in set(P.engine)]
P = P[P.engine.isin(eng)]
P['err'] = P.predicted - P.actual
T = P.groupby(['status', 'position', 'engine']).err.mean().unstack('engine')[[e for e in eng if e in P.engine.unique()]]
n = P[P.engine.eq('p3_robust_native')].groupby(['status', 'position']).size()
T.insert(0, 'n', n)
print('bias'); print(T.round(2).to_string())
G = P.groupby(['status', 'engine']).agg(bias=('err', 'mean'), mae=('err', lambda e: e.abs().mean()), mse=('err', lambda e: (e**2).mean())).unstack('engine')
print(G.round(3).to_string())
# bench within-slate pearson per engine
bp = P[P.status.eq('B')].groupby(['engine', 'slate']).apply(lambda g: np.corrcoef(g.predicted, g.actual)[0, 1]).groupby('engine').mean()
print('bench pearson', bp.round(4).to_dict())
