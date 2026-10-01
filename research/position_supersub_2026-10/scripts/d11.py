"""Within-position calibration slope of event forecasts (starters), pre-2025 vs 2025-26 blocks."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
f = pd.read_pickle(B+'/blocks_components.pkl')
s = f[f.started.astype(bool) & f.position.ne('Unknown')]
rows = []
for (pre, pos), g in s.groupby(['pre', 'position']):
    for e in ['tries', 'metres', 'tackles', 'defenders_beaten', 'tackle_turnover']:
        for comp in ('rob',):
            x = g[f'{comp}_{e}']; y = g[e]
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            b = np.cov(x, y)[0, 1]/x.var()
            rows.append(dict(pre=pre, pos=pos, event=e, n=len(x), ratio=x.mean()/y.mean(), slope=b))
T = pd.DataFrame(rows)
print(T.pivot_table(index=['pos'], columns=['event', 'pre'], values='slope').round(2).to_string())
print(T.pivot_table(index=['pos'], columns=['event', 'pre'], values='ratio').round(2).to_string())
# hooker tries by predicted tercile
g = s[s.position.eq('Hooker')]
g = g.assign(q=g.groupby('pre').rob_tries.transform(lambda v: pd.qcut(v, 3, labels=False)))
print(g.groupby(['pre', 'q']).agg(n=('tries', 'size'), pred=('rob_tries', 'mean'), act=('tries', 'mean')).round(3))
