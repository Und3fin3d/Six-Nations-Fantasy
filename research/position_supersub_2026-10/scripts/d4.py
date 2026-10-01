"""Per-component (robust empirical vs v4 trees) event bias by status x position, pre-2025 blocks + 2025-26 blocks."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
C = pd.read_pickle(S+'/components_raw.pkl')
store = pd.read_pickle(B+'/store.pkl')
KEY = ['fixture_id', 'player_id', 'team']
EV = ['tries', 'tackles', 'metres', 'defenders_beaten', 'tackle_turnover', 'minutes']
print(C.columns[:12].tolist(), C.shape)
f = pd.DataFrame({k: C[('', k)] if ('', k) in C.columns else C[k] for k in KEY+['block', 'position']})
for comp in ('v4', 'remp', 'rob'):
    for e in EV:
        f[f'{comp}_{e}'] = C[(comp, e)].to_numpy()
f = f.merge(store[KEY+['started', 'jersey']+EV], on=KEY, how='left')
f['pre'] = ~f.block.str.contains('2025|2026')
for pre, g in f.groupby('pre'):
    rows = []
    for (st, pos), h in g.groupby(['started', 'position']):
        r = {'pre': pre, 'started': st, 'pos': pos, 'n': len(h)}
        for e in EV:
            a = h[e].sum()
            for comp in ('v4', 'remp', 'rob'):
                r[f'{e[:4]}{e[-3:]}_{comp}'] = h[f'{comp}_{e}'].sum()/max(a, 1e-9)
        rows.append(r)
    T = pd.DataFrame(rows).set_index(['pre', 'started', 'pos'])
    print(T.round(2).to_string())
f.to_pickle(B+'/blocks_components.pkl')
