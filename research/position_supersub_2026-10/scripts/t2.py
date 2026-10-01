"""Bench back-row (and other bench groups) API-component change under S on 6N 2025-26 official slates."""
from common import *
import numpy as np, pandas as pd
from pathlib import Path
from research.position_supersub import load_raw
import build
pd.set_option('display.width', 250)
off = pd.read_pickle(B+'/off.pkl'); off = off[off.competition.eq('six_nations')]
rows = []
for slate, g in off.groupby('slate'):
    f = build.pred_frame(load_raw(Path(B)/'runs'/'status'/'official'/slate/'status.jsonl'), prefix='s_')
    g = g.reset_index(drop=True)
    for c in build.COMPS + ['obs']:
        g['s_'+c] = f['s_'+c].to_numpy()
    rows.append(g)
O = pd.concat(rows)
out = []
for (st, pos), g in O.groupby(['started', 'position']):
    r = {'started': st, 'position': pos, 'n': len(g), 'pred_min': g.p_minutes.mean(), 'act_min': g.a_minutes.mean()}
    for c in build.COMPS + ['obs']:
        r['ref_'+c[2:] if c != 'obs' else 'ref_obs'] = (g['p_'+c]-g['a_'+c]).mean()
        r['S_'+c[2:] if c != 'obs' else 'S_obs'] = (g['s_'+c]-g['a_'+c]).mean()
    out.append(r)
T = pd.DataFrame(out).set_index(['started', 'position'])
cols = ['n', 'pred_min', 'act_min'] + [f'{e}_{c}' for c in ['tries', 'tackles', 'db', 'metres', 'turnover', 'obs'] for e in ('ref', 'S')]
print(T[cols].round(2).to_string())
T.round(3).to_csv(Path(W)/'research'/'position_supersub_2026-10'/'test_api_components_status.csv')
