"""Bias by pre-match team strength (WR points edge) x status x forward/back."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl'); blk = pd.read_pickle(B+'/blocks.pkl')
six = off[off.competition.eq('six_nations')].copy()
blk['pre'] = ~blk.block.str.contains('2025|2026')
wr = pd.read_csv(W+'/data/wr_rankings.csv', parse_dates=['snapshot_date']).sort_values('snapshot_date')


def edge(f):
    f = f.copy(); f['date'] = pd.to_datetime(f.date).dt.tz_localize(None)
    out = []
    for d, g in f.groupby('date'):
        snap = wr[wr.snapshot_date < d]
        snap = snap[snap.snapshot_date.eq(snap.snapshot_date.max())].set_index('team').wr_pts
        out.append(pd.Series(g.team.map(snap).to_numpy(float)-g.opponent.map(snap).to_numpy(float), index=g.index))
    f['edge'] = pd.concat(out)
    return f


def tab(f, y, p, name):
    f = edge(f[np.isfinite(f[y].astype(float))])
    f['tier'] = pd.cut(f.edge, [-100, -5, 5, 100], labels=['underdog', 'even', 'favourite'])
    f['grp'] = pd.Series(np.where(f.started.astype(bool), 'start', 'bench'), index=f.index) + '_' + pd.Series(np.where(f.is_forward, 'fwd', 'back'), index=f.index)
    t = f.groupby(['grp', 'tier']).apply(lambda g: pd.Series(dict(n=len(g), bias=(g[p]-g[y]).mean(), act=g[y].mean(), amin=g.a_minutes.mean(), pmin=g.p_minutes.mean()))).round(2)
    print('=====', name); print(t.unstack(1).to_string())
    return t

T = {}
T['official'] = tab(six, 'actual_pts', 'p_points', 'official 6N (official pts)')
T['devcr'] = tab(dev, 'actual_pts', 'p_points', 'devcr')
T['blk_pre'] = tab(blk[blk.pre], 'a_obs', 'p_obs', 'blocks pre-2025 (API obs)')
T['blk_post'] = tab(blk[~blk.pre], 'a_obs', 'p_obs', 'blocks 2025-26 (API obs)')
pd.concat(T).to_csv(B+'/bias_by_strength.csv')
