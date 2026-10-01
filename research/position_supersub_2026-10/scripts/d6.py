"""Bench minutes: expected vs actual by position/jersey/split; forecast skill."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl'); blk = pd.read_pickle(B+'/blocks.pkl')
six = off[off.competition.eq('six_nations')].copy()
blk['pre'] = ~blk.block.str.contains('2025|2026')
for name, f in (('official', six), ('devcr', dev), ('blk_pre', blk[blk.pre]), ('blk_post', blk[~blk.pre])):
    b = f[~f.started.astype(bool)].copy()
    # bench split: number of forwards on bench per team-fixture
    nf = b.groupby(['fixture_id', 'team']).is_forward.transform('sum')
    b['split'] = nf.astype(int).astype(str)+'-'+(b.groupby(['fixture_id', 'team']).is_forward.transform('size')-nf).astype(int).astype(str)
    t = b.groupby(['jersey']).apply(lambda g: pd.Series(dict(n=len(g), pmin=g.p_minutes.mean(), emp=g.emp_minutes.mean() if 'emp_minutes' in g else np.nan, v4=g.v4_minutes.mean() if 'v4_minutes' in g else np.nan,
        amin=g.a_minutes.mean(), dnp=(g.a_minutes == 0).mean(), mae=(g.p_minutes-g.a_minutes).abs().mean(), corr=np.corrcoef(g.p_minutes, g.a_minutes)[0, 1])))
    print('=====', name); print(t.round(2).to_string())
    print(b.groupby('split').apply(lambda g: pd.Series(dict(n=len(g), pmin=g.p_minutes.mean(), amin=g.a_minutes.mean()))).round(2).to_string())
    print(b.groupby(['split', 'position']).apply(lambda g: pd.Series(dict(n=len(g), pmin=g.p_minutes.mean(), amin=g.a_minutes.mean()))).round(1).to_string())
    print('bench minutes MAE', (b.p_minutes-b.a_minutes).abs().mean().round(2), 'corr', np.corrcoef(b.p_minutes, b.a_minutes)[0, 1].round(3))
