from common import *
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl'); blk = pd.read_pickle(B+'/blocks.pkl')
print(off.groupby('competition').a_minutes.apply(lambda s: s.isna().mean()))
six = off[off.competition.eq('six_nations')].copy()
COMPS = ['c_tries', 'c_assists', 'c_kicking', 'c_db', 'c_offload', 'c_tackles', 'c_turnover', 'c_pens', 'c_cards', 'c_metres']


def table(f, actual='actual_pts', pred='p_points'):
    f = f.copy()
    f['started'] = f.started.astype(bool)
    def agg(g):
        r = dict(n=len(g), pred=g[pred].mean(), act=g[actual].mean(), bias=(g[pred]-g[actual]).mean(),
                 mae=(g[pred]-g[actual]).abs().mean(),
                 pmin=g.p_minutes.mean(), amin=g.a_minutes.mean(), played=(g.a_minutes > 0).mean(),
                 obs_bias=(g.p_obs-g.a_obs).mean())
        ratio = np.where(g.p_minutes > 0, g.a_minutes/g.p_minutes, 0)
        r['min_part'] = (g.p_obs*(1-ratio)).mean()
        r['rate_part'] = (g.p_obs*ratio-g.a_obs).mean()
        return pd.Series(r)
    return f.groupby(['started', 'position']).apply(agg).round(2)

print('=== OFFICIAL 6N 2025-26 (official pts) ===')
print(table(six))
print('=== DEVCR 6N 2023-24 (current-rubric labels) ===')
print(table(dev))
print('=== DEVCR by season ===')
for s, g in dev.groupby('season'):
    print(s); print(table(g)[['n', 'bias', 'pmin', 'amin', 'obs_bias', 'min_part', 'rate_part']])
blk['pre'] = ~blk.block.str.contains('2025|2026')
blk['actual_obs'] = blk.a_obs
print('=== BLOCKS pre-2025 (API obs points; tournament-start forecasts) ===')
print(table(blk[blk.pre], 'a_obs', 'p_obs'))
print('=== BLOCKS 2025-26 ===')
print(table(blk[~blk.pre], 'a_obs', 'p_obs'))
