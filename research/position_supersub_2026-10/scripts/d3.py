from common import *
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl'); blk = pd.read_pickle(B+'/blocks.pkl')
six = off[off.competition.eq('six_nations')]
blk['pre'] = ~blk.block.str.contains('2025|2026')
COMPS = ['c_tries', 'c_assists', 'c_kicking', 'c_db', 'c_offload', 'c_tackles', 'c_turnover', 'c_pens', 'c_cards', 'c_metres']
def comp(f, name):
    f = f.copy(); f['started'] = f.started.astype(bool)
    t = f.groupby(['started', 'position']).apply(lambda g: pd.Series({**{'n': len(g)}, **{c[2:]: (g['p_'+c]-g['a_'+c]).mean() for c in COMPS}, 'obs': (g.p_obs-g.a_obs).mean()}))
    print('===', name); print(t.round(2).to_string())
    return t
out = {}
for name, f in (('official6N', six), ('devcr', dev), ('blocks_pre2025', blk[blk.pre]), ('blocks_2025_26', blk[~blk.pre])):
    out[name] = comp(f, name)
pd.concat(out).round(3).to_csv(B+'/api_component_bias.csv')
# official Ta vs API tackles
P = pd.read_pickle(S+'/anatomy.pkl')
m = six.assign(key=six.player_name.map(__import__('compare_api_official').norm_key))
m = m.merge(P[['season','round','team','key','Ta','MC','Min','DB','OF']], on=['season','round','team','key'], how='inner')
print('official Ta / API tackles', m.Ta.sum()/m.a_tackles.sum(), ' MC/metres', m.MC.sum()/m.a_metres.sum(), ' DB', m.DB.sum()/m.a_defenders_beaten.sum(), ' OF', m.OF.sum()/m.a_offload.sum(), ' Min', m.Min.sum()/m.a_minutes.sum())
print(m.groupby(['started','position']).apply(lambda g: pd.Series(dict(Ta_ratio=g.Ta.sum()/g.a_tackles.sum(), MC_ratio=g.MC.sum()/g.a_metres.sum(), min_ratio=g.Min.sum()/g.a_minutes.sum()))).round(2))
