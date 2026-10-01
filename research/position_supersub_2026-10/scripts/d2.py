from common import *
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
P = pd.read_pickle(S+'/anatomy.pkl')
comps = ['tries','metres','tackles','def_beaten','kicking','breakdown_steal','scrums','potm','offloads','assists','kicks_retained','lineout_steal','pens_conceded','cards','fifty22']
api_side = ['tries','kicking','assists','def_beaten','offloads','tackles','pens_conceded','cards']
P['gap_other'] = P.actual - P.a_sum
rows = []
for (st, pos), g in P.groupby(['started', 'pos']):
    r = {'started': st, 'pos': pos, 'n': len(g), 'bias': (g.pred-g.actual).mean()}
    for c in comps:
        r[c] = (g['p_'+c]-g['a_'+c]).mean()
    r['recon_gap'] = -(g.gap_other).mean()
    rows.append(r)
T = pd.DataFrame(rows).set_index(['started', 'pos'])
T['api_like'] = T[api_side].sum(axis=1)
T['official_only'] = T[['breakdown_steal','potm','scrums','kicks_retained','lineout_steal','fifty22']].sum(axis=1)
print(T.round(2).to_string())
T.round(3).to_csv(B+'/official_component_bias.csv')
# bench back-row deep dive: per-80 official BS & metres vs predicted
g = P[~P.started]
print(g.groupby('pos')[['Min','BS','MC','Ta','POTM']].mean().round(2))
print('pred BS*5 / actual BS*5 by pos (bench)'); print(g.groupby('pos')[['p_breakdown_steal','a_breakdown_steal','p_metres','a_metres','p_potm','a_potm']].mean().round(2))
