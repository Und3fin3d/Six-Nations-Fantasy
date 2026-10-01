"""Sensitivity: development selections with official 2023 round 1 excluded (suspected row shift)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from compare_api_official import norm_key
S = os.environ['OFFICIAL_STATS_SCRATCH']
pd.set_option('display.width', 250)
# --- POTM beta on 2023 rounds 2-4 only
P = pd.read_pickle(S+'/agentA/potm_study.pkl')
P = P[P.calendar_year.eq(2023) & P['round'].between(2, 4)]
def probs(beta, bench=0.05):
    p = pd.Series(0.0, index=P.index)
    for (f, t), g in P.groupby(['fixture_id', 'team']):
        w = np.exp(beta*(g.xp-g.xp.max()))*np.where(g.started, 1, bench)
        p.loc[g.index] = g.pwin.iloc[0]*w/w.sum()
    return p/p.groupby(P.fixture_id).transform('sum')
print('POTM 2023 R2-4 matches', P.fixture_id.nunique())
for beta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
    p = probs(beta); print(' beta', beta, 'match log-loss', round(-np.log(p[P.y == 1]).mean(), 3))
# --- BS shrinkage on 2023 rounds 3-4, history = 2023 rounds >=2 only
D = pd.read_pickle(S+'/agentA/bs_study2.pkl')
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched & J.calendar_year.eq(2023) & J['round'].ge(2) & J.minutes.gt(0)]
J['key'] = J.player_name.map(norm_key)
D = D[D.calendar_year.eq(2023) & D['round'].between(3, 4) & D.position.ne('Unknown')].copy()
y = D.off_BS.fillna(0)
def dev(mu): mu = np.clip(mu, 1e-6, None); return 2*np.mean(np.where(y > 0, y*np.log(y/mu), 0) - (y-mu))
pos = []
for rnd, key, position in zip(D['round'], D.key, D.position):
    h = J[J['round'] < rnd]
    glob = h.off_BS.sum()/h.minutes.sum()*80
    g = h[h.position.eq(position)]
    pos.append((80*g.off_BS.sum() + glob*400)/(g.minutes.sum() + 400))
D['pos_rate'] = pos
expo = D.min_model/80
rows = [dict(krp='model', koff='', dev=dev(D.tt_model), mse=np.mean((y-D.tt_model)**2)),
        dict(krp='pos only', koff='', dev=dev(D.pos_rate*expo), mse=np.mean((y-D.pos_rate*expo)**2))]
for krp in (5, 10, 20):
    prior = D.pos_rate*D[f'rprel{krp}']
    rows.append(dict(krp=krp, koff='none', dev=dev(prior*expo), mse=np.mean((y-prior*expo)**2)))
    for koff in (3, 6, 12, 24):
        b = np.array([J[(J['round'] < r) & J.key.eq(k)].off_BS.sum() for r, k in zip(D['round'], D.key)])
        m = np.array([J[(J['round'] < r) & J.key.eq(k)].minutes.sum()/80 for r, k in zip(D['round'], D.key)])
        lam = (b + koff*prior)/(m + koff)
        rows.append(dict(krp=krp, koff=koff, dev=dev(lam*expo), mse=np.mean((y-lam*expo)**2)))
print('BS 2023 R3-4 (history R2+):'); print(pd.DataFrame(rows).round(4).to_string(index=False))
# --- 2023 definitions without round 1
J2 = pd.read_pickle(S+'/agentA/join6n.pkl'); J2 = J2[J2.off_matched & J2.calendar_year.eq(2023)]
for label, g in (('all R1-4', J2), ('R2-4', J2[J2['round'].ge(2)])):
    X = g[['metres', 'runs']].fillna(0).to_numpy(float)
    a, b = np.linalg.lstsq(X, g.off_MC.fillna(0).to_numpy(float), rcond=None)[0]
    print('2023', label, 'MC~metres+runs a=%.2f b=%.2f' % (a, b), 'BS vs API tt corr', round(np.corrcoef(g.tackle_turnover.fillna(0), g.off_BS.fillna(0))[0, 1], 3),
          'MC vs metres corr', round(np.corrcoef(g.metres.fillna(0), g.off_MC.fillna(0))[0, 1], 3),
          'back-three BS/80', round(80*g[g.position.eq('Back-three')].off_BS.sum()/g[g.position.eq('Back-three')].minutes.sum(), 3))
