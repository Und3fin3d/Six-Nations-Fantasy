"""Pseudo-development selection of BS shrinkage on 2023 (R2-R4) official rows only.
RP features come from seasons completed before the 2025 lock (no 2025/26 outcomes used)."""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
D = pd.read_pickle(S+'/agentA/bs_study2.pkl')
D = D[D.calendar_year.eq(2023) & D['round'].between(2, 4) & D.position.ne('Unknown')].copy()
y = D.off_BS.fillna(0)
def dev(mu): mu = np.clip(mu, 1e-6, None); return 2*np.mean(np.where(y > 0, y*np.log(y/mu), 0) - (y-mu))
# official EB counts within 2023 earlier rounds
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched & J.calendar_year.eq(2023)]
import sys; sys.path.insert(0, '.')
from compare_api_official import norm_key
J['key'] = J.player_name.map(norm_key)
out = []
expo = D.min_model/80
for krp in (5, 10, 20):
    prior = D.pos_rate*D[f'rprel{krp}']
    out.append(dict(krp=krp, koff='none', dev=dev(prior*expo), mse=np.mean((y-prior*expo)**2), corr=np.corrcoef(y, prior*expo)[0, 1]))
    for koff in (3, 6, 12, 24):
        b = []; m = []
        for (rnd, key) in zip(D['round'], D.key):
            h = J[(J['round'] < rnd) & J.key.eq(key)]
            b.append(h.off_BS.sum()); m.append(h.minutes.sum()/80)
        b = np.array(b); m = np.array(m)
        lam = (b + koff*prior)/(m + koff)
        out.append(dict(krp=krp, koff=koff, dev=dev(lam*expo), mse=np.mean((y-lam*expo)**2), corr=np.corrcoef(y, lam*expo)[0, 1]))
out.append(dict(krp='model', koff='', dev=dev(D.tt_model), mse=np.mean((y-D.tt_model)**2), corr=np.corrcoef(y, D.tt_model)[0, 1]))
out.append(dict(krp='pos only', koff='', dev=dev(D.pos_rate*expo), mse=np.mean((y-D.pos_rate*expo)**2), corr=np.corrcoef(y, D.pos_rate*expo)[0, 1]))
print(pd.DataFrame(out).round(4).to_string(index=False))
