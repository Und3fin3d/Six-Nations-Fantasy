"""Within-player status (bench vs start) per-minute rate factors by position x event, by level and era."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
st = pd.read_pickle(B+'/store.pkl')
st['date'] = pd.to_datetime(st.date)
EV = ['tries', 'try_assists', 'defenders_beaten', 'offload', 'tackles', 'tackle_turnover', 'metres', 'runs', 'clean_breaks', 'penalties_conceded', 'missed_tackles', 'turnovers_conceded']


def factors(f, events=EV, iters=4):
    out = {}
    f = f[pd.to_numeric(f.minutes, errors='coerce').gt(0) & f.position.ne('Unknown')].copy()
    m = f.minutes.astype(float).to_numpy()
    g = list(zip(f.position, f.started.astype(bool)))
    gi = pd.Index(sorted(set(g)))
    gidx = gi.get_indexer(g)
    pid = pd.factorize(f.player_id)[0]
    for e in events:
        ok = f[f'available__{e}'].fillna(False).astype(bool).to_numpy() & f[e].notna().to_numpy()
        y = f[e].to_numpy(float)
        s = np.ones(len(gi))
        for _ in range(iters):
            den_p = np.bincount(pid[ok], weights=(m*s[gidx])[ok], minlength=pid.max()+1)
            num_p = np.bincount(pid[ok], weights=y[ok], minlength=pid.max()+1)
            r = np.where(den_p > 0, num_p/np.maximum(den_p, 1e-9), 0)
            num_g = np.bincount(gidx[ok], weights=y[ok], minlength=len(gi))
            den_g = np.bincount(gidx[ok], weights=(m*r[pid])[ok], minlength=len(gi))
            s = np.where(den_g > 0, num_g/np.maximum(den_g, 1e-9), 1)
        for k, (pos, started) in enumerate(gi):
            out[(pos, started, e)] = s[k]
    T = pd.Series(out).unstack(2)
    # report bench relative to start per position
    rel = T.xs(False, level=1)/T.xs(True, level=1)
    return rel

pre = st[st.date < '2025-01-01']
for name, f in (('intl<2025', pre[pre.competition_level.eq('international')]), ('club<2025', pre[pre.competition_level.eq('club')]),
                ('club 2025+', st[(st.date >= '2025-01-01') & st.competition_level.eq('club')]), ('intl 2025+', st[(st.date >= '2025-01-01') & st.competition_level.eq('international')])):
    print('=====', name, len(f)); print(factors(f).round(2).to_string())
