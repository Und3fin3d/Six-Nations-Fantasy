"""Null for super-sub differences: SD of smoothed bench-argmax super-sub points across 'equally informed'
3%-perturbed copies of robust P3 (bench arg-max proxy; nation cap ignored)."""
from common import *
import numpy as np, pandas as pd
P = pd.read_pickle(B+'/runs/eval_official2/players.pkl')
off = pd.read_pickle(B+'/off.pkl')[['slate', 'pool_id', 'a_minutes']]
P = P.merge(off, on=['slate', 'pool_id'], how='left')
rng = np.random.default_rng(5)
rows = []
for comp, pat in (('6N 2025', '2025'), ('6N 2026', '2026_r'), ('NCR 2026', 'ncr')):
    g = P[P.slate.str.contains(pat) & P.status.eq('B')]
    if comp == 'NCR 2026':
        continue  # NCR minutes are not in the research store rows used here
    res = {}
    for eng in ('p3_robust_native', 'status_rates', 'status_shrunk', 'MK', 'status_MK', 'status_shrunk_MK'):
        h = g[g.engine.eq(eng)]
        tot = 0
        for _, s in h.groupby('slate'):
            real = np.where(s.a_minutes.to_numpy() > 0, np.nan_to_num(s.actual.to_numpy()), 0)
            x = s.predicted.to_numpy()
            picks = (x*np.exp(rng.normal(0, .01, (400, len(x))))).argmax(1)
            tot += 3*real[picks].mean()
        res[eng] = tot
    h = g[g.engine.eq('p3_robust_native')]
    copies = []
    for _ in range(300):
        tot = 0
        for _, s in h.groupby('slate'):
            real = np.where(s.a_minutes.to_numpy() > 0, np.nan_to_num(s.actual.to_numpy()), 0)
            x = s.predicted.to_numpy()*np.exp(rng.normal(0, .03, len(s)))
            picks = (x*np.exp(rng.normal(0, .01, (40, len(x))))).argmax(1)
            tot += 3*real[picks].mean()
        copies.append(tot)
    rows.append(dict(comp=comp, **{k: round(v, 1) for k, v in res.items()}, null_mean=np.mean(copies), null_sd=np.std(copies)))
R = pd.DataFrame(rows); print(R.round(1).to_string(index=False))
R.round(1).to_csv(W+'/research/position_supersub_2026-10/test_supersub_null.csv', index=False)
