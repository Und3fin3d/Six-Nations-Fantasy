"""Test analysis: bootstrap MAE/MSE deltas, positional bias (official + API-observable), super-sub table."""
from common import *
import json
import numpy as np, pandas as pd
from pathlib import Path
from research.position_supersub import load_raw
from research.position_supersub_blocks import observable_expected
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
E = Path(W)/'research'/'position_supersub_2026-10'
R = pd.read_csv(B+'/runs/eval_official2/rounds.csv'); P = pd.read_pickle(B+'/runs/eval_official2/players.pkl')
P['comp'] = np.where(P.slate.str.startswith('ncr'), 'NCR 2026', np.where(P.slate.str.contains('2025'), '6N 2025', '6N 2026'))
P = P[np.isfinite(P.actual)]
rng = np.random.default_rng(11)
out = []
for comp, g in P.groupby('comp'):
    for a, b in (('status_rates', 'p3_robust_native'), ('status_shrunk', 'p3_robust_native'), ('status_MK', 'MK'), ('status_shrunk_MK', 'MK'), ('status_shrunk_MK', 'p3_robust_native')):
        x = g[g.engine.eq(a)].reset_index(drop=True); y = g[g.engine.eq(b)].reset_index(drop=True)
        assert (x.pool_id.values == y.pool_id.values).all()
        dmae = (x.predicted-x.actual).abs()-(y.predicted-y.actual).abs()
        dmse = (x.predicted-x.actual)**2-(y.predicted-y.actual)**2
        frame = pd.DataFrame({'slate': x.slate, 'fx': x.fixture_id, 'dmae': dmae, 'dmse': dmse})
        per_fx = frame.groupby(['slate', 'fx'])[['dmae', 'dmse']].mean().reset_index()
        point_mae = frame.groupby('slate').dmae.mean().mean(); point_mse = frame.groupby('slate').dmse.mean().mean()
        bs = []
        for _ in range(2000):
            s = per_fx.groupby('slate', group_keys=False).apply(lambda h: h.sample(len(h), replace=True, random_state=int(rng.integers(1e9))))
            bs.append(s.groupby('slate')[['dmae', 'dmse']].mean().mean().to_numpy())
        bs = np.array(bs)
        rounds_better = int((frame.groupby('slate').dmae.mean() < 0).sum())
        out.append(dict(comp=comp, a=a, b=b, d_mae=point_mae, mae_lo=np.percentile(bs[:, 0], 5), mae_hi=np.percentile(bs[:, 0], 95),
                        d_mse=point_mse, mse_lo=np.percentile(bs[:, 1], 5), mse_hi=np.percentile(bs[:, 1], 95),
                        rounds_mae_better=f"{rounds_better}/{frame.slate.nunique()}"))
BS = pd.DataFrame(out); print(BS.round(4).to_string(index=False)); BS.round(4).to_csv(E/'test_bootstrap.csv', index=False)
# positional bias on official labels (6N only)
six = P[P.comp.ne('NCR 2026')].copy(); six['err'] = six.predicted-six.actual
T = six.groupby(['status', 'position', 'engine']).err.mean().unstack('engine')[['p3_robust_native', 'status_rates', 'status_shrunk', 'MK', 'status_MK', 'status_shrunk_MK', 'empirical_baseline']]
T.insert(0, 'n', six[six.engine.eq('p3_robust_native')].groupby(['status', 'position']).size())
print('official-label bias 6N 2025-26'); print(T.round(2).to_string()); T.round(3).to_csv(E/'test_bias_official_labels.csv')
# API-observable bias on 6N official slates
off = pd.read_pickle(B+'/off.pkl'); off = off[off.competition.eq('six_nations')]
rows = []
for slate, g in off.groupby('slate'):
    st = load_raw(Path(B)/'runs'/'status'/'official'/slate/'status.jsonl')
    g = g.copy(); g['s_obs'] = observable_expected(st); g['k_obs'] = observable_expected(load_raw(Path(B)/'runs'/'status'/'official'/slate/'shrunk.jsonl'))
    rows.append(g)
O = pd.concat(rows)
O['ref_err'] = O.p_obs-O.a_obs; O['st_err'] = O.s_obs-O.a_obs; O['sk_err'] = O.k_obs-O.a_obs
A = O.groupby(['status', 'position']).agg(n=('ref_err', 'size'), ref=('ref_err', 'mean'), status=('st_err', 'mean'), shrunk=('sk_err', 'mean'),
                                         ref_mse=('ref_err', lambda e: (e**2).mean()), status_mse=('st_err', lambda e: (e**2).mean()), shrunk_mse=('sk_err', lambda e: (e**2).mean()))
print('API-observable bias 6N 2025-26'); print(A.round(2).to_string()); A.round(3).to_csv(E/'test_bias_api_observable.csv')
print(O.groupby('status').agg(ref=('ref_err', 'mean'), status=('st_err', 'mean'), ref_mse=('ref_err', lambda e: (e**2).mean()), st_mse=('st_err', lambda e: (e**2).mean()), ref_mae=('ref_err', lambda e: e.abs().mean()), st_mae=('st_err', lambda e: e.abs().mean()), sk=('sk_err', 'mean'), sk_mse=('sk_err', lambda e: (e**2).mean()), sk_mae=('sk_err', lambda e: e.abs().mean())).round(3).T)
# super-sub table 6N
S6 = R[R.competition.eq('six_nations')]
tab = S6.pivot_table(index='slate', columns='engine', values=['supersub', 'supersub_points', 'supersub_pred', 'smoothed_supersub'], aggfunc='first')
cols = []
for eng in ('p3_robust_native', 'status_rates', 'status_shrunk', 'MK', 'status_shrunk_MK'):
    cols += [('supersub', eng), ('supersub_pred', eng), ('supersub_points', eng), ('smoothed_supersub', eng)]
tab = tab[cols]
print(tab.round(1).to_string()); tab.round(2).to_csv(E/'test_supersub_picks.csv')
print(S6.groupby(['season', 'engine'])[['supersub_points', 'smoothed_supersub', 'team_points', 'smoothed_points']].sum().round(1))
