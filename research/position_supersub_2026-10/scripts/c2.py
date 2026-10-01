"""Super-sub ceiling simulation.

Per round, true bench means mu_i = m + b*(pred_i - mean) + u_i, u ~ N(0, s_u^2) with
s_u^2 = var_mu - b^2 var(pred) (persistent player signal the forecast misses). var_mu is the
same-player cross-product covariance of round-centred bench points (different matches, <=730 days).
Reports expected tripled super-sub points per 10 rounds for: random bench player, current argmax,
forecasts that recover a share phi of the missing signal, and the oracle (phi=1).
A player-bootstrap gives an interval for var_mu.
"""
from common import *
import itertools
import numpy as np, pandas as pd
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl')
six = off[off.competition.eq('six_nations')].copy()
rng = np.random.default_rng(7)


def prep(f, y, p):
    f = f[~f.started.astype(bool) & np.isfinite(f[y].astype(float))].copy()
    f['y'] = np.where(f.a_minutes > 0, f[y], 0.0); f['pred'] = f[p]
    f['yc'] = f.y - f.groupby('slate').y.transform('mean'); f['pc'] = f.pred - f.groupby('slate').pred.transform('mean')
    f['date'] = pd.to_datetime(f.date)
    return f


def var_mu(f, players=None):
    prods = []
    groups = dict(tuple(f.groupby('player_id')))
    keys = players if players is not None else list(groups)
    for pid in keys:
        g = groups[pid]
        if len(g) < 2:
            continue
        v = g.yc.to_numpy(); d = g.date.to_numpy()
        for i, j in itertools.combinations(range(len(g)), 2):
            if abs((d[i]-d[j]).astype('timedelta64[D]').astype(int)) <= 730:
                prods.append(v[i]*v[j])
    return float(np.mean(prods))


def simulate(f, vm, phis=(0.0, 0.25, 0.5, 1.0), draws=4000):
    vp = f.pc.var(); b = np.cov(f.pc, f.yc)[0, 1]/vp
    su2 = max(vm - b*b*vp, 0.0)
    out = {'b': b, 'var_mu': vm, 'captured_share': min(b*b*vp/vm, 1.0) if vm > 0 else np.nan, 'sd_u': np.sqrt(su2)}
    rounds = [g for _, g in f.groupby('slate')]
    out['random'] = 3*sum(g.y.mean() for g in rounds)
    out['realised_argmax'] = 3*sum(g.y.iloc[np.argmax(g.pred.to_numpy())] for g in rounds)
    out['hindsight'] = 3*sum(g.y.max() for g in rounds)
    for phi in phis:
        tot = 0.0
        for g in rounds:
            base = b*g.pc.to_numpy()
            n = len(base)
            u_obs = rng.normal(0, np.sqrt(phi*su2), (draws, n))
            score = base + u_obs
            pick = score.argmax(axis=1)
            tot += g.y.mean() + score[np.arange(draws), pick].mean()
        out[f'expected_phi{phi}'] = 3*tot
    return out

rows = []
for name, f in (('official 6N 2025-26', prep(six, 'actual_pts', 'p_points')), ('devcr 6N 2023-24', prep(dev, 'actual_pts', 'p_points'))):
    vm = var_mu(f)
    players = f.player_id.unique()
    boots = [var_mu(f, list(rng.choice(players, len(players)))) for _ in range(200)]
    r = simulate(f, vm)
    lo, hi = np.percentile(boots, [5, 95])
    r_lo, r_hi = simulate(f, lo), simulate(f, hi)
    r['var_mu_90'] = f'[{lo:.2f}, {hi:.2f}]'
    r['oracle_90'] = f"[{r_lo['expected_phi1.0']:.0f}, {r_hi['expected_phi1.0']:.0f}]"
    rows.append(dict(set=name, **r))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(R.round(2).T.to_string())
R.to_csv(B+'/supersub_ceiling.csv', index=False)
