"""Super-sub ceiling: how much of bench-point variance is predictable at all?

Var(mu) estimated from same-player cross-products of bench appearances (different matches), centred on
the round mean of all bench players. Compare with the model's explained variance (corr^2 * var) and
its covariance with outcomes.
"""
from common import *
import numpy as np, pandas as pd, itertools
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl'); blk = pd.read_pickle(B+'/blocks.pkl')
six = off[off.competition.eq('six_nations')].copy()
blk['pre'] = ~blk.block.str.contains('2025|2026')


def decomp(f, y, pred, unit, max_gap_days=730):
    f = f[~f.started.astype(bool) & np.isfinite(f[y].astype(float))].copy()
    f['y'] = f[y]; f['pred'] = f[pred]
    f['yc'] = f.y - f.groupby(unit).y.transform('mean')
    f['pc'] = f.pred - f.groupby(unit).pred.transform('mean')
    tot = f.yc.var()
    f['date'] = pd.to_datetime(f.date)
    prods = []
    for pid, g in f.groupby('player_id'):
        if len(g) < 2:
            continue
        v = g.yc.to_numpy(); d = g.date.to_numpy()
        for i, j in itertools.combinations(range(len(g)), 2):
            if abs((d[i]-d[j]).astype('timedelta64[D]').astype(int)) <= max_gap_days:
                prods.append(v[i]*v[j])
    var_mu = float(np.mean(prods)) if prods else np.nan
    cov = float(np.cov(f.pc, f.yc)[0, 1]); vp = f.pc.var()
    r = np.corrcoef(f.pc, f.yc)[0, 1]
    return dict(n=len(f), pairs=len(prods), sd_y=np.sqrt(tot), var_mu=var_mu, sd_mu=np.sqrt(max(var_mu, 0)), icc=var_mu/tot,
                model_r=r, model_r2=r*r, model_explained=cov*cov/vp, sd_pred=np.sqrt(vp), slope=cov/vp)

rows = []
rows.append(dict(set='official 6N 25-26 (official pts)', **decomp(six, 'actual_pts', 'p_points', 'slate')))
rows.append(dict(set='devcr 6N 23-24 (current-rubric)', **decomp(dev, 'actual_pts', 'p_points', 'slate')))
rows.append(dict(set='blocks pre-2025 (API obs pts)', **decomp(blk[blk.pre], 'a_obs', 'p_obs', 'block')))
rows.append(dict(set='blocks 2025-26 (API obs pts)', **decomp(blk[~blk.pre], 'a_obs', 'p_obs', 'block')))
rows.append(dict(set='official 6N 25-26 (API obs pts)', **decomp(six, 'a_obs', 'p_obs', 'slate')))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(R.round(3).to_string(index=False))
R.to_csv(B+'/ceiling_variance.csv', index=False)
