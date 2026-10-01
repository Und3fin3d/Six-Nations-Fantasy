"""Super-sub anatomy: bench forecasts vs realised, near-ties, position of hindsight best."""
from common import *
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
off = pd.read_pickle(B+'/off.pkl'); dev = pd.read_pickle(B+'/devcr.pkl')
six = off[off.competition.eq('six_nations')]
for name, f in (('official', six), ('devcr', dev)):
    print('=====', name)
    tot = dict(pick=0, best=0, base_pick=0)
    rows = []
    for slate, g in f.groupby('slate', sort=False):
        b = g[g.status.eq('B')].copy()
        b['played'] = b.a_minutes > 0
        b['real'] = np.where(b.played, b.actual_pts, 0)
        b = b.sort_values('p_points', ascending=False)
        top = b.iloc[0]; best = b.loc[b.real.idxmax()]
        bb = b.sort_values('baseline', ascending=False).iloc[0]
        rank_best = int((b.p_points > best.p_points).sum()) + 1
        rows.append(dict(slate=slate, pick=top.player_name, pick_pos=top.position, pick_pred=top.p_points, pick_real=top.real,
                         gap12=b.p_points.iloc[0]-b.p_points.iloc[1], n_within1=int((b.p_points > top.p_points-1).sum()),
                         best=best.player_name, best_pos=best.position, best_pred=best.p_points, best_real=best.real, best_rank=rank_best,
                         base_pick=bb.player_name, base_real=bb.real, mean_bench_real=b.real.mean(), top5_mean=b.real.iloc[:5].mean()))
    R = pd.DataFrame(rows)
    print(R.round(1).to_string(index=False))
    print('sum pick real x3', 3*R.pick_real.sum(), ' best x3', 3*R.best_real.sum(), ' baseline pick x3', 3*R.base_real.sum(),
          ' random bench x3', 3*R.mean_bench_real.sum(), ' top5 avg x3', 3*R.top5_mean.sum())
    # correlation of forecast and outcome among bench players, per slate
    cs = f[f.status.eq('B')].groupby('slate').apply(lambda g: pd.Series(dict(r=np.corrcoef(g.p_points, np.where(g.a_minutes > 0, g.actual_pts, 0))[0, 1], rb=np.corrcoef(g.baseline, np.where(g.a_minutes > 0, g.actual_pts, 0))[0, 1])))
    print('bench within-round corr: model', cs.r.mean().round(3), ' baseline', cs.rb.mean().round(3))
    bb = f[f.status.eq('B')]
    print('bench SD pred', bb.groupby('slate').p_points.std().mean().round(2), ' SD actual', bb.groupby('slate').actual_pts.std().mean().round(2))
