"""Official metres: mapping from API metres + carries forecasts, rolling pre-lock fits."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import lognorm, spearmanr
sys.path.insert(0, '.')
from model.unified.contracts import RawPrediction
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J['match_at'] = pd.to_datetime(J.match_at, utc=True)
OFF = J[J.off_matched & J.minutes.gt(0)].copy()

def floor_pts(mean, disp):
    if mean <= 0: return 0.0
    s2 = np.log1p(disp/mean**2); s, sc = np.sqrt(s2), np.exp(np.log(mean)-s2/2)
    end = int(np.ceil(lognorm.isf(1e-10, s, scale=sc)/10))
    return float(lognorm.sf(10*np.arange(1, end+1), s, scale=sc).sum())

def load_raw(slate):
    for base in ('base', 'devcr'):
        try: return [RawPrediction.from_dict(json.loads(l)) for l in open(f'{S}/runs/{base}/models/{slate}/p3_robust_native.jsonl')]
        except FileNotFoundError: pass

def fit(h, form):
    y = h.off_MC.to_numpy(float)
    if form == 'scale': X = h[['metres']].fillna(0).to_numpy(float)
    else: X = h[['metres', 'runs']].fillna(0).to_numpy(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]

rows = []
for (yr, rnd), cand in J[J.calendar_year.isin([2023, 2025, 2026])].groupby(['calendar_year', 'round']):
    raw = load_raw(f'six_nations_{yr}_r{int(rnd)}')
    if raw is None: continue
    cutoff = cand.match_at.min()
    fc = pd.DataFrame({'fixture_id': [p.fixture_id for p in raw], 'player_id': [p.player_id for p in raw],
                       'm_mean': [p.events['metres'].mean for p in raw], 'm_disp': [p.events['metres'].dispersion for p in raw],
                       'r_mean': [p.events['runs'].mean for p in raw]})
    c = cand[cand.off_matched].merge(fc, on=['fixture_id', 'player_id'], how='left')
    hist = OFF[OFF.match_at < cutoff]
    cur = hist[hist.calendar_year.eq(yr)]
    latest = cur if len(cur) else hist[hist.calendar_year.eq(hist.calendar_year.max())] if len(hist) else hist
    c['base'] = [floor_pts(m, d) for m, d in zip(c.m_mean, c.m_disp)]
    for tag, h in (('latest', latest), ('pooled', hist)):
        if len(h) == 0:
            c[f'{tag}_scale'] = c.base; c[f'{tag}_carry'] = c.base; continue
        k = fit(h, 'scale')[0]
        a, b = fit(h, 'carry')
        for form, mu in (('scale', k*c.m_mean), ('carry', a*c.m_mean + b*c.r_mean)):
            mu = np.clip(mu, 0.01, None); f = mu/c.m_mean
            c[f'{tag}_{form}'] = [floor_pts(m, d*ff**2) for m, d, ff in zip(mu, c.m_disp, f)]
        c[f'coef_{tag}'] = f'k={k:.2f} a={a:.2f} b={b:.2f}'
    c['actual'] = np.floor(c.off_MC.fillna(0)/10)
    rows.append(c)
D = pd.concat(rows, ignore_index=True)
out = []
for (yr), g in D.groupby('calendar_year'):
    for col in ('base', 'latest_scale', 'latest_carry', 'pooled_scale', 'pooled_carry'):
        e = g.actual - g[col]
        out.append(dict(yr=yr, pred=col, mean_pred=g[col].mean(), mean_act=g.actual.mean(), mae=e.abs().mean(), mse=(e**2).mean(),
                        corr=np.corrcoef(g.actual, g[col])[0, 1], within_round_corr=g.groupby('round').apply(lambda x: np.corrcoef(x.actual, x[col])[0, 1]).mean()))
print(pd.DataFrame(out).round(3).to_string(index=False))
print(D.groupby(['calendar_year', 'round'])[['coef_latest', 'coef_pooled']].first().to_string())
print(D.groupby(['calendar_year', 'position'])[['base', 'latest_carry', 'actual']].mean().round(2).unstack(0).to_string())
D.to_pickle(S+'/agentA/metres_study.pkl')
