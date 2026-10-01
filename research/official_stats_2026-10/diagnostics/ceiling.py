"""Ceilings: how much of each official component's variance is learnable at all?"""
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
rng = np.random.default_rng(0)
# --- BS: Poisson floor. Var(y) = E[lambda] + Var(lambda)  -> max R2 = Var(lambda)/Var(y)
B = pd.read_pickle(S+'/agentA/bs_study2.pkl')
for yr in (2023, 2025, 2026):
    g = B[B.calendar_year.eq(yr)]
    y = g.off_BS.fillna(0)
    # condition on minutes played (exposure known ex post): Var(lambda) estimated from overdispersion
    print(yr, 'BS rows', len(g), 'mean', round(y.mean(), 3), 'var', round(y.var(), 3), 'max R2 (Poisson, incl. minutes)', round(1-y.mean()/y.var(), 3))
# player-level split-half reliability of BS per 80 within 2025-26 (odd/even rounds)
g = B[B.calendar_year.isin([2025, 2026])]
odd = g[g['round'] % 2 == 1].groupby('key').agg(b=('off_BS', 'sum'), m=('minutes', 'sum'))
even = g[g['round'] % 2 == 0].groupby('key').agg(b=('off_BS', 'sum'), m=('minutes', 'sum'))
x = odd.join(even, lsuffix='_o', rsuffix='_e').dropna(); x = x[(x.m_o >= 160) & (x.m_e >= 160)]
r = np.corrcoef(80*x.b_o/x.m_o, 80*x.b_e/x.m_e)[0, 1]
print('BS/80 split-half (odd vs even rounds 2025-26) players', len(x), 'r', round(r, 3), 'Spearman-Brown', round(2*r/(1+r), 3))
# simulate: if forecasts = truth (Poisson), what corr between forecast and outcome? use the adapter-like predictor eb pos*rprel10
from math import sqrt
for yr in (2025, 2026):
    g = B[B.calendar_year.eq(yr)]
    lam = (g.pos_rate*g.rprel10*g.min_model/80).fillna(0).to_numpy()
    sims = [np.corrcoef(lam, rng.poisson(lam))[0, 1] for _ in range(500)]
    print(yr, 'if pos*RP forecast were the true mean: expected corr', round(np.mean(sims), 3), '(achieved', round(np.corrcoef(lam, g.off_BS.fillna(0))[0, 1], 3), ')')
# --- POTM ceilings: Bernoulli with p summing to 1 per match
P = pd.read_pickle(S+'/agentA/potm_study.pkl')
P = P[P.calendar_year.isin([2025, 2026])]
def probs(beta, use_pwin=True, bench=0.05, col='xp'):
    p = pd.Series(0.0, index=P.index)
    for (f, t), g in P.groupby(['fixture_id', 'team']):
        w = np.exp(beta*(g[col]-g[col].max()))*np.where(g.started, 1, bench)
        p.loc[g.index] = (g.pwin.iloc[0] if use_pwin else .5)*w/w.sum()
    return p/p.groupby(P.fixture_id).transform('sum')
p = probs(0.1)
sims = []
for _ in range(500):
    y = np.zeros(len(P))
    for f, idx in P.groupby('fixture_id').indices.items():
        y[idx[rng.choice(len(idx), p=p.iloc[idx].to_numpy())]] = 1
    sims.append(np.corrcoef(p, y)[0, 1])
print('POTM: if the frozen p were the truth, expected corr(p, y) =', round(np.mean(sims), 3), '+-', round(np.std(sims), 3), ' achieved', round(np.corrcoef(p, P.y)[0, 1], 3))
print('POTM variance in points: Var(15y) =', round(225*P.y.var(), 3), ' explainable by p (225 Var p) =', round(225*p.var(), 3))
# --- Metres: lognormal model-implied ceiling vs achieved
M = pd.read_pickle(S+'/agentA/metres_study.pkl')
for yr in (2025, 2026):
    g = M[M.calendar_year.eq(yr)]
    mu = g.m_mean.to_numpy(); disp = g.m_disp.to_numpy()
    s2 = np.log1p(disp/mu**2); sims = []
    for _ in range(200):
        x = rng.lognormal(np.log(mu)-s2/2, np.sqrt(s2))
        sims.append(np.corrcoef(g.base, np.floor(x/10))[0, 1])
    print(yr, 'metres: if the API forecast distribution were true, corr(points forecast, outcome) =', round(np.mean(sims), 3), ' achieved', round(np.corrcoef(g.base, g.actual)[0, 1], 3))
# split-half reliability of official metres per 80 within season
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched & J.minutes.gt(20)]
for yr in (2025, 2026):
    g = J[J.calendar_year.eq(yr)]
    o = g[g['round'] % 2 == 1].groupby('player_id').agg(v=('off_MC', 'sum'), m=('minutes', 'sum'))
    e = g[g['round'] % 2 == 0].groupby('player_id').agg(v=('off_MC', 'sum'), m=('minutes', 'sum'))
    x = o.join(e, lsuffix='_o', rsuffix='_e').dropna(); x = x[(x.m_o >= 120) & (x.m_e >= 120)]
    print(yr, 'official metres/80 split-half r', round(np.corrcoef(x.v_o/x.m_o, x.v_e/x.m_e)[0, 1], 3), 'players', len(x))
