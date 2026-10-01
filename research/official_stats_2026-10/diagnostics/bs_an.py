import numpy as np, pandas as pd, sys
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
D = pd.read_pickle(S+'/agentA/bs_study.pkl')
rp = pd.read_csv('data/rp_compstats.csv').drop_duplicates(['key', 'competition', 'season'])
def season_end(s):
    s = str(s)
    return pd.Timestamp(f'{int(s.split("/")[1])}-07-31', tz='UTC') if '/' in s else pd.Timestamp(f'{int(s)}-12-31', tz='UTC')
rp['end'] = rp.season.map(season_end)
# Dev (2023): RP features from seasons ending before the 2025 lock
dev_cut = pd.Timestamp('2025-01-31', tz='UTC')
r = rp[rp.end < dev_cut].groupby('key').agg(tw=('turnovers_won', 'sum'), m=('minutes', 'sum'))
d23 = D.calendar_year.eq(2023)
D.loc[d23, 'rp_m'] = D.loc[d23, 'key'].map(r.m).fillna(0).to_numpy(); D.loc[d23, 'rp_tw'] = D.loc[d23, 'key'].map(r.tw).fillna(0).to_numpy()
print('2023 RP coverage (pseudo-dev)', (D[d23].rp_m > 0).mean())
y = D.off_BS.fillna(0)
def dev(mu, y):
    mu = np.clip(mu, 1e-6, None); return 2*np.mean(np.where(y > 0, y*np.log(y/mu), 0) - (y-mu))
def report(name, mu, mask):
    yy, mm = y[mask], mu[mask]
    return dict(pred=name, dev=round(dev(mm, yy), 4), mse=round(np.mean((yy-mm)**2), 4), corr=round(np.corrcoef(yy, mm)[0, 1], 3), mean_pred=round(mm.mean(), 3), mean_act=round(yy.mean(), 3))
# RP-based rate: shrink player RP tw/80 toward position RP mean (computed on all D rows' players with RP data, per year-pool pre-lock -> approximated via rp table itself)
# position for RP: use the candidate's position
expo = D.min_model/80; expo_a = D.minutes/80
res = []
for k in (2, 5, 10, 20):
    pos_rp = D[D.rp_m > 0].groupby(['calendar_year', 'position']).apply(lambda g: g.rp_tw.sum()/(g.rp_m.sum()/80))
    mu_rp = pd.Series([pos_rp.get((a, b), np.nan) for a, b in zip(D.calendar_year, D.position)], index=D.index)
    D[f'rprate{k}'] = (D.rp_tw + k*mu_rp)/(D.rp_m/80 + k)
    D[f'rprel{k}'] = D[f'rprate{k}']/mu_rp  # relative to position
for yr in (2023, 2025, 2026):
    m = D.calendar_year.eq(yr) & D.position.ne('Unknown')
    out = [report('model_tt', D.tt_model, m), report('pos', D.pos_rate*expo, m)]
    for k in (3, 6, 12): out.append(report(f'eb{k}', D[f'eb{k}']*expo, m))
    for k in (2, 5, 10, 20): out.append(report(f'pos*rprel{k}', D.pos_rate*D[f'rprel{k}']*expo, m))
    for k in (5, 10): out.append(report(f'eb6*rprel{k}', D['eb6']*D[f'rprel{k}']*expo, m))
    out.append(report('ACTMIN pos', D.pos_rate*expo_a, m)); out.append(report('ACTMIN pos*rprel5', D.pos_rate*D.rprel5*expo_a, m))
    print(yr); print(pd.DataFrame(out).to_string(index=False))
# per-80 split: corr of player RP rate with official BS per 80 across players (season level)
for yr in (2023, 2025, 2026):
    g = D[D.calendar_year.eq(yr) & (D.rp_m > 400)].groupby('key').agg(bs=('off_BS', 'sum'), m=('minutes', 'sum'), rp=('rprate5', 'first'), pos=('position', 'first'))
    g = g[g.m >= 160]
    print(yr, 'players', len(g), 'corr official BS/80 vs RP shrunk rate', round(np.corrcoef(80*g.bs/g.m, g.rp)[0, 1], 3))
D.to_pickle(S+'/agentA/bs_study2.pkl')
