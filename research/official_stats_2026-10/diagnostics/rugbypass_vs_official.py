import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
from compare_api_official import norm_key
from scipy.stats import spearmanr
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
off = pd.read_csv('data/official_player_match.csv')
for c in off.columns[5:]: off[c] = pd.to_numeric(off[c], errors='coerce').fillna(0)
os_ = off.groupby(['season', 'key']).agg(BS=('BS', 'sum'), MC=('MC', 'sum'), Min=('Min', 'sum'), n=('Min', 'size')).reset_index()
rp = pd.read_csv('data/rp_compstats.csv')
rp6 = rp[rp.competition.eq('Six Nations')].copy(); rp6['season'] = rp6.season.astype(int)
x = os_.merge(rp6, on=['season', 'key'], how='inner')
print('joined', len(x), x.groupby('season').size().to_dict())
for yr, g in x.groupby('season'):
    print(yr, 'corr BS vs RP turnovers_won', round(np.corrcoef(g.BS, g.turnovers_won)[0, 1], 3), 'sum BS', g.BS.sum(), 'sum tw', g.turnovers_won.sum(),
          'min corr', round(np.corrcoef(g.Min, g.minutes)[0, 1], 3), 'MC vs RP metres', round(np.corrcoef(g.MC, g.metres)[0, 1], 3), 'MC/RPm', round(g.MC.sum()/g.metres.sum(), 3),
          'MC vs RP carries', round(np.corrcoef(g.MC, g.carries)[0, 1], 3))
# API season tackle_turnover too
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched]
J['key'] = J.player_name.map(norm_key)
api = J.groupby(['calendar_year', 'key']).agg(tt=('tackle_turnover', 'sum'), BS=('off_BS', 'sum'), m=('metres', 'sum'), runs=('runs', 'sum'), MC=('off_MC', 'sum')).reset_index()
for yr, g in api.groupby('calendar_year'):
    print(yr, 'season-level corr BS vs API tt', round(np.corrcoef(g.BS, g.tt)[0, 1], 3), ' MC vs API metres', round(np.corrcoef(g.MC, g.m)[0, 1], 3))
# player BS reliability across seasons (per 80, players >= 160 min both)
r = os_.assign(r=80*os_.BS/os_.Min.clip(lower=1))
w = r[r.Min >= 160].pivot(index='key', columns='season', values='r')
for a, b in [(2023, 2025), (2025, 2026), (2023, 2026)]:
    z = w[[a, b]].dropna(); print('BS/80 reliability', a, b, len(z), round(np.corrcoef(z[a], z[b])[0, 1], 3), 'spearman', round(spearmanr(z[a], z[b])[0], 3))
# RP turnovers_won per 80 in prior club seasons vs official BS per 80
