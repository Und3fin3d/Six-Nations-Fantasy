"""BS forecast study: which pre-lock predictor best forecasts official breakdown steals?"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from compare_api_official import norm_key
from model.unified.contracts import RawPrediction
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
J = pd.read_pickle(S+'/agentA/join6n.pkl')
J['key'] = J.player_name.map(norm_key)
J['match_at'] = pd.to_datetime(J.match_at, utc=True)
OFF = J[J.off_matched].copy()
rp = pd.read_csv('data/rp_compstats.csv').drop_duplicates(['key', 'competition', 'season'])
def season_end(s):
    s = str(s)
    return pd.Timestamp(f'{int(s.split("/")[1])}-07-31', tz='UTC') if '/' in s else pd.Timestamp(f'{int(s)}-12-31', tz='UTC')
rp['end'] = rp.season.map(season_end)

def load_raw(slate):
    for base in ('base', 'devcr'):
        p = f'{S}/runs/{base}/models/{slate}/p3_robust_native.jsonl'
        try:
            return [RawPrediction.from_dict(json.loads(l)) for l in open(p)]
        except FileNotFoundError:
            pass

rows = []
for (yr, rnd), cand in J[J.calendar_year.isin([2023, 2025, 2026])].groupby(['calendar_year', 'round']):
    cutoff = cand.match_at.min()
    raw = load_raw(f'six_nations_{yr}_r{int(rnd)}')
    if raw is None: continue
    fc = pd.DataFrame({'fixture_id': [p.fixture_id for p in raw], 'player_id': [p.player_id for p in raw],
                       'tt_model': [p.events['tackle_turnover'].mean for p in raw], 'min_model': [min(p.minutes.mean, 80) for p in raw]})
    c = cand.merge(fc, on=['fixture_id', 'player_id'], how='left')
    hist = OFF[OFF.match_at < cutoff]
    # position prior per 80 from pre-lock official rows (fallback to global)
    g = hist.groupby('position').agg(bs=('off_BS', 'sum'), m=('minutes', 'sum'))
    glob = hist.off_BS.sum()/hist.minutes.sum()*80 if len(hist) else np.nan
    prior = (80*(g.bs+2*glob/80*400)/(g.m+400)).to_dict() if len(hist) else {}
    c['pos_rate'] = c.position.map(prior).fillna(glob)
    # EB player official history: gamma-Poisson with pseudo-exposure k80 matches
    ph = hist.groupby('key').agg(bs=('off_BS', 'sum'), m=('minutes', 'sum'))
    for k80 in (3, 6, 12):
        b = c.key.map(ph.bs).fillna(0); m = c.key.map(ph.m).fillna(0)/80
        c[f'eb{k80}'] = (b + k80*c.pos_rate)/(m + k80)
    # RP turnovers_won rate from completed seasons (all comps)
    r = rp[rp.end < cutoff].groupby('key').agg(tw=('turnovers_won', 'sum'), m=('minutes', 'sum'))
    c['rp_m'] = c.key.map(r.m).fillna(0); c['rp_tw'] = c.key.map(r.tw).fillna(0)
    rows.append(c.assign(cutoff=cutoff))
D = pd.concat(rows, ignore_index=True)
D = D[D.off_matched].copy()
# RP position mean rates per 80 (pre-lock pooled over all completed seasons available) — compute per round
out = []
for (yr, rnd), c in D.groupby(['calendar_year', 'round']):
    c = c.copy()
    r = c[c.rp_m > 0]
    out.append(c)
D = pd.concat(out)
print(D.groupby('calendar_year').apply(lambda g: pd.Series({'rows': len(g), 'rp_cov': (g.rp_m > 0).mean(), 'rp_min_med': g.rp_m.median()})))
D.to_pickle(S+'/agentA/bs_study.pkl')
