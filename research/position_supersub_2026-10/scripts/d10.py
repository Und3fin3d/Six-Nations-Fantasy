"""Front-row try over-forecast: is it club-driven? Split starters by share of prior club minutes."""
from common import *
import json, glob
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
f = pd.read_pickle(B+'/blocks_components.pkl')
st = pd.read_pickle(B+'/store.pkl'); st['match_at'] = pd.to_datetime(st.match_at, utc=True)
cut = {}
for d in sorted(glob.glob(S+'/raw/comparison-raw-*')):
    cut[d.split('raw-')[-1]] = pd.Timestamp(json.load(open(d+'/results/run_manifest.json'))['cutoff'])
rows = []
for block, g in f.groupby('block'):
    h = st[st.match_at < cut[block]]
    mins = h.groupby(['player_id', 'competition_level']).minutes.sum().unstack(fill_value=0)
    g = g.copy()
    g['intl_min'] = g.player_id.map(mins.get('international', pd.Series(dtype=float))).fillna(0)
    g['club_min'] = g.player_id.map(mins.get('club', pd.Series(dtype=float))).fillna(0)
    rows.append(g)
f = pd.concat(rows)
f['club_share'] = f.club_min/(f.club_min+f.intl_min).replace(0, np.nan)
f['grp'] = pd.cut(f.club_share, [-0.01, 0.6, 0.85, 1.0], labels=['intl-heavy', 'mixed', 'club-heavy'])
s = f[f.started.astype(bool)]
for pos in ['Hooker', 'Prop', 'Back-row', 'Scrum-half', 'Back-three']:
    g = s[s.position.eq(pos)]
    t = g.groupby(['pre', 'grp'], observed=True).apply(lambda h: pd.Series(dict(n=len(h), act=h.tries.mean(), remp=h.remp_tries.mean(), v4=h.v4_tries.mean(), rob=h.rob_tries.mean())))
    print('==', pos); print(t.round(3).to_string())
