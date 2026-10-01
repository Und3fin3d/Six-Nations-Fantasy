"""Join store 6N rows (API stats) to official per-stat rows; save to agentA/join6n.pkl."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from official_labels import match_official
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
store = pd.read_csv(S+'/runs/base/inputs/player_match.csv', low_memory=False, dtype={'fixture_id': str, 'player_id': str, 'team': str}, parse_dates=['date', 'match_at'])
print(store.shape)
print([c for c in store.columns[91:]])
six = store[store.competition_id_cache.eq(1266)].copy()
print(six.groupby('calendar_year').size())
off = pd.read_csv('data/official_player_match.csv')
lab = match_official(six, off)
for c in ['Min','Ta','MC','50-22','LS','BS','T','As','C','Pen','DG','YC','RC','Pts','POTM','DB','OF','SW','CPen','KR']:
    six['off_'+c] = pd.to_numeric(lab[c], errors='coerce')
six['off_matched'] = lab['season'].notna()
print(six.groupby('calendar_year').off_matched.agg(['sum','size']))
six.to_pickle(S+'/agentA/join6n.pkl')
