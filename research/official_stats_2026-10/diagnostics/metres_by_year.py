import numpy as np, pandas as pd
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
store = pd.read_csv(S+'/runs/base/inputs/player_match.csv', low_memory=False, dtype={'fixture_id': str, 'player_id': str}, usecols=['fixture_id','player_id','date','competition_id_cache','competition','competition_level','calendar_year','minutes','metres','tackle_turnover','position','source','runs','available__metres'])
intl = store[store.competition_level.eq('international')]
g = intl.groupby(['calendar_year']).apply(lambda g: pd.Series({'n': len(g), 'm80': 80*g.metres.sum()/g.minutes.sum(), 'runs80': 80*g.runs.sum()/g.minutes.sum(), 'mpr': g.metres.sum()/g.runs.sum(), 'tt80': 80*g.tackle_turnover.sum()/g.minutes.sum()}))
print('all internationals'); print(g.round(3).to_string())
six = intl[intl.competition_id_cache.eq(1266)]
print('6N by year/position: API metres per 80')
print(six.groupby(['calendar_year','position']).apply(lambda g: 80*g.metres.sum()/g.minutes.sum()).unstack(0).round(1).to_string())
print('6N metres per run by year/position')
print(six.groupby(['calendar_year','position']).apply(lambda g: g.metres.sum()/g.runs.sum()).unstack(0).round(2).to_string())
# monthly in 2025-2026 internationals
intl = intl.assign(ym=pd.to_datetime(intl.date).dt.to_period('Q'))
print(intl[intl.calendar_year >= 2024].groupby('ym').apply(lambda g: pd.Series({'n': len(g), 'm80': 80*g.metres.sum()/g.minutes.sum(), 'mpr': g.metres.sum()/g.runs.sum()})).round(2).to_string())
club = store[store.competition_level.eq('club')]
club = club.assign(ym=pd.to_datetime(club.date).dt.to_period('Q'))
print('club'); print(club[club.calendar_year >= 2024].groupby('ym').apply(lambda g: pd.Series({'n': len(g), 'm80': 80*g.metres.sum()/max(g.minutes.sum(),1), 'mpr': g.metres.sum()/max(g.runs.sum(),1)})).round(2).to_string())
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched]
print('official MC per 80 by year/position'); print(J.groupby(['calendar_year','position']).apply(lambda g: 80*g.off_MC.sum()/g.minutes.sum()).unstack(0).round(1).to_string())
