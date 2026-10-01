import pandas as pd, numpy as np
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
P = pd.read_pickle(S+'/anatomy.pkl')
for k in ['scrums', 'lineout_steal', 'kicks_retained', 'fifty22']:
    t = P.groupby(['pos', 'started']).apply(lambda g: pd.Series({'pred': g['p_'+k].mean(), 'act': g['a_'+k].mean(),
        'corr': np.corrcoef(g['p_'+k], g['a_'+k])[0, 1] if g['p_'+k].std() > 0 and g['a_'+k].std() > 0 else np.nan}))
    t = t[(t.pred.abs() > 0.02) | (t.act.abs() > 0.02)]
    print(k, 'overall corr', round(np.corrcoef(P['p_'+k], P['a_'+k])[0, 1], 3), 'by season', P.groupby('season').apply(lambda g: pd.Series({'pred': g['p_'+k].mean(), 'act': g['a_'+k].mean()})).round(3).to_dict('index'))
    print(t.round(3).to_string())
