import numpy as np, pandas as pd

class _R:
    def __init__(s,y,X):
        y=np.asarray(y,float); A=np.asarray(X,float); b=np.linalg.lstsq(A,y,rcond=None)[0]
        s.params=pd.Series(b,index=list(X.columns)); r=y-A@b; s.rsquared=1-(r**2).sum()/((y-y.mean())**2).sum()
class sm:
    @staticmethod
    def OLS(y,X):
        class F:
            def fit(self_): return _R(y,X)
        return F()
pd.set_option('display.width', 250)
import os; S=os.environ['OFFICIAL_STATS_SCRATCH']  # holds runs/ (rolling_eval outputs), anatomy.pkl and agentA/ intermediates
J = pd.read_pickle(S+'/agentA/join6n.pkl'); J = J[J.off_matched].copy()
o = {c: J['off_'+c].fillna(0) for c in ['Min','Ta','MC','50-22','LS','BS','T','As','C','Pen','DG','YC','RC','Pts','POTM','DB','OF','SW','CPen','KR']}
fw = J.is_forward.astype(bool)
rec = np.where(fw, 15, 10)*o['T'] + 4*o['As'] + 2*o['C'] + 3*o['Pen'] + 4*o['DG'] + 2*o['DB'] + 2*o['OF'] + o['Ta'] + 5*o['BS'] - o['CPen'] - 5*o['YC'] - 8*o['RC'] + 7*o['50-22'] + 7*o['LS'] + o['SW'] + 2*o['KR'] + 15*o['POTM'] + np.floor(o['MC']/10)
J['rec'] = rec
for yr, g in J.groupby('calendar_year'):
    print(yr, 'exact', round((g.rec == g.off_Pts).mean(), 3), 'corr', round(np.corrcoef(g.rec, g.off_Pts)[0, 1], 4), 'mean gap', round((g.off_Pts-g.rec).mean(), 3))
# regression off_MC ~ metres + runs (no intercept) per year
for yr, g in J.groupby('calendar_year'):
    X = g[['metres', 'runs']].fillna(0)
    m = sm.OLS(g.off_MC.fillna(0), X).fit()
    m1 = sm.OLS(g.off_MC.fillna(0), X[['metres']]).fit()
    print(yr, 'MC ~ metres+runs', m.params.round(3).to_dict(), 'R2', round(m.rsquared, 3), '| MC~metres', m1.params.round(3).to_dict(), 'R2', round(m1.rsquared, 3))
    for pos, h in g.groupby('position'):
        if len(h) < 30: continue
        mm = sm.OLS(h.off_MC.fillna(0), h[['metres', 'runs']].fillna(0)).fit()
        print('   ', pos, mm.params.round(2).to_dict())
