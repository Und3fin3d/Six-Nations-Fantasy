"""Build per-player diagnostic tables: predicted vs actual events/minutes/points."""
from common import *
import json, glob, pickle
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import lognorm
from model.unified.contracts import RawPrediction
from model.unified.rolling_eval import expected_points
from research.reweight_eval import cached_slates

KEY = ['fixture_id', 'player_id', 'team']
EV = ['tries', 'try_assists', 'conversion_goals', 'penalty_goals', 'drop_goals_converted', 'defenders_beaten',
      'offload', 'tackles', 'tackle_turnover', 'penalties_conceded', 'yellow_cards', 'red_cards', 'metres',
      'runs', 'clean_breaks', 'missed_tackles', 'scrums_won', 'lineout_steals', 'fifty_22', 'kicks_retained', 'potm']


def exp_metres_pts(d):
    if d is None or d.mean <= 0:
        return 0.0
    s2 = np.log1p(d.dispersion/d.mean**2); s = np.sqrt(s2); sc = np.exp(np.log(d.mean)-s2/2)
    end = int(np.ceil(lognorm.isf(1e-10, s, scale=sc)/10))
    return float(lognorm.sf(10*np.arange(1, end+1), s, scale=sc).sum())


def comp_pred(p):
    g = lambda e: p.events[e].mean if e in p.events else 0.0
    fw = p.is_forward
    return dict(c_tries=(15 if fw else 10)*g('tries'), c_assists=4*g('try_assists'),
                c_kicking=2*g('conversion_goals')+3*g('penalty_goals')+4*g('drop_goals_converted'),
                c_db=2*g('defenders_beaten'), c_offload=2*g('offload'), c_tackles=g('tackles'),
                c_turnover=5*g('tackle_turnover'), c_pens=-g('penalties_conceded'),
                c_cards=-5*g('yellow_cards')-8*g('red_cards'), c_metres=exp_metres_pts(p.events.get('metres')))


def comp_actual(df):
    fw = df.is_forward.astype(bool)
    n = lambda c: pd.to_numeric(df[c], errors='coerce').fillna(0)
    return pd.DataFrame(dict(c_tries=np.where(fw, 15, 10)*n('tries'), c_assists=4*n('try_assists'),
        c_kicking=2*n('conversion_goals')+3*n('penalty_goals')+4*n('drop_goals_converted'),
        c_db=2*n('defenders_beaten'), c_offload=2*n('offload'), c_tackles=n('tackles'),
        c_turnover=5*n('tackle_turnover'), c_pens=-n('penalties_conceded'),
        c_cards=-5*n('yellow_cards')-8*n('red_cards'), c_metres=np.floor(n('metres')/10)), index=df.index)

COMPS = ['c_tries', 'c_assists', 'c_kicking', 'c_db', 'c_offload', 'c_tackles', 'c_turnover', 'c_pens', 'c_cards', 'c_metres']


def load_raw(path):
    return [RawPrediction.from_dict(json.loads(l)) for l in open(path)]


def pred_frame(raw, prefix='p_'):
    rows = []
    for p in raw:
        r = {'fixture_id': p.fixture_id, 'player_id': p.player_id, 'team': p.team, 'is_forward': p.is_forward,
             'position': p.position, 'player_name': p.player_name, prefix+'minutes': p.minutes.mean}
        for e in EV:
            r[prefix+e] = p.events[e].mean if e in p.events else np.nan
        for k, v in comp_pred(p).items():
            r[prefix+k] = v
        rows.append(r)
    f = pd.DataFrame(rows)
    f[prefix+'obs'] = f[[prefix+c for c in COMPS]].sum(axis=1)
    return f


def attach_actual(f, store):
    cols = KEY+['jersey', 'started', 'minutes', 'competition_level', 'calendar_year', 'margin', 'team_score', 'opp_score', 'opponent', 'date']+[e for e in EV if e in store]
    a = store[cols].rename(columns={e: 'a_'+e for e in EV if e in store}).rename(columns={'minutes': 'a_minutes'})
    f = f.merge(a, on=KEY, how='left', validate='one_to_one')
    ac = comp_actual(f.rename(columns={'a_'+e: e for e in EV}))
    for c in COMPS:
        f['a_'+c] = ac[c]
    f['a_obs'] = f[['a_'+c for c in COMPS]].sum(axis=1)
    return f


def main():
    store = pd.read_pickle(B+'/store.pkl')
    out = []
    # official slates
    for sl in cached_slates(Path(S)/'runs'/'base'):
        raw = load_raw(f'{S}/runs/base/models/{sl.name}/p3_robust_native.jsonl')
        f = pred_frame(raw)
        f['p_points'] = expected_points(raw, sl.competition)
        for comp in ('empirical', 'v4'):
            pth = Path(f'{S}/runs/base/components/{sl.name}/{comp}.jsonl')
            cr = load_raw(pth if pth.exists() else Path(B)/'components'/sl.name/f'{comp}.jsonl')
            f[comp[:3]+'_minutes'] = [p.minutes.mean for p in cr]
        f['actual_pts'] = sl.actual; f['baseline'] = sl.baseline
        f['status'] = sl.pool.status.to_numpy(); f['set'] = 'official'; f['slate'] = sl.name; f['competition'] = sl.competition
        f['season'] = sl.season; f['round'] = sl.round; f['pool_id'] = sl.pool.id.to_numpy()
        out.append(attach_actual(f, store))
    off = pd.concat(out, ignore_index=True); off.to_pickle(B+'/off.pkl')
    out = []
    for sl in pickle.loads((Path(S)/'runs'/'devcr'/'slates.pkl').read_bytes()):
        raw = load_raw(f'{S}/runs/devcr/models/{sl.name}/p3_robust_native.jsonl')
        f = pred_frame(raw)
        f['p_points'] = expected_points(raw, sl.competition)
        for comp in ('empirical', 'v4'):
            cr = load_raw(f'{S}/runs/devcr/models/{sl.name}/{comp}.jsonl')
            f[comp[:3]+'_minutes'] = [p.minutes.mean for p in cr]
        f['actual_pts'] = sl.actual; f['baseline'] = sl.baseline; f['label_source'] = sl.label_source
        f['status'] = sl.pool.status.to_numpy(); f['set'] = 'devcr'; f['slate'] = sl.name; f['competition'] = 'six_nations'
        f['season'] = sl.season; f['round'] = sl.round; f['pool_id'] = sl.pool.id.to_numpy()
        out.append(attach_actual(f, store))
    dev = pd.concat(out, ignore_index=True); dev.to_pickle(B+'/devcr.pkl')
    out = []
    for d in sorted(glob.glob(S+'/raw/comparison-raw-*')):
        block = d.split('raw-')[-1]
        raw = load_raw(d+'/results/p3_robust_native.jsonl')
        f = pred_frame(raw)
        f['p_points'] = expected_points(raw, 'six_nations')
        emp = load_raw(d+'/results/empirical_event.jsonl')
        f['emp_legacy_minutes'] = [p.minutes.mean for p in emp]
        f['block'] = block; f['set'] = 'block'
        out.append(attach_actual(f, store))
    blk = pd.concat(out, ignore_index=True); blk.to_pickle(B+'/blocks.pkl')
    for name, f in (('off', off), ('dev', dev), ('blk', blk)):
        print(name, f.shape, 'missing actual', f.a_minutes.isna().sum())


if __name__ == '__main__':
    main()
