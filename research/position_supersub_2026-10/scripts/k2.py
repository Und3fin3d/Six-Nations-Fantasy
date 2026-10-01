"""Report the fitted within-position shrinkage K and status factors at the 6N 2024 R1 lock (development)."""
from common import *
import json, pickle, numpy as np, pandas as pd
from pathlib import Path
from model.history import past_matches
from model.unified.raw_benchmark.config import STABLE_EVENTS, EXTENDED_EVENTS
from model.unified.raw_benchmark.status_rates import estimate_status_factors, starter_equivalent, within_position_shrinkage_k
from research.context_experiment import load_store
store, _ = load_store(Path(S)/'runs'/'base')
slate = [s for s in pickle.loads((Path(S)/'runs'/'devcr'/'slates.pkl').read_bytes()) if s.name == 'six_nations_2024_r1'][0]
train = past_matches(store, slate.cutoff)
ev = (*STABLE_EVENTS, *EXTENDED_EVENTS)
f = estimate_status_factors(train, ev, slate.cutoff)
k = within_position_shrinkage_k(starter_equivalent(train, f, ev), ev, slate.cutoff)
T = pd.Series({(a, c): v for (a, b, c), v in f.items() if not b}).unstack(1)
out = Path(W)/'research'/'position_supersub_2026-10'
T.round(3).to_csv(out/'status_factors_dev_2024r1.csv')
pd.Series(k).round(0).to_csv(out/'shrinkage_k_dev_2024r1.csv', header=['K_minutes'])
print(T[['tries', 'metres', 'defenders_beaten', 'tackles', 'runs', 'try_assists', 'offload']].round(2).to_string())
print({e: round(v) for e, v in sorted(k.items())})
