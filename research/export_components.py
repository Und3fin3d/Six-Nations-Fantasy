"""Save each robust P3 component's raw forecasts for the official slates.

Loads the fitted ``p3_robust_native`` blend saved by rolling_eval and writes
``components/<slate>/{empirical,v4}.jsonl`` so per-event blend weights can be
studied without refitting. Cache-only.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from model.history import past_matches
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.rolling_eval import official_slates
from research.context_experiment import load_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--slates', nargs='*')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    store, prepared = load_store(args.base)
    for slate in official_slates(store, ('six_nations', 'ncr')):
        if args.slates and slate.name not in args.slates:
            continue
        path = args.base/'models'/slate.name/'p3_robust_native.pkl'
        if not path.exists():
            print(f'{slate.name}: no fitted model', flush=True)
            continue
        train = past_matches(store, slate.cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        _, candidates = build_frozen_feature_frames(train, slate.candidates, v4=True,
                                                    prepared_train=prepared.loc[train.index])
        model = EventWeightedBlend.load(path)
        out = args.base/'components'/slate.name
        out.mkdir(parents=True, exist_ok=True)
        for name, component in (('empirical', model.empirical), ('v4', model.v4)):
            raw = component.predict_frame(candidates)
            (out/f'{name}.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))
        print(f'{slate.name}: saved', flush=True)


if __name__ == '__main__':
    main()
