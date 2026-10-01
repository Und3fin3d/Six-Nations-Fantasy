"""Command line for the large computed-rubric benchmark.

Example::

    OMP_NUM_THREADS=2 python -m research.big_benchmark --base RUNS/base --output OUT --stage all
    python -m research.big_benchmark --base RUNS/base --output OUT --stage report \\
        --engines p3_robust empirical_baseline h1_oct1 h2 mk

``--base`` is a ``model.unified.rolling_eval`` output directory holding
``inputs/player_match.csv`` and ``inputs/training_features.pkl`` (the cached
research store and its prior-only features). Every stage is resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from model.unified.data import ROOT

from . import slates as slate_module

DATA = ROOT/'data'
INPUT_FILES = (
    DATA/'unified'/'p3_hillclimb'/'config.json', DATA/'wr_rankings.csv', DATA/'rp_compstats.csv',
    ROOT/'research'/'hillclimb_2026-10-01'/'team_strength_beta.json',
    ROOT/'research'/'hillclimb_2026-10-02'/'matchup_coefficients.json',
)
STAGES = ('manifest', 'fit', 'score', 'decide', 'report')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(base: Path):
    from research.context_experiment import load_store
    store, prepared = load_store(base)
    if len(store) != len(prepared) or not store.index.equals(prepared.index):
        raise ValueError('training features do not align with the research store')
    return store, prepared


def manifest_stage(base: Path, output: Path, store: pd.DataFrame) -> dict:
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in INPUT_FILES}
    inputs['base/inputs/training_features.pkl'] = sha256(base/'inputs'/'training_features.pkl')
    manifest = slate_module.build_manifest(store, store_sha256=sha256(base/'inputs'/'player_match.csv'), inputs=inputs)
    manifest = slate_module.write_or_check(output/'manifest.json', manifest)
    info = {
        'python': platform.python_version(),
        'packages': {n: version(n) for n in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm')},
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'research'/'big_benchmark').glob('*.py'))},
    }
    (output/'run_info.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print(f"manifest: {manifest['n_slates']} slates, {manifest['n_fixtures']} fixtures, "
          f"{manifest['n_players']} player rows, {len(manifest['locks'])} locks", flush=True)
    return manifest


def fit_stage(store, prepared, manifest: dict, output: Path, locks: list[str] | None) -> None:
    from .fit import fit_lock, lock_name
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    wr = pd.read_csv(DATA/'wr_rankings.csv', parse_dates=['snapshot_date'])
    frame = slate_module.slate_frame(manifest)
    mapping = slate_module.fixture_slates(manifest)
    index = {s['slate']: s['fixtures'] for s in manifest['slates']}
    for lock, group in frame.groupby('lock', sort=True):
        name = lock_name(lock)
        if locks and name not in locks:
            continue
        fixtures = [f for s in group['slate'] for f in index[s]]
        start = time.monotonic()
        summary = fit_lock(store, prepared, lock, mapping.loc[fixtures], output/'locks'/name, config, wr)
        print(f"lock {name}: {summary['candidate_rows']} rows, {len(summary['slates'])} slates, "
              f"{summary['seconds']} ({time.monotonic()-start:.0f}s wall)", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--locks', nargs='*', help='restrict fit/score to these YYYY-MM locks')
    parser.add_argument('--engines', nargs='*', help='engines to score (default: all registered)')
    parser.add_argument('--rubrics', nargs='*', help='rubrics to score (default: all)')
    parser.add_argument('--draws', type=int, default=40, help='jitter draws for smoothed squad points')
    parser.add_argument('--official-rounds', type=Path,
                        help='hillclimb2_eval rounds.csv of the 13 official rounds, for the power comparison')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == 'all' else (args.stage,)
    store = prepared = None
    if {'manifest', 'fit', 'score'} & set(stages):
        store, prepared = load_inputs(args.base)
        manifest = manifest_stage(args.base, args.output, store)
    else:
        manifest = json.loads((args.output/'manifest.json').read_text())
    if 'fit' in stages:
        fit_stage(store, prepared, manifest, args.output, args.locks)
    if 'score' in stages:
        from .score import score_stage
        score_stage(store, manifest, args.output, args.locks, args.engines, args.rubrics)
    if 'decide' in stages:
        from .decision import decide_stage
        decide_stage(manifest, args.output, args.engines, args.rubrics, args.draws)
    if 'report' in stages:
        from .report import report_stage
        report_stage(manifest, args.output, args.engines, args.rubrics, args.official_rounds)


if __name__ == '__main__':
    main()
