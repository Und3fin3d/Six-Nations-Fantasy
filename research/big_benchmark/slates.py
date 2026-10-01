"""Frozen locks, blocks and synthetic slates for the large benchmark.

Design (see ``research/BIG_BENCHMARK_REPORT_2026-10.md``):

* **Slate**: one competition round (``source_round``), the unit a fantasy game
  would lock. Club competitions, the Six Nations and the Nations Championship
  use their own rounds. Every other international fixture (Rugby Championship,
  Pacific Nations Cup, World Cup, Lions tour, tour matches and ad-hoc tests)
  is pooled into one ``intl_<year>_w<week>`` slate per ISO week, like a
  multi-nation test-window game; their 2-fixture rounds could not field a
  15-player squad under a four-per-team cap. Fixtures more than six days
  after a slate's first kickoff (rescheduled games) form their own ``_late``
  slate.
* **Block**: a competition season (``source_season``); pooled international
  weeks use calendar year and window (spring/summer/autumn).
* **Lock**: forecasts are refitted once per calendar month at 00:00 UTC on the
  1st and every slate whose first kickoff falls in that month is forecast from
  that fit, with history strictly before the lock (``model.history.past_matches``:
  kickoff + 3 h availability). Slates never straddle locks; horizons are
  therefore 0-31 days, comparable to the tournament-frozen official slates
  (0-6 weeks). One fit per month bounds compute at ~39 fits for July 2023 to
  September 2026, versus ~600 for per-round refits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPETITIONS = {
    1230: 'top14', 1236: 'urc', 1242: 'super_rugby_pacific', 1218: 'premiership',
    1464: 'champions_cup', 1470: 'challenge_cup', 1254: 'currie_cup', 1260: 'npc',
    2538: 'japan_league_one', 1266: 'six_nations', 1296: 'rugby_championship',
    1272: 'rugby_world_cup', 1326: 'pacific_nations_cup', 1338: 'lions_tour',
    696: 'nations_championship', 30: 'international', 18: 'tour_match',
    2202: 'autumn_nations_cup', 2208: 'tri_nations',
}
WEEKLY = {30, 18, 1338, 1296, 1326, 1272}  # pooled into weekly international slates
EVALUATION_START = pd.Timestamp('2023-07-01', tz='UTC')
LATE_DAYS = 6
MIN_DECISION_FIXTURES = 3
# Candidate post-processing coefficients (team strength, matchup) and the
# kicking exponent were fitted on international tournament blocks that ended
# before 2025; the official 2025/26 slates informed candidate selection.
COEFFICIENT_FIT_END = pd.Timestamp('2025-01-01', tz='UTC')
OFFICIAL_SELECTION = {('six_nations', '2025'), ('six_nations', '2026'), ('nations_championship', '2026')}


def _window(month: int) -> str:
    return 'spring' if month <= 4 else 'summer' if month <= 8 else 'autumn'


def fixture_table(store: pd.DataFrame) -> pd.DataFrame:
    fixtures = store.sort_values(['match_at', 'fixture_id']).drop_duplicates('fixture_id').copy()
    fixtures['match_at'] = pd.to_datetime(fixtures['match_at'], utc=True)
    fixtures['comp'] = fixtures['competition_id_cache'].astype(int).map(COMPETITIONS)
    if fixtures['comp'].isna().any():
        raise ValueError(f'unmapped competitions: {sorted(fixtures.loc[fixtures.comp.isna(), "competition_id_cache"].unique())}')
    return fixtures


def assign_slates(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Add ``slate``, ``block``, ``family`` and ``lock`` columns, one row per fixture."""
    out = fixtures.copy()
    iso = out['match_at'].dt.isocalendar()
    comp_id = out['competition_id_cache'].astype(int)
    weekly = comp_id.isin(WEEKLY)
    season = out['source_season'].astype('Int64').astype(str)
    rounds = pd.to_numeric(out['source_round'], errors='coerce')
    base = np.where(
        weekly,
        'intl_' + iso['year'].astype(str) + '_w' + iso['week'].astype(int).map('{:02d}'.format),
        out['comp'] + '_' + season + '_r' + rounds.fillna(0).astype(int).map('{:02d}'.format))
    out['slate'] = base
    out['block'] = np.where(
        weekly, 'intl_' + out['match_at'].dt.year.astype(str) + '_' + out['match_at'].dt.month.map(_window),
        out['comp'] + '_' + season)
    if (~weekly & rounds.isna()).any():
        raise ValueError('round-based competition fixture without a source round')
    first = out.groupby('slate')['match_at'].transform('min')
    late = out['match_at'] - first > pd.Timedelta(days=LATE_DAYS)
    out.loc[late, 'slate'] = out.loc[late, 'slate'] + '_late'
    first = out.groupby('slate')['match_at'].transform('min')
    out['slate_first_kickoff'] = first
    out['lock'] = first.dt.tz_convert(None).dt.to_period('M').dt.to_timestamp().dt.tz_localize('UTC')
    out['family'] = np.where(out['competition_level'].eq('international'), 'international', 'club')
    return out


def build_manifest(store: pd.DataFrame, *, store_sha256: str, inputs: dict, end: pd.Timestamp | None = None) -> dict:
    fixtures = assign_slates(fixture_table(store))
    fixtures = fixtures[fixtures['slate_first_kickoff'].ge(EVALUATION_START)]
    if end is not None:
        fixtures = fixtures[fixtures['slate_first_kickoff'].lt(end)]
    rows = store[store['fixture_id'].astype(str).isin(fixtures['fixture_id'].astype(str))]
    players = rows.groupby('fixture_id').size()
    slates = []
    for slate, group in fixtures.groupby('slate', sort=False):
        comp = 'international_week' if str(group['slate'].iloc[0]).startswith('intl_') else str(group['comp'].iloc[0])
        season = str(group['source_season'].astype('Int64').iloc[0])
        first = group['match_at'].min()
        slates.append({
            'slate': slate, 'block': str(group['block'].iloc[0]), 'competition': comp,
            'family': str(group['family'].iloc[0]), 'lock': group['lock'].iloc[0].isoformat(),
            'first_kickoff': first.isoformat(), 'last_kickoff': group['match_at'].max().isoformat(),
            'fixtures': sorted(group['fixture_id'].astype(str)),
            'fixture_competitions': sorted(group['comp'].unique().tolist()),
            'players': int(players.reindex(group['fixture_id']).sum()),
            'decision_eligible': bool(group['fixture_id'].nunique() >= MIN_DECISION_FIXTURES),
            'coefficients_in_sample': bool(group['family'].iloc[0] == 'international' and first < COEFFICIENT_FIT_END),
            'official_selection_overlap': (comp, season) in OFFICIAL_SELECTION,
        })
    slates.sort(key=lambda s: (s['lock'], s['first_kickoff'], s['slate']))
    locks = sorted({s['lock'] for s in slates})
    payload = {
        'schema_version': 1,
        'design': 'monthly shared locks; slate = competition round (ISO week for ad-hoc internationals); '
                  'history strictly before lock with kickoff+3h availability',
        'evaluation_start': EVALUATION_START.isoformat(),
        'store_sha256': store_sha256, 'inputs_sha256': inputs,
        'locks': locks, 'slates': slates,
        'n_slates': len(slates), 'n_fixtures': int(sum(len(s['fixtures']) for s in slates)),
        'n_players': int(sum(s['players'] for s in slates)),
    }
    payload['manifest_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def write_or_check(path: Path, manifest: dict) -> dict:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError(f'frozen manifest differs from the current inputs: {path}; use a new output directory')
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def slate_frame(manifest: dict) -> pd.DataFrame:
    frame = pd.DataFrame(manifest['slates'])
    frame['lock'] = pd.to_datetime(frame['lock'], utc=True)
    return frame


def fixture_slates(manifest: dict) -> pd.Series:
    return pd.Series({f: s['slate'] for s in manifest['slates'] for f in s['fixtures']}, name='slate')
