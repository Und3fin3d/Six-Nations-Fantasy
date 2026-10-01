"""Squad decisions on synthetic slates under Six Nations-style rules.

Rules (identical to ``model.unified.rolling_eval.evaluate`` for the Six
Nations): 15 fantasy starters by positional quota (``model.ncr_project.REQUIRED``),
one super-sub who must be a named replacement, a captain who must be a named
starter, at most four players per team, no budget. Captain points count x2;
the super-sub counts x3 when he plays (else 0).

The MILP is the same as ``model.ncr_project.optimise`` written with sparse,
role-specific variables, after removing provably dominated roles:

* a player cannot be the best use of a fantasy-starter slot when strictly
  better players of his position (value, then pool order) span at least
  ``quota + 5`` other teams: at most ``quota`` of them can be in the squad,
  at most four teams can be at the cap, so one of the rest could replace him
  at no loss;
* the same argument restricted to named starters prunes captaincy.

Every replacement keeps a super-sub variable. ``flat`` squad points count every
squad member once (no captain or super-sub multiplier): a lower-variance view of
selection quality that removes the two knife-edge choices. ``tests/test_big_benchmark.py``
checks objective equality against ``optimise`` on random pools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from model.ncr_project import REQUIRED

MAX_PER_TEAM = 4
SQUAD = 16
POS = {'Prop': 'Prop', 'Hooker': 'Hooker', 'Second-row': 'Lock', 'Back-row': 'Loose Forward',
       'Scrum-half': 'Scrum Half', 'Fly-half': 'Fly Half', 'Centre': 'Centre', 'Back-three': 'Back Three'}
# Replacements without a prior recorded start take the conventional role of
# their bench jersey (published with the teamsheet, so known at the lock).
BENCH_JERSEY = {16: 'Hooker', 17: 'Prop', 18: 'Prop', 19: 'Lock', 20: 'Loose Forward',
                21: 'Scrum Half', 22: 'Fly Half', 23: 'Back Three'}
JITTER, SEED = 0.01, 12345


@dataclass(frozen=True)
class Squad:
    starters: np.ndarray  # pool indices of the 15 fantasy starters (captain included)
    captain: int
    sub: int
    objective: float


def pool_positions(position: pd.Series, jersey: pd.Series) -> np.ndarray:
    mapped = position.map(POS)
    fallback = pd.to_numeric(jersey, errors='coerce').map(BENCH_JERSEY)
    return mapped.fillna(fallback).fillna('Unknown').to_numpy(str)


def candidate_roles(pos: np.ndarray, team: np.ndarray, starter: np.ndarray, value: np.ndarray,
                    margin: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Masks of players kept for the fantasy-starter and captain roles."""
    n = len(value)
    keep_x, keep_c = np.zeros(n, bool), np.zeros(n, bool)
    order = np.lexsort((np.arange(n), -value))  # best first, ties by pool order
    for p, quota in REQUIRED.items():
        threshold = quota + margin
        for mask, keep in ((pos == p, keep_x), ((pos == p) & starter, keep_c)):
            teams_seen: set[str] = set()  # teams of every strictly better player so far
            for i in order[mask[order]]:
                if len(teams_seen) - (team[i] in teams_seen) < threshold:
                    keep[i] = True
                elif len(teams_seen) > threshold:
                    break  # no later player can have fewer than `threshold` other teams above him
                teams_seen.add(team[i])
    return keep_x, keep_c


def solve(pos, team, starter, value, sub_value, *, max_per_team: int = MAX_PER_TEAM,
          captain_weight: float = 2.0) -> Squad | None:
    pos, team = np.asarray(pos), np.asarray(team)
    starter, value, sub_value = np.asarray(starter, bool), np.asarray(value, float), np.asarray(sub_value, float)
    keep_x, keep_c = candidate_roles(pos, team, starter, value)
    xs, cs, ss = np.flatnonzero(keep_x), np.flatnonzero(keep_c), np.flatnonzero(~starter)
    roles = [(i, 'x') for i in xs] + [(i, 'c') for i in cs] + [(i, 's') for i in ss]
    nv = len(roles)
    objective = np.array([value[i] if r == 'x' else captain_weight*value[i] if r == 'c' else sub_value[i]
                          for i, r in roles])
    rows, cols, lower, upper = [], [], [], []
    constraint = 0

    def add(members, lo, hi):
        nonlocal constraint
        if not len(members):
            if lo > 0:
                raise _Infeasible
            return
        rows.extend([constraint]*len(members)); cols.extend(members)
        lower.append(lo); upper.append(hi); constraint += 1

    try:
        add([k for k, (_, r) in enumerate(roles) if r == 'c'], 1, 1)
        add([k for k, (_, r) in enumerate(roles) if r == 's'], 1, 1)
        for p, quota in REQUIRED.items():
            add([k for k, (i, r) in enumerate(roles) if r in 'xc' and pos[i] == p], quota, quota)
        for t in np.unique(team):
            add([k for k, (i, _) in enumerate(roles) if team[i] == t], -np.inf, max_per_team)
        by_player: dict[int, list[int]] = {}
        for k, (i, _) in enumerate(roles):
            by_player.setdefault(i, []).append(k)
        for members in by_player.values():
            if len(members) > 1:
                add(members, -np.inf, 1)
    except _Infeasible:
        return None
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(constraint, nv)).tocsr()
    result = milp(c=-objective, constraints=LinearConstraint(matrix, lower, upper),
                  integrality=np.ones(nv), bounds=Bounds(0, 1))
    if not result.success:
        return None
    chosen = [roles[k] for k in np.flatnonzero(result.x > 0.5)]
    starters = np.array(sorted(i for i, r in chosen if r in 'xc'))
    captain = next(i for i, r in chosen if r == 'c')
    sub = next(i for i, r in chosen if r == 's')
    return Squad(starters, int(captain), int(sub), float(objective @ (result.x > 0.5)))


class _Infeasible(Exception):
    pass


def realised_points(squad: Squad, actual: np.ndarray, played: np.ndarray) -> float:
    """Squad score under the ``model.ncr_eval.team_points`` multipliers."""
    total = float(actual[squad.starters].sum() + actual[squad.captain])
    return total + (3.0*float(actual[squad.sub]) if played[squad.sub] else 0.0)


def decide_slate(pool: pd.DataFrame, predictions: dict[str, np.ndarray], actual: np.ndarray,
                 draws: int, seed_key: int) -> list[dict]:
    """Realised, smoothed and hindsight-optimal squad points for every engine on one pool."""
    pos, team = pool['pos'].to_numpy(str), pool['team'].astype(str).to_numpy()
    starter = pool['started'].astype(bool).to_numpy()
    played = pool['minutes'].fillna(0).to_numpy(float) > 0
    bench = (~starter).astype(float)
    oracle = solve(pos, team, starter, actual, 3.0*actual*played*bench)
    if oracle is None:
        return []
    best = realised_points(oracle, actual, played)
    flat_oracle = solve(pos, team, starter, actual, actual*played*bench, captain_weight=1.0)  # members count once
    best_flat = float(actual[flat_oracle.starters].sum() + actual[flat_oracle.sub]*played[flat_oracle.sub])
    rng = np.random.default_rng([SEED, seed_key])
    noise = np.exp(rng.normal(0, JITTER, (draws, len(pool))))
    rows = []
    for engine, value in predictions.items():
        squad = solve(pos, team, starter, value, 3.0*value*bench)
        realised = realised_points(squad, actual, played)
        jittered = [solve(pos, team, starter, value*z, 3.0*value*z*bench) for z in noise]
        smoothed = [realised_points(q, actual, played) for q in jittered]
        smoothed_flat = [float(actual[q.starters].sum() + actual[q.sub]*played[q.sub]) for q in jittered]
        flat = float(actual[squad.starters].sum() + actual[squad.sub]*played[squad.sub])
        rows.append({'engine': engine, 'realised': realised, 'flat': flat,
                     'smoothed': float(np.mean(smoothed)) if draws else np.nan,
                     'smoothed_sd': float(np.std(smoothed)) if draws else np.nan,
                     'smoothed_flat': float(np.mean(smoothed_flat)) if draws else np.nan,
                     'oracle': best, 'regret': best - realised, 'oracle_flat': best_flat,
                     'smoothed_regret': best - float(np.mean(smoothed)) if draws else np.nan,
                     'predicted_objective': squad.objective,
                     'captain_points': float(actual[squad.captain]),
                     'sub_points': float(actual[squad.sub]*played[squad.sub]),
                     'pool_players': int(len(pool))})
    return rows


def decide_stage(manifest: dict, output: Path, engines, rubrics, draws: int) -> None:
    from .score import load_players, ENGINES
    from .rubrics import RUBRICS
    engines = list(engines or ENGINES)
    rubrics = list(rubrics or RUBRICS)
    eligible = {s['slate']: k for k, s in enumerate(manifest['slates']) if s['decision_eligible']}
    path = output/'decisions.csv'
    done = set()
    if path.exists():
        existing = pd.read_csv(path)
        done = set(zip(existing.slate, existing.rubric, existing.engine))
    for lock in manifest['locks']:
        name = pd.Timestamp(lock).strftime('%Y-%m')
        players = load_players(output, name)
        if players is None:
            print(f'decide: lock {name} not scored yet; stopping', flush=True)
            return
        rows = []
        for slate, pool in players.groupby('slate', sort=False):
            if slate not in eligible:
                continue
            for rubric in rubrics:
                todo = [e for e in engines if (slate, rubric, e) not in done]
                if not todo:
                    continue
                scorable = pool[np.isfinite(pool[f'actual__{rubric}'])].reset_index(drop=True)
                scorable = scorable[scorable['pos'].isin(list(REQUIRED))].reset_index(drop=True)
                predictions = {e: scorable[f'pred__{e}__{rubric}'].to_numpy(float) for e in todo}
                result = decide_slate(scorable, predictions, scorable[f'actual__{rubric}'].to_numpy(float),
                                      draws, eligible[slate]*7 + list(RUBRICS).index(rubric))
                for row in result:
                    rows.append({'slate': slate, 'lock': name, 'rubric': rubric, **row,
                                 'excluded_unscorable': int(len(pool) - len(scorable))})
        if rows:
            frame = pd.DataFrame(rows)
            frame.to_csv(path, mode='a', header=not path.exists(), index=False)
            done |= set(zip(frame.slate, frame.rubric, frame.engine))
        print(f'decide: lock {name} done ({len(rows)} rows)', flush=True)
    (output/'decide_info.json').write_text(json.dumps({'draws': draws, 'jitter': JITTER, 'seed': SEED}) + '\n')
