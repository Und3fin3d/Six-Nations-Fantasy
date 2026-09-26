from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from model.ncr_project import REQUIRED


@dataclass(frozen=True)
class SelectionRules:
    budget: float = 100.0
    max_nation: int = 3
    max_hemi: int | None = 10
    captain_status: tuple[str, ...] = ('P',)
    supersub_status: tuple[str, ...] = ('B',)


def standardise(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Role utility must be a finite one-dimensional array')
    scale = float(values.std())
    return (values - values.mean()) / scale if scale > 1e-12 else np.zeros(len(values))


def validate_pool(pool, rules):
    required = ['id', 'team', 'pos', 'value', 'status']
    if rules.max_hemi is not None:
        required.append('hemi')
    if len(pool) < 16 or pool[required].isna().any().any() or pool.id.duplicated().any():
        raise ValueError('A complete, uniquely identified selection pool is required')
    if not pool.pos.isin(REQUIRED).all():
        raise ValueError('Unknown selection position')
    if not np.isfinite(pool.value.to_numpy(float)).all() or (pool.value < 0).any():
        raise ValueError('Prices must be known, finite and nonnegative')
    if np.isnan(rules.budget) or rules.budget <= 0 or rules.max_nation < 1:
        raise ValueError('Invalid selection limits')
    if rules.max_hemi is not None and not pool.hemi.isin([1, 2]).all():
        raise ValueError('Unknown hemisphere in a hemisphere-constrained pool')


def constraints_for(pool, rules):
    validate_pool(pool, rules)
    n = len(pool)
    zero, one, eye = np.zeros(n), np.ones(n), np.eye(n)
    constraints = [
        LinearConstraint(np.r_[one, zero, zero], 16, 16),
        LinearConstraint(np.r_[zero, one, zero], 1, 1),
        LinearConstraint(np.r_[zero, zero, one], 1, 1),
        LinearConstraint(np.c_[-eye, eye, np.zeros((n, n))], -np.inf, 0),
        LinearConstraint(np.c_[-eye, eye, eye], -np.inf, 0),
        LinearConstraint(np.r_[pool.value.to_numpy(float), zero, zero], -np.inf, rules.budget),
    ]
    for team in pool.team.unique():
        mask = pool.team.eq(team).to_numpy(float)
        constraints.append(LinearConstraint(np.r_[mask, zero, zero], -np.inf, rules.max_nation))
    if rules.max_hemi is not None:
        for hemi in (1, 2):
            mask = pool.hemi.eq(hemi).to_numpy(float)
            constraints.append(LinearConstraint(np.r_[mask, zero, zero], -np.inf, rules.max_hemi))
    for position, count in REQUIRED.items():
        mask = pool.pos.eq(position).to_numpy(float)
        constraints.append(LinearConstraint(np.r_[mask, -mask, zero], count, count))
    not_captain = ~pool.status.isin(rules.captain_status).to_numpy()
    not_sub = ~pool.status.isin(rules.supersub_status).to_numpy()
    constraints.append(LinearConstraint(np.r_[zero, zero, not_captain], 0, 0))
    constraints.append(LinearConstraint(np.r_[zero, not_sub, zero], 0, 0))
    return constraints


def verify_selection(squad, rules):
    if len(squad) != 16 or squad.id.duplicated().any():
        raise ValueError('Selection is not sixteen distinct players')
    if squad.is_sub.sum() != 1 or squad.is_capt.sum() != 1 or (squad.is_sub & squad.is_capt).any():
        raise ValueError('Invalid captain or super-sub assignment')
    xv = squad.loc[~squad.is_sub]
    if xv.groupby('pos').size().to_dict() != REQUIRED:
        raise ValueError('Selection violates positional quotas')
    if squad.value.sum() > rules.budget + 1e-7 or squad.groupby('team').size().max() > rules.max_nation:
        raise ValueError('Selection violates budget or country limits')
    if rules.max_hemi is not None and squad.groupby('hemi').size().max() > rules.max_hemi:
        raise ValueError('Selection violates hemisphere limits')
    if not squad.loc[squad.is_capt, 'status'].isin(rules.captain_status).all():
        raise ValueError('Captain violates declared policy eligibility')
    if not squad.loc[squad.is_sub, 'status'].isin(rules.supersub_status).all():
        raise ValueError('Super-sub violates declared policy eligibility')
    return dict(players=len(squad), cost=float(squad.value.sum()),
                max_country_count=int(squad.groupby('team').size().max()),
                max_hemisphere_count=int(squad.groupby('hemi').size().max()) if 'hemi' in squad else None)


def optimise_roles(pool, xv_utility, captain_utility, supersub_utility, *, rules=None, mode='joint'):
    rules = rules or SelectionRules()
    constraints = constraints_for(pool, rules)
    n = len(pool)
    heads = [np.asarray(head, dtype=float) for head in (xv_utility, captain_utility, supersub_utility)]
    if any(head.shape != (n,) or not np.isfinite(head).all() for head in heads):
        raise ValueError('Every candidate must have all finite role utilities before optimisation')
    xv, captain, sub = heads
    zero = np.zeros(n)
    components = [np.r_[xv, -xv, zero], np.r_[zero, sub, zero], np.r_[zero, zero, captain]]
    if mode == 'joint':
        objectives = [sum(components)]
    elif mode == 'sequential':
        objectives = components
    else:
        raise ValueError(f'Unknown selection policy {mode}')
    stages = []
    for objective in objectives:
        result = milp(-objective, integrality=np.ones(3 * n), bounds=Bounds(0, 1),
                      constraints=constraints, options={'mip_rel_gap': 0.0})
        if not result.success or result.mip_gap > 1e-9:
            raise RuntimeError(f'No certified optimal complete squad: {result.message}')
        value = float(objective @ result.x)
        stages.append(dict(objective=value, gap=float(result.mip_gap), nodes=int(result.mip_node_count)))
        constraints.append(LinearConstraint(objective, value - 1e-8, np.inf))
    chosen = result.x[:n] > 0.5
    frame = pool.copy()
    frame['in_squad'] = chosen
    frame['is_sub'] = result.x[n:2*n] > 0.5
    frame['is_capt'] = result.x[2*n:] > 0.5
    selected = frame.loc[chosen].copy()
    legality = verify_selection(selected, rules)
    rounded = np.r_[chosen, frame.is_sub, frame.is_capt].astype(float)
    violation = 0.0
    for constraint in constraints:
        values = constraint.A @ rounded
        violation = max(violation, float(np.maximum(constraint.lb - values, 0).max()),
                        float(np.maximum(values - constraint.ub, 0).max()))
    if violation > 2e-7:
        raise ValueError(f'Rounded selection violates constraints by {violation}')
    rule_record = asdict(rules)
    if not np.isfinite(rule_record['budget']):
        rule_record['budget'] = 'unbounded'
    return selected, dict(mode=mode, rules=rule_record, stages=stages,
                          final_component_objectives=[float(c @ rounded) for c in components],
                          maximum_constraint_violation=violation, **legality)
