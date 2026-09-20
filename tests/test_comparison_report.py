"""The comparison contract must fail closed when an evaluation is missing."""
import numpy as np
import pandas as pd
import pytest

from model.unified.comparison_report import require_complete, paired_interval, SEASONS
from model.unified.domain_experiment import CONTROL
from model.unified.friendly_eval import TARGETS

CANDIDATE = "p3_seedbag_native"


def complete_tables():
    fantasy = pd.DataFrame([
        dict(competition=c,season=y,round=r,slate=f"{c}_{y}_r{r}",engine=e,mae=2.,team_points=100.)
        for (c,y),n in SEASONS.items() for r in range(1,n+1)
        for e in ("empirical_baseline",CONTROL,CANDIDATE)
    ])
    ids = [str(i) for i in range(15)]
    stats = pd.DataFrame([
        dict(engine=e,fixture_id=f,target=t,n=46,mae=1.)
        for e in ("empirical_raw",CONTROL,CANDIDATE) for f in ids for t in TARGETS
    ])
    return fantasy,stats,ids


def test_complete_four_evaluations_are_accepted():
    fantasy,stats,ids = complete_tables()
    summary,per_stat = require_complete(fantasy,stats,ids,CANDIDATE)
    assert len(summary) == 3
    assert summary.fixtures.eq(15).all()
    assert summary.count_stat_mae.eq(1).all()
    assert len(per_stat) == 3 * len(TARGETS)


@pytest.mark.parametrize("damage", ["round","season","comparator","duplicate","nonfinite"])
def test_incomplete_fantasy_comparison_fails(damage):
    fantasy,stats,ids = complete_tables()
    if damage == "round":
        fantasy = fantasy.drop(index=0)
    elif damage == "season":
        fantasy = fantasy[fantasy.season.ne(2025)]
    elif damage == "comparator":
        fantasy = fantasy[fantasy.engine.ne(CONTROL)]
    elif damage == "duplicate":
        fantasy = pd.concat([fantasy,fantasy.iloc[[0]]])
    else:
        fantasy.loc[0,"mae"] = np.nan
    with pytest.raises(ValueError):
        require_complete(fantasy,stats,ids,CANDIDATE)


@pytest.mark.parametrize("damage", ["game","stat","comparator","support","nonfinite","duplicate"])
def test_incomplete_friendly_comparison_fails(damage):
    fantasy,stats,ids = complete_tables()
    if damage == "game":
        stats = stats[stats.fixture_id.ne(ids[-1])]
    elif damage == "stat":
        stats = stats.drop(index=0)
    elif damage == "comparator":
        stats = stats[stats.engine.ne(CONTROL)]
    elif damage == "support":
        stats.loc[0,"n"] = 0
    elif damage == "nonfinite":
        stats.loc[0,"mae"] = np.nan
    else:
        stats = pd.concat([stats,stats.iloc[[0]]])
    with pytest.raises(ValueError):
        require_complete(fantasy,stats,ids,CANDIDATE)


def test_paired_interval_is_reproducible_and_uses_cluster_count():
    result = paired_interval([-2.,1.,4.])
    assert result == paired_interval([-2.,1.,4.])
    assert result["clusters"] == 3
    assert result["difference"] == 1.
    assert result["p05"] <= result["difference"] <= result["p95"]
    with pytest.raises(ValueError):
        paired_interval([])
    with pytest.raises(ValueError):
        paired_interval([np.nan])
