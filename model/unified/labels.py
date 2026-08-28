"""Pooled fantasy-points labels across competitions for the stage-2 stack.

Stage 1 learns rugby events; stage 2 learns the event->fantasy-points ranking.
Its labels are *official* fantasy points, pooled across Six Nations and the
Nations Championship and normalised to a within-round percentile so a single
model spans both scoring systems.  The rubric that produced each label is
carried alongside as a conditioning feature (see :mod:`model.unified.rubric`),
so pooling never teaches contradictory point values -- the model sees which
rules applied.

Keys differ by competition and are kept verbatim for the feature stage:

* Six Nations rows are keyed by the API ``(fixture_id, player_id)`` and join
  directly to the canonical store.
* Nations Championship rows are keyed by the fantasy ``player_id`` and reach
  the store only through ``data/ncr/ncr_player_crosswalk.csv``; unmatched
  players fall back to position priors rather than being dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.ncr_rank_eval import load_actuals

from .data import ROOT

DATA = ROOT / "data"
OUT = DATA / "unified" / "fantasy_labels.csv"

# 2024 has no official fantasy points; 2023 used a materially different rubric
# (mean ~24.5 vs ~16.3), so it is excluded unless explicitly requested with its
# own rubric era.
SIX_NATIONS_TRAIN_SEASONS = (2025,)
SIX_NATIONS_HOLDOUT_SEASON = 2026
NCR_ROUNDS = {
    1: "feeds/players_post_gw1.json",
    2: "feeds/players_gw2.json",
    3: "feeds/players_gw3.json",
}


def _percentile(frame: pd.DataFrame) -> pd.Series:
    """Within-group percentile of official points (ties share the mean rank)."""
    return frame.groupby("group_id")["official_pts"].rank(pct=True, method="average")


def _six_nations_rows(seasons: tuple[int, ...], *, include_2023: bool) -> pd.DataFrame:
    targets = pd.read_csv(DATA / "model_targets.csv", low_memory=False)
    wanted = set(seasons)
    if include_2023:
        wanted.add(2023)
    frame = targets[targets["season"].isin(wanted) & targets["official_pts"].notna()].copy()
    out = pd.DataFrame({
        "competition": "six_nations",
        "rubric_era": frame["season"].map(lambda s: "2023" if s == 2023 else "modern"),
        "season": frame["season"].astype(int),
        "round": frame["round"].astype(int),
        "key_fixture": frame["fixture_id"].astype(str),
        "key_player": frame["player_id"].astype(str),
        "team": frame["team"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["canonical_pos"].astype(str),
        "is_forward": frame["is_forward"].astype(bool),
        "official_pts": frame["official_pts"].astype(float),
    })
    out["group_id"] = "6n-" + out["season"].astype(str) + "-r" + out["round"].astype(str)
    return out


def _ncr_rows() -> pd.DataFrame:
    rows = []
    for gw, feed in NCR_ROUNDS.items():
        actuals = load_actuals(DATA / "ncr" / feed)
        proj = pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
        # The incumbent projection cohort defines the official evaluation set;
        # joining on it guarantees stage-2 covers exactly the incumbent's rows.
        cohort = proj[["id", "name", "team", "pos"]].merge(
            actuals[["id", "actual", "actual_status"]], on="id", validate="one_to_one",
        )
        rows.append(pd.DataFrame({
            "competition": "ncr",
            "rubric_era": "modern",
            "season": 2026,
            "round": gw,
            "key_fixture": f"ncr-gw{gw}",
            "key_player": cohort["id"].astype(int).astype(str),
            "team": cohort["team"].astype(str),
            "player_name": cohort["name"].astype(str),
            "position": cohort["pos"].astype(str),
            "is_forward": cohort["pos"].astype(str).str.lower().str.contains(
                "prop|hook|row|forward|flank|lock|number"),
            "official_pts": cohort["actual"].astype(float),
            "status": cohort["actual_status"].astype(str),
            "group_id": f"ncr-2026-gw{gw}",
        }))
    return pd.concat(rows, ignore_index=True)


def build_fantasy_labels(*, include_2023: bool = False) -> pd.DataFrame:
    """Pool Six Nations and NCR official fantasy points into one label table."""
    if include_2023:
        # 2023 used a different scoring era but rubric.py encodes only the modern
        # rubric, so its labels would be conditioned on the wrong point weights.
        raise NotImplementedError(
            "2023 labels need a 2023 rubric era in rubric.py before they can be pooled")
    six = _six_nations_rows(SIX_NATIONS_TRAIN_SEASONS + (SIX_NATIONS_HOLDOUT_SEASON,),
                            include_2023=include_2023)
    ncr = _ncr_rows()
    pooled = pd.concat([six, ncr], ignore_index=True, sort=False)
    pooled["label_percentile"] = _percentile(pooled).to_numpy()
    # Normalised points (z within group) preserve magnitude for the regression head.
    grp = pooled.groupby("group_id")["official_pts"]
    pooled["label_z"] = ((pooled["official_pts"] - grp.transform("mean"))
                         / grp.transform("std").replace(0, 1.0)).to_numpy()
    if "status" not in pooled:
        pooled["status"] = ""
    pooled["status"] = pooled["status"].fillna("")
    return pooled


def write_fantasy_labels(path: Path = OUT, *, include_2023: bool = False) -> pd.DataFrame:
    frame = build_fantasy_labels(include_2023=include_2023)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
