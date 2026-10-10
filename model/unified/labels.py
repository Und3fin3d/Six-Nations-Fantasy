from __future__ import annotations

import json
import os

import pandas as pd

OUT = "data/unified/fantasy_labels.csv"

SIX_NATIONS_TRAIN_SEASONS = (2025,)
SIX_NATIONS_HOLDOUT_SEASON = 2026
NCR_ROUNDS = {
    1: "feeds/players_post_gw1.json",
    2: "feeds/players_gw2.json",
    3: "feeds/players_gw3.json",
}


def load_actuals(path: str) -> pd.DataFrame:
    players = json.load(open(path))["Data"]["Value"]["Players"]
    return pd.DataFrame([
        {
            "id": int(float(player["id"])),
            "actual_name": player.get("full_name") or player.get("display_name"),
            "actual": float(player.get("cur_gd_points") or 0),
            "actual_status": player.get("player_status") or "",
        }
        for player in players
    ])

def _percentile(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("group_id")["official_pts"].rank(pct=True, method="average")


def _six_nations_rows(seasons: tuple[int, ...], *, include_2023: bool) -> pd.DataFrame:
    targets = pd.read_csv("data/model_targets.csv", low_memory=False)
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
        actuals = load_actuals(f"data/ncr/{feed}")
        proj = pd.read_csv(f"data/ncr/ncr_gw{gw}_projections.csv")
        cohort = proj[["id", "name", "team", "pos"]].merge(
            actuals[["id", "actual", "actual_status"]], on="id",
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
    if include_2023:
        raise NotImplementedError(
            "2023 labels need a 2023 rubric era in rubric.py before they can be pooled")
    six = _six_nations_rows(SIX_NATIONS_TRAIN_SEASONS + (SIX_NATIONS_HOLDOUT_SEASON,),
                            include_2023=include_2023)
    ncr = _ncr_rows()
    pooled = pd.concat([six, ncr], ignore_index=True, sort=False)
    pooled["label_percentile"] = _percentile(pooled).to_numpy()
    grp = pooled.groupby("group_id")["official_pts"]
    pooled["label_z"] = ((pooled["official_pts"] - grp.transform("mean"))
                         / grp.transform("std").replace(0, 1.0)).to_numpy()
    if "status" not in pooled:
        pooled["status"] = ""
    pooled["status"] = pooled["status"].fillna("")
    return pooled


def write_fantasy_labels(path: str = OUT, *, include_2023: bool = False) -> pd.DataFrame:
    frame = build_fantasy_labels(include_2023=include_2023)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
