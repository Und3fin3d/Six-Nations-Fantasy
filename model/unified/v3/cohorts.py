"""Join official fantasy label cohorts to canonical point-in-time match rows."""

from __future__ import annotations

import pandas as pd

from ..data import ROOT

DATA = ROOT / "data"


def _crosswalk() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "ncr" / "ncr_player_crosswalk.csv")
    frame = frame.dropna(subset=["api_player_id"]).copy()
    frame["fantasy_id"] = frame["fantasy_id"].astype(int).astype(str)
    frame["api_player_id"] = frame["api_player_id"].astype(int).astype(str)
    return frame[["fantasy_id", "api_player_id"]].drop_duplicates("fantasy_id")


def match_labels_to_store(labels: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Return every label row with a matched canonical row when available."""
    labels = labels.reset_index(drop=True).copy()
    labels["_label_row_id"] = labels.index
    store = store.copy()
    store["fixture_id"] = store["fixture_id"].astype(str)
    store["player_id"] = store["player_id"].astype(str)
    parts = []
    for (competition, round_no), block in labels.groupby(["competition", "round"], sort=False):
        block = block.copy()
        if competition == "six_nations":
            block["key_fixture"] = block["key_fixture"].astype(str)
            block["key_player"] = block["key_player"].astype(str)
            wanted_fixtures = set(block["key_fixture"])
            sub = store[store["fixture_id"].isin(wanted_fixtures)].copy()
            merged = block.merge(
                sub, left_on=["key_fixture", "key_player"],
                right_on=["fixture_id", "player_id"], how="left",
                validate="one_to_one", suffixes=("_label", ""),
            )
        elif competition == "ncr":
            block["key_player"] = block["key_player"].astype(str)
            block = block.merge(
                _crosswalk(), left_on="key_player", right_on="fantasy_id", how="left",
                validate="many_to_one",
            )
            sub = store[pd.to_numeric(store.get("ncr_gameday"), errors="coerce").eq(int(round_no))]
            merged = block.merge(
                sub, left_on="api_player_id", right_on="player_id", how="left",
                # A small number of fantasy catalogue aliases can resolve to
                # the same API player. Preserve the official cohort here and
                # surface the duplicate in the audit rather than dropping it.
                validate="many_to_one", suffixes=("_label", ""),
            )
        else:
            continue
        merged["store_matched"] = merged["fixture_id"].notna()
        parts.append(merged)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    if out["_label_row_id"].duplicated().any():
        raise AssertionError("label-to-store join inflated the official cohort")
    return out.sort_values("_label_row_id").reset_index(drop=True)
