"""P0.2 — NCR rubric-coverage audit, proxy repair, incumbent POTM ablation.

The v3 development proxy (observable points) omits every NCR event with store
coverage 0 (potm +15, scrums_won +2 front-row, lineout_steals +5,
interceptions +5), producing a large position-systematic bias (GW1 −5.08/row).
This module:

1. quantifies each unobserved event's likely contribution using per-position
   per-80 rates from store rows where the event IS observed (Six Nations
   internationals) — no NCR label is used to build the correction;
2. proposes the repaired proxy = observable + position-level expected
   unobserved-event points (+ scraped POTM where present) and reports how much
   of the official-vs-proxy gap it closes (an instrument calibration, not a
   model-admission decision);
3. ablates the deterministic POTM bump out of the NCR incumbent's projections
   (potm = starter_exp − per80·exp_min/80, `model/ncr_project.py:366-375`) and
   re-measures its top-10 capture, deciding whether a POTM head is worth
   building at all.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..scoring import NationsChampionshipScorer

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
BENCH = DATA / "unified" / "v3" / "benchmark"

UNOBSERVED = ("potm", "scrums_won", "lineout_steals", "interceptions")
FRONT_ROW = {"Prop", "Hooker"}


def _store() -> pd.DataFrame:
    return pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                       parse_dates=["date"])


def position_expectations(store: pd.DataFrame) -> pd.DataFrame:
    """Per-position per-80 expectations for NCR-unobserved events, estimated
    from international rows where the event is observed."""
    intl = store[store["competition_level"].eq("international")].copy()
    weights = NationsChampionshipScorer.weights
    rows = []
    for event in UNOBSERVED:
        available = intl.get(f"available__{event}")
        if available is None:
            continue
        sub = intl[available.astype(bool) & intl["minutes"].gt(0)]
        if sub.empty:
            rows.append({"event": event, "n": 0, "note": "never observed anywhere"})
            continue
        per80 = (pd.to_numeric(sub[event], errors="coerce")
                 / sub["minutes"] * 80).clip(lower=0)
        by_pos = per80.groupby(sub["position"]).mean()
        weight = weights.get(event, 2 if event == "scrums_won" else 0)
        for position, rate in by_pos.items():
            rows.append({"event": event, "position": position, "n": int(len(sub)),
                         "per80": float(rate), "weight": weight,
                         "exp_pts_per80": float(rate) * weight})
    return pd.DataFrame(rows)


def repaired_proxy(decomposition: pd.DataFrame,
                   expectations: pd.DataFrame) -> pd.DataFrame:
    """observable + position-level expected unobserved points (+ scraped POTM)."""
    out = decomposition.copy()
    correction = np.zeros(len(out))
    for event, block in expectations.dropna(subset=["position"]).groupby("event"):
        table = block.set_index("position")["exp_pts_per80"]
        rate = out["position_label"].map(table).fillna(0.0)
        if event == "scrums_won":
            rate = rate.where(out["position_label"].isin(FRONT_ROW), 0.0)
        correction += (rate * out["act_minutes"] / 80.0).to_numpy()
    out["position_correction"] = correction
    potm = pd.read_csv(DATA / "ncr" / "ncr_potm.csv")
    potm_ids = set(potm["player_id"].astype(str))
    is_potm = out["player_id"].astype(str).isin(potm_ids) & out["gw"].eq(2)
    out["potm_correction"] = np.where(is_potm, 15.0, 0.0)
    out["repaired_points"] = (out["observable_points"] + out["position_correction"]
                              + out["potm_correction"])
    return out


def incumbent_potm_ablation() -> pd.DataFrame:
    predictions = pd.read_csv(BENCH / "predictions.csv")
    rows = []
    for gw in (1, 2):
        projection = pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
        projection["potm_term"] = (projection["starter_exp"]
                                   - projection["per80"] * projection["exp_min"] / 80.0)
        projection["ablated_exp"] = projection["starter_exp"] - projection["potm_term"]
        block = predictions[predictions["competition"].eq("ncr")
                            & predictions["engine"].eq("baseline")
                            & predictions["round"].eq(gw)].copy()
        block["key_player"] = block["key_player"].astype(str)
        projection["key_player"] = projection["id"].astype(int).astype(str)
        merged = block.merge(
            projection[["key_player", "ablated_exp", "potm_term"]],
            on="key_player", how="left", validate="one_to_one",
        ).dropna(subset=["incumbent_points"]).reset_index(drop=True)
        renamed = merged.rename(columns={"official_pts": "actual"})
        full = _group_metrics(renamed, "incumbent_points", actual_col="actual")
        ablated = _group_metrics(renamed.dropna(subset=["ablated_exp"]).reset_index(drop=True),
                                 "ablated_exp", actual_col="actual")
        rows.append({
            "gw": gw, "n": len(merged),
            "mean_potm_term": float(merged["potm_term"].mean()),
            "max_potm_term": float(merged["potm_term"].max()),
            "incumbent_top10": full["top_10_capture"],
            "ablated_top10": ablated["top_10_capture"],
            "incumbent_spearman": full["spearman"],
            "ablated_spearman": ablated["spearman"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    store = _store()
    expectations = position_expectations(store)
    decomposition = pd.read_csv(OUT / "p0_error_decomposition.csv")
    labeled = pd.read_csv(OUT / "p0_row_detail.csv") if (OUT / "p0_row_detail.csv").exists() else None
    if labeled is not None:
        decomposition = labeled
    repaired = repaired_proxy(decomposition, expectations)
    before = float((repaired["observable_points"] - repaired["official_pts"]).abs().mean())
    after = float((repaired["repaired_points"] - repaired["official_pts"]).abs().mean())
    bias_before = float((repaired["observable_points"] - repaired["official_pts"]).mean())
    bias_after = float((repaired["repaired_points"] - repaired["official_pts"]).mean())
    from scipy.stats import spearmanr
    rank_before = float(np.mean([
        spearmanr(g["observable_points"], g["official_pts"]).statistic
        for _, g in repaired.groupby("gw")
    ]))
    rank_after = float(np.mean([
        spearmanr(g["repaired_points"], g["official_pts"]).statistic
        for _, g in repaired.groupby("gw")
    ]))
    ablation = incumbent_potm_ablation()
    OUT.mkdir(parents=True, exist_ok=True)
    expectations.to_csv(OUT / "p0_unobserved_event_expectations.csv", index=False)
    repaired.to_csv(OUT / "p0_repaired_proxy_rows.csv", index=False)
    ablation.to_csv(OUT / "p0_incumbent_potm_ablation.csv", index=False)
    summary = {
        "proxy_mae_vs_official": {"before": before, "after": after},
        "proxy_bias_vs_official": {"before": bias_before, "after": bias_after},
        "proxy_within_gw_spearman": {"before": rank_before, "after": rank_after},
        "incumbent_potm_ablation": ablation.to_dict(orient="records"),
        "note": ("Corrections use only cross-competition store rates + scraped "
                 "POTM; official NCR labels enter only as the measurement target "
                 "of this instrument calibration."),
    }
    (OUT / "p0_rubric_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
