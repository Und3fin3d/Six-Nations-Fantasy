"""P0 — repair the ruler before any v4 modelling.

P0.1 Error decomposition of the frozen baseline's NCR predictions into
     appearance / minutes / per-80-rate / rubric-coverage terms, split by
     international-history depth and forward/back (branch rule S1: a
     minutes-dominated result stops the tail/rate program).
P0.2 Full NCR rubric-coverage audit + incumbent POTM ablation.
P0.3 Pinned picker-sensitivity margin (see p0_margin.py).

Read-only over frozen v3 artifacts; writes reports under data/unified/v4/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..features import build_pit_features
from ..gbdt import UniversalGBDT
from ..labels import build_fantasy_labels
from ..scoring import scorer_for
from ..v3.cohorts import match_labels_to_store
from ..v3.harness import attach_match_timestamps

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
BENCH = DATA / "unified" / "v3" / "benchmark"

# Thin history is defined by INTERNATIONAL exposure (red-team correction #2):
# prior international rows in the canonical store at the fixture date.
THIN_INTL_MATCHES = 5


def load_store() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                      parse_dates=["date"])
    store = build_pit_features(attach_match_timestamps(raw))
    is_intl = store["competition_level"].eq("international").astype(float)
    store["intl_prior_matches"] = (
        is_intl.groupby(store["player_id"]).transform(lambda x: x.shift(1).cumsum())
    ).fillna(0.0)
    return store


def _expected_points_terms(model: UniversalGBDT, rows: pd.DataFrame,
                           competition: str) -> pd.DataFrame:
    """Exact per-row decomposition of model-vs-observable points error.

    For played rows the identity
        pred_count - act_count
            = (pred_min - act_min)/80 * pred_rate + act_min/80 * (pred_rate - act_rate)
    splits each event's error into a minutes term and a per-80-rate term.
    Rows the player did not play collapse into a single appearance term.
    """
    scorer = scorer_for(competition)
    weights = dict(scorer.weights)
    if competition == "ncr":
        weights["scrums_won"] = 2  # front-row allocation, linear like the rest
    predictions = model.predict_frame(rows)
    records = []
    for (_, row), prediction in zip(rows.iterrows(), predictions):
        pred_min = float(prediction.minutes.mean)
        act_min = float(pd.to_numeric(row.get("minutes"), errors="coerce") or 0.0)
        expected = minutes_term = rate_term = observable = 0.0
        for event, dist in prediction.events.items():
            weight = weights.get(event)
            if weight is None:
                continue
            pred_count = float(dist.mean)
            available = bool(row.get(f"available__{event}", False))
            act_count = float(pd.to_numeric(row.get(event), errors="coerce") or 0.0) if available else np.nan
            expected += weight * pred_count
            if np.isnan(act_count):
                continue
            observable += weight * act_count
            if act_min > 0 and pred_min > 0:
                pred_rate = 80.0 * pred_count / pred_min
                act_rate = 80.0 * act_count / act_min
                minutes_term += weight * (pred_min - act_min) / 80.0 * pred_rate
                rate_term += weight * act_min / 80.0 * (pred_rate - act_rate)
        played = act_min > 0
        records.append({
            "pred_minutes": pred_min, "act_minutes": act_min, "played": played,
            "expected_points": expected, "observable_points": observable,
            "minutes_term": minutes_term if played else 0.0,
            "rate_term": rate_term if played else 0.0,
            "appearance_term": (expected - observable) if not played else 0.0,
        })
    return pd.DataFrame(records, index=rows.index)


def decompose_ncr(store: pd.DataFrame) -> pd.DataFrame:
    labels = build_fantasy_labels()
    ncr = labels[labels["competition"].eq("ncr")]
    parts = []
    for round_no, block in ncr.groupby("round", sort=True):
        artifact = BENCH / "artifacts" / "baseline" / f"ncr_2026_r{int(round_no)}.pkl"
        model = UniversalGBDT.load(artifact)
        joined = match_labels_to_store(block.reset_index(drop=True), store)
        matched = joined[joined["store_matched"]].copy()
        terms = _expected_points_terms(model, matched, "ncr")
        matched = pd.concat([matched, terms], axis=1)
        matched["gw"] = int(round_no)
        parts.append(matched)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["official_error"] = out["expected_points"] - out["official_pts"]
    out["rubric_term"] = out["observable_points"] - out["official_pts"]
    out["model_term"] = out["expected_points"] - out["observable_points"]
    out["thin_intl"] = pd.to_numeric(out["intl_prior_matches"], errors="coerce") < THIN_INTL_MATCHES
    return out


def _share_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution of each term, overall and by segment."""
    def agg(block: pd.DataFrame, name: str) -> dict:
        model_abs = block["model_term"].abs()
        return {
            "segment": name, "n": len(block),
            "official_mae": block["official_error"].abs().mean(),
            "model_mae_vs_observable": model_abs.mean(),
            "rubric_gap_mae": block["rubric_term"].abs().mean(),
            "appearance_share": (block["appearance_term"].abs().sum()
                                 / max(model_abs.sum(), 1e-9)),
            "minutes_share": (block["minutes_term"].abs().sum()
                              / max(model_abs.sum(), 1e-9)),
            "rate_share": (block["rate_term"].abs().sum()
                           / max(model_abs.sum(), 1e-9)),
        }
    rows = [agg(frame, "all")]
    for gw, block in frame.groupby("gw"):
        rows.append(agg(block, f"gw{gw}"))
    for flag, name in ((True, "thin_intl(<5)"), (False, "rich_intl(>=5)")):
        rows.append(agg(frame[frame["thin_intl"].eq(flag)], name))
    for fwd, name in ((True, "forwards"), (False, "backs")):
        rows.append(agg(frame[frame["is_forward"].eq(fwd)], name))
    rows.append(agg(frame[frame["played"]], "played_only"))
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def verify_ledger_23(store: pd.DataFrame) -> dict:
    """Re-verify the diagnostician's unreproduced numbers (ledger #23)."""
    predictions = pd.read_csv(BENCH / "predictions.csv")
    ncr = predictions[predictions["competition"].eq("ncr")
                      & predictions["engine"].eq("baseline")]
    capture = {}
    for round_no, block in ncr.groupby("round"):
        block = block.reset_index(drop=True)
        base = _group_metrics(block.rename(columns={"official_pts": "actual"}),
                              "predicted_points", actual_col="actual")
        inc = _group_metrics(block.dropna(subset=["incumbent_points"])
                             .rename(columns={"official_pts": "actual"})
                             .reset_index(drop=True),
                             "incumbent_points", actual_col="actual")
        capture[f"gw{round_no}"] = {
            "baseline_top10": round(base["top_10_capture"], 4),
            "incumbent_top10": round(inc["top_10_capture"], 4),
        }
    labels = build_fantasy_labels()
    joined = match_labels_to_store(
        labels[labels["competition"].eq("ncr")].reset_index(drop=True), store)
    matched = joined[joined["store_matched"]]
    intl = pd.to_numeric(matched["intl_prior_matches"], errors="coerce").dropna()
    history = {
        "median_intl_prior": float(intl.median()),
        "share_le_3": float((intl <= 3).mean()),
        "share_lt_5": float((intl < THIN_INTL_MATCHES).mean()),
        "n_thin_lt_5": int((intl < THIN_INTL_MATCHES).sum()),
    }
    return {"per_gw_top10_capture": capture, "ncr_cohort_intl_history": history}


def main() -> None:
    store = load_store()
    decomposition = decompose_ncr(store)
    shares = _share_table(decomposition)
    ledger = verify_ledger_23(store)
    OUT.mkdir(parents=True, exist_ok=True)
    decomposition_cols = [
        "gw", "player_id", "key_player", "player_name_label", "position_label",
        "is_forward", "official_pts",
        "expected_points", "observable_points", "pred_minutes", "act_minutes",
        "played", "thin_intl", "intl_prior_matches", "official_error",
        "model_term", "rubric_term", "appearance_term", "minutes_term", "rate_term",
    ]
    keep = [c for c in decomposition_cols if c in decomposition]
    decomposition[keep].to_csv(OUT / "p0_error_decomposition.csv", index=False)
    shares.to_csv(OUT / "p0_error_shares.csv", index=False)
    (OUT / "p0_ledger23_verification.json").write_text(
        json.dumps(ledger, indent=2) + "\n")
    played = decomposition[decomposition["played"]]
    dominant = shares.loc[shares["segment"].eq("all")].iloc[0]
    verdict = ("minutes-dominated (S1 STOP)" if dominant["minutes_share"] > 0.5
               else "rate-dominated (proceed)")
    lines = [
        "# P0.1 — NCR error decomposition (frozen baseline fold artifacts)", "",
        f"Rows: {len(decomposition)} matched NCR label rows (GW1+GW2); "
        f"played: {len(played)}.",
        f"Thin-intl (<{THIN_INTL_MATCHES} prior intl store rows): "
        f"{int(decomposition['thin_intl'].sum())} rows.", "",
        "Signed identity per played row: expected - observable = minutes_term + "
        "rate_term; DNP rows collapse to appearance_term; "
        "official error additionally carries the rubric_term "
        "(observable - official, the unobserved-events gap).", "",
        "## Mean-absolute shares of model error (vs observable points)", "",
        _markdown_table(shares), "",
        f"## Verdict: **{verdict}**", "",
        "## Ledger #23 re-verification", "",
        "```json", json.dumps(ledger, indent=2), "```", "",
    ]
    (OUT / "p0_error_decomposition.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
