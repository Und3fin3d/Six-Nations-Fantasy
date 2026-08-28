"""Honest cross-competition benchmark for the stage-2 ranking stack (v2).

Protocol (see the approved plan): every reported number is out-of-sample for the
learned stack.

* Six Nations: stage-1 and stage-2 never see 2026.  Stage-1 cut at 2026-02-01,
  stage-2 trained on 2025, evaluated on the sealed 2026 season.
* NCR (leave-one-gameweek-out): GW1 predicted by a stack trained on Six Nations
  labels only (pure cross-competition transfer); GW2 predicted by a stack
  trained on Six Nations + NCR GW1.  Stage-1 is re-cut per gameweek.

Cohort parity is guaranteed by construction: NCR labels are the incumbent
projection cohort, so the stack scores exactly the rows the incumbent does,
with position priors for players the crosswalk cannot place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import ROOT
from .evaluation import tie_aware_top_n
from .gbdt import UniversalGBDT
from .labels import build_fantasy_labels
from .rank_stack import RankStack, build_stage2_features, load_store_features

DATA = ROOT / "data"
OUT = DATA / "unified"
MODELS = OUT / "models"
CUTOFFS = (10, 25, 50, 100)
STAGE1 = {
    "six_nations": MODELS / "unified_gbdt_2026-02-01.pkl",
    "ncr_gw1": MODELS / "unified_gbdt_2026-07-04.pkl",
    "ncr_gw2": MODELS / "unified_gbdt_2026-07-11.pkl",
}


def _group_metrics(cohort: pd.DataFrame, rank_col: str, mae_col: str | None = None,
                   actual_col: str = "official_pts") -> dict:
    """Ranking + error metrics within one round, tie-aware.

    ``rank_col`` orders players (ranking metrics); ``mae_col`` (defaults to
    ``rank_col``) supplies the points forecast for MAE.
    """
    mae_col = mae_col or rank_col
    rank = cohort[rank_col].reset_index(drop=True)
    points = cohort[mae_col].reset_index(drop=True)
    actual = cohort[actual_col].reset_index(drop=True)
    row = {"n": len(cohort),
           "mae": float(np.mean(np.abs(points - actual))),
           "spearman": float(spearmanr(rank, actual).statistic) if len(cohort) > 2 else np.nan}
    for n in CUTOFFS:
        if len(cohort) >= n:
            overlap, capture = tie_aware_top_n(rank, actual, n)
            row[f"top_{n}_overlap"] = overlap
            row[f"top_{n}_capture"] = capture
    return row


def _average(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes("number")
    out = {c: float(numeric[c].mean()) for c in numeric.columns if c != "n"}
    if "n" in numeric.columns:
        out["n"] = int(numeric["n"].sum())
    return out


# --- Six Nations ---------------------------------------------------------------

def six_nations() -> pd.DataFrame:
    stage1 = UniversalGBDT.load(STAGE1["six_nations"])
    store = load_store_features()
    labels = build_fantasy_labels()
    six = labels[labels.competition == "six_nations"]
    train = six[six.season == 2025]
    test = six[six.season == 2026]
    feat_train = build_stage2_features(train, store, stage1)
    feat_test = build_stage2_features(test, store, stage1)
    stack = RankStack().fit(feat_train)
    pred = stack.predict(feat_test)

    incumbent = pd.read_csv(DATA / "model_predictions_2026.csv")
    incumbent["key"] = (incumbent.fixture_id.astype(str) + "|" + incumbent.player_id.astype(str))
    pred = pred.copy()
    pred["key"] = pred.key_fixture.astype(str) + "|" + pred.key_player.astype(str)
    pred = pred.merge(incumbent[["key", "target_pts_hat"]], on="key", how="left")

    rows = []
    for gid, block in pred.groupby("group_id"):
        block = block.reset_index(drop=True)
        rows.append({"model": "Unified v2 stack", "group": gid,
                     **_group_metrics(block, "stack_score", mae_col="expected_points")})
        rows.append({"model": "Unified v1 stage-1", "group": gid,
                     **_group_metrics(block, "s1_exp_points")})
        inc = block.dropna(subset=["target_pts_hat"])
        if len(inc) > 2:
            rows.append({"model": "6N champion", "group": gid,
                         **_group_metrics(inc, "target_pts_hat")})
    return pd.DataFrame(rows)


# --- NCR (leave-one-gameweek-out) ---------------------------------------------

def ncr() -> pd.DataFrame:
    labels = build_fantasy_labels()
    six = labels[labels.competition == "six_nations"]
    six_train = six[six.season == 2025]
    rows = []
    prior_ncr = []  # accumulate earlier NCR gameweeks as training rows
    for gw in (1, 2):
        stage1 = UniversalGBDT.load(STAGE1[f"ncr_gw{gw}"])
        store = load_store_features()
        train_labels = pd.concat([six_train, *prior_ncr], ignore_index=True)
        feat_train = build_stage2_features(train_labels, store, stage1)
        gw_labels = labels[(labels.competition == "ncr") & (labels["round"] == gw)]
        feat_test = build_stage2_features(gw_labels, store, stage1)
        stack = RankStack().fit(feat_train)
        pred = stack.predict(feat_test).reset_index(drop=True)

        proj = pd.read_csv(DATA / f"ncr/ncr_gw{gw}_projections.csv")[["id", "starter_exp"]]
        proj["key_player"] = proj.id.astype(int).astype(str)
        pred = pred.merge(proj[["key_player", "starter_exp"]], on="key_player", how="left")

        rows.append({"model": "Unified v2 stack", "group": f"gw{gw}",
                     **_group_metrics(pred, "stack_score", mae_col="expected_points")})
        rows.append({"model": "Unified v1 stage-1", "group": f"gw{gw}",
                     **_group_metrics(pred, "s1_exp_points")})
        inc = pred.dropna(subset=["starter_exp"])
        rows.append({"model": "NCR incumbent", "group": f"gw{gw}",
                     **_group_metrics(inc, "starter_exp")})
        rows.append({"model": "_coverage", "group": f"gw{gw}", "n": len(pred),
                     "matched": float(pred["s1_matched"].mean())})
        prior_ncr.append(gw_labels)
    return pd.DataFrame(rows)


# --- report --------------------------------------------------------------------

def _table(df: pd.DataFrame, models: list[str]) -> str:
    metric_cols = ["mae", "spearman", "top_10_capture", "top_25_capture",
                   "top_50_capture", "top_100_capture"]
    header = "| Model | " + " | ".join(
        ["MAE", "Spearman", "Top10 cap", "Top25 cap", "Top50 cap", "Top100 cap"]) + " |"
    sep = "|---|" + "---:|" * 6
    lines = [header, sep]
    for model in models:
        block = df[df.model == model]
        if block.empty:
            continue
        agg = _average(block.to_dict("records"))
        cells = []
        for c in metric_cols:
            v = agg.get(c, np.nan)
            cells.append("n/a" if pd.isna(v) else (f"{v:.2f}" if c == "mae" else f"{v:.3f}"))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gate_verdict(df: pd.DataFrame, candidate: str, incumbent: str) -> tuple[bool, list[str]]:
    cand = _average(df[df.model == candidate].to_dict("records"))
    inc = _average(df[df.model == incumbent].to_dict("records"))
    reasons = []
    if cand["mae"] > inc["mae"] * 1.02:
        reasons.append(f"MAE {cand['mae']:.2f} regressed >2% vs {inc['mae']:.2f}")
    gains = []
    for k in ("top_10_capture", "top_25_capture", "top_50_capture", "top_100_capture"):
        if k in cand and k in inc:
            if cand[k] < inc[k] - 0.02:
                reasons.append(f"{k} {cand[k]:.3f} regressed >2pp vs {inc[k]:.3f}")
            gains.append(cand[k] - inc[k])
    if not gains or np.mean(gains) < 0.03:
        reasons.append(f"mean capture gain {np.mean(gains) if gains else 0:+.3f} below +3pp")
    return (not reasons), reasons


def _round_wins(df: pd.DataFrame, a: str, b: str, metric: str, lower_is_better: bool) -> str:
    """How many rounds model ``a`` beats model ``b`` on ``metric``."""
    pa = df[df.model == a].set_index("group")[metric]
    pb = df[df.model == b].set_index("group")[metric]
    common = pa.index.intersection(pb.index)
    if lower_is_better:
        wins = int((pa[common] < pb[common]).sum())
    else:
        wins = int((pa[common] > pb[common]).sum())
    return f"{wins}/{len(common)}"


def render(six: pd.DataFrame, ncr_df: pd.DataFrame) -> str:
    ncr_cov = ncr_df[ncr_df.model == "_coverage"]
    ncr_metrics = ncr_df[ncr_df.model != "_coverage"]
    six_ok, six_reasons = gate_verdict(six, "Unified v2 stack", "6N champion")
    ncr_ok, ncr_reasons = gate_verdict(ncr_metrics, "Unified v2 stack", "NCR incumbent")
    lines = [
        "# Unified supermodel v2 — ranking-first stacked benchmark",
        "",
        "One stage-1 rugby-event model feeds one pooled stage-2 ranker+regressor "
        "conditioned on each competition's scoring rubric (no competition identity "
        "feature). Every number below is out-of-sample for the learned stack; "
        "metrics are computed within each round and averaged. See "
        "`model/unified/benchmark_v2.py` for the protocol.",
        "",
        "## Six Nations (2026 sealed holdout)",
        "",
        _table(six, ["6N champion", "Unified v1 stage-1", "Unified v2 stack"]),
        "",
        "## Nations Championship (leave-one-gameweek-out)",
        "",
        _table(ncr_metrics, ["NCR incumbent", "Unified v1 stage-1", "Unified v2 stack"]),
        "",
        "Cohort parity is by construction (incumbent projection cohort). "
        "Stage-1 feature match rate per gameweek: "
        + ", ".join(f"{r.group} {r.matched:.0%} of {int(r.n)}" for r in ncr_cov.itertuples()) + ".",
        "",
        "## Promotion gate",
        "",
        f"- Six Nations: {'PASS' if six_ok else 'FAIL'}"
        + ("" if six_ok else " — " + "; ".join(six_reasons)),
        f"- NCR: {'PASS' if ncr_ok else 'FAIL'}"
        + ("" if ncr_ok else " — " + "; ".join(ncr_reasons)),
        "",
        f"**Decision: {'PROMOTE' if (six_ok and ncr_ok) else 'DO NOT PROMOTE'}.** "
        "Production routing is unchanged regardless; promotion also requires the "
        "prospective November shadow rounds (GW4–7).",
        "",
        "## Caveats (read before quoting these numbers)",
        "",
        "- The 6N headline is a small-sample edge, not a proven win. Across the 5 "
        f"sealed 2026 rounds, stage-1 beats the 6N champion on MAE in "
        f"{_round_wins(six, 'Unified v1 stage-1', '6N champion', 'mae', True)} rounds and on "
        f"Spearman in {_round_wins(six, 'Unified v1 stage-1', '6N champion', 'spearman', False)}; "
        "no significance test is applied. Read it as \"competitive with, plausibly better than\" "
        "the champion, contingent on `data/model_predictions_2026.csv` being a genuine pre-round forecast.",
        "- The stack-vs-stage-1 comparison is confounded. Stage-2 trains on stage-1 "
        "features that were fit *including* those same training rows (non-OOF), and on only "
        "~674 6N labelled rows (2024 has no fantasy points; the 2023 rubric era is excluded). "
        "\"Learned stack does not beat deterministic stage-1\" is therefore a working hypothesis, "
        "not a clean test of the v2 thesis — out-of-fold stage-1 features and hyperparameter "
        "tuning are the outstanding work.",
        "- Stage-1 artifacts train only through the first 85% of pre-cutoff dates "
        "(the CLI reserves, then discards, a validation tail), so they saw *less* recent data "
        "than the `asof` label implies — a conservative bias for the stage-1 result.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    six = six_nations()
    ncr_df = ncr()
    OUT.mkdir(parents=True, exist_ok=True)
    six.to_csv(OUT / "benchmark_v2_six_nations.csv", index=False)
    ncr_df.to_csv(OUT / "benchmark_v2_ncr.csv", index=False)
    report = render(six, ncr_df)
    (OUT / "benchmark_v2.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
