"""Final v4 report: admitted config, selection evidence, display-only diagnostics.

Run ONLY after the P1/P2 admission verdicts are frozen (p1_verdicts_A_B.json,
p2_verdict.json). The quarantined sets (6N 2026, NCR GW1-2) are scored here for
DISPLAY ONLY — no decision may be revised on their numbers (RESEARCH_PLAN.md S3);
they exist so the report can show the directional NCR read next to the frozen
v3 benchmark rows. Promotion evidence remains the prospective GW4-7 shadows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import ROOT
from ..labels import build_fantasy_labels
from ..v3.benchmark import _cohort_predictions
from ..v3.cohorts import match_labels_to_store
from ..v3.harness import fold_for_block, strict_training_frame
from .experiments import HURDLE_EVENTS, _fold_metrics, build_store
from .gbdt import V4GBDT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
BENCH = DATA / "unified" / "v3" / "benchmark"
BOOTSTRAP = 2000
SEED = 17


def admitted_config() -> str:
    p1 = json.loads((OUT / "p1_verdicts_A_B.json").read_text())
    p2 = json.loads((OUT / "p2_verdict.json").read_text())

    def admitted(v: dict) -> bool:
        return v["mae_holds"] and v["spearman_improves"] and v["capture_improves"]

    winner = None
    if admitted(p1["B"]):
        winner = "B"
    if admitted(p1["A"]) and (winner is None or p1["A"]["spearman_improves"] > p1["B"]["spearman_improves"]):
        winner = winner or "A"
    if winner is None:
        raise SystemExit("P1 admitted nothing; v4 = P0 only (stop rule S2)")
    return f"{winner}+hurdle" if p2["admit"] else winner


def _build(config: str) -> V4GBDT:
    kwargs = dict(weighting="natural")
    if config.startswith("B"):
        kwargs.update(pool_player_id=True, player_effects=True)
    if config.endswith("+hurdle"):
        kwargs.update(hurdle_events=HURDLE_EVENTS)
    return V4GBDT(**kwargs)


def diagnostics(config: str) -> pd.DataFrame:
    """Fit per quarantined leg fold and score the same parity cohorts."""
    store = build_store()
    labels = build_fantasy_labels()
    legs = pd.concat([
        labels[labels["competition"].eq("six_nations") & labels["season"].eq(2026)],
        labels[labels["competition"].eq("ncr")],
    ])
    rows = []
    for group_id, block in legs.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        competition = str(block["competition"].iloc[0])
        matched = match_labels_to_store(block, store)
        matched = matched[matched["store_matched"]]
        cached = OUT / f"diag_{group_id}.csv"
        if cached.exists():
            print(f"[{group_id}] reusing {cached.name}", flush=True)
            predicted = pd.read_csv(cached)
        else:
            fold = fold_for_block(block, store)
            train = strict_training_frame(store, fold)
            model = _build(config)
            print(f"[{group_id}] fitting gbdt_v4[{config}] on {len(train):,} rows", flush=True)
            model.fit(train)
            predicted = _cohort_predictions(model, block, store, competition)
        rows.append({"competition": competition, "group_id": group_id,
                     "engine": "gbdt_v4", **_fold_metrics(predicted, matched)})
        predicted.assign(competition=competition, engine="gbdt_v4").to_csv(
            OUT / f"diag_{group_id}.csv", index=False)
    return pd.DataFrame(rows)


def frozen_reference() -> pd.DataFrame:
    """Frozen v3 benchmark fold metrics for baseline / gbdt_v3 / incumbent."""
    metrics = pd.read_csv(BENCH / "fold_metrics.csv")
    reference = metrics[metrics["phase"].eq("retrospective_reference")].copy()
    return reference.rename(columns={"model": "engine"})


def paired_bootstrap(config: str) -> dict:
    """v4 minus incumbent on pooled NCR GW1-2 (display-only diagnostic)."""
    predictions = pd.read_csv(BENCH / "predictions.csv")
    # Incumbent projections are carried as the incumbent_points column on the
    # frozen benchmark rows (there is no engine=="incumbent" row set).
    incumbent = predictions[predictions["engine"].eq("baseline")
                            & predictions["competition"].eq("ncr")
                            & predictions["incumbent_points"].notna()]
    frames = []
    for group_id in sorted(incumbent["group_id"].unique()):
        v4 = pd.read_csv(OUT / f"diag_{group_id}.csv")
        merged = v4.rename(columns={"predicted_points": "predicted_points_v4"}).merge(
            incumbent[incumbent["group_id"].eq(group_id)]
            [["_label_row_id", "incumbent_points"]],
            on="_label_row_id").rename(columns={"incumbent_points": "predicted_points_inc"})
        merged["group_id"] = group_id
        frames.append(merged)
    pooled = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(SEED)

    def capture(frame: pd.DataFrame, col: str, n: int = 10) -> float:
        top_true = frame.nlargest(n, "official_pts")["official_pts"].sum()
        top_pred = frame.nlargest(n, col)["official_pts"].sum()
        return top_pred / max(top_true, 1e-9)

    mae_diffs, cap_diffs = [], []
    for _ in range(BOOTSTRAP):
        sample = pooled.groupby("group_id", group_keys=False).apply(
            lambda g: g.sample(len(g), replace=True, random_state=rng.integers(2**31)))
        mae_v4 = (sample["predicted_points_v4"] - sample["official_pts"]).abs().mean()
        mae_inc = (sample["predicted_points_inc"] - sample["official_pts"]).abs().mean()
        mae_diffs.append(mae_v4 - mae_inc)
        caps = sample.groupby("group_id").apply(
            lambda g: capture(g, "predicted_points_v4") - capture(g, "predicted_points_inc"))
        cap_diffs.append(float(caps.mean()))
    return {
        "n_rows": int(len(pooled)),
        "mae_diff_mean": float(np.mean(mae_diffs)),
        "mae_diff_p05": float(np.percentile(mae_diffs, 5)),
        "mae_diff_p95": float(np.percentile(mae_diffs, 95)),
        "top10_capture_diff_mean": float(np.mean(cap_diffs)),
        "top10_capture_diff_p05": float(np.percentile(cap_diffs, 5)),
        "top10_capture_diff_p95": float(np.percentile(cap_diffs, 95)),
    }


def render(config: str, diag: pd.DataFrame, boot: dict) -> str:
    p1 = pd.read_csv(OUT / "p1_summary_A_B.csv")
    p2 = json.loads((OUT / "p2_verdict.json").read_text())
    reference = frozen_reference()
    cuts = ["mae", "spearman", "top_10_capture", "top_25_capture",
            "top_50_capture", "top_100_capture"]

    def table(frame: pd.DataFrame, by: str) -> str:
        agg = frame.groupby(by, as_index=False)[cuts].mean(numeric_only=True)
        lines = ["| Model | MAE | Spearman | Top10 | Top25 | Top50 | Top100 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for r in agg.itertuples(index=False):
            lines.append(
                f"| {getattr(r, by)} | {r.mae:.2f} | {r.spearman:.3f} | "
                f"{r.top_10_capture:.1%} | {r.top_25_capture:.1%} | "
                f"{r.top_50_capture:.1%} | {r.top_100_capture:.1%} |")
        return "\n".join(lines)

    diag = diag.assign(engine="gbdt_v4 (display-only)")
    combined = {}
    for competition in ("six_nations", "ncr"):
        ref = reference[reference["competition"].eq(competition)][["engine", *cuts]]
        v4 = diag[diag["competition"].eq(competition)][["engine", *cuts]]
        combined[competition] = pd.concat([v4, ref], ignore_index=True)

    return "\n".join([
        "# Unified supermodel v4 — final report",
        "",
        f"Admitted configuration: **gbdt_v4[{config}]** "
        "(P1 variant B: pooled player id + post-hoc EB player effects + level-split form"
        + ("; P2 hurdle heads admitted)" if config.endswith("+hurdle")
           else "; P2 hurdle heads NOT admitted)"),
        "",
        "Admission decisions (3/3 used, 6N-2025 LORO only — see p1_verdicts_A_B.json,",
        "p2_verdict.json). Quarantined sets below are DISPLAY ONLY (plan §6/S3): no",
        "decision was or may be revised on them. Promotion evidence = prospective NCR",
        "GW4–7 shadows, pooled, burn-once (tier-1 superiority / tier-2 deferred).",
        "",
        "## 6N 2025 selection layer (admission evidence)",
        "", table(p1.rename(columns={"config": "engine"}), "engine"), "",
        f"P2 tail diagnostics: Brier {p2['brier_hurdle']:.4f} (hurdle) vs "
        f"{p2['brier_plain']:.4f} (plain NB); |p90 coverage − 0.90| "
        f"{p2['p90_gap_hurdle']:.3f} vs {p2['p90_gap_plain']:.3f}; "
        f"mean capture {p2['capture_hurdle']:.3f} vs B {p2['capture_B']:.3f} → "
        f"admit={p2['admit']}.",
        "",
        "## Six Nations 2026 (quarantined — display only)",
        "", table(combined["six_nations"], "engine"), "",
        "## NCR GW1–2 (quarantined — display only)",
        "", table(combined["ncr"], "engine"), "",
        "### Paired bootstrap, v4 − incumbent, pooled NCR (display only)",
        "",
        f"- MAE diff {boot['mae_diff_mean']:+.2f} "
        f"(90% CI {boot['mae_diff_p05']:+.2f} … {boot['mae_diff_p95']:+.2f}), "
        f"n={boot['n_rows']}",
        f"- Top-10 capture diff {boot['top10_capture_diff_mean']:+.1%} "
        f"(90% CI {boot['top10_capture_diff_p05']:+.1%} … "
        f"{boot['top10_capture_diff_p95']:+.1%})",
        "",
        "## November runbook (GW4–7)",
        "",
        "```bash",
        "# before each GW lock (repeat --target-gw 4..7):",
        "/tmp/6n-model-pinned/bin/python -m model.unified.v4.assemble fit \\",
        f"    --config '{config}' --cutoff <GW_LOCK_UTC> \\",
        "    --output data/unified/v4/models/gbdt_v4_ncr_gw<N>.pkl",
        "/tmp/6n-model-pinned/bin/python -m model.unified.v4.assemble activate \\",
        "    --model data/unified/v4/models/gbdt_v4_ncr_gw<N>.pkl --target-gw <N>",
        "# gw_update.sh step 6 then freezes the write-once shadow automatically.",
        "```",
        "",
        "After GW7 labels: single pooled look. Tier-1 (2026 promotion) requires the",
        "paired-bootstrap 90% CI to exclude zero on MAE AND mean capture vs the NCR",
        "incumbent while holding 6N — otherwise incumbents stay and the tier-2",
        "non-inferiority clock continues into 2027 (plan §6).",
    ])


def main() -> None:
    config = admitted_config()
    print(f"admitted config: gbdt_v4[{config}]", flush=True)
    diag = diagnostics(config)
    diag.to_csv(OUT / "final_diagnostics.csv", index=False)
    boot = paired_bootstrap(config)
    (OUT / "final_bootstrap.json").write_text(json.dumps(boot, indent=2) + "\n")
    report = render(config, diag, boot)
    (OUT / "REPORT.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
