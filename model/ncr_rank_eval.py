#!/usr/bin/env python3
"""Compare NCR empirical and transplanted champion player rankings.

All comparisons use the same prediction cohort for both models and official
fantasy points as the outcome. Top-N overlap is tie-aware: when the Nth actual
score is tied, each member of that tie receives the fraction of the remaining
places available at the cutoff.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ncr"
CUTOFFS = (10, 25, 50, 100)
ROUNDS = {
    1: {
        "ncr": "ncr_gw1_projections.csv",
        "champion": "ncr_gw1_projections_champion_replay.csv",
        "actual": "feeds/players_post_gw1.json",
        "note": "Champion prediction is a point-in-time replay; its original GW1 CSV was not retained.",
    },
    2: {
        "ncr": "ncr_gw2_projections.csv",
        "champion": "ncr_gw2_projections_champion.csv",
        "actual": "feeds/players_gw2.json",
        "note": "Both prediction CSVs are the saved pre-match artifacts.",
    },
}


def load_actuals(path: Path) -> pd.DataFrame:
    players = json.loads(path.read_text())["Data"]["Value"]["Players"]
    return pd.DataFrame([
        {
            "id": int(float(player["id"])),
            "actual_name": player.get("full_name") or player.get("display_name"),
            "actual": float(player.get("cur_gd_points") or 0),
            "actual_status": player.get("player_status") or "",
        }
        for player in players
    ])


def tie_aware_hits(predicted_ids: set[int], actual: pd.DataFrame, n: int) -> tuple[float, float]:
    """Return expected hits and the actual-points cutoff for a tied top N."""
    cutoff = float(actual["actual"].nlargest(n).iloc[-1])
    strict = set(actual.loc[actual["actual"] > cutoff, "id"].astype(int))
    tied = set(actual.loc[actual["actual"] == cutoff, "id"].astype(int))
    remaining = n - len(strict)
    tie_weight = remaining / len(tied)
    hits = len(predicted_ids & strict) + tie_weight * len(predicted_ids & tied)
    return float(hits), cutoff


def evaluate_round(gw: int, files: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ncr = (pd.read_csv(DATA / files["ncr"])[["id", "name", "starter_exp"]]
           .rename(columns={"starter_exp": "NCR"}))
    champion = (pd.read_csv(DATA / files["champion"])[["id", "starter_exp"]]
                .rename(columns={"starter_exp": "Champion"}))
    actual = load_actuals(DATA / files["actual"])
    cohort = ncr.merge(champion, on="id", validate="one_to_one").merge(
        actual[["id", "actual"]], on="id", validate="one_to_one"
    )
    if cohort["id"].duplicated().any():
        raise AssertionError(f"GW{gw}: duplicate player IDs after join")

    quality = {
        "gw": gw,
        "ncr_players": len(ncr),
        "champion_players": len(champion),
        "common_players": len(cohort),
        "official_players": len(actual),
        "positive_actual_missing": int(
            ((actual["actual"] != 0) & ~actual["id"].isin(cohort["id"])).sum()
        ),
    }
    rows = []
    oracle_ranked = actual.sort_values(["actual", "id"], ascending=[False, True])
    for model in ("NCR", "Champion"):
        rho = float(spearmanr(cohort[model], cohort["actual"]).statistic)
        for n in CUTOFFS:
            predicted_ids = set(cohort.nlargest(n, [model, "id"])["id"].astype(int))
            hits, cutoff = tie_aware_hits(predicted_ids, actual, n)
            oracle = oracle_ranked.head(n)
            captured = float(actual.loc[actual["id"].isin(predicted_ids), "actual"].sum())
            oracle_points = float(oracle["actual"].sum())
            missing = oracle.loc[~oracle["id"].isin(cohort["id"])]
            rows.append({
                "gw": gw,
                "model": model,
                "n": n,
                "tie_adjusted_hits": hits,
                "overlap": hits / n,
                "points_captured": captured / oracle_points,
                "actual_cutoff_points": cutoff,
                "spearman_rho": rho,
                "actual_top_n_missing_from_both_models": len(missing),
            })
    missing_top = oracle_ranked.head(100).loc[
        ~oracle_ranked.head(100)["id"].isin(cohort["id"]),
        ["id", "actual_name", "actual", "actual_status"],
    ].copy()
    missing_top.insert(0, "gw", gw)
    return pd.DataFrame(rows), quality, missing_top


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render(results: pd.DataFrame, quality: pd.DataFrame, missing: pd.DataFrame) -> str:
    mean = (results.groupby(["model", "n"], as_index=False)
            .agg(overlap=("overlap", "mean"), points_captured=("points_captured", "mean")))
    lines = [
        "# NCR empirical vs 6N champion — ranking evaluation",
        "",
        "Official fantasy points are the outcome. Predictions are compared as rankings, so the "
        "models' different point scales do not matter.",
        "",
        "## Two-round average",
        "",
        "| Top N | NCR overlap | Champion overlap | NCR point capture | Champion point capture |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in CUTOFFS:
        a = mean[mean["n"].eq(n)].set_index("model")
        lines.append(
            f"| {n} | **{pct(a.loc['NCR', 'overlap'])}** | "
            f"{pct(a.loc['Champion', 'overlap'])} | "
            f"**{pct(a.loc['NCR', 'points_captured'])}** | "
            f"{pct(a.loc['Champion', 'points_captured'])} |"
        )

    lines.extend(["", "## Round-by-round", ""])
    for gw in sorted(ROUNDS):
        sub = results[results["gw"].eq(gw)]
        rho = sub.drop_duplicates("model").set_index("model")["spearman_rho"]
        lines.extend([
            f"### GW{gw}",
            "",
            f"Spearman rank correlation: **NCR {rho['NCR']:.3f}**, "
            f"champion {rho['Champion']:.3f}.",
            "",
            "| Top N | NCR hits | Champion hits | NCR point capture | Champion point capture |",
            "|---:|---:|---:|---:|---:|",
        ])
        for n in CUTOFFS:
            a = sub[sub["n"].eq(n)].set_index("model")
            lines.append(
                f"| {n} | **{a.loc['NCR', 'tie_adjusted_hits']:.1f}** | "
                f"{a.loc['Champion', 'tie_adjusted_hits']:.1f} | "
                f"**{pct(a.loc['NCR', 'points_captured'])}** | "
                f"{pct(a.loc['Champion', 'points_captured'])} |"
            )
        lines.extend(["", f"_{ROUNDS[gw]['note']}_", ""])

    lines.extend([
        "## Data quality and interpretation",
        "",
        "| GW | NCR predictions | Champion predictions | Common cohort | Official pool | "
        "Non-zero scorers missing from both |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in quality.itertuples(index=False):
        lines.append(
            f"| {row.gw} | {row.ncr_players} | {row.champion_players} | "
            f"{row.common_players} | {row.official_players} | {row.positive_actual_missing} |"
        )
    lines.extend([
        "",
        "The headline overlap uses the full official leaderboard. Players absent from both "
        "prediction pools therefore count as misses. Spearman correlation uses only the common "
        "prediction cohort, isolating model ranking quality from shared lineup-input omissions.",
        "",
        "Top-N hits are tie-adjusted at the actual cutoff. Point capture is the official points "
        "earned by a model's predicted top N divided by the points earned by the hindsight top N.",
        "",
        "Only two rounds are available, so this can compare current ranking quality but cannot "
        "establish a learning curve or prove how the NCR model will improve with more rounds.",
    ])
    if len(missing):
        lines.extend([
            "",
            "### Official top-100 players absent from both prediction pools",
            "",
            "| GW | Player | Actual points | Status |",
            "|---:|---|---:|---|",
        ])
        for row in missing.itertuples(index=False):
            lines.append(f"| {row.gw} | {row.actual_name} | {row.actual:g} | {row.actual_status} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    frames, quality, missing = [], [], []
    for gw, files in ROUNDS.items():
        result, q, absent = evaluate_round(gw, files)
        frames.append(result)
        quality.append(q)
        missing.append(absent)
    results = pd.concat(frames, ignore_index=True)
    quality_df = pd.DataFrame(quality)
    missing_df = pd.concat(missing, ignore_index=True)
    results.to_csv(DATA / "ncr_rank_evaluation.csv", index=False)
    missing_df.to_csv(DATA / "ncr_rank_missing_players.csv", index=False)
    report = render(results, quality_df, missing_df)
    (DATA / "ncr_rank_evaluation.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
