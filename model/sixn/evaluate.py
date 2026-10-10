from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

XV_QUOTA = {"Prop": 2, "Hooker": 1, "Second-row": 2, "Back-row": 3, "Scrum-half": 1, "Fly-half": 1, "Centre": 2, "Back-three": 3}
SUPERSUB_MULTIPLIER = 3.0
COLUMNS = ["mae", "bench_mae", "value_team", "value_xv", "top15", "top30", "capt_top1", "capt_top3", "capt_top5", "spearman_pos"]


def _labelled(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)].copy()


def _pick_xv(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    picked = [gp.nlargest(min(k, len(gp)), score_col) for pos, k in XV_QUOTA.items() if not (gp := df[df["canonical_pos"] == pos]).empty]
    return pd.concat(picked) if picked else df.iloc[0:0].copy()


def _team_value(g: pd.DataFrame, score_col: str) -> float:
    xv = _pick_xv(g, score_col)
    pool = g[~g["player_id"].isin(set(xv["player_id"]))]
    return (
        float(xv["official_pts"].sum())
        + float(xv.loc[xv[score_col].idxmax(), "official_pts"])
        + SUPERSUB_MULTIPLIER * (float(pool.loc[pool[score_col].idxmax(), "official_pts"]) if not pool.empty else 0.0)
    )


def evaluate(pred: pd.DataFrame, score_col: str = "target_pts_hat", *, points: bool = True) -> dict:
    lab = _labelled(pred)
    rounds = [g for _, g in lab.groupby("round")]
    xv = [
        float(_pick_xv(g, score_col)["official_pts"].sum()) / optimal if (optimal := float(_pick_xv(g, "official_pts")["official_pts"].sum())) > 0 else np.nan
        for g in rounds
    ]
    team = []
    for g in rounds:
        if _pick_xv(g, score_col).empty:
            team.append(np.nan)
            continue
        opt_xv = _pick_xv(g, "official_pts")
        opt_pool = g[~g["player_id"].isin(set(opt_xv["player_id"]))]
        optimal = (
            float(opt_xv["official_pts"].sum())
            + (float(opt_xv["official_pts"].max()) if not opt_xv.empty else 0.0)
            + SUPERSUB_MULTIPLIER * (float(opt_pool["official_pts"].max()) if not opt_pool.empty else 0.0)
        )
        team.append(_team_value(g, score_col) / optimal if optimal > 0 else np.nan)
    hits = {n: 0 for n in (1, 3, 5)}
    for g in rounds:
        cap, act = g.loc[g[score_col].idxmax()], g["official_pts"].sort_values(ascending=False).to_numpy()
        for n in hits:
            hits[n] += int(cap["official_pts"] >= act[min(n - 1, len(act) - 1)])
    overlap = lambda n: (
        float(
            np.mean(
                [
                    len(set(g.nlargest(min(n, len(g)), score_col)["player_id"]) & set(g.nlargest(min(n, len(g)), "official_pts")["player_id"])) / min(n, len(g))
                    for g in rounds
                ]
            )
        )
        if rounds
        else np.nan
    )
    num = den = 0.0
    for _, g in lab.dropna(subset=[score_col]).groupby("canonical_pos"):
        if g[score_col].nunique() >= 3 and len(g) >= 5 and np.isfinite(rho := spearmanr(g[score_col], g["official_pts"]).correlation):
            num, den = num + rho * len(g), den + len(g)
    labelled = lab.dropna(subset=[score_col])
    return {
        "value_team": float(np.nanmean(team)),
        "value_xv": float(np.nanmean(xv)),
        **{f"capt_top{n}": hits[n] / len(rounds) if rounds else np.nan for n in hits},
        "top15": overlap(15),
        "top30": overlap(30),
        "spearman_pos": float(num / den) if den else np.nan,
        "mae": float((labelled["official_pts"] - labelled[score_col]).abs().mean()) if points else np.nan,
        "bench_mae": np.nan,
    }


def format_table(rows: dict[str, dict], title: str) -> str:
    head = f"{'engine':16s} " + " ".join(f"{c:>10s}" for c in COLUMNS)
    cells = lambda m: " ".join("       nan" if (v := m.get(c, np.nan)) is None or (isinstance(v, float) and np.isnan(v)) else f"{v:10.3f}" for c in COLUMNS)
    return "\n".join([f"\n=== {title} ===", head, "-" * len(head)] + [f"{name:16s} {cells(m)}" for name, m in rows.items()])
