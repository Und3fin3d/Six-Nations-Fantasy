from __future__ import annotations

import warnings
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from model.sixn import picker
from model.sixn.assemble import assemble_predictions, rank_scores
from model.sixn.baselines import build_matrix, direct_points_oof, predict_rates
from model.sixn.data import load
from model.sixn.splits import component_train
from model.sixn.train_components import predict_rates_lgbm_only, predict_rates_registry, select_components

XV_QUOTA = {"Prop": 2, "Hooker": 1, "Second-row": 2, "Back-row": 3, "Scrum-half": 1, "Fly-half": 1, "Centre": 2, "Back-three": 3}
SUPERSUB_MULTIPLIER = 3.0
COLUMNS = ["mae", "bench_mae", "value_team", "value_xv", "top15", "top30", "capt_top1", "capt_top3", "capt_top5", "spearman_pos"]
ENGINES = ["ensemble", "lgbm_only", "glm", "ridge", "naive"]
MODE = "post_team_sheet"


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
    top = lambda g, col, n: set(g.nlargest(min(n, len(g)), col)["player_id"])
    overlap = lambda n: float(np.mean([len(top(g, score_col, n) & top(g, "official_pts", n)) / min(n, len(g)) for g in rounds])) if rounds else np.nan
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


def build_predictors(registry: pd.DataFrame):
    eng = dict(zip(registry["component"], registry["engine"]))
    return {e: partial(predict_rates, engine=e) for e in ("naive", "ridge", "glm")} | {
        "lgbm_only": predict_rates_lgbm_only, "ensemble": partial(predict_rates_registry, eng=eng),
    }


def direct_points_frame(df: pd.DataFrame, test_season: int, mode: str) -> pd.DataFrame:
    sub = df[df["season"] == test_season]
    yhat = pd.Series(np.nan, index=sub.index, dtype=float)
    tr_idx = np.where((df["season"] < test_season).to_numpy() & df["is_modern"].to_numpy() & df["has_label"].to_numpy())[0]
    if len(tr_idx):
        te_idx = np.where((df["season"] == test_season).to_numpy())[0]
        Xtr, Xte = build_matrix(df, tr_idx, te_idx, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(Xtr, df.iloc[tr_idx]["official_pts"].to_numpy(float))
        yhat.loc[sub.index] = mdl.predict(Xte)
    else:
        oof = direct_points_oof(df, test_season, mode)
        yhat.loc[oof.index] = oof.to_numpy()
    return pd.DataFrame(
        {
            "round": sub["round"].to_numpy(), "fixture_id": sub["fixture_id"].to_numpy(), "player_id": sub["player_id"].to_numpy(),
            "canonical_pos": sub["canonical_pos"].to_numpy(), "is_forward": sub["is_forward"].to_numpy(), "official_pts": sub["official_pts"].to_numpy(),
            "has_label": sub["has_label"].to_numpy(), "is_modern": sub["is_modern"].to_numpy(), "target_pts_hat": yhat.to_numpy(),
        },
        index=sub.index,
    )


def run_season(df, train_idx, test_season, predictors):
    results, preds = {}, {}
    for name in ENGINES:
        preds[name] = assemble_predictions(df, train_idx, test_season, MODE, predictors[name])
        results[name] = evaluate(preds[name])
    results["b3_direct"] = evaluate(direct_points_frame(df, test_season, MODE))
    rk = preds["ensemble"].copy()
    rk["rank_score"] = rank_scores(df, train_idx, np.where((df["season"] == test_season).to_numpy())[0], MODE)
    results["rank_head"] = evaluate(rk, score_col="rank_score", points=False)
    return results


def select_deployable_engine(results: dict[str, dict]) -> str:
    naive = results["naive"]
    candidates = [n for n in ENGINES if results[n]["mae"] <= naive["mae"] and results[n]["value_xv"] >= naive["value_xv"]]
    return min(candidates, key=lambda n: (-results[n]["value_xv"], results[n]["mae"], n)) if candidates else "naive"


def main():
    df = load()
    bt_train_mask = component_train(df, 2025)
    print("Selecting components on 2023+2024 OOF (post_team_sheet) ...")
    registry = select_components(df, bt_train_mask, MODE)
    print(registry[["component", "kind", "engine", "reason"]].to_string(index=False))
    predictors = build_predictors(registry)
    bt_train_idx = np.where(bt_train_mask)[0]
    res25 = run_season(df, bt_train_idx, 2025, predictors)
    print(format_table(res25, f"2025 BACKTEST  (train 2023+2024, mode={MODE})"))
    alt = {name: evaluate(assemble_predictions(df, bt_train_idx, 2025, "pre_team_sheet", predictors[name])) for name in ("naive", "ensemble")}
    print(format_table(alt, "2025 BACKTEST  (mode=pre_team_sheet, anchors only)"))
    engine = select_deployable_engine(res25)
    print(f"\nSelected deployable engine from 2025 backtest: {engine}")
    res26 = run_season(df, np.where(component_train(df, 2026))[0], 2026, predictors)
    print(format_table(res26, f"2026 DEPLOYMENT  (train 2023+2024+2025, mode={MODE})"))
    for season in (2025, 2026):
        pred = picker.predict(df, season)
        pred.to_csv(f"data/model_predictions_{season}.csv", index=False)
        print(f"  refreshed model_predictions_{season}.csv from promoted research config {picker.CFG['name']} ({len(pred)} rows)")
    e, n = res25[engine], res25["naive"]
    print("\n=== VERDICT (2025 backtest, baseline bar) ===")
    print(f"  {engine} value_xv={e['value_xv']:.3f} vs naive {n['value_xv']:.3f} -> {'PASS' if e['value_xv'] >= n['value_xv'] else 'FAIL'}")
    print(f"  {engine} MAE={e['mae']:.3f} vs naive {n['mae']:.3f} -> {'PASS' if e['mae'] <= n['mae'] else 'FAIL'}")
    print(f"  {engine} spearman={e['spearman_pos']:.3f} vs naive {n['spearman_pos']:.3f}")


if __name__ == "__main__":
    main()
