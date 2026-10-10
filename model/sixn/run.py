from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from model.sixn.assemble import assemble_predictions, rank_scores
from model.sixn.baselines import build_matrix, direct_points_oof, predict_rates
from model.sixn.data import load
from model.sixn.evaluate import evaluate, format_table
from model.sixn.splits import component_train
from model.sixn.train_components import predict_rates_lgbm_only, predict_rates_registry, save_registry, select_components

ENGINE_ORDER = ["ensemble", "lgbm_only", "glm", "ridge", "naive"]
DEPLOYABLE_ENGINES = ["ensemble", "lgbm_only", "glm", "ridge", "naive"]


def build_predictors(registry: pd.DataFrame):
    return {
        "naive": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "naive"), "ridge": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "ridge"),
        "glm": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "glm"), "lgbm_only": lambda d, ti, te, mo: predict_rates_lgbm_only(d, ti, te, mo),
        "ensemble": lambda d, ti, te, mo: predict_rates_registry(d, ti, te, mo, registry),
    }


def direct_points_frame(df: pd.DataFrame, test_season: int, mode: str) -> pd.DataFrame:
    sub = df[df["season"] == test_season]
    prior = df[(df["season"] < test_season) & df["is_modern"] & df["has_label"]]
    yhat = pd.Series(np.nan, index=sub.index, dtype=float)
    if len(prior):
        tr_idx = np.where((df["season"] < test_season).to_numpy() & df["is_modern"].to_numpy() & df["has_label"].to_numpy())[0]
        te_idx = np.where((df["season"] == test_season).to_numpy())[0]
        Xtr, Xte, _ = build_matrix(df, tr_idx, te_idx, mode)
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


def run_season(df, train_idx, test_season, mode, predictors, registry):
    results, preds = {}, {}
    for name in ENGINE_ORDER:
        pred, diag = assemble_predictions(df, train_idx, test_season, mode, predictors[name])
        results[name] = evaluate(pred)
        preds[name] = (pred, diag)
    results["b3_direct"] = evaluate(direct_points_frame(df, test_season, mode))
    test_idx = np.where((df["season"] == test_season).to_numpy())[0]
    rk = preds["ensemble"][0].copy()
    rk["rank_score"] = rank_scores(df, train_idx, test_idx, mode)
    results["rank_head"] = evaluate(rk, score_col="rank_score", points=False)
    return results, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-2026", action="store_true", help="keep 2026 sealed")
    ap.add_argument("--mode", default="post_team_sheet", choices=["pre_team_sheet", "post_team_sheet"])
    args = ap.parse_args()

    df = load()

    bt_train_mask = component_train(df, 2025)
    print("Selecting components on 2023+2024 OOF (post_team_sheet) ...")
    registry = select_components(df, bt_train_mask, args.mode)
    save_registry(registry)
    print(registry[["component", "kind", "engine", "reason"]].to_string(index=False))

    predictors = build_predictors(registry)
    bt_train_idx = np.where(bt_train_mask)[0]

    res25, preds25 = run_season(df, bt_train_idx, 2025, args.mode, predictors, registry)
    print(format_table(res25, f"2025 BACKTEST  (train 2023+2024, mode={args.mode})"))

    if args.mode == "post_team_sheet":
        alt = {}
        for name in ("naive", "ensemble"):
            pred, _ = assemble_predictions(df, bt_train_idx, 2025, "pre_team_sheet", predictors[name])
            alt[name] = evaluate(pred)
        print(format_table(alt, "2025 BACKTEST  (mode=pre_team_sheet, anchors only)"))

    selected_engine = select_deployable_engine(res25)
    print(f"\nSelected deployable engine from 2025 backtest: {selected_engine}")
    _accept_and_persist(df, bt_train_idx, 2025, args.mode, predictors, selected_engine)

    if not args.no_2026:
        dep_train_idx = np.where(component_train(df, 2026))[0]
        res26, _ = run_season(df, dep_train_idx, 2026, args.mode, predictors, registry)
        print(format_table(res26, f"2026 DEPLOYMENT  (train 2023+2024+2025, mode={args.mode})"))
        _accept_and_persist(df, dep_train_idx, 2026, args.mode, predictors, selected_engine)

    seasons = [2025] + ([] if args.no_2026 else [2026])
    _persist_promoted_research_config(df, seasons)

    _verdict(res25, selected_engine)


def select_deployable_engine(results: dict[str, dict]) -> str:
    naive = results["naive"]
    candidates = []
    for name in DEPLOYABLE_ENGINES:
        m = results[name]
        if m["mae"] <= naive["mae"] and m["value_xv"] >= naive["value_xv"]:
            candidates.append(name)
    if not candidates:
        return "naive"
    return sorted(candidates, key=lambda n: (-results[n]["value_xv"], results[n]["mae"], n))[0]


def _accept_and_persist(df, train_idx, season, mode, predictors, engine):
    pred, _ = assemble_predictions(df, train_idx, season, mode, predictors[engine])
    out = f"data/model_predictions_{season}.csv"
    pred.to_csv(out, index=False)
    print(f"  saved {os.path.basename(out)} from {engine}  ({len(pred)} rows)")


def _persist_promoted_research_config(df, seasons: list[int]) -> None:
    from model.sixn.picker import PROMOTED_CONFIG, _predict_config, config_from_dict

    cfg = config_from_dict(json.load(open(PROMOTED_CONFIG)))
    for season in seasons:
        pred, _, _ = _predict_config(df, cfg, season)
        out = f"data/model_predictions_{season}.csv"
        pred.to_csv(out, index=False)
        print(f"  refreshed {os.path.basename(out)} from promoted research config {cfg.name} ({len(pred)} rows)")


def _verdict(res25, selected_engine):
    e, n = res25[selected_engine], res25["naive"]
    print("\n=== VERDICT (2025 backtest, baseline bar) ===")
    print(f"  {selected_engine} value_xv={e['value_xv']:.3f} vs naive {n['value_xv']:.3f} -> {'PASS' if e['value_xv'] >= n['value_xv'] else 'FAIL'}")
    print(f"  {selected_engine} MAE={e['mae']:.3f} vs naive {n['mae']:.3f} -> {'PASS' if e['mae'] <= n['mae'] else 'FAIL'}")
    print(f"  {selected_engine} spearman={e['spearman_pos']:.3f} vs naive {n['spearman_pos']:.3f}")


if __name__ == "__main__":
    main()
