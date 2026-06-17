#!/usr/bin/env python3
"""model/run.py  —  run harness (Phase 6).

Backtest (train 2023+2024 -> 2025) and deployment (train 2023+2024+2025 -> 2026),
printing a comparison table across all engines and both feature-view modes:
  ensemble | xgb_only | glm | ridge | naive | b3_direct | rank_head

2026 is sealed: it is evaluated exactly once here, after the 2025 backtest
result is accepted, and is never used to choose components or tune anything.

Selection policy:
  pick the best deployable point engine on the 2025 backtest, requiring it to
  beat naive on both MAE and XV value.  Rank candidates by XV value first, then
  MAE.  This keeps b3_direct/rank_head as diagnostics while allowing xgb_only
  to win if it proves better at the assembled picker objective.

Usage:
  python -m model.run                # backtest + deployment, post-team-sheet headline
  python -m model.run --no-2026      # backtest only (keep 2026 sealed)
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from model import baselines as B
from model.assemble import assemble_predictions, rank_scores
from model.baselines import build_matrix, direct_points_oof, predict_rates
from model.data import DATA, load
from model.evaluate import evaluate, format_table
from model.splits import component_train
from model.train_components import (
    predict_rates_xgb_only,
    predict_rates_registry,
    save_registry,
    select_components,
)

ENGINE_ORDER = ["ensemble", "xgb_only", "glm", "ridge", "naive"]
DEPLOYABLE_ENGINES = ["ensemble", "xgb_only", "glm", "ridge", "naive"]


def build_predictors(registry: pd.DataFrame):
    return {
        "naive": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "naive"),
        "ridge": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "ridge"),
        "glm": lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "glm"),
        "xgb_only": lambda d, ti, te, mo: predict_rates_xgb_only(d, ti, te, mo),
        "ensemble": lambda d, ti, te, mo: predict_rates_registry(d, ti, te, mo, registry),
    }


def direct_points_frame(df: pd.DataFrame, test_season: int, mode: str) -> pd.DataFrame:
    """B3 cross-check frame: predicted modern `official_pts` directly.

    Trains on prior modern-labelled seasons when they exist (2026 deployment);
    otherwise honest in-season GroupKFold OOF (2025 backtest, no prior modern).
    """
    sub = df[df["season"] == test_season]
    prior = df[(df["season"] < test_season) & df["is_modern"] & df["has_label"]]
    yhat = pd.Series(np.nan, index=sub.index, dtype=float)
    if len(prior):
        tr_idx = np.where((df["season"] < test_season).to_numpy()
                          & df["is_modern"].to_numpy() & df["has_label"].to_numpy())[0]
        te_idx = np.where((df["season"] == test_season).to_numpy())[0]
        Xtr, Xte, _ = build_matrix(df, tr_idx, te_idx, mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = RidgeCV(alphas=(1.0, 10.0, 100.0)).fit(
                Xtr, df.iloc[tr_idx]["official_pts"].to_numpy(float))
        yhat.loc[sub.index] = mdl.predict(Xte)
    else:
        oof = direct_points_oof(df, test_season, mode)
        yhat.loc[oof.index] = oof.to_numpy()
    return pd.DataFrame({
        "round": sub["round"].to_numpy(), "fixture_id": sub["fixture_id"].to_numpy(),
        "player_id": sub["player_id"].to_numpy(), "canonical_pos": sub["canonical_pos"].to_numpy(),
        "is_forward": sub["is_forward"].to_numpy(), "official_pts": sub["official_pts"].to_numpy(),
        "has_label": sub["has_label"].to_numpy(), "is_modern": sub["is_modern"].to_numpy(),
        "target_pts_hat": yhat.to_numpy(),
    }, index=sub.index)


def run_season(df, train_idx, test_season, mode, predictors, registry):
    results, preds = {}, {}
    for name in ENGINE_ORDER:
        pred, diag = assemble_predictions(df, train_idx, test_season, mode, predictors[name])
        results[name] = evaluate(pred)
        preds[name] = (pred, diag)
    # B3 direct-points cross-check
    results["b3_direct"] = evaluate(direct_points_frame(df, test_season, mode))
    # optional rank head (ordering overlay on the ensemble frame)
    test_idx = np.where((df["season"] == test_season).to_numpy())[0]
    rk = preds["ensemble"][0].copy()
    rk["rank_score"] = rank_scores(df, train_idx, test_idx, mode)
    results["rank_head"] = evaluate(rk, score_col="rank_score", points=False)
    return results, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-2026", action="store_true", help="keep 2026 sealed")
    ap.add_argument("--mode", default="post_team_sheet",
                    choices=["pre_team_sheet", "post_team_sheet"])
    args = ap.parse_args()

    df = load()

    # ---- component selection on the BACKTEST train years (no 2025/2026 peek) ----
    bt_train_mask = component_train(df, 2025)
    print("Selecting components on 2023+2024 OOF (post_team_sheet) ...")
    registry = select_components(df, bt_train_mask, args.mode)
    save_registry(registry)
    print(registry[["component", "kind", "engine", "reason"]].to_string(index=False))

    predictors = build_predictors(registry)
    bt_train_idx = np.where(bt_train_mask)[0]

    # ---- 2025 backtest ----
    res25, preds25 = run_season(df, bt_train_idx, 2025, args.mode, predictors, registry)
    print(format_table(res25, f"2025 BACKTEST  (train 2023+2024, mode={args.mode})"))

    # mode comparison for the two anchor engines
    if args.mode == "post_team_sheet":
        alt = {}
        for name in ("naive", "ensemble"):
            pred, _ = assemble_predictions(df, bt_train_idx, 2025, "pre_team_sheet",
                                           predictors[name])
            alt[name] = evaluate(pred)
        print(format_table(alt, "2025 BACKTEST  (mode=pre_team_sheet, anchors only)"))

    selected_engine = select_deployable_engine(res25)
    print(f"\nSelected deployable engine from 2025 backtest: {selected_engine}")
    _accept_and_persist(df, bt_train_idx, 2025, args.mode, predictors,
                        selected_engine)

    # ---- 2026 deployment (sealed; evaluated once) ----
    if not args.no_2026:
        dep_train_idx = np.where(component_train(df, 2026))[0]
        res26, _ = run_season(df, dep_train_idx, 2026, args.mode, predictors, registry)
        print(format_table(res26, f"2026 DEPLOYMENT  (train 2023+2024+2025, mode={args.mode})"))
        _accept_and_persist(df, dep_train_idx, 2026, args.mode, predictors,
                            selected_engine)

    seasons = [2025] + ([] if args.no_2026 else [2026])
    _persist_promoted_research_config(df, seasons)

    _verdict(res25, selected_engine)


def select_deployable_engine(results: dict[str, dict]) -> str:
    """Choose the best deployable point model from the 2025 validation table.

    `b3_direct` is diagnostic and `rank_head` is not a calibrated point forecast,
    so neither is eligible.  Candidates must beat naive on both target metrics;
    if none do, fall back to naive.
    """
    naive = results["naive"]
    candidates = []
    for name in DEPLOYABLE_ENGINES:
        m = results[name]
        if m["mae"] <= naive["mae"] and m["value_xv"] >= naive["value_xv"]:
            candidates.append(name)
    if not candidates:
        return "naive"
    return sorted(
        candidates,
        key=lambda n: (-results[n]["value_xv"], results[n]["mae"], n),
    )[0]


def _accept_and_persist(df, train_idx, season, mode, predictors, engine):
    pred, _ = assemble_predictions(df, train_idx, season, mode, predictors[engine])
    out = DATA / f"model_predictions_{season}.csv"
    pred.to_csv(out, index=False)
    print(f"  saved {out.name} from {engine}  ({len(pred)} rows)")


def _persist_promoted_research_config(df, seasons: list[int]) -> None:
    """If the research loop has a promotion decision, make it the final artifact.

    `model.run` still prints the full diagnostic table, but the branch's deployable
    config may include research-only assembly/selection knobs such as `sel_score`.
    """
    from model.research import Config, PROMOTED_CONFIG, _predict_config

    if not PROMOTED_CONFIG.exists():
        return
    cfg = Config(**json.loads(PROMOTED_CONFIG.read_text()))
    for season in seasons:
        pred, _, _ = _predict_config(df, cfg, season)
        out = DATA / f"model_predictions_{season}.csv"
        pred.to_csv(out, index=False)
        print(f"  refreshed {out.name} from promoted research config {cfg.name} "
              f"({len(pred)} rows)")


def _verdict(res25, selected_engine):
    e, n = res25[selected_engine], res25["naive"]
    print("\n=== VERDICT (2025 backtest, baseline bar) ===")
    print(f"  {selected_engine} value_xv={e['value_xv']:.3f} vs naive {n['value_xv']:.3f} "
          f"-> {'PASS' if e['value_xv'] >= n['value_xv'] else 'FAIL'}")
    print(f"  {selected_engine} MAE={e['mae']:.3f} vs naive {n['mae']:.3f} "
          f"-> {'PASS' if e['mae'] <= n['mae'] else 'FAIL'}")
    print(f"  {selected_engine} spearman={e['spearman_pos']:.3f} vs naive {n['spearman_pos']:.3f}")


if __name__ == "__main__":
    main()
