#!/usr/bin/env python3
"""model/research.py  —  Karpathy-style autoresearch loop.

Greedy hill-climb over a queue of principled candidate changes, each evaluated on
the 2025 dev backtest, accepted only on a Pareto improvement (value_xv up, MAE not
worse — or the reverse) that holds on >=3/5 rounds.  See RESEARCH_GOAL.md.

Hard invariants (never relaxed):
  * 2026 is sealed — the loop never reads it (asserted).
  * Component engines are chosen on 2023+2024 component-rate OOF (independent of
    the 2025 points signal); only the assembly knobs are tuned against 2025.
  * Leakage asserts in model.data run at import; kicking-mass gate must hold.

Usage:
  python -m model.research            # one pass over the candidate queue
  python -m model.research --seal-2026  # evaluate the final best on 2026 ONCE
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge

from model import assemble as A
from model import train_components as TC
from model.assemble import assemble_predictions, rank_scores
from model.baselines import MIN_MINUTES, predict_rates
from model.data import load
from model.evaluate import (
    captain_hitrate,
    points_mae,
    spearman_within_pos,
    topn_overlap,
    value_of_xv,
)
from model.splits import component_train, round_iter
from model.train_components import (
    LGBM_COMPS,
    predict_rates_lgbm_only,
    predict_rates_registry,
    save_registry,
    select_components,
)

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
DATA = ROOT / "data"
DEV_SEASON = 2025          # the only modern-labelled dev season
SEALED_SEASON = 2026       # never read inside the loop
MODE = "post_team_sheet"
BEST_CONFIG = RESEARCH / "best_config.json"
PROMOTED_CONFIG = RESEARCH / "promoted_config.json"

# acceptance thresholds (see RESEARCH_GOAL.md)
DELTA_V = 0.003            # min value_xv gain to accept on the value_xv leg
TAU_MAE = 0.02            # max tolerated MAE regression on the value_xv leg
DELTA_M = 0.02            # min MAE gain to accept on the MAE leg
ROUND_ROBUST = 3          # value_xv gain must hold on >= this many of 5 rounds


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """All knobs.  Defaults reproduce the current cdx deployable behaviour."""
    name: str = "baseline"
    note: str = "current cdx selected deployable engine"
    deploy_engine: str = "lgbm_only"       # lgbm_only | bayesian | registry
    # component layer (validated on 2023+2024 OOF)
    lgbm_margin: float = 0.02
    blend_weight: float = 0.5
    # minutes model
    minutes_model: str = "ridge"        # ridge | ridge_interact | poisson
    minutes_alpha: float = 10.0
    # latent / POTM
    potm_pp_weight: float = 0.5
    potm_tau_floor: float = 1.0
    latent_shrink: float = 1.0
    # post-hoc
    recon_calib: str = "none"           # none | linear (forward-chained per round)
    target_prior_blend: float = 0.0     # blend points toward prior position mean
    selector_tilt: float = 0.0          # blend rank head into the XV pick only

    def delta(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


@contextlib.contextmanager
def apply_config(cfg: Config):
    """Thread cfg into the module globals that the pipeline reads, then restore."""
    saved = (
        TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
        A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK,
    )
    TC.LGBM_MARGIN = cfg.lgbm_margin
    TC.BLEND_WEIGHT = cfg.blend_weight
    A.POTM_PP_WEIGHT = cfg.potm_pp_weight
    A.POTM_TAU_FLOOR = cfg.potm_tau_floor
    A.LATENT_SHRINK = cfg.latent_shrink
    try:
        yield
    finally:
        (TC.LGBM_MARGIN, TC.BLEND_WEIGHT,
         A.POTM_PP_WEIGHT, A.POTM_TAU_FLOOR, A.LATENT_SHRINK) = saved


# ---------------------------------------------------------------------------
# minutes variants (default `ridge`+alpha=10 mirrors baselines.expected_minutes)
# ---------------------------------------------------------------------------
def build_minutes(df, train_idx, test_idx, cfg: Config) -> np.ndarray:
    tr = df.iloc[train_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows):
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    te = df.iloc[test_idx]
    feats = ["started", "jersey", "is_forward"]

    def design(rows):
        cols = [rows[feats].astype(float).to_numpy(), prior(rows)[:, None]]
        if cfg.minutes_model == "ridge_interact":
            isf = rows["is_forward"].astype(float).to_numpy()
            cols.append((rows["started"].astype(float).to_numpy() * isf)[:, None])
            cols.append((rows["jersey"].astype(float).to_numpy() * isf)[:, None])
        return np.column_stack(cols)

    Xtr, ytr = design(tr), tr["minutes"].to_numpy(float)
    Xte = design(te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.minutes_model == "poisson":
            reg = PoissonRegressor(alpha=cfg.minutes_alpha, max_iter=500).fit(
                Xtr, np.clip(ytr, 0, None))
        else:
            reg = Ridge(alpha=cfg.minutes_alpha).fit(Xtr, ytr)
    return np.clip(reg.predict(Xte), 0, 80)


# ---------------------------------------------------------------------------
# post-hoc transforms
# ---------------------------------------------------------------------------
def _forward_linear_calib(pred: pd.DataFrame) -> np.ndarray:
    """Per-round linear recalibration of recon_pts_hat -> official_pts, fit only on
    labelled rows from EARLIER rounds (forward-chained, no leakage).  Round 1 and
    thin-history rounds fall back to identity."""
    out = pred["recon_pts_hat"].to_numpy(float).copy()
    lab = (pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)).to_numpy()
    rnd = pred["round"].to_numpy()
    recon = pred["recon_pts_hat"].to_numpy(float)
    off = pred["official_pts"].to_numpy(float)
    for R in np.unique(rnd):
        prev = lab & (rnd < R)
        cur = rnd == R
        if prev.sum() >= 20 and np.ptp(recon[prev]) > 1e-6:
            slope, intercept = np.polyfit(recon[prev], off[prev], 1)
            out[cur] = slope * recon[cur] + intercept
    return np.clip(out, 0.0, None)


def _forward_posmean_blend(
    df: pd.DataFrame, test_idx: np.ndarray, pred: pd.DataFrame, season: int,
    weight: float, *, min_n: int = 40,
) -> np.ndarray:
    """Blend point forecasts toward prior modern-labelled position means.

    This is intentionally gentler than linear calibration: it only adds a small
    prior-pull once enough earlier modern labels exist, and it never uses rows
    dated on/after the round being predicted.
    """
    out = pred["target_pts_hat"].to_numpy(float).copy()
    base = out.copy()
    for _, asof, ridx in round_iter(df, season):
        hist_mask = (
            df["is_modern"].to_numpy()
            & df["has_label"].to_numpy()
            & (df["date"] < asof).to_numpy()
        )
        if hist_mask.sum() < min_n:
            continue
        local = np.searchsorted(test_idx, ridx)
        assert np.array_equal(test_idx[local], ridx)
        hist = df.loc[hist_mask]
        global_mean = float(hist["official_pts"].mean())
        pos_mean = hist.groupby("canonical_pos")["official_pts"].mean()
        prior = df.iloc[ridx]["canonical_pos"].map(pos_mean).fillna(global_mean).to_numpy(float)
        out[local] = (1.0 - weight) * base[local] + weight * prior
    return np.clip(out, 0.0, None)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _add_selector_score(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
    pred: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, str]:
    """Attach the optional XV-selection score used by selector_tilt candidates."""
    if cfg.selector_tilt <= 0:
        return pred, "target_pts_hat"
    rk = rank_scores(df, train_idx, test_idx, MODE)
    out = pred.copy()
    out["rank_score"] = rk
    out["sel_score"] = _zscore(out["target_pts_hat"].to_numpy(float)) + \
        cfg.selector_tilt * _zscore(rk)
    return out, "sel_score"


# ---------------------------------------------------------------------------
# evaluation of one config on the dev season
# ---------------------------------------------------------------------------
_REG_CACHE: dict = {}


def _registry_for(df, train_mask, cfg: Config) -> pd.DataFrame:
    key = (round(cfg.lgbm_margin, 5), round(cfg.blend_weight, 5))
    if key not in _REG_CACHE:
        with apply_config(cfg):
            _REG_CACHE[key] = select_components(df, train_mask, MODE)
    return _REG_CACHE[key]


def _registry_lgbm_count(cfg: Config, registry: pd.DataFrame | None) -> int:
    if cfg.deploy_engine == "lgbm_only":
        return len(LGBM_COMPS)
    if cfg.deploy_engine == "bayesian":
        return 0
    return int((registry["engine"].isin(["lgbm", "blend"])).sum())


def _predict_config(
    df: pd.DataFrame, cfg: Config, season: int,
) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """Build predictions for a fixed config/season without evaluating acceptance."""
    train_mask = component_train(df, season)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where((df["season"] == season).to_numpy())[0]
    registry = None if cfg.deploy_engine in {"lgbm_only", "bayesian"} \
        else _registry_for(df, train_mask, cfg)

    with apply_config(cfg):
        minutes_hat = build_minutes(df, train_idx, test_idx, cfg)

        def predictor(d, ti, te, mo):
            if cfg.deploy_engine == "lgbm_only":
                return predict_rates_lgbm_only(d, ti, te, mo)
            if cfg.deploy_engine == "bayesian":
                return predict_rates(d, ti, te, mo, "bayesian")
            return predict_rates_registry(d, ti, te, mo, registry)

        pred, diag = assemble_predictions(
            df, train_idx, season, MODE, predictor, minutes_hat=minutes_hat)

        if cfg.recon_calib == "linear":
            recon_c = _forward_linear_calib(pred)
            pred = pred.copy()
            pred["recon_pts_hat"] = recon_c
            pred["target_pts_hat"] = recon_c + pred["latent_hat"].to_numpy(float)

        if cfg.target_prior_blend > 0:
            pred = pred.copy()
            pred["target_pts_hat"] = _forward_posmean_blend(
                df, test_idx, pred, season, cfg.target_prior_blend)

        pred, sel_col = _add_selector_score(df, train_idx, test_idx, pred, cfg)

    return pred, sel_col, registry


def _evaluate_prediction(
    pred: pd.DataFrame, sel_col: str, cfg: Config, registry: pd.DataFrame | None,
) -> dict:
    vx, ratios = value_of_xv(pred, sel_col)
    c1, c3 = captain_hitrate(pred, sel_col)
    return {
        "config": cfg.name,
        "value_xv": vx,
        "round_xv": [float(r) for r in ratios],
        "round_mae": _round_mae(pred),
        "mae": points_mae(pred, "target_pts_hat")["overall"],
        "top15": topn_overlap(pred, 15, sel_col),
        "capt_top1": c1,
        "capt_top3": c3,
        "spearman_pos": spearman_within_pos(pred, "target_pts_hat"),
        "registry_lgbm": _registry_lgbm_count(cfg, registry),
    }


def _round_mae(pred: pd.DataFrame) -> list[float]:
    out = []
    for _, g in pred.groupby("round", sort=True):
        lab = g[g["has_label"].astype(bool) & g["is_modern"].astype(bool)]
        out.append(float((lab["official_pts"] - lab["target_pts_hat"]).abs().mean()))
    return out


def dev_evaluate(df, cfg: Config, season: int = DEV_SEASON) -> dict:
    """Run the full Strategy-C assembly for `cfg` on `season` and return metrics."""
    assert season != SEALED_SEASON, "2026 is sealed — the loop must not read it"
    pred, sel_col, registry = _predict_config(df, cfg, season)
    return _evaluate_prediction(pred, sel_col, cfg, registry)


# ---------------------------------------------------------------------------
# acceptance test
# ---------------------------------------------------------------------------
def accept(best: dict, cand: dict) -> tuple[bool, str]:
    dv = cand["value_xv"] - best["value_xv"]
    dm = cand["mae"] - best["mae"]                      # negative = improvement
    rb_v = sum(c > b + 1e-9 for c, b in zip(cand["round_xv"], best["round_xv"]))
    rb_m = sum(c < b - 1e-9 for c, b in zip(cand["round_mae"], best["round_mae"]))
    pareto_v = dv >= DELTA_V and dm <= TAU_MAE
    pareto_m = dm <= -DELTA_M and dv >= -DELTA_V
    if pareto_v and rb_v >= ROUND_ROBUST:
        return True, (f"ACCEPT via value_xv: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")
    if pareto_m and rb_m >= ROUND_ROBUST:
        return True, (f"ACCEPT via mae: dval={dv:+.3f} dmae={dm:+.3f} "
                      f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")
    if pareto_v or pareto_m:
        why = "round robustness failed"
    else:
        why = "no Pareto improvement"
    return (False, f"reject ({why}): dval={dv:+.3f} dmae={dm:+.3f} "
            f"value_rounds_up={rb_v}/5 mae_rounds_up={rb_m}/5")


# ---------------------------------------------------------------------------
# candidate queue — each is (name, note, delta kwargs)
# ---------------------------------------------------------------------------
CANDIDATES = [
    ("deploy_bayesian", "use Bayesian ridge component architecture", dict(deploy_engine="bayesian")),
    ("deploy_registry", "return to OOF-gated component registry", dict(deploy_engine="registry")),
    ("minutes_alpha_3", "tighter minutes ridge", dict(minutes_alpha=3.0)),
    ("minutes_alpha_30", "looser minutes ridge", dict(minutes_alpha=30.0)),
    ("minutes_interact", "started/jersey x is_forward", dict(minutes_model="ridge_interact")),
    ("minutes_poisson", "Poisson minutes head", dict(minutes_model="poisson", minutes_alpha=1.0)),
    ("latent_shrink_05", "shrink set-piece latent 0.5", dict(latent_shrink=0.5)),
    ("latent_shrink_0", "drop set-piece latent (fallback 4)", dict(latent_shrink=0.0)),
    ("potm_pp_07", "POTM lean on points", dict(potm_pp_weight=0.7)),
    ("potm_pp_03", "POTM lean on role prior", dict(potm_pp_weight=0.3)),
    ("potm_tau_2", "softer POTM softmax", dict(potm_tau_floor=2.0)),
    ("lgbm_margin_05", "stricter LGBM gate", dict(lgbm_margin=0.05)),
    ("lgbm_margin_01", "looser LGBM gate", dict(lgbm_margin=0.01)),
    ("blend_07", "heavier LGBM in blend", dict(blend_weight=0.7)),
    ("recon_calib_linear", "forward-chained recon calibration", dict(recon_calib="linear")),
    ("target_posmean_10", "blend target 10% toward prior position mean",
     dict(target_prior_blend=0.10)),
    ("target_posmean_20", "blend target 20% toward prior position mean",
     dict(target_prior_blend=0.20)),
    ("selector_tilt_025", "tilt XV pick toward rank head", dict(selector_tilt=0.25)),
    ("selector_tilt_05", "stronger rank tilt", dict(selector_tilt=0.5)),
]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run_loop(candidate_names: set[str] | None = None, limit: int | None = None) -> tuple[Config, dict, list]:
    df = load()
    RESEARCH.mkdir(exist_ok=True)

    best_cfg = Config()
    best = dev_evaluate(df, best_cfg)
    trials = [{"trial": 0, "name": "baseline", "note": best_cfg.note,
               "accepted": True, "reason": "incumbent",
               **_metric_row(best)}]
    print(f"[0] baseline           value_xv={best['value_xv']:.3f} "
          f"mae={best['mae']:.3f} top15={best['top15']:.3f} "
          f"lgbm={best['registry_lgbm']}")

    candidates = CANDIDATES
    if candidate_names is not None:
        candidates = [c for c in candidates if c[0] in candidate_names]
    if limit is not None:
        candidates = candidates[:limit]

    for i, (name, note, kw) in enumerate(candidates, 1):
        cand_cfg = best_cfg.delta(name=name, note=note, **kw)
        cand = dev_evaluate(df, cand_cfg)
        ok, reason = accept(best, cand)
        print(f"[{i}] {name:20s} value_xv={cand['value_xv']:.3f} "
              f"mae={cand['mae']:.3f} top15={cand['top15']:.3f}  -> "
              f"{'ACCEPT' if ok else 'reject'}  ({reason})")
        trials.append({"trial": i, "name": name, "note": note,
                       "accepted": ok, "reason": reason, **_metric_row(cand)})
        if ok:
            best_cfg, best = cand_cfg, cand

    _write_ledger(trials, best_cfg, best)
    _persist_best(df, best_cfg, best)
    print(f"\nBEST = {best_cfg.name}  value_xv={best['value_xv']:.3f} "
          f"mae={best['mae']:.3f}  (cdx baseline 0.680 / 7.680)")
    return best_cfg, best, trials


def _metric_row(m: dict) -> dict:
    return {k: m[k] for k in ("value_xv", "mae", "top15", "capt_top1",
                              "capt_top3", "spearman_pos", "registry_lgbm")} \
        | {"round_xv": m["round_xv"], "round_mae": m["round_mae"]}


def _write_ledger(trials: list, best_cfg: Config, best: dict) -> None:
    (RESEARCH / "ledger.json").write_text(json.dumps(trials, indent=2))
    (RESEARCH / "best_config.json").write_text(
        json.dumps(dataclasses.asdict(best_cfg), indent=2))
    lines = [f"# Autoresearch ledger — {date.today()}", "",
             f"Dev season {DEV_SEASON} (post_team_sheet). Baseline value_xv=0.680 / MAE=7.680.",
             "2026 sealed — not evaluated here.", "",
             "| # | candidate | value_xv | MAE | top15 | accepted | reason |",
             "|---|---|---|---|---|---|---|"]
    for t in trials:
        lines.append(f"| {t['trial']} | {t['name']} | {t['value_xv']:.3f} | "
                     f"{t['mae']:.3f} | {t['top15']:.3f} | "
                     f"{'yes' if t['accepted'] else 'no'} | {t['reason']} |")
    lines += ["", f"**Best:** `{best_cfg.name}` — value_xv {best['value_xv']:.3f}, "
              f"MAE {best['mae']:.3f}, top15 {best['top15']:.3f}.",
              "", "```json", json.dumps(dataclasses.asdict(best_cfg), indent=2), "```"]
    (RESEARCH / "LEDGER.md").write_text("\n".join(lines))


def _persist_best(df, best_cfg: Config, best: dict) -> None:
    """Refresh the registry + 2025 predictions for the winning config (dev only)."""
    pred, _, registry = _predict_config(df, best_cfg, DEV_SEASON)
    if registry is not None:
        save_registry(registry, DATA / "model_component_registry.csv")
    pred.to_csv(DATA / f"model_predictions_{DEV_SEASON}.csv", index=False)


def seal_2026(best_cfg: Config, *, persist_predictions: bool = True) -> None:
    """Evaluate the final best config on the SEALED season exactly once."""
    df = load()
    pred, sel_col, registry = _predict_config(df, best_cfg, SEALED_SEASON)
    res = _evaluate_prediction(pred, sel_col, best_cfg, registry)
    print("\n=== SEALED 2026 (evaluated once for the final best config) ===")
    print(f"  config={best_cfg.name}  value_xv={res['value_xv']:.3f}  "
          f"mae={res['mae']:.3f}  top15={res['top15']:.3f}")
    note = {"sealed_season": SEALED_SEASON, "config": best_cfg.name,
            "evaluated_on": str(date.today()), **_metric_row(res)}
    if persist_predictions:
        (RESEARCH / "sealed_2026.json").write_text(json.dumps(note, indent=2))
        out = DATA / f"model_predictions_{SEALED_SEASON}.csv"
        pred.to_csv(out, index=False)
        print(f"  saved {out.name} from {best_cfg.name}  ({len(pred)} rows)")
    else:
        (RESEARCH / "sealed_best_check.json").write_text(json.dumps(note, indent=2))
        print("  prediction CSV not overwritten for non-promoted seal check")


def _seal_eval(df, cfg: Config) -> dict:
    pred, sel_col, registry = _predict_config(df, cfg, SEALED_SEASON)
    return _evaluate_prediction(pred, sel_col, cfg, registry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-2026", action="store_true",
                    help="evaluate the saved best config on the sealed season once")
    ap.add_argument("--seal-config", choices=["promoted", "best"], default="promoted",
                    help="which saved config to seal; promoted falls back to best if absent")
    ap.add_argument("--candidate", action="append",
                    help="run only this candidate name; can be passed multiple times")
    ap.add_argument("--limit", type=int,
                    help="run only the first N candidates after filtering")
    args = ap.parse_args()
    if args.seal_2026:
        cfg_path = PROMOTED_CONFIG if args.seal_config == "promoted" else BEST_CONFIG
        if not cfg_path.exists() and args.seal_config == "promoted":
            cfg_path = BEST_CONFIG
        print(f"Sealing config from {cfg_path.relative_to(ROOT)}")
        cfg = Config(**json.loads(cfg_path.read_text()))
        seal_2026(cfg, persist_predictions=(cfg_path == PROMOTED_CONFIG))
    else:
        run_loop(set(args.candidate) if args.candidate else None, args.limit)


if __name__ == "__main__":
    main()
