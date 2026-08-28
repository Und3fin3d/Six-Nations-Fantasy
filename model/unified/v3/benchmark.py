"""Leak-free official-points selection and retrospective reference benchmark."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..features import build_pit_features
from ..gbdt import UniversalGBDT
from ..labels import build_fantasy_labels
from ..scoring import scorer_for
from .blend import EventBlendModel
from .cohorts import match_labels_to_store
from .config import BaselineConfig, GBDTV3Config, NeuralV3Config
from .context import augment_context, restrict_context
from .gbdt import ExposureRateGBDT
from .harness import (
    OUT, attach_match_timestamps, fold_for_block, strict_training_frame,
    sha256, write_fold_manifest,
)
from .metrics import paired_bootstrap_difference
from .neural import ExposureRateNeural

DATA = ROOT / "data"
CUTS = (10, 25, 50, 100)


def _load_configs() -> tuple[BaselineConfig, GBDTV3Config, NeuralV3Config]:
    search = OUT / "search"
    baseline_path = search / "best_baseline.json"
    gbdt_path = search / "best_gbdt.json"
    neural_path = search / "best_neural.json"
    baseline = (
        BaselineConfig.from_dict(json.loads(baseline_path.read_text()))
        if baseline_path.exists() else BaselineConfig()
    )
    gbdt = (
        GBDTV3Config.from_dict(json.loads(gbdt_path.read_text()))
        if gbdt_path.exists() else GBDTV3Config()
    )
    neural = (
        NeuralV3Config.from_dict(json.loads(neural_path.read_text()))
        if neural_path.exists() else NeuralV3Config()
    )
    audit_path = OUT / "audit" / "audit.json"
    appearance = False
    if audit_path.exists():
        appearance = bool(
            json.loads(audit_path.read_text()).get("zero_minutes", {})
            .get("appearance_head_enabled")
        )
    gbdt = replace(gbdt, use_appearance=appearance)
    neural = replace(neural, use_appearance=appearance)
    return baseline, gbdt, neural


def _fit(engine: str, train: pd.DataFrame, config):
    if engine == "baseline":
        return UniversalGBDT(
            random_state=config.seed, weighting=config.weighting,
            time_half_life_days=config.time_half_life_days,
            n_estimators=config.n_estimators, num_leaves=config.num_leaves,
        ).fit(train)
    if engine == "gbdt_v3":
        return ExposureRateGBDT(config=config).fit(train)
    if engine == "neural_v3":
        dates = sorted(pd.to_datetime(train["date"]).dropna().unique())
        start = dates[max(1, int(len(dates) * 0.90)) - 1]
        fit = train[pd.to_datetime(train["date"]) < start]
        validation = train[pd.to_datetime(train["date"]) >= start]
        return ExposureRateNeural(config=config).fit(fit, validation)
    if engine == "blend_v3":
        gbdt_config, neural_config, neural_weight = config
        g_train = restrict_context(train, tuple(gbdt_config.context_blocks))
        n_train = restrict_context(train, tuple(neural_config.context_blocks))
        dates = sorted(pd.to_datetime(n_train["date"]).dropna().unique())
        start = dates[max(1, int(len(dates) * 0.90)) - 1]
        neural_fit = n_train[pd.to_datetime(n_train["date"]) < start]
        neural_validation = n_train[pd.to_datetime(n_train["date"]) >= start]
        return EventBlendModel(
            ExposureRateGBDT(config=gbdt_config).fit(g_train),
            ExposureRateNeural(config=neural_config).fit(
                neural_fit, neural_validation,
            ),
            neural_weight,
        )
    raise ValueError(engine)


def _load(engine: str, path: Path):
    if engine == "baseline":
        return UniversalGBDT.load(path)
    if engine == "gbdt_v3":
        return ExposureRateGBDT.load(path)
    if engine == "neural_v3":
        return ExposureRateNeural.load(path)
    if engine == "blend_v3":
        return EventBlendModel.load(path)
    raise ValueError(engine)


def _score_predictions(model, rows: pd.DataFrame, competition: str) -> np.ndarray:
    scorer = scorer_for(competition)
    predictions = model.predict_frame(rows)
    return np.array([
        scorer.score_prediction(prediction, n=800, seed=701 + i).mean
        for i, prediction in enumerate(predictions)
    ])


def _cohort_predictions(model, labels: pd.DataFrame, feature_store: pd.DataFrame,
                        competition: str) -> pd.DataFrame:
    joined = match_labels_to_store(labels, feature_store)
    output = labels.reset_index(drop=True).copy()
    output["_label_row_id"] = output.index
    matched = joined[joined["store_matched"]].copy()
    matched["predicted_points"] = _score_predictions(model, matched, competition)
    values = matched.set_index("_label_row_id")["predicted_points"]
    output["predicted_points"] = output["_label_row_id"].map(values)
    output["career_matches"] = output["_label_row_id"].map(
        matched.set_index("_label_row_id")["career_matches"]
    )
    output["matched"] = output["predicted_points"].notna()
    # Full-cohort parity: unseen/crosswalk-missing rows receive position medians.
    position_medians = (
        output[output["matched"]].groupby("position")["predicted_points"].median()
    )
    global_median = float(output["predicted_points"].median()) if output["matched"].any() else 0.0
    need = output["predicted_points"].isna()
    output.loc[need, "predicted_points"] = (
        output.loc[need, "position"].map(position_medians).fillna(global_median)
    )
    return output


def _incumbent(block: pd.DataFrame) -> pd.Series:
    competition = block["competition"].iloc[0]
    season = int(block["season"].iloc[0])
    round_no = int(block["round"].iloc[0])
    if competition == "six_nations":
        path = DATA / f"model_predictions_{season}.csv"
        incumbent = pd.read_csv(path)
        incumbent["key_fixture"] = incumbent["fixture_id"].astype(str)
        incumbent["key_player"] = incumbent["player_id"].astype(str)
        key = block.copy()
        key["key_fixture"] = key["key_fixture"].astype(str)
        key["key_player"] = key["key_player"].astype(str)
        merged = key.merge(
            incumbent[["key_fixture", "key_player", "target_pts_hat"]],
            on=["key_fixture", "key_player"], how="left", validate="one_to_one",
        )
        return merged["target_pts_hat"]
    projection = pd.read_csv(DATA / "ncr" / f"ncr_gw{round_no}_projections.csv")
    projection["key_player"] = projection["id"].astype(int).astype(str)
    key = block.copy()
    key["key_player"] = key["key_player"].astype(str)
    return key.merge(
        projection[["key_player", "starter_exp"]], on="key_player",
        how="left", validate="one_to_one",
    )["starter_exp"]


def _engine_store(raw_timed: pd.DataFrame, engine: str, config) -> pd.DataFrame:
    if engine == "baseline":
        blocks = ()
    elif engine == "blend_v3":
        blocks = tuple(dict.fromkeys(
            (*config[0].context_blocks, *config[1].context_blocks)
        ))
    else:
        blocks = tuple(config.context_blocks)
    return build_pit_features(augment_context(raw_timed, blocks))


def _ensure_manifest(path: Path, fold, train, evaluation, store_path, artifacts=()):
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["fold"]["lock_at"] != fold.lock_at:
            raise ValueError(f"existing manifest has different lock: {path}")
        for artifact in artifacts:
            expected = existing.get("artifact_sha256", {}).get(str(artifact))
            if expected != sha256(artifact):
                raise ValueError(f"artifact hash no longer matches immutable manifest: {artifact}")
        return existing
    return write_fold_manifest(
        path, fold, train, evaluation, store_path=store_path, artifacts=artifacts,
    )


def run_benchmark(output_dir: Path = OUT / "benchmark",
                  engines: tuple[str, ...] = ("baseline", "gbdt_v3", "neural_v3")) -> pd.DataFrame:
    neural_status = OUT / "search" / "neural_track.json"
    if "neural_v3" in engines and neural_status.exists():
        status = json.loads(neural_status.read_text())
        if status.get("status") == "stopped":
            print(f"[neural-v3 stopped] {status.get('reason')}", flush=True)
            engines = tuple(engine for engine in engines if engine != "neural_v3")
    labels = build_fantasy_labels()
    raw_path = DATA / "unified" / "player_match.csv"
    raw = pd.read_csv(raw_path, low_memory=False, parse_dates=["date"])
    raw_timed = attach_match_timestamps(raw)
    baseline_config, gbdt_config, neural_config = _load_configs()
    blend_path = OUT / "search" / "blend.json"
    blend = json.loads(blend_path.read_text()) if blend_path.exists() else {}
    if "blend_v3" in engines and not blend.get("eligible"):
        raise ValueError("blend_v3 was requested but did not pass eligibility gates")
    if blend.get("eligible") and "blend_v3" not in engines:
        engines = (*engines, "blend_v3")
    configs = {
        "baseline": baseline_config, "gbdt_v3": gbdt_config, "neural_v3": neural_config,
        "blend_v3": (
            gbdt_config, neural_config, float(blend.get("neural_weight", 0.5))
        ),
    }
    stores = {
        engine: _engine_store(raw_timed, engine, configs[engine]) for engine in engines
    }
    metric_rows = []
    prediction_rows = []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        phase = (
            "selection" if block["competition"].iloc[0] == "six_nations"
            and int(block["season"].iloc[0]) == 2025 else "retrospective_reference"
        )
        for engine in engines:
            store = stores[engine]
            fold = fold_for_block(block, store)
            train = strict_training_frame(store, fold)
            evaluation = match_labels_to_store(block, store)
            matched_evaluation = evaluation[evaluation["store_matched"]]
            suffix = (
                ".pt" if engine == "neural_v3"
                else ".json" if engine == "blend_v3" else ".pkl"
            )
            artifact = output_dir / "artifacts" / engine / f"{fold.label}{suffix}"
            manifest_path = output_dir / "manifests" / engine / f"{fold.label}.json"
            if artifact.exists() != manifest_path.exists():
                raise RuntimeError(
                    f"incomplete immutable fold artifact pair: {artifact}, {manifest_path}"
                )
            if artifact.exists():
                model = _load(engine, artifact)
            else:
                print(f"[{phase}] {engine} {group_id}: train {len(train):,}", flush=True)
                model = _fit(engine, train, configs[engine])
                artifact.parent.mkdir(parents=True, exist_ok=True)
                model.save(artifact)
            artifacts = [artifact]
            if engine == "blend_v3":
                artifacts.extend([
                    artifact.with_suffix(artifact.suffix + ".gbdt.pkl"),
                    artifact.with_suffix(artifact.suffix + ".neural.pt"),
                ])
            _ensure_manifest(
                manifest_path, fold, train, matched_evaluation, raw_path,
                artifacts=tuple(artifacts),
            )
            predicted = _cohort_predictions(
                model, block, store, str(block["competition"].iloc[0]),
            )
            predicted["incumbent_points"] = _incumbent(block).to_numpy()
            predicted["engine"] = engine
            predicted["phase"] = phase
            prediction_rows.append(predicted)
            metrics = _group_metrics(
                predicted.rename(columns={"official_pts": "actual"}),
                "predicted_points", actual_col="actual",
            )
            metric_rows.append({
                "competition": block["competition"].iloc[0],
                "season": int(block["season"].iloc[0]), "round": int(block["round"].iloc[0]),
                "group_id": group_id, "phase": phase, "model": engine, **metrics,
            })
        incumbent_frame = block.copy()
        incumbent_frame["incumbent_points"] = _incumbent(block).to_numpy()
        if incumbent_frame["incumbent_points"].notna().all():
            metrics = _group_metrics(incumbent_frame, "incumbent_points")
            metric_rows.append({
                "competition": block["competition"].iloc[0],
                "season": int(block["season"].iloc[0]), "round": int(block["round"].iloc[0]),
                "group_id": group_id, "phase": phase, "model": "incumbent", **metrics,
            })
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True, sort=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    report, decision = render_report(metrics, predictions)
    (output_dir / "benchmark.md").write_text(report)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return metrics


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["mae", "spearman", *[f"top_{n}_capture" for n in CUTS]]
    return frame.groupby(["phase", "competition", "model"], as_index=False)[cols].mean()


def render_report(metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[str, dict]:
    aggregate = _aggregate(metrics)
    selection = aggregate[aggregate["phase"].eq("selection")]
    incumbent_selection = selection[selection["model"].eq("incumbent")]
    candidate_selection = selection[selection["model"].ne("incumbent")].copy()
    if incumbent_selection.empty:
        raise ValueError("Six Nations 2025 incumbent metrics are required for selection")
    incumbent_row = incumbent_selection.iloc[0]
    selection_rows = []
    for row in candidate_selection.itertuples(index=False):
        capture_diffs = [
            getattr(row, f"top_{n}_capture") - incumbent_row[f"top_{n}_capture"]
            for n in CUTS
        ]
        utility = (
            0.5 * ((incumbent_row["mae"] - row.mae) / max(incumbent_row["mae"], 1e-9))
            + 0.5 * float(np.mean(capture_diffs))
        )
        selection_rows.append({
            "model": row.model, "selection_utility": utility,
            "selection_noninferior": (
                row.mae <= incumbent_row["mae"] * 1.02
                and all(diff >= -0.02 for diff in capture_diffs)
            ),
        })
    selection_policy = pd.DataFrame(selection_rows)
    eligible = selection_policy[selection_policy["selection_noninferior"]]
    ranked = eligible if len(eligible) else selection_policy
    selected = str(ranked.sort_values("selection_utility", ascending=False).iloc[0]["model"])
    reference = aggregate[aggregate["phase"].eq("retrospective_reference")]
    reasons = []
    utility_terms = []
    subgroup_reasons = []
    uncertainty = []
    corrective_margins = []
    for competition in ("six_nations", "ncr"):
        candidate = reference[
            reference["competition"].eq(competition) & reference["model"].eq(selected)
        ]
        incumbent = reference[
            reference["competition"].eq(competition) & reference["model"].eq("incumbent")
        ]
        if candidate.empty or incumbent.empty:
            reasons.append(f"missing retrospective comparison for {competition}")
            continue
        c, b = candidate.iloc[0], incumbent.iloc[0]
        corrective_margins.append(c.mae / max(b.mae, 1e-9) - 1)
        if c.mae > b.mae * 1.02:
            reasons.append(f"{competition}: MAE regression exceeds 2%")
        capture_diffs = []
        for n in CUTS:
            key = f"top_{n}_capture"
            diff = float(c[key] - b[key])
            capture_diffs.append(diff)
            corrective_margins.append(-diff)
            if diff < -0.02:
                reasons.append(f"{competition}: top-{n} capture regression exceeds 2pp")
        utility_terms.append(0.5 * ((b.mae - c.mae) / max(b.mae, 1e-9))
                             + 0.5 * float(np.mean(capture_diffs)))

        block = predictions[
            predictions["phase"].eq("retrospective_reference")
            & predictions["competition"].eq(competition)
            & predictions["engine"].eq(selected)
        ].dropna(subset=["incumbent_points"])
        for subgroup, mask in (
            ("bench", block["status"].astype(str).eq("B")),
            ("low_history", pd.to_numeric(block["career_matches"], errors="coerce").fillna(0).lt(5)),
        ):
            sub = block[mask]
            if len(sub) >= 10:
                cmae = float(np.mean(np.abs(sub["predicted_points"] - sub["official_pts"])))
                imae = float(np.mean(np.abs(sub["incumbent_points"] - sub["official_pts"])))
                if cmae > imae + 0.5:
                    subgroup_reasons.append(
                        f"{competition} {subgroup}: MAE regression {cmae - imae:+.2f} > 0.5"
                    )
                corrective_margins.append((cmae - imae) / 25.0)
        if len(block):
            uncertainty.append({
                "competition": competition,
                **paired_bootstrap_difference(
                    block["official_pts"].to_numpy(float),
                    block["predicted_points"].to_numpy(float),
                    block["incumbent_points"].to_numpy(float),
                ),
            })
    pooled_utility = float(np.mean(utility_terms)) if utility_terms else float("-inf")
    if pooled_utility <= 0:
        reasons.append(f"competition-balanced pooled utility is not positive ({pooled_utility:+.4f})")
    reasons.extend(subgroup_reasons)
    narrow_miss = bool(
        reasons
        and utility_terms
        and pooled_utility > -0.02
        and corrective_margins
        and max(corrective_margins) <= 0.03
        and not any(reason.startswith("missing ") for reason in reasons)
    )
    decision = {
        "selected_on_6n_2025": selected,
        "selection_policy": selection_policy.to_dict(orient="records"),
        "retrospective_gate_passed": not reasons,
        "pooled_utility": pooled_utility,
        "reasons": reasons,
        "paired_bootstrap_mae_difference": uncertainty,
        "promotion": False,
        "promotion_blocker": (
            "retrospective gates failed; GW4-GW7 shadows remain mandatory final evidence"
            if reasons else "requires immutable NCR GW4-GW7 prospective shadow"
        ),
        "corrective_cycle_allowed": narrow_miss,
        "stopping_rule": (
            "one corrective cycle only if corrective_cycle_allowed; otherwise retain incumbents"
        ),
    }

    lines = [
        "# Unified rugby supermodel v3 benchmark", "",
        "Six Nations 2025 is the model-selection layer. Six Nations 2026 and NCR "
        "GW1–2 are retrospective references because their results influenced development.",
        "", f"Selected architecture on 6N 2025: **{selected}**.", "",
    ]
    for phase, title in (
        ("selection", "Six Nations 2025 selection"),
        ("retrospective_reference", "Retrospective references"),
    ):
        lines.extend([f"## {title}", ""])
        view = aggregate[aggregate["phase"].eq(phase)]
        lines.extend([
            "| Competition | Model | MAE | Spearman | Top10 | Top25 | Top50 | Top100 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in view.itertuples(index=False):
            lines.append(
                f"| {row.competition} | {row.model} | {row.mae:.2f} | {row.spearman:.3f} | "
                f"{row.top_10_capture:.1%} | {row.top_25_capture:.1%} | "
                f"{row.top_50_capture:.1%} | {row.top_100_capture:.1%} |"
            )
        lines.append("")
    lines.extend([
        "## Fold-level differences versus incumbent", "",
        "| Phase | Competition | Season | Round | MAE diff | Mean capture diff |",
        "|---|---|---:|---:|---:|---:|",
    ])
    candidate_folds = metrics[metrics["model"].eq(selected)]
    incumbent_folds = metrics[metrics["model"].eq("incumbent")]
    paired = candidate_folds.merge(
        incumbent_folds,
        on=["phase", "competition", "season", "round", "group_id"],
        suffixes=("_candidate", "_incumbent"), validate="one_to_one",
    )
    for row in paired.itertuples(index=False):
        capture_diff = float(np.mean([
            getattr(row, f"top_{n}_capture_candidate")
            - getattr(row, f"top_{n}_capture_incumbent")
            for n in CUTS
        ]))
        lines.append(
            f"| {row.phase} | {row.competition} | {row.season} | {row.round} | "
            f"{row.mae_candidate - row.mae_incumbent:+.2f} | {capture_diff:+.1%} |"
        )
    lines.extend(["", "## Paired bootstrap MAE differences", ""])
    if uncertainty:
        for interval in uncertainty:
            lines.append(
                f"- {interval['competition']}: candidate minus incumbent "
                f"{interval['mean']:+.2f} (5th–95th percentile "
                f"{interval['p05']:+.2f} to {interval['p95']:+.2f})."
            )
    else:
        lines.append("- No complete retrospective comparison was available.")
    lines.append("")
    lines.extend([
        "## Decision", "",
        f"- Retrospective safe-overall gate: **{'PASS' if not reasons else 'FAIL'}**.",
        f"- Competition-balanced pooled utility: {pooled_utility:+.4f}.",
        "- Promotion: **NO** — GW4–GW7 prospective shadow remains mandatory, and "
        "all retrospective gates must also pass.",
        f"- Corrective cycle allowed after a miss: **{'YES' if narrow_miss else 'NO'}**.",
    ])
    if reasons:
        lines.extend(["", *[f"- {reason}" for reason in reasons]])
    lines.append("")
    return "\n".join(lines), decision
