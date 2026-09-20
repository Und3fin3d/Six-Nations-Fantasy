"""Audit and report a complete candidate comparison without choosing new settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.ncr_project import BUDGET, MAX_HEMI, MAX_NATION, REQUIRED
from .contracts import RawPrediction
from .data import ROOT
from .domain_experiment import CONTROL, CONFIG_DIR, KEY
from .friendly_cohort import complete_fixture_cohort
from .friendly_eval import COUNT_TARGETS, TARGETS, summarise_stats
from .rolling_eval import official_slates, expected_points

SEASONS = {("ncr", 2026): 3, ("six_nations", 2025): 5, ("six_nations", 2026): 5}


def _close(actual, expected, message):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True):
        raise ValueError(message)


def _raw(path):
    return [RawPrediction.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires finite paired clusters")
    rng = np.random.default_rng(17)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(clusters=len(values), difference=float(values.mean()),
                p05=float(np.quantile(draws, .05)), p95=float(np.quantile(draws, .95)))


def require_complete(fantasy, stats, fixture_ids, candidate):
    fantasy_engines = {"empirical_baseline", CONTROL, candidate}
    raw_engines = {"empirical_raw", CONTROL, candidate}
    if set(fantasy.engine) != fantasy_engines or set(stats.engine) != raw_engines:
        raise ValueError("missing or unexpected comparator")
    if fantasy.duplicated(["slate", "engine"]).any():
        raise ValueError("duplicate fantasy evaluations")
    expected = {(competition, season, round_, engine)
                for (competition, season), count in SEASONS.items()
                for round_ in range(1, count + 1) for engine in fantasy_engines}
    observed = set(fantasy[["competition", "season", "round", "engine"]].itertuples(index=False, name=None))
    if observed != expected or not np.isfinite(fantasy[["mae", "team_points"]]).all().all():
        raise ValueError("all three complete tournament-seasons are required")
    summary, per_stat = summarise_stats(stats, fixture_ids)
    if not summary.count_support_complete.all() or not np.isfinite(
        summary[["count_stat_mae", "metres_mae", "minutes_mae"]]
    ).all().all():
        raise ValueError("Friendly-15 has incomplete observed support")
    return summary, per_stat


def _table(frame):
    def fmt(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def report(output: Path, previous: Path, destination: Path, candidate: str):
    fixed = json.loads((CONFIG_DIR / "fixtures.json").read_text())
    fixture_ids = [str(row["fixture_id"]) for row in fixed["fixtures"]]
    if len(fixture_ids) != 15 or len(set(fixture_ids)) != 15:
        raise ValueError("expected the fixed fifteen fixtures")
    official_names = {f"{c}_{y}_r{r}" for (c,y),n in SEASONS.items() for r in range(1,n+1)}
    required_jobs = official_names | {f"test_{f}" for f in fixture_ids}
    directories = {p.name: p for p in (output / "jobs").iterdir() if p.is_dir()}
    if set(directories) != required_jobs:
        raise ValueError("comparison must contain exactly 28 completed jobs")
    store = pd.read_csv(output / "inputs/player_match.csv", low_memory=False, dtype={key: str for key in KEY}, parse_dates=["date", "match_at"])
    slates = {s.name: s for s in official_slates(store, ("ncr", "six_nations"))}
    stat_frames, fantasy_frames, manifests = [], [], []
    audits = dict(stat_maes=0, fantasy_maes=0, squad_totals=0, ncr_budget_checks=0,
                  unchanged_controls=0, raw_rescores=0, source_truth_checks=0)
    for name, directory in sorted(directories.items()):
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests.append(manifest)
        old_manifest = json.loads((previous / "jobs" / name / "manifest.json").read_text())
        for key in ("store_sha256", "features_sha256", "weight_config_sha256", "packages",
                    "cutoff", "history_keys_sha256", "candidate_keys"):
            if manifest[key] != old_manifest[key]:
                raise ValueError(f"{name}: comparison input changed: {key}")
        if pd.Timestamp(manifest["training_match_at_max"]) + pd.Timedelta(hours=3) >= pd.Timestamp(manifest["cutoff"]):
            raise ValueError("history completion cutoff failed")
        metrics = pd.read_csv(directory / "metrics.csv", dtype={"fixture_id": str})
        if name.startswith("test_"):
            fixture = name.removeprefix("test_")
            truth = complete_fixture_cohort(store, fixture, ROOT / "data/cache")
            saved = pd.read_csv(directory / "truth.csv", dtype={key: str for key in KEY})
            wanted = list(truth[KEY].astype(str).itertuples(index=False, name=None))
            if list(saved[KEY].astype(str).itertuples(index=False, name=None)) != wanted or len(truth) != 46:
                raise ValueError("friendly teamsheet changed")
            for target in TARGETS:
                _close(saved[target], truth[target], "cached friendly truth differs")
                if not np.array_equal(saved[f"available__{target}"], truth[f"available__{target}"]):
                    raise ValueError("friendly availability differs")
            audits["source_truth_checks"] += 1
            for engine in ("empirical_raw", CONTROL, candidate):
                raw = _raw(directory / f"{engine}.jsonl")
                if [(p.fixture_id,p.player_id,p.team) for p in raw] != wanted:
                    raise ValueError("friendly raw key order differs")
                measured = metrics[metrics.engine.eq(engine)].set_index("target")
                if set(measured.index) != set(TARGETS) or len(measured) != len(TARGETS):
                    raise ValueError("missing or duplicate stat metric")
                for target in TARGETS:
                    actual = pd.to_numeric(truth[target], errors="coerce").to_numpy(float)
                    valid = truth[f"available__{target}"].to_numpy(bool) & np.isfinite(actual)
                    predicted = np.array([(p.minutes if target == "minutes" else p.events[target]).mean for p in raw])
                    _close(np.abs(actual[valid]-predicted[valid]).mean(), measured.loc[target,"mae"], "stat MAE differs")
                    if int(measured.loc[target,"n"]) != int(valid.sum()):
                        raise ValueError("stat support differs")
                    audits["stat_maes"] += 1
                if engine != candidate:
                    old = _raw(previous / "jobs" / name / f"{engine}.jsonl")
                    if [p.to_dict() for p in raw] != [p.to_dict() for p in old]:
                        raise ValueError(f"{name}/{engine}: fixed raw control changed")
                    audits["unchanged_controls"] += 1
            stat_frames.append(metrics)
        else:
            slate = slates[name]
            for engine in ("empirical_baseline", CONTROL, candidate):
                path = directory / name
                predictions = pd.read_csv(path / f"{engine}_predictions.csv")
                if predictions.id.astype(str).tolist() != slate.pool.id.astype(str).tolist():
                    raise ValueError("fantasy player cohort changed")
                _close(predictions.actual, slate.actual, "official fantasy outcome differs")
                metric = metrics[metrics.engine.eq(engine)]
                if len(metric) != 1:
                    raise ValueError("missing or duplicate fantasy result")
                _close(np.abs(predictions.predicted - predictions.actual).mean(), metric.iloc[0].mae, "fantasy MAE differs")
                audits["fantasy_maes"] += 1
                squad = pd.read_csv(path / f"{engine}_squad.csv")
                if len(squad) != 16 or squad.id.nunique() != 16 or squad.is_capt.sum() != 1 or squad.is_sub.sum() != 1:
                    raise ValueError("invalid squad roles/size")
                if (squad.is_capt & squad.is_sub).any() or not squad.loc[squad.is_capt,"status"].eq("P").all() or not squad.loc[squad.is_sub,"status"].eq("B").all():
                    raise ValueError("invalid captain/substitute status")
                positions = squad.loc[~squad.is_sub].groupby("pos").size().to_dict()
                if positions != REQUIRED or squad.team.value_counts().max() > (MAX_NATION if slate.competition == "ncr" else 4):
                    raise ValueError("invalid position/nation constraints")
                if slate.competition == "ncr":
                    if squad.value.sum() > BUDGET + 1e-9 or squad.hemi.value_counts().max() > MAX_HEMI:
                        raise ValueError("invalid NCR budget/hemisphere constraints")
                    audits["ncr_budget_checks"] += 1
                total = 0.0
                for row in squad.itertuples(index=False):
                    base, played, status = slate.team_actuals[str(int(float(row.id)))]
                    multiplier = 2.0 if row.is_capt else (0.0 if played <= 0 else (3.0 if status == "B" else .5)) if row.is_sub else 1.0
                    total += float(base) * multiplier
                _close(total, metric.iloc[0].team_points, "squad total differs")
                audits["squad_totals"] += 1
                if engine != "empirical_baseline":
                    raw = _raw(directory / f"{engine}.jsonl")
                    if [[p.fixture_id,p.player_id,p.team] for p in raw] != manifest["candidate_keys"]:
                        raise ValueError("raw fantasy keys differ")
                    _close(expected_points(raw, slate.competition), predictions.predicted, "raw fantasy rescore differs")
                    audits["raw_rescores"] += 1
                if engine != candidate:
                    old_dir = previous / "jobs" / name / name
                    pd.testing.assert_frame_equal(predictions, pd.read_csv(old_dir / f"{engine}_predictions.csv"), check_exact=True)
                    pd.testing.assert_frame_equal(squad, pd.read_csv(old_dir / f"{engine}_squad.csv"), check_exact=True)
                    audits["unchanged_controls"] += 1
            fantasy_frames.append(metrics)
    for key in ("source_sha256", "store_sha256", "features_sha256", "weight_config_sha256", "packages", "selection_sha256", "candidate_configs"):
        if any(m[key] != manifests[0][key] for m in manifests[1:]):
            raise ValueError(f"jobs disagree on {key}")
    fantasy, stats = pd.concat(fantasy_frames), pd.concat(stat_frames)
    friendly, per_stat = require_complete(fantasy, stats, fixture_ids, candidate)
    summary = fantasy.groupby(["competition", "season", "engine"]).agg(
        mae=("mae","mean"), team_points=("team_points","sum"), rounds=("round","nunique")).reset_index()
    intervals = []
    for (competition, season), group in fantasy.groupby(["competition", "season"]):
        for metric in ("mae", "team_points"):
            values = group.pivot(index="round",columns="engine",values=metric)
            for reference in ("empirical_baseline", CONTROL):
                intervals.append(dict(evaluation=f"{competition}_{season}", metric=metric, reference=reference,
                                      **paired_interval(values[candidate]-values[reference])))
    counts = stats[stats.target.isin(COUNT_TARGETS)].groupby(["engine","fixture_id"]).mae.mean().rename("count_stat_mae").reset_index()
    games = counts.merge(stats[stats.target.eq("metres")][["engine","fixture_id","mae"]].rename(columns={"mae":"metres_mae"}), on=["engine","fixture_id"])
    games = games.merge(stats[stats.target.eq("minutes")][["engine","fixture_id","mae"]].rename(columns={"mae":"minutes_mae"}), on=["engine","fixture_id"])
    for metric in ("count_stat_mae","metres_mae","minutes_mae"):
        values = games.pivot(index="fixture_id",columns="engine",values=metric)
        for reference in ("empirical_raw", CONTROL):
            intervals.append(dict(evaluation="Friendly-15",metric=metric,reference=reference,
                                  **paired_interval(values[candidate]-values[reference])))
    gate = {}
    for (competition,season), group in summary.groupby(["competition","season"]):
        board = group.set_index("engine")
        gate[f"{competition}_{season}"] = bool(board.loc[candidate,"mae"] < board.loc["empirical_baseline","mae"] and board.loc[candidate,"team_points"] > board.loc["empirical_baseline","team_points"])
    board = friendly.set_index("engine")
    gate["Friendly-15"] = bool(board.loc[candidate,"count_stat_mae"] < board.loc["empirical_raw","count_stat_mae"] and all(board.loc[candidate,f"{t}_mae"] <= 1.02 * board.loc["empirical_raw",f"{t}_mae"] for t in ("metres","minutes")))
    destination.mkdir(parents=True,exist_ok=True)
    for filename, frame in (("fantasy_summary.csv",summary),("fantasy_rounds.csv",fantasy),("friendly_summary.csv",friendly),
                            ("stat_mae.csv",per_stat),("friendly_games.csv",games),("intervals.csv",pd.DataFrame(intervals))):
        frame.to_csv(destination / filename,index=False)
    decision = dict(candidate=candidate,point_estimate_gates=gate,all_point_estimates_pass=all(gate.values()),
                    definitive_superiority_established=False,promotion_authorized=False,audits=audits)
    (destination/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    text = "# Candidate: complete four-evaluation comparison\n\n"
    text += "One prespecified native player-identity candidate; seed 17 and all other settings remain fixed. No post-result tuning. Lower MAE and higher team points are better.\n\n## Fantasy comparisons\n\n" + _table(summary)
    text += "\n\n## Friendly-15\n\n" + _table(friendly)
    text += "\n\n## Paired whole-round / whole-fixture intervals\n\nDifferences are candidate minus reference; 90% descriptive intervals, 10,000 resamples, seed 17. These are not adjusted for earlier model research or repeated historical evaluation.\n\n" + _table(pd.DataFrame(intervals))
    text += "\n\n## Audit and decision\n\n```json\n" + json.dumps(decision,indent=2) + "\n```\n"
    text += "\nAll 28 jobs are required: 13 fantasy rounds and the identical 15 friendlies. Each reported stat MAE was recalculated from raw forecasts and cache-rebuilt truth; fantasy MAEs, squad totals, role/position/nation constraints and NCR budgets were checked. The empirical and PR #20 controls match previous saved predictions and squads exactly. Current-run source/input/history manifests agree.\n"
    text += "\n## Limits\n\nSix Nations squads remain price-free diagnostics, not budget-feasible proof. NCR GW3 retains corrected retrospective lineup/prices; GW1 excludes New Zealand and France. Friendly and Six Nations lineups are oracle inputs and kickoff is a lock proxy. Unversioned historical data cannot prove original publication times. All models use the same earlier completed history; no current-game labels enter features. Existing event weights and these historical outcomes have already been examined. NCR GW4-7 is unused. Neither an improved point estimate nor one historical representation test establishes universal superiority. No production switch or automatic PR is authorized.\n"
    text += "\n## Reproduction\n\nUse the pinned Python 3.11 model environment and required LFS inputs. For each fixed fixture run `python -m model.unified.domain_experiment --study identity --output results --job test:<fixture_id>`; for fantasy use each of the thirteen existing slate names. Use a fresh output directory. `python -m model.unified.comparison_report --output results --previous previous_results --destination report --candidate p3_identity_native` requires the previous evidence for control checks.\n"
    (destination/"REPORT.md").write_text(text)
    print(text,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--candidate",required=True)
    args = parser.parse_args()
    report(args.output,args.previous,args.destination,args.candidate)


if __name__ == "__main__":
    main()
