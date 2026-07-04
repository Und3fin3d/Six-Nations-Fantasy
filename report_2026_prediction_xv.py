#!/usr/bin/env python3
"""Build 2026 Six Nations model-picked XVs and compare them with actual points.

The script reads a season prediction artifact such as
`data/model_predictions_2026.csv`, selects a fantasy XV for each round using the
branch's best available model selection score, adds a captain and supersub, and
also selects an overall XV from season aggregates.  For both views it reports
predicted points beside the realised official fantasy points and writes CSV
outputs by default.

Usage:
  python report_2026_prediction_xv.py
  python report_2026_prediction_xv.py --score-col target_pts_hat
  python report_2026_prediction_xv.py --no-write --summary-only
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "data" / "model_predictions_2026.csv"
DEFAULT_CONTEXT = ROOT / "data" / "model_player_match.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data"

XV_QUOTA = {
    "Prop": 2,
    "Hooker": 1,
    "Second-row": 2,
    "Back-row": 3,
    "Scrum-half": 1,
    "Fly-half": 1,
    "Centre": 2,
    "Back-three": 3,
}
POSITION_ORDER = {pos: i for i, pos in enumerate(XV_QUOTA)}
CAPTAIN_MULTIPLIER = 2.0
SUPERSUB_MULTIPLIER = 3.0

# `sel_score` is the research selector when present.  It may not be calibrated
# points, so the report still displays `target_pts_hat` as the predicted score.
SELECTION_SCORE_PRIORITY = (
    "sel_score",
    "selector_pts_hat",
    "target_pts_hat",
    "recon_pts_hat",
    "rank_score",
)
PREDICTED_POINTS_PRIORITY = (
    "target_pts_hat",
    "selector_pts_hat",
    "recon_pts_hat",
    "sel_score",
    "rank_score",
)
CONTEXT_COLUMNS = (
    "season",
    "round",
    "fixture_id",
    "player_id",
    "date",
    "team",
    "opponent",
    "home_away",
    "jersey",
    "started",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 2026 Six Nations predicted XV reports."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Prediction CSV to read.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=DEFAULT_CONTEXT,
        help="Optional model_player_match CSV used to add team/fixture context.",
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--score-col",
        default="auto",
        help="Column used to pick XVs. Defaults to the best available selector.",
    )
    parser.add_argument(
        "--predicted-col",
        default="auto",
        help="Column displayed as predicted points. Defaults to target_pts_hat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSVs.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report but do not write CSV files.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = load_predictions(args.predictions, args.season)
    pred = add_context(pred, args.context, args.season)

    selection_col = resolve_column(
        pred, args.score_col, SELECTION_SCORE_PRIORITY, "selection score"
    )
    predicted_col = resolve_column(
        pred, args.predicted_col, PREDICTED_POINTS_PRIORITY, "predicted points"
    )
    pred = coerce_numeric(pred, [selection_col, predicted_col, "official_pts"])
    pred = labelled_rows(pred)

    round_xv = build_round_xvs(pred, selection_col, predicted_col)
    overall_xv = build_overall_xv(pred, selection_col, predicted_col)
    summary = build_summary(pred, round_xv, overall_xv, predicted_col)

    print_report(
        summary,
        round_xv,
        overall_xv,
        selection_col,
        predicted_col,
        args.season,
        summary_only=args.summary_only,
    )

    if not args.no_write:
        write_outputs(args.output_dir, args.season, round_xv, overall_xv, summary)


def load_predictions(path: Path, season: int) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"prediction file not found: {path}")
    df = pd.read_csv(path)
    required = {
        "season",
        "round",
        "fixture_id",
        "player_id",
        "player_name",
        "canonical_pos",
        "official_pts",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    df = df[df["season"].astype(int) == season].copy()
    if df.empty:
        raise SystemExit(f"{path} has no rows for season {season}")
    return df


def add_context(pred: pd.DataFrame, path: Path, season: int) -> pd.DataFrame:
    if not path.exists():
        return pred
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in CONTEXT_COLUMNS if c in header]
    if not {"season", "round", "fixture_id", "player_id"}.issubset(usecols):
        return pred

    ctx = pd.read_csv(path, usecols=usecols)
    ctx = ctx[ctx["season"].astype(int) == season].drop_duplicates(
        ["season", "round", "fixture_id", "player_id"]
    )
    keys = ["season", "round", "fixture_id", "player_id"]
    add_cols = [c for c in ctx.columns if c not in pred.columns or c in keys]
    return pred.merge(ctx[add_cols], on=keys, how="left", validate="one_to_one")


def resolve_column(
    df: pd.DataFrame, requested: str, priority: Iterable[str], label: str
) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise SystemExit(f"{label} column not found: {requested}")
        return requested
    for col in priority:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    raise SystemExit(f"could not find an auto {label} column in {list(df.columns)}")


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def labelled_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["official_pts"].notna()
    if "has_label" in df.columns:
        mask &= as_bool(df["has_label"])
    if "is_modern" in df.columns:
        mask &= as_bool(df["is_modern"])
    out = df[mask].copy()
    if out.empty:
        raise SystemExit("no labelled rows with official_pts are available to compare")
    return out


def build_round_xvs(
    pred: pd.DataFrame, selection_col: str, predicted_col: str
) -> pd.DataFrame:
    teams = []
    for round_no, group in pred.groupby("round", sort=True):
        xv = pick_xv(group, selection_col, predicted_col)
        xv = add_captain_and_supersub(xv, group, selection_col, predicted_col)
        xv["round"] = round_no
        xv["view"] = "round"
        teams.append(xv)
    return tidy_team(pd.concat(teams, ignore_index=True))


def build_overall_xv(
    pred: pd.DataFrame, selection_col: str, predicted_col: str
) -> pd.DataFrame:
    group_cols = ["player_id", "player_name", "canonical_pos"]
    optional = [c for c in ("team",) if c in pred.columns]

    named_aggs = {
        "rounds_played": ("round", "nunique"),
        "fixtures": ("fixture_id", "nunique"),
        "selection_score": (selection_col, "sum"),
        "predicted_pts": (predicted_col, "sum"),
        "actual_pts": ("official_pts", "sum"),
    }
    for col in optional:
        named_aggs[col] = (col, mode_value)

    agg = pred.groupby(group_cols, as_index=False).agg(**named_aggs)
    supersub_pool = aggregate_supersub_pool(pred, selection_col, predicted_col)
    xv = pick_xv(agg, "selection_score", "predicted_pts")
    xv = add_captain_and_supersub(xv, supersub_pool, "selection_score", "predicted_pts")
    xv["view"] = "overall"
    return tidy_team(xv)


def aggregate_supersub_pool(
    pred: pd.DataFrame, selection_col: str, predicted_col: str
) -> pd.DataFrame:
    pool = pred.copy()
    if "started" in pool.columns:
        pool = pool[~as_bool(pool["started"])].copy()
    if pool.empty:
        return pool

    group_cols = ["player_id", "player_name", "canonical_pos"]
    optional = [c for c in ("team",) if c in pool.columns]
    named_aggs = {
        "rounds_played": ("round", "nunique"),
        "fixtures": ("fixture_id", "nunique"),
        "selection_score": (selection_col, "sum"),
        "predicted_pts": (predicted_col, "sum"),
        "actual_pts": ("official_pts", "sum"),
        "started": ("started", "first"),
    }
    for col in optional:
        named_aggs[col] = (col, mode_value)
    return pool.groupby(group_cols, as_index=False).agg(**named_aggs)


def pick_xv(df: pd.DataFrame, selection_col: str, predicted_col: str) -> pd.DataFrame:
    picked = []
    actual_col = "official_pts" if "official_pts" in df.columns else "actual_pts"
    for pos, quota in XV_QUOTA.items():
        pool = df[df["canonical_pos"] == pos].copy()
        if pool.empty:
            continue
        pool = pool.sort_values(
            [selection_col, predicted_col, actual_col, "player_name"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        pos_pick = pool.head(quota).copy()
        pos_pick["pos_slot"] = np.arange(1, len(pos_pick) + 1)
        pos_pick["quota"] = quota
        picked.append(pos_pick)
    if not picked:
        raise SystemExit("no players could be picked for any XV position")
    out = pd.concat(picked, ignore_index=True)
    out["position_sort"] = out["canonical_pos"].map(POSITION_ORDER)
    return out.sort_values(["position_sort", "pos_slot"]).reset_index(drop=True)


def add_captain_and_supersub(
    xv: pd.DataFrame,
    pool: pd.DataFrame,
    selection_col: str,
    predicted_col: str,
) -> pd.DataFrame:
    team = prepare_team_rows(xv, selection_col, predicted_col)
    team["team_role"] = "XV"
    team["is_captain"] = False
    team["is_supersub"] = False
    team["score_multiplier"] = 1.0

    captain_idx = top_model_index(team, selection_col, predicted_col)
    if captain_idx is not None:
        team.loc[captain_idx, "team_role"] = "Captain"
        team.loc[captain_idx, "is_captain"] = True
        team.loc[captain_idx, "score_multiplier"] = CAPTAIN_MULTIPLIER

    supersub = pick_supersub(pool, set(team["player_id"]), selection_col, predicted_col)
    if supersub is not None:
        supersub = prepare_team_rows(supersub, selection_col, predicted_col)
        supersub["team_role"] = "Supersub"
        supersub["is_captain"] = False
        supersub["is_supersub"] = True
        supersub["score_multiplier"] = SUPERSUB_MULTIPLIER
        supersub["pos_slot"] = np.nan
        supersub["quota"] = 1
        team = pd.concat([team, supersub], ignore_index=True)

    team["fantasy_predicted_pts"] = team["predicted_pts"] * team["score_multiplier"]
    team["fantasy_actual_pts"] = team["actual_pts"] * team["score_multiplier"]
    team["actual_minus_predicted"] = team["actual_pts"] - team["predicted_pts"]
    team["fantasy_actual_minus_predicted"] = (
        team["fantasy_actual_pts"] - team["fantasy_predicted_pts"]
    )
    team["role_sort"] = np.where(
        team["is_supersub"], len(POSITION_ORDER), team["canonical_pos"].map(POSITION_ORDER)
    )
    return team.sort_values(["role_sort", "pos_slot"], na_position="last").reset_index(drop=True)


def prepare_team_rows(
    df: pd.DataFrame, selection_col: str, predicted_col: str
) -> pd.DataFrame:
    out = df.copy()
    if "selection_score" not in out.columns:
        out["selection_score"] = out[selection_col]
    if "predicted_pts" not in out.columns:
        out["predicted_pts"] = out[predicted_col]
    if "actual_pts" not in out.columns:
        actual_col = "official_pts" if "official_pts" in out.columns else "actual_pts"
        out["actual_pts"] = out[actual_col]
    return out


def top_model_index(
    df: pd.DataFrame, selection_col: str, predicted_col: str
) -> Optional[int]:
    if df.empty:
        return None
    sorted_df = df.sort_values(
        [selection_col, predicted_col, "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return int(sorted_df.index[0])


def pick_supersub(
    pool: pd.DataFrame,
    selected_player_ids: set,
    selection_col: str,
    predicted_col: str,
) -> Optional[pd.DataFrame]:
    if pool.empty:
        return None
    candidates = pool[~pool["player_id"].isin(selected_player_ids)].copy()
    if "started" in candidates.columns:
        bench = candidates[~as_bool(candidates["started"])].copy()
        if not bench.empty:
            candidates = bench
    if candidates.empty:
        return None
    idx = top_model_index(candidates, selection_col, predicted_col)
    return candidates.loc[[idx]].copy() if idx is not None else None


def tidy_team(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "view",
        "round",
        "team_role",
        "canonical_pos",
        "pos_slot",
        "player_name",
        "team",
        "opponent",
        "home_away",
        "jersey",
        "started",
        "rounds_played",
        "is_captain",
        "is_supersub",
        "score_multiplier",
        "selection_score",
        "predicted_pts",
        "actual_pts",
        "actual_minus_predicted",
        "fantasy_predicted_pts",
        "fantasy_actual_pts",
        "fantasy_actual_minus_predicted",
        "fixture_id",
        "player_id",
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = [
        c for c in df.columns if c not in cols and c not in {"position_sort", "role_sort"}
    ]
    return df[cols + rest]


def build_summary(
    pred: pd.DataFrame,
    round_xv: pd.DataFrame,
    overall_xv: pd.DataFrame,
    predicted_col: str,
) -> pd.DataFrame:
    rows = []
    for round_no, group in round_xv.groupby("round", sort=True):
        pool = pred[pred["round"] == round_no]
        raw_actual_total = float(group["actual_pts"].sum())
        raw_predicted_total = float(group["predicted_pts"].sum())
        actual_total = float(group["fantasy_actual_pts"].sum())
        predicted_total = float(group["fantasy_predicted_pts"].sum())
        optimal_total = hindsight_team_total(pool)
        rows.append(
            {
                "view": "round",
                "round": round_no,
                "picked_players": int(len(group)),
                "raw_predicted_total": raw_predicted_total,
                "raw_actual_total": raw_actual_total,
                "predicted_total": predicted_total,
                "actual_total": actual_total,
                "actual_minus_predicted_total": actual_total - predicted_total,
                "hindsight_actual_team_total": optimal_total,
                "actual_value_ratio": safe_ratio(actual_total, optimal_total),
            }
        )

    round_rows = pd.DataFrame(rows)
    totals = {
        "view": "rounds_total",
        "round": "all",
        "picked_players": int(round_xv.shape[0]),
        "raw_predicted_total": float(round_rows["raw_predicted_total"].sum()),
        "raw_actual_total": float(round_rows["raw_actual_total"].sum()),
        "predicted_total": float(round_rows["predicted_total"].sum()),
        "actual_total": float(round_rows["actual_total"].sum()),
        "actual_minus_predicted_total": float(
            round_rows["actual_minus_predicted_total"].sum()
        ),
        "hindsight_actual_team_total": float(round_rows["hindsight_actual_team_total"].sum()),
        "actual_value_ratio": safe_ratio(
            round_rows["actual_total"].sum(),
            round_rows["hindsight_actual_team_total"].sum(),
        ),
    }
    overall_raw_actual = float(overall_xv["actual_pts"].sum())
    overall_raw_predicted = float(overall_xv["predicted_pts"].sum())
    overall_actual = float(overall_xv["fantasy_actual_pts"].sum())
    overall_predicted = float(overall_xv["fantasy_predicted_pts"].sum())
    overall_optimal = hindsight_overall_team_total(pred)
    overall = {
        "view": "overall_xv",
        "round": "season",
        "picked_players": int(len(overall_xv)),
        "raw_predicted_total": overall_raw_predicted,
        "raw_actual_total": overall_raw_actual,
        "predicted_total": overall_predicted,
        "actual_total": overall_actual,
        "actual_minus_predicted_total": overall_actual - overall_predicted,
        "hindsight_actual_team_total": overall_optimal,
        "actual_value_ratio": safe_ratio(overall_actual, overall_optimal),
    }
    return pd.concat(
        [round_rows, pd.DataFrame([totals, overall])], ignore_index=True
    )


def aggregate_for_overall(
    pred: pd.DataFrame, predicted_col: str, actual_col: str
) -> pd.DataFrame:
    return (
        pred.groupby(["player_id", "player_name", "canonical_pos"], as_index=False)
        .agg(predicted_pts=(predicted_col, "sum"), official_pts=(actual_col, "sum"))
        .rename(columns={"predicted_pts": predicted_col})
    )


def hindsight_team_total(df: pd.DataFrame) -> float:
    xv = pick_xv(df, "official_pts", "official_pts")
    selected_ids = set(xv["player_id"])
    base_total = float(xv["official_pts"].sum())
    captain_extra = float(xv["official_pts"].max()) if len(xv) else 0.0
    supersub = best_actual_supersub(df, selected_ids)
    return base_total + captain_extra + SUPERSUB_MULTIPLIER * supersub


def hindsight_overall_team_total(pred: pd.DataFrame) -> float:
    agg = (
        pred.groupby(["player_id", "player_name", "canonical_pos"], as_index=False)
        .agg(official_pts=("official_pts", "sum"))
    )
    xv = pick_xv(agg, "official_pts", "official_pts")
    selected_ids = set(xv["player_id"])
    base_total = float(xv["official_pts"].sum())
    captain_extra = float(xv["official_pts"].max()) if len(xv) else 0.0
    supersub_pool = aggregate_supersub_pool(pred, "official_pts", "official_pts")
    supersub = best_actual_supersub(supersub_pool, selected_ids)
    return base_total + captain_extra + SUPERSUB_MULTIPLIER * supersub


def best_actual_supersub(df: pd.DataFrame, selected_player_ids: set) -> float:
    if df.empty:
        return 0.0
    actual_col = "official_pts" if "official_pts" in df.columns else "actual_pts"
    candidates = df[~df["player_id"].isin(selected_player_ids)].copy()
    if "started" in candidates.columns:
        bench = candidates[~as_bool(candidates["started"])].copy()
        if not bench.empty:
            candidates = bench
    if candidates.empty:
        return 0.0
    return float(candidates[actual_col].max())


def hindsight_xv_total(df: pd.DataFrame) -> float:
    total = 0.0
    for pos, quota in XV_QUOTA.items():
        pool = df[df["canonical_pos"] == pos]
        if pool.empty:
            continue
        total += float(pool.nlargest(min(quota, len(pool)), "official_pts")["official_pts"].sum())
    return total


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def mode_value(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def print_report(
    summary: pd.DataFrame,
    round_xv: pd.DataFrame,
    overall_xv: pd.DataFrame,
    selection_col: str,
    predicted_col: str,
    season: int,
    *,
    summary_only: bool,
) -> None:
    print(f"\n{season} Six Nations model XV report")
    print(f"  selecting XVs by: {selection_col}")
    print(f"  predicted points: {predicted_col}")

    print("\nSummary")
    summary_cols = [
        "view",
        "round",
        "picked_players",
        "raw_predicted_total",
        "raw_actual_total",
        "predicted_total",
        "actual_total",
        "actual_minus_predicted_total",
        "hindsight_actual_team_total",
        "actual_value_ratio",
    ]
    print(format_frame(summary[summary_cols]))

    if summary_only:
        return

    team_cols = [
        "team_role",
        "canonical_pos",
        "pos_slot",
        "player_name",
        "team",
        "started",
        "score_multiplier",
        "selection_score",
        "predicted_pts",
        "actual_pts",
        "fantasy_predicted_pts",
        "fantasy_actual_pts",
        "fantasy_actual_minus_predicted",
    ]
    team_cols = [c for c in team_cols if c in round_xv.columns]
    for round_no, group in round_xv.groupby("round", sort=True):
        print(f"\nRound {round_no} predicted XV")
        print(format_frame(group[team_cols]))

    overall_cols = [
        "team_role",
        "canonical_pos",
        "pos_slot",
        "player_name",
        "team",
        "rounds_played",
        "score_multiplier",
        "selection_score",
        "predicted_pts",
        "actual_pts",
        "fantasy_predicted_pts",
        "fantasy_actual_pts",
        "fantasy_actual_minus_predicted",
    ]
    overall_cols = [c for c in overall_cols if c in overall_xv.columns]
    print("\nOverall top predicted XV")
    print(format_frame(overall_xv[overall_cols]))


def format_frame(df: pd.DataFrame) -> str:
    out = df.copy()
    integerish = {"round", "picked_players", "pos_slot", "rounds_played", "jersey"}
    for col in out.select_dtypes(include=[np.number]).columns:
        if col in integerish:
            out[col] = out[col].map(lambda v: int(v) if pd.notna(v) else v)
        else:
            out[col] = out[col].map(lambda v: round(float(v), 2) if pd.notna(v) else v)
    return out.to_string(index=False)


def write_outputs(
    output_dir: Path,
    season: int,
    round_xv: pd.DataFrame,
    overall_xv: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        f"prediction_xv_{season}_rounds.csv": round_xv,
        f"prediction_xv_{season}_overall.csv": overall_xv,
        f"prediction_xv_{season}_summary.csv": summary,
    }
    for name, frame in files.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        print(f"  wrote {display_path(path)}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
