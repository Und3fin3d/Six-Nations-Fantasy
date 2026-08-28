"""Exact-kickoff tournament holdouts for historical raw benchmarking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..schema import EVENTS
from .config import EVALUATION_START, MIN_FIXTURES
from .coverage import sha256

NAMED_COMPETITIONS = {
    1266: "six_nations",
    1296: "rugby_championship",
    1272: "rugby_world_cup",
    1326: "pacific_nations_cup",
    1338: "british_irish_lions",
    696: "nations_championship",
}


@dataclass(frozen=True)
class HistoricalFold:
    label: str
    tournament: str
    calendar_year: int
    cutoff: str
    fixture_ids: tuple[str, ...]
    slate_ids: tuple[str, ...]
    hemisphere: str

    @property
    def cutoff_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff)


def _block_name(row: pd.Series) -> str | None:
    comp_id = pd.to_numeric(pd.Series([row["competition_id_cache"]]), errors="coerce").iloc[0]
    if pd.notna(comp_id) and int(comp_id) in NAMED_COMPETITIONS:
        return NAMED_COMPETITIONS[int(comp_id)]
    if pd.notna(comp_id) and int(comp_id) == 30:
        month = int(pd.Timestamp(row["match_at"]).month)
        if 6 <= month <= 8:
            return "summer_internationals"
        if 10 <= month <= 12:
            return "autumn_internationals"
    return None


def _fixture_slate(row: pd.Series, tournament: str, year: int, *, use_source_round: bool) -> str:
    if use_source_round:
        for column in ("source_game_week", "source_round"):
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return f"{tournament}_{year}_r{int(value)}"
    iso = pd.Timestamp(row["match_at"]).isocalendar()
    return f"{tournament}_{year}_w{int(iso.week):02d}"


def build_folds(store: pd.DataFrame, min_fixtures: int = MIN_FIXTURES) -> list[HistoricalFold]:
    fixtures = (
        store[store["competition_level"].eq("international")]
        .sort_values("match_at")
        .drop_duplicates("fixture_id")
        .copy()
    )
    fixtures["match_at"] = pd.to_datetime(fixtures["match_at"], utc=True)
    fixtures = fixtures[fixtures["match_at"].ge(pd.Timestamp(EVALUATION_START))]
    fixtures["tournament"] = fixtures.apply(_block_name, axis=1)
    fixtures = fixtures.dropna(subset=["tournament"])
    fixtures["calendar_year"] = fixtures["match_at"].dt.year.astype(int)
    folds: list[HistoricalFold] = []
    for (tournament, year), block in fixtures.groupby(["tournament", "calendar_year"], sort=True):
        if block["fixture_id"].nunique() < min_fixtures:
            continue
        block = block.sort_values(["match_at", "fixture_id"])
        source_rounds = pd.to_numeric(block["source_game_week"], errors="coerce")
        if source_rounds.nunique() <= 1:
            source_rounds = pd.to_numeric(block["source_round"], errors="coerce")
        use_source_round = bool(
            source_rounds.nunique() > 1
            or (block["match_at"].max() - block["match_at"].min()) <= pd.Timedelta(days=7)
        )
        slate_by_fixture = {
            str(row.fixture_id): _fixture_slate(
                row._asdict(), str(tournament), int(year), use_source_round=use_source_round,
            )
            for row in block.itertuples(index=False)
        }
        teams = store[store["fixture_id"].astype(str).isin(block["fixture_id"].astype(str))]
        hemispheres = set(teams["hemisphere"].dropna().astype(str)) - {"other"}
        hemisphere = next(iter(hemispheres)) if len(hemispheres) == 1 else "mixed"
        fixture_ids = tuple(block["fixture_id"].astype(str))
        folds.append(HistoricalFold(
            label=f"{tournament}_{int(year)}",
            tournament=str(tournament), calendar_year=int(year),
            cutoff=block["match_at"].min().isoformat(), fixture_ids=fixture_ids,
            slate_ids=tuple(slate_by_fixture[fixture] for fixture in fixture_ids),
            hemisphere=hemisphere,
        ))
    if len({fold.label for fold in folds}) != len(folds):
        raise AssertionError("historical fold labels are not unique")
    return sorted(folds, key=lambda fold: (fold.cutoff_timestamp, fold.label))


def strict_training_frame(store: pd.DataFrame, fold: HistoricalFold) -> pd.DataFrame:
    match_at = pd.to_datetime(store["match_at"], utc=True)
    fixture = store["fixture_id"].astype(str)
    train = store[match_at.lt(fold.cutoff_timestamp) & ~fixture.isin(fold.fixture_ids)].copy()
    if not train.empty and pd.to_datetime(train["match_at"], utc=True).max() >= fold.cutoff_timestamp:
        raise AssertionError("training row is not strictly before tournament cutoff")
    if set(train["fixture_id"].astype(str)) & set(fold.fixture_ids):
        raise AssertionError("evaluation fixture leaked into training")
    return train


def evaluation_frame(store: pd.DataFrame, fold: HistoricalFold) -> pd.DataFrame:
    rows = store[store["fixture_id"].astype(str).isin(fold.fixture_ids)].copy()
    slate = dict(zip(fold.fixture_ids, fold.slate_ids))
    rows["slate_id"] = rows["fixture_id"].astype(str).map(slate)
    if rows.empty or rows["fixture_id"].nunique() != len(fold.fixture_ids):
        raise ValueError(f"{fold.label}: incomplete evaluation fixture cohort")
    return rows


def masked_candidates(evaluation: pd.DataFrame) -> pd.DataFrame:
    candidates = evaluation.copy()
    for target in ("minutes", *EVENTS):
        if target in candidates:
            candidates[target] = pd.NA
        candidates[f"available__{target}"] = False
    candidates["source"] = "raw_benchmark_candidate"
    candidates["source_priority"] = 999
    return candidates


def write_fold_manifest(
    path: Path, fold: HistoricalFold, train: pd.DataFrame, evaluation: pd.DataFrame,
    *, store_path: Path, artifact_paths: tuple[Path, ...] = (), engine: str,
) -> dict:
    payload = {
        "schema_version": 1,
        "engine": engine,
        "fold": asdict(fold),
        "training_rows": int(len(train)),
        "evaluation_rows": int(len(evaluation)),
        "training_fixture_ids": sorted(train["fixture_id"].astype(str).unique().tolist()),
        "evaluation_fixture_ids": sorted(evaluation["fixture_id"].astype(str).unique().tolist()),
        "training_match_at_max": (
            pd.to_datetime(train["match_at"], utc=True).max().isoformat() if len(train) else None
        ),
        "store": str(store_path),
        "store_sha256": sha256(store_path),
        "artifact_sha256": {str(p): sha256(p) for p in artifact_paths if p.exists()},
        "oracle_lineup": True,
        "evaluation_targets_masked_for_features": True,
    }
    payload = json.loads(json.dumps(payload))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise FileExistsError(f"immutable fold manifest differs: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
