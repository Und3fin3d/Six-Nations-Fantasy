"""Canonical timestamps, exclusive folds, manifests, and write-once outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..data import ROOT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v3"


def _utc(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, content: str | bytes) -> None:
    """Create an immutable artifact; refuse silent retrospective regeneration."""
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "x"
    with path.open(mode) as handle:
        handle.write(content)


def attach_match_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical match and fantasy-lock timestamps.

    The historical canonical store records dates but not kick-off times.  Those
    rows receive explicit ``date`` precision.  NCR rows are upgraded to the
    exact fixture and lock times from the fantasy fixture feed.  Six Nations
    rows share the first match date of their round as a conservative round lock,
    which guarantees the entire round is excluded from that fold's training.
    """
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["match_at"] = _utc(out["date"])
    out["lock_at"] = out["match_at"]
    out["timestamp_precision"] = "date"

    fixtures_path = DATA / "ncr" / "ncr_fixtures.csv"
    if fixtures_path.exists():
        fixtures = pd.read_csv(fixtures_path)
        fixtures["match_at_fx"] = _utc(fixtures["game_date"])
        fixtures["lock_at_fx"] = _utc(fixtures["lock_date"])
        fixture_rows = []
        for row in fixtures.itertuples(index=False):
            for team, opponent in ((row.home, row.away), (row.away, row.home)):
                fixture_rows.append({
                    "date_key": pd.Timestamp(row.match_at_fx).date(),
                    "team": str(team), "opponent": str(opponent),
                    "ncr_gameday": int(row.gameday),
                    "ncr_fantasy_match_id": str(row.match_id),
                    "match_at_fx": row.match_at_fx, "lock_at_fx": row.lock_at_fx,
                })
        lookup = pd.DataFrame(fixture_rows)
        # The not-yet-seeded finals contain repeated TBD/TBD placeholders.
        # They cannot match a historical store row and must not invalidate the
        # many-to-one join for real named fixtures.
        lookup = lookup.drop_duplicates(["date_key", "team", "opponent"], keep="first")
        out["date_key"] = out["date"].dt.date
        out = out.merge(
            lookup, on=["date_key", "team", "opponent"], how="left",
            validate="many_to_one",
        )
        matched = out["match_at_fx"].notna()
        out.loc[matched, "match_at"] = out.loc[matched, "match_at_fx"]
        out.loc[matched, "lock_at"] = out.loc[matched, "lock_at_fx"]
        out.loc[matched, "timestamp_precision"] = "fixture"
        out = out.drop(columns=["date_key", "match_at_fx", "lock_at_fx"])

    six = pd.to_numeric(out.get("competition_id"), errors="coerce").eq(1266)
    if six.any():
        group_cols = ["season", "round"]
        locks = out.loc[six].groupby(group_cols)["match_at"].transform("min")
        out.loc[six, "lock_at"] = locks
        out.loc[six, "timestamp_precision"] = "round_date"
    return out


def load_timed_store(path: Path = DATA / "unified" / "player_match.csv",
                     *, context_blocks: Iterable[str] = ()) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False, parse_dates=["date"])
    timed = attach_match_timestamps(raw)
    if context_blocks:
        from .context import augment_context
        timed = augment_context(timed, tuple(context_blocks))
    return timed


@dataclass(frozen=True)
class FoldSpec:
    competition: str
    season: int
    round: int
    lock_at: str
    evaluation_fixture_ids: tuple[str, ...]
    label: str

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.lock_at)


def fold_for_block(block: pd.DataFrame, timed_store: pd.DataFrame) -> FoldSpec:
    competition = str(block["competition"].iloc[0])
    season = int(block["season"].iloc[0])
    round_no = int(block["round"].iloc[0])
    if competition == "six_nations":
        fixture_ids = tuple(sorted(block["key_fixture"].astype(str).unique()))
        eval_rows = timed_store[timed_store["fixture_id"].astype(str).isin(fixture_ids)]
    elif competition == "ncr":
        eval_rows = timed_store[pd.to_numeric(
            timed_store.get("ncr_gameday"), errors="coerce").eq(round_no)]
        fixture_ids = tuple(sorted(eval_rows["fixture_id"].astype(str).unique()))
    else:
        raise ValueError(f"unsupported labelled competition {competition!r}")
    if eval_rows.empty:
        raise ValueError(f"{competition} {season} round {round_no}: no evaluation rows")
    lock = pd.to_datetime(eval_rows["lock_at"], utc=True).min()
    return FoldSpec(
        competition=competition, season=season, round=round_no,
        lock_at=lock.isoformat(), evaluation_fixture_ids=fixture_ids,
        label=f"{competition}_{season}_r{round_no}",
    )


def strict_training_frame(timed_store: pd.DataFrame, fold: FoldSpec) -> pd.DataFrame:
    match_at = pd.to_datetime(timed_store["match_at"], utc=True)
    train = timed_store[match_at < fold.cutoff].copy()
    train = train[~train["fixture_id"].astype(str).isin(fold.evaluation_fixture_ids)]
    if not train.empty:
        assert pd.to_datetime(train["match_at"], utc=True).max() < fold.cutoff
    if set(train["fixture_id"].astype(str)) & set(fold.evaluation_fixture_ids):
        raise AssertionError("evaluation fixtures leaked into training")
    return train


def write_fold_manifest(
    path: Path, fold: FoldSpec, train: pd.DataFrame, evaluation: pd.DataFrame,
    *, store_path: Path, artifacts: Iterable[Path] = (),
) -> dict:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "artifact_sha256": {str(p): sha256(p) for p in artifacts if p.exists()},
    }
    write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
