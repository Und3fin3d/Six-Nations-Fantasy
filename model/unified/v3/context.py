"""Point-in-time-safe optional context and a v3-specific feature encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import ROOT
from ..features import BASE_NUMERIC_FEATURES, CATEGORICAL_FEATURES

DATA = ROOT / "data"
CONTEXT_PREFIXES = ("rolecert_", "style_", "weather_", "wr_")


def _join_external(frame: pd.DataFrame, path, keys: list[str], prefix: str) -> pd.DataFrame:
    if not path.exists():
        return frame
    extra = pd.read_csv(path)
    selected = keys + [c for c in extra if c.startswith(prefix)]
    selected = [c for c in selected if c in extra]
    if not set(keys).issubset(selected):
        return frame
    extra = extra[selected].drop_duplicates(keys)
    left = frame.copy()
    for key in keys:
        if key in {"fixture_id", "player_id", "team_id", "opponent_id"}:
            left[key] = left[key].astype(str)
            extra[key] = extra[key].astype(str)
    return left.merge(extra, on=keys, how="left", validate="many_to_one")


def _add_wr(frame: pd.DataFrame) -> pd.DataFrame:
    path = DATA / "wr_rankings.csv"
    if not path.exists():
        return frame
    ratings = pd.read_csv(path, parse_dates=["snapshot_date"])
    ratings = ratings.dropna(subset=["snapshot_date", "team", "wr_pts"]).sort_values("snapshot_date")
    out = frame.copy()
    out["_row"] = np.arange(len(out))
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    def lookup(name_col: str, output: str) -> pd.DataFrame:
        pieces = []
        for team, rows in out.groupby(name_col, sort=False):
            history = ratings[ratings["team"].eq(team)][["snapshot_date", "wr_pts"]]
            if history.empty:
                part = rows[["_row"]].copy()
                part[output] = np.nan
            else:
                part = pd.merge_asof(
                    rows[["_row", "date"]].sort_values("date"),
                    history.sort_values("snapshot_date"),
                    left_on="date", right_on="snapshot_date", direction="backward",
                )[["_row", "wr_pts"]].rename(columns={"wr_pts": output})
            pieces.append(part)
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["_row", output])

    out = out.merge(lookup("team", "wr_team_pts"), on="_row", how="left", validate="one_to_one")
    out = out.merge(lookup("opponent", "wr_opp_pts"), on="_row", how="left", validate="one_to_one")
    out["wr_margin"] = out["wr_team_pts"] - out["wr_opp_pts"]
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def augment_context(frame: pd.DataFrame, blocks: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    if "wr" in blocks:
        out = _add_wr(out)
    if "roles" in blocks:
        out = _join_external(
            out, DATA / "external_player_roles.csv",
            ["season", "round", "fixture_id", "team_id", "player_id"], "rolecert_",
        )
    if "style" in blocks:
        out = _join_external(
            out, DATA / "external_team_style.csv",
            ["season", "round", "fixture_id", "team_id", "opponent_id"], "style_",
        )
    if "weather" in blocks:
        out = _join_external(
            out, DATA / "external_fixture_weather.csv",
            ["season", "round", "fixture_id"], "weather_",
        )
    return out


def restrict_context(frame: pd.DataFrame, blocks: tuple[str, ...]) -> pd.DataFrame:
    """Drop optional context columns not admitted for one component model."""
    allowed_prefixes = {
        "wr": "wr_", "roles": "rolecert_", "style": "style_", "weather": "weather_",
    }
    allowed = {allowed_prefixes[block] for block in blocks if block in allowed_prefixes}
    drop = [
        column for column in frame
        if column.startswith(CONTEXT_PREFIXES)
        and not any(column.startswith(prefix) for prefix in allowed)
    ]
    return frame.drop(columns=drop)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [c for c in BASE_NUMERIC_FEATURES if c in frame]
    dynamic = [
        c for c in frame
        if c.startswith(("form_per80__", "history_count__", *CONTEXT_PREFIXES))
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return list(dict.fromkeys(base + dynamic))


@dataclass
class V3FeatureEncoder:
    categories: dict[str, dict[str, int]] | None = None
    numeric_columns: list[str] | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "V3FeatureEncoder":
        self.categories = {}
        for col in CATEGORICAL_FEATURES:
            values = frame.get(col, pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
            self.categories[col] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
        self.numeric_columns = numeric_feature_columns(frame)
        numeric = frame.reindex(columns=self.numeric_columns).apply(pd.to_numeric, errors="coerce")
        matrix = numeric.to_numpy(float)
        self.means = numeric.mean(axis=0, skipna=True).fillna(0.0).to_numpy()
        filled = np.where(np.isnan(matrix), self.means, matrix)
        self.scales = np.std(filled, axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.categories is None or self.numeric_columns is None:
            raise RuntimeError("V3FeatureEncoder must be fit before transform")
        cat = np.column_stack([
            frame.get(col, pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
            .map(mapping).fillna(0).astype(int).to_numpy()
            for col, mapping in self.categories.items()
        ])
        numeric = frame.reindex(columns=self.numeric_columns).apply(pd.to_numeric, errors="coerce")
        matrix = numeric.to_numpy(float)
        filled = np.where(np.isnan(matrix), self.means, matrix)
        return cat.astype("int64"), ((filled - self.means) / self.scales).astype("float32")
