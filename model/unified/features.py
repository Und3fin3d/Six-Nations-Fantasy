"""Point-in-time features shared by every universal model adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import EVENTS

CATEGORICAL_FEATURES = (
    "player_id", "position", "team", "opponent", "competition_level",
)
BASE_NUMERIC_FEATURES = (
    "is_forward", "started", "jersey", "days_since_last", "career_matches",
    "recent_minutes", "recent_start_rate", "team_recent_margin",
)


def build_pit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build prior-match-only features; every rolling statistic is shifted."""

    df = frame.sort_values(["date", "fixture_id", "team", "player_id"]).copy()
    df["team_margin"] = pd.to_numeric(df.get("team_score"), errors="coerce") - pd.to_numeric(
        df.get("opp_score"), errors="coerce")
    player_groups = df.groupby("player_id", sort=False)
    df["previous_date"] = player_groups["date"].shift(1)
    df["days_since_last"] = (df["date"] - df["previous_date"]).dt.days.clip(0, 1000)
    df["career_matches"] = player_groups.cumcount().astype(float)
    df["recent_minutes"] = player_groups["minutes"].transform(
        lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
    df["recent_start_rate"] = player_groups["started"].transform(
        lambda x: x.astype(float).shift(1).ewm(span=6, min_periods=1).mean())
    for event in EVENTS:
        values = pd.to_numeric(df[event], errors="coerce")
        valid = df[f"available__{event}"].astype(bool)
        per80 = values.where(valid) / df["minutes"].where(df["minutes"] > 0) * 80
        df[f"form_per80__{event}"] = per80.groupby(df["player_id"]).transform(
            lambda x: x.shift(1).ewm(halflife=4, min_periods=1).mean())
        df[f"history_count__{event}"] = valid.groupby(df["player_id"]).transform(
            lambda x: x.shift(1).fillna(False).cumsum()).astype(float)
    df["team_recent_margin"] = df.groupby("team", sort=False)["team_margin"].transform(
        lambda x: x.shift(1).ewm(span=8, min_periods=1).mean())
    return df.drop(columns=["previous_date", "team_margin"])


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in BASE_NUMERIC_FEATURES if c in frame] + [
        c for c in frame if c.startswith(("form_per80__", "history_count__"))
    ]


@dataclass
class FeatureEncoder:
    """Train-only category vocabularies and numeric normalization."""

    categories: dict[str, dict[str, int]] | None = None
    numeric_columns: list[str] | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        self.categories = {}
        for col in CATEGORICAL_FEATURES:
            values = frame.get(col, pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
            self.categories[col] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
        self.numeric_columns = numeric_feature_columns(frame)
        numeric_frame = frame[self.numeric_columns].astype(float)
        matrix = numeric_frame.to_numpy()
        self.means = numeric_frame.mean(axis=0, skipna=True).fillna(0.0).to_numpy()
        filled = np.where(np.isnan(matrix), self.means, matrix)
        self.scales = np.std(filled, axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.categories is None or self.numeric_columns is None:
            raise RuntimeError("FeatureEncoder must be fit before transform")
        cat = np.column_stack([
            frame.get(col, pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
            .map(mapping).fillna(0).astype(int).to_numpy()
            for col, mapping in self.categories.items()
        ])
        matrix = frame.reindex(columns=self.numeric_columns).astype(float).to_numpy()
        filled = np.where(np.isnan(matrix), self.means, matrix)
        return cat, (filled - self.means) / self.scales
