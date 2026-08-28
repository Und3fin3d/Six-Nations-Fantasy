"""Stage-2 pooled ranking+magnitude model over stage-1 rugby-event forecasts.

Stage 1 (``UniversalGBDT``) predicts competition-independent match events.
Stage 2 learns the event->fantasy-points *ranking* directly from official
points, pooled across Six Nations and the Nations Championship and conditioned
on the scoring rubric (never on a competition identity, which would just
re-create two specialists).  Two LightGBM heads -- a lambdarank ranker and a
points regressor -- are combined; magnitude comes from the regressor, which
predicts official points directly and is anchored by the already
competition-scaled ``s1_exp_points`` feature (so no per-competition calibrator
is needed, and points stay in-range for a competition never trained on).

The feature builder attaches stage-1 outputs to each labelled player-round:

* Six Nations rows join the store directly on the API ``(fixture_id, player_id)``.
* NCR rows reach the store through the fantasy->API crosswalk; players the
  crosswalk cannot place fall back to position priors (``s1_matched == 0``) so
  the evaluation cohort is never silently reduced.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT
from .features import build_pit_features
from .gbdt import UniversalGBDT
from .rubric import rubric_columns, rubric_vector
from .scoring import scorer_for

DATA = ROOT / "data"

# Stage-1 event expectations carried as stage-2 features.
S1_EVENTS = (
    "tries", "try_assists", "tackles", "metres", "defenders_beaten",
    "tackle_turnover", "clean_breaks", "offload", "penalties_conceded",
)
POSITION_CODES = {
    "Prop": 1, "Hooker": 2, "Second-row": 3, "Back-row": 4, "Scrum-half": 5,
    "Fly-half": 6, "Centre": 7, "Back-three": 8,
}
S1_FEATURES = (
    ["s1_exp_points", "s1_minutes", "s1_start_rate", "s1_matched"]
    + [f"s1_ev__{e}" for e in S1_EVENTS]
)
META_FEATURES = ["is_forward", "position_code"]


def _position_code(pos: object) -> int:
    text = str(pos).strip().lower()
    if any(k in text for k in ("prop",)):
        return 1
    if "hook" in text:
        return 2
    if "lock" in text or "second" in text:
        return 3
    if any(k in text for k in ("row", "flank", "number", "loose forward", "back row")):
        return 4
    if "scrum" in text:
        return 5
    if "fly" in text or "10" in text:
        return 6
    if "centre" in text or "center" in text or "midfield" in text:
        return 7
    if any(k in text for k in ("back three", "wing", "full", "back-three")):
        return 8
    return 0


def _stage1_table(store_feat: pd.DataFrame, model: UniversalGBDT, competition: str) -> pd.DataFrame:
    """Run stage 1 over PIT-featured store rows and summarise each prediction."""
    if store_feat.empty:
        return pd.DataFrame(columns=["fixture_id", "player_id", *S1_FEATURES])
    scorer = scorer_for(competition)
    preds = model.predict_frame(store_feat)
    rows = []
    start_rate = store_feat.get("recent_start_rate")
    for i, pred in enumerate(preds):
        summary = scorer.score_prediction(pred, n=800, seed=101 + i)
        row = {
            "fixture_id": str(pred.fixture_id), "player_id": str(pred.player_id),
            "s1_exp_points": float(summary.mean),
            "s1_minutes": float(pred.minutes.mean),
            "s1_start_rate": float(start_rate.iloc[i]) if start_rate is not None
            and pd.notna(start_rate.iloc[i]) else np.nan,
            "s1_matched": 1.0,
        }
        for event in S1_EVENTS:
            row[f"s1_ev__{event}"] = float(pred.events[event].mean) if event in pred.events else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill unmatched rows with position-level medians of the matched rows."""
    out = frame.copy()
    matched = out[out["s1_matched"] == 1.0]
    for col in S1_FEATURES:
        if col == "s1_matched":
            continue
        pos_median = matched.groupby("position_code")[col].median()
        global_median = float(matched[col].median()) if len(matched) else 0.0
        need = out[col].isna()
        out.loc[need, col] = out.loc[need, "position_code"].map(pos_median).fillna(global_median)
    out["s1_matched"] = out["s1_matched"].fillna(0.0)
    out["s1_start_rate"] = out["s1_start_rate"].fillna(out["s1_start_rate"].median())
    return out


def build_stage2_features(
    labels: pd.DataFrame, store_feat: pd.DataFrame, stage1: UniversalGBDT,
    *, crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach stage-1 outputs + rubric vector to labelled player-rounds.

    ``store_feat`` must already be PIT-featured and pre-filtered to the rows the
    labels can legitimately see (i.e. built from history strictly before each
    fixture, and — for the honest protocol — from a stage-1 whose cutoff does not
    postdate the labels).
    """
    labels = labels.copy()
    labels["position_code"] = labels["position"].map(_position_code)
    parts = []
    # Iterate per round (group_id), not per competition: an NCR block spanning
    # several gameweeks must attach each gameweek's own stage-1 features, never
    # the first gameweek's to all of them.
    for group_id, block in labels.groupby("group_id"):
        block = block.copy()
        competition = block["competition"].iloc[0]
        if competition == "six_nations":
            keys = block[["key_fixture", "key_player"]].rename(
                columns={"key_fixture": "fixture_id", "key_player": "player_id"})
            keys["fixture_id"] = keys["fixture_id"].astype(str)
            keys["player_id"] = keys["player_id"].astype(str)
            sub = store_feat[store_feat["fixture_id"].astype(str).isin(set(keys["fixture_id"]))
                             & store_feat["player_id"].astype(str).isin(set(keys["player_id"]))].copy()
            table = _stage1_table(sub, stage1, "six_nations")
            merged = block.merge(table, left_on=["key_fixture", "key_player"],
                                 right_on=["fixture_id", "player_id"], how="left")
        else:
            if crosswalk is None:
                crosswalk = _load_crosswalk()
            block = block.merge(crosswalk, left_on="key_player", right_on="fantasy_id", how="left")
            gw = int(block["round"].iloc[0])
            date = _ncr_gw_date(gw)
            sub = store_feat[(store_feat["date"].dt.normalize() == date)
                             & store_feat["source"].str.startswith("ncr_international")].copy()
            table = _stage1_table(sub, stage1, "ncr")
            merged = block.merge(table, left_on="api_player_id", right_on="player_id",
                                 how="left", suffixes=("", "_s1"))
        rubric = rubric_vector(competition)
        for col, val in zip(rubric_columns(), rubric):
            merged[col] = val
        parts.append(merged)
    features = pd.concat(parts, ignore_index=True, sort=False)
    features = _apply_priors(features)
    keep = ["competition", "group_id", "official_pts", "label_percentile", "label_z",
            "position", "position_code", "is_forward", "player_name", "team", "round",
            "season", "key_fixture", "key_player", "status"]
    keep = [c for c in keep if c in features.columns]
    return features[keep + S1_FEATURES + rubric_columns()]


def feature_columns() -> list[str]:
    return list(S1_FEATURES) + META_FEATURES + rubric_columns()


def _relevance(percentile: pd.Series) -> np.ndarray:
    """Decile relevance for lambdarank (0..9, higher = better)."""
    return np.clip(np.ceil(percentile.to_numpy() * 10) - 1, 0, 9).astype(int)


@dataclass
class RankStack:
    """Pooled ranker + points regressor over stage-1 forecasts.

    Magnitude comes from a regressor that predicts official points directly,
    anchored by the already-competition-scaled ``s1_exp_points`` feature and the
    rubric vector -- so it stays in the right units for a competition it was
    never trained on (the NCR GW1 pure-transfer case) without any per-competition
    calibrator.  ``blend_s1`` shrinks the learned points toward the deterministic
    stage-1 forecast to bound the variance of a head trained on very few rounds.
    """

    ranker_params: dict = field(default_factory=lambda: dict(
        objective="lambdarank", n_estimators=300, learning_rate=.03, num_leaves=15,
        min_child_samples=20, reg_lambda=3.0, subsample=.9, colsample_bytree=.8,
        random_state=17, n_jobs=1, verbosity=-1, label_gain=list(range(10))))
    reg_params: dict = field(default_factory=lambda: dict(
        objective="huber", alpha=.9, n_estimators=400, learning_rate=.03, num_leaves=15,
        min_child_samples=20, reg_lambda=3.0, subsample=.9, colsample_bytree=.8,
        random_state=17, n_jobs=1, verbosity=-1))
    blend_s1: float = 0.35
    columns: list[str] = field(default_factory=feature_columns)
    ranker: object = None
    regressor: object = None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.reindex(columns=self.columns).astype(float).fillna(0.0).to_numpy()

    def _fit_heads(self, frame: pd.DataFrame):
        import lightgbm as lgb
        ordered = frame.sort_values("group_id", kind="stable")
        X = self._matrix(ordered)
        group_sizes = ordered.groupby("group_id", sort=False).size().to_numpy()
        ranker = lgb.LGBMRanker(**self.ranker_params)
        ranker.fit(X, _relevance(ordered["label_percentile"]), group=group_sizes)
        regressor = lgb.LGBMRegressor(**self.reg_params)
        regressor.fit(X, ordered["official_pts"].astype(float).to_numpy())
        return ranker, regressor

    def fit(self, frame: pd.DataFrame) -> "RankStack":
        self.ranker, self.regressor = self._fit_heads(frame)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self._matrix(frame)
        reg_points = np.clip(self.regressor.predict(X), -10, None)
        s1 = frame["s1_exp_points"].to_numpy(float)
        expected = (1 - self.blend_s1) * reg_points + self.blend_s1 * s1
        rank_raw = self.ranker.predict(X)
        out = frame.copy()
        # Ordering blends the learned ranker with the points forecast; the
        # components are exposed separately so experiments can ablate them.
        rz = (rank_raw - rank_raw.mean()) / (rank_raw.std() or 1.0)
        pz = (expected - expected.mean()) / (expected.std() or 1.0)
        out["rank_component"] = rz
        out["points_component"] = pz
        out["stack_score"] = rz + pz
        out["expected_points"] = expected
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "RankStack":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not a RankStack artifact")
        return model


# --- store / crosswalk helpers -------------------------------------------------

def _load_crosswalk() -> pd.DataFrame:
    cw = pd.read_csv(DATA / "ncr/ncr_player_crosswalk.csv")
    cw = cw.dropna(subset=["api_player_id"]).copy()
    cw["fantasy_id"] = cw["fantasy_id"].astype(int).astype(str)
    cw["api_player_id"] = cw["api_player_id"].astype(int).astype(str)
    return cw[["fantasy_id", "api_player_id"]]


def _ncr_gw_date(gw: int) -> pd.Timestamp:
    fx = pd.read_csv(DATA / "ncr/ncr_fixtures.csv")
    dates = pd.to_datetime(fx[fx["gameday"] == gw]["game_date"]).dt.normalize().unique()
    if len(dates) != 1:
        # A gameweek split across days (e.g. a Fri/Sat November round) would
        # push most fixtures onto position priors under the exact-date filter.
        raise ValueError(f"GW{gw} spans {len(dates)} match dates; store filter needs a per-date join")
    return pd.Timestamp(dates[0])


def load_store_features(store_path: Path = DATA / "unified/player_match.csv",
                        *, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """PIT-feature the canonical store, optionally truncated at ``asof``.

    Truncation keeps stage-1 feature generation honest: rows dated on/after the
    stage-1 cutoff are removed so no forecast for a training row can be built
    from contemporaneous history the model was not entitled to see.
    """
    raw = pd.read_csv(store_path, low_memory=False, parse_dates=["date"])
    feat = build_pit_features(raw)
    if asof is not None:
        # Keep prediction rows on the cutoff day (fixtures kick off that day) but
        # drop anything strictly after it.
        feat = feat[feat["date"] <= pd.Timestamp(asof)]
    return feat
