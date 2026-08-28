"""Point-in-time-safe empirical-Bayes form features for unified model v5.

Two pieces:

* :class:`ShrunkTables` — closed-form global tables (club→test calibration,
  position×event per-80 priors, jersey-slot minute priors, level-stratified
  mean totals) fit **per training window** on a frame that must already be
  restricted to rows strictly before the fold lock (plan ledger L1).
* :func:`add_shrunk_features` — row-wise features built only from *prior*
  rows of the same player (shifted, exponentially weighted). Appending future
  rows never changes an earlier row's features, matching the v1 contract in
  `model/unified/features.py`.

The shrinkage formula (plan §4, Family A)::

    shrunk_e = (wm_i·rate_i + c·wm_c·cal_e·rate_c + K·prior_{pos,e})
               / (wm_i + c·wm_c + K)

with ewm-weighted minutes ``wm``, club confidence ``c``, calibration factor
``cal_e`` and shrinkage strength ``K``. ``potm`` is excluded from rate
features: it is a per-fixture award, not a per-minute process.

Corrective-cycle note (`data/unified/v5/corrective_cycle.md`): kill-switch K1
fired on the frozen-protocol benchmark (zero low-history-slice gain), and the
shrunk/slot layers were implicated in the NCR top-10 capture collapse. Both
layers are therefore **off by default** (terminal rung T): this module's
feature blocks are emitted only when ``config.use_eb_features`` /
``config.use_slot_minutes`` are explicitly enabled for ablation forensics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import EVENTS
from .config import V5Config

RATE_FEATURE_EVENTS = tuple(event for event in EVENTS if event != "potm")
LEVEL_INTERNATIONAL = "international"
LEVEL_CLUB = "club"

SHRUNK_PREFIXES = ("shrink_per80__", "shrink_weight__")
SLOT_FEATURES = (
    "slot_p_play", "slot_p_start", "slot_min_start", "slot_min_bench",
    "slot_exp_minutes",
)
HISTORY_FEATURES = ("hist_intl_matches", "hist_club_matches")


def _ewm_sum(series: pd.Series, halflife: float) -> pd.Series:
    """Exponentially weighted sum over each player's *prior* rows only."""
    return series.shift(1).ewm(halflife=halflife, min_periods=1).sum()


@dataclass
class ShrunkTables:
    """Closed-form global tables fit on one training window."""

    cal: dict[str, float] = field(default_factory=dict)
    prior_rate: dict[str, dict[str, float]] = field(default_factory=dict)
    global_rate: dict[str, float] = field(default_factory=dict)
    slot_start_minutes: dict[str, float] = field(default_factory=dict)
    slot_bench_minutes: dict[str, float] = field(default_factory=dict)
    mean_total: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cal": self.cal,
            "prior_rate": self.prior_rate,
            "global_rate": self.global_rate,
            "slot_start_minutes": self.slot_start_minutes,
            "slot_bench_minutes": self.slot_bench_minutes,
            "mean_total": self.mean_total,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ShrunkTables":
        return cls(
            cal={k: float(v) for k, v in payload.get("cal", {}).items()},
            prior_rate={
                k: {p: float(x) for p, x in v.items()}
                for k, v in payload.get("prior_rate", {}).items()
            },
            global_rate={
                k: float(v) for k, v in payload.get("global_rate", {}).items()
            },
            slot_start_minutes={
                k: float(v) for k, v in payload.get("slot_start_minutes", {}).items()
            },
            slot_bench_minutes={
                k: float(v) for k, v in payload.get("slot_bench_minutes", {}).items()
            },
            mean_total={
                k: {p: float(x) for p, x in v.items()}
                for k, v in payload.get("mean_total", {}).items()
            },
        )


def fit_tables(frame: pd.DataFrame, config: V5Config) -> ShrunkTables:
    """Fit all global tables on one training window (caller enforces the lock)."""
    tables = ShrunkTables()
    level = frame["competition_level"].fillna("unknown").astype(str)
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    played = minutes > 0
    intl = level.eq(LEVEL_INTERNATIONAL)
    club = level.eq(LEVEL_CLUB)

    club_min = minutes.where(club).groupby(frame["player_id"]).sum()
    intl_min = minutes.where(intl).groupby(frame["player_id"]).sum()
    dual = club_min.index[
        (club_min >= config.dual_min_minutes)
        & (intl_min.reindex(club_min.index).fillna(0.0) >= config.dual_min_minutes)
    ]

    for event in RATE_FEATURE_EVENTS:
        available = frame.get(
            f"available__{event}", pd.Series(False, index=frame.index)
        ).fillna(False).astype(bool)
        y = pd.to_numeric(frame[event], errors="coerce").where(available).fillna(0.0)
        m = minutes.where(available).fillna(0.0)

        intl_rows = intl & played & available
        total_min = float(m[intl_rows].sum())
        global_rate = (
            80.0 * float(y[intl_rows].sum()) / total_min if total_min > 0 else 0.0
        )
        tables.global_rate[event] = global_rate

        priors: dict[str, float] = {}
        for position, block in frame[intl_rows].groupby("position"):
            pos_min = float(m[block.index].sum())
            pos_rate = (
                80.0 * float(y[block.index].sum()) / pos_min
                if pos_min > 0 else global_rate
            )
            priors[str(position)] = (
                pos_min * pos_rate + config.prior_shrink_minutes * global_rate
            ) / (pos_min + config.prior_shrink_minutes)
        priors["__global__"] = global_rate
        tables.prior_rate[event] = priors

        dual_mask = frame["player_id"].isin(dual)
        di = dual_mask & intl & played & available
        dc = dual_mask & club & played & available
        min_i, min_c = float(m[di].sum()), float(m[dc].sum())
        if min_i > 0 and min_c > 0:
            raw = (float(y[di].sum()) / min_i) / max(float(y[dc].sum()) / min_c, 1e-9)
            raw = float(np.clip(raw, config.cal_clip_lo, config.cal_clip_hi))
            weight = min(min_c / config.cal_shrink_minutes, 1.0)
            tables.cal[event] = 1.0 + (raw - 1.0) * weight
        else:
            tables.cal[event] = 1.0

        totals: dict[str, float] = {}
        for lvl in sorted(level[available].unique()):
            block = available & level.eq(lvl)
            totals[str(lvl)] = float(
                pd.to_numeric(frame.loc[block, event], errors="coerce").mean()
            )
        totals["__all__"] = (
            float(pd.to_numeric(frame.loc[available, event], errors="coerce").mean())
            if available.any() else 0.0
        )
        tables.mean_total[event] = totals

    started = frame["started"].astype(bool)
    bench_played = (~started) & played
    tables.slot_start_minutes["__global__"] = (
        float(minutes[started].mean()) if started.any() else 70.0
    )
    tables.slot_bench_minutes["__global__"] = (
        float(minutes[bench_played].mean()) if bench_played.any() else 24.0
    )
    for position, block in frame.groupby("position"):
        sub_start = started[block.index]
        sub_bench = bench_played[block.index]
        if sub_start.any():
            tables.slot_start_minutes[str(position)] = float(
                minutes[block.index][sub_start].mean()
            )
        if sub_bench.any():
            tables.slot_bench_minutes[str(position)] = float(
                minutes[block.index][sub_bench].mean()
            )
    return tables


def _concat_block(df: pd.DataFrame, block: dict[str, object]) -> pd.DataFrame:
    """Attach a batch of new columns in one concat (avoids fragmentation).

    Any pre-existing columns with the same names are dropped first, keeping
    the overwrite semantics of one-by-one assignment.
    """
    new = pd.DataFrame(block, index=df.index)
    overlap = new.columns.intersection(df.columns)
    if len(overlap):
        df = df.drop(columns=overlap)
    return pd.concat([df, new], axis=1)


def add_shrunk_features(
    frame: pd.DataFrame, tables: ShrunkTables, config: V5Config,
) -> pd.DataFrame:
    """Append level-split shrunk-rate, slot-minute and level-history features.

    Row-wise and point-in-time safe: every aggregate is shifted one match
    within each player before weighting, so appending future rows cannot
    change an earlier row's values. Row count is preserved exactly (sort +
    column ops only); the caller's (fixture_id, player_id, team) key must
    already be unique, as in the canonical store.

    Columns are assembled in dicts and attached with two batched
    ``pd.concat`` calls (scratch block, then feature block) rather than
    ~200 individual inserts, which keeps pandas from emitting
    fragmentation PerformanceWarnings.

    Layer gating (terminal rung T default): with both
    ``config.use_eb_features`` and ``config.use_slot_minutes`` off (the
    post-K1 default) the frame is returned unchanged; each block is emitted
    only under its own flag.
    """
    if not (config.use_eb_features or config.use_slot_minutes):
        return frame

    df = frame.sort_values(["date", "fixture_id", "team", "player_id"]).copy()
    level = df["competition_level"].fillna("unknown").astype(str)
    minutes = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    started_bool = df["started"].astype(bool)

    k = float(config.shrinkage_minutes_k)
    c = float(config.club_confidence)
    half = float(config.ewm_halflife_matches)

    scratch: dict[str, object] = {}
    scratch["_is_intl"] = level.eq(LEVEL_INTERNATIONAL).astype(float)
    scratch["_is_club"] = level.eq(LEVEL_CLUB).astype(float)
    scratch["_played"] = (minutes > 0).astype(float)
    scratch["_started"] = started_bool.astype(float)
    scratch["_bench_play"] = ((minutes > 0) & ~started_bool).astype(float)
    scratch["_min_start"] = minutes * scratch["_started"]
    scratch["_min_bench"] = minutes * scratch["_bench_play"]
    scratch["_squad"] = df.get(
        "available__minutes", pd.Series(True, index=df.index)
    ).fillna(True).astype(bool).astype(float)

    if config.use_eb_features:
        for event in RATE_FEATURE_EVENTS:
            available = df.get(
                f"available__{event}", pd.Series(False, index=df.index)
            ).fillna(False).astype(bool)
            y = pd.to_numeric(df[event], errors="coerce").where(available).fillna(0.0)
            m = minutes.where(available).fillna(0.0)
            scratch[f"_num_i_{event}"] = y * scratch["_is_intl"]
            scratch[f"_min_i_{event}"] = m * scratch["_is_intl"]
            scratch[f"_num_c_{event}"] = y * scratch["_is_club"]
            scratch[f"_min_c_{event}"] = m * scratch["_is_club"]

    temp_cols = list(scratch)
    df = _concat_block(df, scratch)

    players = df.groupby("player_id", sort=False)
    ew = {col: players[col].transform(lambda s: _ewm_sum(s, half)) for col in temp_cols}

    features: dict[str, object] = {}
    if config.use_eb_features:
        for event in RATE_FEATURE_EVENTS:
            cal = float(tables.cal.get(event, 1.0))
            priors = tables.prior_rate.get(event, {"__global__": 0.0})
            global_rate = float(priors.get("__global__", 0.0))
            prior = (
                df["position"].astype(str).map(priors).fillna(global_rate).to_numpy(float)
            )
            wm_i = ew[f"_min_i_{event}"].fillna(0.0).to_numpy(float)
            wm_c = ew[f"_min_c_{event}"].fillna(0.0).to_numpy(float)
            num_i = ew[f"_num_i_{event}"].fillna(0.0).to_numpy(float)
            num_c = ew[f"_num_c_{event}"].fillna(0.0).to_numpy(float)
            rate_i = np.where(wm_i > 0, 80.0 * num_i / np.maximum(wm_i, 1e-9), 0.0)
            rate_c = np.where(wm_c > 0, 80.0 * num_c / np.maximum(wm_c, 1e-9), 0.0)
            denom = wm_i + c * wm_c + k
            features[f"shrink_per80__{event}"] = (
                wm_i * rate_i + c * wm_c * cal * rate_c + k * prior
            ) / denom
            features[f"shrink_weight__{event}"] = (wm_i + c * wm_c) / denom

        features["hist_intl_matches"] = ew["_is_intl"].fillna(0.0)
        features["hist_club_matches"] = ew["_is_club"].fillna(0.0)

    if config.use_slot_minutes:
        def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
            num_v, den_v = num.to_numpy(float), den.to_numpy(float)
            out = np.where(den_v > 0, num_v / np.maximum(den_v, 1e-9), np.nan)
            return pd.Series(out, index=df.index)

        p_play = _ratio(ew["_played"], ew["_squad"])
        p_start = _ratio(ew["_started"], ew["_squad"])
        min_start = _ratio(ew["_min_start"], ew["_started"])
        min_bench = _ratio(ew["_min_bench"], ew["_bench_play"])

        positions = df["position"].astype(str)
        start_prior = positions.map(tables.slot_start_minutes).fillna(
            tables.slot_start_minutes.get("__global__", 70.0)
        )
        bench_prior = positions.map(tables.slot_bench_minutes).fillna(
            tables.slot_bench_minutes.get("__global__", 24.0)
        )
        min_start = min_start.fillna(start_prior)
        min_bench = min_bench.fillna(bench_prior)

        features["slot_p_play"] = p_play
        features["slot_p_start"] = p_start
        features["slot_min_start"] = min_start
        features["slot_min_bench"] = min_bench
        features["slot_exp_minutes"] = (
            p_start.fillna(0.0) * min_start
            + (p_play.fillna(0.0) - p_start.fillna(0.0)).clip(0.0, 1.0) * min_bench
        )
    df = _concat_block(df, features)
    return df.drop(columns=temp_cols)


def shrunk_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns added by this module that the encoder should consume."""
    columns = [
        c for c in frame
        if c.startswith(SHRUNK_PREFIXES) and pd.api.types.is_numeric_dtype(frame[c])
    ]
    columns += [c for c in (*SLOT_FEATURES, *HISTORY_FEATURES) if c in frame]
    return list(dict.fromkeys(columns))
