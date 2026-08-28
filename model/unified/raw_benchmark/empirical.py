"""Competition-independent empirical event-rate model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.rp_rates import _norm, _season_end_year

from ..contracts import EventDistribution, RawPrediction
from ..data import ROOT
from ..schema import distribution_family
from .config import EXTENDED_EVENTS, STABLE_EVENTS

DATA = ROOT / "data"
HALFLIFE_DAYS = 420.0
K = 220.0
CLUB_ATT_CAL, CLUB_DEF_CAL, CLUB_DSC_CAL = 0.84, 0.97, 1.0
CLUB_CONF = 0.55
RP_HALFLIFE_YEARS = 1.6

ATTACK_EVENTS = {
    "tries", "try_assists", "conversion_goals", "missed_conversion_goals",
    "penalty_goals", "missed_penalty_goals", "drop_goals_converted",
    "drop_goal_missed", "defenders_beaten", "clean_breaks", "offload",
    "runs", "passes", "bad_passes", "metres", "fifty_22", "kicks_retained",
}
DEFENCE_EVENTS = {
    "tackles", "missed_tackles", "dominant_tackles", "tackle_turnover",
    "tackle_try_saver", "rucks_won", "rucks_lost", "lineout_steals",
    "scrums_won",
}
DISCIPLINE_EVENTS = {"turnovers_conceded", "penalties_conceded", "yellow_cards", "red_cards"}
RP_ALIASES = {
    "runs": "carries", "offload": "offloads", "tackle_turnover": "turnovers_won",
}


def _calibration(event: str) -> float:
    if event in ATTACK_EVENTS:
        return float(CLUB_ATT_CAL)
    if event in DEFENCE_EVENTS:
        return float(CLUB_DEF_CAL)
    if event in DISCIPLINE_EVENTS:
        return float(CLUB_DSC_CAL)
    return 1.0


def _matchup_group(event: str) -> str:
    if event in ATTACK_EVENTS:
        return "attack"
    if event in DEFENCE_EVENTS:
        return "defence"
    return "neutral"


@dataclass
class EmpiricalEventModel:
    """Recency, club and player-prior model operating only on raw events."""

    asof: str | pd.Timestamp
    events: tuple[str, ...] = (*STABLE_EVENTS, *EXTENDED_EVENTS)
    profiles: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    position_priors: dict[tuple[str, str], float] = field(default_factory=dict)
    rp_priors: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    minutes_by_player: dict[tuple[str, bool], float] = field(default_factory=dict)
    minutes_by_position: dict[tuple[str, bool], float] = field(default_factory=dict)
    player_names: dict[str, str] = field(default_factory=dict)
    dispersion: dict[str, float] = field(default_factory=dict)
    wr_points: dict[str, float] = field(default_factory=dict)
    active_events: set[str] = field(default_factory=set)

    def _weighted_rates(self, frame: pd.DataFrame, level: str) -> dict[tuple[str, str], tuple[float, float]]:
        subset = frame[frame["competition_level"].eq(level)].copy()
        if subset.empty:
            return {}
        asof = pd.Timestamp(self.asof).tz_localize(None) if pd.Timestamp(self.asof).tzinfo else pd.Timestamp(self.asof)
        dates = pd.to_datetime(subset["date"], errors="coerce")
        subset["_weight"] = np.power(0.5, (asof - dates).dt.days.clip(lower=0) / HALFLIFE_DAYS)
        minutes = pd.to_numeric(subset["minutes"], errors="coerce").fillna(0)
        output: dict[tuple[str, str], tuple[float, float]] = {}
        for event in self.events:
            valid = (
                subset[f"available__{event}"].fillna(False).astype(bool)
                & minutes.gt(0) & pd.to_numeric(subset[event], errors="coerce").notna()
            )
            if not valid.any():
                continue
            block = subset[valid].copy()
            block["_weighted_minutes"] = block["_weight"] * pd.to_numeric(block["minutes"], errors="coerce")
            block["_weighted_event"] = block["_weight"] * pd.to_numeric(block[event], errors="coerce")
            grouped = block.groupby(block["player_id"].astype(str))[["_weighted_minutes", "_weighted_event"]].sum()
            for player, row in grouped.iterrows():
                exposure = float(row["_weighted_minutes"])
                if exposure > 0:
                    output[(str(player), event)] = (float(row["_weighted_event"] / exposure * 80), exposure)
        return output

    def _fit_rp_priors(self) -> None:
        path = DATA / "rp_compstats.csv"
        if not path.exists():
            return
        rp = pd.read_csv(path, low_memory=False)
        asof = pd.Timestamp(self.asof).tz_localize(None) if pd.Timestamp(self.asof).tzinfo else pd.Timestamp(self.asof)
        end_year = rp["season"].map(_season_end_year)
        concluded = pd.to_datetime(end_year.astype(str) + "-06-30", errors="coerce")
        rp = rp[concluded.lt(asof)].copy()
        rp["_weight"] = np.power(
            0.5,
            (asof.year - rp["season"].map(_season_end_year)).clip(lower=0) / RP_HALFLIFE_YEARS,
        )
        rp = rp[rp["_weight"].gt(0.15)]
        by_name: dict[tuple[str, str], tuple[float, float]] = {}
        for slug, group in rp.groupby("slug"):
            minutes = pd.to_numeric(group["minutes"], errors="coerce").fillna(0)
            weighted_minutes = float((group["_weight"] * minutes).sum())
            if weighted_minutes < 120:
                continue
            for event in self.events:
                column = RP_ALIASES.get(event, event)
                if column not in group:
                    continue
                values = pd.to_numeric(group[column], errors="coerce")
                valid = values.notna()
                if not valid.any():
                    continue
                rate = float((group.loc[valid, "_weight"] * values[valid]).sum() / weighted_minutes * 80)
                by_name[(_norm(str(slug).replace("-", " ")), event)] = (rate, weighted_minutes)

        # Event-specific calibration against established international players.
        factor: dict[str, float] = {}
        for event in self.events:
            ratios = []
            for player, name in self.player_names.items():
                observed = self.profiles.get((player, event))
                prior = by_name.get((_norm(name), event))
                if observed and prior and observed[1] >= 300 and prior[0] > 1e-6:
                    ratios.append(observed[0] / prior[0])
            factor[event] = float(np.clip(np.median(ratios), 0.5, 1.2)) if len(ratios) >= 20 else 0.84
        self.rp_priors = {
            (name, event): (rate * factor[event], exposure)
            for (name, event), (rate, exposure) in by_name.items()
        }

    def fit(self, frame: pd.DataFrame) -> "EmpiricalEventModel":
        train = frame.copy()
        train["player_id"] = train["player_id"].astype(str)
        self.player_names = train.groupby("player_id")["player_name"].last().astype(str).to_dict()
        international = self._weighted_rates(train, "international")
        club = self._weighted_rates(train, "club")
        keys = set(international) | set(club)
        self.profiles = {}
        for key in keys:
            intl_rate, intl_minutes = international.get(key, (np.nan, 0.0))
            club_rate, club_minutes = club.get(key, (np.nan, 0.0))
            if intl_minutes > 0 and club_minutes > 0:
                club_weight = CLUB_CONF * club_minutes
                rate = (
                    intl_minutes * intl_rate
                    + club_weight * club_rate * _calibration(key[1])
                ) / (intl_minutes + club_weight)
                exposure = intl_minutes + club_weight
            elif intl_minutes > 0:
                rate, exposure = intl_rate, intl_minutes
            elif club_minutes > 0:
                rate, exposure = club_rate * _calibration(key[1]), CLUB_CONF * club_minutes
            else:
                continue
            self.profiles[key] = (float(rate), float(exposure))

        intl = train[train["competition_level"].eq("international")].copy()
        for event in self.events:
            valid = (
                intl[f"available__{event}"].fillna(False).astype(bool)
                & pd.to_numeric(intl["minutes"], errors="coerce").gt(0)
            )
            if not valid.any():
                continue
            self.active_events.add(event)
            block = intl[valid].copy()
            block["_rate"] = (
                pd.to_numeric(block[event], errors="coerce")
                / pd.to_numeric(block["minutes"], errors="coerce").clip(lower=1) * 80
            )
            position = block.groupby("position")["_rate"].mean()
            overall = float(block["_rate"].mean())
            for pos in train["position"].dropna().astype(str).unique():
                self.position_priors[(str(pos), event)] = float(position.get(pos, overall))
            actual = pd.to_numeric(block[event], errors="coerce").to_numpy(float)
            self.dispersion[event] = max(float(np.var(actual)), float(np.mean(actual)), 1e-4)

        self._fit_rp_priors()
        asof = pd.Timestamp(self.asof).tz_localize(None) if pd.Timestamp(self.asof).tzinfo else pd.Timestamp(self.asof)
        dates = pd.to_datetime(train["date"], errors="coerce")
        train["_weight"] = np.power(0.5, (asof - dates).dt.days.clip(lower=0) / HALFLIFE_DAYS)
        minute_valid = train["available__minutes"].fillna(False).astype(bool)
        minute_rows = train[minute_valid].copy()
        for (player, started), group in minute_rows.groupby([minute_rows["player_id"].astype(str), "started"]):
            weight = group["_weight"].to_numpy(float)
            if weight.sum() > 0:
                self.minutes_by_player[(str(player), bool(started))] = float(
                    np.average(pd.to_numeric(group["minutes"], errors="coerce"), weights=weight)
                )
        for (position, started), group in minute_rows.groupby(["position", "started"]):
            weight = group["_weight"].to_numpy(float)
            if weight.sum() > 0:
                self.minutes_by_position[(str(position), bool(started))] = float(
                    np.average(pd.to_numeric(group["minutes"], errors="coerce"), weights=weight)
                )
        self.dispersion["minutes"] = max(float(pd.to_numeric(minute_rows["minutes"], errors="coerce").var()), 1.0)

        rankings = DATA / "wr_rankings.csv"
        if rankings.exists():
            wr = pd.read_csv(rankings, parse_dates=["snapshot_date"])
            snapshot = wr[wr["snapshot_date"].lt(asof)]
            if not snapshot.empty:
                snapshot = snapshot[snapshot["snapshot_date"].eq(snapshot["snapshot_date"].max())]
                self.wr_points = snapshot.set_index("team")["wr_pts"].astype(float).to_dict()
        return self

    def _rate(self, player: str, name: str, position: str, event: str) -> float:
        base = self.position_priors.get((position, event), 0.0)
        rp = self.rp_priors.get((_norm(name), event))
        if rp:
            confidence = min(rp[1], 900.0) / 900.0 * 200.0
            prior = (confidence * rp[0] + 80.0 * base) / (confidence + 80.0)
        else:
            prior = base
        profile = self.profiles.get((player, event))
        if not profile:
            return max(float(prior), 0.0)
        rate, exposure = profile
        return max(float((exposure * rate + K * prior) / (exposure + K)), 0.0)

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        predictions = []
        for row in frame.itertuples(index=False):
            player, name, position = str(row.player_id), str(row.player_name), str(row.position)
            started = bool(row.started)
            minutes = self.minutes_by_player.get(
                (player, started),
                self.minutes_by_position.get((position, started), 70.0 if started else 20.0),
            )
            minutes = float(np.clip(minutes, 0, 80))
            margin = 2.0 * (
                self.wr_points.get(str(row.team), 80.0)
                - self.wr_points.get(str(row.opponent), 80.0)
                + (3.0 if str(getattr(row, "home_away", "")) == "home" else -3.0)
            )
            attack_mult = float(np.clip(1.0 + margin / 60.0, 0.72, 1.35))
            defence_mult = float(np.clip(1.0 - margin / 90.0, 0.80, 1.28))
            events = {}
            for event in sorted(self.active_events):
                rate = self._rate(player, name, position, event)
                group = _matchup_group(event)
                if group == "attack":
                    rate *= attack_mult
                elif group == "defence":
                    rate *= defence_mult
                mean = max(rate * minutes / 80.0, 0.0)
                family = distribution_family(event)
                if family == "bernoulli":
                    mean, dispersion = min(mean, 1.0), 1.0
                elif family == "negative_binomial":
                    variance = max(self.dispersion.get(event, mean + 1.0), mean + 1e-6)
                    dispersion = max(mean * mean / max(variance - mean, 1e-6), 0.05)
                else:
                    dispersion = max(self.dispersion.get(event, 1.0), 1e-6)
                events[event] = EventDistribution(family, float(mean), float(dispersion))
            predictions.append(RawPrediction(
                fixture_id=str(row.fixture_id), player_id=player, player_name=name,
                team=str(row.team), opponent=str(row.opponent), position=position,
                is_forward=bool(row.is_forward), events=events,
                minutes=EventDistribution("lognormal", minutes, self.dispersion.get("minutes", 25.0)),
                metadata={"model": "empirical_event", "competition_level": str(row.competition_level)},
            ))
        return predictions

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "EmpiricalEventModel":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not an EmpiricalEventModel")
        return model
