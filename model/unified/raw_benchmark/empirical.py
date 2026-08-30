"""Competition-independent empirical event-rate model."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class EmpiricalConfig:
    """Tunable surface of the empirical engine.

    Every default reproduces the frozen ``empirical_event`` behaviour exactly,
    so an unset ``RB_EMPIRICAL_VARIANT`` leaves the v1 contract untouched.
    """

    # Prior for a position/event, used when a player has little or no history.
    # "mean_of_ratios" averages per-appearance per-80 rates (frozen behaviour);
    # "exposure_weighted" uses sum(event)/sum(minutes), the unbiased rate.
    position_prior_mode: str = "mean_of_ratios"
    position_prior_recency: bool = False
    # Opponent-strength response. "fixed" is the frozen hand-tuned multiplier,
    # "off" disables it, "damped" scales the margin, "empirical" fits the
    # response per event group from the training fold.
    matchup_mode: str = "fixed"
    matchup_damping: float = 1.0
    matchup_attack_scale: float = 60.0
    matchup_defence_scale: float = 90.0
    # Shrinkage of a player's own history toward the prior, in minutes.
    # "global" keeps K; "empirical_bayes" fits K per event from the training
    # fold's between/within player rate variance.
    shrinkage_mode: str = "global"
    k_shrinkage: float = K
    # Minutes play two roles with different optimal statistics: the reported
    # minutes head is scored by MAE (median-optimal), while the count scaler
    # needs E[minutes] because counts are rate * minutes / 80. "same" keeps the
    # frozen single-table behaviour; "median" fits a separate median head.
    minutes_statistic: str = "mean"
    minutes_head_statistic: str = "same"
    # Counts are predicted as rate * minutes / 80, so minutes noise propagates
    # multiplicatively into every event. "player" is the frozen per-player
    # table; "position" uses the much lower-variance position/started table;
    # "shrunk" blends the two.
    # "eb" fits the player-vs-position blend from the training fold's
    # within/between player minutes variance, so a 30-cap player keeps their
    # own minutes while a 1-cap player falls back to the position mean.
    minutes_mode: str = "player"
    minutes_shrinkage: float = 0.5
    halflife_days: float = HALFLIFE_DAYS
    club_conf: float = CLUB_CONF

    @classmethod
    def from_env(cls) -> "EmpiricalConfig":
        name = os.environ.get("RB_EMPIRICAL_VARIANT", "").strip()
        if not name:
            return cls()
        if name not in VARIANTS:
            raise ValueError(f"unknown RB_EMPIRICAL_VARIANT {name!r}; known: {sorted(VARIANTS)}")
        return VARIANTS[name]


BASE = EmpiricalConfig()
# Candidate queue for the all-rugby hill-climb. Each entry is a small delta off
# the incumbent it was proposed against; see LEDGER.md for hypotheses/outcomes.
_T1 = replace(BASE, minutes_mode="eb", matchup_mode="off")
_T2 = replace(_T1, shrinkage_mode="empirical_bayes")
_T3 = replace(_T2, minutes_head_statistic="median")
VARIANTS: dict[str, EmpiricalConfig] = {
    "base": BASE,
    # T1: shrink the per-player minutes table toward the position/started mean
    # by empirical Bayes, and drop the saturating opponent-strength multiplier.
    "t1_eb_off": _T1,
    # T2: per-event empirical-Bayes shrinkage of player history to the prior.
    "t2_eb_off_ebk": _T2,
    # T3: separate median-valued minutes head (the minutes target is MAE-scored)
    # while counts keep scaling on E[minutes].
    "t3_eb_off_ebk_head": _T3,
    # Rejected in triage, kept so the ledger can be re-derived.
    "r_exposure_prior": replace(_T1, position_prior_mode="exposure_weighted"),
    "r_prior_recency": replace(_T1, position_prior_recency=True),
    "r_minutes_median_both": replace(_T1, minutes_statistic="median"),
    "r_matchup_damped": replace(BASE, minutes_mode="eb", matchup_mode="damped", matchup_damping=0.25),
    "r_minutes_position": replace(BASE, minutes_mode="position", matchup_mode="off"),
}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = weights.sum()
    if total <= 0:
        return float(np.median(values)) if len(values) else 0.0
    cumulative = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(0.5 * total, cumulative, values))


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
    config: EmpiricalConfig = field(default_factory=EmpiricalConfig.from_env)
    event_shrinkage: dict[str, float] = field(default_factory=dict)
    matchup_beta: dict[str, float] = field(default_factory=dict)
    minutes_weight: dict[tuple[str, bool], float] = field(default_factory=dict)
    minutes_eb_m: float = 4.0
    minutes_median_by_player: dict[tuple[str, bool], float] = field(default_factory=dict)
    minutes_median_by_position: dict[tuple[str, bool], float] = field(default_factory=dict)

    def _weighted_rates(self, frame: pd.DataFrame, level: str) -> dict[tuple[str, str], tuple[float, float]]:
        subset = frame[frame["competition_level"].eq(level)].copy()
        if subset.empty:
            return {}
        asof = pd.Timestamp(self.asof).tz_localize(None) if pd.Timestamp(self.asof).tzinfo else pd.Timestamp(self.asof)
        dates = pd.to_datetime(subset["date"], errors="coerce")
        subset["_weight"] = np.power(0.5, (asof - dates).dt.days.clip(lower=0) / self.config.halflife_days)
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

    def _fit_event_shrinkage(self, block: pd.DataFrame, event: str) -> float:
        """Empirical-Bayes shrinkage in minutes: K = within-var / between-var.

        Events whose per-player rates differ little relative to appearance noise
        earn heavy shrinkage toward the position prior; genuinely player-specific
        events earn little. Falls back to the global K when under-determined.
        """
        minutes = pd.to_numeric(block["minutes"], errors="coerce").clip(lower=1)
        counts = pd.to_numeric(block[event], errors="coerce")
        grouped = pd.DataFrame({
            "player": block["player_id"].astype(str),
            "count": counts, "minutes": minutes,
        }).groupby("player")[["count", "minutes"]].sum()
        grouped = grouped[grouped["minutes"] >= 80]
        if len(grouped) < 30:
            return float(self.config.k_shrinkage)
        rates = grouped["count"] / grouped["minutes"] * 80
        grand = float(grouped["count"].sum() / max(float(grouped["minutes"].sum()), 1e-9) * 80)
        # Poisson sampling variance of each player's rate, per 80 minutes.
        within = float(np.mean(grand / (grouped["minutes"] / 80.0)))
        between = float(np.var(rates, ddof=1)) - within
        if not np.isfinite(between) or between <= 1e-9 or grand <= 1e-9:
            return float(self.config.k_shrinkage)
        return float(np.clip(grand / between * 80.0, 20.0, 3000.0))

    def fit(self, frame: pd.DataFrame) -> "EmpiricalEventModel":
        train = frame.copy()
        asof_ts = pd.Timestamp(self.asof)
        asof_ts = asof_ts.tz_localize(None) if asof_ts.tzinfo else asof_ts
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
            if self.config.position_prior_mode == "mean_of_ratios" and not self.config.position_prior_recency:
                position = block.groupby("position")["_rate"].mean()
                overall = float(block["_rate"].mean())
            else:
                if self.config.position_prior_recency:
                    block_dates = pd.to_datetime(block["date"], errors="coerce")
                    block["_pw"] = np.power(
                        0.5,
                        (asof_ts - block_dates).dt.days.clip(lower=0) / self.config.halflife_days,
                    )
                else:
                    block["_pw"] = 1.0
                if self.config.position_prior_mode == "exposure_weighted":
                    # Unbiased rate: total events over total minutes. The frozen
                    # mean-of-ratios estimator is inflated by short appearances,
                    # whose per-80 extrapolation is enormous yet equally weighted.
                    block["_num"] = block["_pw"] * pd.to_numeric(block[event], errors="coerce")
                    block["_den"] = block["_pw"] * pd.to_numeric(
                        block["minutes"], errors="coerce"
                    ).clip(lower=1)
                else:
                    block["_num"] = block["_pw"] * block["_rate"]
                    block["_den"] = block["_pw"]
                scale = 80.0 if self.config.position_prior_mode == "exposure_weighted" else 1.0
                grouped = block.groupby("position")[["_num", "_den"]].sum()
                position = grouped["_num"] / grouped["_den"].replace(0.0, np.nan) * scale
                overall = float(
                    block["_num"].sum() / max(float(block["_den"].sum()), 1e-9) * scale
                )
            for pos in train["position"].dropna().astype(str).unique():
                value = float(position.get(pos, np.nan))
                self.position_priors[(str(pos), event)] = (
                    overall if not np.isfinite(value) else value
                )
            if self.config.shrinkage_mode == "empirical_bayes":
                self.event_shrinkage[event] = self._fit_event_shrinkage(block, event)
            actual = pd.to_numeric(block[event], errors="coerce").to_numpy(float)
            self.dispersion[event] = max(float(np.var(actual)), float(np.mean(actual)), 1e-4)

        self._fit_rp_priors()
        asof = pd.Timestamp(self.asof).tz_localize(None) if pd.Timestamp(self.asof).tzinfo else pd.Timestamp(self.asof)
        dates = pd.to_datetime(train["date"], errors="coerce")
        train["_weight"] = np.power(0.5, (asof - dates).dt.days.clip(lower=0) / self.config.halflife_days)
        minute_valid = train["available__minutes"].fillna(False).astype(bool)
        minute_rows = train[minute_valid].copy()
        # The minutes target is scored by MAE, whose Bayes-optimal point
        # estimate is the median, not the mean.
        use_median = self.config.minutes_statistic == "median"
        head_median = self.config.minutes_head_statistic == "median"

        def _summarise(group: pd.DataFrame) -> float:
            weight = group["_weight"].to_numpy(float)
            values = pd.to_numeric(group["minutes"], errors="coerce").to_numpy(float)
            if use_median:
                return _weighted_median(values, weight)
            return float(np.average(values, weights=weight))

        within, means, sizes, counts = [], [], [], {}
        for (player, started), group in minute_rows.groupby([minute_rows["player_id"].astype(str), "started"]):
            if group["_weight"].sum() <= 0:
                continue
            key = (str(player), bool(started))
            self.minutes_by_player[key] = _summarise(group)
            if head_median:
                self.minutes_median_by_player[key] = _weighted_median(
                    pd.to_numeric(group["minutes"], errors="coerce").to_numpy(float),
                    group["_weight"].to_numpy(float),
                )
            counts[key] = float(group["_weight"].sum())
            values = pd.to_numeric(group["minutes"], errors="coerce").to_numpy(float)
            if len(values) >= 2:
                within.append(float(np.var(values, ddof=1)))
                means.append(float(np.mean(values)))
                sizes.append(len(values))
        for (position, started), group in minute_rows.groupby(["position", "started"]):
            if group["_weight"].sum() > 0:
                self.minutes_by_position[(str(position), bool(started))] = _summarise(group)
                if head_median:
                    self.minutes_median_by_position[(str(position), bool(started))] = _weighted_median(
                        pd.to_numeric(group["minutes"], errors="coerce").to_numpy(float),
                        group["_weight"].to_numpy(float),
                    )
        self.minutes_weight = counts
        # Empirical-Bayes blend point: M = within-player variance / between-player
        # variance, in effective appearances. No hand-set constant. The observed
        # spread of player means already contains sampling noise sigma^2/n, so
        # subtract it to recover the true between-player variance.
        if len(within) >= 30:
            sigma_within = float(np.mean(within))
            harmonic_n = float(len(sizes) / np.sum(1.0 / np.asarray(sizes, dtype=float)))
            tau_between = float(np.var(means, ddof=1)) - sigma_within / max(harmonic_n, 1.0)
            if np.isfinite(tau_between) and tau_between > 1e-6:
                self.minutes_eb_m = float(np.clip(sigma_within / tau_between, 0.25, 50.0))
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
        shrinkage = self.event_shrinkage.get(event, self.config.k_shrinkage)
        return max(float((exposure * rate + shrinkage * prior) / (exposure + shrinkage)), 0.0)

    def _minutes(
        self, player: str, position: str, started: bool,
        by_player: dict, by_position: dict,
    ) -> float:
        fallback = by_position.get((position, started), 70.0 if started else 20.0)
        if self.config.minutes_mode == "position":
            minutes = fallback
        elif self.config.minutes_mode in {"shrunk", "eb"}:
            if self.config.minutes_mode == "eb":
                effective = self.minutes_weight.get((player, started), 0.0)
                weight = effective / (effective + self.minutes_eb_m)
            else:
                weight = float(np.clip(self.config.minutes_shrinkage, 0.0, 1.0))
            minutes = (
                weight * by_player.get((player, started), fallback)
                + (1.0 - weight) * fallback
            )
        else:
            minutes = by_player.get((player, started), fallback)
        return float(np.clip(minutes, 0, 80))

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        predictions = []
        for row in frame.itertuples(index=False):
            player, name, position = str(row.player_id), str(row.player_name), str(row.position)
            started = bool(row.started)
            minutes = self._minutes(
                player, position, started, self.minutes_by_player, self.minutes_by_position,
            )
            head_minutes = minutes
            if self.config.minutes_head_statistic == "median":
                head_minutes = self._minutes(
                    player, position, started,
                    self.minutes_median_by_player, self.minutes_median_by_position,
                )
            margin = 2.0 * (
                self.wr_points.get(str(row.team), 80.0)
                - self.wr_points.get(str(row.opponent), 80.0)
                + (3.0 if str(getattr(row, "home_away", "")) == "home" else -3.0)
            )
            if self.config.matchup_mode == "off":
                attack_mult = defence_mult = 1.0
            else:
                # "damped" keeps the frozen shape but scales how far a ranking
                # gap is allowed to move a raw count.
                scaled = margin * self.config.matchup_damping
                attack_mult = float(np.clip(
                    1.0 + scaled / self.config.matchup_attack_scale, 0.72, 1.35,
                ))
                defence_mult = float(np.clip(
                    1.0 - scaled / self.config.matchup_defence_scale, 0.80, 1.28,
                ))
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
                minutes=EventDistribution("lognormal", head_minutes, self.dispersion.get("minutes", 25.0)),
                metadata={"model": "empirical_event", "competition_level": str(row.competition_level)},
            ))
        return predictions

    def __setstate__(self, state: dict) -> None:
        # Artifacts pickled before the config surface existed carry no config;
        # backfill the frozen defaults so old caches stay loadable.
        state.setdefault("config", EmpiricalConfig())
        state.setdefault("event_shrinkage", {})
        state.setdefault("matchup_beta", {})
        state.setdefault("minutes_weight", {})
        state.setdefault("minutes_eb_m", 4.0)
        state.setdefault("minutes_median_by_player", {})
        state.setdefault("minutes_median_by_position", {})
        self.__dict__.update(state)

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
