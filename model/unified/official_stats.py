"""Six Nations official-statistic adapter (research, opt-in).

Three Six Nations fantasy components are scored from statistics that the
competition-independent raw-event model does not observe directly:

* **Breakdown steals (BS, 5 pts).** The model forecasts the RapidAPI
  ``tackle_turnover`` event, which correlates only 0.03-0.11 with official BS
  per player-match (0.06-0.29 per player-season). RugbyPass/Opta
  ``turnovers_won`` correlates 0.76-0.87 with official BS per player-season.
  This adapter forecasts official BS as ``rate * E[minutes] / 80``, where the
  rate is an empirical-Bayes posterior: an official position rate times the
  player's relative RugbyPass turnovers-won rate (completed seasons only) is the
  prior, updated with the player's own official BS history.
* **Metres (1 pt per 10 official metres).** The official metres definition is
  not the API one, and it changed: in 2023 and 2025 official metres were close
  to API metres plus about 2.3-2.9 m per carry; in 2026 they were about 1.05x
  API metres. The adapter maps ``official = a * E[API metres] + b * E[carries]``
  with ``(a, b)`` fitted on the current season's completed official rounds or,
  before round 1 results, on the latest official season. The lognormal
  coefficient of variation is preserved.
* **Player of the match (POTM, 15 pts).** Exactly one per match; 41/42
  winners in 2023-26 were on the winning team, 41/42 were starters. The adapter
  replaces the per-player POTM mean with ``P(team wins) * softmax(beta * xp)``
  over that team's players (bench weight 0.05), renormalised to one per match.
  ``P(team wins)`` comes from the pre-lock margin Elo; ``xp`` is the player's
  expected Six Nations points excluding POTM.

Only pre-lock information is used: official rows before the lock, RugbyPass
seasons admitted by :func:`model.history.past_seasons`, and match history
before the lock for the Elo state. Other events, minutes and the NCR adapter
are untouched. See ``research/OFFICIAL_STATS_REPORT_2026-10.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from compare_api_official import norm_key
from model.history import past_matches, past_seasons
from official_labels import match_official, official_name_key

from .contracts import EventDistribution, RawPrediction

ROOT = Path(__file__).resolve().parents[2]
SIX_NATIONS_ID = 1266
OFFICIAL_COLUMNS = ('Min', 'MC', 'BS', 'POTM', 'Pts')


@dataclass(frozen=True)
class OfficialStatConfig:
    breakdown_steals: bool = True
    metres: bool = True
    potm: bool = True
    rp_shrink: float = 10.0          # pseudo-80-minute matches toward the position RugbyPass rate
    official_shrink: float = 12.0    # pseudo-80-minute matches toward the RugbyPass-informed prior
    position_shrink_minutes: float = 400.0
    potm_beta: float = 0.1
    potm_bench_weight: float = 0.05


def official_history(store: pd.DataFrame, official: pd.DataFrame | None = None) -> pd.DataFrame:
    """Six Nations store rows joined to the official per-player statistics.

    Returns one row per matched player-match with the store's identity, kickoff,
    role and API events, plus ``off_<column>`` official statistics.
    """
    if official is None:
        official = pd.read_csv(ROOT/'data'/'official_player_match.csv', low_memory=False)
    six = store[pd.to_numeric(store['competition_id_cache'], errors='coerce').eq(SIX_NATIONS_ID)]
    labels = match_official(six, official)
    keep = ['fixture_id', 'player_id', 'team', 'player_name', 'match_at', 'season', 'round', 'position',
            'minutes', 'metres', 'runs', 'started']
    out = six[[c for c in keep if c in six]].copy()
    for column in OFFICIAL_COLUMNS:
        out[f'off_{column}'] = pd.to_numeric(labels[column], errors='coerce')
    out = out[labels['season'].notna().to_numpy()].copy()
    out['key'] = [official_name_key(n, t) for n, t in zip(out.player_name, out.team)]
    out['match_at'] = pd.to_datetime(out['match_at'], utc=True)
    for column in ('minutes', 'metres', 'runs'):
        out[column] = pd.to_numeric(out[column], errors='coerce').fillna(0.0)
    return out.reset_index(drop=True)


def rugbypass_turnovers(compstats: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Turnovers won and minutes per player key over seasons completed before ``cutoff``.

    Keys shared by more than one RugbyPass slug are ambiguous and dropped.
    """
    frame = compstats.drop_duplicates(['key', 'competition', 'season'])
    ambiguous = frame.groupby('key').slug.nunique()
    frame = frame[~frame.key.isin(ambiguous[ambiguous > 1].index)]
    frame = past_seasons(frame, cutoff)
    minutes = pd.to_numeric(frame.minutes, errors='coerce').fillna(0)
    won = pd.to_numeric(frame.turnovers_won, errors='coerce').fillna(0)
    return pd.DataFrame({'tw': won.groupby(frame.key).sum(), 'minutes': minutes.groupby(frame.key).sum()})


def _metres_mapping(history: pd.DataFrame, season: int | None) -> tuple[float, float] | None:
    played = history[history.minutes.gt(0) & history.off_MC.notna()]
    if played.empty:
        return None
    current = played[played.season.eq(season)] if season is not None else played.iloc[:0]
    rows = current if len(current) else played[played.season.eq(played.season.max())]
    design = rows[['metres', 'runs']].to_numpy(float)
    a, b = np.linalg.lstsq(design, rows.off_MC.to_numpy(float), rcond=None)[0]
    return float(a), float(b)


def _scaled(dist: EventDistribution, mean: float) -> EventDistribution:
    mean = max(float(mean), 0.0)
    if dist.family == 'lognormal':
        factor = mean/dist.mean if dist.mean > 0 else 1.0
        return EventDistribution(dist.family, mean, max(dist.dispersion*factor**2, 1e-9))
    if dist.family == 'bernoulli':
        return EventDistribution(dist.family, min(mean, 1.0), dist.dispersion)
    return EventDistribution(dist.family, mean, dist.dispersion)


def win_probability(edge: np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + 10.0**(-np.asarray(edge, float)/400.0))


def potm_probabilities(frame: pd.DataFrame, beta: float, bench_weight: float) -> np.ndarray:
    """``frame`` has fixture_id, team, started, xp and pwin; returns one probability per row."""
    weights = np.zeros(len(frame))
    positions = np.arange(len(frame))
    for _, index in frame.groupby(['fixture_id', 'team']).indices.items():
        rows = frame.iloc[index]
        score = rows.xp.to_numpy(float)
        share = np.exp(beta*(score - score.max()))*np.where(rows.started.astype(bool), 1.0, bench_weight)
        weights[positions[index]] = float(rows.pwin.iloc[0])*share/share.sum()
    totals = pd.Series(weights).groupby(frame.fixture_id.to_numpy()).transform('sum').to_numpy()
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)


@dataclass
class SixNationsOfficialStats:
    """Fit on pre-lock data, then adjust a slate's raw forecasts for Six Nations scoring."""

    config: OfficialStatConfig = OfficialStatConfig()

    def fit(self, official: pd.DataFrame, compstats: pd.DataFrame | None, history: pd.DataFrame,
            cutoff, season: int | None = None) -> 'SixNationsOfficialStats':
        cutoff = pd.Timestamp(cutoff)
        cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
        self.cutoff = cutoff
        past = past_matches(official, cutoff)
        played = past[past.minutes.gt(0) & past.off_BS.notna()]
        self.has_official = len(played) > 0
        if self.has_official:
            overall = played.off_BS.sum()/played.minutes.sum()*80
            by_pos = played.groupby('position').agg(bs=('off_BS', 'sum'), minutes=('minutes', 'sum'))
            k = self.config.position_shrink_minutes
            self.position_rate = ((80*by_pos.bs + overall*k)/(by_pos.minutes + k)).to_dict()
            self.overall_rate = float(overall)
            self.player_bs = played.groupby('key').agg(bs=('off_BS', 'sum'), minutes=('minutes', 'sum'))
        self.turnovers = (rugbypass_turnovers(compstats, cutoff) if compstats is not None
                          else pd.DataFrame(columns=['tw', 'minutes']))
        self.metres_map = _metres_mapping(past, season)
        from .v4.context import pre_match_context
        self.team_state = pre_match_context(past_matches(history, cutoff))[1]
        return self

    # --- breakdown steals -------------------------------------------------
    def breakdown_rates(self, keys: list[str], positions: list[str]) -> np.ndarray:
        prior = np.array([self.position_rate.get(p, self.overall_rate) for p in positions])
        tw = pd.Series(keys).map(self.turnovers.tw).fillna(0).to_numpy(float)
        rp_80 = pd.Series(keys).map(self.turnovers.minutes).fillna(0).to_numpy(float)/80
        # Position RugbyPass rate over this pool's covered players.
        pool = pd.DataFrame({'pos': positions, 'tw': tw, 'm80': rp_80}).drop_duplicates()
        covered = pool[pool.m80 > 0]
        mean_all = covered.tw.sum()/covered.m80.sum() if covered.m80.sum() > 0 else np.nan
        by_pos = covered.groupby('pos').agg(tw=('tw', 'sum'), m80=('m80', 'sum'))
        pos_rp = np.array([(by_pos.tw.get(p, 0) + 10*mean_all)/(by_pos.m80.get(p, 0) + 10)
                           if np.isfinite(mean_all) else np.nan for p in positions])
        k = self.config.rp_shrink
        relative = np.where(np.isfinite(pos_rp) & (pos_rp > 0), (tw + k*pos_rp)/(rp_80 + k)/pos_rp, 1.0)
        prior = prior*relative
        own_bs = pd.Series(keys).map(self.player_bs.bs).fillna(0).to_numpy(float)
        own_80 = pd.Series(keys).map(self.player_bs.minutes).fillna(0).to_numpy(float)/80
        k = self.config.official_shrink
        return (own_bs + k*prior)/(own_80 + k)

    def adjust(self, raw: list[RawPrediction], candidates: pd.DataFrame) -> list[RawPrediction]:
        frame = candidates.reset_index(drop=True)
        if [(p.fixture_id, p.player_id, p.team) for p in raw] != list(zip(
                frame.fixture_id.astype(str), frame.player_id.astype(str), frame.team.astype(str))):
            raise ValueError('forecasts and candidates are misaligned')
        output = list(raw)
        if self.config.breakdown_steals and self.has_official:
            keys = [official_name_key(p.player_name, p.team) for p in raw]
            rates = self.breakdown_rates(keys, [p.position for p in raw])
            for i, (p, rate) in enumerate(zip(output, rates)):
                minutes = float(np.clip(p.minutes.mean, 0, 80))
                events = dict(p.events)
                old = events.get('tackle_turnover', EventDistribution('negative_binomial', 0.0, 1.0))
                events['tackle_turnover'] = _scaled(old, rate*minutes/80)
                output[i] = replace(p, events=events)
        if self.config.metres and self.metres_map is not None:
            a, b = self.metres_map
            for i, p in enumerate(output):
                if 'metres' not in p.events or p.events['metres'].mean <= 0:
                    continue
                carries = p.events['runs'].mean if 'runs' in p.events else 0.0
                events = dict(p.events)
                events['metres'] = _scaled(p.events['metres'], max(a*p.events['metres'].mean + b*carries, 0.01))
                output[i] = replace(p, events=events)
        if self.config.potm:
            from .rolling_eval import expected_points
            from .v4.context import add_candidate_context
            without = [replace(p, events={k: v for k, v in p.events.items() if k != 'potm'}) for p in output]
            edges = add_candidate_context(frame, self.team_state)['ctx__elo_edge'].to_numpy(float)
            table = pd.DataFrame({'fixture_id': frame.fixture_id.astype(str), 'team': frame.team.astype(str),
                                  'started': frame.started.astype(bool),
                                  'xp': expected_points(without, 'six_nations'), 'pwin': win_probability(edges)})
            probabilities = potm_probabilities(table, self.config.potm_beta, self.config.potm_bench_weight)
            for i, (p, q) in enumerate(zip(output, probabilities)):
                events = dict(p.events)
                events['potm'] = EventDistribution('bernoulli', float(np.clip(q, 0.0, 1.0)), 1.0)
                output[i] = replace(p, events=events)
        return output


def load_compstats() -> pd.DataFrame:
    frame = pd.read_csv(ROOT/'data'/'rp_compstats.csv', low_memory=False)
    if 'key' not in frame:
        frame['key'] = frame.slug.str.replace('-', ' ').map(norm_key)
    return frame
