"""Pre-match team and opponent strength features for the v4 event trees.

The v4 trees previously saw team/opponent only as identity codes and the
team's own recent margin. They had no numeric opponent strength and no venue.
These features give every row, club or international, one chronological
margin-of-victory Elo state per team, plus separate points-for/against form.

Each fixture updates its two teams once, even when both sides have player
rows. Training rows receive the state before their fixture; candidate rows
receive the state after every training fixture, so held-out results never
update one another.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction

PREFIX = "ctx__"
COLUMNS = (
    "ctx__team_elo", "ctx__opp_elo", "ctx__elo_edge", "ctx__home",
    "ctx__team_pf", "ctx__team_pa", "ctx__opp_pf", "ctx__opp_pa",
)
INITIAL = 1500.0
K_FACTOR = 24.0
HOME_ADVANTAGE = 45.0
FORM_ALPHA = 2.0 / (8 + 1)  # the span-8 weight already used for team margin form


def _home(value) -> float:
    text = str(value).lower()
    return 1.0 if text == "home" else 0.0 if text == "away" else 0.5


def fixture_results(frame: pd.DataFrame) -> pd.DataFrame:
    """One scored row per fixture, from whichever side has player rows."""
    columns = ["fixture_id", "team", "opponent", "team_score", "opp_score", "home_away", "match_at"]
    rows = frame[[c for c in columns if c in frame]].copy()
    rows["team_score"] = pd.to_numeric(rows["team_score"], errors="coerce")
    rows["opp_score"] = pd.to_numeric(rows["opp_score"], errors="coerce")
    rows = rows.dropna(subset=["team_score", "opp_score"])
    rows["match_at"] = pd.to_datetime(rows["match_at"], utc=True)
    rows = rows.sort_values(["match_at", "fixture_id", "team"]).drop_duplicates("fixture_id")
    return rows.reset_index(drop=True)


class TeamState:
    """Chronological Elo and points form, updated once per fixture."""

    def __init__(self):
        self.elo: dict[str, float] = {}
        self.pf: dict[str, float] = {}
        self.pa: dict[str, float] = {}

    def features(self, team: str, opponent: str, home: float) -> dict[str, float]:
        team_elo = self.elo.get(team, INITIAL)
        opp_elo = self.elo.get(opponent, INITIAL)
        return {
            "ctx__team_elo": team_elo, "ctx__opp_elo": opp_elo,
            "ctx__elo_edge": team_elo - opp_elo + HOME_ADVANTAGE * (2.0 * home - 1.0),
            "ctx__home": home,
            "ctx__team_pf": self.pf.get(team, np.nan), "ctx__team_pa": self.pa.get(team, np.nan),
            "ctx__opp_pf": self.pf.get(opponent, np.nan), "ctx__opp_pa": self.pa.get(opponent, np.nan),
        }

    def update(self, team: str, opponent: str, home: float, scored: float, conceded: float) -> None:
        team_elo = self.elo.get(team, INITIAL)
        opp_elo = self.elo.get(opponent, INITIAL)
        edge = team_elo - opp_elo + HOME_ADVANTAGE * (2.0 * home - 1.0)
        expected = 1.0 / (1.0 + 10.0 ** (-edge / 400.0))
        margin = scored - conceded
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        # Margin-of-victory multiplier with the usual favourite damping.
        winner_edge = edge if margin > 0 else -edge if margin < 0 else 0.0
        multiplier = np.log1p(abs(margin)) * 2.2 / (winner_edge * 0.001 + 2.2)
        delta = K_FACTOR * max(multiplier, 0.5) * (result - expected)
        self.elo[team] = team_elo + delta
        self.elo[opponent] = opp_elo - delta
        for side, points_for, points_against in ((team, scored, conceded), (opponent, conceded, scored)):
            for table, value in ((self.pf, points_for), (self.pa, points_against)):
                previous = table.get(side)
                table[side] = value if previous is None else previous + FORM_ALPHA * (value - previous)


def pre_match_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, TeamState]:
    """Pre-fixture state for every (fixture, team) plus the final state."""
    results = fixture_results(frame)
    state = TeamState()
    records = []
    for row in results.itertuples(index=False):
        home = _home(row.home_away)
        team, opponent = str(row.team), str(row.opponent)
        records.append({"fixture_id": row.fixture_id, "team": team, **state.features(team, opponent, home)})
        records.append({"fixture_id": row.fixture_id, "team": opponent, **state.features(opponent, team, 1.0 - home)})
        state.update(team, opponent, home, float(row.team_score), float(row.opp_score))
    table = pd.DataFrame(records, columns=["fixture_id", "team", *COLUMNS])
    return table, state


def add_training_context(features: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-fixture states; unscored fixtures keep missing values."""
    keyed = table.assign(fixture_id=table.fixture_id.astype(str), team=table.team.astype(str))
    left = features.drop(columns=[c for c in COLUMNS if c in features]).copy()
    left["_fixture"] = left["fixture_id"].astype(str)
    left["_team"] = left["team"].astype(str)
    merged = left.merge(keyed.rename(columns={"fixture_id": "_fixture", "team": "_team"}),
                        on=["_fixture", "_team"], how="left", validate="many_to_one")
    merged.index = features.index
    return merged.drop(columns=["_fixture", "_team"])


def add_candidate_context(candidates: pd.DataFrame, state: TeamState) -> pd.DataFrame:
    """Candidates see only the state after every training fixture."""
    output = candidates.drop(columns=[c for c in COLUMNS if c in candidates]).copy()
    rows = [state.features(str(r.team), str(r.opponent), _home(getattr(r, "home_away", "")))
            for r in output.itertuples(index=False)]
    context = pd.DataFrame(rows, index=output.index, columns=list(COLUMNS))
    return pd.concat([output, context], axis=1)


def team_strength_scale(prediction: RawPrediction, edge: float, beta: Mapping[str, float]) -> RawPrediction:
    """Rescale event means by exp(beta * edge / 400), preserving each event's shape.

    Count dispersion (negative-binomial size) is kept, lognormal variance scales
    with the squared factor so its coefficient of variation is unchanged, and
    Bernoulli probabilities stay within [0, 1]. Minutes are not changed.
    """
    events = {}
    for name, dist in prediction.events.items():
        factor = float(np.exp(beta.get(name, 0.0) * edge / 400.0))
        if dist.family == "bernoulli":
            events[name] = EventDistribution(dist.family, min(dist.mean * factor, 1.0), dist.dispersion)
        elif dist.family == "lognormal":
            events[name] = EventDistribution(dist.family, dist.mean * factor, dist.dispersion * factor ** 2)
        else:
            events[name] = EventDistribution(dist.family, dist.mean * factor, dist.dispersion)
    return replace(prediction, events=events)


@dataclass
class TeamStrengthCalibration:
    """Event-specific matchup calibration on top of a fitted raw-event model.

    ``beta`` comes from team-level Poisson fits of observed totals on the
    wrapped model's forecasts in earlier periods. ``state`` is the team state
    at the prediction lock, so candidates never see their own results.
    """

    model: object
    beta: Mapping[str, float]
    state: TeamState

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        raw = self.model.predict_frame(frame)
        context = add_candidate_context(frame, self.state)
        return [team_strength_scale(p, float(edge), self.beta)
                for p, edge in zip(raw, context["ctx__elo_edge"])]


def fit_team_strength_beta(predictions: pd.DataFrame, events: tuple[str, ...],
                           bound: float = 1.5) -> dict[str, float]:
    """Team-level Poisson fit: observed total ~ forecast total * exp(beta * edge / 400).

    ``predictions`` has one row per (fixture, team) with ``edge``, ``p_<event>``
    (summed forecast means) and ``y_<event>`` (summed observed values, missing
    when unobserved). Events with fewer than 30 observed units keep beta 0.
    """
    from scipy.optimize import minimize_scalar

    beta = {}
    for event in events:
        block = predictions[["edge", f"p_{event}", f"y_{event}"]].dropna()
        block = block[block[f"p_{event}"] > 0]
        if block[f"y_{event}"].sum() <= 30:
            beta[event] = 0.0
            continue
        x = block["edge"].to_numpy(float) / 400.0
        p, y = block[f"p_{event}"].to_numpy(float), block[f"y_{event}"].to_numpy(float)
        loss = lambda b: float(np.sum(p * np.exp(b * x) - y * (np.log(p) + b * x)))
        beta[event] = float(minimize_scalar(loss, bounds=(-bound, bound), method="bounded").x)
    return beta
