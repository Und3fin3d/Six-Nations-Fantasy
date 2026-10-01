"""October 2026 hill-climb candidate: robust P3 + team strength + kicking + empirical blend.

Research candidate only; production routing is unchanged. Starting from the
robust P3 raw forecasts it applies, in order:

1. team-strength event calibration with coefficients fitted on international
   blocks completed before 2025 (``TEAM_STRENGTH_BETA``);
2. team-level goal-kicking concentration with ``gamma = 1.5`` (chosen on the
   same pre-2025 blocks);
3. a fantasy-points blend, ``0.75 * model + 0.25 * empirical fantasy baseline``.

See ``research/HILLCLIMB_REPORT_2026-10-01.md`` for the evidence and limits.

``p3_hillclimb_2026_10b`` (round 2, ``research/HILLCLIMB_PROTOCOL_2026-10-02.md``)
replaces step 1 with matchup calibration (``model.unified.matchup``: Elo edge
plus the opponent's conceded profile, coefficients from pre-2025 blocks),
keeps step 2, and blends ``0.7 * model + 0.3 * empirical``. Every setting was
chosen on development data before the evaluation seasons were scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import RawPrediction
from .kicking import concentrate_kicking
from .matchup import calibrate_matchups
from .v4.context import add_candidate_context, pre_match_context, team_strength_scale

ROOT = Path(__file__).resolve().parents[2]
TEAM_STRENGTH_BETA = ROOT/'research'/'hillclimb_2026-10-01'/'team_strength_beta.json'
KICKING_GAMMA = 1.5
MODEL_WEIGHT = 0.75
MATCHUP_COEFFICIENTS = ROOT/'research'/'hillclimb_2026-10-02'/'matchup_coefficients.json'
MATCHUP_MODEL_WEIGHT = 0.7


def adjust_raw(raw: list[RawPrediction], candidates: pd.DataFrame, history: pd.DataFrame,
               beta: dict[str, float] | None = None) -> list[RawPrediction]:
    """Team-strength calibration then kicking concentration, using only ``history``."""
    beta = json.loads(TEAM_STRENGTH_BETA.read_text()) if beta is None else beta
    state = pre_match_context(history)[1]
    edges = add_candidate_context(candidates, state)['ctx__elo_edge']
    calibrated = [team_strength_scale(p, float(edge), beta) for p, edge in zip(raw, edges)]
    return concentrate_kicking(calibrated, KICKING_GAMMA)


def blend_points(model_points: np.ndarray, empirical_points: np.ndarray) -> np.ndarray:
    return MODEL_WEIGHT*np.asarray(model_points, float) + (1 - MODEL_WEIGHT)*np.asarray(empirical_points, float)


def adjust_raw_matchup(raw: list[RawPrediction], candidates: pd.DataFrame, history: pd.DataFrame,
                       cutoff: pd.Timestamp, coefficients: dict | None = None) -> list[RawPrediction]:
    """Round-2 raw steps: matchup calibration then kicking concentration, using only ``history``."""
    coefficients = json.loads(MATCHUP_COEFFICIENTS.read_text()) if coefficients is None else coefficients
    calibrated = calibrate_matchups(raw, candidates, history, cutoff, coefficients)
    return concentrate_kicking(calibrated, KICKING_GAMMA)


def blend_points_matchup(model_points: np.ndarray, empirical_points: np.ndarray) -> np.ndarray:
    weight = MATCHUP_MODEL_WEIGHT
    return weight*np.asarray(model_points, float) + (1 - weight)*np.asarray(empirical_points, float)
