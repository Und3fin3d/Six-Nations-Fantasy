"""Unified, competition-agnostic rugby event prediction.

The public seam is deliberately small: build a canonical match store, train one
raw-event model, then pass its :class:`RawPrediction` objects through a scoring
adapter.  Competition rules never enter the learned model.
"""

from .contracts import EventDistribution, RawPrediction
from .scoring import NationsChampionshipScorer, SixNationsScorer, scorer_for

__all__ = [
    "EventDistribution",
    "RawPrediction",
    "NationsChampionshipScorer",
    "SixNationsScorer",
    "scorer_for",
]
