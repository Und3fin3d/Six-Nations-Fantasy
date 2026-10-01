"""Large computed-rubric fantasy benchmark over every cached match (research only).

The official benchmark has 13 rounds with official fantasy points. That is far
too few to separate forecasting models whose differences are a few hundredths
of a point. This package scores every cached club and international match
from July 2023 under deterministic, API-observable fantasy rubrics, with
point-in-time forecasts refitted at shared monthly locks, and evaluates both
forecast accuracy and squad-selection decisions on synthetic slates.

Stages (``python -m research.big_benchmark --stage ...``):

``manifest``  frozen locks, blocks and slates plus input hashes;
``fit``       one robust P3 fit per lock, raw forecasts for every candidate,
              pre-lock matchup context and empirical baselines (resumable);
``score``     per-player expected and actual points for every engine and rubric;
``decide``    optimiser squads, smoothed squads and hindsight optima per slate;
``report``    pooled metrics, paired bootstraps and the power analysis.

Nothing here changes production routing, promotes a model or calls RapidAPI.
"""
