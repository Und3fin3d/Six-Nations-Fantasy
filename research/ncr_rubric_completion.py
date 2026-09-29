import argparse
import json
import os

import numpy as np
import pandas as pd

from model.unified.contracts import RawPrediction
from model.unified.scoring import NationsChampionshipScorer
from research.conditional_duration_inputs import digest, write_json
from research.conditional_duration_metrics import only_metres_points


VERSION = 'ncr_missing_attack_terms_v3'
LEGACY = 'ncr_front_row_v2'
REGISTRATION = 'research/DECISION_NCR_RUBRIC_COMPLETENESS_ADDENDUM_2026-09-26.json'


def missing_attack_expectation(predictions, *, metre_floor=None):
    missing = [(prediction.fixture_id, prediction.player_id, event)
               for prediction in predictions for event in ('metres', 'fifty_22') if event not in prediction.events]
    if missing:
        raise ValueError(f'NCR rubric completion needs explicit missing-term forecasts: {missing[:5]}')
    floor = only_metres_points(predictions) if metre_floor is None else np.asarray(metre_floor, dtype=float)
    means = np.array([prediction.events['metres'].mean for prediction in predictions])
    if floor.shape != means.shape or not np.isfinite(floor).all():
        raise ValueError('Metre-floor expectations must cover every player')
    if np.any(floor < np.maximum(0, means / 10 - 1) - 1e-7) or np.any(floor > means / 10 + 1e-7):
        raise ValueError('Metre-floor expectation contradicts its declared raw mean')
    kicks = np.array([prediction.events['fifty_22'].mean for prediction in predictions])
    return floor + 2 * kicks


def expected_ncr_points(predictions, *, version=VERSION, metre_floor=None):
    if version not in (VERSION, LEGACY):
        raise ValueError(f'Unregistered NCR rubric version: {version}')
    scorer = NationsChampionshipScorer(version=LEGACY)
    original = np.array([scorer.expected_points(prediction) for prediction in predictions])
    if version == LEGACY:
        if metre_floor is not None:
            raise ValueError('The preserved v2 contract has no metre-floor term')
        return original
    return original + missing_attack_expectation(predictions, metre_floor=metre_floor)


def complete_native_points(original_points, empirical_predictions):
    original = np.asarray(original_points, dtype=float)
    if original.shape != (len(empirical_predictions),) or not np.isfinite(original).all():
        raise ValueError('Native completion requires the complete unchanged native point pool')
    return original + missing_attack_expectation(empirical_predictions)


def score_ncr_events(events, *, position, version=VERSION):
    if version not in (VERSION, LEGACY):
        raise ValueError(f'Unregistered NCR rubric version: {version}')
    scorer = NationsChampionshipScorer(version=LEGACY)
    original = scorer.score_samples(events, is_forward=position in ('Prop', 'Hooker', 'Lock', 'Loose Forward'), position=position)
    if version == LEGACY:
        return original
    for event in ('metres', 'fifty_22'):
        if event not in events:
            raise ValueError(f'Deterministic NCR rubric completion requires observed {event}, not an implicit zero')
    return original + np.floor(np.asarray(events['metres'], dtype=float) / 10) + 2 * np.asarray(events['fifty_22'], dtype=float)


def run(args):
    if os.path.exists(args.output) or os.path.exists(args.output + '.json'):
        raise FileExistsError('Choose new output paths; existing research evidence is immutable')
    predictions = [RawPrediction.from_dict(json.loads(line)) for line in open(args.forecasts) if line.strip()]
    original = expected_ncr_points(predictions, version=LEGACY)
    completed = expected_ncr_points(predictions)
    frame = pd.DataFrame([dict(fixture_id=p.fixture_id, player_id=p.player_id, team=p.team,
                              original_v2=a, completed_v3=b, missing_terms=b-a)
                          for p, a, b in zip(predictions, original, completed)])
    if frame.duplicated(['fixture_id', 'player_id', 'team']).any():
        raise ValueError('Duplicate forecast identities')
    frame.to_csv(args.output, index=False)
    write_json(args.output + '.json', dict(registration=REGISTRATION, scoring_version=VERSION,
        original_scoring_version=LEGACY, forecast_sha256=digest(args.forecasts), rows=len(frame),
        output_sha256=digest(args.output), source_sha256=digest(__file__),
        missing_other_event_forecasts={event: sum(event not in p.events for p in predictions)
                                      for event in ('lineouts_won', 'lineout_errors', 'handling_errors', 'interceptions')},
        objective='active_unfinished', candidate_selected=False, production_default_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecasts', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
