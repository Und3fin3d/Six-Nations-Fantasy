import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from model.unified.scoring import NationsChampionshipScorer
from research.conditional_duration_inputs import digest, write_json
from research.conditional_duration_metrics import load_outcome_audit


SOURCE = 'data/unified/decision_continuation_2026-09-26/run-36259688139/official/public_pages.json'
SOURCE_BLOB = '338a439767709e8c26aaf4161c641c556f9a95c9'
AUDIT = 'data/unified/decision_continuation_2026-09-26/run-36272929019/outcomes'
RULES = {
    'tries': 12, 'try_assists': 5, 'conversion_goals': 2, 'missed_conversion_goals': -1,
    'penalty_goals': 3, 'missed_penalty_goals': -1, 'drop_goals_converted': 5,
    'fifty_22': 2, 'defenders_beaten': 2, 'offload': 2, 'clean_breaks': 3,
    'metres': 0.1, 'tackles': 1, 'missed_tackles': -1, 'tackle_turnover': 4,
    'interceptions': 5, 'lineouts_won': 1, 'lineout_steals': 5, 'scrums_won': 2,
    'potm': 15, 'penalties_conceded': -1, 'handling_errors': -1, 'lineout_errors': -2,
    'yellow_cards': -5, 'red_cards': -10,
}


def run(args):
    os.makedirs(args.output, exist_ok=False)
    content = open(SOURCE, 'rb').read()
    blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('The independently archived official rule source changed')
    sources = json.loads(content)
    rule = [page for page in sources if page['requested_url'].endswith('/en/rules/')]
    if len(rule) != 1 or 'Every 10 Metres Gained\t+1' not in rule[0]['text'] or 'Successful 50:22\t+2' not in rule[0]['text']:
        raise ValueError('Official primary text does not support the stated missing components')
    pool = load_outcome_audit()
    pool = pool[pool.competition.eq('ncr')].copy()
    labels_path = f'{AUDIT}/bridged_raw_labels.csv'
    labels = pd.read_csv(labels_path, low_memory=False)
    labels = labels[labels.slate.isin(pool.slate.unique())]
    if len(pool) != 758 or len(labels) != 758 or labels.duplicated(['slate', 'id']).any():
        raise ValueError('All three full NCR pools and their unknown raw labels are required')
    index = pd.MultiIndex.from_frame(pool[['slate', 'id']])
    labels = labels.set_index(['slate', 'id']).reindex(index).reset_index()
    if not labels[['slate', 'id']].equals(pool[['slate', 'id']].reset_index(drop=True)):
        raise ValueError('NCR source label alignment changed')
    hashes = {path: digest(path) for path in (SOURCE, labels_path, __file__, 'model/unified/scoring.py', 'model/ncr_project.py')}
    scorer = NationsChampionshipScorer(version='ncr_front_row_v2')
    components = []
    for event, weight in RULES.items():
        actual = pd.to_numeric(labels[event], errors='coerce') if event in labels else pd.Series(np.nan, index=labels.index)
        availability = labels.get(f'available__{event}', pd.Series(False, index=labels.index))
        observed = availability.fillna(False).to_numpy(bool) & np.isfinite(actual.to_numpy(float))
        included_weight = scorer.weights.get(event, 0.0)
        for offset, player in enumerate(pool.itertuples(index=False)):
            eligible = player.pos in ('Prop', 'Hooker') if event == 'scrums_won' else True
            existing_weight = scorer.scrum_weight(player.pos) if event == 'scrums_won' else included_weight
            value = float(actual.iloc[offset]) if observed[offset] else np.nan
            correct = np.floor(value / 10) if event == 'metres' else eligible * weight * value
            components.append(dict(slate=player.slate, id=player.id, name=player.name, pos=player.pos,
                                   status=player.status, event=event, observed=bool(observed[offset]),
                                   value=value, current_weight=existing_weight,
                                   official_weight=weight if eligible else 0.0,
                                   current_known_component=existing_weight*value if observed[offset] else np.nan,
                                   rule_known_component=correct,
                                   component_convention='floor_per_10' if event == 'metres' else 'per_recorded_event'))
    components = pd.DataFrame(components)
    components['known_omitted_component'] = components.rule_known_component - components.current_known_component
    components.to_csv(f'{args.output}/observed_components.csv', index=False)
    rows = []
    for (slate, player_id), group in components.groupby(['slate', 'id'], sort=False):
        old_weights = list(scorer.weights)
        old_parts, old_unknown = [], []
        source = labels[labels.slate.eq(slate) & labels.id.eq(player_id)].iloc[0]
        player = pool[pool.slate.eq(slate) & pool.id.eq(player_id)].iloc[0]
        for event in (*old_weights, 'scrums_won'):
            present = pd.notna(source.get(f'available__{event}')) and bool(source.get(f'available__{event}')) and pd.notna(source.get(event))
            weight = scorer.scrum_weight(player.pos) if event == 'scrums_won' else scorer.weights[event]
            if present:
                old_parts.append(weight * float(source[event]))
            elif weight:
                old_unknown.append(event)
        old_sum = float(sum(old_parts))
        metres = group[group.event.eq('metres')].iloc[0]
        kick = group[group.event.eq('fifty_22')].iloc[0]
        added_metres = metres.rule_known_component if metres.observed else np.nan
        added_kick = kick.rule_known_component if kick.observed else np.nan
        corrected_known = old_sum + (added_metres if metres.observed else 0.0) + (added_kick if kick.observed else 0.0)
        linear_known = old_sum + (metres.value / 10 if metres.observed else 0.0) + (added_kick if kick.observed else 0.0)
        rows.append(dict(slate=slate, id=player_id, name=player['name'], pos=player.pos, status=player.status,
                         official_points=player.official_points, matched_raw_key=player.source_key_matched,
                         old_known_component_sum=old_sum, old_unobserved_components=';'.join(old_unknown),
                         observed_rule_components=int(group.observed.sum()), rule_components=len(group),
                         all_rule_components_observed=bool(group.observed.all()),
                         known_metres_addition=added_metres, known_fifty22_addition=added_kick,
                         omitted_components_corrected_known_sum=corrected_known,
                         residual_before=player.official_points-old_sum,
                         residual_after=player.official_points-corrected_known,
                         linear_metres_residual=player.official_points-linear_known,
                         limitation='Partial reconstruction only; turnover-conceded proxy and unobserved rubric events not resolved. Unknown additions are not asserted zero.'))
    players = pd.DataFrame(rows)
    players.to_csv(f'{args.output}/partial_reconciliation.csv', index=False)
    summary = components.groupby(['event', 'pos'], as_index=False).agg(
        rows=('id', 'size'), observed=('observed', 'sum'), positive=('value', lambda values: values.gt(0).sum()),
        mean_observed=('value', 'mean'), current_weight=('current_weight', 'first'),
        official_weight=('official_weight', 'first'), known_rule_points=('rule_known_component', lambda values: values.sum(min_count=1)),
        known_omitted_points=('known_omitted_component', lambda values: values.sum(min_count=1)))
    summary.to_csv(f'{args.output}/component_coverage.csv', index=False)
    players.groupby(['status', 'pos'], as_index=False).agg(
        rows=('id', 'size'), raw_matches=('matched_raw_key', 'sum'), metres_observed=('known_metres_addition', 'count'),
        mean_known_metres=('known_metres_addition', 'mean'), residual_bias_before=('residual_before', 'mean'),
        residual_bias_after=('residual_after', 'mean'), residual_mae_before=('residual_before', lambda x: x.abs().mean()),
        residual_mae_after=('residual_after', lambda x: x.abs().mean())
    ).to_csv(f'{args.output}/reconciliation_summary.csv', index=False)
    feeds = {}
    for round_number in (1, 2, 3):
        path = f'data/ncr/feeds/players_gw{round_number}.json'
        hashes[path] = digest(path)
        records = json.load(open(path))['Data']['Value']['Players']
        fields = sorted({key for row in records for key in row})
        feeds[str(round_number)] = dict(rows=len(records), fields=fields,
            scoring_fields=[field for field in fields if any(token in field.lower() for token in ('point', 'stat', 'metre', 'break', 'lineout', 'error', 'scrum'))])
    write_json(f'{args.output}/official_feed_fields.json', feeds)
    if hashes != {path: digest(path) for path in hashes}:
        raise ValueError('Source data changed during scoring coverage audit')
    write_json(f'{args.output}/manifest.json', dict(source_hashes=hashes,
        source_text_sha256=rule[0]['text_sha256'], source_retrieved_at=rule[0]['fetched_at'],
        source_url=rule[0]['final_url'], candidate_fits=0, candidate_selections=0, rugby_api_calls=0,
        point_labels_changed=False, official_rows=len(players), matched_raw_rows=int(players.matched_raw_key.sum()),
        unknown_raw_rows=int((~players.matched_raw_key).sum()),
        status='Independent scoring/source coverage audit; not evidence of improved forecast or selection',
        chronology='Current archived rules do not alone establish original July publication time',
        objective='active_unfinished'))
    print(summary.to_string(index=False), flush=True)
    print(pd.read_csv(f'{args.output}/reconciliation_summary.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
