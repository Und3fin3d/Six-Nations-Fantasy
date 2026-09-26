import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter

from ingest_6n import derive_minutes, player_minutes


EXPECTED = 'e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be'


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def write_json(path, value):
    with open(path, 'w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def review(args):
    if digest(args.store) != EXPECTED:
        raise ValueError('Duration review requires the unchanged registered corrected store')
    os.makedirs(args.output, exist_ok=False)
    exceptional, counts = [], Counter()
    with open(args.store, newline='') as stream:
        for row in csv.DictReader(stream):
            counts['rows'] += 1
            available = row['available__minutes'].lower() == 'true'
            minutes = numeric(row['minutes'])
            if available:
                counts['observed_rows'] += 1
                if not math.isfinite(minutes) or minutes < 0 or minutes > 80:
                    exceptional.append(row)
    fields = ['fixture_id', 'player_id', 'player_name', 'team', 'date', 'competition',
              'competition_level', 'source', 'minutes', 'available__minutes', 'provenance__minutes', 'started']
    with open(f'{args.output}/outside_registered_domain.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in exceptional)
    sources, evidence = {}, []
    for fixture in sorted({row['fixture_id'] for row in exceptional}):
        path = f'data/cache/match_{fixture}.json'
        sources[path] = digest(path)
        payload = json.load(open(path)).get('results', {})
        match = payload.get('match') or {}
        events = payload.get('events') or []
        off, on, _ = derive_minutes(events)
        players = []
        for row in (item for item in exceptional if item['fixture_id'] == fixture):
            matches = []
            for side in ('home', 'away'):
                if str(match.get(f'{side}_team')) != row['team']:
                    continue
                for player in (payload.get(side) or {}).get('teamsheet') or []:
                    if str(player.get('player_id')) == row['player_id']:
                        matches.append(player)
            if len(matches) != 1:
                raise ValueError(f'Non-unique cached duration source: {fixture}/{row["player_id"]}/{row["team"]}')
            player = matches[0]
            pid = player['player_id']
            started = not bool(player.get('substitute', False))
            rebuilt = player_minutes(pid, started, off, on)
            if rebuilt != numeric(row['minutes']):
                raise ValueError('Current minute parser does not reproduce the audited source value')
            relevant = [event for event in events if str(event.get('player_1_id')) == str(pid)
                        or str(event.get('player_2_id')) == str(pid)]
            players.append(dict(canonical={key: row.get(key) for key in fields},
                                cached_player=player, rebuilt_minutes=rebuilt, player_events=relevant))
        evidence.append(dict(fixture_id=fixture, match=match, players=players,
                             events_after_80=[event for event in events if numeric(event.get('time')) > 80],
                             event_types=dict(Counter(str(event.get('type')) for event in events)),
                             cache_path=path, cache_sha256=sources[path]))
    write_json(f'{args.output}/source_evidence.json', evidence)
    if any(digest(path) != value for path, value in sources.items()) or digest(args.store) != EXPECTED:
        raise ValueError('Read-only duration evidence changed')
    summary = dict(counts, exceptional_rows=len(exceptional), exceptional_fixtures=len(evidence),
                   exceptional_by_source=dict(Counter(row['source'] for row in exceptional)),
                   exceptional_by_level=dict(Counter(row['competition_level'] for row in exceptional)),
                   range=[min(numeric(row['minutes']) for row in exceptional), max(numeric(row['minutes']) for row in exceptional)] if exceptional else None,
                   store_sha256=EXPECTED, parser_sha256=digest('ingest_6n.py'), source_hashes=sources,
                   script_sha256=digest(__file__), rugby_api_calls=0, candidate_fits=0,
                   data_changed=False, objective='active_unfinished',
                   interpretation='Source review, not inferred true duration or evidence that a model hypothesis failed. No outcome-based candidate choice.')
    write_json(f'{args.output}/manifest.json', summary)
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--output', required=True)
    review(parser.parse_args())
