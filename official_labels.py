import numpy as np
import pandas as pd

from compare_api_official import norm_key


def official_name_key(name, team):
    key = norm_key(name)
    return 'j|brex' if team == 'Italy' and key in {'j|ignaciobrex', 'n|brex'} else key


def match_official(players, official):
    keys = ['season', 'round', 'team', 'name_key']
    rows = players.reindex(columns=['season', 'round', 'team', 'player_name', 'minutes']).copy()
    rows['player_row'] = np.arange(len(rows))
    rows['name_key'] = [official_name_key(n, t) for n, t in zip(rows.player_name, rows.team)]
    labels = official.copy()
    labels['official_row'] = np.arange(len(labels))
    labels['name_key'] = [official_name_key(n, t) for n, t in zip(labels.name, labels.team)]
    rows['ambiguous_player'] = rows.duplicated(keys, keep=False)
    labels['ambiguous_official'] = labels.duplicated(keys, keep=False)
    pairs = rows.merge(labels, on=keys, how='inner')
    ambiguous = pairs.ambiguous_player | pairs.ambiguous_official
    minutes_match = (pd.to_numeric(pairs.minutes) - pd.to_numeric(pairs.get('Min', pd.Series(np.nan, index=pairs.index)))).abs().le(2)
    pairs = pairs[~ambiguous | minutes_match].copy()
    if pairs.player_row.duplicated().any() or pairs.official_row.duplicated().any():
        raise ValueError('Official player identity is ambiguous; explicit source evidence is required')
    pairs['label_source'] = np.where(
        pairs.ambiguous_player | pairs.ambiguous_official,
        'official_workbook_minutes_identity', 'official_workbook_name_identity')
    columns = list(official.columns) + ['label_source']
    result = pairs.set_index('player_row')[columns].reindex(range(len(players)))
    result.index = players.index
    return result
