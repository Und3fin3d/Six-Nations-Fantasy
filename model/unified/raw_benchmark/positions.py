import numpy as np
import pandas as pd

from ..schema import FORWARD_POSITIONS, POSITION_BY_JERSEY


def prior_supported_positions(rows, history):
    starts = history.loc[history.jersey.between(1,15), ['player_id','match_at','jersey']].copy()
    starts['available_at'] = pd.to_datetime(starts.match_at, utc=True) + pd.Timedelta(hours=3)
    starts['prior_position'] = starts.jersey.map(POSITION_BY_JERSEY)
    query = rows[['player_id','match_at']].copy()
    query['match_at'] = pd.to_datetime(query.match_at, utc=True)
    query['row_order'] = np.arange(len(query))
    matched = pd.merge_asof(
        query.sort_values('match_at'),
        starts[['player_id','available_at','prior_position']].sort_values('available_at'),
        left_on='match_at', right_on='available_at', by='player_id', allow_exact_matches=False)
    matched = matched.sort_values('row_order')
    result = rows.copy()
    starting = result.jersey.between(1,15)
    result['position'] = matched.prior_position.to_numpy()
    result.loc[starting,'position'] = result.loc[starting,'jersey'].map(POSITION_BY_JERSEY)
    result['position_source'] = np.where(starting, 'starting_jersey', 'prior_recorded_start')
    result.loc[result.position.isna(),'position_source'] = 'unknown_no_prior_start'
    result['position'] = result.position.fillna('Unknown')
    result['is_forward'] = result.position.isin(FORWARD_POSITIONS)
    return result
