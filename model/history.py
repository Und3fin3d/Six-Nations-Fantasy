from __future__ import annotations

import pandas as pd


def utc_cutoff(value: str | pd.Timestamp) -> pd.Timestamp:
    cutoff = pd.Timestamp(value)
    if pd.isna(cutoff):
        raise ValueError('a finite prediction cutoff is required')
    return cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')


def past_matches(frame: pd.DataFrame, asof: str | pd.Timestamp) -> pd.DataFrame:
    column = 'match_at' if 'match_at' in frame else 'date'
    cutoff = utc_cutoff(asof)
    if column == 'date':
        cutoff = cutoff.normalize()
    else:
        cutoff -= pd.Timedelta(hours=3)
    timestamps = pd.to_datetime(frame[column], errors='coerce', utc=True)
    return frame.loc[timestamps.notna() & timestamps.lt(cutoff)].copy()


def past_seasons(frame: pd.DataFrame, asof: str | pd.Timestamp) -> pd.DataFrame:
    years = frame['season'].astype(str).str.findall(r'\d{4}')
    end_year = years.map(lambda values: int(values[-1]) if values else None)
    split = years.map(len).ge(2)
    year = end_year.where(split, end_year + 1).astype('Int64').astype(str)
    month_day = split.map({True: '-07-01', False: '-01-01'})
    available_at = pd.to_datetime(year + month_day, errors='coerce', utc=True)
    return frame.loc[available_at.lt(utc_cutoff(asof))].copy()
