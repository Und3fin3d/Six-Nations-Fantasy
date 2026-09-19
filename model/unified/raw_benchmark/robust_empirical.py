"""Small competition-independent challenger: safer minutes and rate priors.

The legacy empirical engine remains the control. No fantasy outcomes or
competition identifier select a parameter in this challenger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.history import past_matches, utc_cutoff
from .empirical import EmpiricalEventModel, HALFLIFE_DAYS


class RobustEmpiricalEventModel(EmpiricalEventModel):
    """Exposure-weighted position rates and shrunk international-role minutes."""

    def fit(self, frame: pd.DataFrame) -> 'RobustEmpiricalEventModel':
        train = past_matches(frame, self.asof)
        super().fit(train)
        intl = train[train.competition_level.eq('international')].copy()
        dates = pd.to_datetime(intl.date, utc=True)
        intl['_weight'] = np.power(0.5, (utc_cutoff(self.asof)-dates).dt.days.clip(lower=0)/HALFLIFE_DAYS)
        minutes = pd.to_numeric(intl.minutes, errors='coerce')
        # A one-minute cameo must not count like an eighty-minute observation
        # when estimating a positional per-80 prior.
        for event in self.events:
            value = pd.to_numeric(intl[event], errors='coerce')
            valid = intl[f'available__{event}'].fillna(False).astype(bool) & minutes.gt(0) & value.notna()
            observed = intl.loc[valid, ['position', '_weight']].copy()
            observed['_event'] = value[valid]*observed['_weight']
            observed['_minutes'] = minutes[valid]*observed['_weight']
            sums = observed.groupby('position')[['_event','_minutes']].sum()
            for position, row in sums.iterrows():
                self.position_priors[(str(position),event)] = float(80*row['_event']/row['_minutes'])
        valid = intl.available__minutes.fillna(False).astype(bool) & minutes.notna()
        observed = intl[valid].copy()
        observed['_wm'] = minutes[valid]*observed['_weight']
        self.minutes_by_player = {}
        self.minutes_by_position = {}
        position = observed.groupby(['position','started'])[['_weight','_wm']].sum()
        for (pos, started), row in position.iterrows():
            self.minutes_by_position[(str(pos),bool(started))] = float(row['_wm']/row['_weight'])
        for (player, started), group in observed.groupby(['player_id','started']):
            # Four recency-weighted appearances of prior strength, shared by
            # every tournament. Club starting minutes cannot overwrite a test
            # bench role; there is no ad-hoc NCR minutes adjustment.
            pos = str(group.sort_values('date').iloc[-1].position)
            prior = self.minutes_by_position.get((pos,bool(started)),70.0 if started else 20.0)
            self.minutes_by_player[(str(player),bool(started))] = float(
                (group['_wm'].sum()+4.0*prior)/(group['_weight'].sum()+4.0))
        return self
