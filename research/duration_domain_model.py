import dataclasses

import numpy as np
import pandas as pd

from model.unified.conditional_duration import ConditionalDurationGBDT


DOMAINS = ('bounded', 'source_tail')


def duration_mask(frame, domain):
    if domain not in DOMAINS:
        raise ValueError(f'Unregistered duration domain: {domain}')
    minutes = pd.to_numeric(frame.minutes, errors='coerce').to_numpy(float)
    valid = frame.available__minutes.fillna(False).to_numpy(bool) & np.isfinite(minutes) & (minutes >= 0)
    if domain == 'bounded':
        valid &= minutes <= 80
    return valid


class DurationDomainGBDT(ConditionalDurationGBDT):
    def __init__(self, *, duration_domain, **kwargs):
        super().__init__(**kwargs)
        if duration_domain not in DOMAINS:
            raise ValueError(f'Unregistered duration domain: {duration_domain}')
        self.duration_domain = duration_domain
        self.duration_exclusions = []

    @staticmethod
    def _duration_labels(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Observed duration must be finite and nonnegative')
        return np.where(values == 0, 0, np.minimum(9, np.floor(values / 10) + 1)).astype(int)

    def fit(self, frame):
        valid = duration_mask(frame, self.duration_domain)
        observed = frame.available__minutes.fillna(False).to_numpy(bool)
        self.duration_exclusions = frame.loc[observed & ~valid,
            ['fixture_id', 'player_id', 'team', 'date', 'minutes', 'competition_level', 'started']].to_dict('records')
        working = frame.copy()
        working['available__minutes'] = valid
        self.duration_model = None
        self.duration_classes = np.array([], dtype=int)
        self.duration_centres = np.zeros(10)
        self.duration_variances = np.zeros(10)
        self.duration_counts = np.zeros(10, dtype=int)
        return super().fit(working)

    def predict_frame(self, frame):
        predictions = super().predict_frame(frame)
        return [dataclasses.replace(prediction, metadata={**prediction.metadata,
                'model': f'conditional_duration_{self.duration_domain}_v2',
                'duration_domain': self.duration_domain,
                'duration_source': 'registered_source_proxy_not_corrected_elapsed_time'})
                for prediction in predictions]
