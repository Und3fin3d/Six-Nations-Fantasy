import numpy as np
import pandas as pd

import build_features as features
import build_team_play as teamplay
from model.history import past_matches
from official_labels import match_official


class IncumbentFeatureHistory:
    def __init__(self, root, store, native):
        timestamps = store[['fixture_id', 'match_at']].drop_duplicates('fixture_id')
        timestamps['fixture_id'] = timestamps.fixture_id.astype(int)
        self.players = pd.read_csv(root/'data/api_player_match.csv')
        self.teams = pd.read_csv(root/'data/api_team_match.csv')
        for frame in (self.players, self.teams):
            frame['date'] = pd.to_datetime(frame.date)
        self.players = self.players.merge(timestamps, on='fixture_id', how='left', validate='many_to_one')
        self.teams = self.teams.merge(timestamps, on='fixture_id', how='left', validate='many_to_one')
        official = pd.read_csv(root/'data/official_player_match.csv')
        labels = match_official(self.players[self.players.comp_id.eq(1266)], official.dropna(subset=['Pts']))
        self.players['potm'] = labels.POTM
        self.players['has_potm_label'] = self.players.potm.notna()
        self.players['potm'] = self.players.potm.fillna(0)
        positions = native.drop_duplicates('player_id').set_index('player_id').canonical_pos
        self.players['_matchup_pos'] = self.players.player_id.map(positions)
        self.players['_matchup_is_forward'] = self.players._matchup_pos.isin(features.FORWARD_GROUPS)

    def lock_features(self, native, slate):
        cutoff = slate.cutoff
        players = past_matches(self.players, cutoff)
        teams = past_matches(self.teams, cutoff)
        current = native.season.eq(slate.season) & native['round'].eq(slate.round)
        out = native.copy()
        player_history = dict(tuple(players.groupby('player_id')))
        team_players = dict(tuple(players.groupby('team_id')))
        opponents = dict(tuple(players.groupby('opponent_id')))
        team_history = dict(tuple(teams.groupby('team_id')))
        empty_players, empty_teams = players.iloc[:0], teams.iloc[:0]
        for idx, row in out[current].iterrows():
            values = features.form_role_features(player_history.get(row.player_id, empty_players), row.date, 90.0)
            values.update(features.team_bench_slot_features(team_players.get(row.team_id, empty_players), row.date, row.jersey, 90.0))
            values.update(features.ownteam_features(team_history.get(row.team_id, empty_teams), row.date, 90.0))
            values.update(features.matchup_features(opponents.get(row.opponent_id, empty_players), row.date, row.canonical_pos, row.is_forward, 90.0))
            for target in ('metres', 'tries'):
                values[f'form_hot_{target}'] = values[f'form_per80_{target}'] - row[f'class_per80_{target}']
            values['has_form'] = values['form_n_prior'] > 0
            for column, value in values.items():
                if column in out.columns:
                    out.at[idx, column] = value
        forecasts = self.teamplay_at_lock(teams, slate)
        forecast_cols = [c for c in forecasts if c.startswith('teamplay_') and c in out]
        keyed = forecasts.set_index(['fixture_id', 'team_id'])
        for idx, row in out[current].iterrows():
            out.loc[idx, forecast_cols] = keyed.loc[(row.fixture_id, row.team_id), forecast_cols].to_numpy()
        changes = []
        for col in out.select_dtypes(include=[np.number, bool]).columns:
            if col not in native or not current.any():
                continue
            changed = ~np.isclose(out.loc[current, col].to_numpy(float), native.loc[current, col].to_numpy(float), equal_nan=True)
            if changed.any():
                changes.append({'column': col, 'changed_rows': int(changed.sum())})
        return out, {'feature_changes': changes, 'player_history_rows': len(players), 'team_history_rows': len(teams),
                     'latest_player_match': str(players.match_at.max()), 'latest_team_match': str(teams.match_at.max()),
                     'player_rows_without_kickoff': int(self.players.match_at.isna().sum()),
                     'team_rows_without_kickoff': int(self.teams.match_at.isna().sum())}

    def teamplay_at_lock(self, prior, slate):
        fixtures = slate.candidates.fixture_id.astype(int).unique()
        current = self.teams[self.teams.fixture_id.isin(fixtures)].copy()
        current['date'] = slate.cutoff.tz_localize(None).normalize()
        history = prior[prior.comp_id.eq(teamplay.SIX_NATIONS)]
        raw = teamplay._add_targets(pd.concat([history, current], ignore_index=True))
        ctx = teamplay._merge_wr(teamplay._context_rows(raw, 90.0))
        targets = list({target for target, _ in teamplay.PREDICTION_TARGETS.values()})
        frame = ctx.merge(raw[['fixture_id', 'team_id', *targets]], on=['fixture_id', 'team_id'], validate='one_to_one')
        feature_cols = [c for c in frame if c.startswith(('ctx_', 'teamplay_')) or c == 'home_away_num']
        is_current = frame.fixture_id.isin(fixtures)
        frame.loc[is_current, targets] = np.nan
        lock_day = current.date.iloc[0]
        frame['date'] = np.where(is_current, lock_day, lock_day-pd.Timedelta(days=1))
        out = frame[['fixture_id', 'team_id', 'team', 'opponent_id', 'opponent', 'date',
                     'teamplay_team_n_prior', 'teamplay_opp_n_prior', 'teamplay_h2h_n_prior']].copy()
        for column, (target, kind) in teamplay.PREDICTION_TARGETS.items():
            out[column] = teamplay._predict_by_date(frame, feature_cols, target, kind)
        return teamplay._add_aspects(out)[is_current].copy()
