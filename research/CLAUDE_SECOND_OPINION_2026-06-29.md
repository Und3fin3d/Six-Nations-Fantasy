# Claude Code second opinion - 2026-06-29

Claude Code 2.1.178 was given the corrected experiment evidence and asked for an
independent reasoning review, not implementation verification.

## Opinion

- Drop the closing-market promotion unconditionally. Retrospectively stamped
  closing prices are point-in-time leakage, and better retrospective scores are
  expected when leaked information is useful.
- Keep the opening-odds captain tilt as research-only. Its `+0.001797`
  `value_team` gain is below the predeclared `+0.003` gate and comes from one
  captain decision.
- Restore the clean orchestrator in production.
- Reject the Elo branch. The implementation bug must be fixed, but even
  corrected Elo is too data-hungry and unstable for the available Six Nations
  history.

## Suggested next work

1. Confirmed team-sheet bench roles and direct bench-minutes modelling.
2. Coach-by-position substitution timing priors from play-by-play or substitution
   events.
3. Position-specific opponent matchup features from historical match statistics.

The reviewer also warned that five-round comparisons have very low statistical
power, the clean orchestrator itself must remain point-in-time clean, and a
single-round captain change should not be interpreted as robust evidence.
