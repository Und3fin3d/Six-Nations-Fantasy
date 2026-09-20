# Seed-averaging comparison — initial run is incomplete

No comparison result or promotion decision is valid yet. The initial workflow's shell pipeline hid a Python failure behind `tee`, so its green status does not establish completed model evaluation or a successful audit. The audit rejected the absent NCR round-2 candidate output. The candidate's ensemble alignment check had found duplicate `(fixture_id, player_id, team)` keys in the existing NCR round-2 cohort. The duplicate identities are being diagnosed before any relaxation or correction.

All original raw forecasts, partial results and logs are preserved in workflow `35478343154`. Do not use this run as a complete four-evaluation scoreboard, and do not silently omit NCR round 2. The preceding commit's report contained only the successful test log, not comparison results.

The source regression suite did complete: **186 tests passed in 46.20 seconds** at `3b18a40c8e149bdc52a9e726bb70f1aa3833e420`. That is a code-test result only, not evidence that all real-data cases passed. The completed source-test result remains valid; the comparison/audit success claim is withdrawn.

No model promotion, new PR, main change or change to PR #20.
