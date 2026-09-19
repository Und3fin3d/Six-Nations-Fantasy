# Full-teamsheet correction before acceptance

The original 15 fixture IDs and separate 12 development fixture IDs stay fixed. An input-coverage audit, not a model-error comparison, found that the canonical store tracks only one side of four selected test fixtures (Georgia/Japan, Spain/Fiji, Italy/Chile and Scotland/Tonga). Some development fixtures have the same limitation. The permanent match JSON caches contain both teamsheets.

The metric therefore evaluates every player on both cached teamsheets. No selected fixture is replaced, and no team is dropped to improve an error. Untracked opponents are explicit cold-start cases; their canonical history is limited, with any existing club history still available. Training data stays unchanged to isolate the declared weighting experiment. Positions for new players are inferred from pre-match jersey slots using the existing schema. Missing stat keys remain missing, never copied from an anchor player or filled as zeros.

The original one-sided local smoke test and workflow 35475086184 are superseded, regardless of their outcomes. The workflow was stopped before accepting a candidate; its incomplete-cohort selection, if generated, is invalid. A new full-cohort development selection must be frozen before the Friendly-15 and fantasy confirmation jobs run. Candidate definitions, selection rules and interpretation limits in PLAN.md are unchanged.

Tests assert inclusion of an untracked opponent, preserved zero-minute substitutes, missing-stat isolation, rejection of missing teams/duplicate keys, and masking all current targets before feature construction.
