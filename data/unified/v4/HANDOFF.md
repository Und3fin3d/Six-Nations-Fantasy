# Handoff: research and adversarially plan the next unified rugby model

## Mission

Research a genuinely new competition-independent rugby forecasting model for
`/Users/williamgreenfield/Documents/dev/6n`, then produce a rigorous, executable
plan. **Do not implement or tune a new model in this session.** Planning must
finish, survive independent adversarial review, and be approved by the user
before code or expensive experiments begin.

The objective remains one model that learns general rugby knowledge, predicts
competition-independent outcomes, and can serve both Six Nations and Nations
Championship Rugby without competition-specific fantasy rank stacks. Do not
assume the v3 exposure/rate design is the right foundation. Reuse it, replace
it, or propose a different target structure only when evidence supports that
choice.

## Current evidence: read, do not recreate

Start by reading these artifacts:

- Full v3 benchmark and fold-level comparison:
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/benchmark/benchmark.md`
- Machine-readable decision:
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/benchmark/decision.json`
- Fold metrics and predictions:
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/benchmark/fold_metrics.csv`
  and
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/benchmark/predictions.csv`
- Data/scorer audit:
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/audit/audit.md`
  and its CSVs in the same directory
- Search results and admitted feature blocks:
  `/Users/williamgreenfield/Documents/dev/6n/data/unified/v3/search/`
- v3 implementation and contracts:
  `/Users/williamgreenfield/Documents/dev/6n/model/unified/v3/`
- Stable raw-prediction and scoring seam:
  `/Users/williamgreenfield/Documents/dev/6n/model/unified/contracts.py`
  and
  `/Users/williamgreenfield/Documents/dev/6n/model/unified/scoring.py`
- v3 regression tests:
  `/Users/williamgreenfield/Documents/dev/6n/tests/test_unified_v3.py`
- Workflow documentation:
  `/Users/williamgreenfield/Documents/dev/6n/model/unified/README.md`

Essential result: the leak-free v1-style non-neural baseline was retained.
GBDT-v3 slightly improved official-points MAE but worsened rolling raw-event
deviance by about 6.7% and had inconsistent ranking capture, especially NCR
top-10. All 20 neural-v3 configurations materially regressed versus GBDT-v3
and the neural track stopped after rung one. The selected baseline failed the
retrospective NCR safe-overall gate, so no unified model was promoted. Treat
this as negative evidence to explain, not an invitation to make a more complex
model by default.

The repository has a dirty/untracked worktree containing user work. Preserve
all unrelated changes. This session is read-only except for the final planning
artifact requested below.

## Mandatory independent-agent protocol

Use subagents because the user explicitly requests adversarial, independent
checking. Every subagent must be spawned with `fork_turns="none"` and given a
self-contained task containing the absolute evidence paths above. Agents must
not receive the parent conversation, must not read one another's drafts during
the independent-proposal phase, and must write to separate temporary paths.

Use all available concurrency slots productively, but never claim independence
between agents that inherited or shared conversational context.

### Phase 1: independent diagnosis and proposals

Run three independent workstreams in parallel:

1. **Failure diagnostician** — determine why v1 direct totals beat v3 raw
   exposure/rates on deviance while v3 sometimes improved fantasy MAE. Audit
   target parameterisation, scoring-weighted loss, missingness, minutes and
   appearance assumptions, conditional-rate bias, low-history behaviour,
   domain shift, calibration, and evaluation power. Separate demonstrated
   causes from hypotheses and specify cheap falsification tests.
2. **Model researcher** — propose 2–4 substantially different unified model
   families. At least one must be a disciplined extension of the strong v1
   baseline, and at least one may rethink the target representation. Consider
   partial pooling/hierarchical effects, joint event dependence, empirical
   Bayes shrinkage, hurdle/zero-inflated targets, direct totals plus auxiliary
   rugby tasks, player-state representations, and uncertainty. Do not propose
   fashionable complexity without data-to-parameter and validation arguments.
3. **Evaluation adversary** — attempt to falsify the entire unified-model goal
   and audit whether the current comparison can distinguish model quality from
   sparse NCR noise. Challenge cutoffs, cohorts, specialist comparators,
   top-N capture variance, raw-event weighting, observable-points construction,
   retrospective contamination, multiple-testing risk, and the feasibility of
   promotion before more NCR labels exist. Propose stopping rules.

Each workstream must return:

- claims with direct file/metric evidence;
- unknowns and assumptions;
- ranked experiments with expected information gain, cost and failure signal;
- explicit reasons not to pursue tempting alternatives.

### Phase 2: first plan, still no implementation

The primary agent independently writes an initial plan after reading the three
reports. It must include:

- a causal diagnosis of v3's failure;
- a small hypothesis tree, not a grab bag of models;
- the recommended model family and one credible fallback;
- target definitions and the exact raw-to-fantasy conversion boundary;
- leakage-safe features and prohibited features;
- data sufficiency and parameter-count reasoning;
- rolling folds, selection/reference/prospective layers and immutable artifacts;
- baseline, specialist and ablation comparisons;
- primary metrics, promotion margins and uncertainty reporting;
- compute budgets, successive elimination and explicit stop conditions;
- implementation interfaces, tests, migration and rollback;
- a sequence of cheap discriminating experiments before any full build.

Do not tune against Six Nations 2026 or NCR GW1–2; they remain retrospective
references. Do not weaken the existing safe-overall promotion rule merely to
make a unified model pass.

### Phase 3: independent cross-examination

Spawn at least two **new** adversarial reviewers with `fork_turns="none"`.
They must not be the Phase 1 authors.

- Reviewer A receives only the user objective, evidence paths, hard constraints
  and the primary agent's initial plan. It must identify leakage, circular
  selection, underpowered claims, unjustified complexity and missing falsifiers.
- Reviewer B receives the same materials plus the Phase 1 reports, but no prior
  reviewer discussion. It must construct the strongest alternative plan and
  identify any decision that depends on taste rather than evidence.

The primary agent must answer every material criticism explicitly: accept and
change the plan, reject with file-backed evidence, or mark as an unresolved
decision for the user. Do not silently average incompatible recommendations.

### Phase 4: final red-team gate

After synthesis, spawn one final fresh-context red-team agent using
`fork_turns="none"`. Give it only the proposed final plan, the original mission,
the hard constraints and the evidence paths. Ask it to decide:

- Is the plan temporally leak-free?
- Can its first experiments actually falsify the central hypothesis?
- Does it compare fairly with v1 and both specialist incumbents?
- Is the sample size adequate for every proposed learned component?
- Are promotion and stopping rules precommitted?
- Would a failed result leave the repository and incumbents safe?

The final plan may be delivered only after all correctness objections from this
gate are resolved or clearly surfaced to the user.

## Research constraints

- Preserve `RawPrediction` and deterministic Six Nations/NCR scorers unless a
  proposed interface change has a compelling, backward-compatible migration.
- One global model or globally shared parameters only. No competition-specific
  fantasy-points calibrators, rank stacks or blend weights.
- Existing context data may be audited; no new paid/external-data collection is
  assumed.
- Point-in-time safety is non-negotiable. Historical weather observations are
  not pre-lock forecasts.
- Official points are evaluation outcomes, not reconstructed training truth.
- Use ranks/capture to compare across scoring systems, but retain raw-event and
  calibration diagnostics.
- Keep v1 as the universal baseline and both specialists as deployment
  incumbents.
- Prefer experiments that isolate one disputed modelling assumption at a time.
- More model complexity requires an explicit sample-size, regularisation and
  out-of-time validation justification.
- If evidence says one universal model is not currently achievable, say so and
  design the minimum data-collection/shadow programme needed to revisit it.

## Required output

Write one final artifact only:

`/Users/williamgreenfield/Documents/dev/6n/data/unified/v4/RESEARCH_PLAN.md`

It must contain:

1. executive recommendation and confidence;
2. evidence-backed diagnosis of v3;
3. options considered and rejected;
4. chosen hypothesis and fallback;
5. staged experiment and implementation plan;
6. frozen evaluation protocol and promotion gates;
7. compute/data budget;
8. risks, stopping rules and rollback;
9. an adversarial-review ledger showing each criticism and resolution;
10. unresolved questions requiring user choice.

End the session after presenting this plan and the independent-review verdicts.
Do not modify model code, rerun expensive tuning, or start implementation until
the user explicitly approves the plan.

## Suggested skills

- `diagnose` — disciplined root-cause analysis of v3's failure.
- `grill-with-docs` — challenge the draft against the benchmark and audit files.
- `data-analytics:validate-data` — validate metric/cohort and data-quality claims.
- `data-analytics:metric-diagnostics` — decompose why raw deviance, MAE and
  ranking capture disagree.
- `improve-codebase-architecture` — only after the modelling hypothesis is
  settled, to check interface and migration consequences.

