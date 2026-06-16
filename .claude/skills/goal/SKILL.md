---
name: goal
description: Autoresearch goal for the 6n picker — iteratively improve the 2025 dev selection value (value_xv) and points MAE via a Karpathy-style greedy hill-climb, with 2026 kept sealed. When the user types /goal, run one pass of the loop, read the ledger, report best-vs-baseline, and propose the next principled candidate. Trigger: /goal
---

# /goal — iteratively improve selection MAE + XV value

Branch: `cc`. Style: Karpathy-style autoresearch — a greedy hill-climb over a queue of concrete,
principled candidate changes, each evaluated on the held-out dev signal, accepted only on a Pareto
improvement, every trial logged. Reproducible, **0 API** (local CPU only).

## When `/goal` is invoked, do this
1. Run one pass: `/usr/local/bin/python3.11 -m model.research`.
2. Read `research/LEDGER.md` (+ `research/best_config.json`) and report the best config and its
   `value_xv` / `MAE` / `top15` vs the baseline (0.666 / 7.684 / 0.320), naming which candidates were
   accepted and why.
3. If nothing beat the incumbent, propose the **next** principled candidate (append to
   `CANDIDATES` in `model/research.py`) rather than re-running the same queue. Bias toward changes
   that reduce variance or fix a measured bias — not knob-fishing on 5 rounds.
4. Only when the user explicitly asks to check deployment: `python -m model.research --seal-2026`
   (evaluates the sealed season **once**; the number must never feed back into the search).

## Objective
Maximise **2025 value-of-XV** (the selection metric — picking a good XV is the product) and
minimise **2025 points MAE**, jointly. value_xv is primary; MAE is a co-objective and a guard.

Baseline to beat (current `main`/`cc` ensemble, post_team_sheet):

| metric | value |
|---|---|
| value_xv (2025) | 0.666 |
| MAE (2025) | 7.684 |
| top15 (2025) | 0.320 |
| spearman_pos (2025) | 0.469 |

## Dev signal (and why)
Only **2025** carries modern point labels (674 rows / 5 rounds). 2023 and 2024 have **zero**
modern-labelled rows; **2026 is sealed**. Therefore:
- The points/selection fitness can only be measured on 2025 — a small, 5-round sample.
- The **component-engine choice** is still validated independently on **2023+2024 component-rate
  GroupKFold OOF** (raw counts exist every season, independent of the scoring system). That OOF gate
  is the real anti-overfit guard for the component layer.
- The **assembly knobs** (minutes, POTM, latent shrink, calibration, selector) are tuned on 2025 —
  so a 5-round overfit is the central risk and is guarded explicitly (below).
- **2026 is never read inside the loop.** Evaluate it at most once, after convergence, for the final
  accepted config only; that number never feeds back into the search.

## Acceptance rule (Pareto + robustness)
A candidate replaces the incumbent **iff all hold**:
1. **Pareto improvement** on 2025: value_xv up by ≥ `δ_v = 0.003` with MAE not worse than `τ_mae =
   0.02`; **or** MAE down by ≥ `δ_m = 0.02` with value_xv not worse than `δ_v`.
2. **Round robustness**: the value_xv gain holds on **≥ 3 of 5** rounds (per-round value_xv), so a
   single lucky round can't carry a change.
3. **Hard invariants** (never relaxed): leakage asserts in `model/data.py` pass; the loop touches no
   2026 row; kicking mass on hard non-kickers is exactly 0; all rates ≥ 0; deterministic reruns.

Ties / sub-threshold trials are **rejected** and logged — bias toward the simpler incumbent
("simplest model that beats naive").

## Search
Greedy hill-climb from the incumbent (= current `main` behaviour). Each candidate is a small delta
off the *current* incumbent; accept ⇒ it becomes the new incumbent and later candidates build on it.
When a candidate changes a component-layer knob, the registry is re-derived from 2023+2024 OOF.

## Knobs (`Config` in `model/research.py`, defaults reproduce frozen behaviour)
- minutes model: ridge alpha, started/jersey×is_forward interaction, Poisson minutes
- component LGBM promotion margin (`LGBM_MARGIN`) and blend weight
- POTM expectation: softmax/prior blend weight, temperature floor
- latent shrinkage of `a`/`b` toward 0 (the plan's documented fallback step 4)
- forward-chained recon→official calibration (per-round linear, fit on rounds < R)
- selection-ordering tilt: blend point forecast with the lambdarank head for the XV pick only

## Artifacts
- `model/research.py` — loop engine, config, fitness, candidate registry.
- `research/ledger.json` + `research/LEDGER.md` — every trial, accepted or rejected, with deltas.
- `research/best_config.json` — current incumbent config; `research/sealed_2026.json` — one-shot test.
- `data/model_component_registry.csv`, `data/model_predictions_2025.csv` — refreshed on a new best.
