# v4 P0 — ruler repair: results and decisions (2026-07-17)

All P0 gates evaluated. Code: `model/unified/v4/{p0,p0_rubric,p0_margin}.py`.
Artifacts in this directory. No quarantined set influenced any admission decision;
NCR GW1–2 labels were used only as the measurement target of instrument
calibration (proxy audit) and for the error decomposition of already-frozen v3
predictions, per RESEARCH_PLAN.md §5 P0.

## P0.1 Error decomposition — S1 gate: PASS (rate-dominated, proceed)

505 matched NCR rows, frozen baseline fold artifacts, exact identity
(minutes-term + rate-term for played rows; appearance for DNP):

| share of model error | value |
|---|---|
| per-80 **rate** term | **0.853** |
| minutes term | 0.310 |
| appearance term | 0.003 |

The tail/rate program (P1/P2) is correctly aimed. Backs' rate share is 0.950 —
the attacking-tail story holds.

## Ledger #23 re-verification — one claim confirmed, one REFUTED

- Per-GW top-10 capture CONFIRMED exactly: GW1 52.6% vs 62.6%, GW2 60.2% vs 62.9%.
- "Thin-history is the norm" REFUTED: NCR cohort median is **23** prior
  international store rows; only **11.1%** (56 rows) are thin (<5 intl), not 35.6%≤3.
  ⇒ EB shrinkage (P1) is a narrower lever than Report 2 claimed; the dominant
  error mass is per-80 rate error on **rich-history** players. P1's thin-intl
  kill cohort remains as precommitted, but expectations are tempered.

## P0.2 Rubric audit + POTM — POTM head DROPPED; proxy repair is bias-only

- NCR-unobserved events: potm(+15), scrums_won(+2 FR), lineout_steals(+5),
  interceptions(+5). Feed carries no POTM; only hand-scraped `ncr_potm.csv` (GW2, 6 rows).
- **Incumbent POTM ablation: zero effect** on its top-10 capture (0.6256→0.6256,
  0.6286→0.6286) and Spearman (−0.0003). The bump is a ≤2-pt near-monotone
  transform. **The conditional POTM head is permanently dropped.**
- Proxy repair (position-level expected unobserved points from cross-competition
  store rates + scraped POTM): bias −4.33 → −2.97, MAE 5.40 → 5.21, but
  within-GW Spearman **worsens** 0.889 → 0.864. Per the plan's risk clause the
  repaired proxy is **not** promoted to a ranking-selection criterion:
  **6N-2025 LORO official points is the sole primary selection layer**; the
  proxy (with the bias correction noted) stays a secondary diagnostic.

## P0.3 Pinned picker-sensitivity margin — FROZEN at 0.1 points

Fixed 6N-2025 baseline squad, E|noise|=σ grid 0.1..1.0, 200 seeded reps:
σ=0.1 already changes ≥1 XV pick in ≥50% of reps (σ=0.2 → 97.5%).
**Derived tier-2 non-inferiority margin = 0.1 MAE points (~1.1% of incumbent
NCR MAE)** — tighter than the inherited 2%. Frozen in `p0_margin.json`.

## Consequences for the plan

1. P1 and P2 proceed (S1 passed); P1 judged primarily on overall rank metrics
   with the thin-intl cohort as the precommitted secondary read.
2. No POTM head; no further proxy-repair work.
3. Tier-2 margin = 0.1 pts absolute (harder than 2%); tier-1 unchanged.
